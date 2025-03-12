#!/usr/bin/env python
# CiderPress: Machine-learning based density functional theory calculations
# Copyright (C) 2024 The President and Fellows of Harvard College
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

from abc import ABC, abstractmethod

import numpy as np
from pyscf import dft, gto, lib, scf
from pyscf.dft.gen_grid import Grids
from pyscf.sgx.sgx import sgx_fit

from ciderpress.external.sgx_tools import get_jk_densities

CALC_TYPES = {
    "RHF": scf.hf.RHF,
    "UHF": scf.uhf.UHF,
}


def recursive_remove_none(obj):
    if isinstance(obj, dict):
        return {k: recursive_remove_none(v) for k, v in obj.items() if v is not None}
    else:
        return obj


def recursive_bytes_to_str(obj):
    if isinstance(obj, dict):
        return {k: recursive_bytes_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        lst = [recursive_bytes_to_str(item) for item in obj]
        if isinstance(obj, tuple):
            lst = tuple(lst)
        return lst
    elif isinstance(obj, bytes):
        return obj.decode("utf-8")
    else:
        return obj


class ElectronAnalyzer(ABC):
    """
    A class for generating and storing data derived from a PySCF
    electronic structure calculation, in particular distributions
    on a real-space grid such as the density, exchange (correlation)
    energy density, Coulomb energy density, etc.
    """

    _atype = None

    def __init__(
        self, mol, dm, grids_level=3, mo_occ=None, mo_coeff=None, mo_energy=None
    ):
        if mo_occ is None and hasattr(dm, "mo_occ"):
            self.mo_occ = dm.mo_occ
        else:
            self.mo_occ = mo_occ
        if mo_coeff is None and hasattr(dm, "mo_coeff"):
            self.mo_coeff = dm.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if mo_energy is None and hasattr(dm, "mo_energy"):
            self.mo_energy = dm.mo_energy
        else:
            self.mo_energy = mo_energy
        self.dm = dm
        self.mol = mol
        self.mol.verbose = 3
        self.mol.build()
        self.grids_level = grids_level
        self.grids = Grids(self.mol)
        self.grids.level = self.grids_level
        self._data = {}
        self.grids.build(with_non0tab=False)

    def keys(self):
        return self._data.keys()

    def set(self, k, v):
        self._data[k] = v

    def as_dict(self):
        return {
            "atype": self._atype,
            "data": self._data,
            "dm": self.dm,
            "mo_occ": self.mo_occ,
            "mo_coeff": self.mo_coeff,
            "mo_energy": self.mo_energy,
            "grids_level": self.grids_level,
            "mol": gto.mole.pack(self.mol),
        }

    @staticmethod
    def from_dict(d):
        args = [
            gto.mole.unpack(d["mol"]),
            d["dm"],
        ]
        kwargs = {
            "grids_level": d["grids_level"],
            "mo_occ": d.get("mo_occ"),
            "mo_coeff": d.get("mo_coeff"),
            "mo_energy": d.get("mo_energy"),
        }
        atype = d.get("atype")
        if atype is None:
            cls = ElectronAnalyzer
        elif atype == "RHF" or atype == "RKS":
            cls = RHFAnalyzer
        elif atype == "UHF" or atype == "UKS":
            cls = UHFAnalyzer
        else:
            raise ValueError("Unrecognized analyzer type {}".format(atype))
        analyzer = cls(*args, **kwargs)
        analyzer._data.update(d["data"])
        return analyzer

    def dump(self, fname):
        """
        Dump self to an hdf5 file called fname.

        Args:
            fname (str): Name of file to dump to
        """
        h5dict = recursive_remove_none(self.as_dict())
        lib.chkfile.dump(fname, "analyzer", h5dict)

    @staticmethod
    def load(fname):
        """
        Load instance of cls from hdf5

        Args:
            fname (str): Name of file from which to load
        """
        analyzer_dict = lib.chkfile.load(fname, "analyzer")
        analyzer_dict = recursive_bytes_to_str(analyzer_dict)
        return ElectronAnalyzer.from_dict(analyzer_dict)

    def get(self, name, error_if_missing=True):
        if error_if_missing:
            return self._data[name]
        else:
            return self._data.get(name, None)

    def calculate_vxc(self, xcname, xcfunc=None, grids=None, xctype="MGGA"):
        """
        Computes the XC potential matrix for a functional,
        store it, and return it.

        Args:
            xcname (str): XC functional
            xcfunc (callable): XC function in case one
                wants to use an XC functional not in libxc
            grids (optional): Grid to use to calculate XC.

        Returns:
            vxc (nspin x nao x nao): XC potential matrix
        """
        if not isinstance(xcname, str):
            raise ValueError("XC must be string")
        if self._atype == "RHF":
            calc = dft.RKS(self.mol)
        elif self._atype == "UHF":
            calc = dft.UKS(self.mol)
        else:
            raise NotImplementedError
        calc.xc = xcname
        if grids is not None:
            calc.grids = grids
        if xcfunc is not None:
            calc.define_xc_(xcfunc, xctype=xctype)
        vxc = calc.get_veff(dm=self.rdm1)
        self._data["EXC_{}".format(xcname)] = vxc.exc
        vxc = vxc - vxc.vj
        self._data["VXC_{}".format(xcname)] = vxc
        return vxc

    def calculate_vxc_on_mo(self, xcname, orbs=None, **kwargs):
        """
        Compute contributions of XC potential to the eigenvalues. If
        VXC_{xcname} is not in _data, calculate_vxc is called first.

        Args:
            xcname (str): Name of XC functional (for libxc)
            orbs: Orbital index dictionary for computing orbital derivatives,
                in the format use for descriptors.get_descriptors.
            **kwargs: extra arguments to pass to calculate_vxc, if it is called.

        Returns:
            eigenvalue contributions. If orbs is provided, it will be in
                the same format used by descritpors.get_descriptors.
                Otherwise, it will be a numpy array with all the eigenvalue
                contributions.
        """
        if "VXC_{}".format(xcname) not in self._data.keys():
            vxc = self.calculate_vxc(xcname, **kwargs)
        else:
            vxc = self._data["VXC_{}".format(xcname)]
        assert self.mo_coeff is not None
        if orbs is None:
            orbvals = vxc.dot(self.mo_coeff)
            eigvals = np.einsum("...ui,ui->...i", self.mo_coeff, orbvals)
            self._data["ORBXC_{}".format(xcname)] = eigvals
        else:
            from ciderpress.pyscf.descriptors import (
                get_labels_and_coeffs,
                unpack_eigvals,
            )

            labels, coeffs, en_list, sep_spins = get_labels_and_coeffs(
                orbs, self.mo_coeff, self.mo_occ, self.mo_energy
            )
            if self.mo_occ.ndim == 2:
                orbvals = {}
                for s in range(2):
                    c = np.array(coeffs[s])
                    if len(c) == 0:
                        orbvals[s] = None
                        continue
                    orbvals[s] = np.array(c).dot(vxc[s])
                    orbvals[s] = np.einsum("iu,iu->i", c, orbvals[s])
            else:
                c = np.array(coeffs)
                orbvals = c.dot(vxc)
                orbvals = np.einsum("iu,iu->i", c, orbvals)
            if sep_spins:
                eigvals = (
                    unpack_eigvals(orbs[0], labels[0], orbvals[0]),
                    unpack_eigvals(orbs[1], labels[1], orbvals[1]),
                )
            elif self.mo_occ.ndim == 2:
                eigvals = {}
                for k in list(orbs.keys()):
                    eigvals[k] = {}
                for s in range(2):
                    eigvals_tmp = unpack_eigvals(orbs, labels[s], orbvals[s])
                    for k in list(orbs.keys()):
                        eigvals[k].update(eigvals_tmp[k])
            else:
                eigvals = unpack_eigvals(orbs, labels, orbvals)
            self._data["ORBXC_{}".format(xcname)] = eigvals
        return self._data["ORBXC_{}".format(xcname)]

    @staticmethod
    def from_calc(calc, grids_level=None, store_energy_orig=True):
        """
        NOTE: This has side effects on calc, see notes below.

        Args:
            calc: SCF object from PySCF
            grids_level: Size of PySCF grids for XC integration
            store_energy_orig: Whether to store original xc energy.

        Returns:
            ElectronAnalyzer: analyzer constructed from calc
        """
        if isinstance(calc, scf.uhf.UHF):
            cls = UHFAnalyzer
        elif isinstance(calc, scf.hf.RHF):
            cls = RHFAnalyzer
        else:
            raise ValueError("Must be HF calculation")

        if grids_level is None:
            if isinstance(calc, dft.rks.KohnShamDFT):
                grids_level = calc.grids.level
            else:
                grids_level = 3
        analyzer = cls(
            calc.mol,
            calc.make_rdm1(),
            grids_level=grids_level,
            mo_occ=calc.mo_occ,
            mo_coeff=calc.mo_coeff,
            mo_energy=calc.mo_energy,
        )
        if store_energy_orig and isinstance(calc, dft.rks.KohnShamDFT):
            # save initial exc for reference
            if (calc.scf_summary.get("exc") is None) or (
                grids_level != calc.grids.level
            ):
                # NOTE this should leave calc almost the same
                # as it started, but might make some
                # minor changes if there are non-default
                # settings/screenings on the grids. Also
                # scf_summary will be overwritten
                old_level = calc.grids.level
                calc.grids.level = grids_level
                calc.grids.build()
                if hasattr(calc, "with_df") and hasattr(calc.with_df, "grids"):
                    calc.with_df.build()
                if hasattr(calc, "_numint") and hasattr(calc._numint, "build"):
                    calc._numint.build()
                e_tot = calc.energy_tot(analyzer.dm)
                calc.grids.level = old_level
                calc.grids.build()
            else:
                e_tot = calc.e_tot
            analyzer._data["xc_orig"] = calc.xc
            analyzer._data["exc_orig"] = calc.scf_summary["exc"]
            analyzer._data["e_tot_orig"] = e_tot
        return analyzer

    def get_rho_data(self, overwrite=False):
        if "rho_data" in self.keys() and not overwrite:
            return self.get("rho_data")
        ni = dft.numint.NumInt()
        spinpol = self.dm.ndim == 3
        if spinpol:
            rho_list_a = []
            rho_list_b = []
        else:
            rho_list = []
        for ao, non0, weight, coord in ni.block_loop(self.mol, self.grids, deriv=2):
            if spinpol:
                rho_list_a.append(ni.eval_rho(self.mol, ao, self.dm[0], xctype="MGGA"))
                rho_list_b.append(ni.eval_rho(self.mol, ao, self.dm[1], xctype="MGGA"))
            else:
                rho_list.append(ni.eval_rho(self.mol, ao, self.dm, xctype="MGGA"))
        if spinpol:
            rhoa = np.concatenate(rho_list_a, axis=-1)
            rhob = np.concatenate(rho_list_b, axis=-1)
            self._data["rho_data"] = np.stack([rhoa, rhob], axis=0)
        else:
            self._data["rho_data"] = np.concatenate(rho_list, axis=-1)
        return self._data["rho_data"]

    @abstractmethod
    def perform_full_analysis(self):
        pass

    @property
    def rdm1(self):
        return self.dm

    @property
    def rho_data(self):
        return self.get("rho_data")


class RHFAnalyzer(ElectronAnalyzer):

    _atype = "RHF"

    def get_xc(self, xcname):
        """
        Function for accessing XC functional data from _data
        """
        if xcname in self._data["xc_list"]:
            return self._data["xc"][xcname]
        else:
            raise ValueError("Data not computed for xc functional")

    def get_rs(self, term, omega, tol=1e-9):
        """
        Function for accessing range-separated data from _data
        Finds match based on a tolerance, in case rounding
        issues arise with floats.
        """
        if not (term in ["ee", "ha", "ex"]):
            raise ValueError("Term must be ee, ha, or ex")
        name = "{}_energy_density_rs".format(term)
        for ind, omega_tst in enumerate(self._data["omega_list"]):
            if abs(omega_tst - omega) < tol:
                return self._data[name][ind]
        else:
            raise RuntimeError("RS data not found for omega={}".format(omega))

    def get_xc_energy(self, xcname):
        """
        Store and return the XC energy for the functional given by xcname
        """
        if self._data.get("xc") is None:
            self._data["xc_list"] = []
            self._data["xc"] = {}
        ni = dft.numint.NumInt()
        if isinstance(self, UHFAnalyzer):
            _, exc, _ = ni.nr_uks(
                self.mol, self.grids, xcname, self.dm, relativity=0, hermi=1
            )
        else:
            _, exc, _ = ni.nr_rks(
                self.mol, self.grids, xcname, self.dm, relativity=0, hermi=1
            )
        self._data["xc_list"].append(xcname)
        self._data["xc"][xcname] = exc
        return exc

    def get_ha_energy_density(self):
        self.get_ee_energy_density()
        return self._data["ha_energy_density"]

    def get_ex_energy_density(self):
        self.get_ee_energy_density()
        return self._data["ex_energy_density"]

    get_fx_energy_density = get_ex_energy_density

    def _get_ej_ek(self, omega=None):
        calc_tmp = sgx_fit(scf.RHF(self.mol))
        calc_tmp.build()
        calc_tmp.with_df.grids = self.grids
        if omega is None:
            ej, ek = get_jk_densities(calc_tmp.with_df, self.dm)
        else:
            with calc_tmp.mol.with_range_coulomb(omega):
                ej, ek = get_jk_densities(calc_tmp.with_df, self.dm)
        return ej, ek

    def _get_ee(self, omega=None):
        ej, ek = self._get_ej_ek(omega)
        ej = ej[0]
        ek = 0.5 * ek[0]
        return ej, ek, ej + ek

    def get_ee_energy_density(self):
        """
        Returns the sum of E_{Ha} and E_{X}, i.e. the Coulomb repulsion
        energy of the HF or KS Slater determinant.
        """
        ej, ek, ee = self._get_ee()
        self._data["ha_energy_density"] = ej
        self._data["ex_energy_density"] = ek
        self._data["ee_energy_density"] = ee
        return self._data["ee_energy_density"]

    def get_ee_energy_density_rs(self, omega):
        if self._data.get("omega_list") is None:
            self._data["omega_list"] = []
            self._data["ee_energy_density_rs"] = []
            self._data["ha_energy_density_rs"] = []
            self._data["ex_energy_density_rs"] = []
        ej, ek, ee = self._get_ee(omega)
        self._data["omega_list"].append(omega)
        self._data["ha_energy_density_rs"].append(ej)
        self._data["ex_energy_density_rs"].append(ek)
        self._data["ee_energy_density_rs"].append(ee)
        return self._data["ee_energy_density_rs"][-1]

    def perform_full_analysis(self):
        self.get_rho_data()
        self.get_ee_energy_density()

    @property
    def atype(self):
        return self._atype


RKSAnalyzer = RHFAnalyzer


class UHFAnalyzer(RHFAnalyzer):

    _atype = "UHF"

    def _get_ej_ek(self, omega=None):
        calc_tmp = sgx_fit(scf.UHF(self.mol))
        calc_tmp.build()
        calc_tmp.with_df.grids = self.grids
        if omega is None:
            ej, ek = get_jk_densities(calc_tmp.with_df, self.dm)
        else:
            with calc_tmp.mol.with_range_coulomb(omega):
                ej, ek = get_jk_densities(calc_tmp.with_df, self.dm)
        return ej, ek

    def _get_ee(self, omega=None):
        ej, ek = self._get_ej_ek(omega)
        return ej, ek, ej + ek


UKSAnalyzer = UHFAnalyzer
