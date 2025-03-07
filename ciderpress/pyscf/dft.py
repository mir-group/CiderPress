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

from pyscf import lib
from pyscf.dft.gen_grid import Grids

from ciderpress.dft.model_utils import get_slxc_settings, load_cider_model
from ciderpress.pyscf.gen_cider_grid import CiderGrids
from ciderpress.pyscf.nldf_convolutions import PySCFNLDFInitializer
from ciderpress.pyscf.numint import CiderNumInt, NLDFNLOFNumInt, NLDFNumInt, NLOFNumInt
from ciderpress.pyscf.sdmx import PySCFSDMXInitializer


def make_cider_calc(
    ks,
    mlfunc,
    xmix=1.0,
    xc=None,
    xkernel=None,
    ckernel=None,
    mlfunc_format=None,
    nlc_coeff=None,
    nldf_init=None,
    sdmx_init=None,
    rhocut=None,
):
    """
    Decorate the PySCF DFT object ks with a CIDER functional mlfunc.
    If xc, xkernel, ckernel, and xmix are not specified,
    the equivalent of HF with CIDER in place of EXX is performed.
    The XC energy is::

        E_xc = xmix * E_x^CIDER + (1-xmix) * xkernel + ckernel + xc

    Note the above formula applies even if ``E_x^CIDER`` is a
    full XC functional. If ``E_x^CIDER`` is a full XC functional that
    does not need a baseline, the user should pass ``xmix=1.0``, ``xc=None``,
    ``xkernel=None``, ``ckernel=None`` (the defaults).

    NOTE: Only GGA-level XC functionals can be used with GGA-level
    (orbital-independent) CIDER functionals currently.

    Args:
        ks (pyscf.dft.KohnShamDFT): DFT object
        mlfunc (MappedXC, MappedXC2, str): CIDER exchange functional or file name
        xmix (float): Fraction of CIDER exchange used.
        xc (str or None): If specified, this semi-local XC code is evaluated
             and added to the total XC energy.
        xkernel (str or None): Semi-local X code in libxc. Scaled by (1-xmix).
        ckernel (str or None): Semi-local C code in libxc.
        mlfunc_format (str or None): 'joblib' or 'yaml', specifies the format
            of mlfunc if it is a string corresponding to a file name.
            If unspecified, infer from file extension and raise error
            if file type cannot be determined.
        nlc_coeff (tuple or None):
            VV10 coefficients. If None, VV10 term is not evaluated.
        nldf_init (PySCFNLDFInitializer)
        sdmx_init (PySCFSDMXInitializer)
        rhocut (float)

    Returns:
        A decorated Kohn-Sham object for performing a CIDER calculation.
    """
    mlfunc = load_cider_model(mlfunc, mlfunc_format)
    ks._xc = get_slxc_settings(xc, xkernel, ckernel, xmix)
    # Assign the PySCF-facing functional to be a simple SL
    # functional to avoid hybrid DFT being called.
    # NOTE this might need to be changed to some nicer
    # approach later.
    if mlfunc.settings.sl_settings.level == "MGGA":
        ks.xc = "R2SCAN"
    else:
        ks.xc = "PBE"
    new_ks = _CiderKS(
        ks,
        mlfunc,
        xmix=xmix,
        nldf_init=nldf_init,
        sdmx_init=sdmx_init,
        rhocut=rhocut,
        nlc_coeff=nlc_coeff,
    )
    return lib.set_class(new_ks, (_CiderKS, ks.__class__))


class _CiderKS:

    grids = None

    def __init__(
        self,
        mf,
        mlxc,
        xmix=1.0,
        nldf_init=None,
        sdmx_init=None,
        rhocut=None,
        nlc_coeff=None,
    ):
        self.__dict__.update(mf.__dict__)
        # self.mlxc = None
        # self.xmix = None
        # self.nldf_init = None
        # self.sdmx_init = None
        # self.rhocut = rhocut
        self.set_mlxc(
            mlxc,
            xmix=xmix,
            nldf_init=nldf_init,
            sdmx_init=sdmx_init,
            rhocut=rhocut,
            nlc_coeff=nlc_coeff,
        )

    def set_mlxc(
        self,
        mlxc,
        xmix=1.0,
        nldf_init=None,
        sdmx_init=None,
        rhocut=None,
        nlc_coeff=None,
    ):
        # self.mlxc = mlxc
        # self.xmix = xmix
        # self.nldf_init = nldf_init
        # self.sdmx_init = sdmx_init
        if nldf_init is None and mlxc.settings.has_nldf:
            nldf_init = PySCFNLDFInitializer(mlxc.settings.nldf_settings)
        if sdmx_init is None and mlxc.settings.has_sdmx:
            sdmx_init = PySCFSDMXInitializer(mlxc.settings.sdmx_settings, lowmem=False)
        old_grids = self.grids
        changed = False
        if mlxc.settings.has_nldf:
            if not isinstance(old_grids, CiderGrids):
                changed = True
                self.grids = CiderGrids(self.mol)
        else:
            if old_grids is None or isinstance(old_grids, CiderGrids):
                changed = True
                self.grids = Grids(self.mol)
        if changed:
            for key in (
                "atom_grid",
                "atomic_radii",
                "radii_adjust",
                "radi_method",
                "becke_scheme",
                "prune",
                "level",
            ):
                self.grids.__setattr__(key, old_grids.__getattribute__(key))
        settings = mlxc.settings
        has_nldf = not settings.nldf_settings.is_empty
        has_nlof = not settings.nlof_settings.is_empty
        if has_nldf and has_nlof:
            cls = NLDFNLOFNumInt
        elif has_nldf:
            cls = NLDFNumInt
        elif has_nlof:
            cls = NLOFNumInt
        else:
            cls = CiderNumInt
        self._numint = cls(
            mlxc,
            self._xc,
            nldf_init,
            sdmx_init,
            xmix=xmix,
            rhocut=rhocut,
            nlc_coeff=nlc_coeff,
        )

    def build(self, mol=None, **kwargs):
        self._numint.build(mol=mol)
        return super().build(mol, **kwargs)

    def reset(self, mol=None):
        self._numint.reset(mol=mol)
        return super().reset(mol)

    def method_not_implemented(self, *args, **kwargs):
        raise NotImplementedError

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        new_self = super().density_fit(
            auxbasis=auxbasis, with_df=with_df, only_dfj=only_dfj
        )
        lib.set_class(new_self, (_CiderDF, new_self.__class__))
        return new_self

    def nuc_grad_method(self):
        from pyscf import dft

        has_df = hasattr(self, "with_df") and self.with_df is not None
        if isinstance(self, dft.rks.RKS):
            from ciderpress.pyscf import rks_grad

            if has_df:
                return rks_grad.DFGradients(self)
            else:
                return rks_grad.Gradients(self)
        elif isinstance(self, dft.uks.UKS):
            from ciderpress.pyscf import uks_grad

            if has_df:
                return uks_grad.DFGradients(self)
            else:
                return uks_grad.Gradients(self)

    Gradients = nuc_grad_method

    Hessian = method_not_implemented
    NMR = method_not_implemented
    NSR = method_not_implemented
    Polarizability = method_not_implemented
    RotationalGTensor = method_not_implemented
    MP2 = method_not_implemented
    CISD = method_not_implemented
    CCSD = method_not_implemented
    CASCI = method_not_implemented
    CASSCF = method_not_implemented


class _CiderDF:
    nuc_grad_method = _CiderKS.nuc_grad_method
