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

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from pyscf import dft, gto

from ciderpress.dft.settings import SemilocalSettings, get_alpha, get_s2
from ciderpress.pyscf.analyzers import RHFAnalyzer, UHFAnalyzer
from ciderpress.pyscf.descriptors import (
    get_descriptors,
    get_full_rho,
    get_labels_and_coeffs,
)
from ciderpress.pyscf.tests.utils_for_test import (
    get_random_coords,
    get_rotated_coords,
    get_scaled_mol,
    rotate_molecule,
)

nh3str = """
N        0.000000    0.000000    0.116671
H        0.000000    0.934724   -0.272232
H        0.809495   -0.467362   -0.272232
H       -0.809495   -0.467362   -0.272232
"""


FD_DELTA = 1e-5


class _Grids:
    def __init__(self, mol, coords):
        self.mol = mol
        self.coords = coords
        self.non0tab = None
        self.cutoff = 0
        self.weights = np.ones(coords.shape[0])


class TestSemilocal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np_settings = SemilocalSettings("np")
        cls.ns_settings = SemilocalSettings("ns")
        cls.npa_settings = SemilocalSettings("npa")
        cls.nst_settings = SemilocalSettings("nst")
        cls.mols = [
            gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvp", spin=0),
            gto.M(atom="O 0 0 0; O 0 0 1.2", basis="def2-tzvp", spin=2),
            gto.M(atom="Ar", basis="def2-tzvp", spin=0),
            gto.M(atom=nh3str, spin=0, basis="def2-svp"),
        ]

    def _check_sl_equivalence(self, mol, dm, coords):
        grids = _Grids(mol, coords)

        ni = dft.numint.NumInt()
        rho = get_full_rho(ni, mol, dm, grids, "MGGA")
        if rho.ndim == 2:
            rho = rho[None, :]
        nspin = rho.shape[0]
        rho = rho * nspin
        sigma = np.einsum("sxg,sxg->sg", rho[:, 1:4], rho[:, 1:4])

        if dm.ndim == 3:
            analyzer = UHFAnalyzer(mol, dm)
        else:
            analyzer = RHFAnalyzer(mol, dm)
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        feat_np = get_descriptors(analyzer, self.np_settings)
        feat_ns = get_descriptors(analyzer, self.ns_settings)
        feat_npa = get_descriptors(analyzer, self.npa_settings)
        feat_nst = get_descriptors(analyzer, self.nst_settings)
        if feat_np.ndim == 2:
            feat_np = feat_np[None, :]
            feat_ns = feat_ns[None, :]
            feat_npa = feat_npa[None, :]
            feat_nst = feat_nst[None, :]
        assert_allclose(feat_np[:, 0], rho[:, 0], atol=1e-10)
        assert_allclose(feat_ns[:, 0], rho[:, 0], atol=1e-10)
        assert_allclose(feat_npa[:, 0], rho[:, 0], atol=1e-10)
        assert_allclose(feat_nst[:, 0], rho[:, 0], atol=1e-10)

        s2 = get_s2(rho[:, 0], sigma)
        assert_allclose(feat_np[:, 1], s2, atol=1e-10)
        assert_allclose(feat_ns[:, 1], sigma, atol=1e-10)
        assert_allclose(feat_npa[:, 1], s2, atol=1e-10)
        assert_allclose(feat_nst[:, 1], sigma, atol=1e-10)

        alpha = get_alpha(rho[:, 0], sigma, rho[:, 4])
        assert_allclose(feat_npa[:, 2], alpha, atol=1e-10)
        assert_allclose(feat_nst[:, 2], rho[:, 4], atol=1e-10)

    def _get_fd_feats(
        self,
        k,
        iorb,
        spin,
        settings_list,
        analyzer,
        mo_coeff,
        mo_occ,
        mo_energy,
        inner_level,
        grids,
    ):
        dtmp = {k: [iorb]}
        labels, coeffs, en_list, sep_spins = get_labels_and_coeffs(
            dtmp, mo_coeff, mo_occ, mo_energy
        )
        if k == "U":
            delta = FD_DELTA
        else:
            delta = -1 * FD_DELTA
        if analyzer.dm.ndim == 3:
            coeff_vector = coeffs[0][0] if len(coeffs[0]) > 0 else coeffs[1][0]
        else:
            coeff_vector = coeffs[0]
        dmtmp = analyzer.dm.copy()
        if dmtmp.ndim == 3:
            dmtmp[spin] += delta * np.outer(coeff_vector, coeff_vector)
        else:
            dmtmp[:] += delta * np.outer(coeff_vector, coeff_vector)
        if analyzer.dm.ndim == 3:
            ana_tmp = UHFAnalyzer(analyzer.mol, dmtmp, grids_level=inner_level)
        else:
            ana_tmp = RHFAnalyzer(analyzer.mol, dmtmp, grids_level=inner_level)
        ana_tmp.grids = grids
        ana_tmp.perform_full_analysis()
        flist_p = []
        for settings in settings_list:
            flist_p.append(get_descriptors(ana_tmp, settings))
        dmtmp = analyzer.dm.copy()
        if dmtmp.ndim == 3:
            dmtmp[spin] -= delta * np.outer(coeff_vector, coeff_vector)
        else:
            dmtmp[:] -= delta * np.outer(coeff_vector, coeff_vector)
        if analyzer.dm.ndim == 3:
            ana_tmp = UHFAnalyzer(analyzer.mol, dmtmp, grids_level=inner_level)
        else:
            ana_tmp = RHFAnalyzer(analyzer.mol, dmtmp, grids_level=inner_level)
        ana_tmp.grids = grids
        ana_tmp.perform_full_analysis()
        flist_m = []
        for settings in settings_list:
            flist_m.append(get_descriptors(ana_tmp, settings))
        feat_fd = [
            (feat_p - feat_m)[spin] / (2 * delta)
            for feat_p, feat_m in zip(flist_p, flist_m)
        ]
        return feat_fd

    def _check_sl_occ_derivs(self, analyzer, coords, rtol=1e-3, atol=1e-3):
        mo_occ = analyzer.mo_occ
        mo_coeff = analyzer.mo_coeff
        mo_energy = analyzer.mo_energy
        grids = _Grids(analyzer.mol, coords)
        inner_level = 3
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        orbs = {"U": [2, 0], "O": [0, 1]}

        rho_ref = get_descriptors(analyzer, "l")
        np_ref = get_descriptors(
            analyzer,
            self.np_settings,
        )
        ns_ref = get_descriptors(
            analyzer,
            self.ns_settings,
        )
        npa_ref = get_descriptors(
            analyzer,
            self.npa_settings,
        )
        nst_ref = get_descriptors(
            analyzer,
            self.nst_settings,
        )

        rho_pred, occd_rho, rho_eigvals = get_descriptors(analyzer, "l", orbs=orbs)
        np_pred, occd_np, vi_eigvals = get_descriptors(
            analyzer, self.np_settings, orbs=orbs
        )
        ns_pred, occd_ns, vj_eigvals = get_descriptors(
            analyzer, self.ns_settings, orbs=orbs
        )
        npa_pred, occd_npa, vij_eigvals = get_descriptors(
            analyzer, self.npa_settings, orbs=orbs
        )
        nst_pred, occd_nst, vk_eigvals = get_descriptors(
            analyzer, self.nst_settings, orbs=orbs
        )
        assert_almost_equal(rho_pred, rho_ref, 12)
        assert_almost_equal(np_pred, np_ref, 12)
        assert_almost_equal(ns_pred, ns_ref, 12)
        assert_almost_equal(npa_pred, npa_ref, 12)
        assert_almost_equal(nst_pred, nst_ref, 12)

        for k, orblist in orbs.items():
            for iorb in orblist:
                dtmp = {k: [iorb]}
                print(k, iorb)
                labels, coeffs, en_list, sep_spins = get_labels_and_coeffs(
                    dtmp, mo_coeff, mo_occ, mo_energy
                )
                if np_pred.shape[0] == 2:
                    spin, occd_pred = occd_np[k][iorb]
                else:
                    spin, occd_pred = 0, occd_np[k][iorb]
                dfeat_list = self._get_fd_feats(
                    k,
                    iorb,
                    spin,
                    [
                        self.np_settings,
                        self.ns_settings,
                        self.npa_settings,
                        self.nst_settings,
                    ],
                    analyzer,
                    mo_coeff,
                    mo_occ,
                    mo_energy,
                    inner_level,
                    grids,
                )
                assert_allclose(occd_pred, dfeat_list[0], rtol=rtol, atol=atol)
                if ns_pred.shape[0] == 2:
                    spin, occd_pred = occd_ns[k][iorb]
                else:
                    spin, occd_pred = 0, occd_ns[k][iorb]
                assert_allclose(occd_pred, dfeat_list[1], rtol=rtol, atol=atol)
                if npa_pred.shape[0] == 2:
                    spin, occd_pred = occd_npa[k][iorb]
                else:
                    spin, occd_pred = 0, occd_npa[k][iorb]
                assert_allclose(occd_pred, dfeat_list[2], rtol=rtol, atol=atol)
                if nst_pred.shape[0] == 2:
                    spin, occd_pred = occd_nst[k][iorb]
                else:
                    spin, occd_pred = 0, occd_nst[k][iorb]
                assert_allclose(occd_pred, dfeat_list[3], rtol=rtol, atol=atol)

    def _check_rotation_invariance(self, mol, coords, settings):
        rot_coords = get_rotated_coords(coords, 0.6 * np.pi, axis="x")
        rot_mol = rotate_molecule(mol, 0.6 * np.pi, axis="x")

        ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        ks.xc = "PBE"
        ks.conv_tol = 1e-12
        ks.kernel()
        if mol.spin == 0:
            analyzer = RHFAnalyzer.from_calc(ks)
        else:
            analyzer = UHFAnalyzer.from_calc(ks)
        grids = _Grids(analyzer.mol, coords)
        grids.non0tab = None
        grids.cutoff = 0
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        desc = get_descriptors(analyzer, settings, orbs=None)

        ks = dft.RKS(rot_mol) if rot_mol.spin == 0 else dft.UKS(rot_mol)
        ks.xc = "PBE"
        ks.conv_tol = 1e-12
        ks.kernel()
        if mol.spin == 0:
            analyzer = RHFAnalyzer.from_calc(ks)
        else:
            analyzer = UHFAnalyzer.from_calc(ks)
        grids = _Grids(analyzer.mol, rot_coords)
        grids.non0tab = None
        grids.cutoff = 0
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        rot_desc = get_descriptors(analyzer, settings, orbs=None)

        assert_allclose(rot_desc, desc, rtol=1e-4, atol=1e-4)

    def test_rotation_invariance(self):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvppd")
        ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        ks.xc = "PBE"
        ks.conv_tol = 1e-12
        ks.kernel()
        coords = get_random_coords(ks)
        self._check_rotation_invariance(mol, coords, self.np_settings)
        self._check_rotation_invariance(mol, coords, self.ns_settings)
        self._check_rotation_invariance(mol, coords, self.npa_settings)
        self._check_rotation_invariance(mol, coords, self.nst_settings)

    def _check_uniform_scaling_helper(
        self,
        settings,
        analyzer,
        coords,
        mol,
        scaled_mol,
        lambd,
        rtol,
        atol,
    ):
        analyzer.mol = mol
        grids = _Grids(mol, coords)
        grids.non0tab = None
        grids.cutoff = 0
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        desc = get_descriptors(
            analyzer,
            settings,
            orbs=None,
        )
        grids = _Grids(scaled_mol, coords / lambd)
        grids.non0tab = None
        grids.cutoff = 0
        analyzer.grids = grids
        analyzer.mol = scaled_mol
        analyzer.perform_full_analysis()
        sdesc = get_descriptors(
            analyzer,
            settings,
            orbs=None,
        )
        pows = np.array(settings.get_feat_usps())
        assert_allclose(
            sdesc - desc,
            desc * lambd ** pows[None, :, None] - desc,
            rtol=rtol,
            atol=atol,
        )

    def _check_uniform_scaling(self, analyzer, coords, lambd=1.3, rtol=1e-3, atol=1e-3):
        mol = analyzer.mol.copy()
        scaled_mol = get_scaled_mol(mol, lambd)
        args = [
            self.np_settings,
            analyzer,
            coords,
            mol,
            scaled_mol,
            lambd,
            rtol,
            atol,
        ]
        self._check_uniform_scaling_helper(*args)
        args[0] = self.ns_settings
        self._check_uniform_scaling_helper(*args)
        args[0] = self.npa_settings
        self._check_uniform_scaling_helper(*args)
        args[0] = self.nst_settings
        self._check_uniform_scaling_helper(*args)

    def test_uniform_scaling(self):
        tols = [(5e-4, 5e-4), (5e-4, 5e-4), (3e-3, 3e-3), (1e-3, 1e-3)]
        for mol, (atol, rtol) in zip(self.mols, tols):
            ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            ks.xc = "PBE"
            ks.conv_tol = 1e-12
            ks.kernel()
            coords = get_random_coords(ks)
            if mol.spin != 0:
                analyzer = UHFAnalyzer.from_calc(ks)
            else:
                analyzer = RHFAnalyzer.from_calc(ks)
            self._check_uniform_scaling(analyzer, coords, rtol=rtol, atol=atol)

    def test_sl_equivalence(self):
        mols = self.mols
        tols = [(5e-4, 5e-4), (2e-4, 5e-4), (1e-3, 1e-3), (5e-4, 5e-4)]
        for mol, (atol, rtol) in zip(mols, tols):
            print(mol.atom)
            ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            ks.xc = "PBE"
            ks.conv_tol = 1e-12
            ks.kernel()
            coords = get_random_coords(ks)
            self._check_sl_equivalence(mol, ks.make_rdm1(), coords)

    def test_sl_occ_derivs(self):
        tols = [(1e-3, 1e-3), (1e-3, 1e-3), (3e-3, 3e-3), (1e-3, 1e-3)]
        for mol, (atol, rtol) in zip(self.mols, tols):
            print(mol.atom)
            ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            ks.xc = "PBE"
            ks.kernel()
            coords = get_random_coords(ks)
            dm = ks.make_rdm1()
            if dm.ndim == 3:
                analyzer = UHFAnalyzer.from_calc(ks)
            else:
                analyzer = RHFAnalyzer.from_calc(ks)
            self._check_sl_occ_derivs(analyzer, coords, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main()
