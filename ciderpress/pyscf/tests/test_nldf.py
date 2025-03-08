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
from pyscf.dft.gen_grid import Grids

from ciderpress.dft.settings import (
    NLDFSettingsVI,
    NLDFSettingsVIJ,
    NLDFSettingsVJ,
    NLDFSettingsVK,
)
from ciderpress.pyscf.analyzers import RHFAnalyzer, UHFAnalyzer
from ciderpress.pyscf.debug_numint import get_nldf_numint
from ciderpress.pyscf.descriptors import (
    get_descriptors,
    get_full_rho,
    get_labels_and_coeffs,
)
from ciderpress.pyscf.gen_cider_grid import CiderGrids
from ciderpress.pyscf.nldf_convolutions import PyscfNLDFGenerator
from ciderpress.pyscf.tests.utils_for_test import (
    get_random_coords,
    get_rotated_coords,
    get_scaled_grid,
    rotate_molecule,
)

nh3str = """
N        0.000000    0.000000    0.116671
H        0.000000    0.934724   -0.272232
H        0.809495   -0.467362   -0.272232
H       -0.809495   -0.467362   -0.272232
"""


FD_DELTA = 5e-6
FD_DELTA2 = 1e-4


class _Grids:
    def __init__(self, mol, coords):
        self.mol = mol
        self.coords = coords
        self.non0tab = None
        self.cutoff = 0
        self.weights = np.ones(coords.shape[0])


class _TestNLDFBase:

    _plan_type = None

    @classmethod
    def setUpClass(cls):
        vj_specs = ["se", "se_ar2", "se_a2r4", "se_erf_rinv"]
        theta_params = [1.0, 0.0, 0.03125]
        feat_params = [[2.0, 0.0, 0.04] for i in range(4)]
        feat_params[-1].append(2.0)
        cls.vi_settings = NLDFSettingsVI(
            "MGGA",
            theta_params,
            "one",
            ["se_ap"],
            ["se_grad"],
            [(0, 0)],
        )
        cls.vj_settings = NLDFSettingsVJ(
            "MGGA", theta_params, "one", vj_specs, feat_params
        )
        cls.vij_settings = NLDFSettingsVIJ(
            "MGGA",
            theta_params,
            "one",
            ["se_ap"],
            ["se_grad"],
            [(0, 0)],
            vj_specs,
            feat_params,
        )
        cls.vk_settings = NLDFSettingsVK(
            "MGGA",
            theta_params,
            "one",
            [[1.0, 0.0, 0.02], [2.0, 0.0, 0.04], [4.0, 0.0, 0.08], [8.0, 0.0, 0.16]],
            "exponential",
        )
        cls.vv_gg_kwargs = {
            "a0": theta_params[0],
            "fac_mul": theta_params[2],
        }
        cls.gg_kwargs = {
            "a0": feat_params[0][0],
            "fac_mul": feat_params[0][2],
        }
        cls.mols = [
            gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvp", spin=0),
            gto.M(atom="O 0 0 0; O 0 1.2 0", basis="def2-tzvp", spin=2),
            gto.M(atom="Ar", basis="def2-tzvp", spin=0),
            gto.M(atom=nh3str, spin=0, basis="def2-svp"),
        ]
        cls.o2pa = gto.M(atom="O 0 0 0; O 0 1.2 0", basis="def2-svp", spin=1, charge=1)
        cls.o2pb = gto.M(atom="O 0 0 0; O 0 1.2 0", basis="def2-svp", spin=-1, charge=1)

    def test_same_spin_issue(self):
        ksa = dft.UKS(self.o2pa)
        ksb = dft.UKS(self.o2pb)
        ksa.xc = "HF"
        ksb.xc = "HF"
        ksa.grids.level = 1
        ksb.grids.level = 1
        ksa.conv_tol = 1e-13
        ksb.conv_tol = 1e-13
        ksa.kernel()
        ksb.kernel()
        ana_a = UHFAnalyzer.from_calc(ksa)
        ana_b = UHFAnalyzer.from_calc(ksb)
        orbs = {"U": [0], "O": [0]}
        desca, ddesca, eigvalsa = get_descriptors(ana_a, self.vj_settings, orbs=orbs)
        descb, ddescb, eigvalsb = get_descriptors(ana_b, self.vj_settings, orbs=orbs)
        wt = ksa.grids.weights
        desca *= wt
        descb *= wt
        # We have to take the sum because of symmetry breaking
        assert_allclose(
            desca.sum(axis=2), descb[::-1].sum(axis=2), rtol=1e-4, atol=1e-7
        )
        assert_allclose(
            (ddesca["U"][0][1] * wt).sum(axis=1),
            (ddescb["U"][0][1] * wt).sum(axis=1),
            rtol=1e-3,
            atol=1e-7,
        )
        assert_allclose(
            (ddesca["O"][0][1] * wt).sum(axis=1),
            (ddescb["O"][0][1] * wt).sum(axis=1),
            rtol=1e-3,
            atol=1e-7,
        )

    def _check_nldf_equivalence(
        self, mol, dm, coords, rtol=2e-3, atol=2e-3, plan_type="gaussian"
    ):
        grids = _Grids(mol, coords)
        inner_level = 3
        lambd = 1.6
        beta = 1.6

        _ifeat = get_nldf_numint(
            mol,
            grids,
            dm,
            self.gg_kwargs,
            self.vv_gg_kwargs,
            version="i",
            inner_grids_level=inner_level,
        ).transpose(0, 2, 1)
        ifeat = np.zeros((_ifeat.shape[0], self.vi_settings.nfeat, _ifeat.shape[2]))
        ifeat[:, 0] = _ifeat[:, 3]
        jfeat = get_nldf_numint(
            mol,
            grids,
            dm,
            self.gg_kwargs,
            self.vv_gg_kwargs,
            version="j",
            inner_grids_level=inner_level,
        ).transpose(0, 2, 1)
        kfeat = get_nldf_numint(
            mol,
            grids,
            dm,
            self.gg_kwargs,
            self.vv_gg_kwargs,
            version="k",
            inner_grids_level=inner_level,
        ).transpose(0, 2, 1)

        if dm.ndim == 3:
            analyzer = UHFAnalyzer(mol, dm)
        else:
            analyzer = RHFAnalyzer(mol, dm)
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        n0 = 6
        l1j = _ifeat[:, n0 + 3 : n0 + 6]
        ifeat[:, 1] = np.einsum("sxg,sxg->sg", l1j, l1j)
        ijfeat = np.append(jfeat, ifeat, axis=1)
        ifeat_pred = get_descriptors(
            analyzer,
            self.vi_settings,
            plan_type=plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        assert_allclose(ifeat_pred, ifeat, rtol=rtol, atol=atol)
        jfeat_pred = get_descriptors(
            analyzer,
            self.vj_settings,
            plan_type=plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        assert_allclose(jfeat_pred, jfeat, rtol=rtol, atol=atol)
        ijfeat_pred = get_descriptors(
            analyzer,
            self.vij_settings,
            plan_type=plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        assert_allclose(ijfeat_pred, ijfeat, rtol=rtol, atol=atol)
        kfeat_pred = get_descriptors(
            analyzer,
            self.vk_settings,
            plan_type=plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        assert kfeat_pred.shape == kfeat.shape, "{} {}".format(
            kfeat_pred.shape, kfeat.shape
        )
        for i in range(kfeat.shape[1]):
            assert_allclose(kfeat_pred[:, i], kfeat[:, i], rtol=rtol, atol=atol)

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
        lambd,
        beta,
    ):
        dtmp = {k: [iorb]}
        labels, coeffs, en_list, sep_spins = get_labels_and_coeffs(
            dtmp, mo_coeff, mo_occ, mo_energy
        )
        if k == "U":
            delta = FD_DELTA2
        else:
            delta = -1 * FD_DELTA2
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
            flist_p.append(
                get_descriptors(
                    ana_tmp,
                    settings,
                    plan_type=self._plan_type,
                    aux_lambd=lambd,
                    aug_beta=beta,
                )
            )
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
            flist_m.append(
                get_descriptors(
                    ana_tmp,
                    settings,
                    plan_type=self._plan_type,
                    aux_lambd=lambd,
                    aug_beta=beta,
                )
            )
        feat_fd = [
            (feat_p - feat_m)[spin] / (2 * delta)
            for feat_p, feat_m in zip(flist_p, flist_m)
        ]
        return feat_fd

    def _check_nldf_occ_derivs(self, analyzer, coords, rtol=1e-3, atol=1e-3):
        mo_occ = analyzer.mo_occ
        mo_coeff = analyzer.mo_coeff
        mo_energy = analyzer.mo_energy
        grids = _Grids(analyzer.mol, coords)
        inner_level = 3
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        orbs = {"U": [2, 0], "O": [0, 1]}

        lambd = 1.6
        beta = 1.6

        rho_ref = get_descriptors(analyzer, "l")
        ifeat_ref = get_descriptors(
            analyzer,
            self.vi_settings,
            plan_type=self._plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        jfeat_ref = get_descriptors(
            analyzer,
            self.vj_settings,
            plan_type=self._plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        ijfeat_ref = get_descriptors(
            analyzer,
            self.vij_settings,
            plan_type=self._plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        kfeat_ref = get_descriptors(
            analyzer,
            self.vk_settings,
            plan_type=self._plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )

        rho_pred, occd_rho, rho_eigvals = get_descriptors(analyzer, "l", orbs=orbs)
        ifeat_pred, occd_ifeat, vi_eigvals = get_descriptors(
            analyzer,
            self.vi_settings,
            orbs=orbs,
            plan_type=self._plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        jfeat_pred, occd_jfeat, vj_eigvals = get_descriptors(
            analyzer,
            self.vj_settings,
            orbs=orbs,
            plan_type=self._plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        ijfeat_pred, occd_ijfeat, vij_eigvals = get_descriptors(
            analyzer,
            self.vij_settings,
            orbs=orbs,
            plan_type=self._plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        kfeat_pred, occd_kfeat, vk_eigvals = get_descriptors(
            analyzer,
            self.vk_settings,
            orbs=orbs,
            plan_type=self._plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        assert_almost_equal(rho_pred, rho_ref, 12)
        assert_almost_equal(ifeat_pred, ifeat_ref, 12)
        assert_almost_equal(jfeat_pred, jfeat_ref, 12)
        assert_almost_equal(ijfeat_pred, ijfeat_ref, 12)
        assert_almost_equal(kfeat_pred, kfeat_ref, 12)

        for k, orblist in orbs.items():
            for iorb in orblist:
                dtmp = {k: [iorb]}
                labels, coeffs, en_list, sep_spins = get_labels_and_coeffs(
                    dtmp, mo_coeff, mo_occ, mo_energy
                )
                if ifeat_pred.shape[0] == 2:
                    spin, occd_pred = occd_ifeat[k][iorb]
                else:
                    spin, occd_pred = 0, occd_ifeat[k][iorb]
                dfeat_list = self._get_fd_feats(
                    k,
                    iorb,
                    spin,
                    [
                        self.vi_settings,
                        self.vj_settings,
                        self.vij_settings,
                        self.vk_settings,
                    ],
                    analyzer,
                    mo_coeff,
                    mo_occ,
                    mo_energy,
                    inner_level,
                    grids,
                    lambd,
                    beta,
                )
                # NOTE: vi still a bit less stable, so we use larger tolerance
                assert_allclose(
                    occd_pred, dfeat_list[0], rtol=10 * rtol, atol=10 * atol
                )
                if ijfeat_pred.shape[0] == 2:
                    spin, occd_pred = occd_jfeat[k][iorb]
                else:
                    spin, occd_pred = 0, occd_jfeat[k][iorb]
                assert_allclose(occd_pred, dfeat_list[1], rtol=rtol, atol=atol)
                if ijfeat_pred.shape[0] == 2:
                    spin, occd_pred = occd_ijfeat[k][iorb]
                else:
                    spin, occd_pred = 0, occd_ijfeat[k][iorb]
                assert_allclose(
                    occd_pred, dfeat_list[2], rtol=10 * rtol, atol=10 * atol
                )
                if kfeat_pred.shape[0] == 2:
                    spin, occd_pred = occd_kfeat[k][iorb]
                else:
                    spin, occd_pred = 0, occd_kfeat[k][iorb]
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
        self._check_rotation_invariance(mol, coords, self.vi_settings)
        self._check_rotation_invariance(mol, coords, self.vj_settings)
        self._check_rotation_invariance(mol, coords, self.vij_settings)
        self._check_rotation_invariance(mol, coords, self.vk_settings)

    def _check_uniform_scaling_helper(
        self,
        settings,
        analyzer,
        coords,
        mol,
        inner_grids,
        scaled_inner_grids,
        lambd,
        plan_type,
        rtol,
        atol,
    ):
        analyzer.mol = mol
        scaled_mol = scaled_inner_grids.mol
        grids = _Grids(mol, coords)
        grids.non0tab = None
        grids.cutoff = 0
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        desc = get_descriptors(
            analyzer,
            settings,
            orbs=None,
            plan_type=plan_type,
            inner_grids=inner_grids,
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
            plan_type=plan_type,
            inner_grids=scaled_inner_grids,
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
        inner_grids = CiderGrids(mol)
        inner_grids.build(with_non0tab=False)
        inner_grids.cutoff = 0
        scaled_inner_grids = get_scaled_grid(inner_grids, lambd)
        args = [
            self.vi_settings,
            analyzer,
            coords,
            mol,
            inner_grids,
            scaled_inner_grids,
            lambd,
            self._plan_type,
            rtol,
            atol,
        ]
        self._check_uniform_scaling_helper(*args)
        args[0] = self.vj_settings
        self._check_uniform_scaling_helper(*args)
        args[0] = self.vij_settings
        self._check_uniform_scaling_helper(*args)
        args[0] = self.vk_settings
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

    def _check_nldf_vxc(
        self,
        analyzer,
        rtol=1e-3,
        atol=1e-5,
        interpolator_type="onsite_spline",
        check_feat_equiv=True,
    ):
        mo_occ = analyzer.mo_occ
        mo_coeff = analyzer.mo_coeff
        mo_energy = analyzer.mo_energy
        inner_level = 3
        grids = CiderGrids(analyzer.mol)
        grids.build()
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        rho = analyzer.get_rho_data()
        if rho.ndim == 2:
            rho = np.ascontiguousarray(rho[None, [0, 1, 2, 3, -1]])
        else:
            rho = np.ascontiguousarray(rho[:, [0, 1, 2, 3, -1]])
        orbs = {"U": [0], "O": [0]}

        lambd = 1.6
        beta = 1.6

        all_settings = [
            self.vi_settings,
            self.vj_settings,
            self.vij_settings,
            self.vk_settings,
        ]

        nspin = 1 if analyzer.dm.ndim == 2 else 2

        for settings in all_settings:
            feat_ref, occd_feat, eigvals = get_descriptors(
                analyzer,
                settings,
                orbs=orbs,
                plan_type=self._plan_type,
                aux_lambd=lambd,
                aug_beta=beta,
            )
            feat_ref[..., grids.weights.size - grids.grids_indexer.padding :] = 0.0
            factor = np.mean(1.0 / (0.1 + feat_ref**2), axis=(0, 2))

            def get_e_and_v(feat, use_mean=False):
                sum = (factor[:, None] * feat**2).sum(1)
                if use_mean:
                    sum /= len(factor)
                vrho = sum / (1 + sum)
                e = rho[:, 0] * vrho
                vfeat = rho[:, 0] * 1 / (1 + sum) ** 2
                vfeat = vfeat[:, None, :] * 2 * (factor[:, None] * feat)
                if use_mean:
                    vfeat /= len(factor)
                e = np.dot(e, grids.weights).sum()
                vrho *= grids.weights
                vfeat *= grids.weights
                return e, vrho, vfeat

            nldf_gen = PyscfNLDFGenerator.from_mol_and_settings(
                analyzer.mol,
                grids.grids_indexer,
                nspin,
                settings,
                plan_type=self._plan_type,
                aux_lambd=lambd,
                aug_beta=beta,
                interpolator_type=interpolator_type,
            )
            nldf_pert = PyscfNLDFGenerator.from_mol_and_settings(
                analyzer.mol,
                grids.grids_indexer,
                nspin,
                settings,
                plan_type=self._plan_type,
                aux_lambd=lambd,
                aug_beta=beta,
                interpolator_type=interpolator_type,
            )
            nldf_gen_nsp = PyscfNLDFGenerator.from_mol_and_settings(
                analyzer.mol,
                grids.grids_indexer,
                1,
                settings,
                plan_type=self._plan_type,
                aux_lambd=lambd,
                aug_beta=beta,
                interpolator_type=interpolator_type,
            )
            nldf_gen.interpolator.set_coords(grids.coords)
            nldf_pert.interpolator.set_coords(grids.coords)
            nldf_gen_nsp.interpolator.set_coords(grids.coords)
            feat2 = []
            feat_nsp = []
            for s in range(nspin):
                feat2.append(nldf_gen.get_features(rho_in=rho[s], spin=s))
                feat_nsp.append(nldf_gen_nsp.get_features(rho_in=nspin * rho[s]))
            feat2 = np.stack(feat2)
            if check_feat_equiv:
                for s in range(nspin):
                    assert_allclose(feat_nsp[s], feat_ref[s], atol=1e-7, rtol=1e-7)
                    assert_allclose(feat2[s], feat_ref[s], atol=1e-7, rtol=1e-7)
            all_occd_feat = occd_feat
            for my_key in ["U", "O"]:
                my_orb = {my_key: [0]}
                if nspin == 1:
                    occd_feat = all_occd_feat[my_key][0]
                    spin = 0
                else:
                    spin, occd_feat = all_occd_feat[my_key][0]
                e, vrho_tmp, vfeat = get_e_and_v(feat2, use_mean=False)
                de3 = np.sum(occd_feat * vfeat[spin])
                vrho1 = []
                for s in range(nspin):
                    vrho1.append(nldf_gen.get_potential(vfeat[s], spin=s))
                vrho1 = np.stack(vrho1)
                labels, coeffs, en_list, sep_spins = get_labels_and_coeffs(
                    my_orb, mo_coeff, mo_occ, mo_energy
                )
                if analyzer.dm.ndim == 3:
                    coeff_vector = coeffs[0][0] if len(coeffs[0]) > 0 else coeffs[1][0]
                else:
                    coeff_vector = coeffs[0]
                occd_dm = np.outer(coeff_vector, coeff_vector)
                occd_rho = get_full_rho(
                    dft.numint.NumInt(), analyzer.mol, occd_dm, grids, "MGGA"
                )
                de = np.sum(vrho1[spin] * occd_rho)
                if "U" in my_orb.keys():
                    delta = FD_DELTA
                else:
                    delta = -1 * FD_DELTA
                dmtmp = analyzer.dm.copy()
                if dmtmp.ndim == 3:
                    dmtmp[spin] += delta * occd_dm
                else:
                    dmtmp[:] += delta * occd_dm
                if analyzer.dm.ndim == 3:
                    ana_tmp = UHFAnalyzer(analyzer.mol, dmtmp, grids_level=inner_level)
                else:
                    ana_tmp = RHFAnalyzer(analyzer.mol, dmtmp, grids_level=inner_level)
                ana_tmp.grids = grids
                ana_tmp.perform_full_analysis()
                rho_pert = ana_tmp.get_rho_data()
                if rho_pert.ndim == 2:
                    rho_pert = np.ascontiguousarray(rho_pert[None, [0, 1, 2, 3, -1]])
                else:
                    rho_pert = np.ascontiguousarray(rho_pert[:, [0, 1, 2, 3, -1]])
                feat_pert = []
                for s in range(nspin):
                    feat_pert.append(nldf_pert.get_features(rho_in=rho_pert[s], spin=s))
                feat_pert = np.stack(feat_pert)
                e_pert = get_e_and_v(feat_pert, use_mean=False)[0]
                de2 = (e_pert - e) / delta
                de_tot = 0
                de2_tot = 0
                for i in range(feat2.shape[1]):
                    feat_tmp = feat2.copy()
                    feat_tmp[:, :i] = 0
                    feat_tmp[:, i + 1 :] = 0
                    vrho1 = []
                    _e, vrho_tmp, vfeat = get_e_and_v(feat_tmp)
                    for s in range(nspin):
                        vrho1.append(nldf_gen.get_potential(vfeat[s], spin=s))
                    vrho1 = np.stack(vrho1)
                    feat_tmp = feat_pert.copy()
                    feat_tmp[:, :i] = 0
                    feat_tmp[:, i + 1 :] = 0
                    _e_pert = get_e_and_v(feat_tmp)[0]
                    _de2 = (_e_pert - _e) / delta
                    _de = np.sum(vrho1[spin] * occd_rho)
                    de_tot += _de
                    de2_tot += _de2
                assert_allclose(de, de2, rtol=rtol, atol=atol)
                assert_allclose(de_tot, de2_tot, rtol=2 * rtol, atol=2 * atol)
                assert_allclose(de3, de, rtol=1e-7, atol=1e-7)

    def test_nldf_equivalence(self):
        mols = self.mols
        if self._plan_type == "gaussian":
            tols = [(5e-4, 5e-4), (2e-4, 5e-4), (1e-3, 1e-3), (5e-4, 5e-4)]
        else:
            # NOTE spline is a bit less precise for some reason
            tols = [(1e-3, 1e-3), (1e-3, 1e-3), (3e-3, 3e-3), (1e-3, 1e-3)]
        for mol, (atol, rtol) in zip(mols, tols):
            print(mol.atom)
            ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            ks.xc = "PBE"
            ks.conv_tol = 1e-12
            ks.kernel()
            coords = get_random_coords(ks)
            self._check_nldf_equivalence(
                mol,
                ks.make_rdm1(),
                coords,
                plan_type=self._plan_type,
                rtol=rtol,
                atol=atol,
            )

    def test_nldf_occ_derivs(self):
        tols = [(1e-4, 1e-4), (1e-4, 1e-4), (1e-3, 1e-3), (1e-4, 1e-4)]
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
            self._check_nldf_occ_derivs(analyzer, coords, rtol=rtol, atol=atol)

    def test_nldf_vxc(self):
        # NOTE atol could be 1e-5 for Gaussian interp
        atol = 1e-4
        rtol = 3e-4
        for mol in self.mols:
            print(mol.atom)
            ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            ks.xc = "PBE"
            ks.kernel()
            ref_grids = Grids(mol)
            ref_grids.level = 0
            ref_grids.build()
            dm = ks.make_rdm1()
            if dm.ndim == 3:
                analyzer = UHFAnalyzer.from_calc(ks)
            else:
                analyzer = RHFAnalyzer.from_calc(ks)
            self._check_nldf_vxc(
                analyzer,
                interpolator_type="onsite_spline",
                rtol=rtol,
                atol=atol,
            )
            self._check_nldf_vxc(
                analyzer,
                interpolator_type="onsite_direct",
                rtol=rtol,
                atol=atol,
                check_feat_equiv=False,
            )


class TestNLDFGaussian(_TestNLDFBase, unittest.TestCase):
    _plan_type = "gaussian"


class TestNLDFSpline(_TestNLDFBase, unittest.TestCase):
    _plan_type = "spline"


if __name__ == "__main__":
    unittest.main()
