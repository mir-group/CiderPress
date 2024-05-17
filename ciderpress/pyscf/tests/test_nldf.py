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


FD_DELTA = 1e-4


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
        l0_specs = ["se", "se_r2", "se_apr2", "se_ap", "se_ap2r2", "se_lapl"]
        l1_specs = ["se_rvec", "se_grad"]
        vj_specs = ["se", "se_ar2", "se_a2r4", "se_erf_rinv"]
        l1_dots = [(-1, 1), (0, 1), (0, 0), (1, 1)]
        theta_params = [1.0, 0.0, 0.03125]
        feat_params = [[2.0, 0.0, 0.04] for i in range(4)]
        feat_params[-1].append(2.0)
        cls.vi_settings = NLDFSettingsVI(
            "MGGA",
            theta_params,
            "one",
            l0_specs,
            l1_specs,
            l1_dots,
        )
        cls.vj_settings = NLDFSettingsVJ(
            "MGGA", theta_params, "one", vj_specs, feat_params
        )
        cls.vij_settings = NLDFSettingsVIJ(
            "MGGA",
            theta_params,
            "one",
            l0_specs,
            l1_specs,
            l1_dots,
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
            gto.M(atom="O 0 0 0; O 0 0 1.2", basis="def2-tzvp", spin=2),
            gto.M(atom="Ar", basis="def2-tzvp", spin=0),
            gto.M(atom=nh3str, spin=0, basis="def2-svp"),
        ]

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
        n0 = len(self.vi_settings.l0_feat_specs)
        ifeat[:, :n0] = _ifeat[:, :n0]
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
        rho = get_descriptors(analyzer, "l")
        for i, (j, k) in enumerate(self.vi_settings.l1_feat_dots):
            if j == -1:
                l1j = rho[:, 1:4]
            else:
                l1j = _ifeat[:, n0 + 3 * j : n0 + 3 * j + 3]
            if k == -1:
                l1k = rho[:, 1:4]
            else:
                l1k = _ifeat[:, n0 + 3 * k : n0 + 3 * k + 3]
            ifeat[:, n0 + i] = np.einsum("sxg,sxg->sg", l1j, l1k)
        ijfeat = np.append(jfeat, ifeat, axis=1)

        ifeat_pred = get_descriptors(
            analyzer,
            self.vi_settings,
            plan_type=plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        # TODO uncomment after fixing vi stability
        # import traceback as tb
        # errs = {}
        # for i in range(ifeat_pred.shape[1]):
        #     print(i)
        #     try:
        #         assert_allclose(ifeat_pred[:, i], ifeat[:, i], rtol=rtol, atol=atol)
        #     except AssertionError as e:
        #         errs[i] = ''.join(tb.format_exception(None, e, e.__traceback__))
        # if len(errs) > 0:
        #     estr = ''
        #     for i, err in errs.items():
        #         print(i, err)
        #         estr = estr + 'FEATURE {}\n{}\n'.format(i, err)
        #     raise AssertionError(estr)
        jfeat_pred = get_descriptors(
            analyzer,
            self.vj_settings,
            plan_type=plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        assert_allclose(jfeat_pred, jfeat, rtol=rtol, atol=atol)
        assert_almost_equal(jfeat_pred, jfeat, 3)
        # TODO uncomment after fixing vi stability
        # ijfeat_pred = get_descriptors(
        #     analyzer,
        #     self.vij_settings,
        #     plan_type=plan_type,
        #     aux_lambd=lambd,
        #     aug_beta=beta,
        # )
        # assert_allclose(ijfeat_pred, ijfeat, rtol=rtol, atol=atol)
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

    def _check_nldf_occ_derivs(
        self, analyzer, coords, rtol=1e-3, atol=1e-3, plan_type="spline"
    ):
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
            plan_type=plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        jfeat_ref = get_descriptors(
            analyzer,
            self.vj_settings,
            plan_type=plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        ijfeat_ref = get_descriptors(
            analyzer,
            self.vij_settings,
            plan_type=plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        kfeat_ref = get_descriptors(
            analyzer,
            self.vk_settings,
            plan_type=plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )

        rho_pred, occd_rho, rho_eigvals = get_descriptors(analyzer, "l", orbs=orbs)
        ifeat_pred, occd_ifeat, vi_eigvals = get_descriptors(
            analyzer,
            self.vi_settings,
            orbs=orbs,
            plan_type=plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        jfeat_pred, occd_jfeat, vj_eigvals = get_descriptors(
            analyzer,
            self.vj_settings,
            orbs=orbs,
            plan_type=plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        ijfeat_pred, occd_ijfeat, vij_eigvals = get_descriptors(
            analyzer,
            self.vij_settings,
            orbs=orbs,
            plan_type=plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        kfeat_pred, occd_kfeat, vk_eigvals = get_descriptors(
            analyzer,
            self.vk_settings,
            orbs=orbs,
            plan_type=plan_type,
            aux_lambd=lambd,
            aug_beta=beta,
        )
        assert_almost_equal(rho_pred, rho_ref, 12)
        # TODO uncomment after fixing vi stability
        # assert_almost_equal(ifeat_pred, ifeat_ref, 12)
        assert_almost_equal(jfeat_pred, jfeat_ref, 12)
        # TODO uncomment after fixing vi stability
        # assert_almost_equal(ijfeat_pred, ijfeat_ref, 12)
        assert_almost_equal(kfeat_pred, kfeat_ref, 12)

        for k, orblist in orbs.items():
            for iorb in orblist:
                dtmp = {k: [iorb]}
                print(k, iorb)
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
                if ifeat_pred.shape[0] == 2:
                    spin, occd_pred = occd_ifeat[k][iorb]
                else:
                    spin, occd_pred = 0, occd_ifeat[k][iorb]
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
                ifeat_pert = get_descriptors(
                    ana_tmp,
                    self.vi_settings,
                    plan_type=plan_type,
                    aux_lambd=lambd,
                    aug_beta=beta,
                )
                occd_ifeat_fd = (ifeat_pert - ifeat_ref)[spin] / delta
                for i in range(self.vi_settings.nfeat):
                    # TODO uncomment after fixing vi stability
                    continue
                    assert_allclose(
                        occd_pred[i], occd_ifeat_fd[i], rtol=rtol, atol=atol
                    )
                jfeat_pert = get_descriptors(
                    ana_tmp,
                    self.vj_settings,
                    plan_type=plan_type,
                    aux_lambd=lambd,
                    aug_beta=beta,
                )
                if jfeat_pert.shape[0] == 2:
                    spin, occd_pred = occd_jfeat[k][iorb]
                else:
                    spin, occd_pred = 0, occd_jfeat[k][iorb]
                occd_jfeat_fd = (jfeat_pert - jfeat_ref)[spin] / delta
                assert_allclose(occd_pred, occd_jfeat_fd, rtol=rtol, atol=atol)
                ijfeat_pert = get_descriptors(
                    ana_tmp,
                    self.vij_settings,
                    plan_type=plan_type,
                    aux_lambd=lambd,
                    aug_beta=beta,
                )
                if ijfeat_pert.shape[0] == 2:
                    spin, occd_pred = occd_ijfeat[k][iorb]
                else:
                    spin, occd_pred = 0, occd_ijfeat[k][iorb]
                (ijfeat_pert - ijfeat_ref)[spin] / delta
                # TODO uncomment after fixing vi stability
                # assert_allclose(occd_pred, occd_ijfeat_fd, rtol=rtol, atol=atol)
                kfeat_pert = get_descriptors(
                    ana_tmp,
                    self.vk_settings,
                    plan_type=plan_type,
                    aux_lambd=lambd,
                    aug_beta=beta,
                )
                if kfeat_pert.shape[0] == 2:
                    spin, occd_pred = occd_kfeat[k][iorb]
                else:
                    spin, occd_pred = 0, occd_kfeat[k][iorb]
                occd_kfeat_fd = (kfeat_pert - kfeat_ref)[spin] / delta
                assert_allclose(occd_pred[0], occd_kfeat_fd[0], rtol=rtol, atol=atol)

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
            sdesc, desc * lambd ** pows[None, :, None], rtol=rtol, atol=atol
        )

    def _check_uniform_scaling(
        self, analyzer, coords, lambd=1.3, rtol=1e-2, atol=1e-2, plan_type="spline"
    ):
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
            plan_type,
            rtol,
            atol,
        ]
        # TODO uncomment after fixing vi stability
        # self._check_uniform_scaling_helper(*args)
        args[0] = self.vj_settings
        self._check_uniform_scaling_helper(*args)
        args[0] = self.vij_settings
        # TODO uncomment after fixing vi stability
        # self._check_uniform_scaling_helper(*args)
        args[0] = self.vk_settings
        self._check_uniform_scaling_helper(*args)

    def test_uniform_scaling(self):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvppd")
        ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        ks.xc = "PBE"
        ks.conv_tol = 1e-12
        ks.kernel()
        coords = get_random_coords(ks)
        if mol.spin != 0:
            analyzer = UHFAnalyzer.from_calc(ks)
        else:
            analyzer = RHFAnalyzer.from_calc(ks)
        self._check_uniform_scaling(analyzer, coords, plan_type=self._plan_type)

    def _check_nldf_vxc(
        self,
        analyzer,
        rtol=1e-3,
        atol=1e-5,
        interpolator_type="onsite_spline",
        check_feat_equiv=True,
        feat_equiv_prec=7,
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

        lambd = 2.0
        beta = 2.0

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

            def get_e_and_v(feat):
                sum = (factor[:, None] * feat**2).sum(1)
                vrho = sum / (1 + sum)
                e = rho[:, 0] * vrho
                vfeat = rho[:, 0] * 1 / (1 + sum) ** 2
                vfeat = vfeat[:, None, :] * 2 * (factor[:, None] * feat)
                e = np.dot(e, grids.weights).sum()
                vrho *= grids.weights
                vfeat *= grids.weights
                return e, vrho, vfeat

            if nspin == 1:
                occd_feat = occd_feat["U"][0]
                spin = 0
            else:
                spin, occd_feat = occd_feat["U"][0]
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
                    assert_almost_equal(feat_nsp[s], feat_ref[s], feat_equiv_prec)
                    assert_almost_equal(feat2[s], feat_ref[s], feat_equiv_prec)
            e, vrho_tmp, vfeat = get_e_and_v(feat2)
            de3 = np.sum(occd_feat * vfeat[spin])
            vrho1 = []
            for s in range(nspin):
                vrho1.append(nldf_gen.get_potential(vfeat[s], spin=s))
            vrho1 = np.stack(vrho1)
            labels, coeffs, en_list, sep_spins = get_labels_and_coeffs(
                {"U": [0]}, mo_coeff, mo_occ, mo_energy
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
            if True:  # TODO implement for valence too
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
            # feat_pert = get_descriptors(
            #    ana_tmp, settings,
            #    plan_type=plan_type, aux_lambd=lambd, aug_beta=beta,
            # )
            feat_pert = []
            for s in range(nspin):
                feat_pert.append(nldf_pert.get_features(rho_in=rho_pert[s], spin=s))
            feat_pert = np.stack(feat_pert)
            e_pert = get_e_and_v(feat_pert)[0]
            de2 = (e_pert - e) / delta
            print("energies", e, de, de2, de3)
            de_tot = 0
            de2_tot = 0
            for i in range(feat2.shape[1]):
                feat_tmp = feat2.copy()
                feat_tmp[:, :i] = 0
                feat_tmp[:, i + 1 :] = 0
                vrho1 = []
                _e, vrho_tmp, vfeat = get_e_and_v(feat_tmp)
                _de3 = np.sum(occd_feat * vfeat[spin])
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
                print("energies sub", i, _e, _de, _de2, _de3)
            print("energies tot", de_tot, de2_tot)
            assert_allclose(de, de2, rtol=rtol, atol=1e-5)
            assert_allclose(de3, de, rtol=rtol, atol=1e-5)
            print()

    def test_nldf_equivalence(self):
        mols = self.mols
        tols = [(5e-4, 5e-4), (2e-4, 5e-4), (3e-4, 5e-4), (5e-4, 5e-4)]
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
        mols = self.mols
        tols = [(2e-3, 2e-3), (2e-3, 1e-2), (2e-3, 5e-3), (2e-3, 2e-3)]
        for mol, (atol, rtol) in zip(mols, tols):
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
            self._check_nldf_occ_derivs(
                analyzer, coords, plan_type=self._plan_type, rtol=rtol, atol=atol
            )

    def test_nldf_vxc(self):
        mols = self.mols
        tols = [(2e-3, 5e-3), (2e-3, 5e-3), (2e-3, 5e-3), (2e-3, 2e-3)]
        tols = [(1e-5, 5e-3), (1e-5, 5e-3), (1e-5, 5e-3), (1e-5, 2e-3)]
        for mol, (atol, rtol) in zip(mols, tols):
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
            feat_equiv_prec = 5 if mol.atom == "Ar" else 6
            self._check_nldf_vxc(
                analyzer,
                interpolator_type="onsite_spline",
                rtol=rtol,
                atol=atol,
                feat_equiv_prec=feat_equiv_prec,
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
