import unittest

import numpy as np
from numpy.testing import assert_allclose
from pyscf import dft, gto, scf
from pyscf.dft.gen_grid import Grids
from scipy.special import dawsn, erf, gamma

from ciderpress.dft.plans import SADMPlan, SDMXFullPlan, SDMXPlan
from ciderpress.dft.settings import (
    SADMSettings,
    SDMX1Settings,
    SDMXFullSettings,
    SDMXG1Settings,
    SDMXGSettings,
    SDMXSettings,
)
from ciderpress.external.sgx_tools import get_jk_densities, sgx_fit
from ciderpress.pyscf.analyzers import RHFAnalyzer, UHFAnalyzer
from ciderpress.pyscf.descriptors import get_descriptors, get_labels_and_coeffs
from ciderpress.pyscf.sdmx_slow import EXXSphGenerator, eval_conv_ao
from ciderpress.pyscf.tests.utils_for_test import (
    get_random_coords,
    get_rotated_coords,
    get_scaled_mol,
    rotate_molecule,
)

LDA_FACTOR = -3.0 / 4.0 * (3.0 / np.pi) ** (1.0 / 3)
FD_DELTA = 1e-4

np.random.seed(555)
RANDOM_WT = np.random.normal(size=100)


class _Grids:
    def __init__(self, coords, weights):
        self.coords = coords
        self.weights = weights


def ueg_proj(a, k):
    rta = np.sqrt(a)
    fac = 6 * 2**0.75 * a**0.75 * np.pi**0.25 / k**3
    p1 = k * np.sqrt(np.pi) / rta * np.exp(-k * k / (4 * a))
    p2 = np.pi * erf(k / (2 * rta))
    return fac * (p2 - p1)


def ueg_proj_coul(a, k):
    return (
        12
        * np.pi
        * (k**2 - 2 * a * np.sqrt(k**2 / a) * dawsn(0.5 * np.sqrt(k**2 / a)))
        / k**4
        * (2 * a / np.pi) ** 0.75
    )


class TestXEDSADM(unittest.TestCase):
    def _check_value_at_zero(self, mol):
        ks = dft.RKS(mol)
        ks.xc = "PBE"
        ks.kernel()
        settings = SADMSettings("exact")
        # All the following get the needed precision
        # plan = SADMPlan(settings, 1, 0.01, 1.8, 32, fit_metric='ovlp')
        # plan = SADMPlan(settings, 1, 0.01, 2.0, 28, fit_metric='ovlp')
        plan = SADMPlan(settings, 1, 0.01, 2.2, 26, fit_metric="ovlp")
        exx_gen = EXXSphGenerator(plan)
        dm = ks.make_rdm1()
        calc_tmp = sgx_fit(scf.RHF(mol))
        calc_tmp.build()
        grids = _Grids(np.zeros((1, 3)), np.ones(1))
        calc_tmp.with_df.grids = grids
        ej, ek = get_jk_densities(calc_tmp.with_df, dm)
        ek_pred = exx_gen(dm, mol, grids.coords)
        assert_allclose(ek_pred.item(), 0.5 * ek.item(), rtol=1e-6, atol=1e-7)

    def _check_ueg(self, rho, fit_metric="ovlp"):
        k = (3 * np.pi**2 * rho) ** (1.0 / 3)
        settings = SADMSettings("exact")
        if fit_metric == "coul":
            plan = SADMPlan(settings, 1, 0.00001, 1.6, 40, fit_metric=fit_metric)
        else:
            plan = SADMPlan(settings, 1, 0.00001, 1.6, 40, fit_metric=fit_metric)
        if fit_metric == "ovlp":
            projections = ueg_proj(plan.alphas, k)
        else:
            projections = ueg_proj_coul(plan.alphas, k)
        exx_true = LDA_FACTOR * rho ** (1.0 / 3)
        tmp = plan.fit_matrix.dot(projections)
        exx_pred = -0.25 * tmp.dot(tmp) * rho
        assert_allclose(exx_pred, exx_true, rtol=1e-2, atol=1e-4)
        assert_allclose(
            exx_pred,
            np.array(settings.ueg_vector(rho)) / rho,
            rtol=1e-2,
            atol=1e-4,
        )

        settings = SADMSettings("smooth")
        # fit_metric is ignored here
        plan = SADMPlan(settings, 1, 0.01, 2.0, 28, fit_metric=fit_metric)
        projections = (
            ueg_proj(plan.alphas, k) / (np.pi / (2 * plan.alphas)) ** -0.75
            - ueg_proj(2 * plan.alphas, k) / (np.pi / (4 * plan.alphas)) ** -0.75
        )
        projections *= plan.alpha_norms
        exx_true = -0.8751999726949563 * rho ** (1.0 / 3)
        tmp = plan.fit_matrix.dot(projections)
        exx_pred = -0.25 * tmp.dot(tmp) * rho
        assert_allclose(exx_pred, exx_true, rtol=1e-5, atol=1e-6)
        assert_allclose(
            exx_pred,
            np.array(settings.ueg_vector(rho)) / rho,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_value_at_zero(self):
        self._check_value_at_zero(gto.M(atom="He", basis="def2-tzvppd"))
        self._check_value_at_zero(gto.M(atom="Ar", basis="def2-tzvppd"))
        self._check_value_at_zero(gto.M(atom="Kr", basis="def2-tzvppd"))

    def _check_single_gaussian(self, mode, alpha):
        basis = {"He": [(0, [alpha, 1.0])]}
        mol = gto.M(atom="He", basis=basis)
        dm = np.ones((1, 1))
        settings = SADMSettings(mode)
        plan = SADMPlan(settings, 1, 0.01, 2.2, 26, fit_metric="ovlp")
        exx_gen = EXXSphGenerator(plan)
        coords = np.zeros((1, 3))
        ek_pred = exx_gen(dm, mol, coords)
        if mode == "exact":
            ek_ref = -2 * (alpha / np.pi) ** 2
        else:
            ek_ref = -2.0 / 49 * np.sqrt(alpha / np.pi)
            ek_ref *= (69 - 48 * np.sqrt(2)) * (8 + 9 * np.sqrt(2))
            ek_ref *= (2 * alpha / np.pi) ** 1.5
        assert_allclose(ek_pred, ek_ref, rtol=1e-5, atol=1e-6)

    def test_single_gaussian(self):
        for mode in ["exact", "smooth"]:
            for alpha in [0.02, 0.4, 1.0, 4.6, 20.0]:
                self._check_single_gaussian(mode, alpha)

    def test_ueg(self):
        for metric in ["ovlp", "coul"]:
            self._check_ueg(1.0, fit_metric=metric)
            self._check_ueg(0.34, fit_metric=metric)
            self._check_ueg(2.78, fit_metric=metric)
            self._check_ueg(0.05, fit_metric=metric)
            self._check_ueg(25, fit_metric=metric)

    def _check_occ_derivs(self, analyzer, coords, mode="exact", rtol=1e-4, atol=1e-4):
        mo_occ = analyzer.mo_occ
        mo_coeff = analyzer.mo_coeff
        mo_energy = analyzer.mo_energy
        grids = _Grids(coords, np.ones(coords.shape[0]))
        grids.mol = analyzer.mol
        grids.non0tab = None
        grids.cutoff = 0
        inner_level = 3
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        orbs = {"U": [2, 0], "O": [0, 1]}
        plan = SADMPlan(SADMSettings(mode), 1, 0.003, 2.0, 30, fit_metric="ovlp")
        exxgen = EXXSphGenerator(plan)
        desc0 = get_descriptors(analyzer, plan.settings, orbs=None)
        settings = plan.settings
        desc, ddesc, eigvals = get_descriptors(analyzer, plan.settings, orbs=orbs)
        assert_allclose(desc, desc0)

        feat_ref = exxgen.get_features(analyzer.dm, analyzer.mol, analyzer.grids.coords)
        assert_allclose(desc[0], feat_ref, rtol=1e-3, atol=1e-3)

        for k, orblist in orbs.items():
            for iorb in orblist:
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
                if desc.shape[0] == 2:
                    spin, occd_pred = ddesc[k][iorb]
                else:
                    spin, occd_pred = 0, ddesc[k][iorb]
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
                desc_pert = get_descriptors(ana_tmp, settings)
                occd_fd = (desc_pert - desc) / delta
                assert_allclose(occd_pred, occd_fd[0], rtol=rtol, atol=atol)

    def test_occ_derivs(self):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvppd")
        ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        ks.xc = "PBE"
        ks.kernel()
        ref_grids = Grids(mol)
        ref_grids.level = 0
        ref_grids.build()
        dm = ks.make_rdm1()
        if mol.spin == 0:
            rho = ks._numint.get_rho(mol, dm, ref_grids)
            cond = rho > 1e-2
        else:
            rhoa = ks._numint.get_rho(mol, dm[0], ref_grids)
            rhob = ks._numint.get_rho(mol, dm[1], ref_grids)
            cond = np.logical_and(rhoa > 1e-2, rhob > 1e-2)
        coords = ref_grids.coords[cond]
        inds = np.arange(coords.shape[0])
        np.random.seed(16)
        np.random.shuffle(inds)
        coords = np.ascontiguousarray(coords[inds[:50]])
        if dm.ndim == 3:
            analyzer = UHFAnalyzer.from_calc(ks)
        else:
            analyzer = RHFAnalyzer.from_calc(ks)
        self._check_occ_derivs(analyzer, coords, mode="exact")
        self._check_occ_derivs(analyzer, coords, mode="smooth")

    def _check_potential(self, mode="exact"):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvppd")
        ks = dft.RKS(mol)
        ks.xc = "PBE"
        ks.kernel()
        settings = SADMSettings(mode)
        plan = SADMPlan(settings, 1, 0.01, 2.0, 28, fit_metric="ovlp")
        exx_gen = EXXSphGenerator(plan)
        dm = ks.make_rdm1()
        dm = dm
        calc_tmp = sgx_fit(scf.RHF(mol))
        calc_tmp.grids = dft.gen_grid.Grids(mol)
        calc_tmp.grids.build()
        calc_tmp.build()
        vxc = np.zeros_like(dm)
        grids = calc_tmp.grids
        potential = grids.weights.copy()[None, :]
        exx_gen.get_features(dm, mol, grids.coords)
        exx_gen.get_vxc_(vxc, potential)
        np.random.seed(13)
        dm_pert = np.random.normal(size=dm.shape)
        dm_pert = dm_pert + dm_pert.T
        dm_pert /= np.abs(dm_pert).sum()
        delta = 1e-4
        dmm = dm - 0.5 * delta * dm_pert
        dmp = dm + 0.5 * delta * dm_pert
        ekp = exx_gen(dmp, mol, grids.coords)
        ekm = exx_gen(dmm, mol, grids.coords)
        energy_p = np.dot(ekp, grids.weights)
        energy_m = np.dot(ekm, grids.weights)
        de_ref = (energy_p - energy_m) / delta
        de_pred = (dm_pert * (vxc + vxc.T)).sum()
        assert_allclose(de_pred, de_ref, rtol=1e-8, atol=1e-8)

    def test_potential(self):
        self._check_potential("exact")
        self._check_potential("smooth")


class _TestSDMXParams:

    ueg_rtol = None
    ueg_atol = None
    gaussian_rtol = None
    gaussian_atol = None

    def _get_single_gaussian_ref(self, alpha):
        raise NotImplementedError

    def _get_plan(self, pows=None):
        raise NotImplementedError

    def _get_ueg_plan(self):
        raise NotImplementedError


class _TestSDMXBase(_TestSDMXParams):
    def _check_single_gaussian(self, alpha):
        basis = {"He": [(0, [alpha, 1.0])]}
        mol = gto.M(atom="He", basis=basis)
        dm = np.ones((1, 1))
        plan = self._get_plan()
        exx_gen = EXXSphGenerator(plan)
        coords = np.zeros((1, 3))
        ek_pred = exx_gen(dm, mol, coords)
        ek_ref = self._get_single_gaussian_ref(alpha)
        assert_allclose(
            ek_pred[:, 0], ek_ref, rtol=self.gaussian_rtol, atol=self.gaussian_atol
        )
        # exit()

    def test_single_gaussian(self):
        for alpha in [0.02, 0.4, 1.0, 4.6, 20.0]:
            self._check_single_gaussian(alpha)

    def _check_ueg(self, rho):
        k = (3 * np.pi**2 * rho) ** (1.0 / 3)
        plan = self._get_ueg_plan()
        projections = (
            ueg_proj(plan.alphas, k) / (np.pi / (2 * plan.alphas)) ** -0.75
            - ueg_proj(2 * plan.alphas, k) / (np.pi / (4 * plan.alphas)) ** -0.75
        )
        projections *= plan.alpha_norms
        tmp = np.stack(
            [plan.fit_matrices[i].dot(projections) for i in range(plan.settings.nfeat)]
        )
        exx_pred = -0.25 * np.einsum("xb,xb->x", tmp, tmp) * rho
        exx_ref = np.array(plan.settings.ueg_vector(rho)) / rho
        print(rho)
        assert_allclose(
            exx_pred,
            exx_ref,
            rtol=self.ueg_rtol,
            atol=self.ueg_atol,
        )

    def test_ueg(self):
        self._check_ueg(1.00)
        self._check_ueg(0.34)
        self._check_ueg(2.78)
        self._check_ueg(0.05)
        self._check_ueg(25.0)

    def _check_occ_derivs(self, analyzer, coords, rtol=1e-4, atol=1e-4):
        mo_occ = analyzer.mo_occ
        mo_coeff = analyzer.mo_coeff
        mo_energy = analyzer.mo_energy
        grids = _Grids(coords, np.ones(coords.shape[0]))
        grids.mol = analyzer.mol
        grids.non0tab = None
        grids.cutoff = 0
        inner_level = 3
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        orbs = {"U": [2, 0], "O": [0, 1]}
        plan = self._get_plan()
        exxgen = EXXSphGenerator(plan)
        desc0 = get_descriptors(analyzer, plan.settings, orbs=None)
        settings = plan.settings
        desc, ddesc, eigvals = get_descriptors(analyzer, plan.settings, orbs=orbs)
        assert_allclose(desc, desc0)

        feat_ref = exxgen.get_features(analyzer.dm, analyzer.mol, analyzer.grids.coords)
        assert_allclose(desc[0], feat_ref, rtol=1e-3, atol=1e-3)

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
                if desc.shape[0] == 2:
                    spin, occd_pred = ddesc[k][iorb]
                else:
                    spin, occd_pred = 0, ddesc[k][iorb]
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
                desc_pert = get_descriptors(ana_tmp, settings)
                occd_fd = (desc_pert - desc) / delta
                assert_allclose(occd_pred, occd_fd[0], rtol=rtol, atol=atol)

    def test_occ_derivs(self):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvppd")
        ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        ks.xc = "PBE"
        ks.conv_tol = 1e-12
        ks.kernel()
        dm = ks.make_rdm1()
        coords = get_random_coords(ks)
        if dm.ndim == 3:
            analyzer = UHFAnalyzer.from_calc(ks)
        else:
            analyzer = RHFAnalyzer.from_calc(ks)
        self._check_occ_derivs(analyzer, coords)

    def _check_potential(self, pows, mol, dm):
        plan = self._get_plan(pows=pows)
        exx_gen = EXXSphGenerator(plan)
        calc_tmp = sgx_fit(scf.RHF(mol))
        calc_tmp.grids = dft.gen_grid.Grids(mol)
        calc_tmp.grids.build()
        calc_tmp.build()
        vxc = np.zeros_like(dm)
        grids = calc_tmp.grids
        nfeat = exx_gen.plan.settings.nfeat
        potential = RANDOM_WT[:nfeat, None] * grids.weights.copy()
        # potential = np.stack([potential for _ in range(plan.settings.nfeat)])
        mid = 10000
        exx_gen.get_features(dm, mol, grids.coords[:mid])
        exx_gen.get_vxc_(vxc, potential[..., :mid])
        exx_gen.get_features(dm, mol, grids.coords[mid:])
        exx_gen.get_vxc_(vxc, potential[..., mid:])
        np.random.seed(13)
        dm_pert = np.random.normal(size=dm.shape)
        dm_pert = dm_pert + dm_pert.T
        dm_pert /= np.abs(dm_pert).sum()
        delta = 1e-4
        dmm = dm - 0.5 * delta * dm_pert
        dmp = dm + 0.5 * delta * dm_pert
        ekp = exx_gen(dmp, mol, grids.coords)
        ekm = exx_gen(dmm, mol, grids.coords)
        energy_p = np.sum(ekp * potential)
        energy_m = np.sum(ekm * potential)
        de_ref = (energy_p - energy_m) / delta
        de_pred = (dm_pert * (vxc + vxc.T)).sum()
        assert_allclose(de_pred, de_ref.sum(), rtol=1e-8, atol=1e-8)

    def test_potential(self):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvppd")
        ks = dft.RKS(mol)
        ks.conv_tol = 1e-12
        ks.xc = "PBE"
        ks.kernel()
        dm = ks.make_rdm1()
        self._check_potential([0, 1, 2], mol, dm)
        self._check_potential([0], mol, dm)
        self._check_potential([1], mol, dm)
        self._check_potential([2], mol, dm)

    def _check_uniform_scaling(self, analyzer, coords, lambd=1.3, rtol=1e-4, atol=1e-4):
        grids = _Grids(coords, np.ones(coords.shape[0]))
        grids.mol = analyzer.mol
        grids.non0tab = None
        grids.cutoff = 0
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        plan = self._get_plan()
        desc = get_descriptors(analyzer, plan.settings, orbs=None)
        grids = _Grids(coords / lambd, np.ones(coords.shape[0]))
        grids.mol = analyzer.mol
        grids.non0tab = None
        grids.cutoff = 0
        analyzer.grids = grids
        analyzer.mol = get_scaled_mol(analyzer.mol, lambd)
        analyzer.perform_full_analysis()
        sdesc = get_descriptors(analyzer, plan.settings, orbs=None)
        pows = np.array(plan.settings.get_feat_usps())
        assert_allclose(
            sdesc, desc * lambd ** pows[None, :, None], rtol=rtol, atol=atol
        )

    def test_uniform_scaling(self):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvppd")
        ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        ks.xc = "PBE"
        ks.conv_tol = 1e-12
        ks.kernel()
        dm = ks.make_rdm1()
        coords = get_random_coords(ks)
        if dm.ndim == 3:
            analyzer = UHFAnalyzer.from_calc(ks)
        else:
            analyzer = RHFAnalyzer.from_calc(ks)
        self._check_uniform_scaling(analyzer, coords)

    def _check_rotation_invariance(self, mol, coords):
        rot_coords = get_rotated_coords(coords, 0.6 * np.pi, axis="x")
        rot_mol = rotate_molecule(mol, 0.6 * np.pi, axis="x")
        plan = self._get_plan()

        ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        ks.xc = "PBE"
        ks.conv_tol = 1e-12
        ks.kernel()
        if mol.spin == 0:
            analyzer = RHFAnalyzer.from_calc(ks)
        else:
            analyzer = UHFAnalyzer.from_calc(ks)
        grids = _Grids(coords, np.ones(coords.shape[0]))
        grids.mol = analyzer.mol
        grids.non0tab = None
        grids.cutoff = 0
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        desc = get_descriptors(analyzer, plan.settings, orbs=None)

        ks = dft.RKS(rot_mol) if rot_mol.spin == 0 else dft.UKS(rot_mol)
        ks.xc = "PBE"
        ks.conv_tol = 1e-12
        ks.kernel()
        if mol.spin == 0:
            analyzer = RHFAnalyzer.from_calc(ks)
        else:
            analyzer = UHFAnalyzer.from_calc(ks)
        grids = _Grids(rot_coords, np.ones(coords.shape[0]))
        grids.mol = analyzer.mol
        grids.non0tab = None
        grids.cutoff = 0
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        rot_desc = get_descriptors(analyzer, plan.settings, orbs=None)

        assert_allclose(rot_desc, desc, rtol=1e-4, atol=1e-4)

    def test_rotation_invariance(self):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvppd")
        ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        ks.xc = "PBE"
        ks.conv_tol = 1e-12
        ks.kernel()
        coords = get_random_coords(ks)
        self._check_rotation_invariance(mol, coords)

    def test_eval_conv_gto(self):
        plan = self._get_plan()
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvppd")
        grids = Grids(mol)
        grids.level = 0
        grids.build()
        coords = grids.coords
        cao1 = eval_conv_ao(plan, mol, coords, deriv=1)
        cao_val = cao1[::4]
        cao0 = eval_conv_ao(plan, mol, coords, deriv=0)
        assert_allclose(cao_val, cao0, rtol=1e-13, atol=1e-13)
        delta = 1e-6
        for v in range(3):
            grad_test = cao1[v + 1 :: 4]
            coords[:, v] += 0.5 * delta
            grad_ref = eval_conv_ao(plan, mol, coords, deriv=0)
            coords[:, v] -= delta
            grad_ref -= eval_conv_ao(plan, mol, coords, deriv=0)
            coords[:, v] += 0.5 * delta
            grad_ref /= delta
            assert_allclose(grad_test, grad_ref, rtol=1e-6, atol=1e-8)


class TestSDMX(unittest.TestCase, _TestSDMXBase):
    ueg_rtol = 1e-5
    ueg_atol = 1e-6
    gaussian_rtol = 5e-5
    gaussian_atol = 1e-5

    def _get_single_gaussian_ref(self, alpha):
        refvals = []
        # r^0
        ek_ref = -2.0 / 49 * np.sqrt(alpha / np.pi)
        ek_ref *= (69 - 48 * np.sqrt(2)) * (8 + 9 * np.sqrt(2))
        ek_ref *= (2 * alpha / np.pi) ** 1.5
        refvals.append(ek_ref)
        # r^-1
        ek_ref = (8 + 9 * np.sqrt(2)) / (49 * np.pi)
        ek_ref *= (
            (1 + np.sqrt(8)) * np.pi**1.5
            + 256 * np.pi**2 / gamma(0.25) ** 2
            - 256 * gamma(1.25) ** 2
        )
        ek_ref *= -1 * (2 * alpha / np.pi) ** 1.5
        refvals.append(ek_ref)
        # r^-2
        ek_ref = -0.25 * (8 + 9 * np.sqrt(2)) / (49 * np.pi) * alpha
        ek_ref *= (
            3 * (1 + np.sqrt(32)) * np.pi**1.5
            - 384 * np.pi**2 / gamma(0.25) ** 2
            + 256 * gamma(1.25) ** 2
        )
        ek_ref *= (2 * alpha / np.pi) ** 1.5
        refvals.append(ek_ref)
        return refvals

    def _get_plan(self, pows=None):
        if pows is None:
            pows = [1, 0, 2]
        settings = SDMXSettings(pows)
        plan = SDMXPlan(settings, 1, 0.002, 2.0, 30)
        return plan

    def _get_ueg_plan(self):
        settings = SDMXSettings([0, 1, 2])
        plan = SDMXPlan(settings, 1, 0.002, 1.8, 32)
        return plan


class TestSDMXG(unittest.TestCase, _TestSDMXBase):
    ueg_rtol = 1e-4
    ueg_atol = 1e-5
    gaussian_rtol = 5e-5
    gaussian_atol = 1e-5

    def _get_single_gaussian_ref(self, alpha):
        refvals = []
        fac = (2 * alpha / np.pi) ** 1.5
        # r^-1
        ek_ref = -2.0 / 49 * np.sqrt(alpha / np.pi)
        ek_ref *= (69 - 48 * np.sqrt(2)) * (8 + 9 * np.sqrt(2))
        ek_ref *= fac
        refvals.append(ek_ref)
        # r^0
        ek_ref = (8 + 9 * np.sqrt(2)) / (49 * np.pi)
        ek_ref *= (
            (1 + np.sqrt(8)) * np.pi**1.5
            + 256 * np.pi**2 / gamma(0.25) ** 2
            - 256 * gamma(1.25) ** 2
        )
        ek_ref *= -1 * fac
        refvals.append(ek_ref)
        # r^-2
        ek_ref = -0.25 * (8 + 9 * np.sqrt(2)) / (49 * np.pi) * alpha
        ek_ref *= (
            3 * (1 + np.sqrt(32)) * np.pi**1.5
            - 384 * np.pi**2 / gamma(0.25) ** 2
            + 256 * gamma(1.25) ** 2
        )
        ek_ref *= fac
        refvals.append(ek_ref)
        # grad r^-1
        ek_ref = -3.0 / 49 * np.sqrt(alpha / np.pi)
        ek_ref *= (-2171 + 1536 * np.sqrt(2)) * (8 + 9 * np.sqrt(2))
        ek_ref *= fac
        refvals.append(ek_ref)
        # grad r^0
        ek_ref = -9.0 / (4 * 196 * np.pi) * (8 + 9 * np.sqrt(2))
        ek_ref *= (
            5 * (1 + np.sqrt(8)) * np.pi**1.5
            - 22528 * np.pi**2 / gamma(0.25) ** 2
            + 20480 * gamma(1.25) ** 2
        )
        ek_ref *= fac
        refvals.append(ek_ref)
        # grad r^-2
        ek_ref = -3 * alpha * (8 + 9 * np.sqrt(2)) / (4 * 784 * np.pi)
        ek_ref *= (
            9 * (1 + 4 * np.sqrt(2)) * np.pi**1.5
            + 98304 * np.pi**2 / gamma(0.25) ** 2
            - 90112 * gamma(1.25) ** 2
        )
        ek_ref *= fac
        refvals.append(ek_ref)
        return refvals

    def _get_plan(self, pows=None):
        if pows is None:
            pows = [1, 0, 2]
        settings = SDMXGSettings(pows, len(pows))
        plan = SDMXPlan(settings, 1, 0.002, 2.0, 30)
        return plan

    def _get_ueg_plan(self):
        settings = SDMXGSettings([0, 1, 2], 3)
        plan = SDMXPlan(settings, 1, 0.002, 1.8, 32)
        return plan


class TestSDMX1(unittest.TestCase, _TestSDMXBase):
    ueg_rtol = 1e-4
    ueg_atol = 1e-5
    gaussian_rtol = 5e-5
    gaussian_atol = 1e-5

    def _get_single_gaussian_ref(self, alpha):
        refvals = []
        fac = (2 * alpha / np.pi) ** 1.5
        # r^-1
        ek_ref = -2.0 / 49 * np.sqrt(alpha / np.pi)
        ek_ref *= (69 - 48 * np.sqrt(2)) * (8 + 9 * np.sqrt(2))
        ek_ref *= fac
        refvals.append(ek_ref)
        # r^0
        ek_ref = (8 + 9 * np.sqrt(2)) / (49 * np.pi)
        ek_ref *= (
            (1 + np.sqrt(8)) * np.pi**1.5
            + 256 * np.pi**2 / gamma(0.25) ** 2
            - 256 * gamma(1.25) ** 2
        )
        ek_ref *= -1 * fac
        refvals.append(ek_ref)
        # r^-2
        ek_ref = -0.25 * (8 + 9 * np.sqrt(2)) / (49 * np.pi) * alpha
        ek_ref *= (
            3 * (1 + np.sqrt(32)) * np.pi**1.5
            - 384 * np.pi**2 / gamma(0.25) ** 2
            + 256 * gamma(1.25) ** 2
        )
        ek_ref *= fac
        refvals.append(ek_ref)
        refvals.append(0)
        refvals.append(0)
        refvals.append(0)
        return refvals

    def _get_plan(self, pows=None):
        if pows is None:
            pows = [1, 0, 2]
        settings = SDMX1Settings(pows, len(pows))
        plan = SDMXPlan(settings, 1, 0.002, 2.0, 30)
        return plan

    def _get_ueg_plan(self):
        settings = SDMX1Settings([0, 1, 2], 3)
        plan = SDMXPlan(settings, 1, 0.002, 1.8, 32)
        return plan

    def _check_ueg(self, rho):
        k = (3 * np.pi**2 * rho) ** (1.0 / 3)
        plan = self._get_ueg_plan()
        projections = (
            ueg_proj(plan.alphas, k) / (np.pi / (2 * plan.alphas)) ** -0.75
            - ueg_proj(2 * plan.alphas, k) / (np.pi / (4 * plan.alphas)) ** -0.75
        )
        projections *= plan.alpha_norms
        tmp = np.stack(
            [plan.fit_matrices[i].dot(projections) for i in range(plan.settings.nfeat)]
        )
        tmp[3:] = 0
        exx_pred = -0.25 * np.einsum("xb,xb->x", tmp, tmp) * rho
        exx_ref = np.array(plan.settings.ueg_vector(rho)) / rho
        assert_allclose(
            exx_pred,
            exx_ref,
            rtol=self.ueg_rtol,
            atol=self.ueg_atol,
        )


class TestSDMXG1(unittest.TestCase, _TestSDMXBase):
    ueg_rtol = 1e-4
    ueg_atol = 1e-5
    gaussian_rtol = 5e-5
    gaussian_atol = 5e-6

    def _get_single_gaussian_ref(self, alpha):
        return TestSDMXG._get_single_gaussian_ref(self, alpha) + [0] * 3

    def _get_plan(self, pows=None):
        if pows is None:
            pows = [1, 0, 2]
        settings = SDMXG1Settings(pows, len(pows), len(pows))
        plan = SDMXPlan(settings, 1, 0.002, 2.0, 30)
        return plan

    def _get_ueg_plan(self):
        settings = SDMXG1Settings([0, 1, 2], 3, 3)
        plan = SDMXPlan(settings, 1, 0.002, 1.8, 32)
        return plan

    def _check_ueg(self, rho):
        k = (3 * np.pi**2 * rho) ** (1.0 / 3)
        plan = self._get_ueg_plan()
        projections = (
            ueg_proj(plan.alphas, k) / (np.pi / (2 * plan.alphas)) ** -0.75
            - ueg_proj(2 * plan.alphas, k) / (np.pi / (4 * plan.alphas)) ** -0.75
        )
        projections *= plan.alpha_norms
        tmp = np.stack(
            [plan.fit_matrices[i].dot(projections) for i in range(plan.settings.nfeat)]
        )
        tmp[6:] = 0
        exx_pred = -0.25 * np.einsum("xb,xb->x", tmp, tmp) * rho
        exx_ref = np.array(plan.settings.ueg_vector(rho)) / rho
        assert_allclose(
            exx_pred,
            exx_ref,
            rtol=self.ueg_rtol,
            atol=self.ueg_atol,
        )


class TestSDMXFull(unittest.TestCase, _TestSDMXBase):
    ueg_rtol = 1e-4
    ueg_atol = 1e-5
    gaussian_rtol = 5e-5
    gaussian_atol = 5e-6

    def _get_single_gaussian_ref(self, alpha):
        rta = np.sqrt(alpha)
        (np.sqrt(2) + np.sqrt(0.5)) / 2
        (2 + 0.5) / 2
        l0vals = [
            -0.43194485,
            -0.53353024 * rta,
            -1.11785128 * alpha,
            -1.34126381,
            -0.88212052 * rta,
            -0.69183874 * alpha,
            -0.39261913,
            # -0.39261913,
            -0.49576391 * rta,
            # -0.51194593 * rta,
            -1.07205970 * alpha,
            # -1.21179111 * alpha,
            (-0.94577970 + -1.34126381) / 2,
            (-0.62645876 + -0.88212052) / 2 * rta,
            # (-0.62645876 + num1 * -0.88212052) / 2 * rta,
            (-0.50202339 + -0.69183874) / 2 * alpha,
            # (-0.50202339 + num2 * -0.69183874) / 2 * alpha,
            # -0.20272,
            # -0.303078 * rta,
            # -0.819554 * alpha,
            # -0.353794,
            # -0.239756 * rta,
            # -0.205664 * alpha,
        ]
        l0vals = np.array(l0vals) * (2 * alpha / np.pi) ** 1.5
        return l0vals.tolist() + [0] * 12

    def _get_plan(self, pows=None):
        if pows is None:
            pows = [0, 1, 2]
        npow = len(pows)
        settings = {
            1.00: (pows, [npow, npow, npow, npow]),
            2.00: (pows, [npow, npow, npow, npow]),
        }
        settings = SDMXFullSettings(settings_dict=settings)
        plan = SDMXFullPlan(settings, 1, 0.0005, 1.8, 33)
        return plan

    def _get_ueg_plan(self, pows=None):
        if pows is None:
            pows = [0, 1, 2]
        npow = len(pows)
        settings = {
            1.00: (pows, [npow, npow, npow, npow]),
            1.50: (pows, [npow, npow, npow, npow]),
            2.00: (pows, [npow, npow, npow, npow]),
        }
        settings = SDMXFullSettings(settings_dict=settings)
        plan = SDMXFullPlan(settings, 1, 0.0005, 1.6, 45)
        return plan

    def _check_ueg(self, rho):
        k = (3 * np.pi**2 * rho) ** (1.0 / 3)
        plan = self._get_ueg_plan()
        projections = (
            ueg_proj(plan.alphas, k) / (np.pi / (2 * plan.alphas)) ** -0.75
            - ueg_proj(2 * plan.alphas, k) / (np.pi / (4 * plan.alphas)) ** -0.75
        )
        projections *= plan.alpha_norms
        tmp = np.stack(
            [plan.fit_matrices[i].dot(projections) for i in range(plan.settings.nfeat)]
        )
        tmp[18:] = 0
        exx_pred = -0.25 * np.einsum("xb,xb->x", tmp, tmp) * rho
        exx_ref = np.array(plan.settings.ueg_vector(rho)) / rho
        assert_allclose(
            exx_pred,
            exx_ref,
            rtol=self.ueg_rtol,
            atol=self.ueg_atol,
        )


if __name__ == "__main__":
    unittest.main()
