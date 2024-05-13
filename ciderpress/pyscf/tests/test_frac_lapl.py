import ctypes
import time
import unittest

import numpy as np
from numpy.testing import assert_allclose
from pyscf import dft, gto, lib
from pyscf.dft.gen_grid import Grids
from pyscf.dft.numint import _dot_ao_dm, eval_ao
from scipy.special import erf, gamma

from ciderpress.dft.plans import FracLaplPlan
from ciderpress.dft.settings import FracLaplSettings
from ciderpress.dft.tests.utils_for_test import (
    get_random_coords,
    get_rotated_coords,
    get_scaled_mol,
    rotate_molecule,
)
from ciderpress.pyscf.analyzers import RHFAnalyzer, UHFAnalyzer
from ciderpress.pyscf.descriptors import get_descriptors, get_labels_and_coeffs
from ciderpress.pyscf.frac_lapl import (
    NBINS,
    FLNumInt,
    _odp_dot_sparse_,
    _set_flapl,
    eval_flapl_gto,
    eval_kao,
    libcider,
)


def get_e_and_v(feat):
    energy = (feat * feat / (1 + feat * feat)).sum(axis=1)
    potential = 2 * feat / (1 + feat * feat) ** 2
    return energy, potential


def ueg_proj(a, k):
    rta = np.sqrt(a)
    fac = 6 * 2**0.75 * a**0.75 * np.pi**0.25 / k**3
    p1 = k * np.sqrt(np.pi) / rta * np.exp(-k * k / (4 * a))
    p2 = np.pi * erf(k / (2 * rta))
    return fac * (p2 - p1)


def ueg_vals(k, r):
    return 3 * (np.sin(k * r) - k * r * np.cos(k * r) + 1e-24) / (k * r + 1e-24) ** 3


def prefac_1f1(s, d):
    return 4**s * gamma(d / 2.0 + s) / np.pi ** (d / 2.0) / np.abs(gamma(-s))


def get_flapl(grids, grids_cen, ao, ao_cen, s=0.5):
    if s > 0:
        my_ao = ao_cen[:, None] - ao[None, :].copy()
    else:
        my_ao = ao[:].copy()
    d = 3
    coord = grids.coords[None, :, :] - grids_cen.coords[:, None, :]
    dist = np.linalg.norm(coord, axis=-1) ** (d + 2 * s)
    invdist = 1 / (dist + 1e-6)
    return prefac_1f1(s, d) * np.dot(my_ao * invdist, grids.weights)


def get_flapl2(grids, grids_cen, dm_rg, dm_r, s=0.5):
    if s > 0:
        my_ao = dm_r[:, None] - dm_rg.copy()
    else:
        my_ao = dm_rg[:].copy()
    d = 3
    coord = grids.coords[None, :, :] - grids_cen.coords[:, None, :]
    dist = np.linalg.norm(coord, axis=-1) ** (d + 2 * s)
    invdist = 1 / (dist + 1e-6)
    return prefac_1f1(s, d) * np.dot(my_ao * invdist, grids.weights)


DELTA = 1e-6
FD_DELTA = 1e-4


class _Grids:
    def __init__(self, mol, coords, weights=None):
        self.mol = mol
        self.coords = coords
        self.non0tab = None
        self.cutoff = 0
        if weights is None:
            self.weights = np.ones(coords.shape[0])
        else:
            self.weights = weights


class TestFracLapl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvp", output="/dev/null")
        settings = FracLaplSettings(
            [-1.0, -0.5, 0.25, 0.5], 4, 2, [(-1, 0), (0, 1), (0, 0), (1, 1)]
        )
        settings2 = FracLaplSettings(
            [-1.0, -0.5, 0.25, 0.5],
            4,
            2,
            [(-1, 0), (0, 1), (0, 0), (1, 1)],
            nd1=2,
            ld_dots=[(-1, 0), (0, 1), (0, 0), (1, 1)],
            ndd=2,
        )
        plan = FracLaplPlan(settings, 1)
        numint = FLNumInt(plan)
        cls.mol = mol
        cls.numint = numint
        plan2 = FracLaplPlan(settings2, 1)
        numint2 = FLNumInt(plan2)
        cls.mol = mol
        cls.numint2 = numint2

    def test_frac_lapl_deriv(self):
        x = np.linspace(0.001, 600, 1000000)
        xp = x + 0.5 * DELTA
        xm = x - 0.5 * DELTA
        s = 0.5
        lmax = 6
        _set_flapl(a=0.025, d=0.002, size=6000, lmax=lmax, s=s)
        # fbuf = GLOBAL_FRAC_LAPL_DATA[s]
        for l in range(lmax + 1):
            fp = np.zeros_like(x)
            fm = np.zeros_like(x)
            df_pred = np.zeros_like(x)
            libcider.vec_eval_1f1(
                fp.ctypes.data_as(ctypes.c_void_p),
                xp.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(x.size),
                ctypes.c_int(l),
            )
            libcider.vec_eval_1f1(
                fm.ctypes.data_as(ctypes.c_void_p),
                xm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(x.size),
                ctypes.c_int(l),
            )
            libcider.vec_eval_1f1_grad(
                df_pred.ctypes.data_as(ctypes.c_void_p),
                x.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(x.size),
                ctypes.c_int(l),
            )
            df_ref = (fm - fp) / DELTA
            assert_allclose(df_pred, df_ref, rtol=1e-6, atol=1e-7)

    def _check_vxc(self, mol, v2=False):
        if v2:
            numint = self.numint2
        else:
            numint = self.numint
        ks = dft.RKS(mol)
        ks.xc = "PBE"
        ks.kernel()
        dm = ks.make_rdm1()
        make_rho, nset, nao = numint._gen_rho_evaluator(
            mol, dm, hermi=1, with_lapl=False
        )
        grids = dft.gen_grid.Grids(mol)
        grids.build()
        cutoff = grids.cutoff * 1e2
        nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))
        ao_loc = mol.ao_loc_nr()
        vmat = np.zeros_like(dm)[None, :]
        v1 = np.zeros_like(vmat)
        aow1 = None
        aow2 = None
        for (ao, kao), mask, weight, coords in numint.block_loop(mol, grids, deriv=1):
            rho = make_rho(0, (ao, kao), None, "MGGA")
            feat = numint.plan.get_feat(rho, make_l1_data_copy=False)
            assert feat.ndim == 3
            feat2 = feat.copy() + np.random.normal(size=feat.shape)
            numint.plan.get_feat(rho, feat=feat2, make_l1_data_copy=False)
            assert_allclose(feat2, feat, atol=1e-12)
            edens, vfeat = get_e_and_v(feat2)
            featp = feat2.copy()
            ind_feat = 0
            featp[:, ind_feat, :] += DELTA
            edens_p = get_e_and_v(featp)[0]
            vfeat_fd = (edens_p - edens) / DELTA
            assert_allclose(vfeat_fd, vfeat[:, ind_feat], atol=4e-6)
            np.dot(edens, weight).sum()
            vxc = numint.plan.get_vxc(vfeat)
            for i in range(numint.plan._nsl + numint.plan.settings.nrho):
                rhop = rho.copy()
                rhop[i] += 0.5 * DELTA
                featp = numint.plan.get_feat(rhop, make_l1_data_copy=False)
                ep = get_e_and_v(featp)[0]
                rhop[i] -= DELTA
                featm = numint.plan.get_feat(rhop, make_l1_data_copy=False)
                em = get_e_and_v(featm)[0]
                rhop[i] += 0.5 * DELTA
                de = (ep - em) / DELTA
                assert_allclose(vxc[:, i], de, rtol=1e-7, atol=1e-8)
            wv = vxc * weight
            aow1, aow2 = _odp_dot_sparse_(
                (ao, kao),
                wv[0],
                nbins,
                mask,
                None,
                ao_loc,
                numint.plan.settings,
                vmat=vmat[0],
                v1=v1[0],
                aow1=aow1,
                aow2=aow2,
            )[:2]
        vmat = lib.hermi_sum(vmat, axes=(0, 2, 1))
        vmat += v1
        np.random.seed(6)
        dm_pert = np.random.normal(size=dm.shape)
        dm_pert += dm_pert.T
        dm_pert /= np.linalg.norm(dm_pert)
        make_rho, nset, nao = numint._gen_rho_evaluator(
            mol, dm + 0.5 * DELTA * dm_pert, hermi=1, with_lapl=False
        )
        ep = 0
        for ao, mask, weight, coords in numint.block_loop(mol, grids, deriv=1):
            rho = make_rho(0, ao, None, "MGGA")
            feat = numint.plan.get_feat(rho, make_l1_data_copy=False)
            edens = get_e_and_v(feat)[0]
            ep += np.dot(edens, weight).sum()
        make_rho, nset, nao = numint._gen_rho_evaluator(
            mol, dm - 0.5 * DELTA * dm_pert, hermi=1, with_lapl=False
        )
        em = 0
        for (ao, kao), mask, weight, coords in numint.block_loop(mol, grids, deriv=1):
            rho = make_rho(0, (ao, kao), None, "MGGA")
            feat = numint.plan.get_feat(rho, make_l1_data_copy=False)
            edens = get_e_and_v(feat)[0]
            em += np.dot(edens, weight).sum()
        egrad_pred = (dm_pert * vmat).sum()
        egrad_ref = (ep - em) / DELTA
        print(egrad_pred, egrad_ref)
        assert_allclose(egrad_pred, egrad_ref, rtol=1e-7, atol=1e-10)

    def test_vxc(self):
        self._check_vxc(self.mol)
        self._check_vxc(self.mol, v2=True)

    def _check_rotation_invariance(self, mol, coords, v2=False):
        rot_coords = get_rotated_coords(coords, 0.6 * np.pi, axis="x")
        rot_mol = rotate_molecule(mol, 0.6 * np.pi, axis="x")
        plan = self.numint2.plan if v2 else self.numint.plan

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
        desc = get_descriptors(analyzer, plan.settings, orbs=None)

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
        rot_desc = get_descriptors(analyzer, plan.settings, orbs=None)

        assert_allclose(rot_desc, desc, rtol=1e-4, atol=1e-4)

    def test_rotation_invariance(self):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvppd", output="/dev/null")
        ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        ks.xc = "PBE"
        ks.conv_tol = 1e-12
        ks.kernel()
        coords = get_random_coords(ks)
        self._check_rotation_invariance(mol, coords, v2=False)
        self._check_rotation_invariance(mol, coords, v2=True)

    def _check_uniform_scaling(
        self, analyzer, coords, lambd=1.3, rtol=1e-4, atol=1e-4, v2=False
    ):
        grids = _Grids(analyzer.mol, coords)
        grids.non0tab = None
        grids.cutoff = 0
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        if v2:
            plan = self.numint2.plan
        else:
            plan = self.numint.plan
        desc = get_descriptors(analyzer, plan.settings, orbs=None)
        grids = _Grids(analyzer.mol, coords / lambd)
        grids.non0tab = None
        grids.cutoff = 0
        analyzer.grids = grids
        analyzer.mol = get_scaled_mol(analyzer.mol, lambd)
        grids.mol = analyzer.mol
        analyzer.perform_full_analysis()
        sdesc = get_descriptors(analyzer, plan.settings, orbs=None)
        pows = np.array(plan.settings.get_feat_usps())
        assert_allclose(
            sdesc, desc * lambd ** pows[None, :, None], rtol=rtol, atol=atol
        )

    def test_uniform_scaling(self):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvppd", output="/dev/null")
        ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        ks.xc = "PBE"
        ks.conv_tol = 1e-12
        ks.kernel()
        coords = get_random_coords(ks)
        if mol.spin != 0:
            analyzer = UHFAnalyzer.from_calc(ks)
        else:
            analyzer = RHFAnalyzer.from_calc(ks)
        self._check_uniform_scaling(analyzer, coords, v2=False)
        self._check_uniform_scaling(analyzer, coords, v2=True)

    def _check_occ_derivs(self, analyzer, coords, rtol=1e-4, atol=1e-4, v2=False):
        mo_occ = analyzer.mo_occ
        mo_coeff = analyzer.mo_coeff
        mo_energy = analyzer.mo_energy
        grids = _Grids(analyzer.mol, coords)
        inner_level = 3
        analyzer.grids = grids
        analyzer.perform_full_analysis()
        orbs = {"U": [2, 0], "O": [0, 1]}

        if v2:
            numint = self.numint2
        else:
            numint = self.numint

        desc0 = get_descriptors(analyzer, numint.plan.settings, orbs=None)
        settings = numint.plan.settings
        desc, ddesc, eigvals = get_descriptors(
            analyzer, numint.plan.settings, orbs=orbs
        )
        assert_allclose(desc, desc0)

        make_rho, nset, nao = numint._gen_rho_evaluator(
            analyzer.mol, analyzer.dm, hermi=1, with_lapl=False
        )
        ip0 = 0
        feat_ref = np.zeros((settings.nfeat, grids.coords.shape[0]))
        for ao, mask, weight, coords in numint.block_loop(analyzer.mol, grids, deriv=1):
            ip1 = ip0 + coords.shape[0]
            rho = make_rho(0, ao, None, "MGGA")
            feat_ref[:, ip0:ip1] = numint.plan.get_feat(rho, make_l1_data_copy=False)
            ip0 = ip1
        assert_allclose(desc[0], feat_ref)

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
        mol = self.mol
        ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        ks.xc = "PBE"
        ks.kernel()
        coords = get_random_coords(ks)
        if mol.spin != 0:
            analyzer = UHFAnalyzer.from_calc(ks)
        else:
            analyzer = RHFAnalyzer.from_calc(ks)
        self._check_occ_derivs(analyzer, coords)
        self._check_occ_derivs(analyzer, coords, v2=True)

    def _check_ueg(self, rho, v2=False):
        if v2:
            settings = self.numint2.plan.settings
        else:
            settings = self.numint.plan.settings
        rmin = 1e-7
        d = 0.0001
        N = 300000
        r = rmin * np.exp(d * np.arange(N))
        dr = d * r
        dv = 4 * np.pi * r * r * dr
        k = (rho * 3 * np.pi**2) ** (1.0 / 3)
        ueg_dm = rho * 3 * (np.sin(k * r) - k * r * np.cos(k * r))
        ueg_dm /= (k * r) ** 3
        coords = np.zeros((N, 3))
        coords[:, 2] = r
        grids = _Grids(None, coords, weights=dv)
        grids_cen = _Grids(None, np.zeros((1, 3)))
        ueg_vals_num = []
        for s in settings.slist:
            ueg_vals_num.append(
                get_flapl(grids, grids_cen, ueg_dm, rho * np.ones(1), s=s)[0]
            )
        ueg_preds = settings.ueg_vector(rho)
        print(rho)
        # exclude s=0.5 because it is less stable
        print(ueg_preds[:3], ueg_vals_num[:3])
        assert_allclose(ueg_preds[:3], ueg_vals_num[:3], rtol=1e-2, atol=3e-3)
        assert_allclose(ueg_preds[3], ueg_vals_num[3], rtol=3e-2, atol=1e-2)
        assert_allclose(ueg_preds[4:], 0, rtol=0, atol=0)

    def test_ueg(self):
        self._check_ueg(1.0)
        self._check_ueg(2.3)
        self._check_ueg(0.32)
        self._check_ueg(1.0, v2=True)
        self._check_ueg(2.3, v2=True)
        self._check_ueg(0.32, v2=True)

    def test_eval_flapl_gto(self):
        settings = self.numint.plan.settings

        mol = gto.M(atom="H 0 0 0; F 0 0 1.0", basis="def2-qzvppd", output="/dev/null")
        grids = Grids(mol)
        grids.prune = None
        grids.level = 8
        grids.build()

        grids_cen = _Grids(
            mol,
            np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [10.0, 0.0, 0.0]], dtype=np.float64
            ),
        )

        t0 = time.monotonic()
        ao_ref = eval_ao(mol, coords=grids.coords)
        ao_ref_cen = eval_ao(mol, coords=grids_cen.coords)
        t1 = time.monotonic()
        ao_pred = eval_flapl_gto(settings.slist, mol, coords=grids.coords, debug=True)
        t2 = time.monotonic()
        for i in range(ao_pred.shape[0]):
            assert_allclose(ao_ref, ao_pred[i], rtol=1e-12, atol=1e-12)
        t3 = time.monotonic()
        kao_pred = eval_flapl_gto(settings.slist, mol, coords=grids.coords, debug=False)
        kao_pred_cen = eval_flapl_gto(
            settings.slist, mol, coords=grids_cen.coords, debug=False
        )
        sub_coords = np.ascontiguousarray(grids.coords[::10])
        k1ao_pred = eval_kao(settings.slist, mol, coords=sub_coords, n1=3)
        assert k1ao_pred.shape[0] == 13
        for i in range(3):
            coords = sub_coords.copy()
            coords[:, i] -= 0.5 * DELTA
            kmao = eval_kao(settings.slist, mol, coords=coords)
            coords[:, i] += DELTA
            kpao = eval_kao(settings.slist, mol, coords=coords)
            kfd_ref = (kpao - kmao) / DELTA
            for j in range(3):
                assert_allclose(
                    k1ao_pred[1 + 4 * j + i], kfd_ref[j], rtol=1e-5, atol=1e-6
                )
        t4 = time.monotonic()
        print(t1 - t0, t2 - t1, t4 - t3)

        assert kao_pred.shape == ao_pred.shape
        for j in range(mol.nao_nr()):
            for i, s in enumerate(settings.slist):
                refvals = get_flapl(
                    grids, grids_cen, ao_ref[..., j], ao_ref_cen[..., j], s=s
                )
                if j > 20:
                    continue
                if s < 0:
                    assert_allclose(
                        kao_pred_cen[i, :-1, j],
                        refvals[:-1],
                        rtol=1e-2,
                        atol=1e-2,
                        equal_nan=False,
                    )
                assert_allclose(
                    kao_pred_cen[i, -1, j],
                    refvals[-1],
                    rtol=1e-3,
                    atol=1e-3,
                    equal_nan=False,
                )

        ks = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        ks.xc = "PBE"
        ks.kernel()
        dm = ks.make_rdm1()
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()
        print(ao_ref.shape)
        dm_gu = _dot_ao_dm(mol, ao_ref, dm, None, shls_slice, ao_loc)
        dm_ru = _dot_ao_dm(mol, ao_ref_cen, dm, None, shls_slice, ao_loc)
        dm_rg = np.einsum("ru,gu->rg", ao_ref_cen, dm_gu)
        dm_rr = np.diag(np.einsum("ru,gu->rg", ao_ref_cen, dm_ru))

        make_rho, nset, nao = self.numint._gen_rho_evaluator(
            mol, dm, hermi=1, with_lapl=False
        )
        for ao, mask, weight, coords in self.numint.block_loop(mol, grids_cen, deriv=1):
            rho = make_rho(0, ao, None, "MGGA")
            feat = self.numint.plan.get_feat(rho, make_l1_data_copy=False)

        i = 0
        s = settings.slist[i]
        refvals = get_flapl2(grids, grids_cen, dm_rg, dm_rr, s=s)
        assert_allclose(feat[0, i], refvals, rtol=2e-3, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
