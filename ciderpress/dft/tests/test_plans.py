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
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from ciderpress.dft.plans import (
    NLDFGaussianPlan,
    NLDFSplinePlan,
    SemilocalPlan,
    SemilocalPlan2,
)
from ciderpress.dft.settings import (
    CFC,
    NLDFSettingsVI,
    NLDFSettingsVIJ,
    NLDFSettingsVJ,
    NLDFSettingsVK,
    SemilocalSettings,
)

DELTA = 1e-7


def ovlp_reference(a, b, featid, extra_args=None):
    if featid == "se":
        return (np.pi / (a + b)) ** 1.5
    elif featid == "se_ar2":
        return a * 1.5 * np.pi**1.5 / (a + b) ** 2.5
    elif featid == "se_a2r4":
        return a * a * 15.0 / 4 * np.pi**1.5 / (a + b) ** 3.5
    elif featid == "se_erf_rinv":
        assert extra_args is not None
        c = extra_args[0] * a
        return (np.pi / (a + b)) ** 1.5 / np.sqrt(1 + c / (a + b))
    else:
        raise ValueError


class TestSemilocalPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        N = 50
        cls._x = np.linspace(0, 1, N)

        # construct a reasonable set of densities
        np.random.seed(53)
        directions = 0.25 * np.random.normal(size=(3, N))
        np.random.seed(53)
        alpha = np.random.normal(size=N) ** 2
        rho = 1 + 0.5 * np.sin(np.pi * cls._x)
        drho = directions * rho
        tauw = np.einsum("xg,xg->g", drho, drho) / (8 * rho)
        tau = tauw + alpha * CFC * rho ** (5.0 / 3)
        rho_data = np.concatenate([[rho], drho, [tau]], axis=0)
        assert rho_data.shape[0] == 5
        cls.rho = rho_data[None, :, :]
        cls.sprho = 0.5 * np.stack([rho_data, rho_data])
        cls.gga_rho = rho_data[None, :4, :]
        cls.gga_sprho = 0.5 * np.stack([rho_data[:4], rho_data[:4]])
        np.random.seed(42)
        cls.occd = np.random.normal(size=cls.rho.shape)
        cls.spoccd = np.random.normal(size=cls.sprho.shape)
        cls.gga_occd = np.random.normal(size=cls.gga_rho.shape)
        cls.gga_spoccd = np.random.normal(size=cls.gga_sprho.shape)

        cls.npa = SemilocalSettings(mode="npa")
        cls.nst = SemilocalSettings(mode="nst")
        cls.np = SemilocalSettings(mode="np")
        cls.ns = SemilocalSettings(mode="ns")

    def check_settings(self, settings):
        plan = SemilocalPlan(settings, 1)
        plan2 = SemilocalPlan2(settings, 1)
        if settings.level == "MGGA":
            rho = self.rho
            drho = self.occd
        else:
            rho = self.gga_rho
            drho = self.gga_occd
        rho = rho.copy()
        feat = plan.get_feat(rho)
        sigma = np.einsum("sxg,sxg->sg", rho[:, 1:4], rho[:, 1:4])
        tau = rho[:, 4] if settings.level == "MGGA" else None
        dtau = drho[:, 4] if settings.level == "MGGA" else None
        feat2 = plan2.get_feat(rho[:, 0], sigma, tau)
        assert_allclose(feat, feat2, atol=1e-14)

        def get_e_and_vfeat(feat):
            energy = (1 + (feat**2).sum(axis=1)) ** 0.5
            return (energy, feat / energy[:, None, :])

        vfeat = get_e_and_vfeat(feat)[1]
        vxc1 = plan.get_vxc(rho, vfeat, vxc=None)
        vxc2 = np.zeros_like(vxc1)
        vxc3 = plan.get_vxc(rho, vfeat, vxc=vxc2)
        assert_allclose(vxc2, vxc1, atol=1e-14)
        assert_allclose(vxc3, vxc1, atol=1e-14)

        vfeat = get_e_and_vfeat(feat2)[1]
        shape = rho[:, 0].shape
        vrho, vsigma, vtau = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        plan2.get_vxc(vfeat, rho[:, 0], vrho, sigma, vsigma, tau, vtau)
        assert_allclose(vrho, vxc1[:, 0], atol=1e-14)
        assert_allclose(2 * vsigma[:, None] * rho[:, 1:4], vxc1[:, 1:4], atol=1e-14)
        if plan.level == "MGGA":
            assert_allclose(vtau, vxc1[:, 4], atol=1e-14)
        featb, occd = plan.get_occd(rho, drho)
        dsigma = 2 * np.einsum("sxg,sxg->sg", rho[:, 1:4], drho[:, 1:4])
        feat2b, occd2 = plan2.get_occd(rho[:, 0], drho[:, 0], sigma, dsigma, tau, dtau)
        assert_allclose(occd, occd2, atol=1e-14)
        assert_allclose(featb, feat, atol=1e-14)
        assert_allclose(feat2b, feat2, atol=1e-14)
        assert_allclose((vfeat * occd2).sum(1), (vxc1 * drho).sum(1))

        delta = 1e-5
        for i in range(rho.shape[1]):
            rho[:, i] += 0.5 * delta
            f = plan.get_feat(rho)
            ep = get_e_and_vfeat(f)[0]
            rho[:, i] -= delta
            f = plan.get_feat(rho)
            em = get_e_and_vfeat(f)[0]
            rho[:, i] += 0.5 * delta
            diff = (ep - em) / delta
            assert_allclose(vxc1[:, i], diff, rtol=1e-8, atol=1e-8)

        nsp_feat = feat
        plan = SemilocalPlan(settings, 2)
        plan2 = SemilocalPlan2(settings, 2)
        if settings.level == "MGGA":
            rho = self.sprho
            drho = self.spoccd
        else:
            rho = self.gga_sprho
            drho = self.gga_spoccd
        rho = rho.copy()
        feat = plan.get_feat(rho)
        sigma = np.einsum("sxg,sxg->sg", rho[:, 1:4], rho[:, 1:4])
        tau = rho[:, 4] if settings.level == "MGGA" else None
        dtau = drho[:, 4] if settings.level == "MGGA" else None
        feat2 = plan2.get_feat(rho[:, 0], sigma, tau)

        assert feat.shape[0] == 2
        assert feat.shape[1:] == nsp_feat.shape[1:]
        assert_allclose(feat[0], nsp_feat[0], atol=1e-14)
        assert_allclose(feat[1], nsp_feat[0], atol=1e-14)
        assert_allclose(feat, feat2, atol=1e-14)

        vfeat = get_e_and_vfeat(feat)[1]
        vxc1 = plan.get_vxc(rho, vfeat, vxc=None)
        vxc2 = np.zeros_like(vxc1)
        vxc3 = plan.get_vxc(rho, vfeat, vxc=vxc2)
        assert_allclose(vxc2, vxc1, atol=1e-14)
        assert_allclose(vxc3, vxc1, atol=1e-14)

        vfeat = get_e_and_vfeat(feat2)[1]
        shape = rho[:, 0].shape
        vrho, vsigma, vtau = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        plan2.get_vxc(vfeat, rho[:, 0], vrho, sigma, vsigma, tau, vtau)
        assert_allclose(vrho, vxc1[:, 0], atol=1e-14)
        assert_allclose(2 * vsigma[:, None] * rho[:, 1:4], vxc1[:, 1:4], atol=1e-14)
        if plan.level == "MGGA":
            assert_allclose(vtau, vxc1[:, 4], atol=1e-14)
        featb, occd = plan.get_occd(rho, drho)
        dsigma = 2 * np.einsum("sxg,sxg->sg", rho[:, 1:4], drho[:, 1:4])
        feat2b, occd2 = plan2.get_occd(rho[:, 0], drho[:, 0], sigma, dsigma, tau, dtau)
        assert_allclose(featb, feat, atol=1e-14)
        assert_allclose(occd, occd2, atol=1e-14)
        assert_allclose(feat2b, feat2, atol=1e-14)
        assert_allclose((vfeat * occd2).sum(1), (vxc1 * drho).sum(1), atol=1e-14)

        delta = 1e-5
        for i in range(rho.shape[1]):
            rho[:, i] += 0.5 * delta
            f = plan.get_feat(rho)
            ep = get_e_and_vfeat(f)[0]
            rho[:, i] -= delta
            f = plan.get_feat(rho)
            em = get_e_and_vfeat(f)[0]
            rho[:, i] += 0.5 * delta
            diff = (ep - em) / delta
            assert_allclose(vxc1[:, i], diff, rtol=1e-8, atol=1e-8)

    def test_settings(self):
        self.check_settings(self.npa)
        self.check_settings(self.np)
        self.check_settings(self.nst)
        self.check_settings(self.ns)


class TestNLDFGaussianPlan(unittest.TestCase):

    _x = None
    _a = None
    _b = None
    _c1 = None
    _c2 = None
    alpha0 = None
    lambd = None
    nalpha = None
    PlanClass = NLDFGaussianPlan

    @classmethod
    def setUpClass(cls):
        N = 50
        cls._x = np.linspace(0, 1, N)

        # construct a reasonable set of densities
        np.random.seed(53)
        directions = 0.25 * np.random.normal(size=(3, N))
        np.random.seed(53)
        alpha = np.random.normal(size=N) ** 2
        rho = 1 + 0.5 * np.sin(np.pi * cls._x)
        drho = directions * rho
        tauw = np.einsum("xg,xg->g", drho, drho) / (8 * rho)
        tau = tauw + alpha * CFC * rho ** (5.0 / 3)
        rho_data = np.concatenate([[rho], drho, [tau]], axis=0)
        assert rho_data.shape[0] == 5
        cls.rho = rho_data

        # set up parameters for testing nonlocal parts
        cls._a = 0.07
        cls._b = 5.3
        # test function f(x, r) = c1(x) * exp(-a*r^2) + c2(x) * exp(-b*r^2)
        cls._c1 = 1.3 * cls._x**2
        # cls._c1 = 0 * cls._x ** 2
        cls._c2 = 0.2 - 0.5 * cls._x
        # cls._c2 = 0.0 * cls._x
        cls.alpha0 = 0.003
        cls.lambd = 1.6
        cls.nalpha = 30
        thetap = [2.0, 0.5, 0.0625]
        vb_params = [
            [1.0, 0.25, 0.03125],
            [2.0, 0.50, 0.06250],
            [4.0, 1.00, 0.12500],
        ]
        feat_params = [
            [1.0, 0.25, 0.03125],
            [1.0, 0.00, 0.03125],
            [4.0, 0.00, 0.06250, 1.0],
            [1.0, 0.25, 0.03125],
            [1.0, 0.00, 0.03125],
        ]
        feat_specs = ["se", "se_a2r4", "se_erf_rinv", "se_ar2", "se"]
        i0specs = ["se_lapl", "se_ap2r2", "se_apr2", "se", "se_ap"]
        i1specs = ["se_rvec", "se_grad"]
        i1dots = [(-1, 1), (-1, 0), (1, 0)]
        cls.example_vi_settings = NLDFSettingsVI(
            "MGGA",
            thetap,
            "one",
            i0specs,
            i1specs,
            i1dots,
        )
        cls.example_vj_settings = NLDFSettingsVJ(
            "MGGA",
            thetap,
            "one",
            feat_specs,
            feat_params,
        )
        cls.vb_settings = NLDFSettingsVJ(
            "MGGA",
            thetap,
            "one",
            ["se", "se", "se"],
            vb_params,
        )
        cls.example_vij_settings = NLDFSettingsVIJ(
            "MGGA",
            thetap,
            "one",
            i0specs,
            i1specs,
            i1dots,
            feat_specs,
            feat_params,
        )
        cls.example_vk_settings = NLDFSettingsVK(
            "MGGA",
            thetap,
            "one",
            feat_params[:2],
            rho_damp="exponential",
        )

    def _get_plan(self, settings, nspin, **kwargs):
        lambd = kwargs.pop("lambd", self.lambd)
        alpha0 = kwargs.pop("alpha0", self.alpha0)
        nalpha = kwargs.pop("nalpha", self.nalpha)
        return self.PlanClass(settings, nspin, alpha0, lambd, nalpha, **kwargs)

    def test_eval_feat_exp(self):
        plan = self._get_plan(self.example_vj_settings, 1)
        plan2 = self._get_plan(self.example_vj_settings, 2)
        sigma = np.einsum("xg,xg->g", self.rho[1:4], self.rho[1:4])
        results = []
        for i in range(5):
            results.append(plan.eval_feat_exp((self.rho[0], sigma, self.rho[4]), i=i))
        results.append(plan.eval_feat_exp((self.rho[0], sigma, self.rho[4]), i=-1))
        for i in range(len(results)):
            results[i] = [
                results[i][0],
                results[i][1][0],
                results[i][1][1],
                results[i][1][2],
            ]
        for j in range(4):
            assert_equal(results[0][j], results[3][j])
            assert_equal(results[1][j], results[4][j])
            assert_almost_equal(results[-1][j], 2 * results[0][j])
            assert_almost_equal(results[-1][j], 2 * results[3][j])
        for i in [0, 1, 2, 3, 4, -1]:
            rho_tmp = self.rho[0].copy()
            sigma_tmp = sigma.copy()
            tau_tmp = self.rho[4].copy()
            a, (dadn, dadsigma, dadtau) = plan.eval_feat_exp(
                (rho_tmp, sigma_tmp, tau_tmp), i=i
            )
            a2, (da2dn, da2dsigma, da2dtau) = plan2.eval_feat_exp(
                (0.5 * rho_tmp, 0.25 * sigma_tmp, 0.5 * tau_tmp), i=i
            )
            assert_almost_equal(a2, a, 12)
            assert_almost_equal(da2dn, 2 * dadn, 12)
            assert_almost_equal(da2dsigma, 4 * dadsigma, 12)
            assert_almost_equal(da2dtau, 2 * dadtau, 12)
            inputs = [rho_tmp, sigma_tmp, tau_tmp]
            derivs = [dadn, dadsigma, dadtau]
            for input, deriv in zip(inputs, derivs):
                input[:] += 0.5 * DELTA
                ap = plan.eval_feat_exp((rho_tmp, sigma_tmp, tau_tmp), i=i)[0]
                input[:] -= DELTA
                am = plan.eval_feat_exp((rho_tmp, sigma_tmp, tau_tmp), i=i)[0]
                input[:] += 0.5 * DELTA
                assert_almost_equal(deriv, (ap - am) / DELTA, 7)
            rho_tiny = 5e-11 * np.ones_like(rho_tmp)
            sigma_tiny = sigma_tmp * rho_tiny**2 / rho_tmp**2
            tau_tiny = tau_tmp * rho_tiny / rho_tmp
            a, (dadn, dadsigma, dadtau) = plan.eval_feat_exp(
                (rho_tiny, sigma_tiny, tau_tiny), i=i
            )
            assert (dadn == 0).all()
            assert (dadsigma == 0).all()
            assert (dadtau == 0).all()

    def test_get_interpolation_arguments(self):
        plan = self.PlanClass(
            self.example_vij_settings,
            1,
            self.alpha0,
            self.lambd,
            self.nalpha,
        )
        rho = self.rho[0]
        tau = self.rho[4]
        sigma = np.einsum("xg,xg->g", self.rho[1:4], self.rho[1:4])
        for i in range(-1, 5):
            rho_tmp = rho.copy()
            sigma_tmp = sigma.copy()
            tau_tmp = tau.copy()
            a, (dadn, dadsigma, dadtau) = plan.get_interpolation_arguments(
                (rho_tmp, sigma_tmp, tau_tmp), i=i
            )
            inputs = [rho_tmp, sigma_tmp, tau_tmp]
            derivs = [dadn, dadsigma, dadtau]
            for input, deriv in zip(inputs, derivs):
                input[:] += 0.5 * DELTA
                ap = plan.get_interpolation_arguments(
                    (rho_tmp, sigma_tmp, tau_tmp), i=i
                )[0]
                input[:] -= DELTA
                am = plan.get_interpolation_arguments(
                    (rho_tmp, sigma_tmp, tau_tmp), i=i
                )[0]
                input[:] += 0.5 * DELTA
                assert_almost_equal(deriv, (ap - am) / DELTA, 7)
            rho_tiny = 5e-11 * np.ones_like(rho_tmp)
            sigma_tiny = sigma_tmp * rho_tiny**2 / rho_tmp**2
            tau_tiny = tau_tmp * rho_tiny / rho_tmp
            a, (dadn, dadsigma, dadtau) = plan.eval_feat_exp(
                (rho_tiny, sigma_tiny, tau_tiny), i=i
            )
            assert (dadn == 0).all()
            assert (dadsigma == 0).all()
            assert (dadtau == 0).all()
        try:
            plan.get_interpolation_arguments((rho, sigma, tau), i=5)
            raise AssertionError("This should raise an error; index too high.")
        except ValueError:
            pass
        for i in range(-1, 5):
            try:
                plan.get_interpolation_arguments(
                    (1e5 * rho, 1e10 * sigma, 1e5 * tau), i=i
                )
                raise AssertionError("This should raise an error; exponent too large.")
            except RuntimeError:
                pass

    def check_get_interpolation_coefficients(self, **kwargs):
        plan = self._get_plan(self.example_vij_settings, 1, **kwargs)
        rho = self.rho[0]
        tau = self.rho[4]
        sigma = np.einsum("xg,xg->g", self.rho[1:4], self.rho[1:4])
        feat_parts = self._get_feat(plan.alphas[:, None], "se", None)
        feat_parts *= plan.alpha_norms[:, None]
        for i in range(-1, 5):
            exp_g = plan.eval_feat_exp((rho, sigma, tau), i=i)[0]
            arg_g = plan.get_interpolation_arguments((rho, sigma, tau), i=i)[0]
            p, dp = plan.get_interpolation_coefficients(arg_g, i=i)
            p2, dp2 = np.empty_like(p), np.empty_like(p)
            plan.get_interpolation_coefficients(arg_g, i=i, vbuf=p2, dbuf=dp2)
            assert_equal(p, p2)
            assert_equal(dp, dp2)
            arg_g[:] += 0.5 * DELTA
            pp, _ = plan.get_interpolation_coefficients(arg_g, i=i)
            arg_g[:] -= DELTA
            pm, _ = plan.get_interpolation_coefficients(arg_g, i=i)
            arg_g[:] += 0.5 * DELTA
            assert_almost_equal(dp, (pp - pm) / DELTA, 4)
            p = plan.get_transformed_interpolation_terms(
                p, i=i, fwd=True, inplace=False
            )
            if i == -1:
                feat_id = "se"
            else:
                feat_id = plan.nldf_settings.feat_spec_list[i]
            extra_args = plan._get_extra_args(i) if i > -1 else None
            ref_feat = self._get_feat(exp_g, feat_id, extra_args)
            test_feat = np.einsum("qg,{}->g".format(plan.coef_order), feat_parts, p)
            assert_almost_equal(test_feat, ref_feat, 3)

    def check_interpolation_coefs_stability(self, settings, tol=1e-3, **kwargs):
        if self.PlanClass == NLDFSplinePlan:
            kwargs["spline_size"] = None
        alpha_max = 120
        alpha_min = 0.001
        ratio = alpha_max / alpha_min
        if kwargs["alpha_formula"] == "etb":
            ratio += 1
        kwargs["nalpha"] = int(np.ceil(np.log(ratio) / np.log(kwargs["lambd"]))) + 1
        plan = self._get_plan(settings, 1, **kwargs)
        exp_g = 0.1 * 1.06 ** np.arange(100)
        if isinstance(plan, NLDFSplinePlan):
            arg_g = plan.get_a2q_fast(exp_g)[0]
        else:
            arg_g = exp_g
        p = plan.get_interpolation_coefficients(arg_g, i=-1)[0]
        p = plan.get_transformed_interpolation_terms(p, i=-1, fwd=True)
        rs = 0.02 * (np.exp(0.03 * np.linspace(0, 200)) - 1)
        bas = plan.alpha_norms[:, None] * np.exp(-plan.alphas[:, None] * rs**2)
        exact = np.exp(-exp_g[:, None] * rs**2)
        if plan.coef_order == "gq":
            approx = np.einsum("gq,qr->gr", p, bas)
        else:
            approx = np.einsum("qg,qr->gr", p, bas)
        assert_allclose(approx, exact, rtol=tol, atol=tol)

    def test_get_interpolation_coefficients(self):
        # self.check_get_interpolation_coefficients(alpha_formula='zexp')
        for alpha_formula in ["etb", "zexp"]:
            for coef_order in ["gq", "qg"]:
                self.check_get_interpolation_coefficients(
                    coef_order=coef_order,
                    alpha_formula=alpha_formula,
                )

    def test_interpolation_coefs_stability(self):
        if self.PlanClass == NLDFSplinePlan:
            tols = [1e-3, 1e-3, 3e-4, 1e-4, 3e-5]
        else:
            # Gaussian interpolation is a bit less precise
            # for larger lambd
            tols = [1e-2, 3e-3, 1e-3, 1e-4, 3e-5]
        for alpha_formula in ["etb", "zexp"]:
            for coef_order in ["gq", "qg"]:
                # We want to go to extreme values of lambd to make
                # sure the algorithm remains reasonable even
                # when the overlap condition numbers of huge.
                for lambd, tol in zip([1.8, 1.6, 1.5, 1.4, 1.3], tols):
                    self.check_interpolation_coefs_stability(
                        self.example_vij_settings,
                        alpha_formula=alpha_formula,
                        coef_order=coef_order,
                        lambd=lambd,
                        tol=tol,
                    )
                for lambd, tol in zip([1.8, 1.6, 1.5, 1.4, 1.3], tols):
                    self.check_interpolation_coefs_stability(
                        self.vb_settings,
                        alpha_formula=alpha_formula,
                        coef_order=coef_order,
                        lambd=lambd,
                        tol=tol,
                    )
                if self.PlanClass == NLDFSplinePlan:
                    # The spline plan is systematically improvable IF all the
                    # features are of the `se` form and spline_size == nalpha.
                    # This can help reach high precision
                    for lambd, tol in zip([1.3, 1.2, 1.1], [3e-5, 1e-5, 3e-6]):
                        self.check_interpolation_coefs_stability(
                            self.vb_settings,
                            alpha_formula=alpha_formula,
                            coef_order=coef_order,
                            lambd=lambd,
                            tol=tol,
                        )

    def check_eval_rho_full(self, settings, coef_order="qg"):
        plan = self._get_plan(settings, 1, coef_order=coef_order)
        rho = self.rho[0]
        drho = self.rho[1:4]
        tau = self.rho[4]
        sigma = np.einsum("xg,xg->g", drho, drho)
        feat_parts = self._get_feat(plan.alphas[:, None], "se", None)
        feat_parts *= plan.alpha_norms[:, None]
        plan.get_transformed_interpolation_terms(
            feat_parts.T if coef_order == "gq" else feat_parts,
            i=0,
            fwd=False,
            inplace=True,
        )
        f = plan.zero_coefs_full(
            feat_parts.shape[1],
            buffer_nrows=len(plan.nldf_settings.l1_feat_specs),
        )
        vf = np.zeros_like(f)
        if coef_order == "gq":
            if isinstance(settings, (NLDFSettingsVJ, NLDFSettingsVIJ, NLDFSettingsVK)):
                f[:, : plan.nalpha] = feat_parts.T
                start = plan.nalpha
            else:
                start = 0
            if isinstance(settings, (NLDFSettingsVI, NLDFSettingsVIJ)):
                f[:, start:] = np.linspace(0.5, 1.5, f.shape[0])[:, None]
        else:
            if isinstance(settings, (NLDFSettingsVJ, NLDFSettingsVIJ, NLDFSettingsVK)):
                f[: plan.nalpha] = feat_parts
                start = plan.nalpha
            else:
                start = 0
            if isinstance(settings, (NLDFSettingsVI, NLDFSettingsVIJ)):
                f[start:] = np.linspace(0.5, 1.5, f.shape[1])
        feat, dfeat = plan.eval_rho_full(f, self.rho)
        assert feat.shape == (plan.nldf_settings.nfeat, rho.size)
        assert dfeat.shape == (plan.nldf_settings.num_feat_param_sets, rho.size)
        for i in range(plan.nldf_settings.num_feat_param_sets):
            exp_g = plan.eval_feat_exp((rho, sigma, tau), i=i)[0]
            feat_id = plan.nldf_settings.feat_spec_list[i]
            extra_args = plan._get_extra_args(i) if i > -1 else None
            ref_feat = self._get_feat(exp_g, feat_id, extra_args)
            assert_almost_equal(feat[i], ref_feat, 5)

        def get_e_and_vfeat(feat):
            energy = (1 + (feat**2).sum(axis=0)) ** 0.5
            return (energy, feat / energy)

        vfeat = get_e_and_vfeat(feat)[1]
        vrho_data = np.zeros_like(self.rho)
        vrho = vrho_data[0]
        vdrho = vrho_data[1:4]
        vtau = vrho_data[4]
        plan.eval_vxc_full(vfeat, vrho_data, dfeat, self.rho, vf=vf)
        inputs = [rho, drho[0], drho[1], drho[2], tau]
        derivs = [vrho, vdrho[0], vdrho[1], vdrho[2], vtau]
        if coef_order == "gq":
            inputs += [f.T[q] for q in range(f.shape[1])]
            derivs += [vf.T[q] for q in range(f.shape[1])]
        else:
            inputs += [f[q] for q in range(f.shape[0])]
            derivs += [vf[q] for q in range(f.shape[0])]
        i = 0
        for target, deriv in zip(inputs, derivs):
            i += 1
            target[:] += 0.5 * DELTA
            feat_tmp, _ = plan.eval_rho_full(f, self.rho)
            ep, _ = get_e_and_vfeat(feat_tmp)
            target[:] -= DELTA
            feat_tmp, _ = plan.eval_rho_full(f, self.rho)
            em, _ = get_e_and_vfeat(feat_tmp)
            target[:] += 0.5 * DELTA
            assert_almost_equal(deriv, (ep - em) / DELTA, 6)

    def test_eval_rho_and_vxc_full(self):
        self.check_eval_rho_full(self.example_vij_settings, coef_order="qg")
        self.check_eval_rho_full(self.example_vi_settings, coef_order="qg")
        self.check_eval_rho_full(self.example_vj_settings, coef_order="qg")
        self.check_eval_rho_full(self.example_vk_settings, coef_order="qg")
        self.check_eval_rho_full(self.example_vij_settings, coef_order="gq")
        self.check_eval_rho_full(self.example_vi_settings, coef_order="gq")
        self.check_eval_rho_full(self.example_vj_settings, coef_order="gq")
        self.check_eval_rho_full(self.example_vk_settings, coef_order="gq")

    def check_version_k(self, coef_order="gq"):
        plan = self._get_plan(self.example_vk_settings, 1, coef_order=coef_order)
        rho = self.rho[0]
        sigma = np.einsum("xg,xg->g", self.rho[1:4], self.rho[1:4])
        tau = self.rho[4]
        exp_g = plan.eval_feat_exp((rho, sigma, tau), i=0)[0]
        # make some other exponent for testing
        ref_exp_g = exp_g / (0.4 + exp_g) + 0.5 * exp_g
        # arg1_g = plan.get_interpolation_arguments(rho, sigma, tau, i=-1)[0]
        arg2_g = plan.get_interpolation_arguments((rho, sigma, tau), i=0)[0]
        p1, dp1 = plan.get_interpolation_coefficients(ref_exp_g, i=-1)
        p2, dp2 = plan.get_interpolation_coefficients(arg2_g, i=0)
        plan.get_transformed_interpolation_terms(p2, i=0, fwd=True, inplace=True)
        refval = np.exp(-1.5 * ref_exp_g / exp_g)
        if coef_order == "gq":
            p2 *= plan.alpha_norms
        else:
            p2 *= plan.alpha_norms[:, None]
        testval = np.einsum("{},{}->g".format(coef_order, coef_order), p1, p2)
        assert_almost_equal(testval, refval, 3)

    def test_version_k(self):
        self.check_version_k("gq")
        self.check_version_k("qg")

    def _get_feat(self, arg_g, feat_id, extra_args):
        res = self._c1 * ovlp_reference(arg_g, self._a, feat_id, extra_args=extra_args)
        res += self._c2 * ovlp_reference(arg_g, self._b, feat_id, extra_args=extra_args)
        return res


class TestNLDFSplinePlan(TestNLDFGaussianPlan):

    PlanClass = NLDFSplinePlan

    def _get_plan(self, settings, nspin, **kwargs):
        spline_size = kwargs.pop("spline_size", 160)
        kwargs["spline_size"] = spline_size
        lambd = kwargs.pop("lambd", self.lambd)
        alpha0 = kwargs.pop("alpha0", self.alpha0)
        nalpha = kwargs.pop("nalpha", self.nalpha)
        return self.PlanClass(settings, nspin, alpha0, lambd, nalpha, **kwargs)


if __name__ == "__main__":
    unittest.main()
