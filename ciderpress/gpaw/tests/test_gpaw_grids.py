import unittest

import numpy as np
from numpy.testing import assert_allclose

from ciderpress.gpaw.gpaw_grids import (
    SBTFullGridDescriptor,
    make_radial_derivative_calculator,
)

LMAX = 6


def gaussian(l, alpha, r):
    return r**l * np.exp(-alpha * r * r)


def gaussian_ft(l, alpha, k):
    const = np.pi**1.5 / 2**l / alpha ** (1.5 + l)
    return const * k**l * np.exp((-0.25 / alpha) * k * k)


class TestRadialDerivative(unittest.TestCase):
    def test_radial_deriv(self):
        N = 100
        r_g = 0.01 * (np.exp(0.04 * np.arange(N)) - 1)
        dr_g = 0.01 * 0.04 * np.exp(0.04 * np.arange(N))
        exps = [0.5, 1.0, 2.0]
        nexp = len(exps)
        nl = 5
        reffunc_alg = np.empty((nexp, nl, N))
        refderiv_alg = np.empty((nexp, nl, N))
        deriv_o1, bwd_o1 = make_radial_derivative_calculator(r_g, order=1)
        deriv_o2, bwd_o2 = make_radial_derivative_calculator(r_g, order=2)
        deriv_o3, bwd_o3 = make_radial_derivative_calculator(r_g, order=15)
        for i in range(nexp):
            for j in range(nl):
                myexp = np.exp(-exps[i] * r_g)
                jm1 = max(j - 1, 0)
                # precision is a bit lower for higher l, larger exp
                fac = 0.3**j / exps[i]
                fac = 1
                reffunc_alg[i, j] = fac * r_g**j * myexp
                refderiv_alg[i, j] = fac * (j * r_g**jm1 - exps[i] * r_g**j) * myexp
        testo1_alg = deriv_o1(reffunc_alg)
        testo2_alg = deriv_o2(reffunc_alg)
        testo3_alg = deriv_o3(reffunc_alg)
        assert_allclose(testo1_alg, refderiv_alg, rtol=1e-4, atol=1e-3)
        assert_allclose(testo2_alg, refderiv_alg, rtol=1e-6, atol=1e-5)
        assert_allclose(testo3_alg, refderiv_alg, rtol=1e-7, atol=1e-6)
        np.random.seed(89)
        # test that the backward function matches finite difference
        for i in range(4):
            wt = 1e-6 * np.random.uniform(size=reffunc_alg.shape)
            vderiv_alg = np.random.uniform(size=refderiv_alg.shape) * dr_g
            for fwd_func, bwd_func in zip(
                [deriv_o1, deriv_o2, deriv_o3], [bwd_o1, bwd_o2, bwd_o3]
            ):
                p_alg = fwd_func(reffunc_alg + 0.5 * wt)
                m_alg = fwd_func(reffunc_alg - 0.5 * wt)
                fd_result = np.einsum("alg,alg->", p_alg, vderiv_alg)
                fd_result -= np.einsum("alg,alg->", m_alg, vderiv_alg)
                fd_result /= wt.sum()
                print(np.abs(vderiv_alg).sum())
                ad_alg = bwd_func(vderiv_alg)
                print(np.abs(ad_alg).sum())
                ad_result = np.einsum("alg,alg->", ad_alg, wt)
                ad_result /= wt.sum()
                print(wt.sum(), fd_result, ad_result)
                assert_allclose(fd_result, ad_result)
                print()


class TestSBT(unittest.TestCase):
    def test_sbt_gaussian(self):
        sbtgd = SBTFullGridDescriptor(0.001, 1e12, 0.02, N=1024, lmax=LMAX)
        for l in range(LMAX + 1):
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                scale = alpha ** (1.5 + 0.5 * l)
                f_g = scale * gaussian(l, alpha, sbtgd.r_g)
                fref_k = scale * gaussian_ft(l, alpha, sbtgd.k_g)
                ftest_k = 4 * np.pi * sbtgd.transform_single_fwd(f_g, l)
                print("MINMAX", np.min(sbtgd.k_g), np.max(sbtgd.k_g))
                print(
                    np.linalg.norm(ftest_k - fref_k),
                    np.linalg.norm(fref_k),
                    np.linalg.norm(ftest_k),
                )
                print(np.max(np.abs(ftest_k - fref_k)))
                assert_allclose(ftest_k, fref_k, atol=1e-4, rtol=0)
                ftest_g = (0.25 / np.pi) * sbtgd.transform_single_bwd(ftest_k, l)
                assert_allclose(ftest_g, f_g, atol=1e-4, rtol=0)


if __name__ == "__main__":
    unittest.main()
