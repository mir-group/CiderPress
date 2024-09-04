import unittest

import numpy as np
from numpy.testing import assert_allclose
from pyscf.dft.libxc import eval_xc

from ciderpress.dft.baselines import gga_c_pbe
from ciderpress.dft.settings import ds2

DELTA = 1e-6


def get_fd(feat_index, spin, feat, func):
    feat[spin, feat_index] += 0.5 * DELTA
    ep = func(feat)[0]
    feat[spin, feat_index] -= DELTA
    em = func(feat)[0]
    feat[spin, feat_index] += 0.5 * DELTA
    return (ep - em) / DELTA


class TestBaselines(unittest.TestCase):
    def test_gga_c_pbe(self):
        np.random.seed(42)
        rho1 = np.random.normal(size=(4, 20))
        rho2 = np.random.normal(size=(4, 20))
        rho1[0] = np.abs(rho1[0])
        rho2[0] = np.abs(rho2[0])
        sigma1 = np.einsum("xg,xg->g", rho1[1:4], rho1[1:4])
        sigma2 = np.einsum("xg,xg->g", rho2[1:4], rho2[1:4])
        p1 = sigma1 / (4.0 * (3 * np.pi**2) ** (2.0 / 3) * rho1[0] ** (8.0 / 3))
        p2 = sigma2 / (4.0 * (3 * np.pi**2) ** (2.0 / 3) * rho2[0] ** (8.0 / 3))

        feat = np.stack([rho1[0], p1, 0.5 * p1, 0.25 * rho1[0]])
        e_ref, vxc = eval_xc("GGA_C_PBE", rho1, spin=0)[:2]
        e_test, dedx = gga_c_pbe(feat[None, :])
        for ind in [0, 1]:
            fd = get_fd(ind, 0, feat[None, :], gga_c_pbe)
            assert_allclose(dedx[0, ind], fd, atol=1e-7, rtol=1e-7)
        dp = ds2(rho1[0], sigma1)
        vrho_test = dedx[:, 0] + dedx[:, 1] * dp[0]
        vsigma_test = dedx[:, 1] * dp[1]
        assert_allclose(e_test / rho1[0], e_ref)
        assert_allclose(vrho_test[0], vxc[0])
        assert_allclose(vsigma_test[0], vxc[1])

        sfeat = np.stack([feat, feat])
        e_ref, vxc = eval_xc("GGA_C_PBE", (0.5 * rho1, 0.5 * rho1), spin=1)[:2]
        e_test, dedx = gga_c_pbe(sfeat)
        for s in range(2):
            for ind in range(2):
                fd = get_fd(ind, s, sfeat, gga_c_pbe)
                assert_allclose(dedx[s, ind], fd, atol=1e-7, rtol=1e-7)
        dp = ds2(rho1[0], sigma1)
        vrho_test = dedx[:, 0] + dedx[:, 1] * dp[0]
        vsigma_test = dedx[:, 1] * dp[1]
        assert_allclose(e_test / rho1[0], e_ref)
        assert_allclose(2 * vrho_test, vxc[0].T)
        # Normally would be a factor of 4, but since we have no
        # up-down cross term in sigma, only a factor of 2 needed
        assert_allclose(2 * vsigma_test, vxc[1].T[::2])

        sfeat = np.stack([feat, 0 * feat])
        e_ref, vxc = eval_xc("GGA_C_PBE", (0.5 * rho1, 0.0 * rho1), spin=1)[:2]
        e_test, dedx = gga_c_pbe(sfeat)
        s = 0
        for ind in range(2):
            fd = get_fd(ind, s, sfeat, gga_c_pbe)
            assert_allclose(dedx[s, ind], fd, atol=1e-7, rtol=1e-7)
        dp = ds2(rho1[0], sigma1)
        vrho_test = dedx[:, 0] + dedx[:, 1] * dp[0]
        vsigma_test = dedx[:, 1] * dp[1]
        assert_allclose(e_test * 2 / rho1[0], e_ref, atol=1e-7)
        assert_allclose(2 * vrho_test[0], vxc[0].T[0], atol=1e-7)
        assert_allclose(4 * vsigma_test[0], vxc[1].T[0], atol=1e-7)
        assert_allclose(2 * vrho_test[1], vxc[0].T[1], rtol=1e-2)
        _e = e_test
        _vrho = vrho_test
        _vsigma = vsigma_test

        sfeat = np.stack([0 * feat, feat])
        e_ref, vxc = eval_xc("GGA_C_PBE", (0.0 * rho1, 0.5 * rho1), spin=1)[:2]
        e_test, dedx = gga_c_pbe(sfeat)
        s = 1
        for ind in range(2):
            fd = get_fd(ind, s, sfeat, gga_c_pbe)
            assert_allclose(dedx[s, ind], fd, atol=1e-7, rtol=1e-7)
        dp = ds2(rho1[0], sigma1)
        vrho_test = dedx[:, 0] + dedx[:, 1] * dp[0]
        vsigma_test = dedx[:, 1] * dp[1]
        assert_allclose(e_test * 2 / rho1[0], e_ref, atol=1e-7)
        assert_allclose(2 * vrho_test[1], vxc[0].T[1], atol=1e-7)
        assert_allclose(4 * vsigma_test[1], vxc[1].T[2], atol=1e-7)
        assert_allclose(2 * vrho_test[0], vxc[0].T[0], rtol=1e-2)
        assert_allclose(e_test, _e)
        assert_allclose(vrho_test, _vrho[::-1])
        assert_allclose(vsigma_test, _vsigma[::-1])

        feat2 = np.stack([rho2[0], p2, 0.5 * p2, 0.25 * rho2[0]])
        sfeat = np.stack([feat, feat2])
        e_ref, vxc = eval_xc("GGA_C_PBE", (0.5 * rho1, 0.5 * rho2), spin=1)[:2]
        e_test, dedx = gga_c_pbe(sfeat)
        for s in range(2):
            for ind in range(2):
                fd = get_fd(ind, s, sfeat, gga_c_pbe)
                assert_allclose(dedx[s, ind], fd, atol=1e-7, rtol=1e-7)

        partition = 2 * np.random.uniform(size=20)
        rho1, rho2 = partition * rho1, (2 - partition) * rho1
        sigma1 = np.einsum("xg,xg->g", rho1[1:4], rho1[1:4])
        sigma2 = np.einsum("xg,xg->g", rho2[1:4], rho2[1:4])
        p1 = sigma1 / (4.0 * (3 * np.pi**2) ** (2.0 / 3) * rho1[0] ** (8.0 / 3))
        p2 = sigma2 / (4.0 * (3 * np.pi**2) ** (2.0 / 3) * rho2[0] ** (8.0 / 3))
        feat = np.stack([rho1[0], p1, 0.5 * p1, 0.25 * rho1[0]])
        feat2 = np.stack([rho2[0], p2, 0.5 * p2, 0.25 * rho2[0]])
        sfeat = np.stack([feat, feat2])
        e_ref, vxc = eval_xc("GGA_C_PBE", (0.5 * rho1, 0.5 * rho2), spin=1)[:2]
        e_test, dedx = gga_c_pbe(sfeat)
        for s in range(2):
            for ind in range(2):
                fd = get_fd(ind, s, sfeat, gga_c_pbe)
                assert_allclose(dedx[s, ind], fd, atol=1e-7, rtol=1e-7)
        dp1 = ds2(rho1[0], sigma1)
        dp2 = ds2(rho2[0], sigma2)
        vrho_test = dedx[:, 0] + dedx[:, 1] * np.stack([dp1[0], dp2[0]])
        vsigma_test = dedx[:, 1] * np.stack([dp1[1], dp2[1]])
        assert_allclose(e_test * 2.0 / (rho1[0] + rho2[0]), e_ref, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
