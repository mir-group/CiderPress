"""
The purpose of these tests is to make sure that all the spherical
harmonic derivative coefficients are working correctly.
"""

import ctypes
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

from ciderpress.dft.sph_harm_coeff import (
    clebsch_gordan,
    clebsch_gordan_e3nn,
    get_deriv_ylm_coeff,
    libcider,
)


def get_ylm(r, lmax):
    N = r.shape[0]
    nlm = (lmax + 1) * (lmax + 1)
    rnorm = np.linalg.norm(r, axis=1)
    rhat = r / rnorm[:, np.newaxis]
    ylm = np.zeros((N, nlm))
    libcider.recursive_sph_harm_vec(
        ctypes.c_int(nlm),
        ctypes.c_int(N),
        rhat.ctypes.data_as(ctypes.c_void_p),
        ylm.ctypes.data_as(ctypes.c_void_p),
    )
    for l in range(lmax + 1):
        for m in range(2 * l + 1):
            lm = l * l + m
            ylm[:, lm] *= rnorm**l
    return ylm


def get_ylm_fd(i, r, lmax, delta=1e-6):
    rx = r.copy()
    rx[:, i] += 0.5 * delta
    ylmp = get_ylm(rx, lmax)
    rx[:, i] -= delta
    ylmm = get_ylm(rx, lmax)
    dylm = (ylmp - ylmm) / delta
    return dylm


def get_ylm_ad(r, lmax):
    nlm = (lmax + 1) * (lmax + 1)
    rnorm = np.linalg.norm(r, axis=1)
    rhat = r / rnorm[:, np.newaxis]
    N = r.shape[0]
    ylm = np.zeros((N, nlm))
    dylm_tmp = np.zeros((N, 3, nlm))
    dylm = np.empty((4, N, nlm))
    libcider.recursive_sph_harm_deriv_vec(
        ctypes.c_int(nlm),
        ctypes.c_int(N),
        rhat.ctypes.data_as(ctypes.c_void_p),
        ylm.ctypes.data_as(ctypes.c_void_p),
        dylm_tmp.ctypes.data_as(ctypes.c_void_p),
    )
    dylm[0] = ylm
    dylm[1:] = dylm_tmp.transpose(1, 0, 2)
    for l in range(lmax + 1):
        rlm1 = rnorm ** (l - 1)
        rl = rnorm**l
        for m in range(2 * l + 1):
            lm = l * l + m
            dylm[0, :, lm] *= rl
            dylm[1:, :, lm] *= rlm1
            dylm[1:, :, lm] += ylm[:, lm] * l * rlm1 * rhat.T
    return dylm


class TestYlm(unittest.TestCase):
    def test_ylm(self):
        np.random.seed(32)
        lmax = 10
        N = 15
        nlm = (lmax + 1) * (lmax + 1)
        r = np.random.normal(size=(N, 3))
        dylm_ref = np.zeros((3, N, nlm))
        ylm0 = get_ylm(r, lmax)
        for i in range(3):
            dylm_ref[i] = get_ylm_fd(i, r, lmax)
        dylm_ref2 = get_ylm_ad(r, lmax)
        for lm in range(0, nlm):
            assert_allclose(ylm0[:, lm], dylm_ref2[0, :, lm], rtol=1e-6, atol=1e-5)
            assert_allclose(
                dylm_ref[0, :, lm], dylm_ref2[1, :, lm], rtol=1e-6, atol=1e-5
            )
            assert_allclose(
                dylm_ref[1, :, lm], dylm_ref2[2, :, lm], rtol=1e-6, atol=1e-5
            )
            assert_allclose(
                dylm_ref[2, :, lm], dylm_ref2[3, :, lm], rtol=1e-6, atol=1e-5
            )

        gaunt_coeff = get_deriv_ylm_coeff(lmax)
        ylm0_fit = np.zeros((4, N, nlm))
        igrad = 0
        for l in range(lmax):
            2 * l + 1
            for m in range(2 * l + 1):
                lm = l * l + m
                if igrad == 0:
                    lmt = lm
                else:
                    lmt = (l + 1) * (l + 1) - 1 - m
                lm0 = lmt + 2 * l + 1
                lm1 = lmt + 2 * l + 3
                if igrad == 0:
                    ylm0_fit[0, :, lm0] = ylm0[:, lm]
                    ylm0_fit[1, :, lm1] = ylm0[:, lm]
                else:
                    ylm0_fit[0, :, lm0] = ylm0[:, lm]
                    ylm0_fit[1, :, lm1] = ylm0[:, lm]
        for l in range(lmax + 1):
            2 * l + 1
            for m in range(2 * l + 1):
                lm = l * l + m
                X = ylm0_fit[:, :, lm].T
                X = []
                inds = []
                mm = m - l
                abs(mm)
                f1 = (
                    clebsch_gordan(l - 1, 1, l, mm - 1, 1, mm).n(64)
                    * ((2 * l + 1) * l) ** 0.5
                    / (2) ** 0.5
                )
                f2 = (
                    clebsch_gordan(l - 1, 1, l, mm + 1, -1, mm).n(64)
                    * ((2 * l + 1) * l) ** 0.5
                    / (2) ** 0.5
                    * -1
                )
                if l != 0:
                    fe = clebsch_gordan_e3nn(l - 1, 1, l)
                    fac = ((2 * l + 1) * l) ** 0.5
                    if igrad == 0:
                        igg = 2
                        mmm = m
                        m1m = m - 2
                        m1p = m
                    elif igrad == 1:
                        igg = 0
                        mmm = m
                        m1p = 2 * l - m - 2
                        m1m = 2 * l - m
                    else:
                        igg = 1
                        mmm = m
                        m1m = m - 1
                        m1p = m - 1
                    fxp = fe[m1p, igg, mmm] if mm < l - 1 else 0
                    fxp *= fac
                    fxm = fe[m1m, igg, mmm] if mm > -l + 1 else 0
                    fxm *= fac
                else:
                    fxp = fxm = 0
                if mm == 0:
                    f1 *= np.sqrt(2)
                    f2 *= np.sqrt(2)
                if igrad == 1:
                    mvals = [
                        mm + 1,
                        mm - 1,
                    ]
                else:
                    mvals = [
                        -mm - 1,
                        -mm + 1,
                    ]
                for i in range(2):
                    if abs(mvals[i]) >= l:
                        continue
                    if np.linalg.norm(ylm0_fit[i, :, lm]) == 0:
                        continue
                    for x in X:
                        if np.linalg.norm(x - ylm0_fit[i, :, lm]) < 1e-6:
                            break
                    else:
                        inds.append((i, mvals[i]))
                        X.append(ylm0_fit[i, :, lm])
                if len(X) == 0:
                    continue
                X = np.array(X).T
                y = dylm_ref[igrad, :, lm]
                beta = np.linalg.inv(X.T.dot(X) + 1e-10 * np.identity(X.shape[1]))
                beta = beta.dot(X.T.dot(y))
                err = np.linalg.norm(X.dot(beta) - y)
                assert_almost_equal(err, 0, 4)
        for lm in range(nlm):
            c_xl = np.zeros((N, nlm))
            c_xl[:, lm] = 1
            dylm_pred = np.zeros((3, N, nlm))
            libcider.fill_sph_harm_deriv_coeff(
                c_xl.ctypes.data_as(ctypes.c_void_p),
                dylm_pred.ctypes.data_as(ctypes.c_void_p),
                gaunt_coeff.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(N),
                ctypes.c_int(lmax),
            )
            dylm_pred = (dylm_pred * ylm0).sum(-1)
            assert_almost_equal(dylm_pred[2], dylm_ref2[3, :, lm], 6)
            assert_almost_equal(dylm_pred[0], dylm_ref2[1, :, lm], 6)
            assert_almost_equal(dylm_pred[1], dylm_ref2[2, :, lm], 6)


if __name__ == "__main__":
    unittest.main()
