import ctypes

import numpy as np
from sympy.physics.wigner import clebsch_gordan, real_gaunt

from ciderpress.lib import load_library as load_cider_library

libcider = load_cider_library("libmcider")


def change_basis_real_to_complex(l: int) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / np.sqrt(2)
        q[l + m, l - abs(m)] = -1j / np.sqrt(2)
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / np.sqrt(2)
        q[l + m, l - abs(m)] = 1j * (-1) ** m / np.sqrt(2)
    return (
        -1j
    ) ** l * q  # Added factor of 1j**l to make the Clebsch-Gordan coefficients real


def su2_clebsch_gordan(l1, l2, l3):
    mat = np.zeros(
        [
            2 * l1 + 1,
            2 * l2 + 1,
            2 * l3 + 1,
        ]
    )
    for m1 in range(2 * l1 + 1):
        for m2 in range(2 * l2 + 1):
            for m3 in range(2 * l3 + 1):
                mat[m1, m2, m3] = clebsch_gordan(l1, l2, l3, m1 - l1, m2 - l2, m3 - l3)
    return mat


def clebsch_gordan_e3nn(l1: int, l2: int, l3: int) -> np.ndarray:
    r"""
    The Clebsch-Gordan coefficients of the real irreducible representations of :math:`SO(3)`.
    This function taken from https://github.com/e3nn/e3nn

    Args:
        l1 (int): the representation order of the first irrep
        l2 (int): the representation order of the second irrep
        l3 (int): the representation order of the third irrep

    Returns:
        np.ndarray: the Clebsch-Gordan coefficients
    """
    C = su2_clebsch_gordan(l1, l2, l3)
    Q1 = change_basis_real_to_complex(l1)
    Q2 = change_basis_real_to_complex(l2)
    Q3 = change_basis_real_to_complex(l3)
    C = np.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, np.conj(Q3.T), C)

    assert np.all(np.abs(np.imag(C)) < 1e-5)
    return np.real(C)


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
    xyz = ylm[:, 1:4].copy()
    ylm[:, 3] = xyz[:, 0]
    ylm[:, 1] = xyz[:, 1]
    ylm[:, 2] = xyz[:, 2]
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
    print("N", N)
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
    subval = np.einsum("vgl,gv->gl", dylm[1:], rhat)
    print("SUBVAL", np.linalg.norm(subval))
    for l in range(lmax + 1):
        rlm1 = rnorm ** (l - 1)
        rl = rnorm**l
        for m in range(2 * l + 1):
            lm = l * l + m
            dylm[0, :, lm] *= rl
            dylm[1:, :, lm] *= rlm1
            dylm[1:, :, lm] += ylm[:, lm] * l * rlm1 * rhat.T
    xyz = dylm[..., 1:4].copy()
    dylm[..., 3] = xyz[..., 0]
    dylm[..., 1] = xyz[..., 1]
    dylm[..., 2] = xyz[..., 2]
    return dylm


def get_deriv_ylm_coeff(lmax):
    nlm = (lmax + 1) * (lmax + 1)
    gaunt_coeff = np.zeros((5, nlm))
    for l in range(lmax + 1):
        fe = clebsch_gordan_e3nn(l, 1, l + 1)
        for im in range(2 * l + 1):
            lm = l * l + im
            m = abs(im - l)
            l1m = l + 1 - m
            fac = ((2 * l + 3) * (l + 1)) ** 0.5
            gaunt_coeff[0, lm] = fac * fe[im, 2, im]
            gaunt_coeff[1, lm] = fac * fe[im, 2, im + 2]
            gaunt_coeff[2, lm] = fac * fe[im, 0, 2 * l - im]
            gaunt_coeff[3, lm] = fac * fe[im, 0, 2 * l - im + 2]
            gaunt_coeff[4, lm] = np.sqrt(
                ((2.0 * l + 3) * l1m * (l + 1 + m) / (2 * l + 1))
            )
    return gaunt_coeff


def get_ylm1_coeff(lmax, plus=True):
    nlm = (lmax + 1) * (lmax + 1)
    gaunt_coeff = np.zeros((5, nlm))
    if plus:
        sign = 1
    else:
        sign = -1
    for l in range(lmax + 1):
        for im in range(2 * l + 1):
            lm = l * l + im
            m = im - l
            lp1 = l + sign * 1
            gaunt_coeff[0, lm] = real_gaunt(1, l, lp1, 1, m, m - 1)
            gaunt_coeff[1, lm] = real_gaunt(1, l, lp1, 1, m, m + 1)
            gaunt_coeff[2, lm] = real_gaunt(1, l, lp1, -1, m, m - 1)
            gaunt_coeff[3, lm] = real_gaunt(1, l, lp1, -1, m, m + 1)
            gaunt_coeff[4, lm] = real_gaunt(1, l, lp1, 0, m, m)
    return gaunt_coeff


def get_deriv_ylm_coeff_v2(lmax):
    nlm = (lmax + 1) * (lmax + 1)
    gaunt_coeff = np.zeros((5, nlm))
    for l in range(lmax + 1):
        fe = clebsch_gordan_e3nn(l, 1, l + 1)
        for im in range(2 * l + 1):
            lm = l * l + im
            m = abs(im - l)
            l1m = l + 1 - m
            fac = ((2 * l + 3) * (l + 1)) ** 0.5
            gaunt_coeff[0, lm] = fac * fe[im, 2, im]
            gaunt_coeff[1, lm] = fac * fe[im, 2, im + 2]
            gaunt_coeff[2, lm] = fac * fe[im, 0, 2 * l - im]
            gaunt_coeff[3, lm] = fac * fe[im, 0, 2 * l - im + 2]
            # gaunt_coeff[:4, lm] *= np.sqrt(2 * m + 1) * (-1)**l
            gaunt_coeff[4, lm] = np.sqrt(
                ((2.0 * l + 3) * l1m * (l + 1 + m) / (2 * l + 1))
            )
    return gaunt_coeff


if __name__ == "__main__":
    from numpy.testing import assert_allclose, assert_almost_equal

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
        assert_allclose(dylm_ref[0, :, lm], dylm_ref2[1, :, lm], rtol=1e-6, atol=1e-5)
        assert_allclose(dylm_ref[1, :, lm], dylm_ref2[2, :, lm], rtol=1e-6, atol=1e-5)
        assert_allclose(dylm_ref[2, :, lm], dylm_ref2[3, :, lm], rtol=1e-6, atol=1e-5)

    import time

    t0 = time.monotonic()
    gaunt_coeff = get_deriv_ylm_coeff(lmax)
    t1 = time.monotonic()
    print("TIME TO GEN COEFF", t1 - t0)
    print("GAUNT", gaunt_coeff)
    ylm0_fit = np.zeros((4, N, nlm))
    igrad = 0
    for l in range(lmax):
        mmax = 2 * l + 1
        for m in range(2 * l + 1):
            """
            if igrad == 0:
                lm = l * l + m
            else:
                lm = (l+1) * (l+1) - 1 - m
            lm0 = lm + 2 * l + 1
            lm1 = lm + 2 * l + 3
            ylm0_fit[0, :, lm0] = ylm0[:, lm]
            ylm0_fit[1, :, lm1] = ylm0[:, lm]
            """
            lm = l * l + m
            if igrad == 0:
                lmt = lm
            else:
                lmt = (l + 1) * (l + 1) - 1 - m
            lm0 = lmt + 2 * l + 1
            lm1 = lmt + 2 * l + 3
            # lm2 = lmh + 2 * l + 1
            # lm3 = lmh + 2 * l + 3
            if igrad == 0:
                ylm0_fit[0, :, lm0] = ylm0[:, lm]
                ylm0_fit[1, :, lm1] = ylm0[:, lm]
            else:
                ylm0_fit[0, :, lm0] = ylm0[:, lm]
                ylm0_fit[1, :, lm1] = ylm0[:, lm]
    for l in range(lmax + 1):
        mmax = 2 * l + 1
        for m in range(2 * l + 1):
            lm = l * l + m
            X = ylm0_fit[:, :, lm].T
            X = []
            inds = []
            mm = m - l
            am = abs(mm)
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
                print(mm, m, l, fe.shape)
                # print(fe)
                fac = ((2 * l + 1) * l) ** 0.5
                if igrad == 0:
                    igg = 2
                    mmm = m
                    m1m = m - 2
                    m1p = m
                elif igrad == 1:
                    igg = 0
                    mmm = m
                    # fac *= -1
                    m1p = 2 * l - m - 2
                    m1m = 2 * l - m
                else:
                    igg = 1
                    mmm = m
                    m1m = m - 1
                    m1p = m - 1
                # igg = 2 if igrad == 0 else 0
                # mmm = m if igrad == 0 else 2 * l - m
                fxp = fe[m1p, igg, mmm] if mm < l - 1 else 0
                fxp *= fac
                fxm = fe[m1m, igg, mmm] if mm > -l + 1 else 0
                fxm *= fac
            else:
                fxp = fxm = 0
            if mm == 0:
                f1 *= np.sqrt(2)
                f2 *= np.sqrt(2)
            # print('FACTORC', float(f1), float(f2))
            print("FACTORB", fxp, fxm)
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
            # print('MVALS', mvals)
            for i in range(2):
                # print('MM', m, mm, mvals[i])
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
            # print(X.shape, y.shape)
            beta = np.linalg.inv(X.T.dot(X) + 1e-10 * np.identity(X.shape[1]))
            beta = beta.dot(X.T.dot(y))
            err = np.linalg.norm(X.dot(beta) - y)
            assert_almost_equal(err, 0, 4)
            print(l, m - l, inds, lm, np.round(beta, 6), err)
            # print(gaunt_coeff[2, lm], err)
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
        print(lm)
        # print(r[:, 2])
        # print(ylm0[:, lm])
        # print(dylm_pred[2])
        # print(dylm_ref[2, :, lm])
        assert_almost_equal(dylm_pred[2], dylm_ref2[3, :, lm], 6)
        assert_almost_equal(dylm_pred[0], dylm_ref2[1, :, lm], 6)
        assert_almost_equal(dylm_pred[1], dylm_ref2[2, :, lm], 6)
