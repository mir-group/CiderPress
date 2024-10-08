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

import ctypes

import numpy as np

from ciderpress.dft.settings import LDA_FACTOR
from ciderpress.lib import load_library

xc_helper = load_library("libxc_utils")


def nsp_rho_basline(X0T, e=None, dedx=None):
    nspin, nfeat, nsamp = X0T.shape
    if e is None:
        e = np.zeros(nsamp)
    if dedx is None:
        dedx = np.zeros_like(X0T)
    nspin = X0T.shape[0]
    e[:] += X0T[:, 0].mean(0)
    dedx[0, :] += 1.0 / nspin


def _lda_x_helper(X0T, e, dedx):
    rho = X0T[0]
    e[:] += LDA_FACTOR * rho ** (4.0 / 3)
    dedx[0] += 4.0 / 3 * LDA_FACTOR * rho ** (1.0 / 3)


def _vi_x_damp_helper(X0T, e, dedx):
    rho = X0T[0]
    damp = 2.0 / (1.0 + 0.5 * X0T[3]) ** 2
    ddamp = -1 * damp / (1.0 + 0.5 * X0T[3])
    e[:] += LDA_FACTOR * damp * rho ** (4.0 / 3)
    dedx[0] += 4.0 / 3 * LDA_FACTOR * damp * rho ** (1.0 / 3)
    dedx[3] += LDA_FACTOR * ddamp * rho ** (4.0 / 3)


def _pbe_x_helper(X0T, e, dedx):
    rho = X0T[0]
    s2 = X0T[1]
    kappa = 0.804
    mu = 0.2195149727645171
    mk = mu / kappa
    fac = 1.0 / (1 + mk * s2)
    fx = 1 + kappa - kappa * fac
    dfx = mu * fac * fac
    e[:] += LDA_FACTOR * rho ** (4.0 / 3) * fx
    dedx[0] += 4.0 / 3 * LDA_FACTOR * rho ** (1.0 / 3) * fx
    dedx[1] += LDA_FACTOR * rho ** (4.0 / 3) * dfx


def _chachiyo_x_helper(X0T, e, dedx):
    rho = X0T[0]
    s2 = X0T[1]
    c = 4 * np.pi / 9
    x = c * np.sqrt(s2)
    dx = c / (2 * np.sqrt(s2))
    chfx = (3 * x**2 + np.pi**2 * np.log(1 + x)) / (
        (np.pi**2 + 3 * x) * np.log(1 + x)
    )
    dchfx = (
        -3 * x**2 * (np.pi**2 + 3 * x)
        + 3 * x * (1 + x) * (2 * np.pi**2 + 3 * x) * np.log(1 + x)
        - 3 * np.pi**2 * (1 + x) * np.log(1 + x) ** 2
    ) / ((1 + x) * (np.pi**2 + 3 * x) ** 2 * np.log(1 + x) ** 2)
    dchfx *= dx
    chfx[s2 < 1e-8] = 1 + 8 * s2[s2 < 1e-8] / 27
    dchfx[s2 < 1e-8] = 8.0 / 27
    e[:] += LDA_FACTOR * rho ** (4.0 / 3) * chfx
    dedx[0] += 4.0 / 3 * LDA_FACTOR * rho ** (1.0 / 3) * chfx
    dedx[1] += LDA_FACTOR * rho ** (4.0 / 3) * dchfx


def _sl_x_helper(X0T, _helper):
    nspin, nfeat, nsamp = X0T.shape
    e = np.zeros(nsamp)
    dedx = np.zeros_like(X0T)
    nspin = X0T.shape[0]
    for s in range(nspin):
        _helper(X0T[s], e, dedx[s])
    e[:] /= nspin
    dedx[:] /= nspin
    return e, dedx


def zero_xc(X0T):
    nspin, nfeat, nsamp = X0T.shape
    e = np.zeros(nsamp)
    dedx = np.zeros_like(X0T)
    return e, dedx


def one_xc(X0T):
    nspin, nfeat, nsamp = X0T.shape
    e = np.ones(nsamp)
    dedx = np.zeros_like(X0T)
    return e, dedx


def lda_x(X0T):
    return _sl_x_helper(X0T, _lda_x_helper)


def nlda_x_damp(X0T):
    return _sl_x_helper(X0T, _vi_x_damp_helper)


def gga_x_chachiyo(X0T):
    return _sl_x_helper(X0T, _chachiyo_x_helper)


def gga_x_pbe(X0T):
    return _sl_x_helper(X0T, _pbe_x_helper)


def get_libxc_lda_baseline(xcid, rho):
    if isinstance(xcid, str):
        xcid = LDA_CODES[xcid]
    nspin, size = rho.shape
    rho = np.asfortranarray(rho)
    exc = np.zeros(size)
    vrho = np.zeros_like(rho, order="F")
    xc_helper.get_lda_baseline(
        ctypes.c_int(xcid),
        ctypes.c_int(nspin),
        ctypes.c_int(size),
        rho.ctypes.data_as(ctypes.c_void_p),
        exc.ctypes.data_as(ctypes.c_void_p),
        vrho.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(1e-12),
    )
    return exc, vrho


def get_libxc_gga_baseline(xcid, rho, sigma):
    if isinstance(xcid, str):
        xcid = GGA_CODES[xcid]
    nspin, size = rho.shape
    assert sigma.shape == (2 * nspin - 1, size)
    rho = np.asfortranarray(rho)
    sigma = np.asfortranarray(sigma)
    exc = np.zeros(size)
    vrho = np.zeros_like(rho, order="F")
    vsigma = np.zeros_like(sigma, order="F")
    xc_helper.get_gga_baseline(
        ctypes.c_int(xcid),
        ctypes.c_int(nspin),
        ctypes.c_int(size),
        rho.ctypes.data_as(ctypes.c_void_p),
        sigma.ctypes.data_as(ctypes.c_void_p),
        exc.ctypes.data_as(ctypes.c_void_p),
        vrho.ctypes.data_as(ctypes.c_void_p),
        vsigma.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(1e-12),
    )
    return exc, vrho, vsigma


def get_libxc_mgga_baseline(xcid, rho, sigma, tau):
    if isinstance(xcid, str):
        xcid = MGGA_CODES[xcid]
    nspin, size = rho.shape
    assert sigma.shape == (2 * nspin - 1, size)
    rho = np.asfortranarray(rho)
    # rho = np.ascontiguousarray(rho)
    sigma = np.asfortranarray(sigma)
    # sigma = np.ascontiguousarray(sigma)
    exc = np.zeros(size)
    vrho = np.zeros_like(rho, order="F")
    vsigma = np.zeros_like(sigma, order="F")
    vtau = np.zeros_like(tau, order="F")
    xc_helper.get_mgga_baseline(
        ctypes.c_int(xcid),
        ctypes.c_int(nspin),
        ctypes.c_int(size),
        rho.ctypes.data_as(ctypes.c_void_p),
        sigma.ctypes.data_as(ctypes.c_void_p),
        tau.ctypes.data_as(ctypes.c_void_p),
        exc.ctypes.data_as(ctypes.c_void_p),
        vrho.ctypes.data_as(ctypes.c_void_p),
        vsigma.ctypes.data_as(ctypes.c_void_p),
        vtau.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(1e-12),
    )
    return exc, vrho, vsigma, vtau


def get_libxc_baseline_ss(xcid, rho_tuple):
    nspin = rho_tuple[0].shape[0]
    assert xcid in SS_GGA_CODES
    if nspin == 2:
        _rho_tuple = tuple([r.copy(order="F") for r in rho_tuple])
        for r in _rho_tuple:
            r[1:] = 0.0
        exc_a, vrho_a, vsigma_a = get_libxc_gga_baseline(
            SS_GGA_CODES[xcid], _rho_tuple[0], _rho_tuple[1]
        )
        for r, ref in zip(_rho_tuple, rho_tuple):
            r[:1] = ref[-1]
        exc_b, vrho_b, vsigma_b = get_libxc_gga_baseline(
            SS_GGA_CODES[xcid], _rho_tuple[0], _rho_tuple[1]
        )
        exc = exc_a * rho_tuple[0][0] + exc_b * rho_tuple[0][1]
        vrho_a[1] = vrho_b[0]
        vsigma_a[1] = 0
        vsigma_a[2] = vsigma_b[0]
        vrho = vrho_a
        vsigma = vsigma_a
    else:
        assert nspin == 1
        ngrid = rho_tuple[0].shape[1]
        _rho_tuple = (np.zeros((2, ngrid), order="F"), np.zeros((3, ngrid), order="F"))
        _rho_tuple[0][0] = 0.5 * rho_tuple[0][0]
        _rho_tuple[1][0] = 0.25 * rho_tuple[1][0]
        exc, vrho, vsigma = get_libxc_gga_baseline(
            SS_GGA_CODES[xcid], _rho_tuple[0], _rho_tuple[1]
        )
        exc *= _rho_tuple[0][0] * 2
        vrho = vrho[:1]
        vsigma = 0.5 * vsigma[:1]
    return exc, vrho, vsigma


def get_libxc_baseline_os(xcid, rho_tuple):
    assert xcid in OS_GGA_CODES
    exc, vrho, vsigma = get_libxc_gga_baseline(
        OS_GGA_CODES[xcid], rho_tuple[0], rho_tuple[1]
    )
    exc *= rho_tuple[0].sum(0)
    ss_xcid = "SS_" + xcid[3:]
    exc_ss, vrho_ss, vsigma_ss = get_libxc_baseline_ss(ss_xcid, rho_tuple)
    exc[:] -= exc_ss
    vrho[:] -= vrho_ss
    vsigma[:] -= vsigma_ss
    return exc, vrho, vsigma


def get_libxc_baseline(xcid, rho_tuple):
    for r in rho_tuple:
        assert r.ndim == 2
    if xcid in LDA_CODES:
        res = get_libxc_lda_baseline(xcid, rho_tuple[0])
    elif xcid in GGA_CODES:
        res = get_libxc_gga_baseline(xcid, rho_tuple[0], rho_tuple[1])
    elif xcid in MGGA_CODES:
        res = get_libxc_mgga_baseline(xcid, rho_tuple[0], rho_tuple[1], rho_tuple[2])
    elif xcid in SS_GGA_CODES:
        return get_libxc_baseline_ss(xcid, rho_tuple)
    elif xcid in OS_GGA_CODES:
        return get_libxc_baseline_os(xcid, rho_tuple)
    else:
        raise ValueError("Unsupported xcid {}".format(xcid))
    res[0][:] *= rho_tuple[0].sum(0)
    return res


def get_sigma(X0T):
    const = 4 * (3 * np.pi**2) ** (2.0 / 3)
    rho = X0T[:, 0]
    p = X0T[:, 1]
    sigma_s = const * p * rho ** (8.0 / 3)
    nspin = X0T.shape[0]
    if nspin == 1:
        return rho, sigma_s
    else:
        assert nspin == 2
        sigma = np.zeros((3, X0T.shape[-1]), order="F")
        zeta = (rho[0] - rho[1]) / (rho[0] + rho[1])
        zfac = 0.5 * (1 - zeta * zeta) / (1 + zeta * zeta)
        sigma[0] = sigma_s[0]
        sigma[2] = sigma_s[1]
        sigma[1] = (sigma_s[0] + sigma_s[1]) * zfac
        return 0.5 * rho, 0.25 * sigma


def get_dsigma(X0T, vX0T, vrho, vsigma):
    const = 4 * (3 * np.pi**2) ** (2.0 / 3)
    rho = X0T[:, 0]
    p = X0T[:, 1]
    sigma_s = const * p * rho ** (8.0 / 3)
    nspin = X0T.shape[0]
    if nspin == 1:
        vsigma_s = vsigma
    else:
        assert nspin == 2
        vrho *= 0.5
        vsigma *= 0.25
        zeta = (rho[0] - rho[1]) / (rho[0] + rho[1])
        zfac = 0.5 * (1 - zeta * zeta) / (1 + zeta * zeta)
        dzfac = -2.0 * zeta / (1 + zeta * zeta) ** 2
        vsigma_s = vsigma[::2] + vsigma[1] * zfac
        vzfac = vsigma[1] * (sigma_s[0] + sigma_s[1]) * dzfac
        vX0T[0, 0] += vzfac * 2 * rho[1] / (rho[0] + rho[1]) ** 2
        vX0T[1, 0] -= vzfac * 2 * rho[0] / (rho[0] + rho[1]) ** 2
    vX0T[:, 0] += vrho + vsigma_s * const * p * (8.0 / 3) * rho ** (5.0 / 3)
    vX0T[:, 1] += vsigma_s * const * rho ** (8.0 / 3)


def get_gga_c(xcid, X0T):
    dedx = np.zeros_like(X0T)
    rho, sigma = get_sigma(X0T)
    X0T.shape[0]
    e, vrho, vsigma = get_libxc_gga_baseline(xcid, rho, sigma)
    get_dsigma(X0T, dedx, vrho, vsigma)
    return e * rho.sum(0), dedx


def gga_c_pbe(X0T):
    return get_gga_c(130, X0T)


def mgga_c_r2scan(X0T):
    raise NotImplementedError


BASELINE_CODES = {
    "RHO": nsp_rho_basline,
    "ZERO": zero_xc,
    "ONE": one_xc,
    "LDA_X": lda_x,
    "NLDA_X_DAMP": nlda_x_damp,
    "GGA_X_PBE": gga_x_pbe,
    "GGA_X_CHACHIYO": gga_x_chachiyo,
    "GGA_C_PBE": gga_c_pbe,
}

LDA_CODES = {
    "LDA_X": 1,
    "LDA_C_PW_MOD": 13,
}

GGA_CODES = {
    "GGA_X_PBE": 101,
    "GGA_C_PBE": 130,
    "GGA_X_PBE_SOL": 116,
    "GGA_C_PBE_SOL": 133,
}

MGGA_CODES = {
    "MGGA_X_R2SCAN": 497,
    "MGGA_C_R2SCAN": 498,
}

OS_GGA_CODES = {
    "OS_GGA_C_PBE": 130,
}

OS_MGGA_CODES = {
    "OS_MGGA_C_R2SCAN": 498,
}

SS_GGA_CODES = {
    "SS_GGA_C_PBE": 130,
}

SS_MGGA_CODES = {
    "SS_MGGA_C_R2SCAN": 498,
}
