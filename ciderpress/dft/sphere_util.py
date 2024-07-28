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

import numpy as np
import scipy.special as sp
from scipy.linalg import cho_solve, cholesky, solve_triangular

SQRTPI = np.sqrt(np.pi)

DEFAULT_TOL = 1e-7


def gauss_real(n, a, r):
    return r**n * np.exp(-a * r * r)


def gauss_fft(n, a, k):
    # \int_0^\infty dr r^(2+n) exp(-ar^2) j_n(k*r)
    return (
        2 ** (-2 - n) * a ** (-1.5 - n) * np.exp(-(k**2) / (4 * a)) * k**n * SQRTPI
    )


def gauss_conv_coef_and_expnt(n, alpha, beta):
    # res[0] r^n exp(-res[1] r^2) = \int d^3r' r'^n exp(-beta r^2) exp(-alpha (r-r')^2)
    return (np.pi / alpha) ** 1.5 * (alpha / (beta + alpha)) ** (
        1.5 + n
    ), beta * alpha / (beta + alpha)


def gauss_integral(n, a):
    return 0.5 * a ** (-1.5 - n) * sp.gamma(1.5 + n)


def incomplete_gauss_integral(n, a, r):
    ar2 = a * r * r
    return 0.5 * a ** (-1.5 - n) * sp.gamma(1.5 + n) * sp.gammaincc(1.5 + n, ar2)


def gauss_and_derivs(a, r):
    r = np.sqrt(a) * r
    """
    E**(-r**2)
    (-2*r)/E**r**2
    -2/E**r**2 + (4*r**2)/E**r**2
    (12*r)/E**r**2 - (8*r**3)/E**r**2
    12/E**r**2 - (48*r**2)/E**r**2 + (16*r**4)/E**r**2
    """
    sqe = np.exp(-(r**2))
    vals = np.stack(
        [
            sqe,
            (-2 * r) * sqe,
            -2 * sqe + (4 * r**2) * sqe,
            (12 * r) * sqe - (8 * r**3) * sqe,
            12 * sqe - (48 * r**2) * sqe + (16 * r**4) * sqe,
        ]
    )
    for n in range(vals.shape[0]):
        vals[n] *= np.sqrt(a) ** n
    return vals


def mat_vol(x):
    return np.abs(np.linalg.det(x))


def get_ng_est(ncut, alphas, h_cv, cell_cv, tol=DEFAULT_TOL, ngmul=1):
    """
    Estimate the number of real-space and reciprocal-space
    grid points needed to project the alphas Gaussian basis from an
    atom onto the FFT grid giving by spacing h_cv and cell cell_cv.
    ncut is the cutoff between short-range (real-space) and long-range
    (reciprocal-space) alphas. tol is the threshold below which
    a Gaussian is considered negligible
    """
    # alphas : exponents in **DESCENDING** order
    # ncut : cutoff between short-range basis and long-range
    vfac = 4 * np.pi / 3
    if ncut == -1:  # do everything in reciprocal space
        dvk = (2 * np.pi) ** 3 / mat_vol(cell_cv)
        kmax = np.pi / mat_vol(h_cv) ** (1.0 / 3)
        nk = vfac * kmax**3 / dvk
        print(nk, kmax, dvk)
        return int(np.ceil(nk)), 0.0, kmax
    elif ncut >= len(alphas) - 1:  # do everything in real space
        amin_sr = alphas[-1]
        dvr = mat_vol(h_cv)
        logeps = -np.log(tol)
        rmax = np.sqrt(logeps / amin_sr)
        ng = vfac * rmax**3 / dvr
        return int(np.ceil(ng)), rmax, 0.0
    amin_sr = alphas[ncut]
    amax_lr = alphas[ncut + 1]
    dvr = mat_vol(h_cv)
    dvk = (2 * np.pi) ** 3 / mat_vol(cell_cv)
    logeps = -np.log(tol)
    rmax = np.sqrt(logeps / amin_sr)
    kmax = np.sqrt(4 * amax_lr * logeps)
    ng = vfac * rmax**3 / dvr
    nk = vfac * kmax**3 / dvk
    # print(ng,nk,ng+nk,kmax,dvk)
    return int(ngmul * np.ceil(ng) + np.ceil(nk)), rmax, kmax


def get_ncut_for_kgauss_set(alphas, gd, tol=DEFAULT_TOL):
    ng_est = []
    rcuts = []
    kcuts = []
    for ncut in range(-1, len(alphas)):
        est, rmax, kmax = get_ng_est(ncut, alphas, gd.h_cv, gd.cell_cv, tol=tol)
        ng_est.append(est)
        rcuts.append(rmax)
        kcuts.append(kmax)
    # print('All estimates:', ng_est)
    # print('All rcuts:', rcuts)
    # print('All kcuts:', kcuts)
    ng_est, ncut = np.min(ng_est), np.argmin(ng_est) - 1
    # print('Estimated grid pts for projection = ', ng_est)
    # print('Selecting ncut = {}, with rmax = {} and kmax = {}'.format(ncut, rcuts[ncut], kcuts[ncut]))
    return ncut, rcuts[ncut], kcuts[ncut]


class GaussSetBase:
    def make_gaussf_smoothf(self):
        gaussf = []
        smoothf = []
        r_g = self.r_g
        self.dr_g
        gcut = self.gcut
        alphas = self.alphas
        rcut = r_g[gcut]
        for a in alphas:
            funcs = gauss_and_derivs(a, r_g)
            gaussf.append(funcs[0])
            sf = funcs[0].copy()
            fcut = funcs[:, gcut]
            rd_g = r_g - rcut
            poly = (
                fcut[0]
                + rd_g * fcut[1]
                + rd_g**2 * fcut[2] / 2
                + rd_g**3 * fcut[3] / 6
            )
            vpoly = poly[0]
            dpoly = fcut[1] - rcut * fcut[2] + rcut**2 * fcut[3] / 2
            x = -rcut
            if self.mode == "o3z":
                D = dpoly / x**3 - 5 * vpoly / x**4
                E = -dpoly / x**4 + 4 * vpoly / x**5
                poly += D * rd_g**4 + E * rd_g**5
            elif self.mode == "o2z":
                C = dpoly / x**2 - 4 * vpoly / x**3
                D = -dpoly / x**3 + 3 * vpoly / x**4
                poly += C * rd_g**3 + D * rd_g**4
            elif self.mode == "o3":
                poly += dpoly / (4 * rcut**3) * rd_g**4
            else:
                raise ValueError("Unrecognized fit mode")
            sf[:gcut] = poly[:gcut]
            smoothf.append(sf)
        sinv_list = []
        cl_list = []
        gaussf = np.stack(gaussf)
        smoothf = np.stack(smoothf)
        alpha_grid = alphas[:, None] + alphas
        if self.normalize:
            N = alphas**0.75
            gaussf *= N[:, None]
            smoothf *= N[:, None]
            norm_grid = N[:, None] * N
        else:
            norm_grid = np.ones_like(alpha_grid)
        for l in range(self.lmax + 1):
            # ovlp = np.einsum('ag,bg->ab', gaussf, gaussf * r_g**(2+2*l) * dr_g)
            ovlp = norm_grid * gauss_integral(l, alpha_grid)
            sinv_list.append(np.linalg.inv(ovlp))
            lower = True
            cl_list.append((cholesky(ovlp, lower=lower), lower))
        self.gaussf = gaussf
        self.smoothf = smoothf
        self.shortf = gaussf - smoothf
        self.sinv_list = sinv_list
        self.cl_list = cl_list


class GaussSet(GaussSetBase):
    def __init__(self, alphas, r_g, dr_g, gcut, normalize=True, mode="o2z", lmax=4):
        """
        Modes:
            o2z: Zero at origin, order 2 match
            o3z: Zero at origin, order 3 match
            o3 : Nonzero at origin, order 3 match
        """
        self.alphas = alphas
        self.r_g = r_g
        self.dr_g = dr_g
        self.gcut = gcut
        self.normalize = normalize
        self.mode = mode
        self.lmax = lmax
        self.make_gaussf_smoothf()

    def project(self, funcs, l):
        return np.einsum(
            "...g,ag->...a", funcs, self.gaussf * self.r_g ** (2 + l) * self.dr_g
        )

    def _get_smooth_helper_1(self, gauss_proj, l):
        return gauss_proj @ self.sinv_list[l]

    def _get_smooth_helper_2(self, coef, smoothf, l):
        return self.r_g**l * np.einsum("ag,...a->...g", smoothf, coef)

    def get_smoothf_shortf(self, funcs, l):
        gauss_proj = self.project(funcs, l)
        coef = self._get_smooth_helper_1(gauss_proj, l)
        return self._get_smooth_helper_2(
            coef, self.smoothf, l
        ), self._get_smooth_helper_2(coef, self.shortf, l)

    def get_smoothf_gaussf(self, funcs, l):
        gauss_proj = self.project(funcs, l)
        coef = self._get_smooth_helper_1(gauss_proj, l)
        return self._get_smooth_helper_2(
            coef, self.smoothf, l
        ), self._get_smooth_helper_2(coef, self.gaussf, l)

    def get_smoothf(self, funcs, l):
        gauss_proj = self.project(funcs, l)
        coef = self._get_smooth_helper_1(gauss_proj, l)
        return self._get_smooth_helper_2(coef, self.smoothf, l)


class KGaussSet(GaussSetBase):
    def __init__(
        self,
        alphas,
        sbtgd,
        gcut,
        ncut,
        normalize=True,
        mode="o2z",
        lmax=4,
        tol=DEFAULT_TOL,
    ):
        """
        Modes:
            o2z: Zero at origin, order 2 match
            o3z: Zero at origin, order 3 match
            o3 : Nonzero at origin, order 3 match
        """
        self.alphas = alphas
        self.ncut = ncut
        self.r_g = sbtgd.r_g
        self.dr_g = sbtgd.dr_g
        self.k_g = sbtgd.k_g
        self.dk_g = sbtgd.dk_g
        self.gcut = gcut
        self.normalize = normalize
        self.mode = mode
        self.lmax = lmax
        self.tol = tol
        self.make_gaussf_smoothf()
        self._transform_basis()

    def _transform_basis(self):
        ncut = self.ncut
        self.srf_l = []
        self.mrf_l = []
        self.lrf_l = []
        self.gaussk_l = []
        # self.gaussf_l = []
        # self.shortf_l = []
        # self.smoothf_l = []
        for l in range(self.lmax + 1):
            self.gaussk_l.append(gauss_fft(l, self.alphas[:, None], self.k_g))
            if self.normalize:
                self.gaussk_l[-1] *= self.alphas[:, None] ** 0.75

            # sinv = self.sinv_list[l]
            cl = self.cl_list[l]
            rl = self.r_g**l
            f = rl * self.shortf
            # srf = sinv@f
            srf = cho_solve(cl, f)
            f = rl * self.smoothf
            # mrf = sinv@f
            mrf = cho_solve(cl, f)

            f = self.gaussk_l[-1]
            # lrf = sinv[:,ncut+1:]@f[ncut+1:]
            lrf = solve_triangular(
                cl[0][ncut + 1 :, ncut + 1 :], f[ncut + 1 :], lower=True
            )
            ndiff = f.shape[0] - lrf.shape[0]
            my_zeros = np.zeros((ndiff, lrf.shape[-1]))
            lrf = np.append(my_zeros, lrf, axis=0)
            lrf = solve_triangular(cl[0].T, lrf, lower=False)

            f = self.r_g**l * self.gaussf
            # mrf -= sinv[:,ncut+1:]@f[ncut+1:]
            tmp = solve_triangular(
                cl[0][ncut + 1 :, ncut + 1 :], f[ncut + 1 :], lower=True
            )
            tmp = np.append(my_zeros, tmp, axis=0)
            tmp = solve_triangular(cl[0].T, tmp, lower=False)
            mrf -= tmp
            self.srf_l.append(srf)
            self.mrf_l.append(mrf)
            self.lrf_l.append(lrf)

    def project(self, funcs_k, l):
        # TODO why is the factor of 2/pi here again?
        # I think because \int f(k) g(k) d^3k has (4pi)^2 from the unit sphere and sph_harm, then the FT has a denominator 8pi^3
        return (
            2
            / np.pi
            * np.einsum(
                "...k,ak->...a", funcs_k, self.gaussk_l[l] * self.k_g**2 * self.dk_g
            )
        )

    def project_real(self, funcs, l):
        return np.einsum(
            "...g,ag->...a", funcs, self.gaussf * self.r_g ** (2 + l) * self.dr_g
        )

    def get_components(self, funcs, l, recip=True):
        if recip:
            proj = self.project(funcs, l)
        else:
            proj = self.project_real(funcs, l)
        # TODO this is incorrect because of transform of lrf
        lrf_k = np.einsum("...a,ak->...k", proj, self.lrf_l[l])
        mrf_g = np.einsum("...a,ag->...g", proj, self.mrf_l[l])
        srf_g = np.einsum("...a,ag->...g", proj, self.srf_l[l])
        return lrf_k, mrf_g, srf_g


def make_gaussf_smoothf(alphas, r_g, dr_g, gcut):
    gaussf = []
    smoothf = []
    rcut = r_g[gcut]
    for a in alphas:
        funcs = gauss_and_derivs(a, r_g)
        gaussf.append(funcs[0])
        sf = funcs[0].copy()
        fcut = funcs[:, gcut]
        rd_g = r_g - rcut
        poly = (
            fcut[0] + rd_g * fcut[1] + rd_g**2 * fcut[2] / 2 + rd_g**3 * fcut[3] / 6
        )
        dpoly = fcut[1] - rcut * fcut[2] + rcut**2 * fcut[3] / 2
        poly += dpoly / (4 * rcut**3) * rd_g**4
        sf[:gcut] = poly[:gcut]
        smoothf.append(sf)
    sinv_list = []
    gaussf = np.stack(gaussf)
    smoothf = np.stack(smoothf)
    for l in range(8):
        N = alphas**0.75
        ovlp = np.einsum(
            "a,b,ag,bg->ab", N, N, gaussf, gaussf * r_g ** (2 + 2 * l) * dr_g
        )
        sinv_list.append(np.linalg.inv(ovlp))
    return gaussf, smoothf, sinv_list


def project(funcs, gaussf, alphas, l, r_g, dr_g):
    N = alphas**0.75
    return np.einsum("a,...g,ag->...a", N, funcs, gaussf * r_g ** (2 + l) * dr_g)


def get_smooth_helper_1(gauss_proj, sinv):
    return gauss_proj @ sinv


def get_smooth_helper_2(coef, smoothf, l, r_g, alphas):
    N = alphas**0.75
    return r_g**l * np.einsum("a,ag,...a->...g", N, smoothf, coef)


def get_smooth(funcs, gaussf, smoothf, l, r_g, dr_g, sinv):
    gauss_proj = project(funcs, gaussf, l, r_g, dr_g)
    coef = get_smooth_helper_1(gauss_proj, sinv)
    return get_smooth_helper_2(coef, smoothf, l, r_g)
