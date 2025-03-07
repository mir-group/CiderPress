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
from abc import ABC, abstractmethod

import numpy as np
import scipy.linalg.lapack
from pyscf import lib as pyscflib
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.special import gamma

from ciderpress import lib
from ciderpress.dft.settings import (
    NLDFSettings,
    SDMXBaseSettings,
    SDMXFullSettings,
    dalpha,
    ds2,
    get_alpha,
    get_cider_exponent,
    get_cider_exponent_gga,
    get_s2,
)
from ciderpress.lib import load_library as load_cider_library

libcider = load_cider_library("libmcider")


# map for C code
VJ_ID_MAP = {
    "se": 0,
    "se_ar2": 1,
    "se_a2r4": 2,
    "se_erf_rinv": 3,
}
VI_ID_MAP = {
    "se": 0,
    "se_r2": 1,
    "se_apr2": 2,
    "se_ap": 3,
    "se_ap2r2": 4,
    "se_lapl": 5,
    "se_rvec": 6,
    "se_grad": 7,
}


def _get_ovlp_fit_interpolation_coefficients(
    plan, arg_g, i=-1, local=True, vbuf=None, dbuf=None
):
    """

    Args:
        plan (NLDFAuxiliaryPlan):
        arg_g:
        i:
        local:
        vbuf:
        dbuf:

    Returns:

    """
    ngrids = arg_g.size
    p = plan.empty_coefs(ngrids, local=local, buf=vbuf)
    dp = plan.empty_coefs(ngrids, local=local, buf=dbuf)
    if i == -1:
        feat_id = VJ_ID_MAP["se"]
    else:
        assert 0 <= i < plan.nldf_settings.num_feat_param_sets
        feat_id = VJ_ID_MAP[plan.nldf_settings.feat_spec_list[i]]
    if feat_id == VJ_ID_MAP["se_erf_rinv"]:
        extra_args = plan._get_extra_args(i)
        num_extra_args = len(extra_args)
        extra_args = (ctypes.c_double * num_extra_args)(*extra_args)
    else:
        extra_args = lib.c_null_ptr()
    alphas = plan.local_alphas if local else plan.alphas
    if plan.coef_order == "gq":
        fn = libcider.cider_coefs_gto_gq
        shape = (arg_g.size, len(alphas))
        assert p.shape == shape
        assert dp.shape == shape
    else:
        fn = libcider.cider_coefs_gto_qg
        shape = (len(alphas), arg_g.size)
        assert p.shape == shape
        assert dp.shape == shape
    assert p.flags.c_contiguous
    assert dp.flags.c_contiguous
    assert arg_g.flags.c_contiguous
    assert alphas.flags.c_contiguous
    fn(
        p.ctypes.data_as(ctypes.c_void_p),
        dp.ctypes.data_as(ctypes.c_void_p),
        arg_g.ctypes.data_as(ctypes.c_void_p),
        alphas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(arg_g.size),
        ctypes.c_int(len(alphas)),
        ctypes.c_int(feat_id),
        extra_args,
    )
    return p, dp


def _stable_solve(a, b):
    c_and_l = cho_factor(a)
    return cho_solve(c_and_l, b)


def _construct_cubic_splines(coefs_qi, dense_indexes):
    """
    Construct interpolating splines for NLDFAuxPlanSpline.

    The recipe is from

      https://en.wikipedia.org/wiki/Spline_(mathematics)

    Adapted from GPAW vdW implementation
    """
    q = dense_indexes
    n, N = coefs_qi.shape

    # shape N,n
    y = coefs_qi.T
    a = y
    h = q[1:] - q[:-1]
    # h = np.ones(q.size - 1) * (n - 1) / (N - 1)
    alpha = 3 * (
        (a[2:] - a[1:-1]) / h[1:, np.newaxis] - (a[1:-1] - a[:-2]) / h[:-1, np.newaxis]
    )
    l = np.ones((N, n))
    mu = np.zeros((N, n))
    z = np.zeros((N, n))
    for i in range(1, N - 1):
        l[i] = 2 * (q[i + 1] - q[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / l[i]
    b = np.zeros((N, n))
    c = np.zeros((N, n))
    d = np.zeros((N, n))
    for i in range(N - 2, -1, -1):
        c[i] = z[i] - mu[i] * c[i + 1]
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / 3 / h[i]
    C_aip = np.zeros((n, N, 4))
    C_aip[:, :-1, 0] = a[:-1].T
    C_aip[:, :-1, 1] = b[:-1].T
    C_aip[:, :-1, 2] = c[:-1].T
    C_aip[:, :-1, 3] = d[:-1].T
    C_aip[-1, -1, 0] = 1.0
    return np.ascontiguousarray(C_aip.transpose(1, 0, 2))


def get_rho_tuple_with_grad_cross(rho_data, is_mgga=False):
    assert rho_data.ndim == 3
    drho = rho_data[:, 1:4]
    nspin = rho_data.shape[0]
    sigma = np.empty((2 * nspin - 1, rho_data.shape[-1]), order="F")
    sigma[::2] = np.einsum("sxg,sxg->sg", drho, drho)
    if nspin == 2:
        sigma[1] = np.einsum("xg,xg->g", drho[0], drho[1])
    rho = rho_data[:, 0].copy(order="F")
    if is_mgga:
        return rho, sigma, rho_data[:, 4].copy(order="F")
    else:
        return rho, sigma


def vxc_tuple_to_array(rho_data, vtuple):
    varr = np.zeros_like(rho_data)
    varr[:, 0] = vtuple[0]
    if len(vtuple) > 1:
        varr[:, 1:4] = 2 * vtuple[1][::2, None] * rho_data[:, 1:4]
        if len(vtuple) == 3:
            varr[:, 4] = vtuple[2]
        if rho_data.shape[0] == 2:
            varr[:, 1:4] += vtuple[1][1] * rho_data[::-1, 1:4]
    return varr


def get_rho_tuple(rho_data, with_spin=False, is_mgga=False):
    if with_spin:
        drho = rho_data[:, 1:4]
        sigma = np.einsum("sx...,sx...->s...", drho, drho)
        if is_mgga:
            return rho_data[:, 0].copy(), sigma, rho_data[:, 4].copy()
        else:
            return rho_data[:, 0].copy(), sigma
    else:
        drho = rho_data[1:4]
        sigma = np.einsum("x...,x...->...", drho, drho)
        if is_mgga:
            return rho_data[0], sigma, rho_data[4]
        else:
            return rho_data[0], sigma


def get_drhodf_tuple(rho_data, drhodf_data, with_spin=False, is_mgga=False):
    if with_spin:
        raise NotImplementedError("Have not implemented this function for with_spin")
        drho = rho_data[:, 1:4]
        ddrhodf = drhodf_data[:, 1:4]
        dsigmadf = 2 * np.einsum("sx...,sx...->s...", drho, ddrhodf)
        if is_mgga:
            return drhodf_data[:, 0].copy(), dsigmadf, drhodf_data[:, 4].copy()
        else:
            return drhodf_data[:, 0].copy(), dsigmadf
    else:
        drho = rho_data[1:4]
        ddrhodf = drhodf_data[1:4]
        dsigmadf = 2 * np.einsum("x...,x...->...", drho, ddrhodf)
        if is_mgga:
            return drhodf_data[0], dsigmadf, drhodf_data[4]
        else:
            return drhodf_data[0], dsigmadf


def get_ccl_settings(plan):
    has_vj = "j" in plan.nldf_settings.nldf_type
    ifeat_ids = []
    for spec in plan.nldf_settings.l0_feat_specs:
        ifeat_ids.append(VI_ID_MAP[spec])
    for spec in plan.nldf_settings.l1_feat_specs:
        ifeat_ids.append(VI_ID_MAP[spec])
    return has_vj, ifeat_ids


class _BaseSemilocalPlan:
    def __init__(self, settings, nspin):
        """
        Args:
            settings (SemilocalSettings): Settings for SL features
            nspin (int): Number of spin channels.
                1 for non-spin-polarized and 2 for spin-polarized
        """
        self.settings = settings
        self.nspin = nspin
        if self.settings.mode not in ["npa", "nst", "np", "ns"]:
            raise NotImplementedError

    @property
    def level(self):
        return self.settings.level

    def _fill_feat_npa_(self, rho, sigma, tau, feat):
        assert feat.shape[1] == 3
        feat[:, 0] = rho
        feat[:, 1] = get_s2(rho, sigma)
        feat[:, 2] = get_alpha(rho, sigma, tau)
        feat[:, 0] *= self.nspin
        feat[:, 1:3] /= self.nspin ** (2.0 / 3)

    def _fill_occd_npa_(
        self, feat, occd, rho, drho, sigma, dsigma, tau=None, dtau=None
    ):
        dpdn, dpdsigma = ds2(rho, sigma)
        feat[:, 0] = rho
        feat[:, 1] = get_s2(rho, sigma)
        occd[:, 0] = drho
        occd[:, 1] = dpdn * drho + dpdsigma * dsigma
        if self.level == "MGGA":
            feat[:, 2] = get_alpha(rho, sigma, tau)
            dadn, dadsigma, dadtau = dalpha(rho, sigma, tau)
            occd[:, 2] = dadn * drho + dadsigma * dsigma + dadtau * dtau
        feat[:, 0] *= self.nspin
        feat[:, 1:3] /= self.nspin ** (2.0 / 3)
        occd[:, 0] *= self.nspin
        occd[:, 1:3] /= self.nspin ** (2.0 / 3)

    def _fill_feat_nst_(self, rho, sigma, tau, feat):
        assert feat.shape[1] == 3
        feat[:, 0] = rho
        feat[:, 1] = sigma
        feat[:, 2] = tau
        feat[:, 0] *= self.nspin
        feat[:, 1] *= self.nspin * self.nspin
        feat[:, 2] *= self.nspin

    def _fill_occd_nst_(
        self, feat, occd, rho, drho, sigma, dsigma, tau=None, dtau=None
    ):
        feat[:, 0] = self.nspin * rho
        occd[:, 0] = self.nspin * drho
        feat[:, 1] = sigma * (self.nspin * self.nspin)
        occd[:, 1] = dsigma * (self.nspin * self.nspin)
        if self.level == "MGGA":
            feat[:, 2] = tau * self.nspin
            occd[:, 2] = dtau * self.nspin

    def _fill_feat_np_(self, rho, sigma, feat):
        assert feat.shape[1] == 2
        feat[:, 0] = rho
        feat[:, 1] = get_s2(rho, sigma)
        feat[:, 0] *= self.nspin
        feat[:, 1] /= self.nspin ** (2.0 / 3)

    def _fill_feat_ns_(self, rho, sigma, feat):
        assert feat.shape[1] == 2
        feat[:, 0] = rho
        feat[:, 1] = sigma
        feat[:, 0] *= self.nspin
        feat[:, 1] *= self.nspin * self.nspin

    def get_feat(self, rho, sigma, tau=None):
        for arr in [rho, sigma, tau]:
            assert arr is None or arr.ndim == 2
        feat = np.empty((rho.shape[0], self.settings.nfeat, rho.shape[1]))
        if self.settings.mode == "npa":
            self._fill_feat_npa_(rho, sigma, tau, feat)
        elif self.settings.mode == "nst":
            self._fill_feat_nst_(rho, sigma, tau, feat)
        elif self.settings.mode == "np":
            self._fill_feat_np_(rho, sigma, feat)
        else:
            self._fill_feat_ns_(rho, sigma, feat)
        return feat

    def get_occd(self, rho, drho, sigma, dsigma, tau=None, dtau=None):
        for arr in [rho, drho, sigma, dsigma, tau, dtau]:
            assert arr is None or arr.ndim == 2
        feat = np.empty((rho.shape[0], self.settings.nfeat, rho.shape[1]))
        occd = np.empty((rho.shape[0], self.settings.nfeat, rho.shape[1]))
        if self.settings.mode in ["npa", "np"]:
            self._fill_occd_npa_(feat, occd, rho, drho, sigma, dsigma, tau, dtau)
        elif self.settings.mode in ["nst", "ns"]:
            self._fill_occd_nst_(feat, occd, rho, drho, sigma, dsigma, tau, dtau)
        return feat, occd


class SemilocalPlan(_BaseSemilocalPlan):
    def get_feat(self, rho):
        sigma = np.einsum("sx...,sx...->s...", rho[:, 1:4], rho[:, 1:4])
        if self.level == "MGGA":
            tau = rho[:, 4]
        else:
            tau = None
        return super(SemilocalPlan, self).get_feat(rho[:, 0], sigma, tau)

    def get_occd(self, rho, drho):
        assert rho.ndim == drho.ndim == 3
        sigma = np.einsum("sx...,sx...->s...", rho[:, 1:4], rho[:, 1:4])
        dsigma = 2 * np.einsum("sx...,sx...->s...", rho[:, 1:4], drho[:, 1:4])
        if self.level == "MGGA":
            tau = rho[:, 4]
            dtau = drho[:, 4]
        else:
            tau = None
            dtau = None
        return super(SemilocalPlan, self).get_occd(
            rho[:, 0], drho[:, 0], sigma, dsigma, tau, dtau
        )

    def _fill_vxc_npa_(self, rho, vfeat, vxc):
        sigma = np.einsum("sx...,sx...->s...", rho[:, 1:4], rho[:, 1:4])
        sm23 = self.nspin ** (-2.0 / 3)
        dpdn, dpdsigma = ds2(rho[:, 0], sigma)
        dadn, dadsigma, dadtau = dalpha(rho[:, 0], sigma, rho[:, 4])
        vxc[:, 0] += self.nspin * vfeat[:, 0] + sm23 * (
            vfeat[:, 1] * dpdn + vfeat[:, 2] * dadn
        )
        vxc[:, 1:4] += (
            (sm23 * (vfeat[:, 1] * dpdsigma + vfeat[:, 2] * dadsigma))[:, None, :]
            * 2
            * rho[:, 1:4]
        )
        vxc[:, 4] += sm23 * vfeat[:, 2] * dadtau

    def _fill_vxc_nst_(self, rho, vfeat, vxc):
        vxc[:, 0] += self.nspin * vfeat[:, 0]
        # fmt: off
        vxc[:, 1:4] += (
            (self.nspin * self.nspin * 2 * vfeat[:, 1])[:, None, :]
            * rho[:, 1:4]
        )
        # fmt: on
        vxc[:, 4] += self.nspin * vfeat[:, 2]

    def _fill_vxc_np_(self, rho, vfeat, vxc):
        sigma = np.einsum("sx...,sx...->s...", rho[:, 1:4], rho[:, 1:4])
        sm23 = self.nspin ** (-2.0 / 3)
        dpdn, dpdsigma = ds2(rho[:, 0], sigma)
        vxc[:, 0] += self.nspin * vfeat[:, 0] + sm23 * vfeat[:, 1] * dpdn
        vxc[:, 1:4] += (sm23 * vfeat[:, 1] * dpdsigma)[:, None, :] * 2 * rho[:, 1:4]

    def _fill_vxc_ns_(self, rho, vfeat, vxc):
        vxc[:, 0] += self.nspin * vfeat[:, 0]
        # fmt: off
        vxc[:, 1:4] += (
            (self.nspin * self.nspin * 2 * vfeat[:, 1])[:, None, :]
            * rho[:, 1:4]
        )
        # fmt: on

    def get_vxc(self, rho, vfeat, vxc=None):
        assert rho.ndim == 3
        if vxc is None:
            vxc = np.zeros_like(rho[:, :5])
        if self.settings.mode == "npa":
            self._fill_vxc_npa_(rho, vfeat, vxc)
        elif self.settings.mode == "nst":
            self._fill_vxc_nst_(rho, vfeat, vxc)
        elif self.settings.mode == "np":
            self._fill_vxc_np_(rho, vfeat, vxc)
        else:
            self._fill_vxc_ns_(rho, vfeat, vxc)
        return vxc


class SemilocalPlan2(_BaseSemilocalPlan):
    def _fill_vxc_npa_(self, vfeat, rho, vrho, sigma, vsigma, tau=None, vtau=None):
        sm23 = self.nspin ** (-2.0 / 3)
        dpdn, dpdsigma = ds2(rho, sigma)
        if self.level == "MGGA":
            dadn, dadsigma, dadtau = dalpha(rho, sigma, tau)
            vrho[:] += self.nspin * vfeat[:, 0] + sm23 * (
                vfeat[:, 1] * dpdn + vfeat[:, 2] * dadn
            )
            vsigma[:] += sm23 * (vfeat[:, 1] * dpdsigma + vfeat[:, 2] * dadsigma)
            vtau[:] += sm23 * vfeat[:, 2] * dadtau
        else:
            vrho[:] += self.nspin * vfeat[:, 0] + sm23 * vfeat[:, 1] * dpdn
            vsigma[:] += sm23 * vfeat[:, 1] * dpdsigma

    def _fill_vxc_nst_(self, vfeat, vrho, vsigma, vtau=None):
        vrho[:] += self.nspin * vfeat[:, 0]
        vsigma[:] += self.nspin * self.nspin * vfeat[:, 1]
        if self.level == "MGGA":
            vtau[:] += self.nspin * vfeat[:, 2]

    def get_vxc(self, vfeat, rho, vrho, sigma, vsigma, tau=None, vtau=None):
        for arr in [rho, vrho, sigma, vsigma, tau, vtau]:
            assert arr is None or arr.ndim == 2
        if self.settings.mode in ["npa", "np"]:
            self._fill_vxc_npa_(vfeat, rho, vrho, sigma, vsigma, tau, vtau)
        elif self.settings.mode in ["nst", "ns"]:
            self._fill_vxc_nst_(vfeat, vrho, vsigma, vtau)


class FracLaplPlan:
    """
    Plan for the Fractional Laplacian features. These features do not
    require additional settings for how to compute them, so this
    class basically just stores the settings class and nspin. Currently,
    evaluation of the orbital operations must be implemented
    in each DFT code separately.
    """

    def __init__(self, settings, nspin, nrho_sl=5, drho_start=1):
        """
        Initialize FracLaplPlan instance

        Args:
            settings (FracLaplSettings): settings for the fractional
                laplacian descriptors
        """
        self.settings = settings
        self.nspin = nspin
        self._cached_l1_data = []
        self._cached_ld_data = []
        self._nsl = nrho_sl
        self._i0d = drho_start

    def _clear_l1_cache(self):
        self._cached_l1_data = []
        self._cached_ld_data = []

    def _cache_l1_vectors(self, drho, l1_data, make_copy):
        assert len(l1_data[0]) % 3 == 0
        assert len(l1_data[0]) // 3 == self.settings.nk1
        for i in range(self.settings.nk1):
            self._cached_l1_data.append(l1_data[:, 3 * i : 3 * i + 3])
        self._cached_l1_data.append(drho)
        if make_copy:
            self._cached_l1_data = [v.copy() for v in self._cached_l1_data]

    def _cache_ld_vectors(self, drho, l1_data, make_copy):
        assert len(l1_data[0]) % 3 == 0
        assert len(l1_data[0]) // 3 == self.settings.nd1
        for i in range(self.settings.nk1):
            self._cached_ld_data.append(l1_data[:, 3 * i : 3 * i + 3])
        self._cached_ld_data.append(drho)
        if make_copy:
            self._cached_ld_data = [v.copy() for v in self._cached_ld_data]

    def _cache_all_l1_data(self, rho_data, make_l1_data_copy):
        nk0 = self.settings.nk0
        nk1 = self.settings.nk1
        nd1 = self.settings.nd1
        self._cache_l1_vectors(
            rho_data[:, self._i0d : self._i0d + 3],
            rho_data[:, self._nsl + nk0 : self._nsl + nk0 + 3 * nk1],
            make_l1_data_copy,
        )
        start = self._nsl + nk0 + 3 * nk1
        self._cache_ld_vectors(
            rho_data[:, self._i0d : self._i0d + 3],
            rho_data[:, start : start + 3 * nd1],
            make_l1_data_copy,
        )

    def get_feat(self, rho_data, feat=None, make_l1_data_copy=True):
        self._clear_l1_cache()
        if rho_data.ndim == 2:
            rho_data = rho_data[None, :, :]
        nk0 = self.settings.nk0
        nspin = self.nspin
        assert rho_data.ndim == 3
        assert rho_data.shape[:2] == (nspin, self._nsl + self.settings.nrho)
        l0_data = rho_data[:, self._nsl : self._nsl + nk0]
        self._cache_all_l1_data(rho_data, make_l1_data_copy)
        ngrids = rho_data.shape[2]
        if feat is None:
            feat = np.empty((nspin, self.settings.nfeat, ngrids))
        assert feat.shape == (nspin, self.settings.nfeat, ngrids)
        for i in range(nk0):
            feat[:, i] = l0_data[:, i]
        for i1 in range(len(self.settings.l1_dots)):
            j, k = self.settings.l1_dots[i1]
            feat[:, nk0 + i1] = np.einsum(
                "sxg,sxg->sg", self._cached_l1_data[j], self._cached_l1_data[k]
            )
        nl1 = len(self.settings.l1_dots)
        for i1 in range(len(self.settings.ld_dots)):
            j, k = self.settings.ld_dots[i1]
            feat[:, nk0 + nl1 + i1] = np.einsum(
                "sxg,sxg->sg", self._cached_ld_data[j], self._cached_ld_data[k]
            )
        ndd = self.settings.ndd
        for i in range(ndd):
            feat[:, i - ndd] = rho_data[:, i - ndd]
        return feat

    def get_occd(self, rho_data, occd_data):
        assert rho_data.ndim == 2
        assert self.nspin == 1
        assert occd_data.ndim == 3
        assert occd_data.shape[1:] == rho_data.shape
        norb = occd_data.shape[0]
        nk0 = self.settings.nk0
        ngrids = rho_data.shape[1]
        l0_occd = occd_data[:, self._nsl : self._nsl + nk0]
        self._clear_l1_cache()
        self._cache_all_l1_data(rho_data[None, :], False)
        l1_rho_cache = [v for v in self._cached_l1_data]
        ld_rho_cache = [v for v in self._cached_ld_data]
        self._clear_l1_cache()
        self._cache_all_l1_data(occd_data, False)
        l1_occd_cache = [v for v in self._cached_l1_data]
        ld_occd_cache = [v for v in self._cached_ld_data]
        self._clear_l1_cache()
        occd_feat = np.empty((norb, self.settings.nfeat, ngrids))
        for i in range(nk0):
            occd_feat[:, i] = l0_occd[:, i]
        for i1 in range(len(self.settings.l1_dots)):
            j, k = self.settings.l1_dots[i1]
            occd_feat[:, nk0 + i1] = np.einsum(
                "sxg,oxg->og", l1_rho_cache[j], l1_occd_cache[k]
            ) + np.einsum("sxg,oxg->og", l1_rho_cache[k], l1_occd_cache[j])
        nl1 = len(self.settings.l1_dots)
        for i1 in range(len(self.settings.ld_dots)):
            j, k = self.settings.ld_dots[i1]
            occd_feat[:, nk0 + nl1 + i1] = np.einsum(
                "sxg,oxg->og", ld_rho_cache[j], ld_occd_cache[k]
            ) + np.einsum("sxg,oxg->og", ld_rho_cache[k], ld_occd_cache[j])
        ndd = self.settings.ndd
        for i in range(ndd):
            occd_feat[:, i - ndd] = occd_data[:, i - ndd]
        return occd_feat

    def get_vxc(self, vfeat, vxc=None):
        if vfeat.ndim == 2:
            vfeat = vfeat[None, :, :]
        nk0 = self.settings.nk0
        nk1 = self.settings.nk1
        nd1 = self.settings.nd1
        nspin = self.nspin
        assert vfeat.ndim == 3
        assert vfeat.shape[:2] == (nspin, self.settings.nfeat)
        ngrids = vfeat.shape[2]
        if vxc is None:
            vxc = np.zeros((nspin, self._nsl + self.settings.nrho, ngrids))
        assert vxc.shape == (nspin, self._nsl + self.settings.nrho, ngrids)
        vdrho = vxc[:, self._i0d : self._i0d + 3]
        l0_vxc = vxc[:, self._nsl : self._nsl + nk0]
        l1_vxc = vxc[:, self._nsl + nk0 : self._nsl + nk0 + 3 * nk1]
        start = self._nsl + nk0 + 3 * nk1
        ld_vxc = vxc[:, start : start + 3 * nd1]
        ldd_vxc = vxc[:, start + 3 * nd1 :]
        for i in range(nk0):
            l0_vxc[:, i] += vfeat[:, i]
        for i1 in range(len(self.settings.l1_dots)):
            j, k = self.settings.l1_dots[i1]
            vfeat_tmp = vfeat[:, nk0 + i1 : nk0 + i1 + 1, :]
            if j == -1:
                target = vdrho
            else:
                target = l1_vxc[:, 3 * j : 3 * j + 3]
            target[:] += vfeat_tmp * self._cached_l1_data[k]
            if k == -1:
                target = vdrho
            else:
                target = l1_vxc[:, 3 * k : 3 * k + 3]
            target[:] += vfeat_tmp * self._cached_l1_data[j]
        start = nk0 + len(self.settings.l1_dots)
        for i1 in range(len(self.settings.ld_dots)):
            j, k = self.settings.ld_dots[i1]
            vfeat_tmp = vfeat[:, start + i1 : start + i1 + 1, :]
            if j == -1:
                target = vdrho
            else:
                target = ld_vxc[:, 3 * j : 3 * j + 3]
            target[:] += vfeat_tmp * self._cached_ld_data[k]
            if k == -1:
                target = vdrho
            else:
                target = ld_vxc[:, 3 * k : 3 * k + 3]
            target[:] += vfeat_tmp * self._cached_ld_data[j]
        ndd = self.settings.ndd
        for i in range(ndd):
            ldd_vxc[:, i] += vfeat[:, i - ndd]
        return vxc


class SADMPlan:
    """
    Plan for evaluating exchange energy of spherically averaged density
    matrix. Stores the settings (basically an empty class since the feature
    is not tunable, though local range separation might be added later),
    the nspin, and paramters for an even-tempered Gaussian basis for
    expanding the spherically averaged density matrix. Currently,
    evaluation of the integrals must be implemented in each DFT code separately.
    """

    has_coul_list = False

    def __init__(self, settings, nspin, alpha0, lambd, nalpha, fit_metric="ovlp"):
        """

        Args:
            settings (SADMSettings): Settings for the descriptor.
            nspin (int): Number of spins, 2 for spin-polarized and 1 for
                non-spin-polarized
            alpha0 (float): Minimum exponent. sqrt(1/alpha0) should be at least
                the width of the system, or altneratively the maximum distance
                at which the exchange hole is expected to have significant density.
            lambd (float): ETB lambda parameter
            nalpha (int): Number of ETB basis functions
            fit_metric (str): 'ovlp' or 'coul', the metric for expanding the
                density matrix in the auxiliary basis.
        """
        self.settings = settings
        self.nspin = nspin
        if alpha0 <= 0:
            raise ValueError("alpha0 must be positive")
        self.alpha0 = alpha0
        if lambd <= 1:
            raise ValueError("lambd must be > 1")
        self.lambd = lambd
        if not isinstance(nalpha, int) or nalpha <= 0:
            raise ValueError("nalpha must be positive integer")
        self.nalpha = nalpha
        self.alphas = self.alpha0 * self.lambd ** np.arange(self.nalpha)
        self.alpha_norms = (np.pi / (2 * self.alphas)) ** -0.75
        prod = self.alphas * self.alphas[:, None]
        sum = self.alphas + self.alphas[:, None]
        # coul = ( alpha(r) | 1/(r-r') | beta(r') )
        # coul = 4 * np.sqrt(2) * np.pi / prod**0.25 / sum**0.5
        # coul = ( alpha(r) | 1 / r | beta(r) )
        coul = 4 * np.sqrt(2 / np.pi) * prod**0.75 / sum
        LJ = cholesky(coul, lower=True)
        if settings.mode == "smooth":
            fit_metric = "ovlp"
            r2vals = 2.0 / self.alphas
            vals = np.exp(-self.alphas[None, :] * r2vals[:, None])
            vals *= self.alpha_norms[None, :]
            self.fit_matrix = np.linalg.solve(vals.T, LJ).T
            self.alpha_norms = (
                4 * self.alphas**1.5 / ((4 - np.sqrt(2)) * np.pi**1.5)
            )
        elif fit_metric == "ovlp":
            # < alpha(r) | beta(r) >
            ovlp = 2 * np.sqrt(2) * prod**0.75 / sum**1.5
            c_and_l = cho_factor(ovlp)
            fit = cho_solve(c_and_l, LJ)
            self.fit_matrix = np.ascontiguousarray(fit.T)
        elif fit_metric == "coul":
            self.fit_matrix = solve_triangular(LJ, np.identity(LJ.shape[0]), lower=True)
        else:
            raise ValueError('fit_metric must be "ovlp" or "coul"')
        self.fit_metric = fit_metric

    @property
    def num_l0_feat(self):
        return 1

    @property
    def num_l1_feat(self):
        return 0

    def get_features(self, p_vag, out=None, l0tmp=None, l1tmp=None):
        """
        Should pass out[idm], l0tmp[idm], etc. to this function
        """
        na, ng = p_vag.shape[1:]
        nfeat = self.settings.nfeat
        n0 = self.num_l0_feat
        assert n0 == 1
        assert nfeat == 1
        if out is None:
            out = np.empty((nfeat, ng))
        if l0tmp is None:
            l0tmp = np.empty((n0, na, ng))
        l0tmp[0] = pyscflib.dot(self.fit_matrix, p_vag[0])
        out[0] = pyscflib.einsum("qg,qg->g", l0tmp[0], l0tmp[0])
        out[0] *= -0.25 * self.nspin * self.nspin
        return out

    def get_vxc(self, vxc_ig, l0tmp, l1tmp=None, out=None):
        nfeat = self.settings.nfeat
        n0 = self.num_l0_feat
        assert nfeat == n0 == vxc_ig.shape[0] == 1
        ng = vxc_ig.shape[1]
        if out is None:
            out = np.zeros((1, self.nalpha, ng))
        else:
            out[:] = 0.0
        if nfeat == 0:
            return out
        tmp = -0.25 * vxc_ig[:n0, None] * l0tmp * self.nspin * self.nspin  # fqg
        out[0] += pyscflib.dot(self.fit_matrix.T, tmp[0])
        return out


def _sdmx_einsum(tmp, mystr=None):
    if mystr is not None:
        pass
    elif tmp.ndim == 2:
        mystr = "qg,qg->g"
    elif tmp.ndim == 3:
        mystr = "vqg,vqg->g"
    else:
        raise ValueError
    if tmp.dtype == np.float64:
        return pyscflib.einsum(mystr, tmp, tmp)
    elif tmp.dtype == np.complex128:
        return pyscflib.einsum(mystr, tmp.real, tmp.real) + pyscflib.einsum(
            mystr, tmp.imag, tmp.imag
        )


class SDMXBasePlan:

    has_coul_list: bool = True
    nspin: int
    nalpha: int
    settings: SDMXBaseSettings
    fit_matrices: list
    alphas_norms = None
    alphas = None

    @property
    def num_l0_feat(self):
        raise NotImplementedError

    @property
    def num_l1_feat(self):
        raise NotImplementedError

    def _get_fit_mats(self):
        if self.has_coul_list:
            fit_mats = self.fit_matrices
        else:
            fit_mats = [self.fit_matrix]
        return fit_mats

    def get_features(self, p_vag, out=None, l0tmp=None, l1tmp=None):
        """
        Should pass out[idm], l0tmp[idm], etc. to this function
        """
        na, ng = p_vag.shape[1:]
        nfeat = self.settings.nfeat
        n0 = self.num_l0_feat
        if out is None:
            out = np.empty((nfeat, ng))
        if l0tmp is None:
            l0tmp = np.empty((n0, na, ng))
        elif l1tmp is None:
            l1tmp = np.empty((nfeat - n0, 3, na, ng))
        fit_mats = self._get_fit_mats()
        fac = -0.25 * self.nspin * self.nspin
        for ifeat, fit_matrix in enumerate(fit_mats[:n0]):
            l0tmp[ifeat] = pyscflib.dot(fit_matrix, p_vag[0])
            # out[ifeat] = pyscflib.einsum(
            #     'qg,qg->g', l0tmp[ifeat], l0tmp[ifeat]
            # )
            out[ifeat] = _sdmx_einsum(l0tmp[ifeat])
            out[ifeat] *= fac
        for ifeat, fit_matrix in enumerate(fit_mats[n0:]):
            ifeat_shift = ifeat + n0
            for v in range(3):
                l1tmp[ifeat, v] = pyscflib.dot(fit_matrix, p_vag[v + 1])
            # out[ifeat_shift] = pyscflib.einsum(
            #     'vqg,vqg->g', l1tmp[ifeat], l1tmp[ifeat]
            # )
            out[ifeat_shift] = _sdmx_einsum(l1tmp[ifeat])
            out[ifeat_shift] *= fac  # factor of -1/4 for EXX
        return out

    def get_vxc(self, vxc_ig, l0tmp, l1tmp=None, out=None):
        nfeat = self.settings.nfeat
        assert nfeat == vxc_ig.shape[0]
        ng = vxc_ig.shape[1]
        n0 = self.num_l0_feat
        n1 = self.num_l1_feat
        if n1 == 0:
            nv = 1
        else:
            nv = 4
            assert l1tmp is not None
        if out is None:
            out = np.zeros((nv, self.nalpha, ng), dtype=l0tmp.dtype)
        else:
            out[:] = 0.0
        if nfeat == 0:
            return out
        fit_mats = self._get_fit_mats()
        fac = -0.25 * self.nspin * self.nspin
        tmp = fac * vxc_ig[:n0, None] * l0tmp  # fqg
        # if tmp.dtype == np.complex128:
        #     tmp = tmp.conj()
        for ifeat, fit_matrix in enumerate(fit_mats[:n0]):
            out[0] += pyscflib.dot(fit_matrix.T, tmp[ifeat])
        if n1 > 0:
            tmp1 = fac * vxc_ig[n0:, None, None] * l1tmp
            # if tmp1.dtype == np.complex128:
            #     tmp1 = tmp1.conj()
            for ifeat, fit_matrix in enumerate(self._get_fit_mats()[n0:]):
                for v in range(3):
                    # tmp2[:, v + 1] += fit_matrix.T.dot(tmp1[ifeat, v])
                    out[v + 1] += pyscflib.dot(fit_matrix.T, tmp1[ifeat, v])
        return out


class SDMXPlan(SDMXBasePlan):

    has_coul_list = True

    def __init__(self, settings, nspin, alpha0, lambd, nalpha, fit_metric="ovlp"):
        """

        Args:
            settings (SDMXSettings): Settings for the descriptor.
            nspin (int): Number of spins, 2 for spin-polarized and 1 for
                non-spin-polarized
            alpha0 (float): Minimum exponent. sqrt(1/alpha0) should be at least
                the width of the system, or altneratively the maximum distance
                at which the exchange hole is expected to have significant density.
            lambd (float): ETB lambda parameter
            nalpha (int): Number of ETB basis functions
            fit_metric (str): 'ovlp' or 'coul', the metric for expanding the
                density matrix in the auxiliary basis.
        """
        self.settings = settings
        self.nspin = nspin
        if alpha0 <= 0:
            raise ValueError("alpha0 must be positive")
        self.alpha0 = alpha0
        if lambd <= 1:
            raise ValueError("lambd must be > 1")
        self.lambd = lambd
        if not isinstance(nalpha, int) or nalpha <= 0:
            raise ValueError("nalpha must be positive integer")
        self.nalpha = nalpha
        self.alphas = self.alpha0 * self.lambd ** np.arange(self.nalpha)
        self.alpha_norms = (np.pi / (2 * self.alphas)) ** -0.75
        self.alpha_norms_l1 = np.sqrt(4.0 / 3 * self.alphas) * self.alpha_norms
        prod = self.alphas * self.alphas[:, None]
        sum = self.alphas + self.alphas[:, None]
        # coul = ( alpha(r) | 1 / r | beta(r) )
        coul_list = [
            4
            * np.sqrt(2 / np.pi)
            * prod**0.75
            / sum ** (0.5 * (3 - n))
            * gamma(0.5 * (3 - n))
            for n in settings.pows
        ]
        ndt = self.settings.ndterms
        if ndt > 0:
            coul_list += [
                16
                * np.sqrt(2 / np.pi)
                * prod**1.75
                / sum ** (0.5 * (7 - n))
                * gamma(0.5 * (7 - n))
                for n in settings.pows[:ndt]
            ]
        n1t = self.settings.n1terms
        if n1t > 0:
            coul_list += [
                4
                * np.sqrt(2 / np.pi)
                * prod**0.75
                / sum ** (0.5 * (5 - n))
                * gamma(0.5 * (5 - n))
                for n in settings.pows[:n1t]
            ]
        LJ_list = [cholesky(coul, lower=True) for coul in coul_list]
        fit_metric = "ovlp"
        if settings.mode == "smooth":
            r2vals = 2.0 / self.alphas
            vals = np.exp(-self.alphas[None, :] * r2vals[:, None])
            vals *= self.alpha_norms[None, :]
            self.fit_matrices = [
                np.ascontiguousarray(np.linalg.solve(vals.T, LJ).T) for LJ in LJ_list
            ]
            self.alpha_norms = (
                4 * self.alphas**1.5 / ((4 - np.sqrt(2)) * np.pi**1.5)
            )
        else:
            # < alpha(r) | beta(r) >
            ovlp = 2 * np.sqrt(2) * prod**0.75 / sum**1.5
            c_and_l = cho_factor(ovlp)
            self.fit_matrices = [
                np.ascontiguousarray(cho_solve(c_and_l, LJ).T) for LJ in LJ_list
            ]
            ovlp = 4 * np.sqrt(2) * prod**1.25 / sum**2.5
        self.fit_metric = fit_metric

    @property
    def num_l0_feat(self):
        if self.settings.n1terms == 0:
            return self.settings.nfeat
        else:
            return len(self.settings.pows) + self.settings.ndterms

    @property
    def num_l1_feat(self):
        return self.settings.nfeat - self.num_l0_feat


def _get_int_0(n, prod, asum):
    return (
        4
        * np.sqrt(2 / np.pi)
        * prod**0.75
        / asum ** (0.5 * (3 - n))
        * gamma(0.5 * (3 - n))
    )


def _get_int_d(n, prod, asum):
    return (
        16
        * np.sqrt(2 / np.pi)
        * prod**1.75
        / asum ** (0.5 * (7 - n))
        * gamma(0.5 * (7 - n))
    )


def _get_int_1(n, prod, asum):
    return (
        4
        * np.sqrt(2 / np.pi)
        * prod**0.75
        / asum ** (0.5 * (5 - n))
        * gamma(0.5 * (5 - n))
    )


def _get_int_1d(n, prod, asum):
    return (
        4
        * np.sqrt(2 / np.pi)
        * prod**0.75
        / asum ** (0.5 * (9 - n))
        * gamma(0.5 * (5 - n))
        * ((n - 4) * asum**2 + prod * (35 - 12 * n + n * n))
    )


class SDMXFullPlan(SDMXBasePlan):

    settings: SDMXFullSettings
    fit_metric = "ovlp"

    def __init__(
        self,
        settings,
        nspin,
        alpha0,
        lambd,
        nalpha,
    ):
        """

        Args:
            settings (SDMXFullSettings):
            nspin (int):
            alpha0 (float):
            lambd (float):
            nalpha (int):
        """
        self.settings = settings
        self.nspin = nspin
        if alpha0 <= 0:
            raise ValueError("alpha0 must be positive")
        self.alpha0 = alpha0
        if lambd <= 1:
            raise ValueError("lambd must be > 1")
        self.lambd = lambd
        if not isinstance(nalpha, int) or nalpha <= 0:
            raise ValueError("nalpha must be positive integer")
        self.nalpha = nalpha
        self.alphas = self.alpha0 * self.lambd ** np.arange(self.nalpha)
        self.alpha_norms = (np.pi / (2 * self.alphas)) ** -0.75
        l0mats = []
        l1mats = []
        for ratio in self.settings.ratios:
            r = ratio
            prod = self.alphas * self.alphas[:, None]
            isum = self.alphas + self.alphas[:, None]
            asum = self.alphas / r + r * self.alphas[:, None]
            bsum = self.alphas * r + self.alphas[:, None] / r
            for n, rdr in self.settings.iterate_l0_terms(ratio):
                # num = 0.25 * (ratio ** (0.5 * n) + ratio ** (-0.5 * n))
                num = 0.5
                if not rdr:
                    l0mats.append(
                        num * _get_int_0(n, prod, isum)
                        + 0.25 * _get_int_0(n, prod, asum)
                        + 0.25 * _get_int_0(n, prod, bsum)
                    )
                else:
                    l0mats.append(
                        num * _get_int_d(n, prod, isum)
                        + 0.25 * _get_int_d(n, prod, asum)
                        + 0.25 * _get_int_d(n, prod, bsum)
                    )
            for n, rdr in self.settings.iterate_l1_terms(ratio):
                # num = 0.25 * (ratio ** (0.5 * n) + ratio ** (-0.5 * n))
                num = 0.5
                if not rdr:
                    l1mats.append(
                        num * _get_int_1(n, prod, isum)
                        + 0.25 * _get_int_1(n, prod, asum)
                        + 0.25 * _get_int_1(n, prod, bsum)
                    )
                else:
                    l1mats.append(
                        num * _get_int_1d(n, prod, isum)
                        + 0.25 * _get_int_1d(n, prod, asum)
                        + 0.25 * _get_int_1d(n, prod, bsum)
                    )
        self._num_l0_feat = len(l0mats)
        self._num_l1_feat = len(l1mats)
        coul_list = [m for m in l0mats + l1mats]
        LJ_list = [cholesky(coul, lower=True) for coul in coul_list]
        r2vals = 2.0 / self.alphas
        vals = np.exp(-self.alphas[None, :] * r2vals[:, None])
        vals *= self.alpha_norms[None, :]
        self.fit_matrices = [
            np.ascontiguousarray(np.linalg.solve(vals.T, LJ).T) for LJ in LJ_list
        ]
        for coul, LJ in zip(coul_list, self.fit_matrices):
            tmp = np.linalg.solve(vals.T, coul).T
            tmp = np.linalg.solve(vals.T, tmp)
        self.alpha_norms = 4 * self.alphas**1.5 / ((4 - np.sqrt(2)) * np.pi**1.5)

    @property
    def num_l0_feat(self):
        return self._num_l0_feat

    @property
    def num_l1_feat(self):
        return self._num_l1_feat


class SDMXIntPlan(SDMXBasePlan):
    def __init__(
        self,
        settings,
        nspin,
        alpha0,
        lambd,
        nalpha,
    ):
        """

        Args:
            settings (SDMXFullSettings):
            nspin (int):
            alpha0 (float):
            lambd (float):
            nalpha (int):
        """
        self.settings = settings
        self.nspin = nspin
        if alpha0 <= 0:
            raise ValueError("alpha0 must be positive")
        self.alpha0 = alpha0
        if lambd <= 1:
            raise ValueError("lambd must be > 1")
        self.lambd = lambd
        if not isinstance(nalpha, int) or nalpha <= 0:
            raise ValueError("nalpha must be positive integer")
        self.nalpha = nalpha
        self.alphas = self.alpha0 * self.lambd ** np.arange(self.nalpha)
        self.alpha_norms = (np.pi / (2 * self.alphas)) ** -0.75
        r2vals = 2.0 / self.alphas
        vals = np.exp(-2 * self.alphas[None, :] * r2vals[:, None])
        vals *= self.alpha_norms[None, :] * self.alpha_norms[None, :]
        all_n = []
        self._num_l0_feat = 0
        self._num_l1_feat = 0
        for ratio in self.settings.ratios:
            for n, rdr in self.settings.iterate_l0_terms(ratio):
                if False:  # rdr or ratio != 1:
                    raise NotImplementedError(
                        "Numerical integral SDMX only works for ratio-1, rdr=False"
                    )
                all_n.append(n)
                self._num_l0_feat += 1
            for n, rdr in self.settings.iterate_l1_terms(ratio):
                if False:  # rdr or ratio != 1:
                    raise NotImplementedError(
                        "Numerical integral SDMX only works for ratio-1, rdr=False"
                    )
                all_n.append(n - 2)
                self._num_l1_feat += 1
        wt_dict = []
        for n in all_n:
            refz = True
            alphas = self.alphas
            if refz:
                alphas = np.append(alphas, [alphas[-1] * self.lambd])
            lambd = self.lambd
            mypow = 0.5 + 0.5 / lambd
            # mypow = 0.5
            amid = alphas[alphas.size // 2]
            rvals = (0.25 / mypow**2 / np.sqrt(amid)) * (amid / alphas) ** mypow
            # rvals = 1.0 / alphas**mypow
            if refz:
                rvals_abs = rvals.copy()
                rvals -= np.min(rvals)
            assert np.min(rvals) == rvals[-1]  # check ordering
            rdr = 4 * np.pi * rvals ** (2 - n) * rvals_abs * np.log(lambd**mypow)
            if refz:
                rdr = rdr[:-1]
                rvals = rvals[:-1]
            wt_dict.append(rdr)
        self.alphas = 2.0 / (rvals * rvals)
        self.wt_dict = wt_dict
        self.fit_metric = "ovlp"
        self.alpha_norms = 4 * self.alphas**1.5 / ((4 - np.sqrt(2)) * np.pi**1.5)

    @property
    def num_l0_feat(self):
        return self._num_l0_feat

    @property
    def num_l1_feat(self):
        return self._num_l1_feat

    def get_features(self, p_vag, out=None, l0tmp=None, l1tmp=None):
        """
        Should pass out[idm], l0tmp[idm], etc. to this function
        """
        na, ng = p_vag.shape[1:]
        nfeat = self.settings.nfeat
        n0 = self.num_l0_feat
        n1 = self.num_l1_feat
        if out is None:
            out = np.empty((nfeat, ng))
        if l0tmp is None:
            l0tmp = np.empty((na, ng))
        elif l1tmp is None:
            l1tmp = np.empty((3, na, ng))
        fac = -0.25 * self.nspin * self.nspin
        l0tmp[:] = p_vag[0]
        if l0tmp.dtype == np.float64:
            p0_ag = l0tmp * l0tmp
        else:
            p0_ag = l0tmp.real * l0tmp.real + l0tmp.imag * l0tmp.imag
        if n1 > 0:
            for v in range(3):
                l1tmp[v] = p_vag[v + 1]
            p1_ag = _sdmx_einsum(l1tmp, mystr="vag,vag->ag")
            p1_ag = np.einsum("vag,vag->ag", l1tmp, l1tmp)
        for ifeat in range(n0):
            out[ifeat] = pyscflib.einsum("ag,a->g", p0_ag, self.wt_dict[ifeat])
            out[ifeat] *= fac
        for ifeat in range(n0, n0 + n1):
            out[ifeat] = pyscflib.einsum("ag,a->g", p1_ag, self.wt_dict[ifeat])
            out[ifeat] *= fac
        return out

    def get_vxc(self, vxc_ig, l0tmp, l1tmp=None, out=None):
        nfeat = self.settings.nfeat
        assert nfeat == vxc_ig.shape[0]
        ng = vxc_ig.shape[1]
        n0 = self.num_l0_feat
        n1 = self.num_l1_feat
        if n1 == 0:
            nv = 1
        else:
            nv = 4
            assert l1tmp is not None
        if out is None:
            out = np.zeros((nv, self.nalpha, ng), dtype=l0tmp.dtype)
        else:
            out[:] = 0.0
        if nfeat == 0:
            return out
        fac = -0.25 * self.nspin * self.nspin
        v_ag = np.zeros((self.nalpha, ng))
        for ifeat in range(n0):
            v_ag[:] += self.wt_dict[ifeat][:, None] * vxc_ig[ifeat]
        out[0] = fac * l0tmp * v_ag
        if n1 > 0:
            v_ag[:] = 0
            for ifeat in range(n0, n0 + n1):
                v_ag[:] += self.wt_dict[ifeat][:, None] * vxc_ig[ifeat]
            for v in range(3):
                out[v + 1] = fac * l1tmp[v] * v_ag
        return out


class HybridPlan:
    """Plan for hybrid DFT. Not yet implemented"""

    def __init__(self, settings, nspin):
        """
        Args:
            settings (HybridSettings)
        """
        self.settings = settings
        self.nspin = nspin


class NLDFAuxiliaryPlan(ABC):
    """
    Plan for expanding the NLDFs using an auxiliary basis or spline.
    Stores the list of interpolation exponents and how they were constructed,
    as well as instructions for evaluating the auxiliary coefficients.
    Also has a method to call the relevant C function to get the coefficients.
    """

    def __init__(
        self,
        nldf_settings,
        nspin,
        alpha0,
        lambd,
        nalpha,
        coef_order="gq",
        alpha_formula="etb",
        proc_inds=None,
        rhocut=1e-10,
        expcut=1e-10,
        raise_large_expnt_error=True,
        use_smooth_expnt_cutoff=False,
    ):
        """
        Initialize NLDFAuxiliaryPlan

        Args:
            nldf_settings (NLDFSettings): code-universal NLDF settings
            nspin (int): 1 (non-spin-polarized) or 2 (spin-polarized)
            alpha0 (float): Small-exponent setting for constructing
                list of alphas.
            lambd (float): Inverse density of interpolation.
                Must be > 1. 1=infinitesimal spacing of Gaussian exponents
            nalpha (int): Number of interpolating exponents
            coef_order (str): Whether interpolation/expansion coefficients
                have grid as the slow (first) index ('gq') or exponent ('qg')
            alpha_formula (str): How to generate the interpolating exponents.
                Must be 'etb' (alphas[j] = alpha0 * lambd**j))
                or 'zexp' (alphas[j] = alpha0 * (lambd**j - 1) / (lambd - 1))
            proc_inds (list or np.ndarray): If parallelized, which
                alpha indexes does this process cover. List or array
                of integers.
            rhocut (float): Small-density cutoff for stability
            expcut (float): Small-exponent cutoff for stability
            raise_large_expnt_error (bool, True): If True, raise an error
                if a convolution exponent is larger than the largest interpolation
                value. It is generally best to set this to true to make
                sure a calculation doesn't accidentally lose significant precision
                to going outside the interpolation range.
            use_smooth_expnt_cutoff (bool, False): If True, use a damping function
                for large exponent to smoothly ensure that as the exponent goes
                to infinity, the damped exponent goes to alpha_max. The
                smoothing function is design to contribute insignificantly
                except for near and beyond alpha_max.
        """
        if not isinstance(nldf_settings, NLDFSettings):
            raise ValueError("Require NLDFSettings object")
        self.nldf_settings = nldf_settings
        if alpha0 <= 0:
            raise ValueError("alpha0 must be positive")
        self.alpha0 = np.float64(alpha0)
        if lambd <= 1:
            raise ValueError("lambd must be > 1")
        self.lambd = np.float64(lambd)
        if not isinstance(nalpha, int) or nalpha <= 0:
            raise ValueError("nalpha must be positive integer")
        self.nalpha = nalpha
        if alpha_formula == "zexp":
            self.alpha0 /= self.lambd - 1
        elif alpha_formula != "etb":
            raise ValueError
        self.alpha_formula = alpha_formula
        if coef_order not in ["gq", "qg"]:
            raise ValueError
        self.coef_order = coef_order
        # if alpha_order not in ['increasing', 'decreasing']:
        #    raise ValueError
        # if alpha_order == 'decreasing':
        #    raise NotImplementedError
        # one could hypothetically set these two differently,
        # but probably not useful to do so
        self.alpha_order = "increasing"
        self.normalization = "norm"

        # set the parallelization setting (which indices on this process)
        self.proc_inds = proc_inds

        if nspin not in [1, 2]:
            raise ValueError("nspin must be 1 (non-polarized) or 2 (polarized)")
        self.nspin = nspin
        if rhocut < 0 or expcut < 0:
            raise ValueError("rhocut and expcut must be nonnegative")
        self.rhocut = rhocut / nspin
        self.expcut = expcut

        self.alphas = None
        self.amin = None
        self.local_alphas = None
        self.local_nalpha = None
        self.alpha_norms = None
        self._alpha_transform = None
        self._local_qloc = None
        self._global_qloc = None

        self._l0_start_locs = []
        self._l1_start_locs = []
        self._cached_l1_data = {s: [] for s in range(self.nspin)}
        self._l1_dots = []
        self._cached_p_i_qg = {s: [] for s in range(self.nspin)}

        # set up the indexing helper for version i features
        start_locs = []
        end_locs = []
        loc_curr = 0
        for i in range(len(self.nldf_settings.l0_feat_specs)):
            start_locs.append(loc_curr)
            loc_curr += 1
            end_locs.append(loc_curr)
        self._l0_start_locs = start_locs
        start_locs = []
        end_locs = []
        for i in range(len(self.nldf_settings.l1_feat_specs)):
            start_locs.append(loc_curr)
            loc_curr += 3
            end_locs.append(loc_curr)
        self._l1_start_locs = start_locs
        self._l1_dots = self.nldf_settings.l1_feat_dots

        # set up the list of interpolating exponents
        # and their normalization coefficients
        self.alphas = self.get_q2a(np.arange(self.nalpha, dtype=np.float64))
        if self.alpha_order == "decreasing":
            self.alphas = np.ascontiguousarray(np.flip(self.alphas))
        if self.proc_inds is not None:
            self.local_alphas = np.ascontiguousarray(self.alphas[self.proc_inds])
        else:
            self.local_alphas = self.alphas.copy()
        self.local_nalpha = self.local_alphas.size

        if self.normalization == "one":
            self.alpha_norms = np.ones_like(self.alphas)
        else:
            self.alpha_norms = (np.pi / (2 * self.alphas)) ** -0.75

        if use_smooth_expnt_cutoff:
            # exponent will not overflow when this flag is true
            self._raise_large_expnt_error = False
        else:
            self._raise_large_expnt_error = raise_large_expnt_error
        self._use_smooth_expnt_cutoff = use_smooth_expnt_cutoff
        self._run_setup()

    def new(self, **kwargs):
        new_kwargs = dict(
            nldf_settings=self.nldf_settings,
            nspin=self.nspin,
            alpha0=self.alpha0,
            lambd=self.lambd,
            nalpha=self.nalpha,
            coef_order=self.coef_order,
            alpha_formula=self.alpha_formula,
            proc_inds=self.proc_inds,
            rhocut=self.rhocut,
            expcut=self.expcut,
        )
        new_kwargs.update(kwargs)
        return self.__class__(**new_kwargs)

    @abstractmethod
    def _run_setup(self):
        pass

    @property
    def num_vj(self):
        """Number of vj or vk features"""
        return self.nldf_settings.num_feat_param_sets

    @property
    def num_vi_ints(self):
        return len(self._l0_start_locs) + 3 * len(self._l1_start_locs)

    def get_q2a(self, q):
        """Convert (float) q index to exponent alpha."""
        if self.alpha_formula == "etb":
            return self.alpha0 * self.lambd**q
        else:
            res = self.alpha0 * (self.lambd**q - 1)
            res[0] = self.expcut
            return res

    def _get_extra_args(self, i):
        assert 0 <= i < self.nldf_settings.num_feat_param_sets
        params = self.nldf_settings.feat_params[i]
        # return [] if len(params) < 4 else params[3:]
        # TODO this works for GGA and MGGA but not if there are
        # multiple extra args for the feature specs. This might
        # need to be revised later on.
        if self.nldf_settings.feat_spec_list[i] == "se_erf_rinv":
            return [params[-1]]
        else:
            return []

    def get_function_to_convolve(self, rho_tuple):
        """
        Returns the function that gets convolved to make the nonlocal
        density features. Usually the density rho (settings.rho_mult='one'),
        but can also be other quantities like the rho times the theta
        exponent (rho_mult='expnt').

        Args:
            rho (np.ndarray): Density
            sigma (np.ndarray): Squared gradient of density
            tau (np.ndarray): Kinetic energy density

        Returns:
            tuple(np.ndarray): function to convolve, as well as its derivatives
            with respect to rho, sigma, and tau. Combined, this makes for a
            tuple of four arrays.
        """
        rho = rho_tuple[0]
        if self.nldf_settings.rho_mult == "one":
            da = [
                np.ones_like(rho),
            ] + [np.zeros_like(r) for r in rho_tuple[1:]]
            return rho, tuple(da)
        elif self.nldf_settings.rho_mult == "expnt":
            a, da_tuple = self.eval_feat_exp(rho_tuple, i=-1)
            for da in da_tuple:
                da[:] *= rho
            da_tuple[0][:] += a
            a[:] *= rho
            return a, da_tuple
        else:
            raise NotImplementedError

    def get_rho_tuple(self, rho_data, with_spin=False):
        return get_rho_tuple(
            rho_data, with_spin=with_spin, is_mgga=self.nldf_settings.sl_level == "MGGA"
        )

    def get_drhodf_tuple(self, rho_data, drhodf_data, with_spin=False):
        return get_drhodf_tuple(
            rho_data,
            drhodf_data,
            with_spin=with_spin,
            is_mgga=self.nldf_settings.sl_level == "MGGA",
        )

    def eval_feat_exp(self, rho_tuple, i=-1):
        """
        Evaluate the exponents that determine the length-scale
        of feature i at each grid point. i can take a range
        from -1 (theta exponent) to the number of version j/k
        features minus 1.

        Args:
            rho (np.ndarray): density
            sigma (np.ndarray): squared gradient
            tau (np.ndarray): kinetic energy density
            i: feature id. -1 corresponds to theta exponent

        Returns:
            the cider exponent array (np.ndarray)
        """
        if self.nldf_settings.sl_level == "MGGA":
            rho, sigma, tau = rho_tuple
            if i == -1:
                a0, grad_mul, tau_mul = self.nldf_settings.theta_params
            else:
                if not 0 <= i < self.nldf_settings.num_feat_param_sets:
                    raise ValueError("Feature index out of range")
                a0, grad_mul, tau_mul = self.nldf_settings.feat_params[i][:3]
            a, dadn, dadsigma, dadtau = get_cider_exponent(
                rho,
                sigma,
                tau,
                a0=a0,
                grad_mul=grad_mul,
                tau_mul=tau_mul,
                rhocut=self.rhocut,
                nspin=self.nspin,
            )
            res = a, (dadn, dadsigma, dadtau)
        else:
            rho, sigma = rho_tuple
            if i == -1:
                a0, grad_mul = self.nldf_settings.theta_params[:2]
            else:
                if not 0 <= i < self.nldf_settings.num_feat_param_sets:
                    raise ValueError("Feature index out of range")
                a0, grad_mul = self.nldf_settings.feat_params[i][:2]
            a, dadn, dadsigma = get_cider_exponent_gga(
                rho,
                sigma,
                a0=a0,
                grad_mul=grad_mul,
                rhocut=self.rhocut,
                nspin=self.nspin,
            )
            res = a, (dadn, dadsigma)
        if self._use_smooth_expnt_cutoff:
            derivs = [arr.ctypes.data_as(ctypes.c_void_p) for arr in res[1]]
            nd = len(derivs)
            libcider.smooth_cider_exponents(
                res[0].ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_void_p * nd)(*derivs),
                ctypes.c_double(np.max(self.alphas)),
                ctypes.c_int(a.size),
                ctypes.c_int(nd),
            )
        if self._raise_large_expnt_error and a.size > 0:
            # TODO if rhocut is small, might need to leave some buffer at small rho
            ap = a[rho > self.rhocut]
            if ap.size > 0 and np.max(ap) > np.max(self.alphas):
                # print(np.max(ap), rho[rho > self.rhocut][np.argmax(ap)],
                #      tau[rho > self.rhocut][np.argmax(ap)],
                #      np.max(self.alphas), self.rhocut)
                raise RuntimeError(
                    "NLDF exponent is too large! Please increase nalpha/alpha_max."
                )
        return res

    def _clear_l1_cache(self, s):
        self._cached_l1_data[s] = []

    def _clear_p_cache(self, s):
        self._cached_p_i_qg[s] = []

    def _cache_l1_vectors(self, f_qg, drho, s):
        for loc in self._l1_start_locs:
            self._cached_l1_data[s].append(f_qg[loc : loc + 3].copy())
        self._cached_l1_data[s].append(drho.copy())

    def _cache_p_tensor(self, s, p):
        self._cached_p_i_qg[s].append(p)

    def eval_rho_vj_(self, f_qg, p_i_qg, feat):
        if len(p_i_qg) == 0:
            return feat
        if self.coef_order == "gq":
            p_i_qg = [p.T for p in p_i_qg]
        nalpha = p_i_qg[0].shape[0]
        for i in range(len(p_i_qg)):
            feat[i] = np.einsum("qg,qg->g", f_qg[:nalpha], p_i_qg[i])
        return feat

    def eval_vxc_vj_(self, vfeat, p_i_qg, vf_qg):
        if len(p_i_qg) == 0:
            return vf_qg
        if self.coef_order == "gq":
            p_i_qg = [p.T for p in p_i_qg]
        nalpha = p_i_qg[0].shape[1]
        for i in range(len(p_i_qg)):
            vf_qg[:nalpha] += p_i_qg[i] * vfeat[i]
        return vf_qg

    def eval_rho_vi_(self, f_qg, drho, feat, spin):
        self._clear_l1_cache(spin)
        self._cache_l1_vectors(f_qg, drho, spin)
        i = 0
        l1cache = self._cached_l1_data[spin]
        for loc in self._l0_start_locs:
            feat[i] = f_qg[loc]
            i += 1
        for j, k in self._l1_dots:
            feat[i] = np.einsum("xg,xg->g", l1cache[j], l1cache[k])
            feat[i] *= self.nspin
            i += 1
        return feat

    def eval_vxc_vi_(self, vfeat, vdrho, vf_qg, spin):
        if len(self._cached_l1_data[spin]) != len(self._l1_start_locs) + 1:
            raise ValueError("Must have cached l=1 features to get potential")
        l1cache = self._cached_l1_data[spin]
        i = 0
        for loc in self._l0_start_locs:
            vf_qg[loc] = vfeat[i]
            i += 1
        for j, k in self._l1_dots:
            vfeat[i] *= self.nspin
            if j == -1:
                target = vdrho
            else:
                loc = self._l1_start_locs[j]
                target = vf_qg[loc : loc + 3]
            target[:] += vfeat[i] * l1cache[k]
            if k == -1:
                target = vdrho
            else:
                loc = self._l1_start_locs[k]
                target = vf_qg[loc : loc + 3]
            target[:] += vfeat[i] * l1cache[j]
            i += 1
        # self._clear_l1_cache(spin)

    def eval_rho_full(
        self,
        f,
        rho_data,
        spin=0,
        feat=None,
        dfeat=None,
        cache_p=True,
        apply_transformation=False,
        coeff_multipliers=None,
    ):
        """
        Evaluate the raw features. IMPORTANT: For spin-polarized calculations,
        spin must be passed explicitly and must be different for each
        spin channel, otherwise cached data will get overwritten, leading
        to incorrect gradients later on. If apply_transformation is True, f is overwritten.

        Args:
            f (np.ndarray): Feature interpolation coefficients, of shape
                (ngrids, nalpha) for coef_order=="qg" else (nalpha, ngrids)
            rho_data (np.ndarray): Density vector [n, dn/dx, dn/dy, dn/dz, (tau)]
            spin (int, 0): Spin index of this density, for caching data.
            feat (np.ndarray, None): Shape (nfeat, ngrids) optional buffer
                for features
            dfeat (np.ndarray, None): Shape (nfeat, ngrids) optional buffer
                for feature derivatives
            cache_p (bool, True): Cache nalpha x ngrids-size arrays for
                interpolation coefficients. Slightly faster, but more memory
                intensive.
            apply_transformation (bool, False): For Gaussian interpolation,
                apply the linear transformation from the projection values
                to the interpolation coefficients.
            coeff_multipliers (np.ndarray, None): If not None, multiply the
                interpolation coefficients by this (nalpha,)-shapred array

        Returns:
            (np.ndarray, np.ndarray): Features (nfeat, ngrids)
            and derivative of features (nfeat, ngrids) with respect to
            the feat_params exponent.
        """
        if self.coef_order == "qg":
            f_qg = f
        else:
            f_qg = f.T
        ngrids = f_qg.shape[1]
        rho_tuple = self.get_rho_tuple(rho_data)
        num_vj = self.nldf_settings.num_feat_param_sets
        if feat is None:
            feat = np.empty((self.nldf_settings.nfeat, f_qg.shape[1]))
        if dfeat is None:
            dfeat = np.empty((num_vj, f_qg.shape[1]))
        self._clear_p_cache(spin)
        dbuf = self.empty_coefs(ngrids, local=False)
        for i in range(num_vj):
            a_g = self.get_interpolation_arguments(rho_tuple, i=i)[0]
            p, dp = self.get_interpolation_coefficients(a_g, i=i, dbuf=dbuf)
            if coeff_multipliers is not None:
                p[:] *= coeff_multipliers[:, None]
                dp[:] *= coeff_multipliers[:, None]
            if apply_transformation:
                self.get_transformed_interpolation_terms(p, i=i, fwd=True, inplace=True)
                self.get_transformed_interpolation_terms(
                    dp, i=i, fwd=True, inplace=True
                )
            self.eval_rho_vj_(f_qg[: self.nalpha], [dp], dfeat[i : i + 1])
            self.eval_rho_vj_(f_qg[: self.nalpha], [p], feat[i : i + 1])
            if cache_p:
                self._cache_p_tensor(spin, p)
        start = 0 if self.nldf_settings.nldf_type == "i" else self.nalpha
        self._clear_l1_cache(spin)
        if "i" in self.nldf_settings.nldf_type:
            self.eval_rho_vi_(f_qg[start:], rho_data[1:4], feat[num_vj:], spin=spin)
        feat[:] *= self.nspin
        # dfeat[:] *= self.nspin
        return feat, dfeat

    def eval_occd_full(
        self,
        f,
        rho_data,
        occd_f,
        occd_rho_data,
        apply_transformation=True,
        coeff_multipliers=None,
    ):
        spin = 0
        # if self.nspin != 1:
        #     raise NotImplementedError
        if self.coef_order == "qg":
            f_qg = f
            occd_f_qg = occd_f
        else:
            f_qg = f.T
            occd_f_qg = occd_f.T
        ngrids = f_qg.shape[1]
        rho_tuple = self.get_rho_tuple(rho_data)
        occd_sigma = 2 * np.einsum("xg,xg->g", rho_data[1:4], occd_rho_data[1:4])
        num_vj = self.nldf_settings.num_feat_param_sets
        occd1_feat = np.zeros((self.nldf_settings.nfeat, f_qg.shape[1]))
        occd2_feat = np.zeros((self.nldf_settings.nfeat, f_qg.shape[1]))
        dbuf = self.empty_coefs(ngrids, local=False)
        for i in range(num_vj):
            a_g, da_tuple = self.get_interpolation_arguments(rho_tuple, i=i)
            p, dp = self.get_interpolation_coefficients(a_g, i=i, dbuf=dbuf)
            if coeff_multipliers is not None:
                p[:] *= coeff_multipliers[:, None]
                dp[:] *= coeff_multipliers[:, None]
            occd_a_g = occd_rho_data[0] * da_tuple[0]
            occd_a_g[:] += occd_sigma * da_tuple[1]
            if self.nldf_settings.sl_level == "MGGA":
                occd_a_g[:] += occd_rho_data[4] * da_tuple[2]
            if self.coef_order == "qg":
                dp *= occd_a_g
            else:
                dp *= occd_a_g[:, None]
            if apply_transformation:
                self.get_transformed_interpolation_terms(p, i=i, fwd=True, inplace=True)
                self.get_transformed_interpolation_terms(
                    dp, i=i, fwd=True, inplace=True
                )
            self.eval_rho_vj_(f_qg[: self.nalpha], [dp], occd1_feat[i : i + 1])
            self.eval_rho_vj_(occd_f_qg[: self.nalpha], [p], occd2_feat[i : i + 1])
        start = 0 if self.nldf_settings.nldf_type == "i" else self.nalpha
        self._clear_l1_cache(spin)
        self._cache_l1_vectors(f_qg[start:], rho_data[1:4], spin)
        l1_vals = [v for v in self._cached_l1_data[spin]]
        self._clear_l1_cache(spin)
        self._cache_l1_vectors(occd_f_qg[start:], occd_rho_data[1:4], spin)
        l1_occd = [v for v in self._cached_l1_data[spin]]
        i = 0
        for loc in self._l0_start_locs:
            occd1_feat[num_vj + i] = occd_f_qg[start + loc]
            i += 1
        for j, k in self._l1_dots:
            occd1_feat[num_vj + i] = np.einsum(
                "xg,xg->g", l1_vals[j], l1_occd[k]
            ) + np.einsum("xg,xg->g", l1_occd[j], l1_vals[k])
            i += 1
        return self.nspin * (occd1_feat + occd2_feat)

    def eval_vxc_full(
        self,
        vfeat,
        vrho_data,
        dfeat,
        rho_data,
        spin=0,
        p_i_qg=None,
        vf=None,
    ):
        """

        Args:
            vfeat (np.ndarray): (nfeat, ngrids) derivative of energy density
                with respect to features
            vrho_data (np.ndarray): (nrho, ngrids) derivative of energy density
                with respect to density vector [n, dn/dx, dn/dy, dn/dz, (tau)]
            dfeat (np.ndarray): (nfeat, ngrids) derivative of features
                with respect to NLDF exponents.
            rho_data (np.ndarray): density vector
            p_i_qg (np.ndarray, None): Interpolation coefficients for features. If None,
                the features must be cached using cache_p=True during
                eval_rho_full
            vf (np.ndarray, None): Optional buffer for vf

        Returns:
            (np.ndarray): Shape (nalpha + num_vi_ints + vf_buffer_nslot, ngrids)
            or transpose if coef_order == 'gq'.
            The last vf_buffer_nslot rows (qg) or columns (gq)
            are zeros. The remaining rows are filled with the
            functional derivatives with respect to the nonlocal
            density integrals.
        """
        vfeat[:] *= self.nspin
        if vf is None:
            vf = self.zero_coefs_full(vfeat.shape[1])
        if self.coef_order == "qg":
            vf_qg = vf
        else:
            vf_qg = vf.T
        num_vj = self.nldf_settings.num_feat_param_sets
        if p_i_qg is None:
            assert len(self._cached_p_i_qg[spin]) == num_vj
            p_i_qg = self._cached_p_i_qg[spin]
        if self.nldf_settings.nldf_type == "i":
            start = 0
        else:
            start = self.nalpha
            self.eval_vxc_vj_(vfeat[:num_vj], p_i_qg, vf_qg[:start])
        if "i" in self.nldf_settings.nldf_type:
            self.eval_vxc_vi_(vfeat[num_vj:], vrho_data[1:4], vf_qg[start:], spin)
        num_vj = self.nldf_settings.num_feat_param_sets
        rho_tuple = self.get_rho_tuple(rho_data)
        for i in range(num_vj):
            da_tuple = self.get_interpolation_arguments(rho_tuple, i=i)[1]
            vrho_data[0, :] += da_tuple[0] * dfeat[i] * vfeat[i]
            vrho_data[1:4, :] += 2 * rho_data[1:4] * da_tuple[1] * dfeat[i] * vfeat[i]
            if self.nldf_settings.sl_level == "MGGA":
                vrho_data[4, :] += da_tuple[2] * dfeat[i] * vfeat[i]
        # self._clear_p_cache(spin)
        return vf_qg if self.coef_order == "qg" else vf_qg.T

    @abstractmethod
    def _get_interpolation_arguments(self, rho_tuple, i=-1):
        pass

    @abstractmethod
    def _get_interpolation_coefficients(self, arg_g, i=-1, vbuf=None, dbuf=None):
        pass

    @abstractmethod
    def _get_transformed_interpolation_terms(self, p_xx, i=-1, fwd=True, inplace=False):
        pass

    def get_interpolation_arguments(self, rho_tuple, i=-1):
        if i == -1 and self.nldf_settings.nldf_type == "k":
            return self.eval_feat_exp(rho_tuple, i=i)
        else:
            return self._get_interpolation_arguments(rho_tuple, i=i)

    def get_interpolation_coefficients(self, arg_g, i=-1, vbuf=None, dbuf=None):
        if not (-1 <= i < self.nldf_settings.num_feat_param_sets):
            raise ValueError("Unsupported feature index")
        if i == -1 and self.nldf_settings.nldf_type == "k":
            ngrids = arg_g.size
            nalpha = self.local_nalpha
            if self.coef_order == "gq":
                fn = libcider.cider_coefs_vk1_gq
                shape = (ngrids, nalpha)
            else:
                fn = libcider.cider_coefs_vk1_qg
                shape = (nalpha, ngrids)
            p = np.ndarray(shape, dtype=np.float64, buffer=vbuf)
            dp = np.ndarray(shape, dtype=np.float64, buffer=dbuf)
            fn(
                p.ctypes.data_as(ctypes.c_void_p),
                dp.ctypes.data_as(ctypes.c_void_p),
                arg_g.ctypes.data_as(ctypes.c_void_p),
                self.local_alphas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(arg_g.size),
                ctypes.c_int(self.nalpha),
            )
            return p, dp
        else:
            return self._get_interpolation_coefficients(
                arg_g, i=i, vbuf=vbuf, dbuf=dbuf
            )

    def get_transformed_interpolation_terms(self, p_xx, i=-1, fwd=True, inplace=False):
        if i == -1 and self.nldf_settings.nldf_type == "k":
            return p_xx if inplace else p_xx.copy()
        else:
            return self._get_transformed_interpolation_terms(
                p_xx, i=i, fwd=fwd, inplace=inplace
            )

    def _get_coef_shape(self, ngrids, local, nextra=0):
        nalpha = self.local_nalpha if local else self.nalpha
        nalpha += nextra
        if self.coef_order == "gq":
            return (ngrids, nalpha)
        else:
            return (nalpha, ngrids)

    def empty_coefs(self, ngrids, local=True, buf=None):
        shape = self._get_coef_shape(ngrids, local)
        return np.ndarray(shape, dtype=np.float64, order="C", buffer=buf)

    def zero_coefs(self, ngrids, local=True):
        shape = self._get_coef_shape(ngrids, local)
        return np.zeros(shape, dtype=np.float64, order="C")

    def zero_coefs_full(self, ngrids, buffer_nrows=0):
        nextra = buffer_nrows + self.num_vi_ints
        if self.nldf_settings.nldf_type == "i":
            if self.coef_order == "gq":
                shape = (ngrids, nextra)
            else:
                shape = (nextra, ngrids)
        else:
            shape = self._get_coef_shape(ngrids, False, nextra=nextra)
        return np.zeros(shape, dtype=np.float64, order="C")


class NLDFGaussianPlan(NLDFAuxiliaryPlan):
    def _run_setup(self):
        ovlp, _ = _get_ovlp_fit_interpolation_coefficients(
            self, self.alphas, i=-1, local=False
        )
        ovlp *= self.alpha_norms[None, :] * self.alpha_norms[:, None]
        self._dmul = False
        if self._dmul:
            ref = np.identity(ovlp.shape[0]) * self.alpha_norms[:, None]
            self._alpha_transform = _stable_solve(ovlp, ref)
        else:
            self._alpha_transform = ovlp

    def _get_interpolation_arguments(self, rho_tuple, i=-1):
        return self.eval_feat_exp(rho_tuple, i=i)

    def _get_interpolation_coefficients(
        self, arg_g, i=-1, local=True, vbuf=None, dbuf=None
    ):
        p, dp = _get_ovlp_fit_interpolation_coefficients(
            self,
            arg_g,
            i=i,
            local=local,
            vbuf=vbuf,
            dbuf=dbuf,
        )
        return p, dp

    def _get_transformed_interpolation_terms(self, p_xx, i=-1, fwd=True, inplace=False):
        if self.coef_order == "gq":
            in_size = p_xx.shape[1]
        elif self.coef_order == "qg":
            in_size = p_xx.shape[0]
        else:
            raise ValueError
        if in_size != self.nalpha:
            raise ValueError("Wrong number of interpolating points")
        if fwd:
            transform = self._alpha_transform
        else:
            transform = self._alpha_transform.T
        if self.coef_order == "gq":
            p_qu = p_xx.transpose()
        else:
            p_qu = p_xx
        if self._dmul:
            # TODO parallelize with openmp
            if inplace:
                p_qu[:] = transform.dot(p_qu)
            else:
                p_qu = transform.dot(p_qu)
        else:
            if inplace:
                if fwd:
                    p_qu[:] *= self.alpha_norms[:, None]
                p_qu[:] = _stable_solve(transform, p_qu)
                if not fwd:
                    p_qu[:] *= self.alpha_norms[:, None]
            else:
                if fwd:
                    p_qu = self.alpha_norms[:, None] * p_qu
                p_qu = _stable_solve(transform, p_qu)
                if not fwd:
                    p_qu = self.alpha_norms[:, None] * p_qu
        if self.coef_order == "gq":
            p_xx = p_qu.T
        else:
            p_xx = p_qu
        return p_xx


class NLDFSplinePlan(NLDFAuxiliaryPlan):
    def __init__(
        self,
        nldf_settings,
        nspin,
        alpha0,
        lambd,
        nalpha,
        coef_order="gq",
        alpha_formula="etb",
        proc_inds=None,
        spline_size=None,
        rhocut=1e-10,
        expcut=1e-10,
        raise_large_expnt_error=True,
        use_smooth_expnt_cutoff=False,
    ):
        self._spline_size = nalpha if spline_size is None else spline_size
        self._local_alpha_transform = None
        super(NLDFSplinePlan, self).__init__(
            nldf_settings,
            nspin,
            alpha0,
            lambd,
            nalpha,
            coef_order=coef_order,
            alpha_formula=alpha_formula,
            proc_inds=proc_inds,
            rhocut=rhocut,
            expcut=expcut,
            raise_large_expnt_error=raise_large_expnt_error,
            use_smooth_expnt_cutoff=use_smooth_expnt_cutoff,
        )

    def _run_setup(self):
        ovlp, _ = _get_ovlp_fit_interpolation_coefficients(
            self, self.alphas, i=-1, local=False
        )
        ovlp *= self.alpha_norms[None, :] * self.alpha_norms[:, None]
        self._alpha_transform = []
        self._local_alpha_transform = []
        # interp_indexes = np.linspace(0, self.nalpha - 1, self._spline_size)
        interp_indexes = np.arange(0, self._spline_size).astype(np.float64)
        dense_alphas = self.get_q2a(
            interp_indexes * (self.nalpha - 1) / (self._spline_size - 1)
        )
        for i in range(self.nldf_settings.num_feat_param_sets):
            p, _ = _get_ovlp_fit_interpolation_coefficients(
                self, dense_alphas, i=i, local=False
            )
            if self.coef_order == "gq":
                p = p.T
            p[:, :] *= self.alpha_norms[:, None]
            if p.shape == ovlp.shape:
                diff = np.max(np.abs(p * self.alpha_norms - ovlp))
            else:
                diff = 1e10
            if diff > 1e-14:
                coefs_qi = np.ascontiguousarray(_stable_solve(ovlp, p))
            else:
                coefs_qi = np.identity(p.shape[0]) / self.alpha_norms
            self._alpha_transform.append(
                _construct_cubic_splines(coefs_qi, interp_indexes)
            )
        p, _ = _get_ovlp_fit_interpolation_coefficients(
            self, dense_alphas, i=-1, local=False
        )
        if self.coef_order == "gq":
            p = p.T
        p[:, :] *= self.alpha_norms[:, None]
        # coefs_qi = np.ascontiguousarray(_stable_solve(ovlp, p))
        if p.shape == ovlp.shape:
            diff = np.max(np.abs(p * self.alpha_norms - ovlp))
        else:
            diff = 1e10
        if diff > 1e-14:
            coefs_qi = np.ascontiguousarray(_stable_solve(ovlp, p))
        else:
            coefs_qi = np.identity(p.shape[0]) / self.alpha_norms
        self._alpha_transform.append(_construct_cubic_splines(coefs_qi, interp_indexes))
        self._local_alpha_transform = [
            np.ascontiguousarray(t[:, self.proc_inds, :]) for t in self._alpha_transform
        ]
        if self.alpha_formula == "zexp":
            self.alpha_norms[0] = 0  # for numerical stability

    def get_a2q_fast(self, exp_g):
        """A fast (parallel C-backend) version of get_a2q that
        also has a local option to get the q values just
        for the local alphas."""
        di = np.empty_like(exp_g)
        derivi = np.empty_like(exp_g)
        if self.alpha_formula == "etb":
            fn = libcider.cider_ind_etb
        else:
            fn = libcider.cider_ind_zexp
        fn(
            di.ctypes.data_as(ctypes.c_void_p),
            derivi.ctypes.data_as(ctypes.c_void_p),
            exp_g.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(exp_g.size),
            ctypes.c_double(self.alpha0),
            ctypes.c_double(self.lambd),
        )
        if self._spline_size != self.nalpha:
            di[:] *= (self._spline_size - 1) / (self.nalpha - 1)
            derivi[:] *= (self._spline_size - 1) / (self.nalpha - 1)
        libcider.cider_ind_clip(
            di.ctypes.data_as(ctypes.c_void_p),
            derivi.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self._spline_size - 1),
            ctypes.c_int(exp_g.size),
        )
        return di, derivi

    def _get_interpolation_arguments(self, rho_tuple, i=-1):
        a_g, da_tuple = self.eval_feat_exp(rho_tuple, i=i)
        di, derivi = self.get_a2q_fast(a_g)
        for da in da_tuple:
            da[:] *= derivi
        return di, da_tuple

    def _get_interpolation_coefficients(
        self, arg_g, i=-1, local=True, vbuf=None, dbuf=None
    ):
        ngrids = arg_g.size
        nalpha = self.local_nalpha if local else self.nalpha
        if self.coef_order == "gq":
            fn = libcider.cider_coefs_spline_gq
        else:
            fn = libcider.cider_coefs_spline_qg
        p = self.empty_coefs(ngrids, local=local, buf=vbuf)
        dp = self.empty_coefs(ngrids, local=local, buf=dbuf)
        if local:
            w_kap = self._local_alpha_transform[i]
        else:
            w_kap = self._alpha_transform[i]
        fn(
            p.ctypes.data_as(ctypes.c_void_p),
            dp.ctypes.data_as(ctypes.c_void_p),
            arg_g.ctypes.data_as(ctypes.c_void_p),
            w_kap.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(nalpha),
            ctypes.c_double(self.lambd),
        )
        return p, dp

    def _get_transformed_interpolation_terms(self, p_xx, i=-1, fwd=True, inplace=False):
        return p_xx if inplace else p_xx.copy()


def get_diff_gaussian_ovlp(alphas, alpha_norms, flip=False):
    potrf = scipy.linalg.lapack.get_lapack_funcs("potrf")
    trtri = scipy.linalg.lapack.get_lapack_funcs("trtri")

    alpha_s3 = (np.pi / (alphas + alphas[:, None])) ** 1.5
    alpha_s4 = alpha_s3.copy()
    alpha_s4[1:, 1:] += alpha_s3[:-1, :-1]
    alpha_s4[1:, :] -= alpha_s3[:-1, :]
    alpha_s4[:, 1:] -= alpha_s3[:, :-1]
    ad = np.diag(alpha_s4).copy()
    norm4 = ad
    alpha_s4 /= np.sqrt(ad * ad[:, None])
    alpha_s4 = alpha_s4
    transform = np.identity(alpha_s4.shape[0])
    for i in range(alpha_s4.shape[0] - 1):
        transform[i + 1, i] = -1
    transform /= np.sqrt(ad)[:, None]

    if flip:
        alpha_s4 = np.flip(alpha_s4, axis=0)
        alpha_s4 = np.flip(alpha_s4, axis=1)
    tmp, info = potrf(alpha_s4, lower=True)
    tmp, info = trtri(tmp, lower=True)
    tmp = np.tril(tmp)
    alpha_sinv4 = tmp.T.dot(tmp)
    if flip:
        alpha_sinv4 = np.flip(alpha_sinv4, axis=0)
        alpha_sinv4 = np.flip(alpha_sinv4, axis=1)
    transform = np.identity(tmp.shape[0])
    for i in range(tmp.shape[0] - 1):
        transform[i + 1, i] = -1
    transform /= alpha_norms
    transform /= np.sqrt(norm4[:, None])
    transform = transform

    return alpha_s4, alpha_sinv4, norm4, transform, transform.T.dot(tmp.T)
