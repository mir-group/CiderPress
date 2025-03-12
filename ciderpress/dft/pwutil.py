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

from ciderpress.lib import load_library

pw_cutil = load_library("libpwutil")


def eval_pasdw_funcs(radfuncs_ng, ylm_lg, nlist_i, lmlist_i):
    assert radfuncs_ng.flags.c_contiguous
    assert ylm_lg.flags.c_contiguous
    assert nlist_i.flags.c_contiguous
    assert lmlist_i.flags.c_contiguous
    nlm, ng = ylm_lg.shape
    assert radfuncs_ng.shape[1] == ng
    ni = len(nlist_i)
    assert len(nlist_i) == len(lmlist_i)
    funcs_ig = np.empty((ni, ng), order="C")
    pw_cutil.eval_pasdw_funcs(
        radfuncs_ng.ctypes.data_as(ctypes.c_void_p),
        ylm_lg.ctypes.data_as(ctypes.c_void_p),
        funcs_ig.ctypes.data_as(ctypes.c_void_p),
        nlist_i.ctypes.data_as(ctypes.c_void_p),
        lmlist_i.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ni),
        ctypes.c_int(ng),
        ctypes.c_int(nlm),
    )
    return funcs_ig


def pasdw_reduce(fn, coefs_i, funcs_ig, augfeat_g, indset):
    assert coefs_i.flags.c_contiguous
    assert funcs_ig.flags.c_contiguous
    assert augfeat_g.flags.c_contiguous
    assert indset.flags.c_contiguous
    ni, ng = funcs_ig.shape
    assert coefs_i.size == ni
    assert indset.shape == (ng, 3)
    assert indset.dtype == np.int32
    n1, n2, n3 = augfeat_g.shape
    fn(
        coefs_i.ctypes.data_as(ctypes.c_void_p),
        funcs_ig.ctypes.data_as(ctypes.c_void_p),
        augfeat_g.ctypes.data_as(ctypes.c_void_p),
        indset.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ni),
        ctypes.c_int(ng),
        ctypes.c_int(n1),
        ctypes.c_int(n2),
        ctypes.c_int(n3),
    )


def pasdw_reduce_i(coefs_i, funcs_ig, augfeat_g, indset):
    pasdw_reduce(pw_cutil.pasdw_reduce_i, coefs_i, funcs_ig, augfeat_g, indset)


def pasdw_reduce_g(coefs_i, funcs_ig, augfeat_g, indset):
    pasdw_reduce(pw_cutil.pasdw_reduce_g, coefs_i, funcs_ig, augfeat_g, indset)


def _eval_cubic_spline(fn, funcs_ntp, t_g, dt_g):
    nn, nt, _np = funcs_ntp.shape
    assert _np == 4
    ng = t_g.size
    assert dt_g.size == ng
    assert funcs_ntp.flags.c_contiguous
    assert t_g.flags.c_contiguous
    assert dt_g.flags.c_contiguous
    radfuncs_ng = np.empty((nn, ng), order="C")
    fn(
        funcs_ntp.ctypes.data_as(ctypes.c_void_p),
        radfuncs_ng.ctypes.data_as(ctypes.c_void_p),
        t_g.ctypes.data_as(ctypes.c_void_p),
        dt_g.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nn),
        ctypes.c_int(nt),
        ctypes.c_int(ng),
    )
    return radfuncs_ng


def eval_cubic_spline(funcs_ntp, t_g, dt_g):
    return _eval_cubic_spline(pw_cutil.eval_cubic_spline, funcs_ntp, t_g, dt_g)


def eval_cubic_spline_deriv(funcs_ntp, t_g, dt_g):
    return _eval_cubic_spline(pw_cutil.eval_cubic_spline_deriv, funcs_ntp, t_g, dt_g)


def _eval_cubic_interp(i_g, t_g, c_ip, with_deriv):
    for arr, dtype in zip([i_g, t_g, c_ip], [np.int32, np.float64, np.float64]):
        assert arr.dtype == dtype
        assert arr.flags.c_contiguous
    ng = i_g.size
    ni = c_ip.shape[0]
    assert c_ip.shape[1] == 4
    y_g = np.empty(ng, order="C")
    args = [
        i_g.ctypes.data_as(ctypes.c_void_p),
        t_g.ctypes.data_as(ctypes.c_void_p),
        c_ip.ctypes.data_as(ctypes.c_void_p),
        y_g.ctypes.data_as(ctypes.c_void_p),
    ]
    if with_deriv:
        dy_g = np.empty(ng, order="C")
        args.append(dy_g.ctypes.data_as(ctypes.c_void_p))
        res = (y_g, dy_g)
        fn = pw_cutil.eval_cubic_interp
    else:
        res = y_g
        fn = pw_cutil.eval_cubic_interp_noderiv
    args += [
        ctypes.c_int(ng),
        ctypes.c_int(ni),
    ]
    fn(*args)
    return res


def eval_cubic_interp(i_g, t_g, c_ip):
    return _eval_cubic_interp(i_g, t_g, c_ip, True)


def eval_cubic_interp_noderiv(i_g, t_g, c_ip):
    return _eval_cubic_interp(i_g, t_g, c_ip, False)


def recursive_sph_harm(nlm, rhat_gv):
    n = rhat_gv.shape[0]
    res = np.empty((n, nlm), order="C")
    assert rhat_gv.flags.c_contiguous
    pw_cutil.recursive_sph_harm_vec(
        ctypes.c_int(nlm),
        ctypes.c_int(n),
        rhat_gv.ctypes.data_as(ctypes.c_void_p),
        res.ctypes.data_as(ctypes.c_void_p),
    )
    return res


def recursive_sph_harm_deriv(nlm, rhat_gv):
    n = rhat_gv.shape[0]
    res = np.empty((n, nlm), order="C")
    dres = np.empty((n, 3, nlm), order="C")
    assert rhat_gv.flags.c_contiguous
    pw_cutil.recursive_sph_harm_deriv_vec(
        ctypes.c_int(nlm),
        ctypes.c_int(n),
        rhat_gv.ctypes.data_as(ctypes.c_void_p),
        res.ctypes.data_as(ctypes.c_void_p),
        dres.ctypes.data_as(ctypes.c_void_p),
    )
    return res, dres


def mulexp(F_k, theta_k, k2_k, a, b):
    nk = F_k.size
    assert F_k.size == theta_k.size == k2_k.size
    return pw_cutil.mulexp(
        F_k.ctypes.data_as(ctypes.c_void_p),
        theta_k.ctypes.data_as(ctypes.c_void_p),
        k2_k.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(a),
        ctypes.c_double(b),
        ctypes.c_int(nk),
    )
