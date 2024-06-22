import ctypes

import numpy as np

from ciderpress.dft.futil import sph_nlxc_mod as fnlxc
from ciderpress.lib import load_library

pw_cutil = load_library("libpwutil.so")


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


def eval_cubic_spline(*args):
    return fnlxc.eval_cubic_spline(*args)


def eval_cubic_spline_deriv(*args):
    return fnlxc.eval_cubic_spline_deriv(*args)


def eval_cubic_interp(*args):
    return fnlxc.eval_cubic_interp(*args)


def eval_cubic_interp_noderiv(*args):
    return fnlxc.eval_cubic_interp_noderiv(*args)


def recursive_sph_harm_t2(nlm, rhat_gv):
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


def recursive_sph_harm_t2_deriv(nlm, rhat_gv):
    n = rhat_gv.shape[0]
    res = np.empty((n, nlm), order="C")
    dres = np.empty((n, 3, nlm), order="C")
    assert rhat_gv.flags.c_contiguous
    pw_cutil.recursive_sph_harm_vec(
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
