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
from scipy.linalg import cholesky


def gauss_fft(n, a, k):
    # \int_0^\infty dr r^(2+n) exp(-ar^2) j_n(k*r)
    return (
        2 ** (-2 - n)
        * a ** (-1.5 - n)
        * np.exp(-(k**2) / (4 * a))
        * k**n
        * np.sqrt(np.pi)
    )


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


def get_ffunc(n, r, R):
    x = r / R
    xn = x**n
    y = xn * (x * x - 1) * (x * x - 1)
    y[x > 1] = 0.0
    return y


def get_poly(n, r, R):
    x = r / R
    xn = x**n
    xn[x > 1] = 0.0
    return xn


PFUNC2_NORMS = np.array(
    [
        5.77240697e00,
        3.07271389e00,
        3.14265261e00,
        2.13897490e01,
        1.94289398e01,
        1.33574495e02,
        1.14971278e02,
        7.98033330e02,
        6.65701240e02,
        4.65664626e03,
        3.80496685e03,
    ],
    dtype=np.float64,
)


def get_pfunc(n, r, R, enforce_cutoff=True):
    x = np.pi * r / R
    xp = r / R
    if n == 0:
        y = 0.5 + 0.5 * np.cos(x)
    elif n % 2 == 1:
        y1 = np.sin(0.5 * x) - 0.5 + 0.5 * np.cos(x)
        y1 *= 5
        y = xp ** (n - 1) * np.exp(-(n - 1) * xp**2) * y1
    else:
        y2 = 0.5 - 0.5 * np.cos(2 * x)
        y = xp ** (n - 2) * np.exp(-(n - 2) * xp**2) * y2
    if enforce_cutoff:
        y[r > R] = 0
    return y


def get_pfunc_norm(n, r, R, enforce_cutoff=True):
    if n >= len(PFUNC2_NORMS):
        raise ValueError
    res = get_pfunc(n, r, R, enforce_cutoff=True)
    res *= PFUNC2_NORMS[n] / R**1.5
    return res


def get_pfuncs_k(pfuncs_g, ls, rgd, ns=None):
    if ns is not None:
        l_add = ns - ls
    else:
        l_add = np.zeros_like(ls)
    n_pfuncs = pfuncs_g.shape[0]
    pfuncs_k = pfuncs_g.copy()
    for i in range(n_pfuncs):
        pfuncs_k[i] = rgd.transform_single_fwd(pfuncs_g[i], ls[i], l_add=l_add[i])
    return pfuncs_k


def get_phi_iabk(pfuncs_k, ks, alphas, betas=None):
    if betas is None:
        betas = alphas
    nalpha = alphas.shape[0]
    nbeta = betas.shape[0]
    n_pfuncs = pfuncs_k.shape[0]
    nk = ks.size
    phi_iabk = np.empty((n_pfuncs, nalpha, nbeta, nk))
    k2s = ks * ks
    for a in range(nalpha):
        for b in range(nbeta):
            aexp = alphas[a] + betas[b]
            expnt = 1 / (4 * aexp)
            prefac = (np.pi / aexp) ** 1.5
            kernel = prefac * np.exp(-expnt * k2s)
            for i in range(n_pfuncs):
                phi_iabk[i, a, b, :] = kernel * pfuncs_k[i]
    return phi_iabk


def get_phi_iabg(phi_iabk, ls, rgd):
    phi_iabg = phi_iabk.copy()
    n_pfuncs = phi_iabk.shape[0]
    nalpha = phi_iabk.shape[1]
    nbeta = phi_iabk.shape[2]
    for i in range(n_pfuncs):
        for a in range(nalpha):
            for b in range(nbeta):
                phi_iabg[i, a, b, :] = rgd.transform_single_bwd(
                    phi_iabk[i, a, b, :], ls[i]
                )
    return phi_iabg


def get_dv(rgd):
    return rgd.dr_g * rgd.r_g**2


def get_dvk(rgd):
    return 2 / np.pi * rgd.k_g**2 * rgd.dk_g


def get_p11_matrix(delta_l_pg, rgd, reg=0):
    dv_g = get_dv(rgd)
    p11_l_ii = []
    for l, delta_pg in enumerate(delta_l_pg):
        mat = np.einsum("pg,qg,g->pq", delta_pg, delta_pg, dv_g)
        mat[:] += reg * np.identity(mat.shape[-1])
        p11_l_ii.append(mat)
    return p11_l_ii


def get_p12_p21_matrix(delta_l_pg, phi_jabg, rgd, w_b, nbas_loc):
    dv_g = get_dv(rgd)
    p12_vbja = []
    p21_javb = []
    lp1 = len(delta_l_pg)
    for l in range(lp1):
        j0, j1 = nbas_loc[l], nbas_loc[l + 1]
        mat = np.einsum("vg,jabg,g->javb", delta_l_pg[l], phi_jabg[j0:j1], dv_g)
        mat[:] *= w_b
        p21_javb.append(mat)
        p12_vbja.append(mat.transpose(2, 3, 0, 1))
    return p12_vbja, p21_javb


def get_p22_matrix(phi_iabg, rgd, rcut, l, w_b, nbas_loc, reg=0, cut_func=False):
    dv_g = get_dv(rgd)
    phicut_iabg = phi_iabg[nbas_loc[l] : nbas_loc[l + 1]] * np.sqrt(dv_g)
    if cut_func:
        x = rgd.r_g / (0.75 * rcut)
        if cut_func == 2:
            fcut = np.ones_like(x)
        else:
            fcut = 0.5 + 0.5 * np.cos(np.pi * x)
        fcut[x > 1] = 0.0
        phicut_iabg *= np.sqrt(fcut)
    p22_jaja = np.einsum("iacg,jbcg,c->iajb", phicut_iabg, phicut_iabg, w_b)
    nj, na = p22_jaja.shape[:2]
    p22_ii = p22_jaja.view()
    p22_ii.shape = (nj * na, nj * na)
    # TODO this is close to previous regularization, but not
    # necessarily the optimal regularization
    reg = reg / np.tile(w_b, nj)
    p22_ii[:] += reg * np.identity(nj * na)
    return p22_jaja


def construct_full_p_matrices(p11_l_vv, p12_l_vbja, p21_l_javb, p22_l_jaja, w_b):
    lp1 = len(p11_l_vv)
    p_l_ii = []
    for l in range(lp1):
        nv = p11_l_vv[l].shape[0]
        nb = w_b.size
        N1 = nv * nb
        p11_pbpb = np.zeros((nv, nb, nv, nb))
        for b in range(nb):
            p11_pbpb[:, b, :, b] = p11_l_vv[l] * w_b[b]
        p11_ii = p11_pbpb.reshape(N1, N1)
        p12_vbja = p12_l_vbja[l]
        p21_javb = p21_l_javb[l]
        p22_jaja = p22_l_jaja[l]
        N2 = p22_jaja.shape[0] * p22_jaja.shape[1]
        N = N1 + N2
        pl_ii = np.zeros((N, N))
        pl_ii[:N1, :N1] = p11_ii
        pl_ii[:N1, N1:] = p12_vbja.reshape(N1, N2)
        pl_ii[N1:, :N1] = p21_javb.reshape(N2, N1)
        pl_ii[N1:, N1:] = p22_jaja.reshape(N2, N2)
        p_l_ii.append(pl_ii)
    return p_l_ii


def get_delta_lpg(betas_lb, rcut, rgd, pmin, pmax):
    r_g = rgd.r_g
    dv_g = get_dv(rgd)
    lp1 = len(betas_lb)
    delta_lpg = []
    for l in range(lp1):
        cond = np.logical_and(
            betas_lb[l] > pmin,
            betas_lb[l] < pmax,
        )
        betas = betas_lb[l][cond]
        fcut = (0.5 * np.pi / betas) ** 0.75 * gauss_and_derivs(betas, rcut)
        funcs = (0.5 * np.pi / betas[:, None]) ** 0.75 * np.exp(
            -betas[:, None] * r_g * r_g
        )
        rd_g = rgd.r_g - rcut
        poly = (
            fcut[0, :, None]
            + rd_g * fcut[1, :, None]
            + rd_g**2 * fcut[2, :, None] / 2
            + rd_g**3 * fcut[3, :, None] / 6
        )
        dpoly = fcut[1] - rcut * fcut[2] + rcut**2 * fcut[3] / 2
        poly += dpoly[:, None] / (4 * rcut**3) * rd_g**4
        funcs = funcs - poly
        funcs[:, rgd.r_g > rcut] = 0
        funcs_tmp = funcs * r_g**l * np.sqrt(dv_g)
        ovlp = np.einsum("ig,jg->ij", funcs_tmp, funcs_tmp)
        L = cholesky(ovlp, lower=True)
        basis = np.linalg.solve(L, funcs)
        delta_lpg.append(basis * r_g**l)
    return delta_lpg
