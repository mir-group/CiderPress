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

import os
from os.path import join

import numpy as np
from pyscf.dft.libxc import eval_xc
from pyscf.lib import prange
from pyscf.lib.scipy_helper import pivoted_cholesky
from scipy.linalg import cho_solve, cholesky, qr, solve_triangular

from ciderpress.density import get_ldax, get_ldax_dens

"""
This module contains utilities that are required for computing exchange
energy covariances between systems for the purpose of training to
total exchange energies.
"""

LDA_FACTOR = -0.75 * (3.0 / np.pi) ** (1.0 / 3)


def compute_cov(model, desca, descb):
    return model.covariance(desca, descb)


def compute_heg_covs(model):
    xdesc = np.zeros((1, model.X.shape[1]))
    xdesc[0, 0] = 1
    return model.covariance_wrt_x_notransform(xdesc).reshape(-1, 1), np.zeros(1)


def reduce_model_size_(model, tol=1e-5, nmax=None):
    S = model.gp.kernel_(model.X, model.X)
    normlz = np.power(np.diag(S), -0.5)
    Snorm = np.dot(np.diag(normlz), np.dot(S, np.diag(normlz)))
    odS = np.abs(Snorm)
    np.fill_diagonal(odS, 0.0)
    odSs = np.sum(odS, axis=0)
    sortidx = np.argsort(odSs)

    Ssort = Snorm[np.ix_(sortidx, sortidx)].copy()
    c, piv, r_c = pivoted_cholesky(Ssort, tol=tol)
    if nmax is not None and nmax < r_c:
        r_c = nmax
    idx = sortidx[piv[:r_c]]
    model.X = np.asfortranarray(model.X[idx])
    model.y = np.asfortranarray(model.y[idx])
    model.gp.X_train_ = model.X
    model.gp.y_train_ = model.y
    model.gp.alpha_ = np.zeros(idx.size)
    # print(model.X.shape, model.y.shape, model.alpha_.shape)
    Knew = model.gp.kernel_(model.X, model.X)
    # oldvals = np.linalg.eigvalsh(S)
    vals = np.linalg.eigvalsh(Knew)
    # print('old eigval range', np.min(oldvals), np.max(oldvals))
    print("eigval range", np.min(vals), np.max(vals))
    return model


def compute_tr_covs(model, dir1, formulas=None, unit=None):
    desc = np.load(join(dir1, "desc.npy"))
    rho = np.load(join(dir1, "rho.npy"))
    val = np.load(join(dir1, "val.npy"))
    wt = np.load(join(dir1, "wt.npy"))
    if os.path.exists(join(dir1, "cut.npy")):
        cut = np.load(join(dir1, "cut.npy"))
    else:
        cut = np.array([wt.size])
    if cut[0] != 0:
        cut = np.append([0], cut)
    N1 = cut.size - 1
    np.zeros((N1, N1))
    print()
    print("SIZE", dir1, N1, wt.size)
    print()
    vwrtt_mat = np.zeros((model.X.shape[0], N1))
    exx = np.zeros(N1)
    blksize = 10000
    for i in range(N1):
        wta = wt[cut[i] : cut[i + 1]]
        rhoa = rho[0, cut[i] : cut[i + 1]]
        n43a = get_ldax_dens(rhoa) * wta
        desca = desc[:, cut[i] : cut[i + 1]]
        vala = val[cut[i] : cut[i + 1]]
        if unit is not None:
            assert unit == "eV"
            ### vala *= 0.0367493
            sprefac = 2 * (3 * np.pi * np.pi) ** (1.0 / 3)
            print("MODEL DESC VERSION", model.desc_version)
            if "b" in model.desc_version:
                print("THIS IS A MGGA")
                rho = desca[0].copy()
                sigma = desca[1].copy()
                tau = desca[2].copy()
                alpha = tau - sigma / (8 * rho + 1e-16)
                ntst = alpha.size
                alpha = np.maximum(alpha, 0)
                assert alpha.size == ntst  # make sure I used maximum correctly
                alpha /= (3.0 / 10) * (3 * np.pi**2) ** (2.0 / 3) * rho ** (
                    5.0 / 3
                ) + 1e-16
                desca[2] = alpha
            desca[1] = desca[1] / (sprefac**2 * desca[0] ** (8.0 / 3) + 1e-16)
        ya = np.zeros(wta.size)
        for i0, i1 in prange(0, wta.size, blksize):
            xa = model.get_descriptors(desca.T[i0:i1])
            ya[i0:i1] = model.gp.predict(xa)
        # TODO check below line
        if model.xed_y_converter[3] == 2:
            _get_fx = model.xed_y_converter[2]
            exx_pred = (ya + _get_fx(desca[1])[0]).dot(n43a)
        else:
            exx_pred = (ya + 1).dot(n43a)
        if formulas is None:
            print(
                "exchange energy pred true pred-true",
                exx_pred,
                vala.dot(wta),
                exx_pred - vala.dot(wta),
            )
        else:
            print(
                "exchange energy",
                formulas[i]["els"],
                formulas[i]["nums"],
                exx_pred,
                vala.dot(wta),
                exx_pred - vala.dot(wta),
            )
        exx[i] = np.dot(vala, wta)
        # TODO check below
        if model.xed_y_converter[3] == 2:
            exx[i] -= np.dot(_get_fx(desca[1])[0], n43a)
        else:
            exx[i] -= np.sum(n43a)
        vwrtt_tot = 0
        for i0, i1 in prange(0, desca.shape[1], blksize):
            vwrtt = model.covariance_wrt_train_set(desca.T[i0:i1])
            vwrtt_tot += vwrtt.dot(n43a[i0:i1])
            print(vwrtt_tot.shape)
        vwrtt_mat[:, i] = vwrtt_tot
        vwrtt = None
    # print(vwrtt_mat.shape)
    return vwrtt_mat, exx


def compute_x_pred(dir1, model_type, model=None):
    desc = np.load(join(dir1, "desc.npy"))
    rho = np.load(join(dir1, "rho.npy"))
    val = np.load(join(dir1, "val.npy"))
    wt = np.load(join(dir1, "wt.npy"))
    if os.path.exists(join(dir1, "cut.npy")):
        cut = np.load(join(dir1, "cut.npy"))
    else:
        cut = np.array([wt.size])
    if cut[0] != 0:
        cut = np.append([0], cut)
    N1 = cut.size - 1
    exx_pred = np.zeros(N1)
    blksize = 10000
    if rho.shape[0] == 1:
        rhotmp = rho
        rho = np.zeros((4, rho.shape[1]))
        rho[0] = rhotmp[0]
        sprefac = 2 * (3 * np.pi * np.pi) ** (1.0 / 3)
        n83 = sprefac**2 * rhotmp[0] ** (8.0 / 3)
        rho[1] = np.sqrt(desc[1] * n83)
    for i in range(N1):
        wta = wt[cut[i] : cut[i + 1]]
        rhoa = rho[0, cut[i] : cut[i + 1]]
        n43a = get_ldax_dens(rhoa) * wta
        desca = desc[:, cut[i] : cut[i + 1]]
        vala = val[cut[i] : cut[i + 1]]
        ya = np.zeros(wta.size)
        if model_type == "EXX":
            exx_pred[i] = np.dot(vala, wta)
        elif model_type == "ML":
            for i0, i1 in prange(0, wta.size, blksize):
                ya[i0:i1] = model.get_F(desca[:, i0:i1])
            exx_pred[i] = np.dot(ya, n43a)
        elif model_type == "SL":
            exc = eval_xc(model, rho[:, cut[i] : cut[i + 1]])[0]
            exx_pred[i] = np.dot(exc, wta * rhoa)
        else:
            raise ValueError("Unrecognized model type")
    return exx_pred


def compute_tr_covs_ex(model, dir1):
    cut = np.load(join(dir1, "cut.npy"))
    desc = np.load(join(dir1, "desc.npy"))
    rho = np.load(join(dir1, "rho.npy"))
    val = np.load(join(dir1, "val.npy"))
    wt = np.load(join(dir1, "wt.npy"))
    N1 = cut.size
    np.zeros((N1, N1))
    cut = np.append([0], cut)
    vwrtt_mat = np.zeros((model.X.shape[0], N1))
    exx = np.zeros(N1)
    for i in range(N1):
        wta = wt[cut[i] : cut[i + 1]]
        rhoa = rho[0, cut[i] : cut[i + 1]]
        na = rhoa * wta
        n43a = get_ldax_dens(rhoa) * wta
        n13a = get_ldax(rhoa)
        desca = desc[:, cut[i] : cut[i + 1]]
        vala = val[cut[i] : cut[i + 1]]
        xa = model.get_descriptors(desca.T)
        ya = model.gp.predict(xa)
        # TODO check below line
        _get_fx = model.xed_y_converter[2]
        exx_pred = (ya + _get_fx(desca[1])[0] * n13a).dot(na)
        print(
            "exchange energy pred true pred-true",
            exx_pred,
            vala.dot(wta),
            exx_pred - vala.dot(wta),
        )
        exx[i] = np.dot(vala, wta)
        # TODO check below
        exx[i] -= np.dot(_get_fx(desca[1])[0], n43a)
        vwrtt = model.covariance_wrt_train_set(desca.T)
        vwrtt = vwrtt.dot(na)
        vwrtt_mat[:, i] = vwrtt
        vwrtt = None
    # print(vwrtt_mat.shape)
    return vwrtt_mat, exx


def compute_covs(model, dir1, dir2=None):
    cut = np.load(join(dir1, "cut.npy"))
    desc = np.load(join(dir1, "desc.npy"))
    rho = np.load(join(dir1, "rho.npy"))
    #'settings.yaml'
    val = np.load(join(dir1, "val.npy"))
    wt = np.load(join(dir1, "wt.npy"))
    N1 = cut.size
    cov_mat_mol = np.zeros((N1, N1))
    cut = np.append([0], cut)
    np.zeros(N1)
    print(rho.shape, desc.shape, val.shape, wt.shape)
    for i in range(N1):
        wta = wt[cut[i] : cut[i + 1]]
        rhoa = rho[0, cut[i] : cut[i + 1]]
        n43a = get_ldax_dens(rhoa) * wta
        desca = desc[:, cut[i] : cut[i + 1]]
        xa = model.get_descriptors(desca.T)
        ya = model.gp.predict(xa)
        _get_fx = model.xed_y_converter[2]
        ya = ya + _get_fx(desca[1])[0]
        # ya = model.y_to_xed(ya, desca.T)
        fpred = model.predict(desca.T)
        print("exchange energy", fpred.dot(wt[cut[i] : cut[i + 1]]))
        # print('exchange energy', ya.dot(wt[cut[i]:cut[i+1]]))
        print("exchange energy", ya.dot(n43a))
        print("exact", np.dot(val[cut[i] : cut[i + 1]], wt[cut[i] : cut[i + 1]]))
        vwrtt = model.covariance_wrt_train_set(desca.T)
        vwrtt = vwrtt.dot(n43a)
        print(vwrtt.shape)
        for j in range(N1):
            wtb = wt[cut[j] : cut[j + 1]]
            n43b = get_ldax_dens(rho[0, cut[j] : cut[j + 1]]) * wtb
            descb = desc[:, cut[j] : cut[j + 1]]
            cov_mat = compute_cov(model, desca.T, descb.T)
            cov_ex = cov_mat.dot(n43b)
            cov_ex = n43a.dot(cov_ex)
            cov_mat_mol[i, j] = cov_ex
    print(cov_mat_mol)
    # print(np.linalg.eigvalsh(cov_mat_mol))


def compute_new_alpha(
    model,
    vwrtt_mat,
    exx,
    use_cho=False,
    version=0,
    debug=False,
    rho=False,
    molsigma=None,
    skip_freq=0,
):
    # print(vwrtt_mat.shape, exx.shape, model.X.shape)
    if debug:
        model.gp.optimizer = None
        model.gp.fit(model.X, model.y)
        print("ALPHA", model.gp.alpha)

    Kex = model.gp.kernel_(model.X)
    Kmm = model.gp.kernel_(model.X, model.X)
    M = Kmm.shape[0]
    N = M + vwrtt_mat.shape[1]
    # print(N, M)
    noise_mm = np.diag(model.gp.kernel_(model.X) - Kmm)
    if skip_freq > 0:
        noise_mm = noise_mm[::skip_freq]

    if molsigma is None:
        molsigma = np.sqrt(1e-5)
    if isinstance(molsigma, np.ndarray) and molsigma.size == (N - M):
        print("USING SIGMA LIST")
        noise_nn = np.append(noise_mm, molsigma**2)
    else:
        noise_nn = np.append(noise_mm, [molsigma**2] * (N - M))

    if skip_freq == 0:
        Kmn = np.hstack((Kmm, vwrtt_mat))
    else:
        Kmn = np.hstack((Kmm[:, ::skip_freq], vwrtt_mat))

    if rho == False:
        rho = None
    else:
        rho = model.X[:, 0]

    if version == 9:
        alpha = 1e-9
        Kmn = vwrtt_mat
        L_ = cholesky(Kmm + np.identity(M) * alpha, lower=True)
        Kimn = cho_solve((L_, True), Kmn)
        Knmimn = Kmn.T.dot(Kimn)
        K = Knmimn + np.diag(noise_nn[M - N :]) + np.identity(N - M) * alpha
        yvec = exx.copy()
    elif version == 8:
        noise_param = np.mean(noise_mm)
        print("v8", noise_param)
        K = Kmn.dot(Kmn.T) + noise_param * Kmm + np.identity(noise_mm.size) * 1e-9
        yvec = Kmn.dot(np.append(model.y, exx))
    elif version == 7:
        alpha = 1e-9
        L_ = cholesky(Kmm + np.identity(M) * alpha, lower=True)
        Kimn = cho_solve((L_, True), Kmn)
        Knmimn = Kmn.T.dot(Kimn)
        K = Knmimn + np.diag(noise_nn) + np.identity(noise_nn.size) * alpha
        if skip_freq == 0:
            yvec = np.append(model.y, exx)
        else:
            yvec = np.append(model.y[::skip_freq], exx)
    elif version == 6:
        alpha = 1e-8
        vals, vecs = np.linalg.eigh(Kmm)
        cond = vals > 1e-10
        Kppi, Kpp, Ump = np.diag(1 / vals[cond]), np.diag(vals[cond]), vecs[:, cond]
        # L_ = cholesky(Kmm+np.identity(M)*alpha, lower=True)
        # Kimn = cho_solve((L_, True), Kmn)
        # Knmimn = Kmn.T.dot(Kimn)
        Kpn = Ump.T.dot(Kmn)
        Kipn = Kppi.dot(Kpn)
        Knpipn = Kpn.T.dot(Kipn)
        K = Knpipn + np.diag(noise_nn) + np.identity(noise_nn.size) * alpha
        yvec = np.append(model.y, exx)
    elif version == 5:
        print("v5")
        vals, vecs = np.linalg.eigh(Kmm)
        K = (
            Kmn.dot(Kmn.T)
            + Kmm * np.sqrt(noise_mm[:, None] * noise_mm)
            + 1e-9 * np.identity(M)
        )
        yvec = Kmn.dot(np.append(model.y, exx))
    elif version == 4 and (rho is not None):
        rhomat = np.outer(rho ** (1.0 / 3), rho ** (1.0 / 3))
        Kex *= rhomat
        Kmm *= rhomat
        Kmn[:, :M] *= rhomat
        for j in range(M, N):
            Kmn[:, j] *= rho ** (1.0 / 3)
        noise_param = 1e-5
        K = Kmn.dot(Kmn.T) + noise_param * Kmm + np.identity(M) * 1e-9
        yvec = Kmn.dot(np.append(model.y * rho ** (1.0 / 3), exx))
    elif version == 4:
        vals, vecs = np.linalg.eigh(Kmm)
        cond = vals > 1e-8
        P = cond.sum()
        Kpp, Ump = np.diag(vals[cond]), vecs[:, cond]
        Kpn = Ump.T.dot(Kmn)  # TODO Kmm -> Kmn
        Kpn_n = Kpn / noise_nn  # TODO noise_mm -> noise_nn
        # Kpn_n = Kpn * (noise_mm[:,None] * noise_mm) # TODO noise_mm -> noise_nn
        K = Kpn_n.dot(Kpn.T) + Kpp + 1e-9 * np.identity(P)
        yvec = Kpn_n.dot(np.append(model.y, exx))  # TODO append exx
        # L_ = cholesky(Kmm + np.identity(M) * 1e-10, lower=True)
        # Kimn = cho_solve((L_, True), Kmm)
        # Knmimn = Kmn.T.dot(Kimn)
        # K = None
    elif version == 3:
        if False:
            L_ = cholesky(Kmm, lower=True)
        else:
            q, r = qr(Kmm + np.identity(M) * 1e-10)
            theta = solve_triangular(r, q.T.dot(Kmn))
            K = Kmn.T.dot(theta) + np.diag(noise_nn)
            yvec = np.append(model.y, exx)
    elif version == 2:
        Kmnn = Kmn * (noise_mm[:, None] / noise_nn)
        # Kmnn = Kmn * (1 / noise_nn)
        K = Kmnn.dot(Kmn.T) + noise_mm[:, None] * Kmm + 1e-10 * np.identity(M)
        # K = Kmnn.dot(Kmn.T) + Kmm + 1e-10 * np.identity(M)
        yvec = Kmnn.dot(np.append(model.y, exx))
        # K = Kmm + noise * np.identity(M) ### TODO remove
    elif version == 1:
        noise_mm = np.ones(M) * 1e-6
        # K = Kmm.dot(1/noise_tmp).dot(Kmm) + Kex
        Kmnn = Kmm * (noise_mm[:, None] / noise_mm)
        K = Kmnn.dot(Kmm) + noise_mm[:, None] * Kex
        K += np.identity(M) * model.gp.alpha
        yvec = Kmnn.dot(model.y)
        np.linalg.eigvals(K)
        # print(np.sort(np.real(Keig)))
    else:
        K = Kex + np.identity(M) * model.gp.alpha
        yvec = model.y

    if use_cho:
        L_ = cholesky(K, lower=True)
        alpha_new = cho_solve((L_, True), yvec)
    else:
        print(K.shape, yvec.shape)
        q, r = qr(K)
        alpha_new = solve_triangular(r, q.T.dot(yvec), lower=False)
    if version == 4 and (rho is None):
        alpha_new = Ump.dot(alpha_new)
    elif version == 4:
        alpha_new *= rho ** (1.0 / 3)
    elif version == 6:
        # alpha_new = np.dot(Kimn, alpha_new)
        alpha_new = np.dot(Ump, np.dot(Kipn, alpha_new))
    elif version == 7:
        alpha_new = np.dot(Kimn, alpha_new)
    elif version == 9:
        alpha_new = np.dot(Kimn, alpha_new)

    print(
        np.linalg.norm(model.gp.alpha_),
        np.linalg.norm(alpha_new[:M]),
        np.linalg.norm(alpha_new[:M] - model.gp.alpha_),
    )
    return alpha_new
