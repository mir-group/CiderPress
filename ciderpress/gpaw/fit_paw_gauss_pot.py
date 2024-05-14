import numpy as np
from scipy.linalg import cholesky
from scipy.special import jv

from ciderpress.dft.sphere_util import (
    gauss_and_derivs,
    gauss_conv_coef_and_expnt,
    gauss_fft,
)


def get_pf_integral(n, R):
    # \int_0^R dr r^2 (R^2-r^2)^2 * r^n / R^(n+4)
    return 8 * R * R * R / ((3 + n) * (5 + n) * (7 + n))


def get_ff_integral(n, R):
    # \int_0^R dr r^2 r^n / R^n
    return R * R * R / (3 + n)


def get_pfunc(n, r, R, enforce_cutoff=True):
    res = (R * R - r * r) ** 2 * r**n / R ** (4 + n)
    if enforce_cutoff:
        res[r > R] = 0
    return res


def get_ffunc(n, r, R):
    return r**n / R**n


def get_ffunc2(n, r, R, taper=0.5):
    # x = r/R
    # xn = x**n
    # y = xn * (x*x-4) * (x*x-4)
    # y[x>2] = 0.0
    # return y
    x = r**n / R**n
    return x * np.exp(-0.5 * (n + 1) * (r / R) * (r / R))


def get_ffunc3(n, r, R):
    x = r / R
    xn = x**n
    y = xn * (x * x - 1) * (x * x - 1)
    y[x > 1] = 0.0
    return y


def get_ffunc4_real(n, r, R):
    A = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 1], [0, 1, 2, 3]], dtype=np.float64
    )
    b = np.array([1, n, 0, 0], dtype=np.float64)
    x = np.linalg.solve(A, b)
    dr = r / R
    drp = dr[dr > 1] - 1
    res = dr**n
    res[dr > 1] = x[0] + drp * (x[1] + drp * (x[2] + drp * x[3]))
    res[dr > 2] = 0
    return res


def get_ffunc4_ft(n, l, k, R):
    n = n - l
    assert n % 2 == 0
    n = n // 2
    k = k * R
    prefac = R ** (3 + l + 2 * n)

    j0 = jv(1.5 + n + l, k)
    j1 = jv(2.5 + n + l, k)

    if n == 0:
        func = j0
    elif n == 1:
        func = (3 + 2 * l) * j0 - k * j1
    elif n == 2:
        p0 = -1 * (k * k - (3 + 2 * l) * (5 + 2 * l))
        p1 = -1 * k * (1 + 2 * l)
        func = p0 * j0 + p1 * j1
    else:
        raise ValueError

    res = prefac * func * np.sqrt(0.5 * np.pi) / k ** (n + 1.5)

    thr = 1e-10
    if l == 0:
        res[k < thr] = prefac / (3 + l + 2 * n)
    else:
        thr = thr ** (1.0 / l)
        res[k < thr] = 0

    return res


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


def get_pfunc2(n, r, R, enforce_cutoff=True):
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


def get_pfunc2_norm(n, r, R, enforce_cutoff=True):
    if n >= len(PFUNC2_NORMS):
        raise ValueError
    res = get_pfunc2(n, r, R, enforce_cutoff=True)
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


def get_f_b(theta_aLg, rgd, alphas, k2r=False):
    nalpha = alphas.size
    k2s = rgd.k_g * rgd.k_g
    theta_aLk = theta_aLg.copy()
    rgd.transform_set_fwd(theta_aLg, theta_aLk)
    f_bLk = np.zeros_like(theta_aLk)
    for b in range(nalpha):
        for a in range(nalpha):
            aexp = alphas[a] + alphas[b]
            expnt = 1 / (4 * aexp)
            prefac = (np.pi / aexp) ** 1.5
            kernel = prefac * np.exp(-expnt * k2s)
            f_bLk[b, :, :] += theta_aLk[a, :, :] * kernel
    if k2r:
        f_bLx = f_bLk.copy()
        rgd.transform_set_bwd(f_bLk, f_bLx)
    else:
        f_bLx = f_bLk
    return f_bLx


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


NBAS_LST = [3, 2, 2, 1, 1]
NAO_LST = [3 * 1, 2 * 3, 2 * 5, 1 * 7, 1 * 9]
NBAS_LOC = np.append([0], np.cumsum(NBAS_LST)).astype(np.int32)


def get_phi_ovlp_direct(phi_iabg, rgd, rcut, l, nbas_loc=NBAS_LOC, w_b=None):
    # dv = 4 * np.pi * rgd.dr_g * rgd.r_g**2
    dv = rgd.dr_g * rgd.r_g**2
    if w_b is None:
        w_b = np.ones(phi_iabg.shape[-2])
    phicut_iabg = phi_iabg[nbas_loc[l] : nbas_loc[l + 1]] * np.sqrt(dv)
    phicut_iabg = phicut_iabg[..., rgd.r_g > rcut]
    return np.einsum("iacg,jbcg,c->iajb", phicut_iabg, phicut_iabg, w_b)


def get_phi_solve_direct(phi_iabg, phi_iajb, l, nbas_loc=NBAS_LOC, w_b=None):
    reg = 1e-9
    phi_iabg = phi_iabg[nbas_loc[l] : nbas_loc[l + 1]]
    n_i, n_a = phi_iabg.shape[:2]
    n_ia = n_i * n_a
    return np.linalg.solve(
        phi_iajb.reshape(n_ia, n_ia) + reg * np.identity(n_ia),
        phi_iabg.reshape(n_ia, -1),
    ).reshape(*(phi_iabg.shape))


def get_phi_ovlp_f(phi_iabg, f_Lbg, rgd, l, rcut, nbas_loc=NBAS_LOC, w_b=None):
    # TODO will need L channel on f_bg and construct separate ia
    # matrix for each L
    # dv = 4 * np.pi * rgd.dr_g * rgd.r_g**2
    if w_b is None:
        w_b = np.ones(phi_iabg.shape[-2])
    dv = rgd.dr_g * rgd.r_g**2
    f_bg = f_Lbg[l * l : (l + 1) * (l + 1)] * dv
    phicut_iabg = phi_iabg[nbas_loc[l] : nbas_loc[l + 1], ..., rgd.r_g > rcut]
    fcut_bg = f_bg[..., rgd.r_g > rcut]
    return np.einsum("iabg,mbg,b->ima", phicut_iabg, fcut_bg, w_b)


def get_dv(rgd):
    return rgd.dr_g * rgd.r_g**2


def get_dvk(rgd):
    return 2 / np.pi * rgd.k_g**2 * rgd.dk_g


def get_p11_matrix(delta_lpg, rgd, reg=0):
    dv_g = get_dv(rgd)
    p11_lii = np.einsum("lpg,lqg,g->lpq", delta_lpg, delta_lpg, dv_g)
    p11_lii += reg * np.identity(p11_lii.shape[-1])
    return p11_lii


def get_p12_p21_matrix(delta_lpg, phi_jabg, rgd, llist_j, w_b):
    dv_g = get_dv(rgd)
    nl, Np, ng = delta_lpg.shape
    nj, na, nb, ng = phi_jabg.shape
    p12_pbja = np.zeros((Np, nb, nj, na))
    for j in range(nj):
        l = llist_j[j]
        p12_pbja[:, :, j, :] = np.einsum(
            "pg,abg,g->pba", delta_lpg[l], phi_jabg[j], dv_g
        )
    p12_pbja *= w_b[:, None, None]  # TODO check w_b setup is correct
    return p12_pbja, p12_pbja.transpose(2, 3, 0, 1)


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
    p22_ii = p22_jaja.reshape(nj * na, nj * na)
    p22_ii += reg * np.identity(nj * na)  # TODO make sure this adds to p22_jaja
    return p22_jaja


def construct_full_p_matrices(p11_lpp, p12_pbja, p21_japb, p22_ljaja, w_b, nbas_loc):
    Np = p11_lpp.shape[1]
    nb = p12_pbja.shape[1]
    N1 = Np * nb
    lp1 = p11_lpp.shape[0]
    p_l_ii = []
    for l in range(lp1):
        j0, j1 = nbas_loc[l], nbas_loc[l + 1]
        p11_pbpb = np.zeros((Np, nb, Np, nb))
        for b in range(nb):
            p11_pbpb[:, b, :, b] = p11_lpp[l] * w_b[b]
        p11_ii = p11_pbpb.reshape(N1, N1)
        p12cut_pbja = p12_pbja[:, :, j0:j1]
        p21cut_japb = p21_japb[j0:j1]
        p22_jaja = p22_ljaja[l]
        N2 = p22_jaja.shape[0] * p22_jaja.shape[1]
        N = N1 + N2
        pl_ii = np.zeros((N, N))
        pl_ii[:N1, :N1] = p11_ii
        pl_ii[:N1, N1:] = p12cut_pbja.reshape(N1, N2)
        pl_ii[N1:, :N1] = p21cut_japb.reshape(N2, N1)
        pl_ii[N1:, N1:] = p22_jaja.reshape(N2, N2)
        p_l_ii.append(pl_ii)
    return p_l_ii


def get_delta_lpg(betas, rcut, rgd, thr=6, lmax=4):
    R = rcut
    r_g = rgd.r_g
    dv_g = get_dv(rgd)
    betas = betas[betas * R * R > thr]
    fcut = (0.5 * np.pi / betas) ** 0.75 * gauss_and_derivs(betas, R)
    funcs = (0.5 * np.pi / betas[:, None]) ** 0.75 * np.exp(-betas[:, None] * r_g * r_g)
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
    delta_lpg = np.zeros((lmax + 1, betas.size, r_g.size))
    for l in range(lmax + 1):
        funcs_tmp = funcs * r_g**l * np.sqrt(dv_g)
        ovlp = np.einsum("ig,jg->ij", funcs_tmp, funcs_tmp)
        L = cholesky(ovlp, lower=True)
        basis = np.linalg.solve(L, funcs)
        delta_lpg[l] = basis * r_g**l
    # print('DELTA NORM', np.einsum('lpg,lqg,g->lpq', delta_lpg, delta_lpg, dv_g))
    return delta_lpg


def get_delta_lpk(delta_lpg, rgd):
    delta_lpk = np.zeros_like(delta_lpg)
    for l in range(delta_lpg.shape[0]):
        for p in range(delta_lpg.shape[1]):
            delta_lpk[l, p] = rgd.transform_single_fwd(
                np.ascontiguousarray(delta_lpg[l, p]), l
            )
    return delta_lpk


def get_fit(l=0, RMAX=1.0, rs=None):
    S0 = 3
    ovlp0_p = np.zeros((S0, S0))
    ovlp0_f = np.zeros((S0, S0))
    if rs is None:
        rs = np.linspace(0, RMAX, 100)
    for i in range(S0):
        for j in range(S0):
            ovlp0_p[i, j] = get_pf_integral(2 * (l + i + j), RMAX)
            ovlp0_f[i, j] = get_ff_integral(2 * (l + i + j), RMAX)
    ovlp0_ff_inv = np.linalg.solve(ovlp0_f, np.identity(S0))
    ovlp0_pf_inv = np.linalg.solve(ovlp0_p, np.identity(S0))
    funcs = np.stack([get_pfunc(2 * i + l, rs, RMAX) for i in range(S0)])
    ffuncs = np.stack([get_ffunc(2 * i + l, rs, RMAX) for i in range(S0)])
    return ovlp0_ff_inv, ovlp0_pf_inv, funcs, ffuncs, S0


def get_convs(alphas, beta, l, ortho_funcs, rgd):
    nfunc = ortho_funcs.shape[0]
    ngrid = ortho_funcs.shape[1]
    nalpha = alphas.size
    transforms = np.zeros((nfunc, ngrid))
    convolutions = np.zeros((nalpha, nfunc, ngrid))
    for i in range(nfunc):
        transforms[i] = rgd.transform_single_fwd(ortho_funcs[i], l)
        for a in range(nalpha):
            expnt = alphas[a] + beta
            g_k = gauss_fft(l, expnt, rgd.k_g)
            convolutions[a, i] = rgd.transform_single_bwd(transforms[i] * g_k, l)
    return convolutions


def get_gaussian_convs(alphas, beta, l, gammas, r_g):
    nfunc = gammas.size
    ngrid = r_g.size
    nalpha = alphas.size
    convolutions = np.zeros((nalpha, nfunc, ngrid))
    for i in range(nfunc):
        for a in range(nalpha):
            expnt = alphas[a] + beta
            coef, expnt = gauss_conv_coef_and_expnt(l, expnt, gammas[i])
            convolutions[a, i] = coef * np.exp(-expnt * r_g**2)
    return convolutions
