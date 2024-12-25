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
import time

import numpy as np
from pyscf import lib
from pyscf.dft.numint import _contract_rho, _dot_ao_dm, eval_ao
from pyscf.gto.mole import (
    ANG_OF,
    ATOM_OF,
    NCTR_OF,
    NPRIM_OF,
    PTR_COEFF,
    PTR_COORD,
    PTR_EXP,
)
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools
from pyscf.pbc.dft.numint import eval_ao_kpts
from pyscf.pbc.gto.eval_gto import get_lattice_Ls
from pyscf.pbc.tools.pbc import fft, ifft
from scipy.special import erf

from ciderpress.dft.plans import SADMPlan, SDMXFullPlan, SDMXPlan
from ciderpress.dft.settings import SADMSettings, SDMXFullSettings, SDMXSettings
from ciderpress.lib import load_library as load_cider_library
from ciderpress.pyscf import sdmx_slow as sdmx

libcider = load_cider_library("libmcider")


def build_ft_cell(cell):
    ft_cell = cell.copy()
    bas = cell._bas.copy()
    real_env = cell._env.copy()
    recip_env = cell._env.copy()
    coords = cell.atom_coords(unit="Bohr")
    vol = np.abs(np.linalg.det(cell.lattice_vectors()))
    fac = 4 * np.pi * np.prod(cell.mesh) / vol
    for ib in range(bas.shape[0]):
        l = bas[ib, ANG_OF]
        nctr = bas[ib, NCTR_OF]
        nprim = bas[ib, NPRIM_OF]
        eloc = bas[ib, PTR_EXP]
        cloc = bas[ib, PTR_COEFF]
        coeffs = real_env[cloc : cloc + nprim * nctr]
        coeffs = coeffs.reshape(nctr, nprim)
        exps = real_env[eloc : eloc + nprim]
        recip_env[eloc : eloc + nprim] = 0.25 / exps
        coeff_fac = 0.5 ** (2 + l) * exps ** (-1.5 - l) * np.sqrt(np.pi)
        recip_env[cloc : cloc + nprim * nctr] = fac * (coeffs * coeff_fac).ravel()
        aloc = cell._atm[bas[ib, ATOM_OF], PTR_COORD]
        recip_env[aloc : aloc + 3] = 0
    ft_cell._env = recip_env
    ft_cell._real_atom_coords = coords
    return ft_cell


def get_ao_recip(ftcell, Gkv, deriv=0, out=None):
    Gkv = np.asfortranarray(Gkv)
    if deriv == 0:
        ncomp = 1
    elif deriv == 1:
        ncomp = 4
    else:
        raise NotImplementedError
    ngrids = Gkv.shape[0]
    nao = ftcell.nao_nr()
    if out is None:
        out = np.ndarray(
            (ncomp, nao, ngrids), order="C", dtype=np.complex128
        ).transpose(0, 2, 1)
    aotmp = eval_ao(ftcell, Gkv, deriv=0)
    out[0] = aotmp
    if deriv == 1:
        out[1] = 1j * aotmp * Gkv[:, 0]
        out[2] = 1j * aotmp * Gkv[:, 1]
        out[3] = 1j * aotmp * Gkv[:, 2]
    nao = ftcell.nao_nr()
    ao_loc = ftcell.ao_loc_nr()
    atom_list = np.zeros(nao, dtype=np.int32)
    ang_list = np.zeros(nao, dtype=np.int32)
    for i, b in enumerate(ftcell._bas):
        atom_list[ao_loc[i] : ao_loc[i + 1]] = b[ATOM_OF]
        ang_list[ao_loc[i] : ao_loc[i + 1]] = b[ANG_OF]
    atom_coords = np.ascontiguousarray(ftcell._real_atom_coords)
    for v in range(ncomp):
        libcider.apply_orb_phases(
            out[v].ctypes.data_as(ctypes.c_void_p),
            atom_list.ctypes.data_as(ctypes.c_void_p),
            ang_list.ctypes.data_as(ctypes.c_void_p),
            Gkv.ctypes.data_as(ctypes.c_void_p),
            atom_coords.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(ftcell.natm),
            ctypes.c_int(nao),
            ctypes.c_int(Gkv.shape[0]),
        )
    return out


def _gauss_diff_with_cutoff(a, r, k2):
    k = np.sqrt(k2)
    arg = (1j * k + 2 * a * r) / (2 * np.sqrt(a))
    arg2 = (1j * k - 2 * a * r) / (2 * np.sqrt(a))
    if np.isnan(arg).any():
        raise RuntimeError("nan values in internal CiderPress routine")
    tmp = erf(arg).real - erf(arg2).real
    tmp = tmp * (k * np.sqrt(np.pi) * np.exp(arg * arg))
    g = 2j * np.sqrt(a) * (np.exp(2j * k * r) - 1)
    g[:] *= np.exp(-r * (1j * k + a * r)) / (8 * a**1.5 * k)
    g[k2 < 1e-8] = np.sqrt(np.pi) / (4 * a**1.5) * erf(np.sqrt(a) * r) - np.exp(
        -a * r * r
    ) * r / (2 * a)
    return g


def _get_rcut_for_cell(cell, kpt_dim):
    lattice = cell.lattice_vectors()
    lattice[0] *= kpt_dim[0]
    lattice[1] *= kpt_dim[1]
    lattice[2] *= kpt_dim[2]
    lengths = np.linalg.norm(lattice, axis=1)
    rmax_init = np.max(lengths) + 0.1
    Ls = get_lattice_Ls(cell, rcut=rmax_init)
    dists = np.linalg.norm(Ls, axis=1)
    return np.min(dists[dists > 1e-8])


def precompute_convolutions(expnt, coeff, cell, kpts, itype, kpt_dim=None):
    log = lib.logger.Logger(cell.stdout, cell.verbose)
    log.debug("# Precomputing convolution kernels for SDMX")
    if kpt_dim is None:
        Nk = tools.pbc.get_monkhorst_pack_size(cell, kpts)
    else:
        Nk = kpt_dim
    log.debug("# Nk = %s", Nk)

    kcell = pbcgto.Cell()
    kcell.atom = "H 0. 0. 0."
    kcell.spin = 1
    kcell.unit = "B"
    kcell.verbose = 0
    kcell.a = cell.lattice_vectors() * Nk
    Lc = 1.0 / lib.norm(np.linalg.inv(kcell.a), axis=0)
    log.debug("# Lc = %s", Lc)
    Rin = Lc.min() / 2.0
    alpha_cut = 5.0 / Rin  # any alpha larger than this need not be precomp'd
    alpha = np.sqrt(expnt)
    if alpha > alpha_cut:
        return None
    max(alpha, 0.5 * alpha_cut)
    log.debug("# Rin = %s", Rin)
    log.debug("WS alpha = %s", alpha)
    # kcell.mesh = np.array([4 * int(L * alpha * 3.0) for L in Lc])
    # kcell.mesh = np.array([2 * int(L * alpha_cut * 3.0) for L in Lc])
    kcell.mesh = np.array([2 * int(L * alpha_cut * 3.0) for L in Lc])
    log.debug("# kcell.mesh FFT = %s", kcell.mesh)
    rs = kcell.get_uniform_grids(wrap_around=False)
    kngs = len(rs)
    log.debug("# kcell kngs = %d", kngs)
    corners_coord = lib.cartesian_prod(([0, 1], [0, 1], [0, 1]))
    corners = np.dot(corners_coord, kcell.a)
    r = np.min([lib.norm(rs - c, axis=1) for c in corners], axis=0)
    if itype == "gauss_diff":
        vR = np.exp(-expnt * r * r)
        vR = coeff * vR * (1 - vR)
    elif itype == "gauss_r2":
        tmp = r * r
        vR = coeff * tmp * np.exp(-expnt * tmp)
    else:
        raise ValueError
    vG = (kcell.vol / kngs) * fft(vR, kcell.mesh)
    if abs(vG.imag).max() > 1e-6:
        raise RuntimeError("Unconventional lattice was found")
    ws_conv = {"alpha": alpha, "kcell": kcell, "q": kcell.Gv, "vq": vG.real.copy()}
    log.debug("# Finished precomputing")
    return ws_conv


def precompute_all_convolutions(cell, kpts, alphas, alpha_norms, itype, kpt_dim=None):
    ws_conv_list = []
    for ialpha in range(len(alphas)):
        ws_conv_list.append(
            precompute_convolutions(
                alphas[ialpha], alpha_norms[ialpha], cell, kpts, itype, kpt_dim=kpt_dim
            )
        )
    return ws_conv_list


def get_ao_and_aok(cell, coords, kpt, deriv):
    ao = eval_ao_kpts(cell, coords, kpt, deriv=deriv)[0]
    expir = np.exp(-1j * np.dot(coords, kpt))
    aokG = np.asarray(tools.fftk(np.asarray(ao.T, order="C"), cell.mesh, expir))
    return ao, aokG


def get_ao_and_aok_fast(cell, coords, kpt, deriv):
    recip_cell = build_ft_cell(cell)
    aokG = get_ao_recip(recip_cell, cell.Gv + kpt, deriv=0)[0].T
    ao = fft_fast(aokG, cell.mesh, fwd=False, inplace=False)
    expir = np.exp(1j * np.dot(coords, kpt))
    ao *= expir
    return ao.T, aokG


def get_recip_convolutions(
    cell,
    Gk2,
    alphas,
    alpha_norms,
    itype,
    cutoff_type,
    cutoff_info=None,
    kG=None,
    out=None,
):
    nrecip = Gk2.size  # np.prod(cell.mesh)
    nalpha = len(alphas)
    conv_aG = np.ndarray((nalpha, nrecip), order="C", buffer=out)
    if cutoff_info is None:
        cutoff_info_list = [None] * nalpha
    else:
        cutoff_info_list = cutoff_info
    for ialpha in range(nalpha):
        cutoff_info = cutoff_info_list[ialpha]
        alpha = alphas[ialpha]
        anorm = alpha_norms[ialpha]
        conv_kG = conv_aG[ialpha]
        if itype == "gauss_r2":
            raise NotImplementedError
        elif itype == "gauss_diff":
            if cutoff_type is None or (cutoff_type == "ws" and cutoff_info is None):
                assert conv_kG.flags.c_contiguous
                assert Gk2.flags.c_contiguous
                libcider.recip_conv_kernel_gaussdiff(
                    conv_kG.ctypes.data_as(ctypes.c_void_p),
                    Gk2.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_double(alpha),
                    ctypes.c_double(anorm),
                    ctypes.c_int(Gk2.size),
                )
                """
                expnt = 0.125 / alpha
                conv_kG[:] = np.exp(-expnt * Gk2)
                fac = np.sqrt(2.0) / 16
                fac2 = 4 * np.pi * anorm * np.sqrt(np.pi) / alpha**1.5
                # conv_kG[:] = fac2 * conv_kG * (0.25 * conv_kG - fac)
                conv_kG[:] *= (0.25 * conv_kG - fac)
                conv_kG[:] *= fac2
                """
            elif cutoff_type == "sphere":
                assert cutoff_info is not None
                rcut = _get_rcut_for_cell(cell, cutoff_info)
                conv_kG[:] = _gauss_diff_with_cutoff(alpha, rcut, Gk2)
            elif cutoff_type == "ws":
                assert kG is not None
                kcell = cutoff_info["kcell"]
                q = cutoff_info["q"]
                vq = cutoff_info["vq"]
                """
                conv_kG[:] = 0.0
                gxyz = np.dot(kG, kcell.lattice_vectors().T) / (2 * np.pi)
                gxyz = gxyz.round(decimals=6).astype(int)
                mesh = np.asarray(kcell.mesh)
                gxyz = (gxyz + mesh) % mesh
                qidx = (gxyz[:, 0]*mesh[1] + gxyz[:, 1]) * mesh[2] + gxyz[:, 2]
                maxqv = abs(q).max(axis=0)
                is_lt_maxqv = (abs(kG) <= maxqv).all(axis=1)
                conv_kG[is_lt_maxqv] += vq[qidx[is_lt_maxqv]]
                """
                assert conv_kG.flags.c_contiguous
                if not kG.flags.c_contiguous:
                    kG = np.ascontiguousarray(kG)
                assert kG.ndim == 2
                assert kG.dtype == np.float64
                assert kG.shape[1] == 3
                assert vq.flags.c_contiguous
                maxqv = np.ascontiguousarray(abs(q).max(axis=0).astype(np.float64))
                lattice = np.asarray(
                    kcell.lattice_vectors(), dtype=np.float64, order="C"
                )
                mesh = np.asarray(kcell.mesh, dtype=np.int32, order="C")
                libcider.recip_conv_kernel_ws(
                    conv_kG.ctypes.data_as(ctypes.c_void_p),
                    vq.ctypes.data_as(ctypes.c_void_p),
                    kG.ctypes.data_as(ctypes.c_void_p),
                    lattice.ctypes.data_as(ctypes.c_void_p),
                    maxqv.ctypes.data_as(ctypes.c_void_p),
                    mesh.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(kG.shape[0]),
                    ctypes.c_int(vq.size),
                )
            else:
                raise ValueError
        else:
            raise ValueError
    return conv_aG


def convolve_aos(
    cell, aokG, Gk2, alpha, anorm, itype, cutoff_type, cutoff_info=None, kG=None
):
    conv_kG = get_recip_convolutions(
        cell, Gk2, [alpha], [anorm], itype, cutoff_type, [cutoff_info], kG
    )[0]
    conv_aokG = np.ascontiguousarray(aokG * conv_kG)
    conv_aokR = fft_fast(conv_aokG, cell.mesh, fwd=False, inplace=True)
    return conv_aokR


def fft_grad(f, mesh, Gv):
    igk = _get_igk(Gv, np.complex128, mesh)
    fk = fft(f, mesh)
    # _zero_even_edges_fft(fk, mesh)
    return np.stack(
        [
            ifft(fk * igk[:, 0], mesh),
            ifft(fk * igk[:, 1], mesh),
            ifft(fk * igk[:, 2], mesh),
        ]
    )


def fft_grad_bwd(fx, mesh, Gv):
    igk = _get_igk(Gv, np.complex128, mesh)
    fk = igk[:, 0] * ifft(fx[0], mesh)
    fk += igk[:, 1] * ifft(fx[1], mesh)
    fk += igk[:, 2] * ifft(fx[2], mesh)
    # _zero_even_edges_fft(fk, mesh)
    return fft(fk, mesh)


def fft_grad_fast(f0, mesh, Gv):
    assert f0.flags.c_contiguous
    assert f0.ndim == 2
    assert f0.shape[1] == Gv.shape[0]
    assert Gv.flags.f_contiguous
    assert Gv.ndim == 2
    assert Gv.shape[1] == 3
    assert f0.dtype == np.complex128
    f1 = np.empty((3, f0.shape[0], f0.shape[1]), order="C", dtype=np.complex128)
    libcider.run_ffts(
        f0.ctypes.data_as(ctypes.c_void_p),
        f1[0].ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(1.0),
        (ctypes.c_int * 3)(*mesh),
        ctypes.c_int(1),
        ctypes.c_int(f0.shape[0]),
        ctypes.c_int(1),
        ctypes.c_int(0),
    )
    f1[1] = 1j * f1[0] * Gv[:, 1]
    f1[2] = 1j * f1[0] * Gv[:, 2]
    f1[0] *= 1j * Gv[:, 0]
    scale = 1.0 / np.prod(mesh)
    libcider.run_ffts(
        f1.ctypes.data_as(ctypes.c_void_p),
        None,
        ctypes.c_double(scale),
        (ctypes.c_int * 3)(*mesh),
        ctypes.c_int(0),
        ctypes.c_int(3 * f0.shape[0]),
        ctypes.c_int(1),
        ctypes.c_int(0),
    )
    return f1


def fft_fast(f_in, mesh, fwd=True, inplace=False, scale=None, f_out=None):
    assert f_in.flags.c_contiguous
    assert f_in.dtype == np.complex128
    if scale is None:
        if fwd:
            scale = 1.0
        else:
            scale = 1.0 / np.prod(mesh).astype(np.float64)
    if f_in.shape[-1] == mesh[0] * mesh[1] * (mesh[2] // 2 + 1):
        r2c = 1
    elif f_in.shape[-1] == np.prod(mesh):
        r2c = 0
    else:
        raise ValueError
    if not inplace:
        if f_out is None:
            f_out = np.empty_like(f_in, order="C")
        _f_out = f_out.ctypes.data_as(ctypes.c_void_p)
    else:
        f_out = None
        _f_out = None
    fwd = 1 if fwd else 0
    libcider.run_ffts(
        f_in.ctypes.data_as(ctypes.c_void_p),
        _f_out,
        (ctypes.c_int * 3)(*mesh),
        ctypes.c_double(scale),
        ctypes.c_int(fwd),
        ctypes.c_int(f_in.shape[0]),
        ctypes.c_int(1),
        ctypes.c_int(r2c),
    )
    if not inplace:
        return f_out
    else:
        return f_in


def compute_sdmx_tensor(
    dms, cell, coords, alphas, alpha_norms, itype, kpts, cutoff_type=None, has_l1=False
):
    t0s = time.monotonic()
    Gv = np.asfortranarray(cell.Gv)
    shls_slice = (0, cell.nbas)
    ao_loc = cell.ao_loc_nr()
    nalpha = len(alphas)
    ng = Gv.shape[0]
    ncpa = 4 if has_l1 else 1
    tmp = np.zeros((ncpa, nalpha, ng), dtype=np.complex128)
    if cutoff_type == "ws":
        precomp_list = precompute_all_convolutions(
            cell, kpts, alphas, alpha_norms, itype
        )
    else:
        precomp_list = [None] * len(kpts)
    t0e = time.monotonic()
    t1 = 0
    t2 = 0
    for kn, kpt in enumerate(kpts):
        t1s = time.monotonic()
        # ao, aokG = get_ao_and_aok(cell, coords, kpt, 0)
        ao, aokG = get_ao_and_aok_fast(cell, coords, kpt, 0)
        ao = ao * np.exp(-1j * np.dot(coords, kpt))[:, None]
        # After above op, `ao` var contains conj(ao) e^ikr, which is periodic
        # because conj(ao) = (periodic part) * e^-ikr
        c0 = _dot_ao_dm(cell, ao, dms[kn], None, shls_slice, ao_loc)
        if has_l1:
            c1 = fft_grad_fast(c0.T, cell.mesh, Gv).transpose(0, 2, 1)
            c1[0] += 1j * kpt[0] * c0
            c1[1] += 1j * kpt[1] * c0
            c1[2] += 1j * kpt[2] * c0
        Gk = Gv + kpt
        Gk2 = lib.einsum("gv,gv->g", Gk, Gk)
        t1e = time.monotonic()
        t2s = time.monotonic()
        for ialpha in range(nalpha):
            conv_aokR = convolve_aos(
                cell,
                aokG,
                Gk2,
                alphas[ialpha],
                alpha_norms[ialpha],
                itype,
                cutoff_type,
                cutoff_info=precomp_list[ialpha],
                kG=Gk,
            )
            tmp[0, ialpha] += _contract_rho(conv_aokR.T, c0)
            if has_l1:
                tmp[1, ialpha] -= _contract_rho(conv_aokR.T, c1[0])
                tmp[2, ialpha] -= _contract_rho(conv_aokR.T, c1[1])
                tmp[3, ialpha] -= _contract_rho(conv_aokR.T, c1[2])
        t2e = time.monotonic()
        t1 += t1e - t1s
        t2 += t2e - t2s
    t3s = time.monotonic()
    tmp[:] /= len(kpts)
    if has_l1:
        tmp1 = fft_grad(tmp[0], cell.mesh, Gv)
        tmp[1] += tmp1[0]
        tmp[2] += tmp1[1]
        tmp[3] += tmp1[2]
    t3e = time.monotonic()
    print("SDMX TIMES", t0e - t0s, t1, t2, t3e - t3s)
    return tmp


def _get_grid_info(cell, kpts, dms):
    if len(kpts) == 1 and np.linalg.norm(kpts) < 1e-16 and dms[0].dtype == np.float64:
        rtype = np.float64
        nz = cell.mesh[2] // 2 + 1
        assert cell.Gv.ndim == 2
        assert cell.Gv.shape[1] == 3
        Gv = np.ascontiguousarray(
            cell.Gv.reshape(cell.mesh[0], cell.mesh[1], cell.mesh[2], 3)[
                ..., :nz, :
            ].reshape(-1, 3)
        )
        ng_recip = Gv.shape[0]
        ng_real = ng_recip * 2
        mesh = cell.mesh
        rx = np.fft.fftfreq(mesh[0], 1.0 / mesh[0])
        ry = np.fft.fftfreq(mesh[1], 1.0 / mesh[1])
        # rz = np.fft.fftfreq(mesh[2], 1./mesh[2])
        rz = np.arange(nz)
        b = cell.reciprocal_vectors()
        Gvbase = (rx, ry, rz)
        Gv = np.dot(lib.cartesian_prod(Gvbase), b)
    else:
        rtype = np.complex128
        Gv = cell.Gv
        ng_recip = Gv.shape[0]
        ng_real = ng_recip
    return rtype, Gv, ng_recip, ng_real


def _zero_even_edges_fft(x, mesh):
    # double complex *x, const int num_fft, const int *fftg, const int halfc
    assert x.flags.c_contiguous
    assert x.dtype == np.complex128
    size = x.shape[-1]
    nfft = x.size // size
    assert nfft * size == x.size
    if size == np.prod(mesh):
        halfc = 0
    elif size == mesh[0] * mesh[1] * (mesh[2] // 2 + 1):
        halfc = 1
    else:
        raise ValueError
    libcider.zero_even_edges_fft(
        x.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nfft),
        (ctypes.c_int * 3)(*mesh),
        ctypes.c_int(halfc),
    )


def _get_igk(Gk, rtype, mesh, zero_even_edge=None):
    # default zero_even_edge is True for rtype=np.float64,
    # False for rtype=np.complex128
    igk = np.asfortranarray(Gk * 1j)
    if rtype == np.float64:
        igk.shape = (mesh[0], mesh[1], mesh[2] // 2 + 1, 3)
        if zero_even_edge is None:
            zero_even_edge = True
    else:
        igk.shape = (mesh[0], mesh[1], mesh[2], 3)
        if zero_even_edge is None:
            zero_even_edge = False
    if zero_even_edge:
        if mesh[0] % 2 == 0:
            mid = mesh[0] // 2
            igk[mid, :, :, 0] = 0.0
        if mesh[1] % 2 == 0:
            mid = mesh[1] // 2
            igk[:, mid, :, 1] = 0.0
        if mesh[2] % 2 == 0:
            mid = mesh[2] // 2
            igk[:, :, mid, 2] = 0.0
    igk.shape = (-1, 3)
    return igk


def compute_sdmx_tensor_lowmem(
    dms,
    cell,
    coords,
    alphas,
    alpha_norms,
    itype,
    kpts,
    cutoff_type=None,
    has_l1=False,
    nao_per_block=None,
    kpt_dim=None,
    precomp_list=None,
    spinpol=False,
):
    t0s = time.monotonic()
    nao = cell.nao_nr()
    nalpha = len(alphas)
    ncpa = 4 if has_l1 else 1
    if nao_per_block is None:
        nao_per_block = 8  # max(nao // (2 * ncpa), 4)
    if precomp_list is not None:
        pass
    elif cutoff_type == "ws":
        precomp_list = precompute_all_convolutions(
            cell, kpts, alphas, alpha_norms, itype, kpt_dim=kpt_dim
        )
    else:
        precomp_list = [None] * len(kpts)
    t0e = time.monotonic()
    t1 = 0
    t2 = 0

    rtype, Gv, ng_recip, ng_real = _get_grid_info(cell, kpts, dms)
    nalpha = len(alphas)
    conv_aG = np.ndarray((nalpha, ng_recip), order="C")
    recip_cell = build_ft_cell(cell)
    aobuf = np.ndarray((1, nao, ng_recip), dtype=np.complex128).transpose(0, 2, 1)
    if spinpol:
        nspin = len(dms)
    else:
        nspin = 1
        dms = (dms,)
    tmp = np.zeros((nspin, ncpa, nalpha, ng_real), dtype=rtype)
    c_all = np.ndarray((nspin, nao, ng_recip), dtype=np.complex128)
    bufs = np.ndarray((nspin * ncpa + 1, nao_per_block * ng_recip), dtype=np.complex128)
    for kn, kpt in enumerate(kpts):
        t1s = time.monotonic()
        Gk = Gv + kpt
        Gk2 = lib.einsum("gv,gv->g", Gk, Gk)
        aoG = get_ao_recip(recip_cell, Gk, deriv=0, out=aobuf)[0].T
        # is_gamma = np.linalg.norm(kpt) < 1e-10
        # if is_gamma:
        #     _zero_even_edges_fft(aoG, cell.mesh)
        for s in range(nspin):
            lib.dot(dms[s][kn].T, aoG, c=c_all[s])
        assert aoG.flags.c_contiguous
        assert aoG.shape == (nao, ng_recip)
        get_recip_convolutions(
            cell,
            Gk2,
            alphas,
            alpha_norms,
            itype,
            cutoff_type=cutoff_type,
            cutoff_info=precomp_list,
            kG=Gk,
            out=conv_aG,
        )
        igk = _get_igk(Gk, rtype, cell.mesh)
        t1e = time.monotonic()
        t1 += t1e - t1s
        t2s = time.monotonic()
        for i0, i1 in lib.prange(0, nao, nao_per_block):
            ck = np.ndarray(
                (nspin, ncpa, i1 - i0, ng_recip),
                dtype=np.complex128,
                buffer=bufs[: nspin * ncpa],
            )
            cr = np.ndarray(
                (nspin, ncpa, i1 - i0, ng_real),
                dtype=rtype,
                buffer=bufs[: nspin * ncpa],
            )
            ck[:, 0] = c_all[:, i0:i1]
            if has_l1:
                for s in range(nspin):
                    _mul_z(ck[s, 0], igk[:, 0], ck[s, 1])
                    _mul_z(ck[s, 0], igk[:, 1], ck[s, 2])
                    _mul_z(ck[s, 0], igk[:, 2], ck[s, 3])
            shape = ck.shape
            ck.shape = (nspin * ncpa * (i1 - i0), ng_recip)
            fft_fast(ck, cell.mesh, fwd=False, inplace=True)
            ck.shape = shape
            conv_aog = np.ndarray(
                (i1 - i0, ng_recip), dtype=np.complex128, buffer=bufs[nspin * ncpa]
            )
            conv_aor = np.ndarray(
                (i1 - i0, ng_real), dtype=rtype, buffer=bufs[nspin * ncpa]
            )
            for ialpha in range(nalpha):
                _mul_dz(aoG[i0:i1], conv_aG[ialpha], conv_aog)
                fft_fast(conv_aog, cell.mesh, fwd=False, inplace=True)
                for s in range(nspin):
                    tmp[s, 0, ialpha] += _contract_rho(conv_aor.T, cr[s, 0].T)
                    if has_l1:
                        tmp[s, 1, ialpha] -= _contract_rho(conv_aor.T, cr[s, 1].T)
                        tmp[s, 2, ialpha] -= _contract_rho(conv_aor.T, cr[s, 2].T)
                        tmp[s, 3, ialpha] -= _contract_rho(conv_aor.T, cr[s, 3].T)
        t2e = time.monotonic()
        t2 += t2e - t2s
    t3s = time.monotonic()
    tmp[:] /= len(kpts)
    if rtype == np.float64:
        tmp = np.ascontiguousarray(
            tmp.reshape(
                nspin,
                ncpa,
                nalpha,
                cell.mesh[0],
                cell.mesh[1],
                2 * (cell.mesh[2] // 2 + 1),
            )[..., : cell.mesh[2]].reshape(nspin, ncpa, nalpha, -1)
        )
    if has_l1:
        for s in range(nspin):
            tmp1 = fft_grad(tmp[s, 0], cell.mesh, cell.Gv)
            if rtype == np.float64:
                tmp1 = tmp1.real
            tmp[s, 1] += tmp1[0]
            tmp[s, 2] += tmp1[1]
            tmp[s, 3] += tmp1[2]
    if not spinpol:
        tmp = tmp[0]
    t3e = time.monotonic()
    print("SDMX TIMES", t0e - t0s, t1, t2, t3e - t3s)
    return tmp


def _contract_convolution(tmp, conv_ao, cr, ialpha):
    rtype = tmp.dtype
    if rtype == np.float64:
        fn = libcider.contract_convolution_d
    elif rtype == np.complex128:
        fn = libcider.contract_convolution_z
    else:
        raise ValueError
    assert conv_ao.dtype == rtype
    assert cr.dtype == rtype
    fn(
        tmp[:, ialpha].ctypes.data_as(ctypes.c_void_p),
        conv_ao.ctypes.data_as(ctypes.c_void_p),
        cr.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(tmp.shape[0]),
        ctypes.c_int(conv_ao.shape[0]),
        ctypes.c_int(conv_ao.shape[1]),
        ctypes.c_int(tmp.shape[1]),
    )


def compute_sdmx_tensor_mo(
    mo_coeff,
    mo_occ,
    cell,
    coords,
    alphas,
    alpha_norms,
    itype,
    kpts,
    cutoff_type=None,
    has_l1=False,
    nmo_per_block=None,
    dms=None,
    kpt_dim=None,
    precomp_list=None,
    spinpol=False,
):
    if nmo_per_block is None:
        nmo_per_block = 32
    t0s = time.monotonic()
    nao = cell.nao_nr()
    nalpha = len(alphas)
    ncpa = 4 if has_l1 else 1
    if spinpol:
        rtype, Gv, ng_recip, ng_real = _get_grid_info(cell, kpts, mo_coeff[0])
    else:
        rtype, Gv, ng_recip, ng_real = _get_grid_info(cell, kpts, mo_coeff)
    if precomp_list is not None:
        pass
    elif cutoff_type == "ws":
        precomp_list = precompute_all_convolutions(
            cell, kpts, alphas, alpha_norms, itype, kpt_dim=kpt_dim
        )
    else:
        precomp_list = [None] * len(kpts)
    t0e = time.monotonic()
    t1 = 0
    t2 = 0
    nalpha = len(alphas)
    recip_cell = build_ft_cell(cell)
    conv_aG = np.ndarray((nalpha, ng_recip), order="C")
    aobuf = np.ndarray((1, nao, ng_recip), dtype=np.complex128).transpose(0, 2, 1)
    if spinpol:
        nspin = len(mo_occ)
    else:
        nspin = 1
        mo_coeff = (mo_coeff,)
        mo_occ = (mo_occ,)
    nmo_max = 0
    cmo_lists = []
    for s in range(nspin):
        _cmo_list = []
        for kn in range(len(kpts)):
            occs = mo_occ[s][kn]
            occ_screen = occs > 1e-10
            cmo = mo_coeff[s][kn][:, occ_screen] * np.sqrt(occs[occ_screen])
            _cmo_list.append(np.ascontiguousarray(cmo.T))
            nmo_max = max(nmo_max, _cmo_list[-1].shape[0])
        cmo_lists.append(_cmo_list)
    tmp = np.zeros((nspin, ncpa, nalpha, ng_real), dtype=rtype)
    bufs = np.ndarray((ncpa + 1, nmo_per_block * ng_recip), dtype=np.complex128)
    ck_all_buf = np.ndarray(nmo_max * ng_recip, dtype=np.complex128)
    for kn, kpt in enumerate(kpts):
        Gk = Gv + kpt
        Gk2 = lib.einsum("gv,gv->g", Gk, Gk)
        aoG = get_ao_recip(recip_cell, Gk, deriv=0, out=aobuf)[0].T
        # is_gamma = np.linalg.norm(kpt) < 1e-10
        # if is_gamma:
        #     _zero_even_edges_fft(aoG, cell.mesh)
        assert aoG.flags.c_contiguous
        assert aoG.shape == (nao, ng_recip)
        t1s = time.monotonic()
        get_recip_convolutions(
            cell,
            Gk2,
            alphas,
            alpha_norms,
            itype,
            cutoff_type=cutoff_type,
            cutoff_info=precomp_list,
            kG=Gk,
            out=conv_aG,
        )
        igk = _get_igk(Gk, rtype, cell.mesh)
        t1e = time.monotonic()
        t1 += t1e - t1s
        t2s = time.monotonic()
        for s in range(nspin):
            nmo = cmo_lists[s][kn].shape[0]
            ck_all = np.ndarray((nmo, ng_recip), dtype=np.complex128, buffer=ck_all_buf)
            lib.dot(cmo_lists[s][kn], aoG, c=ck_all)
            for i0, i1 in lib.prange(0, nmo, nmo_per_block):
                ck = np.ndarray(
                    (ncpa, i1 - i0, ng_recip), dtype=np.complex128, buffer=bufs[:ncpa]
                )
                cr = np.ndarray(
                    (ncpa, i1 - i0, ng_real), dtype=rtype, buffer=bufs[:ncpa]
                )
                ck[0] = ck_all[i0:i1]
                if has_l1:
                    _mul_z(ck[0], igk[:, 0], ck[1])
                    _mul_z(ck[0], igk[:, 1], ck[2])
                    _mul_z(ck[0], igk[:, 2], ck[3])
                shape = ck.shape
                ck.shape = (ncpa * (i1 - i0), ng_recip)
                fft_fast(ck, cell.mesh, fwd=False, inplace=True)
                ck.shape = shape
                conv_aog = np.ndarray(
                    (i1 - i0, ng_recip), dtype=np.complex128, buffer=bufs[ncpa]
                )
                conv_aor = np.ndarray(
                    (i1 - i0, ng_real), dtype=rtype, buffer=bufs[ncpa]
                )
                tc, td = 0, 0
                for ialpha in range(nalpha):
                    # conv_ao[:] = ck * conv_aG[ialpha]
                    _mul_dz(ck_all[i0:i1], conv_aG[ialpha], conv_aog)
                    tc -= time.monotonic()
                    fft_fast(conv_aog, cell.mesh, fwd=False, inplace=True)
                    tc += time.monotonic()
                    td -= time.monotonic()
                    _contract_convolution(tmp[s], conv_aor, cr, ialpha)
                    td += time.monotonic()
            t2e = time.monotonic()
        t2 += t2e - t2s
    t3s = time.monotonic()
    tmp[:] /= len(kpts)
    tmp[:, 1:] *= -1
    if rtype == np.float64:
        tmp = np.ascontiguousarray(
            tmp.reshape(
                nspin,
                ncpa,
                nalpha,
                cell.mesh[0],
                cell.mesh[1],
                2 * (cell.mesh[2] // 2 + 1),
            )[..., : cell.mesh[2]].reshape(nspin, ncpa, nalpha, -1)
        )
    if has_l1:
        for s in range(nspin):
            tmp1 = fft_grad(tmp[s, 0], cell.mesh, cell.Gv)
            if rtype == np.float64:
                tmp1 = tmp1.real
            tmp[s, 1] += tmp1[0]
            tmp[s, 2] += tmp1[1]
            tmp[s, 3] += tmp1[2]
    if not spinpol:
        tmp = tmp[0]
    t3e = time.monotonic()
    print("SDMX TIMES", t0e - t0s, t1, t2, t3e - t3s)
    return tmp


def _check_shapes(
    a, b, c, atype=np.complex128, btype=np.complex128, ctype=np.complex128
):
    assert a.shape == c.shape
    assert a.ndim == 2
    assert b.ndim == 1
    assert a.shape[1] == b.shape[0]
    assert a.flags.c_contiguous
    assert b.flags.c_contiguous
    assert c.flags.c_contiguous
    assert a.dtype == atype
    assert b.dtype == btype
    assert c.dtype == ctype


def _mul_add_d(a, b, c):
    dtype = np.float64
    _check_shapes(a, b, c, atype=dtype, btype=dtype, ctype=dtype)
    libcider.parallel_mul_add_d(
        a.ctypes.data_as(ctypes.c_void_p),
        b.ctypes.data_as(ctypes.c_void_p),
        c.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(a.shape[0]),
        ctypes.c_int(a.shape[1]),
    )


def _mul_add_z(a, b, c):
    _check_shapes(a, b, c)
    libcider.parallel_mul_add_z(
        a.ctypes.data_as(ctypes.c_void_p),
        b.ctypes.data_as(ctypes.c_void_p),
        c.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(a.shape[0]),
        ctypes.c_int(a.shape[1]),
    )


def _mul_z(a, b, c):
    _check_shapes(a, b, c)
    libcider.parallel_mul_z(
        a.ctypes.data_as(ctypes.c_void_p),
        b.ctypes.data_as(ctypes.c_void_p),
        c.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(a.shape[0]),
        ctypes.c_int(a.shape[1]),
    )


def _mul_dz(a, b, c):
    _check_shapes(a, b, c, btype=np.float64)
    libcider.parallel_mul_dz(
        a.ctypes.data_as(ctypes.c_void_p),
        b.ctypes.data_as(ctypes.c_void_p),
        c.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(a.shape[0]),
        ctypes.c_int(a.shape[1]),
    )


def _fast_conj(a):
    assert a.flags.c_contiguous
    libcider.fast_conj(
        a.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(a.size),
    )


def _weight_symm_gpts(x, mesh):
    assert x.ndim == 2
    stride = mesh[2] // 2 + 1
    assert x.shape[1] == mesh[0] * mesh[1] * stride
    assert x.flags.c_contiguous
    assert x.dtype == np.complex128
    dim1 = x.shape[0] * mesh[0] * mesh[1]
    dim2 = mesh[2]
    libcider.weight_symm_gpts(
        x.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(dim1),
        ctypes.c_int(dim2),
    )


def compute_sdmx_tensor_vxc(
    vmats,
    cell,
    vgrid,
    alphas,
    alpha_norms,
    itype,
    kpts,
    cutoff_type=None,
    has_l1=False,
    nao_per_block=None,
    kpt_dim=None,
    precomp_list=None,
    spinpol=False,
):
    # NOTE: vgrid should have weight applied already
    if spinpol:
        nspin = len(vmats)
        vmat_list = vmats
    else:
        vmat_list = (vmats,)
        vgrid = (vgrid,)
        nspin = 1
    ncpa = 4 if has_l1 else 1
    nalpha = len(alphas)
    rtype, Gv, ng_recip, ng_real = _get_grid_info(cell, kpts, vmats)
    vgridc_list = []
    vgrid_list = vgrid
    for s in range(nspin):
        vgridc = np.empty((ncpa, nalpha, ng_real), dtype=rtype, order="C")
        if rtype == np.float64:
            vgrid = vgrid_list[s].real
            # above should have no effect since complex part
            # is zero for gamma-point
            assert vgridc.dtype == np.float64
            # assert vgrid.dtype == np.float64
            vgridc.shape = (
                ncpa,
                nalpha,
                cell.mesh[0],
                cell.mesh[1],
                2 * (cell.mesh[2] // 2 + 1),
            )
            vgrid.shape = (ncpa, nalpha, cell.mesh[0], cell.mesh[1], cell.mesh[2])
            nz = cell.mesh[2]
            vgridc[..., :nz] = vgrid
            vgridc[..., nz:] = 0.0
            if has_l1:
                vgridc[0, ..., :nz] += fft_grad_bwd(
                    vgrid[1:4].reshape(3, nalpha, -1), cell.mesh, cell.Gv
                ).real.reshape(*vgrid.shape[1:])
                vgridc[1:] *= -1
            vgridc.shape = (ncpa, nalpha, ng_real)
        else:
            vgridc[:] = vgrid_list[s]
            if has_l1:
                vgridc[0] += fft_grad_bwd(vgridc[1:4], cell.mesh, Gv)
                vgridc[1:] *= -1
        assert vgridc.ndim == 3
        vgridc_list.append(vgridc)

    nao = cell.nao_nr()
    if nao_per_block is None:
        # nao_per_block = max(nao // (2 * ncpa), 4)
        nao_per_block = 8
    if precomp_list is not None:
        pass
    elif cutoff_type == "ws":
        precomp_list = precompute_all_convolutions(
            cell, kpts, alphas, alpha_norms, itype, kpt_dim=kpt_dim
        )
    else:
        precomp_list = [None] * len(alphas)

    conv_aG = np.ndarray((nalpha, ng_recip), order="C")
    recip_cell = build_ft_cell(cell)
    aobuf = np.ndarray((1, nao, ng_recip), dtype=np.complex128).transpose(0, 2, 1)
    bufs = np.ndarray((nspin * ncpa + 1, nao_per_block * ng_recip), dtype=np.complex128)
    dt0, dt1, dt2, dt3 = 0, 0, 0, 0
    dt4 = 0
    dt5 = 0
    for kn, kpt in enumerate(kpts):
        t0 = time.monotonic()
        Gk = Gv + kpt
        Gk2 = lib.einsum("gv,gv->g", Gk, Gk)
        aoG = get_ao_recip(recip_cell, Gk, deriv=0, out=aobuf)[0].T
        # is_gamma = np.linalg.norm(kpt) < 1e-10
        # if is_gamma:
        #     _zero_even_edges_fft(aoG, cell.mesh)
        assert aoG.flags.c_contiguous
        assert aoG.shape == (nao, ng_recip)
        get_recip_convolutions(
            cell,
            Gk2,
            alphas,
            alpha_norms,
            itype,
            cutoff_type=cutoff_type,
            cutoff_info=precomp_list,
            kG=Gk,
            out=conv_aG,
        )
        igk = _get_igk(Gk, rtype, cell.mesh)
        # igk = np.asfortranarray(Gk * 1j)
        t1 = time.monotonic()
        dt4 += t1 - t0
        for i0, i1 in lib.prange(0, nao, nao_per_block):
            t0 = time.monotonic()
            wvr = np.ndarray(
                (nspin, ncpa, i1 - i0, ng_real),
                dtype=rtype,
                buffer=bufs[: nspin * ncpa],
            )
            wvk = np.ndarray(
                (nspin, ncpa, i1 - i0, ng_recip),
                dtype=np.complex128,
                buffer=bufs[: nspin * ncpa],
            )
            wvr[:] = 0.0
            conv_aog = np.ndarray(
                (i1 - i0, ng_recip), dtype=np.complex128, buffer=bufs[nspin * ncpa]
            )
            conv_aor = np.ndarray(
                (i1 - i0, ng_real), dtype=rtype, buffer=bufs[nspin * ncpa]
            )
            t1 = time.monotonic()
            for ialpha in range(nalpha):
                _mul_dz(aoG[i0:i1], conv_aG[ialpha], conv_aog)
                fft_fast(conv_aog, cell.mesh, fwd=False, inplace=True)
                for s in range(nspin):
                    vgridc = vgridc_list[s]
                    if rtype == np.complex128:
                        _mul_add_z(conv_aor, vgridc[0, ialpha], wvr[s, 0])
                        if has_l1:
                            _mul_add_z(conv_aor, vgridc[1, ialpha], wvr[s, 1])
                            _mul_add_z(conv_aor, vgridc[2, ialpha], wvr[s, 2])
                            _mul_add_z(conv_aor, vgridc[3, ialpha], wvr[s, 3])
                    else:
                        _mul_add_d(conv_aor, vgridc[0, ialpha], wvr[s, 0])
                        if has_l1:
                            _mul_add_d(conv_aor, vgridc[1, ialpha], wvr[s, 1])
                            _mul_add_d(conv_aor, vgridc[2, ialpha], wvr[s, 2])
                            _mul_add_d(conv_aor, vgridc[3, ialpha], wvr[s, 3])
            t2 = time.monotonic()
            shape = wvk.shape
            wvk.shape = (nspin * ncpa * (i1 - i0), ng_recip)
            t2b = time.monotonic()
            ngrid64 = np.prod(np.asarray(cell.mesh).astype(np.float64))
            fft_fast(wvk, cell.mesh, fwd=True, inplace=True, scale=1.0 / ngrid64)
            wvk.shape = shape
            _fast_conj(wvk)
            t3 = time.monotonic()
            for s in range(nspin):
                if has_l1:
                    _mul_add_z(wvk[s, 1], igk[:, 0], wvk[s, 0])
                    _mul_add_z(wvk[s, 2], igk[:, 1], wvk[s, 0])
                    _mul_add_z(wvk[s, 3], igk[:, 2], wvk[s, 0])
                vmats = vmat_list[s]
                if vmats[kn].dtype == np.float64:
                    if rtype == np.float64:
                        _weight_symm_gpts(wvk[s, 0], cell.mesh)
                        # need to double count since we have half the plane-waves
                        vmats[kn][i0:i1] += 2 * lib.dot(wvk[s, 0], aoG.T).real
                    else:
                        vmats[kn][i0:i1] += lib.dot(wvk[s, 0], aoG.T).real
                elif vmats[kn].dtype == np.complex128:
                    vmats[kn][i0:i1] += lib.dot(wvk[s, 0], aoG.T)
                else:
                    raise ValueError
            # vmats[kn][:, i0:i1] += np.dot(aoG, wv[0].T)
            t4 = time.monotonic()
            # """
            dt0 += t1 - t0
            dt1 += t2 - t1
            dt2 += t3 - t2b
            dt3 += t4 - t3
            dt5 += t2b - t2
    print("VXC SPLITS", dt0, dt1, dt2, dt3, dt4, dt5)
    return vmats


class PySCFSDMXInitializer:
    def __init__(self, settings, **kwargs):
        self.settings = settings
        self.kwargs = kwargs

    def initialize_sdmx_generator(self, cell, nspin, kpts=None):
        if kpts is None:
            kpts = np.zeros((1, 3))
        res = EXXSphGenerator.from_settings_and_mol(
            self.settings, nspin, cell, kpts, **self.kwargs
        )
        return res


class EXXSphGenerator(sdmx.EXXSphGenerator):
    """
    Object used to evaluate SDMX features within PySCF.
    Note this evaluates the SDMX using FFTs, so the coords
    have to correspond to a uniform grid.
    """

    fast = False
    uses_rblk = False

    def __init__(self, plan, cutoff_type=None, lowmem=False):
        """
        Initialize EXXSphGenerator.

        Args:
            plan (SADMPlan or SDMXBasePlan): Plan for evaluating SDMX features.
        """
        if plan.fit_metric != "ovlp":
            raise NotImplementedError
        self.plan = plan
        self.lowmem = lowmem
        self.cutoff_type = cutoff_type
        self._buf = None
        self._cached_ao_data = None
        self._kpt_dim = None
        self._precomp_list = None

    def reset(self):
        self._buf = None
        self._cached_ao_data = None
        self._kpt_dim = None
        self._precomp_list = None

    @classmethod
    def from_settings_and_mol(
        cls,
        settings,
        nspin,
        cell,
        kpts,
        alpha0=None,
        lambd=None,
        nalpha=None,
        cutoff_type="ws",
        lowmem=False,
    ):
        kpt_dim = tools.pbc.get_monkhorst_pack_size(cell, kpts)
        if lambd is None:
            lambd = 1.8
        if alpha0 is None:
            lengths = np.linalg.norm(cell.lattice_vectors(), axis=1)
            supercell_length = np.max(lengths * kpt_dim)
            max_wpd = 0.5 * supercell_length
            alpha0 = 1.0 / (2 * max_wpd**2)
        if nalpha is None:
            max_exp = np.max(cell._env[cell._bas[:, PTR_EXP]])
            nalpha = 1 + int(1 + np.log(max_exp / alpha0) / np.log(lambd))
            nalpha += 2  # buffer exponents
        if isinstance(settings, SDMXFullSettings):
            plan = SDMXFullPlan(settings, nspin, alpha0, lambd, nalpha)
        elif isinstance(settings, SDMXSettings):
            plan = SDMXPlan(settings, nspin, alpha0, lambd, nalpha)
        elif isinstance(settings, SADMSettings):
            plan = SADMPlan(settings, nspin, alpha0, lambd, nalpha, fit_metric="ovlp")
        else:
            raise ValueError("Invalid type {}".format(type(settings)))
        return cls(plan, cutoff_type=cutoff_type, lowmem=lowmem)

    def _get_buffers(self, aobuf_size, caobuf_size, save_buf):
        if self._buf is not None and self._buf.size >= aobuf_size + caobuf_size:
            buf = self._buf
        else:
            buf = np.empty(aobuf_size + caobuf_size)
        aobuf = buf[:aobuf_size]
        caobuf = buf[aobuf_size : aobuf_size + caobuf_size]
        if save_buf:
            self._buf = buf
        else:
            self._buf = None
        return aobuf, caobuf

    def set_kpt_dim(self, kpt_dim):
        self._kpt_dim = kpt_dim

    def get_dm_convolutions(
        self, dms, cell, coords, kpts=None, cutoff=None, save_buf=False, spinpol=False
    ):
        old_kdim = self._kpt_dim
        self.set_kpt_dim(tools.pbc.get_monkhorst_pack_size(cell, kpts))
        if old_kdim is None or not (old_kdim == self._kpt_dim).all():
            self._precomp_list = precompute_all_convolutions(
                cell,
                kpts,
                self.plan.alphas,
                self.plan.alpha_norms,
                self.plan.settings.integral_type,
                kpt_dim=self._kpt_dim,
            )
        if spinpol:
            assert len(dms) == self.plan.nspin
            ndim = dms[0].ndim
            if ndim == 2:
                dms = (dm[None, :, :] for dm in dms)
            elif ndim == 3:
                pass
            else:
                raise ValueError
            has_mo = True
            for dm in dms:
                has_mo = has_mo and getattr(dm, "mo_coeff", None) is not None
            if has_mo:
                mo_coeff = [dm.mo_coeff for dm in dms]
                mo_occ = [dm.mo_occ for dm in dms]
            else:
                mo_coeff = None
                mo_occ = None
        else:
            assert self.plan.nspin == 1
            ndim = dms.ndim
            if ndim == 2:
                dms = dms[None, :, :]
            elif ndim == 3:
                pass
            else:
                raise ValueError
            has_mo = getattr(dms, "mo_coeff", None) is not None
            if has_mo:
                mo_coeff = dms.mo_coeff
                mo_occ = dms.mo_occ
            else:
                mo_coeff = None
                mo_occ = None
        has_l1 = self.has_l1
        if kpts is None:
            kpts = np.zeros((1, 3))
        if has_mo:
            tmps = compute_sdmx_tensor_mo(
                mo_coeff,
                mo_occ,
                cell,
                coords,
                self.plan.alphas,
                self.plan.alpha_norms,
                self.plan.settings.integral_type,
                kpts=kpts,
                cutoff_type=self.cutoff_type,
                has_l1=has_l1,
                kpt_dim=self._kpt_dim,
                precomp_list=self._precomp_list,
                spinpol=spinpol,
            )
        else:
            tmps = compute_sdmx_tensor_lowmem(
                dms,
                cell,
                coords,
                self.plan.alphas,
                self.plan.alpha_norms,
                self.plan.settings.integral_type,
                kpts=kpts,
                cutoff_type=self.cutoff_type,
                has_l1=has_l1,
                kpt_dim=self._kpt_dim,
                precomp_list=self._precomp_list,
                spinpol=spinpol,
            )
        if not spinpol:
            tmps = tmps[None, :, :, :]
        return tmps

    def get_features_from_convolutions(self, tmps, cell):
        n_out = tmps.shape[0]
        ngrids = tmps.shape[-1]
        n0 = self.plan.num_l0_feat
        nfeat = self.plan.settings.nfeat
        l0tmp2 = np.empty((n_out, n0, self.plan.nalpha, ngrids), dtype=np.complex128)
        l1tmp2 = np.empty(
            (n_out, nfeat - n0, 3, self.plan.nalpha, ngrids), dtype=np.complex128
        )
        out = np.empty((n_out, nfeat, ngrids), dtype=np.float64)
        for idm, tmp in enumerate(tmps):
            self.plan.get_features(
                tmp, out=out[idm], l0tmp=l0tmp2[idm], l1tmp=l1tmp2[idm]
            )
        self._cached_ao_data = (cell, l0tmp2, l1tmp2)
        return out

    def get_features(
        self, dms, cell, coords, kpts=None, cutoff=None, save_buf=False, spinpol=False
    ):
        if spinpol:
            assert len(dms) == self.plan.nspin
            ndim = dms[0].ndim
        else:
            assert self.plan.nspin == 1
            ndim = dms.ndim
        tmps = self.get_dm_convolutions(
            dms,
            cell,
            coords,
            kpts=kpts,
            cutoff=cutoff,
            save_buf=save_buf,
            spinpol=spinpol,
        )
        out = self.get_features_from_convolutions(tmps, cell)
        if ndim == 2:
            out = out[0]
        return out

    def get_convolution_potential(self, vxc_grid, spinpol=False):
        if self._cached_ao_data is None:
            raise RuntimeError
        n_out = self.plan.nspin
        nalpha = self.plan.nalpha
        if spinpol:
            ngrids = vxc_grid[0].shape[-1]
        else:
            ngrids = vxc_grid.shape[-1]
        has_l1 = self.has_l1
        l0tmp2, l1tmp2 = self._cached_ao_data[1:]
        if has_l1:
            ncpa = 4
        else:
            ncpa = 1
        tmp2 = np.zeros((n_out, ncpa, nalpha, ngrids), dtype=np.complex128)
        for idm in range(n_out):
            self.plan.get_vxc(vxc_grid[idm], l0tmp2[idm], l1tmp2[idm], out=tmp2[idm])
        if not spinpol:
            tmp2 = tmp2[0]
        return tmp2

    def get_vxc_from_vconv_(self, vxc_mat, vxc_conv, kpts=None, spinpol=None):
        if self._cached_ao_data is None:
            raise RuntimeError
        cell = self._cached_ao_data[0]
        compute_sdmx_tensor_vxc(
            vxc_mat,
            cell,
            vxc_conv,
            self.plan.alphas,
            self.plan.alpha_norms,
            self.plan.settings.integral_type,
            kpts=kpts,
            cutoff_type=self.cutoff_type,
            has_l1=self.has_l1,
            kpt_dim=self._kpt_dim,
            precomp_list=self._precomp_list,
            spinpol=spinpol,
        )
        return vxc_mat

    def get_vxc_(self, vxc_mat, vxc_grid, kpts=None, spinpol=False):
        vxc_conv = self.get_convolution_potential(vxc_grid, spinpol=spinpol)
        return self.get_vxc_from_vconv_(vxc_mat, vxc_conv, kpts=kpts, spinpol=spinpol)


if __name__ == "__main__":
    pass
