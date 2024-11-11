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
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft.gen_grid import ALIGNMENT_UNIT, make_mask
from pyscf.dft.numint import (
    OCCDROP,
    _contract_rho,
    _contract_rho_sparse,
    _dot_ao_ao_sparse,
    _dot_ao_dm,
    _dot_ao_dm_sparse,
    _empty_aligned,
    _NumIntMixin,
    _scale_ao_sparse,
    _sparse_enough,
)
from pyscf.gto.eval_gto import (
    BLKSIZE,
    CUTOFF,
    NBINS,
    _get_intor_and_comp,
    libcgto,
    make_loc,
    make_screen_index,
)
from pyscf.gto.mole import ANG_OF
from scipy.special import gamma, hyp1f1

from ciderpress.lib import load_library as load_cider_library

libcider = load_cider_library("libmcider")

GLOBAL_FRAC_LAPL_DATA = {}
DEFAULT_FRAC_LAPL_POWER = 0.5
DEFAULT_FRAC_LAPL_A = 0.25
DEFAULT_FRAC_LAPL_D = 0.002
DEFAULT_FRAC_LAPL_SIZE = 4000
XCUT_1F1 = 100


class FracLaplBuf:
    def __init__(self, a, d, size, lmax, s):
        self.a = a
        self.d = d
        self.size = size
        self.lmax = lmax
        self.s = s
        self.buf = np.empty((lmax + 1, 4, size))
        x = a * (np.exp(d * np.arange(size)) - 1)
        f = np.zeros((lmax + 1, size))
        xbig = x[x >= XCUT_1F1]
        xmax = x[x < XCUT_1F1][-1]
        for l in range(lmax + 1):
            f[l] = hyp1f1(1.5 + s + l, 1.5 + l, -x)
            f[l] *= 2 ** (2 * s) * gamma(1.5 + s + l) / gamma(1.5 + l)
            fsmall = f[l, x < XCUT_1F1][-1]
            f[l, x > XCUT_1F1] = fsmall * (xmax / xbig) ** (1.5 + s + l)
        libcider.initialize_spline_1f1(
            self.buf.ctypes.data_as(ctypes.c_void_p),
            f.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.size),
            ctypes.c_int(self.lmax),
        )
        df = np.zeros((lmax + 1, size))
        for l in range(lmax + 1):
            df[l] = hyp1f1(2.5 + s + l, 2.5 + l, -x)
            df[l] *= 2 ** (2 * s) * gamma(2.5 + s + l) / gamma(2.5 + l)
            fsmall = df[l, x < XCUT_1F1][-1]
            df[l, x > XCUT_1F1] = fsmall * (xmax / xbig) ** (2.5 + s + l)
        self.dbuf = np.empty((lmax + 1, 4, size))
        libcider.initialize_spline_1f1(
            self.dbuf.ctypes.data_as(ctypes.c_void_p),
            df.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.size),
            ctypes.c_int(self.lmax),
        )

    def set_data(self):
        # libcider.set_spline_1f1(
        #    self.buf.ctypes.data_as(ctypes.c_void_p),
        #    ctypes.c_double(self.a),
        #    ctypes.c_double(self.d),
        #    ctypes.c_int(self.size),
        #    ctypes.c_int(self.lmax),
        #    ctypes.c_double(self.s),
        # )
        libcider.set_spline_1f1_with_grad(
            self.buf.ctypes.data_as(ctypes.c_void_p),
            self.dbuf.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(self.a),
            ctypes.c_double(self.d),
            ctypes.c_int(self.size),
            ctypes.c_int(self.lmax),
            ctypes.c_double(self.s),
        )


def _set_flapl(a=0.025, d=0.004, size=6000, lmax=6, s=0.5):
    s_key = int(s * 100)
    if s_key not in GLOBAL_FRAC_LAPL_DATA or GLOBAL_FRAC_LAPL_DATA[s_key].lmax < lmax:
        GLOBAL_FRAC_LAPL_DATA[s_key] = FracLaplBuf(a, d, size, lmax, s)
    GLOBAL_FRAC_LAPL_DATA[s_key].set_data()


def set_frac_lapl(s, lmax=6):
    _set_flapl(lmax=lmax, s=s)


def eval_flapl_gto(
    slst,
    mol,
    coords,
    shls_slice=None,
    non0tab=None,
    ao_loc=None,
    cutoff=None,
    out=None,
    debug=False,
    deriv=0,
):
    if non0tab is not None:
        if (non0tab == 0).any():
            # TODO implement some sort of screening later
            raise NotImplementedError
    if mol.cart:
        feval = "GTOval_cart_deriv%d" % deriv
    else:
        feval = "GTOval_sph_deriv%d" % deriv
    eval_name, comp = _get_intor_and_comp(mol, feval)
    comp = len(slst)

    atm = np.asarray(mol._atm, dtype=np.int32, order="C")
    bas = np.asarray(mol._bas, dtype=np.int32, order="C")
    env = np.asarray(mol._env, dtype=np.double, order="C")
    coords = np.asarray(coords, dtype=np.double, order="F")
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ngrids = coords.shape[0]

    if eval_name == "GTOval_sph_deriv0":
        deriv = 0
    elif eval_name == "GTOval_sph_deriv1":
        deriv = 1
    else:
        raise NotImplementedError
    if debug:
        assert deriv == 0
        contract_fn = getattr(libcgto, "GTOcontract_exp0")
        eval_fn = getattr(libcgto, "GTOshell_eval_grid_cart")
    else:
        if deriv == 0:
            contract_fn = getattr(libcider, "GTOcontract_flapl0")
            eval_fn = getattr(libcgto, "GTOshell_eval_grid_cart")
        elif deriv == 1:
            contract_fn = getattr(libcider, "GTOcontract_flapl1")
            eval_fn = getattr(libcgto, "GTOshell_eval_grid_cart_deriv1")
            comp *= 4
        else:
            raise ValueError

    if ao_loc is None:
        ao_loc = make_loc(bas, eval_name)

    if shls_slice is None:
        shls_slice = (0, nbas)
    sh0, sh1 = shls_slice
    nao = ao_loc[sh1] - ao_loc[sh0]
    if "spinor" in eval_name:
        ao = np.ndarray(
            (2, comp, nao, ngrids), dtype=np.complex128, buffer=out
        ).transpose(0, 1, 3, 2)
    else:
        ao = np.ndarray((comp, nao, ngrids), buffer=out).transpose(0, 2, 1)

    if non0tab is None:
        if cutoff is None:
            non0tab = np.ones(((ngrids + BLKSIZE - 1) // BLKSIZE, nbas), dtype=np.uint8)
        else:
            non0tab = make_screen_index(mol, coords, shls_slice, cutoff)

    # drv = getattr(libcgto, eval_name)
    # normal call tree is GTOval_sph_deriv0 -> GTOval_sph -> GTOeval_sph_drv
    drv = getattr(libcgto, "GTOeval_sph_drv")
    for i, s in enumerate(slst):
        set_frac_lapl(s, lmax=mol._bas[:, ANG_OF].max())
        if deriv == 0:
            j = i
            param = (1, 1)
        else:
            j = i * 4
            param = (1, 4)
        drv(
            eval_fn,
            contract_fn,
            ctypes.c_double(1),
            ctypes.c_int(ngrids),
            (ctypes.c_int * 2)(*param),
            (ctypes.c_int * 2)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            ao[j].ctypes.data_as(ctypes.c_void_p),
            coords.ctypes.data_as(ctypes.c_void_p),
            non0tab.ctypes.data_as(ctypes.c_void_p),
            atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p),
        )
    if comp == 1:
        if "spinor" in eval_name:
            ao = ao[:, 0]
        else:
            ao = ao[0]
    return ao


def eval_rho(
    mol,
    all_ao,
    dm,
    flsettings,
    non0tab=None,
    xctype="LDA",
    hermi=0,
    with_lapl=True,
    verbose=None,
):
    ao, kao = all_ao
    xctype = xctype.upper()
    ngrids, nao = ao.shape[-2:]

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    if with_lapl:
        raise NotImplementedError
    else:
        rho = np.empty((5 + flsettings.nrho, ngrids))
        tau_idx = 4

    c0 = _dot_ao_dm(mol, ao[0], dm, non0tab, shls_slice, ao_loc)
    # rho[0] = numpy.einsum('pi,pi->p', ao[0], c0)
    rho[0] = _contract_rho(ao[0], c0)
    nk0 = flsettings.nk0
    nk1 = flsettings.nk1
    nd1 = flsettings.nd1

    for ik in range(nk0):
        ir = ik + 1 + tau_idx
        indk = flsettings.get_ik0(ik)
        rho[ir] = _contract_rho(kao[indk], c0)
        if not hermi:
            rho[ir] += _contract_rho(c0, kao[indk])
            rho[ir] *= 0.5

    for ik in range(nd1):
        ir = (3 * nk1 + nk0) + (1 + tau_idx) + (3 * ik)
        indk = flsettings.get_ik0(ik)
        for i in range(3):
            rho[ir + i] = _contract_rho(c0, kao[indk + 1 + i])

    rho[tau_idx] = 0
    for ik in range(flsettings.ndd):
        ir = (3 * nk1 + nk0) + (1 + tau_idx) + (3 * nd1) + ik
        rho[ir] = 0
    for i in range(1, 4):
        c1 = _dot_ao_dm(mol, ao[i], dm, non0tab, shls_slice, ao_loc)
        # rho[tau_idx] += numpy.einsum('pi,pi->p', c1, ao[i])
        rho[tau_idx] += _contract_rho(ao[i], c1)

        # rho[i] = numpy.einsum('pi,pi->p', c0, ao[i])
        rho[i] = _contract_rho(ao[i], c0)
        if hermi:
            rho[i] *= 2
        else:
            rho[i] += _contract_rho(c1, ao[0])

        for ik in range(flsettings.nk1):
            ir = 3 * ik + i + tau_idx + nk0
            indk = flsettings.get_ik0(ik)
            rho[ir] = _contract_rho(kao[indk], c1)
            if not hermi:
                rho[ir] += _contract_rho(c1, kao[indk])
                rho[ir] *= 0.5

        for ik in range(flsettings.ndd):
            ir = (3 * nk1 + nk0) + (1 + tau_idx) + (3 * nd1) + ik
            indk = flsettings.get_ik0(ik) + i
            rho[ir] += _contract_rho(kao[indk], c1)
            if not hermi:
                rho[ir] += _contract_rho(c1, kao[indk])

    # tau = 1/2 (\nabla f)^2
    rho[tau_idx] *= 0.5
    for ik in range(flsettings.ndd):
        ir = (3 * nk1 + nk0) + (1 + tau_idx) + (3 * nd1) + ik
        if not hermi:
            rho[ir] *= 0.5
    return rho


def eval_rho1(
    mol,
    all_ao,
    dm,
    flsettings,
    screen_index=None,
    xctype="LDA",
    hermi=0,
    with_lapl=True,
    cutoff=None,
    ao_cutoff=CUTOFF,
    pair_mask=None,
    verbose=None,
):
    ao, kao = all_ao
    if flsettings.nd1 != 0 or flsettings.ndd != 0:
        raise NotImplementedError
    xctype = xctype.upper()
    ngrids = ao.shape[-2]

    if cutoff is None:
        cutoff = CUTOFF
    cutoff = min(cutoff, 0.1)
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(ao_cutoff))

    if pair_mask is None:
        ovlp_cond = mol.get_overlap_cond()
        pair_mask = ovlp_cond < -np.log(cutoff)

    ao_loc = mol.ao_loc_nr()
    if with_lapl:
        raise NotImplementedError
    else:
        rho = np.empty((5 + flsettings.nrho, ngrids))
        tau_idx = 4
    c0 = _dot_ao_dm_sparse(ao[0], dm, nbins, screen_index, pair_mask, ao_loc)
    rho[0] = _contract_rho_sparse(ao[0], c0, screen_index, ao_loc)
    nk0 = flsettings.nk0
    nk1 = flsettings.nk1
    nd1 = flsettings.nd1

    for ik in range(nk0):
        # TODO might be able to use some kind of screening here
        ir = ik + 1 + tau_idx
        indk = flsettings.get_ik0(ik)
        # rho[ir] = _contract_rho_sparse(kao[indk], c0, screen_index, ao_loc)
        rho[ir] = _contract_rho(kao[indk], c0)
        if not hermi:
            # rho[ir] += _contract_rho_sparse(c0, kao[indk], screen_index, ao_loc)
            rho[ir] += _contract_rho(c0, kao[indk])
            rho[ir] *= 0.5

    for ik in range(nd1):
        ir = (3 * nk1 + nk0) + (1 + tau_idx) + (3 * ik)
        indk = flsettings.get_ik0(ik)
        for i in range(3):
            # rho[ir + i] = _contract_rho_sparse(
            #    c0, kao[indk + 1 + i], screen_index, ao_loc
            # )
            rho[ir + i] = _contract_rho(c0, kao[indk + 1 + i])

    rho[tau_idx] = 0
    for ik in range(flsettings.ndd):
        ir = (3 * nk1 + nk0) + (1 + tau_idx) + (3 * nd1) + ik
        rho[ir] = 0
    for i in range(1, 4):
        c1 = _dot_ao_dm_sparse(ao[i], dm.T, nbins, screen_index, pair_mask, ao_loc)
        rho[tau_idx] += _contract_rho_sparse(ao[i], c1, screen_index, ao_loc)

        rho[i] = _contract_rho_sparse(ao[i], c0, screen_index, ao_loc)
        if hermi:
            rho[i] *= 2
        else:
            rho[i] += _contract_rho_sparse(c1, ao[0], screen_index, ao_loc)

        for ik in range(flsettings.nk1):
            ir = 3 * ik + i + tau_idx + nk0
            indk = flsettings.get_ik0(ik)
            # rho[ir] = _contract_rho_sparse(kao[indk], c1, screen_index, ao_loc)
            rho[ir] = _contract_rho(kao[indk], c1)
            if not hermi:
                # rho[ir] += _contract_rho_sparse(c1, kao[indk], screen_index, ao_loc)
                rho[ir] += _contract_rho(c1, kao[indk])
                rho[ir] *= 0.5

        for ik in range(flsettings.ndd):
            ir = (3 * nk1 + nk0) + (1 + tau_idx) + (3 * nd1) + ik
            indk = flsettings.get_ik0(ik) + i
            # rho[ir] += _contract_rho_sparse(kao[indk], c1, screen_index, ao_loc)
            rho[ir] += _contract_rho(kao[indk], c1)
            if not hermi:
                # rho[ir] += _contract_rho_sparse(c1, kao[indk], screen_index, ao_loc)
                rho[ir] += _contract_rho(c1, kao[indk])

    rho[tau_idx] *= 0.5
    for ik in range(flsettings.ndd):
        ir = (3 * nk1 + nk0) + (1 + tau_idx) + (3 * nd1) + ik
        if not hermi:
            rho[ir] *= 0.5
    return rho


def _odp_dot_sparse_(
    all_ao,
    wv,
    nbins,
    mask,
    pair_mask,
    ao_loc,
    flsettings,
    aow1=None,
    aow2=None,
    vmat=None,
    v1=None,
):
    ao, kao = all_ao
    wv[0] *= 0.5  # *.5 for v+v.conj().T
    wv[4] *= 0.5  # *.5 for 1/2 in tau
    if flsettings.nrho > 0:
        wv[5:] *= 0.5

    aow1 = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow1)
    nk0 = flsettings.nk0
    nk1 = flsettings.nk1
    nd1 = flsettings.nd1
    ndd = flsettings.ndd
    tau_idx = 4
    if nd1 > 0 or ndd > 0:
        wvtmp = np.zeros(kao.shape[:2])
        for ik in range(nk0):
            ir = ik + 1 + tau_idx
            indk = flsettings.get_ik0(ik)
            wvtmp[indk] = wv[ir]
            # rho[ir] = _contract_rho(kao[indk], c0)
            # if not hermi:
            #    rho[ir] += _contract_rho(c0, kao[indk])
            #    rho[ir] *= 0.5
        for ik in range(nd1):
            ir = (3 * nk1 + nk0) + (1 + tau_idx) + (3 * ik)
            indk = flsettings.get_ik0(ik)
            wvtmp[indk + 1 : indk + 4] = wv[ir : ir + 3]
        aow2 = _scale_ao_sparse(kao, wvtmp, None, ao_loc, out=aow2)
        aow1 += aow2
        vmat = _dot_ao_ao_sparse(
            ao[0], aow1, None, nbins, None, pair_mask, ao_loc, hermi=0, out=vmat
        )
    elif nk0 > 0:
        # TODO might be able to use some kind of screening here
        # aow2 = _scale_ao_sparse(kao, wv[5:5 + nk0], mask, ao_loc, out=aow2)
        # aow1 += aow2
        # vmat = _dot_ao_ao_sparse(ao[0], aow1, None, nbins, mask, pair_mask,
        #                         ao_loc, hermi=0, out=vmat)
        aow2 = _scale_ao_sparse(kao, wv[5 : 5 + nk0], None, ao_loc, out=aow2)
        aow1 += aow2
        vmat = _dot_ao_ao_sparse(
            ao[0], aow1, None, nbins, None, pair_mask, ao_loc, hermi=0, out=vmat
        )
    else:
        vmat = _dot_ao_ao_sparse(
            ao[0], aow1, None, nbins, mask, pair_mask, ao_loc, hermi=0, out=vmat
        )

    if ndd > 0 or nd1 > 0:
        for j in range(3):
            i = j + 1
            # TODO might be able to use some kind of screening here
            aow1 = _scale_ao_sparse(ao[i], wv[4], mask, ao_loc, out=aow1)
            for ik in range(nk1):
                ir = 3 * ik + i + tau_idx + nk0
                indk = flsettings.get_ik0(ik)
                aow2 = _scale_ao_sparse(kao[indk], wv[ir], None, ao_loc, out=aow2)
                aow1 += aow2
            for ik in range(ndd):
                ir = (3 * nk1 + nk0) + (1 + tau_idx) + (3 * nd1) + ik
                indk = flsettings.get_ik0(ik) + i
                aow2 = _scale_ao_sparse(kao[indk], wv[ir], None, ao_loc, out=aow2)
                aow1 += aow2
            vmat = _dot_ao_ao_sparse(
                ao[i], aow1, None, nbins, None, pair_mask, ao_loc, hermi=0, out=vmat
            )
    elif nk1 > 0:
        kwv = np.empty((nk1, wv.shape[-1]))
        gkao = kao[:nk1]
        for j in range(3):
            i = j + 1
            # TODO might be able to use some kind of screening here
            kwv[:] = wv[5 + nk0 + j : 5 + nk0 + 3 * nk1 : 3]
            aow1 = _scale_ao_sparse(ao[i], wv[4], mask, ao_loc, out=aow1)
            # aow2 = _scale_ao_sparse(gkao, kwv, mask, ao_loc, out=aow2)
            # aow1 += aow2
            # vmat = _dot_ao_ao_sparse(ao[i], aow1, None, nbins, mask, pair_mask,
            #                         ao_loc, hermi=0, out=vmat)
            aow2 = _scale_ao_sparse(gkao, kwv, None, ao_loc, out=aow2)
            aow1 += aow2
            vmat = _dot_ao_ao_sparse(
                ao[i], aow1, None, nbins, None, pair_mask, ao_loc, hermi=0, out=vmat
            )
    else:
        for j in range(3):
            i = j + 1
            aow1 = _scale_ao_sparse(ao[i], wv[4], mask, ao_loc, out=aow1)
            v1 = _dot_ao_ao_sparse(
                ao[i], aow1, None, nbins, mask, pair_mask, ao_loc, hermi=1, out=v1
            )

    return aow1, aow2, vmat, v1


def eval_rho2(
    mol,
    all_ao,
    mo_coeff,
    mo_occ,
    flsettings,
    non0tab=None,
    xctype="LDA",
    with_lapl=True,
    verbose=None,
):
    ao, kao = all_ao
    xctype = xctype.upper()
    ngrids, nao = ao.shape[-2:]

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    pos = mo_occ > OCCDROP
    if np.any(pos):
        cpos = np.einsum("ij,j->ij", mo_coeff[:, pos], np.sqrt(mo_occ[pos]))
        if with_lapl:
            raise NotImplementedError
        else:
            rho = np.empty((5 + flsettings.nrho, ngrids))
            tau_idx = 4
        c0 = _dot_ao_dm(mol, ao[0], cpos, non0tab, shls_slice, ao_loc)
        # rho[0] = numpy.einsum('pi,pi->p', c0, c0)
        rho[0] = _contract_rho(c0, c0)
        cklst = []
        for i in range(kao.shape[0]):
            # TODO might be able to use some kind of screening here
            cklst.append(
                _dot_ao_dm(mol, kao[i], cpos, None, shls_slice, ao_loc)
                # _dot_ao_dm(mol, kao[i], cpos, non0tab, shls_slice, ao_loc)
            )
        nk0 = flsettings.nk0
        nk1 = flsettings.nk1
        nd1 = flsettings.nd1
        for ik in range(nk0):
            ir = ik + 1 + tau_idx
            indk = flsettings.get_ik0(ik)
            rho[ir] = _contract_rho(cklst[indk], c0)

        for ik in range(nd1):
            ir = (3 * nk1 + nk0) + (1 + tau_idx) + (3 * ik)
            indk = flsettings.get_ik0(ik)
            for i in range(3):
                rho[ir + i] = _contract_rho(c0, cklst[indk + 1 + i])

        rho[tau_idx] = 0
        for ik in range(flsettings.ndd):
            ir = (3 * nk1 + nk0) + (1 + tau_idx) + (3 * nd1) + ik
            rho[ir] = 0
        for i in range(1, 4):
            c1 = _dot_ao_dm(mol, ao[i], cpos, non0tab, shls_slice, ao_loc)
            rho[i] = _contract_rho(c0, c1) * 2
            rho[tau_idx] += _contract_rho(c1, c1)

            for ik in range(flsettings.nk1):
                ir = 3 * ik + i + tau_idx + nk0
                indk = flsettings.get_ik0(ik)
                rho[ir] = _contract_rho(cklst[indk], c1)

            for ik in range(flsettings.ndd):
                ir = (3 * nk1 + nk0) + (1 + tau_idx) + (3 * nd1) + ik
                indk = flsettings.get_ik0(ik) + i
                rho[ir] += _contract_rho(cklst[indk], c1)

        rho[tau_idx] *= 0.5
        return rho
    else:
        rho = np.zeros((5 + flsettings.nrho, ngrids))
        return rho


def _block_loop(
    ni,
    mol,
    grids,
    nao=None,
    deriv=0,
    max_memory=2000,
    non0tab=None,
    blksize=None,
    buf=None,
):
    """Define this macro to loop over grids by blocks."""
    if grids.coords is None:
        grids.build(with_non0tab=True)
    if nao is None:
        nao = mol.nao
    ngrids = grids.coords.shape[0]
    comp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
    flsettings = ni.plan.settings
    kcomp = flsettings.npow + 3 * flsettings.nd1
    # NOTE to index grids.non0tab, the blksize needs to be an integer
    # multiplier of BLKSIZE
    if blksize is None:
        blksize = int(max_memory * 1e6 / ((comp + kcomp + 1) * nao * 8 * BLKSIZE))
        blksize = max(4, min(blksize, ngrids // BLKSIZE + 1, 1200)) * BLKSIZE
    assert blksize % BLKSIZE == 0

    if non0tab is None and mol is grids.mol:
        non0tab = grids.non0tab
    if non0tab is None:
        non0tab = np.empty(
            ((ngrids + BLKSIZE - 1) // BLKSIZE, mol.nbas), dtype=np.uint8
        )
        non0tab[:] = NBINS + 1  # Corresponding to AO value ~= 1
    screen_index = non0tab

    # the xxx_sparse() functions require ngrids 8-byte aligned
    allow_sparse = ngrids % ALIGNMENT_UNIT == 0

    if buf is None:
        buf = _empty_aligned((comp + kcomp) * blksize * nao)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[ip0:ip1]
        weight = grids.weights[ip0:ip1]
        mask = screen_index[ip0 // BLKSIZE :]
        # TODO: pass grids.cutoff to eval_ao
        ao_buf = np.ndarray((comp + kcomp, weight.size * nao), buffer=buf)
        ao = ni.eval_ao(
            mol, coords, deriv=deriv, non0tab=mask, cutoff=grids.cutoff, out=ao_buf
        )
        kao = ni.eval_kao(
            flsettings.slist,
            mol,
            coords,
            deriv=0,
            non0tab=None,
            cutoff=grids.cutoff,
            out=ao_buf[comp:],
            n1=flsettings.nd1,
        )
        if not allow_sparse and not _sparse_enough(mask):
            # Unset mask for dense AO tensor. It determines which eval_rho
            # to be called in make_rho
            mask = None
        yield (ao, kao), mask, weight, coords


def eval_kao(
    slst,
    mol,
    coords,
    deriv=0,
    shls_slice=None,
    non0tab=None,
    cutoff=None,
    out=None,
    n1=0,
    verbose=None,
):
    assert deriv == 0
    comp = len(slst)
    if mol.cart:
        "GTOval_cart_deriv%d" % deriv
    else:
        "GTOval_sph_deriv%d" % deriv
    if n1 == 0:
        return eval_flapl_gto(
            slst, mol, coords, shls_slice, non0tab, cutoff=cutoff, out=out
        )
    else:
        comp += 3 * n1
        nao = mol.nao_nr()
        ngrids = coords.shape[0]
        if out is None:
            out = np.ndarray((comp, nao, ngrids), buffer=out)
        else:
            assert out.shape[0] == comp
            assert out.flags.c_contiguous
            out = out.reshape((comp, nao, ngrids))
        eval_flapl_gto(
            slst[:n1],
            mol,
            coords,
            shls_slice,
            non0tab,
            cutoff=cutoff,
            out=out[: 4 * n1],
            deriv=1,
        )
        eval_flapl_gto(
            slst[n1:],
            mol,
            coords,
            shls_slice,
            non0tab,
            cutoff=cutoff,
            out=out[4 * n1 :],
        )
        return out.transpose(0, 2, 1)


def nr_rks(
    ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
):
    if relativity != 0:
        raise NotImplementedError

    xctype = "MGGA"

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    vmat = np.zeros((nset, nao, nao))

    def block_loop(ao_deriv):
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory=max_memory
        ):
            for i in range(nset):
                rho = make_rho(i, ao, mask, xctype)
                exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[:2]
                den = rho[0] * weight
                nelec[i] += den.sum()
                excsum[i] += np.dot(den, exc)
                wv = weight * vxc
                yield i, ao, mask, wv

    aow1 = None
    aow2 = None
    pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)
    if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
        raise NotImplementedError("laplacian in meta-GGA method")
    ao_deriv = 1
    v1 = np.zeros_like(vmat)
    for i, ao, mask, wv in block_loop(ao_deriv):
        _odp_dot_sparse_(
            ao,
            wv,
            nbins,
            mask,
            pair_mask,
            ao_loc,
            ni.flsettings,
            aow1=aow1,
            aow2=aow2,
            vmat=vmat[i],
            v1=v1[i],
        )
    vmat = lib.hermi_sum(vmat, axes=(0, 2, 1))
    vmat += v1


class FLNumInt(_NumIntMixin):
    """Numerical integration methods for non-relativistic RKS and UKS
    with fractional Laplacian density descriptors."""

    cutoff = CUTOFF * 1e2  # cutoff for small AO product

    def __init__(self, plan):
        """
        Initialize FLNumInt object.

        Args:
            plan (FracLaplPlan): Plan object for how to evaluate fractional
                Laplacian features
        """
        super(FLNumInt, self).__init__()
        self.plan = plan

    def nr_vxc(
        self,
        mol,
        grids,
        xc_code,
        dms,
        spin=0,
        relativity=0,
        hermi=0,
        max_memory=2000,
        verbose=None,
    ):
        if spin == 0:
            return self.nr_rks(
                mol, grids, xc_code, dms, relativity, hermi, max_memory, verbose
            )
        else:
            return self.nr_uks(
                mol, grids, xc_code, dms, relativity, hermi, max_memory, verbose
            )

    get_vxc = nr_vxc

    nr_rks = nr_rks
    nr_uks = None  # TODO write nr_uks

    # nr_nlc_vxc = numint.nr_nlc_vxc
    nr_sap = nr_sap_vxc = numint.nr_sap_vxc

    make_mask = staticmethod(make_mask)
    eval_ao = staticmethod(numint.eval_ao)
    eval_kao = staticmethod(eval_kao)
    eval_rho = staticmethod(eval_rho)
    eval_rho1 = lib.module_method(eval_rho1, absences=["cutoff"])
    eval_rho2 = staticmethod(eval_rho2)
    get_rho = numint.get_rho

    block_loop = _block_loop

    def _gen_rho_evaluator(self, mol, dms, hermi=0, with_lapl=True, grids=None):
        if getattr(dms, "mo_coeff", None) is not None:
            # TODO: test whether dm.mo_coeff matching dm
            mo_coeff = dms.mo_coeff
            mo_occ = dms.mo_occ
            if isinstance(dms, np.ndarray) and dms.ndim == 2:
                mo_coeff = [mo_coeff]
                mo_occ = [mo_occ]
        else:
            mo_coeff = mo_occ = None

        if isinstance(dms, np.ndarray) and dms.ndim == 2:
            dms = dms[np.newaxis]

        if hermi != 1 and dms[0].dtype == np.double:
            # (D + D.T)/2 because eval_rho computes 2*(|\nabla i> D_ij <j|) instead of
            # |\nabla i> D_ij <j| + |i> D_ij <\nabla j| for efficiency when dm is real
            dms = lib.hermi_sum(np.asarray(dms, order="C"), axes=(0, 2, 1)) * 0.5
            hermi = 1

        nao = dms[0].shape[0]
        ndms = len(dms)

        ovlp_cond = mol.get_overlap_cond()
        if dms[0].dtype == np.double:
            dm_cond = np.max(
                [mol.condense_to_shell(dm, "absmax") for dm in dms], axis=0
            )
            pair_mask = (np.exp(-ovlp_cond) * dm_cond) > self.cutoff
        else:
            pair_mask = ovlp_cond < -np.log(self.cutoff)

        def make_rho(idm, ao, sindex, xctype):
            if sindex is not None and grids is not None:
                return self.eval_rho1(
                    mol,
                    ao,
                    dms[idm],
                    self.plan.settings,
                    sindex,
                    xctype,
                    hermi,
                    with_lapl,
                    cutoff=self.cutoff,
                    ao_cutoff=grids.cutoff,
                    pair_mask=pair_mask,
                )
            elif mo_coeff is not None:
                return self.eval_rho2(
                    mol,
                    ao,
                    mo_coeff[idm],
                    mo_occ[idm],
                    self.plan.settings,
                    sindex,
                    xctype,
                    with_lapl,
                )
            else:
                return self.eval_rho(
                    mol,
                    ao,
                    dms[idm],
                    self.plan.settings,
                    sindex,
                    xctype,
                    hermi,
                    with_lapl,
                )

        return make_rho, ndms, nao
