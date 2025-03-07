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
from ase.utils.timing import Timer
from pyscf import lib
from pyscf.dft import numint
from pyscf.dft.gen_grid import ALIGNMENT_UNIT, BLKSIZE, NBINS
from pyscf.dft.numint import _dot_ao_ao_sparse, _scale_ao_sparse, eval_ao

import ciderpress.pyscf.frac_lapl as nlof
from ciderpress.dft.plans import (
    FracLaplPlan,
    SemilocalPlan,
    get_rho_tuple_with_grad_cross,
    vxc_tuple_to_array,
)
from ciderpress.dft.settings import FeatureSettings
from ciderpress.dft.xc_evaluator import MappedXC
from ciderpress.dft.xc_evaluator2 import MappedXC2
from ciderpress.pyscf.nldf_convolutions import PyscfNLDFGenerator
from ciderpress.pyscf.sdmx import EXXSphGenerator

DEFAULT_RHOCUT = 1e-9


def _tau_dot_sparse(
    bra, ket, wv, nbins, screen_index, pair_mask, ao_loc, out=None, aow=None
):
    """Similar to _tau_dot, while sparsity is explicitly considered. Note the
    return may have ~1e-13 difference to _tau_dot.
    """
    nao = bra.shape[1]
    if out is None:
        out = np.zeros((nao, nao), dtype=bra.dtype)
    hermi = 1
    aow = _scale_ao_sparse(ket[1], wv, screen_index, ao_loc, out=aow)
    _dot_ao_ao_sparse(
        bra[1], aow, None, nbins, screen_index, pair_mask, ao_loc, hermi, out
    )
    aow = _scale_ao_sparse(ket[2], wv, screen_index, ao_loc, out=aow)
    _dot_ao_ao_sparse(
        bra[2], aow, None, nbins, screen_index, pair_mask, ao_loc, hermi, out
    )
    aow = _scale_ao_sparse(ket[3], wv, screen_index, ao_loc, out=aow)
    _dot_ao_ao_sparse(
        bra[3], aow, None, nbins, screen_index, pair_mask, ao_loc, hermi, out
    )
    return out


def _get_sdmx_orbs(ni, mol, coords, ao):
    ni.timer.start("sdmx ao")
    if ni.has_sdmx and ni.sdmxgen.fast:
        if ao is None:
            sdmx_ao = eval_ao(mol, coords, deriv=0)
        else:
            sdmx_ao = ao[0] if ao.ndim == 3 else ao
        sdmx_cao = ni.sdmxgen.get_cao(mol, coords, save_buf=True)
    elif ni.has_sdmx:
        sdmx_ao, sdmx_cao = ni.sdmxgen.get_orb_vals(mol, coords, save_buf=True)
    else:
        sdmx_ao, sdmx_cao = None, None
    ni.timer.stop("sdmx ao")
    return sdmx_ao, sdmx_cao


def nr_rks(
    ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
):
    """

    Args:
        ni (CiderNumInt):
        mol (pyscf.gto.Mole):
        grids (pyscf.dft.gen_grid.Grids):
        xc_code:
        dms:
        relativity:
        hermi:
        max_memory:
        verbose:

    Returns:

    """
    ni.timer.start("nr_rks")
    if relativity != 0:
        raise NotImplementedError

    xctype = ni.settings.sl_settings.level
    ni.initialize_feature_generators(mol, grids, 1)

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))

    if dms.ndim == 2:
        dms = dms[None, :, :]

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    vmat = np.zeros((nset, nao, nao))

    def block_loop(ao_deriv):
        if ni.has_sdmx:
            extra_ao = ni.sdmxgen.get_extra_ao(mol)
        else:
            extra_ao = 0
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory=max_memory, extra_ao=extra_ao
        ):
            sdmx_ao, sdmx_cao = _get_sdmx_orbs(ni, mol, coords, ao)
            for i in range(nset):
                rho = make_rho(i, ao, mask, xctype)
                ni.timer.start("sdmx fwd")
                if ni.has_sdmx:
                    sdmx_feat = ni.sdmxgen.get_features(
                        dms[i],
                        mol,
                        coords,
                        ao=sdmx_ao,
                        cao=sdmx_cao,
                    )
                else:
                    sdmx_feat = None
                ni.timer.stop("sdmx fwd")
                ni.timer.start("xc cider")
                exc, (vxc, vxc_nldf, vxc_sdmx) = ni.eval_xc_cider(
                    xc_code, rho, None, sdmx_feat, deriv=1, xctype=xctype
                )[:2]
                ni.timer.stop("xc cider")
                assert vxc_nldf is None
                ni.timer.start("sdmx bwd")
                if ni.has_sdmx:
                    ni.sdmxgen.get_vxc_(vmat[i], vxc_sdmx[0] * weight)
                ni.timer.stop("sdmx bwd")
                den = rho[0] * weight
                nelec[i] += den.sum()
                excsum[i] += np.dot(den, exc)
                wv = weight * vxc
                yield i, ao, mask, wv
        # if ni.has_sdmx:
        #     ni.sdmxgen.reset_buffers()

    buffers = None
    pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)
    if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
        raise NotImplementedError("laplacian in meta-GGA method")
    ao_deriv = 1
    v1 = np.zeros_like(vmat)
    for i, ao, mask, wv in block_loop(ao_deriv):
        vmats, buffers = ni.contract_wv(
            ao,
            wv,
            nbins,
            mask,
            pair_mask,
            ao_loc,
            vmats=(vmat[i], v1[i]),
            buffers=buffers,
        )
    vmat = lib.hermi_sum(vmat, axes=(0, 2, 1))
    vmat += v1

    if ni.has_sdmx:
        ni.sdmxgen._cached_ao_data = None

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]
    if isinstance(dms, np.ndarray):
        dtype = dms.dtype
    else:
        dtype = np.result_type(*dms)
    if vmat.dtype != dtype:
        vmat = np.asarray(vmat, dtype=dtype)
    # if ni.has_sdmx:
    #    ni.sdmxgen.reset_buffers()
    ni.timer.stop("nr_rks")
    return nelec, excsum, vmat


def nr_uks(
    ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
):
    ni.timer.start("nr_uks")
    if relativity != 0:
        raise NotImplementedError

    xctype = ni.settings.sl_settings.level
    ni.initialize_feature_generators(mol, grids, 2)

    dma, dmb = numint._format_uks_dm(dms)
    if dma.ndim == 2:
        dma = dma[None, :, :]
        dmb = dmb[None, :, :]
        is_2d = True
    else:
        is_2d = False
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))
    if ni.has_sdmx:
        dm_for_sdmx = [np.stack([dma[i], dmb[i]]) for i in range(nset)]
    else:
        dm_for_sdmx = None

    nelec = np.zeros((2, nset))
    excsum = np.zeros(nset)
    vmat = np.zeros((2, nset, nao, nao))

    def block_loop(ao_deriv):
        if ni.has_sdmx:
            extra_ao = ni.sdmxgen.get_extra_ao(mol)
        else:
            extra_ao = 0
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory=max_memory, extra_ao=extra_ao
        ):
            sdmx_ao, sdmx_cao = _get_sdmx_orbs(ni, mol, coords, ao)
            for i in range(nset):
                rho_a = make_rhoa(i, ao, mask, xctype)
                rho_b = make_rhob(i, ao, mask, xctype)
                rho = (rho_a, rho_b)
                ni.timer.start("sdmx fwd")
                if ni.has_sdmx:
                    sdmx_feat = ni.sdmxgen.get_features(
                        dm_for_sdmx[i],
                        mol,
                        coords,
                        ao=sdmx_ao,
                        cao=sdmx_cao,
                    )
                else:
                    sdmx_feat = None
                ni.timer.stop("sdmx fwd")
                ni.timer.start("xc cider")
                exc, (vxc, vxc_nldf, vxc_sdmx) = ni.eval_xc_cider(
                    xc_code, rho, None, sdmx_feat, deriv=1, xctype=xctype
                )[:2]
                ni.timer.stop("xc cider")
                assert vxc_nldf is None
                ni.timer.start("sdmx bwd")
                if ni.has_sdmx:
                    ni.sdmxgen.get_vxc_(vmat[:, i], vxc_sdmx * weight)
                ni.timer.stop("sdmx bwd")
                den_a = rho_a[0] * weight
                den_b = rho_b[0] * weight
                nelec[0, i] += den_a.sum()
                nelec[1, i] += den_b.sum()
                excsum[i] += np.dot(den_a, exc)
                excsum[i] += np.dot(den_b, exc)
                wv = weight * vxc
                yield i, ao, mask, wv

    buffers = None
    pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)
    if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
        raise NotImplementedError("laplacian in meta-GGA method")
    ao_deriv = 1
    v1 = np.zeros_like(vmat)
    for i, ao, mask, wv in block_loop(ao_deriv):
        buffers = ni.contract_wv(
            ao,
            wv[0],
            nbins,
            mask,
            pair_mask,
            ao_loc,
            vmats=(vmat[0, i], v1[0, i]),
            buffers=buffers,
        )[1]
        buffers = ni.contract_wv(
            ao,
            wv[1],
            nbins,
            mask,
            pair_mask,
            ao_loc,
            vmats=(vmat[1, i], v1[1, i]),
            buffers=buffers,
        )[1]
    vmat = lib.hermi_sum(vmat.reshape(-1, nao, nao), axes=(0, 2, 1)).reshape(
        2, nset, nao, nao
    )
    vmat += v1

    if ni.has_sdmx:
        ni.sdmxgen._cached_ao_data = None

    if isinstance(dma, np.ndarray) and is_2d:
        vmat = vmat[:, 0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]

    dtype = np.result_type(dma, dmb)
    if vmat.dtype != dtype:
        vmat = np.asarray(vmat, dtype=dtype)
    ni.timer.stop("nr_uks")
    return nelec, excsum, vmat


def nr_rks_nldf(
    ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
):
    """

    Args:
        ni (NLDFNumInt):
        mol (pyscf.gto.Mole):
        grids (pyscf.dft.gen_grid.Grids):
        xc_code (str):
        dms (numpy.ndarray):
        relativity:
        hermi:
        max_memory:
        verbose:

    Returns:

    """
    if not hasattr(grids, "grids_indexer"):
        raise ValueError("Grids object must have indexer for NLDF evaluation")

    if relativity != 0:
        raise NotImplementedError

    xctype = ni.settings.sl_settings.level
    ni.initialize_feature_generators(mol, grids, 1)
    assert ni.has_nldf

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))

    if dms.ndim == 2:
        dms = dms[None, :, :]

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    vmat = np.zeros((nset, nao, nao))
    nrho = 5 if xctype == "MGGA" else 4
    if not ni.settings.nlof_settings.is_empty:
        nrho += ni.settings.nlof_settings.nrho
    rho_full = np.zeros((nset, nrho, grids.weights.size), dtype=np.float64, order="C")
    wv_full = np.zeros((nset, nrho, grids.weights.size), dtype=np.float64, order="C")

    num_nldf = ni.settings.nldf_settings.nfeat
    vxc_nldf_full = np.zeros(
        (nset, num_nldf, grids.weights.size), dtype=np.float64, order="C"
    )

    def block_loop(ao_deriv):
        ip0 = 0
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory=max_memory
        ):
            ip1 = ip0 + weight.size
            for i in range(nset):
                yield i, ip0, ip1, ao, mask, weight, coords
            ip0 = ip1

    if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
        raise NotImplementedError("laplacian in meta-GGA method")
    ao_deriv = 1
    for i, ip0, ip1, ao, mask, weight, coords in block_loop(ao_deriv):
        rho_full[i, :, ip0:ip1] = make_rho(i, ao, mask, xctype)

    par_atom = False
    extra_ao = ni.nldfgen.get_extra_ao()
    if ni.has_sdmx:
        extra_ao += ni.sdmxgen.get_extra_ao(mol)
    if par_atom:
        raise NotImplementedError
    else:
        nldf_feat = []
        for idm in range(nset):
            nldf_feat.append(ni.nldfgen.get_features(rho_full[idm]))
        ip0 = 0
        for mask, weight, coords in ni.extra_block_loop(
            mol, grids, max_memory=max_memory, extra_ao=extra_ao
        ):
            ip1 = ip0 + weight.size
            sdmx_ao, sdmx_cao = _get_sdmx_orbs(ni, mol, coords, None)
            for idm in range(nset):
                rho = rho_full[idm, :, ip0:ip1]
                if ni.has_sdmx:
                    sdmx_feat = ni.sdmxgen.get_features(
                        dms[idm], mol, coords, ao=sdmx_ao, cao=sdmx_cao
                    )
                else:
                    sdmx_feat = None
                exc, (vxc, vxc_nldf, vxc_sdmx) = ni.eval_xc_cider(
                    xc_code,
                    rho,
                    nldf_feat[idm][:, ip0:ip1],
                    sdmx_feat,
                    deriv=1,
                    xctype=xctype,
                )[:2]
                if ni.has_sdmx:
                    ni.sdmxgen.get_vxc_(vmat[idm], vxc_sdmx[0] * weight)
                vxc_nldf_full[idm, :, ip0:ip1] = vxc_nldf * weight
                den = rho[0] * weight
                nelec[idm] += den.sum()
                excsum[idm] += np.dot(den, exc)
                wv_full[idm, :, ip0:ip1] = weight * vxc
            ip0 = ip1
        for idm in range(nset):
            wv_full[idm, :, :] += ni.nldfgen.get_potential(vxc_nldf_full[idm])

    buffers = None
    pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)
    v1 = np.zeros_like(vmat)
    for i, ip0, ip1, ao, mask, weight, coords in block_loop(ao_deriv):
        wv = wv_full[i, :, ip0:ip1]
        vmats, buffers = ni.contract_wv(
            ao,
            wv,
            nbins,
            mask,
            pair_mask,
            ao_loc,
            vmats=(vmat[i], v1[i]),
            buffers=buffers,
        )
    vmat = lib.hermi_sum(vmat, axes=(0, 2, 1))
    vmat += v1

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]
    if isinstance(dms, np.ndarray):
        dtype = dms.dtype
    else:
        dtype = np.result_type(*dms)
    if vmat.dtype != dtype:
        vmat = np.asarray(vmat, dtype=dtype)
    return nelec, excsum, vmat


def nr_uks_nldf(
    ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
):
    ni.timer.start("nr_uks")
    if relativity != 0:
        raise NotImplementedError

    xctype = ni.settings.sl_settings.level
    ni.initialize_feature_generators(mol, grids, 2)

    dma, dmb = numint._format_uks_dm(dms)
    if dma.ndim == 2:
        dma = dma[None, :, :]
        dmb = dmb[None, :, :]
        is_2d = True
    else:
        is_2d = False
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))
    if ni.has_sdmx:
        dm_for_sdmx = [np.stack([dma[i], dmb[i]]) for i in range(nset)]
    else:
        dm_for_sdmx = None

    nelec = np.zeros((2, nset))
    excsum = np.zeros(nset)
    vmat = np.zeros((2, nset, nao, nao))
    nrho = 5 if xctype == "MGGA" else 4
    if not ni.settings.nlof_settings.is_empty:
        nrho += ni.settings.nlof_settings.nrho
    rhoa_full = np.zeros((nset, nrho, grids.weights.size), dtype=np.float64, order="C")
    rhob_full = np.zeros((nset, nrho, grids.weights.size), dtype=np.float64, order="C")
    wva_full = np.zeros((nset, nrho, grids.weights.size), dtype=np.float64, order="C")
    wvb_full = np.zeros((nset, nrho, grids.weights.size), dtype=np.float64, order="C")

    num_nldf = ni.settings.nldf_settings.nfeat
    vxc_nldf_full = np.zeros(
        (nset, 2, num_nldf, grids.weights.size), dtype=np.float64, order="C"
    )

    def block_loop(ao_deriv):
        ip0 = 0
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory=max_memory
        ):
            ip1 = ip0 + weight.size
            for i in range(nset):
                yield i, ip0, ip1, ao, mask, weight, coords
            ip0 = ip1

    if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
        raise NotImplementedError("laplacian in meta-GGA method")
    ao_deriv = 1
    for i, ip0, ip1, ao, mask, weight, coords in block_loop(ao_deriv):
        rhoa_full[i, :, ip0:ip1] = make_rhoa(i, ao, mask, xctype)
        rhob_full[i, :, ip0:ip1] = make_rhob(i, ao, mask, xctype)

    par_atom = False
    extra_ao = ni.nldfgen.get_extra_ao()
    if ni.has_sdmx:
        extra_ao += ni.sdmxgen.get_extra_ao(mol)
    if par_atom:
        raise NotImplementedError
    else:
        nldf_feat = []
        for idm in range(nset):
            nldf_feat.append(
                np.stack(
                    [
                        ni.nldfgen.get_features(rhoa_full[idm], spin=0),
                        ni.nldfgen.get_features(rhob_full[idm], spin=1),
                    ]
                )
            )
        ip0 = 0
        for mask, weight, coords in ni.extra_block_loop(
            mol, grids, max_memory=max_memory, extra_ao=extra_ao
        ):
            ip1 = ip0 + weight.size
            sdmx_ao, sdmx_cao = _get_sdmx_orbs(ni, mol, coords, None)
            for idm in range(nset):
                rho_a = rhoa_full[idm, :, ip0:ip1]
                rho_b = rhob_full[idm, :, ip0:ip1]
                rho = (rho_a, rho_b)
                if ni.has_sdmx:
                    sdmx_feat = ni.sdmxgen.get_features(
                        dm_for_sdmx[i],
                        mol,
                        coords,
                        ao=sdmx_ao,
                        cao=sdmx_cao,
                    )
                else:
                    sdmx_feat = None
                exc, (vxc, vxc_nldf, vxc_sdmx) = ni.eval_xc_cider(
                    xc_code,
                    rho,
                    nldf_feat[idm][..., ip0:ip1],
                    sdmx_feat,
                    deriv=1,
                    xctype=xctype,
                )[:2]
                if ni.has_sdmx:
                    ni.sdmxgen.get_vxc_(vmat[:, i], vxc_sdmx * weight)
                vxc_nldf_full[idm, ..., ip0:ip1] = vxc_nldf * weight
                den_a = rho_a[0] * weight
                den_b = rho_b[0] * weight
                nelec[0, i] += den_a.sum()
                nelec[1, i] += den_b.sum()
                excsum[i] += np.dot(den_a, exc)
                excsum[i] += np.dot(den_b, exc)
                wva_full[idm, :, ip0:ip1] = weight * vxc[0]
                wvb_full[idm, :, ip0:ip1] = weight * vxc[1]
            ip0 = ip1
        for idm in range(nset):
            wva_full[idm, :, :] += ni.nldfgen.get_potential(
                vxc_nldf_full[idm, 0], spin=0
            )
            wvb_full[idm, :, :] += ni.nldfgen.get_potential(
                vxc_nldf_full[idm, 1], spin=1
            )

    buffers = None
    pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)
    if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
        raise NotImplementedError("laplacian in meta-GGA method")
    ao_deriv = 1
    v1 = np.zeros_like(vmat)
    for i, ip0, ip1, ao, mask, weight, coords in block_loop(ao_deriv):
        wva = wva_full[i, :, ip0:ip1]
        wvb = wvb_full[i, :, ip0:ip1]
        buffers = ni.contract_wv(
            ao,
            wva,
            nbins,
            mask,
            pair_mask,
            ao_loc,
            vmats=(vmat[0, i], v1[0, i]),
            buffers=buffers,
        )[1]
        buffers = ni.contract_wv(
            ao,
            wvb,
            nbins,
            mask,
            pair_mask,
            ao_loc,
            vmats=(vmat[1, i], v1[1, i]),
            buffers=buffers,
        )[1]

    vmat = lib.hermi_sum(vmat.reshape(-1, nao, nao), axes=(0, 2, 1)).reshape(
        2, nset, nao, nao
    )
    vmat += v1

    if ni.has_sdmx:
        ni.sdmxgen._cached_ao_data = None

    if isinstance(dma, np.ndarray) and is_2d:
        vmat = vmat[:, 0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]

    dtype = np.result_type(dma, dmb)
    if vmat.dtype != dtype:
        vmat = np.asarray(vmat, dtype=dtype)
    ni.timer.stop("nr_uks")
    return nelec, excsum, vmat


class CiderNumIntMixin:

    sl_plan: SemilocalPlan
    fl_plan: FracLaplPlan
    sdmxgen: EXXSphGenerator
    nldfgen: PyscfNLDFGenerator
    settings: FeatureSettings
    xmix: float
    rhocut: float

    def nlc_coeff(self, xc_code):
        if self._nlc_coeff is not None:
            return self._nlc_coeff
        else:
            return super().nlc_coeff(xc_code)

    @property
    def settings(self):
        return self.mlxc.settings

    @property
    def has_sdmx(self):
        return self.settings.has_sdmx

    @property
    def has_nldf(self):
        return self.settings.has_nldf

    def build(self, mol=None):
        self.mol = mol
        self.timer = Timer()
        self.sl_plan = None
        self.fl_plan = None
        self.sdmxgen = None
        self.nldfgen = None

    def reset(self, mol=None):
        self.mol = mol
        self.sl_plan = None
        self.fl_plan = None
        self.sdmxgen = None
        self.nldfgen = None

    def initialize_feature_generators(self, mol, grids, nspin):
        self.sl_plan = SemilocalPlan(self.settings.sl_settings, nspin)
        self.fl_plan = FracLaplPlan(self.settings.nlof_settings, nspin)
        cond = self.sdmxgen is None
        cond = cond or self.mol != mol
        cond = cond or self.sdmxgen.plan.nspin != nspin
        cond = cond and self.settings.has_sdmx
        if cond:
            self.sdmxgen = self.sdmx_init.initialize_sdmx_generator(mol, nspin)
        self.mol = mol

    def eval_xc_cider(
        self,
        xc_code,
        rho,
        nldf_feat,
        sdmx_feat,
        deriv=1,
        omega=None,
        xctype=None,
        verbose=None,
    ):
        xc_code = self.slxc
        rho = np.asarray(rho, order="C", dtype=np.double)
        if deriv > 1:
            raise NotImplementedError
        if rho.ndim == 2:
            closed = True
            rho = rho[None, :]
        else:
            closed = False
        nfeat = self.settings.nfeat
        ngrids = rho.shape[-1]
        nspin = rho.shape[0]
        vxc = np.zeros_like(rho)
        xctype = self._xc_type(xc_code)
        if xctype == "LDA":
            nvar = 1
        elif xctype == "GGA":
            nvar = 4
        elif xctype == "MGGA":
            nvar = 5
        else:
            exc = np.zeros(ngrids)
        if xctype in ["LDA", "GGA", "MGGA"]:
            if nspin == 1:
                rhotmp = np.ascontiguousarray(rho[0, :nvar])
                exc, vxc[0, :nvar] = self.eval_xc_eff(
                    xc_code,
                    rhotmp,
                    deriv=deriv,
                    omega=omega,
                    xctype=xctype,
                    verbose=verbose,
                )[:2]
            else:
                exc, vxc[:, :nvar] = self.eval_xc_eff(
                    xc_code,
                    rho[:, :nvar],
                    deriv=deriv,
                    omega=omega,
                    xctype=xctype,
                    verbose=verbose,
                )[:2]

        start = 0
        X0T = np.empty((nspin, nfeat, ngrids))
        has_sl = not self.settings.sl_settings.is_empty
        has_nldf = not self.settings.nldf_settings.is_empty
        has_nlof = not self.settings.nlof_settings.is_empty
        has_sdmx = not self.settings.sdmx_settings.is_empty
        if has_sl:
            nfeat_tmp = self.settings.sl_settings.nfeat
            X0T[:, start : start + nfeat_tmp] = self.sl_plan.get_feat(rho[:, :5])
            start += nfeat_tmp
        if has_nldf:
            if nldf_feat.ndim == 2:
                nldf_feat = nldf_feat[None, :]
            assert nldf_feat.shape[0] == nspin
            assert nldf_feat.ndim == 3
            nfeat_tmp = self.settings.nldf_settings.nfeat
            X0T[:, start : start + nfeat_tmp] = nldf_feat
            start += nfeat_tmp
        if has_nlof:
            nfeat_tmp = self.settings.nlof_settings.nfeat
            X0T[:, start : start + nfeat_tmp] = self.fl_plan.get_feat(rho)
            start += nfeat_tmp
        if has_sdmx:
            if sdmx_feat.ndim == 2:
                sdmx_feat = sdmx_feat[None, :]
            assert sdmx_feat.shape[0] == nspin
            assert sdmx_feat.ndim == 3
            nfeat_tmp = self.settings.sdmx_settings.nfeat
            X0T[:, start : start + nfeat_tmp] = sdmx_feat
            start += nfeat_tmp
        if start != nfeat:
            raise RuntimeError("nfeat mismatch, this should not happen!")

        X0TN = self.settings.normalizers.get_normalized_feature_vector(X0T)
        xmix = self.xmix
        if isinstance(self.mlxc, MappedXC):
            exc_ml, dexcdX0TN_ml = self.mlxc(X0TN, rhocut=self.rhocut)
        elif isinstance(self.mlxc, MappedXC2):
            rho_tuple = get_rho_tuple_with_grad_cross(rho, is_mgga=True)
            exc_ml, dexcdX0TN_ml, vrho_tuple = self.mlxc(
                X0TN, rho_tuple, rhocut=self.rhocut
            )
            vxc[:] += xmix * vxc_tuple_to_array(rho, vrho_tuple)
        else:
            raise TypeError("mlxc must be MappedXC or MappedXC2")
        exc_ml *= xmix
        dexcdX0TN_ml *= xmix
        vxc_ml = self.settings.normalizers.get_derivative_wrt_unnormed_features(
            X0T, dexcdX0TN_ml
        )
        exc[:] += exc_ml / (rho[:, 0].sum(axis=0) + 1e-16)

        start = 0
        if has_sl:
            nfeat_tmp = self.settings.sl_settings.nfeat
            self.sl_plan.get_vxc(
                rho[:, :5], vxc_ml[:, start : start + nfeat_tmp], vxc=vxc[:, :5]
            )
            start += nfeat_tmp
        if has_nldf:
            nfeat_tmp = self.settings.nldf_settings.nfeat
            vxc_nldf = vxc_ml[:, start : start + nfeat_tmp]
            start += nfeat_tmp
        else:
            vxc_nldf = None
        if has_nlof:
            nfeat_tmp = self.settings.nlof_settings.nfeat
            self.fl_plan.get_vxc(vxc_ml[:, start : start + nfeat_tmp], vxc=vxc)
            start += nfeat_tmp
        if has_sdmx:
            nfeat_tmp = self.settings.sdmx_settings.nfeat
            vxc_sdmx = vxc_ml[:, start : start + nfeat_tmp]
            start += nfeat_tmp
        else:
            vxc_sdmx = None
        if start != nfeat:
            raise RuntimeError("nfeat mismatch, this should not happen!")

        if closed:
            vxc = vxc[0]

        return exc, (vxc, vxc_nldf, vxc_sdmx), None, None


class CiderNumInt(CiderNumIntMixin, numint.NumInt):
    def __init__(
        self, mlxc, slxc, nldf_init, sdmx_init, xmix=1.0, rhocut=None, nlc_coeff=None
    ):
        """

        Args:
            mlxc (MappedXC): Model for XC energy
            slxc (str): semilocal contribution to XC energy
            nldf_init (PySCFNLDFInitializer)
            sdmx_init (PySCFSDMXInitializer)
            xmix (float): Mixing fraction of ML functional
            rhocut (float): Low density cutoff for numerical stability
        """
        self.mlxc = mlxc
        self.slxc = slxc
        self.xmix = xmix
        self.rhocut = DEFAULT_RHOCUT if rhocut is None else rhocut
        self.mol = None
        self.nldf_init = nldf_init
        self.sdmx_init = sdmx_init
        self.sdmxgen = None
        self.nldfgen = None
        self._nlc_coeff = nlc_coeff
        super(CiderNumInt, self).__init__()

    def method_not_implemented(self, *args, **kwargs):
        raise NotImplementedError

    nr_rks = nr_rks
    nr_uks = nr_uks

    nr_rks_fxc = method_not_implemented
    nr_uks_fxc = method_not_implemented
    nr_rks_fxc_st = method_not_implemented
    cache_xc_kernel = method_not_implemented
    cache_xc_kernel1 = method_not_implemented

    def contract_wv(
        self,
        ao,
        wv,
        nbins,
        mask,
        pair_mask,
        ao_loc,
        buffers=None,
        vmats=None,
    ):
        if vmats is None:
            vmats = (None, None)
        if not wv.flags.c_contiguous:
            wv = np.ascontiguousarray(wv)
        vmat, v1 = vmats
        aow = buffers
        wv[0] *= 0.5
        aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
        vmat = _dot_ao_ao_sparse(
            ao[0], aow, None, nbins, mask, pair_mask, ao_loc, hermi=0, out=vmat
        )
        if len(wv) > 4:
            wv[4] *= 0.5
            v1 = _tau_dot_sparse(
                ao, ao, wv[4], nbins, mask, pair_mask, ao_loc, out=v1, aow=aow
            )
        return (vmat, v1), aow

    def block_loop(
        self,
        mol,
        grids,
        nao=None,
        deriv=0,
        max_memory=2000,
        non0tab=None,
        blksize=None,
        buf=None,
        extra_ao=0,
    ):
        """Define this macro to loop over grids by blocks.
        Same as pyscf block_loop except extra_ao allows accounting for
        extra ao terms in blksize calculation.
        """
        if grids.coords is None:
            grids.build(with_non0tab=True)
        if nao is None:
            nao = mol.nao
        ngrids = grids.coords.shape[0]
        comp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
        # NOTE to index grids.non0tab, the blksize needs to be an integer
        # multiplier of BLKSIZE
        if blksize is None:
            blksize = int(
                max_memory * 1e6 / (((comp + 1) * nao + extra_ao) * 8 * BLKSIZE)
            )
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
        allow_sparse = ngrids % ALIGNMENT_UNIT == 0 and nao > numint.SWITCH_SIZE

        if buf is None:
            buf = numint._empty_aligned(comp * blksize * nao)
        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            mask = screen_index[ip0 // BLKSIZE :]
            ao = self.eval_ao(
                mol, coords, deriv=deriv, non0tab=mask, cutoff=grids.cutoff, out=buf
            )
            if not allow_sparse and not numint._sparse_enough(mask):
                # Unset mask for dense AO tensor. It determines which eval_rho
                # to be called in make_rho
                mask = None
            yield ao, mask, weight, coords


class _FLNumIntMixin:

    feat_plan = None
    settings: FeatureSettings = None

    nr_rks = nr_rks
    nr_uks = nr_uks

    nr_nlc_vxc = numint.nr_nlc_vxc
    nr_sap = nr_sap_vxc = numint.nr_sap_vxc

    make_mask = staticmethod(nlof.make_mask)
    eval_ao = staticmethod(numint.eval_ao)
    eval_kao = staticmethod(nlof.eval_kao)
    eval_rho = staticmethod(nlof.eval_rho)
    eval_rho1 = lib.module_method(nlof.eval_rho1, absences=["cutoff"])
    eval_rho2 = staticmethod(nlof.eval_rho2)
    get_rho = numint.get_rho

    def block_loop(
        ni,
        mol,
        grids,
        nao=None,
        deriv=0,
        max_memory=2000,
        non0tab=None,
        blksize=None,
        buf=None,
        extra_ao=None,
    ):
        """Define this macro to loop over grids by blocks. Expands on
        CiderNumInt.block_loop by also returning the fractional laplacian-
        like operators on the orbitals, as set up by the FracLaplSettings
        """
        if grids.coords is None:
            grids.build(with_non0tab=True)
        if nao is None:
            nao = mol.nao
        ngrids = grids.coords.shape[0]
        comp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
        flsettings = ni.settings.nlof_settings
        kcomp = flsettings.npow + 3 * flsettings.nd1
        # NOTE to index grids.non0tab, the blksize needs to be an integer
        # multiplier of BLKSIZE
        if blksize is None:
            blksize = int(
                max_memory * 1e6 / (((comp + kcomp + 1) * nao + extra_ao) * 8 * BLKSIZE)
            )
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
            buf = nlof._empty_aligned((comp + kcomp) * blksize * nao)
        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            mask = screen_index[ip0 // BLKSIZE :]
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
            if not allow_sparse and not nlof._sparse_enough(mask):
                # Unset mask for dense AO tensor. It determines which eval_rho
                # to be called in make_rho
                mask = None
            yield (ao, kao), mask, weight, coords

    def contract_wv(
        self,
        ao,
        wv,
        nbins,
        mask,
        pair_mask,
        ao_loc,
        buffers=None,
        vmats=None,
    ):
        if buffers is None:
            buffers = (None, None)
        if vmats is None:
            vmats = (None, None)
        aow1, aow2, vmat, v1 = nlof._odp_dot_sparse_(
            ao,
            wv,
            nbins,
            mask,
            pair_mask,
            ao_loc,
            self.settings.nlof_settings,
            aow1=buffers[0],
            aow2=buffers[1],
            vmat=vmats[0],
            v1=vmats[1],
        )
        return (vmat, v1), (aow1, aow2)

    _gen_rho_evaluator = nlof.FLNumInt._gen_rho_evaluator


class _NLDFMixin:

    nr_rks = nr_rks_nldf
    nr_uks = nr_uks_nldf

    def extra_block_loop(
        self,
        mol,
        grids,
        max_memory=2000,
        non0tab=None,
        blksize=None,
        make_mask=False,
        extra_ao=None,
    ):
        if grids.coords is None:
            grids.build(with_non0tab=True)
        ngrids = grids.coords.shape[0]
        # NOTE to index grids.non0tab, the blksize needs to be an integer
        # multiplier of BLKSIZE
        if blksize is None:
            blksize = int(max_memory * 1e6 / ((1 + extra_ao) * 8 * BLKSIZE))
            blksize = max(4, min(blksize, ngrids // BLKSIZE + 1, 1200)) * BLKSIZE
        assert blksize % BLKSIZE == 0

        if make_mask:
            if non0tab is None and mol is grids.mol:
                non0tab = grids.non0tab
            if non0tab is None:
                non0tab = np.empty(
                    ((ngrids + BLKSIZE - 1) // BLKSIZE, mol.nbas), dtype=np.uint8
                )
                non0tab[:] = NBINS + 1  # Corresponding to AO value ~= 1
            screen_index = non0tab
        else:
            screen_index = None

        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            if make_mask:
                mask = screen_index[ip0 // BLKSIZE :]
            else:
                mask = None
            yield mask, weight, coords


class NLOFNumInt(_FLNumIntMixin, CiderNumInt):
    pass


class NLDFNumInt(_NLDFMixin, CiderNumInt):

    grids = None

    def initialize_feature_generators(self, mol, grids, nspin):
        cond = self.nldfgen is None
        cond = cond or self.grids != grids
        cond = cond or self.mol != mol
        cond = cond or self.nldfgen.plan.nspin != nspin
        if cond:
            self.nldfgen = self.nldf_init.initialize_nldf_generator(
                mol, grids.grids_indexer, nspin
            )
            self.nldfgen.interpolator.set_coords(grids.coords)
        super().initialize_feature_generators(mol, grids, nspin)
        self.grids = grids


class NLDFNLOFNumInt(_NLDFMixin, _FLNumIntMixin, CiderNumInt):

    grids = None

    def initialize_feature_generators(self, mol, grids, nspin):
        cond = self.nldfgen is None
        cond = cond or self.grids != grids
        cond = cond or self.mol != mol
        cond = cond or self.nldfgen.plan.nspin != nspin
        if cond:
            self.nldfgen = self.nldf_init.initialize_nldf_generator(
                mol, grids.grids_indexer, nspin
            )
        super().initialize_feature_generators(mol, grids, nspin)
        self.grids = grids
