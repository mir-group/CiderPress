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

import sys

import numpy as np
from pyscf import lib
from pyscf.pbc.dft import numint
from pyscf.pbc.dft.gen_grid import UniformGrids
from pyscf.pbc.lib.kpts import KPoints

from ciderpress.dft.plans import SemilocalPlan
from ciderpress.pyscf.numint import CiderNumIntMixin
from ciderpress.pyscf.pbc.util import FFTInterpolator

DEFAULT_RHOCUT = 1e-8


def _get_kpts_12(kpts, kpts_band):
    if kpts is None:
        kpts = np.zeros((1, 3))
    elif isinstance(kpts, KPoints):
        kpts = kpts.kpts
    if kpts.ndim == 1:
        kpts = kpts[None, :]
    kpts2 = kpts
    if kpts_band is not None:
        kpts1 = _kpts1 = np.reshape(kpts_band, (-1, 3))
        dk = abs(kpts[:, None] - kpts1).max(axis=2)
        k_idx, kband_idx = np.where(dk < numint.KPT_DIFF_TOL)
        where = np.empty(len(kpts1), dtype=int)
        where[kband_idx] = k_idx
        kband_mask = np.ones(len(kpts1), dtype=bool)
        kband_mask[kband_idx] = False
        kpts1_uniq = kpts1[kband_mask]
        if kpts1_uniq.size > 0:
            kpts_all = np.vstack([kpts, kpts1_uniq])
            where[kband_mask] = len(kpts) + np.arange(kpts1_uniq.shape[0])
        else:
            kpts_all = kpts2
        kpts1 = np.asarray([kpts_all[k] for k in where])
        assert (np.abs(kpts1 - _kpts1) < 1e-15).all()
    else:
        kpts1 = kpts2
        where = np.arange(len(kpts1))
    return kpts1, kpts2


def nr_rks(
    ni,
    cell,
    grids,
    xc_code,
    dms,
    spin=0,
    relativity=0,
    hermi=1,
    kpts=None,
    kpts_band=None,
    max_memory=2000,
    verbose=None,
):
    if ni.dense_mesh is not None and isinstance(grids, UniformGrids):
        return nr_rks_dense(
            ni,
            cell,
            grids,
            xc_code,
            dms,
            spin,
            relativity,
            hermi,
            kpts,
            kpts_band,
            max_memory,
            verbose,
        )
    max_memory = 2000
    ni.timer.start("nr_rks")
    if relativity != 0:
        raise NotImplementedError

    xctype = "MGGA"
    ni.initialize_feature_generators_kpts(cell, grids, 1, kpts)

    kpts1, kpts2 = _get_kpts_12(kpts, kpts_band)

    ao_deriv = 1
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms, hermi, False)

    is_sdmx_rblk = ni.has_sdmx and ni.sdmxgen.uses_rblk
    is_sdmx_rglob = ni.has_sdmx and not ni.sdmxgen.uses_rblk

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    shls_slice = (0, cell.nbas)
    ao_loc = cell.ao_loc
    vmat = [0] * nset
    v_hermi = 1
    if is_sdmx_rglob:
        assert isinstance(grids, UniformGrids)
        sdmx_feat_all = ni.sdmxgen.get_features(
            dms,
            cell,
            grids.coords,
            kpts2,
            spinpol=False
            # TODO set nao_per_block for mem
        )
        vxc_sdmx_all = np.empty_like(sdmx_feat_all)
    else:
        sdmx_feat_all = None
        vxc_sdmx_all = None
    i0 = 0
    i1 = 0
    for ao_k1, ao_k2, mask, weight, coords in ni.block_loop(
        cell, grids, nao, ao_deriv, kpts, kpts_band, max_memory
    ):
        i1 = i0 + weight.size
        ni.timer.start("sdmx ao")
        if is_sdmx_rblk:
            raise NotImplementedError
            # sdmx_ao_k1 = ao_k1[0] if ao_k1.ndim == 3 else ao_k1
            # sdmx_ao_k2 = ao_k2[0] if ao_k2.ndim == 3 else ao_k2
            # sdmx_data = ni.sdmxgen.get_cao(cell, coords, save_buf=True)
        else:
            sdmx_ao_k1, sdmx_ao_k2, sdmx_data = None, None, None
            sdmx_cao_k2 = None, None
        ni.timer.stop("sdmx ao")
        for i in range(nset):
            rho = make_rho(i, ao_k2, mask, xctype).real
            ni.timer.start("sdmx fwd")
            if is_sdmx_rblk:
                sdmx_feat = ni.sdmxgen.get_features(
                    dms[i],
                    cell,
                    coords,
                    ao=sdmx_ao_k2,
                    data=sdmx_data,
                )
            elif is_sdmx_rglob:
                sdmx_feat = sdmx_feat_all[i, ..., i0:i1]
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
            if is_sdmx_rblk:
                ni.sdmxgen.get_vxc_(
                    vmat[i], vxc_sdmx[0] * weight, ao=sdmx_ao_k1, cao=sdmx_cao_k2
                )
            elif is_sdmx_rglob:
                vxc_sdmx_all[..., i0:i1] = vxc_sdmx * weight
            ni.timer.stop("sdmx bwd")
            den = rho[0] * weight
            nelec[i] += den.sum()
            excsum[i] += np.dot(den, exc)
            wv = weight * vxc
            vmat[i] += ni._vxc_mat(
                cell, ao_k1, wv, mask, xctype, shls_slice, ao_loc, v_hermi
            )
        i0 = i1
    if is_sdmx_rglob:
        assert nset == 1
        ni.sdmxgen.get_vxc_(vmat[0], vxc_sdmx_all, kpts=kpts1, spinpol=False)
    vmat = np.stack(vmat)
    # call swapaxes method to swap last two indices because vmat may be a 3D
    # array (nset,nao,nao) in single k-point mode or a 4D array
    # (nset,nkpts,nao,nao) in k-points mode
    vmat = vmat + vmat.conj().swapaxes(-2, -1)
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]
    return nelec, excsum, vmat


def nr_rks_dense(
    ni,
    cell,
    grids,
    xc_code,
    dms,
    spin=0,
    relativity=0,
    hermi=1,
    kpts=None,
    kpts_band=None,
    max_memory=2000,
    verbose=None,
):
    max_memory = 2000
    ni.timer.start("nr_rks")
    if relativity != 0:
        raise NotImplementedError

    xctype = "MGGA"
    ni.initialize_feature_generators_kpts(cell, grids, 1, kpts)
    kpts1, kpts2 = _get_kpts_12(kpts, kpts_band)
    ao_deriv = 1
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms, hermi, False)

    assert nset == 1

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    shls_slice = (0, cell.nbas)
    ao_loc = cell.ao_loc
    vmat = [0] * nset
    v_hermi = 1
    if ni.has_sdmx:
        assert isinstance(grids, UniformGrids)
        sdmx_feat_all = ni.sdmxgen.get_dm_convolutions(
            dms,
            cell,
            grids.coords,
            kpts2,
            spinpol=False,
            # TODO set nao_per_block for mem
        )
    else:
        sdmx_feat_all = None
    i0 = 0
    i1 = 0
    rho_all = np.empty((5, grids.weights.size), dtype=np.float64)
    for ao_k1, ao_k2, mask, weight, coords in ni.block_loop(
        cell, grids, nao, ao_deriv, kpts, kpts_band, max_memory
    ):
        i1 = i0 + weight.size
        for i in range(nset):
            rho_all[:, i0:i1] = make_rho(i, ao_k2, mask, xctype).real
        i0 = i1

    assert grids.weights.size == np.prod(cell.mesh)
    wv_all = np.zeros((5, grids.weights.size), dtype=np.float64)
    ffti_rho = FFTInterpolator(cell.mesh, ni.dense_mesh, r2c=True, num_fft_buffer=5)
    rho_dense = ffti_rho.interpolate(rho_all, fwd=True)
    delta = 1e-18
    rho_damp = np.sqrt(rho_dense[0] * rho_dense[0] + delta)
    rhoc = rho_dense[0].copy()
    tau_damp = np.sqrt(rho_dense[4] * rho_dense[4] + delta)
    tauc = rho_dense[4].copy()
    rho_dense[0] = rho_damp
    rho_dense[4] = tau_damp
    rho_damp[:] = rhoc / rho_damp
    tau_damp[:] = tauc / tau_damp
    if ni.has_sdmx:
        r2c = sdmx_feat_all.dtype == np.float64
        ffti_sdmx = FFTInterpolator(cell.mesh, ni.dense_mesh, r2c=r2c)
        sdmx_dense = ffti_sdmx.interpolate(sdmx_feat_all, fwd=True)
        # sdmx_dense[..., cond] = 0.0
    else:
        ffti_sdmx = None
    blksize = 10000  # TODO adapt based on memory
    weight = np.sum(grids.weights) / np.prod(ni.dense_mesh)
    for i0, i1 in lib.prange(0, np.prod(ni.dense_mesh), blksize):
        if ni.has_sdmx:
            sdmx_feat = ni.sdmxgen.get_features_from_convolutions(
                sdmx_dense[..., i0:i1], cell
            )
        exc, (_vxc, _vxc_nldf, _vxc_sdmx) = ni.eval_xc_cider(
            xc_code, rho_dense[..., i0:i1], None, sdmx_feat, deriv=1, xctype=xctype
        )[:2]
        den = weight * rho_dense[0, i0:i1]
        nelec[0] += den.sum()
        excsum[0] += np.dot(den, exc)
        assert _vxc_nldf is None
        rho_dense[..., i0:i1] = _vxc
        rho_dense[0, i0:i1] *= rho_damp[i0:i1]
        rho_dense[1, i0:i1] *= tau_damp[i0:i1]
        if ni.has_sdmx:
            _vxc_sdmx = ni.sdmxgen.get_convolution_potential(_vxc_sdmx, spinpol=False)
            sdmx_dense[..., i0:i1] = _vxc_sdmx
    vxc = rho_dense
    vxc_sdmx = sdmx_dense
    # vxc[:, cond] = 0.0
    # uniform weights on denser grid, volume / ngrids
    wv_all = ffti_rho.interpolate(vxc, fwd=False)
    if ni.has_sdmx:
        # vxc_sdmx[..., cond] = 0.0
        wv_sdmx = ffti_sdmx.interpolate(vxc_sdmx, fwd=False)
    weight = grids.weights
    wv_all[:] *= weight
    wv_sdmx[:] *= weight

    i0 = 0
    for ao_k1, ao_k2, mask, weight, coords in ni.block_loop(
        cell, grids, nao, ao_deriv, kpts, kpts_band, max_memory
    ):
        i1 = i0 + weight.size
        for i in range(nset):
            vmat[i] += ni._vxc_mat(
                cell, ao_k1, wv_all[:, i0:i1], mask, xctype, shls_slice, ao_loc, v_hermi
            )
        i0 = i1
    if ni.has_sdmx:
        ni.sdmxgen.get_vxc_from_vconv_(vmat[0], wv_sdmx, kpts=kpts1, spinpol=False)
    vmat = np.stack(vmat)
    vmat = vmat + vmat.conj().swapaxes(-2, -1)
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]
    return nelec, excsum, vmat


def nr_uks(
    ni,
    cell,
    grids,
    xc_code,
    dms,
    spin=1,
    relativity=0,
    hermi=1,
    kpts=None,
    kpts_band=None,
    max_memory=2000,
    verbose=None,
):
    max_memory = 2000
    ni.timer.start("nr_rks")
    if relativity != 0:
        raise NotImplementedError

    xctype = "MGGA"
    ni.initialize_feature_generators_kpts(cell, grids, 2, kpts)

    kpts1, kpts2 = _get_kpts_12(kpts, kpts_band)

    ao_deriv = 1
    dma, dmb = numint._format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(cell, dma, hermi, False)[:2]
    make_rhob = ni._gen_rho_evaluator(cell, dmb, hermi, False)[0]

    is_sdmx_rblk = ni.has_sdmx and ni.sdmxgen.uses_rblk
    is_sdmx_rglob = ni.has_sdmx and not ni.sdmxgen.uses_rblk

    nelec = np.zeros((2, nset))
    excsum = np.zeros(nset)

    shls_slice = (0, cell.nbas)
    ao_loc = cell.ao_loc
    vmata = [0] * nset
    vmatb = [0] * nset
    v_hermi = 1
    if is_sdmx_rglob:
        assert isinstance(grids, UniformGrids)
        sdmx_feat_all = ni.sdmxgen.get_features(
            (dma, dmb),
            cell,
            grids.coords,
            kpts2,
            spinpol=True
            # TODO set nao_per_block for mem
        )
        vxc_sdmx_all = np.empty_like(sdmx_feat_all)
    else:
        sdmx_feat_all = None
        vxc_sdmx_all = None
    i0 = 0
    i1 = 0
    if nset != 1:
        raise NotImplementedError  # TODO handle multiple sets
    for ao_k1, ao_k2, mask, weight, coords in ni.block_loop(
        cell, grids, nao, ao_deriv, kpts, kpts_band, max_memory
    ):
        i1 = i0 + weight.size
        ni.timer.start("sdmx ao")
        if is_sdmx_rblk:
            raise NotImplementedError
        ni.timer.stop("sdmx ao")
        for i in range(nset):
            rho_a = make_rhoa(i, ao_k2, mask, xctype).real
            rho_b = make_rhob(i, ao_k2, mask, xctype).real
            rho = (rho_a, rho_b)
            if is_sdmx_rblk:
                raise NotImplementedError
            elif is_sdmx_rglob:
                sdmx_feat = sdmx_feat_all[:, ..., i0:i1]
            else:
                sdmx_feat = None
            ni.timer.start("xc cider")
            exc, (vxc, vxc_nldf, vxc_sdmx) = ni.eval_xc_cider(
                xc_code, rho, None, sdmx_feat, deriv=1, xctype=xctype
            )[:2]
            ni.timer.stop("xc cider")
            assert vxc_nldf is None
            if is_sdmx_rblk:
                raise NotImplementedError
            elif is_sdmx_rglob:
                vxc_sdmx_all[..., i0:i1] = vxc_sdmx * weight
            dena = rho_a[0] * weight
            denb = rho_b[0] * weight
            nelec[0, i] += dena.sum()
            nelec[1, i] += denb.sum()
            excsum[i] += dena.dot(exc)
            excsum[i] += denb.dot(exc)
            wv = weight * vxc
            vmata[i] += ni._vxc_mat(
                cell, ao_k1, wv[0], mask, xctype, shls_slice, ao_loc, v_hermi
            )
            vmatb[i] += ni._vxc_mat(
                cell, ao_k1, wv[1], mask, xctype, shls_slice, ao_loc, v_hermi
            )
        i0 = i1
    if is_sdmx_rglob:
        assert nset == 1
        ni.sdmxgen.get_vxc_(
            (vmata[0], vmatb[0]), vxc_sdmx_all, kpts=kpts1, spinpol=True
        )
    vmat = np.stack([vmata, vmatb])
    vmat = vmat + vmat.conj().swapaxes(-2, -1)
    if nset == 1:
        nelec = nelec[:, 0]
        excsum = excsum[0]
        vmat = vmat[:, 0]
    return nelec, excsum, vmat


class CiderNumInt(CiderNumIntMixin, numint.NumInt):
    def __init__(
        self,
        mlxc,
        nldf_init,
        sdmx_init,
        xmix=1.0,
        rhocut=None,
        nlc_coeff=None,
        dense_mesh=None,
    ):
        """

        Args:
            mlxc (MappedXC): Model for XC energy
            nldf_init (PySCFNLDFInitializer)
            sdmx_init (PySCFSDMXInitializer)
            xmix (float): Mixing fraction of ML functional
            rhocut (float): Low density cutoff for numerical stability
            dense_mesh (3-tuple or array): Denser mesh for XC integration
        """
        self.mlxc = mlxc
        self.xmix = xmix
        self.rhocut = DEFAULT_RHOCUT if rhocut is None else rhocut
        self.mol = None
        self.nldf_init = nldf_init
        self.sdmx_init = sdmx_init
        self.sdmxgen = None
        self.nldfgen = None
        self._nlc_coeff = nlc_coeff
        self.dense_mesh = dense_mesh
        super(CiderNumInt, self).__init__()

    @lib.with_doc(nr_rks.__doc__)
    def nr_rks(
        self,
        cell,
        grids,
        xc_code,
        dms,
        relativity=0,
        hermi=1,
        kpt=np.zeros(3),
        kpts_band=None,
        max_memory=2000,
        verbose=None,
    ):
        if kpts_band is not None:
            # To compute Vxc on kpts_band, convert the NumInt object to KNumInt object.
            ni = self.view(CiderKNumInt)
            nao = dms.shape[-1]
            dms = dms.reshape(-1, nao, nao)
            return ni.nr_rks(
                cell,
                grids,
                xc_code,
                dms,
                relativity,
                hermi,
                kpt.reshape(1, 3),
                kpts_band,
                max_memory,
                verbose,
            )
        spin = 0
        return nr_rks(
            self,
            cell,
            grids,
            xc_code,
            dms,
            spin,
            relativity,
            hermi,
            kpt,
            kpts_band,
            max_memory,
            verbose,
        )

    @lib.with_doc(nr_uks.__doc__)
    def nr_uks(
        self,
        cell,
        grids,
        xc_code,
        dms,
        relativity=0,
        hermi=1,
        kpt=np.zeros(3),
        kpts_band=None,
        max_memory=2000,
        verbose=None,
    ):
        if kpts_band is not None:
            # To compute Vxc on kpts_band, convert the NumInt object to KNumInt object.
            ni = self.view(CiderKNumInt)
            nao = dms.shape[-1]
            dms = dms.reshape(2, -1, nao, nao)
            return ni.nr_uks(
                cell,
                grids,
                xc_code,
                dms,
                relativity,
                hermi,
                kpt.reshape(1, 3),
                kpts_band,
                max_memory,
                verbose,
            )
        spin = 1
        return nr_uks(
            self,
            cell,
            grids,
            xc_code,
            dms,
            spin,
            relativity,
            hermi,
            kpt,
            kpts_band,
            max_memory,
            verbose,
        )

    def method_not_implemented(self, *args, **kwargs):
        raise NotImplementedError

    nr_rks_fxc = method_not_implemented
    nr_uks_fxc = method_not_implemented
    nr_rks_fxc_st = method_not_implemented
    cache_xc_kernel = method_not_implemented
    cache_xc_kernel1 = method_not_implemented

    def initialize_feature_generators_kpts(self, cell, grids, nspin, kpts):
        self.sl_plan = SemilocalPlan(self.settings.sl_settings, nspin)
        self.fl_plan = None
        cond = self.sdmxgen is None
        cond = cond or self.cell != cell
        cond = cond or self.sdmxgen.plan.nspin != nspin
        cond = cond and self.settings.has_sdmx
        if cond:
            self.sdmxgen = self.sdmx_init.initialize_sdmx_generator(cell, nspin, kpts)
        self.cell = cell


class CiderKNumInt(CiderNumIntMixin, numint.KNumInt):
    def __init__(
        self,
        mlxc,
        slxc,
        nldf_init,
        sdmx_init,
        xmix=1.0,
        rhocut=None,
        nlc_coeff=None,
        dense_mesh=None,
    ):
        """

        Args:
            mlxc (MappedXC): Model for XC energy
            slxc (str): Semilocal part of XC functional
            nldf_init (PySCFNLDFInitializer)
            sdmx_init (PySCFSDMXInitializer)
            xmix (float): Mixing fraction of ML functional
            rhocut (float): Low density cutoff for numerical stability
            dense_mesh (3-tuple or array): Denser mesh for XC integrations
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
        self.dense_mesh = dense_mesh
        super(CiderKNumInt, self).__init__()

    @lib.with_doc(nr_rks.__doc__)
    def nr_rks(
        self,
        cell,
        grids,
        xc_code,
        dms,
        relativity=0,
        hermi=1,
        kpts=None,
        kpts_band=None,
        max_memory=2000,
        verbose=None,
        **kwargs
    ):
        if kpts is None:
            if "kpt" in kwargs:
                sys.stderr.write(
                    "WARN: KNumInt.nr_rks function finds keyword "
                    'argument "kpt" and converts it to "kpts"\n'
                )
                kpts = kwargs["kpt"]
            else:
                kpts = self.kpts
        kpts = kpts.reshape(-1, 3)

        return nr_rks(
            self,
            cell,
            grids,
            xc_code,
            dms,
            0,
            0,
            hermi,
            kpts,
            kpts_band,
            max_memory,
            verbose,
        )

    @lib.with_doc(nr_uks.__doc__)
    def nr_uks(
        self,
        cell,
        grids,
        xc_code,
        dms,
        relativity=0,
        hermi=1,
        kpts=None,
        kpts_band=None,
        max_memory=2000,
        verbose=None,
        **kwargs
    ):
        if kpts is None:
            if "kpt" in kwargs:
                sys.stderr.write(
                    "WARN: KNumInt.nr_uks function finds keyword "
                    'argument "kpt" and converts it to "kpts"\n'
                )
                kpts = kwargs["kpt"]
            else:
                kpts = self.kpts
        kpts = kpts.reshape(-1, 3)

        return nr_uks(
            self,
            cell,
            grids,
            xc_code,
            dms,
            1,
            0,
            hermi,
            kpts,
            kpts_band,
            max_memory,
            verbose,
        )

    def method_not_implemented(self, *args, **kwargs):
        raise NotImplementedError

    def initialize_feature_generators_kpts(self, cell, grids, nspin, kpts):
        self.sl_plan = SemilocalPlan(self.settings.sl_settings, nspin)
        self.fl_plan = None
        cond = self.sdmxgen is None
        cond = cond or self.cell != cell
        cond = cond or self.sdmxgen.plan.nspin != nspin
        cond = cond and self.settings.has_sdmx
        if cond:
            self.sdmxgen = self.sdmx_init.initialize_sdmx_generator(cell, nspin, kpts)
        self.cell = cell

    nr_rks_fxc = method_not_implemented
    nr_uks_fxc = method_not_implemented
    nr_rks_fxc_st = method_not_implemented
    cache_xc_kernel = method_not_implemented
    cache_xc_kernel1 = method_not_implemented
