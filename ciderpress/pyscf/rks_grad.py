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
import pyscf.df.grad.rks as rks_grad_df
from pyscf import lib
from pyscf.dft import gen_grid
from pyscf.grad import rks as rks_grad
from pyscf.grad.rks import (
    _gga_grad_sum_,
    _tau_grad_dot_,
    get_nlc_vxc,
    get_nlc_vxc_full_response,
    grids_noresponse_cc,
    grids_response_cc,
)
from pyscf.lib import logger


def get_veff(ks_grad, mol=None, dm=None):
    if mol is None:
        mol = ks_grad.mol
    if dm is None:
        dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = rks_grad._initialize_grids(ks_grad)

    # NOTE pyscf has hyb params here, shouldn't be needed for now

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory * 0.9 - mem_now)
    if ks_grad.grid_response:
        exc, vxc = get_vxc_full_response(
            ni, mol, grids, mf.xc, dm, max_memory=max_memory, verbose=ks_grad.verbose
        )
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = get_nlc_vxc_full_response(
                ni,
                mol,
                nlcgrids,
                xc,
                dm,
                max_memory=max_memory,
                verbose=ks_grad.verbose,
            )
            exc += enlc
            vxc += vnlc
        logger.debug1(ks_grad, "sum(grids response) %s", exc.sum(axis=0))
    else:
        exc, vxc = get_vxc(
            ni, mol, grids, mf.xc, dm, max_memory=max_memory, verbose=ks_grad.verbose
        )
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = get_nlc_vxc(
                ni,
                mol,
                nlcgrids,
                xc,
                dm,
                max_memory=max_memory,
                verbose=ks_grad.verbose,
            )
            vxc += vnlc
    t0 = logger.timer(ks_grad, "vxc", *t0)

    has_df = hasattr(mf, "with_df") and mf.with_df is not None
    if not has_df:
        if not ni.libxc.is_hybrid_xc(mf.xc):
            vj = ks_grad.get_j(mol, dm)
            vxc += vj
        else:
            raise NotImplementedError("Hybrid DFT with CIDER functionals")
    else:
        if not ni.libxc.is_hybrid_xc(mf.xc):
            vj = ks_grad.get_j(mol, dm)
            vxc += vj
            if ks_grad.auxbasis_response:
                e1_aux = vj.aux.sum((0, 1))
        else:
            raise NotImplementedError("Hybrid DFT with CIDER functionals")

        if ks_grad.auxbasis_response:
            logger.debug1(ks_grad, "sum(auxbasis response) %s", e1_aux.sum(axis=0))
            vxc = lib.tag_array(vxc, exc1_grid=exc, aux=e1_aux)
        else:
            vxc = lib.tag_array(vxc, exc1_grid=exc)
        return vxc

    return lib.tag_array(vxc, exc1_grid=exc)


def get_vxc(
    ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
):
    if not ni.settings.nldf_settings.is_empty:
        return get_vxc_nldf(
            ni,
            mol,
            grids,
            xc_code,
            dms,
            relativity=relativity,
            hermi=hermi,
            max_memory=max_memory,
            verbose=verbose,
        )
    ni.timer.start("nr_rks")
    if relativity != 0:
        raise NotImplementedError

    xctype = ni.settings.sl_settings.level
    ni.initialize_feature_generators(mol, grids, 1)

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()

    if dms.ndim == 2:
        dms = dms[None, :, :]

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)

    ao_deriv = 2
    vmat = np.zeros((nset, 3, nao, nao))

    def block_loop(ao_deriv):
        if ni.has_sdmx:
            raise NotImplementedError
        else:
            extra_ao = 0
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory=max_memory, extra_ao=extra_ao
        ):
            for i in range(nset):
                rho = make_rho(i, ao, mask, xctype)
                ni.timer.start("xc cider")
                exc, (vxc, vxc_nldf, vxc_sdmx) = ni.eval_xc_cider(
                    xc_code, rho, None, None, deriv=1, xctype=xctype
                )[:2]
                ni.timer.stop("xc cider")
                assert vxc_nldf is None
                den = rho[0] * weight
                nelec[i] += den.sum()
                excsum[i] += np.dot(den, exc)
                wv = weight * vxc
                yield i, ao, mask, wv

    for idm, ao, mask, wv in block_loop(ao_deriv):
        wv[0] *= 0.5
        _gga_grad_sum_(vmat[idm], mol, ao, wv, mask, ao_loc)
        if xctype == "MGGA":
            wv[4] *= 0.5
            _tau_grad_dot_(vmat[idm], mol, ao, wv[4], mask, ao_loc, True)

    exc = None
    if nset == 1:
        vmat = vmat[0]
    # - sign because nabla_X = -nabla_x
    return exc, -vmat


def get_vxc_nldf(
    ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
):
    if not hasattr(grids, "grids_indexer"):
        raise ValueError("Grids object must have indexer for NLDF evaluation")

    if relativity != 0:
        raise NotImplementedError

    xctype = ni.settings.sl_settings.level
    ni.initialize_feature_generators(mol, grids, 1)
    assert ni.has_nldf

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()

    if dms.ndim == 2:
        dms = dms[None, :, :]

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    ao_deriv = 2
    vmat = np.zeros((nset, 3, nao, nao))
    nrho = 5 if xctype == "MGGA" else 4
    if not ni.settings.nlof_settings.is_empty:
        raise NotImplementedError
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
    for i, ip0, ip1, ao, mask, weight, coords in block_loop(ao_deriv):
        rho_full[i, :, ip0:ip1] = make_rho(i, ao, mask, xctype)

    par_atom = False
    extra_ao = ni.nldfgen.get_extra_ao()
    if ni.has_sdmx:
        raise NotImplementedError
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
            for idm in range(nset):
                rho = rho_full[idm, :, ip0:ip1]
                sdmx_feat = None
                exc, (vxc, vxc_nldf, vxc_sdmx) = ni.eval_xc_cider(
                    xc_code,
                    np.ascontiguousarray(rho),
                    np.ascontiguousarray(nldf_feat[idm][:, ip0:ip1]),
                    sdmx_feat,
                    deriv=1,
                    xctype=xctype,
                )[:2]
                vxc_nldf_full[idm, :, ip0:ip1] = vxc_nldf * weight
                den = rho[0] * weight
                nelec[idm] += den.sum()
                excsum[idm] += np.dot(den, exc)
                wv_full[idm, :, ip0:ip1] = weight * vxc
            ip0 = ip1
        for idm in range(nset):
            wv_full[idm, :, :] += ni.nldfgen.get_potential(vxc_nldf_full[idm])

    for i, ip0, ip1, ao, mask, weight, coords in block_loop(ao_deriv):
        for idm in range(nset):
            rho = np.ascontiguousarray(rho_full[idm, :, ip0:ip1])
            wv = np.ascontiguousarray(wv_full[idm][:, ip0:ip1])
            wv[0] *= 0.5
            _gga_grad_sum_(vmat[idm], mol, ao, wv[:4], mask, ao_loc)
            if xctype == "MGGA":
                wv[4] *= 0.5
                _tau_grad_dot_(vmat[idm], mol, ao, wv[4], mask, ao_loc, True)

    exc = None
    if nset == 1:
        vmat = vmat[0]
    # - sign because nabla_X = -nabla_x
    return exc, -vmat


def get_vxc_full_response(
    ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
):
    if not ni.settings.nldf_settings.is_empty:
        return get_vxc_nldf_full_response(
            ni,
            mol,
            grids,
            xc_code,
            dms,
            relativity=relativity,
            hermi=hermi,
            max_memory=max_memory,
            verbose=verbose,
        )

    if not ni.settings.sdmx_settings.is_empty:
        raise NotImplementedError
    if not ni.settings.nlof_settings.is_empty:
        raise NotImplementedError
    xctype = ni.settings.sl_settings.level
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()

    excsum = np.zeros((mol.natm, 3))
    vmat = np.zeros((3, nao, nao))

    ao_deriv = 2
    for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
        mask = gen_grid.make_mask(mol, coords)
        ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask, cutoff=grids.cutoff)
        rho = make_rho(0, ao[:10], mask, xctype)
        exc, (vxc, vxc_nldf, vxc_sdmx) = ni.eval_xc_cider(
            xc_code,
            rho,
            None,
            None,
            deriv=1,
            xctype=xctype,
        )[:2]
        wv = weight * vxc
        wv[0] *= 0.5
        vtmp = np.zeros((3, nao, nao))
        _gga_grad_sum_(vtmp, mol, ao, wv, mask, ao_loc)
        if xctype == "MGGA":
            wv[4] *= 0.5  # for the factor 1/2 in tau
            _tau_grad_dot_(vtmp, mol, ao, wv[4], mask, ao_loc, True)
        vmat += vtmp
        # response of weights
        excsum += np.einsum("r,r,nxr->nx", exc, rho[0], weight1)
        # response of grids coordinates
        excsum[atm_id] += np.einsum("xij,ji->x", vtmp, dms) * 2
        rho = vxc = wv = None
    # - sign because nabla_X = -nabla_x
    return excsum, -vmat


def get_vxc_nldf_full_response(
    ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
):
    if not hasattr(grids, "grids_indexer"):
        raise ValueError("Grids object must have indexer for NLDF evaluation")

    if relativity != 0:
        raise NotImplementedError

    xctype = ni.settings.sl_settings.level
    assert ni.has_nldf
    ni.initialize_feature_generators(mol, grids, 1)

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    if nset != 1:
        raise NotImplementedError
    ao_loc = mol.ao_loc_nr()

    assert dms.ndim == 2

    nelec = np.zeros(nset)
    vmat = np.zeros((3, nao, nao))
    nrho = 5 if xctype == "MGGA" else 4
    if not ni.settings.nlof_settings.is_empty:
        raise NotImplementedError

    num_nldf = ni.settings.nldf_settings.nfeat
    vxc_nldf_full = np.zeros(
        (nset, num_nldf, grids.weights.size), dtype=np.float64, order="C"
    )

    ga_loc = ni.nldfgen.interpolator.grid_loc_atom
    assert ga_loc is not None
    rho_full = np.zeros((nset, nrho, ga_loc[-1]), dtype=np.float64, order="C")
    wv_full = np.zeros((nset, nrho, ga_loc[-1]), dtype=np.float64, order="C")
    exc = np.zeros(ga_loc[-1], dtype=np.float64, order="C")
    all_coords = np.zeros((ga_loc[-1], 3), dtype=np.float64, order="C")

    ao_deriv = 1
    for atm_id, (coords, weight) in enumerate(grids_noresponse_cc(grids)):
        ip0, ip1 = ga_loc[atm_id : atm_id + 2]
        mask = gen_grid.make_mask(mol, coords)
        ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask, cutoff=grids.cutoff)
        rho_full[0, :, ip0:ip1] = make_rho(0, ao, mask, xctype)
        all_coords[ip0:ip1] = coords

    par_atom = False
    extra_ao = ni.nldfgen.get_extra_ao()
    if ni.has_sdmx:
        raise NotImplementedError
    if par_atom:
        raise NotImplementedError
    else:
        nldf_feat = [
            ni.nldfgen.get_features(rho_full[0], map_grids=False, grad_mode=True)
        ]
        ip0 = 0
        for mask, weight, coords in ni.extra_block_loop(
            mol, grids, max_memory=max_memory, extra_ao=extra_ao
        ):
            ip1 = ip0 + weight.size
            ip1 = min(ip1, grids.grids_indexer.idx_map.size)
            weight = weight[: ip1 - ip0]
            inds = grids.grids_indexer.idx_map[ip0:ip1]
            rho = rho_full[0][..., inds]
            exc[inds], (vxc, vxc_nldf, vxc_sdmx) = ni.eval_xc_cider(
                xc_code,
                rho,
                nldf_feat[0][:, ip0:ip1],
                None,
                deriv=1,
                xctype=xctype,
            )[:2]
            vxc_nldf_full[0][:, ip0:ip1] = vxc_nldf * weight
            den = rho[0] * weight
            nelec[0] += den.sum()
            wv_full[0][:, inds] = weight * vxc
            ip0 = ip1
        wvtmp, cidergg_g, excsum = ni.nldfgen.get_potential(
            vxc_nldf_full[0], map_grids=False, grad_mode=True
        )
        # excsum[:] = 0
        wv_full[0] += wvtmp

    exc_ref = 0
    ao_deriv = 2
    for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
        mask = gen_grid.make_mask(mol, coords)
        ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask, cutoff=grids.cutoff)
        ip0, ip1 = ga_loc[atm_id : atm_id + 2]
        rho = np.ascontiguousarray(rho_full[0, :, ip0:ip1])
        wv = np.ascontiguousarray(wv_full[0][:, ip0:ip1])
        vtmp = np.zeros((3, nao, nao))
        wv[0] *= 0.5
        _gga_grad_sum_(vtmp, mol, ao, wv[:4], mask, ao_loc)
        if xctype == "MGGA":
            wv[4] *= 0.5
            _tau_grad_dot_(vtmp, mol, ao, wv[4], mask, ao_loc, True)
        vmat += vtmp

        # response of weights
        excsum += np.einsum("r,r,nxr->nx", exc[ip0:ip1], rho[0], weight1)
        # response of grids coordinates
        excsum[atm_id] += np.einsum("xij,ji->x", vtmp, dms) * 2
        excsum += np.dot(weight1, cidergg_g[ip0:ip1])
        exc_ref += np.sum(exc[ip0:ip1] * rho[0] * weight)
        ip0 = ip1
    # - sign because nabla_X = -nabla_x
    return excsum, -vmat


class Gradients(rks_grad.Gradients):

    get_veff = get_veff


class DFGradients(rks_grad_df.Gradients):

    get_veff = get_veff
