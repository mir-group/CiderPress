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
from pyscf import lib
from pyscf.df.grad import uks as uks_grad_df
from pyscf.dft import gen_grid, numint
from pyscf.grad import rks as rks_grad
from pyscf.grad import uks as uks_grad
from pyscf.grad.rks import grids_noresponse_cc, grids_response_cc
from pyscf.lib import logger


def get_veff(ks_grad, mol=None, dm=None):
    """
    First order derivative of DFT effective potential matrix (wrt electron coordinates)

    Args:
        ks_grad : grad.uhf.Gradients or grad.uks.Gradients object
    """
    if mol is None:
        mol = ks_grad.mol
    if dm is None:
        dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = rks_grad._initialize_grids(ks_grad)

    ni = mf._numint
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
            enlc, vnlc = rks_grad.get_nlc_vxc_full_response(
                ni,
                mol,
                nlcgrids,
                xc,
                dm[0] + dm[1],
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
            enlc, vnlc = rks_grad.get_nlc_vxc(
                ni,
                mol,
                nlcgrids,
                xc,
                dm[0] + dm[1],
                max_memory=max_memory,
                verbose=ks_grad.verbose,
            )
            vxc += vnlc
    t0 = logger.timer(ks_grad, "vxc", *t0)

    has_df = hasattr(mf, "with_df") and mf.with_df is not None
    if not has_df:
        # no density fitting
        if not ni.libxc.is_hybrid_xc(mf.xc):
            vj = ks_grad.get_j(mol, dm)
            vxc += vj[0] + vj[1]
        else:
            raise NotImplementedError("Hybrid DFT with CIDER functionals")

        return lib.tag_array(vxc, exc1_grid=exc)
    else:
        # density fitting is used
        if not ni.libxc.is_hybrid_xc(mf.xc):
            vj = ks_grad.get_j(mol, dm)
            vxc += vj[0] + vj[1]
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
    ni.timer.start("nr_uks")
    if relativity != 0:
        raise NotImplementedError

    if not ni.settings.sdmx_settings.is_empty:
        raise NotImplementedError("SDMX forces")
    if not ni.settings.nlof_settings.is_empty:
        raise NotImplementedError("NLOF forces")

    xctype = ni.settings.sl_settings.level
    ni.initialize_feature_generators(mol, grids, 2)

    dma, dmb = numint._format_uks_dm(dms)
    assert dma.ndim == 2
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]
    ao_loc = mol.ao_loc_nr()
    extra_ao = 0
    ao_deriv = 2

    vmat = np.zeros((2, 3, nao, nao))
    for ao, mask, weight, coords in ni.block_loop(
        mol, grids, nao, ao_deriv, max_memory=max_memory, extra_ao=extra_ao
    ):
        rho_a = make_rhoa(0, ao, mask, xctype)
        rho_b = make_rhob(0, ao, mask, xctype)
        rho = (rho_a, rho_b)
        ni.timer.start("xc cider")
        exc, (vxc, vxc_nldf, vxc_sdmx) = ni.eval_xc_cider(
            xc_code, rho, None, None, deriv=1, xctype=xctype
        )[:2]
        ni.timer.stop("xc cider")
        assert vxc_nldf is None
        wv = weight * vxc
        wv[:, 0] *= 0.5
        rks_grad._gga_grad_sum_(vmat[0], mol, ao, wv[0], mask, ao_loc)
        rks_grad._gga_grad_sum_(vmat[1], mol, ao, wv[1], mask, ao_loc)
        if xctype == "MGGA":
            wv[:, 4] *= 0.5
            rks_grad._tau_grad_dot_(vmat[0], mol, ao, wv[0, 4], mask, ao_loc, True)
            rks_grad._tau_grad_dot_(vmat[1], mol, ao, wv[1, 4], mask, ao_loc, True)

    exc = np.zeros((mol.natm, 3))
    # - sign because nabla_X = -nabla_x
    return exc, -vmat


def get_vxc_nldf(
    ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
):
    ni.timer.start("nr_uks")
    if relativity != 0:
        raise NotImplementedError

    if not ni.settings.sdmx_settings.is_empty:
        raise NotImplementedError("SDMX forces")
    if not ni.settings.nlof_settings.is_empty:
        raise NotImplementedError("NLOF forces")

    xctype = ni.settings.sl_settings.level
    ni.initialize_feature_generators(mol, grids, 2)

    dma, dmb = numint._format_uks_dm(dms)
    assert dma.ndim == 2
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]
    assert nset == 1
    ao_loc = mol.ao_loc_nr()
    nrho = 5 if xctype == "MGGA" else 4
    if not ni.settings.nlof_settings.is_empty:
        nrho += ni.settings.nlof_settings.nrho
    rhoa_full = np.zeros((nrho, grids.weights.size), dtype=np.float64, order="C")
    rhob_full = np.zeros((nrho, grids.weights.size), dtype=np.float64, order="C")
    wva_full = np.zeros((nrho, grids.weights.size), dtype=np.float64, order="C")
    wvb_full = np.zeros((nrho, grids.weights.size), dtype=np.float64, order="C")

    num_nldf = ni.settings.nldf_settings.nfeat
    vxc_nldf_full = np.zeros(
        (2, num_nldf, grids.weights.size), dtype=np.float64, order="C"
    )

    def block_loop(ao_deriv):
        ip0 = 0
        for ao, mask, weight, coords in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory=max_memory
        ):
            ip1 = ip0 + weight.size
            yield ip0, ip1, ao, mask, weight, coords
            ip0 = ip1

    if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
        raise NotImplementedError("laplacian in meta-GGA method")
    ao_deriv = 1
    for ip0, ip1, ao, mask, weight, coords in block_loop(ao_deriv):
        rhoa_full[:, ip0:ip1] = make_rhoa(0, ao, mask, xctype)
        rhob_full[:, ip0:ip1] = make_rhob(0, ao, mask, xctype)

    par_atom = False
    extra_ao = ni.nldfgen.get_extra_ao()
    if par_atom:
        raise NotImplementedError
    else:
        nldf_feat = np.stack(
            [
                ni.nldfgen.get_features(rhoa_full, spin=0),
                ni.nldfgen.get_features(rhob_full, spin=1),
            ]
        )
        ip0 = 0
        for mask, weight, coords in ni.extra_block_loop(
            mol, grids, max_memory=max_memory, extra_ao=extra_ao
        ):
            ip1 = ip0 + weight.size
            rho_a = rhoa_full[:, ip0:ip1]
            rho_b = rhob_full[:, ip0:ip1]
            rho = (rho_a, rho_b)
            exc, (vxc, vxc_nldf, vxc_sdmx) = ni.eval_xc_cider(
                xc_code,
                rho,
                nldf_feat[..., ip0:ip1],
                None,
                deriv=1,
                xctype=xctype,
            )[:2]
            vxc_nldf_full[..., ip0:ip1] = vxc_nldf * weight
            wva_full[:, ip0:ip1] = weight * vxc[0]
            wvb_full[:, ip0:ip1] = weight * vxc[1]
            ip0 = ip1
        wva_full[:, :] += ni.nldfgen.get_potential(vxc_nldf_full[0], spin=0)
        wvb_full[:, :] += ni.nldfgen.get_potential(vxc_nldf_full[1], spin=1)

    ao_deriv = 2
    vmat = np.zeros((2, 3, nao, nao))
    for ip0, ip1, ao, mask, weight, coords in block_loop(ao_deriv):
        rho_a = np.ascontiguousarray(rhoa_full[..., ip0:ip1])
        rho_b = np.ascontiguousarray(rhob_full[..., ip0:ip1])
        rho = (rho_a, rho_b)
        wva = np.ascontiguousarray(wva_full[..., ip0:ip1])
        wvb = np.ascontiguousarray(wvb_full[..., ip0:ip1])
        wva[0] *= 0.5
        wvb[0] *= 0.5
        rks_grad._gga_grad_sum_(vmat[0], mol, ao, wva, mask, ao_loc)
        rks_grad._gga_grad_sum_(vmat[1], mol, ao, wvb, mask, ao_loc)
        if xctype == "MGGA":
            wva[4] *= 0.5
            wvb[4] *= 0.5
            rks_grad._tau_grad_dot_(vmat[0], mol, ao, wva[4], mask, ao_loc, True)
            rks_grad._tau_grad_dot_(vmat[1], mol, ao, wvb[4], mask, ao_loc, True)

    exc = np.zeros((mol.natm, 3))
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
    ni.timer.start("nr_uks")
    if relativity != 0:
        raise NotImplementedError

    if not ni.settings.sdmx_settings.is_empty:
        raise NotImplementedError("SDMX forces")
    if not ni.settings.nlof_settings.is_empty:
        raise NotImplementedError("NLOF forces")

    xctype = ni.settings.sl_settings.level
    ni.initialize_feature_generators(mol, grids, 2)

    dma, dmb = numint._format_uks_dm(dms)
    assert dma.ndim == 2
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]
    ao_loc = mol.ao_loc_nr()
    ao_deriv = 2
    excsum = 0

    vmat = np.zeros((2, 3, nao, nao))
    for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
        mask = gen_grid.make_mask(mol, coords)
        ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask, cutoff=grids.cutoff)
        rho_a = make_rhoa(0, ao, mask, xctype)
        rho_b = make_rhob(0, ao, mask, xctype)
        rho = (rho_a, rho_b)
        ni.timer.start("xc cider")
        exc, (vxc, vxc_nldf, vxc_sdmx) = ni.eval_xc_cider(
            xc_code, rho, None, None, deriv=1, xctype=xctype
        )[:2]
        ni.timer.stop("xc cider")
        assert vxc_nldf is None
        wv = weight * vxc
        wv[:, 0] *= 0.5
        if xctype == "MGGA":
            wv[:, 4] *= 0.5

        vtmp = np.zeros((3, nao, nao))
        rks_grad._gga_grad_sum_(vtmp, mol, ao, wv[0], mask, ao_loc)
        if xctype == "MGGA":
            rks_grad._tau_grad_dot_(vtmp, mol, ao, wv[0, 4], mask, ao_loc, True)
        vmat[0] += vtmp
        excsum += np.einsum("r,r,nxr->nx", exc, rho_a[0] + rho_b[0], weight1)
        excsum[atm_id] += np.einsum("xij,ji->x", vtmp, dms[0]) * 2

        vtmp = np.zeros((3, nao, nao))
        rks_grad._gga_grad_sum_(vtmp, mol, ao, wv[1], mask, ao_loc)
        if xctype == "MGGA":
            rks_grad._tau_grad_dot_(vtmp, mol, ao, wv[1, 4], mask, ao_loc, True)
        vmat[1] += vtmp
        excsum[atm_id] += np.einsum("xij,ji->x", vtmp, dms[1]) * 2

    # - sign because nabla_X = -nabla_x
    return excsum, -vmat


def get_vxc_nldf_full_response(
    ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None
):
    ni.timer.start("nr_uks")
    if relativity != 0:
        raise NotImplementedError

    if not ni.settings.sdmx_settings.is_empty:
        raise NotImplementedError("SDMX forces")
    if not ni.settings.nlof_settings.is_empty:
        raise NotImplementedError("NLOF forces")

    xctype = ni.settings.sl_settings.level
    ni.initialize_feature_generators(mol, grids, 2)

    dma, dmb = numint._format_uks_dm(dms)
    assert dma.ndim == 2
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]
    ao_loc = mol.ao_loc_nr()
    ao_deriv = 2
    excsum = 0

    vmat = np.zeros((2, 3, nao, nao))
    nrho = 5 if xctype == "MGGA" else 4
    if not ni.settings.nlof_settings.is_empty:
        raise NotImplementedError

    num_nldf = ni.settings.nldf_settings.nfeat
    vxc_nldf_full = np.zeros(
        (2, num_nldf, grids.weights.size), dtype=np.float64, order="C"
    )

    ga_loc = ni.nldfgen.interpolator.grid_loc_atom
    assert ga_loc is not None
    rho_full = np.zeros((2, nrho, ga_loc[-1]), dtype=np.float64, order="C")
    wv_full = np.zeros((2, nrho, ga_loc[-1]), dtype=np.float64, order="C")
    all_coords = np.zeros((ga_loc[-1], 3), dtype=np.float64, order="C")
    exc = np.zeros(ga_loc[-1], dtype=np.float64, order="C")

    ao_deriv = 1
    for atm_id, (coords, weight) in enumerate(grids_noresponse_cc(grids)):
        ip0, ip1 = ga_loc[atm_id : atm_id + 2]
        mask = gen_grid.make_mask(mol, coords)
        ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask, cutoff=grids.cutoff)
        rho_full[0, :, ip0:ip1] = make_rhoa(0, ao, mask, xctype)
        rho_full[1, :, ip0:ip1] = make_rhob(0, ao, mask, xctype)
        all_coords[ip0:ip1] = coords

    par_atom = False
    extra_ao = ni.nldfgen.get_extra_ao()
    if ni.has_sdmx:
        raise NotImplementedError
    if par_atom:
        raise NotImplementedError
    else:
        nldf_feat = np.stack(
            [
                ni.nldfgen.get_features(
                    rho_full[0], map_grids=False, grad_mode=True, spin=0
                ),
                ni.nldfgen.get_features(
                    rho_full[1], map_grids=False, grad_mode=True, spin=1
                ),
            ]
        )
        ip0 = 0
        for mask, weight, coords in ni.extra_block_loop(
            mol, grids, max_memory=max_memory, extra_ao=extra_ao
        ):
            ip1 = ip0 + weight.size
            ip1 = min(ip1, grids.grids_indexer.idx_map.size)
            weight = weight[: ip1 - ip0]
            inds = grids.grids_indexer.idx_map[ip0:ip1]
            rho = (rho_full[0][..., inds], rho_full[1][..., inds])
            exc[inds], (vxc, vxc_nldf, vxc_sdmx) = ni.eval_xc_cider(
                xc_code,
                rho,
                np.ascontiguousarray(nldf_feat[..., ip0:ip1]),
                None,
                deriv=1,
                xctype=xctype,
            )[:2]
            vxc_nldf_full[..., ip0:ip1] = vxc_nldf * weight
            wv_full[..., inds] = weight * vxc
            ip0 = ip1
        wvtmp, cidergg_g, excsum = ni.nldfgen.get_potential(
            vxc_nldf_full[0], map_grids=False, grad_mode=True, spin=0
        )
        wv_full[0] += wvtmp
        wvtmp, _cidergg_g, _excsum = ni.nldfgen.get_potential(
            vxc_nldf_full[1], map_grids=False, grad_mode=True, spin=1
        )
        cidergg_g += _cidergg_g
        excsum += _excsum
        wv_full[1] += wvtmp

    ao_deriv = 2
    for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grids)):
        mask = gen_grid.make_mask(mol, coords)
        ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask, cutoff=grids.cutoff)
        ip0, ip1 = ga_loc[atm_id : atm_id + 2]

        mask = gen_grid.make_mask(mol, coords)
        ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask, cutoff=grids.cutoff)
        rho_a = np.ascontiguousarray(rho_full[0, :, ip0:ip1])
        rho_b = np.ascontiguousarray(rho_full[1, :, ip0:ip1])
        wv = np.ascontiguousarray(wv_full[..., ip0:ip1])
        wv[:, 0] *= 0.5
        if xctype == "MGGA":
            wv[:, 4] *= 0.5

        vtmp = np.zeros((3, nao, nao))
        rks_grad._gga_grad_sum_(vtmp, mol, ao, wv[0], mask, ao_loc)
        if xctype == "MGGA":
            rks_grad._tau_grad_dot_(vtmp, mol, ao, wv[0, 4], mask, ao_loc, True)
        vmat[0] += vtmp
        excsum += np.einsum("r,r,nxr->nx", exc[ip0:ip1], rho_a[0] + rho_b[0], weight1)
        excsum[atm_id] += np.einsum("xij,ji->x", vtmp, dms[0]) * 2

        vtmp = np.zeros((3, nao, nao))
        rks_grad._gga_grad_sum_(vtmp, mol, ao, wv[1], mask, ao_loc)
        if xctype == "MGGA":
            rks_grad._tau_grad_dot_(vtmp, mol, ao, wv[1, 4], mask, ao_loc, True)
        vmat[1] += vtmp
        excsum[atm_id] += np.einsum("xij,ji->x", vtmp, dms[1]) * 2

        excsum += np.dot(weight1, cidergg_g[ip0:ip1])
        ip0 = ip1
    # - sign because nabla_X = -nabla_x
    return excsum, -vmat


class Gradients(uks_grad.Gradients):

    get_veff = get_veff


class DFGradients(uks_grad_df.Gradients):

    get_veff = get_veff
