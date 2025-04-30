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
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor
from gpaw.sphere.lebedev import R_nv, Y_nL, weight_n
from gpaw.utilities.timing import Timer
from gpaw.xc.pawcorrection import rnablaY_nLv
from gpaw.xc.vdw import spline
from scipy.linalg import cho_factor, cho_solve, cholesky

from ciderpress.data import get_hgbs_max_exps
from ciderpress.dft import pwutil
from ciderpress.dft.density_util import get_sigma
from ciderpress.dft.grids_indexer import AtomicGridsIndexer
from ciderpress.dft.lcao_convolutions import (
    ANG_OF,
    PTR_EXP,
    ATCBasis,
    get_etb_from_expnt_range,
    get_gamma_lists_from_etb_list,
)
from ciderpress.dft.plans import NLDFAuxiliaryPlan, libcider
from ciderpress.gpaw.fit_paw_gauss_pot import get_dv, get_ffunc, get_pfunc_norm
from ciderpress.gpaw.gpaw_grids import (
    SBTFullGridDescriptor,
    make_radial_derivative_calculator,
)
from ciderpress.gpaw.interp_paw import (
    DiffPAWXCCorrection,
    calculate_cider_paw_correction,
)


def get_ag_indices(fft_obj, gd, shape, spos_c, rmax, buffer=0, get_global_disps=False):
    center = np.round(spos_c * shape).astype(int)
    disp = np.empty(3, dtype=int)
    lattice = gd.cell_cv
    vol = np.abs(np.linalg.det(lattice))
    for i in range(3):
        res = np.cross(lattice[(i + 1) % 3], lattice[(i + 2) % 3])
        disp[i] = np.ceil(np.linalg.norm(res) * rmax / vol * shape[i]) + 1 + buffer
    indices = [
        np.arange(center[i] - disp[i], center[i] + disp[i] + 1) for i in range(3)
    ]
    fdisps = []
    for i in range(3):
        fdisps.append(indices[i].astype(np.float64) / shape[i] - spos_c[i])
    indices = [ind % s for ind, s in zip(indices, shape)]
    if get_global_disps:
        return indices, fdisps
    lbound_inds, ubound_inds = fft_obj.get_bound_inds()
    conds = []
    for i in range(3):
        conds.append(
            np.logical_and(indices[i] >= lbound_inds[i], indices[i] < ubound_inds[i])
        )
    local_indices = [indices[i][conds[i]] - lbound_inds[i] for i in range(3)]
    local_fdisps = [fdisps[i][conds[i]] for i in range(3)]
    return local_indices, local_fdisps


def get_convolution_data(settings, alphas, alpha_norms, betas, beta_norms):
    """
    Prefactors and (reciprocal-space) exponents for vi, vj, vij convolutions
    """
    if "j" in settings.version:
        expnts = betas[:, None] + alphas
        expnts = 1.0 / (4 * expnts)
        facs = beta_norms[:, None] * alpha_norms
        facs[:] *= (4 * np.pi * expnts) ** 1.5
        k2pows = [0] * len(alphas)
    else:
        expnts = np.empty((0, alphas.size))
        facs = np.empty((0, alphas.size))
        k2pows = []
    if settings.version != "j":
        vi_facs = []
        vi_expnts = []
        viexp = 1.0 / (4 * alphas)
        vifac = alpha_norms * (4 * np.pi * viexp) ** 1.5
        for spec in settings.l0_feat_specs:
            vi_expnts.append(viexp)
            if spec == "se":
                vi_facs.append(vifac)
                k2pows.append(0)
            elif spec == "se_ap":
                vi_facs.append(vifac * alphas)
                k2pows.append(0)
            elif spec == "se_lapl":
                vi_facs.append(vifac)
                k2pows.append(1)
            else:
                raise NotImplementedError
        for spec in settings.l1_feat_specs:
            vi_expnts.append(viexp)
            if spec == "se_grad":
                vi_facs.append(vifac)
                k2pows.append(1)
            else:
                raise NotImplementedError
        expnts = np.append(expnts, vi_expnts, axis=0)
        facs = np.append(facs, vi_facs, axis=0)
    k2pows = np.asarray(k2pows, order="C", dtype=np.int32)
    return expnts, facs, k2pows


PSETUP_LIST1 = ([0, 2, 1, 2], [0, 0, 1, 2])
PSETUP_LIST2 = ([0, 2, 4, 1, 3, 2, 4, 3, 4], [0, 0, 0, 1, 1, 2, 2, 3, 4])
PSETUP_LIST3 = (
    [0, 2, 4, 6, 1, 3, 5, 2, 4, 6, 3, 5, 4, 6],
    [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
)
PSETUP_LIST4 = (
    [0, 2, 4, 6, 8, 1, 3, 5, 7, 2, 4, 6, 8, 3, 5, 7, 4, 6, 8],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4],
)


def get_psetup_func_counts(Z, big=False):
    if Z > 36:
        if big:
            return PSETUP_LIST4
        else:
            return PSETUP_LIST3
    elif Z > 18:
        if big:
            return PSETUP_LIST3
        else:
            return PSETUP_LIST2
    elif Z > 2:
        if big:
            return PSETUP_LIST3
        else:
            return PSETUP_LIST2
    else:
        if big:
            return PSETUP_LIST2
        else:
            return PSETUP_LIST1


class SBTGridContainer:
    def __init__(
        self,
        rgd,
        big_rgd,
    ):
        self.rgd = rgd
        self.big_rgd = big_rgd

    @classmethod
    def from_setup(cls, setup, rmin=1e-4, d=0.03, encut=1e6, N=None):
        rmax = setup.rgd.r_g[-1]
        if N is not None:
            assert isinstance(N, int)
            d = np.log(rmax / rmin) / (N - 1)
        else:
            N = np.ceil(np.log(rmax / rmin) / d) + 1
        xcc = setup.xc_correction
        lmax = int(np.sqrt(Y_nL.shape[1]) + 1e-8) - 1
        big_rgd = SBTFullGridDescriptor(rmin, encut, d, N=N, lmax=lmax)
        xrgd = xcc.rgd
        rmax = xrgd.r_g[-1] - 0.2  # TODO be more precise
        N = int(np.log(rmax / big_rgd.a) / big_rgd.d) + 1
        ergd = big_rgd.new(N)

        return cls(ergd, big_rgd)


class FastAtomPASDWSlice:
    def __init__(
        self,
        indset,
        g,
        dg,
        rad_g,
        rhat_gv,
        rcut,
        psetup,
        dv,
        h,
        ovlp_fit=False,
        store_funcs=False,
    ):
        self.t_g = np.ascontiguousarray(g.astype(np.int32))
        self.dt_g = np.ascontiguousarray(dg.astype(np.float64))
        self.rad_g = rad_g
        self.h = h
        self.rhat_gv = rhat_gv
        self.rcut = rcut
        self.psetup = psetup
        self.dv = dv
        if ovlp_fit:
            self.setup_ovlpt()
        else:
            self.sinv_pf = None
        self.store_funcs = store_funcs
        self._funcs_gi = None
        if self.store_funcs:
            self._funcs_gi = self.get_funcs()
        self.indset = np.ascontiguousarray(indset.astype(np.int64))

    @classmethod
    def from_gd_and_setup(
        cls,
        gd,
        spos_c,
        psetup,
        fft_obj,
        rmax=0,
        sphere=True,
        ovlp_fit=False,
        store_funcs=False,
        is_global=False,
    ):
        if ovlp_fit and not is_global:
            raise ValueError("Need global grid set for overlap fitting")
        rgd = psetup.interp_rgd
        # Make sure the grid is linear, as this is assumed
        # by the calculation of dg below and by the get_grads function.
        assert isinstance(rgd, EquidistantRadialGridDescriptor)
        if rmax == 0:
            rmax = psetup.rcut
        shape = gd.get_size_of_global_array()
        indices, fdisps = get_ag_indices(
            fft_obj, gd, shape, spos_c, rmax, buffer=0, get_global_disps=is_global
        )
        # NOTE: If is_global, indset is incorrect and will cause problems.
        # When is_global is True, this object should only be used for calculating
        # the overlap fitting, not for going between atoms and FFT grids
        indset, rad_g, rhat_gv = fft_obj.get_radial_info_for_atom(indices, fdisps)
        if sphere:
            cond = rad_g <= rmax
            indset = indset[cond]
            rad_g = rad_g[cond]
            rhat_gv = rhat_gv[cond]

        h = rgd.r_g[1] - rgd.r_g[0]
        dg = rad_g / h
        g = np.floor(dg).astype(np.int32)
        g = np.minimum(g, rgd.r_g.size - 1)
        dg -= g

        return cls(
            indset,
            g,
            dg,
            rad_g,
            rhat_gv,
            rmax,
            psetup,
            gd.dv,
            h,
            ovlp_fit=ovlp_fit,
            store_funcs=store_funcs,
        )

    def get_funcs(self, pfunc=True):
        if (self.store_funcs) and (self._funcs_gi is not None) and (pfunc):
            return self._funcs_gi

        if pfunc:
            funcs_xtp = self.psetup.pfuncs_ntp
            xlist_i = self.psetup.nlist_i
        else:
            funcs_xtp = self.psetup.ffuncs_jtp
            xlist_i = self.psetup.jlist_i

        radfuncs_ng = pwutil.eval_cubic_spline(
            funcs_xtp,
            self.t_g,
            self.dt_g,
        )
        lmax = self.psetup.lmax
        Lmax = (lmax + 1) * (lmax + 1)
        ylm = pwutil.recursive_sph_harm(Lmax, self.rhat_gv)
        funcs_ig = pwutil.eval_pasdw_funcs(
            radfuncs_ng,
            np.ascontiguousarray(ylm.T),
            xlist_i,
            self.psetup.lmlist_i,
        )
        return funcs_ig

    def get_grads(self, pfunc=True):
        if pfunc:
            funcs_xtp = self.psetup.pfuncs_ntp
            xlist_i = self.psetup.nlist_i
        else:
            funcs_xtp = self.psetup.ffuncs_jtp
            xlist_i = self.psetup.jlist_i

        radfuncs_ng = pwutil.eval_cubic_spline(
            funcs_xtp,
            self.t_g,
            self.dt_g,
        )
        radderivs_ng = (
            pwutil.eval_cubic_spline_deriv(
                funcs_xtp,
                self.t_g,
                self.dt_g,
            )
            / self.h
        )
        lmax = self.psetup.lmax
        Lmax = (lmax + 1) * (lmax + 1)
        ylm, dylm = pwutil.recursive_sph_harm_deriv(Lmax, self.rhat_gv)
        dylm /= self.rad_g[:, None, None] + 1e-8  # TODO right amount of regularization?
        rodylm = np.ascontiguousarray(np.einsum("gv,gvL->Lg", self.rhat_gv, dylm))
        funcs_ig = pwutil.eval_pasdw_funcs(
            radderivs_ng,
            np.ascontiguousarray(ylm.T),
            xlist_i,
            self.psetup.lmlist_i,
        )
        funcs_ig -= pwutil.eval_pasdw_funcs(
            radfuncs_ng,
            rodylm,
            xlist_i,
            self.psetup.lmlist_i,
        )
        funcs_vig = self.rhat_gv.T[:, None, :] * funcs_ig
        for v in range(3):
            funcs_vig[v] += pwutil.eval_pasdw_funcs(
                radfuncs_ng,
                np.ascontiguousarray(dylm[:, v].T),
                xlist_i,
                self.psetup.lmlist_i,
            )
        return funcs_vig

    def get_stress_funcs(self, pfunc=True):
        funcs_vig = self.get_grads(pfunc=pfunc)
        r_vg = self.rad_g * self.rhat_gv.T
        return r_vg[:, None, None, :] * funcs_vig

    @property
    def num_funcs(self):
        return len(self.psetup.nlist_i)

    @property
    def num_grids(self):
        return self.rad_g.size

    def setup_ovlpt(self):
        rad_pfuncs_ng = pwutil.eval_cubic_spline(
            self.psetup.pfuncs_ntp,
            self.t_g,
            self.dt_g,
        )
        rad_ffuncs_nj = pwutil.eval_cubic_spline(
            self.psetup.ffuncs_jtp,
            self.t_g,
            self.dt_g,
        )
        lmax = self.psetup.lmax
        Lmax = (lmax + 1) * (lmax + 1)
        ylm = pwutil.recursive_sph_harm(Lmax, self.rhat_gv)
        pfuncs_ig = pwutil.eval_pasdw_funcs(
            rad_pfuncs_ng,
            np.ascontiguousarray(ylm.T),
            self.psetup.nlist_i,
            self.psetup.lmlist_i,
        )
        ffuncs_ig = pwutil.eval_pasdw_funcs(
            rad_ffuncs_nj,
            np.ascontiguousarray(ylm.T),
            self.psetup.jlist_i,
            self.psetup.lmlist_i,
        )
        ovlp_pf = np.einsum("pg,fg->pf", pfuncs_ig, ffuncs_ig) * self.dv
        print(
            "OVLPS",
            np.linalg.eigvals(ovlp_pf.T),
            np.linalg.eigvals(self.psetup.exact_ovlp_pf),
        )
        self.sinv_pf = np.linalg.solve(ovlp_pf.T, self.psetup.exact_ovlp_pf)

    def get_ovlp_deriv(self, pfuncs_ig, pgrads_vig, stress=False):
        ffuncs_ig = self.get_funcs(False)
        fgrads_vig = self.get_grads(False)
        if stress:
            ovlp_pf = np.einsum("pg,fg->pf", pfuncs_ig, ffuncs_ig) * self.dv
            dr_vg = self.rad_g * self.rhat_gv.T
            dovlp_vvpf = np.einsum("ug,pg,vfg->uvpf", dr_vg, pfuncs_ig, fgrads_vig)
            dovlp_vvpf += np.einsum("ug,vpg,fg->uvpf", dr_vg, pgrads_vig, ffuncs_ig)
            dovlp_vvpf *= self.dv
            for v in range(3):
                dovlp_vvpf[v, v] += ovlp_pf
            B_vvfq = -1 * np.einsum("uvpf,pq->uvfq", dovlp_vvpf, self.sinv_pf)
            ni = ovlp_pf.shape[0]
            X_vvpq = np.empty((3, 3, ni, ni))
            for v1 in range(3):
                for v2 in range(3):
                    X_vvpq[v1, v2] = np.linalg.solve(ovlp_pf.T, B_vvfq[v1, v2])
            return X_vvpq
        else:
            ovlp_pf = np.einsum("pg,fg->pf", pfuncs_ig, ffuncs_ig) * self.dv
            dovlp_vpf = np.einsum("pg,vfg->vpf", pfuncs_ig, fgrads_vig) + np.einsum(
                "vpg,fg->vpf", pgrads_vig, ffuncs_ig
            )
            dovlp_vpf *= self.dv
            # d(sinv_pf)/dR = (ovlp_pf^T)^-1 * (d(ovlp_pf)/dR)^T * sinv_pf
            B_vfq = -1 * np.einsum("vpf,pq->vfq", dovlp_vpf, self.sinv_pf)
            ni = ovlp_pf.shape[0]
            X_vpq = np.empty((3, ni, ni))
            for v in range(3):
                X_vpq[v] = np.linalg.solve(ovlp_pf.T, B_vfq[v])
            return X_vpq


class _PAWCiderContribs:

    nspin: int
    grids_indexer: AtomicGridsIndexer
    plan: NLDFAuxiliaryPlan

    def __init__(self, plan, cider_kernel, atco, xcc, gplan):
        self.plan = plan
        self.cider_kernel = cider_kernel
        self._atco = atco
        self._ang_w_g = weight_n
        self._rad_w_g = xcc.rgd.dv_g
        self.w_g = (xcc.rgd.dv_g[:, None] * weight_n).ravel()
        self.r_g = xcc.rgd.r_g
        (
            self._get_rad_deriv,
            self._get_rad_deriv_bwd,
        ) = make_radial_derivative_calculator(self.r_g, order=4)
        self.xcc = xcc
        # TODO might want more control over grids_indexer nlm,
        # currently it is just controlled by size of Y_nL in GPAW.
        self.grids_indexer = AtomicGridsIndexer.make_single_atom_indexer(
            Y_nL, self.r_g, DY_nLv=rnablaY_nLv
        )
        self.grids_indexer.set_weights(self.w_g)
        self.timer = Timer()

    def _apply_ang_weight_(self, f_gq):
        assert f_gq.flags.c_contiguous
        f_gq.shape = (self._rad_w_g.size, self._ang_w_g.size, f_gq.shape[-1])
        f_gq[:] *= self._ang_w_g[:, None]
        f_gq.shape = (-1, f_gq.shape[-1])

    def _apply_rad_weight_(self, f_gq):
        assert f_gq.flags.c_contiguous
        f_gq.shape = (self._rad_w_g.size, self._ang_w_g.size, f_gq.shape[-1])
        f_gq[:] *= self._rad_w_g[:, None, None]
        f_gq.shape = (-1, f_gq.shape[-1])

    @property
    def nlm(self):
        return self.grids_indexer.nlm

    @classmethod
    def from_plan(cls, plan, gplan, cider_kernel, Z, xcc, beta=1.8):
        lmax = int(np.sqrt(Y_nL.shape[1] + 1e-8)) - 1
        rmax = np.max(xcc.rgd.r_g)
        # TODO tune this to find optimal basis size
        min_exps = 0.125 / (rmax * rmax) * np.ones(4)
        max_exps = get_hgbs_max_exps(Z)[:4]
        etb = get_etb_from_expnt_range(
            lmax, beta, min_exps, max_exps, 0.0, 1e99, lower_fac=1.0, upper_fac=4.0
        )
        inputs_to_atco = get_gamma_lists_from_etb_list([etb])
        atco = ATCBasis(*inputs_to_atco)
        return cls(plan, cider_kernel, atco, xcc, gplan)

    @property
    def is_mgga(self):
        return self.plan.nldf_settings.sl_level == "MGGA"

    @property
    def nspin(self):
        return self.plan.nspin

    def get_kinetic_energy(self, ae, D_sp=None):
        D_sp = self._D_sp if D_sp is None else D_sp
        nspins = D_sp.shape[0]
        xcc = self.xcc
        if ae:
            tau_pg = xcc.tau_pg
            tauc_g = xcc.tauc_g / (np.sqrt(4 * np.pi) * nspins)
        else:
            tau_pg = xcc.taut_pg
            tauc_g = xcc.tauct_g / (np.sqrt(4 * np.pi) * nspins)
        nn = tau_pg.shape[-1] // tauc_g.shape[0]
        tau_sg = np.dot(D_sp, tau_pg)
        tau_sg.shape = (tau_sg.shape[0], -1, nn)
        tau_sg[:] += tauc_g[:, None]
        tau_sg.shape = (tau_sg.shape[0], -1)
        return tau_sg

    def contract_kinetic_potential(self, dedtau_sg, ae):
        xcc = self.xcc
        if ae:
            tau_pg = xcc.tau_pg
            sign = 1.0
        else:
            tau_pg = xcc.taut_pg
            sign = -1.0
        wt = self.w_g * sign
        self._dH_sp += np.dot(dedtau_sg * wt, tau_pg.T)

    def vec_radial_vars(self, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae):
        nspin = len(n_sLg)
        ngrid = n_sLg.shape[-1] * weight_n.size
        nx = 5 if self.is_mgga else 4
        rho_sxg = np.empty((nspin, nx, ngrid))
        rho_sxg[:, 0] = np.dot(Y_nL, n_sLg).transpose(1, 2, 0).reshape(nspin, -1)
        b_vsgn = np.dot(rnablaY_nLv.transpose(0, 2, 1), n_sLg).transpose(1, 2, 3, 0)
        b_vsgn[..., 1:, :] /= self.xcc.rgd.r_g[1:, None]
        b_vsgn[..., 0, :] = b_vsgn[..., 1, :]
        a_sgn = np.dot(Y_nL, dndr_sLg).transpose(1, 2, 0)
        b_vsgn += R_nv.T[:, None, None, :] * a_sgn[None, :, :]
        N = Y_nL.shape[0]
        e_g = self.xcc.rgd.empty(N).reshape(-1)
        e_g[:] = 0
        rho_sxg[:, 1:4] = b_vsgn.transpose(1, 0, 2, 3).reshape(nspin, 3, -1)
        if self.is_mgga:
            rho_sxg[:, 4] = self.get_kinetic_energy(ae)
        vrho_sxg = np.zeros_like(rho_sxg)
        return e_g, rho_sxg, vrho_sxg

    @property
    def atco_inp(self):
        return self._atco

    @property
    def atco_feat(self):
        # TODO this will need to be different for vi and vij
        return self._atco

    def convert_rad2orb_(
        self, nspin, x_sgLq, x_suq, rad2orb=True, inp=True, indexer=None
    ):
        if inp:
            atco = self.atco_inp
        else:
            atco = self.atco_feat
        if indexer is None:
            indexer = self.grids_indexer
        for s in range(nspin):
            atco.convert_rad2orb_(
                x_sgLq[s],
                x_suq[s],
                indexer,
                indexer.rad_arr,
                rad2orb=rad2orb,
                offset=0,
            )

    def set_D_sp(self, D_sp, xcc):
        self._dH_sp = np.zeros_like(D_sp)
        self._D_sp = D_sp

    def get_dH_sp(self):
        return self._dH_sp

    def get_paw_atom_contribs(self, rho_sxg):
        nspin = self.nspin
        assert len(rho_sxg) == nspin
        x_gq = self.grids_indexer.empty_gq(nalpha=self.plan.nalpha)
        x_srLq = np.stack(
            [
                self.grids_indexer.empty_rlmq(nalpha=self.plan.nalpha)
                for s in range(nspin)
            ]
        )
        for s in range(nspin):
            rho_tuple = self.plan.get_rho_tuple(rho_sxg[s])
            arg_g = self.plan.get_interpolation_arguments(rho_tuple, i=-1)[0]
            fun_g = self.plan.get_function_to_convolve(rho_tuple)[0]
            fun_g[:] *= self.w_g
            p_gq = self.plan.get_interpolation_coefficients(arg_g.ravel(), i=-1)[0]
            x_gq[:] = p_gq * fun_g[:, None]
            self.grids_indexer.reduce_angc_ylm_(x_srLq[s], x_gq, a2y=True, offset=0)
        return x_srLq

    """
    def get_paw_atom_contribs_en(self, e_g, rho_sxg, vrho_sxg, f_srLq, feat_only=False):
        # NOTE that grid index order is different than previous implementation
        # gn instead of previous ng
        nspin = rho_sxg.shape[0]
        ngrid = rho_sxg.shape[2]
        f_srLq = f_srLq.copy()
        vf_srLq = np.empty_like(f_srLq)
        feat_sig = np.zeros((nspin, self.plan.nldf_settings.nfeat, ngrid))
        dfeat_sig = np.zeros((nspin, self.plan.num_vj, ngrid))
        f_gq = self.grids_indexer.empty_gq(
            nalpha=self.plan.nalpha + self.plan.num_vi_ints
        )
        n1 = len(self.plan.nldf_settings.l1_feat_specs)
        n0 = f_gq.shape[1] - 3 * n1
        if n1 > 0:
            f1_srLq = f_srLq[..., -n1:].copy()
            df1_sLqr = self._get_rad_deriv(f1_srLq.transpose(0, 2, 3, 1))
            df1_sLqr[:] *= self._dgdr
            print(
                "HII",
                f1_srLq[0, :, 0, 0].tolist(),
                df1_sLqr[0, 0, 0].tolist(),
                self.r_g.tolist(),
            )
            f1_srLq[:] /= self.r_g[:, None, None] + 1e-10
            f_srLq[..., -n1:] = df1_sLqr.transpose(0, 3, 1, 2)
            tmp_gq = np.empty((f_gq.shape[0], n1))
            rhat_gv = (np.ones((self.r_g.size, 1, 3)) * R_nv).reshape(-1, 3)
            ixlist = [(0, "x"), (1, "y"), (2, "z")]
        for s in range(nspin):
            self.grids_indexer.reduce_angc_ylm_(f_srLq[s], f_gq, a2y=False, offset=0)
            if n1 > 0:
                # go backwards to avoid overwriting
                for j in range(n1 - 1, -1, -1):
                    nstart = n0 + 3 * j
                    f_gq[:, nstart : nstart + 3] = f_gq[:, n0 + j][:, None] * rhat_gv
                for i, x in ixlist:
                    self.grids_indexer.reduce_angc_ylm_(
                        f1_srLq[s], tmp_gq, a2y=False, offset=0, deriv=x
                    )
                    for j in range(n1):
                        f_gq[:, n0 + 3 * j + i] += tmp_gq[:, j]
            self.plan.eval_rho_full(
                f_gq,
                rho_sxg[s],
                spin=s,
                feat=feat_sig[s],
                dfeat=dfeat_sig[s],
                cache_p=True,
                # TODO might need to be True for gaussian interp
                apply_transformation=False,
                # coeff_multipliers=self.plan.alpha_norms,
            )
        if feat_only:
            return feat_sig
        # print(feat_sig[0, 0, :: self._ang_w_g.size])
        print("FEAT", feat_sig[0].sum(axis=-1), dfeat_sig[0].sum(axis=-1))
        sigma_xg = get_sigma(rho_sxg[:, 1:4])
        dedsigma_xg = np.zeros_like(sigma_xg)
        nspin = feat_sig.shape[0]
        dedn_sg = np.zeros_like(vrho_sxg[:, 0])
        e_g[:] = 0.0
        args = [
            e_g,
            rho_sxg[:, 0].copy(),
            dedn_sg,
            sigma_xg,
            dedsigma_xg,
        ]
        if rho_sxg.shape[1] == 4:
            args.append(feat_sig)
        elif rho_sxg.shape[1] == 5:
            dedtau_sg = np.zeros_like(vrho_sxg[:, 0])
            args += [
                rho_sxg[:, 4].copy(),
                dedtau_sg,
                feat_sig,
            ]
        else:
            raise ValueError
        vfeat_sig = self.cider_kernel.calculate(*args)
        vrho_sxg[:, 0] = dedn_sg
        if rho_sxg.shape[1] == 5:
            vrho_sxg[:, 4] = dedtau_sg
        vrho_sxg[:, 1:4] = 2 * dedsigma_xg[::2, None, :] * rho_sxg[:, 1:4]
        if nspin == 2:
            vrho_sxg[:, 1:4] += dedsigma_xg[1, None, :] * rho_sxg[::-1, 1:4]
        # print(vfeat_sig.sum())
        for s in range(nspin):
            f_gq[:] = self.plan.eval_vxc_full(
                vfeat_sig[s],
                vrho_sxg[s],
                dfeat_sig[s],
                rho_sxg[s],
                spin=s,
            )
            # self._apply_ang_weight_(f_gq)
            f_gq[:] *= self.w_g[:, None]
            if n1 > 0:
                tmp_rLq = f1_srLq[s].copy()
                f1_srLq[s] = 0.0
                for i, x in ixlist:
                    for j in range(n1):
                        tmp_gq[:, j] = f_gq[:, n0 + 3 * j + i]
                    self.grids_indexer.reduce_angc_ylm_(
                        tmp_rLq, tmp_gq, a2y=True, offset=0, deriv=x
                    )
                    f1_srLq[s] += tmp_rLq
                # go forward to avoid overwriting
                for j in range(0, n1):
                    nstart = n0 + 3 * j
                    tmp = np.einsum("gv,gv->g", f_gq[:, nstart : nstart + 3], rhat_gv)
                    f_gq[:, n0 + j] = tmp
            self.grids_indexer.reduce_angc_ylm_(vf_srLq[s], f_gq, a2y=True, offset=0)
        if n1 > 0:
            df1_sLqr[:] = vf_srLq[..., -n1:].transpose(0, 2, 3, 1).copy()
            f1_srLq[:] /= self.r_g[:, None, None] + 1e-10
            print(
                np.sum(f1_srLq),
                np.sum(np.abs(f1_srLq)),
                vf_srLq.sum(),
                np.abs(vf_srLq).sum(),
            )
            print(np.abs(df1_sLqr).sum())
            df1_sLqr[:] *= self._dgdr
            tmp = self._get_rad_deriv_bwd(df1_sLqr)
            print(np.abs(tmp).sum())
            f1_srLq[:] += tmp.transpose(0, 3, 1, 2)
            print(
                np.sum(f1_srLq),
                np.sum(np.abs(f1_srLq)),
                vf_srLq.sum(),
                np.abs(vf_srLq).sum(),
            )
            # print(np.abs(f1_srLq).sum(axis=1), np.abs(vf_srLq).sum(axis=(1, 3)))
            # print(f1_srLq[0, :, 0, 0])
            vf_srLq[..., -n1:] = f1_srLq
            # vf_srLq[:, :10, :, -n1:] = 0  # f1_srLq
        # vf_srLq[:] *= self._rad_w_g[:, None, None]
        return vf_srLq
    """

    def get_paw_atom_contribs_en(self, e_g, rho_sxg, vrho_sxg, f_srLq, feat_only=False):
        # NOTE that grid index order is different than previous implementation
        # gn instead of previous ng
        nspin = rho_sxg.shape[0]
        ngrid = rho_sxg.shape[2]
        f_srLq = np.ascontiguousarray(f_srLq)
        vf_srLq = np.empty_like(f_srLq)
        feat_sig = np.zeros((nspin, self.plan.nldf_settings.nfeat, ngrid))
        dfeat_sig = np.zeros((nspin, self.plan.num_vj, ngrid))
        f_gq = self.grids_indexer.empty_gq(nalpha=self.plan.nalpha)
        for s in range(nspin):
            self.grids_indexer.reduce_angc_ylm_(f_srLq[s], f_gq, a2y=False, offset=0)
            self.plan.eval_rho_full(
                f_gq,
                rho_sxg[s],
                spin=s,
                feat=feat_sig[s],
                dfeat=dfeat_sig[s],
                cache_p=True,
                # TODO might need to be True for gaussian interp
                apply_transformation=False,
                # coeff_multipliers=self.plan.alpha_norms,
            )
        if feat_only:
            return feat_sig
        featw_ig = feat_sig.sum(axis=0) * self.w_g
        print("FEAT", featw_ig.sum(axis=-1))
        sigma_xg = get_sigma(rho_sxg[:, 1:4])
        dedsigma_xg = np.zeros_like(sigma_xg)
        nspin = feat_sig.shape[0]
        dedn_sg = np.zeros_like(vrho_sxg[:, 0])
        e_g[:] = 0.0
        args = [
            e_g,
            rho_sxg[:, 0].copy(),
            dedn_sg,
            sigma_xg,
            dedsigma_xg,
        ]
        if rho_sxg.shape[1] == 4:
            args.append(feat_sig)
        elif rho_sxg.shape[1] == 5:
            dedtau_sg = np.zeros_like(vrho_sxg[:, 0])
            args += [
                rho_sxg[:, 4].copy(),
                dedtau_sg,
                feat_sig,
            ]
        else:
            raise ValueError
        vfeat_sig = self.cider_kernel.calculate(*args)
        vrho_sxg[:, 0] = dedn_sg
        if rho_sxg.shape[1] == 5:
            vrho_sxg[:, 4] = dedtau_sg
        vrho_sxg[:, 1:4] = 2 * dedsigma_xg[::2, None, :] * rho_sxg[:, 1:4]
        if nspin == 2:
            vrho_sxg[:, 1:4] += dedsigma_xg[1, None, :] * rho_sxg[::-1, 1:4]
        for s in range(nspin):
            f_gq[:] = self.plan.eval_vxc_full(
                vfeat_sig[s],
                vrho_sxg[s],
                dfeat_sig[s],
                rho_sxg[s],
                spin=s,
            )
            f_gq[:] *= self.w_g[:, None]
            self.grids_indexer.reduce_angc_ylm_(vf_srLq[s], f_gq, a2y=True, offset=0)
        return vf_srLq

    def get_paw_atom_contribs_pot(self, rho_sxg, vrho_sxg, vx_srLq):
        nspin = self.nspin
        vx_gq = self.grids_indexer.empty_gq(nalpha=self.plan.nalpha)
        for s in range(nspin):
            self.grids_indexer.reduce_angc_ylm_(vx_srLq[s], vx_gq, a2y=False, offset=0)
            rho_tuple = self.plan.get_rho_tuple(rho_sxg[s])
            arg_g, darg_g = self.plan.get_interpolation_arguments(rho_tuple, i=-1)
            fun_g, dfun_g = self.plan.get_function_to_convolve(rho_tuple)
            p_gq, dp_gq = self.plan.get_interpolation_coefficients(arg_g, i=-1)
            dedarg_g = np.einsum("gq,gq->g", dp_gq, vx_gq) * fun_g
            dedfun_g = np.einsum("gq,gq->g", p_gq, vx_gq)
            vrho_sxg[s, 0] += dedarg_g * darg_g[0] + dedfun_g * dfun_g[0]
            tmp = dedarg_g * darg_g[1] + dedfun_g * dfun_g[1]
            vrho_sxg[s, 1:4] += 2 * rho_sxg[s, 1:4] * tmp
            if len(darg_g) == 3:
                vrho_sxg[s, 4] += dedarg_g * darg_g[2] + dedfun_g * dfun_g[2]

    def grid2aux(self, f_sgLq):
        raise NotImplementedError

    def aux2grid(self, f_sxq):
        raise NotImplementedError

    def x2u(self, f_sxq):
        raise NotImplementedError

    def calculate_y_terms(self, xt_sgLq, dx_sgLq, projector):
        na = projector.alphas_ps.size
        self._projector = projector
        t0 = time.monotonic()
        dx_skLq = self.grid2aux(dx_sgLq)
        th = time.monotonic()
        dy_skLq = self.perform_convolution_fwd(dx_skLq)
        t1 = time.monotonic()
        c_siq, dxt_skLq, df_sgLq = projector.get_c_and_df(dy_skLq)
        t2 = time.monotonic()
        xt_skLq = self.grid2aux(xt_sgLq)
        xt_skLq[..., :na] += dxt_skLq
        fr_skLq = self.perform_convolution_fwd(xt_skLq)
        fr_skLq[:] *= self._projector.dv_k[:, None, None]
        fr_sgLq = self.aux2grid(fr_skLq)
        t3 = time.monotonic()
        print("YTIMES", th - t0, t1 - th, t2 - t1, t3 - t2)
        return fr_sgLq, df_sgLq, c_siq

    def calculate_vx_terms(self, vfr_sgLq, vdf_sgLq, vc_siq, projector):
        na = projector.alphas_ps.size
        self._projector = projector
        vfr_skLq = self.grid2aux(vfr_sgLq)
        vfr_skLq[:] *= self._projector.dv_k[:, None, None]
        vxt_skLq = self.perform_convolution_bwd(vfr_skLq)
        vxt_skLa = vxt_skLq[..., :na].copy()
        vxt_sgLq = self.aux2grid(vxt_skLq)
        vdy_skLq = projector.get_vy_and_vyy(vc_siq, vxt_skLa, vdf_sgLq)
        vdx_skLq = self.perform_convolution_bwd(vdy_skLq)
        vdx_sgLq = self.aux2grid(vdx_skLq)
        return vxt_sgLq, vdx_sgLq


class PAWCiderContribsRecip(_PAWCiderContribs):
    def __init__(self, plan, cider_kernel, atco, xcc, gplan):
        super(PAWCiderContribsRecip, self).__init__(
            plan, cider_kernel, atco, xcc, gplan
        )
        self._atco_recip = self._atco.get_reciprocal_atco()
        self._galphas = gplan.alphas.copy()
        self._gnorms = gplan.alpha_norms.copy()
        self._grid_indexers = {}

    def _make_grid_indexers(self, nlm):
        if nlm not in self._grid_indexers:
            y = Y_nL[:, :nlm]
            r_g = self.grids_indexer.rad_arr
            k_g = self._projector.k_k
            rind = AtomicGridsIndexer.make_single_atom_indexer(y, r_g)
            kind = AtomicGridsIndexer.make_single_atom_indexer(y, k_g)
            self._grid_indexers[nlm] = (rind, kind)
        return self._grid_indexers[nlm]

    def r2k(self, in_sxLq, fwd=True):
        in_sxLq = np.ascontiguousarray(in_sxLq)
        nspin, nx, nlm, nq = in_sxLq.shape
        indexers = self._make_grid_indexers(nlm)
        if fwd:
            atco_inp = self.atco_inp
            atco_out = self._atco_recip
            in_indexer, out_indexer = indexers
        else:
            atco_inp = self._atco_recip
            atco_out = self.atco_feat
            out_indexer, in_indexer = indexers
        nx = out_indexer.rad_arr.size
        shape = (nspin, nx, nlm, nq)
        out_sxLq = np.zeros(shape)
        assert in_sxLq.flags.c_contiguous
        assert out_sxLq.flags.c_contiguous
        for s in range(nspin):
            tmp_uq = np.zeros((atco_inp.nao, nq))
            atco_inp.convert_rad2orb_(
                in_sxLq[s],
                tmp_uq,
                in_indexer,
                in_indexer.rad_arr,
                rad2orb=True,
                offset=0,
            )
            self.atco_inp.solve_atc_coefs_arr_(tmp_uq)
            atco_out.convert_rad2orb_(
                out_sxLq[s],
                tmp_uq,
                out_indexer,
                out_indexer.rad_arr,
                rad2orb=False,
                offset=0,
            )
        return out_sxLq

    def grid2aux(self, f_sgLq):
        return self.r2k(f_sgLq, fwd=True)

    def aux2grid(self, f_skLq):
        return self.r2k(f_skLq, fwd=False)

    def calc_conv_skLq(self, in_skLq, alphas=None, alpha_norms=None):
        in_skLq = np.ascontiguousarray(in_skLq)
        out_skLq = np.empty_like(in_skLq)
        k_g = np.ascontiguousarray(self._projector.k_k)
        nspin, nk, nlm, nq = in_skLq.shape
        if alphas is None:
            alphas = self.plan.alphas
            alpha_norms = self.plan.alpha_norms
        assert nk == k_g.size
        assert nq == alphas.size
        t0 = time.monotonic()
        libcider.atc_reciprocal_convolution(
            in_skLq.ctypes.data_as(ctypes.c_void_p),
            out_skLq.ctypes.data_as(ctypes.c_void_p),
            k_g.ctypes.data_as(ctypes.c_void_p),
            alphas.ctypes.data_as(ctypes.c_void_p),
            alpha_norms.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nspin),
            ctypes.c_int(nk),
            ctypes.c_int(nlm),
            ctypes.c_int(nq),
        )
        t1 = time.monotonic()
        print("TIME", t1 - t0)
        return out_skLq

    def perform_convolution_fwd(self, f_skLq):
        return self.calc_conv_skLq(f_skLq)

    def perform_convolution_bwd(self, f_skLq):
        return self.calc_conv_skLq(f_skLq)

    def perform_fitting_convolution_bwd(self, f_skLq):
        return self.calc_conv_skLq(
            f_skLq,
            self.sbt_rgd,
            alphas=self._galphas,
            alpha_norms=self._gnorms,
        )

    perform_fitting_convolution_fwd = perform_fitting_convolution_bwd


PAWCiderContribs = PAWCiderContribsRecip


class CiderRadialFeatureCalculator:
    def __init__(self, xc):
        self.xc = xc
        self.mode = "feature"

    def __call__(self, rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae=True):
        e_g, rho_sxg, vrho_sxg = self.xc.vec_radial_vars(
            n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae
        )
        return self.xc.get_paw_atom_contribs(rho_sxg)


class CiderRadialEnergyCalculator:
    def __init__(self, xc):
        self.xc = xc
        self.mode = "energy"

    def __call__(self, rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, f_srLq, ae=True):
        e_g, rho_sxg, vrho_sxg = self.xc.vec_radial_vars(
            n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae
        )
        print("AE", ae)
        vf_srLq = self.xc.get_paw_atom_contribs_en(
            e_g, rho_sxg, vrho_sxg, f_srLq, feat_only=False
        )
        if rho_sxg.shape[1] == 5:
            self.xc.contract_kinetic_potential(vrho_sxg[:, 4], ae)
        return e_g, vrho_sxg[:, 0], vrho_sxg[:, 1:4], vf_srLq


class CiderRadialPotentialCalculator:
    def __init__(self, xc):
        self.xc = xc
        self.mode = "potential"

    def __call__(self, rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, vx_srLq, ae=True):
        e_g, rho_sxg, vrho_sxg = self.xc.vec_radial_vars(
            n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae
        )
        self.xc.get_paw_atom_contribs_pot(rho_sxg, vrho_sxg, vx_srLq)
        if rho_sxg.shape[1] == 5:
            self.xc.contract_kinetic_potential(vrho_sxg[:, 4], ae)
        return vrho_sxg[:, 0], vrho_sxg[:, 1:4]


class CiderRadialExpansion:
    def __init__(self, rcalc, ft_srLq=None, df_srLq=None):
        self.rcalc = rcalc
        self.ft_srLq = ft_srLq
        self.df_srLq = df_srLq
        assert self.rcalc.mode in [
            "feature",
            "energy",
            "potential",
        ]

    def __call__(self, rgd, D_sLq, n_qg, dndr_qg, nc0_sg, dnc0dr_sg, ae=True):
        n_sLg = np.dot(D_sLq, n_qg)
        n_sLg[:, 0] += nc0_sg
        dndr_sLg = np.dot(D_sLq, dndr_qg)
        dndr_sLg[:, 0] += dnc0dr_sg
        nspins, Lmax, nq = D_sLq.shape
        if self.rcalc.mode == "feature":
            # We call it wx_srLq because it is multiplied
            # by the grid weights already
            if self.ft_srLq is None:
                wx_srLq = self.rcalc(
                    rgd, n_sLg, Y_nL[:, :Lmax], dndr_sLg, rnablaY_nLv[:, :Lmax], ae
                )
            else:
                f_srLq = self.ft_srLq + self.df_srLq if ae else self.ft_srLq
                wx_srLq = self.rcalc(
                    rgd,
                    n_sLg,
                    Y_nL[:, :Lmax],
                    dndr_sLg,
                    rnablaY_nLv[:, :Lmax],
                    f_srLq,
                    ae,
                )
            return wx_srLq
        elif self.rcalc.mode == "energy":
            dEdD_sqL = np.zeros((nspins, nq, Lmax))
            f_srLq = self.ft_srLq + self.df_srLq if ae else self.ft_srLq
            e_g, dedn_sg, dedgrad_svg, vf_srLq = self.rcalc(
                rgd,
                n_sLg,
                Y_nL[:, :Lmax],
                dndr_sLg,
                rnablaY_nLv[:, :Lmax],
                f_srLq,
                ae=ae,
            )
            nn = Y_nL.shape[0]
            nq = f_srLq.shape[-1]
            dedn_sgn = dedn_sg.reshape(nspins, -1, nn)
            dedgrad_svgn = dedgrad_svg.reshape(nspins, 3, -1, nn)
            dedgrad_sgn = np.einsum("svgn,nv->sgn", dedgrad_svgn, R_nv)
            dEdD_sqn = np.einsum("qg,sgn->sqn", n_qg * rgd.dv_g, dedn_sgn)
            dEdD_sqn += np.einsum("qg,sgn->sqn", dndr_qg * rgd.dv_g, dedgrad_sgn)
            dEdD_sqL = np.einsum(
                "sqn,nL->sqL", dEdD_sqn, weight_n[:, None] * Y_nL[:, :Lmax]
            )
            tmp = rgd.dr_g * rgd.r_g
            B_svqn = np.einsum("qg,svgn->svqn", n_qg * tmp, dedgrad_svgn)
            dEdD_sqL += np.einsum(
                "nLv,svqn->sqL",
                (4 * np.pi) * weight_n[:, None, None] * rnablaY_nLv[:, :Lmax],
                B_svqn,
            )
            E = rgd.integrate(e_g.reshape(-1, nn).dot(weight_n))
            return E, dEdD_sqL, vf_srLq
        else:
            f_srLq = self.ft_srLq + self.df_srLq if ae else self.ft_srLq
            dedn_sg, dedgrad_svg = self.rcalc(
                rgd,
                n_sLg,
                Y_nL[:, :Lmax],
                dndr_sLg,
                rnablaY_nLv[:, :Lmax],
                f_srLq,
                ae=ae,
            )
            nn = Y_nL.shape[0]
            nq = f_srLq.shape[-1]
            dedn_sgn = dedn_sg.reshape(nspins, -1, nn)
            dedgrad_svgn = dedgrad_svg.reshape(nspins, 3, -1, nn)
            dedgrad_sgn = np.einsum("svgn,nv->sgn", dedgrad_svgn, R_nv)
            dEdD_sqn = np.einsum("qg,sgn->sqn", n_qg * rgd.dv_g, dedn_sgn)
            dEdD_sqn += np.einsum("qg,sgn->sqn", dndr_qg * rgd.dv_g, dedgrad_sgn)
            dEdD_sqL = np.einsum(
                "sqn,nL->sqL", dEdD_sqn, weight_n[:, None] * Y_nL[:, :Lmax]
            )
            tmp = rgd.dr_g * rgd.r_g
            B_svqn = np.einsum("qg,svgn->svqn", n_qg * tmp, dedgrad_svgn)
            dEdD_sqL += np.einsum(
                "nLv,svqn->sqL",
                (4 * np.pi) * weight_n[:, None, None] * rnablaY_nLv[:, :Lmax],
                B_svqn,
            )
            return dEdD_sqL


class FastPASDWCiderKernel:
    """
    This class is a bit confusing because there are a lot of intermediate
    terms and contributions for the pseudo and all-electron parts of the
    feature convolutions. Here is a summary of the terms:
        xt  : The theta functions for the pseudo-density
        x   : The theta functions for the all-electron density
        dx  : x - xt
        yt  : Raw convolutions xt
        dy  : Raw convolutions of dx
        c   : Coefficients for functions to smoothly fit the (non-smooth)
              dx such that dy and dfr match outside the core radius
        dfr : The values of the convolutions of the PASDW projector functions,
              weighted by c
        fr  : dfr + yt
        ft  : fr + contributions to the convolutions from other atoms, which
              are computed based on the projections of the FFT convolutions
              onto smooth atomic functions.
        df  : Projection of (dy - dfr) onto some basis that decays outside
              the core radius. The idea is that df holds all the high-frequency
              parts of the convolutions and (approximately) vanishes
              outside the core radius.
        f   : df + ft, should accurately approximate the "true" all-electron
              features in the core radius
        v*  : Derivative of the XC energy with respect to *.
              vf, vdf, vft, vfr, etc... are all defined like this.

    After the underscore on each array, the indices are indicated:
        a : Atom
        L : l,m spherical harmonic index
        i : Projector function index
        u : Dense atomic density fitting basis set index
        q : Index for the theta functions or convolutions
        g : Radial real-space grid index
        k : Radial reciprocal-space grid index
        s : Spin index
    """

    is_cider_functional = True
    PAWCiderContribs = PAWCiderContribs

    fr_asgLq: dict
    df_asgLq: dict
    vfr_asgLq: dict
    vdf_asgLq: dict
    # TODO if calculating the density (which is done 3X per loop in
    # this implementation) and potential (done 2X per loop)
    # ever becomes a bottleneck, it can
    # be saved in the below variables to avoid recomputing.
    # rho_asxg: dict
    # vrho_asxg: dict
    # rhot_asxg: dict
    # vrhot_asxg: dict

    def __init__(
        self,
        cider_kernel,
        plan,
        gd,
        cut_xcgrid,
        timer=None,
    ):
        if timer is None:
            from gpaw.utilities.timing import Timer

            self.timer = Timer()
        self.gd = gd
        self.cider_kernel = cider_kernel
        self.Nalpha_small = plan.nalpha
        self.cut_xcgrid = cut_xcgrid
        self.plan = plan
        self.is_mgga = plan.nldf_settings.sl_level == "MGGA"

    @property
    def amin(self):
        return self.plan.alpha0

    @property
    def lambd(self):
        return self.plan.lambd

    @property
    def alphas(self):
        return self.plan.alphas

    def get_D_asp(self):
        return self.atomdist.to_work(self.dens.D_asp)

    def initialize(self, density, atomdist, atom_partition, setups):
        self.dens = density
        self.atomdist = atomdist
        self.atom_partition = atom_partition
        self.setups = setups

    def initialize_more_things(self, setups=None):
        self.nspin = self.dens.nt_sg.shape[0]
        if setups is None:
            setups = self.dens.setups
        for setup in setups:
            if not hasattr(setup, "cider_contribs") or setup.cider_contribs is None:
                setup.xc_correction = DiffPAWXCCorrection.from_setup(
                    setup, build_kinetic=True, ke_order_ng=False
                )
                # TODO for some reason, in the core there are sometimes
                # exponents that are very big (z**2 * 2000 required).
                # However, the SCF energy and stress are essentially
                # unchanged by not including these very high exponents,
                # and they don't seem physically relevant. So it would be good
                # to set this lower by eliminating these few high-exponent points
                # somehow. In the meantime, we set the energy cutoff to a large
                # value and pass raise_large_expnt_error=False
                # to the initializer below just in case to avoid crashes.
                encut = np.float64(setup.Z**2 * 200)
                if encut - 1e-7 <= np.max(self.alphas):
                    encut0 = np.max(self.alphas)
                    Nalpha = self.alphas.size
                else:
                    amin = np.float64(self.amin)
                    Nalpha = int(np.ceil(np.log(encut / amin) / np.log(self.lambd))) + 1
                    nam1 = np.float64(Nalpha - 1)
                    encut0 = amin * self.lambd**nam1
                    assert encut0 >= encut - 1e-6, "Math went wrong {} {}".format(
                        encut0, encut
                    )
                    assert (
                        encut0 / self.lambd < encut
                    ), "Math went wrong {} {} {} {}".format(
                        encut0, encut, self.lambd, encut0 / self.lambd
                    )
                atom_plan = self.plan.new(nalpha=Nalpha, use_smooth_expnt_cutoff=True)
                setup.cider_contribs = self.PAWCiderContribs.from_plan(
                    atom_plan,
                    self.plan,
                    self.cider_kernel,
                    setup.Z,
                    setup.xc_correction,
                    beta=1.6,
                )
                if setup.cider_contribs.plan is not None:
                    assert (
                        abs(np.max(setup.cider_contribs.plan.alphas) / encut0 - 1)
                        < 1e-8
                    ), "{}".format(np.max(setup.cider_contribs.plan.alphas) - encut0)
                setup.ps_setup = PSSetup.from_setup(setup)
                setup.pa_setup = PASetup.from_setup(setup)
                sbt_rgd = SBTGridContainer.from_setup(
                    setup, rmin=1e-4, N=1024, encut=5e7, d=0.02
                ).big_rgd
                setup.cider_proj = CiderCoreTermProjector.from_atco_and_setup(
                    self.plan,
                    atom_plan,
                    setup.ps_setup,
                    setup.cider_contribs.atco_inp,
                    sbt_rgd,
                    encut0,
                    # TODO this is grid_nlm
                    # setup.cider_contribs.nlm,
                )

    def _multiply_spherical_terms_by_cutoff_(self, setup, x_sgLq):
        slgd = setup.xc_correction.rgd
        RCUT = np.max(setup.rcut_j)
        rcut = RCUT
        rmax = slgd.r_g[-1]
        fcut = 0.5 + 0.5 * np.cos(np.pi * (slgd.r_g - rcut) / (rmax - rcut))
        fcut[slgd.r_g < rcut] = 1.0
        fcut[slgd.r_g > rmax] = 0.0
        x_sgLq *= fcut[:, None, None]

    def calculate_paw_cider_features(self, setups, D_asp):
        """
        Computes the contributions to the convolutions F_beta arising
        from the PAW contributions. Projects them onto
        the PAW grids, and also computes the compensation function
        coefficients c_sia that reproduce the F_beta contributions
        outside the augmentation region.

        Args:
            setups: GPAW atomic setups
            D_asp: Atomic PAW density matrices
        """
        if len(D_asp.keys()) == 0:
            return {}
        a0 = list(D_asp.keys())[0]
        nspin = D_asp[a0].shape[0]
        assert (nspin == 1) or self.is_cider_functional
        self.fr_asgLq = {}
        self.df_asgLq = {}
        c_asiq = {}

        for a, D_sp in D_asp.items():
            t0 = time.monotonic()
            setup = setups[a]

            self.timer.start("coefs")
            rcalc = CiderRadialFeatureCalculator(setup.cider_contribs)
            expansion = CiderRadialExpansion(rcalc)
            dx_sgLq, xt_sgLq = calculate_cider_paw_correction(
                expansion, setup, D_sp, separate_ae_ps=True
            )
            dx_sgLq -= xt_sgLq
            self.timer.stop()
            self._multiply_spherical_terms_by_cutoff_(setup, xt_sgLq)
            self.timer.start("transform and convolve")
            t1 = time.monotonic()
            fr_sgLq, df_sgLq, c_siq = setup.cider_contribs.calculate_y_terms(
                xt_sgLq, dx_sgLq, setup.cider_proj
            )
            t2 = time.monotonic()
            self.timer.stop()
            self.df_asgLq[a] = df_sgLq
            self.fr_asgLq[a] = fr_sgLq
            c_asiq[a] = c_siq
            print("OVERALL", t1 - t0, t2 - t1)
        return c_asiq

    def _add_smooth_nonlocal_terms_(
        self, psetup, fr_sgLq, ft_sgLq, D_siq, dv_g, qmin, qmax
    ):
        ni = psetup.ni
        nspin = D_siq.shape[0]
        for i in range(ni):
            L = psetup.lmlist_i[i]
            n = psetup.nlist_i[i]
            Dref_sq = (psetup.pfuncs_ng[n] * dv_g).dot(fr_sgLq[:, :, L, qmin:qmax])
            j = psetup.jlist_i[i]
            for s in range(nspin):
                ft_sgLq[s, :, L, qmin:qmax] += (
                    D_siq[s, i] - Dref_sq[s]
                ) * psetup.ffuncs_jg[j][:, None]

    def _add_smooth_nonlocal_potential_(
        self, psetup, vfr_sgLq, vft_sgLq, dvD_siq, dv_g, qmin, qmax
    ):
        # TODO C function for this loop? can go in pa_setup
        ni = psetup.ni
        for i in range(ni):
            L = psetup.lmlist_i[i]
            n = psetup.nlist_i[i]
            j = psetup.jlist_i[i]
            dvD_sq = np.einsum(
                "sgq,g->sq",
                vft_sgLq[:, :, L, qmin:qmax],
                psetup.ffuncs_jg[j],
            )
            dvD_siq[:, i] = dvD_sq
            # We need to multiply the term here by dv_g because vfr_sgLq
            # has already been multiplied by the quadrature weights.
            vfr_sgLq[:, :, L, qmin:qmax] -= (
                dvD_sq[:, None, :] * psetup.pfuncs_ng[n][:, None]
            ) * dv_g[:, None]

    def calculate_paw_cider_energy(self, setups, D_asp, D_asiq):
        """
        Taking the projections of the pseudo-convolutions F_beta onto the PASDW
        projectors (coefficients given by D_sabi), as well as the
        reference F_beta contributions from part (1), computes the PAW
        XC energy and potential, as well as the functional derivative
        with respect to D_sabi

        Args:
            setups: GPAW atomic setups
            D_asp: Atomic PAW density matrices
            D_asiq: Atomic PAW PASDW projections
        """
        deltaE = {}
        deltaV = {}
        if len(D_asp.keys()) == 0:
            return {}, {}, {}
        a0 = list(D_asp.keys())[0]
        nspin = D_asp[a0].shape[0]
        dvD_asiq = {}
        self.vfr_asgLq = {}
        self.vdf_asgLq = {}

        for a, D_sp in D_asp.items():
            setup = setups[a]
            psetup = setup.pa_setup
            ni = psetup.ni
            Nalpha_sm = D_asiq[a].shape[-1]
            dv_g = get_dv(setup.xc_correction.rgd)
            df_sgLq = self.df_asgLq[a]
            ft_sgLq = np.zeros_like(df_sgLq)
            fr_sgLq = self.fr_asgLq[a]
            self._add_smooth_nonlocal_terms_(
                psetup, fr_sgLq, ft_sgLq, D_asiq[a], dv_g, 0, Nalpha_sm
            )
            ft_sgLq[:] += fr_sgLq
            rcalc = CiderRadialEnergyCalculator(setup.cider_contribs)
            expansion = CiderRadialExpansion(
                rcalc,
                ft_sgLq,
                df_sgLq,
            )
            # The vf_sgLq and vft_sgLq returned by the function below have
            # already been multiplied by the real-space quadrature weights,
            # so dv_g is not applied to them again.
            deltaE[a], deltaV[a], vf_sgLq, vft_sgLq = calculate_cider_paw_correction(
                expansion,
                setup,
                D_sp,
                separate_ae_ps=True,
            )
            vfr_sgLq = vf_sgLq - vft_sgLq
            vft_sgLq[:] = vfr_sgLq
            dvD_asiq[a] = np.zeros((nspin, ni, Nalpha_sm))
            self._add_smooth_nonlocal_potential_(
                psetup, vfr_sgLq, vft_sgLq, dvD_asiq[a], dv_g, 0, Nalpha_sm
            )
            self.vfr_asgLq[a] = vfr_sgLq
            self.vdf_asgLq[a] = vf_sgLq
            # This step saves some memory since fr_asgLq and df_asgLq are not
            # needed anymore after this step.
            self.fr_asgLq[a] = None
            self.df_asgLq[a] = None
        return dvD_asiq, deltaE, deltaV

    def calculate_paw_cider_potential(self, setups, D_asp, vc_asiq):
        """
        Given functional derivatives with respect to the augmentation function
        coefficients (c_sabi), as well as other atom-centered functional
        derivative contributions stored from part (2), compute the XC potential
        arising from the nonlocal density dependence of the CIDER features.

        Args:
            setups: GPAW atomic setups
            D_asp: Atomic PAW density matrices
            vc_asiq: functional derivatives with respect to coefficients c_asiq
        """
        dH_asp = {}
        if len(D_asp.keys()) == 0:
            return {}
        a0 = list(D_asp.keys())[0]
        nspin = D_asp[a0].shape[0]
        assert (nspin == 1) or self.is_cider_functional

        for a, D_sp in D_asp.items():
            setup = setups[a]
            vxt_sgLq, vdx_sgLq = setup.cider_contribs.calculate_vx_terms(
                self.vfr_asgLq[a], self.vdf_asgLq[a], vc_asiq[a], setup.cider_proj
            )
            self._multiply_spherical_terms_by_cutoff_(setup, vxt_sgLq)
            F_sbLg = vdx_sgLq - vxt_sgLq
            y_sbLg = vxt_sgLq.copy()
            rcalc = CiderRadialPotentialCalculator(setup.cider_contribs)
            expansion = CiderRadialExpansion(
                rcalc,
                F_sbLg,
                y_sbLg,
            )
            dH_asp[a] = calculate_cider_paw_correction(
                expansion, setup, D_sp, separate_ae_ps=True
            )
        return dH_asp

    def calculate_paw_feat_corrections(
        self,
        setups,
        D_asp,
        D_asiq=None,
        vc_asiq=None,
    ):
        """
        This function performs one of three PAW correction routines,
        depending on the inputs.
            1) If D_sabi is None and vc_sabi is None: Computes the
            contributions to the convolutions F_beta arising
            from the PAW contributions. Projects them onto
            the PAW grids, and also computes the compensation function
            coefficients c_sia that reproduce the F_beta contributions
            outside the augmentation region.
            2) If D_sabi is NOT None (df_asbLg must also be given): Taking the
            projections of the pseudo-convolutions F_beta onto the PASDW
            projectors (coefficients given by D_sabi), as well as the
            reference F_beta contributions from part (1), computes the PAW
            XC energy and potential, as well as the functional derivative
            with respect to D_sabi
            3) If D_sabi is None and vc_sabi is NOT None: Given functional
            derivatives with respect to the augmentation function coefficients
            (c_sabi), as well as other atom-centered functional derivative
            contributions stored from part (2), compute the XC potential arising
            from the nonlocal density dependence of the CIDER features.

        Args:
            setups: GPAW atomic setups
            D_asp: Atomic PAW density matrices
            D_asiq: Atomic PAW PASDW projections
            vc_asiq: functional derivatives with respect to coefficients c_sabi
        """
        if D_asiq is None and vc_asiq is None:
            return self.calculate_paw_cider_features(setups, D_asp)
        elif D_asiq is not None:
            return self.calculate_paw_cider_energy(setups, D_asp, D_asiq)
        else:
            assert vc_asiq is not None
            return self.calculate_paw_cider_potential(setups, D_asp, vc_asiq)


class _PASDWData:
    def __init__(self, interp_rgd, paw_rgd, nlist_j, llist_j, rcut):
        self.interp_rgd = interp_rgd
        self.paw_rgd = paw_rgd
        self.rcut = rcut
        self.nlist_j = None
        self.llist_j = None
        self.nlist_i = None
        self.llist_i = None
        self.lmlist_i = None
        self.jlist_i = None
        self.jloc_l = None
        self.exact_ovlp_pf = None
        self.pfuncs_ng = None
        self.pfuncs_ntp = None
        self.ffuncs_jg = None
        self.ffuncs_jtp = None
        self.ni = None
        self.nj = None
        self.nn = None
        self.lmax = None
        self._setup_basis(nlist_j, llist_j)
        self._setup_pfuncs_and_ffuncs()

    @property
    def rmax(self):
        return self.paw_rgd.r_g[-1]

    @property
    def rcut_func(self):
        return self.rcut

    def _setup_basis(self, nlist_j, llist_j):
        lp1 = np.max(llist_j) + 1
        nbas_lst = [0] * lp1
        for l in llist_j:
            nbas_lst[l] += 1
        jloc_l = np.append([0], np.cumsum(nbas_lst)).astype(np.int32)
        lmlist_i = []
        jlist_i = []
        i = 0
        for j, n in enumerate(nlist_j):
            l = llist_j[j]
            for m in range(2 * l + 1):
                lm = l * l + m
                lmlist_i.append(lm)
                jlist_i.append(j)
                i += 1
        lmlist_i = np.array(lmlist_i, dtype=np.int32)
        jlist_i = np.array(jlist_i, dtype=np.int32)

        self.nlist_j = np.array(nlist_j).astype(np.int32)
        self.llist_j = np.array(llist_j).astype(np.int32)
        self.nlist_i = self.nlist_j[jlist_i]
        self.llist_i = self.llist_j[jlist_i]
        self.lmlist_i = np.array(lmlist_i).astype(np.int32)
        self.jlist_i = np.array(jlist_i).astype(np.int32)
        self.jloc_l = np.array(jloc_l).astype(np.int32)

        self.nj = len(self.nlist_j)
        self.ni = len(self.nlist_i)
        self.nn = np.max(self.nlist_j) + 1
        self.lmax = np.max(self.llist_j)

    def get_pfunc(self, n, r_g):
        raise NotImplementedError

    def _setup_pfuncs_and_ffuncs(self):
        raise NotImplementedError

    def get_filt(self, r_g):
        filt = 0.5 + 0.5 * np.cos(np.pi * r_g / self.rcut)
        filt[r_g > self.rcut] = 0.0
        return filt

    def get_fcut(self, r):
        R = self.rcut
        fcut = np.ones_like(r)
        fcut[r > R] = 0.5 + 0.5 * np.cos(np.pi * (r[r > R] - R) / R)
        fcut[r > 2 * R] = 0.0
        return fcut

    def map_to_spline(self, funcs_t):
        assert funcs_t.ndim == 1
        nt = funcs_t.size
        return spline(np.arange(nt).astype(np.float64), funcs_t)

    def map_many_to_spline(self, funcs_nt):
        nn, nt = funcs_nt.shape
        funcs_ntp = np.empty((nn, nt, 4))
        for n in range(nn):
            funcs_ntp[n] = self.map_to_spline(funcs_nt[n])
        return funcs_ntp

    def get_setup_vals(self):
        nj = np.max(self.jloc_l)
        r_g = self.paw_rgd.r_g
        ng = r_g.size
        dr_g = self.paw_rgd.dr_g
        nn = np.max(self.nlist_j) + 1
        nt = self.interp_rgd.r_g.size
        return nn, nj, ng, nt, r_g, dr_g

    @classmethod
    def from_setup(cls, setup):
        rgd = setup.xc_correction.rgd
        rcut = np.max(setup.rcut_j)
        nt = 100
        interp_rgd = EquidistantRadialGridDescriptor(h=rcut * 1.0 / (nt - 1), N=nt)
        nlist_j, llist_j = get_psetup_func_counts(setup.Z)
        return cls(interp_rgd, rgd, nlist_j, llist_j, rcut)


class PASetup(_PASDWData):
    def get_pfunc(self, n, r_g):
        return get_ffunc(n, r_g, 0.8 * self.rmax)

    def _setup_pfuncs_and_ffuncs(self):
        nn, nj, ng, nt, r_g, dr_g = self.get_setup_vals()
        interp_r_g = self.interp_rgd.r_g
        ffuncs_jg = np.zeros((nj, ng))
        ffuncs_jt = np.zeros((nj, nt))
        dv_g = r_g * r_g * dr_g
        filt = self.get_filt(r_g)
        pfuncs_nt = np.empty((nn, nt), dtype=np.float64, order="C")
        pfuncs_ng = np.empty((nn, ng), dtype=np.float64, order="C")
        for n in range(nn):
            pfuncs_nt[n, :] = self.get_pfunc(n, interp_r_g)
            pfuncs_ng[n, :] = self.get_pfunc(n, r_g)
        for l in range(self.lmax + 1):
            jmin, jmax = self.jloc_l[l], self.jloc_l[l + 1]
            ovlp = np.einsum(
                "ig,jg,g->ij",
                pfuncs_ng[self.nlist_j[jmin:jmax]],
                pfuncs_ng[self.nlist_j[jmin:jmax]],
                dv_g * filt,
            )
            c_and_l = cho_factor(ovlp)
            ffuncs_jg[jmin:jmax] = cho_solve(
                c_and_l, pfuncs_ng[self.nlist_j[jmin:jmax]]
            )
            ffuncs_jt[jmin:jmax] = cho_solve(
                c_and_l, pfuncs_nt[self.nlist_j[jmin:jmax], :]
            )
        ffuncs_jtp = self.map_many_to_spline(ffuncs_jt)
        pfuncs_ng[:] *= filt
        pfuncs_nt[:] *= self.get_filt(interp_r_g)
        pfuncs_ntp = self.map_many_to_spline(pfuncs_nt)

        self.exact_ovlp_pf = np.identity(self.ni)
        self.pfuncs_ng = pfuncs_ng
        self.pfuncs_ntp = pfuncs_ntp
        self.ffuncs_jg = ffuncs_jg
        self.ffuncs_jtp = ffuncs_jtp


class PSSetup(_PASDWData):
    def get_pfunc(self, n, r_g):
        return get_pfunc_norm(n, r_g, self.rcut)

    def _setup_pfuncs_and_ffuncs(self):
        nn, nj, ng, nt, r_g, dr_g = self.get_setup_vals()
        interp_r_g = self.interp_rgd.r_g
        rgd = self.paw_rgd
        dv_g = get_dv(rgd)
        ffuncs_jg = np.zeros((nj, ng))
        exact_ovlp_pf = np.zeros((self.ni, self.ni))
        ilist = np.arange(self.ni)

        pfuncs_nt = np.empty((nn, nt), dtype=np.float64, order="C")
        pfuncs_ng = np.empty((nn, ng), dtype=np.float64, order="C")
        for n in range(nn):
            pfuncs_nt[n, :] = self.get_pfunc(n, interp_r_g)
            pfuncs_ng[n, :] = self.get_pfunc(n, r_g)
        pfuncs_ntp = self.map_many_to_spline(pfuncs_nt)

        for j in range(self.nj):
            n = self.nlist_j[j]
            l = self.llist_j[j]
            ffuncs_jg[j] = pfuncs_ng[n]
            i0s = ilist[self.jlist_i == j]
            for j1 in range(self.nj):
                if self.llist_j[j1] == l:
                    i1s = ilist[self.jlist_i == j1]
                    n1 = self.nlist_j[j1]
                    ovlp = np.dot(pfuncs_ng[n] * pfuncs_ng[n1], dv_g)
                    for i0, i1 in zip(i0s.tolist(), i1s.tolist()):
                        exact_ovlp_pf[i0, i1] = ovlp

        self.exact_ovlp_pf = exact_ovlp_pf
        self.pfuncs_ng = pfuncs_ng
        self.pfuncs_ntp = pfuncs_ntp
        self.ffuncs_jg = ffuncs_jg
        self.ffuncs_jtp = pfuncs_ntp.copy()


class CiderCoreTermProjector:
    def __init__(
        self, gplan, aplan, smooth_kbasis, smooth_rbasis, loc_kbasis, loc_rbasis
    ):
        # all electron density plan (from the atomic grid)
        self.aplan: NLDFAuxiliaryPlan = aplan
        # pseudo-density plan (from the FFT grid)
        self.gplan: NLDFAuxiliaryPlan = gplan
        # smooth basis on reciprocal space grid
        self.smooth_kbasis: RFC = smooth_kbasis
        # smooth basis on real-space grid
        self.smooth_rbasis: RFC = smooth_rbasis
        # localized basis on reciprocal space grid
        self.loc_kbasis: RFC = loc_kbasis
        # localized basis on real-space grid
        self.loc_rbasis: RFC = loc_rbasis
        # weights on beta channels for fitting
        self.w_b = np.ones(self.alphas_ae.size) / self.alpha_norms_ae**2
        self._check_rfcs()
        # TODO version i
        self._initialize()

    @property
    def alphas_ae(self):
        return self.aplan.alphas

    @property
    def alpha_norms_ae(self):
        return self.aplan.alpha_norms

    @property
    def alphas_ps(self):
        return self.gplan.alphas

    @property
    def alpha_norms_ps(self):
        return self.gplan.alpha_norms

    @property
    def k_k(self):
        return self.loc_kbasis.r_g

    @property
    def dv_k(self):
        return self.loc_kbasis.dv_g

    @property
    def r_g(self):
        return self.loc_rbasis.r_g

    @property
    def dv_g(self):
        return self.loc_rbasis.dv_g

    @property
    def lmax(self):
        return self.loc_kbasis.lmax

    @property
    def nu(self):
        return self.loc_kbasis.nu

    @property
    def ni(self):
        return self.smooth_kbasis.nu

    def _check_rfcs(self):
        # make sure lmaxs and basis sizes are the same
        assert (self.loc_kbasis.iloc_l == self.loc_rbasis.iloc_l).all()
        assert self.loc_kbasis.lmax == self.smooth_kbasis.lmax
        assert self.loc_rbasis.lmax == self.smooth_kbasis.lmax

        # make sure grids are compatible
        k1_g = self.smooth_kbasis.r_g
        k2_g = self.loc_kbasis.r_g
        assert (np.abs(k1_g - k2_g) < 1e-10).all()

        # make sure things are orthogonal
        jloc_l = self.loc_kbasis.iloc_l
        for l in range(self.lmax + 1):
            j0, j1 = jloc_l[l], jloc_l[l + 1]
            lfuncs_jk = self.loc_kbasis.funcs_ig[j0:j1]
            ovlp = np.einsum("ik,jk,k->ij", lfuncs_jk, lfuncs_jk, self.dv_k)
            assert (np.abs(ovlp - np.identity(ovlp.shape[0])) < 1e-4).all()
            lfuncs_jg = self.loc_rbasis.funcs_ig[j0:j1]
            ovlp = np.einsum("ig,jg,g->ij", lfuncs_jg, lfuncs_jg, self.dv_g)
            assert (np.abs(ovlp - np.identity(ovlp.shape[0])) < 1e-4).all()

    def _get_p21_matrix(self):
        p21_l_javb = []
        jloc_l = self.smooth_kbasis.iloc_l
        vloc_l = self.loc_kbasis.iloc_l
        kernel_kba = self._kernel_kba
        for l in range(self.lmax + 1):
            lsmooth_jk = self.smooth_kbasis.funcs_ig[jloc_l[l] : jloc_l[l + 1]]
            lloc_vk = self.loc_kbasis.funcs_ig[vloc_l[l] : vloc_l[l + 1]]
            phi_jabk = np.einsum("jk,kba->jabk", lsmooth_jk, kernel_kba)
            p21_javb = np.einsum("vk,jabk,k->javb", lloc_vk, phi_jabk, self.dv_k)
            p21_javb[:] *= self.w_b
            p21_l_javb.append(p21_javb)
        return p21_l_javb

    def _get_p22_matrix(self, reg=0):
        kernel_kba = self._kernel_kba
        kernel_aak = np.einsum("kba,kbc,b->ack", kernel_kba, kernel_kba, self.w_b)
        kernel_aak[:] *= self.dv_k
        p22_l_jaja = []
        jloc_l = self.smooth_kbasis.iloc_l
        for l in range(self.lmax + 1):
            lfuncs_jk = self.smooth_kbasis.funcs_ig[jloc_l[l] : jloc_l[l + 1]]
            p22_jaja = np.einsum("abk,ik,jk->iajb", kernel_aak, lfuncs_jk, lfuncs_jk)
            p22_jaja = np.ascontiguousarray(p22_jaja)
            nj, na = p22_jaja.shape[:2]
            p22_ii = p22_jaja.view()
            p22_ii.shape = (nj * na, nj * na)
            my_reg = reg / np.tile(self.w_b[:na], nj)
            p22_ii[:] += my_reg * np.identity(nj * na)
            p22_l_jaja.append(p22_jaja)
        return p22_l_jaja

    def _combine_matrices(self, p21_l, p22_l, reg=0):
        z_l = []
        invw = 1.0 / self.w_b / (1 + reg)
        for l in range(len(p21_l)):
            p21 = p21_l[l].copy()
            p21iw = p21 * invw
            p22 = p22_l[l].copy()
            nj, na, nv, nb = p21.shape
            p21.shape = (nj * na, nv * nb)
            p21iw.shape = (nj * na, nv * nb)
            p22.shape = (nj * na, nj * na)
            m = p22 - (p21iw).dot(p21.T)
            mqtinvd = -1.0 * p21iw
            v = np.append(mqtinvd, np.identity(m.shape[0]), axis=1)
            u = cholesky(m, lower=True)
            z = np.linalg.solve(u, v)
            z_l.append(z)
        return z_l

    def _initialize(self):
        expnts, facs, k2pows = get_convolution_data(
            self.gplan.nldf_settings,
            self.gplan.alphas,
            self.gplan.alpha_norms,
            self.aplan.alphas,
            self.aplan.alpha_norms,
        )
        nb, na = expnts.shape
        k2_g = self.k_k * self.k_k
        self._kernel_kba = facs * np.exp(-expnts * k2_g[:, None, None])
        # self._kernel_bak[na:] = 0.0
        p21 = self._get_p21_matrix()
        p22 = self._get_p22_matrix(reg=1e-8)
        self._zmat_l = self._combine_matrices(p21, p22, reg=1e-8)

    def get_c_and_df(self, y_skLb):
        t0 = time.monotonic()
        y_skLb = y_skLb * self.dv_k[:, None, None]
        t1 = time.monotonic()
        yy_skLa = np.einsum("sklb,kba,b->skla", y_skLb, self._kernel_kba, self.w_b)
        t2 = time.monotonic()
        nspin, nk, nlm, nb = y_skLb.shape
        na = yy_skLa.shape[-1]
        c_sia = np.zeros((nspin, self.ni, na))
        df_sub = np.zeros((nspin, self.nu, nb))
        all_b1_sub = self.loc_kbasis.grid2basis_spin(y_skLb)
        all_b2_sia = self.smooth_kbasis.grid2basis_spin(yy_skLa)
        for s in range(nspin):
            for l in range(self.lmax + 1):
                nm = 2 * l + 1
                for m in range(nm):
                    uloc_l = self.loc_kbasis.uloc_l
                    slice1 = slice(uloc_l[l] + m, uloc_l[l + 1] + m, nm)
                    b1 = all_b1_sub[s][slice1] * self.w_b
                    uloc_l = self.smooth_kbasis.uloc_l
                    slice2 = slice(uloc_l[l] + m, uloc_l[l + 1] + m, nm)
                    b2 = all_b2_sia[s][slice2]
                    b = self.cat_b_vecs(b1, b2)
                    x = self._zmat_l[l].dot(b)
                    x = self._zmat_l[l].T.dot(x)
                    n1 = b1.size
                    df_sub[s, slice1] += x[:n1].reshape(-1, nb)
                    c_sia[s, slice2] += x[n1:].reshape(-1, na)
        df_sub[:] += all_b1_sub
        df_sgLb = self.loc_rbasis.basis2grid_spin(df_sub)
        dxt_sgLa = self.smooth_kbasis.basis2grid_spin(c_sia)
        t3 = time.monotonic()
        print("TIMES", t1 - t0, t2 - t1, t3 - t2)
        return c_sia, dxt_sgLa, df_sgLb

    def get_vy_and_vyy(self, vc_sia, vdxt_skLa, vdf_sgLb):
        vc_sia[:] += self.smooth_kbasis.grid2basis_spin(vdxt_skLa)
        vdf_sub = self.loc_rbasis.grid2basis_spin(vdf_sgLb)
        nspin, ni, na = vc_sia.shape
        nu, nb = vdf_sub.shape[1:]
        all_vb1_sub = vdf_sub.copy()
        all_vb2_sia = np.zeros_like(vc_sia)
        for s in range(nspin):
            for l in range(self.lmax + 1):
                nm = 2 * l + 1
                for m in range(nm):
                    uloc_l = self.loc_kbasis.uloc_l
                    slice1 = slice(uloc_l[l] + m, uloc_l[l + 1] + m, nm)
                    x1 = vdf_sub[s, slice1].ravel()
                    uloc_l = self.smooth_kbasis.uloc_l
                    slice2 = slice(uloc_l[l] + m, uloc_l[l + 1] + m, nm)
                    x2 = vc_sia[s, slice2].ravel()
                    x = self.cat_b_vecs(x1, x2)
                    b = self._zmat_l[l].dot(x)
                    b = self._zmat_l[l].T.dot(b)
                    n1 = x1.size
                    all_vb1_sub[s][slice1] += self.w_b * b[:n1].reshape(-1, nb)
                    all_vb2_sia[s][slice2] += b[n1:].reshape(-1, na)
        vy_skLb = self.loc_kbasis.basis2grid_spin(all_vb1_sub)
        vyy_skLa = self.smooth_kbasis.basis2grid_spin(all_vb2_sia)
        t0 = time.monotonic()
        vy_skLb[:] += np.einsum(
            "skla,kba,b->sklb", vyy_skLa, self._kernel_kba, self.w_b
        )
        t1 = time.monotonic()
        print("VY_TIME", t1 - t0)
        vy_skLb[:] *= self.dv_k[:, None, None]
        return vy_skLb

    def cat_b_vecs(self, b1, b2):
        return np.append(b1.flatten(), b2.flatten())

    def coef_to_real_smooth(self, c_siq, out=None, check_q=True, use_ffuncs=False):
        pass

    def real_to_coef_smooth(self, xt_sgLq, out=None, check_q=True, use_ffuncs=False):
        pass

    def recip_to_real(self, x_skLq):
        pass

    def real_to_recip(self, x_sgLq):
        pass

    @classmethod
    def from_atco_and_setup(
        cls,
        gplan: NLDFAuxiliaryPlan,
        aplan: NLDFAuxiliaryPlan,
        ps_setup: PSSetup,
        atco: ATCBasis,
        sbt_rgd: SBTFullGridDescriptor,
        pmax: float,
    ):
        lmax = max(np.max(ps_setup.llist_j), np.max(atco.bas[:, ANG_OF]))
        betas_lv = _get_beta_basis(atco, ps_setup.rcut, pmax, lmax)
        vcount_l = [len(b) for b in betas_lv]
        vloc_l = np.append([0], np.cumsum(vcount_l)).astype(np.int32)
        rcut = ps_setup.rcut
        paw_dv_g = ps_setup.paw_rgd.dr_g * ps_setup.paw_rgd.r_g**2
        dv_g = sbt_rgd.dr_g * sbt_rgd.r_g**2
        dv_k = 2 / np.pi * sbt_rgd.dk_g * sbt_rgd.k_g**2
        pawbas_l_vg = _get_loc_basis(betas_lv, rcut, ps_setup.paw_rgd.r_g)
        sbtbas_l_vg = _get_loc_basis(betas_lv, rcut, sbt_rgd.r_g)
        pawbas_l_vg = _orthogonalize_basis(pawbas_l_vg, dv_g=paw_dv_g)[0]
        sbtbas_l_vg = _orthogonalize_basis(sbtbas_l_vg, dv_g=dv_g)[0]
        sbtbas_l_vk = []
        for l in range(len(sbtbas_l_vg)):
            sbtbas_vk = np.empty_like(sbtbas_l_vg[l])
            for v in range(sbtbas_vk.shape[0]):
                sbtbas_vk[v] = sbt_rgd.transform_single_fwd(sbtbas_l_vg[l][v], l)
            sbtbas_l_vk.append(sbtbas_vk)
        # pawbas_l_vg, coeff_l_vv = _orthogonalize_basis(pawbas_l_vg, dv_g=paw_dv_g)
        # sbtbas_l_vk, coeff_l_vv = _orthogonalize_basis(sbtbas_l_vk, coeff_l_pp=coeff_l_vv)
        loc_basis_pg = np.concatenate(pawbas_l_vg, axis=0)
        loc_basis_pk = np.concatenate(sbtbas_l_vk, axis=0)
        pfuncs_jg = [ps_setup.get_pfunc(n, sbt_rgd.r_g) for n in ps_setup.nlist_j]
        pfuncs_jk = []
        for j, pfunc_g in enumerate(pfuncs_jg):
            pfuncs_jk.append(sbt_rgd.transform_single_fwd(pfunc_g, ps_setup.llist_j[j]))
        pfuncs_jk = np.asarray(pfuncs_jk)
        pfuncs_jg = np.asarray([ps_setup.pfuncs_ng[n] for n in ps_setup.nlist_j])
        smooth_kbasis = RFC(ps_setup.jloc_l, pfuncs_jk, sbt_rgd.k_g, dv_k)
        smooth_rbasis = RFC(ps_setup.jloc_l, pfuncs_jg, ps_setup.paw_rgd.r_g, paw_dv_g)
        loc_rbasis = RFC(vloc_l, loc_basis_pg, ps_setup.paw_rgd.r_g, paw_dv_g)
        loc_kbasis = RFC(vloc_l, loc_basis_pk, sbt_rgd.k_g, dv_k)
        return cls(gplan, aplan, smooth_kbasis, smooth_rbasis, loc_kbasis, loc_rbasis)


def _get_beta_basis(atco, rcut, pmax, lmax):
    ls = atco.bas[:, ANG_OF]
    coefs = atco.env[atco.bas[:, PTR_EXP]]
    betas_lv = []
    for l in range(lmax + 1):
        betas_lv.append(coefs[ls == l])
    pmin = 2 / rcut**2
    if pmax is None:
        raise ValueError
    else:
        pmax = 2 * pmax
    for l in range(lmax + 1):
        cond = np.logical_and(
            betas_lv[l] > pmin,
            betas_lv[l] < pmax,
        )
        betas_lv[l] = betas_lv[l][cond]
    return betas_lv


def _gauss_and_derivs(a, r):
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


def _get_loc_basis(betas_lv, rcut, r_g):
    delta_l_pg = []
    for l in range(len(betas_lv)):
        betas = betas_lv[l]
        fcut = (0.5 * np.pi / betas) ** 0.75 * _gauss_and_derivs(betas, rcut)
        funcs = (0.5 * np.pi / betas[:, None]) ** 0.75 * np.exp(
            -betas[:, None] * r_g * r_g
        )
        rd_g = r_g - rcut
        poly = (
            fcut[0, :, None]
            + rd_g * fcut[1, :, None]
            + rd_g**2 * fcut[2, :, None] / 2
            + rd_g**3 * fcut[3, :, None] / 6
        )
        dpoly = fcut[1] - rcut * fcut[2] + rcut**2 * fcut[3] / 2
        poly += dpoly[:, None] / (4 * rcut**3) * rd_g**4
        funcs = funcs - poly
        funcs[:, r_g > rcut] = 0
        funcs[:] *= r_g**l
        delta_l_pg.append(funcs)
    return delta_l_pg


def _get_loc_basis_grad_ovlp(betas_lv, rcut, r_g, dr_g):
    dv_g = r_g * r_g * dr_g
    coeff_l_pp = []
    for l in range(len(betas_lv)):
        betas = betas_lv[l]
        fcut = (0.5 * np.pi / betas) ** 0.75 * _gauss_and_derivs(betas, rcut)
        funcs = (0.5 * np.pi / betas[:, None]) ** 0.75 * np.exp(
            -betas[:, None] * r_g * r_g
        )
        gfuncs = funcs * (-2 * betas[:, None] * r_g)
        rd_g = r_g - rcut
        poly = (
            fcut[0, :, None]
            + rd_g * fcut[1, :, None]
            + rd_g**2 * fcut[2, :, None] / 2
            + rd_g**3 * fcut[3, :, None] / 6
        )
        gpoly = (
            +fcut[1, :, None]
            + rd_g * fcut[2, :, None]
            + rd_g**2 * fcut[3, :, None] / 2
        )
        dpoly = fcut[1] - rcut * fcut[2] + rcut**2 * fcut[3] / 2
        poly += dpoly[:, None] / (4 * rcut**3) * rd_g**4
        gpoly += dpoly[:, None] / (rcut**3) * rd_g**3
        funcs = funcs - poly
        funcs[:, r_g > rcut] = 0
        gfuncs[:] *= r_g**l
        if l > 0:
            gfuncs[:] += l * r_g ** (l - 1) * funcs
        funcs[:] *= r_g**l
        # spherical harmonic term
        funcs_tmp = funcs * np.sqrt(dr_g * l * (l + 1))
        # radial term
        gfuncs_tmp = gfuncs * np.sqrt(dv_g)
        ovlp = np.einsum("ig,jg->ij", funcs_tmp, funcs_tmp)
        ovlp += np.einsum("ig,jg->ij", gfuncs_tmp, gfuncs_tmp)
        L = cholesky(ovlp, lower=True)
        coeff_l_pp.append(L)
    return coeff_l_pp


def _orthogonalize_basis(delta_l_pg, coeff_l_pp=None, dv_g=None):
    orth_l_pg = []
    if coeff_l_pp is None:
        coeff_l_pp = []
        assert dv_g is not None
        calc_coeff = True
    else:
        calc_coeff = False
    for l in range(len(delta_l_pg)):
        if calc_coeff:
            funcs_tmp = delta_l_pg[l] * np.sqrt(dv_g)
            ovlp = np.einsum("ig,jg->ij", funcs_tmp, funcs_tmp)
            L = cholesky(ovlp, lower=True)
            coeff_l_pp.append(L)
        orth_l_pg.append(np.linalg.solve(coeff_l_pp[l], delta_l_pg[l]))
    return orth_l_pg, coeff_l_pp


class RFC:
    def __init__(self, iloc_l, funcs_ig, r_g, dv_g, lmax=None):
        self.funcs_ig = np.ascontiguousarray(funcs_ig)
        self.lmax = len(iloc_l) - 2
        if lmax is not None:
            assert lmax >= self.lmax
            self.lmax = lmax
            while len(iloc_l) < lmax + 2:
                iloc_l.append(iloc_l[-1])
        self.r_g = r_g
        self.dv_g = dv_g
        self.nu = 0
        self.uloc_l = [0]
        self.iloc_l = np.asarray(iloc_l, order="C", dtype=np.int32)
        for l in range(self.iloc_l.size - 1):
            nj = self.iloc_l[l + 1] - self.iloc_l[l]
            self.nu += (2 * l + 1) * nj
            self.uloc_l.append(self.nu)
        self.uloc_l = np.asarray(self.uloc_l, order="C", dtype=np.int32)

    @property
    def nlm(self):
        return (self.lmax + 1) * (self.lmax + 1)

    def basis2grid(self, p_uq, p_rlmq, fwd=True):
        assert p_uq.flags.c_contiguous
        assert p_uq.ndim == 2
        nalpha = p_uq.shape[1]
        assert p_uq.shape[0] == self.nu
        ngrid = self.funcs_ig.shape[1]
        if p_rlmq is None:
            p_rlmq = np.zeros((ngrid, self.nlm, nalpha))
        else:
            assert p_rlmq.shape == (ngrid, self.nlm, nalpha)
            assert p_rlmq.flags.c_contiguous
        iloc_l = np.asarray(self.iloc_l, dtype=np.int32, order="C")
        uloc_l = np.asarray(self.uloc_l, dtype=np.int32, order="C")
        if fwd:
            fn = libcider.contract_orb_to_rad_num
        else:
            fn = libcider.contract_rad_to_orb_num
        fn(
            p_rlmq.ctypes.data_as(ctypes.c_void_p),
            p_uq.ctypes.data_as(ctypes.c_void_p),
            self.funcs_ig.ctypes.data_as(ctypes.c_void_p),
            iloc_l.ctypes.data_as(ctypes.c_void_p),
            uloc_l.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(ngrid),
            ctypes.c_int(self.nlm),
            ctypes.c_int(nalpha),
        )

    def basis2grid_spin(self, p_suq, p_srlmq=None):
        assert p_suq.flags.c_contiguous
        assert p_suq.ndim == 3
        nspin = p_suq.shape[0]
        nalpha = p_suq.shape[2]
        assert p_suq.shape[1] == self.nu
        ngrid = self.funcs_ig.shape[1]
        if p_srlmq is None:
            p_srlmq = np.zeros((nspin, ngrid, self.nlm, nalpha))
        for s in range(nspin):
            self.basis2grid(p_suq[s], p_srlmq[s], fwd=True)
        return p_srlmq

    def grid2basis_spin(self, p_srlmq, p_suq=None):
        assert p_srlmq.flags.c_contiguous
        assert p_srlmq.ndim == 4
        nspin = p_srlmq.shape[0]
        nalpha = p_srlmq.shape[3]
        assert p_srlmq.shape[1] == self.funcs_ig.shape[1]
        assert p_srlmq.shape[2] == self.nlm
        if p_suq is None:
            p_suq = np.zeros((nspin, self.nu, nalpha))
        for s in range(nspin):
            self.basis2grid(p_suq[s], p_srlmq[s], fwd=False)
        return p_suq
