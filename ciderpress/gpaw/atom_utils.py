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
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor
from gpaw.sphere.lebedev import R_nv, Y_nL, weight_n
from gpaw.utilities.timing import Timer
from gpaw.xc.pawcorrection import rnablaY_nLv
from gpaw.xc.vdw import spline
from scipy.linalg import cho_factor, cho_solve

from ciderpress.data import get_hgbs_max_exps
from ciderpress.dft import pwutil
from ciderpress.dft.density_util import get_sigma
from ciderpress.dft.grids_indexer import AtomicGridsIndexer
from ciderpress.dft.lcao_convolutions import (
    ANG_OF,
    PTR_COEFF,
    PTR_EXP,
    ATCBasis,
    ConvolutionCollection,
    get_etb_from_expnt_range,
    get_gamma_lists_from_etb_list,
)
from ciderpress.dft.plans import NLDFAuxiliaryPlan, get_ccl_settings, libcider
from ciderpress.gpaw.fit_paw_gauss_pot import (
    construct_full_p_matrices,
    get_delta_lpg,
    get_dv,
    get_dvk,
    get_ffunc,
    get_p11_matrix,
    get_p12_p21_matrix,
    get_p22_matrix,
    get_pfunc_norm,
    get_pfuncs_k,
    get_phi_iabg,
    get_phi_iabk,
    get_poly,
)
from ciderpress.gpaw.gpaw_grids import SBTFullGridDescriptor
from ciderpress.gpaw.interp_paw import (
    DiffPAWXCCorrection,
    calculate_cider_paw_correction,
)

USE_GAUSSIAN_PAW_CONV = False
DEFAULT_CIDER_PAW_ALGO = "v1"
USE_GAUSSIAN_SBT = True
# if USE_GAUSSIAN_PAW_CONV is True, USE_GAUSSIAN_SBT must be True
USE_GAUSSIAN_SBT = USE_GAUSSIAN_SBT or USE_GAUSSIAN_PAW_CONV


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

    def __init__(
        self, plan, cider_kernel, atco, xcc, gplan, paw_algo=DEFAULT_CIDER_PAW_ALGO
    ):
        self.plan = plan
        self.cider_kernel = cider_kernel
        self._atco = atco
        self.w_g = (xcc.rgd.dv_g[:, None] * weight_n).ravel()
        self.r_g = xcc.rgd.r_g
        self.xcc = xcc
        # TODO might want more control over grids_indexer nlm,
        # currently it is just controlled by size of Y_nL in GPAW.
        self.grids_indexer = AtomicGridsIndexer.make_single_atom_indexer(Y_nL, self.r_g)
        self.grids_indexer.set_weights(self.w_g)
        self._paw_algo = paw_algo
        self.timer = Timer()

    @property
    def nlm(self):
        return self.grids_indexer.nlm

    @classmethod
    def from_plan(cls, plan, gplan, cider_kernel, Z, xcc, paw_algo, beta=1.8):
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
        return cls(plan, cider_kernel, atco, xcc, gplan, paw_algo)

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

    def calculate_y_terms_v1(self, xt_sgLq, dx_sgLq, psetup):
        na = psetup.alphas.size
        self.sbt_rgd = psetup.sbt_rgd
        dx_sxq = self.grid2aux(dx_sgLq)
        dy_sxq = self.perform_convolution_fwd(dx_sxq)
        dy1_sxq = dy_sxq[..., :na] * psetup.w_b
        dy2_sxq = dy_sxq[..., na:]
        yy_sxq = self.perform_fitting_convolution_bwd(dy1_sxq)

        df_sgLq = np.zeros_like(dx_sgLq)
        c_siq, df_sgLq[..., :na] = psetup.get_c_and_df(dy1_sxq, yy_sxq)
        df_sgLq[..., na:] = psetup.get_df_only(dy2_sxq)
        xt_sgLq[..., :na] += (
            psetup.coef_to_real(c_siq) * get_dv(psetup.slrgd)[:, None, None]
        )
        xt_sxq = self.grid2aux(xt_sgLq)
        fr_sxq = self.perform_convolution_fwd(xt_sxq)
        fr_sgLq = self.aux2grid(fr_sxq)
        return fr_sgLq, df_sgLq, c_siq

    def calculate_vx_terms_v1(self, vfr_sgLq, vdf_sgLq, vc_siq, psetup):
        na = psetup.alphas.size
        self.sbt_rgd = psetup.sbt_rgd
        vfr_sxq = self.grid2aux(vfr_sgLq)
        vxt_sxq = self.perform_convolution_bwd(vfr_sxq)
        vxt_sgLq = self.aux2grid(vxt_sxq)
        vdy2_sxq = psetup.get_vdf_only(np.ascontiguousarray(vdf_sgLq[..., na:]))
        vc_siq[:] += psetup.real_to_coef(
            vxt_sgLq[..., :na] * get_dv(psetup.slrgd)[:, None, None]
        )
        vdy1_sxq, vyy_sxq = psetup.get_vy_and_vyy(vc_siq, vdf_sgLq[..., :na])

        vdy1_sxq[:] += self.perform_fitting_convolution_fwd(vyy_sxq)
        vdy_sxq = np.concatenate([vdy1_sxq * psetup.w_b, vdy2_sxq], axis=-1)
        vdx_sxq = self.perform_convolution_bwd(vdy_sxq)
        vdx_sgLq = self.aux2grid(vdx_sxq)
        return vxt_sgLq, vdx_sgLq

    def calculate_y_terms_v2(self, xt_sgLq, dx_sgLq, psetup):
        na = psetup.alphas.size
        self.sbt_rgd = psetup.sbt_rgd
        c_siq = psetup.real_to_coef(
            np.ascontiguousarray(dx_sgLq[..., :na]), use_ffuncs=True
        )
        dxt_sgLq = (
            psetup.coef_to_real(c_siq, use_ffuncs=False)
            * get_dv(psetup.slrgd)[:, None, None]
        )
        xt_sgLq[..., :na] += dxt_sgLq
        dx_sgLq[..., :na] -= dxt_sgLq

        xt_sxq = self.grid2aux(xt_sgLq)
        yt_sxq = self.perform_convolution_fwd(xt_sxq)
        fr_sgLq = self.aux2grid(yt_sxq)
        dx_sxq = self.grid2aux(dx_sgLq)
        dy_sxq = self.perform_convolution_fwd(dx_sxq)
        df_sgLq = psetup.get_df_only(dy_sxq)
        return fr_sgLq, df_sgLq, c_siq

    def calculate_vx_terms_v2(self, vfr_sgLq, vdf_sgLq, vc_siq, psetup):
        na = psetup.alphas.size
        self.sbt_rgd = psetup.sbt_rgd
        vdy_sxq = psetup.get_vdf_only(np.ascontiguousarray(vdf_sgLq))
        vdx_sxq = self.perform_convolution_bwd(vdy_sxq)
        vdx_sgLq = self.aux2grid(vdx_sxq)
        vyt_sxq = self.grid2aux(vfr_sgLq)
        vxt_sxq = self.perform_convolution_bwd(vyt_sxq)
        vxt_sgLq = self.aux2grid(vxt_sxq)

        vc_siq = vc_siq + psetup.real_to_coef(
            (vxt_sgLq - vdx_sgLq)[..., :na] * get_dv(psetup.slrgd)[:, None, None],
            use_ffuncs=False,
        )
        vdx_sgLq[..., :na] += psetup.coef_to_real(vc_siq, use_ffuncs=True)
        return vxt_sgLq, vdx_sgLq

    def calculate_y_terms(self, xt_sgLq, dx_sgLq, psetup):
        if self._paw_algo == "v2":
            return self.calculate_y_terms_v2(xt_sgLq, dx_sgLq, psetup)
        else:
            return self.calculate_y_terms_v1(xt_sgLq, dx_sgLq, psetup)

    def calculate_vx_terms(self, vfr_sgLq, vdf_sgLq, vc_siq, psetup):
        if self._paw_algo == "v2":
            return self.calculate_vx_terms_v2(vfr_sgLq, vdf_sgLq, vc_siq, psetup)
        else:
            return self.calculate_vx_terms_v1(vfr_sgLq, vdf_sgLq, vc_siq, psetup)


class PAWCiderContribsRecip(_PAWCiderContribs):
    def __init__(self, plan, cider_kernel, atco, xcc, gplan, paw_algo):
        super(PAWCiderContribsRecip, self).__init__(
            plan, cider_kernel, atco, xcc, gplan, paw_algo
        )
        self._atco_recip = self._atco.get_reciprocal_atco()
        self._galphas = gplan.alphas.copy()
        self._gnorms = gplan.alpha_norms.copy()

    def r2k(self, in_sxLq, rgd, fwd=True):
        # TODO need to work out atco in and out
        # when nldf is not vj
        if fwd:
            atco_inp = self.atco_inp
            atco_out = self._atco_recip
            in_g = self.grids_indexer.rad_arr
            out_g = rgd.k_g
        else:
            atco_inp = self._atco_recip
            atco_out = self.atco_feat
            in_g = rgd.k_g
            out_g = self.grids_indexer.rad_arr
        in_sxLq = np.ascontiguousarray(in_sxLq)
        nspin, nx, nlm, nq = in_sxLq.shape
        nx = out_g.size
        shape = (nspin, nx, nlm, nq)
        out_sxLq = np.zeros(shape)
        y = Y_nL[:, :nlm]
        # TODO make the indexers once in advance and store them
        in_indexer = AtomicGridsIndexer.make_single_atom_indexer(y, in_g)
        out_indexer = AtomicGridsIndexer.make_single_atom_indexer(y, out_g)
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
            libcider.solve_atc_coefs_arr(
                self.atco_inp.atco_c_ptr,
                tmp_uq.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(tmp_uq.shape[-1]),
            )
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
        return self.r2k(f_sgLq, self.sbt_rgd, fwd=True)

    def aux2grid(self, f_sxq):
        f_sxq = f_sxq * get_dvk(self.sbt_rgd)[:, None, None]
        return self.r2k(f_sxq, self.sbt_rgd, fwd=False)

    def x2u(self, f_skLq):
        nspin = f_skLq.shape[0]
        f_skLq = f_skLq * get_dvk(self.sbt_rgd)[:, None, None]
        atco_inp = self._atco_recip
        rgd = self.sbt_rgd
        k_g = rgd.k_g
        nspin, nk, nlm, nq = f_skLq.shape
        y = Y_nL[:, :nlm]
        out_suq = np.zeros((nspin, atco_inp.nao, nq))
        indexer = AtomicGridsIndexer.make_single_atom_indexer(y, k_g)
        for s in range(nspin):
            atco_inp.convert_rad2orb_(
                f_skLq[s],
                out_suq[s],
                indexer,
                indexer.rad_arr,
                rad2orb=True,
                offset=0,
            )
            libcider.solve_atc_coefs_arr(
                self.atco_inp.atco_c_ptr,
                out_suq[s].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(out_suq.shape[-1]),
            )
        return out_suq

    def u2x(self, f_suq):
        nspin, nu, nq = f_suq.shape
        atco_inp = self._atco_recip
        rgd = self.sbt_rgd
        k_g = rgd.k_g
        nk = k_g.size
        nlm = self.nlm
        out_skLq = np.zeros((nspin, nk, nlm, nq))
        y = Y_nL[:, :nlm]
        indexer = AtomicGridsIndexer.make_single_atom_indexer(y, k_g)
        for s in range(nspin):
            libcider.solve_atc_coefs_arr(
                self.atco_inp.atco_c_ptr,
                f_suq[s].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(f_suq.shape[-1]),
            )
            atco_inp.convert_rad2orb_(
                out_skLq[s],
                f_suq[s],
                indexer,
                indexer.rad_arr,
                rad2orb=False,
                offset=0,
            )
        return out_skLq

    def calc_conv_skLq(self, in_skLq, rgd, alphas=None, alpha_norms=None):
        in_skLq = np.ascontiguousarray(in_skLq)
        out_skLq = np.empty_like(in_skLq)
        k_g = np.ascontiguousarray(rgd.k_g)
        nspin, nk, nlm, nq = in_skLq.shape
        if alphas is None:
            alphas = self.plan.alphas
            alpha_norms = self.plan.alpha_norms
        assert nk == k_g.size
        assert nq == alphas.size
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
        return out_skLq

    def perform_convolution_fwd(self, f_skLq):
        return self.calc_conv_skLq(f_skLq, self.sbt_rgd)

    def perform_convolution_bwd(self, f_skLq):
        return self.calc_conv_skLq(f_skLq, self.sbt_rgd)

    def perform_fitting_convolution_bwd(self, f_skLq):
        return self.calc_conv_skLq(
            f_skLq,
            self.sbt_rgd,
            alphas=self._galphas,
            alpha_norms=self._gnorms,
        )

    perform_fitting_convolution_fwd = perform_fitting_convolution_bwd


class PAWCiderContribsOrb(_PAWCiderContribs):
    def __init__(self, plan, cider_kernel, atco, xcc, gplan, paw_algo):
        super(PAWCiderContribsOrb, self).__init__(
            plan, cider_kernel, atco, xcc, gplan, paw_algo
        )
        has_vj, ifeat_ids = get_ccl_settings(plan)
        # TODO currently assuming ccl.atco_inp == ccl.atco_out,
        # but this is not checked anywhere
        ccl = ConvolutionCollection(
            atco, atco, plan.alphas, plan.alpha_norms, has_vj, ifeat_ids
        )
        self.ccl = ccl
        self.ccl.compute_integrals_()
        self.ccl2 = ConvolutionCollection(
            self.ccl.atco_inp,
            self.ccl.atco_out,
            gplan.alphas,
            gplan.alpha_norms,
            ccl._has_vj,
            ccl._ifeat_ids,
        )
        self.ccl2.compute_integrals_()
        self.ccl.solve_projection_coefficients()

    def perform_convolution(
        self,
        theta_sgLq,
        fwd=True,
        out_sgLq=None,
        return_orbs=False,
        take_orbs=False,
        indexer_in=None,
        indexer_out=None,
    ):
        theta_sgLq = np.ascontiguousarray(theta_sgLq)
        nspin = self.nspin
        nalpha = self.ccl.nalpha
        nbeta = self.ccl.nbeta
        is_not_vi = self.plan.nldf_settings.nldf_type != "i"
        inp_shape = (nspin, self.ccl.atco_inp.nao, nalpha)
        out_shape = (nspin, self.ccl.atco_out.nao, nbeta)
        if indexer_in is None:
            indexer_in = self.grids_indexer
        if indexer_out is None:
            indexer_out = self.grids_indexer
        if out_sgLq is None and not return_orbs:
            if fwd:
                nout = nbeta
            else:
                nout = nalpha
            out_sgLq = np.stack(
                [indexer_out.empty_rlmq(nalpha=nout) for s in range(nspin)]
            )
            out_sgLq[:] = 0.0
        if fwd:
            i_inp = -1
            i_out = 0
            if take_orbs:
                theta_suq = theta_sgLq
                assert theta_suq.shape == inp_shape
            else:
                theta_suq = np.zeros(inp_shape, order="C")
            conv_svq = np.zeros(out_shape, order="C")
        else:
            i_inp = 0
            i_out = -1
            if take_orbs:
                theta_suq = theta_sgLq
                assert theta_suq.shape == out_shape
            else:
                theta_suq = np.zeros(out_shape, order="C")
            conv_svq = np.zeros(inp_shape, order="C")
        if not take_orbs:
            self.convert_rad2orb_(
                nspin, theta_sgLq, theta_suq, rad2orb=True, inp=fwd, indexer=indexer_in
            )
        for s in range(nspin):
            # TODO need to be careful about this get_transformed_interpolation_terms
            # call. It is currently no-op due to the use of cubic splines rather than
            # Gaussian interpolation, but it could be a problem with the fitting
            # procedure later on.
            if (fwd) or (is_not_vi):
                self.plan.get_transformed_interpolation_terms(
                    theta_suq[s, :, : self.plan.nalpha], i=i_inp, fwd=fwd, inplace=True
                )
            self.ccl.multiply_atc_integrals(theta_suq[s], output=conv_svq[s], fwd=fwd)
            if (not fwd) or (is_not_vi):
                self.plan.get_transformed_interpolation_terms(
                    conv_svq[s, :, : self.plan.nalpha],
                    i=i_out,
                    fwd=not fwd,
                    inplace=True,
                )
        if return_orbs:
            return conv_svq
        # TODO this should be a modified atco for l1 interpolation
        self.convert_rad2orb_(
            nspin, out_sgLq, conv_svq, rad2orb=False, inp=not fwd, indexer=indexer_out
        )
        return out_sgLq

    def x2u(self, f_sxq):
        return f_sxq

    def u2x(self, f_suq):
        return f_suq

    def grid2aux(self, f_sgLq):
        nspin = f_sgLq.shape[0]
        nalpha = f_sgLq.shape[-1]
        f_svq = np.zeros((nspin, self.ccl.atco_inp.nao, nalpha))
        self.convert_rad2orb_(
            nspin, f_sgLq, f_svq, rad2orb=True, inp=True, indexer=self.grids_indexer
        )
        return f_svq

    def aux2grid(self, f_svq):
        nspin = f_svq.shape[0]
        nalpha = f_svq.shape[-1]
        f_sgLq = self.grids_indexer.empty_rlmq(nalpha=nalpha, nspin=nspin)
        f_sgLq[:] = 0.0
        self.convert_rad2orb_(
            nspin, f_sgLq, f_svq, rad2orb=False, inp=True, indexer=self.grids_indexer
        )
        return f_sgLq

    def perform_convolution_fwd(self, f_sxq):
        return self.perform_convolution(
            f_sxq,
            fwd=True,
            return_orbs=True,
            take_orbs=True,
        )

    def perform_convolution_bwd(self, f_sxq):
        return self.perform_convolution(
            f_sxq,
            fwd=False,
            return_orbs=True,
            take_orbs=True,
        )

    def perform_fitting_convolution_bwd(self, f_sxq):
        nspin = f_sxq.shape[0]
        conv_sxq = np.zeros_like(f_sxq)
        for s in range(nspin):
            self.ccl2.multiply_atc_integrals(f_sxq[s], output=conv_sxq[s], fwd=False)
            libcider.solve_atc_coefs_arr_ccl(
                self.ccl._ccl,
                conv_sxq[s].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(conv_sxq.shape[-1]),
                ctypes.c_int(0),
            )
        return conv_sxq

    def perform_fitting_convolution_fwd(self, f_sxq):
        nspin = f_sxq.shape[0]
        conv_sxq = np.zeros_like(f_sxq)
        for s in range(nspin):
            assert f_sxq[s].flags.c_contiguous
            libcider.solve_atc_coefs_arr_ccl(
                self.ccl._ccl,
                f_sxq[s].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(f_sxq.shape[-1]),
                ctypes.c_int(1),
            )
            self.ccl2.multiply_atc_integrals(f_sxq[s], output=conv_sxq[s], fwd=True)
        return conv_sxq


if USE_GAUSSIAN_PAW_CONV:
    PAWCiderContribs = PAWCiderContribsOrb
else:
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
        g : Radial grid index
        s : Spin index
    """

    is_cider_functional = True
    PAWCiderContribs = PAWCiderContribs

    fr_asgLq: dict
    df_asgLq: dict
    vfr_asgLq: dict
    vdf_asgLq: dict
    # TODO if calculating the density (which done 3X per loop in
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
        paw_algo=DEFAULT_CIDER_PAW_ALGO,
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
        self._paw_algo = paw_algo
        if self._paw_algo not in ["v1", "v2"]:
            raise ValueError("Supported paw_algo are v1, v2, got {}".format(paw_algo))

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
                # TODO it would be good to be able to remove this.
                # Old version of the code used these settings:
                # setup, rmin=setup.xc_correction.rgd.r_g[0]+1e-5,
                # N=1024, encut=1e6, d=0.018
                setup.nlxc_correction = SBTGridContainer.from_setup(
                    setup,
                    rmin=1e-4,
                    N=512,
                    encut=5e4,
                    d=0.02,
                )
                # TODO for some reason, in the core there are sometimes
                # exponents that are very big (z**2 * 2000 required).
                # However, the SCF energy and stress are essentially
                # unchanged by not including these very high exponents,
                # and they don't seem physically relevant. So it would be good
                # to set this lower by eliminating these few high-exponent points.
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
                    paw_algo=self._paw_algo,
                )
                if setup.cider_contribs.plan is not None:
                    assert (
                        abs(np.max(setup.cider_contribs.plan.alphas) / encut0 - 1)
                        < 1e-8
                    ), "{}".format(np.max(setup.cider_contribs.plan.alphas) - encut0)
                if self._paw_algo == "v1":
                    pss_cls = PSmoothSetupV1
                else:
                    pss_cls = PSmoothSetupV2
                setup.ps_setup = pss_cls.from_setup_and_atco(
                    setup,
                    setup.cider_contribs.atco_inp,
                    self.alphas,
                    self.plan.alpha_norms,
                    encut0,
                    grid_nlm=setup.cider_contribs.nlm,
                )
                setup.pa_setup = PAugSetup.from_setup(setup)

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
            setup = setups[a]
            psetup = setup.ps_setup

            self.timer.start("coefs")
            rcalc = CiderRadialFeatureCalculator(setup.cider_contribs)
            expansion = CiderRadialExpansion(rcalc)
            dx_sgLq, xt_sgLq = calculate_cider_paw_correction(
                expansion, setup, D_sp, separate_ae_ps=True
            )
            dx_sgLq -= xt_sgLq
            self.timer.stop()
            """
            slgd = setup.xc_correction.rgd
            RCUT = np.max(setup.rcut_j)
            rcut = RCUT
            rmax = slgd.r_g[-1]
            fcut = 0.5 + 0.5 * np.cos(np.pi * (slgd.r_g - rcut) / (rmax - rcut))
            fcut[slgd.r_g < rcut] = 1.0
            fcut[slgd.r_g > rmax] = 0.0
            xt_sgLq *= fcut[:, None, None]
            """
            self.timer.start("transform and convolve")
            fr_sgLq, df_sgLq, c_siq = setup.cider_contribs.calculate_y_terms(
                xt_sgLq, dx_sgLq, psetup
            )
            self.timer.stop()
            self.df_asgLq[a] = df_sgLq
            self.fr_asgLq[a] = fr_sgLq
            c_asiq[a] = c_siq
        return c_asiq

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
            # TODO C function for this loop? can go in pa_setup
            for i in range(ni):
                L = psetup.lmlist_i[i]
                n = psetup.nlist_i[i]
                Dref_sq = (psetup.pfuncs_ng[n] * dv_g).dot(fr_sgLq[:, :, L, :Nalpha_sm])
                j = psetup.jlist_i[i]
                for s in range(nspin):
                    ft_sgLq[s, :, L, :Nalpha_sm] += (
                        D_asiq[a][s, i] - Dref_sq[s]
                    ) * psetup.ffuncs_jg[j][:, None]
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
            # TODO C function for this loop? can go in pa_setup
            for i in range(ni):
                L = psetup.lmlist_i[i]
                n = psetup.nlist_i[i]
                j = psetup.jlist_i[i]
                dvD_sq = np.einsum(
                    "sgq,g->sq",
                    vft_sgLq[:, :, L, :Nalpha_sm],
                    psetup.ffuncs_jg[j],
                )
                dvD_asiq[a][:, i] = dvD_sq
                # We need to multiply the term here by dv_g because vfr_sgLq
                # has already been multiplied by the quadrature weights.
                vfr_sgLq[:, :, L, :Nalpha_sm] -= (
                    dvD_sq[:, None, :] * psetup.pfuncs_ng[n][:, None]
                ) * dv_g[:, None]
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
            psetup = setup.ps_setup
            vxt_sgLq, vdx_sgLq = setup.cider_contribs.calculate_vx_terms(
                self.vfr_asgLq[a], self.vdf_asgLq[a], vc_asiq[a], psetup
            )
            """
            slgd = setup.xc_correction.rgd
            RCUT = np.max(setup.rcut_j)
            rcut = RCUT
            rmax = slgd.r_g[-1]
            fcut = 0.5 + 0.5 * np.cos(np.pi * (slgd.r_g - rcut) / (rmax - rcut))
            fcut[slgd.r_g < rcut] = 1.0
            fcut[slgd.r_g > rmax] = 0.0
            vxt_sgLq *= fcut[:, None, None]
            """
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


class PASDWData:
    def __init__(
        self,
        pfuncs_ng,
        pfuncs_ntp,
        interp_rgd,
        slrgd,
        nlist_j,
        llist_j,
        lmlist_i,
        jlist_i,
        sbt_rgd,
        alphas,
        nbas_loc,
        rcut_func,
        rcut_feat,
        Z=None,
        alphas_ae=None,
        alpha_norms=None,
        pmax=None,
        grid_nlm=None,
    ):
        self.nn = pfuncs_ng.shape[0]
        self.ng = pfuncs_ng.shape[1]
        self.nt = pfuncs_ntp.shape[1]
        self.ni = lmlist_i.shape[0]
        self.nj = len(nlist_j)
        self.lmax = np.max(llist_j)
        if grid_nlm is not None:
            lmax = int(np.sqrt(grid_nlm) + 1e-8) - 1
            assert lmax >= self.lmax
            self.lmax = lmax
        self.Z = Z

        self.pfuncs_ng = pfuncs_ng
        self.pfuncs_ntp = pfuncs_ntp

        self.nbas_loc = nbas_loc
        self.nlist_j = np.array(nlist_j).astype(np.int32)
        self.llist_j = np.array(llist_j).astype(np.int32)
        self.nlist_i = self.nlist_j[jlist_i]
        self.llist_i = self.llist_j[jlist_i]
        self.lmlist_i = np.array(lmlist_i).astype(np.int32)
        self.jlist_i = np.array(jlist_i).astype(np.int32)

        self.interp_rgd = interp_rgd
        self.slrgd = slrgd
        self.sbt_rgd = sbt_rgd
        self.rmax = self.slrgd.r_g[-1]
        self.rcut_func = rcut_func
        self.rcut_feat = rcut_feat

        self.alphas = alphas
        self.alphas_ae = alphas_ae
        if alpha_norms is None:
            alpha_norms = np.ones_like(alphas_ae)
        self.alpha_norms = alpha_norms
        self.pmax = pmax
        self.nalpha = len(self.alphas)
        self._grid_nlm = grid_nlm


class PAugSetup(PASDWData):
    def initialize_g2a(self):
        nn = self.nn
        nj = self.nj
        ng = self.ng
        r_g = self.slrgd.r_g
        dr_g = self.slrgd.dr_g
        nlist_j = self.nlist_j

        pfuncs_ng = self.pfuncs_ng
        pfuncs_ntp = self.pfuncs_ntp
        ffuncs_jg = np.zeros((nj, ng))
        ffuncs_jt = np.zeros((nj, self.interp_rgd.r_g.size))
        dv_g = r_g * r_g * dr_g
        filt = self.get_filt(r_g)
        for l in range(self.lmax + 1):
            jmin, jmax = self.nbas_loc[l], self.nbas_loc[l + 1]
            ovlp = np.einsum(
                "ig,jg,g->ij",
                pfuncs_ng[nlist_j[jmin:jmax]],
                pfuncs_ng[nlist_j[jmin:jmax]],
                dv_g * filt,
            )
            c_and_l = cho_factor(ovlp)
            ffuncs_jg[jmin:jmax] = cho_solve(c_and_l, pfuncs_ng[nlist_j[jmin:jmax]])
            ffuncs_jt[jmin:jmax] = cho_solve(
                c_and_l, pfuncs_ntp[nlist_j[jmin:jmax], :, 0]
            )
        self.ffuncs_jtp = np.zeros((nj, self.interp_rgd.r_g.size, 4))
        for j in range(nj):
            self.ffuncs_jtp[j] = spline(
                np.arange(self.interp_rgd.r_g.size).astype(np.float64),
                ffuncs_jt[j],
            )
        self.exact_ovlp_pf = np.identity(self.ni)
        self.ffuncs_jg = ffuncs_jg

        self.pfuncs_ng *= self.get_filt(self.slrgd.r_g)
        for n in range(nn):
            self.pfuncs_ntp[n] = spline(
                np.arange(self.interp_rgd.r_g.size).astype(np.float64),
                get_ffunc(n, self.interp_rgd.r_g, self.rcut_func)
                * self.get_filt(self.interp_rgd.r_g),
            )
        # TODO should work but dangerous, make cleaner way to set rcut_func
        self.rcut_func = self.rcut_feat

    def get_filt(self, r_g):
        filt = 0.5 + 0.5 * np.cos(np.pi * r_g / self.rcut_feat)
        filt[r_g > self.rcut_feat] = 0.0
        return filt

    def get_fcut(self, r):
        R = self.rcut_feat
        fcut = np.ones_like(r)
        fcut[r > R] = 0.5 + 0.5 * np.cos(np.pi * (r[r > R] - R) / R)
        fcut[r > 2 * R] = 0.0
        return fcut

    @classmethod
    def from_setup(cls, setup):
        rgd = setup.xc_correction.rgd
        RCUT_FAC = 1.0
        rcut_feat = np.max(setup.rcut_j) * RCUT_FAC
        rmax = np.max(setup.xc_correction.rgd.r_g)
        rcut_func = rmax * 0.8

        nt = 100
        interp_rgd = EquidistantRadialGridDescriptor(h=rcut_func / (nt - 1), N=nt)
        interp_r_g = interp_rgd.r_g
        ng = rgd.r_g.size

        nlist_j, llist_j = get_psetup_func_counts(setup.Z)
        lp1 = np.max(llist_j) + 1
        nbas_lst = [0] * lp1
        for l in llist_j:
            nbas_lst[l] += 1
        nbas_loc = np.append([0], np.cumsum(nbas_lst)).astype(np.int32)
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

        nn = np.max(nlist_j) + 1

        pfuncs_ng = np.zeros((nn, ng), dtype=np.float64, order="C")
        pfuncs_ntp = np.zeros((nn, nt, 4), dtype=np.float64, order="C")
        for n in range(nn):
            pfunc_t = get_ffunc(n, interp_r_g, rcut_func)
            pfuncs_ng[n, :] = get_ffunc(n, rgd.r_g, rcut_func)
            pfuncs_ntp[n, :, :] = spline(
                np.arange(interp_r_g.size).astype(np.float64), pfunc_t
            )

        psetup = cls(
            pfuncs_ng,
            pfuncs_ntp,
            interp_rgd,
            setup.xc_correction.rgd,
            nlist_j,
            llist_j,
            lmlist_i,
            jlist_i,
            setup.nlxc_correction.big_rgd,
            [],
            nbas_loc,
            rcut_func,
            rcut_feat,
            Z=setup.Z,
        )
        psetup.initialize_g2a()

        return psetup


class RadialFunctionCollection:
    def __init__(self, iloc_l, funcs_ig, r_g, dv_g, lmax=None):
        self.iloc_l = iloc_l
        self.funcs_ig = np.ascontiguousarray(funcs_ig)
        self.ovlps_l = None
        self.jloc_l = None
        self.nu = None
        self.nv = None
        self.atco: ATCBasis = None
        self.atco_recip: ATCBasis = None
        self.lmax = len(iloc_l) - 2
        if lmax is not None:
            assert lmax >= self.lmax
            self.lmax = lmax
        self.r_g = r_g
        self.dv_g = dv_g

    @property
    def nlm(self):
        return (self.lmax + 1) * (self.lmax + 1)

    def ovlp_with_atco(self, atco: ATCBasis):
        bas, env = atco.bas, atco.env
        r_g = self.r_g
        dv_g = self.dv_g
        ovlps_l = []
        jcount_l = []
        ijcount_l = []
        self.nu = 0
        uloc_l = [0]
        vloc_l = [0]
        self.nv = 0
        # Check there is large enough l for basis projection
        assert np.max(bas[:, ANG_OF]) >= self.iloc_l.size - 2
        for l in range(self.iloc_l.size - 1):
            cond = bas[:, ANG_OF] == l
            exps_j = env[bas[cond, PTR_EXP]]
            coefs_j = env[bas[cond, PTR_COEFF]]
            basis_jg = coefs_j[:, None] * np.exp(-exps_j[:, None] * r_g * r_g)
            basis_jg[:] *= r_g**l * dv_g
            funcs_jg = self.funcs_ig[self.iloc_l[l] : self.iloc_l[l + 1]]
            ovlp = np.einsum("ig,jg->ij", funcs_jg, basis_jg)
            ovlps_l.append(ovlp)
            ni = self.iloc_l[l + 1] - self.iloc_l[l]
            nj = exps_j.size
            self.nu += (2 * l + 1) * ni
            self.nv += (2 * l + 1) * nj
            uloc_l.append(self.nu)
            vloc_l.append(self.nv)
            jcount_l.append(nj)
            ijcount_l.append(nj * ni)
        self.ovlps_l = ovlps_l
        self._ovlps_l = np.ascontiguousarray(
            np.concatenate([ovlp.ravel() for ovlp in self.ovlps_l])
        )
        self.jloc_l = np.append([0], np.cumsum(jcount_l))
        self.uloc_l = np.asarray(uloc_l, order="C", dtype=np.int32)
        self.vloc_l = np.asarray(vloc_l, order="C", dtype=np.int32)
        self.atco = atco
        self.atco_recip = atco.get_reciprocal_atco()

    def expand_to_grid(self, x_g, recip=False):
        assert self.atco is not None
        x_g = np.ascontiguousarray(x_g)
        if recip:
            atco = self.atco_recip
            fac = 0.5 * np.pi
        else:
            atco = self.atco
            fac = 1.0
        funcs_ig = np.zeros((self.iloc_l[-1], x_g.size))
        for l in range(self.iloc_l.size - 1):
            i0, i1 = self.iloc_l[l : l + 2]
            for i in range(i0, i1):
                in_j = np.ascontiguousarray(self.ovlps_l[l][i - i0])
                libcider.expand_to_grid(
                    in_j.ctypes.data_as(ctypes.c_void_p),
                    funcs_ig[i].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(l),
                    ctypes.c_int(0),
                    x_g.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(x_g.size),
                    atco._atco,
                )
        funcs_ig[:] *= fac
        return funcs_ig

    def _fft_helper(self, in_ig, xin_g, xout_g, dv_g, atco_in, atco_out):
        assert self.atco is not None
        xin_g = np.ascontiguousarray(xin_g)
        xout_g = np.ascontiguousarray(xout_g)
        dv_g = np.ascontiguousarray(dv_g)
        assert in_ig.shape == (self.iloc_l[-1], xin_g.size)
        out_ig = np.zeros((self.iloc_l[-1], xout_g.size))
        for l in range(self.iloc_l.size - 1):
            i0, i1 = self.iloc_l[l : l + 2]
            for i in range(i0, i1):
                in_g = np.ascontiguousarray(in_ig[i])
                p_j = np.zeros_like(self.ovlps_l[l][i - i0], order="C")
                libcider.contract_from_grid(
                    in_g.ctypes.data_as(ctypes.c_void_p),
                    p_j.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(l),
                    ctypes.c_int(0),
                    xin_g.ctypes.data_as(ctypes.c_void_p),
                    dv_g.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(xin_g.size),
                    atco_in._atco,
                )
                libcider.expand_to_grid(
                    p_j.ctypes.data_as(ctypes.c_void_p),
                    out_ig[i].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(l),
                    ctypes.c_int(0),
                    xout_g.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(xout_g.size),
                    atco_out._atco,
                )
        return out_ig

    def orb2k(self, f_suq):
        if not f_suq.flags.c_contiguous:
            f_suq = np.ascontiguousarray(f_suq)
        nspin, nu, nq = f_suq.shape
        atco_inp = self.atco_recip
        rgd = self.sbt_rgd
        k_g = rgd.k_g
        nk = k_g.size
        nlm = self.nlm
        out_skLq = np.zeros((nspin, nk, nlm, nq))
        y = Y_nL[:, :nlm]
        indexer = AtomicGridsIndexer.make_single_atom_indexer(y, k_g)
        for s in range(nspin):
            libcider.solve_atc_coefs_arr(
                self.atco_inp.atco_c_ptr,
                f_suq[s].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(f_suq.shape[-1]),
            )
            atco_inp.convert_rad2orb_(
                out_skLq[s],
                f_suq[s],
                indexer,
                indexer.rad_arr,
                rad2orb=False,
                offset=0,
            )
        return out_skLq

    def fft(self, funcs_ig, r_g, k_g, dv_g):
        atco_in = self.atco
        atco_out = self.atco_recip
        return self._fft_helper(funcs_ig, r_g, k_g, dv_g, atco_in, atco_out)

    def ifft(self, funcs_ig, r_g, k_g, dv_g):
        atco_in = self.atco_recip
        atco_out = self.atco
        return self._fft_helper(funcs_ig, k_g, r_g, dv_g, atco_in, atco_out)

    def convert(self, p_uq, fwd=True):
        assert p_uq.ndim == 2
        assert p_uq.flags.c_contiguous
        nalpha = p_uq.shape[1]
        if fwd:
            iloc_l = np.asarray(self.iloc_l, dtype=np.int32, order="C")
            jloc_l = np.asarray(self.jloc_l, dtype=np.int32, order="C")
            p_vq = np.empty((self.nv, nalpha), order="C")
            assert p_uq.shape[0] == self.nu
        else:
            iloc_l = np.asarray(self.jloc_l, dtype=np.int32, order="C")
            jloc_l = np.asarray(self.iloc_l, dtype=np.int32, order="C")
            p_vq = np.empty((self.nu, nalpha), order="C")
            assert p_uq.shape[0] == self.nv
        p_vq[:] = 0
        if nalpha > 0:
            libcider.convert_atomic_radial_basis(
                p_uq.ctypes.data_as(ctypes.c_void_p),  # input
                p_vq.ctypes.data_as(ctypes.c_void_p),  # output
                self._ovlps_l.ctypes.data_as(ctypes.c_void_p),  # matrices
                iloc_l.ctypes.data_as(ctypes.c_void_p),  # locs for p_uq
                jloc_l.ctypes.data_as(ctypes.c_void_p),  # locs for p_vq
                ctypes.c_int(nalpha),  # last axis of p_uq
                ctypes.c_int(self.lmax),  # maximum l value of basis
                ctypes.c_int(1 if fwd else 0),  # fwd
            )
        return p_vq

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


class _PSmoothSetupBase(PASDWData):
    @classmethod
    def from_setup_and_atco(cls, setup, atco, alphas, alpha_norms, pmax, grid_nlm):
        rgd = setup.xc_correction.rgd
        rcut_feat = np.max(setup.rcut_j)
        rcut_func = rcut_feat * 1.00

        nlist_j, llist_j = get_psetup_func_counts(setup.Z)
        lp1 = int(np.sqrt(grid_nlm + 1e-8))
        nbas_lst = [0] * lp1
        for l in llist_j:
            nbas_lst[l] += 1
        nbas_loc = np.append([0], np.cumsum(nbas_lst)).astype(np.int32)
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

        nt = 100
        interp_rgd = EquidistantRadialGridDescriptor(h=rcut_func / (nt - 1), N=nt)
        interp_r_g = interp_rgd.r_g
        ng = rgd.r_g.size
        nn = np.max(nlist_j) + 1

        # TODO not as much initialization needed here after sbt_rgd removed
        pfuncs_ng = np.zeros((nn, ng), dtype=np.float64, order="C")
        pfuncs_ntp = np.zeros((nn, nt, 4), dtype=np.float64, order="C")
        pfuncs_nt = np.zeros((nn, nt), dtype=np.float64, order="C")
        for n in range(nn):
            pfuncs_nt[n, :] = get_pfunc_norm(n, interp_r_g, rcut_func)
            pfuncs_ng[n, :] = get_pfunc_norm(n, rgd.r_g, rcut_func)

        for n in range(nn):
            pfuncs_ntp[n, :, :] = spline(
                np.arange(interp_r_g.size).astype(np.float64), pfuncs_nt[n]
            )

        psetup = cls(
            pfuncs_ng,
            pfuncs_ntp,
            interp_rgd,
            rgd,
            nlist_j,
            llist_j,
            lmlist_i,
            jlist_i,
            setup.nlxc_correction.big_rgd,
            alphas,
            nbas_loc,
            rcut_func,
            rcut_feat,
            Z=setup.Z,
            alphas_ae=alphas,
            alpha_norms=alpha_norms,
            pmax=pmax,
            grid_nlm=grid_nlm,
        )

        psetup._initialize(atco)

        return psetup

    def _initialize(self, atco):
        pass

    def _get_betas(self, rgd, atco):
        ls = atco.bas[:, ANG_OF]
        coefs = atco.env[atco.bas[:, PTR_EXP]]
        betas_lv = []
        for l in range(np.max(ls) + 1):
            betas_lv.append(coefs[ls == l])
        pmin = 2 / self.rcut_feat**2
        if self.pmax is None:
            raise ValueError
        else:
            pmax = 2 * self.pmax

        return get_delta_lpg(betas_lv, self.rcut_feat, rgd, pmin, pmax + 1e-8)

    def k2orb(self, f_skLq):
        f_skLq = f_skLq * get_dvk(self.sbt_rgd)[:, None, None]
        if not f_skLq.flags.c_contiguous:
            f_skLq = np.ascontiguousarray(f_skLq)
        nspin = f_skLq.shape[0]
        atco_inp = self.pcol.atco_recip
        rgd = self.sbt_rgd
        k_g = rgd.k_g
        nspin, nk, nlm, nq = f_skLq.shape
        y = Y_nL[:, :nlm]
        out_suq = np.zeros((nspin, atco_inp.nao, nq))
        indexer = AtomicGridsIndexer.make_single_atom_indexer(y, k_g)
        for s in range(nspin):
            atco_inp.convert_rad2orb_(
                f_skLq[s],
                out_suq[s],
                indexer,
                indexer.rad_arr,
                rad2orb=True,
                offset=0,
            )
            libcider.solve_atc_coefs_arr(
                self.pcol.atco.atco_c_ptr,
                out_suq[s].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(out_suq.shape[-1]),
            )
        return out_suq

    def orb2k(self, f_suq):
        if not f_suq.flags.c_contiguous:
            f_suq = np.ascontiguousarray(f_suq)
        nspin, nu, nq = f_suq.shape
        atco_inp = self.pcol.atco_recip
        rgd = self.sbt_rgd
        k_g = rgd.k_g
        nk = k_g.size
        nlm = self._grid_nlm
        out_skLq = np.zeros((nspin, nk, nlm, nq))
        y = Y_nL[:, :nlm]
        indexer = AtomicGridsIndexer.make_single_atom_indexer(y, k_g)
        for s in range(nspin):
            libcider.solve_atc_coefs_arr(
                self.pcol.atco.atco_c_ptr,
                f_suq[s].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(f_suq.shape[-1]),
            )
            atco_inp.convert_rad2orb_(
                out_skLq[s],
                f_suq[s],
                indexer,
                indexer.rad_arr,
                rad2orb=False,
                offset=0,
            )
        return out_skLq

    def get_df_only(self, y_svq):
        if not USE_GAUSSIAN_PAW_CONV:
            y_svq = self.k2orb(y_svq)
        else:
            y_svq = np.ascontiguousarray(y_svq)
        nspin, nv, nq = y_svq.shape
        df_suq = np.zeros((nspin, self.pcol.nu, nq))
        for s in range(nspin):
            df_suq[s] = self.pcol.convert(y_svq[s], fwd=False)
        return self.pcol.basis2grid_spin(df_suq)

    def get_vdf_only(self, vdf_sgLq):
        vdf_suq = self.pcol.grid2basis_spin(vdf_sgLq)
        nspin, nu, nq = vdf_suq.shape
        vy_svq = np.zeros((nspin, self.pcol.nv, nq))
        for s in range(nspin):
            vy_svq[s] = self.pcol.convert(vdf_suq[s], fwd=True)
        if not USE_GAUSSIAN_PAW_CONV:
            vy_svq = self.orb2k(vy_svq)
        return vy_svq

    def coef_to_real(self, c_siq, out=None, check_q=True, use_ffuncs=False):
        if check_q:
            nq = self.alphas.size
        else:
            nq = c_siq.shape[-1]
        ng = self.slrgd.r_g.size
        nlm = self.pcol.nlm
        nspin = c_siq.shape[0]
        if out is None:
            xt_sgLq = np.zeros((nspin, ng, nlm, nq))
        else:
            xt_sgLq = out
            assert xt_sgLq.shape == (nspin, ng, nlm, nq)
        if use_ffuncs:
            for s in range(nspin):
                for i in range(self.ni):
                    L = self.lmlist_i[i]
                    j = self.jlist_i[i]
                    xt_sgLq[s, :, L, :] += c_siq[s, i, :] * self.ffuncs_jg[j, :, None]
        else:
            for s in range(nspin):
                for i in range(self.ni):
                    L = self.lmlist_i[i]
                    n = self.nlist_i[i]
                    xt_sgLq[s, :, L, :] += c_siq[s, i, :] * self.pfuncs_ng[n, :, None]
        return xt_sgLq

    def real_to_coef(self, xt_sgLq, out=None, check_q=True, use_ffuncs=False):
        if check_q:
            nq = self.alphas.size
        else:
            nq = xt_sgLq.shape[-1]
        ng = self.slrgd.r_g.size
        nlm = self.pcol.nlm
        nspin = xt_sgLq.shape[0]
        assert xt_sgLq.shape[1:] == (ng, nlm, nq)
        if out is None:
            c_siq = np.zeros((nspin, self.ni, nq))
        else:
            c_siq = out
            assert c_siq.shape == (nspin, self.ni, nq)
        if use_ffuncs:
            for s in range(nspin):
                for i in range(self.ni):
                    L = self.lmlist_i[i]
                    j = self.jlist_i[i]
                    c_siq[s, i, :] += np.dot(self.ffuncs_jg[j, :], xt_sgLq[s, :, L, :])
        else:
            for s in range(nspin):
                for i in range(self.ni):
                    L = self.lmlist_i[i]
                    n = self.nlist_i[i]
                    c_siq[s, i, :] += np.dot(self.pfuncs_ng[n, :], xt_sgLq[s, :, L, :])
        return c_siq


class PSmoothSetupV1(_PSmoothSetupBase):
    def _initialize(self, atco):
        rgd = self.slrgd
        nj = self.nj
        ng = self.ng
        nlist_j = self.nlist_j
        pfuncs_ng = self.pfuncs_ng

        dv_g = get_dv(rgd)
        ffuncs_jg = np.zeros((nj, ng))
        self.exact_ovlp_pf = np.zeros((self.ni, self.ni))
        self.ffuncs_jtp = np.zeros((nj, self.interp_rgd.r_g.size, 4))
        ilist = np.arange(self.ni)

        for j in range(self.nj):
            n = nlist_j[j]
            l = self.llist_j[j]
            ffuncs_jg[j] = pfuncs_ng[n]
            self.ffuncs_jtp[j] = self.pfuncs_ntp[n]
            i0s = ilist[self.jlist_i == j]
            for j1 in range(self.nj):
                if self.llist_j[j1] == l:
                    i1s = ilist[self.jlist_i == j1]
                    n1 = nlist_j[j1]
                    ovlp = np.dot(pfuncs_ng[n] * pfuncs_ng[n1], dv_g)
                    for i0, i1 in zip(i0s.tolist(), i1s.tolist()):
                        self.exact_ovlp_pf[i0, i1] = ovlp
        self.ffuncs_jg = ffuncs_jg
        rfc_lmax = int(np.sqrt(self._grid_nlm + 1e-8)) - 1
        self.jcol = RadialFunctionCollection(
            self.nbas_loc, ffuncs_jg, rgd.r_g, dv_g, lmax=rfc_lmax
        )
        self.w_b = np.ones(self.alphas_ae.size) / self.alpha_norms**2
        delta_l_pg = self._get_betas(self.slrgd, atco)
        delta_l_pg2 = self._get_betas(self.sbt_rgd, atco)
        jloc_l = [0] + [delta_pg.shape[0] for delta_pg in delta_l_pg]
        jloc_l = np.cumsum(jloc_l)
        delta_pg = np.concatenate(delta_l_pg, axis=0)
        self.pcol = RadialFunctionCollection(
            jloc_l, delta_pg, self.slrgd.r_g, get_dv(self.slrgd), lmax=rfc_lmax
        )
        self.pcol.ovlp_with_atco(atco)
        self.jcol.ovlp_with_atco(atco)
        rgd = self.sbt_rgd
        if USE_GAUSSIAN_SBT:
            pfuncs_jk = self.jcol.expand_to_grid(rgd.k_g, recip=True)
        else:
            ng2 = self.sbt_rgd.r_g.size
            nn = self.pfuncs_ng.shape[0]
            pfuncs2_ng = np.zeros((nn, ng2), dtype=np.float64, order="C")
            for n in range(nn):
                pfuncs2_ng[n, :] = get_pfunc_norm(n, self.sbt_rgd.r_g, self.rcut_func)
            pfuncs_jg = np.stack([pfuncs2_ng[n] for n in self.nlist_j])
            pfuncs_jk = get_pfuncs_k(
                pfuncs_jg, self.llist_j, self.sbt_rgd, ns=self.nlist_j
            )
            self.pfuncs_jk = pfuncs_jk
        phi_jabk = get_phi_iabk(pfuncs_jk, rgd.k_g, self.alphas, betas=self.alphas_ae)
        phi_jabk[:] *= self.alpha_norms[:, None, None]
        phi_jabk[:] *= self.alpha_norms[:, None]
        REG11 = 1e-6
        REG22 = 1e-5
        FAC22 = 1e-2
        if USE_GAUSSIAN_SBT:
            phi_jabg = np.zeros_like(phi_jabk)
            for a in range(phi_jabk.shape[1]):
                for b in range(phi_jabk.shape[2]):
                    phi_jabg[:, a, b] = self.jcol.ifft(
                        phi_jabk[:, a, b], rgd.r_g, rgd.k_g, get_dvk(rgd)
                    )
        else:
            phi_jabg = get_phi_iabg(phi_jabk, self.llist_j, self.sbt_rgd)
        p11_l_vv = get_p11_matrix(delta_l_pg, self.slrgd, reg=REG11)
        p12_l_vbja, p21_l_javb = get_p12_p21_matrix(
            delta_l_pg2, phi_jabg, rgd, self.w_b, self.nbas_loc
        )
        p22_l_jaja = []
        for l in range(self.lmax + 1):
            p22_l_jaja.append(
                get_p22_matrix(
                    phi_jabg,
                    rgd,
                    self.rcut_feat,
                    l,
                    self.w_b,
                    self.nbas_loc,
                    reg=REG22,
                )
            )
            p22_l_jaja[-1] += FAC22 * get_p22_matrix(
                phi_jabg,
                rgd,
                self.rcut_feat,
                l,
                self.w_b,
                self.nbas_loc,
                cut_func=True,
            )
        p_l_ii = construct_full_p_matrices(
            p11_l_vv,
            p12_l_vbja,
            p21_l_javb,
            p22_l_jaja,
            self.w_b,
        )
        self.p_l_ii = p_l_ii
        self.p_cl_l_ii = []
        for l in range(self.lmax + 1):
            c_and_l = cho_factor(p_l_ii[l])
            self.p_cl_l_ii.append(c_and_l)

    def _c_and_df_loop_orb(self, y_svq, yy_svq):
        if USE_GAUSSIAN_PAW_CONV:
            y_svq = np.ascontiguousarray(y_svq)
            yy_svq = np.ascontiguousarray(yy_svq)
        else:
            y_svq = self.k2orb(y_svq)
            yy_svq = self.k2orb(yy_svq)
        nspin, nv, nb = y_svq.shape
        assert nv == self.pcol.nv
        na = self.alphas.size
        c_sia = np.zeros((nspin, self.ni, na))
        df_sub = np.zeros((nspin, self.pcol.nu, nb))
        for s in range(nspin):
            all_b1_ub = self.pcol.convert(y_svq[s], fwd=False)
            all_b2_ia = self.jcol.convert(yy_svq[s], fwd=False)
            for l in range(self.lmax + 1):
                nm = 2 * l + 1
                for m in range(nm):
                    uloc_l = self.pcol.uloc_l
                    slice1 = slice(uloc_l[l] + m, uloc_l[l + 1] + m, nm)
                    b1 = all_b1_ub[slice1]
                    uloc_l = self.jcol.uloc_l
                    slice2 = slice(uloc_l[l] + m, uloc_l[l + 1] + m, nm)
                    b2 = all_b2_ia[slice2]
                    b = self.cat_b_vecs(b1, b2)
                    x = cho_solve(self.p_cl_l_ii[l], b)
                    n1 = b1.size
                    df_sub[s, slice1] += x[:n1].reshape(-1, nb)
                    c_sia[s, slice2] += x[n1:].reshape(-1, na)
        return c_sia, self.pcol.basis2grid_spin(df_sub)

    def _c_and_df_loop_recip(self, y_svq, yy_skLq):
        y_svq = self.k2orb(y_svq)
        yy_skLq[:] *= get_dvk(self.sbt_rgd)[:, None, None]
        nspin, nv, nb = y_svq.shape
        assert nv == self.pcol.nv
        na = self.alphas.size
        c_sia = np.zeros((nspin, self.ni, na))
        df_sub = np.zeros((nspin, self.pcol.nu, nb))
        for s in range(nspin):
            all_b1_ub = self.pcol.convert(y_svq[s], fwd=False)
            for l in range(self.lmax + 1):
                nm = 2 * l + 1
                for m in range(nm):
                    uloc_l = self.pcol.uloc_l
                    slice1 = slice(uloc_l[l] + m, uloc_l[l + 1] + m, nm)
                    b1 = all_b1_ub[slice1]
                    uloc_l = self.jcol.uloc_l
                    slice2 = slice(uloc_l[l] + m, uloc_l[l + 1] + m, nm)
                    jl = self.nbas_loc[l]
                    ju = self.nbas_loc[l + 1]
                    b2 = np.einsum(
                        "jk,kq->jq",
                        self.pfuncs_jk[jl:ju],
                        yy_skLq[s, :, l * l + m, :],
                    )
                    b = self.cat_b_vecs(b1, b2)
                    x = cho_solve(self.p_cl_l_ii[l], b)
                    n1 = b1.size
                    df_sub[s, slice1] += x[:n1].reshape(-1, nb)
                    c_sia[s, slice2] += x[n1:].reshape(-1, na)
        return c_sia, self.pcol.basis2grid_spin(df_sub)

    def get_c_and_df(self, y_svq, yy_svq):
        assert yy_svq.shape == y_svq.shape
        if not USE_GAUSSIAN_PAW_CONV and not USE_GAUSSIAN_SBT:
            return self._c_and_df_loop_recip(y_svq, yy_svq)
        else:
            return self._c_and_df_loop_orb(y_svq, yy_svq)

    def _vy_and_vyy_loop_orb(self, vc_sia, vdf_sub):
        nspin, ni, na = vc_sia.shape
        nb = vdf_sub.shape[-1]
        assert ni == self.ni
        vy_svq = np.zeros((nspin, self.pcol.nv, nb))
        vyy_svq = np.zeros((nspin, self.pcol.nv, na))
        for s in range(nspin):
            all_vb1_ub = np.zeros((self.pcol.nu, nb))
            all_vb2_ia = np.zeros((ni, na))
            for l in range(self.lmax + 1):
                nm = 2 * l + 1
                for m in range(nm):
                    uloc_l = self.pcol.uloc_l
                    slice1 = slice(uloc_l[l] + m, uloc_l[l + 1] + m, nm)
                    x1 = vdf_sub[s, slice1].ravel()
                    uloc_l = self.jcol.uloc_l
                    slice2 = slice(uloc_l[l] + m, uloc_l[l + 1] + m, nm)
                    x2 = vc_sia[s, slice2].ravel()
                    x = self.cat_b_vecs(x1, x2)
                    b = cho_solve(self.p_cl_l_ii[l], x)
                    n1 = x1.size
                    all_vb1_ub[slice1] += b[:n1].reshape(-1, nb)
                    all_vb2_ia[slice2] += b[n1:].reshape(-1, na)
            vy_svq[s] = self.pcol.convert(all_vb1_ub, fwd=True)
            vyy_svq[s] = self.jcol.convert(all_vb2_ia, fwd=True)
        if not USE_GAUSSIAN_PAW_CONV:
            vy_svq = self.orb2k(vy_svq)
            vyy_svq = self.orb2k(vyy_svq)
        return vy_svq, vyy_svq

    def _vy_and_vyy_loop_recip(self, vc_sia, vdf_sub):
        nspin, ni, na = vc_sia.shape
        nb = vdf_sub.shape[-1]
        assert ni == self.ni
        vy_svq = np.zeros((nspin, self.pcol.nv, nb))
        vyy_skLq = np.zeros((nspin, self.sbt_rgd.r_g.size, self._grid_nlm, na))
        for s in range(nspin):
            all_vb1_ub = np.zeros((self.pcol.nu, nb))
            for l in range(self.lmax + 1):
                nm = 2 * l + 1
                for m in range(nm):
                    uloc_l = self.pcol.uloc_l
                    slice1 = slice(uloc_l[l] + m, uloc_l[l + 1] + m, nm)
                    x1 = vdf_sub[s, slice1].ravel()
                    uloc_l = self.jcol.uloc_l
                    slice2 = slice(uloc_l[l] + m, uloc_l[l + 1] + m, nm)
                    x2 = vc_sia[s, slice2].ravel()
                    x = self.cat_b_vecs(x1, x2)
                    b = cho_solve(self.p_cl_l_ii[l], x)
                    n1 = x1.size
                    all_vb1_ub[slice1] += b[:n1].reshape(-1, nb)
                    jl = self.nbas_loc[l]
                    ju = self.nbas_loc[l + 1]
                    vyy_skLq[s, :, l * l + m, :] += np.einsum(
                        "jk,jq->kq", self.pfuncs_jk[jl:ju], b[n1:].reshape(-1, na)
                    )
            vy_svq[s] = self.pcol.convert(all_vb1_ub, fwd=True)
        vy_svq = self.orb2k(vy_svq)
        return vy_svq, vyy_skLq

    def get_vy_and_vyy(self, vc_sia, vdf_sgLq):
        vdf_sub = self.pcol.grid2basis_spin(np.ascontiguousarray(vdf_sgLq))
        if not USE_GAUSSIAN_PAW_CONV and not USE_GAUSSIAN_SBT:
            return self._vy_and_vyy_loop_recip(vc_sia, vdf_sub)
        else:
            return self._vy_and_vyy_loop_orb(vc_sia, vdf_sub)

    def cat_b_vecs(self, b1, b2):
        return np.append(b1.flatten(), b2.flatten())


class PSmoothSetupV2(_PSmoothSetupBase):
    def _initialize(self, atco):
        rgd = self.slrgd
        nj = self.nj
        ng = self.ng
        nlist_j = self.nlist_j
        pfuncs_ng = self.pfuncs_ng

        dv_g = get_dv(rgd)
        ffuncs_jg = np.zeros((nj, ng))
        ffuncs_jt = np.zeros((nj, self.interp_rgd.r_g.size))
        for j in range(self.nj):
            n = self.nlist_j[j]
            ffuncs_jg[j] = get_poly(n, rgd.r_g, self.rcut_feat)
            ffuncs_jt[j] = get_poly(n, self.interp_rgd.r_g, self.rcut_feat)
        for l in range(self.lmax + 1):
            jmin, jmax = self.nbas_loc[l], self.nbas_loc[l + 1]
            ovlp = np.einsum(
                "ig,jg,g->ij",
                pfuncs_ng[nlist_j[jmin:jmax]],
                ffuncs_jg[jmin:jmax],
                dv_g,
            )
            ffuncs_jg[jmin:jmax] = np.linalg.solve(ovlp.T, ffuncs_jg[jmin:jmax])
            ffuncs_jt[jmin:jmax] = np.linalg.solve(ovlp.T, ffuncs_jt[jmin:jmax])
        self.ffuncs_jtp = np.zeros((nj, self.interp_rgd.r_g.size, 4))
        for j in range(nj):
            self.ffuncs_jtp[j] = spline(
                np.arange(self.interp_rgd.r_g.size).astype(np.float64),
                ffuncs_jt[j],
            )
        self.exact_ovlp_pf = np.identity(self.ni)
        self.ffuncs_jg = ffuncs_jg
        delta_l_pg = self._get_betas(self.slrgd, atco)
        jloc_l = [0] + [delta_pg.shape[0] for delta_pg in delta_l_pg]
        jloc_l = np.cumsum(jloc_l)
        delta_pg = np.concatenate(delta_l_pg, axis=0)
        rfc_lmax = int(np.sqrt(self._grid_nlm + 1e-8)) - 1
        self.pcol = RadialFunctionCollection(
            jloc_l, delta_pg, self.slrgd.r_g, get_dv(self.slrgd), lmax=rfc_lmax
        )
        self.pcol.ovlp_with_atco(atco)
