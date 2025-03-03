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
from gpaw.sphere.lebedev import R_nv, Y_nL, weight_n
from gpaw.xc.pawcorrection import rnablaY_nLv
from numpy import pi, sqrt

from ciderpress.gpaw.atom_utils import (
    CiderRadialExpansion,
    FastPASDWCiderKernel,
    PAWCiderContribs,
    calculate_cider_paw_correction,
)
from ciderpress.gpaw.fit_paw_gauss_pot import get_dv


def get_kinetic_energy_sl(xcc, ae, D_sp):
    nspins = D_sp.shape[0]
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


def vec_radial_vars_sl(xcc, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae, is_mgga, D_sp):
    nspin = len(n_sLg)
    ngrid = n_sLg.shape[-1] * weight_n.size
    nx = 5 if is_mgga else 4
    rho_sxg = np.empty((nspin, nx, ngrid))
    rho_sxg[:, 0] = np.dot(Y_nL, n_sLg).transpose(1, 2, 0).reshape(nspin, -1)
    b_vsgn = np.dot(rnablaY_nLv.transpose(0, 2, 1), n_sLg).transpose(1, 2, 3, 0)
    b_vsgn[..., 1:, :] /= xcc.rgd.r_g[1:, None]
    b_vsgn[..., 0, :] = b_vsgn[..., 1, :]
    a_sgn = np.dot(Y_nL, dndr_sLg).transpose(1, 2, 0)
    b_vsgn += R_nv.T[:, None, None, :] * a_sgn[None, :, :]
    N = Y_nL.shape[0]
    e_g = xcc.rgd.empty(N).reshape(-1)
    e_g[:] = 0
    rho_sxg[:, 1:4] = b_vsgn.transpose(1, 0, 2, 3).reshape(nspin, 3, -1)
    if is_mgga:
        rho_sxg[:, 4] = get_kinetic_energy_sl(xcc, ae, D_sp)
    vrho_sxg = np.zeros_like(rho_sxg)
    return e_g, rho_sxg, vrho_sxg


def get_kinetic_energy_and_occd(xcc, ae, D_sp, DD_op):
    nspins = D_sp.shape[0]
    if ae:
        tau_pg = xcc.tau_pg
        tauc_g = xcc.tauc_g / (np.sqrt(4 * np.pi) * nspins)
    else:
        tau_pg = xcc.taut_pg
        tauc_g = xcc.tauct_g / (np.sqrt(4 * np.pi) * nspins)
    nn = tau_pg.shape[-1] // tauc_g.shape[0]
    tau_sg = np.dot(D_sp, tau_pg)
    dtaudf_og = np.dot(DD_op, tau_pg)
    tau_sg.shape = (tau_sg.shape[0], -1, nn)
    tau_sg[:] += tauc_g[:, None]
    tau_sg.shape = (tau_sg.shape[0], -1)
    return tau_sg, dtaudf_og


def vec_radial_vars_occd(
    xcc,
    n_sLg,
    dndf_oLg,
    Y_nL,
    dndr_sLg,
    dndrdf_oLg,
    rnablaY_nLv,
    ae,
    D_sp,
    DD_op,
    is_mgga,
):
    nspin = len(n_sLg)
    norb = len(dndf_oLg)
    ngrid = n_sLg.shape[-1] * weight_n.size
    nx = 5 if is_mgga else 4

    rho_sxg = np.empty((nspin, nx, ngrid))
    drhodf_oxg = np.empty((norb, nx, ngrid))
    rho_sxg[:, 0] = np.dot(Y_nL, n_sLg).transpose(1, 2, 0).reshape(nspin, -1)
    drhodf_oxg[:, 0] = np.dot(Y_nL, dndf_oLg).transpose(1, 2, 0).reshape(norb, -1)

    b_vsgn = np.dot(rnablaY_nLv.transpose(0, 2, 1), n_sLg).transpose(1, 2, 3, 0)
    b_vsgn[..., 1:, :] /= xcc.rgd.r_g[1:, None]
    b_vsgn[..., 0, :] = b_vsgn[..., 1, :]
    a_sgn = np.dot(Y_nL, dndr_sLg).transpose(1, 2, 0)
    b_vsgn += R_nv.T[:, None, None, :] * a_sgn[None, :, :]
    rho_sxg[:, 1:4] = b_vsgn.transpose(1, 0, 2, 3).reshape(nspin, 3, -1)

    b_vogn = np.dot(rnablaY_nLv.transpose(0, 2, 1), dndf_oLg).transpose(1, 2, 3, 0)
    b_vogn[..., 1:, :] /= xcc.rgd.r_g[1:, None]
    b_vogn[..., 0, :] = b_vogn[..., 1, :]
    a_ogn = np.dot(Y_nL, dndrdf_oLg).transpose(1, 2, 0)
    b_vogn += R_nv.T[:, None, None, :] * a_ogn[None, :, :]
    drhodf_oxg[:, 1:4] = b_vogn.transpose(1, 0, 2, 3).reshape(norb, 3, -1)

    if is_mgga:
        rho_sxg[:, 4], drhodf_oxg[:, 4] = get_kinetic_energy_and_occd(
            xcc, ae, D_sp, DD_op
        )
    return rho_sxg, drhodf_oxg


def calculate_cider_paw_correction_deriv(
    expansion, setup, D_sp, DD_op=None, addcoredensity=True, has_cider_contribs=True
):
    xcc = setup.xc_correction
    rgd = xcc.rgd
    nspins = len(D_sp)

    if has_cider_contribs:
        setup.cider_contribs.set_D_sp(D_sp, setup.xc_correction)
        setup.cider_contribs._DD_op = DD_op

    if addcoredensity:
        nc0_sg = rgd.empty(nspins)
        nct0_sg = rgd.empty(nspins)
        dc0_sg = rgd.empty(nspins)
        dct0_sg = rgd.empty(nspins)
        nc0_sg[:] = sqrt(4 * pi) / nspins * xcc.nc_g
        nct0_sg[:] = sqrt(4 * pi) / nspins * xcc.nct_g
        dc0_sg[:] = sqrt(4 * pi) / nspins * xcc.dc_g
        dct0_sg[:] = sqrt(4 * pi) / nspins * xcc.dct_g
        if xcc.nc_corehole_g is not None and nspins == 2:
            nc0_sg[0] -= 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g
            nc0_sg[1] += 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g
            dc0_sg[0] -= 0.5 * sqrt(4 * pi) * xcc.dc_corehole_g
            dc0_sg[1] += 0.5 * sqrt(4 * pi) * xcc.dc_corehole_g
    else:
        nc0_sg = 0
        nct0_sg = 0
        dc0_sg = 0
        dct0_sg = 0

    D_sLq = np.inner(D_sp, xcc.B_pqL.T)
    DD_oLq = np.inner(DD_op, xcc.B_pqL.T)

    res = expansion(rgd, D_sLq, DD_oLq, xcc.n_qg, xcc.d_qg, nc0_sg, dc0_sg, ae=True)
    rest = expansion(
        rgd, D_sLq, DD_oLq, xcc.nt_qg, xcc.dt_qg, nct0_sg, dct0_sg, ae=False
    )

    return res, rest


class PAWCiderFeatContribs(PAWCiderContribs):
    def get_kinetic_energy_and_occd(self, ae):
        nspins = self._D_sp.shape[0]
        xcc = self.xcc
        if ae:
            tau_pg = xcc.tau_pg
            tauc_g = xcc.tauc_g / (np.sqrt(4 * np.pi) * nspins)
        else:
            tau_pg = xcc.taut_pg
            tauc_g = xcc.tauct_g / (np.sqrt(4 * np.pi) * nspins)
        nn = tau_pg.shape[-1] // tauc_g.shape[0]
        tau_sg = np.dot(self._D_sp, tau_pg)
        dtaudf_og = np.dot(self._DD_op, tau_pg)
        tau_sg.shape = (tau_sg.shape[0], -1, nn)
        tau_sg[:] += tauc_g[:, None]
        tau_sg.shape = (tau_sg.shape[0], -1)
        return tau_sg, dtaudf_og

    def vec_radial_vars_occd(
        self, n_sLg, dndf_oLg, Y_nL, dndr_sLg, dndrdf_oLg, rnablaY_nLv, ae
    ):
        nspin = len(n_sLg)
        norb = len(dndf_oLg)
        ngrid = n_sLg.shape[-1] * weight_n.size
        nx = 5 if self.is_mgga else 4

        rho_sxg = np.empty((nspin, nx, ngrid))
        drhodf_oxg = np.empty((norb, nx, ngrid))
        rho_sxg[:, 0] = np.dot(Y_nL, n_sLg).transpose(1, 2, 0).reshape(nspin, -1)
        drhodf_oxg[:, 0] = np.dot(Y_nL, dndf_oLg).transpose(1, 2, 0).reshape(norb, -1)

        b_vsgn = np.dot(rnablaY_nLv.transpose(0, 2, 1), n_sLg).transpose(1, 2, 3, 0)
        b_vsgn[..., 1:, :] /= self.xcc.rgd.r_g[1:, None]
        b_vsgn[..., 0, :] = b_vsgn[..., 1, :]
        a_sgn = np.dot(Y_nL, dndr_sLg).transpose(1, 2, 0)
        b_vsgn += R_nv.T[:, None, None, :] * a_sgn[None, :, :]
        rho_sxg[:, 1:4] = b_vsgn.transpose(1, 0, 2, 3).reshape(nspin, 3, -1)

        b_vogn = np.dot(rnablaY_nLv.transpose(0, 2, 1), dndf_oLg).transpose(1, 2, 3, 0)
        b_vogn[..., 1:, :] /= self.xcc.rgd.r_g[1:, None]
        b_vogn[..., 0, :] = b_vogn[..., 1, :]
        a_ogn = np.dot(Y_nL, dndrdf_oLg).transpose(1, 2, 0)
        b_vogn += R_nv.T[:, None, None, :] * a_ogn[None, :, :]
        drhodf_oxg[:, 1:4] = b_vogn.transpose(1, 0, 2, 3).reshape(norb, 3, -1)

        if self.is_mgga:
            rho_sxg[:, 4], drhodf_oxg[:, 4] = self.get_kinetic_energy_and_occd(ae)
        return rho_sxg, drhodf_oxg

    def get_paw_atom_contribs_occd(self, p_o, rho_sxg, drhodf_oxg):
        nspin = self.nspin
        norb = len(p_o)
        assert len(rho_sxg) == nspin
        x_gq = self.grids_indexer.empty_gq(nalpha=self.plan.nalpha)
        osize = self.plan.nalpha * (nspin + norb)
        x_orLq = self.grids_indexer.empty_rlmq(nalpha=osize)
        x_orLq.shape = (nspin + norb, x_orLq.shape[0], x_orLq.shape[1], x_gq.shape[-1])
        dat = [None] * nspin
        for s in range(nspin):
            rho_tuple = self.plan.get_rho_tuple(rho_sxg[s])
            arg_g, darg_g = self.plan.get_interpolation_arguments(rho_tuple, i=-1)
            fun_g, dfun_g = self.plan.get_function_to_convolve(rho_tuple)
            fun_g[:] *= self.w_g
            dfun_g = [dfun * self.w_g for dfun in dfun_g]
            p_gq, dp_gq = self.plan.get_interpolation_coefficients(arg_g.ravel(), i=-1)
            x_gq[:] = p_gq * fun_g[:, None]
            dat[s] = (darg_g, fun_g, dfun_g, p_gq, dp_gq)
            self.grids_indexer.reduce_angc_ylm_(x_orLq[s], x_gq, a2y=True, offset=0)
        for o in range(norb):
            s = p_o[o][0]
            x_gq[:] = 0
            darg_g, fun_g, dfun_g, p_gq, dp_gq = dat[s]
            drhodf_tuple = self.plan.get_drhodf_tuple(rho_sxg[s], drhodf_oxg[o])
            darg_df = 0
            dfun_df = 0
            for darg, dfun, drho in zip(darg_g, dfun_g, drhodf_tuple):
                darg_df += darg * drho
                dfun_df += dfun * drho
            darg_df[:] *= fun_g
            x_gq[:] = darg_df[..., None] * dp_gq + dfun_df[..., None] * p_gq
            self.grids_indexer.reduce_angc_ylm_(x_orLq[nspin + o], x_gq, a2y=True)
        return x_orLq

    def get_paw_atom_feat(self, rho_sxg, f_srLq):
        nspin, nx, ngrid = rho_sxg.shape
        nfeat = self.plan.nldf_settings.nfeat
        x_sig = np.zeros((nspin, nfeat, ngrid))
        f_gq = self.grids_indexer.empty_gq(nalpha=self.plan.nalpha)
        for s in range(nspin):
            self.grids_indexer.reduce_angc_ylm_(f_srLq[s], f_gq, a2y=False, offset=0)
            x_sig[s] = self.plan.eval_rho_full(
                f_gq,
                rho_sxg[s],
                # coeff_multipliers=self.plan.alpha_norms,
            )[0]
        return x_sig

    def get_paw_atom_feat_occd(self, p_o, rho_sxg, drhodf_oxg, f_srLq, f_orLq):
        nspin, nx, ngrid = rho_sxg.shape
        norb = len(p_o)
        nfeat = self.plan.nldf_settings.nfeat
        x_sig = np.zeros((nspin, nfeat, ngrid))
        dxdf_oig = np.zeros((norb, nfeat, ngrid))
        f_gq = self.grids_indexer.empty_gq(nalpha=self.plan.nalpha)
        dat = [None] * nspin
        for s in range(nspin):
            self.grids_indexer.reduce_angc_ylm_(f_srLq[s], f_gq, a2y=False, offset=0)
            x_sig[s] = self.plan.eval_rho_full(
                f_gq,
                rho_sxg[s],
                # coeff_multipliers=self.plan.alpha_norms,
            )[0]
            dat[s] = f_gq.copy()
        for o, p in enumerate(p_o):
            self.grids_indexer.reduce_angc_ylm_(f_orLq[o], f_gq, a2y=False, offset=0)
            s = p[0]
            dxdf_oig[o] = self.plan.eval_occd_full(
                dat[s],
                rho_sxg[s],
                f_gq,
                drhodf_oxg[o],
                # coeff_multipliers=self.plan.alpha_norms,
            )
        return x_sig, dxdf_oig


class PASDWCiderFeatureKernel(FastPASDWCiderKernel):
    PAWCiderContribs = PAWCiderFeatContribs

    def initialize_more_things(self, setups=None):
        self.nspin = self.dens.nt_sg.shape[0]
        if setups is None:
            setups = self.dens.setups
        for setup in setups:
            setup.cider_contribs = None
        super().initialize_more_things(setups=setups)

    def calculate_paw_cider_features_p1(self, setups, D_asp, DD_aop, p_o):
        if len(D_asp.keys()) == 0:
            return {}
        a0 = list(D_asp.keys())[0]
        nspin = D_asp[a0].shape[0]
        assert (nspin == 1) or self.is_cider_functional
        self.fr_asgLq = {}
        self.df_asgLq = {}
        self.c_asiq = {}

        for a, D_sp in D_asp.items():
            setup = setups[a]
            psetup = setup.ps_setup

            self.timer.start("coefs")
            rcalc = CiderRadialThetaDerivCalculator(setup.cider_contribs)
            expansion = CiderRadialDerivExpansion(rcalc, p_o)
            dx_sgLq, xt_sgLq = calculate_cider_paw_correction_deriv(
                expansion, setup, D_sp, DD_op=DD_aop[a]
            )
            dx_sgLq -= xt_sgLq
            self.timer.stop()
            self.timer.start("transform and convolve")
            fr_sgLq, df_sgLq, c_siq = setup.cider_contribs.calculate_y_terms(
                xt_sgLq, dx_sgLq, psetup
            )
            self.timer.stop()
            self.df_asgLq[a] = df_sgLq
            self.fr_asgLq[a] = fr_sgLq
            self.c_asiq[a] = c_siq
        return self.c_asiq

    def calculate_paw_cider_features_p2(self, setups, D_asp, D_aoiq, DD_aop, p_o):
        if len(D_asp.keys()) == 0:
            if DD_aop is None or p_o is None:
                return {}, {}
            else:
                return {}, {}, {}, {}
        a0 = list(D_asp.keys())[0]
        nspin = D_asp[a0].shape[0]
        if DD_aop is None or p_o is None:
            norb = 0
        else:
            norb = DD_aop[a0].shape[0]
        self.vfr_asgLq = {}
        self.vdf_asgLq = {}
        ae_feat_asig = {}
        ps_feat_asig = {}
        ae_dfeat_aoig = {}
        ps_dfeat_aoig = {}

        for a, D_sp in D_asp.items():
            setup = setups[a]
            psetup = setup.pa_setup
            ni = psetup.ni
            Nalpha_sm = D_aoiq[a].shape[-1]
            dv_g = get_dv(setup.xc_correction.rgd)
            df_ogLq = self.df_asgLq[a]
            ft_ogLq = np.zeros_like(df_ogLq)
            fr_ogLq = self.fr_asgLq[a]
            # TODO C function for this loop? can go in pa_setup
            for i in range(ni):
                L = psetup.lmlist_i[i]
                n = psetup.nlist_i[i]
                Dref_sq = (psetup.pfuncs_ng[n] * dv_g).dot(fr_ogLq[:, :, L, :Nalpha_sm])
                j = psetup.jlist_i[i]
                for s in range(nspin + norb):
                    ft_ogLq[s, :, L, :Nalpha_sm] += (
                        D_aoiq[a][s, i] - Dref_sq[s]
                    ) * psetup.ffuncs_jg[j][:, None]
            ft_ogLq[:] += fr_ogLq
            if norb == 0:
                rcalc = CiderRadialFeatCalculator(setup.cider_contribs)
                expansion = CiderRadialExpansion(rcalc, ft_ogLq, df_ogLq)
                res = calculate_cider_paw_correction(
                    expansion, setup, D_sp, separate_ae_ps=True
                )
                ae_feat_asig[a], ps_feat_asig[a] = res
            else:
                rcalc = CiderRadialFeatDerivCalculator(setup.cider_contribs)
                expansion = CiderRadialDerivExpansion(rcalc, p_o, ft_ogLq, df_ogLq)
                res = calculate_cider_paw_correction_deriv(
                    expansion, setup, D_sp, DD_op=DD_aop[a]
                )
                ae_feat_asig[a], ae_dfeat_aoig[a] = res[0]
                ps_feat_asig[a], ps_dfeat_aoig[a] = res[1]
        if norb == 0:
            return ae_feat_asig, ps_feat_asig
        else:
            return ae_feat_asig, ps_feat_asig, ae_dfeat_aoig, ps_dfeat_aoig


def calculate_paw_sl_features(setups, D_asp, is_mgga):
    if len(D_asp.keys()) == 0:
        return {}, {}
    ae_feat_asig = {}
    ps_feat_asig = {}
    for a, D_sp in D_asp.items():
        setup = setups[a]
        rcalc = CiderRadialDensityCalculator(setup.xc_correction, D_sp, is_mgga)
        expansion = CiderRadialExpansion(rcalc)
        ae_feat_asig[a], ps_feat_asig[a] = calculate_cider_paw_correction(
            expansion, setup, D_sp, has_cider_contribs=False, separate_ae_ps=True
        )
    return ae_feat_asig, ps_feat_asig


def calculate_paw_sl_features_deriv(setups, D_asp, DD_aop, p_o, is_mgga):
    if len(D_asp.keys()) == 0:
        return {}, {}, {}, {}
    ae_feat_asig = {}
    ps_feat_asig = {}
    ae_dfeat_aoig = {}
    ps_dfeat_aoig = {}
    for a, D_sp in D_asp.items():
        setup = setups[a]
        rcalc = CiderRadialDensityDerivCalculator(
            setup.xc_correction, D_sp, DD_aop[a], is_mgga
        )
        expansion = CiderRadialDerivExpansion(rcalc, p_o)
        (
            (ae_feat_asig[a], ae_dfeat_aoig[a]),
            (ps_feat_asig[a], ps_dfeat_aoig[a]),
        ) = calculate_cider_paw_correction_deriv(
            expansion, setup, D_sp, DD_op=DD_aop[a], has_cider_contribs=False
        )
    return ae_feat_asig, ps_feat_asig, ae_dfeat_aoig, ps_dfeat_aoig


class CiderRadialThetaDerivCalculator:
    def __init__(self, xc):
        self.xc = xc
        self.mode = "step1"

    def __call__(
        self,
        rgd,
        p_o,
        n_sLg,
        dndf_oLg,
        Y_nL,
        dndr_sLg,
        dndrdf_oLg,
        rnablaY_nLv,
        ae,
    ):
        rho_sxg, drhodf_oxg = self.xc.vec_radial_vars_occd(
            n_sLg, dndf_oLg, Y_nL, dndr_sLg, dndrdf_oLg, rnablaY_nLv, ae
        )
        return self.xc.get_paw_atom_contribs_occd(p_o, rho_sxg, drhodf_oxg)


class CiderRadialFeatCalculator:
    def __init__(self, xc):
        self.xc = xc
        self.mode = "feature"

    def __call__(self, rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, f_orLq, ae):
        rho_sxg = self.xc.vec_radial_vars(n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae)[1]
        return self.xc.get_paw_atom_feat(rho_sxg, f_orLq)


class CiderRadialDensityCalculator:
    def __init__(self, xcc, D_sp, is_mgga):
        self.xcc = xcc
        self.D_sp = D_sp
        self.is_mgga = is_mgga
        self.mode = "feature"

    def __call__(self, rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae):
        return vec_radial_vars_sl(
            self.xcc, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae, self.is_mgga, self.D_sp
        )[1]


class CiderRadialFeatDerivCalculator:
    def __init__(self, xc):
        self.xc = xc
        self.mode = "step2"

    def __call__(
        self,
        rgd,
        p_o,
        n_sLg,
        dndf_oLg,
        Y_nL,
        dndr_sLg,
        dndrdf_oLg,
        rnablaY_nLv,
        f_srLq,
        f_orLq,
        ae,
    ):
        rho_sxg, drhodf_oxg = self.xc.vec_radial_vars_occd(
            n_sLg, dndf_oLg, Y_nL, dndr_sLg, dndrdf_oLg, rnablaY_nLv, ae
        )
        return self.xc.get_paw_atom_feat_occd(p_o, rho_sxg, drhodf_oxg, f_srLq, f_orLq)


class CiderRadialDensityDerivCalculator:
    def __init__(self, xcc, D_sp, DD_op, is_mgga):
        self.xcc = xcc
        self.D_sp = D_sp
        self.DD_op = DD_op
        self.is_mgga = is_mgga
        self.mode = "rho"

    def __call__(self, n_sLg, dndf_oLg, Y_nL, dndr_sLg, dndrdf_oLg, rnablaY_nLv, ae):
        rho_sxg, drhodf_oxg = vec_radial_vars_occd(
            self.xcc,
            n_sLg,
            dndf_oLg,
            Y_nL,
            dndr_sLg,
            dndrdf_oLg,
            rnablaY_nLv,
            ae,
            self.D_sp,
            self.DD_op,
            self.is_mgga,
        )
        return rho_sxg, drhodf_oxg


class CiderRadialDerivExpansion:
    def __init__(self, rcalc, p_o, ft_orLq=None, df_orLq=None):
        self.rcalc = rcalc
        self.p_o = p_o
        self.ft_orLq = ft_orLq
        self.df_orLq = df_orLq
        assert self.rcalc.mode in ["step1", "step2", "rho"]

    def __call__(self, rgd, D_sLq, DD_oLq, n_qg, dndr_qg, nc0_sg, dnc0dr_sg, ae=True):
        n_sLg = np.dot(D_sLq, n_qg)
        n_sLg[:, 0] += nc0_sg
        dndf_oLg = np.dot(DD_oLq, n_qg)

        dndr_sLg = np.dot(D_sLq, dndr_qg)
        dndr_sLg[:, 0] += dnc0dr_sg
        dndrdf_oLg = np.dot(DD_oLq, dndr_qg)

        nspins, Lmax, nq = D_sLq.shape

        if self.rcalc.mode == "step1":
            wx_orLq = self.rcalc(
                rgd,
                self.p_o,
                n_sLg,
                dndf_oLg,
                Y_nL[:, :Lmax],
                dndr_sLg,
                dndrdf_oLg,
                rnablaY_nLv[:, :Lmax],
                ae,
            )
            return wx_orLq
        elif self.rcalc.mode == "step2":
            f_orLq = self.ft_orLq + self.df_orLq if ae else self.ft_orLq
            feat_xg, dfeat_xg = self.rcalc(
                rgd,
                self.p_o,
                n_sLg,
                dndf_oLg,
                Y_nL[:, :Lmax],
                dndr_sLg,
                dndrdf_oLg,
                rnablaY_nLv[:, :Lmax],
                f_orLq[:nspins],
                f_orLq[nspins:],
                ae,
            )
            return feat_xg, dfeat_xg
        elif self.rcalc.mode == "rho":
            feat_xg, dfeat_xg = self.rcalc(
                n_sLg,
                dndf_oLg,
                Y_nL[:, :Lmax],
                dndr_sLg,
                dndrdf_oLg,
                rnablaY_nLv[:, :Lmax],
                ae,
            )
            return feat_xg, dfeat_xg
        else:
            raise ValueError
