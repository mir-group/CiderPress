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
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor
from gpaw.sphere.lebedev import Y_nL
from gpaw.xc.vdw import spline
from scipy.interpolate import interp1d
from scipy.linalg import cho_factor, cho_solve

from ciderpress.dft import pwutil
from ciderpress.gpaw.config import GPAW_USE_NEW_PLANS
from ciderpress.gpaw.etb_util import ETBProjector
from ciderpress.gpaw.fit_paw_gauss_pot import (
    construct_full_p_matrices,
    get_delta_lpg,
    get_delta_lpk,
    get_dv,
    get_dvk,
    get_ffunc3,
    get_p11_matrix,
    get_p12_p21_matrix,
    get_p22_matrix,
    get_pfunc2_norm,
    get_pfuncs_k,
    get_phi_iabg,
    get_phi_iabk,
)
from ciderpress.gpaw.gpaw_grids import SBTFullGridDescriptor
from ciderpress.gpaw.interp_paw import (
    CiderRadialEnergyCalculator,
    CiderRadialEnergyCalculator2,
    CiderRadialExpansion,
    CiderRadialFeatureCalculator,
    DiffPAWXCCorrection,
    calculate_cider_paw_correction,
)
from ciderpress.gpaw.paw_cider_kernel import (
    MetaPAWCiderContribUtils,
    PAWCiderContribUtils,
    PAWCiderKernelShell,
)


def _interpc(func, ref_rgd, r_g):
    return interp1d(
        np.arange(ref_rgd.r_g.size),
        func,
        kind="cubic",
        axis=-1,
    )(ref_rgd.r2g(r_g))


def get_ag_indices(gd, shape, spos_c, rmax, buffer=0):
    ng3 = np.prod(shape)
    center = np.round(spos_c * shape).astype(int)
    disp = np.empty(3, dtype=int)
    lattice = gd.cell_cv
    vol = np.abs(np.linalg.det(lattice))
    dv = vol / ng3
    for i in range(3):
        res = np.cross(lattice[(i + 1) % 3], lattice[(i + 2) % 3])
        # TODO unnecessarily conservative buffer?
        disp[i] = np.ceil(np.linalg.norm(res) * rmax / vol * shape[i]) + 1 + buffer
    indices = [
        np.arange(center[i] - disp[i], center[i] + disp[i] + 1) for i in range(3)
    ]
    xtmp = np.meshgrid(*indices, indexing="ij")
    sposa_g = np.stack(xtmp, axis=3) / shape - spos_c
    cart_sposa_g = sposa_g.dot(gd.cell_cv)
    rad_g = np.linalg.norm(cart_sposa_g, axis=-1)
    rhat_g = cart_sposa_g / np.maximum(rad_g[..., None], 1e-8)
    rhat_g[rad_g < 1e-8] = 0.0
    return center, disp, indices, rad_g, rhat_g, dv, sposa_g[0, 0, 0]


def get_atomic_convolution(self, y_sbLk, xcc):
    k_g = xcc.big_rgd.k_g
    return np.ascontiguousarray(self.calc_y_sbLk(y_sbLk.transpose(0, 2, 3, 1), ks=k_g))


def calculate_paw_cider_features(self, setups, D_asp):
    self.yref_asbLg = {}
    if len(D_asp.keys()) == 0:
        return {}, {}
    a0 = list(D_asp.keys())[0]
    nspin = D_asp[a0].shape[0]
    assert (nspin == 1) or self.is_cider_functional
    y_asbLg = {}
    c_sabi = {s: {} for s in range(nspin)}

    for a, D_sp in D_asp.items():
        setup = setups[a]
        psetup = setup.pasdw_setup
        Nalpha_sm = psetup.phi_jabg.shape[1]
        xcc = setup.nlxc_correction
        rgd = setup.nlxc_correction.big_rgd
        slgd = setup.xc_correction.rgd
        RCUT = np.max(setup.rcut_j)

        self.timer.start("coefs")
        rcalc = CiderRadialFeatureCalculator(setup.cider_contribs)
        expansion = CiderRadialExpansion(rcalc)
        dx_sbgL, dxt_sbgL = calculate_cider_paw_correction(
            expansion, setup, D_sp, separate_ae_ps=True
        )
        dx_sbgL -= dxt_sbgL
        self.timer.stop()

        rcut = RCUT
        rmax = slgd.r_g[-1]
        fcut = 0.5 + 0.5 * np.cos(np.pi * (slgd.r_g - rcut) / (rmax - rcut))
        fcut[slgd.r_g < rcut] = 1.0
        fcut[slgd.r_g > rmax] = 0.0
        dxt_sbgL *= fcut[:, None]

        self.timer.start("transform and convolve")
        dx_sbLk = setup.etb_projector.r2k(dx_sbgL.transpose(0, 1, 3, 2))
        dxt_sbLk = setup.etb_projector.r2k(dxt_sbgL.transpose(0, 1, 3, 2))

        y_sbLk = get_atomic_convolution(setup.cider_contribs, dx_sbLk, xcc)
        yref_sbLk = get_atomic_convolution(setup.cider_contribs, dxt_sbLk, xcc)
        yref_sbLg = 2 / np.pi * setup.etb_projector.k2r(yref_sbLk)
        self.timer.stop()

        self.timer.start("separate long and short-range")
        c_sib, df_sLpb = psetup.get_c_and_df(y_sbLk[:, :Nalpha_sm], realspace=False)
        yt_sbLg = psetup.get_f_realspace_contribs(c_sib, sl=True)
        df_sLpb = np.append(
            df_sLpb,
            psetup.get_df_only(y_sbLk[:, Nalpha_sm:], rgd, sl=False, realspace=False),
            axis=-1,
        )
        df_sbLg = psetup.get_df_realspace_contribs(df_sLpb, sl=True)
        y_asbLg[a] = df_sbLg.copy()
        self.yref_asbLg[a] = yref_sbLg
        self.yref_asbLg[a][:, :Nalpha_sm] += yt_sbLg
        self.timer.stop()
        for s in range(nspin):
            c_sabi[s][a] = c_sib[s].T

    return c_sabi, y_asbLg


def calculate_paw_cider_energy(self, setups, D_asp, D_sabi, df_asbLg):
    deltaE = {}
    deltaV = {}
    if len(D_asp.keys()) == 0:
        return {}, {}, {}
    a0 = list(D_asp.keys())[0]
    nspin = D_asp[a0].shape[0]
    assert (nspin == 1) or self.is_cider_functional
    self.f_asbgr = {}
    self.dvyref_asbLg = {}
    self.dvdf_asbLg = {}
    dvD_sabi = {s: {} for s in range(nspin)}

    for a, D_sp in D_asp.items():
        setup = setups[a]
        psetup = setup.paonly_setup
        ni = psetup.ni
        slgd = setup.xc_correction.rgd
        Nalpha_sm = D_sabi[0][a].shape[0]

        self.timer.start("XC part")
        dv_g = get_dv(slgd)
        yref_sbLg = self.yref_asbLg[a]
        f_sbLg = np.zeros_like(df_asbLg[a])
        for i in range(ni):
            L = psetup.lmlist_i[i]
            n = psetup.nlist_i[i]
            Dref_sb = np.einsum(
                "sbg,g->sb",
                yref_sbLg[:, :Nalpha_sm, L],
                psetup.pfuncs_ng[n] * dv_g,
            )
            j = psetup.jlist_i[i]
            for s in range(nspin):
                f_sbLg[s, :Nalpha_sm, L, :] += (
                    D_sabi[s][a][:, i, None] - Dref_sb[s, :, None]
                ) * psetup.ffuncs_jg[j]
        f_sbLg[:, :] += yref_sbLg
        if GPAW_USE_NEW_PLANS:
            rcalc = CiderRadialEnergyCalculator2(setup.cider_contribs, a)
        else:
            rcalc = CiderRadialEnergyCalculator(setup.cider_contribs, a, mode="energy")
        expansion = CiderRadialExpansion(
            rcalc,
            f_sbLg,
            df_asbLg[a],
            f_in_sph_harm=True,
        )
        deltaE[a], deltaV[a], dvdf_sbgL, xt_sbgL = calculate_cider_paw_correction(
            expansion,
            setup,
            D_sp,
            separate_ae_ps=True,
        )
        dvf_sbLg = (dvdf_sbgL - xt_sbgL).transpose(0, 1, 3, 2)
        dvdf_sbLg = dvdf_sbgL.transpose(0, 1, 3, 2)

        self.f_asbgr[a] = np.einsum("...Lg,nL->...gn", f_sbLg, Y_nL)

        dvyref_sbLg = dvf_sbLg.copy()
        for s in range(nspin):
            dvD_sabi[s][a] = np.zeros((Nalpha_sm, ni))
        for i in range(ni):
            L = psetup.lmlist_i[i]
            n = psetup.nlist_i[i]
            j = psetup.jlist_i[i]
            dvD_sb = np.einsum(
                "sbg,g->sb",
                dvf_sbLg[:, :Nalpha_sm, L],
                psetup.ffuncs_jg[j] * dv_g,
            )
            for s in range(nspin):
                dvD_sabi[s][a][:, i] = dvD_sb[s]
            dvyref_sbLg[:, :Nalpha_sm, L, :] -= dvD_sb[..., None] * psetup.pfuncs_ng[n]
        self.dvyref_asbLg[a] = dvyref_sbLg
        self.dvdf_asbLg[a] = dvdf_sbLg

        self.timer.stop("XC part")

    return dvD_sabi, deltaE, deltaV


def calculate_paw_cider_potential(self, setups, D_asp, vc_sabi):
    dH_asp = {}
    if len(D_asp.keys()) == 0:
        return {}
    a0 = list(D_asp.keys())[0]
    nspin = D_asp[a0].shape[0]
    assert (nspin == 1) or self.is_cider_functional

    for a, D_sp in D_asp.items():
        setup = setups[a]
        psetup = setup.pasdw_setup
        ni = psetup.ni
        xcc = setup.nlxc_correction
        slgd = setup.xc_correction.rgd
        Nalpha_sm = psetup.phi_jabg.shape[1]
        RCUT = np.max(setup.rcut_j)

        vc_sib = np.zeros((nspin, ni, Nalpha_sm))
        for s in range(nspin):
            vc_sib[s] = vc_sabi[s][a].T
        vyref_sbLg = self.dvyref_asbLg[a]
        vdf_sbLg = self.dvdf_asbLg[a]

        vyt_sbLg = vyref_sbLg[:, :Nalpha_sm]
        vc_sib += psetup.get_vf_realspace_contribs(vyt_sbLg, slgd, sl=True)
        vdf_sLpb = psetup.get_vdf_realspace_contribs(vdf_sbLg, slgd, sl=True)
        vy_sbLk = psetup.get_v_from_c_and_df(
            vc_sib, vdf_sLpb[..., :Nalpha_sm], realspace=False
        )
        vy_sbLk = np.append(
            vy_sbLk,
            psetup.get_vdf_only(vdf_sLpb[..., Nalpha_sm:], realspace=False),
            axis=1,
        )

        vyref_sbLk = setup.etb_projector.r2k(vyref_sbLg)
        vdx_sbLk = get_atomic_convolution(setup.cider_contribs, vy_sbLk, xcc)
        vxt_sbLk = get_atomic_convolution(setup.cider_contribs, vyref_sbLk, xcc)

        vdx_sbLg = 2 / np.pi * setup.etb_projector.k2r(vdx_sbLk)
        vxt_sbLg = 2 / np.pi * setup.etb_projector.k2r(vxt_sbLk)

        rcut = RCUT
        rmax = slgd.r_g[-1]
        fcut = 0.5 + 0.5 * np.cos(np.pi * (slgd.r_g - rcut) / (rmax - rcut))
        fcut[slgd.r_g < rcut] = 1.0
        fcut[slgd.r_g > rmax] = 0.0
        vxt_sbLg *= fcut

        F_sbLg = vdx_sbLg - vxt_sbLg
        y_sbLg = vxt_sbLg.copy()

        rcalc = CiderRadialEnergyCalculator(setup.cider_contribs, a, mode="potential")
        expansion = CiderRadialExpansion(
            rcalc,
            F_sbLg,
            y_sbLg,
            f_in_sph_harm=True,
        )
        dH_asp[a] = calculate_cider_paw_correction(
            expansion, setup, D_sp, separate_ae_ps=True
        )

    return dH_asp


def calculate_paw_feat_corrections_pasdw(
    self,
    setups,
    D_asp,
    D_sabi=None,
    df_asbLg=None,
    vc_sabi=None,
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
        D_sabi: Atomic PAW PASDW projections
        df_asbLg: delta-F_beta convolutions
        vc_sabi: functional derivatives with respect to coefficients c_sabi
    """
    if D_sabi is None and vc_sabi is None:
        return calculate_paw_cider_features(self, setups, D_asp)
    elif D_sabi is not None:
        assert df_asbLg is not None
        return calculate_paw_cider_energy(self, setups, D_asp, D_sabi, df_asbLg)
    else:
        assert vc_sabi is not None
        return calculate_paw_cider_potential(self, setups, D_asp, vc_sabi)


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
        # rcut = np.max(setup.rcut_j)
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


class PASDWCiderKernel(PAWCiderKernelShell):

    calculate_paw_feat_corrections = calculate_paw_feat_corrections_pasdw

    def __init__(
        self,
        cider_kernel,
        Nalpha,
        lambd,
        encut,
        world,
        timer,
        Nalpha_small,
        cut_xcgrid,
        **kwargs
    ):
        self.gd = kwargs.get("gd")
        self.bas_exp_fit = kwargs.get("bas_exp_fit")
        if (self.gd is None) or (self.bas_exp_fit is None):
            raise ValueError("Must provide gd and bas_exp_fit for PASDWCiderKernel")
        super(PASDWCiderKernel, self).__init__(world, timer)
        self.cider_kernel = cider_kernel
        self.lambd = lambd
        self.Nalpha_small = Nalpha_small
        self.cut_xcgrid = cut_xcgrid
        self._amin = np.min(self.bas_exp_fit)
        self.is_mgga = kwargs.get("is_mgga") or False

    def initialize_more_things(self, setups=None):
        self.nspin = self.dens.nt_sg.shape[0]

        if setups is None:
            setups = self.dens.setups
        for a, setup in enumerate(setups):
            if not hasattr(setup, "nlxc_correction"):
                # rcut = setup.xc_correction.rgd.r_g[-1]
                setup.xc_correction = DiffPAWXCCorrection.from_setup(
                    setup, build_kinetic=self.is_mgga
                )
                # TODO is this grid sufficient (large enough/small enough k and r)?
                setup.nlxc_correction = SBTGridContainer.from_setup(
                    # setup, rmin=setup.xc_correction.rgd.r_g[0]+1e-5, N=1024, encut=1e6, d=0.018
                    setup,
                    rmin=1e-4,
                    N=1024,
                    encut=5e6,
                    d=0.015,
                )

                encut = setup.Z**2 * 10
                if encut - 1e-7 <= np.max(self.bas_exp_fit):
                    encut0 = np.max(self.bas_exp_fit)
                    Nalpha = self.bas_exp_fit.size
                else:
                    Nalpha = (
                        int(np.ceil(np.log(encut / self._amin) / np.log(self.lambd)))
                        + 1
                    )
                    encut0 = self._amin * self.lambd ** (Nalpha - 1)
                    assert encut0 >= encut - 1e-6, "Math went wrong {} {}".format(
                        encut0, encut
                    )
                    assert (
                        encut0 / self.lambd < encut
                    ), "Math went wrong {} {} {} {}".format(
                        encut0, encut, self.lambd, encut0 / self.lambd
                    )
                PAWClass = (
                    MetaPAWCiderContribUtils if self.is_mgga else PAWCiderContribUtils
                )
                setup.cider_contribs = PAWClass(
                    self.cider_kernel,
                    self.dens.nt_sg.shape[0],
                    encut0,
                    self.lambd,
                    self.timer,
                    Nalpha,
                    self.cut_xcgrid,
                )
                # assert abs(np.max(setup.cider_contribs.bas_exp) - encut0) < 1e-10
                if setup.cider_contribs._plan is not None:
                    assert (
                        abs(np.max(setup.cider_contribs._plan.alphas) - encut0) < 1e-10
                    )

                setup.pasdw_setup = PSOnlySetup.from_setup(
                    setup,
                    self.bas_exp_fit,
                    self.bas_exp_fit,
                    pmax=encut0,
                )
                setup.paonly_setup = PAOnlySetup.from_setup(setup)
                setup.etb_projector = ETBProjector(
                    setup.Z,
                    setup.xc_correction.rgd,
                    setup.nlxc_correction.big_rgd,
                    alpha_min=2 * self._amin,
                )


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
    ):
        self.nn = pfuncs_ng.shape[0]
        self.ng = pfuncs_ng.shape[1]
        self.nt = pfuncs_ntp.shape[1]
        self.ni = lmlist_i.shape[0]
        self.nj = len(nlist_j)
        self.lmax = np.max(llist_j)
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


class PAOnlySetup(PASDWData):
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
                get_ffunc3(n, self.interp_rgd.r_g, self.rcut_func)
                * self.get_filt(self.interp_rgd.r_g),
            )

        self.rcut_func = (
            self.rcut_feat
        )  # TODO should work but dangerous, make cleaner way to set rcut_func

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
        RCUT_FAC = 1.0  # TODO was 1.3
        rcut_feat = np.max(setup.rcut_j) * RCUT_FAC
        # rcut_func = rcut_feat * 2.0
        rmax = np.max(setup.xc_correction.rgd.r_g)
        rcut_func = rmax * 0.8

        nt = 100
        interp_rgd = EquidistantRadialGridDescriptor(h=rcut_func / (nt - 1), N=nt)
        interp_r_g = interp_rgd.r_g
        ng = rgd.r_g.size

        if setup.Z > 1000:
            nlist_j = [0, 2, 4, 6, 8, 1, 3, 5, 7, 2, 4, 6, 8, 3, 5, 7, 4, 6, 8]
            llist_j = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4]
            nbas_lst = [5, 4, 4, 3, 3]
            nbas_loc = np.append([0], np.cumsum(nbas_lst)).astype(np.int32)
            lmlist_i = []
            jlist_i = []
        elif setup.Z > 18:
            nlist_j = [0, 2, 4, 6, 1, 3, 5, 2, 4, 6, 3, 5, 4, 6]
            llist_j = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
            nbas_lst = [4, 3, 3, 2, 2]
            nbas_loc = np.append([0], np.cumsum(nbas_lst)).astype(np.int32)
            lmlist_i = []
            jlist_i = []
        elif setup.Z > 0:
            nlist_j = [0, 2, 4, 1, 3, 2, 4, 3, 4]
            llist_j = [0, 0, 0, 1, 1, 2, 2, 3, 4]
            nbas_lst = [3, 2, 2, 1, 1]
            nbas_loc = np.append([0], np.cumsum(nbas_lst)).astype(np.int32)
            lmlist_i = []
            jlist_i = []
        else:
            nlist_j = [0, 2, 1, 2]
            llist_j = [0, 0, 1, 2]
            nbas_lst = [2, 1, 1]
            nbas_loc = np.append([0], np.cumsum(nbas_lst)).astype(np.int32)
            lmlist_i = []
            jlist_i = []

        nn = np.max(nlist_j) + 1

        pfuncs_ng = np.zeros((nn, ng), dtype=np.float64, order="C")
        pfuncs_ntp = np.zeros((nn, nt, 4), dtype=np.float64, order="C")
        for n in range(nn):
            pfunc_t = get_ffunc3(n, interp_r_g, rcut_func)
            pfuncs_ng[n, :] = get_ffunc3(n, rgd.r_g, rcut_func)
            pfuncs_ntp[n, :, :] = spline(
                np.arange(interp_r_g.size).astype(np.float64), pfunc_t
            )

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


class PSOnlySetup(PASDWData):
    def _initialize_ffunc(self):
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

    def initialize_a2g(self):
        self.w_b = np.ones(self.alphas_ae.size)  # / self.alpha_norms**2
        # self.w_b = self.alphas_ae**-0.5
        if self.pmax is None:
            pmax = 4 * np.max(self.alphas_ae)
        else:
            pmax = 2 * self.pmax
        pmin = 2 / self.rcut_feat**2
        lambd = 1.8
        N = int(np.ceil(np.log(pmax / pmin) / np.log(lambd))) + 1
        dd = np.log(pmax / pmin) / (N - 1)
        alphas_bas = pmax * np.exp(-dd * np.arange(N))
        self.delta_lpg = get_delta_lpg(
            alphas_bas, self.rcut_feat, self.sbt_rgd, thr=pmin, lmax=self.lmax
        )
        self.delta_lpk = get_delta_lpk(self.delta_lpg, self.sbt_rgd)
        self.deltasl_lpg = get_delta_lpg(
            alphas_bas, self.rcut_feat, self.slrgd, thr=pmin, lmax=self.lmax
        )
        rgd = self.sbt_rgd
        pfuncs_jg = np.stack([self.pfuncs_ng2[n] for n in self.nlist_j])
        pfuncs_k = get_pfuncs_k(pfuncs_jg, self.llist_j, rgd, ns=self.nlist_j)
        phi_jabk = get_phi_iabk(pfuncs_k, rgd.k_g, self.alphas, betas=self.alphas_ae)
        # phi_jabk[:] *= self.alpha_norms[:, None, None]
        # phi_jabk[:] *= self.alpha_norms[:, None]
        REG11 = 1e-6
        REG22 = 1e-5
        FAC22 = 1e-2
        phi_jabg = get_phi_iabg(phi_jabk, self.llist_j, rgd)
        p11_lpp = get_p11_matrix(self.delta_lpg, rgd, reg=REG11)
        p12_pbja, p21_japb = get_p12_p21_matrix(
            self.delta_lpg, phi_jabg, rgd, self.llist_j, self.w_b
        )
        p22_ljaja = []
        for l in range(self.lmax + 1):
            p22_ljaja.append(
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
            p22_ljaja[-1] += FAC22 * get_p22_matrix(
                phi_jabg,
                rgd,
                self.rcut_feat,
                l,
                self.w_b,
                self.nbas_loc,
                cut_func=True,
            )
        p_l_ii = construct_full_p_matrices(
            p11_lpp,
            p12_pbja,
            p21_japb,
            p22_ljaja,
            self.w_b,
            self.nbas_loc,
        )
        self.phi_jabk = phi_jabk
        self.phi_jabg = phi_jabg
        self.p_l_ii = p_l_ii
        self.pinv_l_ii = []
        self.p_cl_l_ii = []
        for l in range(self.lmax + 1):
            c_and_l = cho_factor(p_l_ii[l])
            idn = np.identity(p_l_ii[l].shape[0])
            self.p_cl_l_ii.append(c_and_l)
            self.pinv_l_ii.append(cho_solve(c_and_l, idn))
        self.phi_sr_jabg = _interpc(self.phi_jabg, rgd, self.slrgd.r_g)

    def get_c_and_df(self, y_sbLg, realspace=True):
        nspin, nb, Lmax, ng = y_sbLg.shape
        rgd = self.sbt_rgd
        Nalpha_sm = self.alphas.size
        NP = self.delta_lpg.shape[1]
        c_sib = np.zeros((nspin, self.ni, Nalpha_sm))
        df_sLpb = np.zeros((nspin, Lmax, NP, nb))
        for s in range(nspin):
            for l in range(self.lmax + 1):
                Lmin = l * l
                dL = 2 * l + 1
                for L in range(Lmin, Lmin + dL):
                    if realspace:
                        b1 = self.get_b1_pb(y_sbLg[s, :, L, :], l, rgd)
                        b2 = self.get_b2_ja(y_sbLg[s, :, L, :], l, rgd)
                    else:
                        b1 = self.get_b1_pb_recip(y_sbLg[s, :, L, :], l, rgd)
                        b2 = self.get_b2_ja_recip(y_sbLg[s, :, L, :], l, rgd)
                    b = self.cat_b_vecs(b1, b2)
                    # x = self.pinv_l_ii[l].dot(b)
                    x = cho_solve(self.p_cl_l_ii[l], b)
                    N1 = b1.size
                    df_sLpb[s, L] += x[:N1].reshape(NP, nb)
                    ilist = self.lmlist_i == L
                    c_sib[s, ilist, :] += x[N1:].reshape(-1, Nalpha_sm)
        return c_sib, df_sLpb

    def get_v_from_c_and_df(self, vc_sib, vdf_sLpb, realspace=True):
        nspin, Lmax, NP, nb = vdf_sLpb.shape
        rgd = self.sbt_rgd
        Nalpha_sm = self.alphas.size
        ng = rgd.r_g.size
        vy_sbLg = np.zeros((nspin, nb, Lmax, ng))
        for s in range(nspin):
            for l in range(self.lmax + 1):
                Lmin = l * l
                dL = 2 * l + 1
                for L in range(Lmin, Lmin + dL):
                    ilist = self.lmlist_i == L
                    x1 = vdf_sLpb[s, L].flatten()
                    x2 = vc_sib[s, ilist, :].flatten()
                    x = self.cat_b_vecs(x1, x2)
                    # b = self.pinv_l_ii[l].dot(x)
                    b = cho_solve(self.p_cl_l_ii[l], x)
                    N1 = x1.size
                    if realspace:
                        vy_sbLg[s, :, L, :] += self.get_vb1_pb(
                            b[:N1].reshape(NP, nb), l
                        )
                        vy_sbLg[s, :, L, :] += self.get_vb2_ja(
                            b[N1:].reshape(-1, Nalpha_sm), l
                        )
                    else:
                        vy_sbLg[s, :, L, :] += self.get_vb1_pb_recip(
                            b[:N1].reshape(NP, nb), l
                        )
                        vy_sbLg[s, :, L, :] += self.get_vb2_ja_recip(
                            b[N1:].reshape(-1, Nalpha_sm), l
                        )
        return vy_sbLg

    def get_f_realspace_contribs(self, c_sib, sl=False):
        Nalpha = self.alphas_ae.size
        phi_jabg = self.phi_sr_jabg if sl else self.phi_jabg
        ng = phi_jabg.shape[-1]
        nspin, _, nb = c_sib.shape
        Lmax = 25  # np.max(self.lmlist_i) + 1
        yt_sbLg = np.zeros((nspin, Nalpha, Lmax, ng))
        for i in range(self.ni):
            L = self.lmlist_i[i]
            j = self.jlist_i[i]
            yt_sbLg[:, :, L, :] += np.einsum(
                "abg,sa->sbg",
                phi_jabg[j],
                c_sib[:, i, :],
            )
        return yt_sbLg

    def get_vf_realspace_contribs(self, vyt_sbLg, rgd, sl=False):
        if rgd is None:
            dv_g = np.ones(vyt_sbLg.shape[-1])
        else:
            dv_g = get_dv(rgd)
        phi_jabg = self.phi_sr_jabg if sl else self.phi_jabg
        nspin, nb = vyt_sbLg.shape[:2]
        vc_sib = np.zeros((nspin, self.ni, nb))
        for i in range(self.ni):
            L = self.lmlist_i[i]
            j = self.jlist_i[i]
            vc_sib[:, i, :] += np.einsum(
                "abg,sbg,g->sa",
                phi_jabg[j],
                vyt_sbLg[:, :, L, :],
                dv_g,
            )
        return vc_sib

    def get_df_realspace_contribs(self, df_sLpb, sl=False):
        df_sbLp = df_sLpb.transpose(0, 3, 1, 2)
        delta_lpg = self.deltasl_lpg if sl else self.delta_lpg
        ng = delta_lpg.shape[-1]
        nspin, Lmax, NP, nb = df_sLpb.shape
        y_sbLg = np.zeros((nspin, nb, Lmax, ng))
        for s in range(nspin):
            for l in range(self.lmax + 1):
                for L in range(l * l, (l + 1) * (l + 1)):
                    y_sbLg[s, :, L, :] += df_sbLp[s, :, L, :].dot(delta_lpg[l])
        return y_sbLg

    def get_df_only(self, y_sbLg, rgd, sl=False, realspace=True):
        nspin, nb, Lmax, ng = y_sbLg.shape
        NP = self.delta_lpg.shape[1]
        if realspace:
            delta_lpg = self.deltasl_lpg if sl else self.delta_lpg
            dv_g = get_dv(rgd)
        else:
            assert not sl
            delta_lpg = self.delta_lpk
            dv_g = get_dvk(rgd)
        df_sLpb = np.zeros((nspin, Lmax, NP, nb))
        for l in range(self.lmax + 1):
            L0 = l * l
            dL = 2 * l + 1
            for L in range(L0, L0 + dL):
                df_sLpb[:, L, :, :] += np.einsum(
                    "sbg,pg,g->spb", y_sbLg[:, :, L, :], delta_lpg[l], dv_g
                )
        return df_sLpb

    def get_vdf_only(self, df_sLpb, sl=False, realspace=True):
        nspin, Lmax, NP, nb = df_sLpb.shape
        if realspace:
            delta_lpg = self.deltasl_lpg if sl else self.delta_lpg
        else:
            assert not sl
            delta_lpg = self.delta_lpk
        ng = delta_lpg[0].shape[-1]
        y_sbLg = np.zeros((nspin, nb, Lmax, ng))
        for l in range(self.lmax + 1):
            L0 = l * l
            dL = 2 * l + 1
            for L in range(L0, L0 + dL):
                y_sbLg[:, :, L, :] += np.einsum(
                    "spb,pg->sbg", df_sLpb[:, L, :, :], delta_lpg[l]
                )
        return y_sbLg

    def get_vdf_realspace_contribs(self, vy_sbLg, rgd, sl=False):
        if rgd is None:
            dv_g = np.ones(vy_sbLg.shape[-1])
        else:
            dv_g = get_dv(rgd)
        delta_lpg = self.deltasl_lpg if sl else self.delta_lpg
        NP = delta_lpg.shape[1]
        nspin, nb, Lmax, ng = vy_sbLg.shape
        vdf_sbLp = np.zeros((nspin, nb, Lmax, NP))
        for s in range(nspin):
            for l in range(self.lmax + 1):
                for L in range(l * l, (l + 1) * (l + 1)):
                    vdf_sbLp[s, :, L, :] += np.einsum(
                        "bg,pg,g->bp",
                        vy_sbLg[s, :, L, :],
                        delta_lpg[l],
                        dv_g,
                    )
        return vdf_sbLp.transpose(0, 2, 3, 1)

    def get_b1_pb(self, FL_bg, l, rgd):
        dv_g = get_dv(rgd)
        return np.einsum("bg,pg,g->pb", FL_bg, self.delta_lpg[l], dv_g) * self.w_b

    def get_b2_ja(self, FL_bg, l, rgd):
        dv_g = get_dv(rgd)
        return np.einsum(
            "bg,jabg,b,g->ja",
            FL_bg,
            self.phi_jabg[self.nbas_loc[l] : self.nbas_loc[l + 1]],
            self.w_b,
            dv_g,
        )

    def get_vb1_pb(self, b_pb, l):
        return np.einsum("pb,pg->bg", b_pb, self.delta_lpg[l]) * self.w_b[:, None]

    def get_vb2_ja(self, b_ja, l):
        return (
            np.einsum(
                "ja,jabg->bg",
                b_ja,
                self.phi_jabg[self.nbas_loc[l] : self.nbas_loc[l + 1]],
            )
            * self.w_b[:, None]
        )

    def get_b1_pb_recip(self, FL_bg, l, rgd):
        dv_g = get_dvk(rgd)
        return np.einsum("bg,pg,g->pb", FL_bg, self.delta_lpk[l], dv_g) * self.w_b

    def get_b2_ja_recip(self, FL_bg, l, rgd):
        dv_g = get_dvk(rgd)
        return np.einsum(
            "bg,jabg,b,g->ja",
            FL_bg,
            self.phi_jabk[self.nbas_loc[l] : self.nbas_loc[l + 1]],
            self.w_b,
            dv_g,
        )

    def get_vb1_pb_recip(self, b_pb, l):
        return np.einsum("pb,pg->bg", b_pb, self.delta_lpk[l]) * self.w_b[:, None]

    def get_vb2_ja_recip(self, b_ja, l):
        return (
            np.einsum(
                "ja,jabg->bg",
                b_ja,
                self.phi_jabk[self.nbas_loc[l] : self.nbas_loc[l + 1]],
            )
            * self.w_b[:, None]
        )

    def cat_b_vecs(self, b1, b2):
        return np.append(b1.flatten(), b2.flatten())

    @classmethod
    def from_setup(cls, setup, alphas, alphas_ae, alpha_norms=None, pmax=None):

        rgd = setup.xc_correction.rgd
        rcut_feat = np.max(setup.rcut_j)
        rcut_func = rcut_feat * 1.00

        if setup.Z > 100:
            nlist_j = [0, 2, 4, 6, 1, 3, 5, 2, 4, 6, 3, 5, 4, 6]
            llist_j = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
            nbas_lst = [4, 3, 3, 2, 2]
            nbas_loc = np.append([0], np.cumsum(nbas_lst)).astype(np.int32)
            lmlist_i = []
            jlist_i = []
        elif setup.Z > 0:
            nlist_j = [0, 2, 4, 1, 3, 2, 4, 3, 4]
            llist_j = [0, 0, 0, 1, 1, 2, 2, 3, 4]
            nbas_lst = [3, 2, 2, 1, 1]
            nbas_loc = np.append([0], np.cumsum(nbas_lst)).astype(np.int32)
            lmlist_i = []
            jlist_i = []
        else:
            nlist_j = [0, 2, 1, 2]
            llist_j = [0, 0, 1, 2]
            nbas_lst = [2, 1, 1]
            nbas_loc = np.append([0], np.cumsum(nbas_lst)).astype(np.int32)
            lmlist_i = []
            jlist_i = []

        nt = 100
        interp_rgd = EquidistantRadialGridDescriptor(h=rcut_func / (nt - 1), N=nt)
        interp_r_g = interp_rgd.r_g
        ng = rgd.r_g.size
        sbt_rgd = setup.nlxc_correction.big_rgd
        ng2 = sbt_rgd.r_g.size
        nn = np.max(nlist_j) + 1

        pfuncs_ng = np.zeros((nn, ng), dtype=np.float64, order="C")
        pfuncs_ng2 = np.zeros((nn, ng2), dtype=np.float64, order="C")
        pfuncs_ntp = np.zeros((nn, nt, 4), dtype=np.float64, order="C")
        pfuncs_nt = np.zeros((nn, nt), dtype=np.float64, order="C")
        for n in range(nn):
            pfuncs_nt[n, :] = get_pfunc2_norm(n, interp_r_g, rcut_func)
            pfuncs_ng[n, :] = get_pfunc2_norm(n, rgd.r_g, rcut_func)
            pfuncs_ng2[n, :] = get_pfunc2_norm(n, sbt_rgd.r_g, rcut_func)

        """
        # THIS CODE BLOCK orthogonalizes the basis functions,
        # but does not seem to be important currently.
        def _helper(tmp_ng):
            ovlp = np.einsum('ng,mg,g->nm', tmp_ng, tmp_ng, get_dv(rgd))
            L = np.linalg.cholesky(ovlp)
            tmp2_ng = np.linalg.solve(L, tmp_ng)
            ovlp2 = np.einsum('ng,mg,g->nm', tmp_ng, tmp2_ng, get_dv(rgd))
            ovlp3 = np.einsum('ng,mg,g->nm', tmp2_ng, tmp2_ng, get_dv(rgd))
            rval = np.linalg.solve(L, np.identity(tmp_ng.shape[0]))
            return rval
        C_nn = np.zeros((nn,nn))
        C_nn[-1::-2,-1::-2] = _helper(pfuncs_ng[-1::-2])
        C_nn[-2::-2,-2::-2] = _helper(pfuncs_ng[-2::-2])
        pfuncs_nt = C_nn.dot(pfuncs_nt)
        pfuncs_ng = C_nn.dot(pfuncs_ng)
        pfuncs_ng2 = C_nn.dot(pfuncs_ng2)
        """

        for n in range(nn):
            pfuncs_ntp[n, :, :] = spline(
                np.arange(interp_r_g.size).astype(np.float64), pfuncs_nt[n]
            )

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
            alphas_ae=alphas_ae,
            alpha_norms=alpha_norms,
            pmax=pmax,
        )

        psetup.pfuncs_ng2 = pfuncs_ng2
        psetup._initialize_ffunc()
        psetup.initialize_a2g()

        return psetup


class AtomPASDWSlice:
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
        if len(indset) > 0:
            self.indset = np.ascontiguousarray(np.stack(indset).T.astype(np.int32))
        else:
            self.indset = np.empty(0, dtype=np.int32)
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

    @classmethod
    def from_gd_and_setup(
        cls,
        gd,
        spos_c,
        psetup,
        rmax=0,
        sphere=True,
        ovlp_fit=False,
        store_funcs=False,
    ):
        rgd = psetup.interp_rgd
        if rmax == 0:
            rmax = psetup.rcut
        shape = gd.get_size_of_global_array()
        center, disp, indices, rad_g, rhat_gv, dv, sposa_g = get_ag_indices(
            gd, shape, spos_c, rmax, buffer=0
        )

        indices = [ind % s for ind, s in zip(indices, shape)]
        indset = np.meshgrid(*indices, indexing="ij")
        indset = [inds.flatten() for inds in indset]
        rad_g = rad_g.flatten()
        rhat_gv = rhat_gv.reshape(-1, 3)

        if sphere:
            cond = rad_g <= rmax
            indset = [inds[cond] for inds in indset]
            rad_g = rad_g[cond]
            rhat_gv = rhat_gv[cond]

        h = rgd.r_g[1] - rgd.r_g[0]
        dg = rad_g / h  # TODO only for equidist grid
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
        ylm = pwutil.recursive_sph_harm_t2(Lmax, self.rhat_gv)
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
        ylm, dylm = pwutil.recursive_sph_harm_t2_deriv(Lmax, self.rhat_gv)
        dylm /= self.rad_g[:, None, None] + 1e-8  # TODO right amount of regularization?
        rodylm = np.ascontiguousarray(np.einsum("gv,gvL->Lg", self.rhat_gv, dylm))
        # dylm = np.dot(dylm, drhat_g.T)
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
        ylm = pwutil.recursive_sph_harm_t2(Lmax, self.rhat_gv)
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
            B_vfq = -1 * np.einsum("vpf,pq->vfq", dovlp_vpf, self.sinv_pf)
            ni = ovlp_pf.shape[0]
            X_vpq = np.empty((3, ni, ni))
            for v in range(3):
                X_vpq[v] = np.linalg.solve(ovlp_pf.T, B_vfq[v])
            return X_vpq
