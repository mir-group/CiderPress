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
from gpaw.atom.radialgd import AERadialGridDescriptor
from gpaw.sphere.lebedev import R_nv, Y_nL, weight_n
from gpaw.xc.gga import GGA, radial_gga_vars
from gpaw.xc.mgga import MGGA
from gpaw.xc.pawcorrection import rnablaY_nLv
from numpy import pi, sqrt
from scipy.interpolate import interp1d

from ciderpress.gpaw.gpaw_grids import GCRadialGridDescriptor


def create_kinetic_diffpaw(xcc, ny, phi_jg, tau_ypg, _interpc, r_g_new):
    nj = len(phi_jg)
    dphidr_jg = np.zeros(np.shape(phi_jg))
    for j in range(nj):
        phi_g = phi_jg[j]
        xcc.rgd.derivative(phi_g, dphidr_jg[j])

    phi_jg = _interpc(phi_jg)
    dphidr_jg = _interpc(dphidr_jg)

    # second term
    for y in range(ny):
        i1 = 0
        p = 0
        Y_L = xcc.Y_nL[y]
        for j1, l1, L1 in xcc.jlL:
            for j2, l2, L2 in xcc.jlL[i1:]:
                c = Y_L[L1] * Y_L[L2]
                temp = c * dphidr_jg[j1] * dphidr_jg[j2]
                tau_ypg[y, p, :] += temp
                p += 1
            i1 += 1
    # first term
    for y in range(ny):
        i1 = 0
        p = 0
        rnablaY_Lv = xcc.rnablaY_nLv[y, : xcc.Lmax]
        Ax_L = rnablaY_Lv[:, 0]
        Ay_L = rnablaY_Lv[:, 1]
        Az_L = rnablaY_Lv[:, 2]
        for j1, l1, L1 in xcc.jlL:
            for j2, l2, L2 in xcc.jlL[i1:]:
                temp = Ax_L[L1] * Ax_L[L2] + Ay_L[L1] * Ay_L[L2] + Az_L[L1] * Az_L[L2]
                temp *= phi_jg[j1] * phi_jg[j2]
                # temp[1:] /= r_g_new[1:]**2
                # temp[0] = temp[1]
                temp /= r_g_new**2
                if r_g_new[0] == 0:
                    temp[0] = temp[1]
                tau_ypg[y, p, :] += temp
                p += 1
            i1 += 1
    tau_ypg *= 0.5


class DiffPAWXCCorrection:
    def __init__(
        self,
        rgd,
        nc_g,
        nct_g,
        nc_corehole_g,
        dc_g,
        dct_g,
        dc_corehole_g,
        B_pqL,
        n_qg,
        nt_qg,
        d_qg,
        dt_qg,
        e_xc0,
        Lmax,
        tau_npg=None,
        taut_npg=None,
        tauc_g=None,
        tauct_g=None,
        ke_order_ng=True,
    ):
        self.rgd = rgd
        self.nc_g = nc_g
        self.nct_g = nct_g
        self.dc_g = dc_g
        self.dct_g = dct_g
        self.nc_corehole_g = nc_corehole_g
        self.dc_corehole_g = dc_corehole_g
        self.B_pqL = B_pqL
        self.n_qg = n_qg
        self.nt_qg = nt_qg
        self.d_qg = d_qg
        self.dt_qg = dt_qg
        self.e_xc0 = e_xc0
        self.Lmax = Lmax
        self.tau_npg = tau_npg
        self.taut_npg = taut_npg
        self.tauc_g = tauc_g
        self.tauct_g = tauct_g
        self.Y_nL = Y_nL
        if self.tau_npg is not None:
            NP = self.tau_npg.shape[1]
            if ke_order_ng:
                # radial coordinate is last index
                self.tau_pg = np.ascontiguousarray(
                    self.tau_npg.transpose(1, 0, 2).reshape(NP, -1)
                )
                self.taut_pg = np.ascontiguousarray(
                    self.taut_npg.transpose(1, 0, 2).reshape(NP, -1)
                )
            else:
                # angular coordinate is last index
                self.tau_pg = np.ascontiguousarray(
                    self.tau_npg.transpose(1, 2, 0).reshape(NP, -1)
                )
                self.taut_pg = np.ascontiguousarray(
                    self.taut_npg.transpose(1, 2, 0).reshape(NP, -1)
                )

    @classmethod
    def from_setup(cls, setup, build_kinetic=False, ke_order_ng=True):

        if hasattr(setup, "old_xc_correction"):
            xcc = setup.old_xc_correction
        else:
            setup.old_xc_correction = setup.xc_correction
            xcc = setup.xc_correction

        tab = np.array((2, 10, 18, 36, 54, 86, 118))
        (setup.Z > tab).sum()
        rcut = xcc.rgd.r_g[-1]
        if setup.Z > 36 and hasattr(setup.rgd, "a") and hasattr(setup.rgd, "b"):
            rgd = AERadialGridDescriptor(setup.rgd.a, setup.rgd.b, setup.rgd.N)
            gcut = rgd.ceil(rcut)
            rgd = rgd.new(gcut)
        else:
            rgd = GCRadialGridDescriptor(int(setup.Z), 200)
            gcut = rgd.ceil(rcut)
            rgd = rgd.make_cut(gcut)

        def _interpc(func):
            return interp1d(
                np.arange(xcc.rgd.r_g.size),
                func,
                kind="cubic",
            )(xcc.rgd.r2g(rgd.r_g))

        core_dens = {}
        names = ["nc_g", "nct_g", "nc_corehole_g"]
        n_g_list = [xcc.nc_g, xcc.nct_g, xcc.nc_corehole_g]
        if True:
            names += ["tauc_g", "tauct_g"]
            n_g_list += [xcc.tauc_g, xcc.tauct_g]
        for name, n_g in zip(names, n_g_list):
            if n_g is None:
                continue
            core_dens[name] = n_g
            core_dens["d" + name] = xcc.rgd.derivative(n_g)

        d_qg = [xcc.rgd.derivative(n_g) for n_g in xcc.n_qg]
        dt_qg = [xcc.rgd.derivative(n_g) for n_g in xcc.nt_qg]

        if build_kinetic:
            nii = xcc.nii
            nn = len(xcc.rnablaY_nLv)
            ng = rgd.r_g.shape[0]
            tau_npg = np.zeros((nn, nii, ng))
            taut_npg = np.zeros((nn, nii, ng))
            create_kinetic_diffpaw(xcc, nn, xcc.phi_jg, tau_npg, _interpc, rgd.r_g)
            create_kinetic_diffpaw(xcc, nn, xcc.phit_jg, taut_npg, _interpc, rgd.r_g)
        else:
            tau_npg = None
            taut_npg = None

        return cls(
            rgd,
            _interpc(core_dens["nc_g"]),
            _interpc(core_dens["nct_g"]),
            _interpc(core_dens["nc_corehole_g"])
            if xcc.nc_corehole_g is not None
            else None,
            _interpc(core_dens["dnc_g"]),
            _interpc(core_dens["dnct_g"]),
            _interpc(core_dens["dnc_corehole_g"])
            if xcc.nc_corehole_g is not None
            else None,
            xcc.B_pqL,
            np.array([_interpc(n_g) for n_g in xcc.n_qg]),
            np.array([_interpc(nt_g) for nt_g in xcc.nt_qg]),
            np.array([_interpc(n_g) for n_g in d_qg]),
            np.array([_interpc(nt_g) for nt_g in dt_qg]),
            xcc.e_xc0,
            xcc.Lmax,
            tau_npg,
            taut_npg,
            _interpc(core_dens["tauc_g"]),
            _interpc(core_dens["tauct_g"]),
            ke_order_ng=ke_order_ng,
        )


def calculate_cider_paw_correction(
    expansion,
    setup,
    D_sp,
    dEdD_sp=None,
    addcoredensity=True,
    a=None,
    separate_ae_ps=False,
    has_cider_contribs=True,
):
    xcc = setup.xc_correction
    rgd = xcc.rgd
    nspins = len(D_sp)

    if has_cider_contribs and expansion.rcalc.mode != "semilocal":
        setup.cider_contribs.set_D_sp(D_sp, setup.xc_correction)

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

    res = expansion(rgd, D_sLq, xcc.n_qg, xcc.d_qg, nc0_sg, dc0_sg, ae=True)
    rest = expansion(rgd, D_sLq, xcc.nt_qg, xcc.dt_qg, nct0_sg, dct0_sg, ae=False)

    if expansion.rcalc.mode == "feature":
        if separate_ae_ps:
            return res, rest
        else:
            return res - rest
    elif expansion.rcalc.mode == "debug_nosph":
        return res, rest
    elif expansion.rcalc.mode in ["energy", "energy_v2"]:
        e, dEdD_sqL, x_sbgL = res
        et, dEtdD_sqL, xt_sbgL = rest

        dEdD_sp = np.inner(
            (dEdD_sqL - dEtdD_sqL).reshape((nspins, -1)),
            xcc.B_pqL.reshape((len(xcc.B_pqL), -1)),
        )
        dEdD_sp += setup.cider_contribs.get_dH_sp()

        if addcoredensity:
            ediff = e - et - xcc.e_xc0
        else:
            ediff = e - et
        ndiff = x_sbgL.shape[1] - xt_sbgL.shape[1]
        shape = (x_sbgL.shape[0], ndiff, x_sbgL.shape[2], x_sbgL.shape[3])
        if separate_ae_ps:
            return ediff, dEdD_sp, x_sbgL, np.append(xt_sbgL, np.zeros(shape), axis=1)
        else:
            xdiff = x_sbgL - np.append(xt_sbgL, np.zeros(shape), axis=1)
            return ediff, dEdD_sp, xdiff
    elif expansion.rcalc.mode == "potential":
        dEdD_sqL = res
        dEtdD_sqL = rest
        dEdD_sp = np.inner(
            (dEdD_sqL - dEtdD_sqL).reshape((nspins, -1)),
            xcc.B_pqL.reshape((len(xcc.B_pqL), -1)),
        )
        dEdD_sp += setup.cider_contribs.get_dH_sp()
        return dEdD_sp
    elif expansion.rcalc.mode == "semilocal":
        e, dEdD_sqL = res
        et, dEtdD_sqL = rest

        if dEdD_sp is not None:
            dEdD_sp += np.inner(
                (dEdD_sqL - dEtdD_sqL).reshape((nspins, -1)),
                xcc.B_pqL.reshape((len(xcc.B_pqL), -1)),
            )

        if addcoredensity:
            return e - et - xcc.e_xc0
        else:
            return e - et
    else:
        raise ValueError(
            "Unsupported mode for CIDER PAW correction: {}.".format(
                expansion.rcalc.mode
            )
        )


class CiderRadialExpansion:
    def __init__(self, rcalc, F_sbgr=None, y_sbLg=None, f_in_sph_harm=False):
        self.rcalc = rcalc
        if F_sbgr is not None:
            nspins, nb = F_sbgr.shape[:2]
            if f_in_sph_harm:
                F_sbLg = F_sbgr
                self.F_sbg = np.einsum("sbLg,nL->sbng", F_sbLg, Y_nL).reshape(
                    nspins, nb, -1
                )
            else:
                self.F_sbg = F_sbgr.transpose(0, 1, 3, 2).reshape(nspins, nb, -1)
            self.y_sbg = np.einsum("sbLg,nL->sbng", y_sbLg, Y_nL).reshape(
                nspins, nb, -1
            )
        else:
            self.F_sbg = None
            self.y_sbg = None
        assert self.rcalc.mode in [
            "feature",
            "energy",
            "energy_v2",
            "potential",
            "semilocal",
            "debug_nosph",
        ]

    def __call__(self, rgd, D_sLq, n_qg, dndr_qg, nc0_sg, dnc0dr_sg, ae=True):
        n_sLg = np.dot(D_sLq, n_qg)
        n_sLg[:, 0] += nc0_sg

        dndr_sLg = np.dot(D_sLq, dndr_qg)
        dndr_sLg[:, 0] += dnc0dr_sg

        nspins, Lmax, nq = D_sLq.shape
        dEdD_sqL = np.zeros((nspins, nq, Lmax))

        if self.rcalc.mode == "feature":

            x_sbg = self.rcalc(
                rgd, n_sLg, Y_nL[:, :Lmax], dndr_sLg, rnablaY_nLv[:, :Lmax], ae=ae
            )
            nb = x_sbg.shape[1]
            wx_sbng = x_sbg.reshape(nspins, nb, Y_nL.shape[0], -1)

            return (
                4
                * np.pi
                * np.einsum("sbng,nL->sbgL", wx_sbng, weight_n[:, None] * Y_nL)
            )
        elif self.rcalc.mode == "debug_nosph":
            if self.F_sbg is None:
                feat_xg = self.rcalc(
                    rgd, n_sLg, Y_nL[:, :Lmax], dndr_sLg, rnablaY_nLv[:, :Lmax], ae=ae
                )
            else:
                feat_xg = self.rcalc(
                    rgd,
                    n_sLg,
                    Y_nL[:, :Lmax],
                    dndr_sLg,
                    rnablaY_nLv[:, :Lmax],
                    self.F_sbg,
                    self.y_sbg,
                    ae=ae,
                )
            shape = feat_xg.shape[:-1]
            shape = shape + (Y_nL.shape[0], -1)
            feat_xng = feat_xg.reshape(*shape)
            return feat_xng
        elif self.rcalc.mode == "feat_grid":
            pass
        elif self.rcalc.mode == "semilocal":
            E = 0.0
            for n, Y_L in enumerate(Y_nL[:, :Lmax]):
                w = weight_n[n]
                rnablaY_Lv = rnablaY_nLv[n, :Lmax]
                e_g, dedn_sg, b_vsg, a_sg, dedsigma_xg = self.rcalc(
                    rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv
                )
                dEdD_sqL += np.dot(rgd.dv_g * dedn_sg, n_qg.T)[:, :, np.newaxis] * (
                    w * Y_L
                )
                dEdD_sqL += (
                    2
                    * np.dot(rgd.dv_g * dedsigma_xg[::nspins] * a_sg, dndr_qg.T)[
                        :, :, np.newaxis
                    ]
                    * (w * Y_L)
                )
                if nspins == 2:
                    dEdD_sqL += np.dot(
                        rgd.dv_g * dedsigma_xg[1] * a_sg[::-1], dndr_qg.T
                    )[:, :, np.newaxis] * (w * Y_L[:Lmax])
                dedsigma_xg *= rgd.dr_g
                B_vsg = dedsigma_xg[::2] * b_vsg
                if nspins == 2:
                    B_vsg += 0.5 * dedsigma_xg[1] * b_vsg[:, ::-1]
                B_vsq = np.dot(B_vsg, n_qg.T)
                dEdD_sqL += 8 * pi * w * np.inner(rnablaY_Lv, B_vsq.T).T
                E += w * rgd.integrate(e_g)

            return E, dEdD_sqL

        elif self.rcalc.mode == "energy_v2":
            e_g, dedn_sg, dedgrad_svg, v_sbg = self.rcalc(
                rgd,
                n_sLg,
                Y_nL[:, :Lmax],
                dndr_sLg,
                rnablaY_nLv[:, :Lmax],
                self.F_sbg,
                self.y_sbg,
                ae=ae,
            )
            nb = v_sbg.shape[1]
            nn = Y_nL.shape[0]

            dedn_sng = dedn_sg.reshape(nspins, nn, -1)
            dedgrad_svng = dedgrad_svg.reshape(nspins, 3, nn, -1)
            # b_vsng = b_vsg.reshape(3, nspins, nn, -1)
            # a_sng = a_sg.reshape(nspins, nn, -1)

            dedgrad_sng = np.einsum("svng,nv->sng", dedgrad_svng, R_nv)
            # va_sLg = np.einsum("sng,nL->sLg", va_sng, weight_n[:, None] * Y_nL[:, :Lmax])
            dEdD_snq = np.dot(rgd.dv_g * dedn_sng, n_qg.T)
            dEdD_snq += np.dot(rgd.dv_g * dedgrad_sng, dndr_qg.T)
            # dEdD_sqL = np.einsum("sLg,qg->sqL", va_sLg, dndr_qg)
            dEdD_sqL = np.einsum(
                "snq,nL->sqL", dEdD_snq, weight_n[:, None] * Y_nL[:, :Lmax]
            )

            B_svnq = np.dot(dedgrad_svng * rgd.dr_g * rgd.r_g, n_qg.T)
            dEdD_sqL += np.einsum(
                "nLv,svnq->sqL",
                (4 * np.pi) * weight_n[:, None, None] * rnablaY_nLv[:, :Lmax],
                B_svnq,
            )

            E = weight_n.dot(rgd.integrate(e_g.reshape(nn, -1)))
            v_sbgL = np.einsum(
                "sbng,nL->sbgL",
                v_sbg.reshape(nspins, nb, nn, -1),
                weight_n[:, None] * Y_nL,
            )
            return E, dEdD_sqL, 4 * np.pi * v_sbgL

        else:
            mode = self.rcalc.mode
            if mode == "energy":
                e_g, dedn_sg, b_vsg, a_sg, dedsigma_xg, v_sbg = self.rcalc(
                    rgd,
                    n_sLg,
                    Y_nL[:, :Lmax],
                    dndr_sLg,
                    rnablaY_nLv[:, :Lmax],
                    self.F_sbg,
                    self.y_sbg,
                    ae=ae,
                )
                nb = v_sbg.shape[1]
            else:
                dedn_sg, b_vsg, a_sg, dedsigma_xg = self.rcalc(
                    rgd,
                    n_sLg,
                    Y_nL[:, :Lmax],
                    dndr_sLg,
                    rnablaY_nLv[:, :Lmax],
                    self.F_sbg,
                    self.y_sbg,
                    ae=ae,
                )

            nn = Y_nL.shape[0]

            dedn_sng = dedn_sg.reshape(nspins, nn, -1)
            dedsigma_xng = dedsigma_xg.reshape(2 * nspins - 1, nn, -1)
            b_vsng = b_vsg.reshape(3, nspins, nn, -1)
            a_sng = a_sg.reshape(nspins, nn, -1)

            dEdD_snq = np.dot(rgd.dv_g * dedn_sng, n_qg.T)
            dEdD_snq += 2 * np.dot(rgd.dv_g * dedsigma_xng[::nspins] * a_sng, dndr_qg.T)
            if nspins == 2:
                dEdD_snq += np.dot(rgd.dv_g * dedsigma_xng[1] * a_sng[::-1], dndr_qg.T)

            dEdD_sqL = np.einsum(
                "snq,nL->sqL", dEdD_snq, weight_n[:, None] * Y_nL[:, :Lmax]
            )

            dedsigma_xng *= rgd.dr_g
            B_vsng = dedsigma_xng[::2] * b_vsng
            if nspins == 2:
                B_vsng += 0.5 * dedsigma_xng[1] * b_vsng[:, ::-1]
            B_vsnq = np.dot(B_vsng, n_qg.T)
            dEdD_sqL += (
                8
                * np.pi
                * np.einsum(
                    "nLv,vsnq->sqL",
                    weight_n[:, None, None] * rnablaY_nLv[:, :Lmax],
                    B_vsnq,
                )
            )

            if mode == "energy":
                E = weight_n.dot(rgd.integrate(e_g.reshape(nn, -1)))
                v_sbgL = np.einsum(
                    "sbng,nL->sbgL",
                    v_sbg.reshape(nspins, nb, nn, -1),
                    weight_n[:, None] * Y_nL,
                )
                return E, dEdD_sqL, 4 * np.pi * v_sbgL
            else:
                return dEdD_sqL


def vec_radial_gga_vars(rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv):
    nspins = len(n_sLg)

    n_sg = np.dot(Y_nL, n_sLg).transpose(1, 0, 2).reshape(nspins, -1)

    a_sg = np.dot(Y_nL, dndr_sLg).transpose(1, 0, 2)
    b_vsg = np.dot(rnablaY_nLv.transpose(0, 2, 1), n_sLg).transpose(1, 2, 0, 3)
    N = Y_nL.shape[0]
    nx = 2 * nspins - 1

    sigma_xg = rgd.empty((nx, N))
    sigma_xg[::2] = (b_vsg**2).sum(0)
    if nspins == 2:
        sigma_xg[1] = (b_vsg[:, 0] * b_vsg[:, 1]).sum(0)
    sigma_xg[:, :, 1:] /= rgd.r_g[1:] ** 2
    sigma_xg[:, :, 0] = sigma_xg[:, :, 1]
    sigma_xg[::2] += a_sg**2
    if nspins == 2:
        sigma_xg[1] += a_sg[0] * a_sg[1]

    sigma_xg = sigma_xg.reshape(nx, -1)
    a_sg = a_sg.reshape(nspins, -1)
    b_vsg = b_vsg.reshape(3, nspins, -1)

    e_g = rgd.empty(N).reshape(-1)
    dedn_sg = rgd.zeros((nspins, N)).reshape(nspins, -1)
    dedsigma_xg = rgd.zeros((nx, N)).reshape(nx, -1)
    return e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, a_sg, b_vsg


def vec_radial_gga_vars_v2(rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv):
    nspins = len(n_sLg)

    n_sg = np.dot(Y_nL, n_sLg).transpose(1, 0, 2).reshape(nspins, -1)

    b_vsng = np.dot(rnablaY_nLv.transpose(0, 2, 1), n_sLg).transpose(1, 2, 0, 3)
    b_vsng[..., 1:] /= rgd.r_g[1:]
    b_vsng[..., 0] = b_vsng[..., 1]

    a_sng = np.dot(Y_nL, dndr_sLg).transpose(1, 0, 2)
    b_vsng += R_nv.T[:, None, :, None] * a_sng[None, :, :]
    N = Y_nL.shape[0]

    e_g = rgd.empty(N).reshape(-1)
    e_g[:] = 0
    dedn_sg = rgd.zeros((nspins, N)).reshape(nspins, -1)
    dedgrad_svg = rgd.zeros((nspins, 3, N)).reshape(nspins, 3, -1)
    grad_svg = b_vsng.transpose(1, 0, 2, 3).reshape(nspins, 3, -1)
    return e_g, n_sg, dedn_sg, grad_svg, dedgrad_svg


class CiderRadialFeatureCalculator:
    def __init__(self, xc):
        self.xc = xc
        self.mode = "feature"

    def __call__(self, rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae=True):
        (e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, a_sg, b_vsg) = vec_radial_gga_vars(
            rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv
        )
        return self.xc.get_paw_atom_contribs(n_sg, sigma_xg, ae=ae)[0]


class DiffSLRadialCalculator:
    def __init__(self, kernel):
        self.kernel = kernel
        self.mode = "semilocal"

    def __call__(self, rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv):
        (e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, a_sg, b_vsg) = radial_gga_vars(
            rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv
        )
        self.kernel.calculate(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg)
        return e_g, dedn_sg, b_vsg, a_sg, dedsigma_xg


class CiderRadialEnergyCalculator:
    def __init__(self, xc, a=None, mode="energy"):
        self.xc = xc
        self.a = a
        self.mode = mode
        assert mode in ["energy", "potential"]

    def __call__(self, rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, F_sbg, y_sbg, ae=True):
        (e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, a_sg, b_vsg) = vec_radial_gga_vars(
            rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv
        )

        if self.mode == "energy":
            v_sag = self.xc.get_paw_atom_evxc(
                e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, y_sbg, F_sbg, self.a, ae
            )
            return e_g, dedn_sg, b_vsg, a_sg, dedsigma_xg, v_sag
        else:
            self.xc.get_paw_atom_cider_pot(
                n_sg, dedn_sg, sigma_xg, dedsigma_xg, y_sbg, F_sbg, ae
            )

            return dedn_sg, b_vsg, a_sg, dedsigma_xg


class CiderRadialEnergyCalculator2:
    def __init__(self, xc, a=None):
        self.mode = "energy_v2"
        self.xc = xc
        self.a = a

    def __call__(self, rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, F_sbg, y_sbg, ae=True):
        (e_g, n_sg, dedn_sg, grad_svg, dedgrad_svg) = vec_radial_gga_vars_v2(
            rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv
        )
        v_sag = self.xc.get_paw_atom_evxc_v2(
            e_g, n_sg, dedn_sg, grad_svg, dedgrad_svg, y_sbg, F_sbg, self.a, ae
        )
        return e_g, dedn_sg, dedgrad_svg, v_sag


class DiffGGA(GGA):
    def __init__(self, kernel, stencil=2):
        super(DiffGGA, self).__init__(kernel, stencil)

    def get_setup_name(self):
        return "PBE"

    def todict(self):
        return {
            "_cider_type": "DiffGGA",
            "type": self.kernel.type,
            "name": self.kernel.name,
        }

    def initialize(self, density, hamiltonian, wfs):
        super(DiffGGA, self).initialize(density, hamiltonian, wfs)
        for setup in density.setups:
            if setup.xc_correction is None:
                continue
            if not isinstance(setup.xc_correction, DiffPAWXCCorrection):
                setup.xc_correction = DiffPAWXCCorrection.from_setup(
                    setup, build_kinetic=True, ke_order_ng=False
                )

    def calculate_paw_correction(
        self, setup, D_sp, dEdD_sp=None, addcoredensity=True, a=None
    ):
        if setup.xc_correction is None:
            return 0.0
        rcalc = DiffSLRadialCalculator(self.kernel)
        expansion = CiderRadialExpansion(rcalc)
        return calculate_cider_paw_correction(
            expansion, setup, D_sp, dEdD_sp, addcoredensity, a
        )


class DiffMGGA(MGGA):
    def __init__(self, kernel, stencil=2):
        super(DiffMGGA, self).__init__(kernel, stencil)

    def todict(self):
        return {
            "_cider_type": "DiffMGGA",
            "type": self.kernel.type,
            "name": self.kernel.name,
        }

    def initialize(self, density, hamiltonian, wfs):
        super(DiffMGGA, self).initialize(density, hamiltonian, wfs)
        for setup in density.setups:
            if setup.xc_correction is None:
                raise NotImplementedError("SL MGGA not supported with NCPP")
            if not isinstance(setup.xc_correction, DiffPAWXCCorrection):
                setup.xc_correction = DiffPAWXCCorrection.from_setup(
                    setup, build_kinetic=True
                )

    def calculate_paw_correction(
        self, setup, D_sp, dEdD_sp=None, addcoredensity=True, a=None
    ):
        if hasattr(self, "D_sp"):
            raise ValueError
        self.D_sp = D_sp
        self.n = 0
        self.ae = True
        self.xcc = setup.xc_correction
        if self.xcc is None:
            return 0.0
        self.dEdD_sp = dEdD_sp
        self.tau_npg = self.xcc.tau_npg
        self.taut_npg = self.xcc.taut_npg

        rcalc = self.create_mgga_radial_calculator()
        expansion = CiderRadialExpansion(rcalc)
        # The damn thing uses too many 'self' variables to define a clean
        # integrator object.
        E = calculate_cider_paw_correction(
            expansion, setup, D_sp, dEdD_sp, addcoredensity, a
        )
        del self.D_sp, self.n, self.ae, self.xcc, self.dEdD_sp
        return E

    def create_mgga_radial_calculator(self):
        class MockKernel:
            def __init__(self, mgga):
                self.mgga = mgga

            def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
                self.mgga.mgga_radial(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)

        return DiffSLRadialCalculator(MockKernel(self))


class DiffMGGA2(DiffMGGA):
    def initialize(self, density, hamiltonian, wfs):
        super(DiffMGGA, self).initialize(density, hamiltonian, wfs)
        for setup in density.setups:
            if setup.xc_correction is None:
                raise NotImplementedError("SL MGGA not supported with NCPP")
            if not isinstance(setup.xc_correction, DiffPAWXCCorrection):
                setup.xc_correction = DiffPAWXCCorrection.from_setup(
                    setup, build_kinetic=True, ke_order_ng=False
                )
