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

LDA_FACTOR = -3.0 / 4.0 * (3.0 / np.pi) ** (1.0 / 3)
CFC = (3.0 / 10) * (3 * np.pi**2) ** (2.0 / 3)


def dtauw(rho, sigma):
    return -sigma / (8 * rho**2 + 1e-16), 1 / (8 * rho + 1e-16)


def get_uniform_tau(rho):
    return (3.0 / 10) * (3 * np.pi**2) ** (2.0 / 3) * rho ** (5.0 / 3)


def get_single_orbital_tau(rho, mag_grad):
    return mag_grad**2 / (8 * rho + 1e-16)


def get_s2(rho, sigma):
    b = 2 * (3 * np.pi * np.pi) ** (1.0 / 3)
    s = np.sqrt(sigma) / (b * rho ** (4.0 / 3) + 1e-16)
    return s * s


def ds2(rho, sigma):
    # s = |nabla n| / (b * n)
    b = 2 * (3 * np.pi * np.pi) ** (1.0 / 3)
    s = np.sqrt(sigma) / (b * rho ** (4.0 / 3) + 1e-16)
    s2 = s**2
    return -8.0 * s2 / (3 * rho + 1e-16), 1 / (b * rho ** (4.0 / 3) + 1e-16) ** 2


ALPHA_TOL = 1e-10


def get_alpha(rho, sigma, tau):
    rho = rho + ALPHA_TOL
    tau0 = get_uniform_tau(rho)
    tauw = get_single_orbital_tau(rho, np.sqrt(sigma))
    # TODO this numerical stability trick is a bit of a hack.
    # Should make spline support small negative alpha
    # instead, for the sake of clean code and better stability.
    return np.maximum((tau - tauw), 0) / tau0


def dalpha(rho, sigma, tau):
    rho = rho + ALPHA_TOL
    tau0 = get_uniform_tau(rho)
    tauw = sigma / (8 * rho)
    dwdn, dwds = -sigma / (8 * rho * rho), 1 / (8 * rho)
    dadn = 5.0 * (tauw - tau) / (3 * tau0 * rho) - dwdn / tau0
    dadsigma = -dwds / tau0
    dadtau = 1 / tau0
    cond = (tau - tauw) / tau0 < -0.1
    dadn[cond] = 0
    dadsigma[cond] = 0
    dadtau[cond] = 0
    return dadn, dadsigma, dadtau


def get_exponent_b(rho_data, a0=1.0, fac_mul=0.03125, amin=0.0625, nspin=1):
    fac = fac_mul * 1.2 * (6 * np.pi**2) ** (2.0 / 3) / np.pi
    rho = rho_data[0] + 1e-16
    tau = rho_data[5]
    if nspin == 1:
        B = np.pi / 2 ** (2.0 / 3) * (a0 - fac)
    else:
        B = np.pi * (a0 - fac)
    C = np.pi / 2 ** (2.0 / 3) * fac / CFC
    ascale = B * rho ** (2.0 / 3) + C * tau / rho
    # return ascale,\
    #       2*B/(3*rho**(1./3)) - (C*tau/rho)/rho,\
    #       np.zeros_like(ascale),\
    #       C/rho
    dadrho = 2 * B / (3 * rho ** (1.0 / 3)) - (C * tau / rho) / rho
    dadtau = C / rho
    cond = ascale < amin
    ascale[cond] = amin * np.exp(ascale[cond] / amin - 1)
    dadrho[cond] *= ascale[cond] * np.exp(ascale[cond] / amin - 1)
    dadtau[cond] *= ascale[cond] * np.exp(ascale[cond] / amin - 1)
    return ascale, dadrho, np.zeros_like(ascale), dadtau


def get_exponent_d(rho_data, a0=1.0, fac_mul=0.03125, amin=0.0625, nspin=1):
    rho = rho_data[0]
    sigma = np.einsum("ij,ij->j", rho_data[1:4], rho_data[1:4])
    CFC = 0.3 * (3 * np.pi**2) ** (2.0 / 3)
    fac = fac_mul * 1.2 * (6 * np.pi**2) ** (2.0 / 3) / np.pi
    if nspin == 1:
        B = np.pi / 2 ** (2.0 / 3) * a0
    else:
        B = np.pi * a0
    C = np.pi / 2 ** (2.0 / 3) * fac / CFC
    tau = sigma / (8 * rho + 1e-16)
    ascale = B * rho ** (2.0 / 3) + C * tau / (rho + 1e-16)
    dadrho = 2 * B / (3 * rho ** (1.0 / 3) + 1e-16) - 2 * (
        C * tau / (rho * rho + 1e-16)
    )
    dadsigma = C / (8 * rho * rho + 1e-16)
    cond = ascale < amin
    ascale[cond] = amin * np.exp(ascale[cond] / amin - 1)
    dadrho[cond] *= ascale[cond] * np.exp(ascale[cond] / amin - 1)
    dadsigma[cond] *= ascale[cond] * np.exp(ascale[cond] / amin - 1)
    return ascale, dadrho, dadsigma


def v_basis_transform(rho, sigma, tau, v_npalpha):
    v_nst = np.zeros(v_npalpha.shape)
    # dE/dn lines 1-3
    v_nst[0] = v_npalpha[0]
    dpdn, dpdsigma = ds2(rho, sigma)
    # dE/dn line 4 term 1
    v_nst[0] += v_npalpha[1] * dpdn
    # dE/dsigma term 1
    v_nst[1] += v_npalpha[1] * dpdsigma
    dadn, dadsigma, dadtau = dalpha(rho, sigma, tau)
    # dE/dn line 4 term 2
    v_nst[0] += v_npalpha[3] * dadn
    # dE/dsigma term 2
    v_nst[1] += v_npalpha[3] * dadsigma
    # dE/dtau
    v_nst[3] = v_npalpha[3] * dadtau
    return v_nst


def v_basis_transform_f(rho, sigma, tau, taux, v_npax):
    v_nst = np.zeros(v_npax.shape)
    # dE/dn lines 1-3
    v_nst[0] = v_npax[0]
    dpdn, dpdsigma = ds2(rho, sigma)
    # dE/dn line 4 term 1
    v_nst[0] += v_npax[1] * dpdn
    # dE/dsigma term 1
    v_nst[1] += v_npax[1] * dpdsigma
    dadn, dadsigma, dadtau = dalpha(rho, sigma, tau)
    # dE/dn line 4 term 2
    v_nst[0] += v_npax[3] * dadn
    # dE/dsigma term 2
    v_nst[1] += v_npax[3] * dadsigma
    # dE/dtau
    v_nst[3] = v_npax[3] * dadtau
    # dE/dtau from tau/taux
    v_nst[3] += v_npax[4] / (taux + 1e-16)
    # dE/taux from tau/taux
    v_nst[4] -= v_npax[4] * tau / (taux + 1e-16) ** 2
    return v_nst


def v_basis_transform_d(rho, sigma, v_np):
    v_nst = np.zeros(v_np.shape)
    # dE/dn lines 1-3
    v_nst[0] = v_np[0]
    dpdn, dpdsigma = ds2(rho, sigma)
    # dE/dn line 4 term 1
    v_nst[0] += v_np[1] * dpdn
    # dE/dsigma term 1
    v_nst[1] += v_np[1] * dpdsigma
    return v_nst


def functional_derivative_loop_b(mlfunc, dEddesc, rho, sigma, tau):
    N = dEddesc.shape[0]
    v_npa = np.zeros((4, N))
    dedf = np.zeros((3, N))

    for i, d in enumerate(mlfunc.desc_order[:6]):  # TODO fix indexing
        if d == 0:
            v_npa[0] += dEddesc[:, i]
        elif d == 1:
            v_npa[1] += dEddesc[:, i]
        elif d == 2:
            v_npa[3] += dEddesc[:, i]
        else:
            if d == 3:
                dedf[0] += dEddesc[:, i]
            elif d == 4:
                dedf[1] += dEddesc[:, i]
            elif d == 5:
                dedf[2] += dEddesc[:, i]
            elif d == 6:
                raise ValueError("d={} not supported".format(d))
                # tmp = 2 * dEddesc[:,i] * a * feat[3:6]
                # dedf[3:6] += tmp
                # deda += dEddesc[:,i] * np.einsum('ij,ij->j', feat[3:6], feat[3:6])
            elif d == 7:
                raise ValueError("d={} not supported".format(d))
                # v_aniso += dEddesc[:,i] * feat[3:6]
                # dedf[3:6] += dEddesc[:,i] * gri # gradrho * rho^-1
            else:
                raise ValueError("d={} not recognized".format(d))

    v_nst = v_basis_transform(rho, sigma, tau, v_npa)
    return v_nst, dedf


def functional_derivative_loop_h(mlfunc, dEddesc, rho, sigma, tau):
    N = dEddesc.shape[0]
    v_npa = np.zeros((4, N))
    dedf = np.zeros((3, N))

    for i, d in enumerate(mlfunc.desc_order[:6]):  # TODO fix indexing
        if d == 0:
            v_npa[0] += dEddesc[:, i]
        elif d == 1:
            v_npa[1] += dEddesc[:, i]
        elif d == 2:
            v_npa[3] += dEddesc[:, i]
        else:
            dedf[d - 3] += dEddesc[:, i]

    v_nst = v_basis_transform(rho, sigma, tau, v_npa)
    return v_nst, dedf


def functional_derivative_loop_f(mlfunc, dEddesc, rho, sigma, tau, taux):
    N = dEddesc.shape[0]
    v_npax = np.zeros((5, N))
    dedf = np.zeros((3, N))

    for i, d in enumerate(mlfunc.desc_order[:7]):  # TODO fix indexing
        if d == 0:
            v_npax[0] += dEddesc[:, i]
        elif d == 1:
            v_npax[1] += dEddesc[:, i]
        elif d == 2:
            v_npax[3] += dEddesc[:, i]
        elif d == 6:
            v_npax[4] += dEddesc[:, i]
        else:
            if d == 3:
                dedf[0] += dEddesc[:, i]
            elif d == 4:
                dedf[1] += dEddesc[:, i]
            elif d == 5:
                dedf[2] += dEddesc[:, i]
            else:
                raise ValueError("d={} not recognized".format(d))

    v_nst = v_basis_transform_f(rho, sigma, tau, taux, v_npax)
    return v_nst, dedf


def functional_derivative_loop_d(mlfunc, dEddesc, rho, sigma):
    N = dEddesc.shape[0]
    v_np = np.zeros((2, N))
    dedf = np.zeros((3, N))

    for i, d in enumerate(mlfunc.desc_order[:5]):  # TODO fix indexing
        if d == 0:
            v_np[0] += dEddesc[:, i]
        elif d == 1:
            v_np[1] += dEddesc[:, i]
        elif d == 2:
            dedf[0] += dEddesc[:, i]
        elif d == 3:
            dedf[1] += dEddesc[:, i]
        elif d == 4:
            dedf[2] += dEddesc[:, i]
        else:
            raise ValueError("d={} not recognized".format(d))

    v_ns = v_basis_transform_d(rho, sigma, v_np)
    return v_ns, dedf


def contract_descriptors_l0(rho, sigma, tau, feat):
    assert feat.shape[0] == 3
    sprefac = 2 * (3 * np.pi * np.pi) ** (1.0 / 3)
    shape = list(feat.shape)
    shape[0] += 3
    cfeat = np.zeros(tuple(shape))
    # alpha = np.maximum(tau - sigma/(8*rho), 0)
    # alpha /= (3.0/10) * (3*np.pi**2)**(2.0/3) * rho**(5./3) + 1e-16
    alpha = get_alpha(rho, sigma, tau)
    s2 = sigma / ((sprefac * rho ** (4.0 / 3)) ** 2 + 1e-16)
    cfeat[0] = rho
    cfeat[1] = s2
    cfeat[2] = alpha
    cfeat[3] = feat[0]
    cfeat[4] = feat[1]
    cfeat[5] = feat[2]
    return cfeat


def contract_descriptors_h(rho, sigma, tau, feat):
    sprefac = 2 * (3 * np.pi * np.pi) ** (1.0 / 3)
    shape = list(feat.shape)
    shape[0] += 3
    cfeat = np.zeros(tuple(shape))
    # alpha = np.maximum(tau - sigma/(8*rho), 0)
    # alpha /= (3.0/10) * (3*np.pi**2)**(2.0/3) * rho**(5./3) + 1e-16
    alpha = get_alpha(rho, sigma, tau)
    s2 = sigma / ((sprefac * rho ** (4.0 / 3)) ** 2 + 1e-16)
    cfeat[0] = rho
    cfeat[1] = s2
    cfeat[2] = alpha
    cfeat[3:] = feat
    return cfeat


def contract_descriptors_l0_f(rho, sigma, tau, feat, taux):
    assert feat.shape[0] == 3
    sprefac = 2 * (3 * np.pi * np.pi) ** (1.0 / 3)
    shape = list(feat.shape)
    shape[0] += 4
    cfeat = np.zeros(tuple(shape))
    alpha = tau - sigma / (8 * rho)
    alpha /= (3.0 / 10) * (3 * np.pi**2) ** (2.0 / 3) * rho ** (5.0 / 3) + 1e-16
    s2 = sigma / ((sprefac * rho ** (4.0 / 3)) ** 2 + 1e-16)
    cfeat[0] = rho
    cfeat[1] = s2
    cfeat[2] = alpha
    cfeat[3] = feat[0]
    cfeat[4] = feat[1]
    cfeat[5] = feat[2]
    cfeat[6] = tau / (taux + 1e-16)
    return cfeat


def contract_descriptors_l0_d(rho, sigma, feat):
    assert feat.shape[0] == 3
    sprefac = 2 * (3 * np.pi * np.pi) ** (1.0 / 3)
    shape = list(feat.shape)
    shape[0] = 5
    cfeat = np.zeros(tuple(shape))
    s2 = sigma / ((sprefac * rho ** (4.0 / 3)) ** 2 + 1e-16)
    cfeat[0] = rho
    cfeat[1] = s2
    cfeat[2] = feat[0]
    cfeat[3] = feat[1]
    cfeat[4] = feat[2]
    return cfeat


def call_xc_kernel_gga(
    self, e_g, nt_sg, sigma_xg, feat_sg, v_sg, dedsigma_xg, vfeat_sg, RHOCUT=1e-8
):
    LDA_FACTOR = -3.0 / 4.0 * (3.0 / np.pi) ** (1.0 / 3)
    xfac = self.xmix * LDA_FACTOR
    nspin = nt_sg.shape[0]
    contracted_desc = {}
    # TODO SHOULD CHANGE BASELINE a0, etc. to be from the q exponent, because
    # that one is the same for every features and used to reference the coords in the solid-state version.
    F = {}
    dF = {}
    gshape = nt_sg[0].shape
    Nfeat = feat_sg[0].shape[0]
    for s in range(nspin):
        rho = nspin * nt_sg[s].reshape(-1)
        sigma = nspin * nspin * sigma_xg[2 * s].reshape(-1)
        feat = nspin * feat_sg[s].reshape(feat_sg[s].shape[0], -1)
        contracted_desc[s] = contract_descriptors_l0_d(rho, sigma, feat)[
            self.mlfunc.desc_order[:5]
        ]
        F[s], dF[s] = self.mlfunc.get_F_and_derivative(contracted_desc[s])

        cond = rho > RHOCUT
        gcond = cond.reshape(gshape)
        e_g[gcond] += (
            xfac * np.abs(rho) ** (4.0 / 3) * np.sign(rho) * F[s] / nspin
        ).reshape(*gshape)[gcond]
        dEddesc = (xfac * np.abs(rho) ** (4.0 / 3) * np.sign(rho)).reshape(-1, 1) * dF[
            s
        ]
        v_nst, vfeat = functional_derivative_loop_d(self.mlfunc, dEddesc, rho, sigma)

        v_sg[s, gcond] += (xfac * 4.0 / 3 * np.abs(rho) ** (1.0 / 3) * F[s]).reshape(
            *gshape
        )[gcond]
        vfeat_sg[s][:, gcond] += vfeat.reshape(Nfeat, *gshape)[:, gcond]
        v_sg[s, gcond] += v_nst[0].reshape(*gshape)[gcond]
        dedsigma_xg[2 * s, gcond] += nspin * v_nst[1].reshape(*gshape)[gcond]
        rho = sigma = feat = None


def call_xc_kernel_mgga(
    self,
    e_g,
    nt_sg,
    sigma_xg,
    tau_sg,
    feat_sg,
    v_sg,
    dedsigma_xg,
    dedtau_sg,
    vfeat_sg,
    RHOCUT=1e-8,
):
    LDA_FACTOR = -3.0 / 4.0 * (3.0 / np.pi) ** (1.0 / 3)
    xfac = self.xmix * LDA_FACTOR
    nspin = nt_sg.shape[0]
    contracted_desc = {}
    # TODO SHOULD CHANGE BASELINE a0, etc. to be from the q exponent, because
    # that one is the same for every features and used to reference the coords in the solid-state version.
    F = {}
    dF = {}
    gshape = nt_sg[0].shape
    Nfeat = feat_sg[0].shape[0]
    for s in range(nspin):
        rho = nspin * nt_sg[s].reshape(-1)
        sigma = nspin * nspin * sigma_xg[2 * s].reshape(-1)
        tau = nspin * tau_sg[s].reshape(-1)
        feat = nspin * feat_sg[s].reshape(feat_sg[s].shape[0], -1)
        contracted_desc[s] = contract_descriptors_h(rho, sigma, tau, feat)
        F[s], dF[s] = self.mlfunc.get_F_and_derivative(contracted_desc[s])

        cond = rho > RHOCUT
        gcond = cond.reshape(gshape)
        e_g[gcond] += (
            xfac * np.abs(rho) ** (4.0 / 3) * np.sign(rho) * F[s] / nspin
        ).reshape(*gshape)[gcond]
        dEddesc = (xfac * np.abs(rho) ** (4.0 / 3) * np.sign(rho)).reshape(-1, 1) * dF[
            s
        ]
        v_nst, vfeat = functional_derivative_loop_h(
            self.mlfunc, dEddesc, rho, sigma, tau
        )

        v_sg[s, gcond] += (xfac * 4.0 / 3 * np.abs(rho) ** (1.0 / 3) * F[s]).reshape(
            *gshape
        )[gcond]
        vfeat_sg[s][:, gcond] += vfeat.reshape(Nfeat, *gshape)[:, gcond]
        v_sg[s, gcond] += v_nst[0].reshape(*gshape)[gcond]
        dedsigma_xg[2 * s, gcond] += nspin * v_nst[1].reshape(*gshape)[gcond]
        dedtau_sg[s, gcond] += v_nst[3].reshape(*gshape)[gcond]
        rho = sigma = feat = None


def call_xc_kernel_mmgga(
    self,
    e_g,
    nt_sg,
    sigma_xg,
    tau_sg,
    feat_sg,
    taux_sg,
    v_sg,
    dedsigma_xg,
    dedtau_sg,
    vfeat_sg,
    dedtaux_sg,
    RHOCUT=1e-8,
):
    LDA_FACTOR = -3.0 / 4.0 * (3.0 / np.pi) ** (1.0 / 3)
    xfac = self.xmix * LDA_FACTOR
    nspin = nt_sg.shape[0]
    contracted_desc = {}
    # TODO SHOULD CHANGE BASELINE a0, etc. to be from the q exponent, because
    # that one is the same for every features and used to reference the coords in the solid-state version.
    F = {}
    dF = {}
    gshape = nt_sg[0].shape
    Nfeat = feat_sg[0].shape[0]
    for s in range(nspin):
        rho = nspin * nt_sg[s].reshape(-1)
        sigma = nspin * nspin * sigma_xg[2 * s].reshape(-1)
        tau = nspin * tau_sg[s].reshape(-1)
        feat = nspin * feat_sg[s].reshape(feat_sg[s].shape[0], -1)
        taux = nspin * taux_sg[s].reshape(-1)
        contracted_desc[s] = contract_descriptors_l0_f(rho, sigma, tau, feat, taux)[
            self.mlfunc.desc_order[:7]
        ]
        F[s], dF[s] = self.mlfunc.get_F_and_derivative(contracted_desc[s])

        cond = rho > RHOCUT
        gcond = cond.reshape(gshape)
        e_g[gcond] += (
            xfac * np.abs(rho) ** (4.0 / 3) * np.sign(rho) * F[s] / nspin
        ).reshape(*gshape)[gcond]
        dEddesc = (xfac * np.abs(rho) ** (4.0 / 3) * np.sign(rho)).reshape(-1, 1) * dF[
            s
        ]
        v_nst, vfeat = functional_derivative_loop_f(
            self.mlfunc, dEddesc, rho, sigma, tau, taux
        )

        v_sg[s, gcond] += (xfac * 4.0 / 3 * np.abs(rho) ** (1.0 / 3) * F[s]).reshape(
            *gshape
        )[gcond]
        vfeat_sg[s][:, gcond] += vfeat.reshape(Nfeat, *gshape)[:, gcond]
        v_sg[s, gcond] += v_nst[0].reshape(*gshape)[gcond]
        dedsigma_xg[2 * s, gcond] += nspin * v_nst[1].reshape(*gshape)[gcond]
        dedtau_sg[s, gcond] += v_nst[3].reshape(*gshape)[gcond]
        dedtaux_sg[s, gcond] += v_nst[4].reshape(*gshape)[gcond]
        rho = sigma = feat = None


def call_xc_kernel_debug_gga(
    self, e_g, nt_sg, sigma_xg, feat_sg, v_sg, dedsigma_xg, vfeat_sg
):
    LDA_FACTOR = -3.0 / 4.0 * (3.0 / np.pi) ** (1.0 / 3)
    xfac = self.xmix * LDA_FACTOR
    nspin = nt_sg.shape[0]
    contracted_desc = {}
    # TODO SHOULD CHANGE BASELINE a0, etc. to be from the q exponent, because
    # that one is the same for every features and used to reference the coords in the solid-state version.
    F = {}
    dF = {}
    gshape = nt_sg[0].shape
    feat_sg[0].shape[0]
    for s in range(nspin):
        rho = nspin * nt_sg[s].reshape(-1)
        sigma = nspin * nspin * sigma_xg[2 * s].reshape(-1)
        feat = nspin * feat_sg[s].reshape(feat_sg[s].shape[0], -1)
        contracted_desc[s] = contract_descriptors_l0_d(rho, sigma, feat)[
            self.mlfunc.desc_order[:5]
        ]
        F[s], dF[s] = self.mlfunc.get_F_and_derivative(contracted_desc[s])

        cond = rho > 1e-7
        gcond = cond.reshape(gshape)
        e_g[gcond] += (
            xfac * np.abs(rho) ** (4.0 / 3) * np.sign(rho) * F[s] / nspin
        ).reshape(*gshape)[gcond]
        dEddesc = (xfac * np.abs(rho) ** (4.0 / 3) * np.sign(rho)).reshape(-1, 1) * dF[
            s
        ]
        v_nst, vfeat = functional_derivative_loop_d(self.mlfunc, dEddesc, rho, sigma)

        v_sg[s, gcond] += (xfac * 4.0 / 3 * np.abs(rho) ** (1.0 / 3) * F[s]).reshape(
            *gshape
        )[gcond]
        # TODO removing vfeat to test PAW energy
        # vfeat_sg[s][:,gcond] += vfeat.reshape(Nfeat,*gshape)[:,gcond]
        v_sg[s, gcond] += v_nst[0].reshape(*gshape)[gcond]
        dedsigma_xg[2 * s, gcond] += nspin * v_nst[1].reshape(*gshape)[gcond]
        rho = sigma = feat = None
    contracted_desc = F = dF = None
