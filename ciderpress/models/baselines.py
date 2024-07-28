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

from ciderpress.density import LDA_FACTOR


def nsp_rho_basline(X0T, e=None, dedx=None):
    nspin, nfeat, nsamp = X0T.shape
    if e is None:
        e = np.zeros(nsamp)
    if dedx is None:
        dedx = np.zeros_like(X0T)
    nspin = X0T.shape[0]
    e[:] += X0T[:, 0].mean(0)
    dedx[0, :] += 1.0 / nspin


def _lda_x_helper(X0T, e, dedx):
    rho = X0T[0]
    e[:] += LDA_FACTOR * rho ** (4.0 / 3)
    dedx[0] += 4.0 / 3 * LDA_FACTOR * rho ** (1.0 / 3)


def _vi_x_damp_helper(X0T, e, dedx):
    rho = X0T[0]
    damp = 2.0 / (1.0 + 0.5 * X0T[3]) ** 2
    ddamp = -1 * damp / (1.0 + 0.5 * X0T[3])
    e[:] += LDA_FACTOR * damp * rho ** (4.0 / 3)
    dedx[0] += 4.0 / 3 * LDA_FACTOR * damp * rho ** (1.0 / 3)
    dedx[3] += LDA_FACTOR * ddamp * rho ** (4.0 / 3)


def _pbe_x_helper(X0T, e, dedx):
    rho = X0T[0]
    s2 = X0T[1]
    kappa = 0.804
    mu = 0.2195149727645171
    mk = mu / kappa
    fac = 1.0 / (1 + mk * s2)
    fx = 1 + kappa - kappa * fac
    dfx = mu * fac * fac
    e[:] += LDA_FACTOR * rho ** (4.0 / 3) * fx
    dedx[0] += 4.0 / 3 * LDA_FACTOR * rho ** (1.0 / 3) * fx
    dedx[1] += LDA_FACTOR * rho ** (4.0 / 3) * dfx


def _chachiyo_x_helper(X0T, e, dedx):
    rho = X0T[0]
    s2 = X0T[1]
    c = 4 * np.pi / 9
    x = c * np.sqrt(s2)
    dx = c / (2 * np.sqrt(s2))
    chfx = (3 * x**2 + np.pi**2 * np.log(1 + x)) / (
        (np.pi**2 + 3 * x) * np.log(1 + x)
    )
    dchfx = (
        -3 * x**2 * (np.pi**2 + 3 * x)
        + 3 * x * (1 + x) * (2 * np.pi**2 + 3 * x) * np.log(1 + x)
        - 3 * np.pi**2 * (1 + x) * np.log(1 + x) ** 2
    ) / ((1 + x) * (np.pi**2 + 3 * x) ** 2 * np.log(1 + x) ** 2)
    dchfx *= dx
    chfx[s2 < 1e-8] = 1 + 8 * s2[s2 < 1e-8] / 27
    dchfx[s2 < 1e-8] = 8.0 / 27
    e[:] += LDA_FACTOR * rho ** (4.0 / 3) * chfx
    dedx[0] += 4.0 / 3 * LDA_FACTOR * rho ** (1.0 / 3) * chfx
    dedx[1] += LDA_FACTOR * rho ** (4.0 / 3) * dchfx


def _sl_x_helper(X0T, _helper):
    nspin, nfeat, nsamp = X0T.shape
    e = np.zeros(nsamp)
    dedx = np.zeros_like(X0T)
    nspin = X0T.shape[0]
    for s in range(nspin):
        _helper(X0T[s], e, dedx[s])
    e[:] /= nspin
    dedx[:] /= nspin
    return e, dedx


def zero_xc(X0T):
    nspin, nfeat, nsamp = X0T.shape
    e = np.zeros(nsamp)
    dedx = np.zeros_like(X0T)
    return e, dedx


def one_xc(X0T):
    nspin, nfeat, nsamp = X0T.shape
    e = np.ones(nsamp)
    dedx = np.zeros_like(X0T)
    return e, dedx


def lda_x(X0T):
    return _sl_x_helper(X0T, _lda_x_helper)


def nlda_x_damp(X0T):
    return _sl_x_helper(X0T, _vi_x_damp_helper)


def gga_x_chachiyo(X0T):
    return _sl_x_helper(X0T, _chachiyo_x_helper)


def gga_x_pbe(X0T):
    return _sl_x_helper(X0T, _pbe_x_helper)


BASELINE_CODES = {
    "RHO": nsp_rho_basline,
    "ZERO": zero_xc,
    "ONE": one_xc,
    "LDA_X": lda_x,
    "NLDA_X_DAMP": nlda_x_damp,
    "GGA_X_PBE": gga_x_pbe,
    "GGA_X_CHACHIYO": gga_x_chachiyo,
}
