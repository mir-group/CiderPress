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


def get_sigma(grad_svg):
    nspin = grad_svg.shape[0]
    shape = grad_svg.shape[2:]
    if nspin not in [1, 2]:
        raise ValueError("Only 1 or 2 spins supported")
    sigma_xg = np.empty((2 * nspin - 1,) + shape)
    sigma_xg[::2] = np.einsum("sv...,sv...->s...", grad_svg, grad_svg)
    if nspin == 2:
        sigma_xg[1] = np.einsum("v...,v...->...", grad_svg[0], grad_svg[1])
    return sigma_xg


def get_dsigmadf(grad_svg, dgraddf_ovg, spins):
    norb = len(spins)
    assert dgraddf_ovg.shape[0] == norb
    shape = grad_svg.shape[2:]
    dsigmadf_og = np.empty((norb,) + shape)
    for o, s in enumerate(spins):
        dsigmadf_og[o] = 2 * np.einsum("vg,vg->g", grad_svg[s], dgraddf_ovg[o])
    return dsigmadf_og
