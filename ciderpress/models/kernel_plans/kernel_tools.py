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

from ciderpress.models.kernels import (
    DiffAntisymRBF,
    DiffConstantKernel,
    SubsetARBF,
    SubsetRBF,
)


def get_rbf_kernel(
    indexes, length_scale, scale=1.0, opt_hparams=False, min_lscale=None
):
    if min_lscale is None:
        min_lscale = 0.01
    length_scale_bounds = (min_lscale, 10) if opt_hparams else "fixed"
    scale_bounds = (1e-5, 1e3) if opt_hparams else "fixed"
    return DiffConstantKernel(scale, constant_value_bounds=scale_bounds) * SubsetRBF(
        indexes,
        length_scale=length_scale[indexes],
        length_scale_bounds=length_scale_bounds,
    )


def get_antisym_rbf_kernel(length_scale, scale=1.0, opt_hparams=False, min_lscale=None):
    if min_lscale is None:
        min_lscale = 0.01
    length_scale_bounds = (min_lscale, 10) if opt_hparams else "fixed"
    scale_bounds = (1e-5, 1e3) if opt_hparams else "fixed"
    length_scale = np.append(
        0.5 * (length_scale[0] + length_scale[1]), length_scale[2:]
    )
    return DiffConstantKernel(
        scale, constant_value_bounds=scale_bounds
    ) * DiffAntisymRBF(
        length_scale=length_scale,
        length_scale_bounds=length_scale_bounds,
    )


def get_agpr_kernel(
    sinds,
    ainds,
    length_scale,
    scale=None,
    order=2,
    nsingle=1,
    opt_hparams=False,
    min_lscale=None,
):
    print(sinds, ainds, length_scale[sinds], length_scale[ainds])
    if min_lscale is None:
        min_lscale = 0.01
    length_scale_bounds = (min_lscale, 10) if opt_hparams else "fixed"
    scale_bounds = (1e-5, 1e5) if opt_hparams else "fixed"
    if scale is None:
        scale = [1.0] * (order + 1)
    if nsingle == 0:
        singles = None
    else:
        singles = SubsetRBF(
            sinds,
            length_scale=length_scale[sinds],
            length_scale_bounds=length_scale_bounds,
        )
    cov_kernel = SubsetARBF(
        ainds,
        order=order,
        length_scale=length_scale[ainds],
        scale=scale,
        length_scale_bounds=length_scale_bounds,
        scale_bounds=scale_bounds,
    )
    if singles is None:
        return cov_kernel
    else:
        return singles * cov_kernel
