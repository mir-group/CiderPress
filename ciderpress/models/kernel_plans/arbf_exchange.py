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

from ciderpress.dft.xc_evaluator import SplineSetEvaluator
from ciderpress.models.kernel_plans.kernel_tools import get_agpr_kernel
from ciderpress.models.kernel_plans.map_tools import get_mapped_gp_evaluator_additive


def get_kernel(
    natural_scale=None,
    natural_lscale=None,
    scale_factor=None,
    lscale_factor=None,
    **kwargs,
):
    # leave out density from feature vector
    slice(1, None, None)
    print(lscale_factor, natural_lscale)
    print(scale_factor, natural_scale)
    return get_agpr_kernel(
        slice(0, 1, None),
        slice(1, None, None),
        lscale_factor * natural_lscale,
        scale=[1e-5, 1e-5, scale_factor * natural_scale],
        order=2,
        nsingle=1,
    )


def mapping_plan(dft_kernel):
    scale, ind_sets, spline_grids, coeff_sets = get_mapped_gp_evaluator_additive(
        dft_kernel.kernel,
        dft_kernel.X1ctrl,
        dft_kernel.alpha,
        dft_kernel.feature_list,
    )
    # TODO this needs to get converted into a class
    # that evaluates a specific kernel's contribution to the
    # XC energy. This function should then be a component
    # called by DFTKernel.map().
    # Then these mapping plans need to be combined within some
    # overarching evaluator class, which should be created
    # by a function like MOLGP.map().
    return SplineSetEvaluator(scale, ind_sets, spline_grids, coeff_sets)
