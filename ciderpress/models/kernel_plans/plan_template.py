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


def get_kernel(
    natural_scale=None,
    natural_lscale=None,
    scale_factor=None,
    lscale_factor=None,
):
    """
    Args:
        natural_scale (float, None): Natural scale for covariance,
            typically set based on variance of target over training data.
        natural_lscale (array, None): Natural length scale for features,
            typically set based on variance of features over the
            training data.
        scale_factor (float, None): Multiplicative factor to tune
            covariance scale
        lscale_factor (float or array, None): Multiplicative factor
            to tune length scale.

    Returns:
        An sklearn kernel
    """


def mapping_plan(model):
    """
    A function that maps a model based on the kernel above to 'fast'
    functions like splines and polynomial evaluations, then returns
    an Evaluator object based on the mapping.

    Args:
        model:

    Returns:

    """
