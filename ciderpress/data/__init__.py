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

import os

import numpy as np

DATA_PATH = os.path.dirname(os.path.abspath(__file__))


def get_hgbs_min_exps(Z):
    dpath = os.path.join(DATA_PATH, "expnt_mins.npy")
    return np.load(dpath)[int(Z)]


def get_hgbs_max_exps(Z):
    dpath = os.path.join(DATA_PATH, "expnt_maxs.npy")
    return np.load(dpath)[int(Z)]
