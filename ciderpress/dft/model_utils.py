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

import joblib
import yaml

from ciderpress.dft.xc_evaluator import MappedXC
from ciderpress.dft.xc_evaluator2 import MappedXC2


def load_cider_model(mlfunc, mlfunc_format):
    if isinstance(mlfunc, str):
        if mlfunc_format is None:
            if mlfunc.endswith(".yaml"):
                mlfunc_format = "yaml"
            elif mlfunc.endswith(".joblib"):
                mlfunc_format = "joblib"
            else:
                raise ValueError("Unsupported file format")
        if mlfunc_format == "yaml":
            with open(mlfunc, "r") as f:
                mlfunc = yaml.load(f, Loader=yaml.CLoader)
        elif mlfunc_format == "joblib":
            mlfunc = joblib.load(mlfunc)
        else:
            raise ValueError("Unsupported file format")
    if not isinstance(mlfunc, (MappedXC, MappedXC2)):
        raise ValueError("mlfunc must be MappedXC")
    return mlfunc


def get_slxc_settings(xc, xkernel, ckernel, xmix):
    if xc is None:
        # xc is another way to specify non-mixed part of kernel
        xc = ""
    if ckernel is not None:
        xc = ckernel + " + " + xc
    if xkernel is not None:
        xc = "{} * {} + {}".format(1 - xmix, xkernel, xc)
    if xc.endswith(" + "):
        xc = xc[:-3]
    return xc
