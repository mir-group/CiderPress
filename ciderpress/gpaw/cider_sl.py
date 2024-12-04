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
import yaml

from ciderpress.gpaw.cider_kernel import CiderGGAHybridKernel, CiderMGGAHybridKernel
from ciderpress.gpaw.interp_paw import DiffGGA, DiffMGGA


class SLCiderGGAHybridWrapper(CiderGGAHybridKernel):
    def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
        super(SLCiderGGAHybridWrapper, self).calculate(
            e_g,
            n_sg,
            v_sg,
            sigma_xg,
            dedsigma_xg,
            np.zeros(
                (
                    n_sg.shape[0],
                    3,
                )
                + e_g.shape,
                dtype=e_g.dtype,
            ),
        )


class SLCiderMGGAHybridWrapper(CiderMGGAHybridKernel):
    def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg, tau_sg, dedtau_sg):
        super(SLCiderMGGAHybridWrapper, self).calculate(
            e_g,
            n_sg,
            v_sg,
            sigma_xg,
            dedsigma_xg,
            tau_sg,
            dedtau_sg,
            np.zeros(
                (
                    n_sg.shape[0],
                    3,
                )
                + e_g.shape,
                dtype=e_g.dtype,
            ),
        )


class _SLCiderBase:
    def get_setup_name(self):
        return "PBE"

    def todict(self):
        kparams = self.kernel.todict()
        return {
            "kernel_params": kparams,
        }

    def get_mlfunc_data(self):
        return yaml.dump(self.kernel.mlfunc, Dumper=yaml.CDumper)


class SLCiderGGA(_SLCiderBase, DiffGGA):
    def todict(self):
        d = super(SLCiderGGA, self).todict()
        d["_cider_type"] = "SLCiderGGA"
        return d


class SLCiderMGGA(_SLCiderBase, DiffMGGA):
    def todict(self):
        d = super(SLCiderMGGA, self).todict()
        d["_cider_type"] = "SLCiderMGGA"
        return d
