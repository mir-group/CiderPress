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

import unittest

import numpy as np
from gpaw.mpi import MPIComm, world
from gpaw.pw.descriptor import PWDescriptor
from numpy.testing import assert_allclose

from ciderpress.gpaw.nldf_interface import GridDescriptor, LibCiderPW, wrap_fft_mpi


class TestLibCiderPW(unittest.TestCase):
    def test_pwutil(self):
        np.random.seed(98)
        comm = MPIComm(world)
        N_c = [100, 80, 110]
        cell_cv = np.diag(np.ones(3))
        ciderpw = LibCiderPW(N_c, cell_cv, comm)
        ciderpw.initialize_backend()
        gd = GridDescriptor(N_c, cell_cv, comm=MPIComm(world))
        pd = PWDescriptor(180, gd, dtype=float)
        wrapper = wrap_fft_mpi(pd)
        input = np.random.normal(size=N_c)[None, :, :, :]
        input = gd.distribute(input)
        output_ref = pd.fft(input)
        sum = np.abs(output_ref).sum()
        sum = pd.gd.comm.sum_scalar(sum)
        for i in range(20):
            output_test = wrapper.fft(input)
        input_test = wrapper.ifft(output_test)
        input_ref = pd.ifft(output_ref)
        assert_allclose(output_test[0], output_ref)
        assert_allclose(input_test[0], input_ref)


if __name__ == "__main__":
    unittest.main()
