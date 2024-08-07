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

import ctypes
import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from pyscf import gto
from pyscf.dft.gen_grid import libdft

from ciderpress.pyscf.gen_cider_grid import CiderGrids


class TestAtomicGridsIndexer(unittest.TestCase):
    def test_a2y(self):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvp")
        grids = CiderGrids(mol)
        n = 194
        grids.atom_grid = {"H": (30, n), "F": (50, n)}
        grids.prune = None
        grids.build()
        grid = np.empty((n, 4))
        libdft.MakeAngularGrid(grid.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n))
        ang_wt = np.ascontiguousarray(grid[:, 3])
        indexer = grids.grids_indexer
        nalpha = 10
        theta_rlmq = indexer.empty_rlmq(nalpha)
        theta_new_rlmq = indexer.empty_rlmq(nalpha)
        theta_gq = indexer.empty_gq(nalpha)
        np.random.seed(89)
        theta_rlmq[:] = np.random.normal(size=theta_rlmq.shape)
        indexer.reduce_angc_ylm_(theta_rlmq, theta_gq, a2y=False)
        shape = theta_gq.shape
        theta_gq.shape = (-1, n, nalpha)
        theta_gq *= ang_wt[None, :, None] * 4 * np.pi
        theta_gq.shape = shape
        indexer.reduce_angc_ylm_(theta_new_rlmq, theta_gq, a2y=True)
        assert_almost_equal(theta_new_rlmq, theta_rlmq)

        theta_big_gq = np.zeros((theta_gq.shape[0], nalpha + 7))
        indexer.reduce_angc_ylm_(theta_rlmq, theta_gq, a2y=False)
        indexer.reduce_angc_ylm_(theta_rlmq, theta_big_gq, a2y=False, offset=3)
        assert (theta_big_gq[:, :3] == 0).all()
        assert (theta_big_gq[:, -4:] == 0).all()
        assert_almost_equal(theta_big_gq[:, 3:-4], theta_gq, 14)
        theta_big_gq[:, :3] = np.random.normal(size=theta_big_gq[:, :3].shape)
        theta_big_gq[:, -4:] = np.random.normal(size=theta_big_gq[:, -4:].shape)
        theta_new2_rlmq = np.empty_like(theta_new_rlmq)
        indexer.reduce_angc_ylm_(theta_new_rlmq, theta_gq, a2y=True, offset=0)
        indexer.reduce_angc_ylm_(theta_new2_rlmq, theta_big_gq, a2y=True, offset=3)
        # NOTE this test is fine on laptop with 14 digits but fails on Github CI
        assert_almost_equal(theta_new2_rlmq, theta_new_rlmq, 13)


if __name__ == "__main__":
    unittest.main()
