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

from numpy.testing import assert_almost_equal
from pyscf import dft, gto

from ciderpress.pyscf.dft import make_cider_calc

"""
This is a high-level test to make sure RKS SCF energies are consistent between
commits. It also checks that the functionals are not accidentally changed.
"""

XCVALS = {
    "CIDER23X_SL_GGA": -199.442504774708,
    "CIDER23X_NL_GGA": -199.410917905451,
    "CIDER23X_SL_MGGA": -199.41699396454,
    "CIDER23X_NL_MGGA": -199.422085280523,
    "CIDER23X_NL_MGGA_PBE": -199.421990307797,
    "CIDER23X_NL_MGGA_DTR": -199.423911593205,
    "CIDER24Xne": -199.4190650929805,
    "CIDER24Xe": -199.4241297851031,
}


def run_functional(xcname):
    mol = gto.M(
        atom="F 0 0 0; F 0 0 1.420608",
        basis="def2-qzvppd",
        verbose=4,
        output="/dev/null",
    )
    ks = dft.RKS(mol)
    ks.xc = "PBE"
    ks.grids.level = 3

    xcfile = "functionals/{}.yaml".format(xcname)
    ks = make_cider_calc(
        ks, xcfile, xmix=0.25, xkernel="GGA_X_PBE", ckernel="GGA_C_PBE"
    )
    ks = ks.density_fit(auxbasis="def2-universal-jfit")
    ks.small_rho_cutoff = 0.0
    etot = ks.kernel()
    mol.stdout.close()
    assert_almost_equal(etot, XCVALS[xcname], 5)


class _BaseTest:

    xcname = None

    def test_functional(self):
        run_functional(self.xcname)


class TestSLGGA(unittest.TestCase, _BaseTest):
    xcname = "CIDER23X_SL_GGA"


class TestNLGGA(unittest.TestCase, _BaseTest):
    xcname = "CIDER23X_NL_GGA"


class TestSLMGGA(unittest.TestCase, _BaseTest):
    xcname = "CIDER23X_SL_MGGA"


class TestNLMGGA(unittest.TestCase, _BaseTest):
    xcname = "CIDER23X_NL_MGGA"


class TestNLMGGAPBE(unittest.TestCase, _BaseTest):
    xcname = "CIDER23X_NL_MGGA_PBE"


class TestNLMGGADTR(unittest.TestCase, _BaseTest):
    xcname = "CIDER23X_NL_MGGA_DTR"


class Test24ne(unittest.TestCase, _BaseTest):
    xcname = "CIDER24Xne"


class Test24e(unittest.TestCase, _BaseTest):
    xcname = "CIDER24Xe"


if __name__ == "__main__":
    unittest.main()
