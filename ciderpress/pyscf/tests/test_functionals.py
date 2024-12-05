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

import pyscf.dft.radi
from numpy.testing import assert_almost_equal
from pyscf import dft, gto

from ciderpress.pyscf.dft import make_cider_calc

"""
This is a high-level test to make sure RKS SCF energies are consistent between
commits. It also checks that the functionals are not accidentally changed.
"""

# This is for backward consistency
pyscf.dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False


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
    etot = ks.kernel()
    mol.stdout.close()
    assert_almost_equal(etot, XCVALS[xcname], 5)


class TestFunctionals(unittest.TestCase):
    def test_semilocal_gga(self):
        run_functional("CIDER23X_SL_GGA")

    def test_nonlocal_gga(self):
        run_functional("CIDER23X_NL_GGA")

    def test_semilocal_mgga(self):
        run_functional("CIDER23X_SL_MGGA")

    def test_nonlocal_mgga(self):
        run_functional("CIDER23X_NL_MGGA")

    def test_nonlocal_mgga_pbe(self):
        run_functional("CIDER23X_NL_MGGA_PBE")

    def test_nonlocal_mgga_dtr(self):
        run_functional("CIDER23X_NL_MGGA_DTR")

    def test_cider24ne(self):
        run_functional("CIDER24Xne")

    def test_cider24e(self):
        run_functional("CIDER24Xe")


if __name__ == "__main__":
    unittest.main()
