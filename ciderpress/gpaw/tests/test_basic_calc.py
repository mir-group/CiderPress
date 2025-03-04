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
from ase.build import bulk
from gpaw import PW
from numpy.testing import assert_almost_equal

from ciderpress.gpaw.calculator import CiderGPAW, get_cider_functional

NKPT = 4
REFERENCE_ENERGIES = {
    "CIDER23X_SL_GGA": -12.866290987790224,
    "CIDER23X_NL_GGA": -13.046146980191777,
    "CIDER23X_SL_MGGA": -12.265893307629582,
    "CIDER23X_NL_MGGA_DTR": -12.53022643638701,
}
USE_AUGMENT_GRIDS = True


def run_calc(xc, spinpol, setups="paw"):
    atoms = bulk("Si")
    mlfunc = "functionals/{}.yaml".format(xc)
    xc = get_cider_functional(
        mlfunc,
        xmix=0.25,
        qmax=300,
        lambd=1.8,
        pasdw_store_funcs=True,
        pasdw_ovlp_fit=True,  # not USE_FAST_GPAW,
        use_paw=False if setups == "sg15" else True,
    )

    atoms.calc = CiderGPAW(
        h=0.18,  # use a reasonably small grid spacing
        xc=xc,  # assign the CIDER functional to xc
        mode=PW(520),  # plane-wave mode with 520 eV cutoff.
        txt="-",  # output file, '-' for stdout
        occupations={"name": "fermi-dirac", "width": 0.01},
        kpts={"size": (NKPT, NKPT, NKPT), "gamma": False},  # kpt mesh parameters
        convergence={"energy": 1e-7},  # convergence energy in eV/electron
        spinpol=spinpol,
        setups=setups,
        parallel={"augment_grids": USE_AUGMENT_GRIDS},
    )
    etot = atoms.get_potential_energy()  # run the calculation
    return etot


def generate_test(xcname):
    e_ref = REFERENCE_ENERGIES[xcname]

    def run_test(self):
        with np.errstate(all="ignore"):
            e_rks = run_calc(xcname, False)
            e_uks = run_calc(xcname, True)
            assert_almost_equal(e_rks, e_ref, 6)
            assert_almost_equal(e_uks, e_ref, 6)

    return run_test


class TestEnergy(unittest.TestCase):

    test_sl_gga = generate_test("CIDER23X_SL_GGA")

    test_nl_gga = generate_test("CIDER23X_NL_GGA")

    test_sl_mgga = generate_test("CIDER23X_SL_MGGA")

    test_nl_mgga = generate_test("CIDER23X_NL_MGGA_DTR")


if __name__ == "__main__":
    unittest.main()
