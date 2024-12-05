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
from ase.parallel import parprint
from gpaw import GPAW, PW, Mixer

from ciderpress.gpaw.calculator import get_cider_functional

USE_STORED_REF = True
USE_AUGMENT_GRIDS = True


def _run_pw_si_stress(xc, use_pp=False, s_numerical=None):
    # This is based on a test from GPAW
    k = 3
    si = bulk("Si")
    si.calc = GPAW(
        mode=PW(250),
        mixer=Mixer(0.7, 5, 50.0),
        xc=xc,
        kpts=(k, k, k),
        convergence={"energy": 1e-8},
        parallel={"augment_grids": USE_AUGMENT_GRIDS},
        setups="sg15" if use_pp else "paw",
        txt="si.txt",
    )

    si.set_cell(
        np.dot(si.cell, [[1.02, 0, 0.03], [0, 0.99, -0.02], [0.2, -0.01, 1.03]]),
        scale_atoms=True,
    )
    si.get_potential_energy()

    # Trigger nasty bug (fixed in !486):
    si.calc.wfs.pt.blocksize = si.calc.wfs.pd.maxmyng - 1
    s_analytical = si.get_stress()
    if s_numerical is None:
        s_numerical = si.calc.calculate_numerical_stress(si, 1e-5)
    s_err = s_numerical - s_analytical

    parprint("Analytical stress:\n", s_analytical)
    parprint("Numerical stress:\n", s_numerical)
    parprint("Error in stress:\n", s_err)
    assert np.all(abs(s_err) < 1e-4)


def run_pw_si_stress(xc, use_pp=False, s_numerical=None):
    with np.errstate(all="ignore"):
        _run_pw_si_stress(xc, use_pp=use_pp, s_numerical=s_numerical)


def get_xc(fname, use_paw=True):
    return get_cider_functional(
        fname,
        qmax=300,
        lambd=1.8,
        xmix=0.25,
        pasdw_ovlp_fit=False,
        pasdw_store_funcs=True,
        use_paw=use_paw,
    )


class TestStress(unittest.TestCase):
    def test_nl_gga(self):
        xc = get_xc("functionals/CIDER23X_NL_GGA.yaml")
        if USE_STORED_REF:
            s_numerical = np.array(
                [
                    0.00070889,
                    -0.03356804,
                    -0.03070831,
                    -0.02075609,
                    0.12834234,
                    0.01009197,
                ]
            )
            run_pw_si_stress(xc, s_numerical=s_numerical)
        else:
            run_pw_si_stress(xc, s_numerical=None)

    def test_nl_mgga(self):
        xc = get_xc("functionals/CIDER23X_NL_MGGA_DTR.yaml")
        if USE_STORED_REF:
            s_numerical = np.array(
                [
                    -0.00761698,
                    -0.04097609,
                    -0.03780994,
                    -0.022234,
                    0.14451133,
                    0.00919595,
                ]
            )
            run_pw_si_stress(xc, s_numerical=s_numerical)
        else:
            run_pw_si_stress(xc, s_numerical=None)

    def test_sl_gga(self):
        xc = get_xc("functionals/CIDER23X_SL_GGA.yaml")
        if USE_STORED_REF:
            s_numerical = np.array(
                [
                    -0.01818833,
                    -0.05194489,
                    -0.04841279,
                    -0.02068873,
                    0.13185673,
                    0.0095163,
                ]
            )
            run_pw_si_stress(xc, s_numerical=s_numerical)
        else:
            run_pw_si_stress(xc, s_numerical=None)

    def test_sl_mgga(self):
        xc = get_xc("functionals/CIDER23X_SL_MGGA.yaml")
        if USE_STORED_REF:
            s_numerical = np.array(
                [
                    -0.01197671,
                    -0.04729093,
                    -0.04347899,
                    -0.02241506,
                    0.1438053,
                    0.00938071,
                ]
            )
            run_pw_si_stress(xc, s_numerical=s_numerical)
        else:
            run_pw_si_stress(xc, s_numerical=None)

    def test_nl_gga_pp(self):
        xc = get_xc("functionals/CIDER23X_NL_GGA.yaml", use_paw=False)
        if USE_STORED_REF:
            s_numerical = np.array(
                [
                    0.0208956,
                    -0.01761566,
                    -0.00972346,
                    -0.02004954,
                    0.12474854,
                    0.01070437,
                ]
            )
            run_pw_si_stress(xc, use_pp=True, s_numerical=s_numerical)
        else:
            run_pw_si_stress(xc, use_pp=True, s_numerical=None)

    def test_nl_mgga_pp(self):
        with self.assertRaises(NotImplementedError):
            xc = get_xc("functionals/CIDER23X_NL_MGGA.yaml", use_paw=False)
            run_pw_si_stress(xc, use_pp=True, s_numerical=None)


if __name__ == "__main__":
    unittest.main()
