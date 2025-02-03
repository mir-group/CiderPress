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
from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import world

from ciderpress.gpaw.calculator import get_cider_functional
from ciderpress.gpaw.tests import equal


def numeric_force(atoms, a, i, d=0.001, get_xc=None):
    """Compute numeric force on atom with index a, Cartesian component i,
    with finite step of size d
    """
    p0 = atoms.get_positions()
    p = p0.copy()
    p[a, i] += d
    atoms.set_positions(p, apply_constraint=False)
    eplus = atoms.get_potential_energy()
    p[a, i] -= 2 * d
    atoms.set_positions(p, apply_constraint=False)
    eminus = atoms.get_potential_energy()
    atoms.set_positions(p0, apply_constraint=False)
    return (eminus - eplus) / (2 * d)


def _run_cider_forces(functional, get_xc=None):
    m = [2.2, 2.2]
    fe = bulk("Fe", crystalstructure="bcc")
    fe = fe * [2, 1, 1]
    fe.set_initial_magnetic_moments(m)
    if get_xc is not None:
        functional = get_xc()
    fe.calc = GPAW(
        mode=PW(500),
        h=0.2,
        occupations=FermiDirac(width=0.1),
        xc=functional,
        kpts=(2, 4, 4),
        convergence={"energy": 1e-8},
        parallel={"domain": min(2, world.size), "augment_grids": True},
        symmetry="off",
        txt="fe.txt",
    )
    pos = fe.get_positions()
    pos[0, 1] += 0.08
    fe.set_positions(pos)
    fe.get_potential_energy()
    f1 = fe.get_forces()[0, 2]

    f2 = numeric_force(fe, 0, 2, 0.001, get_xc=get_xc)
    print((f1, f2, f1 - f2))
    # This is a really loose bound but same order as with PBE
    # Just to make things things aren't totally broken
    equal(f1, f2, 0.02)


def run_cider_forces(functional, get_xc=None):
    with np.errstate(all="ignore"):
        _run_cider_forces(functional, get_xc=get_xc)


class TestForce(unittest.TestCase):
    def test_nl_mgga(self):
        def get_xc():
            return get_cider_functional(
                "functionals/CIDER23X_NL_MGGA_DTR.yaml",
                qmax=300,
                lambd=1.8,
                xmix=0.25,
                pasdw_ovlp_fit=False,
                pasdw_store_funcs=False,
            )

        run_cider_forces(get_xc())


if __name__ == "__main__":
    unittest.main()
