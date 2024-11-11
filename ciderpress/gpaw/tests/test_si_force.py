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
from ase import Atoms
from gpaw import GPAW, PW, FermiDirac, Mixer

from ciderpress.gpaw.calculator import get_cider_functional
from ciderpress.gpaw.tests import equal

USE_AUGMENT_GRIDS = True


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
    a = 5.45
    bulk = Atoms(
        symbols="Si8",
        positions=[
            (0, 0, 0.1 / a),
            (0, 0.5, 0.5),
            (0.5, 0, 0.5),
            (0.5, 0.5, 0),
            (0.25, 0.25, 0.25),
            (0.25, 0.75, 0.75),
            (0.75, 0.25, 0.75),
            (0.75, 0.75, 0.25),
        ],
        pbc=True,
    )
    bulk.set_cell((a, a, a), scale_atoms=True)
    if get_xc is not None:
        functional = get_xc()
    calc = GPAW(
        h=0.20,
        mode=PW(350),
        xc=functional,
        nbands="150%",
        occupations=FermiDirac(width=0.01),
        kpts=(3, 3, 3),
        convergence={"energy": 1e-8},
        mixer=Mixer(0.7, 8, 50),
        parallel={"augment_grids": USE_AUGMENT_GRIDS},
    )
    bulk.calc = calc
    f1 = bulk.get_forces()[0, 2]
    bulk.get_potential_energy()

    f2 = numeric_force(bulk, 0, 2, 0.001, get_xc=get_xc)
    print((f1, f2, f1 - f2))
    equal(f1, f2, 0.001)


def run_cider_forces(functional, get_xc=None):
    with np.errstate(all="ignore"):
        _run_cider_forces(functional, get_xc=get_xc)


class TestForce(unittest.TestCase):
    def test_sl_gga(self):
        run_cider_forces(get_cider_functional("functionals/CIDER23X_SL_GGA.yaml"))

    def test_sl_mgga(self):
        run_cider_forces(get_cider_functional("functionals/CIDER23X_SL_MGGA.yaml"))

    def test_nl_gga(self):
        def get_xc():
            return get_cider_functional(
                "functionals/CIDER23X_NL_GGA.yaml",
                qmax=300,
                lambd=1.8,
                xmix=0.25,
                pasdw_ovlp_fit=True,
                pasdw_store_funcs=True,
            )

        run_cider_forces(get_xc())

    def test_nl_mgga(self):
        def get_xc():
            return get_cider_functional(
                "functionals/CIDER23X_NL_MGGA_DTR.yaml",
                qmax=300,
                lambd=1.8,
                xmix=0.25,
                pasdw_ovlp_fit=True,
                pasdw_store_funcs=False,
            )

        run_cider_forces(get_xc())


if __name__ == "__main__":
    unittest.main()
