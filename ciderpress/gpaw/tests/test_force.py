import unittest

import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, FermiDirac, Mixer

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
    bulk.set_cell(
        np.dot(bulk.cell, [[1.02, 0, 0.03], [0, 0.99, -0.02], [0.2, -0.01, 1.03]]),
        scale_atoms=True,
    )
    if get_xc is not None:
        functional = get_xc()
    calc = GPAW(
        h=0.2,
        mode=PW(350),
        xc=functional,
        nbands="150%",
        occupations=FermiDirac(width=0.01),
        kpts=(3, 3, 3),
        convergence={"energy": 1e-7},
        parallel={"augment_grids": True},
        mixer=Mixer(0.7, 8, 50),
    )
    bulk.calc = calc
    f1 = bulk.get_forces()[0, 2]
    bulk.get_potential_energy()

    f2 = numeric_force(bulk, 0, 2, 0.001, get_xc=get_xc)
    print((f1, f2, f1 - f2))
    equal(f1, f2, 0.005)


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
