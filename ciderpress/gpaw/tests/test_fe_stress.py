import unittest

import numpy as np
from ase.build import bulk
from ase.parallel import parprint
from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import world

from ciderpress.gpaw.calculator import get_cider_functional

USE_STORED_REF = True


def _run_pw_si_stress(xc, use_pp=False, s_numerical=None):
    # This is based on a test from GPAW
    k = 3
    m = [2.2]
    si = bulk("Fe")
    si.set_initial_magnetic_moments(m)
    si.calc = GPAW(
        mode=PW(800),
        h=0.15,
        occupations=FermiDirac(width=0.1),
        xc=xc,
        kpts=(k, k, k),
        convergence={"energy": 1e-8},
        parallel={"domain": min(2, world.size), "augment_grids": True},
        txt="fe.txt",
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
        fast=True,
    )


class TestStress(unittest.TestCase):
    def test_nl_mgga(self):
        xc = get_xc("functionals/CIDER23X_NL_MGGA_DTR.yaml")
        if USE_STORED_REF:
            s_numerical = np.array(
                [
                    -0.03445521,
                    -0.1184332,
                    -0.09615671,
                    -0.03262395,
                    0.2034452,
                    0.01842121,
                ]
            )
            run_pw_si_stress(xc, s_numerical=s_numerical)
        else:
            run_pw_si_stress(xc, s_numerical=None)


if __name__ == "__main__":
    unittest.main()
