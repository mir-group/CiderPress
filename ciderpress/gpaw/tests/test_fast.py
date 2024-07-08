import unittest

import numpy as np
from ase.build import bulk
from gpaw import PW
from numpy.testing import assert_almost_equal

from ciderpress.gpaw.calculator import CiderGPAW, get_cider_functional


def run_calc(xc, spinpol, setups="paw", fast=False):
    atoms = bulk("Si")
    mlfunc = "functionals/{}.yaml".format(xc)
    xc = get_cider_functional(
        mlfunc,
        xmix=0.25,
        qmax=300,
        lambd=1.8,
        pasdw_store_funcs=False,
        pasdw_ovlp_fit=True,
        use_paw=False if setups == "sg15" else True,
        fast=fast,
    )

    atoms.calc = CiderGPAW(
        h=0.18,  # use a reasonably small grid spacing
        xc=xc,  # assign the CIDER functional to xc
        mode=PW(520),  # plane-wave mode with 520 eV cutoff.
        txt="-",  # output file, '-' for stdout
        occupations={"name": "fermi-dirac", "width": 0.01},
        kpts={"size": (12, 12, 12), "gamma": False},  # kpt mesh parameters
        convergence={"energy": 1e-5},  # convergence energy in eV/electron
        spinpol=spinpol,
        setups=setups,
        parallel={"augment_grids": True},
    )
    etot = atoms.get_potential_energy()  # run the calculation
    return etot


def generate_test(xcname):
    def run_test(self):
        with np.errstate(all="ignore"):
            # e_rks_slow = run_calc(xcname, False, setups="sg15", fast=False)
            # e_uks_slow = run_calc(xcname, True, setups="sg15", fast=False)
            e_rks_fast = run_calc(xcname, False, setups="sg15", fast=True)
            # e_uks_fast = run_calc(xcname, True, setups="sg15", fast=True)
            # assert_almost_equal(e_rks_slow, e_uks_slow)
            assert_almost_equal(e_rks_fast, e_rks_slow)
            # assert_almost_equal(e_uks_fast, e_uks_slow)

    return run_test


class TestEnergy(unittest.TestCase):

    test_nl_gga = generate_test("CIDER23X_NL_GGA")


if __name__ == "__main__":
    unittest.main()
