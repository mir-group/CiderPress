import unittest

import numpy as np
from ase.build import bulk
from gpaw import PW
from numpy.testing import assert_almost_equal

from ciderpress.gpaw.calculator import CiderGPAW, get_cider_functional

USE_FAST_GPAW = True


def run_calc(xc, spinpol, setups="paw"):
    atoms = bulk("Si")
    mlfunc = "functionals/{}.yaml".format(xc)
    xc = get_cider_functional(
        mlfunc,
        xmix=0.25,
        qmax=300,
        lambd=1.8,
        pasdw_store_funcs=False,
        pasdw_ovlp_fit=not USE_FAST_GPAW,
        use_paw=False if setups == "sg15" else True,
        fast=USE_FAST_GPAW,
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


def generate_test(xcname, e_ref):
    def run_test(self):
        with np.errstate(all="ignore"):
            e_rks = run_calc(xcname, False)
            e_uks = run_calc(xcname, True)
            assert_almost_equal(e_rks, e_ref, 6)
            assert_almost_equal(e_uks, e_ref, 6)

    return run_test


class TestEnergy(unittest.TestCase):

    test_sl_gga = generate_test("CIDER23X_SL_GGA", -12.868728302199766)

    test_nl_gga = generate_test("CIDER23X_NL_GGA", -13.045337504398542)

    test_sl_mgga = generate_test("CIDER23X_SL_MGGA", -12.267997644473239)

    test_nl_mgga = generate_test("CIDER23X_NL_MGGA_DTR", -12.537128965314368)


if __name__ == "__main__":
    unittest.main()
