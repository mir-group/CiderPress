import unittest

import numpy as np
from ase.build import bulk
from ase.parallel import parprint
from gpaw import GPAW, PW, Mixer
from gpaw.mpi import world

from ciderpress.gpaw.calculator import get_cider_functional

USE_STORED_REF = True


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
        parallel={"domain": min(2, world.size)},
        setups="sg15" if use_pp else "paw",
        txt="si.txt",
    )

    si.set_cell(
        np.dot(si.cell, [[1.02, 0, 0.03], [0, 0.99, -0.02], [0.2, -0.01, 1.03]]),
        scale_atoms=True,
    )

    etot = si.get_potential_energy()
    print(etot)

    # Trigger nasty bug (fixed in !486):
    si.calc.wfs.pt.blocksize = si.calc.wfs.pd.maxmyng - 1

    s_analytical = si.get_stress()
    parprint(s_analytical)
    # TEST_CIDER_GGA Numerical
    # [-0.00261187 -0.03790705 -0.03193711 -0.0209582   0.13427714  0.00928778]
    # TEST_CIDER_MGGA Numerical
    # [-0.00681636 -0.04026119 -0.03689781 -0.02227667  0.14441494  0.00907815]
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
        qmax=120,
        lambd=1.8,
        xmix=0.25,
        pasdw_ovlp_fit=True,
        pasdw_store_funcs=True,
        use_paw=use_paw,
    )


class TestStress(unittest.TestCase):
    def test_nl_gga(self):
        xc = get_xc("functionals/CIDER23X_NL_GGA.yaml")
        if USE_STORED_REF:
            s_numerical = np.array(
                [
                    0.00041062,
                    -0.03364324,
                    -0.03095064,
                    -0.02067178,
                    0.12828351,
                    0.01013872,
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
                    -0.00737784,
                    -0.04081598,
                    -0.03757383,
                    -0.0222981,
                    0.14454075,
                    0.00917949,
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
                    0.02085583,
                    -0.01765656,
                    -0.00976421,
                    -0.02005148,
                    0.12475251,
                    0.01070791,
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
