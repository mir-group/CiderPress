import unittest

import numpy as np
from ase.build import bulk
from ase.parallel import parprint
from gpaw import GPAW, PW, Mixer, restart
from gpaw.mpi import world
from gpaw.xc.libxc import LibXC
from numpy.testing import assert_almost_equal, assert_equal

from ciderpress.gpaw.calculator import (
    CiderGPAW,
    DiffGGA,
    DiffMGGA,
    get_cider_functional,
)


def run_load_write(xc, use_pp=False, is_cider=False, is_nl=False):
    k = 3
    si = bulk("Si")
    mypar = {"domain": min(2, world.size), "augment_grids": True}
    kwargs = dict(
        mode=PW(250),
        mixer=Mixer(0.7, 5, 50.0),
        xc=xc,
        kpts=(k, k, k),
        convergence={"energy": 1e-8},
        parallel=mypar,
        occupations={"name": "fermi-dirac", "width": 0.0},
        setups="sg15" if use_pp else "paw",
        txt="si.txt",
    )
    si.calc = CiderGPAW(**kwargs)

    e0 = si.get_potential_energy()
    calc = si.calc

    calc.write("_tmp.gpw")
    si1, calc1 = restart("_tmp.gpw", Class=CiderGPAW, parallel=mypar)
    if is_cider and is_nl:
        xc0 = calc.hamiltonian.xc
        xc1 = calc1.hamiltonian.xc
        assert_equal(xc0.encut, xc1.encut)
        assert_equal(xc0.Nalpha, xc1.Nalpha)
        assert_equal(xc0.lambd, xc1.lambd)
    calc1.new(mode=PW(250))
    e1 = si1.get_potential_energy()
    assert_almost_equal(e0, e1)
    if is_cider and is_nl:
        assert_almost_equal(xc0.Nalpha, xc1.Nalpha)
    calc.new(mode=PW(320))
    calc1.new(mode=PW(320))
    e2 = calc.get_potential_energy()
    e3 = calc1.get_potential_energy()
    assert_almost_equal(e2, e3)
    if is_cider:
        new_kwargs = {k: v for k, v in kwargs.items()}
        new_kwargs["xc"] = si.calc.hamiltonian.xc.todict()
        new_kwargs["mlfunc_data"] = si.calc.hamiltonian.xc.get_mlfunc_data()
        si.calc = CiderGPAW(**new_kwargs)
        e4 = si.get_potential_energy()
        assert_almost_equal(e0, e4)
    # Test equivalence to normal GPAW
    si.calc = GPAW(**kwargs)
    e5 = si.get_potential_energy()
    assert_almost_equal(e0, e5)
    del si
    del calc
    del calc1


def get_xc(fname, use_paw=True):
    return get_cider_functional(
        fname,
        qmax=300,
        lambd=1.8,
        xmix=0.25,
        pasdw_ovlp_fit=True,
        pasdw_store_funcs=False,
        use_paw=use_paw,
    )


class TestReadWrite(unittest.TestCase):
    def test_nl_ml(self):
        for fname in [
            "functionals/CIDER23X_NL_GGA.yaml",
            "functionals/CIDER23X_NL_MGGA.yaml",
        ]:
            for use_paw in [False, True]:
                xc = get_xc(fname, use_paw)
                parprint("TEST", fname, use_paw)
                with np.errstate(all="ignore"):
                    run_load_write(xc, use_pp=not use_paw, is_cider=True, is_nl=True)

    def test_sl_ml(self):
        for fname in [
            "functionals/CIDER23X_SL_GGA.yaml",
            "functionals/CIDER23X_SL_MGGA.yaml",
        ]:
            xc = get_xc(fname)
            parprint("TEST", fname)
            with np.errstate(all="ignore"):
                run_load_write(xc, is_cider=True)

    def test_sl_ne(self):
        for xc in [
            DiffGGA(LibXC("PBE")),
            DiffMGGA(LibXC("MGGA_X_R2SCAN+MGGA_C_R2SCAN")),
        ]:
            parprint("TEST", xc.kernel.name)
            with np.errstate(all="ignore"):
                run_load_write(xc)

        for xc in ["PBE", "MGGA_X_R2SCAN+MGGA_C_R2SCAN"]:
            parprint("TEST", xc)
            with np.errstate(all="ignore"):
                run_load_write(xc)


if __name__ == "__main__":
    unittest.main()
