import unittest

from numpy.testing import assert_almost_equal
from pyscf import dft, gto, lib

from ciderpress.pyscf.dft import make_cider_calc

CONV_TOL = 1e-12

SETTINGS = {
    "xkernel": "GGA_X_PBE",
    "ckernel": "GGA_C_PBE",
    "xmix": 0.25,
    "grid_level": 2,
    "debug": False,
    "amax": 1000.0,
    "cider_lmax": 8,
    "lambd": 1.8,
    "aux_beta": 1.8,
}


def build_ks_calc(mol, mlfunc):
    ks = dft.UKS(mol)
    ks.grids.level = 2
    ks = make_cider_calc(ks, mlfunc, **SETTINGS)
    return ks


def setUpModule():
    global mol, mlfuncs, mf1, mf2, mf3
    mlfuncs = [
        "functionals/{}.yaml".format(fname)
        for fname in [
            "CIDER23X_SL_GGA",
            "CIDER23X_SL_MGGA",
            "CIDER23X_NL_GGA",
            "CIDER23X_NL_MGGA",
        ]
    ]
    mol = gto.Mole()
    mol.verbose = 4
    mol.output = "/dev/null"
    mol.atom.extend(
        [["O", (0.0, 0.0, 0.0)], [1, (0.0, -0.757, 0.587)], [1, (0.0, 0.757, 0.587)]]
    )
    mol.basis = "6-31g"
    mol.charge = 1
    mol.spin = 1
    mol.build()
    mf1 = build_ks_calc(mol, mlfuncs[1])
    mf2 = build_ks_calc(mol, mlfuncs[3])
    mf1.conv_tol = CONV_TOL
    mf1.kernel()
    mf2.conv_tol = CONV_TOL
    mf2.kernel()
    mf3 = build_ks_calc(mol, mlfuncs[3], False)
    mf3.grids.level = 3
    mf3.grids.build()
    mf3.conv_tol = CONV_TOL
    mf3.kernel()


def tearDownModule():
    global mol, mlfuncs, mf1, mf2, mf3
    mol.stdout.close()
    del mol, mlfuncs, mf1, mf2, mf3


class KnownValues(unittest.TestCase):
    def test_fd_cider_grad_nogrid(self):
        g = mf3.nuc_grad_method().kernel()
        mol1 = mol.copy()
        mf_scanner = mf3.as_scanner()

        e1 = mf_scanner(
            mol1.set_geom_("O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587")
        )
        e2 = mf_scanner(
            mol1.set_geom_("O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587")
        )
        assert_almost_equal(g[0, 2], (e1 - e2) / 2e-4 * lib.param.BOHR, 3)

    def test_finite_difference(self):
        g = mf1.nuc_grad_method().set(grid_response=True).kernel()
        mol1 = mol.copy()
        mf_scanner = mf1.as_scanner()

        e1 = mf_scanner(
            mol1.set_geom_("O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587")
        )
        e2 = mf_scanner(
            mol1.set_geom_("O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587")
        )
        # TODO precision for LDA is 6
        assert_almost_equal(g[0, 2], (e1 - e2) / 2e-4 * lib.param.BOHR, 5)

    def test_finite_difference_nl(self):
        g = mf2.nuc_grad_method().set(grid_response=True).kernel()
        mol1 = mol.copy()
        mf_scanner = mf2.as_scanner()

        e1 = mf_scanner(
            mol1.set_geom_("O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587")
        )
        e2 = mf_scanner(
            mol1.set_geom_("O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587")
        )
        # TODO precision for LDA is 6
        assert_almost_equal(g[0, 2], (e1 - e2) / 2e-4 * lib.param.BOHR, 6)


if __name__ == "__main__":
    unittest.main()
