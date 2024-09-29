import unittest

import yaml
from numpy.testing import assert_almost_equal
from pyscf import dft, gto, lib

from ciderpress.pyscf.dft import make_cider_calc
from ciderpress.pyscf.nldf_convolutions import PySCFNLDFInitializer

CONV_TOL = 1e-12

SETTINGS = {
    "xkernel": "GGA_X_PBE",
    "ckernel": "GGA_C_PBE",
    "xmix": 0.25,
}


NLDF_SETTINGS = {
    "lmax": 8,
    "aux_lambd": 1.8,
    "aug_beta": 1.8,
    "alpha_max": 1000.0,
}


def build_ks_calc(mol, mlfunc, df=False):
    assert mol.spin == 0
    with open(mlfunc, "r") as f:
        mlfunc = yaml.load(f, Loader=yaml.CLoader)
    nldf_init = PySCFNLDFInitializer(mlfunc.settings.nldf_settings, **NLDF_SETTINGS)
    ks = dft.RKS(mol)
    if df:
        ks = ks.density_fit()
    ks.grids.level = 1
    ks = make_cider_calc(ks, mlfunc, nldf_init=nldf_init, **SETTINGS)
    ks.small_rho_cutoff = 0.0
    ks.conv_tol = CONV_TOL
    return ks


def setUpModule():
    global mol, mlfuncs, mf1, mf2, mf3, mf4, mf5
    mlfuncs = [
        "functionals/{}.yaml".format(fname)
        for fname in [
            "CIDER23X_SL_GGA",
            "CIDER23X_SL_MGGA",
            "CIDER23X_NL_GGA",
            "CIDER23X_NL_MGGA_DTR",
        ]
    ]
    mol = gto.Mole()
    mol.verbose = 4
    mol.output = "/dev/null"
    mol.atom.extend(
        [["O", (0.0, 0.0, 0.0)], [1, (0.0, -0.757, 0.587)], [1, (0.0, 0.757, 0.587)]]
    )
    mol.basis = "6-31g"
    mol.build()
    mf1 = build_ks_calc(mol, mlfuncs[1])
    mf2 = build_ks_calc(mol, mlfuncs[3])
    mf1.kernel()
    mf2.kernel()
    mf3 = build_ks_calc(mol, mlfuncs[3])
    mf3.grids.level = 3
    mf3.grids.build()
    mf3.kernel()
    mf4 = build_ks_calc(mol, mlfuncs[3], df=True)
    mf4.kernel()
    mf5 = build_ks_calc(mol, mlfuncs[3], df=False).density_fit()
    mf5.kernel()
    if (
        not mf1.converged
        or not mf2.converged
        or not mf3.converged
        or not mf4.converged
        or not mf5.converged
    ):
        raise RuntimeError(
            "{} {} {} {} {}".format(
                mf1.converged,
                mf2.converged,
                mf3.converged,
                mf4.converged,
                mf5.converged,
            )
        )


def tearDownModule():
    global mol, mlfuncs, mf1, mf2, mf3, mf4, mf5
    mol.stdout.close()
    del mol, mlfuncs, mf1, mf2, mf3, mf4, mf5


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
        assert_almost_equal(g[0, 2], (e1 - e2) / 2e-4 * lib.param.BOHR, 7)

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
        assert_almost_equal(g[0, 2], (e1 - e2) / 2e-4 * lib.param.BOHR, 7)

    def test_finite_difference_df(self):
        g = mf4.nuc_grad_method().set(grid_response=True).kernel()
        mol1 = mol.copy()
        mf_scanner = mf4.as_scanner()

        e1 = mf_scanner(
            mol1.set_geom_("O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587")
        )
        e2 = mf_scanner(
            mol1.set_geom_("O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587")
        )
        assert_almost_equal(g[0, 2], (e1 - e2) / 2e-4 * lib.param.BOHR, 7)

        g2 = mf5.nuc_grad_method().set(grid_response=True).kernel()
        assert_almost_equal(g2, g, 9)


if __name__ == "__main__":
    unittest.main()
