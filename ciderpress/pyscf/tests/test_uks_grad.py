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

NULL_SETTINGS = {
    "xkernel": "GGA_X_PBE",
    "ckernel": "GGA_C_PBE",
    "xmix": 0.00,
}

XC_SETTINGS = {
    "xkernel": None,
    "ckernel": None,
    "xc": None,
    "xmix": 1.00,
}

NLDF_SETTINGS = {
    "lmax": 8,
    "aux_lambd": 1.8,
    "aug_beta": 1.8,
    "alpha_max": 1000.0,
}


def build_ks_calc(mol, mlfunc, df=False, alt_settings=None):
    if alt_settings is None:
        settings = SETTINGS
    else:
        settings = alt_settings
    with open(mlfunc, "r") as f:
        mlfunc = yaml.load(f, Loader=yaml.CLoader)
    nldf_init = PySCFNLDFInitializer(mlfunc.settings.nldf_settings, **NLDF_SETTINGS)
    ks = dft.UKS(mol)
    ks.grids.level = 1
    if df:
        ks = ks.density_fit()
    ks = make_cider_calc(ks, mlfunc, nldf_init=nldf_init, **settings)
    ks.conv_tol = CONV_TOL
    return ks


def setUpModule():
    global mol, mlfuncs, mf1, mf2, mf3, mf4, mf5, mf6, mf7
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
    mol.charge = 1
    mol.spin = 1
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
    mf6 = build_ks_calc(mol, mlfuncs[3], df=True, alt_settings=NULL_SETTINGS)
    mf6.kernel()
    mf7 = dft.UKS(mol).density_fit()
    mf7.xc = "PBE"
    mf7.grids.level = 1
    mf7.conv_tol = CONV_TOL
    mf7.kernel()
    if (
        not mf1.converged
        or not mf2.converged
        or not mf3.converged
        or not mf4.converged
        or not mf5.converged
        or not mf6.converged
        or not mf7.converged
    ):
        raise RuntimeError(
            "{} {} {} {} {} {} {}".format(
                mf1.converged,
                mf2.converged,
                mf3.converged,
                mf4.converged,
                mf5.converged,
                mf6.converged,
                mf7.converged,
            )
        )


def tearDownModule():
    global mol, mlfuncs, mf1, mf2, mf3, mf4, mf5, mf6, mf7
    mol.stdout.close()
    del mol, mlfuncs, mf1, mf2, mf3, mf4, mf5, mf6, mf7


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

        g3 = mf6.nuc_grad_method().set(grid_response=True).kernel()
        g4 = mf7.nuc_grad_method().set(grid_response=True).kernel()
        assert_almost_equal(g3, g4, 9)

    def _check_fd(self, functional):
        import os

        path = "{}/functionals/{}.yaml".format(os.getcwd(), functional)
        if not os.path.exists(path):
            self.skipTest("Functional not available, skipping...{}".format(path))
        ks = build_ks_calc(mol, path, alt_settings=XC_SETTINGS)
        ks.kernel()
        g = ks.nuc_grad_method().set(grid_response=True).kernel()
        mol1 = mol.copy()

        mf_scanner = ks.as_scanner()

        e1 = mf_scanner(
            mol1.set_geom_("O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587")
        )
        e2 = mf_scanner(
            mol1.set_geom_("O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587")
        )
        assert_almost_equal(g[0, 2], (e1 - e2) / 2e-4 * lib.param.BOHR, 6)

    def test_vk_xc(self):
        self._check_fd("EXAMPLE_VK_FUNCTIONAL")

    def test_vij_xc(self):
        self._check_fd("EXAMPLE_VIJ_FUNCTIONAL")


if __name__ == "__main__":
    unittest.main()
