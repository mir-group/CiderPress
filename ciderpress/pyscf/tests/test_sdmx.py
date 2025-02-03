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

import ctypes
import time
import unittest

import numpy as np
from numpy.testing import assert_allclose
from pyscf import gto, lib
from pyscf.dft.gen_grid import Grids
from pyscf.gto.mole import ANG_OF, ATOM_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_EXP

from ciderpress.dft.settings import SDMXG1Settings, SDMXGSettings
from ciderpress.dft.sph_harm_coeff import get_deriv_ylm_coeff
from ciderpress.lib import load_library as load_cider_library
from ciderpress.pyscf.sdmx import eval_conv_sh
from ciderpress.pyscf.sdmx_slow import (
    PySCFSDMXInitializer,
    eval_conv_ao,
    eval_conv_ao_fast,
)

libcider = load_cider_library("libmcider")


def _get_ylm(lmax, r, grad=False):
    nlm = (lmax + 1) * (lmax + 1)
    rnorm = np.linalg.norm(r + 1e-32, axis=1)
    r = np.ascontiguousarray(r / rnorm[:, None])
    if grad:
        ylm = np.zeros((r.shape[0], nlm))
        dylm = np.zeros((r.shape[0], 3, nlm))
        if lmax == 0:
            ylm[:, 0] = 1.0 / np.sqrt(4 * np.pi)
            dylm[:, :, 0] = 0
        else:
            libcider.recursive_sph_harm_deriv_vec(
                ctypes.c_int(nlm),
                ctypes.c_int(r.shape[0]),
                r.ctypes.data_as(ctypes.c_void_p),
                ylm.ctypes.data_as(ctypes.c_void_p),
                dylm.ctypes.data_as(ctypes.c_void_p),
            )
        for l in range(lmax + 1):
            i0, i1 = l * l, (l + 1) * (l + 1)
            rpl = rnorm[:, None] ** l
            rplm1 = rnorm[:, None] ** (l - 1)
            dylm[:, :, i0:i1] *= rplm1[..., None]
            dylm[:, :, i0:i1] += (
                ylm[:, None, i0:i1] * l * (rplm1[:, :, None] * r[:, :, None])
            )
            ylm[:, i0:i1] *= rpl
        ylm = np.concatenate([ylm[:, None, :], dylm], axis=1)
    else:
        ylm = np.zeros((r.shape[0], nlm))
        if lmax == 0:
            ylm[:, 0] = 1.0 / np.sqrt(4 * np.pi)
        else:
            libcider.recursive_sph_harm_vec(
                ctypes.c_int(nlm),
                ctypes.c_int(r.shape[0]),
                r.ctypes.data_as(ctypes.c_void_p),
                ylm.ctypes.data_as(ctypes.c_void_p),
            )
        for l in range(lmax + 1):
            ylm[:, l * l : (l + 1) * (l + 1)] *= rnorm[:, None] ** l
    return ylm


def get_reference_ylm(mol, lmax_list, coords, grad=False, xyz=False):
    ylms = []
    atom_coords = mol.atom_coords(unit="Bohr")
    for i in range(mol.natm):
        ylms.append(_get_ylm(lmax_list[i], coords - atom_coords[i], grad=grad))
        if xyz and lmax_list[i] > 0:
            tmp = ylms[-1][..., [3, 1, 2]].copy()  # x, y, z
            ylms[-1][..., 1:4] = tmp
    return np.concatenate(ylms, axis=-1)


def get_test_ylm(mol, lmax_list, coords, grad=False, xyz=False):
    ylm_atom_loc = np.zeros(mol.natm + 1, dtype=np.int32)
    for i, l in enumerate(lmax_list):
        ylm_atom_loc[i + 1] = ylm_atom_loc[i] + (l + 1) * (l + 1)
    if grad:
        ylm = np.empty((4, ylm_atom_loc[-1], coords.shape[0]))
    else:
        ylm = np.empty((ylm_atom_loc[-1], coords.shape[0]))
    coords = np.asarray(coords, dtype=np.double, order="F")
    atom_coords = np.ascontiguousarray(mol.atom_coords(unit="Bohr"))
    libcider.SDMXylm_loop(
        ctypes.c_int(coords.shape[0]),
        ylm.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        ylm_atom_loc.ctypes.data_as(ctypes.c_void_p),
        atom_coords.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(mol.natm),
    )
    if grad:
        gaunt_coeff = get_deriv_ylm_coeff(np.max(lmax_list))
        libcider.SDMXylm_grad(
            ctypes.c_int(coords.shape[0]),
            ylm.ctypes.data_as(ctypes.c_void_p),
            gaunt_coeff.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(gaunt_coeff.shape[1]),
            ylm_atom_loc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mol.natm),
        )
    if xyz:
        libcider.SDMXylm_yzx2xyz(
            ctypes.c_int(coords.shape[0]),
            ctypes.c_int(4 if grad else 1),
            ylm.ctypes.data_as(ctypes.c_void_p),
            ylm_atom_loc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mol.natm),
        )
    return ylm


class TestSDMX(unittest.TestCase):
    def test_ylm(self):
        for lmax_list in [[3, 4], [3, 0], [2, 1], [4, 4], [5, 6]]:
            mol = gto.M(
                atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvp", output="/dev/null"
            )

            grids = Grids(mol)
            grids.level = 0
            grids.build()

            ref_ylm = get_reference_ylm(mol, lmax_list, grids.coords)
            test_ylm = get_test_ylm(mol, lmax_list, grids.coords)
            assert_allclose(ref_ylm, test_ylm.T, atol=1e-8, rtol=1e-10)

            ref_ylm = get_reference_ylm(mol, lmax_list, grids.coords, xyz=True)
            test_ylm = get_test_ylm(mol, lmax_list, grids.coords, xyz=True)
            assert_allclose(ref_ylm, test_ylm.T, atol=1e-8, rtol=1e-10)

            ref_ylm = get_reference_ylm(mol, lmax_list, grids.coords, grad=True)
            test_ylm = get_test_ylm(mol, lmax_list, grids.coords, grad=True)
            assert_allclose(ref_ylm[:, 0, :], test_ylm[0].T, atol=1e-8, rtol=1e-10)
            assert_allclose(ref_ylm[:, 1, :], test_ylm[1].T, atol=1e-8, rtol=1e-10)
            assert_allclose(ref_ylm[:, 2, :], test_ylm[2].T, atol=1e-8, rtol=1e-10)
            assert_allclose(
                ref_ylm[:, 3, :], test_ylm[3].T[:, :], atol=1e-8, rtol=1e-10
            )

            ref_ylm = get_reference_ylm(
                mol, lmax_list, grids.coords, grad=True, xyz=True
            )
            test_ylm = get_test_ylm(mol, lmax_list, grids.coords, grad=True, xyz=True)
            assert_allclose(ref_ylm[:, 0, :], test_ylm[0].T, atol=1e-8, rtol=1e-10)
            assert_allclose(ref_ylm[:, 1, :], test_ylm[1].T, atol=1e-8, rtol=1e-10)
            assert_allclose(ref_ylm[:, 2, :], test_ylm[2].T, atol=1e-8, rtol=1e-10)
            assert_allclose(
                ref_ylm[:, 3, :], test_ylm[3].T[:, :], atol=1e-8, rtol=1e-10
            )

            mol.stdout.close()

    def test_cao(self):
        settings0 = SDMXGSettings([0, 1, 2], 2)
        settings1 = SDMXG1Settings([0, 1, 2], 2, 2)
        init0 = PySCFSDMXInitializer(settings0)
        init1 = PySCFSDMXInitializer(settings1)
        inits = {0: init0, 1: init1}
        for itype in ["gauss_r2", "gauss_diff"]:
            for basis in ["def2-svp", "def2-qzvppd", "cc-pvtz"]:
                for deriv in [0, 1]:
                    mol = gto.M(
                        atom="H 0 0 0; F 0 0 0.9", basis=basis, output="/dev/null"
                    )
                    grids = Grids(mol)
                    grids.level = 0
                    grids.build()
                    coords = grids.coords
                    plan = inits[deriv].initialize_sdmx_generator(mol, 1).plan
                    plan.settings._integral_type = itype

                    cao_ref = eval_conv_ao_fast(
                        plan,
                        mol,
                        coords,
                        deriv=deriv,
                    )
                    cao_test = eval_conv_ao(plan, mol, coords, deriv=deriv)
                    assert_allclose(cao_test, cao_ref, atol=1e-8, rtol=1e-10)

                    mol.stdout.close()

    def test_cao_speed(self):
        settings0 = SDMXGSettings([0, 1, 2], 2)
        settings1 = SDMXG1Settings([0, 1, 2], 2, 2)
        init0 = PySCFSDMXInitializer(settings0)
        init1 = PySCFSDMXInitializer(settings1)
        inits = {0: init0, 1: init1}
        for itype in ["gauss_r2", "gauss_diff"]:
            for basis in ["def2-svp", "def2-qzvppd", "cc-pvtz"]:
                for deriv in [0, 1]:
                    mol = gto.M(
                        atom="H 0 0 0; F 0 0 0.9", basis=basis, output="/dev/null"
                    )
                    grids = Grids(mol)
                    grids.level = 3
                    grids.build()
                    blksize = 10000
                    plan = inits[deriv].initialize_sdmx_generator(mol, 1).plan
                    plan.settings._integral_type = itype
                    ncomp = plan.nalpha * (1 + 3 * deriv)
                    nao = mol.nao_nr()
                    cao_buf = np.empty((ncomp, blksize, nao))
                    ylm_buf = np.empty((25, blksize, nao))
                    t0 = time.monotonic()
                    for i0, i1 in lib.prange(0, grids.coords.shape[0], blksize):
                        eval_conv_ao_fast(
                            plan,
                            mol,
                            grids.coords[i0:i1],
                            deriv=deriv,
                            out=cao_buf,
                            ylm_buf=ylm_buf,
                        )
                    t1 = time.monotonic()
                    cao_buf = np.empty((ncomp, blksize, nao))
                    t2 = time.monotonic()
                    for i0, i1 in lib.prange(0, grids.coords.shape[0], blksize):
                        eval_conv_ao(
                            plan, mol, grids.coords[i0:i1], deriv=deriv, out=cao_buf
                        )
                    t3 = time.monotonic()

                    cao_buf = np.empty((ncomp, blksize, nao))
                    t4 = time.monotonic()
                    for i0, i1 in lib.prange(0, grids.coords.shape[0], blksize):
                        eval_conv_sh(
                            plan, mol, grids.coords[i0:i1], deriv=deriv, out=cao_buf
                        )
                    t5 = time.monotonic()
                    print(t1 - t0, t3 - t2, t5 - t4)
                    print()

                    mol.stdout.close()

    def test_sdmx_module(self):
        from pyscf import dft

        from ciderpress.pyscf.sdmx import PySCFSDMXInitializer as NewSDMXInit

        settings0 = SDMXGSettings([0, 1, 2], 2)
        settings1 = SDMXG1Settings([0, 1, 2], 2, 2)
        init0 = PySCFSDMXInitializer(settings0)
        init1 = PySCFSDMXInitializer(settings1)
        inits = {0: init0, 1: init1}
        init0 = NewSDMXInit(settings0)
        init1 = NewSDMXInit(settings1)
        new_inits = {0: init0, 1: init1}
        for itype in ["gauss_diff"]:
            for basis in ["def2-svp", "def2-tzvpd", "def2-qzvppd", "aug-cc-pvtz"]:
                for deriv in [0, 1]:
                    mol = gto.M(
                        atom="H 0 0 0; F 0 0 0.9", basis=basis, output="/dev/null"
                    )
                    grids = Grids(mol)
                    grids.level = 1
                    grids.build()
                    plan = inits[deriv].initialize_sdmx_generator(mol, 1).plan
                    plan.settings._integral_type = itype
                    refgen = inits[deriv].initialize_sdmx_generator(mol, 1)
                    testgen = new_inits[deriv].initialize_sdmx_generator(mol, 1)

                    ks = dft.RKS(mol)
                    ks.xc = "PBE"
                    ks.kernel()
                    dm = ks.make_rdm1()

                    t0 = time.monotonic()
                    ref_feat = refgen.get_features(dm, mol, grids.coords, save_buf=True)
                    t1 = time.monotonic()
                    test_feat = testgen.get_features(
                        dm, mol, grids.coords, save_buf=True
                    )
                    t2 = time.monotonic()
                    print(t1 - t0, t2 - t1)

                    np.random.seed(42)
                    vxc_grid = np.random.normal(size=ref_feat.shape) ** 2
                    nao = mol.nao_nr()
                    vxc_mat_ref = np.zeros((nao, nao))
                    vxc_mat_test = np.zeros((nao, nao))
                    refgen.get_vxc_(vxc_mat_ref, vxc_grid)
                    testgen.get_vxc_(vxc_mat_test, vxc_grid)

                    assert_allclose(test_feat, ref_feat, rtol=1e-8, atol=1e-10)
                    assert_allclose(vxc_mat_test, vxc_mat_ref, rtol=1e-8, atol=1e-10)

                    mol.stdout.close()

    def test_conv_shl(self):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-svp", output="/dev/null")
        grids = Grids(mol)
        grids.level = 0
        grids.build()
        coords = grids.coords

        alphas = np.asarray([0.04, 1.0, 11.0], dtype=np.float64)
        alpha_norms = np.asarray([0.5, 1.0, 1.5], dtype=np.float64)

        plan = (alphas, alpha_norms, "gauss_r2")

        # ATOM_OF, ANG_OF, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_EXP

        csh = eval_conv_sh(plan, mol, coords, deriv=0)
        assert csh.shape == (len(alphas), coords.shape[0], mol.nbas)
        assert not np.isnan(csh).any()
        csh_ref = np.zeros_like(csh)

        bas = mol._bas
        env = mol._env
        atom_coord = mol.atom_coords(unit="Bohr")
        for ib in range(mol.nbas):
            ia = bas[ib, ATOM_OF]
            l = bas[ib, ANG_OF]
            nc = bas[ib, NCTR_OF]
            nprim = bas[ib, NPRIM_OF]
            assert nc == 1
            dr = coords - atom_coord[ia]
            rr = np.linalg.norm(dr, axis=1) ** 2
            rr = rr[:, None]
            exps = env[bas[ib, PTR_EXP] : bas[ib, PTR_EXP] + nprim]
            coeffs = env[bas[ib, PTR_COEFF] : bas[ib, PTR_COEFF] + nprim]
            for ialpha in range(len(alphas)):
                conv_alpha = alphas[ialpha]
                conv_alpha_coeff = alpha_norms[ialpha]
                conv_exp = exps * conv_alpha / (exps + conv_alpha)
                r0mul = (1.5 + l) / (exps + conv_alpha) - l / conv_alpha
                r2mul = conv_exp * (1.0 / conv_alpha - 1.0 / (exps + conv_alpha))
                arr = np.exp(-conv_exp * rr)
                eprim = (r0mul + rr * r2mul) * arr
                conv_coeff = (
                    1.0
                    * (np.pi / conv_alpha) ** 1.5
                    * conv_alpha_coeff
                    * (conv_alpha / (exps + conv_alpha)) ** (1.5 + l)
                ) * coeffs
                csh_ref[ialpha, :, ib] = np.dot(eprim, conv_coeff)

        assert_allclose(csh, csh_ref)

        mol.stdout.close()


if __name__ == "__main__":
    unittest.main()
