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
from pyscf import dft, gto
from pyscf.dft.numint import _contract_rho, _dot_ao_dm, eval_ao
from pyscf.pbc import dft as pbcdft
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools

from ciderpress.dft.settings import SDMX1Settings, SDMXSettings
from ciderpress.pyscf import sdmx_slow as sdmx
from ciderpress.pyscf.pbc.sdmx_fft import (
    build_ft_cell,
    compute_sdmx_tensor_lowmem,
    compute_sdmx_tensor_mo,
    compute_sdmx_tensor_vxc,
    convolve_aos,
    get_ao_and_aok,
    get_ao_recip,
    libcider,
    precompute_all_convolutions,
)


class TestSDMXFFT(unittest.TestCase):
    def test_ao(self):
        basis = "gth-szv"
        pseudo = "gth-pade"
        L = 4.65
        bl = 0.9
        atom = "H {:.2f} {:.2f} {:.2f}; F {:.2f} {:.2f} {:.2f}".format(
            0.5 * L,
            0.5 * L,
            0.5 * (L - bl),
            0.5 * L,
            0.5 * L,
            0.5 * (L + bl),
        )
        mol = gto.M(atom=atom, basis=basis, pseudo=pseudo)
        ks = dft.RKS(mol)
        ks.xc = "PBE"
        ks.kernel()
        dm = ks.make_rdm1()
        lattice = np.diag(L * np.ones(3))
        cell = pbcgto.M(atom=atom, a=lattice, basis=basis, pseudo=pseudo)

        pbcks = pbcdft.RKS(cell)
        pbcks.build()
        grids = pbcks.grids

        cond = np.linalg.norm(grids.coords - 0.5 * L * np.ones(3), axis=1) < 1.5
        ao, aok = get_ao_and_aok(cell, grids.coords, np.zeros(3), 0)
        aor = np.asarray(tools.ifft(np.asarray(aok, order="C"), cell.mesh))
        aomol = eval_ao(mol, grids.coords[cond]).T
        assert_allclose(ao.T.real, aor.real, atol=1e-7, rtol=1e-7)
        assert_allclose(aor.real[..., cond], aomol, atol=1e-2)
        assert_allclose(ao.T.real[..., cond], aomol, atol=1e-2)

        settings = SDMXSettings([0])
        settings1 = SDMX1Settings([0], 1)

        alphas = np.array([1.0, 3.5])
        alpha_norms = np.array([3.2, 0.3])
        cao_ref = sdmx.eval_conv_ao(
            (settings, alphas, alpha_norms), mol, grids.coords[cond], deriv=0
        )
        G2 = np.einsum("gv,gv->g", cell.Gv, cell.Gv)
        ws_convs = precompute_all_convolutions(
            cell, np.zeros((1, 3)), alphas, alpha_norms, "gauss_diff"
        )
        for ialpha in range(alphas.size):
            cao_test = convolve_aos(
                cell,
                aok,
                G2,
                alphas[ialpha],
                alpha_norms[ialpha],
                "gauss_diff",
                "ws",
                cutoff_info=ws_convs[ialpha],
                kG=cell.Gv,
            )
            cao_test0 = cao_test[..., cond].real
            assert_allclose(cao_test[..., cond].real, cao_ref[ialpha].T, atol=2e-2)
            cao_test = convolve_aos(
                cell, aok, G2, alphas[ialpha], alpha_norms[ialpha], "gauss_diff", None
            )
            cao_test1 = cao_test[..., cond].real
            assert_allclose(cao_test0, cao_test1, atol=1e-2)
            assert_allclose(cao_test[..., cond].real, cao_ref[ialpha].T, atol=2e-2)

        p_ag = compute_sdmx_tensor_lowmem(
            dm[None, :, :],
            cell,
            grids.coords,
            alphas,
            alpha_norms,
            "gauss_diff",
            np.zeros((1, 3)),
            cutoff_type="ws",
        )
        ao = eval_ao(mol, grids.coords, deriv=0)
        cao = sdmx.eval_conv_ao(
            (settings, alphas, alpha_norms), mol, grids.coords, deriv=0, shls_slice=None
        )
        ncpa = 1
        nalpha = len(alphas)
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()
        c0 = _dot_ao_dm(mol, ao, dm, None, shls_slice, ao_loc)
        tmp = np.empty((ncpa, nalpha, grids.coords.shape[0]))
        for v in range(ncpa):
            for ialpha in range(nalpha):
                tmp[v, ialpha] = _contract_rho(cao[ialpha * ncpa + v], c0)
        assert_allclose(p_ag[0][..., cond], tmp[0][..., cond], atol=1e-3, rtol=1e-4)

        p_ag = compute_sdmx_tensor_lowmem(
            dm[None, :, :],
            cell,
            grids.coords,
            alphas,
            alpha_norms,
            "gauss_diff",
            np.zeros((1, 3)),
            cutoff_type="ws",
            has_l1=True,
        )
        ao = eval_ao(mol, grids.coords, deriv=0)
        cao = sdmx.eval_conv_ao(
            (settings1, alphas, alpha_norms),
            mol,
            grids.coords,
            deriv=1,
            shls_slice=None,
        )
        ncpa = 4
        nalpha = len(alphas)
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()
        c0 = _dot_ao_dm(mol, ao, dm, None, shls_slice, ao_loc)
        tmp = np.empty((ncpa, nalpha, grids.coords.shape[0]))
        for v in range(ncpa):
            for ialpha in range(nalpha):
                tmp[v, ialpha] = _contract_rho(cao[ialpha * ncpa + v], c0)
        assert_allclose(p_ag[0, :, cond], tmp[0, :, cond], atol=1e-3, rtol=1e-4)
        assert_allclose(p_ag[1, :, cond], tmp[1, :, cond], atol=1e-3, rtol=1e-4)
        assert_allclose(p_ag[2, :, cond], tmp[2, :, cond], atol=1e-3, rtol=1e-4)
        assert_allclose(p_ag[3, :, cond], tmp[3, :, cond], atol=1e-3, rtol=1e-4)

    @unittest.skip("high-memory")
    def test_kpts(self):
        from pyscf.pbc.tools import pyscf_ase

        basis = "gth-szv"
        pseudo = "gth-pade"
        from ase.build import bulk

        struct = bulk("C", cubic=True)
        struct * [2, 2, 2]
        struct211 = struct * [2, 1, 1]
        struct.cell
        unitcell = pbcgto.Cell()
        unitcell.a = struct.cell
        unitcell.atom = pyscf_ase.ase_atoms_to_pyscf(struct)
        unitcell.basis = basis
        unitcell.pseudo = pseudo
        unitcell.space_group_symmetry = (True,)
        unitcell.symmorphic = True
        unitcell.mesh = [64, 64, 64]
        unitcell.verbose = 4
        unitcell.build()
        kpts = unitcell.make_kpts(
            [2, 2, 2],
            space_group_symmetry=True,
            time_reversal_symmetry=True,
        )
        ks = pbcdft.KRKS(unitcell, kpts)
        ks.xc = "PBE"
        ks.kernel()

        kpt = kpts.kpts[3]
        recip_cell = build_ft_cell(unitcell)
        t0 = time.monotonic()
        aok_test = get_ao_recip(recip_cell, unitcell.Gv + kpt, deriv=0)
        t1 = time.monotonic()
        print("TEST ORB TIME RECIP", t1 - t0)
        aor_ref, aok_ref = get_ao_and_aok(unitcell, ks.grids.coords, kpt, deriv=0)
        norm = 1.0  # 4 * np.pi * np.prod(unitcell.mesh) / np.abs(np.linalg.det(unitcell.lattice_vectors()))
        # assert_allclose(norm * np.abs(aok_test[0]), np.abs(aok_ref.T), atol=1e-6, rtol=1e-6)
        assert_allclose(norm * aok_test[0], aok_ref.T, atol=1e-6, rtol=1e-6)
        assert_allclose(aok_test[0], aok_ref.T, atol=1e-6, rtol=1e-6)

        assert aok_test[0].flags.f_contiguous
        t0 = time.monotonic()
        for mklpar in [0, 1]:
            _aok_test = aok_test[0].T.copy()
            assert _aok_test.flags.c_contiguous
            assert _aok_test.T.shape == aok_test[0].shape
            libcider.run_ffts(
                _aok_test.ctypes.data_as(ctypes.c_void_p),
                None,
                ctypes.c_double(1.0 / np.prod(unitcell.mesh)),
                (ctypes.c_int * 3)(*unitcell.mesh),
                ctypes.c_int(0),
                ctypes.c_int(aok_test.shape[-1]),
                ctypes.c_int(mklpar),
                ctypes.c_int(0),
            )
            t1 = time.monotonic()
            print("MKL FFT TIME", t1 - t0)
            eikr = np.exp(1j * ks.grids.coords.dot(kpt))
            assert_allclose(_aok_test.T * eikr[:, None], aor_ref, atol=1e-6, rtol=1e-6)
        aok_ref = aok_test = aor_ref = eikr = None

        alphas = [1.0]
        alpha_norms = [1.0]
        dm1 = kpts.transform_dm(ks.make_rdm1())
        p1_ag = compute_sdmx_tensor_lowmem(
            dm1,
            unitcell,
            ks.grids.coords,
            alphas,
            alpha_norms,
            "gauss_diff",
            kpts.kpts,
            cutoff_type="ws",
            has_l1=False,
        )
        for ind in [3, 4, 7]:
            for has_l1 in [True, False]:
                dm1p = [dm.copy() for dm in dm1]
                delta = 1e-4
                delta2 = 0.1
                dm1p[1][ind, ind] += 0.5 * delta
                p1_ag = compute_sdmx_tensor_lowmem(
                    dm1p,
                    unitcell,
                    ks.grids.coords,
                    alphas,
                    alpha_norms,
                    "gauss_diff",
                    kpts.kpts,
                    cutoff_type="ws",
                    has_l1=has_l1,
                )
                dm1p[1][ind, ind] -= delta
                p1_ag = compute_sdmx_tensor_lowmem(
                    dm1p,
                    unitcell,
                    ks.grids.coords,
                    alphas,
                    alpha_norms,
                    "gauss_diff",
                    kpts.kpts,
                    cutoff_type="ws",
                    has_l1=has_l1,
                )
                mo_coeff = kpts.transform_mo_coeff(ks.mo_coeff)
                mo_occ = kpts.transform_mo_occ(ks.mo_occ)
                p1mo_ag = compute_sdmx_tensor_mo(
                    mo_coeff,
                    mo_occ,
                    unitcell,
                    ks.grids.coords,
                    alphas,
                    alpha_norms,
                    "gauss_diff",
                    kpts.kpts,
                    cutoff_type="ws",
                    has_l1=has_l1,
                )
                vgrid = p1mo_ag / np.sqrt(delta2 + p1mo_ag * p1mo_ag)
                vmats = [np.zeros_like(dm) for dm in dm1]
                compute_sdmx_tensor_vxc(
                    vmats,
                    unitcell,
                    vgrid,
                    alphas,
                    alpha_norms,
                    "gauss_diff",
                    kpts.kpts,
                    cutoff_type="ws",
                    has_l1=has_l1,
                )

        supercell = pbcgto.Cell()
        supercell.a = struct211.cell
        supercell.atom = pyscf_ase.ase_atoms_to_pyscf(struct211)
        supercell.basis = basis
        supercell.pseudo = pseudo
        supercell.verbose = 4
        supercell.space_group_symmetry = (True,)
        supercell.symmorphic = True
        supercell.mesh = np.asarray(
            [unitcell.mesh[0] * 2, unitcell.mesh[1], unitcell.mesh[2]],
            dtype=np.int32,
        )
        supercell.build()
        kpts2 = supercell.make_kpts(
            [1, 2, 2],
            space_group_symmetry=True,
            time_reversal_symmetry=True,
        )
        ks2 = pbcdft.KRKS(supercell, kpts2)
        ks2.xc = "PBE"
        ks2.kernel()

        dm1 = kpts.transform_dm(ks.make_rdm1())
        dm2 = kpts2.transform_dm(ks2.make_rdm1())

        alphas = [1.0]
        alpha_norms = [1.0]
        p1_ag = compute_sdmx_tensor_lowmem(
            dm1,
            unitcell,
            ks.grids.coords,
            alphas,
            alpha_norms,
            "gauss_diff",
            kpts.kpts,
            cutoff_type="ws",
            has_l1=False,
        )
        p2_ag = compute_sdmx_tensor_lowmem(
            dm2,
            supercell,
            ks2.grids.coords,
            alphas,
            alpha_norms,
            "gauss_diff",
            kpts2.kpts,
            cutoff_type="ws",
            has_l1=False,
        )
        ngrids = ks.grids.coords.shape[0]
        tol = 1e-3
        assert_allclose(p1_ag, p2_ag[:, :, :ngrids], atol=tol, rtol=tol)
        assert_allclose(p1_ag, p2_ag[:, :, ngrids:], atol=tol, rtol=tol)

        alphas = [1.0, 3.0, 0.25, 0.5] * 2
        alpha_norms = [1.0, 1.0, 1.0, 1.0] * 2
        p1_ag = compute_sdmx_tensor_lowmem(
            dm1,
            unitcell,
            ks.grids.coords,
            alphas,
            alpha_norms,
            "gauss_diff",
            kpts.kpts,
            cutoff_type="ws",
            has_l1=True,
        )
        p2_ag = compute_sdmx_tensor_lowmem(
            dm2,
            supercell,
            ks2.grids.coords,
            alphas,
            alpha_norms,
            "gauss_diff",
            kpts2.kpts,
            cutoff_type="ws",
            has_l1=True,
        )
        mo_coeff = kpts2.transform_mo_coeff(ks2.mo_coeff)
        mo_occ = kpts2.transform_mo_occ(ks2.mo_occ)
        p2mo_ag = compute_sdmx_tensor_mo(
            mo_coeff,
            mo_occ,
            supercell,
            ks2.grids.coords,
            alphas,
            alpha_norms,
            "gauss_diff",
            kpts2.kpts,
            cutoff_type="ws",
            has_l1=True,
        )
        ngrids = ks.grids.coords.shape[0]
        vgrid = p2mo_ag / np.sqrt(delta2 + p2mo_ag * p2mo_ag)
        vmats = [np.zeros_like(dm) for dm in dm2]
        t0 = time.monotonic()
        compute_sdmx_tensor_vxc(
            vmats,
            supercell,
            vgrid,
            alphas,
            alpha_norms,
            "gauss_diff",
            kpts2.kpts,
            cutoff_type="ws",
            has_l1=True,
        )
        t1 = time.monotonic()
        print("VXC time", t1 - t0)
        tol = 1e-3
        for v in range(4):
            assert_allclose(p1_ag[v], p2_ag[v, :, :ngrids], atol=tol, rtol=tol)
            assert_allclose(p1_ag[v], p2_ag[v, :, ngrids:], atol=tol, rtol=tol)
            assert_allclose(p2mo_ag[v], p2_ag[v], atol=tol, rtol=tol)

    def test_full_dft_run(self):
        from pyscf.pbc.tools import pyscf_ase

        from ciderpress.pyscf.pbc import dft

        basis = "gth-szv"
        pseudo = "gth-pade"
        from ase.build import bulk

        struct = bulk("MgO", crystalstructure="rocksalt", a=4.19)
        kpt_grid = [2, 2, 2]
        unitcell = pbcgto.Cell()
        unitcell.a = struct.cell
        unitcell.atom = pyscf_ase.ase_atoms_to_pyscf(struct)
        unitcell.basis = basis
        unitcell.pseudo = pseudo
        unitcell.space_group_symmetry = (True,)
        unitcell.symmorphic = True
        unitcell.mesh = [48, 48, 48]
        unitcell.verbose = 3
        unitcell.build()
        kpts = unitcell.make_kpts(
            kpt_grid,
            space_group_symmetry=True,
            time_reversal_symmetry=True,
        )
        ks = pbcdft.KRKS(unitcell, kpts)
        ks.xc = "r2SCAN"
        mlfunc = "functionals/CIDER24Xe.yaml"

        cider_ks = dft.make_cider_calc(
            ks, mlfunc, xmix=0.25, xkernel="GGA_X_PBE", ckernel="GGA_C_PBE"
        )
        ks = None
        cider_ks.kernel()

    @unittest.skip("high-memory")
    def test_dense_xc(self):
        from pyscf.pbc.tools import pyscf_ase

        from ciderpress.pyscf.pbc import dft

        basis = "gth-dzv"
        pseudo = "gth-pade"
        from ase.build import bulk

        struct = bulk("MgO", crystalstructure="rocksalt", a=4.19)
        kpt_grid = [2, 2, 2]
        unitcell = pbcgto.Cell()
        unitcell.a = struct.cell
        unitcell.atom = pyscf_ase.ase_atoms_to_pyscf(struct)
        unitcell.basis = basis
        unitcell.pseudo = pseudo
        unitcell.space_group_symmetry = (True,)
        unitcell.symmorphic = True
        unitcell.mesh = [48, 48, 48]
        # unitcell.mesh = [54, 54, 54]
        # unitcell.mesh = [64, 64, 64]
        unitcell.verbose = 4
        unitcell.build()
        kpts = unitcell.make_kpts(
            kpt_grid,
            space_group_symmetry=True,
            time_reversal_symmetry=True,
        )
        ks = pbcdft.KRKS(unitcell, kpts)
        mlfunc = "functionals/CIDER24Xe.yaml"
        cider_ks = dft.make_cider_calc(
            ks,
            mlfunc,
            xmix=0.25,
            xkernel="GGA_X_PBE",
            ckernel="GGA_C_PBE",
            dense_mesh=[96] * 3,
        )
        cider_ks.kernel()

    def test_full_dft_run_uks(self):
        from pyscf.pbc.tools import pyscf_ase

        from ciderpress.pyscf.pbc import dft

        basis = "gth-szv"
        pseudo = "gth-pade"
        from ase.build import bulk

        struct = bulk("MgO", crystalstructure="rocksalt", a=4.19)
        kpt_grid = [2, 2, 2]
        unitcell = pbcgto.Cell()
        unitcell.a = struct.cell
        unitcell.atom = pyscf_ase.ase_atoms_to_pyscf(struct)
        unitcell.basis = basis
        unitcell.pseudo = pseudo
        unitcell.space_group_symmetry = (True,)
        unitcell.symmorphic = True
        unitcell.mesh = [48, 48, 48]
        unitcell.verbose = 4
        unitcell.build()
        kpts = unitcell.make_kpts(
            kpt_grid,
            space_group_symmetry=True,
            time_reversal_symmetry=True,
        )
        ks = pbcdft.KUKS(unitcell, kpts)
        ks.xc = "r2SCAN"
        mlfunc = "functionals/CIDER24Xe.yaml"

        t0 = time.monotonic()
        ks.kernel()
        t1 = time.monotonic()
        t2 = time.monotonic()
        cider_ks = dft.make_cider_calc(
            ks, mlfunc, xmix=0.25, xkernel="GGA_X_PBE", ckernel="GGA_C_PBE"
        )
        ks = None
        cider_ks.kernel()
        t3 = time.monotonic()
        print(t1 - t0, t3 - t2)


if __name__ == "__main__":
    unittest.main()
