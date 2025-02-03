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
import unittest

import numpy
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from pyscf import gto
from pyscf.dft.gen_grid import Grids, libdft
from pyscf.dft.numint import eval_ao
from pyscf.gto import ANG_OF, PTR_COEFF, PTR_EXP

from ciderpress.dft.lcao_interpolation import (
    ATCBasis,
    LCAOInterpolator,
    LCAOInterpolatorDirect,
    libcider,
)
from ciderpress.pyscf.gen_cider_grid import CiderGrids
from ciderpress.pyscf.nldf_convolutions import (
    aug_etb_for_cider,
    get_gamma_lists_from_mol,
)


class TestLCAOInterpolator(unittest.TestCase):

    atco = None
    mol = None
    mol1 = None
    molsquish = None
    molsquish1 = None
    interpolator = None
    n0 = 0
    n1 = 0

    @classmethod
    def setUpClass(cls):
        molstr = "H 0 0 0; F 0 0 0.9"
        # molstr = 'Ne'

        mol = gto.M(atom=molstr, basis="def2-tzvpd")
        basis = aug_etb_for_cider(mol, lmax=6)
        cls.mol = gto.M(atom=molstr, basis=basis)
        # For evaluating reference values; set F to 000 since
        # everything is referenced to nuclear coordinates here.

        cls.molsquish = gto.M(atom="H 0 0 0; F 0 0 0", basis=basis)
        molsquish1 = gto.M(atom="H 0 0 0; F 0 0 0", basis=basis)

        # cls.molsquish = gto.M(atom='Ne', basis=basis)
        # molsquish1 = gto.M(atom='Ne', basis=basis)

        molsquish1._env[molsquish1._bas[:, PTR_COEFF]] *= (
            2 * molsquish1._env[molsquish1._bas[:, PTR_EXP]]
        )
        mol1 = gto.M(atom=molstr, basis=basis)
        mol1._env[mol1._bas[:, PTR_COEFF]] *= 2 * mol1._env[mol1._bas[:, PTR_EXP]]
        cls.molsquish1 = molsquish1
        cls.mol1 = mol1
        dat = get_gamma_lists_from_mol(cls.mol)
        cls.atco = ATCBasis(*dat)
        cls.n0 = 5
        cls.n1 = 2

        # initialize the interpolator
        cls.interpolator = LCAOInterpolator(
            cls.mol.atom_coords(unit="Bohr"),
            cls.atco,
            cls.n0,
            cls.n1,
            dparam=0.02,
            nrad=400,
        )

    def get_true_values(self, coords, f_uq, mol, mol1, byatom=True):
        atco = self.atco
        interpolator = self.interpolator

        atom_loc = gto.mole.aoslice_by_atom(mol)
        atom_loc = np.ascontiguousarray(
            np.append(atom_loc[:, 2], [atom_loc[-1, 3]])
        ).astype(np.int32)
        atom_coords = mol.atom_coords(unit="Bohr")
        ao = eval_ao(mol, coords, deriv=0)
        dao = eval_ao(mol, coords, deriv=1)[1:4]
        ao1 = eval_ao(mol1, coords, deriv=0)
        ao1 = np.ascontiguousarray(np.stack([ao1, ao1, ao1]))
        for ia in range(atco.natm):
            ao1[:, :, atom_loc[ia] : atom_loc[ia + 1]] *= (coords - atom_coords[ia]).T[
                :, :, None
            ]
        ao1 += dao  # + ao1 * coords.T[:, :, None]
        # construct true values at the selected coordinates
        n0 = self.n0
        n1 = self.n1
        if byatom:
            shape = (atco.natm, coords.shape[0])
        else:
            shape = (coords.shape[0],)
        ftrue_argq = np.zeros(shape + (n0,))
        f1mtrue_argq = np.zeros(shape + (n1, 3))
        f1true_argq = np.zeros(shape + (n1,))
        fcheck_uq = f_uq.copy()
        ao_loc = mol.ao_loc_nr()
        for shl in range(mol.nbas):
            if mol._bas[shl, ANG_OF] == 1:
                # pyscf orders l=1 spherical harmonics differently
                toset = fcheck_uq[ao_loc[shl] : ao_loc[shl] + 3].copy()
                toset = toset[[2, 0, 1]]
                fcheck_uq[ao_loc[shl] : ao_loc[shl] + 3] = toset
        if byatom:
            for ia in range(atco.natm):
                ftrue_argq[ia] = np.einsum(
                    "uq,gu->gq",
                    fcheck_uq[atom_loc[ia] : atom_loc[ia + 1], : interpolator._n0],
                    ao[:, atom_loc[ia] : atom_loc[ia + 1]],
                )
                f1true_argq[ia] = np.einsum(
                    "uq,gu->gq",
                    fcheck_uq[atom_loc[ia] : atom_loc[ia + 1], -interpolator._n1 :],
                    ao[:, atom_loc[ia] : atom_loc[ia + 1]],
                )
                f1mtrue_argq[ia] = np.einsum(
                    "uq,xgu->gqx",
                    fcheck_uq[
                        atom_loc[ia] : atom_loc[ia + 1],
                        -2 * interpolator._n1 : -interpolator._n1,
                    ],
                    ao1[:, :, atom_loc[ia] : atom_loc[ia + 1]],
                )
        else:
            ftrue_argq = np.einsum("uq,gu->gq", fcheck_uq[:, : interpolator._n0], ao)
            f1mtrue_argq = np.einsum(
                "uq,xgu->gqx",
                fcheck_uq[:, -2 * interpolator._n1 : -interpolator._n1],
                ao1,
            )
            for ia in range(atco.natm):
                f1true_argq = np.einsum(
                    "uq,gu->gq",
                    fcheck_uq[atom_loc[ia] : atom_loc[ia + 1], -interpolator._n1 :],
                    ao[:, atom_loc[ia] : atom_loc[ia + 1]],
                )
                f1mtrue_argq[:] += (
                    f1true_argq[..., None] * (coords - atom_coords[ia])[:, None, :]
                )
        return ftrue_argq, f1true_argq, f1mtrue_argq

    def test_conv2spline(self):
        atco = self.atco
        interpolator = self.interpolator

        # check that splines have the right values
        bas, env = atco.bas, atco.env
        ls = bas[:, ANG_OF]
        coefs = env[bas[:, PTR_COEFF]]
        exps = env[bas[:, PTR_EXP]]
        rs = interpolator.rads[:, None]
        w_rs = rs**ls * coefs * np.exp(-exps * rs**2)
        assert_almost_equal(interpolator.w0_rsp[..., 0], w_rs)
        bas, env = interpolator.l1atco.bas, interpolator.l1atco.env
        ls = bas[:, ANG_OF]
        coefs = env[bas[:, PTR_COEFF]]
        exps = env[bas[:, PTR_EXP]]
        rs = interpolator.rads[:, None]
        w_rs = rs**ls * coefs * np.exp(-exps * rs**2)
        assert_almost_equal(interpolator.wm_rsp[..., 0], w_rs)

        def get_e_and_vfeat(f_arlpq):
            wts = (
                interpolator.aparam
                * interpolator.dparam
                * np.exp(interpolator.dparam * np.arange(interpolator.nrad))
            )
            wts *= np.exp(-0.3 * interpolator.rads) / 1e4
            shape = f_arlpq.shape
            energy = (
                (f_arlpq**2 * wts[:, None, None, None]).reshape(-1, shape[-1]).sum()
            )
            potential = 2 * f_arlpq * wts[:, None, None, None]
            return (energy, potential)

        np.random.seed(13)
        f_uq = np.random.normal(size=(atco.nao, interpolator.num_in))
        fpred_arlpq = interpolator.conv2spline(f_uq)
        energy, vfpred_arlpq = get_e_and_vfeat(fpred_arlpq)
        vf_uq = interpolator.spline2conv(vfpred_arlpq)
        for q in range(f_uq.shape[1]):
            # TODO numbers in this test are huge, could use a better fd test
            df_uq = np.zeros_like(f_uq)
            df_uq[:, q] = 1e-8 * np.random.normal(size=atco.nao)
            fperturb_pred_arlpq = interpolator.conv2spline(f_uq + df_uq)
            energy_perturb, _ = get_e_and_vfeat(fperturb_pred_arlpq)
            fd_deriv = energy_perturb - energy
            assert_allclose(np.sum(vf_uq * df_uq), fd_deriv, rtol=2e-4)
        n = 194
        grid = numpy.empty((n, 4))
        libdft.MakeAngularGrid(grid.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n))
        ang_coords = np.ascontiguousarray(grid[:, :3])
        coords = np.einsum("ix,j->jix", ang_coords, interpolator.rads).reshape(-1, 3)

        ylm_gl = np.zeros((n, interpolator.nlm))
        libcider.recursive_sph_harm_vec(
            ctypes.c_int(interpolator.nlm),
            ctypes.c_int(n),
            ang_coords.ctypes.data_as(ctypes.c_void_p),
            ylm_gl.ctypes.data_as(ctypes.c_void_p),
        )
        fpred_argq = np.einsum(
            "arlq,gl->argq",
            fpred_arlpq[:, :, :, 0, :],
            ylm_gl,
        ).reshape(atco.natm, -1, interpolator.num_out)

        for r in range(len(interpolator.rads)):
            rmin, rmax = r, r + 1
            gmin, gmax = rmin * n, rmax * n
            ftrue_argq, f1true_argq, f1mtrue_argq = self.get_true_values(
                coords[gmin:gmax], f_uq, self.molsquish, self.molsquish1, byatom=True
            )
            assert_almost_equal(
                fpred_argq[:, gmin:gmax, ..., : interpolator._n0], ftrue_argq
            )
            assert_almost_equal(
                fpred_argq[:, gmin:gmax, ..., -interpolator._n1 :], f1true_argq
            )
            assert_almost_equal(
                fpred_argq[:, gmin:gmax, ..., interpolator._n0 : -interpolator._n1],
                f1mtrue_argq.reshape(
                    fpred_argq.shape[0], gmax - gmin, 3 * interpolator._n1
                ),
            )

    def test_interpolate(self):
        mol = self.mol
        atco = self.atco
        interpolator = self.interpolator

        np.random.seed(13)
        f_uq = np.random.normal(size=(atco.nao, interpolator.num_in))

        f_arlpq = interpolator.conv2spline(f_uq)

        grids = Grids(mol)
        grids.level = 0
        grids.build()
        coords = grids.coords

        wt = grids.weights
        wt[np.isnan(wt)] = 0
        interpolator.set_coords(coords)
        fdirect_gq = interpolator.project_orb2grid(f_uq)

        def get_e_and_vfeat(ff_gq):
            shape = ff_gq.shape
            ff_gq = ff_gq.copy()
            ff_gq[:, -interpolator._n1 :] = 0.0
            energy = (ff_gq**2 * wt[:, None]).reshape(-1, shape[-1]).sum()
            potential = 2 * ff_gq * wt[:, None]
            potential[:, -interpolator._n1] = 0.0
            return (energy, potential)

        f0_gq = interpolator.interpolate_fwd(f_arlpq)
        f_gq = np.zeros_like(f0_gq)
        f_gq = interpolator.interpolate_fwd(f_arlpq, f_gq=f_gq)
        energy, potential = get_e_and_vfeat(f_gq)
        potential2 = potential.copy()
        vf_arlpq = interpolator.interpolate_bwd(potential)
        assert_almost_equal(f_gq, f0_gq, 14)
        assert_almost_equal(fdirect_gq, f_gq, 14)
        vf_uq = interpolator.spline2conv(vf_arlpq)
        vfdirect_uq = interpolator.project_grid2orb(potential2)
        assert_almost_equal(vfdirect_uq, vf_uq, 14)

        for q in range(f_arlpq.shape[-1]):
            # TODO could use a cleaner fd test
            df_arlpq = np.zeros_like(f_arlpq)
            df_arlpq[..., q] = 1e-8 * np.random.normal(size=f_arlpq.shape[:-1])
            fperturb_gq = interpolator.interpolate_fwd(f_arlpq + df_arlpq)
            energy_perturb, _ = get_e_and_vfeat(fperturb_gq)
            fd_deriv = energy_perturb - energy
            assert_allclose(np.sum(vf_arlpq * df_arlpq), fd_deriv, rtol=2e-3)

        ftrue_gq, f1true_gq, f1mtrue_gq = self.get_true_values(
            coords, f_uq, self.mol, self.mol1, byatom=False
        )
        f1mtrue_gq = f1mtrue_gq.reshape(f_gq.shape[0], 3 * interpolator._n1)
        ffull_gq = np.concatenate([ftrue_gq, f1mtrue_gq, f1true_gq], axis=1)
        energy_true, potential_true = get_e_and_vfeat(ffull_gq)
        assert_almost_equal(energy, energy_true, 2)
        atol = 1e-2
        rtol = 1e-2
        atol_small = 1e-5
        rtol_small = 1e-5

        assert not np.isnan(f_gq).any()

        assert_allclose(
            f_gq[..., : interpolator._n0].T,
            ftrue_gq.T,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            (f_gq[..., : interpolator._n0].T * wt).sum(axis=1),
            (ftrue_gq.T * wt).sum(axis=1),
            atol=atol_small,
            rtol=rtol_small,
        )

        # this test won't pass because last few indices are zeroed
        """
        assert_allclose(
            f_gq[..., -interpolator._n1:].T,
            f1true_gq.T,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            (f_gq[..., -interpolator._n1:].T * wt).sum(axis=1),
            (f1true_gq.T * wt).sum(axis=1),
            atol=atol_small,
            rtol=rtol_small,
        )"""

        assert_allclose(
            f_gq[..., interpolator._n0 : -interpolator._n1].T,
            f1mtrue_gq.T,
            atol=atol,
            rtol=rtol,
        )
        assert_allclose(
            (f_gq[..., interpolator._n0 : -interpolator._n1].T * wt).sum(axis=1),
            (f1mtrue_gq.T * wt).sum(axis=1),
            atol=1e-3,
            rtol=rtol_small,
        )

    def test_equivalence(self):
        mol = self.mol
        atco = self.atco
        interpolator0 = self.interpolator

        np.random.seed(13)
        f_uq = np.random.normal(size=(atco.nao, interpolator0.num_in))
        grids = CiderGrids(mol)
        grids.atom_grid = {"Ne": (10, 194), "H": (8, 194), "F": (12, 194)}
        grids.prune = None
        grids.build()
        coords = grids.coords
        indexer = grids.grids_indexer

        interpolator1 = LCAOInterpolatorDirect(
            indexer,
            interpolator0.atom_coords,
            atco,
            self.n0,
            self.n1,
            onsite_direct=False,
            aparam=interpolator0.aparam,
            dparam=interpolator0.dparam,
            nrad=interpolator0.nrad,
        )
        interpolator2 = LCAOInterpolatorDirect(
            indexer,
            interpolator0.atom_coords,
            atco,
            self.n0,
            self.n1,
            onsite_direct=True,
            aparam=interpolator0.aparam,
            dparam=interpolator0.dparam,
            nrad=interpolator0.nrad,
        )

        wt = grids.weights
        wt[np.isnan(wt)] = 0
        interpolator0.set_coords(coords)
        interpolator1.set_coords(coords)
        interpolator2.set_coords(coords)
        f0_gq = interpolator0.project_orb2grid(f_uq)
        f1_gq = interpolator1.project_orb2grid(f_uq)
        f2_gq = interpolator2.project_orb2grid(f_uq)
        ngnpp = interpolator1.grids_indexer.idx_map.size
        ftrue_gq, f1true_gq, f1mtrue_gq = self.get_true_values(
            coords, f_uq, self.mol, self.mol1, byatom=False
        )
        f1mtrue_gq = f1mtrue_gq.reshape(f0_gq.shape[0], 3 * interpolator0._n1)
        ffull_gq = np.concatenate([ftrue_gq, f1mtrue_gq, f1true_gq], axis=1)

        f0_gq[ngnpp:] = 0.0
        ffull_gq[ngnpp:] = 0.0
        assert_almost_equal(f1_gq, f0_gq)
        for q in range(f2_gq.shape[1] - interpolator0._n1):
            assert_allclose(
                f2_gq[:, q],
                f0_gq[:, q],
                atol=1e-2,
                rtol=1e-2,
            )
            assert_allclose(
                f0_gq[:, q],
                ffull_gq[:, q],
                atol=1e-2,
                rtol=1e-2,
            )
            assert_allclose(
                f2_gq[:, q],
                ffull_gq[:, q],
                atol=1e-4,
                rtol=1e-4,
            )

        def get_e_and_vfeat(ff_gq):
            shape = ff_gq.shape
            ff_gq = ff_gq.copy()
            if ff_gq.ndim == 2:
                ff_gq[:, -interpolator0._n1 :] = 0.0
                energy = (ff_gq**2 * wt[:, None]).reshape(-1, shape[-1]).sum()
                potential = 2 * ff_gq * wt[:, None]
                potential[:, -interpolator0._n1] = 0.0
            else:
                energy = (ff_gq**2 * wt).sum()
                potential = 2 * ff_gq * wt
            return (energy, potential)

        energy, potential = get_e_and_vfeat(ffull_gq)
        energy0, potential0 = get_e_and_vfeat(f0_gq)
        energy1, potential1 = get_e_and_vfeat(f1_gq)
        energy2, potential2 = get_e_and_vfeat(f2_gq)
        assert_almost_equal(energy0, energy, 1)
        assert_almost_equal(energy1, energy, 1)
        assert_almost_equal(energy2, energy, 2)
        assert_almost_equal(potential0, potential, 5)
        assert_almost_equal(potential1, potential, 5)
        assert_almost_equal(potential2, potential, 5)

        f0_uq = interpolator0.project_grid2orb(potential)
        f1_uq = interpolator1.project_grid2orb(potential)
        f2_uq = interpolator2.project_grid2orb(potential)
        assert_almost_equal(f1_uq, f0_uq)
        assert_almost_equal(f2_uq, f0_uq, 3)


if __name__ == "__main__":
    unittest.main()
