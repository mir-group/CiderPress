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

import numpy as np
from pyscf import gto
from pyscf.dft.gen_grid import Grids

from ciderpress.pyscf.gen_cider_grid import CiderGrids


def get_scaled_mol(mol, lambd):
    """
    Create a scaled version of PySCF Mole.
    Useful for testing uniform scaling behavior

    Args:
        mol (pyscf.gto.Mole): molecule obect. Should
            be build and have basis set.
        lambd (float): Uniform scaling factor

    Returns:
        new_mol (pyscf.gto.Mole): Scaled Mole
    """
    new_basis = {}
    lambd2 = lambd**2
    lambd32 = lambd**1.5
    for k, v in mol._basis.items():
        bas = []
        for cg in v:  # cg = [l, [alpha, c1, c2...], [alpha, c1, ...]]
            l, coeff_lists = cg[0], cg[1:]
            new_cg = [l]
            for ac in coeff_lists:
                new_cg.append([ac[0] * lambd2])
                for c in ac[1:]:
                    new_cg[-1].append(c * lambd32)
            bas.append(new_cg)
        new_basis[k] = bas
    atom = mol._atom
    new_atom = []
    for at in atom:
        new_atom.append((at[0], [c / lambd for c in at[1]]))
    return gto.M(
        atom=new_atom,
        basis=new_basis,
        unit="Bohr",
        spin=mol.spin,
        charge=mol.charge,
    )


def get_scaled_grid(grids, lambd):
    """
    Create a scaled version of PySCF grids.
    Useful for testing uniform scaling behavior

    Args:
        grids (pyscf.dft.gen_grid.Grids or CiderGrids): Original Grid
        lambd (float): Uniform scaling factor

    Returns:
        new_grids: Same type as grids, mol and grids
            are scaled by lambd
    """
    new_mol = get_scaled_mol(grids.mol, lambd)
    cls = type(grids)
    if cls == CiderGrids:
        new_grids = cls(mol=new_mol, lmax=grids.lmax)
    else:
        new_grids = cls(mol=new_mol)
    new_grids.weights = grids.weights / lambd**3
    new_grids.coords = grids.coords / lambd
    if cls == CiderGrids:
        new_grids.gen_atomic_grids(
            grids.mol,
            grids.atom_grid,
            grids.radi_method,
            grids.level,
            grids.prune,
            build_indexer=True,
        )
        new_grids.grids_indexer.rad_arr /= lambd
        new_grids.grids_indexer.all_weights = (
            grids.grids_indexer.all_weights / lambd**3
        )
        new_grids.grids_indexer.idx_map = grids.grids_indexer.idx_map
        if hasattr(new_grids.grids_indexer, "all_coords"):
            new_grids.grids_indexer.all_coords /= lambd
    return new_grids


def rotate_molecule(mol, theta, axis="z"):
    mol2 = mol.copy()
    coords = mol2.atom_coords(unit=mol2.unit)
    new_coords = get_rotated_coords(coords, theta, axis)
    mol2.set_geom_(new_coords, unit=mol2.unit)
    return mol2


def get_rotated_coords(coords, theta, axis="z"):
    cost, sint = np.cos(theta), np.sin(theta)
    if axis == "x":
        R = np.array([[1, 0, 0], [0, cost, -sint], [0, sint, cost]])
    elif axis == "y":
        R = np.array([[cost, 0, sint], [0, 1, 0], [-sint, 0, cost]])
    elif axis == "z":
        R = np.array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])
    else:
        raise ValueError
    return coords.dot(R.T)


def get_random_coords(ks):
    ref_grids = Grids(ks.mol)
    ref_grids.level = 0
    ref_grids.build()
    dm = ks.make_rdm1()
    if ks.mol.spin == 0:
        rho = ks._numint.get_rho(ks.mol, dm, ref_grids)
        cond = rho > 1e-2
    else:
        rhoa = ks._numint.get_rho(ks.mol, dm[0], ref_grids)
        rhob = ks._numint.get_rho(ks.mol, dm[1], ref_grids)
        cond = np.logical_and(rhoa > 1e-2, rhob > 1e-2)
    coords = ref_grids.coords[cond]
    inds = np.arange(coords.shape[0])
    np.random.seed(16)
    np.random.shuffle(inds)
    return np.ascontiguousarray(coords[inds[:50]])
