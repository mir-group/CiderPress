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

import numpy as np
from pyscf import gto
from pyscf.dft.gen_grid import (
    LEBEDEV_NGRID,
    LEBEDEV_ORDER,
    NELEC_ERROR_TOL,
    Grids,
    _default_ang,
    _default_rad,
    _padding_size,
    arg_group_grids,
    libdft,
    logger,
    nwchem_prune,
    radi,
)

from ciderpress.dft.grids_indexer import AtomicGridsIndexer, libcider

CIDER_DEFAULT_LMAX = 10


class CiderGrids(Grids):
    """
    Attributes (in addition to those in pyscf Grids object):
        lmax (int): Maximum l value for integration of CIDER features
        nlm (int): (lmax + 1)**2
        grids_indexer (AtomicGridsIndexer): object to sort and index
            grids for CIDER features.
    """

    grids_indexer: AtomicGridsIndexer

    _keys = {"grids_indexer", "nlm", "lmax"}

    def __init__(self, mol, lmax=CIDER_DEFAULT_LMAX):
        super(CiderGrids, self).__init__(mol)
        self.lmax = lmax
        self.nlm = (lmax + 1) * (lmax + 1)
        self.grids_indexer = None

    def gen_atomic_grids(
        self,
        mol,
        atom_grid=None,
        radi_method=None,
        level=None,
        prune=None,
        build_indexer=False,
        **kwargs
    ):
        """
        Same as gen_atomic_grids for PySCF grids, except it also
        uses the outputs of gen_atomic_grids_cider to initialize the
        grids_indexer object for CIDER feature routines.
        """
        if atom_grid is None:
            atom_grid = self.atom_grid
        if radi_method is None:
            radi_method = self.radi_method
        if level is None:
            level = self.level
        if prune is None:
            prune = self.prune
        (
            atom_grids_tab,
            lmax_tab,
            rad_loc_tab,
            ylm_tab,
            ylm_loc_tab,
            rad_tab,
            dr_tab,
        ) = gen_atomic_grids_cider(
            mol, atom_grid, self.radi_method, level, prune, **kwargs
        )
        if build_indexer:
            self.grids_indexer = AtomicGridsIndexer.from_tabs(
                mol, self.lmax, rad_loc_tab, ylm_loc_tab, rad_tab, ylm_tab
            )
        return atom_grids_tab

    def build(self, mol=None, with_non0tab=False, sort_grids=True, **kwargs):
        """
        Build the grids. Same as with PySCF grids, but also stores an idx_map
        in the grids_indexer object so that one can map between the atomic grids
        and sorted grids.
        """
        if mol is None:
            mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        atom_grids_tab = self.gen_atomic_grids(
            mol,
            self.atom_grid,
            self.radi_method,
            self.level,
            self.prune,
            build_indexer=True,
            **kwargs
        )
        self.coords, self.weights = self.get_partition(
            mol, atom_grids_tab, self.radii_adjust, self.atomic_radii, self.becke_scheme
        )
        self.grids_indexer.set_weights(self.weights)

        if sort_grids:
            idx = arg_group_grids(mol, self.coords)
            self.coords = self.coords[idx]
            self.weights = self.weights[idx]
            self.grids_indexer.set_idx(idx)
        else:
            self.grids_indexer.set_idx(np.arange(self.weights.size))

        if self.alignment > 1:
            padding = _padding_size(self.size, self.alignment)
            logger.debug(self, "Padding %d grids", padding)
            if padding > 0:
                self.coords = np.vstack(
                    [self.coords, np.repeat([[1e-4] * 3], padding, axis=0)]
                )
                self.weights = np.hstack([self.weights, np.zeros(padding)])
                self.grids_indexer.set_padding(padding)

        self.coords = np.asarray(self.coords, order="C")
        self.weights = np.asarray(self.weights, order="C")

        if with_non0tab:
            self.non0tab = self.make_mask(mol, self.coords)
            self.screen_index = self.non0tab
        else:
            self.screen_index = self.non0tab = None
        logger.info(self, "tot grids = %d", len(self.weights))
        return self

    def reset(self, mol=None):
        self.grids_indexer = None
        return super().reset(mol=mol)

    def prune_by_density_(self, rho, threshold=0):
        """
        Prune grids if the electron density on the grid is small.
        Only difference from PySCF version is that grids_indexer.set_idx
        is called to make sure the map between unsorted and sorted/screened
        grids is correct.
        """
        if threshold == 0:
            return self

        mol = self.mol
        n = np.dot(rho, self.weights)
        if abs(n - mol.nelectron) < NELEC_ERROR_TOL * n:
            rho *= self.weights
            idx = abs(rho) > threshold / self.weights.size
            logger.debug(
                self, "Drop grids %d", self.weights.size - np.count_nonzero(idx)
            )
            self.coords = np.asarray(self.coords[idx], order="C")
            self.weights = np.asarray(self.weights[idx], order="C")
            old_idx = self.grids_indexer.get_idx()
            old_idx_size = old_idx.size
            self.grids_indexer.set_idx(old_idx[idx[:old_idx_size]])
            if self.alignment > 1:
                padding = _padding_size(self.size, self.alignment)
                logger.debug(self, "prune_by_density_: %d padding grids", padding)
                if padding > 0:
                    self.coords = np.vstack(
                        [self.coords, np.repeat([[1e-4] * 3], padding, axis=0)]
                    )
                    self.weights = np.hstack([self.weights, np.zeros(padding)])
                self.grids_indexer.set_padding(padding)
            self.non0tab = self.make_mask(mol, self.coords)
            self.screen_index = self.non0tab
            self.coords = np.asarray(self.coords, order="C")
            self.weights = np.asarray(self.weights, order="C")
        return self


LMAX_DICT = {v: k // 2 for k, v in LEBEDEV_ORDER.items()}


def gen_atomic_grids_cider(
    mol,
    atom_grid=None,
    radi_method=radi.gauss_chebyshev,
    level=3,
    prune=nwchem_prune,
    full_lmax=CIDER_DEFAULT_LMAX,
    **kwargs
):
    if atom_grid is None:
        atom_grid = {}
    if isinstance(atom_grid, (list, tuple)):
        atom_grid = dict([(mol.atom_symbol(ia), atom_grid) for ia in range(mol.natm)])
    atom_grids_tab = {}
    lmax_tab = {}
    rad_loc_tab = {}
    ylm_tab = {}
    ylm_loc_tab = {}
    rad_tab = {}
    dr_tab = {}
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)

        if symb not in atom_grids_tab:
            chg = gto.charge(symb)
            if symb in atom_grid:
                n_rad, n_ang = atom_grid[symb]
                if n_ang not in LEBEDEV_NGRID:
                    raise ValueError("Unsupported angular grids %d" % n_ang)
            else:
                n_rad = _default_rad(chg, level)
                n_ang = _default_ang(chg, level)
            rad, dr = radi_method(n_rad, chg, ia, **kwargs)

            rad_weight = 4 * np.pi * rad**2 * dr

            if callable(prune):
                angs = prune(chg, rad, n_ang)
            else:
                angs = [n_ang] * n_rad
            logger.debug(
                mol, "atom %s rad-grids = %d, ang-grids = %s", symb, n_rad, angs
            )

            angs = np.array(angs)
            coords = []
            vol = []
            rad_loc = np.array([0], dtype=np.int32)
            lmaxs = np.array([LMAX_DICT[ang] for ang in angs]).astype(np.int32)
            lmaxs = np.minimum(lmaxs, full_lmax)
            nlm = (full_lmax + 1) * (full_lmax + 1)
            ylm_full = np.empty((0, nlm), order="C")
            ylm_loc = []
            rads = []
            drs = []
            for n in sorted(set(angs)):
                grid = np.empty((n, 4))
                libdft.MakeAngularGrid(
                    grid.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n)
                )
                idx = np.where(angs == n)[0]
                yloc_curr = ylm_full.shape[0]
                ylm = np.zeros((n, nlm), order="C")
                sphgd = np.ascontiguousarray(grid[:, :3])
                libcider.recursive_sph_harm_vec(
                    ctypes.c_int(nlm),
                    ctypes.c_int(n),
                    sphgd.ctypes.data_as(ctypes.c_void_p),
                    ylm.ctypes.data_as(ctypes.c_void_p),
                )
                lmax_shl = LMAX_DICT[n]
                nlm_shl = (lmax_shl + 1) * (lmax_shl + 1)
                ylm[:, nlm_shl:] = 0.0
                ylm_full = np.append(ylm_full, ylm, axis=0)
                coords.append(
                    np.einsum("i,jk->ijk", rad[idx], grid[:, :3]).reshape(-1, 3)
                )
                vol.append(np.einsum("i,j->ij", rad_weight[idx], grid[:, 3]).ravel())
                rads.append(rad[idx])
                drs.append(dr[idx])
                rad_loc = np.append(
                    rad_loc, rad_loc[-1] + n * np.arange(1, len(idx) + 1)
                )
                ylm_loc.append(yloc_curr * np.ones(idx.size, dtype=np.int32))
            atom_grids_tab[symb] = (np.vstack(coords), np.hstack(vol))
            lmax_tab[symb] = lmaxs
            rad_loc_tab[symb] = rad_loc
            ylm_tab[symb] = ylm_full
            ylm_loc_tab[symb] = np.concatenate(ylm_loc).astype(np.int32)
            rad_tab[symb] = np.concatenate(rads).astype(np.float64)
            dr_tab[symb] = np.concatenate(drs).astype(np.float64)

    return atom_grids_tab, lmax_tab, rad_loc_tab, ylm_tab, ylm_loc_tab, rad_tab, dr_tab
