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

from ciderpress.lib import load_library

libcider = load_library("libmcider")


class AtomicGridsIndexer:
    """
    This class is used to map between atomic grids ordering (which is used
    to integrate the theta functions for CIDER features) and sorted
    grids ordering (which is used to integrate the XC in PySCF).

    It also stores a variety of indexing arrays for performing integrals
    on individual atoms.

    Attributes:
        natm (int): number of atoms
        lmax (int): maximum l value for integration
        nlm (int): (lmax + 1)**2
        rad_arr (np.ndarray(np.float64), shape (nrad,)):
            Array of unique radial coordinates on the grid, sorted by atom.
        ar_loc (np.ndarray(np.int32), shape (nrad + 1,)):
            For each unique radial coordinate, this is the atom on which
            it is centered. I.e. the atom of rad_arr[r] is ar_loc[r]
        ra_loc (np.nddarray(np.int32), shape (natm + 1,)):
            For each atom a, ra_loc[a] is the first index in rad_arr that
            belongs to atom a, and ra_loc[a+1] - 1 is the last index that
            belongs to atom a.
        rad_loc (np.ndarray(np.int32), shape (nrad + 1,)):
            Specifies the range of indexes with a given radial coordinate.
            I.e. all_coords[rad_loc[r] : rad_loc[r+1]] have radial coordinate
            rad_arr[r].
        ylm (np.ndarray(np.float64), shape (num_ang_coord, nlm)):
            The stored spherical harmonics centered on each atom. Shape is
            the total number of unique angular coordinates X the number
            of spherical harmonics per coordinate (lmax + 1)**2.
        ylm_loc (np.ndarray(np.int32), shape (num_ang_coord,)):
            The spherical harmonics on all_coords[rad_loc[r] : rad_loc[r + 1]]
            are ylm[ylm_loc[r]]
        dirs (np.ndarray(np.float64), shape (num_ang_coord, 3)):
            Same as ylm, but just contains (x, y, z) instead of the spherical
            harmonics. Can be indexed by ylm_loc just like ylm
        all_weights (np.ndarray(np.float64)), shape (rad_loc[-1],)):
            All weights for the original, unsorted, unpruned grids, ordered
            by atom, then radial coordinate, then angular coordinate.
            Single flattened array for convenience.
        idx_map (np.ndarray(np.int32), shape (grids.weights.size,)):
            Indexing for map from a sorted/pruned grid to the unsorted,
            unpruned, atom-ordered grid.
            I.e. weights = all_weights[idx_map]
                 coords = all_coords[idx_map]
    """

    def __init__(self, natm, lmax, rad_arr, ar_loc, ra_loc, rad_loc, ylm, ylm_loc):
        """
        Initialize AtomicGridsIndexer. See attributes for argument details.

        Args:
            natm (int): Number of atoms
            lmax (int): Maximum l value
            rad_arr (np.ndarray): Radial coordinates
            ar_loc (np.ndarray): First radial coordinate index for each atom
            ra_loc (np.ndarray): Atom id of each radial coordinate
            rad_loc (np.ndarray): Location of each radial coordinate on
                the full grids.
            ylm (np.ndarray): Spherical harmonics
            ylm_loc (np.ndarray): Location of spherical harmonics for each
                radial coordinate.
        """
        self.natm = natm
        self.lmax = lmax
        self.nlm = (lmax + 1) * (lmax + 1)
        assert self.nlm == ylm.shape[1]
        self.rad_arr = rad_arr
        self.ar_loc = ar_loc
        self.ra_loc = ra_loc
        self.rad_loc = rad_loc
        self.ylm = ylm
        self.dirs = np.ascontiguousarray(ylm[:, [3, 1, 2]] * np.sqrt(4 * np.pi / 3))
        self.ylm_loc = ylm_loc
        self.all_weights = None
        self.idx_map = None
        self.iatom_list = None
        self.ga_loc = self.rad_loc[self.ra_loc]
        self.padding = 0

    @property
    def nrad(self):
        return self.rad_arr.size

    @property
    def ngrids(self):
        if self.all_weights is None:
            raise RuntimeError("weights must be set to get ngrids")
        return self.all_weights.size

    @classmethod
    def from_tabs(cls, mol, lmax, rad_loc_tab, ylm_loc_tab, rad_tab, ylm_tab):
        """
        Initialize AtomicGridsIndexer from tabulated elemental loc arrays.
        See Attributes for more details on what each component means.

        Args:
            mol (pyscf.gto.Mole): Molecule object
            lmax (int): Maximum l value
            rad_loc_tab (dict): Dictionary containing rad_loc array
                for each element in mol.
            ylm_loc_tab (dict): Dictionary containing ylm_loc array
                for each element in mol.
            rad_tab (dict): Dictionary containing array of radial
                coordinates for each element.
            ylm_tab (dict): Dictionary containing array of spherical
                harmonics for each element.

        Returns:
            AtomicGridsIndexer object
        """
        nlm = (lmax + 1) * (lmax + 1)
        full_rad_loc = np.array([0], dtype=np.int32)
        ar_loc = []
        ra_loc = [0]
        rads = []
        full_ylm_loc = np.array([], dtype=np.int32)
        full_ylm = np.empty((0, nlm), dtype=np.float64)
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            nrad = rad_loc_tab[symb].size - 1
            start = full_rad_loc[-1]
            full_rad_loc = np.append(full_rad_loc, rad_loc_tab[symb][1:] + start)
            full_ylm_loc = np.append(
                full_ylm_loc, ylm_loc_tab[symb] + full_ylm.shape[0]
            )
            full_ylm = np.append(full_ylm, ylm_tab[symb], axis=0)
            rads.append(rad_tab[symb])
            rad_loc = full_rad_loc.size - 1
            ra_loc.append(rad_loc)
            ar_loc.append(ia * np.ones(nrad, dtype=np.int32))
        return cls(
            mol.natm,
            lmax,
            rad_arr=np.ascontiguousarray(np.concatenate(rads).astype(np.float64)),
            ar_loc=np.ascontiguousarray(np.concatenate(ar_loc).astype(np.int32)),
            ra_loc=np.array(ra_loc, dtype=np.int32, order="C"),
            rad_loc=np.ascontiguousarray(full_rad_loc.astype(np.int32)),
            ylm=np.ascontiguousarray(full_ylm.astype(np.float64)),
            ylm_loc=np.ascontiguousarray(full_ylm_loc.astype(np.int32)),
        )

    def set_padding(self, padding):
        """Padding is how much bigger target grid is than the grid
        constructed from all_coords[idx]"""
        self.padding = padding

    def set_weights(self, weights):
        """
        Set the atom-ordered, unsorted, unscreened weights.
        """
        self.all_weights = np.asarray(weights, order="C")

    def set_idx(self, idx):
        """
        Set the idx map between the atom-ordered grids and sorted grids.
        atom_ordered_grids[get_idx()] = sorted_screened_grids
        """
        self.idx_map = np.asarray(idx, order="C")
        tmp = np.arange(self.all_weights.size)
        for a in range(self.natm):
            tmp[self.ga_loc[a] : self.ga_loc[a + 1]] = a
        self.iatom_list = np.asarray(tmp[self.idx_map], order="C", dtype=np.int32)

    def get_idx(self):
        """
        Get the idx map between the atom-ordered grids and sorted grids.
        atom_ordered_grids[get_idx()] = sorted_screened_grids
        """
        return self.idx_map

    def empty_rlmq(self, nalpha=1, nspin=None):
        """
        Get an empty array for storying theta_rlmq.

        Args:
            nalpha (int): Number of interpolation points for CIDER kernel.

        Returns:
            np.ndarray with shape (nrad, nlm, nalpha)
        """
        if nspin is None:
            shape = (self.nrad, self.nlm, nalpha)
        else:
            shape = (nspin, self.nrad, self.nlm, nalpha)
        return np.empty(shape, order="C", dtype=np.float64)

    def empty_gq(self, nalpha=1, nspin=None):
        """
        Get an empty array for storing theta_gq.

        Args:
            nalpha (int): Number of interpolation points for CIDER kernel.

        Returns:
            np.ndarray with shape (ngrids, nalpha)
        """
        if nspin is None:
            shape = (self.ngrids, nalpha)
        else:
            shape = (nspin, self.ngrids, nalpha)
        return np.empty(shape, dtype=np.float64)

    def reduce_angc_ylm_(self, theta_rlmq, theta_gq, a2y=True, offset=None):
        """
        Project a function between angular coordinates and spherical harmonics.
        NOTE that this is an in-place operation (with the output being
        theta_rlmq if a2y=True and theta_gq if a2y=False), but the data
        in the output array does not need to be initialized, as it is
        overwritten on output, not added to.

        Args:
            theta_rlmq (np.ndarray): Shape (len(rad_arr), nlm, nalpha)
                Theta projected onto radial coordinates and spherical harmonics.
            theta_gq (np.ndarray): Shape (ngrids, nalpha)
            a2y (bool): If True, project real-space to spherical harmonics.
                If False, project spherical harmonics onto real-space coords.
        """
        if offset is None:
            offset = 0
        nalpha = theta_rlmq.shape[-1]
        stride = theta_gq.shape[-1]
        assert theta_rlmq.flags.c_contiguous
        assert theta_rlmq.shape == (self.nrad, self.nlm, nalpha)
        assert theta_gq.flags.c_contiguous
        assert theta_gq.shape == (self.all_weights.size, stride)
        assert theta_rlmq.dtype == np.float64
        assert theta_gq.dtype == np.float64
        assert nalpha + offset <= stride
        if a2y:
            fn = libcider.reduce_angc_to_ylm
        else:
            fn = libcider.reduce_ylm_to_angc
        fn(
            theta_rlmq.ctypes.data_as(ctypes.c_void_p),
            self.ylm.ctypes.data_as(ctypes.c_void_p),
            theta_gq.ctypes.data_as(ctypes.c_void_p),
            self.rad_loc.ctypes.data_as(ctypes.c_void_p),
            self.ylm_loc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nalpha),
            ctypes.c_int(self.nrad),
            ctypes.c_int(theta_gq.shape[0]),
            ctypes.c_int(self.nlm),
            ctypes.c_int(stride),
            ctypes.c_int(offset),
        )

    @classmethod
    def make_single_atom_indexer(cls, Y_nL, r_g):
        nn, nlm = Y_nL.shape
        lmax = int(np.sqrt(nlm + 1e-8)) - 1
        ng = r_g.size
        rad_arr = np.ascontiguousarray(r_g.astype(np.float64))
        ar_loc = np.zeros(ng, dtype=np.int32, order="C")
        ra_loc = np.asarray([0, ng], dtype=np.int32, order="C")
        rad_loc = np.ascontiguousarray((nn * np.arange(0, ng + 1)).astype(np.int32))
        ylm = np.ascontiguousarray(Y_nL.astype(np.float64))
        ylm_loc = np.zeros((ng,), dtype=np.int32)
        return cls(1, lmax, rad_arr, ar_loc, ra_loc, rad_loc, ylm, ylm_loc)
