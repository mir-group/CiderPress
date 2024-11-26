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
from pyscf import lib as pyscflib

from ciderpress import lib
from ciderpress.dft.lcao_convolutions import (
    ATCBasis,
    atom_loc_from_bas,
    get_gamma_lists_from_bas_and_env,
    libcider,
)
from ciderpress.dft.sph_harm_coeff import get_deriv_ylm_coeff

ATOM_OF = 0
ANG_OF = 1


def get_deriv_mol(old_bas, old_env):
    assert old_bas.ndim == 2 and old_bas.shape[1] == 8
    bas = []
    for b in old_bas:
        if b[ANG_OF] == 0:
            continue
        else:
            bas.append(b)
    bas = np.asarray(bas, order="C", dtype=np.int32)
    bas[:, ANG_OF] -= 1
    bas[:, 2:5] = 0
    loc = [0]
    for b in bas:
        loc.append(loc[-1] + 2 * b[ANG_OF] + 1)
    assert loc[-1] == np.sum(2 * bas[:, ANG_OF] + 1)
    loc = np.asarray(loc, order="C", dtype=np.int32)
    return bas, old_env.copy(), loc, atom_loc_from_bas(bas)


class LCAOInterpolator:
    """
    rads = aparam * (exp(dparam * arange(nrad)) - 1)
    """

    onsite_direct = False
    _ga_loc = None

    def __init__(
        self,
        atom_coords,
        atco,
        n0,
        n1,
        aparam=0.03,
        dparam=0.04,
        nrad=200,
    ):
        """
        Args:
            atom_coords (np.ndarray) : atomic coordinates (natm, 3)
            atco (ATCBasis) : atomic basis for interpolator. This is typically
                the atco_out object of a ConvolutionCollection
            n0 (int) : Number of l=0 features
            n1 (int) : Number of l=1 features
            aparam (float) : A parameter for radial grid.
            dparam (float) : D parameter for radial grid.
            nrad (int) : size of radial grid
        """
        assert atom_coords.shape == (atco.natm, 3)
        self.atom_coords = np.ascontiguousarray(atom_coords)
        self._aparam = aparam
        self._dparam = dparam
        self._nrad = nrad
        self.atco = atco
        lmax = np.max(atco.bas[:, ANG_OF])
        self.lmax = lmax
        self.nlm = (lmax + 1) * (lmax + 1)
        self.rads = np.asarray(
            aparam * (np.exp(dparam * np.arange(nrad)) - 1),
            dtype=np.float64,
            order="C",
        )
        self.w0_rsp = None
        self.wm_rsp = None
        self._n0 = n0
        self._n1 = n1
        self._gaunt_coeff = None
        self._l1bas = None
        self._l1loc = None
        self.l1atco = None

        self._l0bas = self.atco.bas
        self._l0env = self.atco.env
        self._l0loc = self.atco.ao_loc
        self._l0_atom_loc = self.atco.atom_loc
        self._ga_loc_ptr = lib.c_null_ptr()
        self._iatom_loc_ptr = lib.c_null_ptr()

        def _call_spline_(w_rsp, bas, env):
            nbas = len(bas)
            shls_slice = (0, nbas)
            libcider.compute_spline_maps(
                w_rsp.ctypes.data_as(ctypes.c_void_p),
                self.rads.ctypes.data_as(ctypes.c_void_p),
                bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbas),
                env.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int * 2)(*shls_slice),
                ctypes.c_int(self.nrad),
            )

        shape = (nrad, self.atco.nbas, 4)
        if self._n0 > 0:
            self.w0_rsp = np.empty(shape, dtype=np.float64, order="C")
            _call_spline_(self.w0_rsp, self._l0bas, self._l0env)
        if self._n1 > 0:
            dbas, denv, dloc, aloc = get_deriv_mol(self._l0bas, self._l0env)
            dat = get_gamma_lists_from_bas_and_env(dbas, denv)
            self.l1atco = ATCBasis(*dat)
            self.wm_rsp = np.empty((nrad, len(dbas), 4), dtype=np.float64, order="C")
            self._gaunt_coeff = get_deriv_ylm_coeff(self.lmax)
            self._l1bas = dbas
            self._l1loc = dloc
            self._l1env = denv
            self._l1_atom_loc = aloc
            _call_spline_(self.wm_rsp, self._l1bas, self._l1env)

        self.is_num_ai_setup = False

    @property
    def num_in(self):
        """Get the number of (orbital-space) inputs (i.e. nbeta for ccl)"""
        return self._n0 + 2 * self._n1

    @property
    def num_out(self):
        """Get the number of (spline-space) outputs"""
        return self._n0 + 4 * self._n1

    @property
    def aparam(self):
        """Parameter a for radial spline grid (see __init__ docstring)."""
        return self._aparam

    @property
    def dparam(self):
        """Parameter d for radial spline grid (see __init__ docstring)."""
        return self._dparam

    @property
    def nrad(self):
        """Size of radial grid fpr spline"""
        return self._nrad

    @property
    def ngrids(self):
        if self.all_coords is None:
            raise RuntimeError("Uninitialized")
        else:
            return self.all_coords.shape[0]

    @classmethod
    def from_ccl(cls, atom_coords, ccl, aparam=0.03, dparam=0.04, nrad=200):
        """
        Initialize an instance of LCAOInterpolator from a
        ConvolutionCollection instance.

        Args:
            atom_coords (np.ndarray): (natm, 3) atomic coordinates
            ccl (ConvolutionCollection): ConvolutionCollection object
                from which to build the interpolator.
            aparam (float) : A parameter for radial grid.
            dparam (float) : D parameter for radial grid.
            nrad (int) : size of radial grid

        Returns:
            New LCAOInterpolator
        """
        return cls(
            atom_coords,
            ccl.atco_out,
            ccl.n0,
            ccl.n1,
            aparam=aparam,
            dparam=dparam,
            nrad=nrad,
        )

    def _orb2spline_(
        self, atco, f_arlpq, f_uq, w_rsp, nalpha, offset_spline, offset_orb, fwd
    ):
        """
        Helper function for converting between the orbital and spline bases.

        Args:
            f_arlpq (np.ndarray): Spline basis tensor
            f_uq (np.ndarray): Orbital basis tensor (in atco basis)
            w_rsp (np.ndarray): Splines for the orbitals
            offset_spline (int): q offset for the spline basis
            offset_orb (int): q offset for the orbital basis
            fwd (bool): fwd=True takes f_uq and writes f_arlpq.
                        fwd=False takes f_arlpq and writes f_uq.
        """
        assert f_arlpq.shape == (atco.natm, self.nrad, self.nlm, 4, self.num_out)
        assert f_uq.shape[0] == atco.nao
        assert w_rsp is not None
        assert f_arlpq.flags.c_contiguous
        assert f_uq.flags.c_contiguous
        assert w_rsp.flags.c_contiguous
        orb_stride = f_uq.shape[-1]
        spline_stride = f_arlpq.shape[-1]
        assert nalpha + offset_orb <= orb_stride
        assert nalpha + offset_spline <= spline_stride
        if fwd:
            fn = libcider.project_conv_to_spline
        else:
            fn = libcider.project_spline_to_conv
        fn(
            f_arlpq.ctypes.data_as(ctypes.c_void_p),
            f_uq.ctypes.data_as(ctypes.c_void_p),
            w_rsp.ctypes.data_as(ctypes.c_void_p),
            atco.atco_c_ptr,
            ctypes.c_int(nalpha),
            ctypes.c_int(self.nrad),
            ctypes.c_int(self.nlm),
            ctypes.c_int(orb_stride),
            ctypes.c_int(spline_stride),
            ctypes.c_int(offset_spline),
            ctypes.c_int(offset_orb),
        )

    def _fill_l1_coeff_(self, f_uq, f1_uq, offset, offset1, fwd):
        stride = f_uq.shape[1]
        stride1 = f1_uq.shape[1]
        if fwd:
            fn = libcider.fill_l1_coeff_fwd
        else:
            fn = libcider.fill_l1_coeff_bwd
        fn(
            f_uq.ctypes.data_as(ctypes.c_void_p),
            f1_uq.ctypes.data_as(ctypes.c_void_p),
            self._gaunt_coeff.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.nlm),
            self.atco.atco_c_ptr,
            self.l1atco.atco_c_ptr,
            ctypes.c_int(stride),
            ctypes.c_int(offset),
            ctypes.c_int(stride1),
            ctypes.c_int(offset1),
        )

    def conv2spline(self, f_uq, f_arlpq=None, return_f1_uq=False):
        """
        Takes a set of convolutions stored in orbital space and transforms
        it into the spline basis

        Args:
            f_uq (np.ndarray): Orbital basis convolutions
            f_arlpq (np.ndarray): Spline basis convolutions

        Returns:
            f_arlpq (np.ndarray)
        """
        assert f_uq.flags.c_contiguous
        assert f_uq.shape == (self.atco.nao, self.num_in), (
            f_uq.shape,
            (self.atco.nao, self.num_in),
        )
        if f_arlpq is None:
            f_arlpq = np.zeros((self.atco.natm, self.nrad, self.nlm, 4, self.num_out))
        if self._n0 > 0:
            self._orb2spline_(
                self.atco, f_arlpq, f_uq, self.w0_rsp, self._n0, 0, 0, True
            )
        f1_uq = None
        if self._n1 > 0:
            f1_uq = np.zeros((self.l1atco.nao, 3 * self._n1))
            for i1 in range(self._n1):
                self._fill_l1_coeff_(f_uq, f1_uq, self._n0 + i1, 3 * i1, True)
            self._orb2spline_(
                self.l1atco,
                f_arlpq,
                f1_uq,
                self.wm_rsp,
                self._n1 * 3,
                self._n0,
                0,
                True,
            )
            offset_spline = self._n0 + 3 * self._n1
            offset_orb = self._n0 + self._n1
            self._orb2spline_(
                self.atco,
                f_arlpq,
                f_uq,
                self.w0_rsp,
                self._n1,
                offset_spline,
                offset_orb,
                True,
            )
        if return_f1_uq:
            return f_arlpq, f1_uq
        else:
            return f_arlpq

    def spline2conv(self, f_arlpq, f_uq=None, f1_uq=None):
        """
        Takes a set of convolutions stored in spline basis and transforms
        it into the orbital basis. Backwards version of spline2conv.

        Args:
            f_arlpq (np.ndarray): Spline basis convolutions
            f_uq (np.ndarray): Orbital basis convolutions

        Returns:
            f_uq (np.ndarray)
        """
        assert f_arlpq.flags.c_contiguous
        if f_uq is None:
            f_uq = np.zeros((self.atco.nao, self.num_in))
        assert f_uq.shape == (self.atco.nao, self.num_in)
        if self._n0 > 0:
            self._orb2spline_(
                self.atco, f_arlpq, f_uq, self.w0_rsp, self._n0, 0, 0, False
            )
        if self._n1 > 0:
            if f1_uq is None:
                f1_uq = np.zeros((self.l1atco.nao, 3 * self._n1))
            self._orb2spline_(
                self.l1atco,
                f_arlpq,
                f1_uq,
                self.wm_rsp,
                self._n1 * 3,
                self._n0,
                0,
                False,
            )
            for i1 in range(self._n1):
                self._fill_l1_coeff_(f_uq, f1_uq, self._n0 + i1, 3 * i1, False)
            offset_spline = self._n0 + 3 * self._n1
            offset_orb = self._n0 + self._n1
            self._orb2spline_(
                self.atco,
                f_arlpq,
                f_uq,
                self.w0_rsp,
                self._n1,
                offset_spline,
                offset_orb,
                False,
            )
        return f_uq

    @property
    def natm(self):
        return self.atco.natm

    def _set_num_ai(self, all_coords):
        """
        This function computes the number of coordinates in
        all_coords that fall in each spline "box" for the radial
        interpolation grid on each atom. The result is then
        used to create the _loc_ai member, which in turn
        is used by the _compute_sline_ind_order function
        to order grid coordinates by their spline box on a given atom.
        """
        if all_coords is None:
            assert self.is_num_ai_setup
            return
        all_coords = np.ascontiguousarray(all_coords)
        assert all_coords.shape[1] == 3
        ngrids_tot = all_coords.shape[0]
        self.num_ai = np.empty((self.natm, self.nrad), dtype=np.int32, order="C")
        """libcider.compute_num_spline_contribs(
            self.num_ai.ctypes.data_as(ctypes.c_void_p),
            all_coords.ctypes.data_as(ctypes.c_void_p),
            self.atom_coords.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(self.aparam),
            ctypes.c_double(self.dparam),
            ctypes.c_int(self.natm),
            ctypes.c_int(ngrids_tot),
            ctypes.c_int(self.nrad),
            self._ga_loc_ptr,
        )"""
        libcider.compute_num_spline_contribs_new(
            self.num_ai.ctypes.data_as(ctypes.c_void_p),
            all_coords.ctypes.data_as(ctypes.c_void_p),
            self.atom_coords.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(self.aparam),
            ctypes.c_double(self.dparam),
            ctypes.c_int(self.natm),
            ctypes.c_int(ngrids_tot),
            ctypes.c_int(self.nrad),
            self._iatom_loc_ptr,
        )
        self.all_coords = all_coords
        self._loc_ai = np.ascontiguousarray(
            (
                np.append(
                    np.zeros((self.natm, 1), dtype=np.int32),
                    np.cumsum(self.num_ai, axis=1),
                    axis=1,
                )
            ).astype(np.int32)
        )
        # buffers
        self._ind_ord_fwd = np.empty(ngrids_tot, dtype=np.int32, order="C")
        self._ind_ord_bwd = np.empty(ngrids_tot, dtype=np.int32, order="C")
        self._coords_ord = np.empty((ngrids_tot, 3), dtype=np.float64, order="C")
        # end buffers
        self._maxg = np.max(self.num_ai)

        self.is_num_ai_setup = True

    def _compute_spline_ind_order(self, a):
        """
        Assuming _set_num_ai has been called previously, order
        self.all_coords such that each coordinate is in increasing
        order of spline index for the radial interpolation grid
        on atom a.
        """
        if not self.is_num_ai_setup:
            raise RuntimeError
        ngrids_tot = self.all_coords.shape[0]
        nrad = self.num_ai.shape[1]
        assert self.all_coords.shape[1] == 3
        assert self.all_coords.flags.c_contiguous
        libcider.compute_spline_ind_order_new(
            self._loc_ai[a].ctypes.data_as(ctypes.c_void_p),
            self.all_coords.ctypes.data_as(ctypes.c_void_p),
            self.atom_coords[a].ctypes.data_as(ctypes.c_void_p),
            self._coords_ord.ctypes.data_as(ctypes.c_void_p),
            self._ind_ord_fwd.ctypes.data_as(ctypes.c_void_p),
            self._ind_ord_bwd.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(self.aparam),
            ctypes.c_double(self.dparam),
            ctypes.c_int(ngrids_tot),
            ctypes.c_int(nrad),
            self._iatom_loc_ptr,
            ctypes.c_int(a),
        )
        ngrids = np.max(self._loc_ai)
        nlm = self.nlm
        self._auxo_gl_buf = np.ndarray((ngrids, nlm), order="C")
        self._auxo_gp_buf = np.ndarray((ngrids, 4), order="C")
        return self._coords_ord, self._ind_ord_fwd

    def _eval_spline_bas_single(self, a):
        """
        Note that _set_num_ai must have been called previously, because
        it is required for _compute_spline_ind_order to work, which
        is called by this function. TODO might be better to have a cleaner
        solution for this algorithm.
        """
        self._compute_spline_ind_order(a)
        ngrids = self._coords_ord.shape[0]
        if self.onsite_direct:
            # TODO maybe a bit inefficient, should clean this up
            # when I get around to parallelizing _compute_spline_ind_order
            ngrids -= np.sum(self.grids_indexer.iatom_list == a)
        nlm = self.nlm
        nrad = self.nrad
        atm_coord = self.atom_coords[a]
        auxo_gl = np.ndarray((ngrids, nlm), order="C", buffer=self._auxo_gl_buf)
        auxo_gp = np.ndarray((ngrids, 4), order="C", buffer=self._auxo_gp_buf)
        libcider.compute_spline_bas_separate(
            auxo_gl.ctypes.data_as(ctypes.c_void_p),
            auxo_gp.ctypes.data_as(ctypes.c_void_p),
            self._coords_ord.ctypes.data_as(ctypes.c_void_p),
            atm_coord.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(1),
            ctypes.c_int(ngrids),
            ctypes.c_int(nrad),
            ctypes.c_int(nlm),
            ctypes.c_double(self.aparam),
            ctypes.c_double(self.dparam),
        )
        return auxo_gl, auxo_gp

    def _eval_spline_bas_single_deriv(self, a):
        self._compute_spline_ind_order(a)
        ngrids = self._coords_ord.shape[0]
        if self.onsite_direct:
            # TODO maybe a bit inefficient, should clean this up
            # when I get around to parallelizing _compute_spline_ind_order
            ngrids -= np.sum(self.grids_indexer.iatom_list == a)
        nlm = self.nlm
        nrad = self.nrad
        atm_coord = self.atom_coords[a]
        auxo_vgl = np.ndarray((4, ngrids, nlm), order="C")
        auxo_vgp = np.ndarray((4, ngrids, 4), order="C")
        libcider.compute_spline_bas_separate_deriv(
            auxo_vgl.ctypes.data_as(ctypes.c_void_p),
            auxo_vgp.ctypes.data_as(ctypes.c_void_p),
            self._coords_ord.ctypes.data_as(ctypes.c_void_p),
            atm_coord.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(1),
            ctypes.c_int(ngrids),
            ctypes.c_int(nrad),
            ctypes.c_int(nlm),
            ctypes.c_double(self.aparam),
            ctypes.c_double(self.dparam),
        )
        return auxo_vgl, auxo_vgp

    def _call_l1_fill(self, f_gq, atom_coord, fwd):
        if fwd:
            l1_fn = libcider.add_lp1_term_fwd
        else:
            l1_fn = libcider.add_lp1_term_bwd
        for i1 in range(self._n1):
            ig = self.num_out + i1 - self._n1
            ix = self._n0 + 3 * i1
            l1_fn(
                f_gq.ctypes.data_as(ctypes.c_void_p),
                self.all_coords.ctypes.data_as(ctypes.c_void_p),
                atom_coord.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(f_gq.shape[0]),
                ctypes.c_int(ig),
                ctypes.c_int(ix + 0),
                ctypes.c_int(ix + 1),
                ctypes.c_int(ix + 2),
                ctypes.c_int(f_gq.shape[1]),
            )

    def _call_l1_fill_grad(self, excsum, f_gq, vf_gq, ia):
        natm = self.atco.natm
        iatom_list = self.grids_indexer.iatom_list
        n = iatom_list.size
        for i1 in range(self._n1):
            ig = self.num_out + i1 - self._n1
            ix = self._n0 + 3 * i1
            libcider.add_lp1_term_grad(
                excsum.ctypes.data_as(ctypes.c_void_p),
                f_gq.ctypes.data_as(ctypes.c_void_p),
                vf_gq.ctypes.data_as(ctypes.c_void_p),
                iatom_list.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(ia),
                ctypes.c_int(natm),
                ctypes.c_int(n),
                ctypes.c_int(ig),
                ctypes.c_int(ix + 0),
                ctypes.c_int(ix + 1),
                ctypes.c_int(ix + 2),
                ctypes.c_int(f_gq.shape[1]),
            )

    def _interpolate_nopar_atom(self, f_arlpq, f_gq, fwd):
        assert self.is_num_ai_setup
        ngrids_tot = self.all_coords.shape[0]
        nalpha = f_arlpq.shape[-1]
        assert f_arlpq.shape[-1] == f_gq.shape[-1]
        if fwd:
            fn = libcider.compute_mol_convs_single_new
        else:
            fn = libcider.compute_pot_convs_single_new
        for a in range(self.atco.natm):
            auxo_gl, auxo_gp = self._eval_spline_bas_single(a)
            if self.onsite_direct and len(self._loc_ai) == 1:
                if not fwd:
                    f_arlpq[:] = 0
            else:
                ngrids = ngrids_tot
                if self.onsite_direct:
                    ngrids -= self._ga_loc[a + 1] - self._ga_loc[a]
                if not fwd:
                    self._call_l1_fill(f_gq, self.atom_coords[a], fwd)
                fn(
                    f_gq.ctypes.data_as(ctypes.c_void_p),
                    f_arlpq[a].ctypes.data_as(ctypes.c_void_p),
                    auxo_gl.ctypes.data_as(ctypes.c_void_p),
                    auxo_gp.ctypes.data_as(ctypes.c_void_p),
                    self._loc_ai[a].ctypes.data_as(ctypes.c_void_p),
                    self._ind_ord_fwd.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nalpha),
                    ctypes.c_int(self.nrad),
                    ctypes.c_int(ngrids),
                    ctypes.c_int(self.nlm),
                    ctypes.c_int(self._maxg),
                )
                if fwd:
                    self._call_l1_fill(f_gq, self.atom_coords[a], fwd)

    def _contract_grad_terms(self, excsum, f_g, a, v):
        iatom_list = self.grids_indexer.iatom_list
        assert iatom_list is not None
        assert iatom_list.flags.c_contiguous
        ngrids = iatom_list.size
        libcider.contract_grad_terms_parallel(
            excsum.ctypes.data_as(ctypes.c_void_p),
            f_g.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.atco.natm),
            ctypes.c_int(a),
            ctypes.c_int(v),
            ctypes.c_int(ngrids),
            iatom_list.ctypes.data_as(ctypes.c_void_p),
        )

    def _interpolate_nopar_atom_deriv(self, f_arlpq, f_gq):
        assert self.is_num_ai_setup
        ngrids_tot = self.all_coords.shape[0]
        nalpha = f_arlpq.shape[-1]
        assert f_arlpq.shape[-1] == f_gq.shape[-1]
        fn = libcider.compute_mol_convs_single_new
        excsum = np.zeros((self.atco.natm, 3))
        ftmp_gq = np.empty_like(f_gq)
        for a in range(self.atco.natm):
            auxo_vgl, auxo_vgp = self._eval_spline_bas_single_deriv(a)
            if self.onsite_direct and len(self._loc_ai) == 1:
                pass
            else:
                ngrids = ngrids_tot
                if self.onsite_direct:
                    ngrids -= self._ga_loc[a + 1] - self._ga_loc[a]
                args = [
                    ftmp_gq.ctypes.data_as(ctypes.c_void_p),
                    f_arlpq[a].ctypes.data_as(ctypes.c_void_p),
                    auxo_vgl[0].ctypes.data_as(ctypes.c_void_p),
                    auxo_vgp[0].ctypes.data_as(ctypes.c_void_p),
                    self._loc_ai[a].ctypes.data_as(ctypes.c_void_p),
                    self._ind_ord_fwd.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nalpha),
                    ctypes.c_int(self.nrad),
                    ctypes.c_int(ngrids),
                    ctypes.c_int(self.nlm),
                    ctypes.c_int(self._maxg),
                ]
                if self._n1 > 0:
                    ftmp_gq[:] = 0
                    fn(*args)
                    self._call_l1_fill_grad(excsum, ftmp_gq, f_gq, a)
                for v in range(3):
                    ftmp_gq[:] = 0
                    args[2] = auxo_vgl[0].ctypes.data_as(ctypes.c_void_p)
                    args[3] = auxo_vgp[v + 1].ctypes.data_as(ctypes.c_void_p)
                    fn(*args)
                    args[2] = auxo_vgl[v + 1].ctypes.data_as(ctypes.c_void_p)
                    args[3] = auxo_vgp[0].ctypes.data_as(ctypes.c_void_p)
                    fn(*args)
                    self._call_l1_fill(ftmp_gq, self.atom_coords[a], True)
                    ftmp = pyscflib.einsum("gq,gq->g", ftmp_gq, f_gq)
                    self._contract_grad_terms(excsum, ftmp, a, v)
        return excsum

    def set_coords(self, coords):
        self._set_num_ai(coords)

    def interpolate_fwd(self, f_arlpq, f_gq=None):
        """

        Args:
            f_arlpq:
            f_gq: output
            atom_par (bool): If True, parallelize evaluation over atoms. Note,
                this can get quite memory intensive, so if atom_par=True,
                the user should only pass reasonably small sets of coords
                to this function.

        Returns:
            f_gq (np.ndarray): shape (coords.shape[0], f_arlpq.shape[-1])
        """
        if not self.is_num_ai_setup:
            raise ValueError("Need to set up indexes and coordinates first")
        assert self.all_coords is not None
        assert self.all_coords.ndim == 2
        assert self.all_coords.shape[1] == 3
        if f_gq is None:
            f_gq = np.zeros((self.all_coords.shape[0], self.num_out))
        self._interpolate_nopar_atom(f_arlpq, f_gq, True)
        return f_gq

    def interpolate_grad(self, f_arlpq, f_gq):
        if not self.is_num_ai_setup:
            raise ValueError("Need to set up indexes and coordinates first")
        assert self.all_coords is not None
        assert self.all_coords.ndim == 2
        assert self.all_coords.shape[1] == 3
        excsum = self._interpolate_nopar_atom_deriv(f_arlpq, f_gq)
        return excsum

    def interpolate_bwd(self, f_gq, f_arlpq=None):
        if not self.is_num_ai_setup:
            raise ValueError("Need to set up indexes and coordinates first")
        assert self.all_coords is not None
        assert self.all_coords.ndim == 2
        assert self.all_coords.shape[1] == 3
        if f_arlpq is None:
            f_arlpq = np.zeros((self.atco.natm, self.nrad, self.nlm, 4, self.num_out))
        self._interpolate_nopar_atom(f_arlpq, f_gq, False)
        return f_arlpq

    def project_orb2grid(self, f_uq, f_gq=None):
        """

        Args:
            f_uq:
            f_gq:

        Returns:

        """
        f_arlpq = self.conv2spline(f_uq)
        f_gq = self.interpolate_fwd(f_arlpq=f_arlpq, f_gq=f_gq)
        return f_gq

    def project_orb2grid_grad(self, f_uq, f_gq):
        f_arlpq = self.conv2spline(f_uq)
        excsum = self.interpolate_grad(f_arlpq=f_arlpq, f_gq=f_gq)
        return excsum

    def project_grid2orb(self, f_gq, f_uq=None):
        f_arlpq = self.interpolate_bwd(f_gq=f_gq)
        f_uq = self.spline2conv(f_arlpq, f_uq=f_uq)
        return f_uq


class LCAOInterpolatorDirect(LCAOInterpolator):
    def __init__(
        self,
        grids_indexer,
        atom_coords,
        atco,
        n0,
        n1,
        aparam=0.03,
        dparam=0.04,
        nrad=200,
        onsite_direct=True,
    ):
        """
        This modified version of LCAOInterpolator takes an AtomicGridsIndexer
        as an argument, which allows it to support computing onsite
        convolution terms directly rather than via spline.
        This can be done by calling interpolate_direct_fwd or
        interpolate_direct_bwd.

        Args:
            grids_indexer (AtomicGridsIndexer):
            atom_coords:
            atco:
            n0:
            n1:
            aparam:
            dparam:
            nrad:
            onsite_direct
        """
        super(LCAOInterpolatorDirect, self).__init__(
            atom_coords, atco, n0, n1, aparam=aparam, dparam=dparam, nrad=nrad
        )
        self.grids_indexer = grids_indexer
        self._ga_loc = np.ascontiguousarray(self.grids_indexer.ga_loc)
        self.onsite_direct = onsite_direct
        if self.onsite_direct:
            self._ga_loc_ptr = self._ga_loc.ctypes.data_as(ctypes.c_void_p)
            self._iatom_loc_ptr = self.grids_indexer.iatom_list.ctypes.data_as(
                ctypes.c_void_p
            )
        else:
            self._ga_loc_ptr = lib.c_null_ptr()
            self._iatom_loc_ptr = lib.c_null_ptr()

    @property
    def grid_loc_atom(self):
        return self._ga_loc.copy()

    def set_coords(self, coords):
        assert coords.shape[0] == (
            self.grids_indexer.idx_map.size + self.grids_indexer.padding
        )
        self._set_num_ai(coords[: self.grids_indexer.idx_map.size])

    def _run_onsite_orb2grid(self, atco, f_uq, f_gq, nalpha, offset1, offset2, fwd):
        f_rlmq = self.grids_indexer.empty_rlmq(nalpha)
        f_rlmq[:] = 0.0
        if fwd:
            atco.convert_rad2orb_(
                f_rlmq,
                f_uq,
                self.grids_indexer,
                self.grids_indexer.rad_arr,
                rad2orb=not fwd,
                offset=offset1,
            )
            self.grids_indexer.reduce_angc_ylm_(
                f_rlmq,
                f_gq,
                a2y=not fwd,
                offset=offset2,
            )
        else:
            self.grids_indexer.reduce_angc_ylm_(
                f_rlmq,
                f_gq,
                a2y=not fwd,
                offset=offset2,
            )
            atco.convert_rad2orb_(
                f_rlmq,
                f_uq,
                self.grids_indexer,
                self.grids_indexer.rad_arr,
                rad2orb=not fwd,
                offset=offset1,
            )

    def _run_onsite_lp1(self, f_gq, fwd):
        if fwd:
            fn = libcider.add_lp1_onsite_new_fwd
        else:
            fn = libcider.add_lp1_onsite_new_bwd
        for i1 in range(self._n1):
            ig = self.num_out + i1 - self._n1
            ix = self._n0 + 3 * i1
            nf = f_gq.shape[1]
            fn(
                f_gq.ctypes.data_as(ctypes.c_void_p),
                self.grids_indexer.rad_arr.ctypes.data_as(ctypes.c_void_p),
                self.grids_indexer.rad_loc.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(self.grids_indexer.nrad),
                self.grids_indexer.dirs.ctypes.data_as(ctypes.c_void_p),
                self.grids_indexer.ylm_loc.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nf),
                ctypes.c_int(ig),
                ctypes.c_int(ix + 0),
                ctypes.c_int(ix + 1),
                ctypes.c_int(ix + 2),
            )

    def project_orb2grid(self, f_uq, f_gq=None):
        """

        Args:
            f_uq:
            f_gq:

        Returns:

        """
        ngpp = self.all_coords.shape[0] + self.grids_indexer.padding
        if f_gq is None:
            f_gq = np.zeros((ngpp, self.num_out))
        assert f_gq.shape == (ngpp, self.num_out)
        out_gq = f_gq[: self.all_coords.shape[0]]
        in_uq = f_uq
        assert out_gq.flags.c_contiguous
        if self.onsite_direct:
            f_arlpq, f1_uq = self.conv2spline(in_uq, return_f1_uq=True)
            # f_arlpq = self.conv2spline(in_uq, return_f1_uq=False)
            tmp_gq = np.zeros((self.grids_indexer.all_weights.size, self.num_out))
            # tmp_gq = np.zeros(
            #    (self.grids_indexer.all_weights.size, self._n0)
            # )
            self._run_onsite_orb2grid(self.atco, in_uq, tmp_gq, self._n0, 0, 0, True)
            if self._n1 > 0:
                self._run_onsite_orb2grid(
                    self.l1atco, f1_uq, tmp_gq, 3 * self._n1, 0, self._n0, True
                )
                o1 = self._n0 + self._n1
                o2 = self._n0 + 3 * self._n1
                self._run_onsite_orb2grid(
                    self.atco, in_uq, tmp_gq, self._n1, o1, o2, True
                )
                self._run_onsite_lp1(tmp_gq, True)
            out_gq[:] += tmp_gq[self.grids_indexer.idx_map]
        else:
            f_arlpq = self.conv2spline(in_uq, return_f1_uq=False)
        self.interpolate_fwd(f_arlpq=f_arlpq, f_gq=out_gq)
        return f_gq

    def project_grid2orb(self, f_gq, f_uq=None):
        ngpp = self.all_coords.shape[0] + self.grids_indexer.padding
        assert f_gq.shape == (ngpp, self.num_out)
        in_gq = f_gq[: self.all_coords.shape[0]]
        if f_uq is None:
            f_uq = np.zeros((self.atco.nao, self.num_in))
        out_uq = f_uq
        assert in_gq.flags.c_contiguous
        f_arlpq = self.interpolate_bwd(in_gq)
        if self.onsite_direct:
            tmp_gq = np.zeros((self.grids_indexer.all_weights.size, self.num_out))
            tmp_gq[self.grids_indexer.idx_map] = in_gq
            if self._n1 > 0:
                self._run_onsite_lp1(tmp_gq, False)
            self._run_onsite_orb2grid(self.atco, out_uq, tmp_gq, self._n0, 0, 0, False)
            if self._n1 > 0:
                f1_uq = np.zeros((self.l1atco.nao, 3 * self._n1))
                self._run_onsite_orb2grid(
                    self.l1atco, f1_uq, tmp_gq, 3 * self._n1, 0, self._n0, False
                )
                o1 = self._n0 + self._n1
                o2 = self._n0 + 3 * self._n1
                self._run_onsite_orb2grid(
                    self.atco, out_uq, tmp_gq, self._n1, o1, o2, False
                )
            else:
                f1_uq = None
            # f_arlpq = self.interpolate_bwd(in_gq)
            self.spline2conv(f_arlpq, f_uq=out_uq, f1_uq=f1_uq)
        else:
            # f_arlpq = self.interpolate_bwd(in_gq)
            self.spline2conv(f_arlpq, f_uq=out_uq)
        return out_uq

    @classmethod
    def from_ccl(
        cls,
        grids_indexer,
        atom_coords,
        ccl,
        aparam=0.03,
        dparam=0.04,
        nrad=200,
        onsite_direct=True,
    ):
        """
        Initialize an instance of LCAOInterpolator from a
        ConvolutionCollection instance.

        Args:
            atom_coords (np.ndarray): (natm, 3) atomic coordinates
            ccl (ConvolutionCollection): ConvolutionCollection object
                from which to build the interpolator.
            aparam (float) : A parameter for radial grid.
            dparam (float) : D parameter for radial grid.
            nrad (int) : size of radial grid
            onsite_direct (bool) :

        Returns:
            New LCAOInterpolator
        """
        return cls(
            grids_indexer,
            atom_coords,
            ccl.atco_out,
            ccl.n0,
            ccl.n1,
            aparam=aparam,
            dparam=dparam,
            nrad=nrad,
            onsite_direct=onsite_direct,
        )
