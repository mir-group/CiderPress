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
import scipy.special

from ciderpress import lib
from ciderpress.dft.grids_indexer import AtomicGridsIndexer
from ciderpress.lib import load_library as load_cider_library

libcider = load_cider_library("libmcider")
libcider.get_atco_nbas.restype = ctypes.c_int

DEFAULT_AUX_BETA = 1.6
DEFAULT_CIDER_LMAX = 10
ATOM_OF = 0
ANG_OF = 1
PTR_EXP = 5
PTR_COEFF = 6

IFEAT_ID_TO_CONTRIB = {
    0: 0,  # squared-exponential
    1: 1,  # R2 x squared-exp
    2: 2,  # AR2 x squared-exp
    3: 7,  # A x squared-exp
    4: 8,  # A2R2 x squared-exp
    5: 9,  # Laplacian
    6: (4, 5),  # DR x squared-exp
    7: (3, 6),  # A DR x squared-exp
}


def atom_loc_from_bas(bas):
    natm = np.max(bas[:, ATOM_OF]) + 1  # max ia + 1 is number of atoms
    atom_loc = np.zeros(natm + 1, dtype=np.int32)
    for b in bas:
        atom_loc[b[ATOM_OF] + 1] += 1
    atom_loc = np.cumsum(atom_loc)
    return np.asarray(atom_loc, dtype=np.int32, order="C")


# from PySCF, helper function for normalization coeff
def gaussian_int(n, alpha):
    n1 = (n + 1) * 0.5
    return scipy.special.gamma(n1) / (2.0 * alpha**n1)


# from PySCF, helper function for normalization coeff
def gto_norm(l, expnt):
    if np.all(l >= 0):
        return 1 / np.sqrt(gaussian_int(l * 2 + 2, 2 * expnt))
    else:
        raise ValueError("l should be >= 0")


def get_gamma_lists_from_bas_and_env(bas, env):
    natm = np.max(bas[:, ATOM_OF]) + 1
    atom2l0 = [0]
    lmaxs = []
    gamma_loc = [0]
    all_exps = []
    all_coefs = []
    for ia in range(natm):
        ls = bas[bas[:, ATOM_OF] == ia, ANG_OF]
        # ls must be ascending
        assert np.all(ls[1:] >= ls[:-1])
        lmax = np.max(ls)
        exps = []
        coefs = []
        lmaxs.append(np.max(ls))
        for l in range(lmax + 1):
            cond = np.logical_and(bas[:, ATOM_OF] == ia, bas[:, ANG_OF] == l)
            exps_l = env[bas[cond, PTR_EXP]]
            coefs_l = env[bas[cond, PTR_COEFF]]
            gamma_loc.append(len(exps_l))
            exps.append(exps_l)
            coefs.append(coefs_l)
        all_exps.append(np.concatenate(exps))
        all_coefs.append(np.concatenate(coefs))
        atom2l0.append(len(gamma_loc) - 1)
    gamma_loc = np.asarray(np.cumsum(gamma_loc), dtype=np.int32, order="C")
    lmaxs = np.asarray(lmaxs, dtype=np.int32, order="C")
    all_exps = np.asarray(np.concatenate(all_exps), dtype=np.float64, order="C")
    all_coefs = np.asarray(np.concatenate(all_coefs), dtype=np.float64, order="C")
    atom2l0 = np.asarray(atom2l0, dtype=np.int32, order="C")
    return atom2l0, lmaxs, gamma_loc, all_coefs, all_exps


def get_convolution_expnts_from_expnts(
    gammas, atom2l0, lmaxs, gamma_loc, all_exps, gbuf=np.inf
):
    new_gamma_loc = [0]
    new_all_exps = []
    new_all_coefs = []
    for ia in range(len(atom2l0) - 1):
        lmax = lmaxs[ia]
        new_exps = []
        new_coefs = []
        for l in range(lmax + 1):
            loc0 = gamma_loc[atom2l0[ia] + l]
            loc1 = gamma_loc[atom2l0[ia] + l + 1]
            old_exps = all_exps[loc0:loc1]
            gmax = np.max(old_exps)
            cond = gammas < gmax * gbuf
            new_exps_l = gammas[cond]
            new_coefs_l = gto_norm(l, gammas[cond])
            new_gamma_loc.append(len(new_exps_l))
            new_exps.append(new_exps_l)
            new_coefs.append(new_coefs_l)
        new_all_exps.append(np.concatenate(new_exps))
        new_all_coefs.append(np.concatenate(new_coefs))
    gamma_loc = np.asarray(np.cumsum(new_gamma_loc), dtype=np.int32, order="C")
    lmaxs = np.asarray(lmaxs, dtype=np.int32, order="C")
    all_exps = np.asarray(np.concatenate(new_all_exps), dtype=np.float64, order="C")
    all_coefs = np.asarray(np.concatenate(new_all_coefs), dtype=np.float64, order="C")
    assert all_exps.size == all_coefs.size == gamma_loc[-1]
    atom2l0 = np.asarray(atom2l0, dtype=np.int32, order="C")
    return atom2l0, lmaxs, gamma_loc, all_coefs, all_exps


class ATCBasis:
    """
    A wrapper for the atc_basis_set C object, which is used to store
    a simple (but usually large), uncontracted Gaussian basis along
    with various index lists and other data for use in performing
    projections and manipulations of data in this basis on individual
    atoms.
    """

    def __init__(self, atom2l0, lmaxs, gamma_loc, coefs, exps, lower=True):
        """
        Initialize ATCBasis.

        Args:
            atom2l0 (np.ndarray): int32 array of length natm + 1 such that
                gamma_loc[atom2l0[ia] : atom2l0[ia + 1]] contains the global
                gamma_loc indexes for atom ia.
            lmaxs (np.ndarray) : int32, maximum l value for each atom.
            gamma_loc (np.ndarray): int32, a shell indexing list.
                gamma_loc[atom2l0[ia] + l] is the index of the first shell of
                atom ia with angular momentum l.
                [gamma_loc[atom2l0[ia] + l], atom2lo[ia] + l + 1]) is the range
                of indexes with atom ia and angular momentum l. These ranges
                are used to index coefs and exps.
            coefs (np.ndarray): float64, list of coefficients for
                normalizing Gaussian basis functions.
            exps (np.ndarray): float64, list of exponents for Gaussian
                basis functions.
            lower (bool): Whether to store lower or upper matrix for
                Cholesky factorizations within the C atc_basis_set object.
        """
        atco = lib.c_null_ptr()
        gamma_loc = np.asarray(gamma_loc, dtype=np.int32, order="C")
        lmaxs = np.asarray(lmaxs, dtype=np.int32, order="C")
        exps = np.asarray(exps, dtype=np.float64, order="C")
        coefs = np.asarray(coefs, dtype=np.float64, order="C")
        atom2l0 = np.asarray(atom2l0, dtype=np.int32, order="C")
        self.natm = len(atom2l0) - 1
        libcider.generate_atc_basis_set(
            ctypes.byref(atco),
            atom2l0.ctypes.data_as(ctypes.c_void_p),
            lmaxs.ctypes.data_as(ctypes.c_void_p),
            gamma_loc.ctypes.data_as(ctypes.c_void_p),
            exps.ctypes.data_as(ctypes.c_void_p),
            coefs.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(len(atom2l0) - 1),
            ctypes.c_char(b"L" if lower else b"U"),
        )
        self._atco = atco

    def __del__(self):
        """Free C data associated with this object."""
        libcider.free_atc_basis_set(self._atco)

    @property
    def atco_c_ptr(self):
        """Get a pointer to the C data for this object."""
        return self._atco

    @property
    def ao_loc(self):
        ao_loc = np.cumsum(2 * self.bas[:, ANG_OF] + 1)
        ao_loc = np.append([0], ao_loc)
        return np.asarray(ao_loc, dtype=np.int32, order="C")

    @property
    def atom_loc(self):
        return atom_loc_from_bas(self.bas)

    @property
    def nao(self):
        """Get the number of atomic orbitals in this atom-centered basis."""
        return libcider.get_atco_nao(self._atco)

    @property
    def nbas(self):
        """Get the number of shells in this atom-centered basis."""
        return libcider.get_atco_nbas(self._atco)

    @property
    def bas(self):
        """Get a copy of the shell information for this basis."""
        bas = np.zeros((self.nbas, 8), order="C", dtype=np.int32)
        libcider.get_atco_bas(
            bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.cast(self._atco, ctypes.c_void_p),
        )
        return bas

    @property
    def env(self):
        """Get a copy of the exponents and coefficients for this basis."""
        nbas = libcider.get_atco_nbas(self._atco)
        env = np.zeros((2 * nbas,), order="C", dtype=np.float64)
        libcider.get_atco_env(env.ctypes.data_as(ctypes.c_void_p), self._atco)
        return env

    def convert_rad2orb_(
        self, theta_rlmq, p_uq, loc, rads, rad2orb=True, offset=None, zero_output=True
    ):
        """
        Take a set of functions stored on radial grids and spherical harmonics
        and project it onto the atomic orbital basis. This is an in-place
        operation, where theta_rlmq is written if rad2orb is False and
        p_uq is written if rad2orb is True.
        - If rad2orb is True, loc has
          shape (natm + 1,), and theta_rlmq[loc[ia] : loc[ia+1]] is
          the set of functions on atom ia. This corresponds to
          the ra_loc member of an AtomicGridsIndexer
        - If rad2orb is False, loc has shape (nrad,), and theta_rlmq[r]
          is located on atom loc[r]. This corresponds to the ar_loc
          member of an AtomicGridsIndexer

        Args:
            theta_rlmq (np.ndarray): float64, shape (nrad, nlm, nalpha)
            p_uq (np.ndarray): float64, shape (ngrids, nalpha)
            loc (np.ndarray, AtomicGridsIndexer):
                An int32 array whose shape depends on whether rad2orb is True.
            rads (np.ndarray): float64 list of radial coordinates with size
                theta_rlmq.shape[0]. Corresponds to rad_arr member of
                AtomicGridsIndexer.
            rad2orb (bool): Whether to convert radial coordinates to orbital
                basis (True) or vice versa (False).
            offset (int): starting index in p_uq to add to/from
            zero_output (bool): Whether to set output zero before
                adding to it. Default to true. Set to False if you
                want to add to the existing values in the output
                array.
        """
        if isinstance(loc, AtomicGridsIndexer):
            if rad2orb:
                loc = loc.ra_loc
            else:
                loc = loc.ar_loc
        nalpha = theta_rlmq.shape[-1]
        stride = p_uq.shape[1]
        if offset is None:
            offset = 0
        else:
            assert offset >= 0
        nrad = rads.size
        assert loc.flags.c_contiguous
        assert loc.dtype == np.int32
        assert loc.ndim == 1
        assert rads.flags.c_contiguous
        assert rads.dtype == np.float64
        assert theta_rlmq.ndim == 3
        assert theta_rlmq.shape[0] == nrad
        assert theta_rlmq.flags.c_contiguous
        assert theta_rlmq.dtype == np.float64
        assert p_uq.ndim == 2
        assert p_uq.shape[0] == self.nao
        assert p_uq.flags.c_contiguous
        assert p_uq.dtype == np.float64
        assert offset + nalpha <= stride
        if rad2orb:
            assert loc.size == self.natm + 1
            fn = libcider.contract_rad_to_orb
            if zero_output:
                p_uq[:, offset : offset + nalpha] = 0.0
        else:
            assert loc.size == nrad
            fn = libcider.contract_orb_to_rad
            if zero_output:
                theta_rlmq[:] = 0.0
        fn(
            theta_rlmq.ctypes.data_as(ctypes.c_void_p),
            p_uq.ctypes.data_as(ctypes.c_void_p),
            loc.ctypes.data_as(ctypes.c_void_p),
            rads.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(rads.size),
            ctypes.c_int(theta_rlmq.shape[1]),
            self.atco_c_ptr,
            ctypes.c_int(nalpha),
            ctypes.c_int(stride),
            ctypes.c_int(offset),
        )


class ConvolutionCollection:
    """
    ConvolutionCollection is an object to assist in performing
    nonlocal density feature convolutions in a Gaussian basis set.
    It stores Gaussian integrals corresponding to convolutions
    of the input basis atco_inp by the squared-exponential
    functions with exponents alphas, projected onto the output
    basis atco_out. One can perform forward and backward convolutions.
    """

    is_vk = False

    def __init__(
        self,
        atco_inp,
        atco_out,
        alphas,
        alpha_norms,
        has_vj=True,
        ifeat_ids=None,
    ):
        """
        Initialize ConvolutionCollection.

        Args:
            atco_inp (ATCBasis): Input basis for convolutions
            atco_out (ATCBasis): Output basis for convolutions
            alphas (np.ndarray): float64 list of interpolating exponents
            alpha_norms (np.ndarray): float64 list of normalization
                factors for interpolation exponent functions.
            has_vj (bool): Whether to perform version j convolutions.
            ifeat_ids (list): Version i feature contributions to compute.
                Must be in strictly increasing order so that output
                can be processed by interpolation routines.
        """
        self.atco_inp = atco_inp
        self.atco_out = atco_out
        self._alphas = np.ascontiguousarray(alphas, dtype=np.float64)
        self._alpha_norms = np.ascontiguousarray(alpha_norms, dtype=np.float64)
        self._nalpha = self._alphas.size
        assert self._alpha_norms.size == self._nalpha
        if ifeat_ids is None:
            ifeat_ids = []
        icontrib0_ids = []
        icontrib1m_ids = []
        icontrib1p_ids = []
        for ifid in ifeat_ids:
            if ifid not in IFEAT_ID_TO_CONTRIB:
                raise ValueError("Unrecognized feature")
            icid = IFEAT_ID_TO_CONTRIB[ifid]
            if isinstance(icid, int):
                if len(icontrib1m_ids) > 0:
                    raise ValueError("l0 features must preceed l1 in ifeat_ids")
                icontrib0_ids.append(icid)
            else:
                assert len(icid) == 2
                icontrib1m_ids.append(icid[0])
                icontrib1p_ids.append(icid[1])
        self._ifeat_ids = ifeat_ids
        self._icontrib0_ids = icontrib0_ids
        self._icontrib1m_ids = icontrib1m_ids
        self._icontrib1p_ids = icontrib1p_ids
        self._icontrib_ids = np.ascontiguousarray(
            np.concatenate([icontrib0_ids, icontrib1m_ids, icontrib1p_ids]),
            dtype=np.int32,
        )
        if len(self._icontrib_ids) == 0 and not has_vj:
            raise ValueError("Trying to initialize an empty collection")
        self._nbeta = len(self._icontrib_ids)
        if has_vj:
            self._nbeta += self._nalpha
        self._has_vj = has_vj
        self._integrals_initialized = False
        self._ccl = lib.c_null_ptr()
        libcider.generate_convolution_collection(
            ctypes.byref(self._ccl),
            self.atco_inp.atco_c_ptr,
            self.atco_out.atco_c_ptr,
            self._alphas.ctypes.data_as(ctypes.c_void_p),
            self._alpha_norms.ctypes.data_as(ctypes.c_void_p),
            self._icontrib_ids.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self._nalpha),
            ctypes.c_int(len(self._icontrib_ids)),
            ctypes.c_int(1 if self._has_vj else 0),
        )

    def __del__(self):
        """Free C data associated with this object."""
        libcider.free_convolution_collection(self._ccl)

    @property
    def n0(self):
        n0 = len(self._icontrib0_ids)
        if self._has_vj:
            n0 += self.nalpha
        return n0

    @property
    def n1(self):
        return len(self._icontrib1m_ids)

    @property
    def nalpha(self):
        """Number of inputs for the convolutions."""
        return self._nalpha

    @property
    def nbeta(self):
        """Total number of outputs of the convolutions."""
        return self._nbeta

    @property
    def num_out(self):
        return self.nbeta

    def compute_integrals_(self):
        """Compute and store the Gaussian integrals for
        performing convolutions."""
        if not self._integrals_initialized:
            libcider.generate_atc_integrals_all(self._ccl)
            self._integrals_initialized = True

    def solve_projection_coefficients(self):
        """Multiply convolution integral tensor by the inverse
        overlap matrices of the input and output bases."""
        if not self._integrals_initialized:
            self.compute_integrals_()
        libcider.solve_atc_coefs(self._ccl)

    def multiply_atc_integrals(self, input, output=None, fwd=True):
        """
        Perform the convolutions from the input orbital basis
        to the output orbital basis. If fwd is True, the input orbital
        basis is atco_inp and the output orbital basis is atco_out.
        If fwd is False, the input orbital basis is atco_out and the
        output orbital basis is atco_inp (i.e. the convolution is done
        in reverse.)

        Args:
            input (np.ndarray) : float64 array with shape
                (atco_inp, nalpha) for fwd=True and (atco_out, nbeta)
                for fwd=False
            output (np.ndarray, optional) : float64 array with shape
                (atco_out, nbeta) for fwd=True and (atco_inp, nalpha)
                for fwd=False. NOTE: If passed, should be all zeros.
                If None, output is initialized within the function
                and then returned.
            fwd (bool): Whether convolution is performed forward or backward.

        Returns:
            output
        """
        if output is None:
            if fwd:
                output = np.zeros((self.atco_out.nao, self.nbeta))
            else:
                output = np.zeros((self.atco_inp.nao, self.nalpha))
        if not self._integrals_initialized:
            raise RuntimeError("Must initialize integrals before calling")
        assert input.flags.c_contiguous
        assert output.flags.c_contiguous
        if fwd:
            assert input.shape == (self.atco_inp.nao, self.nalpha), (
                input.shape,
                (self.atco_inp.nao, self.nalpha),
            )
            assert output.shape == (self.atco_out.nao, self.nbeta), (
                output.shape,
                (self.atco_out.nao, self.nbeta),
            )
        else:
            assert output.shape == (self.atco_inp.nao, self.nalpha), (
                output.shape,
                (self.atco_inp.nao, self.nalpha),
            )
            assert input.shape == (self.atco_out.nao, self.nbeta), (
                input.shape,
                (self.atco_out.nao, self.nbeta),
            )
        libcider.multiply_atc_integrals(
            input.ctypes.data_as(ctypes.c_void_p),
            output.ctypes.data_as(ctypes.c_void_p),
            self._ccl,
            ctypes.c_int(1 if fwd else 0),
        )
        return output


class ConvolutionCollectionK(ConvolutionCollection):
    """
    Modified ConvolutionCollection for version k features. This is needed
    because multiplication by the ovlp_mats tensor in C is different
    for version k.
    """

    is_vk = True

    def __init__(self, atco_inp, atco_out, alphas, alpha_norms):
        """Simplified initializer for version K features.
        See ConvolutionCollection.__init__ for details. feat_id is the
        same as ifeat_ids except only one input is allowed instead of a list."""
        super(ConvolutionCollectionK, self).__init__(
            atco_inp, atco_out, alphas, alpha_norms, has_vj=False, ifeat_ids=[0]
        )

    @property
    def n0(self):
        return self.nalpha

    @property
    def num_out(self):
        return self.nalpha

    def multiply_atc_integrals(self, input, output=None, fwd=True):
        """
        Modified multiply_atc_integrals function for version k

        Args:
            input (np.ndarray) : float64 array with shape
                (atco_inp, nalpha) for fwd=True and (atco_out, nalpha)
                for fwd=False
            output (np.ndarray, optional) : float64 array with shape
                (atco_out, nalpha) for fwd=True and (atco_inp, nalpha)
                for fwd=False. NOTE: If passed, should be all zeros.
                If None, output is initialized within the function
                and then returned.
            fwd (bool): Whether convolution is performed forward or backward.

        Returns:
            output
        """
        if fwd:
            atco_inp = self.atco_inp
            atco_out = self.atco_out
        else:
            atco_inp = self.atco_out
            atco_out = self.atco_inp
        if output is None:
            output = np.zeros((atco_inp.nao, self.nalpha))
        if not self._integrals_initialized:
            raise RuntimeError("Must initialize integrals before calling")
        assert input.flags.c_contiguous
        assert output.flags.c_contiguous
        assert input.shape == (atco_inp.nao, self.nalpha)
        assert output.shape == (atco_out.nao, self.nalpha)
        libcider.multiply_atc_integrals_vk(
            input.ctypes.data_as(ctypes.c_void_p),
            output.ctypes.data_as(ctypes.c_void_p),
            self._ccl,
            ctypes.c_int(1 if fwd else 0),
        )
        return output
