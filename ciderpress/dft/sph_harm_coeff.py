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
from sympy.physics.wigner import clebsch_gordan

from ciderpress.lib import load_library as load_cider_library

libcider = load_cider_library("libmcider")


def change_basis_real_to_complex(l: int) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / np.sqrt(2)
        q[l + m, l - abs(m)] = -1j / np.sqrt(2)
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / np.sqrt(2)
        q[l + m, l - abs(m)] = 1j * (-1) ** m / np.sqrt(2)
    # Added factor of 1j**l to make the Clebsch-Gordan coefficients real
    return (-1j) ** l * q


def su2_clebsch_gordan(l1, l2, l3):
    mat = np.zeros(
        [
            2 * l1 + 1,
            2 * l2 + 1,
            2 * l3 + 1,
        ]
    )
    for m1 in range(2 * l1 + 1):
        for m2 in range(2 * l2 + 1):
            for m3 in range(2 * l3 + 1):
                mat[m1, m2, m3] = clebsch_gordan(l1, l2, l3, m1 - l1, m2 - l2, m3 - l3)
    return mat


def clebsch_gordan_e3nn(l1: int, l2: int, l3: int) -> np.ndarray:
    r"""
    The Clebsch-Gordan coefficients of the real irreducible representations of :math:`SO(3)`.
    This function taken from https://github.com/e3nn/e3nn

    Args:
        l1 (int): the representation order of the first irrep
        l2 (int): the representation order of the second irrep
        l3 (int): the representation order of the third irrep

    Returns:
        np.ndarray: the Clebsch-Gordan coefficients
    """
    C = su2_clebsch_gordan(l1, l2, l3)
    Q1 = change_basis_real_to_complex(l1)
    Q2 = change_basis_real_to_complex(l2)
    Q3 = change_basis_real_to_complex(l3)
    C = np.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, np.conj(Q3.T), C)

    assert np.all(np.abs(np.imag(C)) < 1e-5)
    return np.real(C)


def get_deriv_ylm_coeff(lmax):
    nlm = (lmax + 1) * (lmax + 1)
    gaunt_coeff = np.zeros((5, nlm))
    for l in range(lmax + 1):
        fe = clebsch_gordan_e3nn(l, 1, l + 1)
        for im in range(2 * l + 1):
            lm = l * l + im
            m = abs(im - l)
            l1m = l + 1 - m
            fac = ((2 * l + 3) * (l + 1)) ** 0.5
            gaunt_coeff[0, lm] = fac * fe[im, 2, im]
            gaunt_coeff[1, lm] = fac * fe[im, 2, im + 2]
            gaunt_coeff[2, lm] = fac * fe[im, 0, 2 * l - im]
            gaunt_coeff[3, lm] = fac * fe[im, 0, 2 * l - im + 2]
            gaunt_coeff[4, lm] = np.sqrt(
                ((2.0 * l + 3) * l1m * (l + 1 + m) / (2 * l + 1))
            )
    return gaunt_coeff
