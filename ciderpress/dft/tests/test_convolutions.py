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

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from pyscf import gto
from pyscf.gto.mole import ANG_OF, ATOM_OF, PTR_COEFF, PTR_EXP

from ciderpress.dft.lcao_convolutions import (
    ATCBasis,
    ConvolutionCollection,
    ConvolutionCollectionK,
    get_convolution_expnts_from_expnts,
)
from ciderpress.pyscf.nldf_convolutions import (
    aug_etb_for_cider,
    get_gamma_lists_from_mol,
)


class TestC(unittest.TestCase):
    def test_initialize_atc(self):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvpd")
        basis = aug_etb_for_cider(mol)[0]
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis=basis)
        dat = get_gamma_lists_from_mol(mol)
        atco = ATCBasis(*dat)
        nbas = atco.nbas
        assert nbas == mol.nbas
        bas = atco.bas
        env = atco.env
        assert_equal(bas[:, ANG_OF], mol._bas[:, ANG_OF])
        assert_equal(bas[:, ATOM_OF], mol._bas[:, ATOM_OF])
        assert_almost_equal(env[bas[:, PTR_EXP]], mol._env[mol._bas[:, PTR_EXP]], 14)
        assert_almost_equal(
            env[bas[:, PTR_COEFF]], mol._env[mol._bas[:, PTR_COEFF]], 14
        )

    def test_initialize_ccl(self):
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="def2-tzvpd")
        basis = aug_etb_for_cider(mol)[0]
        mol = gto.M(atom="H 0 0 0; F 0 0 0.9", basis=basis)
        dat = get_gamma_lists_from_mol(mol)
        atco_inp = ATCBasis(*dat)
        gammas = 0.01 * 2.8 ** np.arange(10)
        new_dat = get_convolution_expnts_from_expnts(
            gammas, dat[0], dat[1], dat[2], dat[4], gbuf=4.0
        )
        atco_out = ATCBasis(*new_dat)
        alphas = gammas / 2
        input = np.zeros((atco_inp.nao, alphas.size))
        np.random.seed(42)
        input += np.abs(np.random.uniform(size=input.shape))
        alphas_norms = (np.pi / (2 * alphas)) ** -0.75
        ccl = ConvolutionCollection(
            atco_inp,
            atco_out,
            alphas,
            alphas_norms,
            has_vj=True,
            ifeat_ids=[0],
        )
        ccl.compute_integrals_()
        output = ccl.multiply_atc_integrals(input, fwd=True)
        assert (output > 0).all()

        alpha_big = 1e32
        ccl = ConvolutionCollection(
            atco_inp,
            atco_inp,
            [alpha_big],
            [(np.pi / (2 * alpha_big)) ** -0.75],
            has_vj=True,
        )
        ccl.compute_integrals_()
        input = np.zeros((atco_inp.nao, 1))
        ovlp = mol.intor("int1e_ovlp_sph")
        cond = mol._bas[:, ATOM_OF] == 0
        nao_split = np.sum(2 * mol._bas[cond, ANG_OF] + 1)
        input += np.random.uniform(size=input.shape)
        output_ref = np.append(
            ovlp[:nao_split, :nao_split].dot(input[:nao_split, 0]),
            ovlp[nao_split:, nao_split:].dot(input[nao_split:, 0]),
        )
        output = ccl.multiply_atc_integrals(input, fwd=True)
        assert_almost_equal(output[:, 0], output_ref)

        ccl.solve_projection_coefficients()
        input0 = input.copy()
        input = np.append(
            ovlp[:nao_split, :nao_split].dot(input0[:nao_split]),
            ovlp[nao_split:, nao_split:].dot(input0[nao_split:]),
        )[:, None]
        output = ccl.multiply_atc_integrals(input, fwd=True)
        assert_almost_equal(output, input0)

        alpha_big = 1e32
        ccl = ConvolutionCollectionK(
            atco_inp,
            atco_inp,
            [alpha_big],
            [(np.pi / (alpha_big)) ** -1.5],
        )
        ccl.compute_integrals_()
        ccl.solve_projection_coefficients()
        input0 = input.copy()
        input = np.append(
            ovlp[:nao_split, :nao_split].dot(input0[:nao_split]),
            ovlp[nao_split:, nao_split:].dot(input0[nao_split:]),
        )[:, None]
        output = ccl.multiply_atc_integrals(input, fwd=True)
        assert_almost_equal(output, input0, 6)

        input = np.zeros((atco_inp.nao, alphas.size))
        input += np.abs(np.random.uniform(size=input.shape))
        ccl = ConvolutionCollection(
            atco_inp,
            atco_out,
            alphas,
            alphas_norms,
            has_vj=False,
            ifeat_ids=[7],
        )
        ccl.compute_integrals_()
        ccl.solve_projection_coefficients()
        output = ccl.multiply_atc_integrals(input, fwd=True)
        assert np.isfinite(output).all()

    def test_convert_rad2orb_(self):
        atom = "Ne"
        mol = gto.M(atom=atom, basis="def2-tzvpd")
        basis = aug_etb_for_cider(mol, lmax=10, beta=2.8)[0]
        mol = gto.M(atom=atom, basis=basis)
        lmax = np.max(mol._bas[:, ANG_OF])
        nlm = (lmax + 1) * (lmax + 1)
        dat = get_gamma_lists_from_mol(mol)
        atco = ATCBasis(*dat)
        nbas = atco.nbas
        assert nbas == mol.nbas
        nq = 3
        nrad = 650
        aparam = 0.02
        dparam = 0.02
        rad = aparam * (np.exp(dparam * np.arange(nrad)) - 1)
        drad = rad**2 * aparam * dparam * np.exp(dparam * np.arange(nrad))
        theta_rlmq = np.zeros((nrad, nlm, nq))
        bas_list = [
            (6, 1, 0),
            (46, 1, 4),
            (10, 2, 0),
            (46, 0, 8),
            (70, 1, 10),
            (70, 1, 20),
        ]
        p_uq_ref = np.zeros((mol.nao_nr(), nq))
        ao_loc = mol.ao_loc_nr()
        for ibas, ind, m in bas_list:
            l = mol._bas[ibas, ANG_OF]
            coef = mol._env[mol._bas[ibas, PTR_COEFF]]
            expnt = mol._env[mol._bas[ibas, PTR_EXP]]
            theta_rlmq[:, l * l + m, ind] += (
                coef * rad**l * np.exp(-expnt * rad**2) * drad
            )
            p_uq_ref[ao_loc[ibas] + m, ind] += 1
        p_uq = np.zeros((mol.nao_nr(), nq))
        pbig_uq = np.zeros((mol.nao_nr(), nq + 5))
        loc = np.asarray([0, len(rad)], dtype=np.int32, order="C")
        atco.convert_rad2orb_(theta_rlmq, p_uq, loc, rad, True)
        atco.convert_rad2orb_(theta_rlmq, pbig_uq, loc, rad, True, offset=3)
        assert_equal(pbig_uq[:, :3], np.zeros_like(pbig_uq[:, :3]))
        assert_equal(pbig_uq[:, -2:], np.zeros_like(pbig_uq[:, -2:]))
        assert_almost_equal(pbig_uq[:, 3:-2], p_uq, 14)
        ovlp = mol.intor("int1e_ovlp_sph")
        p_uq = np.linalg.solve(ovlp, p_uq)
        assert_almost_equal(p_uq, p_uq_ref, 7)
        theta_test = np.zeros_like(theta_rlmq)
        theta_test2 = np.zeros_like(theta_rlmq)
        loc = np.zeros(len(rad), dtype=np.int32)
        np.random.seed(102)
        pbig_uq[:, 3:-2] = p_uq
        pbig_uq[:, :3] = np.random.normal(size=pbig_uq[:, :3].shape)
        pbig_uq[:, -2:] = np.random.normal(size=pbig_uq[:, -2:].shape)
        atco.convert_rad2orb_(theta_test, p_uq, loc, rad, False)
        atco.convert_rad2orb_(theta_test2, pbig_uq, loc, rad, False, offset=3)
        assert_almost_equal(theta_test2, theta_test, 14)
        theta_rlmq[:] = 0
        for ibas, ind, m in bas_list:
            l = mol._bas[ibas, ANG_OF]
            coef = mol._env[mol._bas[ibas, PTR_COEFF]]
            expnt = mol._env[mol._bas[ibas, PTR_EXP]]
            theta_rlmq[:, l * l + m, ind] += coef * rad**l * np.exp(-expnt * rad**2)
        assert_almost_equal(theta_test, theta_rlmq, 5)


if __name__ == "__main__":
    unittest.main()
