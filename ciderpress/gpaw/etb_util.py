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
from scipy.linalg import cholesky
from scipy.linalg.lapack import dtrtri

from ciderpress.dft.sphere_util import gauss_fft, gauss_real
from ciderpress.gpaw.fit_paw_gauss_pot import get_dv
from ciderpress.gpaw.gpaw_grids import SBTFullGridDescriptor
from ciderpress.pyscf.nldf_convolutions import aug_etb_for_cider


class ETBProjector:
    def __init__(self, Z, rgd=None, kgd=None, beta=1.8, alpha_min=None):
        Z = int(Z)
        spin = 0 if Z % 2 == 0 else 1
        tmp_mol = gto.M(atom=str(Z), basis="ano-rcc", spin=spin)
        basis = aug_etb_for_cider(
            tmp_mol, beta=beta, upper_fac=10, lower_fac=0.25, lmax=5
        )
        basis = list(basis.values())[0]
        expnt_dict = {}
        coeff_dict = {}
        self.bas_l_jg = {}
        self.bas_l_jk = {}
        lp1 = 5
        self.lp1 = lp1
        for l in range(lp1):
            expnt_dict[l] = []
        for l, lst in basis:
            if l < lp1:
                expnt_dict[l].append(lst[0])
        if alpha_min is not None:
            for l in range(lp1):
                mini = np.min(expnt_dict[l])
                while mini > alpha_min:
                    expnt_dict[l].append(mini / beta)
                    mini = np.min(expnt_dict[l])
        for l in range(lp1):
            expnt_dict[l] = np.array(expnt_dict[l], dtype=np.float64)
            coeff_dict[l] = gto.mole.gto_norm(l, expnt_dict[l])
            ovlp = gto.mole.gaussian_int(
                l * 2 + 2, expnt_dict[l] + expnt_dict[l][:, None]
            )
            ovlp *= coeff_dict[l] * coeff_dict[l][:, None]
            L = cholesky(ovlp, lower=True)
            Linv = np.tril(dtrtri(L, lower=True)[0])
            if rgd is not None:
                gauss_jg = coeff_dict[l][:, None] * gauss_real(
                    l, expnt_dict[l][:, None], rgd.r_g
                )
                # bas_jg = solve_triangular(L, gauss_jg, lower=True, trans=0)
                bas_jg = Linv.dot(gauss_jg)
                self.bas_l_jg[l] = bas_jg
                np.einsum("ig,jg,g->ij", bas_jg, bas_jg, rgd.r_g**2 * rgd.dr_g)
                # print('R', np.linalg.norm(normg-np.identity(normg.shape[0])))
            if kgd is not None:
                gauss_jk = coeff_dict[l][:, None] * gauss_fft(
                    l, expnt_dict[l][:, None], kgd.k_g
                )
                # bas_jk = solve_triangular(L, gauss_jk, lower=True, trans=0)
                bas_jk = Linv.dot(gauss_jk)
                self.bas_l_jk[l] = bas_jk
        self.rgd = rgd
        self.kgd = kgd
        self.expnt_dict = expnt_dict
        self.coeff_dict = coeff_dict

    def r2k(self, f_xLg):
        f_xLg.shape[-2]
        assert f_xLg.shape[-1] == self.rgd.r_g.size
        kshape = f_xLg.shape[:-1] + (self.kgd.k_g.size,)
        f_xLk = np.zeros(kshape)
        dv_g = get_dv(self.rgd)
        for l in range(self.lp1):
            LMIN = l * l
            LMAX = LMIN + 2 * l + 1
            f_xLj = np.einsum(
                "...Lg,jg,g->...Lj", f_xLg[..., LMIN:LMAX, :], self.bas_l_jg[l], dv_g
            )
            f_xLk[..., LMIN:LMAX, :] = np.einsum(
                "...Lj,jk->...Lk", f_xLj, self.bas_l_jk[l]
            )
        return f_xLk

    def k2r(self, f_xLk):
        f_xLk.shape[-2]
        assert f_xLk.shape[-1] == self.kgd.k_g.size
        gshape = f_xLk.shape[:-1] + (self.rgd.r_g.size,)
        f_xLg = np.zeros(gshape)
        dv_k = self.kgd.k_g**2 * self.kgd.dk_g
        for l in range(self.lp1):
            LMIN = l * l
            LMAX = LMIN + 2 * l + 1
            f_xLj = np.einsum(
                "...Lk,jk,k->...Lj", f_xLk[..., LMIN:LMAX, :], self.bas_l_jk[l], dv_k
            )
            f_xLg[..., LMIN:LMAX, :] = np.einsum(
                "...Lj,jg->...Lg", f_xLj, self.bas_l_jg[l]
            )
        return f_xLg


if __name__ == "__main__":
    rgd = SBTFullGridDescriptor(1e-4, 1e6, 0.015, N=1024)
    ETBProjector(1, rgd=rgd, kgd=rgd)
