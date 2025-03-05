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

import sys

from pyscf import dft, gto

from ciderpress.pyscf.dft import make_cider_calc

# This is a simple example for running a PySCF calculation with
# a CIDER functional.
# This example runs a "surrogate" PBE0 calculation, where a CIDER exchange
# functional is used in place of exact exchange.
# It takes one command line argument, which is the functional to use.
# For example, you could use "CIDER23X_NL_MGGA_DTR" or "CIDER23X_SL_GGA".

mlfunc = "functionals/{}.yaml".format(sys.argv[1])

mol = gto.M(
    atom="F 0.0 0.0 0.0; F 0.0 0.0 1.42",
    basis="def2-tzvp",
)

ks = dft.RKS(mol)
ks = make_cider_calc(
    # KohnShamDFT object
    ks,
    # MappedXC, MappedXC2, or path to yaml/joblib file with one of these classes
    mlfunc,
    # Semi-local exchange and correlation parts
    xkernel="GGA_X_PBE",
    ckernel="GGA_C_PBE",
    # exact exchange mixing parameter
    xmix=0.25,
)
ks = ks.density_fit()
ks.with_df.auxbasis = "def2-universal-jfit"
ks.kernel()
