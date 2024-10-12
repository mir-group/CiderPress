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

import os

scalapack = False
mklpath = os.environ["CONDA_PREFIX"]
omppath = os.environ["CONDA_PREFIX"]

mpi = False
fftw = True
compiler = "gcc"

libraries = [
    "fftw3",
    "fftw3_omp",
    "xc",
    "mkl_intel_lp64",
    "mkl_intel_thread",
    "mkl_core",
    "iomp5",
    "pthread",
]
library_dirs += [f"{omppath}/lib", f"{mklpath}/lib", f"{mklpath}/lib/intel64"]
include_dirs += [f"{omppath}/include", f"{mklpath}/include"]

extra_compile_args += ["-O3", "-std=c99", "-w", "-m64"]

define_macros += [
    ("GPAW_NO_UNDERSCORE_CBLACS", "1"),
    ("GPAW_NO_UNDERSCORE_CSCALAPACK", "1"),
    ("GPAW_NO_UNDERSCORE_SCALAPACK", "1"),
]
