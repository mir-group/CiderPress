#!/usr/bin/env bash

cp .github/workflows/nodef_condarc ~/.condarc
micromamba install mkl mkl-devel mkl-service mkl_fft mkl_random
micromamba install openmpi openmpi-mpicc
micromamba install libxc conda-forge::fftw
