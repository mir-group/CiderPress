#!/usr/bin/env bash

cp .github/workflows/nodef_condarc ~/.condarc
micromamba install mkl"<=2024.0" mkl-devel"<=2024.0" mkl-service"<=2024.0" mkl_fft mkl_random
micromamba install libxc conda-forge::fftw
if [ "$RUNNER_OS" == "Linux" ]
then
    pip3 install torch --index-url https://download.pytorch.org/whl/cu118
elif [ "$RUNNER_OS" == "macOS" ]
then
    pip3 install torch
else
    echo "$RUNNER_OS not supported"
    exit 1
fi
