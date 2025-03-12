#!/usr/bin/env bash
./.github/workflows/apt_deps.sh
pip install pytest
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
if [ "$RUNNER_OS" == "macOS" ]; then
    export CC=gcc-14
    export CXX=g++-14
    export C_INCLUDE_PATH=$(brew --prefix)/include
    export LIBRARY_PATH=$(brew --prefix)/lib
    export LD_LIBRARY_PATH=$(brew --prefix)/lib
fi
export CMAKE_CONFIGURE_ARGS="-DBUILD_LIBXC=1 -DBUILD_FFTW=1 -DBUILD_WITH_MKL=0 -DBUILD_WITH_MPI=0 -DBUILD_MARCH_NATIVE=0"
pip install .
python scripts/download_functionals.py
./.github/workflows/run_tests.sh
