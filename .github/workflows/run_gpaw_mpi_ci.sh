#!/usr/bin/env bash
./.github/workflows/apt_deps.sh
./.github/workflows/mpi_apt_deps.sh

if [ "$RUNNER_OS" == "macOS" ]; then
    export CC=gcc-14
    export CXX=g++-14
fi

CMAKE_CONFIGURE_ARGS="-DBUILD_LIBXC=on" pip install .
python scripts/download_functionals.py

# Slightly hacky, link gpaw to our internal libxc
export CIDER_DEPS=$(python -c "import ciderpress, os; print(os.path.dirname(ciderpress.__file__))")/lib/deps
export LIBRARY_PATH=$LIBRARY_PATH:$CIDER_DEPS/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CIDER_DEPS/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$CIDER_DEPS/include
export CIDERDIR=$PWD
cd ..
git clone https://gitlab.com/gpaw/gpaw.git
cd gpaw
cp $CIDERDIR/.github/workflows/gpaw_blas_siteconfig.py siteconfig.py
pip install .
cd ..
gpaw install-data --sg15 --register $PWD
gpaw install-data --register $PWD
cd $CIDERDIR

./.github/workflows/run_gpaw_tests.sh
