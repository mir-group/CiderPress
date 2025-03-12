#!/usr/bin/env bash
./.github/workflows/apt_deps.sh
pip install pytest
if [ "$RUNNER_OS" == "macOS" ]; then
    export CC=gcc-14
    export CXX=g++-14
    export C_INCLUDE_PATH=$(brew --prefix)/include
    export LIBRARY_PATH=$(brew --prefix)/lib
    export LD_LIBRARY_PATH=$(brew --prefix)/lib
fi
pip install .
python scripts/download_functionals.py

# Slightly hacky, link gpaw to our internal libxc
export CIDER_DEPS=$(python -c "import ciderpress, os; print(os.path.dirname(ciderpress.__file__))")/lib/deps
export LIBRARY_PATH=$LIBRARY_PATH:$CIDER_DEPS/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CIDER_DEPS/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$CIDER_DEPS/include
pip install gpaw
gpaw install-data --sg15 --register $PWD
gpaw install-data --register $PWD

./.github/workflows/run_gpaw_tests.sh
