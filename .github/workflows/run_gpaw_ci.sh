#!/usr/bin/env bash
sh .github/workflows/ci_${RUNNER_OS}/apt_deps.sh
pip install pytest
CMAKE_CONFIGURE_ARGS="-DBUILD_LIBXC=on" pip install .
python scripts/download_functionals.py

# Slightly hacky, link gpaw to our internal libxc
export CIDER_DEPS=$(python -c "import ciderpress, os; print(os.path.dirname(ciderpress.__file__))")/lib/deps
export LIBRARY_PATH=$LIBRARY_PATH:$CIDER_DEPS/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CIDER_DEPS/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$CIDER_DEPS/include
pip install gpaw
gpaw install-data --sg15 --register $PWD
gpaw install-data --register $PWD

sh .github/workflows/run_gpaw_tests.sh
