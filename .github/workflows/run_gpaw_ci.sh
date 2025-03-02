#!/usr/bin/env bash
sh .github/workflows/apt_deps.sh
pip install pytest
CMAKE_CONFIGURE_ARGS="-DBUILD_LIBXC=on" pip install .
python scripts/download_functionals.py

# Slightly hacky, link gpaw to our internal libxc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/ciderpress/lib/deps/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$PWD/ciderpress/lib/deps/include
pip install gpaw
gpaw install-data --sg15 --register $PWD
gpaw install-data --register $PWD

sh .github/workflows/run_gpaw_tests.sh
