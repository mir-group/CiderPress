#!/usr/bin/env bash
sh .github/workflows/ci_${RUNNER_OS}/apt_deps.sh
pip install pytest
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
CMAKE_CONFIGURE_ARGS="-DBUILD_LIBXC=on" pip install .
python scripts/download_functionals.py
sh .github/workflows/run_tests.sh
