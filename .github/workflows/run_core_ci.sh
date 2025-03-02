#!/usr/bin/env bash
sh .github/workflows/apt_deps.sh
pip install pytest
pip install .
python scripts/download_functionals.py
sh .github/workflows/run_tests.sh
