#!/usr/bin/env bash
sh .github/workflows/apt_deps.sh
pip install gpaw
gpaw install-data --sg15 --register $PWD
gpaw install-data --register $PWD
pip install .
python scripts/download_functionals.py
sh .github/workflows/run_gpaw_tests.sh
