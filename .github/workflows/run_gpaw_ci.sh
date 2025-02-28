#!/usr/bin/env bash
sh apt_deps.sh
pip install gpaw
gpaw install-data --sg15 --register $PWD
gpaw install-data --register $PWD
pip install .
sh run_tests.sh
