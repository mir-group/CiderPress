#!/usr/bin/env bash
sh .github/workflows/apt_deps.sh
pip install .
sh .github/workflows/run_tests.sh
