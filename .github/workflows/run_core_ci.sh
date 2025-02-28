#!/usr/bin/env bash
sh apt_deps.sh
pip install .
sh .github/workflows/run_tests.sh
