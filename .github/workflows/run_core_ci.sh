#!/usr/bin/env bash
sh apt_deps.sh
pip install .
sh run_tests.sh
