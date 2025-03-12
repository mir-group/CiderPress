#!/usr/bin/env bash
export OMP_NUM_THREADS=2
export PYTHONPATH=$(pwd):$PYTHONPATH
ulimit -s 20000

# See https://github.com/pytest-dev/pytest/issues/1075
version=$(python -c 'import sys; print("{0}.{1}".format(*sys.version_info[:2]))')
PYTHONHASHSEED=0 pytest ciderpress/ -s -c pytest.ini ciderpress
