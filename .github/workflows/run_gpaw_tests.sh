#!/usr/bin/env bash
export OMP_NUM_THREADS=1
export PYTHONPATH=$(pwd):$PYTHONPATH
ulimit -s 20000

version=$(python -c 'import sys; print("{0}.{1}".format(*sys.version_info[:2]))')
python -m unittest ciderpress.gpaw.tests.test_descriptors.TestDescriptors.test_eigval_vs_fd
pytest ciderpress/gpaw -s -c pytest_mpi.ini ciderpress
