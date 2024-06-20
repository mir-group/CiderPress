#!/usr/bin/env bash
export OMP_NUM_THREADS=1
export PYTHONPATH=$(pwd):$PYTHONPATH
ulimit -s 20000

version=$(python -c 'import sys; print("{0}.{1}".format(*sys.version_info[:2]))')
mpirun -np 2 --oversubscribe pytest ciderpress/gpaw -s -c pytest_mpi.ini ciderpress
