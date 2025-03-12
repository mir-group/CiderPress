#!/usr/bin/env bash
export OMP_NUM_THREADS=1
export PYTHONPATH=$(pwd):$PYTHONPATH
ulimit -s 20000

version=$(python -c 'import sys; print("{0}.{1}".format(*sys.version_info[:2]))')
mpirun -np 2 --oversubscribe gpaw python -m unittest ciderpress.gpaw.tests.test_basic_calc
mpirun -np 2 --oversubscribe gpaw python -m unittest ciderpress.gpaw.tests.test_si_force
mpirun -np 2 --oversubscribe gpaw python -m unittest ciderpress.gpaw.tests.test_si_stress
