#!/usr/bin/env bash
sh .github/workflows/apt_deps.sh
sudo apt-get -qq install openmpi-bin openmpi-common libopenmpi-dev
sudo apt-get -qq install libscalapack-openmpi-dev
pip install gpaw

export CIDERDIR=$PWD
cd ..
git clone https://gitlab.com/gpaw/gpaw.git
cd gpaw
cp $CIDERDIR/.github/workflows/gpaw_blas_siteconfig.py siteconfig.py
pip install .
cd ..
gpaw install-data --sg15 --register $PWD
gpaw install-data --register $PWD
cd $CIDERDIR

CMAKE_CONFIGURE_ARGS="-DBUILD_LIBXC=on" pip install .
python scripts/download_functionals.py
sh .github/workflows/run_gpaw_tests.sh
