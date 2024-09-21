micromamba install mkl mkl-devel mkl-service mkl_fft mkl_random
micromamba install openmpi openmpi-mpicc
micromamba install libxc conda-forge::fftw
micromamba install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
# TODO below can be removed once next PySCF released
pip install git+https://github.com/pyscf/pyscf/commit/943d39a970a3c30ce6cd73ac3c5bf386155d388e
