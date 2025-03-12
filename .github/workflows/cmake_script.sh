#!/usr/bin/env bash

export MY_PYTHON_LIBDIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
export MY_PYTHON_INCDIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')

echo $MY_PYTHON_LIBDIR $MY_PYTHON_INCDIR

cmake -DCMAKE_PYTHON_LIBRARY_PATH=$MY_PYTHON_LIBDIR \
	-DCMAKE_PYTHON_INCLUDE_PATH=$MY_PYTHON_INCDIR \
	-DCMAKE_BUILD_TYPE=Release \
	-DBLA_VENDOR=Intel10_64lp_seq -DBUILD_FFTW=1 ..
make
