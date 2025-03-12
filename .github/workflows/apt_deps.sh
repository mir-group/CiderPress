#!/usr/bin/env bash

if [ "$RUNNER_OS" == "Linux" ]
then
    sudo apt-get -qq install \
	gcc \
	libblas-dev \
	liblapack-dev \
	cmake \
	curl
elif [ "$RUNNER_OS" == "macOS" ]
then
    brew install \
        gcc@14 \
        libomp \
        openblas \
        lapack \
        cmake \
        curl \
        libxc \
        fftw \
        wget
else
    echo "$RUNNER_OS not supported"
    exit 1
fi
