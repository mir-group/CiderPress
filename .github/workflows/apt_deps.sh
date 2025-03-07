#!/usr/bin/env bash

if [ "$RUNNER_OS" == "Linux" ]
then
    echo "HI"
    sudo apt-get -qq install \
	gcc \
	libblas-dev \
	liblapack-dev \
	cmake \
	curl
elif [ "$RUNNER_OS" == "macOS" ]
then
    brew install gcc \
        libomp \
        openblas \
        lapack \
        cmake \
        curl
else
    echo "$RUNNER_OS not supported"
    exit 1
fi
