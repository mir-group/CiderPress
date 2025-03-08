#!/usr/bin/env bash

if [ "$RUNNER_OS" == "Linux" ]; then
    sudo apt-get -qq install openmpi-bin openmpi-common libopenmpi-dev
    sudo apt-get -qq install libscalapack-openmpi-dev
elif [ "$RUNNER_OS" == "macOS" ]; then
    brew install open-mpi scalapack
else
    echo "$RUNNER_OS not supported"
    exit 1
fi
