#!/usr/bin/env bash
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
pip install .
