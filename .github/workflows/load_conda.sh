# micromamba init
micromamba activate run_tests
export LD_LIBRARY_PATH=$(python .github/workflows/get_base.py)/lib:$LD_LIBRARY_PATH
