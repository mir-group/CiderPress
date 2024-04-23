# CiderPress

Tools for training and evaluating CIDER functionals for use in Density Functional Theory calculations. Interfaces to the GPAW and PySCF codes are included.

## What is the CIDER formalism?

Machine Learning (ML) has recently gained attention as a means to fit more accurate Exchange-Correlation (XC) functionals for use in Density Functional Theory (DFT). We have developed CIDER, a set of features, models, and training techniques for efficiently learning the exchange energy, with an eye toward learning full XC functionals. **CIDER** stands for **C**ompressed scale-**I**nvariant **DE**nsity **R**epresentation, which refers to the fact that the descriptors are invariant under squishing or expanding of the density while maintaining its shape. This property makes it efficient for learning the XC functional, especially the exchange energy.

## WARNING: The CiderPress Code Base is Experimental

We want to make clear that both the code and the functionals themselves are experimental. The code base will likely change significantly in the next few years. Therefore, please read the installation guidance, usage instructions, examples, and known issues thoroughly before using CiderPress.

## Installation

We do not yet have a PyPI package for CiderPress, as the setup procedure and dependencies are a bit complex and might undergo significant changes when we release the full version. However, we have done our best to make installation fast and straightforward.
We recommend creating a conda environment from scratch for setting up CiderPressLite as described below, as this makes it much easier to quickly install compatible version of the dependencies of CiderPress, PySCF, and GPAW. In case you want to install using a different setup, here is a list of dependencies:
- Python 3.9-3.11 (Python 3.12 not supported)
- An installation of PySCF
- An installation of GPAW compiled with LibXC and FFTW (if you want to run periodic DFT calculations)
- Intel Math Kernel Library
- The Python package requirements in `requirements.txt`
- Fortran, C, and C++ compilers with OpenMP support.
One of the requirements in `requirements.txt` is the Intel Math Kernel Library. This module is particularly important because the CiderPress C extensions need to link to it and make use of the MKL DFTI library. Please make sure that either your MKL headers and shared libraries are in the `include` and `lib` directories of your Python environment, respectively, or that they are  in your `C_INCLUDE_PATH` and `LIBRARY_PATH`/`LD_LIBRARY_PATH`, respectively.

### Easy Installation with Conda, Micromamba, etc.

This section covers how to install CiderPress and its dependencies from a fresh conda environment. Micromamba is also supported; you will just need to replace the 'conda' commands with 'micromamba' below.

1. Create a new conda environment.
```bash
conda create -n <my_env> python=3.11
conda activate <my_env>
```
Python 3.9-3.11 (not 3.12) are supported. In later versions, we will tryo to move away from deprecated `distutils` tools so that we can support Python 3.12.

2. Install dependencies (three options).
**Option 1**: (Preferred option for multi-node jobs) If you provide your own MPI, C compiler, and Fortran compiler, you can install all the other dependencies using `nocomp_env.yml`:
```bash
conda env update --file <CiderPressLite>/nocomp_env.yaml
```
**Option 2**: The simplest way to do install everything is to use `full_env.yml`, which also installs the mpicc, gcc, g++, and gfortran compilers. Note that this approach will likely not work if you need to run MPI jobs on a cluster, as the conda openmpi will not be linked to the scheduler properly.
```bash
conda env update --file <CiderPressLite>/full_env.yaml
```
**Option 3**: As one more option, you can install components more manually. Start with the compilers and libxc. You can also use your own compilers, but this step helps ensure that everything is compatible. Note that all these compilers (plus libxc) must be installed via conda in one step, or else there will be compiler and library compatibility issues.
```bash
conda install openmpi-mpicc gfortran libxc conda-forge::fftw gxx gcc
```
You may exclude `openmpi-mpicc`, `fftw`, and `libxc` if you will not be using the GPAW interface. Next, Install MKL
```bash
conda install mkl mkl-devel mkl-include mkl_fft
```
Finally, install the other dependencies
```bash
cd <CiderPressLite>
pip install -r requirements.txt
```

3. Build C and Fortran extensions and install CiderPress
```bash
python setup.py build install
```

4. (If using GPAW) Install GPAW from source. We recommend using our siteconfig.py to link gpaw to MPI and MKL for simplicity and speed. (gitlab.com/gpaw/gpaw)
```bash
cd <place you want to save the GPAW source>
git clone https://gitlab.com/gpaw/gpaw.git
cd gpaw
cp <CiderPressLite>/gpaw_siteconfig.py .
python setup.py build install
```
**Note**: Currently CiderPress does not support the new GPAW version (gpaw.new), but we plan to support it in the future.

## How can I run a CIDER calculation?

CIDER calculations can be run in PySCF (for non-periodic, all-electron calculations) and GPAW (for periodic, plane-wave PAW calculations) using the functional initializers `ciderpress.dft.ri_cider.setup_cider_calc` and `ciderpress.gpaw.cider_paw.get_cider_functional`, respectively. Periodic PySCF calculations are not yet supported. See `examples/pyscf/simple_calc.py` and `examples/gpaw/simple_calc.py` for a demonstration of setting up a typical calculation, and refer to the docstrings of the initializers for a more detailed explanation of all the input options. As explained in the docstrings, the defaults are sufficient for most of the input options. The recommended functional for most applications in which the goal is to reproduce hybrid DFT is the NL-MGGA-DTR exchange functional (provided in a separate repository), which is a meta-GGA with nonlocal features of the density.

## How can I train a CIDER functional?

We are currently keeping the various scripts and workflow tools to train CIDER functionals in a separate, private code base. We plan to release this eventually, but it requires significantly more development before public release. CiderPress currently only contains the ML model classes themselves, but not the various training tools. If you are interested in training your own CIDER model, we suggest reaching out to us to discuss (email kylebystrom@gmail.com).

## Known Issues

CiderPress has a few known issues that we are currently investigating. Please be aware of these when attempting calculations with CIDER functionals. We will make a note and publish a new release when we fix these issues. If you run into any other problems, please post an issue on the Github repository.
* For some periodic systems in GPAW within the PAW formalism, significant numerical instability issues arise for the nonlocal functionals. In our experience thus far, these issues are uncommon and seem to be caused by the nonlocal PAW corrections to the CIDER features as opposed to the functionals themselves.
* For the GPAW interface, the memory overhead for the nonlocal features can be fairly high, occasionally causing memory issues. Please be aware that you might need to allocate more meory for a nonlocal CIDER calculation than for, say, a PBE calculation.
* For the PySCF interface, there are (mostly minor) convergence issues for some systems. These issues are much less common and less severe for our most robust functionals (like NL-MGGA-DTR). Even for NL-MGGA-DTR, occasionally a system will not quite converge. Usually the energy convergence is fine, but the orbital gradients are somewhat unstable; it might be necessary to set `conv_tol_grad` to a higher value than the default. These issues are likely a mix between inherent functional stability and the stability of the fast feature evaluation algorithm.
* The code spits out a lot of divide-by-zero and invalid value warnings from numpy, which occur because (as with many functionals) some terms in CIDER functionals become numerically unstable at very small densities. These issues are corrected by setting the XC energy and potential at very low density to zero, and we will clean up various warnings and unnecessary debug statements as soon as possible.
* The construction of the CIDER PAW corrections within GPAW have a very small numerical stability issue that results in different energies on different runs (with energy differences of roughly $10^{-11}$ eV). The difference is so small that it is insignificant for most applications, but it might affect finite difference calculations with very small perturbations.

## Questions and Comments

Find a bug? Areas of code unclearly documented? Other questions? Feel free to contact
Kyle Bystrom at kylebystrom@gmail.com AND/OR create an issue on the Github page at https://github.com/mir-group/CiderPressLite.
