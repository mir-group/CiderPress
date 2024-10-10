# CiderPress

Tools for training and evaluating CIDER functionals for use in Density Functional Theory calculations. Interfaces to the GPAW and PySCF codes are included.

## What is the CIDER formalism?

Machine Learning (ML) has recently gained attention as a means to fit more accurate Exchange-Correlation (XC) functionals for use in Density Functional Theory (DFT). We have developed CIDER, a set of features, models, and training techniques for efficiently learning the exchange and correlation functionals. **CIDER** stands for **C**ompressed scale-**I**nvariant **DE**nsity **R**epresentation, which refers to the fact that the descriptors are invariant under squishing or expanding of the density while maintaining its shape. This property makes it efficient for learning the XC functional, especially the exchange energy.

## **WARNING**: The CiderPress Code Base is Experimental

We want to make clear that both the code and the functionals themselves are experimental. The code base will likely change significantly in the next few years. Therefore, please read the installation guidance, usage instructions, examples, and known issues thoroughly before using CiderPress.

## Installation

We do not yet have a PyPI package for CiderPress, as the setup procedure and dependencies are complex and a work in progress. We are currently working on updating the build environment to enable simple installation through pip/conda/mamba.
In the meantime, we have done our best to make installation from source work smoothly.
We recommend creating a conda environment from scratch for setting up CiderPress as described below, as this makes it much easier to quickly install compatible version of the dependencies of CiderPress, PySCF, and GPAW. In case you want to install using a different setup, here is a list of dependencies that are not Python packages:
- Python 3.9-3.12
- Intel Math Kernel Library (MKL)
- Libxc
- C and C++ compilers with OpenMP support.
If you have conda, you can install the MKL using
```
conda install mkl"<=2024.0" mkl-devel"<=2024.0" mkl-service"<=2024.0" mkl_fft mkl_random
```
The `<=2024.0` is to fix a compatibility issue with PyTorch and MKL, so you can remove it if you don't need PyTorch (i.e. if you don't want to use CIDER24X models). In principle, it is also possible to pip install the MKL dependencies, but we have had trouble getting the libraries to link. Please make sure MKL and Libxc can be found by cmake before building.

Once you have those dependencies set up, you can install CiderPress simply by
```
cd <CiderPress>
pip install .
```
The pip command will install all the other Python dependencies, including the quantum chemistry package PySCF.

To use previously developed functionals, you can download them using
```
cd <CiderPress>
python scripts/download_functionals.py
```
See the `examples` directory for details on how to load and use the functionals.

If you want to run plane-wave DFT calculations, you must also install GPAW with LibXC and FFTW. GPAW uses a `siteconfig.py` file to customize the libraries it links to. This repo's `.github/workflows/gpaw_siteconfig.py` could be useful for compiling GPAW with MKL, LibXC, and FFTW.

**NOTE**: GPAW uses MPI for parallelization, and the CiderPress extensions must also link to MPI to run parallel GPAW calculations. Make sure cmake can find OpenMPI or an equivalent installation and that you have a working `mpicc` compiler before building CiderPress and GPAW together.

**NOTE**: The CiderPress C extensions must use the same OpenMP as PySCF and GPAW, otherwise you will run into parallelization issues and code crashes. `gpaw_siteconfig.py` and `ciderpress/lib/CMakeLists.txt` assume Intel's `iomp5` as the OpenMP library by default. If you are using GNU OpenMP, you should change `iomp5` to `gomp` in `gpaw_siteconfig.py` and change
```
set(MKL_INTERFACE lp64)
```
to
```
set(MKL_INTERFACE lp64)
set(MKL_THREADING gnu_thread)
```
in `ciderpress/lib/CMakeLists.txt`. We are working on making this build setup more portable so the user doesn't have to figure this out by hand.

**NOTE**: To run the CIDER24X functionals, you also need to install pytorch.

### Easy Installation with Conda, Micromamba, etc.

This section covers how to install CiderPress and its dependencies from a fresh conda environment. Micromamba is also supported; you will just need to replace the 'conda' commands with 'micromamba' below.

0. Make sure you have a C compiler installed. (Or you can install one through conda after creating your environment in step 1.)

1. Create a new conda environment.
```bash
conda create -n <my_env> python=3.11
conda activate <my_env>
```
Python 3.9-3.12 are supported.

2. Install dependencies. The scripts `.github/workflows/mm_install_torch.sh` and `.github/workflows/mm_install_mpi.h` can both be used to set up an environment for running CIDER calculations. `mm_install_torch.sh` installs MKL, libxc, FFTW, and pytorch, so it is useful if you want to run calculations with CIDER24X functionals, which require pytorch. `mm_install_mpi.sh` installs MKL, libxc, FFTW, OpenMPI, and mpicc, so it is useful if you want to run GPAW calculations. Note that the conda MPI installation might not work well for multi-node jobs on clusters, so you might want to use your own MPI/mpicc instead if that is your use case. Single-node jobs should work fine with conda's MPI.

3. Build C extensions and install CiderPress.
```bash
pip install .
```

4. (If using GPAW) Install GPAW from source. We recommend using our `siteconfig.py` to link gpaw to MPI and MKL for simplicity and speed. (You can download GPAW at gitlab.com/gpaw/gpaw.)
```bash
cd <place you want to save the GPAW source>
git clone https://gitlab.com/gpaw/gpaw.git
cd gpaw
cp <CiderPress>/.github/workflows/gpaw_siteconfig.py .
python setup.py build install
```
**Note**: Currently CiderPress does not support the new GPAW version (`gpaw.new`), but we plan to support it in the future.

## How can I run a CIDER calculation?

CIDER calculations can be run in PySCF (for non-periodic, all-electron calculations) and GPAW (for periodic, plane-wave PAW calculations) using the functional initializers `ciderpress.pyscf.dft.make_cider_calc` and `ciderpress.gpaw.calculator.get_cider_functional`, respectively. Periodic PySCF calculations are not yet supported, except for the CIDER24X functionals with uniform grids and pseudopotentials. See `examples/pyscf/simple_calc.py` and `examples/gpaw/simple_calc.py` for a demonstration of setting up a typical calculation, and refer to the docstrings of the initializers for a more detailed explanation of all the input options. As explained in the docstrings, the defaults are sufficient for most of the input options. The recommended functional for most applications in which the goal is to reproduce hybrid DFT is the `CIDER23X_NL_MGGA_DTR` exchange functional, which is a meta-GGA with nonlocal features of the density.
The more recent `CIDER24Xne` and `CIDER24Xe` functionals use more powerful descriptors and are therefore more accurate than any of the `CIDER23X` functionals, but they are also more expensive and only available for use in PySCF. `CIDER24Xe` is fit to molecular HOMO-LUMO gaps, so it might be useful for properties where band gaps are important.

## How can I train a CIDER functional?

The basic ML training framework for CiderPress is stored in `ciderpress.models`. You are free to use these tools if you find them helpful, but they are not yet documented or prepared for widespread use. Also, CiderPress currently only contains the ML model classes themselves, but not the various training tools. If you are interested in training your own CIDER model, we suggest reaching out to us to discuss (email kylebystrom@gmail.com).

## Known Issues

CiderPress has a few known issues that we are currently investigating. Please be aware of these when attempting calculations with CIDER functionals. We will make a note and publish a new release when we fix these issues. If you run into any other problems, please post an issue on the Github repository.
* For some periodic systems in GPAW within the PAW formalism, significant numerical instability issues arise for the nonlocal functionals. In our experience thus far, these issues are uncommon and seem to be caused by the nonlocal PAW corrections to the CIDER features as opposed to the functionals themselves.
* For the GPAW interface, the memory overhead for the nonlocal features can be fairly high, occasionally causing memory issues. Please be aware that you might need to allocate more memory for a nonlocal CIDER calculation than for, say, a PBE calculation.
* For the PySCF interface, there are (mostly minor) convergence issues for some systems. These issues are much less common and less severe for our most robust functionals (like NL-MGGA-DTR). Even for NL-MGGA-DTR, occasionally a system will not quite converge. Usually the energy convergence is fine, but the orbital gradients are somewhat unstable; it might be necessary to set `conv_tol_grad` to a higher value than the default. These issues are likely a mix between inherent functional stability and the stability of the fast feature evaluation algorithm.
* The code spits out a lot of divide-by-zero and invalid value warnings from numpy, which occur because (as with many functionals) some terms in CIDER functionals become numerically unstable at very small densities. These issues are corrected by setting the XC energy and potential at very low density to zero, and we will clean up various warnings and unnecessary debug statements as soon as possible.
* The construction of the CIDER PAW corrections within GPAW have a very small numerical stability issue that results in different energies on different runs (with energy differences of roughly $10^{-11}$ eV). The difference is so small that it is insignificant for most applications, but it might affect finite difference calculations with very small perturbations.

## Questions and Comments

Find a bug? Areas of code unclearly documented? Other questions? Feel free to contact
Kyle Bystrom at kylebystrom@gmail.com AND/OR create an issue on the Github page at https://github.com/mir-group/CiderPress.

## Citing

If you find CiderPress or CIDER functionals useful in your research, please cite the following article
```
@article{PhysRevB.110.075130,
  title = {Nonlocal machine-learned exchange functional for molecules and solids},
  author = {Bystrom, Kyle and Kozinsky, Boris},
  journal = {Phys. Rev. B},
  volume = {110},
  issue = {7},
  pages = {075130},
  numpages = {30},
  year = {2024},
  month = {Aug},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.110.075130},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.110.075130}
}
```
The above article introduces the CIDER23X functionals and much of the algorithms in CiderPress. If you use the CIDER24X functionals, please also cite
```
@article{doi:10.1021/acs.jctc.4c00999,
  author = {Bystrom, Kyle and Falletta, Stefano and Kozinsky, Boris},
  title = {Training Machine-Learned Density Functionals on Band Gaps},
  journal = {Journal of Chemical Theory and Computation},
  volume = {20},
  number = {17},
  pages = {7516-7532},
  year = {2024},
  doi = {10.1021/acs.jctc.4c00999},
  note ={PMID: 39178337},
  URL = {https://doi.org/10.1021/acs.jctc.4c00999},
  eprint = {https://doi.org/10.1021/acs.jctc.4c00999}
}
```
