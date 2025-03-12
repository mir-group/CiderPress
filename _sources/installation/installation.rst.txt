.. highlight:: bash

Overview and Installation Instructions
======================================

CiderPress is a Python/C package that provides tools for training and evaluating CIDER Exchange-Correlation (XC) functionals for use in Density Functional Theory calculations. Interfaces to the `GPAW <https://gpaw.readthedocs.io/>`_ and `PySCF <https://pyscf.org/>`_ codes are included.

What is the CIDER formalism?
----------------------------

Machine Learning (ML) has recently gained attention as a means to fit more accurate Exchange-Correlation (XC) functionals for use in Density Functional Theory (DFT). We have developed CIDER, a set of features, models, and training techniques for efficiently learning the exchange and correlation functionals. **CIDER** stands for **C**\ ompressed scale-\ **I**\ nvariant **DE**\ nsity **R**\ epresentation, which refers to the fact that the descriptors are invariant under squishing or expanding of the density while maintaining its shape. This property makes it efficient for learning the XC functional, especially the exchange energy.

**WARNING**: The CiderPress Code Base is Experimental
-----------------------------------------------------

Both the code and the functionals themselves are experimental. The code base will likely change significantly in the next few years. Therefore, please read the installation guidance, usage instructions, examples, and known issues thoroughly before using CiderPress.

Installation
------------

Installation of CiderPress requires the following:

* Python 3.9-3.12
* BLAS and LAPACK
* C and C++ compilers with OpenMP support.

CiderPress uses cmake to build its C backend. If you use ``pip``, cmake is automatically installed as a dependency to enable the build process. The C compiler and linear algebra libraries must be findable by cmake.

Also, if you use ``pip``, all Python package dependencies are installed automatically, with the exception of the optional dependency Pytorch. Pytorch is only needed if you plan to use the ``CIDER24X`` functionals. To install Pytorch with CUDA 11.8, you can use the following: ::

    pip3 install torch --index-url https://download.pytorch.org/whl/cu118

If you want to run plane-wave DFT calculations, you must also install GPAW with LibXC and FFTW. GPAW uses a ``siteconfig.py`` file to customize the libraries it links to. This repo's ``.github/workflows/gpaw_siteconfig.py`` could be useful for compiling GPAW with MKL, LibXC, and FFTW.

If you wish to run parallel calculations with GPAW, you should also have an MPI installation on your system along with an ``mpicc`` compiler. CiderPress has only been tested with OpenMPI but in principle should be compatible with MPICH and Intel MPI as well. The CiderPress build will automatically detect whether MPI is available, and if so it will build an MPI-parallel version of the GPAW interface.

The rest of the code is parallelized with OpenMP only for now.

Installation from PyPI
~~~~~~~~~~~~~~~~~~~~~~

You can install with the usual::

    pip install ciderpress

Currently only the sdist is available (no wheels yet), so it will take some time to build. The C backend of CiderPress is built using cmake, so you can customize the installation by setting the ``CMAKE_CONFIGURE_ARGS`` environment variable. For example, by default, CiderPress builds its own FFTW and searches for a (non-MKL) BLAS/LAPACK installation to link to. To use the Intel Math Kernel Library (MKL) as the FFT and linear algebra backend instead, use the following: ::

    export CMAKE_CONFIGURE_ARGS="-DBUILD_WITH_MKL=ON"
    pip install ciderpress

NOTE: If CiderPress will link to MKL as its linear algebra backend, make sure that you set ``-DBUILD_WITH_MKL=ON``. Otherwise, CiderPress might link to MKL and FFTW, which can cause runtime crashes because MKL contains an FFTW wrapper with identical function names.

Here is a list of cmake build options with their default values:

* ``BUILD_WITH_MKL (OFF)``: If ON, use Intel Math Kernel Library as the linear algebra and FFT backends. If OFF, link to whatever BLAS/LAPACK version is found by cmake and link to FFTW as the FFT backend.
* ``BUILD_LIBXC (OFF)``: If ON, libxc is downloaded, compiled, and linked to CiderPress during the compilation process. If OFF, a libxc installation must be available to link to and findable by cmake at compile time.
* ``BUILD_FFTW (ON)``: Ignored if ``BUILD_WITH_MKL=ON``. Otherwise, if ON, FFTW is built and linked to by cmake during compilation. If OFF, an FFTW installation must be available to link to and findable by cmake at compile time.
* ``BUILD_MARCH_NATIVE (OFF)``: If ON, use the ``-march=native`` C compiler flag, which enables instruction sets on the CPU used for compilation, potentially resulting in higher performance.

To further customize the installation, you can build from source and edit the ``CMakeLists.txt`` files in ``ciderpress/lib`` and its subdirectories.

Build from Source
~~~~~~~~~~~~~~~~~

You can also build from source. If you clone the CiderPress repository, you can enter the repository directory and simply type::

    pip install .

Alternatively, to build the C extensions "in-place," you can use cmake directly as follows: ::

    cd ciderpress/lib
    mkdir build
    cd build
    cmake <CMAKE_ARGS> ..
    make

You can also use the cmake configuration arguments listed above with these approaches. Note that if you use ``pip``, ``cmake`` will be installed as a dependency. If you build from source directly, you must have ``cmake`` installed on your system.

Installation in a Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installation in a conda environment follows the same procedure as above, but with the added benefit that non-Python dependencies can be installed using conda. For example, you can install the MKL using::

    conda install mkl"<=2024.0" mkl-devel"<=2024.0" mkl-service"<=2024.0" mkl_fft mkl_random

The ``<=2024.0`` is to fix a compatibility issue with PyTorch and MKL, so you can remove it if you don't need PyTorch (i.e. if you don't want to use CIDER24X models). In principle, it is also possible to pip install the MKL dependencies, but we have had trouble getting the libraries to link. Then you install CiderPress using MKL::

    CMAKE_CONFIGURE_ARGS=``-DBUILD_WITH_MKL=ON`` pip install .

Step-by-step Installation with Conda, Micromamba, etc.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section covers how to install CiderPress and its dependencies from a fresh conda environment. Micromamba is also supported; you will just need to replace the ``conda`` commands with ``micromamba`` below.

1. Make sure you have a C compiler installed. (Or you can install one through conda after creating your environment in step 1.)

2. Create a new conda environment.::

    conda create -n <my_env> python=3.11
    conda activate <my_env>

   Python 3.9-3.12 are supported.

3. Install dependencies. The scripts ``.github/workflows/mm_install_torch.sh`` and
   ``.github/workflows/mm_install_mpi.h`` can both be used to set up an environment
   for running CIDER calculations. ``mm_install_torch.sh`` installs MKL, libxc, FFTW,
   and pytorch, so it is useful if you want to run calculations with CIDER24X
   functionals, which require pytorch. ``mm_install_mpi.sh`` installs MKL, libxc,
   FFTW, OpenMPI, and mpicc, so it is useful if you want to run GPAW calculations.
   Note that the conda MPI installation might not work well for multi-node jobs on
   clusters, so you might want to use your own MPI/mpicc instead if that
   is your use case. Single-node jobs should work fine with conda's MPI.

4. Build C extensions and install CiderPress.::

    pip install .

5. (If using GPAW) Install GPAW from source. We recommend using our ``gpaw_siteconfig.py`` to link gpaw to
   MPI and MKL for simplicity and speed. (You can download GPAW at gitlab.com/gpaw/gpaw.)::

    cd <place you want to save the GPAW source>
    git clone https://gitlab.com/gpaw/gpaw.git
    cd gpaw
    cp <CiderPress>/.github/workflows/gpaw_siteconfig.py siteconfig.py
    python setup.py build install

   **Note**: Currently CiderPress does not support the new GPAW version (``gpaw.new``), but we plan to support it in the future.

Notes on External Code Performance and Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* CiderPress automatically installs PySCF as a dependency, and GPAW can be installed simply by ``pip install gpaw``. However, both codes will in general have better performance if compiled from source. See the `PySCF installation instructions <https://pyscf.org/install.html>`_ and the `GPAW installation instructions <https://gpaw.readthedocs.io/install.html>`_ for details.
* GPAW uses MPI for parallelization, and the CiderPress extensions must also link to MPI to run parallel GPAW calculations. Make sure cmake can find OpenMPI or an equivalent installation and that you have a working ``mpicc`` compiler before building CiderPress and GPAW together.
* The CiderPress C extensions must use the same OpenMP as PySCF and GPAW, otherwise you will run into parallelization issues and code crashes. The ``gpaw_siteconfig.py`` provided in the CiderPress repository (see Step 5 of step-by-step instructions above) assumes Intel's ``iomp5`` as the OpenMP library by default. If you are using GNU OpenMP, you should change ``iomp5`` to ``gomp`` in ``gpaw_siteconfig.py``. CiderPress will find OpenMP automatically using cmake, so please make sure this version is the one used to build PySCF and GPAW.
* To run the CIDER24X functionals, you also need to install Pytorch.

How can I run a CIDER calculation?
----------------------------------

The ML models that define CIDER functionals are stored in ``yaml`` files. To use previously developed functionals, you can download them using::

    cd <CiderPress>
    python scripts/download_functionals.py

See the ``examples`` directory for details on how to load and use the functionals.

CIDER calculations can be run in PySCF (for non-periodic, all-electron calculations) and GPAW (for periodic, plane-wave PAW calculations) using the functional initializers ``ciderpress.pyscf.dft.make_cider_calc`` and ``ciderpress.gpaw.calculator.get_cider_functional``, respectively. Periodic PySCF calculations are not yet supported, except for the CIDER24X functionals with uniform grids and pseudopotentials. See ``examples/pyscf/simple_calc.py`` and ``examples/gpaw/simple_calc.py`` for a demonstration of setting up a typical calculation, and refer to the docstrings of the initializers for a more detailed explanation of all the input options. As explained in the docstrings, the defaults are sufficient for most of the input options. The recommended functional for most applications in which the goal is to reproduce hybrid DFT is the ``CIDER23X_NL_MGGA_DTR`` exchange functional, which is a meta-GGA with nonlocal features of the density.

The more recent ``CIDER24Xne`` and ``CIDER24Xe`` functionals use more powerful descriptors and are therefore more accurate than any of the ``CIDER23X`` functionals, but they are also more expensive and only available for use in PySCF. ``CIDER24Xe`` is fit to molecular HOMO-LUMO gaps, so it might be useful for properties where band gaps are important.

How can I train a CIDER functional?
-----------------------------------

The basic ML training framework for CiderPress is stored in ``ciderpress.models``. CiderPress currently only contains the ML model classes themselves, but not the various training tools needed to set up the training databases. If you are interested in training your own CIDER model, we suggest reaching out to us to discuss (email kylebystrom@gmail.com).

Known Issues
------------

CiderPress has a few known issues that we are currently investigating. Please be aware of these when attempting calculations with CIDER functionals. We will make a note and publish a new release when we fix these issues. If you run into any other problems, please post an issue on the Github repository.

* For some periodic systems in GPAW within the PAW formalism, significant numerical instability issues arise for the nonlocal functionals. In our experience thus far, these issues are uncommon and seem to be caused by the nonlocal PAW corrections to the CIDER features as opposed to the functionals themselves.
* For the GPAW interface, the memory overhead for the nonlocal features can be fairly high, occasionally causing memory issues. Please be aware that you might need to allocate more memory for a nonlocal CIDER calculation than for, say, a PBE calculation.
* For the PySCF interface, there are (mostly minor) convergence issues for some systems. These issues are much less common and less severe for our most robust functionals (like NL-MGGA-DTR). Even for NL-MGGA-DTR, occasionally a system will not quite converge. Usually the energy convergence is fine, but the orbital gradients are somewhat unstable; it might be necessary to set ``conv_tol_grad`` to a higher value than the default. These issues are likely a mix between inherent functional stability and the stability of the fast feature evaluation algorithm.
* The code spits out a lot of divide-by-zero and invalid value warnings from numpy, which occur because (as with many functionals) some terms in CIDER functionals become numerically unstable at very small densities. These issues are corrected by setting the XC energy and potential at very low density to zero, and we will clean up various warnings and unnecessary debug statements as soon as possible.
* The construction of the CIDER PAW corrections within GPAW have a very small numerical stability issue that results in different energies on different runs (with energy differences of roughly :math:`10^{-11}` eV). The difference is so small that it is insignificant for most applications, but it might affect finite difference calculations with very small perturbations.

Questions and Comments
----------------------

Find a bug? Areas of code unclearly documented? Other questions? Feel free to contact
Kyle Bystrom at kylebystrom@gmail.com AND/OR create an issue on the `Github page <https://github.com/mir-group/CiderPress>`_.

Citing
------

If you find CiderPress or CIDER functionals useful in your research, please cite the following article::

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

The above article introduces the CIDER23X functionals and much of the algorithms in CiderPress. If you use the CIDER24X functionals, please also cite::

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
