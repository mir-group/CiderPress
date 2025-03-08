PySCF Interface
===============

The PySCF interface allows CIDER functionals to be used in the PySCF
software package. Most CIDER models are only availabe in the
non-periodic version of PySCF, but SDMX functionals can be performed
with periodic boundary conditions if pseudopotentials and a
uniform XC integration grid are used. We note that the latter
feature is particularly experimental.

This documentation assumes you are familiar with the PySCF code and
have a working installation of the software.
For PySCF documentation, please see the `PySCF <https://pyscf.org/>`_
website.

The main module CiderPress users need to be familiar with is
:py:mod:`ciderpress.pyscf.dft`, which contains tools to
turn a standard Kohn-Sham DFT calculation into one that uses
a CIDER functional. See the module documentation for details.

For those interested in the experimental periodic boundary
condition feature for SDMX functionals, please see the
:py:mod:`ciderpress.pyscf.pbc.dft` module documentation.

.. toctree::
   :maxdepth: 2

   dft
   pbc/dft
   analyzers
   descriptors

