.. _models_module:

The Models Module
=================

The :py:mod:`ciderpress.models` module is the workhorse for training
Cider exchange-correlation functionals and mapping them to efficient
models for evaluation in DFT codes. The :py:mod:`train` module
contains the :py:class:`MOLGP` and :py:class:`MOLGP2` classes,
which are used to construct Gaussian process models for the exchange
and correlation energies. While these Gaussian process classes
are custom to CiderPress, the kernel functions used to construct
the covariance matrix are built on top of those in the scikit-learn
package.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   train
   dft_kernel
   kernels
   kernel_tools

