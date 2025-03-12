Density and Orbital Features in CiderPress
==========================================

To predict the exchange-correlation energy :math:`E_{\text{xc}}`, we need
to train a machine learning model :math:`e_{\text{xc}}(\mathbf{x})` such that

.. math:: E_{\text{xc}} = \int \text{d}^3\mathbf{r}\, e_\text{xc}(\mathbf{x}[n_1](\mathbf{r}))

where :math:`\mathbf{x}[n_1](\mathbf{r})` is a position-dependent feature vector that is
a functional of the density :math:`n(\mathbf{r})` in "pure" Kohn-Sham DFT and a 
functional of the density matrix :math:`n_1(\mathbf{r}, \mathbf{r}')` in the case of
"generalized" Kohn-Sham DFT. Cider provides several types of feature that can
be used as input to the ML model. These features
can be divided into four categories:

* :ref:`Semilocal Features (SL) <sl_feat>`: Same features as in conventional GGA/meta-GGA functionals (i.e. :math:`n`, :math:`\nabla n`, :math:`\tau`).
  NOTE: All Cider models must include semilocal features.
  They need not be used explicitly in the model, but evaluating
  them is required to evalute baseline functionals and other quantities in the code.
* :ref:`Nonlocal Density Features (NLDF) <nldf_feat>`: These features are constructed by integrating the density
  over a real-space kernel function to characterize the shape of the density around a point :math:`\mathbf{r}`.
* :ref:`Nonlocal Orbital Features (NLOF) <nlof_feat>`: EXPERIMENTAL, NOT TESTED, NOT RECOMMENDED FOR USE.
  One coordinate of the density matrix is operated on by a fractional Laplacian.
* :ref:`Smooth Density Matrix Exchange (SDMX) <sdmx_feat>`:
  One coordinate of the density matrix is convolved at different length scales. Then these convolutions
  are contracted to approximately characterize the shape of the density matrix around a point :math:`\mathbf{r}`.

The set of features to be used in a model is specified using the :class:`FeatureSettings` object. To see
the code API for setting up feature settings, see the :ref:`Settings module <settings_module>` section. To see
mathematical descriptions and physical intuition for the different types of features, see
the subsections below.

.. toctree::
   :maxdepth: 1

   sl
   nldf
   nlof
   sdmx

