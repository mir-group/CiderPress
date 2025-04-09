DFT Module
==========

The :py:mod:`ciderpress.dft` module contains many of the core utilities
of the CiderPress code. The most important of these are:

* The :py:mod:`ciderpress.dft.settings` module, which consists of a set
  of classes for specifying the types of features to be computed
  for an ML model along with the parametrizations of those features.
* The :py:mod:`ciderpress.dft.plans` module, which provides classes
  that specify *how* a given set of features is to be computed.
  For example, an instance of :py:class:`NLDFSettingsVJ` from the
  :py:mod:`settings` module specifies that version-j :ref:`NLDF <nldf_feat>`
  features are to be computed, and an instance :py:class:`NLDFSplinePlan`
  from :py:mod:`plans` instructs CiderPress how to compute these
  features using cubic spline interpolation (see
  :ref:`NLDF Numerical Evaluation <nldf_numerical>`).
* The :py:mod:`ciderpress.dft.feat_normalizer` module, which provides
  utilities to transform "raw" features (which might not be scale-invariant)
  to scale-invariant "normalized features". Note it is not necessary to make
  every feature scale-invariant unless you want to enforce the uniform
  scaling rule for exchange.
* The :py:mod:`ciderpress.dft.transform_data` module, which provides
  utilities to transform "normalized" features (which do not necessarily fall
  in a finite interval, making them unwieldy for ML models) into
  "transformed" features suitable for direct input into Gaussian process
  regression.
* The :py:mod:`ciderpress.dft.xc_evaluator` and :py:mod:`ciderpress.dft.xc_evaluator2`
  modules, which provide tools to efficiently evaluate trained CIDER models.

The APIs of these modules are documentation in the subsections below.

.. toctree::
   :maxdepth: 1

   settings
   plans
   feat_normalizer
   transform_data
   xc_evaluator
