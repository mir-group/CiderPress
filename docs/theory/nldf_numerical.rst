.. _nldf_numerical:

Numerical Evaluation of NLDF Features
=====================================

Computing :ref:`Nonlocal Density Features <nldf_feat>` (NLDFs) using simple
numerical integration can be unwieldy and computationally costly, so in
practical calculations, the feature integrals are expanded as a sum
of convolutions. The expansion is slightly different for each
feature version.

For version j, the general form of the feature is

.. math::

   G_i[n](\mathbf{r}) = \int \text{d}^3\mathbf{r} k(a_0[n](\mathbf{r}'), a_i(\mathbf{r}), |\mathbf{r}-\mathbf{r}'|) n(\mathbf{r}')

In the context of implementing a nonlocal van der Waals density functional, :footcite:t:`Roman-Perez2009`
found that the kernel :math:`k(a_0[n](\mathbf{r}'), a_i(\mathbf{r}), |\mathbf{r}-\mathbf{r}'|)`
can be approximated as

.. math::

   k(a, b, r) \approx \sum_\alpha \sum_\beta k(\alpha, \beta, r) p_\alpha(a) p_\beta(b)

where :math:`\alpha` and :math:`\beta` are a set of interpolating points that span the range
of values taken by :math:`a_0` and :math:`a_i` over the density distribution,
and :math:`p_\alpha(a)` is a cubic spline that is :math:`1` when :math:`a=\alpha` and :math:`0`
when :math:`a=\beta` (with :math:`\beta` being another interpolation point not equal to :math:`\alpha`).
Because the interpolation points are constants independent of the density, the above approximation
converts the feature integral into a sum over convolutions.

The :py:class:`ciderpress.dft.plans.NLDFSplinePlan` class implements this interpolation
approach.

The version j integration kernel :math:`k(a, b, r)` is separable in :math:`a` and :math:`b`:

.. math:: k(a, b, r) = \exp(-(a+b)r^2) = \exp(-a r^2) \exp(-b r^2)

Because of this, the interpolations can also be performed by expanding :math:`\exp(-a r^2)`
as a linear combination of the interpolation functions :math:`\exp(-\alpha r^2)`. This
modified approach is implemented by the :py:class:`ciderpress.dft.plans.NLDFGaussianPlan` class.

The details of how this interpolation approximation is used to compute the NLDFs
is dependent on the periodicity, type of grid, and type of basis set used in a DFT
calculation. For details on the implementation of this approach for isolated Gaussian-type
orbital calculations and periodic plane-wave DFT calculations, see :footcite:t:`CIDER23X`.

The other features versions (i and k) require slightly modified version of this approach,
but the basic idea is the same and the implementation quite similar.

