.. _sdmx_feat:

Smooth Density Matrix Exchange (SDMX)
=====================================

SDMX features\ :footcite:p:`CIDER24X` are nonlocal featurizations of the density matrix.
The basic idea of SDMX is to "smooth" out the density matrix around a reference point
:math:`\mathbf{r}` and also project it onto low-order spherical harmonics
(i.e. principle quantum numbers of :math:`\ell=0` and :math:`\ell=1`).
As a result, one can obtain very simple (but nonlocal) proxies for the shape of
the exchange hole :math:`|n_1(\mathbf{r},\mathbf{r}')|^2`, which can be used for fitting
both the exchange and correlation energies. The formulas and details for the different
SDMX versions are described below. These features take significant inspiration from the
"Rung 3.5" functionals developed by Janesko *et al.*.\ :footcite:p:`Janesko2010,Janesko2013,Janesko2014,Janesko2018`
We also plan to implement Rung 3.5 functionals and add them as an alternative
to SDMX features, but this development has not been done yet. The key difference between
SDMX and the original Rung 3.5 approach is that SMDX features are strictly quadratic
functionals of the density matrix, while Rung 3.5 functionals are not.

One key consideration in the design of the SDMX features is their behavior under
:ref:`uniform scaling <unif_scaling>` of the density.
In keeping with the design objective of enabling the incorporation of exact constraints
into ML functionals, all SDMX features have power law behavior under uniform scaling, as discussed below.

The construction of the most basic SDMX feature starts with a smoothed, spherically averaged density matrix

.. math:: \rho^0(R; \mathbf{r}) = \int \text{d}^3 \mathbf{r}'\, h(|\mathbf{r}'-\mathbf{r}|; R) n_1(\mathbf{r}', \mathbf{r})

where :math:`R` is a length-scale parameter, and :math:`h` is a smooth convolution kernel.
Then, one can construct the set of features :math:`H_j^0(\mathbf{r})` as

.. math:: H_j^0(\mathbf{r}) = 4\pi \int \text{d} R\, R^{2-j} \left|\rho^0(R; \mathbf{r})\right|^2

where :math:`j` is real. Currently we use :math:`j\in\{0,1,2\}`, but other values of :math:`j`
(including non-integers) are possible. However, :math:`j<0` and :math:`j>2` might cause
numerical stability and normalization issues. Currently, :math:`h(u; R)` takes the form

.. math:: h(u; R) = \left(\frac{2}{\pi}\right)^{3/2} \frac{4}{4-\sqrt{2}}\frac{e^{-2u^2/R^2}}{R^3} \left(1 - e^{-2u^2/R^2}\right)

Different functions could be chosen, but the important aspect of the smoothing
functions is that convolving the density matrix with it yields a smoothed
approximation to the spherically averaged density matrix at distance :math:`R`,
and that the level of smoothing increases as :math:`R` increases. These properties
are necessary 1) to obey uniform scaling rules and 2) to ensure that :math:`\rho^0(R; \mathbf{r})`
can be expanded efficiently using a Gaussian basis. To evaluate :math:`H_j^0(\mathbf{r})`,
the smoothed density matrix :math:`\rho^0(R; \mathbf{r})` is evaluated at a discrete set
of distances :math:`R_i` and then interpolated over :math:`R` using a Gaussian basis;
:math:`H_j^0` can then be evaluated analytically within the Gaussian basis. Note that under
:ref:`uniform scaling <unif_scaling>`, this feature scales as :math:`\lambda^{3+j}`, i.e.

.. math:: H_j^0[n_\lambda](\mathbf{r})=\lambda^{3+j} H_j^0[n](\lambda\mathbf{r})

The complexity of the SDMX features can be increased by also calculating expectation
values of the gradients of the smoothed density matrix with respect to the length-scale
:math:`R`, as follows:

.. math:: H_j^{0\text{d}}(\mathbf{r}) = 4\pi \int \text{d} R\, R^{4-j} \left|\frac{\partial}{\partial R} \rho^0(R; \mathbf{r})\right|^2

In practice, these features can be obtained at almost no computational overhead compared
to the original :math:`H_j^0` features because the bottleneck computational operations
are the same as those needed to compute :math:`H_j^0`.
The uniform scaling behavior of :math:`H_j^{0\text{d}}` is also :math:`\lambda^{3+j}`.

The angular complexity of the SDMX features can be increased by computing
(in addition to :math:`\rho^0`) the following higher-order quantity

.. math:: \boldsymbol{\rho}^1(R; \mathbf{r}) = \int \text{d}^3\mathbf{r}' \left[ \nabla h(|\mathbf{r}'-\mathbf{r}|; R) \right]  n_1(\mathbf{r}', \mathbf{r})

which involves the gradient of :math:`h(|\mathbf{r}'-\mathbf{r}|; R)` with
respect to :math:`\mathbf{r}`. Then, one can compute new features

.. math:: H_j^\text{1}(\mathbf{r}) = 4\pi \int \text{d}R\, R^{4-j} \left|\boldsymbol{\rho}^1(R; \mathbf{r})\right|^2

These are more costly to compute than :math:`H_j^0` or :math:`H_j^\text{0d}`,
but they provide significantly more information to the model because they
contain :math:`\ell=1` angular information. The uniform scaling behavior of
:math:`H_j^1` is also :math:`\lambda^{3+j}`.

Finally, one can combine the radial and Cartesian derivatives to construct one more type of feature

.. math:: H_j^\text{1d}(\mathbf{r}) = 4\pi \int \text{d}R\, R^{6-j} \left|\frac{\partial}{\partial R} \boldsymbol{\rho}^1(R; \mathbf{r})\right|^2

which also scales as :math:`\lambda^{3+j}` under uniform scaling and comes at very little
additional computational cost if :math:`\boldsymbol{\rho}^1(R; \mathbf{r})`
was already computed to evaluate :math:`H_j^\text{1}(\mathbf{r})`.

.. footbibliography::

