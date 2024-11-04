.. _nldf_feat:

Nonlocal Density Features (NLDF)
================================

Nonlocal density features, or NLDFs, are the original type of nonlocal feature used
to train Cider models. The general idea is to take the density :math:`n(\mathbf{r})`,
multiply it by a kernel function :math:`k(\mathbf{r}-\mathbf{r}')`, and integrate
over real space to obtain features :math:`G(\mathbf{r})` that are nonlocal in the density:

.. math:: G[n](\mathbf{r}) = \int \text{d}^3\mathbf{r} k(\mathbf{r}-\mathbf{r}') n(\mathbf{r}')

This is a fairly simple convolution-based descriptor, but it has a significant shortcoming.
In order to guarantee a reasonable and finite value for :math:`G(\mathbf{r})`,
:math:`k(\mathbf{r}-\mathbf{r}')` must decay exponetially on some length-scale
as :math:`|\mathbf{r}-\mathbf{r}'|` grows large. However, choosing a single
length-scale will result in nontrivial behavior of the feature under
:ref:`Uniform Scaling <unif_scaling>`. To cure this problem, we need a
density-dependent kernel :math:`k[n](\mathbf{r}-\mathbf{r}')`. For a scaled
density :math:`n_\lambda(\mathbf{r})=\lambda^3 n(\lambda\mathbf{r})`,
if :math:`k[n_\lambda](\mathbf{r}-\mathbf{r}')=\lambda^j k[n](\lambda(\mathbf{r}-\mathbf{r}'))`,
then

.. math:: G[n_\lambda](\mathbf{r}) = \lambda^j G[n](\lambda\mathbf{r})

These features can then be reguralized to be scale-invariant and used in exchange functional models.

There are three types of NLDF implemented in CiderPress currently: versions ``i``, ``j``, and ``k``.
It turns out that version ``i`` and ``j`` features can be efficiently computed together, so
they can be combined into one feature version ``ij``.


Version J
---------

Version J is the NLDF implemented in :footcite:t:`CIDER23X`. It takes the form

.. math:: G_i[n](\mathbf{r}) = \int \text{d}^3\mathbf{r} \exp(-(a_i[n](\mathbf{r})+a_0[n](\mathbf{r}))|\mathbf{r}-\mathbf{r}'|^2) n(\mathbf{r}')

For a functional with GGA semilocal features, :math:`a_i[n](\mathbf{r})` (and :math:`a_0[n](\mathbf{r})`) take the form

.. math:: a_i[n](\mathbf{r}) = \pi\left(\frac{n}{2}\right)^{2/3} \left[A_i + B_i\left(\frac{|\nabla n|^2}{8n\tau_0}\right)\right]

For a functional with meta-GGA semilocal features, :math:`a_i[n](\mathbf{r})` takes the form

.. math:: a_i[n](\mathbf{r}) = \pi\left(\frac{n}{2}\right)^{2/3} \left[A_i + B_i\left(\frac{|\nabla n|^2}{8n\tau_0}\right) + C_i\left(\frac{\tau}{\tau_0}-1\right)\right]

:math:`A_i,B_i,C_i` are tunable parameters. Conventionally :math:`B_i=0` is used for the meta-GGA case (with :math:`C_i` finite),
but one can choose nonzero values in the CiderPress code. Using different :math:`A_i,B_i,C_i` for different :math:`i`
allows one to compute multiple features simultaneously (at roughly the same cost as one feature using the
fast NLDF evaluation algorithm). However, :math:`A_0,B_0,C_0` must be the same for all features. Both the GGA and meta-GGA
constructions of :math:`a_i[n](\mathbf{r})` obey

.. math:: a_i[n_\lambda](\mathbf{r}) = \lambda^2 a_i[n](\lambda \mathbf{r})

As a result of the above relationship, :math:`G_i[n](\mathbf{r})` is scale-invariant:

.. math:: G_i[n_\lambda](\mathbf{r}) = G_i[n](\lambda\mathbf{r})

In CiderPress, there is also an **experimental** feature to multiply the density :math:`n(\mathbf{r}')` by another
function :math:`b(\mathbf{r}')` before integrating over the kernel function. In this case,
:math:`G_i[n](\mathbf{r})` has the same uniform scaling behavior as :math:`b(\mathbf{r}')`.

Version I
---------

Version I modifies Version J in two ways. First, :math:`a_i[n](\mathbf{r})`, the exponent contribution computed
at point :math:`mathbf{r}` is not used; only :math:`a_0[n](\mathbf{r}')`. Second, several different options
are introduced for the integration kernel, resulting in a general form

.. math:: G_*[n](\mathbf{r}) = \int \text{d}^3\mathbf{r} k_*(a_0[n](\mathbf{r}), |\mathbf{r}-\mathbf{r}'|) n(\mathbf{r}')

The :math:`*` symbol is a stand-in for the form of integration kernel used. The options are listed below:

* ``se``: :math:`k_\text{se}(a, r) = \text{e}^{-ar^2}`
* ``se_r2``: :math:``
* ``se_apr2`` :math:`k_\text{se_apr2}(a, r) = a r^2 \text{e}^{-ar^2}`
* ``se_ap``: :math:`k_\text{se_ap}(a, r) = a \text{e}^{-ar^2}`
* ``se_ap2r2``: :math:`k_\text{se_ap2r2}(a, r) = a^2 r^2 \text{e}^{-ar^2}`
* ``se_lapl``: :math:`k_\text{se_lapl}(a, r) = 4 k_\text{se_ap2r2}(a, r) - 2 k_\text{se_ap}(a, r)`

There are also two options for vector features of the form

.. math:: \mathbf{g}_*[n](\mathbf{r}) = \int \text{d}^3\mathbf{r} \,(\mathbf{r}'-\mathbf{r})\,k_*(a_0[n](\mathbf{r}), |\mathbf{r}-\mathbf{r}'|) n(\mathbf{r}')

There are two options for :math:`k` in the above formula:

* ``se_grad``: :math:`k_\text{se_grad}(a, r) = k_\text{se_ap}(a, r)`
* ``se_rvec``: :math:`k_\text{se_grad}(a, r) = k_\text{se}(a, r)`

To construct rotationally invariant descriptors, the vector integrals :math:`\mathbf{g}_*[n](\mathbf{r})`
must be dotted with the density gradient or with another vector integral. For example,
one could have :math:`G=\mathbf{g}_\text{se\_grad} \cdot \mathbf{g}_\text{se\_grad}`

As with Version J, there is experimental support for multiplying math:`n(\mathbf{r}')` by another
function :math:`b(\mathbf{r}')` before integrating.

Unforunately, while all the above Version I variants are supported in the code (in any combination),
most of them either have numerical precision issues (i.e. they are difficult to compute with high
precision using the fast NLDF algorithms) or are physically unrealistic (being too large in the core
region or similar issues). The two most physically and numerically sensible seem to be ``se_ap``
and ``se_grad`` dotted with itself.

Version K
---------

Version K is a modification of Version J meant to remove the need for the squared-exponential kernel
to depend on the density at :math:`\mathbf{r}'`. However, without the dependence on :math:`a_0[n](\mathbf{r}')`,
the NLDFs can have large contributions from the core electrons in the valence region, which 
is physically unrealistic. To fix this, we multiply the density by a function that
decays when :math:`a_i(\mathbf{r})<<a_0(\mathbf{r}')`:

.. math:: G_i[n](\mathbf{r}) = \int \text{d}^3\mathbf{r} \exp(-a_i[n](\mathbf{r})|\mathbf{r}-\mathbf{r}'|^2) \exp(-3a_0[n](\mathbf{r}')/2a_i[n](\mathbf{r})) n(\mathbf{r}')

We have alternative options for the damping function, but the above exponential is the only supported
one currently.

.. footbibliography::

