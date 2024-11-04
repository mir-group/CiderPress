.. _sl_feat:

Semilocal Features (SL)
=======================

Semilocal features are the ingredient commonly used in semilocal DFT.
They most basic ingredient is the electron density :math:`n(\mathbf{r})`,
which is defined as

.. math:: n(\mathbf{r}) = \sum_i f_i |\phi_i(\mathbf{r})|^2

with :math:`\phi_i(\mathbf{r})` being the Kohn-Sham orbitals. A functional
constructed from :math:`n(\mathbf{r})` only is called a local density
approximation (LDA). The gradient of the density :math:`\nabla n(\mathbf{r})`
is also commonly used and results in a generalized-gradient approximation (GGA).
Lastly, the kinetic energy density

.. math:: \tau(\mathbf{r}) = \frac{1}{2} \sum_i f_i |\nabla\phi_i(\mathbf{r})|^2

can be used, resulting in a meta-generalized gradient approximation (meta-GGA or MGGA).
To help enforce physical constraints such as uniform scaling, regularized
features are often introduced, including the reduced gradient\ :footcite:p:`Perdew1996`

.. math:: p = \frac{|\nabla n|^2}{2(3\pi^2)^{1/3}n^{4/3}}

and the iso-orbital indicator\ :footcite:p:`Sun2013`

.. math:: \alpha = \frac{\tau - \tau_W}{\tau_0}

where :math:`\tau_W=|\nabla n|^2/8n` is the single-electron kinetic energy,
and :math:`\tau_0=\frac{3}{10}(3\pi^2)^{2/3}n^{5/3}` is the kinetic energy
density of the uniform electron gas.

Note that for simplicity, all of these definitions are provided for the
non-spin-polarized case, where the orbital occupations :math:`f_i\in [0,2]`

In Cider, the choice of semilocal features is specified using the class:`SemilocalSettings`
class in :py:mod:`ciderpress.dft.settings`. There is only one argument to this
class, :py:obj:`mode`, which specifies which features to compute. There are four choices:

* ``ns``: :math:`\mathbf{x}_\text{sl} = [n, |\nabla n|^2]`
* ``np``: :math:`\mathbf{x}_\text{sl} = [n, p]`
* ``nst``: :math:`\mathbf{x}_\text{sl} = [n, |\nabla n|^2, \tau]`
* ``npa``: :math:`\mathbf{x}_\text{sl} = [n, p, \alpha]`

**IMPORTANT NOTE 1**: If the mode is ``ns`` or ``np``, the kinetic energy density :math:`\tau`
is not computed, so it cannot be used at any point in the calculation. This means
that :ref:`Nonlocal Density Features <nldf_feat>` must be instantiated with
``sl_level="GGA"``. If desired, orbital-dependent features can still be incorporated
via :ref:`Smooth Density Matrix Exchange <sdmx_feat>`. If the mode is ``nst`` or ``npa``,
:math:`\tau` is always computed.

**IMPORTANT NOTE 2**: The regularized features :math:`p` and :math:`\alpha` are
scale-invariant, meaning that
an exchange functional trained with these features and
a proper exchange functional baseline will obey the
uniform scaling rule (see :ref:`Uniform Scaling Constraints <unif_scaling>`).
The raw features :math:`n`, :math:`|\nabla n|^2`, and :math:`\tau` are not
scale-invariant, and CiderPress does not automatically regularize these features
as it does with nonlocal features. Therefore, these features must be regularized in
the :py:mod:`ciderpress.dft.transform_data` module to enforce the uniform scaling
constraint in trained models. (TODO need to elaborate on this).

See the :class:`SemilocalSettings` class in the :ref:`Feature Settings <settings_module>`
documentation for more details on the API for setting up these features.

.. footbibliography::

