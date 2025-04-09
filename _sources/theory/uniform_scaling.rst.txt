.. _unif_scaling:

Uniform Scaling
===============

Uniform scaling is a critical concept for understanding density functional
design. Consider a given electron density distribution :math:`n(\mathbf{r})`.
We can consider a transformed density of the form

.. math:: n_\lambda(\mathbf{r}) = \lambda^3 n(\lambda \mathbf{r})

This transformation is called *uniform coordinate scaling*,\ :footcite:p:`Martin2004`
because it essentially involves redefining
:math:`\mathbf{r}\leftarrow\lambda\mathbf{r}`, which squishes
(:math:`\lambda>0`) or expands (:math:`\lambda<0`) the
density while maintaining its relative shape. The :math:`\lambda^3`
prefactor ensures that the particle number is conserved, i.e.

.. math:: \int \text{d}^3\mathbf{r} n_\lambda(\mathbf{r}) = \int \text{d}^3\mathbf{r} n(\mathbf{r})

One can also consider uniform scaling of the density *matrix*:

.. math:: n_1^\lambda(\mathbf{r}, \mathbf{r}') = \lambda^3 n_1(\lambda \mathbf{r}, \lambda \mathbf{r}')

NOTE: For orbital-dependent functionals, scaling the density and scaling the density matrix are
not precisely equivalent, because there is a distribution of possible density
matrices that can yield a given density. If the orbital-dependent potential
causes the orbitals to rearrange themselves when uniform scaling occurs, the
lowest-energy density matrix for a density :math:`n_\lambda` will not necessarily
be :math:`n_1^\lambda`, i.e. the scaled density matrix obtained
from the lowest-energy density matrix with density :math:`n`. This is a very
subtle point, however, and does not usually make a big impact, so in most
cases we will always refer to scaling the density :math:`n_\lambda` for
simplicity, even when orbital-dependent quantities are involved.
For more details, see :footcite:t:`Gorling1995`.

The exact exchange functional :math:`E_\text{x}[n]` has a simple,
exact behavior under uniform scaling:\ :footcite:p:`Levy1985`

.. math:: E_\text{x}[n_\lambda] = \lambda E_\text{x}[n]

The correlation functional :math:`E_\text{c}[n]` does not have such simple
behavior under uniform scaling, but it does obey the limits\ :footcite:p:`Kaplan2023`

.. math::

   \lim_{\lambda\rightarrow 0} E_\text{c}[n_\lambda] &= \lambda C_0[n] \\
   \lim_{\lambda\rightarrow \infty} E_\text{c}[n_\lambda] &> -\infty

Therefore, it can be helpful to design features with simple, well-understood
behavior under uniform scaling. In particular, if the feature vector :math:`\mathbf{x}`
for the ML model is *scale-invariant*, i.e. if

.. math:: \mathbf{x}[n_\lambda](\mathbf{r}) = \mathbf{x}[n](\lambda \mathbf{r})

then an exchange functional of the form

.. math:: E_\text{x}[n] = \int \text{d}^3\mathbf{r} e_\text{x}^\text{ML}(\mathbf{x}(\mathbf{r})

obeys the uniform scaling rule for exchange (:math:`E_\text{x}[n_\lambda] = \lambda E_\text{x}[n]`).
Similar, scale-invariant features can also be useful for correlation functionals because
their behavior under uniform scaling will be the same as the behavior of the multiplicative
baseline functional used for training. If the baseline model has reasonable behavior under
uniform scaling (such as PBE/SCAN), this could help make more physically realistic models.
(However, it could also needlessly restrict the model's flexibility, so there are trade-offs involved).

.. footbibliography::

