Machine Learning Framework for CIDER Functionals
================================================

This page covers the machine learning framework used to train exchange-correlation
functionals in the CIDER formalism, which is built on Gaussian process regression
(GPR).

Gaussian Process Regression
---------------------------

Gaussian process regression is a nonparametric, Bayesian statistical learning model.
The detailed theory of Gaussian processes can be found in the excellent textbook
by Rasmussen and Williams. :footcite:p:`Rasmussen2006` This section covers the application of Gaussian
process regression to functional design. More details on the formalism
can be found in :footcite:t:`CIDER23X` and :footcite:t:`CIDER24X`

Consider the problem of learning some function :math:`f(\mathbf{x})`, with
:math:`\mathbf{x}` being a feature vector of independent variables.
Let :math:`\mathbf{X}` be a feature matrix, where each row :math:`\mathbf{x}_i`
is the feature vector for training point :math:`i`, and let :math:`\mathbf{y}`
be a target vector, where each element :math:`y_i` is the observed value of
the target function for training point :math:`i`. Then, there are two key
components necessary to construct a Gaussian process regression for
:math:`f(\mathbf{x})`. The first is the covariance kernel
:math:`k(\mathbf{x}, \mathbf{x}')`, which defines the covariance between the
values of the predictive function for two points, i.e.
:math:`k(\mathbf{x}, \mathbf{x}')=\text{Cov}(f(\mathbf{x}), f(\mathbf{x}'))`.
The matrix of covariances between training points is denoted :math:`\mathbf{K}`
with matrix elements :math:`K_{ij}=k(\mathbf{x}_i, \mathbf{x}_j)`.
The second is the noise labels :math:`\sigma_i`, indicating the estimated
prior uncertainty on each training observation :math:`i`. We will denote the
diagonal matrix whose entries are the noise covariances :math:`{\sigma_i^{}}^2`
as :math:`\boldsymbol{\Sigma}_\text{noise}`. Using these components, the
predictive function for a test point :math:`\mathbf{x}_*`
takes the form :footcite:p:`Rasmussen2006`

.. math:: f(\mathbf{x}_*) = \sum_\alpha k(\mathbf{x}_*, \mathbf{x}_i) \alpha_i

with the following definition for the weight vector :math:`\boldsymbol{\alpha}`:

.. math:: \boldsymbol{\alpha} = \left(\mathbf{K} + \boldsymbol{\Sigma}_\text{noise}\right)^{-1} \mathbf{y}

In the case of fitting energies of molecules and solids, we need to fit an
extensive quantity in which the training labels are integrals of the predictive
function :math:`f(\mathbf{x})` over real space. The next section covers
adjustments to the Gaussian process model necessary to perform this task.

Fitting Total Energy Data
-------------------------

Consider an extensive quantity :math:`F`, which is a contribution to the
total electronic energy of a chemical system. In the case of CiderPress,
this quanity will be the exchange, correlation, or exchange-correlation energy.
We can fit :math:`F` by learning a function that gets integrated over
real-space to yield :math:`F` for a given system:

.. math:: F = \int \text{d}^3\mathbf{r}\,f\left(\mathbf{x}(\mathbf{r})\right) \label{eq:F_int}

In the above equation, :math:`f\left(\mathbf{x}(\mathbf{r})\right)` is the predictive
function for the energy density to be learned by the Gaussian process.
In practice, the integral must be performed numerically. For a given chemical system
indexed by :math:`m`, we write
    
.. math:: F^m = \sum_{g\in m} w_g^m f\left(\mathbf{x}_g^m\right) \label{eq:extensive_functional}

where :math:`g` indexes quadrature points and :math:`w_g^m` are the respective quadrature weights.
The covariances between the numerical integrals :math:`F^m` and :math:`F^n`
can be written as

.. math:: \text{Cov}(F^m, F^n) = \sum_{g \in m} \sum_{h \in n} w_g^m w_h^n k(\mathbf{x}_g^m, \mathbf{x}_h^m)

where :math:`k(\mathbf{x}, \mathbf{x}')` is the covariance kernel for
:math:`f(\mathbf{x})`. Computing the above double numerical integral directly
would be expensive. To overcome this issue, we define a small set of "control points"
:math:`\tilde{\mathbf{x}}_a` and approximate :math:`\text{Cov}(F^m, F^n)`
using a resolution-of-the-identity approximation:

.. math::
    \text{Cov}(F^m, F^n) = K_{mn} &\approx \tilde{\mathbf{k}}_m \tilde{\mathbf{K}}^{-1} \tilde{\mathbf{k}}_n \label{eq:roi_cov} \\
    \left(\tilde{\mathbf{K}}\right)_{ab} &= k(\tilde{\mathbf{x}}_a, \tilde{\mathbf{x}}_b) \\
    \left(\tilde{\mathbf{k}}_m\right)_a &= \sum_{g\in m} w_g^m k(\mathbf{x}_g^m, \tilde{\mathbf{x}}_a)

Using this definition of the covariance kernel, the predictive function can be expressed as

.. math::
    f(\mathbf{x}_*) &= \sum_a k(\mathbf{x}_*, \mathbf{\tilde{x}}_a) \alpha_a \label{eq:gp_sum_formula} \\
    \boldsymbol{\alpha} &= \sum_m \mathbf{\tilde{k}}_m \left\{\left[\mathbf{K} + \boldsymbol{\Sigma}_\text{noise}\right]^{-1} \mathbf{y}\right\}_m \label{eq:gp_predictive_cider3}

with :math:`\mathbf{y}` being the vector of training labels :math:`F^m`.

Fitting Eigenvalues
-------------------

The Gaussian process scheme discussed above can be extended to fit the eigenvalues
of the Kohn-Sham Hamiltonian.:footcite:p:`CIDER24X`
The :math:`i`-th eigenvalue :math:`\epsilon_i^m` of chemical system :math:`m` is
the partial derivate of the total energy with respect to the occupation
number :math:`f_i^m` of the orbital:

.. math:: \epsilon_i^m = \frac{\partial E}{\partial f_i^m}

Most of the Kohn-Sham eigenvalues :math:`\epsilon_i^m` are fictional and lack
explicit physical meaning, but the LUMO and HOMO eigenvalues of the exact
functional correspond to the electron affinity and negative of the ionization
potential, respectively. Therefore, we are interested in explicitly
fitting the derivative of our target quantity :math:`\frac{\partial F}{\partial f_i^m}`.
Details of how these 

.. footbibliography::

