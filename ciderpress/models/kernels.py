#!/usr/bin/env python
# CiderPress: Machine-learning based density functional theory calculations
# Copyright (C) 2024 The President and Fellows of Harvard College
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

# sklearn kernels for CIDER models.


from inspect import signature

import numpy as np
from scipy.special import comb
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    Exponentiation,
    GenericKernelMixin,
    Hyperparameter,
    Kernel,
    Product,
    StationaryKernelMixin,
    Sum,
    WhiteKernel,
    _num_samples,
    cdist,
)


def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError(
            "Anisotropic kernel must have the same number of "
            "dimensions as data (%d!=%d)" % (length_scale.shape[0], X.shape[1])
        )
    return length_scale


class DiffKernelMixin:
    def __add__(self, b):
        if not isinstance(b, Kernel):
            return DiffSum(self, DiffConstantKernel(b))
        return DiffSum(self, b)

    def __radd__(self, b):
        if not isinstance(b, Kernel):
            return DiffSum(DiffConstantKernel(b), self)
        return DiffSum(b, self)

    def __mul__(self, b):
        if not isinstance(b, Kernel):
            return DiffProduct(self, DiffConstantKernel(b))
        return DiffProduct(self, b)

    def __rmul__(self, b):
        if not isinstance(b, Kernel):
            return DiffProduct(DiffConstantKernel(b), self)
        return DiffProduct(b, self)

    def __pow__(self, b):
        return DiffExponentiation(self, b)

    def k_and_deriv(self, X, Y=None):
        raise NotImplementedError


class DiffSum(DiffKernelMixin, Sum):
    def k_and_deriv(self, X, Y=None):
        k1, dk1 = self.k1.k_and_deriv(X, Y)
        k2, dk2 = self.k2.k_and_deriv(X, Y)
        return k1 + k2, dk1 + dk2


class DiffProduct(DiffKernelMixin, Product):
    def k_and_deriv(self, X, Y=None):
        k1, dk1 = self.k1.k_and_deriv(X, Y)
        k2, dk2 = self.k2.k_and_deriv(X, Y)
        return k1 * k2, k1[..., None] * dk2 + k2[..., None] * dk1


class DiffExponentiation(DiffKernelMixin, Exponentiation):
    def k_and_deriv(self, X, Y=None):
        k, dk = self.kernel.k_and_deriv(X, Y)
        return (
            k**self.exponent,
            (self.exponent * k ** (self.exponent - 1))[:, :, np.newaxis] * dk,
        )


class DiffTransform(DiffKernelMixin, Kernel):
    def __init__(self, kernel, matrix, std=None, avg=None):
        self.kernel = kernel
        self.matrix = matrix
        self.std = std
        self.avg = avg

    def get_params(self, deep=True):
        params = dict(kernel=self.kernel, exponent=self.exponent)
        if deep:
            deep_items = self.kernel.get_params().items()
            params.update(("kernel__" + k, val) for k, val in deep_items)
        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            r.append(
                Hyperparameter(
                    "kernel__" + hyperparameter.name,
                    hyperparameter.value_type,
                    hyperparameter.bounds,
                    hyperparameter.n_elements,
                )
            )
        return r

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        self.kernel.theta = theta

    @property
    def bounds(self):
        return self.kernel.bounds

    def __eq__(self, b):
        if type(self) is not type(b):
            return False
        return self.kernel == b.kernel and self.exponent == b.exponent

    def _transform(self, X):
        if X is None:
            return None
        if self.avg is not None:
            X = X - self.avg
        if self.std is not None:
            X = X / self.std
        return X.dot(self.matrix)

    def _transform_bwd(self, X):
        if X is None:
            return None
        X = X.dot(self.matrix.T)
        if self.std is not None:
            X = X / self.std
        return X

    def __call__(self, X, Y=None, eval_gradient=False):
        return self.kernel(
            self._transform(X), Y=self._transform(Y), eval_gradient=eval_gradient
        )

    def diag(self, X):
        return self.kernel.diag(self._transform(X))

    def k_and_deriv(self, X, Y=None):
        k, dk = self.kernel.k_and_deriv(self._transform(X), self._transform(Y))
        return k, self._transform_bwd(dk)

    def __repr__(self):
        return "{0} ** {1}".format(self.kernel, self.exponent)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.kernel.is_stationary()

    @property
    def requires_vector_input(self):
        """Returns whether the kernel is defined on discrete structures."""
        return self.kernel.requires_vector_input


class DiffWhiteKernel(DiffKernelMixin, WhiteKernel):
    def k_and_deriv(self, X, Y=None):
        k = self.__call__(X, Y)
        shape = k.shape + (X.shape[1],)
        return k, np.zeros(shape, dtype=k.dtype)


class DiffConstantKernel(DiffKernelMixin, ConstantKernel):
    def k_and_deriv(self, X, Y=None):
        k = self.__call__(X, Y)
        shape = k.shape + (X.shape[1],)
        return k, np.zeros(shape, dtype=k.dtype)


class DiffRBF(DiffKernelMixin, RBF):
    def k_and_deriv(self, X, Y=None):
        """
        Compute the kernel and its derivative with
        respect to the input X.

        Args:
            X: Inputs with respect to which the
               kernel derivative is evaluated.
            Y: Other input. If None, set to X. Note:
                If Y is None i.e. X, Y is still assumed
                stationary, i.e. the return value is
                0.5 * d k(X,X,) / dX

        Returns:
            k, (X.shape[0], Y.shape[0])
            dk (X.shape[0], Y.shape[0], X.shape[1])
        """
        if Y is None:
            Y = X
        k = self.__call__(X, Y)
        dk = k[:, :, None] * (Y[None, :, :] - X[:, None, :])
        dk /= self.length_scale**2
        return k, dk


class DiffAntisymRBF(DiffRBF):
    """
    This is like a regular RBF kernel, except it obeys a sort
    of "antisymmetry" property where
    k(x0, x1, x2, ....; x0', x1', x2' ....) = 0
    if x0 == x1 or x0' == x1'.
    This property applies only to the first two features. It could
    be useful for enforcing exact constraints.
    """

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise NotImplementedError(
                "eval_gradient not implemented for this kernel yet"
            )
        length_scale = _check_length_scale(X[:, 1:], self.length_scale)
        if Y is None:
            Y = X.copy()
        XT = X[:, 2:] / length_scale[1:]
        YT = Y[:, 2:] / length_scale[1:]
        dists = cdist(XT, YT, metric="sqeuclidean")
        KT = np.exp(-0.5 * dists)
        XS = X[:, :2] / length_scale[0]
        YS = Y[:, :2] / length_scale[0]
        dists = cdist(XS[:, 0:1], YS[:, 0:1], metric="sqeuclidean")
        KS = np.exp(-0.5 * dists)
        dists = cdist(XS[:, 0:1], YS[:, 1:2], metric="sqeuclidean")
        KS[:] -= np.exp(-0.5 * dists)
        dists = cdist(XS[:, 1:2], YS[:, 0:1], metric="sqeuclidean")
        KS[:] -= np.exp(-0.5 * dists)
        dists = cdist(XS[:, 1:2], YS[:, 1:2], metric="sqeuclidean")
        KS[:] += np.exp(-0.5 * dists)
        return KS * KT

    def k_and_deriv(self, X, Y=None):
        length_scale = _check_length_scale(X[:, 1:], self.length_scale)
        if Y is None:
            Y = X.copy()
        XT = X[:, 2:] / length_scale[1:]
        YT = Y[:, 2:] / length_scale[1:]
        DK = np.zeros((XT.shape[0], YT.shape[0], X.shape[1]))
        dists = cdist(XT, YT, metric="sqeuclidean")
        KT = np.exp(-0.5 * dists)
        XS = X[:, :2] / length_scale[0]
        YS = Y[:, :2] / length_scale[0]

        dists = cdist(XS[:, 0], YS[:, 0], metric="sqeuclidian")
        tmp = np.exp(-0.5 * dists)
        DK[..., 0] = tmp * (YS[:, 0] - XS[:, 0])
        KS = tmp

        dists = cdist(XS[:, 0], YS[:, 1], metric="sqeuclidian")
        tmp = np.exp(-0.5 * dists)
        DK[..., 0] += tmp * (XS[:, 0] - YS[:, 1])
        KS[:] -= tmp

        dists = cdist(XS[:, 1], YS[:, 0], metric="sqeuclidian")
        tmp = np.exp(-0.5 * dists)
        DK[..., 1] = tmp * (XS[:, 1] - YS[:, 0])
        KS[:] -= tmp

        dists = cdist(XS[:, 1], YS[:, 1], metric="sqeuclidian")
        tmp = np.exp(-0.5 * dists)
        DK[..., 1] += tmp * (YS[:, 1] - XS[:, 1])
        KS[:] += tmp

        DK[..., 2:] = (KS * KT)[..., None] * (YT[None, :, :] - XT[:, None, :])
        DK[..., 2:] /= length_scale[1:]
        DK[..., :2] /= length_scale[0]
        return KS * KT, DK


class DiffLinearKernel(DiffKernelMixin, Kernel):
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError
        k = X.dot(Y.T)
        if eval_gradient:
            dk = np.empty((X.shape[0], X.shape[0], 0))
            return k, dk
        else:
            return k

    def diag(self, X):
        return np.einsum("if,if->i", X, X)

    def is_stationary(self):
        return False

    def k_and_deriv(self, X, Y=None):
        if Y is None:
            Y = X
        k = self.__call__(X, Y)
        return k, np.tile(Y, (X.shape[0], 1, 1))


class DiffPolyKernel(DiffKernelMixin, Kernel):
    def __init__(self, gamma=1.0, gamma_bounds=(1e-5, 1e5), order=4, factorial=True):
        self.gamma = gamma
        self.gamma_bounds = gamma_bounds
        self.order = order
        self.factorial = factorial

    @property
    def anisotropic(self):
        return np.iterable(self.gamma) and len(self.gamma) > 1

    @property
    def hyperparameter_gamma(self):
        if self.anisotropic:
            return Hyperparameter(
                "gamma", "numeric", self.gamma_bounds, len(self.gamma)
            )
        return Hyperparameter("gamma", "numeric", self.gamma_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        k = 1.0
        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError
        optg = not self.hyperparameter_gamma.fixed
        if eval_gradient and optg:
            dk = 0
        dot1 = (self.gamma * X).dot(Y.T)
        dotn = 1
        for n in range(1, self.order + 1):
            if self.factorial:
                if eval_gradient and optg:
                    dk += dotn
                dotn *= dot1 / n
            else:
                if eval_gradient and optg:
                    dk += n * dotn
                dotn *= dot1
            k += dotn
        if eval_gradient:
            if optg:
                if self.anisotropic:
                    dk = dk[:, :, None] * X[:, None, :] * X[None, :, :] * self.gamma
                else:
                    dk = (dk * dot1)[:, :, np.newaxis]
            else:
                dk = np.empty((X.shape[0], X.shape[0], 0), dtype=X.dtype)
            return k, dk
        return k

    def diag(self, X):
        k = 1.0
        dot1 = np.einsum("ij,ij->i", self.gamma * X, X)
        dotn = 1
        for n in range(1, self.order + 1):
            if self.factorial:
                dotn *= dot1 / n
            else:
                dotn *= dot1
            k += dotn
        return k

    def is_stationary(self):
        return False

    def k_and_deriv(self, X, Y=None):
        if Y is None:
            Y = X
        k = 1.0
        dk = 0.0
        dot1 = (self.gamma * X).dot(Y.T)
        dotn = 1
        for n in range(1, self.order + 1):
            if self.factorial:
                dk += dotn
                dotn *= dot1 / n
            else:
                dk += dotn * n
                dotn *= dot1
            k += dotn
        dk = self.gamma * dk[:, :, None] * Y[None, :, :]
        return k, dk


class _IndexMixin:
    def get_params(self, deep=True):
        params = dict()
        cls = self._base_cls
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        init_sign = signature(init)
        args, varargs = [], []
        for parameter in init_sign.parameters.values():
            if parameter.kind != parameter.VAR_KEYWORD and parameter.name != "self":
                args.append(parameter.name)
            if parameter.kind == parameter.VAR_POSITIONAL:
                varargs.append(parameter.name)

        if len(varargs) != 0:
            raise RuntimeError(
                "scikit-learn kernels should always "
                "specify their parameters in the signature"
                " of their __init__ (no varargs)."
                " %s doesn't follow this convention." % (cls,)
            )
        for arg in args:
            params[arg] = getattr(self, arg)
        if deep:
            param_keys = list(params.keys())
            for k in param_keys:
                if hasattr(params[k], "get_params"):
                    deep_items = params[k].get_params().items()
                    params.update(
                        (("{}__{}".format(k, kk), val) for kk, val in deep_items)
                    )

        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""
        r = [
            getattr(self, attr)
            for attr in dir(self)
            if attr.startswith("hyperparameter_")
        ]
        for attr in self.get_params():
            if hasattr(attr, "hyperparameters"):
                for hparam in attr.hyperparameters:
                    r.append(hparam)
        return r


class _SubsetMixin(_IndexMixin):
    """
    A Mixin that evalutes the kernel on a subset of features.
    To use, make a class SubsetClass(_SubsetMixin, BaseClass).
    BaseClass must be the second parent.
    """

    def __init__(self, indexes, *args, **kwargs):
        """
        Initialize SubsetKernel
        Args:
            indexes: list/tuple of indexes or slice
            *args and **kwargs: arguments to parent kernel
        """
        self._locked = False
        self.indexes = indexes
        self._base_cls = self.__class__.__bases__[1]  # Base Kernel
        self._base_cls.__init__(self, *args, **kwargs)

    def _index_and_lock(self, X):
        if self._locked:
            return False, X
        else:
            self._locked = True
            return True, X[:, self.indexes]

    def __call__(self, X, Y=None, eval_gradient=False, **kwargs):
        """
        Call the kernel

        Args:
            X: First matrix
            Y: Second matrix
            eval_gradient: Whether to evaluate gradient. X must be None
            **kwargs: Passed to base kernel. Primarily for using the
                get_sub_kernels option of ARBF for spline mapping.

        Returns:
            Return values of base kernel (usually kernel matrix and, if
                eval_gradient is True, its gradient wrt hyperparams)
        """
        if self._locked:
            return self._base_cls.__call__(
                self, X, Y=Y, eval_gradient=eval_gradient, **kwargs
            )
        self._locked = True
        if Y is not None:
            Y = Y[:, self.indexes]
        X = X[:, self.indexes]
        result = self._base_cls.__call__(
            self, X, Y=Y, eval_gradient=eval_gradient, **kwargs
        )
        self._locked = False
        return result

    def diag(self, X):
        if self._locked:
            return self._base_cls.diag(self, X)
        self._locked = True
        result = self._base_cls.diag(self, X[:, self.indexes])
        self._locked = False
        return result

    def k_and_deriv(self, X, Y=None):
        if self._locked:
            return self._base_cls.k_and_deriv(self, X, Y=Y)
        self._locked = True
        if Y is not None:
            Y = Y[:, self.indexes]
            shape = (X.shape[0], Y.shape[0], X.shape[1])
        else:
            shape = (X.shape[0], X.shape[0], X.shape[1])
        dk = np.zeros(shape)
        X = X[:, self.indexes]
        k, dk[:, :, self.indexes] = self._base_cls.k_and_deriv(self, X, Y=Y)
        self._locked = False
        return k, dk


class _SpinSymMixin(_IndexMixin):
    """
    A Mixin that evalutes the kernel assuming spin-symmetry
    between the alpha_ind features and beta_ind features.
    To use, make a class SpinSymClass(_SpinSymMixin, BaseClass).
    BaseClass must be the second parent.
    """

    def __init__(self, alpha_ind, beta_ind, *args, **kwargs):
        """
        Args:
            alpha_ind: list/tuple of indexes or slice for alpha spin
            beta_ind: list/tuple of indexes or slice for beta spin
            *args and **kwargs: arguments to parent kernel
        """
        self._locked = False
        self.alpha_ind = alpha_ind
        self.beta_ind = beta_ind
        self._base_cls = self.__class__.__bases__[1]  # Base Kernel
        self._base_cls.__init__(self, *args, **kwargs)

    def __call__(self, X, Y=None, eval_gradient=False, **kwargs):
        """
        Call the kernel

        Args:
            X: First matrix
            Y: Second matrix
            eval_gradient: Whether to evaluate gradient. X must be None
            **kwargs: Passed to base kernel. Primarily for using the
                get_sub_kernels option of ARBF for spline mapping.

        Returns:
            Return values of base kernel (usually kernel matrix and, if
                eval_gradient is True, its gradient wrt hyperparams)
        """
        if self._locked:
            return self._base_cls.__call__(
                self, X, Y=Y, eval_gradient=eval_gradient, **kwargs
            )
        self._locked = True
        if Y is not None:
            Y = np.vstack((Y[:, self.alpha_ind], Y[:, self.beta_ind]))
        else:
            Y = None
        X = np.vstack((X[:, self.alpha_ind], X[:, self.beta_ind]))
        NX = X.shape[0] // 2
        NY = Y.shape[0] // 2 if Y is not None else NX
        k = self._base_cls.__call__(self, X, Y=Y, eval_gradient=eval_gradient, **kwargs)
        if eval_gradient:
            k, dk = k
        k = k[:NX] + k[NX:]
        k = k[:, :NY] + k[:, NY:]
        self._locked = False
        if eval_gradient:
            dk = dk[:NX] + dk[NX:]
            dk = dk[:, :NY] + dk[:, NY:]
            return k, dk
        return k

    def diag(self, X):
        if self._locked:
            return self._base_cls.diag(self, X)
        self._locked = True
        XA = X[:, self.alpha_ind]
        XB = X[:, self.beta_ind]
        diag = self._base_cls.diag(self, XA)
        diag += self._base_cls.diag(self, XB)
        diag += np.diag(self._base_cls.__call__(self, XA, XB))
        diag += np.diag(self._base_cls.__call__(self, XB, XA))
        self._locked = False
        return diag

    def k_and_deriv(self, X, Y=None):
        if self._locked:
            return self._base_cls.k_and_deriv(self, X, Y=Y)
        self._locked = True
        Nfeat = X.shape[1]
        if Y is not None:
            Y = np.vstack((Y[:, self.alpha_ind], Y[:, self.beta_ind]))
        else:
            Y = None
        X = np.vstack((X[:, self.alpha_ind], X[:, self.beta_ind]))
        NX = X.shape[0] // 2
        NY = Y.shape[0] // 2 if Y is not None else NX
        k, dk = self._base_cls.k_and_deriv(self, X, Y=Y)
        k = k[:NX] + k[NX:]
        k = k[:, :NY] + k[:, NY:]
        dk = dk[:, :NY] + dk[:, NY:]
        dkfull = np.zeros((NX, NY, Nfeat))
        dkfull[:, :, self.alpha_ind] = dk[:NX]
        dkfull[:, :, self.beta_ind] = dk[NX:]
        self._locked = False
        return k, dkfull


class PartialRBF(DiffRBF):
    """
    Child class of sklearn RBF which only acts on the slice X[:,start:]
    (or X[:,active_dims] if and only if active_dims is supplied).
    start is ignored if active_dims is supplied.
    """

    def __init__(
        self,
        length_scale=1.0,
        length_scale_bounds=(1e-5, 1e5),
        start=0,
        active_dims=None,
    ):
        super(PartialRBF, self).__init__(length_scale, length_scale_bounds)
        self.start = start
        self.active_dims = active_dims

    def __call__(self, X, Y=None, eval_gradient=False):
        if self.active_dims is None:
            X = X[:, self.start :]
            if Y is not None:
                Y = Y[:, self.start :]
        else:
            X = X[:, self.active_dims]
            if Y is not None:
                Y = Y[:, self.active_dims]
        return super(PartialRBF, self).__call__(X, Y, eval_gradient)

    def k_and_deriv(self, X, Y=None):
        if self.active_dims is None:
            X = X[:, self.start :]
            if Y is not None:
                Y = Y[:, self.start :]
        else:
            X = X[:, self.active_dims]
            if Y is not None:
                Y = Y[:, self.active_dims]
        return super(PartialRBF, self).k_and_deriv(X, Y)


class DiffARBF(DiffRBF):
    """
    Additive RBF kernel of Duvenaud et al.
    """

    def __init__(
        self,
        order=1,
        length_scale=1.0,
        scale=None,
        length_scale_bounds=(1e-5, 1e5),
        scale_bounds=(1e-5, 1e5),
    ):
        """
        Args:
            order (int): Order of kernel
            length_scale (float or array): length scale of kernel
            scale (array): coefficients of each order, starting with
                0 and ascending.
            length_scale_bounds: bounds of length_scale
            scale_bounds: bounds of scale
        """
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.order = order
        self.scale = scale
        self.scale_bounds = scale_bounds

    @property
    def hyperparameter_scale(self):
        return Hyperparameter("scale", "numeric", self.scale_bounds, len(self.scale))

    def diag(self, X):
        nfeat = X.shape[1]
        comb_list = []
        for n in range(self.order + 1):
            comb_list.append(comb(nfeat, n))
        comb_list = np.array(comb_list)
        return np.ones(X.shape[0]) * np.sum(self.scale * comb_list)

    def __call__(self, X, Y=None, eval_gradient=False, get_sub_kernels=False):
        if self.anisotropic:
            num_scale = len(self.length_scale)
        else:
            num_scale = 1
        if Y is None:
            Y = X
        diff = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) / self.length_scale
        k0 = np.exp(-0.5 * diff**2)
        sk = []
        if eval_gradient:
            deriv_size = 0
            if not self.hyperparameter_length_scale.fixed:
                deriv_size += num_scale
            else:
                num_scale = 0
            if not self.hyperparameter_scale.fixed:
                deriv_size += len(self.scale)
            derivs = np.zeros((X.shape[0], Y.shape[0], deriv_size))
        for i in range(self.order):
            sk.append(np.sum(k0 ** (i + 1), axis=-1))
        en = [1]
        for n in range(self.order):
            en.append(sk[n] * (-1) ** n)
            for k in range(n):
                en[-1] += (-1) ** k * en[n - k] * sk[k]
            en[-1] /= n + 1
        res = 0
        for n in range(self.order + 1):
            res += self.scale[n] * en[n]
            if eval_gradient and not self.hyperparameter_scale.fixed:
                derivs[:, :, num_scale + n] = self.scale[n] * en[n]
        kernel = res
        if get_sub_kernels:
            return kernel, en

        if eval_gradient and not self.hyperparameter_length_scale.fixed:
            inds = np.arange(X.shape[1])
            for ind in inds:
                den = [np.zeros(X.shape)]
                sktmp = []
                if self.order > 0:
                    den.append(np.ones(k0[:, :, ind].shape))
                for n in range(1, self.order):
                    sktmp.append(
                        np.sum(k0[..., :ind] ** n, axis=-1)
                        + np.sum(k0[..., ind + 1 :] ** n, axis=-1)
                    )
                    den.append(sktmp[n - 1] * (-1) ** (n + 1))
                    for k in range(n - 1):
                        den[-1] += (-1) ** k * den[n - k] * sktmp[k]
                res = 0
                if self.order > 0:
                    res += self.scale[1] * den[1]
                for n in range(2, self.order + 1):
                    res += self.scale[n] * den[n] / (n - 1)
                res *= diff[:, :, ind] ** 2 * k0[:, :, ind]
                if self.anisotropic:
                    derivs[:, :, ind] = res
                else:
                    derivs[:, :, 0] += res

        if eval_gradient:
            return kernel, derivs
        return kernel

    def k_and_deriv(self, X, Y=None):
        if Y is None:
            Y = X
        diff = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) / self.length_scale
        k0 = np.exp(-0.5 * diff**2)
        sk = []
        for i in range(self.order):
            sk.append(np.sum(k0 ** (i + 1), axis=-1))
        en = [1]
        for n in range(self.order):
            en.append(sk[n] * (-1) ** n)
            for k in range(n):
                en[-1] += (-1) ** k * en[n - k] * sk[k]
            en[-1] /= n + 1
        res = 0
        for n in range(self.order + 1):
            res += self.scale[n] * en[n]
        kernel = res

        shape = kernel.shape + (X.shape[1],)
        dk = np.zeros(shape, dtype=kernel.dtype)
        for ind in range(X.shape[1]):
            den = [np.zeros(X.shape)]
            sktmp = []
            if self.order > 0:
                den.append(np.ones(k0[:, :, ind].shape))
            for n in range(1, self.order):
                sktmp.append(
                    np.sum(k0[..., :ind] ** n, axis=-1)
                    + np.sum(k0[..., ind + 1 :] ** n, axis=-1)
                )
                den.append(sktmp[n - 1] * (-1) ** (n + 1))
                for k in range(n - 1):
                    den[-1] += (-1) ** k * den[n - k] * sktmp[k]
            res = 0
            if self.order > 0:
                res += self.scale[1] * den[1]
            for n in range(2, self.order + 1):
                res += self.scale[n] * den[n] / (n - 1)
            res *= -1 * diff[:, :, ind] * k0[:, :, ind]
            dk[:, :, ind] = res
        dk /= self.length_scale

        return kernel, dk


class DiffAdditiveMixin(DiffKernelMixin):
    """
    Need to define is_stationary, _get_k0_dk0_train,
    and _get_k0_dk0_eval.
    """

    def __init__(
        self,
        order=1,
        length_scale=1.0,
        scale=None,
        length_scale_bounds=(1e-5, 1e5),
        scale_bounds=(1e-5, 1e5),
    ):
        """
        Args:
            order (int): Order of kernel
            length_scale (float or array): length scale of kernel
            scale (array): coefficients of each order, starting with
                0 and ascending.
            length_scale_bounds: bounds of length_scale
            scale_bounds: bounds of scale
        """
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.order = order
        self.scale = scale
        self.scale_bounds = scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def is_stationary(self):
        raise NotImplementedError

    def diag(self, X):
        """
        NOTE: Quite inefficient, but this is not used much for the
        functionals anyway.
        """
        K = self(X)
        return np.diag(K)

    def get_k0_for_mapping(self, X, Y, lscale):
        raise NotImplementedError

    def _get_k0_dk0_train(self, X, Y, eval_gradient):
        raise NotImplementedError

    def _get_k0_dk0_eval(self, X, Y, eval_gradient):
        raise NotImplementedError

    @property
    def num_scale(self):
        if self.anisotropic:
            return len(self.length_scale)
        else:
            return 1

    @property
    def hyperparameter_scale(self):
        return Hyperparameter("scale", "numeric", self.scale_bounds, len(self.scale))

    def get_zero_derivs(self, X, Y):
        deriv_size = 0
        if not self.hyperparameter_length_scale.fixed:
            deriv_size += self.num_scale
        if not self.hyperparameter_scale.fixed:
            deriv_size += len(self.scale)
        return np.zeros((X.shape[0], Y.shape[0], deriv_size))

    def __call__(self, X, Y=None, eval_gradient=False, get_sub_kernels=False):
        if Y is None:
            Y = X
        k0, dk0 = self._get_k0_dk0_train(X, Y, eval_gradient)
        if eval_gradient:
            derivs = self.get_zero_derivs(X, Y)
        sk = []
        for i in range(self.order):
            sk.append(np.sum(k0 ** (i + 1), axis=-1))
        en = [1]
        for n in range(self.order):
            en.append(sk[n] * (-1) ** n)
            for k in range(n):
                en[-1] += (-1) ** k * en[n - k] * sk[k]
            en[-1] /= n + 1
        res = 0
        for n in range(self.order + 1):
            res += self.scale[n] * en[n]
            if eval_gradient and not self.hyperparameter_scale.fixed:
                derivs[:, :, self.num_scale + n] = self.scale[n] * en[n]
        kernel = res
        if get_sub_kernels:
            return kernel, en

        if eval_gradient and not self.hyperparameter_length_scale.fixed:
            inds = np.arange(X.shape[1])
            for ind in inds:
                den = [np.zeros(X.shape)]
                sktmp = []
                if self.order > 0:
                    den.append(np.ones(k0[:, :, ind].shape))
                for n in range(1, self.order):
                    sktmp.append(
                        np.sum(k0[..., :ind] ** n, axis=-1)
                        + np.sum(k0[..., ind + 1 :] ** n, axis=-1)
                    )
                    den.append(sktmp[n - 1] * (-1) ** (n + 1))
                    for k in range(n - 1):
                        den[-1] += (-1) ** k * den[n - k] * sktmp[k]
                res = 0
                if self.order > 0:
                    res += self.scale[1] * den[1]
                for n in range(2, self.order + 1):
                    res += self.scale[n] * den[n] / (n - 1)
                res *= dk0[:, :, ind]
                if self.anisotropic:
                    derivs[:, :, ind] = res
                else:
                    derivs[:, :, 0] += res

        if eval_gradient:
            return kernel, derivs
        return kernel

    def k_and_deriv(self, X, Y=None):
        if Y is None:
            Y = X
        k0, dk0 = self._get_k0_dk0_eval(X, Y, True)
        sk = []
        for i in range(self.order):
            sk.append(np.sum(k0 ** (i + 1), axis=-1))
        en = [1]
        for n in range(self.order):
            en.append(sk[n] * (-1) ** n)
            for k in range(n):
                en[-1] += (-1) ** k * en[n - k] * sk[k]
            en[-1] /= n + 1
        res = 0
        for n in range(self.order + 1):
            res += self.scale[n] * en[n]
        kernel = res

        shape = kernel.shape + (X.shape[1],)
        dk = np.zeros(shape, dtype=kernel.dtype)
        for ind in range(X.shape[1]):
            den = [np.zeros(X.shape)]
            sktmp = []
            if self.order > 0:
                den.append(np.ones(k0[:, :, ind].shape))
            for n in range(1, self.order):
                sktmp.append(
                    np.sum(k0[..., :ind] ** n, axis=-1)
                    + np.sum(k0[..., ind + 1 :] ** n, axis=-1)
                )
                den.append(sktmp[n - 1] * (-1) ** (n + 1))
                for k in range(n - 1):
                    den[-1] += (-1) ** k * den[n - k] * sktmp[k]
            res = 0
            if self.order > 0:
                res += self.scale[1] * den[1]
            for n in range(2, self.order + 1):
                res += self.scale[n] * den[n] / (n - 1)
            res *= dk0[:, :, ind]
            dk[:, :, ind] = res

        return kernel, dk


class DiffARBFV2(DiffAdditiveMixin, Kernel):
    """
    Additive RBF kernel of Duvenaud et al.
    """

    def is_stationary(self):
        return True

    def get_k0_for_mapping(self, X, Y, lscale):
        diff = X[:, None] - Y[None, :] / lscale
        return np.exp(-0.5 * diff * diff)

    def _get_k0_dk0_train(self, X, Y, eval_gradient):
        diff = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) / self.length_scale
        k0 = np.exp(-0.5 * diff**2)
        if eval_gradient:
            dk0 = diff**2 * k0
        else:
            dk0 = None
        return k0, dk0

    def _get_k0_dk0_eval(self, X, Y, eval_gradient):
        diff = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) / self.length_scale
        k0 = np.exp(-0.5 * diff**2)
        if eval_gradient:
            dk0 = -diff * k0 / self.length_scale
        else:
            dk0 = None
        return k0, dk0


class DiffAddLLRBF(DiffAdditiveMixin, Kernel):
    """
    Additive RBF kernel of Duvenaud et al.
    """

    def __init__(
        self,
        order=1,
        alpha=1.0,
        length_scale=1.0,
        scale=None,
        length_scale_bounds=(1e-5, 1e5),
        scale_bounds=(1e-5, 1e5),
    ):
        """
        Args:
            order (int): Order of kernel
            alpha (float): Rational quadratic order
            length_scale (float or array): length scale of kernel
            scale (array): coefficients of each order, starting with
                0 and ascending.
            length_scale_bounds: bounds of length_scale
            scale_bounds: bounds of scale
        """
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.order = order
        self.alpha = alpha
        self.scale = scale
        self.scale_bounds = scale_bounds

    def is_stationary(self):
        return False

    def get_k0_for_mapping(self, X, Y, lscale):
        diff = (X[:, np.newaxis] - Y[np.newaxis, :]) / lscale
        dot = (X[:, np.newaxis] * Y[np.newaxis, :]) / (self.alpha * lscale**2)
        return (1 + dot) * np.exp(-0.5 * diff**2)

    def _get_k0_dk0_train(self, X, Y, eval_gradient):
        invssq = 1.0 / (self.alpha * self.length_scale**2)
        diff = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) / self.length_scale
        k0 = np.exp(-0.5 * diff**2)
        dot = 1 + invssq * X[:, np.newaxis, :] * Y[np.newaxis, :, :]
        if eval_gradient:
            dk0 = diff**2 * k0
            ddot = -2 * invssq * X[:, np.newaxis, :] * Y[np.newaxis, :, :]
            dk0 = dk0 * dot + ddot * k0
        else:
            dk0 = None
        k0 = dot * k0
        return k0, dk0

    def _get_k0_dk0_eval(self, X, Y, eval_gradient):
        invssq = 1.0 / (self.alpha * self.length_scale**2)
        diff = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) / self.length_scale
        k0 = np.exp(-0.5 * diff**2)
        dot = 1 + invssq * X[:, np.newaxis, :] * Y[np.newaxis, :, :]
        if eval_gradient:
            dk0 = -1.0 * diff * k0 / self.length_scale
            ddot = invssq * Y[np.newaxis, :, :]
            dk0 = dk0 * dot + ddot * k0
        else:
            dk0 = None
        k0 = dot * k0
        return k0, dk0


class DiffAddRQ(DiffAdditiveMixin, Kernel):
    """
    Additive rational quadratic kernel
    """

    def __init__(
        self,
        order=1,
        alpha=1.0,
        length_scale=1.0,
        scale=None,
        length_scale_bounds=(1e-5, 1e5),
        scale_bounds=(1e-5, 1e5),
    ):
        """
        Args:
            order (int): Order of kernel
            alpha (float): Rational quadratic order
            length_scale (float or array): length scale of kernel
            scale (array): coefficients of each order, starting with
                0 and ascending.
            length_scale_bounds: bounds of length_scale
            scale_bounds: bounds of scale
        """
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.order = order
        self.alpha = alpha
        self.scale = scale
        self.scale_bounds = scale_bounds

    def is_stationary(self):
        return False

    def get_k0_for_mapping(self, X, Y, lscale):
        diff = X[:, np.newaxis] - Y[np.newaxis, :]
        inv_scale = 1.0 / (2 * self.alpha * lscale**2)
        return (1 + diff * diff * inv_scale) ** (-self.alpha)

    def _get_k0_dk0_train(self, X, Y, eval_gradient):
        diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
        inv_scale = 1.0 / (2 * self.alpha * self.length_scale**2)
        alpha = self.alpha
        k0 = (1 + diff * diff * inv_scale) ** (-alpha)
        if eval_gradient:
            dk0 = (
                2
                * diff
                * diff
                * (1 + diff * diff * inv_scale) ** (-1 - alpha)
                * inv_scale
                * alpha
            )
        else:
            dk0 = None
        return k0, dk0

    def _get_k0_dk0_eval(self, X, Y, eval_gradient):
        diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
        inv_scale = 1.0 / (2 * self.alpha * self.length_scale**2)
        alpha = self.alpha
        k0 = (1 + diff * diff * inv_scale) ** (-alpha)
        if eval_gradient:
            dk0 = (
                -2
                * diff
                * (1 + diff * diff * inv_scale) ** (-1 - alpha)
                * inv_scale
                * alpha
            )
        else:
            dk0 = None
        return k0, dk0


def qarbf_args(arbf_base):
    ndim = len(arbf_base.length_scale)
    length_scale = arbf_base.length_scale
    scale = [arbf_base.scale[0]]
    scale += [arbf_base.scale[1]] * (ndim)
    scale += [arbf_base.scale[2]] * (ndim * (ndim - 1) // 2)
    return ndim, np.array(length_scale), scale


def arbf_args(arbf_base):
    ndim = len(arbf_base.length_scale)
    length_scale = arbf_base.length_scale
    order = arbf_base.order
    scale = [arbf_base.scale[0]]
    if order > 0:
        scale += [arbf_base.scale[1]] * (ndim)
    if order > 1:
        scale += [arbf_base.scale[2]] * (ndim * (ndim - 1) // 2)
    if order > 2:
        scale += [arbf_base.scale[3]] * (ndim * (ndim - 1) * (ndim - 2) // 6)
    if order > 3:
        raise ValueError("Order too high for mapping")
    return ndim, np.array(length_scale), scale, order


class QARBF(StationaryKernelMixin, Kernel):
    """
    ARBF, except order is restricted to 2 and
    the algorithm is more efficient.
    """

    def __init__(self, ndim, length_scale, scale, scale_bounds=(1e-5, 1e5)):
        super(QARBF, self).__init__()
        self.ndim = ndim
        self.scale = scale
        self.scale_bounds = scale_bounds
        self.length_scale = length_scale

    @property
    def hyperparameter_scale(self):
        return Hyperparameter("scale", "numeric", self.scale_bounds, len(self.scale))

    def diag(self, X):
        return np.diag(self.__call__(X))

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        diff = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) / self.length_scale
        k0 = np.exp(-0.5 * diff**2)
        sk = np.zeros((X.shape[0], Y.shape[0], len(self.scale)))
        sk[:, :, 0] = self.scale[0]
        t = 1
        for i in range(self.ndim):
            sk[:, :, t] = self.scale[t] * k0[:, :, i]
            t += 1
        for i in range(self.ndim - 1):
            for j in range(i + 1, self.ndim):
                sk[:, :, t] = self.scale[t] * k0[:, :, i] * k0[:, :, j]
                t += 1
        k = np.sum(sk, axis=-1)
        print(self.scale)
        if eval_gradient:
            return k, sk
        return k

    def get_sub_kernel(self, inds, scale_ind, X, Y):
        if inds is None:
            return self.scale[0] * np.ones((X.shape[0], Y.shape[0]))
        if isinstance(inds, int):
            diff = (X[:, np.newaxis, inds] - Y[np.newaxis, :]) / self.length_scale[inds]
            k0 = np.exp(-0.5 * diff**2)
            return self.scale[scale_ind] * k0
        else:
            diff = (
                X[:, np.newaxis, inds[0]] - Y[np.newaxis, :, 0]
            ) / self.length_scale[inds[0]]
            k0 = np.exp(-0.5 * diff**2)
            diff = (
                X[:, np.newaxis, inds[1]] - Y[np.newaxis, :, 1]
            ) / self.length_scale[inds[1]]
            k0 *= np.exp(-0.5 * diff**2)
            return self.scale[scale_ind] * k0

    def get_funcs_for_spline_conversion(self):
        funcs = [lambda x, y: self.get_sub_kernel(None, 0, x, y)]
        t = 1
        for i in range(self.ndim):
            funcs.append(lambda x, y: self.get_sub_kernel(i, t, x, y))
            t += 1
        for i in range(self.ndim - 1):
            for j in range(i + 1, self.ndim):
                funcs.append(lambda x, y: self.get_sub_kernel((i, j), t, x, y))
                t += 1
        return funcs


class PartialARBF(DiffARBF):
    """
    ARBF where subset of X is selected.
    """

    def __init__(
        self,
        order=1,
        length_scale=1.0,
        length_scale_bounds=(1e-5, 1e5),
        scale=1.0,
        scale_bounds=(1e-5, 1e5),
        start=1,
        active_dims=None,
    ):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.order = order
        self.scale = scale  # if (scale is None) else [1.0] * (order + 1)
        self.scale_bounds = scale_bounds
        self.start = start
        self.active_dims = active_dims

    def __call__(self, X, Y=None, eval_gradient=False, get_sub_kernels=False):
        # hasattr check for back-compatibility
        if not np.iterable(self.scale):
            self.scale = [self.scale] * (self.order + 1)
        if (not hasattr(self, "active_dims")) or (self.active_dims is None):
            X = X[:, self.start :]
            if Y is not None:
                Y = Y[:, self.start :]
        else:
            X = X[:, self.active_dims]
            if Y is not None:
                Y = Y[:, self.active_dims]
        return super(PartialARBF, self).__call__(X, Y, eval_gradient, get_sub_kernels)


class SingleRBF(RBF):
    """
    RBF kernel with single index of X selected.
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), index=1):
        super(SingleRBF, self).__init__(length_scale, length_scale_bounds)
        self.index = index

    def __call__(self, X, Y=None, eval_gradient=False):

        X = X[:, self.index : self.index + 1]
        if Y is not None:
            Y = Y[:, self.index : self.index + 1]
        return super(SingleRBF, self).__call__(X, Y, eval_gradient)


class SingleDot(DotProduct):
    """
    DotProduct kernel with single index of X selected.
    """

    def __init__(self, sigma_0=1.0, sigma_0_bounds=(1e-05, 100000.0), index=0):
        super(SingleDot, self).__init__(sigma_0, sigma_0_bounds)
        self.index = index

    def __call__(self, X, Y=None, eval_gradient=False):
        X = X[:, self.index : self.index + 1]
        if Y is not None:
            Y = Y[:, self.index : self.index + 1]
        return super(SingleDot, self).__call__(X, Y, eval_gradient)


class DensityNoise(StationaryKernelMixin, GenericKernelMixin, Kernel):
    def __init__(self, index=0):
        self.index = index

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = np.diag(self.diag(X))
            if eval_gradient:
                grad = np.empty((_num_samples(X), _num_samples(X), 0))
                return K, grad
            else:
                return K
        else:
            return np.zeros((_num_samples(X), _num_samples(Y)))

    def diag(self, X):
        return 1 / X[:, self.index] ** 2


class ExponentialDensityNoise(StationaryKernelMixin, GenericKernelMixin, Kernel):
    def __init__(self, exponent=1.0, exponent_bounds=(0.1, 10)):
        self.exponent = exponent
        self.exponent_bounds = exponent_bounds

    @property
    def hyperparameter_exponent(self):
        return Hyperparameter("exponent", "numeric", self.exponent_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = np.diag(self.diag(X))
            if eval_gradient and not self.hyperparameter_exponent.fixed:
                rho = X[:, 0]
                grad = np.empty((_num_samples(X), _num_samples(X), 1))
                grad[:, :, 0] = np.diag(
                    -self.exponent * np.log(rho) / rho**self.exponent
                )
                return K, grad
            elif eval_gradient:
                grad = np.zeros((X.shape[0], X.shape[0], 0))
                return K, grad
            else:
                return K
        else:
            return np.zeros((_num_samples(X), _num_samples(Y)))

    def diag(self, X):
        return 1 / X[:, 0] ** self.exponent

    def __repr__(self):
        return "{0}(exponent={1:.3g})".format(self.__class__.__name__, self.exponent)


class FittedDensityNoise(StationaryKernelMixin, GenericKernelMixin, Kernel):
    """
    Kernel to model the noise of the exchange enhancement factor based
    on the density. 1 / (1 + decay_rate * rho)
    """

    def __init__(self, decay_rate=4.0, decay_rate_bounds=(1e-5, 1e5)):
        self.decay_rate = decay_rate
        self.decay_rate_bounds = decay_rate_bounds

    @property
    def hyperparameter_decay_rate(self):
        return Hyperparameter("decay_rate", "numeric", self.decay_rate_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = np.diag(self.diag(X))
            if eval_gradient and not self.hyperparameter_decay_rate.fixed:
                rho = X[:, 0]
                grad = np.empty((_num_samples(X), _num_samples(X), 1))
                grad[:, :, 0] = np.diag(-rho / (1 + self.decay_rate * rho) ** 2)
                return K, grad
            elif eval_gradient:
                grad = np.zeros((X.shape[0], X.shape[0], 0))
                return K, grad
            else:
                return K
        else:
            return np.zeros((_num_samples(X), _num_samples(Y)))

    def diag(self, X):
        rho = X[:, 0]
        return 1 / (1 + self.decay_rate * rho)

    def __repr__(self):
        return "{0}(decay_rate={1:.3g})".format(
            self.__class__.__name__, self.decay_rate
        )


class ADKernel(Kernel):
    def __init__(self, k, active_dims):
        self.k = k
        self.active_dims = active_dims

    def get_params(self, deep=True):
        params = dict(k=self.k, active_dims=self.active_dims)
        if deep:
            deep_items = self.k.get_params().items()
            params.update(("k__" + k, val) for k, val in deep_items)

        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        return [
            Hyperparameter(
                "k__" + hyperparameter.name,
                hyperparameter.value_type,
                hyperparameter.bounds,
                hyperparameter.n_elements,
            )
            for hyperparameter in self.k.hyperparameters
        ]

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.
        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.
        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return self.k.theta

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.
        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        self.k.theta = theta

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.
        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        return self.k.bounds

    def __eq__(self, b):
        return self.k == b.k and self.active_dims == b.active_dims

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.k.is_stationary()

    @property
    def requires_vector_input(self):
        """Returns whether the kernel is stationary."""
        return self.k.requires_vector_input

    def __call__(self, X, Y=None, eval_gradient=False):
        X = X[:, self.active_dims]
        if Y is not None:
            Y = Y[:, self.active_dims]
        return self.k.__call__(X, Y, eval_gradient)

    def diag(self, X):
        return self.k.diag(X)

    def __repr__(self):
        return self.k.__repr__()


class SpinSymKernel(ADKernel):
    def __init__(self, k, up_active_dims, down_active_dims):
        self.k = k
        self.up_active_dims = up_active_dims
        self.down_active_dims = down_active_dims

    def get_params(self, deep=True):
        params = dict(
            k=self.k,
            up_active_dims=self.up_active_dims,
            down_active_dims=self.down_active_dims,
        )
        if deep:
            deep_items = self.k.get_params().items()
            params.update(("k__" + k, val) for k, val in deep_items)

        return params

    def __call__(self, X, Y=None, eval_gradient=False):
        Xup = X[:, self.up_active_dims]
        if Y is not None:
            Yup = Y[:, self.up_active_dims]
        else:
            Yup = None
        kup = self.k.__call__(Xup, Yup, eval_gradient)
        Xdown = X[:, self.down_active_dims]
        if Y is not None:
            Ydown = Y[:, self.down_active_dims]
        else:
            Ydown = None
        kdown = self.k.__call__(Xdown, Ydown, eval_gradient)
        if eval_gradient:
            return kup[0] + kdown[0], kup[1] + kdown[1]
        else:
            return kup + kdown


class SubsetRBF(_SubsetMixin, DiffRBF):
    pass


class SubsetARBF(_SubsetMixin, DiffARBF):
    pass


class SubsetAddLLRBF(_SubsetMixin, DiffAddLLRBF):
    pass


class SubsetAddRQ(_SubsetMixin, DiffAddRQ):
    pass


class SubsetPoly(_SubsetMixin, DiffPolyKernel):
    pass


class SpinSymRBF(_SpinSymMixin, DiffRBF):
    pass


class SpinSymARBF(_SpinSymMixin, DiffARBF):
    pass


class SpinSymPoly(_SpinSymMixin, DiffPolyKernel):
    pass
