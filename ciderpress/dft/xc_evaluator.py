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

import ctypes

import numpy as np
import yaml
from interpolation.splines.eval_cubic_numba import (
    vec_eval_cubic_splines_G_1,
    vec_eval_cubic_splines_G_2,
    vec_eval_cubic_splines_G_3,
    vec_eval_cubic_splines_G_4,
)

from ciderpress.lib import load_library
from ciderpress.models.kernels import (
    DiffConstantKernel,
    DiffProduct,
    DiffRBF,
    SubsetRBF,
)

libcider = load_library("libmcider")


def get_vec_eval(grid, coeffs, X, N):
    """
    Call the numba-accelerated spline evaluation routines from the
    interpolation package. Also returns derivatives

    Args:
        grid: start and end points + number of grids in each dimension
        coeffs: coefficients of the spline
        X: coordinates to interpolate
        N: dimension of the interpolation (between 1 and 4, inclusive)
    """
    coeffs = np.expand_dims(coeffs, coeffs.ndim)
    y = np.zeros((X.shape[0], 1))
    dy = np.zeros((X.shape[0], N, 1))
    a_, b_, orders = zip(*grid)
    if N == 1:
        vec_eval_cubic_splines_G_1(a_, b_, orders, coeffs, X, y, dy)
    elif N == 2:
        vec_eval_cubic_splines_G_2(a_, b_, orders, coeffs, X, y, dy)
    elif N == 3:
        vec_eval_cubic_splines_G_3(a_, b_, orders, coeffs, X, y, dy)
    elif N == 4:
        vec_eval_cubic_splines_G_4(a_, b_, orders, coeffs, X, y, dy)
    else:
        raise ValueError("invalid dimension N")
    return np.squeeze(y, -1), np.squeeze(dy, -1)


class XCEvalSerializable:
    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d):
        raise NotImplementedError

    def dump(self, fname):
        """
        Save the Evaluator to a file name fname as yaml format.
        """
        state_dict = self.to_dict()
        with open(fname, "w") as f:
            yaml.dump(state_dict, f)

    @classmethod
    def load(cls, fname):
        """
        Load an instance of this class from yaml
        """
        with open(fname, "r") as f:
            state_dict = yaml.load(f, Loader=yaml.CLoader)
        return cls.from_dict(state_dict)


class KernelEvalBase:

    mode = None
    feature_list = None
    _mul_basefunc = None
    _add_basefunc = None

    @property
    def N1(self):
        raise NotImplementedError

    def get_descriptors(self, X0T, force_polarize=False):
        """
        Compute and return transformed descriptor matrix X1.

        Args:
            rho: Density/Kinetic energy density representation.
            X0 (nspin, N0, Nsamp): Raw features
            nspin (1 or 2): Number of spins

        Returns:
            X1 (Nsamp, N1)
        """
        nspin, N0, Nsamp = X0T.shape
        if force_polarize and self.mode == "POL" and nspin == 1:
            X0T = np.concatenate([X0T, X0T], axis=0)
            nspin = 2
        N1 = self.N1
        if self.mode == "SEP" or self.mode == "POL":
            X1 = np.zeros((nspin, Nsamp, N1))
            for s in range(nspin):
                self.feature_list.fill_vals_(X1[s].T, X0T[s])
            X1 = X1.reshape(nspin * Nsamp, N1)
        elif self.mode == "NPOL":
            X0T_sum = X0T.mean(0)
            X1 = np.zeros((Nsamp, N1))
            self.feature_list.fill_vals_(X1.T, X0T_sum)
        else:
            raise NotImplementedError
        if force_polarize and self.mode == "POL":
            return X1.reshape(nspin, Nsamp, N1)
        return X1

    def get_descriptors_with_mul(self, X0T, multiplier):
        """
        Compute and return transformed descriptor matrix X1.

        Args:
            rho: Density/Kinetic energy density representation.
            X0 (nspin, N0, Nsamp): Raw features
            nspin (1 or 2): Number of spins

        Returns:
            X1 (Nsamp, N1)
        """
        nspin, N0, Nsamp = X0T.shape
        N1 = self.N1
        if self.mode == "SEP" or self.mode == "POL":
            X1 = np.zeros((nspin, Nsamp, N1))
            for s in range(nspin):
                self.feature_list.fill_vals_(X1[s].T, X0T[s])
                X1[s] *= multiplier[:, None]
            X1 = X1.reshape(nspin * Nsamp, N1)
        elif self.mode == "NPOL":
            X0T_sum = X0T.mean(0)
            X1 = np.zeros((Nsamp, N1))
            self.feature_list.fill_vals_(X1.T, X0T_sum)
            X1[:] *= multiplier[:, None]
        else:
            raise NotImplementedError
        return X1

    def _baseline(self, X0T, base_func):
        nspin, N0, Nsamp = X0T.shape
        if self.mode == "SEP":
            ms = []
            dms = []
            for s in range(nspin):
                m, dm = base_func(X0T[s : s + 1])
                ms.append(m / nspin)
                dms.append(dm / nspin)
            return np.stack(ms), np.concatenate(dms, axis=0)
        elif self.mode == "NPOL" or self.mode == "POL":
            ms = []
            dms = []
            m, dm = base_func(X0T)
            return m, dm
        else:
            raise NotImplementedError

    def multiplicative_baseline(self, X0T):
        return self._baseline(X0T, self._mul_basefunc)

    def additive_baseline(self, X0T):
        return self._baseline(X0T, self._add_basefunc)

    def apply_descriptor_grad(self, X0T, dfdX1, force_polarize=False):
        """

        Args:
            X0T (nspin, N0, Nsamp): raw feature descriptors
            dfdX1 (nspin * Nsamp, N1): derivative with respect to transformed
                descriptors X1

        Returns:
            dfdX0T (nspin, N0, Nsamp): derivative with respect
                to raw descriptors X0.
        """
        nspin, N0, Nsamp = X0T.shape
        N1 = self.N1
        if force_polarize and dfdX1.shape[0] == 2 and nspin == 1:
            dfdX1 = dfdX1[:1]
        if self.mode == "SEP" or self.mode == "POL":
            dfdX0T = np.zeros_like(X0T)
            dfdX1 = dfdX1.reshape(nspin, Nsamp, N1)
            for s in range(nspin):
                self.feature_list.fill_derivs_(dfdX0T[s], dfdX1[s].T, X0T[s])
        elif self.mode == "NPOL":
            dfdX0T = np.zeros_like(X0T)
            self.feature_list.fill_derivs_(dfdX0T[0], dfdX1.T, X0T.mean(0))
            for s in range(1, nspin):
                dfdX0T[s] = dfdX0T[0]
            dfdX0T /= nspin
        else:
            raise NotImplementedError
        return dfdX0T

    def apply_baseline(self, X0T, f, dfdX0T=None, add_base=True):
        """

        Args:
            f:
            dfdX0T:
            add_base:

        Returns:

        """
        m, dm = self.multiplicative_baseline(X0T)
        add_base = add_base and self.additive_baseline is not None
        if add_base:
            a, da = self.additive_baseline(X0T)

        res = f * m
        if add_base:
            res += a
        if dfdX0T is not None:
            if self.mode == "SEP":
                dres = dfdX0T * m[:, np.newaxis] + f[:, np.newaxis] * dm
            elif self.mode == "NPOL" or self.mode == "POL":
                dres = dfdX0T * m + f * dm
            else:
                raise NotImplementedError
            if add_base:
                dres += da
            return res, dres
        return res


class FuncEvaluator:
    def build(self, *args, **kwargs):
        pass

    def __call__(self, X1, res=None, dres=None):
        raise NotImplementedError


class KernelEvaluator(FuncEvaluator, XCEvalSerializable):
    def __init__(self, kernel, X1ctrl, alpha):
        self.X1ctrl = X1ctrl
        self.kernel = kernel
        self.alpha = alpha

    def __call__(self, X1, res=None, dres=None):
        if res is None:
            res = np.zeros(X1.shape[0])
        elif res.shape != X1.shape[:1]:
            raise ValueError
        if dres is None:
            dres = np.zeros(X1.shape)
        elif dres.shape != X1.shape:
            raise ValueError
        N = X1.shape[0]
        dn = 2000
        for i0 in range(0, N, dn):
            i1 = min(N, i0 + dn)
            k, dk = self.kernel.k_and_deriv(X1[i0:i1], self.X1ctrl)
            res[i0:i1] += k.dot(self.alpha)
            dres[i0:i1] += np.einsum("gcn,c->gn", dk, self.alpha)
        return res, dres


class RBFEvaluator(FuncEvaluator, XCEvalSerializable):
    _fn = libcider.evaluate_se_kernel

    def __init__(self, kernel, X1ctrl, alpha):
        if isinstance(kernel, DiffProduct):
            assert isinstance(kernel.k1, DiffConstantKernel)
            scale = kernel.k1.constant_value
            kernel = kernel.k2
        else:
            assert isinstance(kernel, DiffRBF)
            scale = 1.0
        if isinstance(kernel, SubsetRBF):
            if isinstance(kernel.indexes, slice):
                i = kernel.indexes
                start = i.start
                step = i.step if i.step is not None else 1
                stop = (
                    i.stop
                    if i.stop is not None
                    else (len(kernel.length_scale) + i.start) // step
                )
                indexes = [i for i in range(start, stop, step)]
            indexes = np.array(indexes, dtype=np.int32)
        else:
            indexes = np.arange(len(kernel.length_scale), dtype=np.int32)
        self._X1ctrl = np.ascontiguousarray(X1ctrl)
        self._alpha = np.ascontiguousarray(alpha * scale)
        self._exps = np.ascontiguousarray(0.5 / kernel.length_scale**2)
        self._nctrl, self._nfeat = self._X1ctrl.shape[-2:]
        self._indexes = np.ascontiguousarray(indexes)

    def __call__(self, X1, res=None, dres=None):
        X1 = np.ascontiguousarray(X1[..., self._indexes])
        if res is None:
            res = np.zeros(X1.shape[0])
        elif res.shape != (X1.shape[-2],):
            raise ValueError
        if dres is None:
            dres = np.zeros(X1.shape)
        elif dres.shape != X1.shape:
            raise ValueError
        n = X1.shape[-2]
        for arr in [res, dres, X1]:
            assert arr.flags.c_contiguous
        self._fn(
            res.ctypes.data_as(ctypes.c_void_p),
            dres.ctypes.data_as(ctypes.c_void_p),
            X1.ctypes.data_as(ctypes.c_void_p),
            self._X1ctrl.ctypes.data_as(ctypes.c_void_p),
            self._alpha.ctypes.data_as(ctypes.c_void_p),
            self._exps.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(n),
            ctypes.c_int(self._nctrl),
            ctypes.c_int(self._nfeat),
        )
        return res, dres


class AntisymRBFEvaluator(RBFEvaluator):
    _fn = libcider.evaluate_se_kernel_antisym

    def __init__(self, kernel, X1ctrl, alpha):
        super(AntisymRBFEvaluator, self).__init__(kernel, X1ctrl, alpha)
        if isinstance(kernel, DiffProduct):
            assert isinstance(kernel.k1, DiffConstantKernel)
            kernel.k1.constant_value
            kernel = kernel.k2
        else:
            assert isinstance(kernel, DiffRBF)
        self._indexes = np.arange(len(kernel.length_scale) + 1, dtype=np.int32)

    def __call__(self, X1, res=None, dres=None):
        assert X1.ndim == 2 or (X1.ndim == 3 and X1.shape[0] == 1)
        return super(AntisymRBFEvaluator, self).__call__(X1, res, dres)


class SpinRBFEvaluator(RBFEvaluator):
    _fn = libcider.evaluate_se_kernel_spin

    def __init__(self, kernel, X1ctrl, alpha):
        super(SpinRBFEvaluator, self).__init__(kernel, X1ctrl, alpha)
        if isinstance(kernel, DiffProduct):
            self._alpha *= kernel.k1.constant_value

    def __call__(self, X1, res=None, dres=None):
        assert X1.ndim == 3 and X1.shape[0] == 2
        return super(SpinRBFEvaluator, self).__call__(X1, res, dres)


class SplineSetEvaluator(FuncEvaluator, XCEvalSerializable):
    def __init__(self, scale, ind_sets, spline_grids, coeff_sets, const=0):
        self.scale = scale
        self.nterms = len(self.scale)
        assert len(ind_sets) == self.nterms
        self.ind_sets = ind_sets
        assert len(spline_grids) == self.nterms
        self.spline_grids = spline_grids
        assert len(coeff_sets) == self.nterms
        self.coeff_sets = coeff_sets
        self.const = const

    def __call__(self, X1, res=None, dres=None):
        """
        Note: This function adds to res and dres if they are passed,
        rather than writing them from scratch.

        Args:
            X1:
            res:
            dres:

        Returns:

        """
        if res is None:
            res = np.zeros(X1.shape[0]) + self.const
        elif res.shape != X1.shape[:1]:
            raise ValueError
        else:
            res[:] += self.const
        if dres is None:
            dres = np.zeros(X1.shape)
        elif dres.shape != X1.shape:
            raise ValueError
        for t in range(self.nterms):
            ind_set = self.ind_sets[t]
            y, dy = get_vec_eval(
                self.spline_grids[t], self.coeff_sets[t], X1[:, ind_set], len(ind_set)
            )
            res[:] += y * self.scale[t]
            dres[:, ind_set] += dy * self.scale[t]
        return res, dres

    def to_dict(self):
        return {
            "scale": np.array(self.scale),
            "ind_sets": self.ind_sets,
            "spline_grids": self.spline_grids,
            "coeff_sets": self.coeff_sets,
            "const": self.const,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["scale"],
            d["ind_sets"],
            d["spline_grids"],
            d["coeff_sets"],
            const=d["const"],
        )


class NNEvaluator(FuncEvaluator, XCEvalSerializable):
    def __init__(self, model, device=None):
        self.model = model
        self.device = device
        import torch
        from torch.autograd import grad

        self.cuda_is_available = torch.cuda.is_available
        self.device_init = torch.device
        self.tensor_init = torch.tensor
        self.eval_grad = grad
        self.set_device(device=device)

    def set_device(self, device=None):
        if device is None:
            if self.cuda_is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        if isinstance(device, str):
            device = self.device_init(device)
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()

    def build(self, *args, **kwargs):
        self.set_device()

    def __call__(self, X1, res=None, dres=None):
        if res is None:
            res = np.zeros(X1.shape[0])
        elif res.shape != X1.shape[:1]:
            raise ValueError
        if dres is None:
            dres = np.zeros(X1.shape)
        elif dres.shape != X1.shape:
            raise ValueError
        X1_torch = self.tensor_init(X1, device=self.device, requires_grad=True)
        self.model.zero_grad()
        output = self.model(X1_torch)
        output_grad = self.eval_grad(output.sum(), X1_torch, retain_graph=False)[0]
        res[:] += output.cpu().detach().numpy()
        dres[:] += output_grad.cpu().detach().numpy()
        return res, dres


class GlobalLinearEvaluator(FuncEvaluator, XCEvalSerializable):
    def __init__(self, consts):
        self.consts = np.asarray(consts, dtype=np.float64, order="C")

    def __call__(self, X1, res=None, dres=None):
        if res is None:
            res = np.zeros(X1.shape[0])
        elif res.shape != X1.shape[:1]:
            raise ValueError
        if dres is None:
            dres = np.zeros(X1.shape)
        elif dres.shape != X1.shape:
            raise ValueError
        res[:] += X1.dot(self.consts)
        dres[:] += self.consts
        return res, dres


class MappedDFTKernel(KernelEvalBase, XCEvalSerializable):
    """
    This class evaluates the XC term arising from a single DFTKernel
    object, using one or a list of FuncEvaluator objects.
    """

    def __init__(
        self,
        fevals,
        feature_list,
        mode,
        multiplicative_baseline,
        additive_baseline=None,
    ):
        self.fevals = fevals
        if not isinstance(self.fevals, list):
            self.fevals = [self.fevals]
        for feval in self.fevals:
            if not isinstance(feval, FuncEvaluator):
                raise ValueError
        self.mode = mode
        self.feature_list = feature_list
        self._mul_basefunc = multiplicative_baseline
        self._add_basefunc = additive_baseline

    @property
    def N1(self):
        return self.feature_list.nfeat

    def __call__(self, X0T, add_base=True, rhocut=0):
        X1 = self.get_descriptors(X0T, force_polarize=True)
        Nsamp_internal = X1.shape[-2]
        f = np.zeros(Nsamp_internal)
        df = np.zeros_like(X1)
        for feval in self.fevals:
            feval(X1, f, df)
        if self.mode == "SEP":
            f = f.reshape(X0T.shape[0], -1)
        dfdX0T = self.apply_descriptor_grad(X0T, df, force_polarize=True)
        res, dres = self.apply_baseline(X0T, f, dfdX0T)
        if rhocut > 0:
            if self.mode == "SEP":
                cond = X0T[:, 0] < rhocut
                for s in range(X0T.shape[0]):
                    res[s][cond[s]] = 0.0
                    dres[s][:, cond[s]] = 0.0
            else:
                cond = X0T[:, 0].sum(0) < rhocut
                res[..., cond] = 0.0
                dres[..., cond] = 0.0
        if self.mode == "SEP":
            res = res.sum(0)
        return res, dres

    def to_dict(self):
        return {
            "fevals": [fe.to_dict() for fe in self.fevals],
            "feature_list": self.feature_list.to_dict(),
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, d):
        raise NotImplementedError


class ModelWithNormalizer:
    def __init__(self, model, normalizer):
        """
        Initialize a ModelWithNormalizer.

        Args:
            model (MappedFunctional): Model evaluator
            normalizer (FeatNormalizerList): Instructions for normalizing
                the features.
        """
        if model.nfeat != normalizer.nfeat:
            raise ValueError
        self.model = model
        self.normalizer = normalizer

    def get_x0_usps(self):
        usp0 = self.model.settings.get_feat_usps()
        usp1 = self.normalizer.get_usps()
        return [u0 + u1 for u0, u1 in zip(usp0, usp1)]

    @property
    def nfeat(self):
        return self.normalizer.nfeat

    def __call__(self, X0T, rho_data, rhocut=0):
        assert X0T.ndim == 3
        nspin = X0T.shape[0]
        X0Tnorm = X0T.copy()
        rho = nspin * rho_data[:, 0]
        sigma = (
            nspin * nspin * np.einsum("xg,xg->g", rho_data[:, 1:4], rho_data[:, 1:4])
        )
        tau = nspin * rho_data[:, 4]
        vrho = np.zeros_like(rho)
        vsigma = np.zeros_like(rho)
        vtau = np.zeros_like(rho)
        for i in range(self.nfeat):
            self.normalizer[i].apply_norm_fwd_(X0Tnorm[:, i], rho, sigma, tau)
        res, dres = self.model(X0Tnorm, rhocut=rhocut)
        for i in range(self.nfeat):
            self.normalizer[i].apply_norm_bwd_(
                dres[:, i], X0T[:, i], rho, sigma, tau, vrho, vsigma, vtau
            )
        return res, dres


class MappedXC:
    def __init__(self, mapped_kernels, settings, libxc_baseline=None):
        """

        Args:
            mapped_kernels (list of DFTKernel):
            settings (FeatureSettings):
            libxc_baseline (str):
        """
        self.kernels = mapped_kernels
        self.settings = settings
        self.libxc_baseline = libxc_baseline

    def __call__(self, X0T, rhocut=0):
        """
        Evaluate functional from normalized features
        Args:
            X0T (array): normalized features
            rhocut (float): low-density cutoff

        Returns:
            res, dres (array, array), XC energy contribution
                and functional derivative
        """
        res, dres = 0, 0
        for kernel in self.kernels:
            tmp, dtmp = kernel(X0T, rhocut=rhocut)
            res += tmp
            dres += dtmp
        return res, dres

    @property
    def normalizer(self):
        return self.settings.normalizers

    @property
    def nfeat(self):
        return self.normalizer.nfeat
