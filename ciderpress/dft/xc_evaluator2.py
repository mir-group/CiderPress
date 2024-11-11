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

import numpy as np

from ciderpress.dft.baselines import get_libxc_baseline
from ciderpress.dft.xc_evaluator import FuncEvaluator, XCEvalSerializable


class KernelEvalBase2:

    mode = None
    feature_list = None
    _mul_basefunc: str = None
    _add_basefunc: str = None

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

    def _get_baseline(self, xcid, rho_tuple):
        if self.mode == "SEP":
            nspin = rho_tuple[0].shape[0]
            if nspin == 1:
                res = list(get_libxc_baseline(xcid, rho_tuple))
                res[0] = res[0][None, :]
                return tuple(res)
            else:
                assert nspin == 2
                sep_res = [np.zeros_like(rho_tuple[0]), np.zeros_like(rho_tuple[0])]
                if len(rho_tuple) > 1:
                    sep_res.append(np.zeros_like(rho_tuple[1]))
                if len(rho_tuple) > 2:
                    sep_res.append(np.zeros_like(rho_tuple[2]))
                sep_res = tuple(sep_res)
                for s in range(nspin):
                    tuple_s = [2 * rho_tuple[0][s : s + 1]]
                    if len(rho_tuple) > 1:
                        tuple_s.append(4 * rho_tuple[1][2 * s : 2 * s + 1])
                    if len(rho_tuple) > 2:
                        tuple_s.append(2 * rho_tuple[2][s : s + 1])
                    res = get_libxc_baseline(xcid, tuple_s)
                    sep_res[0][s] = 0.5 * res[0]
                    sep_res[1][s] = res[1]
                    if len(res) > 2:
                        sep_res[2][2 * s] = 2 * res[2]
                    if len(res) > 3:
                        sep_res[3][s] = res[3]
                return sep_res
        else:
            return get_libxc_baseline(xcid, rho_tuple)

    def multiplicative_baseline(self, rho_tuple):
        return self._get_baseline(self._mul_basefunc, rho_tuple)

    def additive_baseline(self, rho_tuple):
        if self._add_basefunc is None:
            return None
        return self._get_baseline(self._add_basefunc, rho_tuple)

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

    def apply_libxc_baseline_(self, f, dfdX1, rho_tuple, vrho_tuple, add_base=True):
        m_res = self.multiplicative_baseline(rho_tuple)
        add_base = add_base and self._add_basefunc is not None
        for vrho, vm in zip(vrho_tuple, m_res[1:]):
            if self.mode == "SEP" and vrho.shape[0] == 3:
                vrho[::2] += vm[::2] * f
            else:
                vrho[:] += vm * f
        if self.mode == "SEP":
            assert m_res[0].shape == f.shape == dfdX1.shape[:2]
        elif self.mode == "POL":
            assert m_res[0].shape == f.shape == (dfdX1.shape[1],)
        else:
            assert m_res[0].shape == f.shape == (dfdX1.shape[0],)
        dfdX1[:] *= m_res[0][..., None]
        f[:] *= m_res[0]
        if add_base:
            a_res = self.additive_baseline(rho_tuple)
            f[:] += a_res[0]
            for vrho, va in zip(vrho_tuple, a_res[1:]):
                vrho[:] += va


class MappedDFTKernel2(KernelEvalBase2, XCEvalSerializable):
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
        """
        fevals: FuncEvaluator
        """
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

    def __call__(self, X0T, rho_tuple, vrho_tuple, rhocut=0):
        X1 = self.get_descriptors(X0T, force_polarize=True)
        Nsamp_internal = X1.shape[-2]
        f = np.zeros(Nsamp_internal)
        df = np.zeros_like(X1)
        for feval in self.fevals:
            feval(X1, f, df)
        if self.mode == "POL" and X0T.shape[0] == 1:
            df = 2 * df[:1]
        if self.mode == "SEP":
            f = f.reshape(X0T.shape[0], -1)
            df = df.reshape(X0T.shape[0], -1, self.N1)
        if rhocut > 0:
            cond = rho_tuple[0] < rhocut
            if self.mode == "SEP":
                f[cond] = 0.0
                df[cond] = 0.0
            else:
                scond = rho_tuple[0].sum(0) < rhocut
                f[scond] = 0.0
                if self.mode == "POL":
                    df[cond, :] = 0.0
                else:
                    df[scond, :] = 0.0
        self.apply_libxc_baseline_(f, df, rho_tuple, vrho_tuple)
        dfdX0T = self.apply_descriptor_grad(X0T, df, force_polarize=True)
        if self.mode == "SEP":
            f = f.sum(0)
        return f, dfdX0T

    def to_dict(self):
        return {
            "fevals": [fe.to_dict() for fe in self.fevals],
            "feature_list": self.feature_list.to_dict(),
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, d):
        raise NotImplementedError


class MappedXC2:
    def __init__(self, mapped_kernels, settings, libxc_baseline=None):
        """

        Args:
            mapped_kernels (list of MappedDFTKernel2):
            settings (FeatureSettings):
            libxc_baseline (str):
        """
        self.kernels = mapped_kernels
        self.settings = settings
        self.libxc_baseline = libxc_baseline

    def __call__(self, X0T, rho_tuple, vrho_tuple=None, rhocut=0):
        """
        Evaluate functional from normalized features
        Args:
            X0T (array): normalized features
            rhocut (float): low-density cutoff

        Returns:
            res, dres (array, array), XC energy contribution
                and functional derivative
        """
        vrho_tuple = tuple([np.zeros_like(r, order="F") for r in rho_tuple])
        res, dres = 0, 0
        for kernel in self.kernels:
            tmp, dtmp = kernel(X0T, rho_tuple, vrho_tuple, rhocut=rhocut)
            res += tmp
            dres += dtmp
        return res, dres, vrho_tuple

    @property
    def normalizer(self):
        return self.settings.normalizers

    @property
    def nfeat(self):
        return self.normalizer.nfeat
