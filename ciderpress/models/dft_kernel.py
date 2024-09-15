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
from pyscf.lib.scipy_helper import pivoted_cholesky

from ciderpress.dft.xc_evaluator import KernelEvalBase, MappedDFTKernel
from ciderpress.dft.xc_evaluator2 import KernelEvalBase2, MappedDFTKernel2


class DFTKernel(KernelEvalBase):
    """
    Raw descriptors are denoted X0.
    Transformed descriptors are denoted as X1.
    Nctrl is the number of control points for the kernel.
    N0 is the number of raw features.
    N1 is the number of transformed features.

    Attributes:
        mode (str): One of (SEP, POL, OSPOL, SSPOL, NPOL).
        X1ctrl (Nctrl, N1): Control points for kernel.

    k: kernel
    dkdX0T: kernel derivative with respect to raw features.
    m: Multiplicative baseline
    a: Additive baseline
    dm: Derivative of multiplicative baseline wrt features
    da: Derivative of additive baseline wrt features
    SEP array shapes:
        k: (Nctrl, nspin, Nsamp)
        m and a: (nspin, Nsamp)
        dkdX0T: (Nctrl, nspin, N0, Nsamp)
        dm and da: (nspin, N0, Nsamp)
    NPOL, POL, OSPOL, SSPOL array shapes:
        k: (Nctrl, Nsamp)
        m and a: Nsamp
        dkdX0t: (Nctrl, nspin, N0, Nsamp)
        dm and da: (N0, nspin, Nsamp)
    """

    def __init__(
        self,
        kernel,
        feature_list,
        mode,
        multiplicative_baseline,
        additive_baseline=None,
        ctrl_tol=1e-5,
        ctrl_nmax=None,
        component=None,
    ):
        self.kernel = kernel
        self.feature_list = feature_list
        self.mode = mode
        self._mul_basefunc = multiplicative_baseline
        self._add_basefunc = additive_baseline
        self.ctrl_tol = ctrl_tol
        self.ctrl_nmax = ctrl_nmax
        if component is None:
            component = "x"
        assert component in ["x", "c", "xc"]
        self.component = component

        self.X1ctrl = None
        self.alpha = None
        self.base_dict = {}
        self.cov_dict = {}
        self.dbase_dict = {}
        self.dcov_dict = {}
        self.rxn_cov_list = []

    def set_kernel(self, kernel):
        self.kernel = kernel

    @property
    def N1(self):
        return self.feature_list.nfeat

    @property
    def Nctrl(self):
        if self.X1ctrl is None:
            raise ValueError("X1ctrl not set.")
        return self.X1ctrl.shape[0]

    def _reduce_npts(self, X):
        if self.mode == "POL":
            saa = self.kernel(X[0], X[0])
            sbb = self.kernel(X[1], X[1])
            sab = self.kernel(X[0], X[1])
            sba = self.kernel(X[1], X[0])
            S = saa * sbb + sab * sba
        else:
            S = self.kernel(X, X)
        normlz = np.power(np.diag(S), -0.5)
        Snorm = np.dot(np.diag(normlz), np.dot(S, np.diag(normlz)))
        odS = np.abs(Snorm)
        np.fill_diagonal(odS, 0.0)
        odSs = np.sum(odS, axis=0)
        sortidx = np.argsort(odSs)

        Ssort = Snorm[np.ix_(sortidx, sortidx)].copy()
        c, piv, r_c = pivoted_cholesky(Ssort, tol=self.ctrl_tol)
        if self.ctrl_nmax is not None and self.ctrl_nmax < r_c:
            r_c = self.ctrl_nmax
        idx = sortidx[piv[:r_c]]
        if self.mode == "POL":
            return np.asfortranarray(X[:, idx])
        else:
            return np.asfortranarray(X[idx])

    def X0Tlist_to_X1array(self, X0T_list):
        X1_list = []
        for X0T in X0T_list:
            X1_list.append(self.get_descriptors(X0T, force_polarize=True))
        if self.mode == "POL":
            X1 = np.concatenate(X1_list, axis=1)
        else:
            X1 = np.concatenate(X1_list, axis=0)
        return X1

    def X0Tlist_to_X1array_mul(self, X0T_list, mul_list):
        if self.mode == "POL":
            raise NotImplementedError
        X1_list = []
        for mult, X0T in zip(mul_list, X0T_list):
            X1_list.append(self.get_descriptors_with_mul(X0T, mult))
        X1 = np.concatenate(X1_list, axis=0)
        return X1

    def set_control_points(self, X0T_list, reduce=True):
        X1 = self.X0Tlist_to_X1array(X0T_list)
        if reduce:
            X1 = self._reduce_npts(X1)
        self.X1ctrl = X1

    def get_kctrl(self):
        if self.mode == "POL":
            kaa = self.kernel(self.X1ctrl[0], self.X1ctrl[0])
            kbb = self.kernel(self.X1ctrl[1], self.X1ctrl[1])
            kab = self.kernel(self.X1ctrl[0], self.X1ctrl[1])
            kba = self.kernel(self.X1ctrl[1], self.X1ctrl[0])
            k = kaa * kbb + kab * kba
        else:
            k = self.kernel(self.X1ctrl, self.X1ctrl)
        self.Kmm = k
        return self.Kmm

    def get_k(self, X0T):
        nspin, N0, Nsamp = X0T.shape
        X1 = self.get_descriptors(X0T)
        if self.mode == "POL":
            if nspin == 1:
                X1 = np.concatenate([X1, X1], axis=0)
            elif nspin != 2:
                raise ValueError
            X1 = X1.reshape(2, Nsamp, self.N1)
            kaa = self.kernel(X1[0], self.X1ctrl[0])
            kbb = self.kernel(X1[1], self.X1ctrl[1])
            kab = self.kernel(X1[0], self.X1ctrl[1])
            kba = self.kernel(X1[1], self.X1ctrl[0])
            k = kaa * kbb + kab * kba
        else:
            k = self.kernel(X1, self.X1ctrl)
        if self.mode == "SEP":
            k = k.T.reshape(self.Nctrl, nspin, Nsamp)
        else:
            k = k.T
        return k

    def get_k_and_deriv(self, X0T):
        """
        Return Knm

        Args:
            X0T:

        Returns:

        """
        nspin, N0, Nsamp = X0T.shape
        X1 = self.get_descriptors(X0T)
        # k is (nspin * Nsamp, Nctrl)
        # dkdX1 is (nspin * Nsamp, Nctrl, N1)
        if self.mode == "POL":
            if nspin == 1:
                X1 = np.concatenate([X1, X1], axis=0)
            elif nspin != 2:
                raise ValueError
            X1 = X1.reshape(2, Nsamp, self.N1)
            kaa, dkaa = self.kernel.k_and_deriv(X1[0], self.X1ctrl[0])
            kbb, dkbb = self.kernel.k_and_deriv(X1[1], self.X1ctrl[1])
            kab, dkab = self.kernel.k_and_deriv(X1[0], self.X1ctrl[1])
            kba, dkba = self.kernel.k_and_deriv(X1[1], self.X1ctrl[0])
            k = kaa * kbb + kab * kba
            dkdX1a = dkaa * kbb + dkab * kba
            if nspin == 1:
                dkdX1 = dkdX1a
            else:
                dkdX1b = dkbb * kaa + dkba * kab
                dkdX1 = np.concatenate([dkdX1a, dkdX1b], axis=0)
        else:
            k, dkdX1 = self.kernel.k_and_deriv(X1, self.X1ctrl)
        if self.mode == "SEP":
            # k is (Nctrl, nspin, Nsamp)
            k = k.T.reshape(self.Nctrl, nspin, Nsamp)
        else:
            k = k.T
        # dkdX1 is (Nctrl, nspin * Nsamp, N1)
        dkdX1 = dkdX1.transpose(1, 0, 2)
        dkdX0T = np.empty((self.Nctrl, nspin, N0, Nsamp))
        for i in range(self.Nctrl):
            dkdX0T[i] = self.apply_descriptor_grad(X0T, dkdX1[i])
        return k, dkdX0T

    def map(self, mapping_plan):
        fevals = mapping_plan(self)
        return MappedDFTKernel(
            fevals,
            self.feature_list,
            self.mode,
            self._mul_basefunc,
            self._add_basefunc,
        )


class DFTKernel2(KernelEvalBase2, DFTKernel):
    def __init__(
        self,
        kernel,
        feature_list,
        mode,
        multiplicative_baseline,
        additive_baseline=None,
        ctrl_tol=1e-5,
        ctrl_nmax=None,
        component=None,
    ):
        self.kernel = kernel
        self.feature_list = feature_list
        self.mode = mode
        self._mul_basefunc = multiplicative_baseline
        self._add_basefunc = additive_baseline
        assert isinstance(multiplicative_baseline, str)
        assert isinstance(additive_baseline, str)
        self.ctrl_tol = ctrl_tol
        self.ctrl_nmax = ctrl_nmax
        if component is None:
            component = "x"
        assert component in ["x", "c", "xc"]
        self.component = component

        self.X1ctrl = None
        self.alpha = None
        self.base_dict = {}
        self.cov_dict = {}
        self.dbase_dict = {}
        self.dcov_dict = {}
        self.rxn_cov_list = []

    @property
    def N1(self):
        return self.feature_list.nfeat

    def map(self, mapping_plan):
        fevals = mapping_plan(self)
        return MappedDFTKernel2(
            fevals,
            self.feature_list,
            self.mode,
            self._mul_basefunc,
            self._add_basefunc,
        )
