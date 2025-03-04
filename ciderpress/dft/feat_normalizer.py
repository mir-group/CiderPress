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

"""
This class provides tools for normalizing features to have more
desirable behavior under uniform scaling.
"""
from abc import ABC, abstractmethod

import numpy as np

CFC = (3.0 / 10) * (3 * np.pi**2) ** (2.0 / 3)


class FeatNormalizer(ABC):
    def _setup_fwd(self, x, xn):
        if xn is None:
            xn = np.empty_like(x)
        return xn

    def _setup_bwd(self, x, dfdx, dfdrho, dfdinh):
        if dfdx is None:
            dfdx = np.empty_like(x)
        if dfdrho is None:
            dfdrho = np.zeros_like(x)
        if dfdinh is None:
            dfdinh = np.zeros_like(x)
        return dfdx, dfdrho, dfdinh

    @abstractmethod
    def get_usp(self):
        pass

    @abstractmethod
    def fill_fwd(self, x, rho, inh, xn=None):
        """

        Args:
            x:
            rho:
            inh:
            xn: Can be empty

        Returns:

        """

    @abstractmethod
    def fill_bwd(self, dfdxn, x, rho, inh, dfdx=None, dfdrho=None, dfdinh=None):
        """

        Args:
            dfdxn:
            x:
            rho:
            inh:
            dfdx: Can be empty
            dfdrho: Must be initialized because it is added to
            dfdinh: Must be initialized because it is added to

        Returns:

        """

    @abstractmethod
    def get_normed_feature_deriv(self, x, rho, inh, dx, drho, dinh):
        pass

    @abstractmethod
    def get_ueg(self, rho=1.0):
        pass


class ConstantNormalizer(FeatNormalizer):
    def __init__(self, const):
        self.const = const

    def get_usp(self):
        return 0

    def fill_fwd(self, x, rho, inh, xn=None):
        xn = self._setup_fwd(x, xn)
        xn[:] = x * self.const
        return xn

    def fill_bwd(self, dfdxn, x, rho, inh, dfdx=None, dfdrho=None, dfdinh=None):
        dfdx, dfdrho, dfdinh = self._setup_bwd(x, dfdx, dfdrho, dfdinh)
        dfdx[:] = dfdxn * self.const
        return dfdx, dfdrho, dfdinh

    def get_normed_feature_deriv(self, x, rho, inh, dx, drho, dinh):
        return dx * self.const

    def get_ueg(self, rho=1.0):
        return self.const


class DensityNormalizer(FeatNormalizer):
    def __init__(self, const, power):
        self.const = const
        self.power = power

    def get_usp(self):
        return 3 * self.power

    def fill_fwd(self, x, rho, inh, xn=None):
        xn = self._setup_fwd(x, xn)
        xn[:] = x * self.const * rho**self.power
        return xn

    def fill_bwd(self, dfdxn, x, rho, inh, dfdx=None, dfdrho=None, dfdinh=None):
        dfdx, dfdrho, dfdinh = self._setup_bwd(x, dfdx, dfdrho, dfdinh)
        fac = self.const * rho**self.power
        dfdx[:] = dfdxn * fac
        dfdrho[:] += dfdxn * self.power * fac * x / rho
        return dfdx, dfdrho, dfdinh

    def get_normed_feature_deriv(self, x, rho, inh, dx, drho, dinh):
        fac = self.const * rho**self.power
        return fac * (dx + self.power * x * drho / rho)

    def get_ueg(self, rho=1.0):
        return self.const * rho**self.power


class InhomogeneityNormalizer(FeatNormalizer):
    def __init__(self, const1, const2, power):
        self.const1 = const1
        self.const2 = const2
        self.power = power

    def get_usp(self):
        return 0.0

    def fill_fwd(self, x, rho, inh, xn=None):
        xn = self._setup_fwd(x, xn)
        inh = 1 + self.const2 * inh
        xn[:] = x * self.const1 * inh**self.power
        return xn

    def fill_bwd(self, dfdxn, x, rho, inh, dfdx=None, dfdrho=None, dfdinh=None):
        dfdx, dfdrho, dfdinh = self._setup_bwd(x, dfdx, dfdrho, dfdinh)
        inh = 1 + self.const2 * inh
        fac = self.const1 * inh**self.power
        dfdx[:] = dfdxn * fac
        dfdinh[:] += (
            dfdxn * x * self.const1 * self.const2 * self.power * inh ** (self.power - 1)
        )
        return dfdx, dfdrho, dfdinh

    def get_normed_feature_deriv(self, x, rho, inh, dx, drho, dinh):
        inh = 1 + self.const2 * inh
        fac = self.const1 * inh**self.power
        return fac * dx + x * self.const1 * self.const2 * dinh * self.power * inh ** (
            self.power - 1
        )

    def get_ueg(self, rho=1.0):
        return self.const1


class GeneralNormalizer(FeatNormalizer):
    def __init__(self, const1, const2, power1, power2):
        self.const1 = const1
        self.const2 = const2
        self.power1 = power1
        self.power2 = power2

    def get_usp(self):
        return 3 * self.power1

    def fill_fwd(self, x, rho, inh, xn=None):
        xn = self._setup_fwd(x, xn)
        fac = self.const1 * rho**self.power1
        inh = 1 + self.const2 * inh
        xn[:] = x * fac * inh**self.power2
        return xn

    def fill_bwd(self, dfdxn, x, rho, inh, dfdx=None, dfdrho=None, dfdinh=None):
        dfdx, dfdrho, dfdinh = self._setup_bwd(x, dfdx, dfdrho, dfdinh)
        fac1 = self.const1 * rho**self.power1
        inh = 1 + self.const2 * inh
        fac2 = inh**self.power2
        dfdx[:] = dfdxn * fac1 * fac2
        dfdrho[:] += dfdxn * x * fac2 * fac1 * self.power1 / rho
        dfdinh[:] += (
            dfdxn * x * fac1 * self.power2 * self.const2 * inh ** (self.power2 - 1)
        )
        return dfdx, dfdrho, dfdinh

    def get_normed_feature_deriv(self, x, rho, inh, dx, drho, dinh):
        fac1 = self.const1 * rho**self.power1
        inh = 1 + self.const2 * inh
        fac2 = inh**self.power2
        res = dx * fac1 * fac2
        res += x * fac2 * fac1 * self.power1 / rho * drho
        res += x * fac1 * self.power2 * self.const2 * inh ** (self.power2 - 1) * dinh
        return res

    def get_ueg(self, rho=1.0):
        return self.const1 * rho**self.power1


def get_invariant_normalizer_from_exponent_params(power, a0, tau_mul):
    tau_fac = tau_mul * 1.2 * (6 * np.pi**2) ** (2.0 / 3) / np.pi
    B = np.pi / 2 ** (2.0 / 3) * (a0 - tau_fac)
    C = np.pi / 2 ** (2.0 / 3) * tau_fac  # / CFC
    const2 = C / B
    const1 = B**power
    return InhomogeneityNormalizer(const1, const2, power)


def get_normalizer_from_exponent_params(rho_pow, exp_pow, a0, tau_mul, gga=False):
    power1 = rho_pow + 2.0 / 3 * exp_pow
    power2 = exp_pow
    tau_fac = tau_mul * 1.2 * (6 * np.pi**2) ** (2.0 / 3) / np.pi
    if gga:
        B = np.pi / 2 ** (2.0 / 3) * a0
    else:
        B = np.pi / 2 ** (2.0 / 3) * (a0 - tau_fac)
    C = np.pi / 2 ** (2.0 / 3) * tau_fac  # / CFC
    const2 = C / B
    const1 = B**exp_pow
    return GeneralNormalizer(const1, const2, power1, power2)


class FeatNormalizerList:
    def __init__(self, normalizers, slmode, cutoff=1e-10):  # , i0=3, i1=None):
        """
        Args:
            normalizers (list[FeatNormalizer or None]):
            list of feature normalizers
        """
        self.slmode = slmode
        self._normalizers = normalizers
        self.cutoff = cutoff
        # self.i0 = i0
        # self.i1 = self.nfeat + i0 if i1 is None else i1
        # if self.i1 - self.i0 != self.nfeat:
        #    raise ValueError("Index range must correspond to number"
        #                     "of normalizers provided.")

    @property
    def nfeat(self):
        return len(self._normalizers)

    def get_usps(self):
        usps = []
        for n in self._normalizers:
            if n is None:
                usps.append(0)
            else:
                usps.append(n.get_usp())
        return usps

    def __getitem__(self, item):
        return self._normalizers[item]

    def __setitem__(self, key, value):
        raise RuntimeError("Cannot set element of normalizer list")

    def _check_shape(self, x, ndim=3):
        assert ndim == 3 or ndim == 2, "Internal error"
        if x.ndim != ndim:
            raise ValueError("Expected {}-dimensional array".format(ndim))
        if x.shape[-2] != self.nfeat:
            raise ValueError("Array must have size nfeat")

    def _get_rho_and_inh(self, X0T):
        rho_term = np.maximum(X0T[:, 0], self.cutoff)
        grad_term = X0T[:, 1]
        if self.slmode == "npa":
            tau_term = X0T[:, 2]
            inh_term = tau_term + 5.0 / 3 * grad_term
        elif self.slmode == "nst":
            tau_term = X0T[:, 2]
            inh_term = tau_term / (CFC * rho_term ** (5.0 / 3))
        elif self.slmode == "np":
            inh_term = 5.0 / 3 * grad_term
        else:
            inh_term = grad_term / (8 * CFC * rho_term ** (8.0 / 3))
        return rho_term, inh_term

    def _get_drho_and_dinh(self, X0T, DX0T):
        rho = np.maximum(X0T[0], self.cutoff)
        drho = DX0T[0]
        grad = X0T[1]
        dgrad = DX0T[1]
        if self.slmode == "npa":
            tau = X0T[2]
            dtau = DX0T[2]
            dinh = dtau + 5.0 / 3 * dgrad
        elif self.slmode == "nst":
            tau = X0T[2]
            dtau = DX0T[2]
            dinh = dtau / (CFC * rho ** (5.0 / 3))
            dinh -= 5.0 / 3 * tau / (CFC * rho ** (8.0 / 3)) * drho
        elif self.slmode == "np":
            dinh = 5.0 / 3 * dgrad
        else:
            dinh = dgrad / (8 * CFC * rho ** (8.0 / 3))
            dinh -= 8.0 / 3 * grad / (8 * CFC * rho ** (11.0 / 3)) * drho
        return drho, dinh

    def get_normalized_feature_vector(self, X0T):
        # This function should be in the settings class, rather than
        # a plan, because it is called from models, which do not
        # require knowledge of the plan.
        self._check_shape(X0T)
        X0TN = np.empty_like(X0T)
        rho_term, inh_term = self._get_rho_and_inh(X0T)
        nfeat = self.nfeat
        for i in range(nfeat):
            if self[i] is not None:
                self[i].fill_fwd(X0T[:, i], rho_term, inh_term, xn=X0TN[:, i])
            else:
                X0TN[:, i] = X0T[:, i]
        return X0TN

    def get_derivative_of_normed_features(self, X0T, DX0T):
        self._check_shape(X0T, ndim=2)
        self._check_shape(DX0T, ndim=2)
        DX0TN = np.empty_like(DX0T)
        rho, inh = self._get_rho_and_inh(X0T[None, :])
        rho, inh = rho[0], inh[0]
        drho, dinh = self._get_drho_and_dinh(X0T, DX0T)
        for i in range(self.nfeat):
            if self[i] is not None:
                DX0TN[i] = self[i].get_normed_feature_deriv(
                    X0T[i], rho, inh, DX0T[i], drho, dinh
                )
            else:
                DX0TN[i] = DX0T[i]
        return DX0TN

    def get_derivative_wrt_unnormed_features(self, X0T, df_dX0TN):
        self._check_shape(X0T)
        self._check_shape(df_dX0TN)
        df_dX0T = np.empty_like(X0T)
        rho_term, inh_term = self._get_rho_and_inh(X0T)
        dfdrho = np.zeros_like(rho_term)
        dfdinh = np.zeros_like(inh_term)
        cond = rho_term < self.cutoff
        rho_term[cond] = self.cutoff
        nfeat = self.nfeat
        for i in range(nfeat):
            if self[i] is not None:
                self[i].fill_bwd(
                    df_dX0TN[:, i],
                    X0T[:, i],
                    rho_term,
                    inh_term,
                    dfdx=df_dX0T[:, i],
                    dfdrho=dfdrho,
                    dfdinh=dfdinh,
                )
            else:
                df_dX0T[:, i] = df_dX0TN[:, i]
        df_dX0T[:, 0] += dfdrho
        if self.slmode == "npa":
            df_dX0T[:, 1] += 5.0 / 3 * dfdinh
            df_dX0T[:, 2] += dfdinh
        elif self.slmode == "nst":
            df_dX0T[:, 0] -= dfdinh * 5.0 / 3 * inh_term / rho_term
            df_dX0T[:, 2] += dfdinh / (CFC * rho_term ** (5.0 / 3))
        elif self.slmode == "np":
            df_dX0T[:, 1] += 5.0 / 3 * dfdinh
        else:
            df_dX0T[:, 0] -= dfdinh * 8.0 / 3 * inh_term / rho_term
            df_dX0T[:, 1] += dfdinh / (8 * CFC * rho_term ** (8.0 / 3))
        for s in range(cond.shape[0]):
            df_dX0T[s, ..., cond[s]] = 0.0
        return df_dX0T

    def ueg_vector(self, rho=1.0):
        norms = []
        for n in self._normalizers:
            if n is None:
                norms.append(1.0)
            else:
                norms.append(n.get_ueg(rho))
        return np.array(norms)
