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

from ciderpress.dft.settings import get_cider_exponent


def regularize_cider_exponent_(a, dadrho, dadsigma, dadtau, amin):
    cond = a < amin
    a[cond] = amin * np.exp(a[cond] / amin - 1)
    grad_damp = a[cond] * np.exp(a[cond] / amin - 1)
    dadrho[cond] *= grad_damp
    dadsigma[cond] *= grad_damp
    dadtau[cond] *= grad_damp


class FeatNormalizer(ABC):
    @abstractmethod
    def evaluate(self, rho, sigma, tau):
        pass

    @abstractmethod
    def evaluate_with_grad(self, rho, sigma, tau):
        pass

    def apply_norm_fwd_(self, feat, rho, sigma, tau):
        feat[:] *= self.evaluate(rho, sigma, tau)

    def apply_norm_bwd_(self, vfeat, feat, rho, sigma, tau, vrho, vsigma, vtau):
        c, d1, d2, d3 = self.evaluate_with_grad(rho, sigma, tau)
        vfeat[:] *= c
        vrho[:] += d1 * vfeat * feat
        vsigma[:] += d2 * vfeat * feat
        vtau[:] += d3 * vfeat * feat

    @abstractmethod
    def get_usp(self):
        pass


class ConstantNormalizer(FeatNormalizer):
    def __init__(self, const):
        self.const = const

    def evaluate(self, rho, sigma, tau):
        res = np.ones_like(rho)
        res[:] *= self.const
        return res

    def evaluate_with_grad(self, rho, sigma, tau):
        res = self.evaluate(rho, sigma, tau)
        dres = np.zeros_like(rho)
        return res, dres, dres, dres

    def get_usp(self):
        return 0


class DensityPowerNormalizer(FeatNormalizer):
    def __init__(self, const, power, cutoff=1e-10):
        self.const = const
        self.power = power
        self.cutoff = cutoff

    def evaluate(self, rho, sigma, tau):
        rho = np.maximum(self.cutoff, rho)
        return self.const * rho**self.power

    def evaluate_with_grad(self, rho, sigma, tau):
        cond = rho < self.cutoff
        rho = np.maximum(self.cutoff, rho)
        res = self.const * rho**self.power
        dres = self.power * res / rho
        dres[cond] = 0
        zeros = np.zeros_like(res)
        return res, dres, zeros, zeros

    def get_usp(self):
        return 3 * self.power


class CiderExponentNormalizer(FeatNormalizer):
    def __init__(self, const1, const2, const3, power, cutoff=1e-10):
        self.const1 = const1
        self.const2 = const2
        self.const3 = const3
        self.power = power
        self.cutoff = cutoff

    def evaluate(self, rho, sigma, tau):
        rho = np.maximum(self.cutoff, rho)
        return (
            get_cider_exponent(
                rho,
                sigma,
                tau,
                a0=self.const1,
                grad_mul=self.const2,
                tau_mul=self.const3,
                rhocut=self.cutoff,
                nspin=1,
            )[0]
            ** self.power
        )

    def evaluate_with_grad(self, rho, sigma, tau):
        cond = rho < self.cutoff
        res0, d1, d2, d3 = get_cider_exponent(
            rho,
            sigma,
            tau,
            a0=self.const1,
            grad_mul=self.const2,
            tau_mul=self.const3,
            rhocut=self.cutoff,
            nspin=1,
        )
        res = res0**self.power
        tmp = self.power * res0 ** (self.power - 1)
        d1 *= tmp
        d2 *= tmp
        d3 *= tmp
        d1[cond] = 0
        d2[cond] = 0
        d3[cond] = 0
        return res, d1, d2, d3

    def get_usp(self):
        return 2 * self.power


class GeneralNormalizer(FeatNormalizer):
    def __init__(self, const1, const2, const3, power1, power2, cutoff=1e-10):
        self.norm1 = CiderExponentNormalizer(const1, const2, const3, power1, cutoff)
        self.norm2 = DensityPowerNormalizer(1.0, power2, cutoff)

    def evaluate(self, rho, sigma, tau):
        return self.norm1.evaluate(rho, sigma, tau) * self.norm2.evaluate(
            rho, sigma, tau
        )

    def evaluate_with_grad(self, rho, sigma, tau):
        n1, d1a, d1b, d1c = self.norm1.evaluate_with_grad(rho, sigma, tau)
        n2, d2a, d2b, d2c = self.norm2.evaluate_with_grad(rho, sigma, tau)
        return (
            n1 * n2,
            n1 * d2a + n2 * d1a,
            n1 * d2b + n2 * d1b,
            n1 * d2c + n2 * d1c,
        )

    def get_usp(self):
        return self.norm1.get_usp() + self.norm2.get_usp()


class FeatNormalizerList:
    def __init__(self, normalizers):
        """
        Args:
            normalizers (list[FeatNormalizer]): list of feature normalizers
        """
        self._normalizers = normalizers

    @property
    def nfeat(self):
        return len(self._normalizers)

    def get_usps(self):
        return [n.get_usp() for n in self._normalizers]

    def __getitem__(self, item):
        return self._normalizers[item]

    def __setitem__(self, key, value):
        raise RuntimeError("Cannot set element of normalizer list")
