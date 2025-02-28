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

from abc import ABC, abstractmethod

import numpy as np
import yaml


class FeatureNormalizer(ABC):
    code = None

    @property
    @abstractmethod
    def num_arg(self):
        pass

    @abstractmethod
    def bounds(self):
        pass

    @abstractmethod
    def fill_feat_(self, y, x):
        pass

    @abstractmethod
    def fill_deriv_(self, dfdx, dfdy, x):
        pass

    @classmethod
    def from_dict(cls, d):
        if d["code"] in ALL_CLASS_DICT:
            return ALL_CLASS_DICT[d["code"]].from_dict(d)
        else:
            raise ValueError("Unrecognized code: {}".format(d["code"]))


class LMap(FeatureNormalizer):
    code = "L"

    def __init__(self, i, bounds=None):
        self.i = i
        self._bounds = bounds or (-np.inf, np.inf)

    @property
    def bounds(self):
        return self._bounds

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        y[:] = x[self.i]

    def fill_deriv_(self, dfdx, dfdy, x):
        dfdx[self.i] += dfdy[:]

    def as_dict(self):
        return {
            "code": self.code,
            "i": self.i,
            "bounds": self.bounds,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], bounds=d.get("bounds"))


class UMap(FeatureNormalizer):
    code = "U"

    def __init__(self, i, gamma, bounds=None):
        self.i = i
        self.gamma = gamma
        self._bounds = bounds or (0, 1)

    @property
    def bounds(self):
        return self._bounds

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        y[:] = self.gamma * x[self.i] / (1 + self.gamma * x[self.i])

    def fill_deriv_(self, dfdx, dfdy, x):
        i = self.i
        dfdx[i] += dfdy * self.gamma / (1 + self.gamma * x[i]) ** 2

    def as_dict(self):
        return {
            "code": self.code,
            "i": self.i,
            "gamma": self.gamma,
            "bounds": self.bounds,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["gamma"], bounds=d.get("bounds"))


class TMap(FeatureNormalizer):
    code = "T"

    def __init__(self, i, j, bounds=None):
        self.i = i
        self.j = j
        self._bounds = bounds or (0, 1)

    @property
    def bounds(self):
        return self._bounds

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        i, j = self.i, self.j
        tmp = x[j] + 5.0 / 3 * x[i]
        y[:] = tmp / (1 + tmp)

    def fill_deriv_(self, dfdx, dfdy, x):
        i, j = self.i, self.j
        tmp = x[j] + 5.0 / 3 * x[i]
        dtmp = 1.0 / ((1 + tmp) * (1 + tmp))
        dtmp *= dfdy
        dfdx[i] += 5.0 / 3 * dtmp
        dfdx[j] += dtmp

    def as_dict(self):
        return {
            "code": self.code,
            "i": self.i,
            "j": self.j,
            "bounds": self.bounds,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["j"], bounds=d.get("bounds"))


def get_vmap_heg_value(heg, gamma):
    return (heg * gamma) / (1 + heg * gamma)


class VMap(FeatureNormalizer):
    code = "V"

    def __init__(self, i, gamma, scale=1.0, center=0.0, bounds=None):
        self.i = i
        self.gamma = gamma
        self.scale = scale
        self.center = center
        self._bounds = bounds or (-self.center, self.scale - self.center)

    @property
    def bounds(self):
        return self._bounds

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        i = self.i
        y[:] = -self.center + self.scale * self.gamma * x[i] / (1 + self.gamma * x[i])

    def fill_deriv_(self, dfdx, dfdy, x):
        i = self.i
        dfdx[i] += dfdy * self.scale * self.gamma / (1 + self.gamma * x[i]) ** 2

    def as_dict(self):
        return {
            "code": self.code,
            "i": self.i,
            "gamma": self.gamma,
            "scale": self.scale,
            "center": self.center,
            "bounds": self.bounds,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["gamma"], d["scale"], d["center"], bounds=d.get("bounds"))


class VZMap(FeatureNormalizer):
    code = "VZ"

    def __init__(self, i, gamma, scale=1.0, center=0.0, bounds=None):
        self.i = i
        self.gamma = gamma
        self.scale = scale
        self.center = center
        self._bounds = bounds or (-self.center, self.scale - self.center)

    @property
    def bounds(self):
        return self._bounds

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        xi = x[self.i]
        y[:] = -self.center + self.scale * self.gamma * (xi + xi * xi) / (
            1 + self.gamma * (xi + xi * xi)
        )

    def fill_deriv_(self, dfdx, dfdy, x):
        xi = x[self.i]
        fac = dfdy * self.scale * self.gamma
        fac /= (1 + self.gamma * (xi + xi * xi)) ** 2
        dfdx[self.i] += fac
        dfdx[self.i] += fac * (self.gamma + 1) * xi
        dfdx[self.i] += fac * (self.gamma - 1) * xi * (3 * xi + 2 * xi * xi)

    def as_dict(self):
        return {
            "code": self.code,
            "i": self.i,
            "gamma": self.gamma,
            "scale": self.scale,
            "center": self.center,
            "bounds": self.bounds,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["gamma"], d["scale"], d["center"], bounds=d.get("bounds"))


class V2Map(FeatureNormalizer):
    code = "V2"

    def __init__(self, i, j):
        self.i = i
        self.j = j

    @property
    def bounds(self):
        return (-1.0, 1.0)

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        i, j = self.i, self.j
        a = 2**1.5
        b = 0.5 / (a - 1)
        tmp = b * (a * x[i] - x[j])
        y[:] = tmp / (1 + tmp)
        tmp = 0.5 * x[j]
        y[:] -= tmp / (1 + tmp)

    def fill_deriv_(self, dfdx, dfdy, x):
        i, j = self.i, self.j
        a = 2**1.5
        b = 0.5 / (a - 1)
        tmp = a * x[i] - x[j]
        tmp = b / (1 + b * tmp) ** 2
        dfdx[i] += dfdy * tmp * a
        dfdx[j] -= dfdy * tmp
        tmp = 0.5 / (1 + 0.5 * x[j]) ** 2
        dfdx[j] -= dfdy * tmp

    def as_dict(self):
        return {"code": self.code, "i": self.i, "j": self.j}

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["j"])


class V3Map(FeatureNormalizer):
    code = "V3"

    def __init__(self, i, j, gamma):
        self.i = i
        self.j = j
        self.gamma = gamma

    @property
    def bounds(self):
        return (-1.0, 1.0)

    @property
    def num_arg(self):
        return 2

    def fill_feat_(self, y, x):
        i, j = self.i, self.j
        y[:] = self.gamma * x[i] / (1 + self.gamma * x[i])
        y[:] -= self.gamma * x[j] / (1 + self.gamma * x[j])

    def fill_deriv_(self, dfdx, dfdy, x):
        i, j = self.i, self.j
        dfdx[i] += dfdy * self.gamma / (1 + self.gamma * x[i]) ** 2
        dfdx[j] -= dfdy * self.gamma / (1 + self.gamma * x[j]) ** 2

    def as_dict(self):
        return {
            "code": self.code,
            "i": self.i,
            "j": self.j,
            "gamma": self.gamma,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["j"], d["gamma"])


class V4Map(FeatureNormalizer):
    code = "V4"

    def __init__(self, i, j, gamma):
        self.i = i
        self.j = j
        self.gamma = gamma

    @property
    def bounds(self):
        return (-0.5, 0.5)

    @property
    def num_arg(self):
        return 2

    def fill_feat_(self, y, x):
        i, j = self.i, self.j
        tmp = np.exp(self.gamma * (x[i] - x[j]))
        y[:] = 1 / (1 + tmp) - 0.5

    def fill_deriv_(self, dfdx, dfdy, x):
        i, j = self.i, self.j
        tmp = np.exp(self.gamma * (x[i] - x[j]))
        tmp = dfdy * self.gamma * tmp / (1 + tmp) ** 2
        dfdx[i] -= tmp
        dfdx[j] += tmp

    def as_dict(self):
        return {
            "code": self.code,
            "i": self.i,
            "j": self.j,
            "gamma": self.gamma,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["j"], d["gamma"])


class WMap(FeatureNormalizer):
    code = "W"

    def __init__(self, i, j, k, gammai, gammaj):
        self.i = i
        self.j = j
        self.k = k
        self.gammai = gammai
        self.gammaj = gammaj

    @property
    def bounds(self):
        return (-1, 1)

    @property
    def num_arg(self):
        return 3

    def fill_feat_(self, y, x):
        i, j, k = self.i, self.j, self.k
        gammai, gammaj = self.gammai, self.gammaj
        y[:] = (gammai * np.sqrt(gammaj / (1 + gammaj * x[j])) * x[k]) / (
            1 + gammai * x[i]
        )

    def fill_deriv_(self, dfdx, dfdy, x):
        i, j, k = self.i, self.j, self.k
        gammai, gammaj = self.gammai, self.gammaj
        dfdx[i] -= dfdy * (
            (gammai**2 * np.sqrt(gammaj / (1 + gammaj * x[j])) * x[k])
            / (1 + gammai * x[i]) ** 2
        )
        dfdx[j] -= (
            dfdy
            * (gammai * (gammaj / (1 + gammaj * x[j])) ** 1.5 * x[k])
            / (2.0 * (1 + gammai * x[i]))
        )
        dfdx[k] += (
            dfdy
            * (gammai * np.sqrt(gammaj / (1 + gammaj * x[j])))
            / (1 + gammai * x[i])
        )

    def as_dict(self):
        return {
            "code": self.code,
            "i": self.i,
            "j": self.j,
            "k": self.k,
            "gammai": self.gammai,
            "gammaj": self.gammaj,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["j"], d["k"], d["gammai"], d["gammaj"])


class XMap(FeatureNormalizer):
    code = "X"

    def __init__(self, i, j, k, gammai, gammaj):
        self.i = i
        self.j = j
        self.k = k
        self.gammai = gammai
        self.gammaj = gammaj

    @property
    def bounds(self):
        return (-1, 1)

    @property
    def num_arg(self):
        return 3

    def fill_feat_(self, y, x):
        i, j, k = self.i, self.j, self.k
        gammai, gammaj = self.gammai, self.gammaj
        y[:] = (
            np.sqrt(gammai / (1 + gammai * x[i]))
            * np.sqrt(gammaj / (1 + gammaj * x[j]))
            * x[k]
        )

    def fill_deriv_(self, dfdx, dfdy, x):
        i, j, k = self.i, self.j, self.k
        gammai, gammaj = self.gammai, self.gammaj
        dfdx[i] -= dfdy * (
            (
                gammai
                * np.sqrt(gammai / (1 + gammai * x[i]))
                * np.sqrt(gammaj / (1 + gammaj * x[j]))
                * x[k]
            )
            / (2 + 2 * gammai * x[i])
        )
        dfdx[j] -= dfdy * (
            (
                gammaj
                * np.sqrt(gammai / (1 + gammai * x[i]))
                * np.sqrt(gammaj / (1 + gammaj * x[j]))
                * x[k]
            )
            / (2 + 2 * gammaj * x[j])
        )
        dfdx[k] += (
            dfdy
            * np.sqrt(gammai / (1 + gammai * x[i]))
            * np.sqrt(gammaj / (1 + gammaj * x[j]))
        )

    def as_dict(self):
        return {
            "code": self.code,
            "i": self.i,
            "j": self.j,
            "k": self.k,
            "gammai": self.gammai,
            "gammaj": self.gammaj,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["j"], d["k"], d["gammai"], d["gammaj"])


class YMap(FeatureNormalizer):
    code = "Y"

    def __init__(self, i, j, k, l, gammai, gammaj, gammak):
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        self.gammai = gammai
        self.gammaj = gammaj
        self.gammak = gammak

    @property
    def bounds(self):
        return (-1, 1)

    @property
    def num_arg(self):
        return 4

    def fill_feat_(self, y, x):
        i, j, k, l = self.i, self.j, self.k, self.l
        gammai, gammaj, gammak = self.gammai, self.gammaj, self.gammak
        y[:] = (
            np.sqrt(gammai / (1 + gammai * x[i]))
            * x[l]
            * np.sqrt(gammaj / (1 + gammaj * x[j]))
            * np.sqrt(gammak / (1 + gammak * x[k]))
        )

    def fill_deriv_(self, dfdx, dfdy, x):
        i, j, k, l = self.i, self.j, self.k, self.l
        gammai, gammaj, gammak = self.gammai, self.gammaj, self.gammak
        dfdx[i] -= dfdy * (
            (
                gammai
                * np.sqrt(gammai / (1 + gammai * x[i]))
                * x[l]
                * np.sqrt(gammaj / (1 + gammaj * x[j]))
                * np.sqrt(gammak / (1 + gammak * x[k]))
            )
            / (2 + 2 * gammai * x[i])
        )
        dfdx[j] -= dfdy * (
            (
                gammaj
                * np.sqrt(gammai / (1 + gammai * x[i]))
                * x[l]
                * np.sqrt(gammaj / (1 + gammaj * x[j]))
                * np.sqrt(gammak / (1 + gammak * x[k]))
            )
            / (2 + 2 * gammaj * x[j])
        )
        dfdx[k] -= dfdy * (
            (
                gammak
                * np.sqrt(gammai / (1 + gammai * x[i]))
                * x[l]
                * np.sqrt(gammaj / (1 + gammaj * x[j]))
                * np.sqrt(gammak / (1 + gammak * x[k]))
            )
            / (2 + 2 * gammak * x[k])
        )
        dfdx[l] += (
            dfdy
            * np.sqrt(gammai / (1 + gammai * x[i]))
            * np.sqrt(gammaj / (1 + gammaj * x[j]))
            * np.sqrt(gammak / (1 + gammak * x[k]))
        )

    def as_dict(self):
        return {
            "code": self.code,
            "i": self.i,
            "j": self.j,
            "k": self.k,
            "l": self.l,
            "gammai": self.gammai,
            "gammaj": self.gammaj,
            "gammak": self.gammak,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["i"], d["j"], d["k"], d["l"], d["gammai"], d["gammaj"], d["gammak"]
        )


class ZMap(FeatureNormalizer):
    code = "Z"

    def __init__(self, i, gamma, scale=1.0, center=0.0, bounds=None):
        self.i = i
        self.gamma = gamma
        self.scale = scale
        self.center = center
        self._bounds = bounds or (-self.center, self.scale - self.center)

    @property
    def bounds(self):
        return self._bounds

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        i = self.i
        y[:] = -self.center + self.scale / (1 + self.gamma * x[i] ** 2)

    def fill_deriv_(self, dfdx, dfdy, x):
        i = self.i
        dfdx[i] -= (
            2
            * dfdy
            * self.scale
            * self.gamma
            * x[i]
            / (1 + self.gamma * x[i] ** 2) ** 2
        )

    def as_dict(self):
        return {
            "code": self.code,
            "i": self.i,
            "gamma": self.gamma,
            "scale": self.scale,
            "center": self.center,
            "bounds": self.bounds,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["gamma"], d["scale"], d["center"], bounds=d.get("bounds"))


class EMap(FeatureNormalizer):
    code = "E"

    def __init__(self, i, scale=1.0, center=0.0, bounds=None):
        self.i = i
        self.scale = scale
        self.center = center
        self._bounds = bounds or (0, 1)

    @property
    def bounds(self):
        return self._bounds

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        i = self.i
        y[:] = np.exp(-self.scale * x[i] + self.center)

    def fill_deriv_(self, dfdx, dfdy, x):
        i = self.i
        dfdx[i] -= dfdy * self.scale * np.exp(-self.scale * x[i] + self.center)

    def as_dict(self):
        return {
            "code": self.code,
            "i": self.i,
            "scale": self.scale,
            "center": self.center,
            "bounds": self.bounds,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["scale"], d["center"], bounds=d.get("bounds"))


class SignedUMap(FeatureNormalizer):
    code = "SU"

    def __init__(self, i, gamma):
        self.i = i
        self.gamma = gamma

    @property
    def bounds(self):
        return (-1, 1)

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        i = self.i
        y[:] = x[i] / np.sqrt(self.gamma + x[i] * x[i])

    def fill_deriv_(self, dfdx, dfdy, x):
        i = self.i
        dfdx[i] += dfdy * self.gamma / (self.gamma + x[i] * x[i]) ** 1.5

    def as_dict(self):
        return {"code": self.code, "i": self.i, "gamma": self.gamma}

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["gamma"])


class SLNMap(FeatureNormalizer):
    code = "SLN"

    def __init__(self, i, gamma):
        self.i = i
        self.gamma = gamma

    @property
    def bounds(self):
        return (0, 1)

    @property
    def num_arg(self):
        return 1

    def fill_feat_(self, y, x):
        rho = x[self.i]
        const = 3.0 * self.gamma / (4 * np.pi)
        rss = (1 + const * rho) ** (-1.0 / 3)
        y[:] = rss

    def fill_deriv_(self, dfdx, dfdy, x):
        rho = x[self.i]
        const = 3.0 * self.gamma / (4 * np.pi)
        dfdx[self.i] -= dfdy * const / 3.0 * (1 + const * rho) ** (-4.0 / 3)

    def as_dict(self):
        return {"code": self.code, "i": self.i, "gamma": self.gamma}

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["gamma"])


class SLXMap(FeatureNormalizer):
    """
    gamma * s^2 / (1 + gamma * s^2), where s^2 is the squared reduced gradient
    s^2 = sigma / C n^8/3
    """

    code = "SLX"

    def __init__(self, i, j, gamma):
        self.i = i
        self.j = j
        self.gamma = gamma

    @property
    def bounds(self):
        return (0, 1)

    @property
    def num_arg(self):
        return 2

    def fill_feat_(self, y, x):
        rho = np.maximum(x[self.i], 1e-10)
        rho83 = rho ** (8.0 / 3)
        const = 2 * (3 * np.pi**2) ** (1.0 / 3)
        const = self.gamma / const**2
        sigma = const * x[self.j]
        y[:] = sigma / (rho83 + sigma)

    def fill_deriv_(self, dfdx, dfdy, x):
        rho = np.maximum(x[self.i], 1e-10)
        rho83 = rho ** (8.0 / 3)
        rho53 = (8.0 / 3) * rho ** (5.0 / 3)
        const = 2 * (3 * np.pi**2) ** (1.0 / 3)
        const = self.gamma / const**2
        sigma = const * x[self.j]
        fac = rho83 + sigma
        fac[:] = fac * fac
        fac[:] = dfdy / fac
        dfdx[self.i] -= fac * sigma * rho53
        dfdx[self.j] += fac * rho83 * const

    def as_dict(self):
        return {"code": self.code, "i": self.i, "j": self.j, "gamma": self.gamma}

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["j"], d["gamma"])


class SLBMap(FeatureNormalizer):
    """
    beta = (tau - tauw) / (tau + tau0)
    """

    code = "SLB"

    const = 0.3 * (3 * np.pi**2) ** (2.0 / 3)

    def __init__(self, i, j, k):
        self.i = i
        self.j = j
        self.k = k

    @property
    def bounds(self):
        return (-1, 1)

    @property
    def num_arg(self):
        return 3

    def fill_feat_(self, y, x):
        rho = np.maximum(x[self.i], 1e-10)
        tau0 = self.const * rho ** (5.0 / 3)
        tauw = x[self.j] / (8 * rho)
        tau = x[self.k]
        # y[:] = -1 + 2 * (tau - tauw) / (tau0 + tau - tauw)
        y[:] = -1 + 2 * (tau - tauw) / (tau + tau0)

    def fill_deriv_(self, dfdx, dfdy, x):
        rho = np.maximum(x[self.i], 1e-10)
        tau0 = self.const * rho ** (5.0 / 3)
        tauw = x[self.j] / (8 * rho)
        tau = x[self.k]
        fac = 1.0 / (tau + tau0)
        v0 = 2 * dfdy * (tauw - tau) * fac * fac
        vw = -2 * dfdy * fac
        vt = 2 * dfdy * (tau0 + tauw) * fac * fac
        dfdx[self.i] += v0 * self.const * (5.0 / 3) * rho ** (2.0 / 3)
        dfdx[self.i] -= vw * tauw / rho
        dfdx[self.j] += vw / (8 * rho)
        dfdx[self.k] += vt

    def as_dict(self):
        return {"code": self.code, "i": self.i, "j": self.j, "k": self.k}

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["j"], d["k"])


class SLTMap(FeatureNormalizer):
    """
    beta = (tau - tau0) / (tau + tau0)
    """

    code = "SLT"

    const = 0.3 * (3 * np.pi**2) ** (2.0 / 3)

    def __init__(self, i, j):
        self.i = i
        self.j = j

    @property
    def bounds(self):
        return (-1, 1)

    @property
    def num_arg(self):
        return 2

    def fill_feat_(self, y, x):
        rho = np.maximum(x[self.i], 1e-10)
        tau0 = self.const * rho ** (5.0 / 3)
        tau = x[self.j]
        y[:] = (tau - tau0) / (tau + tau0)

    def fill_deriv_(self, dfdx, dfdy, x):
        rho = np.maximum(x[self.i], 1e-10)
        tau0 = self.const * rho ** (5.0 / 3)
        tau = x[self.j]
        fac = 1.0 / (tau + tau0)
        v0 = -2 * dfdy * tau * fac * fac
        vt = 2 * dfdy * tau0 * fac * fac
        dfdx[self.i] += v0 * self.const * (5.0 / 3) * rho ** (2.0 / 3)
        dfdx[self.j] += vt

    def as_dict(self):
        return {"code": self.code, "i": self.i, "j": self.j}

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["j"])


class SLTWMap(FeatureNormalizer):
    """
    beta = (tauw - tau0) / (tauw + tau0)
    """

    code = "SLTW"

    const = 0.3 * (3 * np.pi**2) ** (2.0 / 3)

    def __init__(self, i, j):
        self.i = i
        self.j = j

    @property
    def bounds(self):
        return (-1, 1)

    @property
    def num_arg(self):
        return 2

    def fill_feat_(self, y, x):
        rho = np.maximum(x[self.i], 1e-10)
        tau0 = self.const * rho ** (5.0 / 3)
        tauw = x[self.j] / (8 * rho)
        y[:] = (tauw - tau0) / (tauw + tau0)

    def fill_deriv_(self, dfdx, dfdy, x):
        rho = np.maximum(x[self.i], 1e-10)
        tau0 = self.const * rho ** (5.0 / 3)
        tauw = x[self.j] / (8 * rho)
        fac = 1.0 / (tauw + tau0)
        v0 = -2 * dfdy * tauw * fac * fac
        vw = 2 * dfdy * tau0 * fac * fac
        dfdx[self.i] += v0 * self.const * (5.0 / 3) * rho ** (2.0 / 3)
        dfdx[self.i] -= vw * tauw / rho
        dfdx[self.j] += vw / (8 * rho)

    def as_dict(self):
        return {"code": self.code, "i": self.i, "j": self.j}

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["j"])


class SLDMap(FeatureNormalizer):
    code = "SLD"
    const = 0.3 * (3 * np.pi**2) ** (2.0 / 3)

    def __init__(self, i, j, k):
        self.i = i
        self.j = j
        self.k = k

    @property
    def bounds(self):
        return (-1, 1)

    @property
    def num_arg(self):
        return 3

    def fill_feat_(self, y, x):
        rho = np.maximum(x[self.i], 1e-10)
        tau0 = self.const * rho ** (5.0 / 3)
        tauw = x[self.j] / (8 * rho)
        tau = x[self.k]
        y[:] = tau / (tau + tau0)
        y[:] -= tauw / (tauw + tau0)

    def fill_deriv_(self, dfdx, dfdy, x):
        rho = np.maximum(x[self.i], 1e-10)
        tau0 = self.const * rho ** (5.0 / 3)
        tauw = x[self.j] / (8 * rho)
        tau = x[self.k]
        fac1 = 1.0 / (tau + tau0)
        fac2 = 1.0 / (tauw + tau0)
        v0 = -dfdy * tau * fac1 * fac1
        v0 += dfdy * tauw * fac2 * fac2
        vw = -dfdy * tau0 * fac2 * fac2
        vt = dfdy * tau0 * fac1 * fac1
        dfdx[self.i] += v0 * self.const * (5.0 / 3) * rho ** (2.0 / 3)
        dfdx[self.i] -= vw * tauw / rho
        dfdx[self.j] += vw / (8 * rho)
        dfdx[self.k] += vt

    def as_dict(self):
        return {"code": self.code, "i": self.i, "j": self.j, "k": self.k}

    @classmethod
    def from_dict(cls, d):
        return cls(d["i"], d["j"], d["k"])


class OmegaMap(FeatureNormalizer):
    def __init__(self, i_n, i_s, i_alpha, c, B, C, bounds=None):
        self.i_n = i_n
        self.i_s = i_s
        self.i_alpha = i_alpha
        self.c = c
        self.B = B
        self.C = C
        self._bounds = bounds or (0, 1)

    def set_current_molecule_id(self, mol_id):
        self.current_molecule_id = mol_id

    @property
    def bounds(self):
        return self._bounds

    @property
    def num_arg(self):
        return 3

    def fill_feat_(self, y, x, mol_id=None):
        if x.size == 0:
            raise ValueError("x is a zero-size array")
        n = x[self.i_n]
        s2 = x[self.i_s]
        alpha = x[self.i_alpha]
        if n.size == 0 or s2.size == 0 or alpha.size == 0:
            raise ValueError("n, s2, or alpha is a zero-size array")
        n_nan_mask = np.isnan(n)
        n_zero_mask = n == 0
        s2_nan_mask = np.isnan(s2)
        alpha_nan_mask = np.isnan(alpha)

        n = np.abs(n)
        n[n_nan_mask | n_zero_mask] = 1e-10

        s2[s2_nan_mask] = 0
        alpha[alpha_nan_mask] = 0

        s2 = np.clip(s2, -1e10, 1e10)
        alpha = np.clip(alpha, -1e10, 1e10)

        inner_term = np.maximum(self.B + self.C * (alpha + 5 / 3 * s2), 1e-10)
        omega = np.sqrt(n ** (2 / 3) * inner_term)
        denominator = np.maximum(1 + self.c * omega, 1e-10)
        y[:] = self.c * omega / denominator

    def fill_deriv_(self, dfdx, dfdy, x):
        n = np.maximum(np.abs(x[self.i_n]), 1e-10)
        s2 = np.clip(x[self.i_s], -1e10, 1e10)
        alpha = np.clip(x[self.i_alpha], -1e10, 1e10)

        inner_term = np.maximum(self.B + self.C * (alpha + 5 / 3 * s2), 1e-10)
        omega = np.sqrt(n ** (2 / 3) * inner_term)
        denom = np.maximum((1 + self.c * omega) ** 2, 1e-10)

        term1 = np.divide(dfdy * self.c, 3 * denom, where=denom != 0)
        term2 = np.power(n, -2 / 3, where=n != 0)
        term3 = np.sqrt(np.abs(inner_term))
        dfdx[self.i_n] += term1 * term2 * term3

        term4 = np.divide(
            dfdy * self.c, 2 * denom * omega, where=(denom != 0) & (omega != 0)
        )
        term5 = np.power(n, 2 / 3, where=n != 0)
        dfdx[self.i_s] += term4 * term5 * self.C * 5 / 3
        dfdx[self.i_alpha] += term4 * term5 * self.C

    def as_dict(self):
        return {
            "code": "Omega",
            "i_n": self.i_n,
            "i_s": self.i_s,
            "i_alpha": self.i_alpha,
            "c": self.c,
            "B": self.B,
            "C": self.C,
            "bounds": self.bounds,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["i_n"],
            d["i_s"],
            d["i_alpha"],
            d["c"],
            d["B"],
            d["C"],
            bounds=d.get("bounds"),
        )


ALL_CLASS_DICT = {}
ALL_CLASSES = [
    LMap,
    UMap,
    TMap,
    VMap,
    VZMap,
    V2Map,
    V3Map,
    V4Map,
    WMap,
    XMap,
    YMap,
    ZMap,
    EMap,
    SignedUMap,
    SLNMap,
    SLXMap,
    SLBMap,
    SLTMap,
    SLTWMap,
    SLDMap,
    OmegaMap,
]
for cls in ALL_CLASSES:
    assert cls.code not in ALL_CLASS_DICT
    ALL_CLASS_DICT[cls.code] = cls


class FeatureList:
    def __init__(self, feat_list):
        self.feat_list = feat_list

    @property
    def nfeat(self):
        return len(self.feat_list)

    def __getitem__(self, index):
        return self.feat_list[index]

    def __call__(self, xdesc):
        # xdesc (nsamp, ninp)
        xdesc = xdesc.T
        # now xdesc (ninp, nsamp)
        tdesc = np.zeros((self.nfeat, xdesc.shape[1]))
        for i in range(self.nfeat):
            self.feat_list[i].fill_feat_(tdesc[i], xdesc)
        return tdesc.T

    @property
    def bounds_list(self):
        return [f.bounds for f in self.feat_list]

    def fill_vals_(self, tdesc, xdesc):
        for i in range(self.nfeat):
            self.feat_list[i].fill_feat_(tdesc[i], xdesc)

    def fill_derivs_(self, dfdx, dfdy, xdesc):
        """
        Fill dfdx wth the derivatives of f with respect
        to the raw features x, given the derivative
        with respect to transformed features dfdy.

        Args:
            dfdx (nraw, nsamp): Output. Derivative of output wrt
                raw features x
            dfdy (ntrans, nsamp): Input. Derivative of output wrt transformed
                features y
            xdesc (nraw, nsamp): Raw features x
        """
        for i in range(self.nfeat):
            self.feat_list[i].fill_deriv_(dfdx, dfdy[i], xdesc)

    def as_dict(self):
        d = {"feat_list": [f.as_dict() for f in self.feat_list]}
        return d

    def dump(self, fname):
        d = self.as_dict()
        with open(fname, "w") as f:
            yaml.dump(d, f)

    @classmethod
    def from_dict(cls, d):
        return cls(
            [
                FeatureNormalizer.from_dict(d["feat_list"][i])
                for i in range(len(d["feat_list"]))
            ]
        )

    @classmethod
    def load(cls, fname):
        with open(fname, "r") as f:
            d = yaml.load(f, Loader=yaml.Loader)
        return cls.from_dict(d)


"""
Examples of FeatureList objects:
center0a = get_vmap_heg_value(2.0, gamma0a)
center0b = get_vmap_heg_value(8.0, gamma0b)
center0c = get_vmap_heg_value(0.5, gamma0c)
lst = [
        UMap(1, gammax),
        VMap(2, 1, scale=2.0, center=1.0),
        VMap(3, gamma0a, scale=1.0, center=center0a),
        UMap(4, gamma1),
        UMap(5, gamma2),
        WMap(1, 5, 6, gammax, gamma2),
        XMap(1, 4, 7, gammax, gamma1),
        VMap(8, gamma0b, scale=1.0, center=center0b),
        VMap(9, gamma0c, scale=1.0, center=center0c),
        YMap(1, 4, 5, 10, gammax, gamma1, gamma2),
]
flst = FeatureList(lst)

center0a = get_vmap_heg_value(2.0, gamma0a)
center0b = get_vmap_heg_value(3.0, gamma0b)
UMap(1, gammax)
VMap(2, 1, scale=2.0, center=1.0)
VMap(3, gamma0a, scale=1.0, center=center0a)
UMap(4, gamma1)
UMap(5, gamma2)
WMap(1, 5, 6, gammax, gamma2)
XMap(1, 4, 7, gammax, gamma1)
VMap(8, gamma0b, scale=1.0, center=center0b)
YMap(1, 4, 5, 9, gammax, gamma1, gamma2)
"""
