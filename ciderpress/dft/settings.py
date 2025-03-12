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
from scipy.special import gamma as gamma_func

from ciderpress.dft.feat_normalizer import (
    ConstantNormalizer,
    DensityNormalizer,
    FeatNormalizerList,
    get_normalizer_from_exponent_params,
)

LDA_FACTOR = -3.0 / 4.0 * (3.0 / np.pi) ** (1.0 / 3)
CFC = (3.0 / 10) * (3 * np.pi**2) ** (2.0 / 3)
ALPHA_TOL = 1e-10


def _check_l1_dots(l1_dots, nl1):
    for dot in l1_dots:
        if not isinstance(dot, (tuple, list)) or len(dot) != 2:
            raise ValueError("l1_dots items must be 2-tuples")
        for d in dot:
            if not (-1 <= d < nl1):
                raise ValueError("Illegal index in feat_dots")


def _get_fl_ueg(s):
    tol = 1e-10
    # first, handle the numerically unstable cases
    if s < 0:
        sign = -1
    else:
        sign = 1
    if np.abs(s + 0.5) < tol:
        return 1.5
    if np.abs(s) < tol:
        return 1.0
    elif np.abs(s - 0.5) < tol:
        return 0.75
    elif np.abs(s - 1.0) < tol:
        return 0.6
    # fmt: off
    return (
        sign * 4 * np.pi * 3 * np.cos(np.pi * s)
        * gamma_func(-1 - 2 * s) / (3 + 2 * s) * 4**s
        * gamma_func(1.5 + s)
        / (np.pi**1.5 * np.abs(gamma_func(-s)))
    )
    # fmt: on


def get_cider_exponent(
    rho, sigma, tau, a0=1.0, grad_mul=0.0, tau_mul=0.03125, rhocut=1e-10, nspin=1
):
    """
    Evaluate an NLDF length-scale exponent at the MGGA level.
    """
    if nspin > 2:
        raise ValueError
    if isinstance(rho, np.ndarray):
        isarray = True
    else:
        isarray = False
        rho = np.asarray([rho], dtype=np.float64)
        sigma = np.asarray([sigma], dtype=np.float64)
        tau = np.asarray([tau], dtype=np.float64)
    tau_fac = tau_mul * 1.2 * (6 * np.pi**2) ** (2.0 / 3) / np.pi
    cond = rho < rhocut
    rho = rho.copy()
    rho[cond] = rhocut
    sigma[cond] = 0
    tau[cond] = 0
    if nspin == 1:
        B = np.pi / 2 ** (2.0 / 3) * (a0 - tau_fac)
    else:
        B = np.pi * (a0 - tau_fac)
    C = np.pi / 2 ** (2.0 / 3) * tau_fac / CFC
    ascale = B * rho ** (2.0 / 3) + C * tau / rho
    dadrho = 2 * B / (3 * rho ** (1.0 / 3)) - (C * tau / rho) / rho
    dadtau = C / rho
    if grad_mul != 0:
        grad_fac = grad_mul * 1.2 * (6 * np.pi**2) ** (2.0 / 3) / np.pi
        grad_fac *= np.pi / 2 ** (2.0 / 3) * 0.125 / CFC
        ascale += grad_fac * sigma / (rho * rho)
        dadsigma = grad_fac / (rho * rho)
        dadrho -= 2 * grad_fac * sigma / (rho * rho * rho)
    else:
        dadsigma = np.zeros_like(ascale)
    dadrho[cond] = 0
    dadsigma[cond] = 0
    dadtau[cond] = 0
    if not isarray:
        ascale = ascale.item()
        dadrho = dadrho.item()
        dadsigma = dadsigma.item()
        dadtau = dadtau.item()
    return ascale, dadrho, dadsigma, dadtau


def get_cider_exponent_gga(rho, sigma, a0=1.0, grad_mul=0.03125, rhocut=1e-10, nspin=1):
    """
    Evaluate an NLDF length-scale exponent at the GGA level.
    """
    if nspin > 2:
        raise ValueError
    if isinstance(rho, np.ndarray):
        isarray = True
    else:
        isarray = False
        rho = np.asarray([rho], dtype=np.float64)
        sigma = np.asarray([sigma], dtype=np.float64)
    cond = rho < rhocut
    rho = rho.copy()
    rho[cond] = rhocut
    sigma[cond] = 0
    if nspin == 1:
        B = np.pi / 2 ** (2.0 / 3) * a0
    else:
        B = np.pi * a0
    ascale = B * rho ** (2.0 / 3)
    dadrho = 2 * B / (3 * rho ** (1.0 / 3))
    if grad_mul != 0:
        grad_fac = grad_mul * 1.2 * (6 * np.pi**2) ** (2.0 / 3) / np.pi
        grad_fac *= np.pi / 2 ** (2.0 / 3) * 0.125 / CFC
        ascale += grad_fac * sigma / (rho * rho)
        dadsigma = grad_fac / (rho * rho)
        dadrho -= 2 * grad_fac * sigma / (rho * rho * rho)
    else:
        dadsigma = np.zeros_like(ascale)
    dadrho[cond] = 0
    dadsigma[cond] = 0
    if not isarray:
        ascale = ascale.item()
        dadrho = dadrho.item()
        dadsigma = dadsigma.item()
    return ascale, dadrho, dadsigma


def _get_ueg_expnt(aval, tval, rho):
    return get_cider_exponent(
        rho,
        0,
        CFC * rho ** (5.0 / 3),
        a0=aval,
        grad_mul=0,
        tau_mul=tval,
        rhocut=0,
        nspin=1,
    )[0]


class BaseSettings(ABC):
    """
    This is a base class for storing the settings for different
    types of density/density matrix feature in CiderPress.
    Settings objects indicate which features must be evaluated,
    and with which hyperparameters, to use as input to an ML functional.
    """

    @property
    @abstractmethod
    def nfeat(self):
        """
        Returns:
            Number of features in this feature set.
        """

    @property
    def is_empty(self):
        """
        Return true of this settings object specifies zero features.
        """
        return self.nfeat == 0

    @abstractmethod
    def get_feat_usps(self):
        """
        Returns:
            A list of floats with the uniform scaling power for each feature.
        """

    @abstractmethod
    def ueg_vector(self, rho=1.0):
        """
        Return the feature vector for the uniform electron gas
        as a function of the density rho

        Args:
            rho (float): Value of the density

        Returns:
            np.ndarray(float): UEG feature vector
        """

    def get_reasonable_normalizer(self):
        """
        Returns:
            norms (list[FeatNormalizer]): list of normalizers for these features
        """


class EmptySettings(BaseSettings):
    """
    The EmptySettings class is a representation of a feature set containing
    zero features. It is used when a certain type of feature is not
    present in a model. (For example, if a model does not use SDMX
    features, that model's FeatureSettings.sdmx_settings will be
    an EmptySettings instance.)
    """

    @property
    def nfeat(self):
        return 0

    def get_feat_usps(self):
        return []

    def ueg_vector(self, rho=1.0):
        return np.array([])

    def get_reasonable_normalizer(self):
        return []


class FracLaplSettings(BaseSettings):
    """
    Settings for Fractional Laplacian feature set.
    """

    def __init__(self, slist, nk0, nk1, l1_dots, nd1=0, ld_dots=None, ndd=0):
        """
        Initialize FracLaplSettings object. Note, in the documentation below
        on the indexing of the features, fl_rho is the part of the density
        ingredient vector AFTER the semilocal part.

        Note on the l=1 feature indexing. If l1_dots[i] = (j, k), then::

            fl_feat[i+nk0] = einsum(
                'xg,xg->g',
                fl_rho[3*j+nk0 : 3*j+3+nk0],
                fl_rho[3*k+nk0 : 3*k+3+nk0],
            )

        If ld_dots[i] = (j, k), then::

            nstart = nk0 + 3 * nk1
            fl_feat[i + nk0 + len(l1_dots)] = einsum(
                'xg,xg->g',
                fl_rho[3*j+nstart : 3*j+3+nstart],
                fl_rho[3*k+nstart : 3*k+3+nstart],
            )
            nstart = nk0 + 3 * nk1 + 3 * nd1
            fl_feat[i + nk0 + len(l1_dots) + len(ld_dots)] = fl_rho[i + nstart]

        Args:
            slist (list of float): s parameters for the fractional
                Laplacian. For each s, :math:`(-\\Delta)^s \\phi_i` is computed
                for each single-particle orbital.
            nk0 (int): Number of scalar (l=0) features. Must be <= len(slist).
                fl_feat[i] = fl_rho[i] = :math:`(-\\Delta')^s \\rho(r, r') |_{r'=r}`,
                with s=slist[i].
            nk1 (int): Number of vector (l=1) features F_s^1. Must be <= len(slist).
                fl_rho[3*i+nk0 : 3*i+3+nk0] =
                :math:`\\nabla'' (-\\Delta')^s \\rho(r'', r') |_{r'=r,r''=r}`,
                with s=slist[i].
            l1_dots (list of (int, int)): List of index tuples for contracting
                the l=1 F_s^1 features. -1 indicates to use the semilocal
                density gradient.
            nd1 (int): Number of vector features F_s^d. Must be ``<= len(slist)``
                With ``nstart = nk0 + 3 * nk1``,
                ``fl_rho[i * 3 + nstart : (i+1) * 3 + nstart] =``
                :math:`\\nabla' (-\\Delta')^s \\rho(r, r') |_{r'=r}`,
                with s=slist[i].
            ld_dots (list of (int, int)): Same as l1_dots but for the F_s^d
                features. -1 indicates to use the semilocal density gradient.
            ndd (int): Numer of dot product features :math:`F_s^{dd}`. Must be <= nd1
        """
        self.slist = slist
        assert nk0 <= self.npow
        assert nk1 <= self.npow
        assert nd1 <= self.npow
        assert ndd <= nd1
        self.l1_dots = l1_dots
        self._ndd = ndd
        self._nd1 = nd1
        if ld_dots is None:
            ld_dots = []
        self._ld_dots = ld_dots
        self._nk0 = nk0
        self._nk1 = nk1
        ir = 0
        self.a0_inds = []
        self.a1_inds = []
        for ik in range(self.nk0):
            self.a0_inds.append(ir)
            ir += 1
        for ik in range(self.nk1):
            self.a1_inds.append(ir)
            ir += 3
        ir += 3 * self.nd1 + self.ndd
        self._size = ir
        _check_l1_dots(l1_dots, self.nk1)

    @property
    def npow(self):
        return len(self.slist)

    @property
    def size(self):
        return self._size

    @property
    def nd1(self):
        return self._nd1

    @property
    def ndd(self):
        return self._ndd

    def get_ik0(self, ind):
        return ind + 3 * min(self.nd1, ind)

    @property
    def nk0(self):
        return self._nk0

    @property
    def nk1(self):
        return self._nk1

    @property
    def nrho(self):
        return self.nk0 + 3 * self.nk1 + 3 * self.nd1 + self.ndd

    @property
    def ld_dots(self):
        return self._ld_dots

    @property
    def nfeat(self):
        return self.nk0 + len(self.l1_dots) + len(self.ld_dots) + self.ndd

    @property
    def nglob(self):
        return 0

    @staticmethod
    def get_usp(s):
        return 3 + 2 * s

    def get_rho_usps(self):
        usp_set = [self.get_usp(self.slist[n]) for n in range(len(self.slist))]
        usps = []
        for n in range(self.nk0):
            usps.append(usp_set[n])
        for n in range(self.nk1):
            for i in range(3):
                usps.append(usp_set[n] + 1)
        usps.append(SPEC_USPS["grad_rho"])  # for the density gradient
        return usps

    def get_feat_usps(self):
        usp_set = [self.get_usp(self.slist[n]) for n in range(len(self.slist))]
        # -1 index for grad, relies on input checking in __init__
        usp_set.append(3)
        usps = []
        for n in range(self.nk0):
            usps.append(usp_set[n])
        for j, k in self.l1_dots:
            if j < 0:
                assert j == -1
                j = len(usp_set) + j
            if k < 0:
                assert k == -1
                k = len(usp_set) + k
            usps.append(usp_set[j] + usp_set[k] + 2)
        for j, k in self.ld_dots:
            if j < 0:
                assert j == -1
                j = len(usp_set) + j
            if k < 0:
                assert k == -1
                k = len(usp_set) + k
            usps.append(usp_set[j] + usp_set[k] + 2)
        for n in range(self.ndd):
            usps.append(usp_set[n] + 2)
        return usps

    def ueg_vector(self, rho=1.0):
        kf = (rho * 3 * np.pi**2) ** (1.0 / 3)
        vec = [
            kf ** (3 + 2 * s) / (np.pi**2 * (3 + 2 * s))
            for s in self.slist[: self.nk0]
        ]
        # TODO ndd components actually aren't zero for UEG
        return np.array(
            vec + [0] * (len(self.l1_dots) + len(self.ld_dots) + self.ndd),
            dtype=np.float64,
        )

    def get_reasonable_normalizer(self):
        usps = self.get_feat_usps()
        norms = []
        for i in range(self.nk0):
            rho_pow = -0.5
            exp_pow = 0.5 * (1.5 - usps[i])
            norms.append(
                get_normalizer_from_exponent_params(rho_pow, exp_pow, 1.0, 0.03125)
            )
        for i, (j, k) in enumerate(self.l1_dots):
            rho_pow = -1.0
            if j == -1:
                rho_pow -= 0.5
            if k == -1:
                rho_pow -= 0.5
            exp_pow = -0.5 * (usps[i + self.nk0] + 3 * rho_pow)
            norms.append(
                get_normalizer_from_exponent_params(rho_pow, exp_pow, 1.0, 0.03125)
            )
        start = self.nk0 + len(self.l1_dots)
        for i, (j, k) in enumerate(self.ld_dots):
            rho_pow = -1.0
            if j == -1:
                rho_pow -= 0.5
            if k == -1:
                rho_pow -= 0.5
            exp_pow = -0.5 * (usps[i + start] + 3 * rho_pow)
            norms.append(
                get_normalizer_from_exponent_params(rho_pow, exp_pow, 1.0, 0.03125)
            )
        for i in range(self.ndd):
            rho_pow = -0.5
            exp_pow = 0.5 * (1.5 - usps[i - self.ndd])
            norms.append(
                get_normalizer_from_exponent_params(rho_pow, exp_pow, 1.0, 0.03125)
            )
        return norms


class SDMXBaseSettings(BaseSettings):
    """
    Base object for the SDMX-like settings.
    """

    @property
    def ndterms(self):
        return 0

    @property
    def n1terms(self):
        return 0

    @property
    def integral_type(self):
        if hasattr(self, "_integral_type"):
            return self._integral_type
        else:
            return "gauss_diff"


class SADMSettings(SDMXBaseSettings):
    """
    Settings for Spherically Averaged Density Matrix descriptor, i.e.
    the self-repulsion of the exchange hole constructed only from
    the spherically averaged density matrix at a point.

    WARNING: DEPRECATED, especially 'exact' which is not numerically
    accurate. Use SDMXSettings and its variants instead.
    """

    def __init__(self, mode):
        if mode not in ["exact", "smooth"]:
            raise ValueError("Mode must be exact or smooth, got {}".format(mode))
        self.mode = mode

    @property
    def ueg_const(self):
        if self.mode == "smooth":
            return -0.8751999726949563
        else:
            return LDA_FACTOR

    @property
    def nfeat(self):
        return 1

    @staticmethod
    def get_rho_usps():
        return [4]

    def get_feat_usps(self):
        return self.get_rho_usps()

    def ueg_vector(self, rho=1.0):
        return [self.ueg_const * rho ** (4.0 / 3)]

    @property
    def nglob(self):
        return 0

    def get_reasonable_normalizer(self):
        return [DensityNormalizer(1.0 / self.ueg_const, power=(-4.0 / 3))]


class SDMXSettings(SADMSettings):
    def __init__(self, pows):
        """
        Initialize SDMX settings.

        Args:
            pows (list of int): for each number n in pows,
                int dR R^{2-n} rho_smooth(R) is computed. Technically,
                n can be any float, but it should be 0, 1, or 2 for
                normalizability and ability to compute UEG values.
                If not 0, 1, or 2, UEG limit cannot be computed,
                and features might be poorly defined/numerically inaccurate.
        """
        super(SDMXSettings, self).__init__("smooth")
        self.pows = pows

    @property
    def ueg_const(self):
        consts = []
        for pow in self.pows:
            if pow == 0:
                consts.append(-0.5831374308696952)
            elif pow == 1:
                consts.append(-0.875199972960095)
            elif pow == 2:
                consts.append(-2.073307129927323)
            else:
                raise NotImplementedError("Only have UEG for pow=0,1,2")
        return consts

    @property
    def nfeat(self):
        return len(self.pows)

    def get_feat_usps(self):
        return [3 + n for n in self.pows]

    def ueg_vector(self, rho=1.0):
        return [u * rho ** (1 + n / 3.0) for u, n in zip(self.ueg_const, self.pows)]

    @property
    def nglob(self):
        return 0

    def get_reasonable_normalizer(self):
        return [
            DensityNormalizer(1.0 / u, power=(-1 - n / 3.0))
            for u, n in zip(self.ueg_const, self.pows)
        ]


class SDMXGSettings(SDMXSettings):
    def __init__(self, pows, ndt):
        """
        Initialize SDMXGSettings

        Args:
            pows (list of int): list of 0, 1, 2, see SDMXSettings docstring.
            ndt (int): Number of gradient features H_n^d. Will compute features
                for the first ndt values of n in pows.
        """
        super(SDMXGSettings, self).__init__(pows)
        assert ndt <= len(pows)
        self._ndt = ndt

    @property
    def ndterms(self):
        return self._ndt

    @property
    def ueg_const(self):
        consts = super(SDMXGSettings, self).ueg_const
        for pow in self.pows[: self.ndterms]:
            if pow == 0:
                consts.append(-2.145747704025017)
            elif pow == 1:
                consts.append(-1.7176832947050569)
            elif pow == 2:
                consts.append(-1.526151911492354)
            else:
                raise NotImplementedError("Only have UEG for pow=0,1,2")
        return consts

    @property
    def nfeat(self):
        return len(self.pows) + self.ndterms

    def get_feat_usps(self):
        usps = super(SDMXGSettings, self).get_feat_usps()
        return usps + usps[: self.ndterms]

    def ueg_vector(self, rho=1.0):
        return [
            u * rho ** (1 + n / 3.0)
            for u, n in zip(self.ueg_const, self.pows + self.pows[: self.ndterms])
        ]

    @property
    def nglob(self):
        return 0

    def get_reasonable_normalizer(self):
        return [
            DensityNormalizer(1.0 / u, power=(-1 - n / 3.0))
            for u, n in zip(self.ueg_const, self.pows + self.pows[: self.ndterms])
        ]


class SDMX1Settings(SDMXSettings):
    def __init__(self, pows, n1):
        """
        Initialize SDMX1Settings

        Args:
            pows (list of int): list of 0, 1, 2, see SDMXSettings docstring.
            n1 (int): Number of gradient features H_n^1. Will compute features
                for the first n1 values of n in pows.
        """
        super(SDMX1Settings, self).__init__(pows)
        self.pows = pows
        self._n1 = n1
        assert self._n1 <= len(self.pows)

    @property
    def nfeat(self):
        return len(self.pows) + self._n1

    @property
    def n1terms(self):
        return self._n1

    def get_feat_usps(self):
        res = [3 + n for n in self.pows]
        return res + res[: self._n1]

    def ueg_vector(self, rho=1.0):
        return [u * rho ** (1 + n / 3.0) for u, n in zip(self.ueg_const, self.pows)] + [
            0
        ] * self._n1

    @property
    def nglob(self):
        return 0

    def get_reasonable_normalizer(self):
        ueg_const = self.ueg_const + self.ueg_const[: self._n1]
        pows = self.pows + self.pows[: self._n1]
        return [
            DensityNormalizer(1.0 / u, power=(-1 - n / 3.0))
            for u, n in zip(ueg_const, pows)
        ]


class SDMXG1Settings(SDMXGSettings):
    def __init__(self, pows, nd, n1):
        """
        Initialize SDMXG1Settings

        Args:
            pows (list of int): list of 0, 1, 2, see SDMXSettings docstring.
            nd (int): Number of gradient features H_n^d. Will compute features
                for the first nd values of n in pows.
            n1 (int): Number of gradient features H_n^1. Will compute features
                for the first n1 values of n in pows.
        """
        super(SDMXG1Settings, self).__init__(pows, nd)
        self._n1 = n1

    @property
    def n1terms(self):
        return self._n1

    @property
    def nfeat(self):
        return super(SDMXG1Settings, self).nfeat + self.n1terms

    def get_feat_usps(self):
        res = super(SDMXG1Settings, self).get_feat_usps()
        return res + res[: self._n1]

    def ueg_vector(self, rho=1.0):
        return super(SDMXG1Settings, self).ueg_vector(rho) + [0] * self._n1

    def get_reasonable_normalizer(self):
        norms = super(SDMXG1Settings, self).get_reasonable_normalizer()
        return norms + [
            DensityNormalizer(1.0 / u, power=(-1 - n / 3.0))
            for u, n in zip(self.ueg_const[: self._n1], self.pows[: self._n1])
        ]


class SDMXFullSettings(SDMXBaseSettings):
    mode = "smooth"

    def __init__(self, settings_dict=None):
        """
        Args:
            settings_dict (dict): Each key is float. Each value
                is a 2-tuple. The first item is a list of pows.
                The second item is a 4-list/tuple with the number
                of pows for 0, d, 1, and 1d features.
        """
        self._settings = settings_dict
        try:
            self._check_settings()
        except AssertionError:
            raise ValueError("Invalid settings")

    def _check_settings(self):
        for k, v in self._settings.items():
            assert k >= 1.0
            assert isinstance(k, (int, float))
            assert isinstance(v, (tuple, list))
            assert len(v) == 2
            assert isinstance(v[0], (tuple, list))
            assert isinstance(v[1], (tuple, list))
            assert len(v[1]) == 4
            npow = len(v[0])
            for num in v[1]:
                assert num <= npow

    def _get_num_feat(self, i):
        n = 0
        for k, v in self._settings.items():
            n += v[1][i]
        return n

    @property
    def n0terms(self):
        return self._get_num_feat(0)

    @property
    def ndterms(self):
        return self._get_num_feat(1)

    @property
    def n1terms(self):
        return self._get_num_feat(2)

    @property
    def n1dterms(self):
        return self._get_num_feat(3)

    @property
    def ratios(self):
        return sorted(list(self._settings.keys()))

    def iterate_l0_terms(self, r):
        v0, v1 = self._settings[r]
        for i in range(v1[0]):
            yield v0[i], False
        for i in range(v1[1]):
            yield v0[i], True

    def iterate_l1_terms(self, r):
        v0, v1 = self._settings[r]
        for i in range(v1[2]):
            yield v0[i], False
        for i in range(v1[3]):
            yield v0[i], True

    @property
    def nfeat(self):
        return self.n0terms + self.ndterms + self.n1terms + self.n1dterms

    def _get_ueg_const(self):
        known_ueg_vals = [
            0.5831374308696956,
            0.8751999729599736,
            2.073307129927323,
            2.1457477040250175,
            1.7176832947040594,
            1.526151911492354,
            0.5198776758462206,
            0.8101255945354239,
            1.9940403510297529,
            1.6566509762225055,
            1.3731808970090098,
            1.2642399626720755,
            0.4226571198848177,
            0.7049490157407584,
            1.859127630633614,
            1.0360245380146442,
            0.9154428039042883,
            0.8992798146562154,
        ]
        known_uegs = [
            (1.0, 0, False),
            (1.0, 1, False),
            (1.0, 2, False),
            (1.0, 0, True),
            (1.0, 1, True),
            (1.0, 2, True),
            (1.5, 0, False),
            (1.5, 1, False),
            (1.5, 2, False),
            (1.5, 0, True),
            (1.5, 1, True),
            (1.5, 2, True),
            (2.0, 0, False),
            (2.0, 1, False),
            (2.0, 2, False),
            (2.0, 0, True),
            (2.0, 1, True),
            (2.0, 2, True),
        ]
        known_dict = {k: -1 * v for k, v in zip(known_uegs, known_ueg_vals)}
        return known_dict

    def ueg_vector(self, rho=1.0):
        # (ratio, n, rdr)
        known_dict = self._get_ueg_const()
        for k, v in known_dict.items():
            k2 = (1.0, k[1], k[2])
            known_dict[k] = 0.5 * (known_dict[k] + known_dict[k2])
        uegvec = []
        usps = self.get_feat_usps()
        i = 0
        for ratio in self.ratios:
            for n, rdr in self.iterate_l0_terms(ratio):
                try:
                    uegvec.append(known_dict[ratio, n, rdr] * rho ** (usps[i] / 3.0))
                    i += 1
                except KeyError:
                    raise NotImplementedError("Only support ratio=1,1.5,2, pow=0,1,2")
        for ratio in self.ratios:
            for _ in self.iterate_l1_terms(ratio):
                uegvec.append(0)
        return uegvec

    def get_feat_usps(self):
        usps = []
        for ratio in self.ratios:
            for n, rdr in self.iterate_l0_terms(ratio):
                usps.append(3 + n)
        for ratio in self.ratios:
            for n, rdr in self.iterate_l1_terms(ratio):
                usps.append(3 + n)
        return usps

    def get_reasonable_normalizer(self):
        known_dict = self._get_ueg_const()
        norms = []
        for ratio in self.ratios:
            for n, rdr in self.iterate_l0_terms(ratio):
                try:
                    u = known_dict[ratio, n, rdr]
                except KeyError:
                    raise NotImplementedError("Only support ratio=1,1.5,2, pow=0,1,2")
                norms.append(DensityNormalizer(1.0 / u, power=(-1 - n / 3.0)))
            for n, rdr in self.iterate_l1_terms(ratio):
                try:
                    u = known_dict[ratio, n, rdr]
                except KeyError:
                    raise NotImplementedError("Only support ratio=1,1.5,2, pow=0,1,2")
                norms.append(DensityNormalizer(1.0 / u, power=(-1 - n / 3.0)))
        return norms


class HybridSettings(BaseSettings):
    """
    TODO very rough draft of hybrid settings, to be fully implemented later.
    """

    def __init__(self, alpha, beta, local):
        self.alpha = alpha
        self.beta = beta
        self.local = local

    @property
    def size(self):
        if not self.local:
            return 0
        elif self.beta:  # beta is screened component
            return 2
        else:
            return 1

    @property
    def nglob(self):
        if self.local:
            return 0
        elif self.beta:
            return 2
        else:
            return 1

    @property
    def nfeat(self):
        # TODO
        return 0

    @property
    def get_feat_usps(self):
        raise NotImplementedError

    @property
    def ueg_vector(self, rho=1.0):
        raise NotImplementedError

    def get_reasonable_normalizer(self):
        raise NotImplementedError


ALLOWED_I_SPECS_L0 = ["se", "se_r2", "se_apr2", "se_ap", "se_ap2r2", "se_lapl"]
"""
Allowed spec strings for version i l=0 features.

se: squared-exponential

se_r2: squared-exponential times r^2

se_apr2: squared-exponential times the exponent of the
r' coordinate times r^2

se_ap: squared-exponential times the exponent of the
r' coordinate.

se_ap2r2: squared-exponential times exponent^2 times r^2

se_lapl: laplacian of squared-exponential
"""

ALLOWED_I_SPECS_L1 = ["se_grad", "se_rvec"]
"""
Allowed spec strings for version i l=1 features

se_grad: gradient of squared-exponential

se_rvec: squared-exponential times vector (r'-r)
"""

ALLOWED_J_SPECS = ["se", "se_ar2", "se_a2r4", "se_erf_rinv"]
"""
Allowed spec strings for version j features.
Version k features have the same allowed spec strings

se: squared-exponential

se_ar2: squared-exponential * a * r^2

se_a2r4: squared-exponential * a^2 * r^4

se_erf_rinv: squared-exponential * 1/r with short-range
erf damping
"""
ALLOWED_K_SPECS = ALLOWED_J_SPECS
"""
See ``ALLOWED_J_SPECS``
"""

ALLOWED_RHO_MULTS = ["one", "expnt"]
"""
These strings specify the allowed options for what value
to multiply the density by before integrating it to construct
NLDF features. The options are:

one: Identity, i.e. multiply density by 1

expnt: Multiply the density by the NLDF exponent specified
by the theta_params. (NOTE: Experimental, not thoroughly tested.)
"""

ALLOWED_RHO_DAMPS = ["exponential"]
"""
These strings specify the allowed options for how to "damp"
the density for the version k features. Currently the only allowed
option is "exponential", which results in the integral
:math:`\\int g[n](|r-r'|) n(r') exp(-3 a_0[n](r') / 2 a_i[n](r))`,
where :math:`a_0` is the exponent given by ``theta_params``
and :math:`a_i` is an exponet given by ``feat_params``.
"""

SPEC_USPS = {
    "se": 0,
    "se_r2": -2,
    "se_ar2": 0,
    "se_a2r4": 0,
    "se_erf_rinv": 0,
    "se_ap": 2,
    "se_apr2": 0,
    "se_ap2r2": 2,
    "se_lapl": 2,
    "se_grad": 1,
    "se_rvec": -1,
    "grad_rho": 4,
}
"""
Uniform-scaling powers (USPs) descibe how features scale as the
density is scaled by n_lambda(r) = lambda^3 n(lambda r). If the
USP of a functional F is u, then
F[n_lambda](r) = lambda^u F[n](lambda r)
"""
RHO_MULT_USPS = {
    "one": 0,
    "expnt": 2,
}


class NLDFSettings(BaseSettings):
    """
    NLDFSettings contains the settings for the nonlocal density features,
    which form the core of the CIDER framework. This is an abstract class
    since the feature settings depend on the version. Subclasses are available
    for versions i, j, ij, and k features.
    """

    version = None

    def __init__(
        self,
        sl_level,
        theta_params,
        rho_mult,
    ):
        """
        Initialize NLDFSettings.

        Args:
            sl_level (str): "GGA" or "MGGA", the level of semilocal ingredients
                used to construct the length-scale exponent.
            theta_params (np.ndarray(2 or 3)):
                Settings for the squared-exponential kernel exponent for the r'
                (integrated) coordinate of the features. For version 'k', this
                'exponent' is not used in the squared-exponential but rather
                within the density damping scheme (see rho_damp).
                Should be an array of 3 floats [a0, grad_mul, tau_mul].
                tau_mul is ignored if sl_level="GGA" and may therefore be excluded.
            rho_mult (str): Multiply the density that gets integrated
                by a prefactor. Options: See ALLOWED_RHO_MULTS.
        """
        self._sl_level = sl_level
        self.theta_params = theta_params
        self.rho_mult = rho_mult
        self.l0_feat_specs = []
        self.l1_feat_specs = []
        self.l1_feat_dots = []
        self.feat_params = []
        self._check_params(self.theta_params)
        if self.sl_level not in ["GGA", "MGGA"]:
            raise ValueError("sl_level must be GGA or MGGA")
        if self.rho_mult not in ALLOWED_RHO_MULTS:
            raise ValueError("Unsupported rho_mult")

    @property
    def sl_level(self):
        # backward compatibility
        if hasattr(self, "_sl_level"):
            return self._sl_level
        else:
            return "MGGA"

    def _check_params(self, params, spec="se"):
        try:
            assert isinstance(params, list)
            for num in params:
                assert isinstance(num, (float, int))
            assert params[0] > 0
            assert params[1] >= 0
            if self.sl_level == "MGGA":
                assert params[2] >= 0
            n = 3 if self.sl_level == "MGGA" else 2
            if spec == "se_erf_rinv":
                assert len(params) == n + 1
            else:
                assert len(params) == n
        except AssertionError:
            raise ValueError("Unsupported feature params")

    @staticmethod
    def _check_specs(specs, allowed):
        for spec in specs:
            if spec not in allowed:
                raise ValueError("Unsupported feature specs")

    @property
    @abstractmethod
    def num_vi_feats(self):
        """Return the number of version i-type features."""

    @property
    @abstractmethod
    def num_feat_param_sets(self):
        """Returns: Number of unique feature parameter sets"""

    @abstractmethod
    def get_feat_usps(self):
        """Returns a list containing the usp of each contracted feature."""

    @property
    @abstractmethod
    def feat_spec_list(self):
        """Return a list containing feature specs in appropriate order."""

    @property
    def nfeat(self):
        """Number of contracted, scale-invariant features."""
        return self.num_feat_param_sets + self.num_vi_feats

    @property
    def nldf_type(self):
        """Same as version, represents type of NLDFs being computed."""
        return self.version

    def _ueg_rho_mult(self, rho):
        if self.rho_mult == "one":
            rho_mult = 1
        elif self.rho_mult == "expnt":
            rho_mult = _get_ueg_expnt(self.theta_params[0], self.theta_params[2], rho)
        else:
            raise NotImplementedError
        return rho_mult

    @abstractmethod
    def get_reasonable_normalizer(self):
        pass


class NLDFSettingsVI(NLDFSettings):

    version = "i"

    def __init__(
        self,
        sl_level,
        theta_params,
        rho_mult,
        l0_feat_specs,
        l1_feat_specs,
        l1_feat_dots,
    ):
        """
        Initialize NLDFSettingsVI object

        Args:
            sl_level (str): "GGA" or "MGGA", the level of semilocal ingredients
                used to construct the length-scale exponent.
            theta_params (np.ndarray):
                Settings for the squared-exponential kernel exponent for the r'
                (integrated) coordinate of the features. For version 'k', this
                'exponent' is not used in the squared-exponential but rather
                within the density damping scheme (see rho_damp).
            rho_mult (str): Multiply the density that gets integrated
                by a prefactor. Options: See ALLOWED_RHO_MULTS.
            l0_feat_specs (list of str): Each item in the list is a str
                specifying the formula to be used for the scalar (l=0)
                features. See ALLOWED_I_SPECS_L0 for allowed values.
            l1_feat_specs (list of str): Each item in the list is a str
                specifying the formula for the vector (l=1) features.
                See ALLOWED_I_SPECS_L1 for allowed values.
            l1_feat_dots (list of (int, int)): The vector features
                must be contracted with each other (via dot product) to
                form scalar features for the ML model. Each item i in the
                list is a 2-tuple with indexes j,k for features to contract.
                -1 refers to the semilocal density gradient.
                l1feat_ig = einsum('xg,xg->g', l1ints_jg, l1ints_kg)
        """
        super(NLDFSettingsVI, self).__init__(sl_level, theta_params, rho_mult)
        self.l0_feat_specs = l0_feat_specs
        self.l1_feat_specs = l1_feat_specs
        self.l1_feat_dots = l1_feat_dots
        self._check_specs(self.l0_feat_specs, ALLOWED_I_SPECS_L0)
        self._check_specs(self.l1_feat_specs, ALLOWED_I_SPECS_L1)
        _check_l1_dots(self.l1_feat_dots, len(self.l1_feat_specs))

    @property
    def num_feat_param_sets(self):
        return 0

    @property
    def num_vi_feats(self):
        return len(self.l0_feat_specs) + len(self.l1_feat_dots)

    @property
    def feat_spec_list(self):
        return self.l0_feat_specs + self.l1_feat_specs

    def get_feat_usps(self):
        usp0 = RHO_MULT_USPS[self.rho_mult]
        usps = []
        for spec in self.l0_feat_specs:
            usps.append(usp0 + SPEC_USPS[spec])
        for (
            j,
            k,
        ) in self.l1_feat_dots:
            spec1 = "grad_rho" if j == -1 else self.l1_feat_specs[j]
            spec2 = "grad_rho" if k == -1 else self.l1_feat_specs[k]
            usps.append(usp0 + SPEC_USPS[spec1] + SPEC_USPS[spec2])
        return usps

    def ueg_vector(self, rho=1.0):
        l0ueg = []
        l1ueg = [0] * len(self.l1_feat_dots)
        a0 = self.theta_params[0]
        if self.sl_level == "MGGA":
            t0 = self.theta_params[2]
        else:
            t0 = self.theta_params[1]
        rho_mult = self._ueg_rho_mult(rho)
        for spec in self.l0_feat_specs:
            # 'se', 'se_r2', 'se_apr2', 'se_ap', 'se_ap2r2', 'se_lapl'
            expnt = _get_ueg_expnt(a0, t0, rho)
            integral = (np.pi / expnt) ** 1.5
            if spec == "se":
                integral *= 1
            elif spec == "se_r2":
                integral *= 1.5 / expnt
            elif spec == "se_apr2":
                integral *= 1.5
            elif spec == "se_ap":
                integral *= expnt
            elif spec == "se_ap2r2":
                integral *= 1.5 * expnt
            elif spec == "se_lapl":
                integral *= 4 * expnt
            else:
                raise ValueError
            l0ueg.append(rho * rho_mult * integral)
        return np.append(l0ueg, l1ueg).astype(np.float64)

    def get_reasonable_normalizer(self):
        nvi = self.num_vi_feats
        usps = self.get_feat_usps()[self.nfeat - nvi :]
        uegs = self.ueg_vector()[self.nfeat - nvi :]
        a0 = self.theta_params[0]
        if self.sl_level == "MGGA":
            tau_mul = self.theta_params[2]
        else:
            tau_mul = self.theta_params[1]
        norms = []
        for i in range(nvi):
            usp = usps[i]
            ueg = uegs[i]
            if usp == 0 and ueg != 0:
                norms.append(ConstantNormalizer(2.0 / ueg))
            elif usp == 0:
                norms.append(None)
            elif usp == -2:
                norms.append(get_normalizer_from_exponent_params(0.0, 1.0, a0, tau_mul))
            elif usp == 2:
                norms.append(
                    get_normalizer_from_exponent_params(0.0, -1.0, a0, tau_mul)
                )
            elif usp == 5:
                norms.append(
                    get_normalizer_from_exponent_params(-1.0, -1.0, a0, tau_mul)
                )
            else:
                raise NotImplementedError
        return norms


class NLDFSettingsVJ(NLDFSettings):

    version = "j"

    def __init__(
        self,
        sl_level,
        theta_params,
        rho_mult,
        feat_specs,
        feat_params,
    ):
        """
        Initialize NLDFSettingsVJ

        Args:
            sl_level (str): "GGA" or "MGGA", the level of semilocal ingredients
                used to construct the length-scale exponent.
            theta_params (np.ndarray):
                Settings for the squared-exponential kernel exponent for the r'
                (integrated) coordinate of the features. For version 'k', this
                'exponent' is not used in the squared-exponential but rather
                within the density damping scheme (see rho_damp).
            rho_mult (str): Multiply the density that gets integrated
                by a prefactor. Options: See ALLOWED_RHO_MULTS.
            feat_specs (list of str):
                Each item in the list is a string specifying the formula
                to be used for a feature (see ALLOWED_J_SPECS for options).
                feat_specs[i] uses the parameterization of feat_params[i].
                feat_specs and feat_params must be the same length.
            feat_params (list of np.ndarray):
                Each item in the list is an array with the parameters for the
                feature corresponding to the feat_specs above. Typically, each array
                has three numbers [a0, grad_mul, tau_mul], except erf_rinv which
                has an additional parameter erf_mul for the ratio of the
                erf / rinv exponent to the squared-exponential exponent.
                tau_mul is ignored if sl_level="GGA" and may therefore be excluded.
        """
        super(NLDFSettingsVJ, self).__init__(sl_level, theta_params, rho_mult)
        self.feat_params = feat_params
        self.feat_specs = feat_specs
        self._check_specs(self.feat_specs, ALLOWED_J_SPECS)
        if len(self.feat_params) != len(self.feat_specs):
            raise ValueError("specs and params must have same length")
        for s, p in zip(self.feat_specs, self.feat_params):
            self._check_params(p, spec=s)

    @property
    def num_vi_feats(self):
        return 0

    @property
    def feat_spec_list(self):
        return self.feat_specs

    @property
    def num_feat_param_sets(self):
        return len(self.feat_params)

    def get_feat_usps(self):
        usp0 = RHO_MULT_USPS[self.rho_mult]
        usps = []
        for spec in self.feat_specs:
            usps.append(usp0 + SPEC_USPS[spec])
        return usps

    def ueg_vector(self, rho=1.0):
        ueg_feats = []
        a0t = self.theta_params[0]
        if self.sl_level == "MGGA":
            t0t = self.theta_params[2]
        else:
            t0t = self.theta_params[1]
        rho_mult = self._ueg_rho_mult(rho)
        for spec, params in zip(self.feat_specs, self.feat_params):
            a0i = params[0]
            if self.sl_level == "MGGA":
                t0i = params[2]
            else:
                t0i = params[1]
            a0 = a0i + a0t
            t0 = t0i + t0t
            expnt = _get_ueg_expnt(a0, t0, rho)
            expnt2 = _get_ueg_expnt(a0i, t0i, rho)
            integral = (np.pi / expnt) ** 1.5
            if spec == "se":
                integral *= 1
            elif spec == "se_ar2":
                integral *= 1.5 * expnt2 / expnt
            elif spec == "se_a2r4":
                integral *= 3.75 * (expnt2 / expnt) ** 2
            elif spec == "se_erf_rinv":
                expnt3 = expnt2 * params[-1]
                integral *= np.sqrt(expnt / (expnt + expnt3))
            else:
                raise ValueError
            ueg_feats.append(rho * rho_mult * integral)
        return np.asarray(ueg_feats, dtype=np.float64)

    def get_reasonable_normalizer(self):
        nvj = self.num_feat_param_sets
        uegs = self.ueg_vector()[:nvj]
        usps = self.get_feat_usps()[:nvj]
        a0 = self.theta_params[0]
        if self.sl_level == "MGGA":
            tau_mul = self.theta_params[2]
        else:
            tau_mul = self.theta_params[1]
        norms = []
        for i in range(nvj):
            if usps[i] == 0:
                norms.append(ConstantNormalizer(2.0 / uegs[i]))
            else:
                norms.append(
                    get_normalizer_from_exponent_params(
                        0.0, -0.5 * usps[i], a0, tau_mul
                    )
                )
        return norms


class NLDFSettingsVIJ(NLDFSettings):
    """
    Note:
        When storing features in array, j should go first, followed by i.
    """

    version = "ij"

    def __init__(
        self,
        sl_level,
        theta_params,
        rho_mult,
        l0_feat_specs_i,
        l1_feat_specs_i,
        l1_feat_dots_i,
        feat_specs_j,
        feat_params_j,
    ):
        """
        Initialize NLDFSettingsVIJ object

        Args:
            sl_level (str): "GGA" or "MGGA", the level of semilocal ingredients
                used to construct the length-scale exponent.
            theta_params (np.ndarray):
                Settings for the squared-exponential kernel exponent for the r'
                (integrated) coordinate of the features. For version 'k', this
                'exponent' is not used in the squared-exponential but rather
                within the density damping scheme (see rho_damp).
            rho_mult (str): Multiply the density that gets integrated
                by a prefactor. Options: See ALLOWED_RHO_MULTS.
            l0_feat_specs_i (list of str): Each item in the list is a str
                specifying the formula to be used for the scalar (l=0)
                features. See ALLOWED_I_SPECS_L0 for allowed values.
            l1_feat_specs_i (list of str): Each item in the list is a str
                specifying the formula for the vector (l=1) features.
                See ALLOWED_I_SPECS_L1 for allowed values.
            l1_feat_dots_i (list of (int, int)): The vector features
                must be contracted with each other (via dot product) to
                form scalar features for the ML model. Each item i in the
                list is a 2-tuple with indexes j,k for features to contract.
                -1 refers to the semilocal density gradient.
                l1feat_ig = einsum('xg,xg->g', l1ints_jg, l1ints_kg)
            feat_specs_j (list of str):
                Each item in the list is a string specifying the formula
                to be used for a feature (see ALLOWED_J_SPECS for options).
                feat_specs[i] uses the parameterization of feat_params[i].
                feat_specs and feat_params must be the same length.
            feat_params_j (list of np.ndarray):
                Each item in the list is an array with the parameters for the
                feature corresponding to the feat_specs above. Typically, each array
                has three numbers [a0, grad_mul, tau_mul], except erf_rinv which
                has an additional parameter erf_mul for the ratio of the
                erf / rinv exponent to the squared-exponential exponent.
                tau_mul is ignored if sl_level="GGA" and may therefore be excluded.
        """
        super(NLDFSettingsVIJ, self).__init__(sl_level, theta_params, rho_mult)
        self.l0_feat_specs = l0_feat_specs_i
        self.l1_feat_specs = l1_feat_specs_i
        self.l1_feat_dots = l1_feat_dots_i
        self.feat_specs = feat_specs_j
        self.feat_params = feat_params_j
        self._check_specs(self.l0_feat_specs, ALLOWED_I_SPECS_L0)
        self._check_specs(self.l1_feat_specs, ALLOWED_I_SPECS_L1)
        self._check_specs(self.feat_specs, ALLOWED_J_SPECS)
        for dot in self.l1_feat_dots:
            if not isinstance(dot, (tuple, list)) or len(dot) != 2:
                raise ValueError("l1_feat_dots items must be 2-tuples")
            for d in dot:
                if not (-1 <= d < len(self.l1_feat_specs)):
                    raise ValueError("Illegal index in feat_dots")
        if len(self.feat_params) != len(self.feat_specs):
            raise ValueError("specs and params must have same length")
        for s, p in zip(self.feat_specs, self.feat_params):
            self._check_params(p, spec=s)

    @property
    def num_feat_param_sets(self):
        return len(self.feat_params)

    @property
    def feat_spec_list(self):
        return self.feat_specs + self.l0_feat_specs + self.l1_feat_specs

    @property
    def num_vi_feats(self):
        return len(self.l0_feat_specs) + len(self.l1_feat_dots)

    def get_feat_usps(self):
        usp0 = RHO_MULT_USPS[self.rho_mult]
        usps = []
        for spec in self.feat_specs:
            usps.append(usp0 + SPEC_USPS[spec])
        for spec in self.l0_feat_specs:
            usps.append(usp0 + SPEC_USPS[spec])
        for (
            j,
            k,
        ) in self.l1_feat_dots:
            spec1 = "grad_rho" if j == -1 else self.l1_feat_specs[j]
            spec2 = "grad_rho" if k == -1 else self.l1_feat_specs[k]
            usps.append(usp0 + SPEC_USPS[spec1] + SPEC_USPS[spec2])
        return usps

    def ueg_vector(self, rho=1.0):
        # a little hacky, but easiest way to avoid
        # redundant code right now
        return np.append(
            NLDFSettingsVJ.ueg_vector(self, rho=rho),
            NLDFSettingsVI.ueg_vector(self, rho=rho),
        )

    def get_reasonable_normalizer(self):
        return NLDFSettingsVJ.get_reasonable_normalizer(
            self
        ) + NLDFSettingsVI.get_reasonable_normalizer(self)


class NLDFSettingsVK(NLDFSettings):

    version = "k"

    def __init__(
        self,
        sl_level,
        theta_params,
        rho_mult,
        feat_params,
        rho_damp,
    ):
        """
        Initialize NLDFSettingsVK

        Args:
            sl_level (str): "GGA" or "MGGA", the level of semilocal ingredients
                used to construct the length-scale exponent.
            theta_params (np.ndarray):
                Settings for the squared-exponential kernel exponent for the r'
                (integrated) coordinate of the features. For version 'k', this
                'exponent' is not used in the squared-exponential but rather
                within the density damping scheme (see rho_damp).
            rho_mult (str): Multiply the density that gets integrated
                by a prefactor. Options: See ALLOWED_RHO_MULTS.
            feat_params (list of np.ndarray):
                Each item in the list is an array with the parameters for the
                feature corresponding to the feat_specs above. Typically, each array
                has three numbers [a0, grad_mul, tau_mul], except erf_rinv which
                has an additional parameter erf_mul for the ratio of the
                erf / rinv exponent to the squared-exponential exponent.
                tau_mul is ignored if sl_level="GGA" and may therefore be excluded.
            rho_damp (str): Specifies the damping for the
                density. Options: 'none', 'exponential', 'asymptotic_const'
        """
        super(NLDFSettingsVK, self).__init__(sl_level, theta_params, rho_mult)
        self.feat_specs = ["se"] * len(feat_params)
        self.feat_params = feat_params
        self._check_specs(self.feat_specs, ALLOWED_K_SPECS)
        for s, p in zip(self.feat_specs, self.feat_params):
            self._check_params(p, spec=s)
        self.rho_damp = rho_damp
        if self.rho_damp not in ALLOWED_RHO_DAMPS:
            raise ValueError("rho_damp argument must be in ALLOWED_RHO_DAMPS.")

    @property
    def num_feat_param_sets(self):
        return len(self.feat_params)

    @property
    def feat_spec_list(self):
        return self.feat_specs

    @property
    def num_vi_feats(self):
        return 0

    def get_feat_usps(self):
        usp0 = RHO_MULT_USPS[self.rho_mult]
        return [usp0 + SPEC_USPS[spec] for spec in self.feat_specs]

    def ueg_vector(self, rho=1.0):
        a0t = self.theta_params[0]
        if self.sl_level == "MGGA":
            t0t = self.theta_params[2]
        else:
            t0t = self.theta_params[1]
        rho_mult = self._ueg_rho_mult(rho)
        expnt_theta = _get_ueg_expnt(a0t, t0t, rho)
        ueg_feats = []
        for spec, params in zip(self.feat_specs, self.feat_params):
            if self.sl_level == "MGGA":
                t0i = params[2]
            else:
                t0i = params[1]
            expnt = _get_ueg_expnt(params[0], t0i, rho)
            # integral part
            integral = (np.pi / expnt) ** 1.5
            # damping part
            integral *= np.exp(-1.5 * expnt_theta / expnt)
            # multipliers for more complicated integrals
            if spec == "se":
                integral *= 1
            elif spec == "se_ar2":
                integral *= 1.5
            elif spec == "se_a2r4":
                integral *= 3.75
            elif spec == "se_erf_rinv":
                expnt3 = expnt * params[-1]
                integral *= np.sqrt(expnt / (expnt + expnt3))
            else:
                raise ValueError
            ueg_feats.append(rho * rho_mult * integral)
        return np.asarray(ueg_feats, dtype=np.float64)

    def get_reasonable_normalizer(self):
        nvk = self.num_feat_param_sets
        uegs = self.ueg_vector()
        usps = self.get_feat_usps()
        norms = []
        for i in range(nvk):
            if usps[i] == 0:
                norms.append(ConstantNormalizer(2.0 / uegs[i]))
            else:
                a0 = self.theta_params[0]
                if self.sl_level == "MGGA":
                    tau_mul = self.theta_params[2]
                else:
                    tau_mul = self.theta_params[1]
                norms.append(
                    get_normalizer_from_exponent_params(
                        0.0, -0.5 * usps[i], a0, tau_mul
                    )
                )
        return norms


class SemilocalSettings(BaseSettings):
    """
    Semilocal feature set. Currently only supports meta-GGA.
    Should not be edited from default in general.
    """

    def __init__(self, mode="nst"):
        # mode should be nst or npa (MGGA) or ns or np (GGA).
        self.level = None
        if mode in ["nst", "npa"]:
            self.level = "MGGA"
        elif mode in ["ns", "np"]:
            self.level = "GGA"
        else:
            raise ValueError("Mode must be nst, npa, ns, or np.")
        self.mode = mode

    @property
    def nfeat(self):
        if self.level == "MGGA":
            return 3
        else:
            return 2

    def get_feat_usps(self):
        if self.mode == "nst":
            return [3, 8, 5]
        elif self.mode == "npa":
            return [3, 0, 0]
        elif self.mode == "ns":
            return [3, 8]
        else:
            return [3, 0]

    def ueg_vector(self, rho=1.0):
        if self.mode == "nst":
            return np.array([rho, 0.0, CFC * rho ** (5.0 / 3)])
        elif self.mode == "npa":
            return np.array([rho, 0.0, 1.0])
        else:
            return np.array([rho, 0.0])

    def get_reasonable_normalizer(self):
        return [None] * self.nfeat


class FeatureSettings(BaseSettings):
    """
    The FeatureSettings object is a container for the settings of each
    type of feature in the model.
    """

    def __init__(
        self,
        sl_settings=None,
        nldf_settings=None,
        nlof_settings=None,
        sdmx_settings=None,
        hyb_settings=None,
        normalizers=None,
    ):
        """
        Initialize FeatureSettings.

        Args:
            sl_settings (SemilocalSettings or EmptySettings):
                Semilocal feature settings
            nldf_settings (NLDFSettings or EmptySettings):
                Nonlocal density feature settings
            nlof_settings (FracLaplSettings or EmptySettings):
                Nonlocal orbital feature settings
            sdmx_settings (SDMXBaseSettings or EmptySettings):
                Spherically averaged EXX settings
            hyb_settings (HybridSettings or EmptySettings):
                (Local) hybrid DFT settings
            normalizers (FeatNormalizerList): List of normalizers for features
        """
        self.sl_settings = EmptySettings() if sl_settings is None else sl_settings
        self.nldf_settings = EmptySettings() if nldf_settings is None else nldf_settings
        self.nlof_settings = EmptySettings() if nlof_settings is None else nlof_settings
        self.sdmx_settings = EmptySettings() if sdmx_settings is None else sdmx_settings
        self.hyb_settings = EmptySettings() if hyb_settings is None else hyb_settings
        self.normalizers = (
            FeatNormalizerList([None] * self.nfeat, slmode=self.sl_settings.mode)
            if normalizers is None
            else normalizers
        )
        if self.hyb_settings.nfeat != 0:
            raise NotImplementedError("Hybrid DFT")

    @property
    def has_sl(self):
        return not self.sl_settings.is_empty

    @property
    def has_nldf(self):
        return not self.nldf_settings.is_empty

    @property
    def has_nlof(self):
        return not self.nlof_settings.is_empty

    @property
    def has_sdmx(self):
        return not self.sdmx_settings.is_empty

    @property
    def nfeat(self):
        return (
            self.sl_settings.nfeat
            + self.nldf_settings.nfeat
            + self.nlof_settings.nfeat
            + self.sdmx_settings.nfeat
            + self.hyb_settings.nfeat
        )

    def get_feat_loc(self):
        return np.cumsum(
            [
                0,
                self.sl_settings.nfeat,
                self.nldf_settings.nfeat,
                self.nlof_settings.nfeat,
                self.sdmx_settings.nfeat,
                self.hyb_settings.nfeat,
            ]
        )

    def get_feat_loc_dict(self):
        loc = self.get_feat_loc()
        return {
            "sl": loc[0],
            "nldf": loc[1],
            "nlof": loc[2],
            "sadm": loc[3],
            "hyb": loc[4],
            "end": loc[5],
        }

    def get_feat_usps(self, with_normalizers=False):
        res = np.concatenate(
            [
                self.sl_settings.get_feat_usps(),
                self.nldf_settings.get_feat_usps(),
                self.nlof_settings.get_feat_usps(),
                self.sdmx_settings.get_feat_usps(),
                self.hyb_settings.get_feat_usps(),
            ]
        )
        if with_normalizers:
            res += np.array(self.normalizers.get_usps())
        return res

    def ueg_vector(self, rho=1.0, with_normalizers=False):
        res = np.concatenate(
            [
                self.sl_settings.ueg_vector(rho),
                self.nldf_settings.ueg_vector(rho),
                self.nlof_settings.ueg_vector(rho),
                self.sdmx_settings.ueg_vector(rho),
                self.hyb_settings.ueg_vector(rho),
            ]
        )
        if with_normalizers:
            res *= self.normalizers.ueg_vector(rho)
        return res

    def get_reasonable_normalizer(self):
        return (
            self.sl_settings.get_reasonable_normalizer()
            + self.nldf_settings.get_reasonable_normalizer()
            + self.nlof_settings.get_reasonable_normalizer()
            + self.sdmx_settings.get_reasonable_normalizer()
            + self.hyb_settings.get_reasonable_normalizer()
        )

    def assign_reasonable_normalizer(self):
        self.normalizers = FeatNormalizerList(
            self.get_reasonable_normalizer(), slmode=self.sl_settings.mode
        )


def dtauw(rho, sigma):
    return -sigma / (8 * rho**2 + 1e-16), 1 / (8 * rho + 1e-16)


def get_uniform_tau(rho):
    return (3.0 / 10) * (3 * np.pi**2) ** (2.0 / 3) * rho ** (5.0 / 3)


def get_single_orbital_tau(rho, mag_grad):
    return mag_grad**2 / (8 * rho + 1e-16)


def get_s2(rho, sigma):
    # TODO should this cutoff not be needed if everything else is stable?
    # rho = np.maximum(1e-10, rho)
    cond = rho < ALPHA_TOL
    b = 2 * (3 * np.pi * np.pi) ** (1.0 / 3)
    s = np.sqrt(sigma) / (b * rho ** (4.0 / 3) + 1e-16)
    s[cond] = 0.0
    return s * s


def ds2(rho, sigma):
    # s = |nabla n| / (b * n)
    # TODO should this cutoff not be needed if everything else is stable?
    cond = rho < ALPHA_TOL
    rho = np.maximum(ALPHA_TOL, rho)
    b = 2 * (3 * np.pi * np.pi) ** (1.0 / 3)
    s = np.sqrt(sigma) / (b * rho ** (4.0 / 3) + 1e-16)
    s2 = s**2
    res = -8.0 * s2 / (3 * rho + 1e-16), 1 / (b * rho ** (4.0 / 3) + 1e-16) ** 2
    res[0][cond] = 0.0
    res[1][cond] = 0.0
    return res


def get_alpha(rho, sigma, tau):
    cond = rho < ALPHA_TOL
    rho = np.maximum(ALPHA_TOL, rho)
    tau0 = get_uniform_tau(rho)
    tauw = get_single_orbital_tau(rho, np.sqrt(sigma))
    # TODO this numerical stability trick is a bit of a hack.
    # Should make spline support small negative alpha
    # instead, for the sake of clean code and better stability.
    alpha = np.maximum((tau - tauw), 0) / tau0
    alpha[cond] = 0
    return alpha
    # return np.maximum((tau - tauw), 0) / tau0


def dalpha(rho, sigma, tau):
    cond = rho < ALPHA_TOL
    rho = np.maximum(ALPHA_TOL, rho)
    tau0 = get_uniform_tau(rho)
    tauw = sigma / (8 * rho)
    dwdn, dwds = -sigma / (8 * rho * rho), 1 / (8 * rho)
    dadn = 5.0 * (tauw - tau) / (3 * tau0 * rho) - dwdn / tau0
    dadsigma = -dwds / tau0
    dadtau = 1 / tau0
    # cond = (tau - tauw) / tau0 < -0.1
    dadn[cond] = 0
    dadsigma[cond] = 0
    dadtau[cond] = 0
    return dadn, dadsigma, dadtau
