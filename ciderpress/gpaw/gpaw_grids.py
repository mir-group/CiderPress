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
from gpaw.atom.radialgd import RadialGridDescriptor
from pyscf.dft.radi import BRAGG_RADII

from ciderpress.lib import load_library as load_cider_library

cutil = load_cider_library("libsbt")
cutil.spherical_bessel_transform_setup.restype = ctypes.c_void_p


def _interp_helper(f_g, dg):
    N = f_g.shape[-1]
    g = dg.astype(int)
    g = np.minimum(g, N - 2)
    g = np.maximum(g, 0)
    dg -= g
    return (1 - dg) * f_g[..., g] + dg * f_g[..., g + 1]


class SBTRadialGridDescriptor(RadialGridDescriptor):
    def __init__(self, a, d, N=1000, default_spline_points=25):
        """Radial grid descriptor for Spherical Bessel Transform
        using the NUMSBT algorithm.

        The radial grid is::

                      dg
            r(g) = a(e  ),  g = 0, 1, ..., N - 1
        """

        self.a = a
        self.d = d
        g = np.arange(N)
        r_g = a * np.exp(d * g)
        dr_g = r_g * d
        RadialGridDescriptor.__init__(self, r_g, dr_g, default_spline_points)

    def r2g(self, r):
        g = np.zeros_like(r)
        g[r <= self.a] = 0  # NOTE: truncates g
        g[r > self.a] = np.log(r[r > self.a] / self.a) / self.d
        return g

    def d2gdr2(self):
        raise NotImplementedError

    def interpolate(self, f_g, r_x):
        return _interp_helper(f_g, self.r2g(r_x))

    def interpolate_self(self, f_g, rgd, zero_extrap=False):
        if zero_extrap:
            N = rgd.r_g.shape[0]
            g = rgd.r2g(self.r_g)
            res = _interp_helper(f_g, g)
            res[g > N - 1] = 0
            return res
        else:
            return _interp_helper(f_g, rgd.r2g(self.r_g))

    def new(self, N):
        return SBTRadialGridDescriptor(self.a, self.d, N)


class SBTFullGridDescriptor(SBTRadialGridDescriptor):
    def __init__(self, a_g, encut, d, N=1000, lmax=4, default_spline_points=25):
        """Radial grid descriptor for Spherical Bessel Transform
        using the NUMSBT algorithm.

        The radial grid is::

                      dg
            r(g) = a(e  ),  g = 0, 1, ..., N - 1
        """
        self.a = a_g
        self.d = d
        self.lmax = lmax
        g = np.arange(N)
        r_g = a_g * np.exp(d * g)
        dr_g = r_g * d
        self.k_g = np.zeros_like(r_g)

        self.sbt_desc = cutil.spherical_bessel_transform_setup(
            ctypes.c_double(encut),
            ctypes.c_int(lmax),
            ctypes.c_int(N),
            r_g.ctypes.data_as(ctypes.c_void_p),
            self.k_g.ctypes.data_as(ctypes.c_void_p),
        )

        self.a_k = self.k_g[0]
        self.dk_g = self.k_g * d

        RadialGridDescriptor.__init__(self, r_g, dr_g, default_spline_points)

    def __del__(self):
        cutil.free_sbt_descriptor(ctypes.c_void_p(self.sbt_desc))

    def k2g(self, r):
        g = np.zeros_like(r)
        g[r <= self.a_k] = 0  # NOTE: truncates g
        g[r > self.a_k] = np.log(r[r > self.a_k] / self.a_k) / self.d
        return g

    def interpolate_ks(self, f_k, k_x):
        return _interp_helper(f_k, self.k2g(k_x))

    def transform_single_fwd(self, f_g, l, f_k=None, l_add=0):
        if f_k is None:
            f_k = np.empty_like(f_g)
        cutil.wave_spherical_bessel_transform(
            ctypes.c_void_p(self.sbt_desc),
            f_g.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(l),
            f_k.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(l_add),
        )
        return f_k

    def transform_single_bwd(self, f_k, l, f_g=None):
        if f_g is None:
            f_g = np.empty_like(f_k)
        cutil.inverse_wave_spherical_bessel_transform(
            ctypes.c_void_p(self.sbt_desc),
            f_k.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(l),
            f_g.ctypes.data_as(ctypes.c_void_p),
        )
        return f_g

    def transform_set_fwd(self, f_xLg, f_xLk=None, l_add=0):
        Lmax = f_xLg.shape[-2]
        assert Lmax <= (self.lmax + 1) * (self.lmax + 1)
        nk = f_xLg.shape[-1]
        nset = f_xLg.size // nk
        f_xLg = np.ascontiguousarray(f_xLg)
        if f_xLk is None:
            f_xLk = np.zeros_like(f_xLg, order="C")
        else:
            f_xLk = np.ascontiguousarray(f_xLk)
        cutil.transform_set_fwd(
            ctypes.c_void_p(self.sbt_desc),
            f_xLg.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(Lmax),
            ctypes.c_int(nset),
            ctypes.c_int(nk),
            f_xLk.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(l_add),
        )
        return f_xLk

    def transform_set_bwd(self, f_xLk, f_xLg=None):
        Lmax = f_xLk.shape[-2]
        assert Lmax <= (self.lmax + 1) * (self.lmax + 1)
        nk = f_xLk.shape[-1]
        nset = f_xLk.size // nk
        f_xLk = np.ascontiguousarray(f_xLk)
        if f_xLg is None:
            f_xLg = np.zeros_like(f_xLk, order="C")
        else:
            f_xLg = np.ascontiguousarray(f_xLg)
        cutil.transform_set_bwd(
            ctypes.c_void_p(self.sbt_desc),
            f_xLk.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(Lmax),
            ctypes.c_int(nset),
            ctypes.c_int(nk),
            f_xLg.ctypes.data_as(ctypes.c_void_p),
        )
        return f_xLg


# from pyscf
def becke(n, charge, *args, **kwargs):
    """Becke, JCP 88, 2547 (1988); DOI:10.1063/1.454033"""
    if charge == 1:
        rm = BRAGG_RADII[charge]
    else:
        rm = BRAGG_RADII[charge] * 0.5
    t, w = np.polynomial.chebyshev.chebgauss(n)
    r = (1 + t) / (1 - t) * rm
    w *= 2 / (1 - t) ** 2 * rm * np.sqrt(1 - t**2)
    return r, w, rm


class GCRadialGridDescriptor(RadialGridDescriptor):
    def __init__(self, Z, N=100, default_spline_points=25):
        r_g, dr_g, self.rm = becke(N, Z)
        r_g = np.flip(r_g)
        dr_g = np.flip(dr_g)
        self.npt = N
        RadialGridDescriptor.__init__(self, r_g, dr_g, default_spline_points)

    def r2g(self, r):
        t = (r - self.rm) / (r + self.rm)
        t = np.arccos(t) / np.pi * 2 * self.npt
        return (self.npt - 1) - (t - 1) / 2

    def derivative(self, f, o=None):
        raise NotImplementedError

    def derivative2(self, f, o=None):
        raise NotImplementedError

    def make_cut(self, N):
        return CutGCRadialGridDescriptor(self.r_g[:N], self.dr_g[:N], self.rm, self.npt)


class CutGCRadialGridDescriptor(GCRadialGridDescriptor):
    def __init__(self, r_g, dr_g, rm, npt, default_spline_points=25):
        self.rm = rm
        self.npt = npt
        RadialGridDescriptor.__init__(self, r_g, dr_g, default_spline_points)
