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
from pyscf import lib

from ciderpress.dft.plans import libcider


class FFTInterpolator:
    def __init__(self, mesh1, mesh2, r2c=False, num_fft_buffer=4):
        self._mesh1 = np.asarray(mesh1, dtype=np.int32, order="C")
        self._mesh2 = np.asarray(mesh2, dtype=np.int32, order="C")
        self._r2c = r2c
        if self._r2c:
            size1_z = self._mesh1[2] // 2 + 1
            size2_z = self._mesh2[2] // 2 + 1
        else:
            size1_z = self._mesh1[2]
            size2_z = self._mesh2[2]
        b1size = self._mesh1[0] * self._mesh1[1] * size1_z
        b2size = self._mesh2[0] * self._mesh2[1] * size2_z
        self._num_buf = num_fft_buffer
        self._buffer1 = np.empty(
            num_fft_buffer * b1size, dtype=np.complex128, order="C"
        )
        self._buffer2 = np.empty(
            num_fft_buffer * b2size, dtype=np.complex128, order="C"
        )
        self._inverse_scale1 = 1.0 / np.prod(self._mesh1).astype(np.float64)
        self._inverse_scale2 = 1.0 / np.prod(self._mesh2).astype(np.float64)

    @property
    def dtype(self):
        return np.float64 if self._r2c else np.complex128

    @property
    def mesh1(self):
        return self._mesh1.copy()

    @property
    def mesh2(self):
        return self._mesh2.copy()

    def interpolate(self, input, out=None, fwd=True):
        if fwd:
            in_mesh = self._mesh1
            out_mesh = self._mesh2
            in_buf = self._buffer1
            out_buf = self._buffer2
            inverse_scale = self._inverse_scale1
        else:
            in_mesh = self._mesh2
            out_mesh = self._mesh1
            in_buf = self._buffer2
            out_buf = self._buffer1
            inverse_scale = self._inverse_scale2
        inp = input.view()
        assert inp.shape[-1] == np.prod(in_mesh)
        assert inp.dtype == self.dtype
        assert inp.flags.c_contiguous
        inp.shape = (-1, inp.shape[-1])
        if out is None:
            out = np.empty(
                (inp.shape[0], np.prod(out_mesh)), dtype=self.dtype, order="C"
            )
        result = out.view()
        if input.ndim == 1:
            result.shape = (
                1,
                np.prod(out_mesh),
            )
        else:
            result.shape = input.shape[:-1] + (np.prod(out_mesh),)
        if input.ndim == 1 and out.ndim == 2:
            result = result[0]
        assert out.shape[-1] == np.prod(out_mesh)
        assert out.dtype == self.dtype
        assert out.flags.c_contiguous
        out.shape = (-1, out.shape[-1])
        assert inp.shape[0] == out.shape[0]
        num_fft = inp.shape[0]
        for i0, i1 in lib.prange(0, num_fft, self._num_buf):
            libcider.run_ffts(
                inp[i0:i1].ctypes.data_as(ctypes.c_void_p),
                in_buf.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(1.0),
                (ctypes.c_int * 3)(*in_mesh),
                ctypes.c_int(1),
                ctypes.c_int(i1 - i0),
                ctypes.c_int(1),
                ctypes.c_int(1 if self._r2c else 0),
            )
            """
            libcider.run_ffts(
                self._buffer1.ctypes.data_as(ctypes.c_void_p),
                out[i0:i1].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(self._inverse_scale),
                (ctypes.c_int * 3)(*out_mesh),
                ctypes.c_int(0),
                ctypes.c_int(i1 - i0),
                ctypes.c_int(1),
                ctypes.c_int(1 if self._r2c else 0),
            )
            """
            libcider.map_between_fft_meshes(
                in_buf.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int * 3)(*in_mesh),
                out_buf.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int * 3)(*out_mesh),
                ctypes.c_double(1.0),
                ctypes.c_int(1 if self._r2c else 0),
                ctypes.c_int(i1 - i0),
            )
            libcider.run_ffts(
                out_buf.ctypes.data_as(ctypes.c_void_p),
                out[i0:i1].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(inverse_scale),
                (ctypes.c_int * 3)(*out_mesh),
                ctypes.c_int(0),
                ctypes.c_int(i1 - i0),
                ctypes.c_int(1),
                ctypes.c_int(1 if self._r2c else 0),
            )
            # """
        return result
