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
from numpy.fft import fftn, ifftn, irfftn, rfftn
from numpy.testing import assert_allclose

from ciderpress.pyscf.pbc.sdmx_fft import libcider


def _call_mkl_test(x, fwd, mesh):
    nx, ny, nz = mesh
    nzc = nz // 2 + 1
    if fwd:
        xr = x
        xk = np.empty((nx, ny, nzc), order="C", dtype=np.complex128)
    else:
        xk = x
        xr = np.empty((nx, ny, nz), order="C", dtype=np.float64)
    assert xr.shape == (nx, ny, nz)
    assert xk.shape == (nx, ny, nzc)
    libcider.test_fft3d(
        xr.ctypes.data_as(ctypes.c_void_p),
        xk.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nx),
        ctypes.c_int(ny),
        ctypes.c_int(nz),
        ctypes.c_int(1 if fwd else 0),
    )
    if fwd:
        return xk
    else:
        return xr


def main():
    np.random.seed(34)
    meshes = [
        [32, 100, 19],
        [32, 100, 20],
        [31, 99, 19],
        [2, 2, 4],
        [81, 81, 81],
        [80, 80, 80],
    ]

    for mesh in meshes:
        kmesh = [mesh[0], mesh[1], mesh[2] // 2 + 1]
        xrin = np.random.normal(size=mesh).astype(np.float64)
        xkin1 = np.random.normal(size=kmesh)
        xkin2 = np.random.normal(size=kmesh)
        xkin = np.empty(kmesh, dtype=np.complex128)
        xkin.real = xkin1
        xkin.imag = xkin2
        xkin[:] = rfftn(xrin, norm="backward")
        xkc = fftn(xrin.astype(np.complex128), norm="backward")
        xkin2 = xkin.copy()
        if mesh[2] % 2 == 0:
            xkin2[0, 0, 0] = xkin2.real[0, 0, 0]
            xkin2[0, 0, -1] = xkin2.real[0, 0, -1]
            for ind in [0, -1]:
                for i in range(xkin2.shape[-3]):
                    for j in range(xkin2.shape[-2]):
                        tmp1 = xkin2[i, j, ind]
                        tmp2 = xkin2[-i, -j, ind]
                        xkin2[i, j, ind] = 0.5 * (tmp1 + tmp2.conj())
                        xkin2[-i, -j, ind] = 0.5 * (tmp1.conj() + tmp2)

        xk_np = rfftn(xrin)
        xk_mkl = _call_mkl_test(xrin, True, mesh)
        assert (xk_np.shape == np.array(kmesh)).all()
        assert (xk_mkl.shape == np.array(kmesh)).all()

        xr2_np = ifftn(xkc.copy(), s=mesh, norm="forward")
        xr_np = irfftn(xkin.copy(), s=mesh, norm="forward")
        xr3_np = irfftn(xkin2.copy(), s=mesh, norm="forward")
        xr_mkl = _call_mkl_test(xkin, False, mesh)
        xr2_mkl = _call_mkl_test(xkin2, False, mesh)
        assert (xr_np.shape == np.array(mesh)).all()
        assert (xr_mkl.shape == np.array(mesh)).all()
        assert_allclose(xr2_np.imag, 0, atol=1e-9)
        assert_allclose(xr2_np, xr3_np, atol=1e-9)
        assert_allclose(xr2_mkl, xr3_np, atol=1e-9)
        assert_allclose(xr_mkl, xr_np, atol=1e-9)


if __name__ == "__main__":
    main()
