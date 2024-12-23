import ctypes
import itertools
import unittest

import numpy as np
from numpy.testing import assert_allclose

from ciderpress.lib import load_library
from ciderpress.lib.fft_plan import FFTWrapper
from ciderpress.lib.mpi_fft_plan import MPIFFTWrapper, libfft

libpwutil = load_library("libpwutil.so")
libpwutil.mpi_ensure_initialized()

SEED = 1684

ALL_DIMS = [
    [13],
    [16],
    [13, 9],
    [13, 8],
    [16, 18],
    [13, 9, 17],
    [13, 9, 8],
    [16, 18, 12],
    [4, 4, 4, 4],
]


class TestFFT(unittest.TestCase):
    def run_single_test(self, dims, ntrans, fwd, r2c, inplace, batch_first):
        np.random.seed(SEED)
        ishape = [ntrans] + dims if batch_first else dims + [ntrans]
        axes = [i for i in range(len(dims))]
        if batch_first:
            for i in range(len(axes)):
                axes[i] += 1
        if r2c:
            input_arr = np.empty(shape=ishape, dtype=np.float64)
            input_arr[:] = np.random.normal(size=ishape)
        else:
            input_arr = np.empty(shape=ishape, dtype=np.complex128)
            input_arr.real[:] = np.random.normal(size=ishape)
            input_arr.imag[:] = np.random.normal(size=ishape)
        if not fwd:
            if r2c:
                input_arr = np.fft.rfftn(input_arr, axes=axes)
            else:
                input_arr = np.fft.fftn(input_arr, axes=axes)
        if fwd and r2c:
            ref_output_arr = np.fft.rfftn(input_arr, axes=axes)
        elif fwd:
            ref_output_arr = np.fft.fftn(input_arr, axes=axes)
        elif r2c:
            ref_output_arr = np.fft.irfftn(input_arr, axes=axes, s=dims)
        else:
            ref_output_arr = np.fft.ifftn(input_arr, axes=axes)
        wrapper = FFTWrapper(
            dims,
            ntransform=ntrans,
            r2c=r2c,
            fwd=fwd,
            inplace=inplace,
            batch_first=batch_first,
        )
        input_arr.shape = wrapper.input_shape
        ref_output_arr.shape = wrapper.output_shape
        test_output_arr = wrapper.call(input_arr)
        if not fwd:
            test_output_arr /= np.prod(dims)
        test_output_arr = test_output_arr.reshape(ref_output_arr.shape)
        tol = 1e-15 * 10 ** len(dims)
        assert_allclose(test_output_arr, ref_output_arr, atol=tol, rtol=0)

    def test_cider_fft(self):
        tf = [True, False]
        nts = [1, 5]
        settings_gen = itertools.product(ALL_DIMS, nts, tf, tf, tf, tf)
        for dims, nt, fwd, r2c, inplace, batch_first in settings_gen:
            self.run_single_test(dims, nt, fwd, r2c, inplace, batch_first)

    def _check_mpi1(self, array_size):
        np.random.seed(SEED)
        nproc = ctypes.c_int(0)
        rank = ctypes.c_int(0)
        libfft.cider_fft_world_size_and_rank(
            ctypes.byref(nproc),
            ctypes.byref(rank),
        )
        nproc = nproc.value
        rank = rank.value
        dtype = np.complex128
        matrix = np.random.normal(size=array_size).astype(dtype)
        nx, ny, nz = matrix.shape
        xpp = -(-nx // nproc)
        ypp = -(-ny // nproc)
        my_ox = min(xpp * rank, nx)
        my_nx = min(xpp * (rank + 1), nx) - my_ox
        my_oy = min(ypp * rank, ny)
        my_ny = min(ypp * (rank + 1), ny) - my_oy
        my_matrix = matrix[my_ox : my_ox + my_nx].copy()
        my_matrixt = matrix.transpose(1, 0, 2)[my_oy : my_oy + my_ny].copy()
        my_size = max(xpp * ny * nz, ypp * nx * nz)
        my_test_matrix = np.ndarray(my_size, dtype=dtype)
        my_test_matrix[: my_matrix.size] = my_matrix.ravel()
        libfft.transpose_data_loop_world(
            ctypes.c_int(nx),
            ctypes.c_int(ny),
            ctypes.c_int(nz),
            ctypes.c_int(xpp),
            ctypes.c_int(ypp),
            my_test_matrix.ctypes.data_as(ctypes.c_void_p),
        )
        assert_allclose(
            my_test_matrix[: my_matrixt.size], my_matrixt.ravel(), rtol=0, atol=1e-15
        )
        libfft.transpose_data_loop_world(
            ctypes.c_int(ny),
            ctypes.c_int(nx),
            ctypes.c_int(nz),
            ctypes.c_int(ypp),
            ctypes.c_int(xpp),
            my_test_matrix.ctypes.data_as(ctypes.c_void_p),
        )
        assert_allclose(
            my_test_matrix[: my_matrix.size], my_matrix.ravel(), rtol=0, atol=1e-15
        )

    def _check_mpi2(self, dims, nt, r2c=False):
        array_size = dims + [nt]
        np.random.seed(SEED)
        nproc = ctypes.c_int(0)
        rank = ctypes.c_int(0)
        libfft.cider_fft_world_size_and_rank(
            ctypes.byref(nproc),
            ctypes.byref(rank),
        )
        nproc = nproc.value
        rank = rank.value
        dtype = np.float64 if r2c else np.complex128
        matrix = np.random.normal(size=array_size).astype(dtype)
        if r2c:
            kmatrix = np.fft.rfftn(matrix, axes=(0, 1, 2))
        else:
            kmatrix = np.fft.fftn(matrix, axes=(0, 1, 2))
        nx, ny, nz, ntr = matrix.shape
        xpp = -(-nx // nproc)
        ypp = -(-ny // nproc)
        my_ox = min(xpp * rank, nx)
        my_nx = min(xpp * (rank + 1), nx) - my_ox
        my_oy = min(ypp * rank, ny)
        my_ny = min(ypp * (rank + 1), ny) - my_oy
        my_matrix = matrix[my_ox : my_ox + my_nx].copy(order="C")
        my_kmatrix = kmatrix.transpose(1, 0, 2, 3)[my_oy : my_oy + my_ny].copy()
        wr = MPIFFTWrapper(dims, ntransform=nt, r2c=r2c)
        my_test_kmatrix = wr.call(my_matrix, fwd=True)
        assert_allclose(my_test_kmatrix, my_kmatrix, rtol=0, atol=1e-12)
        my_test_matrix = wr.call(my_kmatrix, fwd=False)
        my_test_matrix /= np.prod(dims)
        assert_allclose(my_test_matrix, my_matrix, rtol=0, atol=1e-12)

    def check_mpi(self, shape):
        self._check_mpi1(shape)
        for nt in [1, 2, 5]:
            for r2c in [True, False]:
                self._check_mpi2(shape, nt, r2c=r2c)

    def test_mpi(self):
        self.check_mpi([19, 12, 3])
        self.check_mpi([7, 7, 8])
        self.check_mpi([18, 23, 20])
        self.check_mpi([3, 5, 24])


if __name__ == "__main__":
    unittest.main()
