import ctypes
import unittest

import numpy as np
from numpy.testing import assert_allclose

from ciderpress.lib import load_library
from ciderpress.lib.mpi_fft_plan import MPIFFTWrapper, libfft

libpwutil = load_library("libpwutil")
libpwutil.mpi_ensure_initialized()
libfft.cider_fft_set_nthread(ctypes.c_int(-1))

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

    def test_mpi_time(self):
        import time

        dims = [210, 192, 121]
        # dims = [105, 128, 121]
        nt = 10
        array_size = dims + [nt]
        np.random.seed(SEED)
        nproc = ctypes.c_int(0)
        rank = ctypes.c_int(0)
        libfft.cider_fft_set_nthread(ctypes.c_int(-1))
        libfft.cider_fft_world_size_and_rank(
            ctypes.byref(nproc),
            ctypes.byref(rank),
        )
        nproc = nproc.value
        rank = rank.value
        r2c = True
        dtype = np.float64 if r2c else np.complex128
        matrix = np.random.normal(size=array_size).astype(dtype)
        nx = matrix.shape[0]
        xpp = -(-nx // nproc)
        my_ox = min(xpp * rank, nx)
        my_nx = min(xpp * (rank + 1), nx) - my_ox
        my_matrix = matrix[my_ox : my_ox + my_nx].copy(order="C")
        t0 = time.monotonic()
        wr = MPIFFTWrapper(dims, ntransform=nt, r2c=r2c)
        my_test_kmatrix = wr.call(my_matrix, fwd=True)
        my_test_matrix = wr.call(my_test_kmatrix, fwd=False)
        t1 = time.monotonic()
        print("TIME", t1 - t0)
        print(my_test_kmatrix.sum())
        print(my_test_matrix.sum())


if __name__ == "__main__":
    unittest.main()
