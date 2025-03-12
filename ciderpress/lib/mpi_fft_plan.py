import ctypes

import numpy as np

from ciderpress.lib import load_library

libfft = load_library("libfft_wrapper")

libfft.allocate_mpi_fft3d_plan_world.restype = ctypes.c_void_p
libfft.malloc_fft_plan_in_array.restype = ctypes.c_void_p
libfft.malloc_fft_plan_out_array.restype = ctypes.c_void_p
libfft.get_fft_plan_in_array.restype = ctypes.c_void_p
libfft.get_fft_plan_out_array.restype = ctypes.c_void_p
libfft.get_fft_input_size.restype = ctypes.c_int
libfft.get_fft_output_size.restype = ctypes.c_int


class MPIFFTWrapper:
    def __init__(self, dims, ntransform=1, r2c=False):
        self._dims = dims
        self._ntransform = ntransform
        self._r2c = r2c
        rshape = [d for d in dims]
        if r2c:
            kshape = [d for d in dims[:-1]] + [dims[-1] // 2 + 1]
        else:
            kshape = [d for d in dims]
        rshape.append(self._ntransform)
        kshape.append(self._ntransform)
        dims = np.asarray(dims, dtype=np.int32, order="C")
        self._ptr = ctypes.c_void_p(
            libfft.allocate_mpi_fft3d_plan_world(
                dims.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(1 if r2c else 0),
                ctypes.c_int(self._ntransform),
            )
        )
        nproc = ctypes.c_int(0)
        rank = ctypes.c_int(0)
        libfft.cider_fft_world_size_and_rank(
            ctypes.byref(nproc),
            ctypes.byref(rank),
        )
        nproc = nproc.value
        rank = rank.value
        nx, ny, nz = dims
        xpp = -(-nx // nproc)
        ypp = -(-ny // nproc)
        my_ox = min(xpp * rank, nx)
        my_nx = min(xpp * (rank + 1), nx) - my_ox
        my_oy = min(ypp * rank, ny)
        my_ny = min(ypp * (rank + 1), ny) - my_oy
        self._nproc = nproc
        self._rank = rank
        self._xpp = xpp
        self._ypp = ypp
        self._my_ox = my_ox
        self._my_nx = my_nx
        self._my_oy = my_oy
        self._my_ny = my_ny
        self._rshape = (self._my_nx, self._dims[1], self._dims[2], self._ntransform)
        if r2c:
            self._kshape = [self._dims[0], self._dims[1], self._dims[2] // 2 + 1]
        else:
            self._kshape = self._dims
        self._kshape = (self._my_ny, self._kshape[0], self._kshape[2], self._ntransform)

    def __del__(self):
        libfft.free_mpi_fft3d_plan(self._ptr)

    def _execute_fwd(self):
        libfft.execute_mpi_fft3d_fwd(self._ptr)

    def _execute_bwd(self):
        libfft.execute_mpi_fft3d_bwd(self._ptr)

    def call(self, x, fwd=True):
        if fwd:
            if self._r2c:
                assert x.dtype == np.float64
            else:
                assert x.dtype == np.complex128
            assert x.shape == self._rshape, "{} {}".format(x.shape, self._rshape)
        else:
            assert x.dtype == np.complex128
            assert x.shape == self._kshape, "{} {}".format(x.shape, self._kshape)
        dtype = np.float64 if (self._r2c and not fwd) else np.complex128
        out = np.empty(self._kshape if fwd else self._rshape, dtype=dtype)
        libfft.write_mpi_fft3d_input(
            self._ptr,
            x.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(1 if fwd else 0),
        )
        if fwd:
            self._execute_fwd()
        else:
            self._execute_bwd()
        libfft.read_mpi_fft3d_output(
            self._ptr,
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(1 if fwd else 0),
        )
        return out
