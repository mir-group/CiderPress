import ctypes

import numpy as np

from ciderpress.lib import load_library

libfft = load_library("libfft_wrapper")

libfft.allocate_fftnd_plan.restype = ctypes.c_void_p
libfft.malloc_fft_plan_in_array.restype = ctypes.c_void_p
libfft.malloc_fft_plan_out_array.restype = ctypes.c_void_p
libfft.get_fft_plan_in_array.restype = ctypes.c_void_p
libfft.get_fft_plan_out_array.restype = ctypes.c_void_p
libfft.get_fft_input_size.restype = ctypes.c_int
libfft.get_fft_output_size.restype = ctypes.c_int


class FFTWrapper:
    def __init__(
        self, dims, ntransform=1, fwd=True, r2c=False, inplace=False, batch_first=True
    ):
        self._dims = dims
        self._ntransform = ntransform
        self._fwd = fwd
        self._r2c = r2c
        self._inplace = inplace
        self._batch_first = batch_first
        rshape = [d for d in dims]
        if r2c:
            kshape = [d for d in dims[:-1]] + [dims[-1] // 2 + 1]
        else:
            kshape = [d for d in dims]
        if batch_first:
            rshape.insert(0, self._ntransform)
            kshape.insert(0, self._ntransform)
        else:
            rshape.append(self._ntransform)
            kshape.append(self._ntransform)
        if fwd:
            self._inshape = tuple(rshape)
            self._outshape = tuple(kshape)
        else:
            self._inshape = tuple(kshape)
            self._outshape = tuple(rshape)
        dims = np.asarray(dims, dtype=np.int32)
        self._ptr = ctypes.c_void_p(
            libfft.allocate_fftnd_plan(
                ctypes.c_int(len(dims)),
                dims.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(1 if fwd else 0),
                ctypes.c_int(1 if r2c else 0),
                ctypes.c_int(self._ntransform),
                ctypes.c_int(1 if inplace else 0),
                ctypes.c_int(1 if batch_first else 0),
            )
        )
        self._in = ctypes.c_void_p(libfft.malloc_fft_plan_in_array(self._ptr))
        if self._inplace:
            self._out = None
        else:
            self._out = ctypes.c_void_p(libfft.malloc_fft_plan_out_array(self._ptr))
        libfft.initialize_fft_plan(
            self._ptr,
            self._in,
            self._out,
        )

    def __del__(self):
        libfft.free_fft_plan(self._ptr)
        libfft.free_fft_array(self._in)
        if not self._inplace:
            libfft.free_fft_array(self._out)

    @property
    def input_shape(self):
        return self._inshape

    @property
    def output_shape(self):
        return self._outshape

    def call(self, x):
        if x.shape != self._inshape:
            raise ValueError(f"Expected input of shape {self._inshape}, got {x.shape}")
        dtype = np.float64 if (self._r2c and not self._fwd) else np.complex128
        out = np.empty(self._outshape, dtype=dtype)
        libfft.write_fft_input(self._ptr, x.ctypes.data_as(ctypes.c_void_p))
        libfft.execute_fft_plan(self._ptr)
        libfft.read_fft_output(self._ptr, out.ctypes.data_as(ctypes.c_void_p))
        return out
