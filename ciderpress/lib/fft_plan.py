import ctypes

import numpy as np

from ciderpress.lib import load_library

libfft = load_library("libfft_wrapper.so")

libfft.allocate_fftnd_plan.restype = ctypes.c_void_p
libfft.malloc_fft_plan_in_array.restype = ctypes.c_void_p
libfft.malloc_fft_plan_out_array.restype = ctypes.c_void_p
libfft.get_fft_plan_in_array.restype = ctypes.c_void_p
libfft.get_fft_plan_out_array.restype = ctypes.c_void_p
libfft.get_fft_input_size.restype = ctypes.c_int
libfft.get_fft_output_size.restype = ctypes.c_int


class FFTWrapper:
    def __init__(self, dims, fwd=True, r2c=False, inplace=False, batch_first=True):
        self._dims = dims
        self._fwd = fwd
        self._r2c = r2c
        self._inplace = inplace
        self._batch_first = batch_first
        dims = np.asarray(dims, dtype=np.int32)
        self._ptr = ctypes.c_void_p(
            libfft.allocate_fftnd_plan(
                ctypes.c_int(len(dims)),
                dims.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(1 if fwd else 0),
                ctypes.c_int(1 if r2c else 0),
                ctypes.c_int(1),
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

    def call(self, x):
        dtype = np.float64 if (self._r2c and not self._fwd) else np.complex128
        out = np.empty(libfft.get_fft_output_size(self._ptr), dtype=dtype)
        libfft.write_fft_input(self._ptr, x.ctypes.data_as(ctypes.c_void_p))
        libfft.execute_fft_plan(self._ptr)
        libfft.read_fft_output(self._ptr, out.ctypes.data_as(ctypes.c_void_p))
        return out
