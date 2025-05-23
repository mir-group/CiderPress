import ctypes
import itertools
import unittest

import numpy as np
from numpy.testing import assert_allclose

from ciderpress.lib.fft_plan import FFTWrapper, libfft

# libfft.cider_fft_set_nthread(ctypes.c_int(-1))

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

    def test_time(self):
        libfft.cider_fft_set_nthread(ctypes.c_int(-1))
        import time

        dims = [105, 128, 121]
        nt = 10
        batch_first = False
        if batch_first:
            array_size = [nt] + dims
        else:
            array_size = dims + [nt]
        np.random.seed(SEED)
        r2c = True
        dtype = np.float64 if r2c else np.complex128
        matrix = np.random.normal(size=array_size).astype(dtype)
        my_matrix = matrix.copy(order="C")
        t0 = time.monotonic()
        fwr = FFTWrapper(
            dims,
            ntransform=nt,
            fwd=True,
            r2c=r2c,
            batch_first=batch_first,
            inplace=True,
        )
        bwr = FFTWrapper(
            dims,
            ntransform=nt,
            fwd=False,
            r2c=r2c,
            batch_first=batch_first,
            inplace=True,
        )
        my_test_kmatrix = fwr.call(my_matrix)
        my_test_matrix = bwr.call(my_test_kmatrix)
        t1 = time.monotonic()
        print("TIME", t1 - t0)
        print(my_test_kmatrix.sum())
        print(my_test_matrix.sum())


if __name__ == "__main__":
    unittest.main()
