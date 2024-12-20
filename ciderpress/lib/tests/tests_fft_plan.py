import unittest

import numpy as np
from numpy.testing import assert_allclose

from ciderpress.lib.fft_plan import FFTWrapper

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
]


class TestFFT(unittest.TestCase):
    def run_single_test(self, dims, fwd, r2c, inplace, batch_first):
        np.random.seed(SEED)
        if r2c:
            input_arr = np.empty(shape=dims, dtype=np.float64)
            input_arr[:] = np.random.normal(size=dims)
        else:
            input_arr = np.empty(shape=dims, dtype=np.complex128)
            input_arr.real[:] = np.random.normal(size=dims)
            input_arr.imag[:] = np.random.normal(size=dims)
        if not fwd:
            if r2c:
                input_arr = np.fft.rfftn(input_arr)
            else:
                input_arr = np.fft.fftn(input_arr)
        if fwd and r2c:
            ref_output_arr = np.fft.rfftn(input_arr)
        elif fwd:
            ref_output_arr = np.fft.fftn(input_arr)
        elif r2c:
            ref_output_arr = np.fft.irfftn(input_arr, s=dims)
        else:
            ref_output_arr = np.fft.ifftn(input_arr)
        wrapper = FFTWrapper(
            dims, r2c=r2c, fwd=fwd, inplace=inplace, batch_first=batch_first
        )
        test_output_arr = wrapper.call(input_arr)
        if not fwd:
            test_output_arr /= np.prod(dims)
        test_output_arr = test_output_arr.reshape(ref_output_arr.shape)
        print(input_arr.shape, input_arr.dtype)
        print(dims, r2c, fwd)
        tol = 1e-15 * 10 ** len(dims)
        assert_allclose(test_output_arr, ref_output_arr, atol=tol, rtol=0)

    def test_cider_fft(self):
        for dims in ALL_DIMS:
            for r2c in [True, False]:
                for fwd in [False, True]:
                    for inplace in [False]:
                        for batch_first in [False, True]:
                            self.run_single_test(dims, fwd, r2c, inplace, batch_first)


if __name__ == "__main__":
    unittest.main()
