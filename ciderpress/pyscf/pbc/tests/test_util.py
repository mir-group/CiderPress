import unittest

import numpy as np
from numpy.testing import assert_allclose
from pyscf import lib

from ciderpress.pyscf.pbc.util import FFTInterpolator

FD_DELTA = 1e-6
NUM_FFT_TEST = 10
FFT_PHASES = [
    [1, 4, 3],
    [-3, 0, 0],
    [0, 4, 1],
]
np.random.seed(57)
FFT_REAL_CONSTS = np.random.uniform(size=3)
FFT_COMPLEX_CONSTS = np.random.uniform(size=3) + 1j * np.random.uniform(size=3)
FFT_REAL_VCONSTS = np.random.uniform(size=3)
FFT_COMPLEX_VCONSTS = np.random.uniform(size=3) + 1j * np.random.uniform(size=3)

FFT_REAL_10 = np.random.uniform(size=(10, 3))
FFT_COMPLEX_10 = np.random.uniform(size=(10, 3)) + 1j * np.random.uniform(size=(10, 3))
FFT_REAL_V10 = np.random.uniform(size=(10, 3))
FFT_COMPLEX_V10 = np.random.uniform(size=(10, 3)) + 1j * np.random.uniform(size=(10, 3))


def get_mesh_coords(m):
    x = np.linspace(0, 1, m[0] + 1, dtype=np.float64)[:-1]
    y = np.linspace(0, 1, m[1] + 1, dtype=np.float64)[:-1]
    z = np.linspace(0, 1, m[2] + 1, dtype=np.float64)[:-1]
    return lib.cartesian_prod((x, y, z))


def eval_test_function(x, consts, add_const=True):
    total = 5.0 if add_const else 0
    if consts.ndim == 1:
        total += consts[0] * np.cos(2 * np.pi * x.dot(FFT_PHASES[0]))
        total += consts[1] * np.sin(2 * np.pi * x.dot(FFT_PHASES[1]))
        total += consts[2] * np.sin(2 * np.pi * x.dot(FFT_PHASES[2]))
    else:
        total += consts[:, 0:1] * np.cos(2 * np.pi * x.dot(FFT_PHASES[0]))
        total += consts[:, 1:2] * np.sin(2 * np.pi * x.dot(FFT_PHASES[1]))
        total += consts[:, 2:3] * np.sin(2 * np.pi * x.dot(FFT_PHASES[2]))
    return total


class TestFFTInterpolator(unittest.TestCase):
    def _check_fft(self, m1, m2, r2c):
        x1 = get_mesh_coords(m1)
        x2 = get_mesh_coords(m2)
        consts = FFT_REAL_CONSTS if r2c else FFT_COMPLEX_CONSTS
        consts = consts.copy()
        vconsts = FFT_REAL_VCONSTS if r2c else FFT_COMPLEX_VCONSTS
        vconsts = vconsts.copy()
        func1 = eval_test_function(x1, consts)
        func2 = eval_test_function(x2, consts)
        eval_test_function(x1, vconsts)
        v2 = eval_test_function(x2, vconsts)
        ffti = FFTInterpolator(m1, m2, r2c)
        out = ffti.interpolate(func1, fwd=True)
        assert_allclose(out, func2)
        assert out.size == np.prod(m2)
        out = ffti.interpolate(func2, fwd=False)
        assert_allclose(out, func1)
        assert out.size == np.prod(m1)

        consts[0] += 0.5 * FD_DELTA
        func1p = eval_test_function(x1, consts)
        outp = ffti.interpolate(func1p, fwd=True)
        consts[0] -= FD_DELTA
        func1m = eval_test_function(x1, consts)
        consts[0] += FD_DELTA
        outm = ffti.interpolate(func1m, fwd=True)
        vref = ((outp - outm) / FD_DELTA * v2).mean()
        v1test = ffti.interpolate(v2, fwd=False)
        f1test = eval_test_function(x1, np.array([1.0, 0.0, 0.0]), add_const=False)
        vtest = (v1test * f1test).mean()
        assert_allclose(vtest, vref)

        consts = FFT_REAL_10 if r2c else FFT_COMPLEX_10
        consts = consts.copy()
        vconsts = FFT_REAL_V10 if r2c else FFT_COMPLEX_V10
        vconsts = vconsts.copy()
        func1 = eval_test_function(x1, consts)
        func2 = eval_test_function(x2, consts)
        ffti = FFTInterpolator(m1, m2, r2c)
        out = ffti.interpolate(func1, fwd=True)
        assert_allclose(out, func2)
        assert out.size == 10 * np.prod(m2)
        out = ffti.interpolate(func2, fwd=False)
        assert_allclose(out, func1)
        assert out.size == 10 * np.prod(m1)

    def test_all(self):
        self._check_fft([13, 13, 13], [13, 13, 13], False)
        self._check_fft([13, 13, 13], [13, 13, 13], True)
        self._check_fft([12, 12, 12], [12, 12, 12], True)
        self._check_fft([12, 12, 12], [12, 12, 12], False)
        self._check_fft([12, 12, 12], [24, 24, 24], False)
        self._check_fft([12, 12, 12], [24, 24, 24], True)
        self._check_fft([11, 11, 11], [16, 16, 16], False)
        self._check_fft([11, 11, 11], [16, 16, 16], True)
        self._check_fft([11, 11, 11], [13, 21, 21], False)
        self._check_fft([11, 11, 11], [21, 13, 21], True)
        self._check_fft([17, 11, 11], [10, 17, 13], False)
        self._check_fft([17, 11, 11], [10, 17, 13], True)


if __name__ == "__main__":
    unittest.main()
