import unittest

import numpy as np
from numpy.testing import assert_allclose

from ciderpress.gpaw.gpaw_grids import SBTFullGridDescriptor

LMAX = 6


def gaussian(l, alpha, r):
    return r**l * np.exp(-alpha * r * r)


def gaussian_ft(l, alpha, k):
    const = np.pi**1.5 / 2**l / alpha ** (1.5 + l)
    return const * k**l * np.exp((-0.25 / alpha) * k * k)


class TestSBT(unittest.TestCase):
    def test_sbt_gaussian(self):
        sbtgd = SBTFullGridDescriptor(0.001, 1e12, 0.02, N=1024, lmax=LMAX)
        for l in range(LMAX + 1):
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                scale = alpha ** (1.5 + 0.5 * l)
                f_g = scale * gaussian(l, alpha, sbtgd.r_g)
                fref_k = scale * gaussian_ft(l, alpha, sbtgd.k_g)
                ftest_k = 4 * np.pi * sbtgd.transform_single_fwd(f_g, l)
                print("MINMAX", np.min(sbtgd.k_g), np.max(sbtgd.k_g))
                print(
                    np.linalg.norm(ftest_k - fref_k),
                    np.linalg.norm(fref_k),
                    np.linalg.norm(ftest_k),
                )
                print(np.max(np.abs(ftest_k - fref_k)))
                assert_allclose(ftest_k, fref_k, atol=1e-4, rtol=0)
                ftest_g = (0.25 / np.pi) * sbtgd.transform_single_bwd(ftest_k, l)
                assert_allclose(ftest_g, f_g, atol=1e-4, rtol=0)


if __name__ == "__main__":
    unittest.main()
