import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.special import lpmn

from ciderpress.dft.futil import fast_sph_harm as fsh


class TestSphHarm(unittest.TestCase):
    def test_sph_harm():
        x = np.linspace(-2, 2, 401)
        assert 0 in x
        assert -1 in x
        assert 1 in x
        m = 8
        n = 8
        N = x.size
        pm, pd = fsh.lpmn_vec(m, n, x)
        assert pm.shape == (m + 1, n + 1, N)
        assert pd.shape == (m + 1, n + 1, N)
        pm_ref, pd_ref = np.zeros_like(pm), np.zeros_like(pd)
        for i in range(N):
            pm_ref[:, :, i], pd_ref[:, :, i] = lpmn(m, n, x[i])
        diff = np.abs(pd_ref - pd)
        diff[np.isnan(diff)] = 0.0
        assert_almost_equal(pm, pm_ref, 14)
        assert_almost_equal(pd, pd_ref, 14)


if __name__ == "__main__":
    unittest.main()
