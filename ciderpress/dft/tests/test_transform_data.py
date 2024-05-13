import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from ciderpress.dft.transform_data import (
    FeatureList,
    LMap,
    SignedUMap,
    TMap,
    UMap,
    VMap,
    WMap,
    XMap,
    YMap,
    ZMap,
)

TMP_TEST = "test_files/tmp"

TEST_VEC = np.array(
    [
        [0.0, 0.4, 8.0],
        [0.0, 0.8, 5.0],
        [0.0, 2.1, 3.0],
        [0.0, 5.0, 20.0],
        [-12.0, 0.0, 5.0],
        [0.0, 5.0, 20.0],
    ]
)


def get_grad_fd(vec, feat_list, delta=1e-8):
    deriv = np.zeros(vec.shape)
    for i in range(vec.shape[0]):
        dvec = vec.copy()
        dvec[i] += delta
        deriv[i] += np.sum((feat_list(dvec.T).T - feat_list(vec.T).T) / delta, axis=0)
    return deriv


def get_grad_a(vec, feat_list):
    deriv = np.zeros(vec.shape)
    feat = np.zeros((feat_list.nfeat, vec.shape[1]))
    feat_list.fill_vals_(feat, vec)
    fderiv = np.ones(feat.shape)
    feat_list.fill_derivs_(deriv, fderiv, vec)
    return deriv


class TestFeatureNormalizer(unittest.TestCase):
    def test_lmap(self):
        flist = FeatureList([LMap(0), LMap(3)])
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada)

    def test_tmap(self):
        flist = FeatureList([TMap(0, 0, 1), TMap(1, 3, 2)])
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada)

    def test_umap(self):
        flist = FeatureList([UMap(0, 0, 0.25), UMap(1, 3, 0.25)])
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada)

    def test_signed_umap(self):
        flist = FeatureList([SignedUMap(0, 0.25), SignedUMap(3, 0.25)])
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada)

    def test_vmap(self):
        flist = FeatureList(
            [VMap(1, 0.25, scale=2.0, center=1.0), VMap(5, 0.5, scale=2.0, center=1.0)]
        )
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada)

    def test_wmap(self):
        flist = FeatureList([WMap(0, 1, 4, 0.5, 0.5)])
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada)

    def test_xmap(self):
        flist = FeatureList([XMap(0, 1, 4, 0.5, 0.5)])
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada)

    def test_ymap(self):
        flist = FeatureList([YMap(0, 1, 3, 4, 0.5, 0.3, 0.2)])
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada)

    def test_zmap(self):
        flist = FeatureList(
            [ZMap(1, 0.25, scale=2.0, center=1.0), ZMap(5, 0.5, scale=2.0, center=1.0)]
        )
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada)

    def test_integration(self):
        flist = FeatureList(
            [
                UMap(0, 0.25),
                UMap(1, 3, 0.25),
                VMap(1, 0.25, scale=2.0, center=1.0),
                WMap(0, 1, 4, 0.5, 0.5),
                XMap(0, 1, 4, 0.5, 0.5),
                YMap(0, 1, 3, 4, 0.5, 0.3, 0.2),
            ]
        )
        gradfd = get_grad_fd(TEST_VEC, flist)
        grada = get_grad_a(TEST_VEC, flist)
        assert_almost_equal(gradfd, grada, 6)


if __name__ == "__main__":
    unittest.main()
