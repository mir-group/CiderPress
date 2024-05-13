import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from ciderpress.dft.feat_normalizer import (
    CFC,
    ConstantNormalizer,
    DensityNormalizer,
    FeatNormalizerList,
    get_normalizer_from_exponent_params,
)
from ciderpress.dft.settings import get_alpha, get_s2
from ciderpress.dft.tests import debug_normalizers as rho_normalizers

DELTA = 1e-7


def eval_fd(func, inps, i):
    inps = [v.copy() for v in inps]
    inps[i] += 0.5 * DELTA
    fp = func(*inps)
    inps[i] -= DELTA
    fm = func(*inps)
    return (fp - fm) / DELTA


class TestFeatNormalizer(unittest.TestCase):

    _x = None
    norm_list: FeatNormalizerList = None
    ref_norm_list: rho_normalizers.FeatNormalizerList = None

    @classmethod
    def setUpClass(cls):
        N = 50
        cls._x = np.linspace(0, 1, N)
        np.random.seed(53)
        directions = 0.25 * np.random.normal(size=(3, N))
        np.random.seed(53)
        alpha = np.random.normal(size=N) ** 2
        rho = 1 + 0.5 * np.sin(np.pi * cls._x)
        drho = directions * rho
        tauw = np.einsum("xg,xg->g", drho, drho) / (8 * rho)
        tau = tauw + alpha * CFC * rho ** (5.0 / 3)
        rho_data = np.concatenate([[rho], drho, [tau]], axis=0)
        assert rho_data.shape[0] == 5
        cls.rho = rho_data
        sigma = np.einsum("xg,xg->g", drho, drho)
        cls.norm_list = FeatNormalizerList(
            [
                None,
                None,
                None,
                ConstantNormalizer(3.14),
                DensityNormalizer(1.2, 0.75),
                get_normalizer_from_exponent_params(0.0, 1.5, 1.0, 0.03125),
                get_normalizer_from_exponent_params(
                    0.75, 1.5, 1.0 * 1.2, 0.03125 * 1.2
                ),
            ]
        )
        np.random.seed(72)
        feat = np.random.normal(size=(cls.norm_list.nfeat - 3, N))
        dfeat = np.random.normal(size=(cls.norm_list.nfeat, N))
        cls.x = np.stack(
            [
                rho,
                get_s2(rho, sigma),
                get_alpha(rho, sigma, tau),
            ]
            + [feat[i] for i in range(feat.shape[0])]
        )[None, :, :]
        cls.dx = dfeat
        cls.ref_norm_list = rho_normalizers.FeatNormalizerList(
            [
                rho_normalizers.ConstantNormalizer(3.14),
                rho_normalizers.DensityPowerNormalizer(1.2, 0.75),
                rho_normalizers.CiderExponentNormalizer(1.0, 0.0, 0.03125, 1.5),
                rho_normalizers.GeneralNormalizer(
                    1.0 * 1.2, 0.0, 0.03125 * 1.2, 1.5, 0.75
                ),
            ]
        )

    def test_evaluate(self):
        rho_data = self.rho.copy()
        rho = rho_data[0]
        sigma = np.einsum("xg,xg->g", rho_data[1:4], rho_data[1:4])
        tau = rho_data[4]
        initial_vals = []

        feat = self.norm_list.get_normalized_feature_vector(self.x)
        dfeat = self.norm_list.get_derivative_of_normed_features(self.x[0], self.dx)
        featp = self.norm_list.get_normalized_feature_vector(self.x + DELTA * self.dx)
        featm = self.norm_list.get_normalized_feature_vector(self.x - DELTA * self.dx)
        dfeat_ref = (featp - featm) / (2 * DELTA)
        for i in range(self.norm_list.nfeat):
            assert_almost_equal(dfeat[i], dfeat_ref[0, i])
        v = 2 * feat
        vfeat = self.norm_list.get_derivative_wrt_unnormed_features(self.x, v)
        for i in range(self.norm_list.nfeat):
            x = self.x.copy()
            x[:, i] += 0.5 * DELTA
            featp = self.norm_list.get_normalized_feature_vector(x)
            ep = (featp**2).sum(axis=1)
            x[:, i] -= DELTA
            featm = self.norm_list.get_normalized_feature_vector(x)
            em = (featm**2).sum(axis=1)
            assert_almost_equal(vfeat[:, i], (ep - em) / DELTA, 5)

        for i in range(self.ref_norm_list.nfeat):
            res0 = self.ref_norm_list[i].evaluate(rho, sigma, tau)
            res, drdn, drds, drdt = self.ref_norm_list[i].evaluate_with_grad(
                rho, sigma, tau
            )
            inps = [rho, sigma, tau]
            drdn_ref = eval_fd(self.ref_norm_list[i].evaluate, inps, 0)
            drds_ref = eval_fd(self.ref_norm_list[i].evaluate, inps, 1)
            drdt_ref = eval_fd(self.ref_norm_list[i].evaluate, inps, 2)
            assert_almost_equal(res, res0)
            assert_almost_equal(feat[0, i + 3], res0 * self.x[0, 3 + i])
            assert_almost_equal(drdn, drdn_ref)
            assert_almost_equal(drds, drds_ref)
            assert_almost_equal(drdt, drdt_ref)
            initial_vals.append((res, drdn, drds, drdt))

        lambd = 1.7
        rho = lambd**3 * rho_data[0]
        sigma = lambd**8 * np.einsum("xg,xg->g", rho_data[1:4], rho_data[1:4])
        tau = lambd**5 * rho_data[4]
        for i in range(self.ref_norm_list.nfeat):
            res, drdn, drds, drdt = self.ref_norm_list[i].evaluate_with_grad(
                rho, sigma, tau
            )
            res_ref, drdn_ref, drds_ref, drdt_ref = initial_vals[i]
            usp = self.ref_norm_list[i].get_usp()
            assert_almost_equal(res, lambd**usp * res_ref)
            assert_almost_equal(drdn, lambd ** (usp - 3) * drdn_ref)
            assert_almost_equal(drds, lambd ** (usp - 8) * drds_ref)
            assert_almost_equal(drdt, lambd ** (usp - 5) * drdt_ref)

        x = self.x.copy()
        x[:, 0] *= lambd**3
        scaled_feat = self.norm_list.get_normalized_feature_vector(x)
        usps = np.array(self.norm_list.get_usps())
        usps[0] = 3
        scaled_ref = lambd ** (usps[None, :, None]) * feat
        for i in range(scaled_ref.shape[1]):
            assert_almost_equal(scaled_feat[:, i], scaled_ref[:, i])


if __name__ == "__main__":
    unittest.main()
