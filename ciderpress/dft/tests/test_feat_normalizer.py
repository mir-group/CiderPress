#!/usr/bin/env python
# CiderPress: Machine-learning based density functional theory calculations
# Copyright (C) 2024 The President and Fellows of Harvard College
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

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
        s2 = get_s2(rho, sigma)
        alpha = get_alpha(rho, sigma, tau)
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
            ],
            slmode="npa",
        )
        np.random.seed(72)
        feat = np.random.normal(size=(cls.norm_list.nfeat - 3, N))
        dfeat = np.random.normal(size=(cls.norm_list.nfeat, N))
        dfeat_np = np.concatenate([dfeat[:2], dfeat[3:]], axis=0)
        dfeat_nst = dfeat.copy()
        dfeat_ns = dfeat_np.copy()
        C = 2 * (3 * np.pi**2) ** (1.0 / 3)
        CF = 0.3 * (3 * np.pi**2) ** (2.0 / 3)
        dsigma = rho ** (8.0 / 3) * dfeat[1]
        dsigma += 8.0 / 3 * rho ** (5.0 / 3) * s2 * dfeat[0]
        dsigma *= C * C
        dtau = 5.0 / 3 * rho ** (2.0 / 3) * (alpha + 5.0 / 3 * s2) * dfeat[0]
        dtau += 5.0 / 3 * rho ** (5.0 / 3) * dfeat[1]
        dtau += rho ** (5.0 / 3) * dfeat[2]
        dtau *= CF
        dfeat_ns[1] = dsigma
        dfeat_nst[1] = dsigma
        dfeat_nst[2] = dtau
        feat_list = [feat[i] for i in range(feat.shape[0])]
        cls.x = cls.x_npa = np.stack(
            [
                rho,
                s2,
                alpha,
            ]
            + feat_list
        )[None, :, :]
        cls.dx = dfeat
        cls.dx_1e = cls.dx.copy()
        cls.dx_1e[2] = 0
        cls.dx_np = dfeat_np
        cls.dx_ns = dfeat_ns
        cls.dx_nst = dfeat_nst
        cls.ref_norm_list = rho_normalizers.FeatNormalizerList(
            [
                rho_normalizers.ConstantNormalizer(3.14),
                rho_normalizers.DensityPowerNormalizer(1.2, 0.75),
                rho_normalizers.CiderExponentNormalizer(1.0, 0.0, 0.03125, 1.5),
                rho_normalizers.GeneralNormalizer(
                    1.0 * 1.2, 0.0, 0.03125 * 1.2, 1.5, 0.75
                ),
            ],
        )
        cls.x_np = np.stack([rho, s2] + feat_list)[None, :, :]
        cls.x_1e = np.stack([rho, s2, np.zeros_like(rho)] + feat_list)[None, :, :]
        cls.x_nst = np.stack([rho, sigma, tau] + feat_list)[None, :, :]
        cls.x_ns = np.stack([rho, sigma] + feat_list)[None, :, :]
        cls.norm_list_np = FeatNormalizerList(
            [
                None,
                None,
                ConstantNormalizer(3.14),
                DensityNormalizer(1.2, 0.75),
                get_normalizer_from_exponent_params(0.0, 1.5, 1.0, 0.03125),
                get_normalizer_from_exponent_params(
                    0.75, 1.5, 1.0 * 1.2, 0.03125 * 1.2
                ),
            ],
            slmode="np",
        )
        cls.norm_list_nst = FeatNormalizerList(
            [
                None,
                DensityNormalizer(0.25 / (3 * np.pi**2) ** (2.0 / 3), -8.0 / 3),
                DensityNormalizer(10.0 / 3 / (3 * np.pi**2) ** (2.0 / 3), -5.0 / 3),
                ConstantNormalizer(3.14),
                DensityNormalizer(1.2, 0.75),
                get_normalizer_from_exponent_params(0.0, 1.5, 1.0, 0.03125),
                get_normalizer_from_exponent_params(
                    0.75, 1.5, 1.0 * 1.2, 0.03125 * 1.2
                ),
            ],
            slmode="nst",
        )
        cls.norm_list_ns = FeatNormalizerList(
            [
                None,
                DensityNormalizer(0.25 / (3 * np.pi**2) ** (2.0 / 3), -8.0 / 3),
                ConstantNormalizer(3.14),
                DensityNormalizer(1.2, 0.75),
                get_normalizer_from_exponent_params(0.0, 1.5, 1.0, 0.03125),
                get_normalizer_from_exponent_params(
                    0.75, 1.5, 1.0 * 1.2, 0.03125 * 1.2
                ),
            ],
            slmode="ns",
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

    def test_evaluate_nst(self):
        feat_ref = self.norm_list.get_normalized_feature_vector(self.x)
        feat_ref[:, 2] += 5.0 / 3 * feat_ref[:, 1]
        feat = self.norm_list_nst.get_normalized_feature_vector(self.x_nst)
        dfeat_ref = self.norm_list.get_derivative_of_normed_features(self.x[0], self.dx)
        dfeat_ref[2] += 5.0 / 3 * dfeat_ref[1]
        dfeat = self.norm_list_nst.get_derivative_of_normed_features(
            self.x_nst[0], self.dx_nst
        )
        featp = self.norm_list_nst.get_normalized_feature_vector(
            self.x_nst + DELTA * self.dx_nst
        )
        featm = self.norm_list_nst.get_normalized_feature_vector(
            self.x_nst - DELTA * self.dx_nst
        )
        dfeat_ref2 = ((featp - featm) / (2 * DELTA))[0]
        for i in range(self.norm_list.nfeat):
            assert_almost_equal(feat[:, i], feat_ref[:, i])
            assert_almost_equal(dfeat[i], dfeat_ref[i])
            assert_almost_equal(dfeat[i], dfeat_ref2[i])
        v = 2 * feat
        vfeat = self.norm_list_nst.get_derivative_wrt_unnormed_features(self.x_nst, v)
        for i in range(self.norm_list.nfeat):
            x = self.x_nst.copy()
            x[:, i] += 0.5 * DELTA
            featp = self.norm_list_nst.get_normalized_feature_vector(x)
            ep = (featp**2).sum(axis=1)
            x[:, i] -= DELTA
            featm = self.norm_list_nst.get_normalized_feature_vector(x)
            em = (featm**2).sum(axis=1)
            assert_almost_equal(vfeat[:, i], (ep - em) / DELTA, 5)
        usps = self.norm_list_nst.get_usps()
        usps_ref = self.norm_list.get_usps()
        muls = [0, 8, 5, 0, 0, 0]
        for usp, usp_ref, mul in zip(usps, usps_ref, muls):
            assert usp + mul == usp_ref, "{} {} {}".format(usp, usp_ref, mul)

    def test_evaluate_np(self):
        feat_ref = self.norm_list.get_normalized_feature_vector(self.x_1e)
        assert_almost_equal(feat_ref[:, 2], 0)
        feat_ref = feat_ref[:, [0, 1, 3, 4, 5, 6]]
        feat = self.norm_list_np.get_normalized_feature_vector(self.x_np)
        dfeat_ref = self.norm_list.get_derivative_of_normed_features(
            self.x_1e[0], self.dx_1e
        )
        dfeat_ref = dfeat_ref[[0, 1, 3, 4, 5, 6]]
        dfeat = self.norm_list_np.get_derivative_of_normed_features(
            self.x_np[0], self.dx_np
        )
        featp = self.norm_list_np.get_normalized_feature_vector(
            self.x_np + DELTA * self.dx_np
        )
        featm = self.norm_list_np.get_normalized_feature_vector(
            self.x_np - DELTA * self.dx_np
        )
        dfeat_ref2 = ((featp - featm) / (2 * DELTA))[0]
        for i in range(self.norm_list_np.nfeat):
            assert_almost_equal(feat[:, i], feat_ref[:, i])
            assert_almost_equal(dfeat[i], dfeat_ref[i])
            assert_almost_equal(dfeat[i], dfeat_ref2[i])
        v = 2 * feat
        vfeat = self.norm_list_np.get_derivative_wrt_unnormed_features(self.x_np, v)
        for i in range(self.norm_list_np.nfeat):
            x = self.x_np.copy()
            x[:, i] += 0.5 * DELTA
            featp = self.norm_list_np.get_normalized_feature_vector(x)
            ep = (featp**2).sum(axis=1)
            x[:, i] -= DELTA
            featm = self.norm_list_np.get_normalized_feature_vector(x)
            em = (featm**2).sum(axis=1)
            assert_almost_equal(vfeat[:, i], (ep - em) / DELTA, 5)
        usps = self.norm_list_np.get_usps()
        usps_ref = np.array(self.norm_list.get_usps())[[0, 1, 3, 4, 5, 6]]
        muls = [0, 0, 0, 0, 0, 0]
        for usp, usp_ref, mul in zip(usps, usps_ref, muls):
            assert usp + mul == usp_ref, "{} {} {}".format(usp, usp_ref, mul)

    def test_evaluate_ns(self):
        feat_ref = self.norm_list_np.get_normalized_feature_vector(self.x_np)
        feat = self.norm_list_ns.get_normalized_feature_vector(self.x_ns)
        dfeat_ref = self.norm_list_np.get_derivative_of_normed_features(
            self.x_np[0], self.dx_np
        )
        dfeat = self.norm_list_ns.get_derivative_of_normed_features(
            self.x_ns[0], self.dx_ns
        )
        featp = self.norm_list_ns.get_normalized_feature_vector(
            self.x_ns + DELTA * self.dx_ns
        )
        featm = self.norm_list_ns.get_normalized_feature_vector(
            self.x_ns - DELTA * self.dx_ns
        )
        dfeat_ref2 = ((featp - featm) / (2 * DELTA))[0]
        for i in range(self.norm_list_np.nfeat):
            assert_almost_equal(feat[:, i], feat_ref[:, i])
            assert_almost_equal(dfeat[i], dfeat_ref[i])
            assert_almost_equal(dfeat[i], dfeat_ref2[i])
        v = 2 * feat
        vfeat = self.norm_list_ns.get_derivative_wrt_unnormed_features(self.x_ns, v)
        for i in range(self.norm_list_np.nfeat):
            x = self.x_ns.copy()
            x[:, i] += 0.5 * DELTA
            featp = self.norm_list_ns.get_normalized_feature_vector(x)
            ep = (featp**2).sum(axis=1)
            x[:, i] -= DELTA
            featm = self.norm_list_ns.get_normalized_feature_vector(x)
            em = (featm**2).sum(axis=1)
            assert_almost_equal(vfeat[:, i], (ep - em) / DELTA, 5)
        usps = self.norm_list_ns.get_usps()
        usps_ref = self.norm_list_np.get_usps()
        muls = [0, 8, 0, 0, 0]
        for usp, usp_ref, mul in zip(usps, usps_ref, muls):
            assert usp + mul == usp_ref, "{} {} {}".format(usp, usp_ref, mul)


if __name__ == "__main__":
    unittest.main()
