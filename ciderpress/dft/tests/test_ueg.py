import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from ciderpress.dft.debug_numint import get_get_exponent, get_nonlocal_features
from ciderpress.dft.settings import (
    ALLOWED_I_SPECS_L0,
    ALLOWED_I_SPECS_L1,
    ALLOWED_J_SPECS,
    NLDFSettingsVI,
    NLDFSettingsVIJ,
    NLDFSettingsVJ,
    NLDFSettingsVK,
)

# constants for uniform grid
M = 100
XMAX = 10.0
N = 2 * M + 1
H = (XMAX / M) ** 3
TOL = 10


class TestUEGVector(unittest.TestCase):
    def _run_ueg_test(self, rho_const, gg_kwargs, vv_gg_kwargs):
        tau_const = 0.3 * (3 * np.pi**2) ** (2.0 / 3) * rho_const ** (5.0 / 3)

        x = np.linspace(-XMAX, XMAX, N)
        y = np.linspace(-XMAX, XMAX, N)
        z = np.linspace(-XMAX, XMAX, N)

        xg, yg, zg = np.meshgrid(x, y, z)
        coords = np.stack([xg.flatten(), yg.flatten(), zg.flatten()]).T
        coords = np.ascontiguousarray(coords)

        G = coords.shape[0]

        vvrho = np.zeros((5, G))
        vvrho[0] += rho_const
        vvrho[4] += tau_const
        vvweight = H * np.ones(G)
        rho = vvrho[:, :1].copy()

        get_exponent_r = get_get_exponent(gg_kwargs)
        get_exponent_rp = get_get_exponent(vv_gg_kwargs)
        theta_params = [vv_gg_kwargs["a0"], 0.0, vv_gg_kwargs["fac_mul"]]
        feat_params = []
        feat_params_k = []
        for i in range(4):
            feat_params.append([gg_kwargs["a0"], 0.0, gg_kwargs["fac_mul"]])
            feat_params_k.append(
                [
                    gg_kwargs["a0"] * 2 ** (i - 1),
                    0.0,
                    gg_kwargs["fac_mul"] * 2 ** (i - 1),
                ]
            )
        feat_params[3].append(2.0)

        isettings = NLDFSettingsVI(
            "MGGA",
            theta_params,
            "one",
            l0_feat_specs=ALLOWED_I_SPECS_L0,
            l1_feat_specs=ALLOWED_I_SPECS_L1,
            l1_feat_dots=[(-1, 0), (-1, 1), (0, 1)],
        )
        jsettings = NLDFSettingsVJ(
            "MGGA",
            theta_params,
            "one",
            feat_specs=ALLOWED_J_SPECS,
            feat_params=feat_params,
        )
        ijsettings = NLDFSettingsVIJ(
            "MGGA",
            theta_params,
            "one",
            l0_feat_specs_i=ALLOWED_I_SPECS_L0,
            l1_feat_specs_i=ALLOWED_I_SPECS_L1,
            l1_feat_dots_i=[(-1, 0), (-1, 1), (0, 1)],
            feat_specs_j=ALLOWED_J_SPECS,
            feat_params_j=feat_params,
        )
        ksettings = NLDFSettingsVK(
            "MGGA",
            theta_params,
            "one",
            feat_params=feat_params_k,
            rho_damp="exponential",
        )

        zero = np.zeros((1, 3))
        ifeat = get_nonlocal_features(
            rho,
            zero,
            vvrho,
            vvweight,
            coords,
            get_exponent_r,
            get_exponent_rp,
            version="i",
        )[0]
        ifeat0 = ifeat[:6]
        ifeat1 = np.zeros(3)
        ifeat = np.append(ifeat0, ifeat1)
        jfeat = get_nonlocal_features(
            rho,
            zero,
            vvrho,
            vvweight,
            coords,
            get_exponent_r,
            get_exponent_rp,
            version="j",
        )[0]
        kfeat = get_nonlocal_features(
            rho,
            zero,
            vvrho,
            vvweight,
            coords,
            get_exponent_r,
            get_exponent_rp,
            version="k",
        )[0]
        ijfeat = np.append(jfeat, ifeat)

        ifeat_ana = isettings.ueg_vector(rho_const)
        jfeat_ana = jsettings.ueg_vector(rho_const)
        ijfeat_ana = ijsettings.ueg_vector(rho_const)
        kfeat_ana = ksettings.ueg_vector(rho_const)
        assert_almost_equal(ijfeat_ana, np.append(jfeat_ana, ifeat_ana), TOL)

        assert_almost_equal(ifeat_ana, ifeat, TOL)
        assert_almost_equal(jfeat_ana, jfeat, TOL)
        assert_almost_equal(ijfeat_ana, ijfeat, TOL)
        assert_almost_equal(kfeat_ana, kfeat, TOL)

    def test_ueg(self):
        num = 1.0
        vv_gg_kwargs = {"a0": 1.0 * num, "fac_mul": 0.03125 * num, "amin": 0.0625 * num}
        num *= 1
        gg_kwargs = {"a0": 1.0 * num, "fac_mul": 0.03125 * num, "amin": 0.0625 * num}
        self._run_ueg_test(1.0, gg_kwargs, vv_gg_kwargs)
        num = 2.3
        vv_gg_kwargs = {"a0": 1.0 * num, "fac_mul": 0.03125 * num, "amin": 0.0625 * num}
        num *= 0.7
        gg_kwargs = {"a0": 1.0 * num, "fac_mul": 0.03125 * num, "amin": 0.0625 * num}
        self._run_ueg_test(0.8, gg_kwargs, vv_gg_kwargs)


if __name__ == "__main__":
    unittest.main()
