import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from ciderpress.models.gp import ALGPR


class TestALGPR(unittest.TestCase):
    def test_fit_single(self):
        # from https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.gaussian_process.kernels import ConstantKernel as C

        def f(x):
            return x * np.sin(x)

        X = np.atleast_2d([1.0, 2.0, 5.0, 6.0, 7.0, 8.0]).T
        y = f(X).ravel()
        y += np.random.normal(0, 5 + np.random.random(y.shape))
        x = np.atleast_2d(np.linspace(0, 10, 1000)).T
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1, 1e2))

        gp = GaussianProcessRegressor(kernel=kernel)
        gp.fit(X, y)
        X_train_std = gp.X_train_
        y_train_std = gp.y_train_
        y_pred_std, sigma_std = gp.predict(x, return_std=True)
        K_inv_std = gp._K_inv
        L_std = gp.L_

        print(gp.kernel_)

        gp = ALGPR(kernel=gp.kernel_, optimizer=None)
        gp.fit(X[:-1], y[:-1])
        gp.predict(x)
        gp.fit_single(X[-1:], y[-1])
        K_inv_al = gp._K_inv
        L_al = gp.L_
        X_train_al = gp.X_train_
        y_train_al = gp.y_train_
        y_pred_al, sigma_al = gp.predict(x, return_std=True)

        print(gp.kernel_)

        for mat in (L_std, L_al, L_std - L_al):
            for row in mat:
                print(row)
            print()

        assert_almost_equal(X_train_al, X_train_std)
        assert_almost_equal(y_train_al, y_train_std)
        assert_almost_equal(y_pred_al, y_pred_std)
        assert_almost_equal(sigma_al, sigma_std)
        assert_almost_equal(K_inv_al, K_inv_std)
        assert_almost_equal(L_al, L_std)
