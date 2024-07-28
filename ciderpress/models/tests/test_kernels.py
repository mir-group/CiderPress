import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from ciderpress.models import kernels


class EdgeCaseKernel(kernels.DiffRBF):
    def diag(self, X):
        return np.diag(self(X))

    def __call__(self, X, Y=None, eval_gradient=False):
        K = self.k_and_deriv(X, Y)[0]
        if eval_gradient:
            length_scale = self.length_scale
            if self.hyperparameter_length_scale.fixed:
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                dists = (X[:, None, :] - X[None, :, :]) / length_scale
                dists = np.einsum("ijk,ijk->ij", dists, dists)
                K_gradient = (K * dists)[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale**2
                )
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def k_and_deriv(self, X, Y=None):
        if Y is None:
            Y = X
        k = kernels.DiffRBF.__call__(self, X, Y)
        dk = k[:, :, None] * (Y[None, :, :] - X[:, None, :])
        dk /= self.length_scale**2
        return k, dk


class ParentTest(unittest.TestCase):

    Kernel = kernels.DiffRBF
    args = []
    kwargs = {}
    Sigma = 0.2
    XSize = 50
    YSize = 30
    Nfeat = 4
    Delta = 1e-7

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        X = np.random.uniform(low=0.0, high=1.0, size=(cls.XSize, cls.Nfeat))
        Y = np.random.uniform(low=0.0, high=1.0, size=(cls.YSize, cls.Nfeat))
        f = (
            0.5 * X[:, 0] ** 2
            - 0.2 * X[:, 1]
            + 0.8 * (1 - X[:, 2]) * (1 - X[:, 3]) ** 2
        )
        f += np.random.normal(scale=cls.Sigma, size=cls.XSize)
        DX = np.empty((cls.Nfeat, 2, cls.XSize, cls.Nfeat))
        for i in range(cls.Nfeat):
            DX[i, 0] = X
            DX[i, 1] = X
            DX[i, 0, :, i] -= 0.5 * cls.Delta
            DX[i, 1, :, i] += 0.5 * cls.Delta
        cls.X = X
        cls.Y = Y
        cls.f = f
        cls.DX = DX

    def test_deep(self):
        kernel = self.Kernel(*(self.args), **(self.kwargs))
        other_kernel = self.Kernel(*(self.args), **(self.kwargs))
        n = 2.5
        m = 1.5
        ekernel = kernels.DiffExponentiation(kernel, n)
        skernel = kernel + other_kernel
        mkernel = kernel * other_kernel
        const = kernels.DiffConstantKernel(m)
        ckernel1 = const * kernel
        ckernel2 = kernel * const
        X, Y = self.X, self.Y
        k0 = kernel(X, Y)
        k1 = ekernel(X, Y)
        assert_almost_equal(k0**n, k1)
        kx1, dk1theta = kernel(X, eval_gradient=True)
        kx2, dk2theta = ekernel(X, eval_gradient=True)
        assert_almost_equal((n * kx1 ** (n - 1))[:, :, np.newaxis] * dk1theta, dk2theta)
        kx2, dk2theta = skernel(X, eval_gradient=True)
        assert_almost_equal(2 * kx1, kx2)
        assert_almost_equal(np.concatenate([dk1theta, dk1theta], axis=2), dk2theta)
        kx2, dk2theta = mkernel(X, eval_gradient=True)
        assert_almost_equal(kx1**2, kx2)
        ref = kx1[:, :, np.newaxis] * dk1theta
        assert_almost_equal(np.concatenate([ref, ref], axis=2), dk2theta)
        kx2, dk2theta = ckernel1(X, eval_gradient=True)
        assert_almost_equal(m * kx1, kx2)
        ref = np.append((m * kx1)[:, :, np.newaxis], m * dk1theta, axis=2)
        assert_almost_equal(ref, dk2theta)
        kx2, dk2theta = ckernel2(X, eval_gradient=True)
        assert_almost_equal(m * kx1, kx2)
        ref = np.append(m * dk1theta, (m * kx1)[:, :, np.newaxis], axis=2)
        assert_almost_equal(ref, dk2theta)

        k1, dk1feat = kernel.k_and_deriv(X, Y)
        k2, dk2feat = ekernel.k_and_deriv(X, Y)
        assert_almost_equal((n * k1 ** (n - 1))[:, :, np.newaxis] * dk1feat, dk2feat)
        k2, dk2feat = skernel.k_and_deriv(X, Y)
        assert_almost_equal(2 * k1, k2)
        assert_almost_equal(2 * dk1feat, dk2feat)
        k2, dk2feat = mkernel.k_and_deriv(X, Y)
        assert_almost_equal(k1**2, k2)
        assert_almost_equal(2 * k1[:, :, np.newaxis] * dk1feat, dk2feat)
        for ckernel in [ckernel1, ckernel2]:
            k2, dk2feat = ckernel.k_and_deriv(X, Y)
            assert_almost_equal(m * k1, k2)
            assert_almost_equal(m * dk1feat, dk2feat)

    def test_diag(self):
        kernel = self.Kernel(*(self.args), **(self.kwargs))
        k1 = kernel(self.X)
        k2 = kernel.diag(self.X)
        assert_almost_equal(np.diag(k1), k2)

    def test_theta_gradient(self):
        kernel = self.Kernel(*(self.args), **(self.kwargs))
        theta = kernel.theta.copy()
        k0 = kernel(self.X)
        k, dk = kernel(self.X, eval_gradient=True)
        assert_almost_equal(k, k0)
        assert dk.shape[-1] == len(theta)
        for i in range(len(theta)):
            theta_tmp = theta.copy()
            theta_tmp[i] -= 0.5 * self.Delta
            kernel.theta = theta_tmp
            km = kernel(self.X)
            theta_tmp[i] += self.Delta
            kernel.theta = theta_tmp
            kp = kernel(self.X)
            kernel.theta = theta
            assert_almost_equal(dk[:, :, i], (kp - km) / self.Delta)

    def test_k_and_deriv(self):
        kernel = self.Kernel(*(self.args), **(self.kwargs))
        X, Y, DX = self.X, self.Y, self.DX
        k0, dk0 = kernel.k_and_deriv(X)
        k1, dk1 = kernel.k_and_deriv(X, X)
        assert_almost_equal(k0, k1)
        assert_almost_equal(dk0, dk1)
        k = kernel(X)
        assert_almost_equal(k0, k)
        k = kernel(X, X)
        assert_almost_equal(k0, k)
        k, dk = kernel.k_and_deriv(X, Y)
        kt = kernel(X, Y)
        assert_almost_equal(k, kt)
        for i in range(self.Nfeat):
            km, _ = kernel.k_and_deriv(DX[i, 0], Y)
            kp, _ = kernel.k_and_deriv(DX[i, 1], Y)
            assert_almost_equal(dk[:, :, i], (kp - km) / self.Delta)
        k, dk = kernel.k_and_deriv(X)
        for i in range(self.Nfeat):
            km, _ = kernel.k_and_deriv(DX[i, 0], X)
            kp, _ = kernel.k_and_deriv(DX[i, 1], X)
            assert_almost_equal(dk[:, :, i], (kp - km) / self.Delta)


class SubsetParentTest(ParentTest):

    Kernel = kernels.SubsetRBF
    args = [slice(1, 4, 2)]

    def test_subset_works(self):
        indexes = self.args[0]
        kernel1 = self.Kernel(*(self.args), **(self.kwargs))
        kernel2 = self.Kernel.__bases__[1](*(self.args[1:]), **(self.kwargs))
        Xsub = self.X[:, indexes]
        Ysub = self.Y[:, indexes]
        k1 = kernel1(self.X, self.Y)
        k2 = kernel2(Xsub, Ysub)
        assert_almost_equal(k1, k2)
        k1, dk1 = kernel1(self.X, eval_gradient=True)
        k2, dk2 = kernel2(Xsub, eval_gradient=True)
        assert_almost_equal(k1, k2)
        assert_almost_equal(dk1, dk2)
        k1, dk1 = kernel1.k_and_deriv(self.X, self.Y)
        dk2 = np.zeros((self.X.shape[0], self.Y.shape[0], self.X.shape[1]))
        k2, dk2[:, :, indexes] = kernel2.k_and_deriv(Xsub, Ysub)
        assert_almost_equal(k1, k2)
        assert_almost_equal(dk1, dk2)


class SpinSymParentTest(ParentTest):

    Kernel = kernels.SpinSymRBF
    args = ([1, 3], [2, 0])

    def test_subset_works(self):
        alpha_ind, beta_ind = self.args[:2]
        kernel1 = self.Kernel(*(self.args), **(self.kwargs))
        kernel2 = self.Kernel.__bases__[1](*(self.args[2:]), **(self.kwargs))
        Xsub = np.vstack([self.X[:, alpha_ind], self.X[:, beta_ind]])
        Ysub = np.vstack([self.Y[:, alpha_ind], self.Y[:, beta_ind]])
        k1 = kernel1(self.X, self.Y)
        k2 = kernel2(Xsub, Ysub)
        NX, NY = self.X.shape[0], self.Y.shape[0]
        k2 = k2[:NX] + k2[NX:]
        k2 = k2[:, :NY] + k2[:, NY:]
        assert_almost_equal(k1, k2)
        k1, dk1 = kernel1(self.X, eval_gradient=True)
        k2, dk2 = kernel2(Xsub, eval_gradient=True)
        k2 = k2[:NX] + k2[NX:]
        k2 = k2[:, :NX] + k2[:, NX:]
        dk2 = dk2[:NX] + dk2[NX:]
        dk2 = dk2[:, :NX] + dk2[:, NX:]
        assert_almost_equal(k1, k2)
        assert_almost_equal(dk1, dk2)
        k1, dk1 = kernel1.k_and_deriv(self.X, self.Y)
        k2, dk2 = kernel2.k_and_deriv(Xsub, Ysub)
        k2 = k2[:NX] + k2[NX:]
        k2 = k2[:, :NY] + k2[:, NY:]
        dk2 = dk2[:, :NY] + dk2[:, NY:]
        dk2full = np.zeros_like(dk1)
        dk2full[:, :, alpha_ind] = dk2[:NX]
        dk2full[:, :, beta_ind] = dk2[NX:]
        assert_almost_equal(k1, k2)
        assert_almost_equal(dk1, dk2full)

    def test_spinsym_works(self):
        alpha_ind, beta_ind = self.args[:2]
        kernel1 = self.Kernel(*(self.args), **(self.kwargs))
        k1 = kernel1(self.X, self.Y)
        X2 = self.X.copy()
        X2[:, alpha_ind] = self.X[:, beta_ind]
        X2[:, beta_ind] = self.X[:, alpha_ind]
        Y2 = self.Y.copy()
        Y2[:, alpha_ind] = self.Y[:, beta_ind]
        Y2[:, beta_ind] = self.Y[:, alpha_ind]
        k2 = kernel1(self.X, Y2)
        assert_almost_equal(k1, k2)
        k2 = kernel1(X2, self.Y)
        assert_almost_equal(k1, k2)
        k2 = kernel1(X2, Y2)
        assert_almost_equal(k1, k2)
        k1 = kernel1(self.X)
        k2 = kernel1(X2)
        assert_almost_equal(k1, k2)
        k1, dk1 = kernel1.k_and_deriv(self.X, self.Y)
        k2, dk2 = kernel1.k_and_deriv(self.X, Y2)
        assert_almost_equal(k1, k2)
        assert_almost_equal(dk1, dk2)
        k2, dk2 = kernel1.k_and_deriv(X2, self.Y)
        dk2swap = dk2.copy()
        dk2swap[:, :, alpha_ind] = dk2[:, :, beta_ind]
        dk2swap[:, :, beta_ind] = dk2[:, :, alpha_ind]
        assert_almost_equal(k1, k2)
        assert_almost_equal(dk1, dk2swap)
        k1, dk1 = kernel1.k_and_deriv(self.X)
        k2, dk2 = kernel1.k_and_deriv(self.X, X2)
        assert_almost_equal(k1, k2)
        assert_almost_equal(dk1, dk2)


class TestDiffRBF(ParentTest):
    Kernel = kernels.DiffRBF
    args = []
    kwargs = {"length_scale": np.array([0.1, 10.0, 1.0, 2.0])}


class TestDiffTransform(ParentTest):
    Kernel = kernels.DiffTransform
    args = []
    Delta = 1e-6
    kwargs = {
        "kernel": kernels.DiffRBF(length_scale=np.array([0.1, 10.0, 1.0, 2.0])),
        "matrix": np.linspace(-1, 1, 16).reshape(4, 4),
        "avg": np.array([0.5, 1.2, -0.8, 0.1]),
        "std": np.array([0.8, 3.0, 0.2, 1.3]),
    }


class TestFixedDiffRBF(ParentTest):
    Kernel = kernels.DiffRBF
    args = []
    kwargs = {
        "length_scale": np.array([0.1, 10.0, 1.0, 2.0]),
        "length_scale_bounds": "fixed",
    }


class TestEdgeCases1(ParentTest):
    Kernel = EdgeCaseKernel
    args = []
    kwargs = {"length_scale": np.array([0.1, 10.0, 1.0, 2.0])}


class TestEdgeCases2(ParentTest):
    Kernel = EdgeCaseKernel
    args = []
    kwargs = {}


class TestEdgeCases3(ParentTest):
    Kernel = EdgeCaseKernel
    args = []
    kwargs = {"length_scale_bounds": "fixed"}


class TestDiffARBF(ParentTest):
    Kernel = kernels.DiffARBF
    args = []
    kwargs = {
        "order": 2,
        "length_scale": np.array([0.15, 12.0, 1.2, 2.3]),
        "scale": np.array([1e-1, 1e-2, 1.0]),
    }


class TestDiffARBFO3(ParentTest):
    Kernel = kernels.DiffARBF
    args = []
    kwargs = {
        "order": 3,
        "length_scale": np.array([0.15, 12.0, 1.2, 2.3]),
        "scale": np.array([1e-1, 1e-2, 1.0, 1.0]),
    }


class TestDiffARBFV2(ParentTest):
    Kernel = kernels.DiffARBFV2
    args = []
    kwargs = {
        "order": 3,
        "length_scale": np.array([0.15, 12.0, 1.2, 2.3]),
        "scale": np.array([1e-1, 1e-2, 1.0, 1.0]),
    }


class TestDiffAddLLRBF(ParentTest):
    Kernel = kernels.DiffAddLLRBF
    Delta = 1e-6
    args = []
    kwargs = {
        "order": 3,
        "length_scale": np.array([0.7, 3.0, 1.2, 2.3]),
        "scale": np.array([1e-1, 1e-2, 1.0, 1.0]),
    }


class TestDiffAddLLRBF2(ParentTest):
    Kernel = kernels.DiffAddLLRBF
    Delta = 1e-6
    args = []
    kwargs = {
        "order": 3,
        "alpha": 4.0,
        "length_scale": np.array([0.7, 3.0, 1.2, 2.3]),
        "scale": np.array([1e-1, 1e-2, 1.0, 1.0]),
    }


class TestDiffAddRQ(ParentTest):
    Kernel = kernels.DiffAddRQ
    args = []
    kwargs = {
        "order": 3,
        "alpha": 1.5,
        "length_scale": np.array([0.15, 12.0, 1.2, 2.3]),
        "scale": np.array([1e-1, 1e-2, 1.0, 1.0]),
    }


class TestFixedDiffARBF1(ParentTest):
    Kernel = kernels.DiffARBF
    args = []
    kwargs = {
        "order": 2,
        "length_scale": np.array([0.15, 12.0, 1.2, 2.3]),
        "scale": np.array([1e-1, 1e-2, 1.0]),
        "length_scale_bounds": "fixed",
        "scale_bounds": "fixed",
    }


class TestFixedDiffARBF2(ParentTest):
    Kernel = kernels.DiffARBF
    args = []
    kwargs = {
        "order": 2,
        "length_scale": np.array([0.15, 12.0, 1.2, 2.3]),
        "scale": np.array([1e-1, 1e-2, 1.0]),
        "scale_bounds": "fixed",
    }


class TestFixedDiffARBF3(ParentTest):
    Kernel = kernels.DiffARBF
    args = []
    kwargs = {
        "order": 2,
        "length_scale": np.array([0.15, 12.0, 1.2, 2.3]),
        "scale": np.array([1e-1, 1e-2, 1.0]),
        "length_scale_bounds": "fixed",
    }


class TestDiffPolyKernel1(ParentTest):
    Kernel = kernels.DiffPolyKernel
    args = []
    kwargs = {
        "order": 4,
        "gamma": 0.65,
    }


class TestDiffPolyKernel2(ParentTest):
    Kernel = kernels.DiffPolyKernel
    args = []
    kwargs = {
        "order": 4,
        "gamma": np.array([0.1, 0.2, 1.0, 0.5]),
    }


class TestDiffPolyKernel3(ParentTest):
    Kernel = kernels.DiffPolyKernel
    args = []
    kwargs = {
        "order": 4,
        "gamma": np.array([0.1, 0.2, 1.0, 0.5]),
        "factorial": False,
    }


class TestDiffPolyKernel1(ParentTest):
    Kernel = kernels.DiffPolyKernel
    args = []
    kwargs = {
        "order": 4,
        "gamma": 0.65,
        "gamma_bounds": "fixed",
    }


class TestFixedDiffPolyKernel2(ParentTest):
    Kernel = kernels.DiffPolyKernel
    args = []
    kwargs = {
        "order": 4,
        "gamma": np.array([0.1, 0.2, 1.0, 0.5]),
        "gamma_bounds": "fixed",
    }


class TestSubsetRBF(SubsetParentTest):
    Kernel = kernels.SubsetRBF
    args = ([3, 1],)
    kwargs = {"length_scale": np.array([0.1, 2.0])}


class TestSubsetARBF(SubsetParentTest):
    Kernel = kernels.SubsetARBF
    args = ([3, 1],)
    kwargs = {
        "order": 2,
        "length_scale": np.array([0.15, 1.2]),
        "scale": np.array([1e-1, 1e-2, 1.0]),
    }


class TestSubsetPoly(SubsetParentTest):
    Kernel = kernels.SubsetPoly
    args = ([3, 1],)
    kwargs = {
        "order": 4,
        "gamma": np.array([0.1, 0.5]),
        "factorial": False,
    }


class TestSpinSymRBF(SpinSymParentTest):
    Kernel = kernels.SpinSymRBF
    args = (slice(0, 2), slice(2, 4))
    kwargs = {"length_scale": np.array([0.1, 2.0])}


class TestSpinSymARBF(SpinSymParentTest):
    Kernel = kernels.SpinSymARBF
    args = (slice(0, 2), slice(2, 4))
    kwargs = {
        "order": 2,
        "length_scale": np.array([0.15, 1.2]),
        "scale": np.array([1e-1, 1e-2, 1.0]),
    }


class TestSpinSymPoly(SpinSymParentTest):
    Kernel = kernels.SpinSymPoly
    args = (slice(0, 2), slice(2, 4))
    kwargs = {
        "order": 4,
        "gamma": np.array([0.1, 0.5]),
        "factorial": False,
    }


if __name__ == "__main__":
    unittest.main()
