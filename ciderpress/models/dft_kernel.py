import numpy as np
from pyscf.lib.scipy_helper import pivoted_cholesky

from ciderpress.dft.xc_evaluator import KernelEvalBase, MappedDFTKernel


class DFTKernel(KernelEvalBase):
    """
    Raw descriptors are denoted X0.
    Transformed descriptors are denoted as X1.
    Nctrl is the number of control points for the kernel.
    N0 is the number of raw features.
    N1 is the number of transformed features.

    Attributes:
        mode (str): One of (SEP, POL, OSPOL, SSPOL, NPOL).
        X1ctrl (Nctrl, N1): Control points for kernel.

    k: kernel
    dkdX0T: kernel derivative with respect to raw features.
    m: Multiplicative baseline
    a: Additive baseline
    dm: Derivative of multiplicative baseline wrt features
    da: Derivative of additive baseline wrt features
    SEP array shapes:
        k: (Nctrl, nspin, Nsamp)
        m and a: (nspin, Nsamp)
        dkdX0T: (Nctrl, nspin, N0, Nsamp)
        dm and da: (nspin, N0, Nsamp)
    NPOL, POL, OSPOL, SSPOL array shapes:
        k: (Nctrl, Nsamp)
        m and a: Nsamp
        dkdX0t: (Nctrl, nspin, N0, Nsamp)
        dm and da: (N0, nspin, Nsamp)
    """

    def __init__(
        self,
        kernel,
        feature_list,
        mode,
        multiplicative_baseline,
        additive_baseline=None,
        ctrl_tol=1e-5,
        ctrl_nmax=None,
        component=None,
    ):
        self.kernel = kernel
        self.feature_list = feature_list
        self.mode = mode
        self._mul_basefunc = multiplicative_baseline
        self._add_basefunc = additive_baseline
        self.ctrl_tol = ctrl_tol
        self.ctrl_nmax = ctrl_nmax
        if component is None:
            component = "x"
        assert component in ["x", "c", "xc"]
        self.component = component

        self.X1ctrl = None
        self.alpha = None
        self.base_dict = {}
        self.cov_dict = {}
        self.dbase_dict = {}
        self.dcov_dict = {}
        self.rxn_cov_list = []

    def set_kernel(self, kernel):
        self.kernel = kernel

    @property
    def N1(self):
        return self.feature_list.nfeat

    @property
    def Nctrl(self):
        if self.X1ctrl is None:
            raise ValueError("X1ctrl not set.")
        return self.X1ctrl.shape[0]

    def _reduce_npts(self, X):
        S = self.kernel(X, X)
        normlz = np.power(np.diag(S), -0.5)
        Snorm = np.dot(np.diag(normlz), np.dot(S, np.diag(normlz)))
        odS = np.abs(Snorm)
        np.fill_diagonal(odS, 0.0)
        odSs = np.sum(odS, axis=0)
        sortidx = np.argsort(odSs)

        Ssort = Snorm[np.ix_(sortidx, sortidx)].copy()
        c, piv, r_c = pivoted_cholesky(Ssort, tol=self.ctrl_tol)
        if self.ctrl_nmax is not None and self.ctrl_nmax < r_c:
            r_c = self.ctrl_nmax
        idx = sortidx[piv[:r_c]]
        return np.asfortranarray(X[idx])

    def X0Tlist_to_X1array(self, X0T_list):
        X1_list = []
        for X0T in X0T_list:
            X1_list.append(self.get_descriptors(X0T))
        X1 = np.concatenate(X1_list, axis=0)
        return X1

    def X0Tlist_to_X1array_mul(self, X0T_list, mul_list):
        X1_list = []
        for mult, X0T in zip(mul_list, X0T_list):
            X1_list.append(self.get_descriptors_with_mul(X0T, mult))
        X1 = np.concatenate(X1_list, axis=0)
        return X1

    def set_control_points(self, X0T_list, reduce=True):
        X1 = self.X0Tlist_to_X1array(X0T_list)
        if reduce:
            X1 = self._reduce_npts(X1)
        self.X1ctrl = X1

    def get_kctrl(self):
        k = self.kernel(self.X1ctrl, self.X1ctrl)
        self.Kmm = k
        return self.Kmm

    def get_k(self, X0T):
        nspin, N0, Nsamp = X0T.shape
        X1 = self.get_descriptors(X0T)
        k = self.kernel(X1, self.X1ctrl)
        if self.mode == "SEP":
            k = k.T.reshape(self.Nctrl, nspin, Nsamp)
        return k

    def get_k_and_deriv(self, X0T):
        """
        Return Knm

        Args:
            X0T:

        Returns:

        """
        nspin, N0, Nsamp = X0T.shape
        X1 = self.get_descriptors(X0T)
        # k is (nspin * Nsamp, Nctrl)
        # dkdX1 is (nspin * Nsamp, Nctrl, N1)
        k, dkdX1 = self.kernel.k_and_deriv(X1, self.X1ctrl)
        if self.mode == "SEP":
            # k is (Nctrl, nspin, Nsamp)
            k = k.T.reshape(self.Nctrl, nspin, Nsamp)
        else:
            raise NotImplementedError
        # dkdX1 is (Nctrl, nspin * Nsamp, N1)
        dkdX1 = dkdX1.transpose(1, 0, 2)
        dkdX0T = np.empty((self.Nctrl, nspin, N0, Nsamp))
        for i in range(self.Nctrl):
            dkdX0T[i] = self.apply_descriptor_grad(X0T, dkdX1[i])
        return k, dkdX0T

    def map(self, mapping_plan):
        fevals = mapping_plan(self)
        return MappedDFTKernel(
            fevals,
            self.feature_list,
            self.mode,
            self._mul_basefunc,
            self._add_basefunc,
        )
