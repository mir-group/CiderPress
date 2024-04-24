import numpy as np


class ModelWithNormalizer:
    def __init__(self, model, normalizer):
        """
        Initialize a ModelWithNormalizer.

        Args:
            model (MappedFunctional): Model evaluator
            normalizer (FeatNormalizerList): Instructions for normalizing
                the features.
        """
        if model.nfeat != normalizer.nfeat:
            raise ValueError
        self.model = model
        self.normalizer = normalizer

    def get_x0_usps(self):
        usp0 = self.model.settings.get_feat_usps()
        usp1 = self.normalizer.get_usps()
        return [u0 + u1 for u0, u1 in zip(usp0, usp1)]

    @property
    def nfeat(self):
        return self.normalizer.nfeat

    def __call__(self, X0T, rho_data, rhocut=0):
        assert X0T.ndim == 3
        nspin = X0T.shape[0]
        X0Tnorm = X0T.copy()
        rho = nspin * rho_data[:, 0]
        sigma = (
            nspin * nspin * np.einsum("xg,xg->g", rho_data[:, 1:4], rho_data[:, 1:4])
        )
        tau = nspin * rho_data[:, 4]
        vrho = np.zeros_like(rho)
        vsigma = np.zeros_like(rho)
        vtau = np.zeros_like(rho)
        for i in range(self.nfeat):
            self.normalizer[i].apply_norm_fwd_(X0Tnorm[:, i], rho, sigma, tau)
        res, dres = self.model(X0Tnorm, rhocut=rhocut)
        for i in range(self.nfeat):
            self.normalizer[i].apply_norm_bwd_(
                dres[:, i], X0T[:, i], rho, sigma, tau, vrho, vsigma, vtau
            )
        return res, dres


class MappedXC:
    def __init__(self, mapped_kernels, settings, libxc_baseline=None):
        """

        Args:
            mapped_kernels (list of DFTKernel):
            settings (FeatureSettings):
            libxc_baseline (str):
        """
        self.kernels = mapped_kernels
        self.settings = settings
        self.libxc_baseline = libxc_baseline

    def set_baseline_mode(self, mode):
        # TODO baseline can be GPAW or PySCF mode.
        # Need to implement for more complicated XC.
        raise NotImplementedError

    def __call__(self, X0T, rhocut=0):
        """
        Evaluate functional from normalized features
        Args:
            X0T (array): normalized features
            rhocut (float): low-density cutoff

        Returns:
            res, dres (array, array), XC energy contribution
                and functional derivative
        """
        res, dres = 0, 0
        for kernel in self.kernels:
            tmp, dtmp = kernel(X0T, rhocut=rhocut)
            res += tmp
            dres += dtmp
        return res, dres

    @property
    def normalizer(self):
        return self.settings.normalizers

    @property
    def nfeat(self):
        return self.normalizer.nfeat
