import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from ciderpress.density import get_ldax, get_ldax_dens, get_xed_from_y, get_y_from_xed
from ciderpress.models.kernels import *

SCALE_FAC = (6 * np.pi**2) ** (2.0 / 3) / (16 * np.pi)


def xed_to_y_scan(xed, rho_data):
    pbex = eval_xc("SCAN,", rho_data)[0] * rho_data[0]
    return (xed - pbex) / (ldax(rho_data[0]) - 1e-7)


def y_to_xed_scan(y, rho_data):
    yp = y * ldax(rho_data[0])
    pbex = eval_xc("SCAN,", rho_data)[0] * rho_data[0]
    return yp + pbex


def xed_to_y_lda(xed, rho, s2=None):
    return get_y_from_xed(xed, rho)


def y_to_xed_lda(y, rho, s2=None):
    return get_xed_from_y(y, rho)


def chachiyo_fx(s2):
    c = 4 * np.pi / 9
    x = c * np.sqrt(s2)
    dx = c / (2 * np.sqrt(s2))
    Pi = np.pi
    Log = np.log
    chfx = (3 * x**2 + Pi**2 * Log(1 + x)) / ((Pi**2 + 3 * x) * Log(1 + x))
    dchfx = (
        -3 * x**2 * (Pi**2 + 3 * x)
        + 3 * x * (1 + x) * (2 * Pi**2 + 3 * x) * Log(1 + x)
        - 3 * Pi**2 * (1 + x) * Log(1 + x) ** 2
    ) / ((1 + x) * (Pi**2 + 3 * x) ** 2 * Log(1 + x) ** 2)
    dchfx *= dx
    chfx[s2 < 1e-8] = 1 + 8 * s2[s2 < 1e-8] / 27
    dchfx[s2 < 1e-8] = 8.0 / 27
    return chfx, dchfx


def xed_to_y_chachiyo(xed, rho, s2):
    return xed / get_ldax_dens(rho) - chachiyo_fx(s2)[0]


def y_to_xed_chachiyo(y, rho, s2):
    return (y + chachiyo_fx(s2)[0]) * get_ldax_dens(rho)


def pbe_fx(s2):
    kappa = 0.804
    mu = 0.2195149727645171
    mk = mu / kappa
    fac = 1.0 / (1 + mk * s2)
    fx = 1 + kappa - kappa * fac
    dfx = mu * fac * fac
    return fx, dfx


def xed_to_y_pbe(xed, rho, s2):
    return xed / get_ldax_dens(rho) - pbe_fx(s2)[0]


def y_to_xed_pbe(y, rho, s2):
    return (y + pbe_fx(s2)[0]) * get_ldax_dens(rho)


# For ex/elec training
def xed_to_dex_chachiyo(xed, rho, s2):
    return xed / rho - chachiyo_fx(s2)[0] * get_ldax(rho)


def dex_to_xed_chachiyo(dex, rho, s2):
    return dex * rho + get_ldax_dens(rho) * chachiyo_fx(s2)[0]


def get_unity():
    return 1


def get_identity(x):
    return x


XED_Y_CONVERTERS = {
    # method_name: (xed_to_y, y_to_xed, fx_baseline, nfeat--rho, s2, alpha...)
    "LDA": (xed_to_y_lda, y_to_xed_lda, get_unity, 1),
    "PBE": (xed_to_y_pbe, y_to_xed_pbe, pbe_fx, 2),
    "CHACHIYO": (xed_to_y_chachiyo, y_to_xed_chachiyo, chachiyo_fx, 2),
    "CHACHIYO_EX": (xed_to_dex_chachiyo, dex_to_xed_chachiyo, chachiyo_fx, 2),
}


def get_rbf_kernel(length_scale, scale=1.0, opt_hparams=True, min_lscale=None):
    if min_lscale is None:
        min_lscale = 0.01
    length_scale_bounds = (min_lscale, 10) if opt_hparams else "fixed"
    scale_bounds = (1e-5, 1e3) if opt_hparams else "fixed"
    return ConstantKernel(scale, constant_value_bounds=scale_bounds) * PartialRBF(
        length_scale=length_scale, length_scale_bounds=length_scale_bounds, start=1
    )


def get_agpr_kernel(
    length_scale, scale=None, order=2, nsingle=1, opt_hparams=True, min_lscale=None
):
    if min_lscale is None:
        min_lscale = 0.01
    length_scale_bounds = (min_lscale, 10) if opt_hparams else "fixed"
    scale_bounds = (1e-5, 1e5) if opt_hparams else "fixed"
    start = 1
    if scale is None:
        scale = [1.0] * (order + 1)
    if nsingle == 0:
        singles = None
    elif nsingle == 1:
        singles = SingleRBF(
            length_scale=length_scale[0],
            index=start,
            length_scale_bounds=length_scale_bounds,
        )
    else:
        active_dims = np.arange(start, start + nsingle).tolist()
        singles = PartialRBF(
            length_scale=length_scale[:nsingle],
            active_dims=active_dims,
            length_scale_bounds=length_scale_bounds,
        )
    cov_kernel = PartialARBF(
        order=order,
        length_scale=length_scale[nsingle:],
        scale=scale,
        length_scale_bounds=length_scale_bounds,
        scale_bounds=scale_bounds,
        start=start + nsingle,
    )
    if singles is None:
        return cov_kernel
    else:
        return singles * cov_kernel


def get_density_noise_kernel(noise0=1e-5, noise1=1e-3):
    wk0 = WhiteKernel(noise_level=noise0, noise_level_bounds=(1e-6, 1e-3))
    wk1 = WhiteKernel(noise_level=noise1)
    return wk0 + wk1 * DensityNoise()


def get_noise_kernel(noise0=0.01, opt_noise=True, scale_ldax=True):
    print("NOISE", noise0, opt_noise, scale_ldax)
    if opt_noise:
        bounds = (1e-8, 1e-2)
    else:
        bounds = "fixed"
    kernel = WhiteKernel(noise_level=noise0**2, noise_level_bounds=bounds)
    if scale_ldax:
        kernel = kernel * DensityNoise()
    return kernel


def get_exp_density_noise_kernel(noise0=1e-5, noise1=1e-5):
    wk0 = WhiteKernel(noise_level=noise0, noise_level_bounds=(1e-6, 1e-3))
    wk1 = WhiteKernel(noise_level=noise1, noise_level_bounds=(1e-7, 1e-3))
    return wk0 + wk1 * ExponentialDensityNoise()


def get_fitted_density_noise_kernel(
    decay1=50.0, decay2=1e6, noise0=1e-6, noise1=0.0004, noise2=0.2
):
    rhok1 = FittedDensityNoise(decay_rate=decay1, decay_rate_bounds="fixed")
    rhok2 = FittedDensityNoise(decay_rate=decay2, decay_rate_bounds="fixed")
    wk = WhiteKernel(noise_level=noise0, noise_level_bounds=(1e-6, 1e-3))
    wk1 = WhiteKernel(noise_level=noise1)
    wk2 = WhiteKernel(noise_level=noise2)
    noise_kernel = wk + wk1 * rhok1 + wk2 * rhok2
    return noise_kernel


class DFTGPR:
    def __init__(
        self,
        feature_list,
        desc_order=None,
        xed_y_converter=XED_Y_CONVERTERS["LDA"],
        init_kernel=None,
        use_algpr=False,
    ):
        """
        Args:
            feature_list (e.g. dft.transform_data.FeatureList):
                An object containing an nfeat property which, when
                called, transforms the raw input descriptors to
                features for the GP to use.
        """
        num_desc = feature_list.nfeat
        if desc_order is None:
            desc_order = np.arange(num_desc)

        self.desc_order = desc_order
        self.feature_list = feature_list
        self.xed_y_converter = xed_y_converter

        if init_kernel is None:
            rbf = RBF([1.0] * num_desc, length_scale_bounds=(1.0e-5, 1.0e5))
            wk = WhiteKernel(noise_level=1.0e-3, noise_level_bounds=(1e-05, 1.0e5))
            kernel = rbf + wk
        else:
            kernel = init_kernel
        self.X = None
        self.y = None
        self.gp = GaussianProcessRegressor(kernel=kernel)
        self.init_kernel = kernel

    def get_descriptors(self, x):
        return np.append(x[:, :1], self.feature_list(x[:, self.desc_order]), axis=1)

    def transform_k_derivs(self, dkdx, xdesc):
        # xdesc (nsamp, ninp)
        out = np.zeros_like(xdesc)
        self.feature_list.fill_derivs_(out, dkdx.T[1:], xdesc.T)
        out[:, 0] += dkdx[:, 0]
        return out

    def _xedy(self, y, x, code):
        if self.xed_y_converter[-1] == 1:
            return self.xed_y_converter[code](y, x[:, 0])
        elif self.xed_y_converter[-1] == 2:
            return self.xed_y_converter[code](y, x[:, 0], x[:, 1])
        else:
            return self.xed_y_converter[code](y)

    def xed_to_y(self, y, x):
        return self._xedy(y, x, 0)

    def y_to_xed(self, y, x):
        return self._xedy(y, x, 1)

    def fit(self, xdesc, xed, optimize_theta=True, add_heg=False, add_tail=False):
        if optimize_theta:
            optimizer = "fmin_l_bfgs_b"
        else:
            optimizer = None
        self.gp.optimizer = optimizer
        self.X = self.get_descriptors(xdesc)
        self.y = self.xed_to_y(xed, xdesc)
        print(np.isnan(self.X).sum(), np.isnan(self.y).sum())
        print(self.X.shape, self.y.shape)
        if add_heg:
            hegx = 0 * self.X[0]
            # set the density to be large -> low uncertainty.
            hegx[0] = 1e8
            # Assume heg y-value is zero.
            hegy = 0
            self.y = np.append([hegy], self.y)
            self.X = np.append([hegx], self.X, axis=0)
        if add_tail:
            tailx = 0 * self.X[0]
            tailx[0] = 1e8
            tailx[1] = 1.0
            tailx[2] = 1.0
            tailx[3:] = [b[0] for b in self.feature_list.bounds_list[2:]]
            tailx2 = tailx.copy()
            tailx2[2] = -1.0
            taily = 0
            self.y = np.append([taily, taily], self.y)
            self.X = np.append([tailx, tailx2], self.X, axis=0)
        self.gp.fit(self.X, self.y)
        self.gp.set_params(kernel=self.gp.kernel_)

    def covariance(self, xdesc1, xdesc2):
        X1 = self.get_descriptors(xdesc1)
        X2 = self.get_descriptors(xdesc2)
        return self.gp.kernel(X1, X2)

    def covariance_wrt_train_set(self, xdesc):
        return self.gp.kernel(self.X, self.get_descriptors(xdesc))

    def cov_deriv_wrt_train_set(self, xdesc):
        xtr = self.get_descriptors(xdesc)
        k, deriv = self.gp.k_and_deriv(xtr, self.X).T
        deriv = self.transform_k_derivs(deriv, xdesc)
        return k, deriv

    def covariance_wrt_x_notransform(self, xdesc):
        return self.gp.kernel(self.X, xdesc)

    def refit(self, optimize_theta=True):
        if optimize_theta:
            optimizer = "fmin_l_bfgs_b"
        else:
            optimizer = None
        self.gp.optimizer = optimizer
        self.gp.fit(self.X, self.y)
        self.gp.set_params(kernel=self.gp.kernel_)

    def predict(self, xdesc, return_std=False):
        X = self.get_descriptors(xdesc)
        y = self.gp.predict(X, return_std=return_std)
        if return_std:
            return self.y_to_xed(y[0], xdesc), y[1] * get_ldax_dens(xdesc[:, 0])
        else:
            return self.y_to_xed(y, xdesc)

    # WARNING: Assumes HEG represented by zero-vector
    def add_heg_limit(self):
        # set the feature vector to the HEG (all zeros).
        hegx = 0 * self.X[0]
        # set the density to be large -> low uncertainty.
        hegx[0] = 1e8
        # Assume heg y-value is zero.
        hegy = 0
        self.y = np.append([hegy], self.y)
        self.X = np.append([hegx], self.X, axis=0)
        self.gp.y_train_ = self.y
        self.gp.X_train_ = self.X
        K = self.gp.kernel_(self.gp.X_train_)
        K[np.diag_indices_from(K)] += self.gp.alpha
        # from sklearn gpr
        from scipy.linalg import cho_solve, cholesky

        try:
            self.gp.L_ = cholesky(K, lower=True)  # Line 2
            self.gp._K_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = (
                "The kernel, %s, is not returning a "
                "positive definite matrix. Try gradually "
                "increasing the 'alpha' parameter of your "
                "GaussianProcessRegressor estimator." % self.gp.kernel_,
            ) + exc.args
            raise
        self.gp.alpha_ = cho_solve((self.gp.L_, True), self.gp.y_train_)  # Line 3

    @classmethod
    def from_settings(cls, X, y, feature_list, args):
        if args.desc_order is None:
            desc_order = np.arange(X.shape[1])
        else:
            desc_order = args.desc_order
        X = X[:, desc_order]

        if args.use_ex_kernel:
            xed_y_converter = XED_Y_CONVERTERS["CHACHIYO_EX"]
        else:
            xed_y_converter = XED_Y_CONVERTERS[args.xed_y_code]

        if args.length_scale is None:
            XT = feature_list(X)
            length_scale = np.std(XT, axis=0) * args.length_scale_mul
        else:
            length_scale = args.length_scale

        yt = xed_y_converter[0](y, X[:, 0], X[:, 1])
        if args.use_ex_kernel:
            scale_var = np.var(yt / (get_ldax(X[:, 0]) + 1e-10))
        else:
            scale_var = np.var(yt)
        scale_var *= args.scale_mul

        if args.agpr:
            if args.agpr_scale is not None:
                cov_kernel = get_agpr_kernel(
                    length_scale,
                    args.agpr_scale,
                    args.agpr_order,
                    args.agpr_nsingle,
                    opt_hparams=args.opt_hparams,
                    min_lscale=args.min_lscale,
                )
            # elif args.opt_hparams:
            #    cov_kernel = get_agpr_kernel(length_scale, [scale_var, scale_var, scale_var],
            #                                 args.agpr_order, args.agpr_nsingle,
            #                                 opt_hparams=args.opt_hparams, min_lscale=args.min_lscale)
            else:
                cov_kernel = get_agpr_kernel(
                    length_scale,
                    [1e-5, 1e-5, scale_var],
                    args.agpr_order,
                    args.agpr_nsingle,
                    opt_hparams=args.opt_hparams,
                    min_lscale=args.min_lscale,
                )
        else:
            cov_kernel = get_rbf_kernel(
                length_scale,
                scale_var,
                opt_hparams=args.opt_hparams,
                min_lscale=args.min_lscale,
            )

        if args.use_ex_kernel:
            cov_kernel = (
                SingleDot(index=0, sigma_0=0.0, sigma_0_bounds="fixed") * cov_kernel
            )
        noise_kernel = get_noise_kernel(
            noise0=args.noise,
            opt_noise=args.optimize_noise,
            scale_ldax=(not args.use_ex_kernel),
        )
        init_kernel = cov_kernel + noise_kernel

        gpr = cls(feature_list, desc_order, xed_y_converter, init_kernel)
        gpr.a0 = args.gg_a0
        gpr.fac_mul = args.gg_facmul
        gpr.amin = args.gg_amin
        gpr.desc_version = args.version[:1].lower()
        if gpr.desc_version in ["b", "d", "f"]:
            gpr.vvmul = args.gg_vvmul

        if not args.optimize_noise:
            gpr.ex_sgn = -1
        else:
            gpr.ex_sgn = 1

        return gpr


class EXGPR(DFTGPR):
    def get_descriptors(self, x):
        return np.append(
            self.ex_sgn * get_ldax(x[:, 0])[:, None],
            self.feature_list(x[:, self.desc_order]),
            axis=1,
        )
