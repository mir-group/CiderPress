from abc import ABC, abstractmethod

import numpy as np
import yaml
from interpolation.splines.eval_cubic_numba import (
    vec_eval_cubic_splines_1,
    vec_eval_cubic_splines_2,
    vec_eval_cubic_splines_3,
    vec_eval_cubic_splines_4,
    vec_eval_cubic_splines_G_1,
    vec_eval_cubic_splines_G_2,
    vec_eval_cubic_splines_G_3,
    vec_eval_cubic_splines_G_4,
)


def get_vec_eval(grid, coeffs, X, N):
    """
    Call the numba-accelerated spline evaluation routines from the
    interpolation package. Also returns derivatives
    Args:
        grid: start and end points + number of grids in each dimension
        coeffs: coefficients of the spline
        X: coordinates to interpolate
        N: dimension of the interpolation (between 1 and 4, inclusive)
    """
    coeffs = np.expand_dims(coeffs, coeffs.ndim)
    y = np.zeros((X.shape[0], 1))
    dy = np.zeros((X.shape[0], N, 1))
    a_, b_, orders = zip(*grid)
    if N == 1:
        vec_eval_cubic_splines_G_1(a_, b_, orders, coeffs, X, y, dy)
    elif N == 2:
        vec_eval_cubic_splines_G_2(a_, b_, orders, coeffs, X, y, dy)
    elif N == 3:
        vec_eval_cubic_splines_G_3(a_, b_, orders, coeffs, X, y, dy)
    elif N == 4:
        vec_eval_cubic_splines_G_4(a_, b_, orders, coeffs, X, y, dy)
    else:
        raise ValueError("invalid dimension N")
    return np.squeeze(y, -1), np.squeeze(dy, -1)


def get_cubic(grid, coeffs, X, N):
    """
    Call the numba-accelerated spline evaluation routines from the
    interpolation package.
    Args:
        grid: start and end points + number of grids in each dimension
        coeffs: coefficients of the spline
        X: coordinates to interpolate
        N: dimension of the interpolation (between 1 and 4, inclusive)
    """
    coeffs = np.expand_dims(coeffs, coeffs.ndim)
    y = np.zeros((X.shape[0], 1))
    a_, b_, orders = zip(*grid)
    if N == 1:
        vec_eval_cubic_splines_1(a_, b_, orders, coeffs, X, y)
    elif N == 2:
        vec_eval_cubic_splines_2(a_, b_, orders, coeffs, X, y)
    elif N == 3:
        vec_eval_cubic_splines_3(a_, b_, orders, coeffs, X, y)
    elif N == 4:
        vec_eval_cubic_splines_4(a_, b_, orders, coeffs, X, y)
    else:
        raise ValueError("invalid dimension N")
    return np.squeeze(y, -1)


class Evaluator:
    """
    The Evaluator module computes the exchange energy densities
    and exchange enhancement factors for spline-interpolated XC functionals.
    """

    def __init__(
        self,
        scale,
        ind_sets,
        spline_grids,
        coeff_sets,
        xed_y_converter,
        feature_list,
        desc_order,
        const=0,
        desc_version="c",
        a0=8.0,
        fac_mul=0.25,
        amin=0.05,
        args=None,
    ):
        """
        Args:
            scale: lists of weights for the spline functions
            ind_sets: List of index sets. For each set of indices, passes
                the features at those indices to the spline interpolation
                functions.
            spline_grids: grids on which the splines are evaluated
            coeff_sets: Coefficients for the spline interpolation
            xed_y_converter (tuple): set of functions for convert between the
                exchange energy density (xed) and exchange enhancement
                factor y
                    (xed_to_y, y_to_xed, eval_baseline_y,
                    1 if baseline y is LDA, 2 if GGA)
            feature_list (FeatureList): FeatureList object
                (ciderpress.models.transform_data.FeatureList) containing
                the features to pass to the model.
            desc_order (list): indexes the descriptors before passing them
                to the features_list
            const: constant term to add to the spline
            desc_version ('c'): Version of the descriptors used ('a' or 'c')
            a0, fac_mul, amin: parameters passed to CIDER feature generator.
                (see ciderpress.dft.get_gaussian_grid_a/c)
        """
        self.scale = scale
        self.nterms = len(self.scale)
        self.ind_sets = ind_sets
        self.spline_grids = spline_grids
        self.coeff_sets = coeff_sets
        self.xed_y_converter = xed_y_converter
        self.feature_list = feature_list
        self.nfeat = feature_list.nfeat
        self.desc_order = desc_order
        self.const = const
        self.desc_version = desc_version
        self.a0 = a0
        self.fac_mul = fac_mul
        self.amin = amin
        self.args = args
        if self.desc_version in ["b", "d"] and self.args is not None:
            self.vvmul = args.gg_vvmul

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

    def get_descriptors(self, x):
        return self.feature_list(x[:, self.desc_order])

    def predict_from_desc(self, X, vec_eval=False, max_order=3, min_order=0):
        res = np.zeros(X.shape[0]) + self.const
        if vec_eval:
            dres = np.zeros(X.shape)
        for t in range(self.nterms):
            if (not isinstance(self.ind_sets[t], int)) and len(
                self.ind_sets[t]
            ) < min_order:
                continue
            if isinstance(self.ind_sets[t], int) or len(self.ind_sets[t]) <= max_order:
                ind_set = self.ind_sets[t]
                if vec_eval:
                    y, dy = get_vec_eval(
                        self.spline_grids[t],
                        self.coeff_sets[t],
                        X[:, ind_set],
                        len(ind_set),
                    )
                    res += y * self.scale[t]
                    dres[:, ind_set] += dy * self.scale[t]
                else:
                    res += (
                        get_cubic(
                            self.spline_grids[t],
                            self.coeff_sets[t],
                            X[:, ind_set],
                            len(ind_set),
                        )
                        * self.scale[t]
                    )
        if vec_eval:
            return res, dres
        else:
            return res

    def predict(self, X, vec_eval=False):
        desc = self.get_descriptors(X)
        F = self.predict_from_desc(desc, vec_eval=vec_eval)
        if vec_eval:
            F = F[0]
        return self.y_to_xed(F, X)

    def to_dict(self):
        return {
            "scale": np.array(self.scale),
            "ind_sets": self.ind_sets,
            "spline_grids": self.spline_grids,
            "coeff_sets": self.coeff_sets,
            "xed_y_converter": self.xed_y_converter,
            "feature_list": self.feature_list,
            "desc_order": self.desc_order,
            "const": self.const,
            "desc_version": self.desc_version,
            "a0": self.a0,
            "fac_mul": self.fac_mul,
            "amin": self.amin,
            "args": self.args,
        }

    @classmethod
    def from_dict(cls, state_dict):
        return cls(
            state_dict["scale"],
            state_dict["ind_sets"],
            state_dict["spline_grids"],
            state_dict["coeff_sets"],
            state_dict["xed_y_converter"],
            state_dict["feature_list"],
            state_dict["desc_order"],
            const=state_dict["const"],
            desc_version=state_dict["desc_version"],
            a0=state_dict["a0"],
            fac_mul=state_dict["fac_mul"],
            amin=state_dict["amin"],
            args=state_dict["args"],
        )

    def dump(self, fname):
        """
        Save the Evaluator to a file name fname as yaml format.
        """
        state_dict = self.to_dict()
        with open(fname, "w") as f:
            yaml.dump(state_dict, f)

    @classmethod
    def load(cls, fname):
        """
        Load an instance of this class from yaml
        """
        with open(fname, "r") as f:
            state_dict = yaml.load(f, Loader=yaml.CLoader)
        return cls.from_dict(state_dict)


class MLFunctional(ABC):
    """
    Abstract base class for Machine-Learned functionals
    to be evaluated using ciderpress.dft.NLNumInt
    """

    def get_F(self, X):
        """
        Get exchange enhancement factor.
        """
        return self.get_F_and_derivative(X)[0]

    def get_derivative(self, X):
        """
        Get derivative of exchange enhancement factor
        with respect to descriptors.
        """
        return self.get_F_and_derivative(X)[1]

    @abstractmethod
    def get_F_and_derivative(self, X):
        """
        Get exchange enhancement factor and its derivative
        with respect to descriptors.
        """


#######################################################
# Semi-local functionals and simple helper functions. #
#######################################################


def identity(x):
    return x.copy()


def square(x):
    return x**2


def single(x):
    return np.ones(x.shape)


def extract_kernel_components(kernel):
    """
    For an sklearn Kernel object composed of
    three kernels of the form:
    kernel = (const * comp) + (noise),
    extract the three components
    """
    return kernel.k1.k1, kernel.k1.k2, kernel.k2


class Descriptor:
    """
    Old version of the Descriptor class (replaced with
    the models.transform_data module). Used in Semi-local
    functional examples below.
    """

    def __init__(self, code, transform=identity, transform_deriv=single, mul=1.0):
        self._transform = transform
        self._transform_deriv = transform_deriv
        self.code = code
        self.mul = mul

    def transform_descriptor(self, desc, deriv=0):
        if deriv == 0:
            return self._transform(desc[self.code])
        else:
            return self._transform(desc[self.code]), self._transform_deriv(
                desc[self.code]
            )


kappa = 0.804
mu = 0.2195149727645171


class PBEFunctional(MLFunctional):
    """
    PBE Exchange functional using the MLFunctional class
    """

    def __init__(self):
        self.desc_list = [Descriptor(1)]
        self.y_to_f_mul = None

    def get_F(self, X):
        p = X.flatten() ** 2
        return 1 + kappa - kappa / (1 + mu * p / kappa)

    def get_derivative(self, X):
        p = X.flatten() ** 2
        return (mu / (1 + mu * p / kappa) ** 2).reshape(-1, 1)


class SCANFunctional(MLFunctional):
    """
    SCAN Exchange functional using the MLFunctional class
    """

    def __init__(self):
        self.desc_list = [Descriptor(1), Descriptor(2)]
        self.y_to_f_mul = None

    def get_F(self, X):
        p = X[:, 0] ** 2
        s = X[:, 0]
        alpha = X[:, 1]
        muak = 10.0 / 81
        k1 = 0.065
        b2 = np.sqrt(5913 / 405000)
        b1 = (511 / 13500) / (2 * b2)
        b3 = 0.5
        b4 = muak**2 / k1 - 1606 / 18225 - b1**2
        h0 = 1.174
        a1 = 4.9479
        c1 = 0.667
        c2 = 0.8
        dx = 1.24
        tmp1 = muak * p
        tmp2 = 1 + b4 * p / muak * np.exp(-np.abs(b4) * p / muak)
        tmp3 = b1 * p + b2 * (1 - alpha) * np.exp(-b3 * (1 - alpha) ** 2)
        x = tmp1 * tmp2 + tmp3**2
        h1 = 1 + k1 - k1 / (1 + x / k1)
        gx = 1 - np.exp(-a1 / np.sqrt(s + 1e-9))
        -a1 / 4 * (s + 1e-9) ** (-2.5) * np.exp(-a1 / np.sqrt(s + 1e-9))
        fx = np.exp(-c1 * alpha / (1 - alpha)) * (alpha < 1) - dx * np.exp(
            c2 / (1 - alpha)
        ) * (alpha > 1)
        fx[np.isnan(fx)] = 0
        assert not np.isnan(fx).any()
        Fscan = gx * (h1 + fx * (h0 - h1))
        return Fscan

    def get_derivative(self, X):
        p = X[:, 0] ** 2
        s = X[:, 0]
        alpha = X[:, 1]
        muak = 10.0 / 81
        k1 = 0.065
        b2 = np.sqrt(5913 / 405000)
        b1 = (511 / 13500) / (2 * b2)
        b3 = 0.5
        b4 = muak**2 / k1 - 1606 / 18225 - b1**2
        h0 = 1.174
        a1 = 4.9479
        c1 = 0.667
        c2 = 0.8
        dx = 1.24
        tmp1 = muak * p
        tmp2 = 1 + b4 * p / muak * np.exp(-np.abs(b4) * p / muak)
        tmp3 = b1 * p + b2 * (1 - alpha) * np.exp(-b3 * (1 - alpha) ** 2)
        x = tmp1 * tmp2 + tmp3**2
        h1 = 1 + k1 - k1 / (1 + x / k1)
        gx = 1 - np.exp(-a1 / np.sqrt(s + 1e-9))
        dgdp = -a1 / 4 * (s + 1e-9) ** (-2.5) * np.exp(-a1 / np.sqrt(s + 1e-9))
        fx = np.exp(-c1 * alpha / (1 - alpha)) * (alpha < 1) - dx * np.exp(
            c2 / (1 - alpha)
        ) * (alpha > 1)
        fx[np.isnan(fx)] = 0
        assert not np.isnan(fx).any()
        gx * (h1 + fx * (h0 - h1))
        dxdp = (
            muak * tmp2
            + tmp1
            * (
                b4 / muak * np.exp(-np.abs(b4) * p / muak)
                - b4 * np.abs(b4) * p / muak**2 * np.exp(-np.abs(b4) * p / muak)
            )
            + 2 * tmp3 * b1
        )
        dxda = (
            2
            * tmp3
            * (
                -b2 * np.exp(-b3 * (1 - alpha) ** 2)
                + 2 * b2 * b3 * (1 - alpha) ** 2 * np.exp(-b3 * (1 - alpha) ** 2)
            )
        )
        dhdx = 1 / (1 + x / k1) ** 2
        dhdp = dhdx * dxdp
        dhda = dhdx * dxda
        dfda = (-c1 * alpha / (1 - alpha) ** 2 - c1 / (1 - alpha)) * np.exp(
            -c1 * alpha / (1 - alpha)
        ) * (alpha < 1) - dx * c2 / (1 - alpha) ** 2 * np.exp(c2 / (1 - alpha)) * (
            alpha > 1
        )
        dfda[np.isnan(dfda)] = 0

        dFdp = dgdp * (h1 + fx * (h0 - h1)) + gx * (1 - fx) * dhdp
        dFda = gx * (dhda - fx * dhda + dfda * (h0 - h1))
        return np.array([dFdp, dFda]).T


#########################
# GP-based functionals. #
#########################


class GPFunctional(MLFunctional):
    # TODO: This setup currently assumes that the gp directly
    # predict F_X - 1. This will not always be the case.

    def __init__(self, gpr):
        # Assumes kernel_ is (const * rbf) + noise
        msg = """
        GPFunctional is provided as a reference only. Its functional
        derivatives are buggy, so please do not use it for practical
        calculations.

        gpr is a ciderpress.models.gp.DFTGPR object
        """
        import warnings

        warnings.warn(msg)
        from sklearn.gaussian_process.kernels import RBF

        cov = gpr.gp.kernel_.k1
        self.alpha_ = cov.k1.constant_value * gpr.gp.alpha_
        self.kernel = RBF(length_scale=cov.k2.length_scale)
        self.X_train_ = gpr.gp.X_train_[:, 1:]
        self.feature_list = gpr.feature_list
        self.nfeat = self.feature_list.nfeat
        self.desc_version = gpr.args.version[:1]
        self.a0 = gpr.args.gg_a0
        self.amin = gpr.args.gg_amin
        self.fac_mul = gpr.args.gg_facmul
        self.vvmul = gpr.args.gg_vvmul
        self.desc_order = gpr.args.desc_order
        self.fx_baseline = gpr.xed_y_converter[2]
        self.fxb_num = gpr.xed_y_converter[3]
        self.args = gpr.args

    def get_F_and_derivative(self, X):
        rho = X[0]
        mat = np.zeros((self.nfeat, X.shape[1]))
        self.feature_list.fill_vals_(mat, X)

        # X has shape n_test, n_desc
        # X_train_ has shape n_train, n_desc
        k = self.kernel(mat.T, self.X_train_)
        F = k.dot(self.alpha_)
        k * self.alpha_
        # shape n_test, n_desc
        kaxt = np.dot(k * self.alpha_, self.X_train_)
        np.dot(k, self.alpha_)
        dF = kaxt - mat.T * F.reshape(-1, 1)
        dF /= self.kernel.length_scale**2
        dFddesc = np.zeros(X.shape).T
        self.feature_list.fill_derivs_(dFddesc.T, dF.T, X)

        highcut = 1e-3
        lowcut = 1e-6
        rhocut = np.maximum(rho[rho < highcut], lowcut)
        xcut = np.log(rhocut / lowcut) / np.log(highcut / lowcut)
        F[rho < highcut] *= 0.5 * (1 - np.cos(np.pi * xcut))
        dFddesc[rho < highcut, :] *= 0.5 * (1 - np.cos(np.pi * xcut[:, np.newaxis]))
        dFddesc[rho < lowcut, :] = 0

        if self.fxb_num == 1:
            chfx = 1
        elif self.fxb_num == 2:
            chfx, dchfx = self.fx_baseline(X[1])
            dFddesc[:, 1] += dchfx
        else:
            raise ValueError("Unsupported basline fx order.")
        F += chfx

        F[rho < 1e-9] = 0
        dFddesc[rho < 1e-9, :] = 0

        return F, dFddesc

    def get_F(self, X):
        return self.get_F_and_derivative(X)[0]

    def get_derivative(self, X):
        return self.get_F_and_derivative(X)[1]


class NormGPFunctional(MLFunctional, Evaluator):
    """
    MLFunctional to evaluate spline-interpolated GP
    functionals using the utilities in the Evaluator class.
    """

    def __init__(self, *args, **kwargs):
        """
        Same arguments as Evaluator.
        """
        Evaluator.__init__(self, *args, **kwargs)
        self.fxb_num = self.xed_y_converter[3]
        self.fx_baseline = self.xed_y_converter[2]

    def get_F_and_derivative(self, X):
        rho = X[0]
        mat = np.zeros((self.nfeat, X.shape[1]))
        self.feature_list.fill_vals_(mat, X)
        F, dF = self.predict_from_desc(mat.T, vec_eval=True)
        dFddesc = np.zeros(X.shape).T
        self.feature_list.fill_derivs_(dFddesc.T, dF.T, X)

        highcut = 1e-8
        lowcut = 1e-9
        rhocut = np.maximum(rho[rho < highcut], lowcut)
        xcut = np.log(rhocut / lowcut) / np.log(highcut / lowcut)
        F[rho < highcut] *= 0.5 * (1 - np.cos(np.pi * xcut))
        dFddesc[rho < highcut, :] *= 0.5 * (1 - np.cos(np.pi * xcut[:, np.newaxis]))
        dFddesc[rho < lowcut, :] = 0

        if self.fxb_num == 1:
            chfx = 1
        elif self.fxb_num == 2:
            chfx, dchfx = self.fx_baseline(X[1])
            dFddesc[:, 1] += dchfx
        else:
            raise ValueError("Unsupported baseline fx order.")
        F += chfx

        if rho is not None:
            F[rho < 1e-9] = 0
            dFddesc[rho < 1e-9, :] = 0

        return F, dFddesc
