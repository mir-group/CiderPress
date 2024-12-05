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

import os

import numpy as np
from pyscf.lib import chkfile, prange
from scipy.linalg import cho_solve, cholesky

from ciderpress.dft.plans import get_rho_tuple_with_grad_cross, vxc_tuple_to_array
from ciderpress.dft.settings import FeatureSettings
from ciderpress.dft.xc_evaluator import MappedXC
from ciderpress.dft.xc_evaluator2 import MappedXC2


def strk_to_tuplek(d, ref=False):
    if not isinstance(d, dict):
        raise NotImplementedError
    nd = {}
    for occ, vocc in d.items():
        for num, vnum in vocc.items():
            if ref:
                nd[occ, int(num)] = vnum
                continue
            if isinstance(vnum, (tuple, list)):
                s = vnum[0]
                v = vnum[1]
            else:
                s = 0
                v = vnum
            nd[occ, int(num)] = (s, v)
    return nd


class MOLGP:
    """
    Gaussian process model for the exchange-correlation functional
    or its components.
    """

    def __init__(
        self,
        kernels,
        settings,
        libxc_baseline=None,
        default_noise=0.030,
    ):
        """
        Args:
            kernels (list[DFTKernel]): List of kernels that sum to the XC energy.
            settings (DescParams or FeatureSettings): Settings for the
                features. Specifies what the feature vector is.
            libxc_baseline (str or None): Additional baseline functional to
                evaluate using the libxc library.
            default_noise (float): Default noise hyperparameter, used if noise
                is not provided for a particular data point
        """
        self.libxc_baseline = libxc_baseline
        if not isinstance(settings, FeatureSettings):
            raise ValueError("settings must be FeatureSettings")
        self.settings = settings
        if self.libxc_baseline is not None:
            raise NotImplementedError
        if not isinstance(kernels, list):
            kernels = [kernels]
        if len(kernels) < 1:
            raise ValueError("Need at least 1 covariance kernel")
        self.kernels = kernels
        self.default_noise = default_noise

        self.numerical_epsilon = 1e-9
        self.args = None

        self.reset_reactions()
        self.Xctrl_exch = None
        self.Xctrl_corr = None
        self.y = None
        self.K_ = None
        self.Kcov_ = None
        self.alpha_mol_ = None
        self.y_mol_ = None
        self.ks_baseline_dict = {}
        self.exx_ref_dict = {}
        self.dexx_ref_dict = {}
        # Reference exact exchange values with baseline subtracted
        self.exch_ref_dict = {}
        # Vector of exchange covariances with control points
        self.exch_cov_dict = {}
        # Vector of derivative of exchange covariances with control points,
        # taken with respect to orbital occupation
        self.exch_dcov_dict = {}
        # Baseline for XC energy
        self.corr_bas_dict = {}
        # Vector of correlation covariances with control points
        self.corr_cov_dict = {}
        # Vector of derivative of correlation covariances with control points
        self.corr_dcov_dict = {}

    def map(self, mapping_plans):
        """
        Map the MOLGP model to an Evaluator object that can efficiently
        evaluate the XC energy and which can also be serialized.

        Returns:
            list(callable): Functions that map each
                kernel to a MappedDFTKernel object
        """
        mapped_kernels = [
            kernel.map(mapping_plan)
            for kernel, mapping_plan in zip(self.kernels, mapping_plans)
        ]
        return MappedXC(
            mapped_kernels,
            self.settings,
            libxc_baseline=self.libxc_baseline,
        )

    @property
    def num_kernels(self):
        return len(self.kernels)

    @property
    def xkernels(self):
        return [k for k in self.kernels if k.component == "x"]

    @property
    def ckernels(self):
        return [k for k in self.kernels if k.component != "x"]

    def fit(self, x=None, sigma_min=0.25):
        Kimn_list = []
        Knmimn = 0
        for kernel in self.kernels:
            Kmm = kernel.get_kctrl()
            Kmn = np.stack(kernel.rxn_cov_list).T
            M = Kmm.shape[0]
            mini_noise = self.numerical_epsilon * np.identity(M)
            L_ = cholesky(Kmm + mini_noise, lower=True)
            Kimn = cho_solve((L_, True), Kmn)
            Knmimn += Kmn.T.dot(Kimn)
            Kimn_list.append(Kimn)
        noise_nn = np.array(self.rxn_noise_list)
        noise_nn = noise_nn**2  # get noise covariance from noise std deviation
        if x is not None:
            Knmimn[:] *= x[0] ** 2
            Kimn_list = [x[0] ** 2 * k for k in Kimn_list]
            noise_nn[:] *= sigma_min + x[1] ** 2
        y = np.array(self.rxn_ref_list)
        if len(noise_nn) == 0:
            raise ValueError("Need to set reactions using add_reactions")
        K = Knmimn + np.diag(noise_nn)
        K += self.numerical_epsilon * np.identity(noise_nn.size)
        self.Kcov_ = Knmimn
        self.K_ = K
        L_ = cholesky(K, lower=True)
        alpha_new = cho_solve((L_, True), y)
        self.alpha_mol_ = alpha_new
        self.y_mol_ = y
        for ik, kernel in enumerate(self.kernels):
            kernel.alpha = np.dot(Kimn_list[ik], alpha_new)

    def compute_likelihood(self, x=None, sigma_min=0.25):
        assert self.alpha_mol_ is not None
        if x is None:
            x = np.array([1.0, 1.0])
        y = self.y_mol_
        noise = (sigma_min + x[1] ** 2) * (self.K_ - self.Kcov_)
        Kfull = x[0] ** 2 * self.Kcov_ + noise
        Lfull = cholesky(Kfull, lower=True)
        vec = np.linalg.solve(Lfull, y)
        likelihood = -0.5 * vec.dot(vec)
        # likelihood = -0.5 * np.dot(y, np.linalg.solve(Kfull, y))
        likelihood -= 0.5 * np.linalg.slogdet(Kfull)[1]
        n = y.size
        likelihood -= n / 2 * np.log(2 * np.pi)
        return likelihood

    def optimize_cov_and_noise_(self, refit=True, sigma_min=0.5):
        assert self.alpha_mol_ is not None
        from scipy.optimize import minimize

        def fun(x, *args):
            return -1 * self.compute_likelihood(x, sigma_min=sigma_min)

        res = minimize(fun, np.array([1.0, 1.0]), tol=1e-3)
        print(res.success, res.fun, res.x)
        if refit:
            self.fit(x=res.x, sigma_min=sigma_min)

    def set_control_points(self, X0T_list, reduce=True):
        """
        Transform raw descriptors X and use them as
        control points.

        Args:
            X: Numpy array of raw features
        """
        for kernel in self.kernels:
            kernel.set_control_points(X0T_list, reduce=reduce)

    def _get_normalized_features(self, desc):
        return self.settings.normalizers.get_normalized_feature_vector(desc)

    def _get_normalized_feature_derivs(self, desc, ddesc):
        return self.settings.normalizers.get_derivative_of_normed_features(desc, ddesc)

    @staticmethod
    def load_data(ddir, mol_id, get_orb_deriv):
        fname = os.path.join(ddir["REF"], mol_id + ".hdf5")
        all_data = chkfile.load(fname, "train_data")
        all_data["desc"] = []
        all_data["ddesc"] = []
        for feat_type in ["SL", "NLDF", "NLOF", "SDMX", "HYB"]:
            if ddir[feat_type] is None:
                if feat_type == "SL":
                    raise ValueError("Need semilocal features")
                else:
                    continue
            else:
                fname = os.path.join(ddir[feat_type], mol_id + ".hdf5")
                data = chkfile.load(fname, "train_data")
                all_data["desc"].append(data["desc"])
                if get_orb_deriv or (get_orb_deriv is None and "ddesc" in data):
                    all_data["ddesc"].append(data["ddesc"])
                    get_orb_deriv = True
                else:
                    get_orb_deriv = False
        all_data["desc"] = np.concatenate(all_data["desc"], axis=-2)
        assert all_data["desc"].ndim == 3
        if get_orb_deriv:
            ddesc = all_data["ddesc"]
            all_data["ddesc"] = {}
            ddesc0 = ddesc[0]
            for k1, v1 in ddesc0.items():
                all_data["ddesc"][k1] = {}
                for k2, v2 in ddesc0[k1].items():
                    all_data["ddesc"][k1][k2] = []
                    if all_data["desc"].shape[0] == 2:
                        assert len(v2) == 2
                        spin = ddesc0[k1][k2][0]
                        all_data["ddesc"][k1][k2] = []
                        for sub_ddesc in ddesc:
                            assert spin == sub_ddesc[k1][k2][0]
                            all_data["ddesc"][k1][k2].append(sub_ddesc[k1][k2][1])
                        all_data["ddesc"][k1][k2] = (
                            spin,
                            np.concatenate(all_data["ddesc"][k1][k2], axis=0),
                        )
                        assert all_data["ddesc"][k1][k2][1].ndim == 2
                    else:
                        for sub_ddesc in ddesc:
                            all_data["ddesc"][k1][k2].append(sub_ddesc[k1][k2])
                        all_data["ddesc"][k1][k2] = np.concatenate(
                            all_data["ddesc"][k1][k2], axis=0
                        )
                        assert all_data["ddesc"][k1][k2].ndim == 2
        else:
            all_data.pop("ddesc")
        return all_data

    def _compute_mol_covs(
        self, ddir, mol_ids, kernel, get_orb_deriv=None, save_refs=False
    ):
        # mode should be x or c
        blksize = 10000
        if self.args is not None:
            debug_model = self.args.debug_model
            debug_spline = self.args.debug_spline
        else:
            debug_model = None
            debug_spline = None

        deriv = None
        for i, mol_id in enumerate(mol_ids):
            print("MOL ID", mol_id)
            data = self.load_data(ddir, mol_id, get_orb_deriv)
            if deriv is None:
                if get_orb_deriv is None:
                    if "ddesc" in data:
                        deriv = True
                    else:
                        deriv = False
                else:
                    deriv = get_orb_deriv
            if deriv:
                assert "ddesc" in data

            weights = data["wt"]
            # nspin = data['nspin']
            desc = data["desc"]
            vwrtt_tot = 0
            baseline = 0
            if debug_spline is not None:
                etst = 0
            if deriv:
                ddesc = strk_to_tuplek(data["ddesc"])
                self.dexx_ref_dict[mol_id] = strk_to_tuplek(data["dval"], ref=True)
                dvwrtt_tot = {k: 0 for k in ddesc.keys()}
                dbaseline = {k: 0 for k in ddesc.keys()}
            for i0, i1 in prange(0, weights.size, blksize):
                X0T = self._get_normalized_features(desc[..., i0:i1])
                wt = weights[i0:i1]
                m, dm = kernel.multiplicative_baseline(X0T)
                a, da = kernel.additive_baseline(X0T)
                if debug_spline is not None:
                    tmp, dtmp = debug_spline(X0T, rhocut=1e-7)
                    etst += np.dot(tmp, wt)
                # k is (Nctrl, nspin, Nsamp)
                # dk is (Nctrl, nspin, N0, Nsamp)
                if deriv:
                    k, dkdX0T = kernel.get_k_and_deriv(X0T)
                else:
                    k = kernel.get_k(X0T)
                    dkdX0T = None
                # TODO setting nan to zero could cover up more serious issues,
                # but it is the easiest way to take care of small density
                # training data without editing functions used at eval-time.
                if kernel.mode == "SEP":
                    cond = X0T[:, 0] < 1e-6
                    m[cond] = 0.0
                    a[cond] = 0.0
                    k[:, cond] = 0.0
                    for s in range(X0T.shape[0]):
                        dm[s][:, cond[s]] = 0.0
                        da[s][:, cond[s]] = 0.0
                        if deriv:
                            dkdX0T[:, s][:, :, cond[s]] = 0.0
                else:
                    cond = X0T[:, 0].sum(0) < 1e-6
                    m[..., cond] = 0.0
                    dm[..., cond] = 0.0
                    a[..., cond] = 0.0
                    da[..., cond] = 0.0
                    k[..., cond] = 0.0
                    if deriv:
                        print(dkdX0T.shape)
                        dkdX0T[..., cond] = 0.0
                if kernel.mode == "SEP":
                    km = (k * m).sum(1)
                    if deriv:
                        dkm = dkdX0T * m[:, None, :]
                        dkm += k[:, :, None, :] * dm
                else:
                    km = k * m
                    if deriv:
                        dkm = dkdX0T * m + dm * k[:, None, None, :]
                vwrtt_tot += (km * wt).sum(axis=1)
                baseline += (a * wt).sum()
                if deriv:
                    for orb, (s, ddesc_tmp) in ddesc.items():
                        # ddesc_tmp has shape (N0, Nsamp)
                        # avoid overwriting slice
                        ddesc_tmp = wt * self._get_normalized_feature_derivs(
                            desc[s, :, i0:i1], ddesc_tmp[:, i0:i1]
                        )
                        dvwrtt_tot[orb] += np.einsum("cdn,dn->c", dkm[:, s], ddesc_tmp)
                        dbaseline[orb] += (da[s] * ddesc_tmp).sum()
            kernel.cov_dict[mol_id] = vwrtt_tot
            kernel.base_dict[mol_id] = baseline
            if debug_model is not None:
                diff = debug_model.kernels[0].alpha.dot(vwrtt_tot)
                print("REF VWRTT", diff, kernel.base_dict[mol_id])
            if debug_spline is not None:
                print("DEBUG EN", etst)
            print("BASELINE", baseline)
            if save_refs:
                self.exx_ref_dict[mol_id] = (data["val"] * weights).sum()
                print("exx_ref", self.exx_ref_dict[mol_id])
                self.ks_baseline_dict[mol_id] = data["e_tot_orig"] - data["exc_orig"]
            if deriv:
                kernel.dcov_dict[mol_id] = dvwrtt_tot
                kernel.dbase_dict[mol_id] = dbaseline
                print("DBASELINE", dbaseline)
                if debug_model is not None:
                    diff = {
                        k: debug_model.kernels[0].alpha.dot(dvwrtt)
                        for k, dvwrtt in dvwrtt_tot.items()
                    }
                    print("REF DVWRTT", diff, kernel.dbase_dict[mol_id])

    def store_mol_covs(self, ddir, mol_ids, get_orb_deriv=None, get_correlation=True):
        """
        Reads features (and optionally their derivatives with respect to orbital
        occupations) and computes covariances with the control point sets of all
        the kernels of the models (or only the exchange components if
        get_correlation=False). Also stores the reference energy data
        in the dataset

        Args:
            ddir (str or dict[str]): If using DescParams (old version), should
                be single directory with the feature vectors. If using
                FeatureSettings (new version), should be dictionary of
                directories, with 'REF', 'SL', 'NLDF', 'NLOF', 'SDMX', and
                'HYB' being the keys. All are optional except 'SL' and 'REF'.
                'HYB' stands for reference data. It should contain
                reference energies and raw, spin-polarized
                density/gradient/kinetic energy data.
            mol_ids (list[str]): system ids to search for in ddir
            get_orb_deriv (bool): Whether to read the orbital occupation
                derivatives. If None, reads them iff they are available.
                If True, reads them and throws an error if they are not
                available. If False, does not read them.
            get_correlation (bool): Whether to compute covariance for the
                correlation kernels (True) or just the exchange kernels
                (False).
        """
        for i, kernel in enumerate(self.kernels):
            print(ddir, get_orb_deriv, get_correlation)
            save_refs = i == 0
            if isinstance(get_orb_deriv, (list, tuple)):
                deriv = get_orb_deriv[i]
            else:
                deriv = get_orb_deriv
            if get_correlation or kernel.component == "x":
                self._compute_mol_covs(
                    ddir, mol_ids, kernel, get_orb_deriv=deriv, save_refs=save_refs
                )

    def reset_reactions(self):
        self.rxn_ref_list = []
        self.rxn_noise_list = []
        for kernel in self.kernels:
            kernel.rxn_cov_list = []

    def add_reactions(self, rxn_list):
        """
        Store the reactions in rxn_list in the model. These will be used
        as training points when fit() is called. Note: For every mol_id
        present in the reactions, the corresponding system must have been
        added to the covariance dictionary via the store_mol_covs
        function.

        Args:
            rxn_list: List of tuples. In each tuple, element 0
                is an integer mode (mode: 0 (X), 1 (C), or 2 (XC)),
                and element 1 is a reaction dict. Each dict should have

                * structs: struct codes list
                  (If an element in structs is a tuple, the first
                  tuple element is a structure code, and the second
                  is an orbital code.)
                * counts: count codes list
                * energy: Energy of the reaction
                * unit: Eh per (rxn energy unit)
                * noise (optional): noise in Eh for the reaction
        """
        for mode, rxn in rxn_list:
            if mode == 1:
                raise NotImplementedError
            elif mode > 2:
                raise ValueError("Unsupported mode")
            rxn_ref = 0
            if mode == 0:
                for sysid, count in zip(rxn["structs"], rxn["counts"]):
                    if isinstance(sysid, tuple):
                        rxn_ref += count * self.dexx_ref_dict[sysid[0]][sysid[1]]
                    else:
                        rxn_ref += count * self.exx_ref_dict[sysid]
            for kernel in self.xkernels:
                rxn_cov = 0
                for sysid, count in zip(rxn["structs"], rxn["counts"]):
                    if isinstance(sysid, tuple):
                        rxn_cov += count * kernel.dcov_dict[sysid[0]][sysid[1]]
                        rxn_ref -= count * kernel.dbase_dict[sysid[0]][sysid[1]]
                    else:
                        rxn_cov += count * kernel.cov_dict[sysid]
                        rxn_ref -= count * kernel.base_dict[sysid]
                kernel.rxn_cov_list.append(rxn_cov)
            if mode == 2:
                if rxn.get("unit") is None:
                    rxn["unit"] = 0.00159360109742136  # kcal/mol per Ha
                rxn_ref += rxn["energy"] * rxn["unit"]
                for sysid, count in zip(rxn["structs"], rxn["counts"]):
                    rxn_ref -= count * self.ks_baseline_dict[sysid]
                for kernel in self.ckernels:
                    rxn_cov = 0
                    for sysid, count in zip(rxn["structs"], rxn["counts"]):
                        if isinstance(sysid, tuple):
                            rxn_cov += count * kernel.dcov_dict[sysid[0]][sysid[1]]
                            rxn_ref -= count * kernel.dbase_dict[sysid[0]][sysid[1]]
                        else:
                            rxn_cov += count * kernel.cov_dict[sysid]
                            rxn_ref -= count * kernel.base_dict[sysid]
                    kernel.rxn_cov_list.append(rxn_cov)
            else:
                for kernel in self.ckernels:
                    kernel.rxn_cov_list.append(np.zeros(kernel.Nctrl))
            self.rxn_ref_list.append(rxn_ref)
            if rxn.get("noise") is not None:
                noise = rxn["noise"]
            elif rxn.get("noise_factor") is not None:
                noise = rxn["noise_factor"] * self.default_noise
            else:
                noise = self.default_noise
            if rxn.get("noise_rel_factor") is not None:
                noise += rxn["noise_rel_factor"] * np.abs(rxn_ref)
            if rxn.get("weight") is not None:
                noise /= np.sqrt(rxn["weight"])
            self.rxn_noise_list.append(noise)


class MOLGP2(MOLGP):
    """
    Gaussian process model for the exchange-correlation functional
    or its components.
    """

    def __init__(
        self,
        kernels,
        settings,
        libxc_baseline=None,
        default_noise=0.030,
    ):
        """
        Same as MOLGP, except kernels should be of type DFTKernel2.
        This requires a new class because DFTKernel2 has a different approach
        to evaluating baseline functional contributions, so a few
        functions in MOLGP need to be modified.

        Args:
            kernels (list[DFTKernel2]): List of kernels that sum to the XC energy.
            settings (DescParams or FeatureSettings): Settings for the
                features. Specifies what the feature vector is.
            libxc_baseline (str or None): Additional baseline functional to
                evaluate using the libxc library.
            default_noise (float): Default noise hyperparameter, used if noise
                is not provided for a particular data point
        """
        super(MOLGP2, self).__init__(
            kernels,
            settings,
            libxc_baseline=libxc_baseline,
            default_noise=default_noise,
        )

    def _compute_mol_covs(
        self, ddir, mol_ids, kernel, get_orb_deriv=None, save_refs=False
    ):
        # need to have different _compute_mol_covs function to handle baselines differently.
        blksize = 10000

        deriv = None
        for mol_id in mol_ids:
            print("MOL ID", mol_id)
            data = self.load_data(ddir, mol_id, get_orb_deriv)
            if deriv is None:
                if get_orb_deriv is None:
                    if "ddesc" in data:
                        deriv = True
                    else:
                        deriv = False
                else:
                    deriv = get_orb_deriv
            if deriv:
                assert "ddesc" in data
                assert "drho_data" in data

            weights = data["wt"]
            nspin = data["nspin"]
            desc = data["desc"]
            rho_data = data["rho_data"] / nspin
            vwrtt_tot = 0
            baseline = 0
            is_mgga = self.settings.sl_settings.level
            if deriv:
                ddesc = strk_to_tuplek(data["ddesc"])
                drho_data = strk_to_tuplek(data["drho_data"])
                for orb, (s, drho) in drho_data.items():
                    drho[:] /= nspin
                self.dexx_ref_dict[mol_id] = strk_to_tuplek(data["dval"], ref=True)
                dvwrtt_tot = {k: 0 for k in ddesc.keys()}
                dbaseline = {k: 0 for k in ddesc.keys()}
            for i0, i1 in prange(0, weights.size, blksize):
                X0T = self._get_normalized_features(desc[..., i0:i1])
                rho_tuple = get_rho_tuple_with_grad_cross(
                    rho_data[..., i0:i1], is_mgga=is_mgga
                )
                wt = weights[i0:i1]
                m_res = kernel.multiplicative_baseline(rho_tuple)
                a_res = kernel.additive_baseline(rho_tuple)
                dm = vxc_tuple_to_array(rho_data[..., i0:i1], m_res[1:])
                m = m_res[0]
                if a_res is None:
                    da = np.zeros_like(dm)
                    a = np.zeros_like(m)
                else:
                    da = vxc_tuple_to_array(rho_data[..., i0:i1], a_res[1:])
                    a = a_res[0]

                # k is (Nctrl, nspin, Nsamp)
                # dk is (Nctrl, nspin, N0, Nsamp)
                if deriv:
                    k, dkdX0T = kernel.get_k_and_deriv(X0T)
                else:
                    k = kernel.get_k(X0T)
                    dkdX0T = None
                # TODO setting nan to zero could cover up more serious issues,
                # but it is the easiest way to take care of small density
                # training data without editing functions used at eval-time.
                cond = rho_tuple[0] < 1e-6
                if kernel.mode == "SEP":
                    m[cond] = 0.0
                    a[cond] = 0.0
                    k[:, cond] = 0.0
                    if deriv:
                        for s in range(X0T.shape[0]):
                            dm[s][:, cond[s]] = 0.0
                            da[s][:, cond[s]] = 0.0
                            dkdX0T[:, s][:, :, cond[s]] = 0.0
                else:
                    if cond.shape[0] == 1:
                        scond = cond[0]
                    else:
                        scond = np.logical_and(cond[0], cond[1])
                    m[..., scond] = 0.0
                    a[..., scond] = 0.0
                    k[..., scond] = 0.0
                    if deriv:
                        for s in range(X0T.shape[0]):
                            dkdX0T[:, s, cond[s], :] = 0.0
                            dm[s][:, cond[s]] = 0.0
                            da[s][:, cond[s]] = 0.0
                if kernel.mode == "SEP":
                    km = (k * m).sum(1)
                    if deriv:
                        dkm1 = dkdX0T * m[:, None, :]
                        dkm2 = k[:, :, None, :] * dm
                else:
                    km = k * m
                    if deriv:
                        dkm1 = dkdX0T * m
                        dkm2 = k[:, None, None, :] * dm
                vwrtt_tot += (km * wt).sum(axis=1)
                baseline += (a * wt).sum()
                if deriv:
                    for orb, (s, ddesc_tmp) in ddesc.items():
                        # ddesc_tmp has shape (N0, Nsamp)
                        # avoid overwriting slice
                        ddesc_tmp = wt * self._get_normalized_feature_derivs(
                            desc[s, :, i0:i1], ddesc_tmp[:, i0:i1]
                        )
                        drho_tmp = wt * drho_data[orb][:, i0:i1]
                        dvwrtt_tot[orb] += np.einsum("cdn,dn->c", dkm1[:, s], ddesc_tmp)
                        dvwrtt_tot[orb] += np.einsum("cdn,dn->c", dkm2[:, s], drho_tmp)
                        dbaseline[orb] += np.dot(da * drho_tmp, wt)
            kernel.cov_dict[mol_id] = vwrtt_tot
            kernel.base_dict[mol_id] = baseline
            if save_refs:
                self.exx_ref_dict[mol_id] = (data["val"] * weights).sum()
                self.ks_baseline_dict[mol_id] = data["e_tot_orig"] - data["exc_orig"]
            if deriv:
                kernel.dcov_dict[mol_id] = dvwrtt_tot
                kernel.dbase_dict[mol_id] = dbaseline

    def map(self, mapping_plans):
        """
        Map the MOLGP model to an Evaluator object that can efficiently
        evaluate the XC energy and which can also be serialized.

        Returns:
            list(callable): Functions that map each
                kernel to a MappedDFTKernel object
        """
        mapped_kernels = [
            kernel.map(mapping_plan)
            for kernel, mapping_plan in zip(self.kernels, mapping_plans)
        ]
        return MappedXC2(
            mapped_kernels,
            self.settings,
            libxc_baseline=self.libxc_baseline,
        )
