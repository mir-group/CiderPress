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

import numpy as np

from ciderpress.dft.lcao_interpolation import LCAOInterpolatorDirect


class LCAONLDFGenerator:
    def __init__(self, plan, ccl, interpolator, grids_indexer):
        """

        Args:
            plan (NLDFAuxiliaryPlan):
            ccl (ConvolutionCollection):
            interpolator (LCAOInterpolator):
            grids_indexer (AtomicGridsIndexer):
        """
        self.plan = plan
        self.ccl = ccl
        self.interpolator = interpolator
        self.grids_indexer = grids_indexer
        self._cache = {s: None for s in range(plan.nspin)}
        nalpha = self.plan.nalpha
        self._uq_buf = np.empty((self.ccl.atco_inp.nao, nalpha))
        self._vq_buf = np.empty((self.ccl.atco_out.nao, self.ccl.num_out))
        self._rlmq_buf = self.grids_indexer.empty_rlmq(nalpha)

    @property
    def natm(self):
        return self.interpolator.atom_coords.shape[0]

    def _perform_fwd_convolution(self, theta_gq, grad_mode=False):
        theta_rlmq = self._rlmq_buf
        theta_uq = self._uq_buf
        conv_vq = self._vq_buf
        conv_vq[:] = 0.0
        theta_uq[:] = 0.0
        theta_rlmq[:] = 0.0
        self.grids_indexer.reduce_angc_ylm_(theta_rlmq, theta_gq, a2y=True)
        self.ccl.atco_inp.convert_rad2orb_(
            theta_rlmq,
            theta_uq,
            self.grids_indexer,
            self.grids_indexer.rad_arr,
            rad2orb=True,
            offset=0,
        )
        self.plan.get_transformed_interpolation_terms(
            theta_uq, i=-1, fwd=True, inplace=True
        )
        self.ccl.multiply_atc_integrals(theta_uq, output=conv_vq, fwd=True)
        if self.plan.nldf_settings.nldf_type != "i":
            self.plan.get_transformed_interpolation_terms(
                conv_vq[:, : self.plan.nalpha], i=0, fwd=False, inplace=True
            )
        if grad_mode:
            return self.interpolator.project_orb2grid(conv_vq), conv_vq.copy()
        else:
            return self.interpolator.project_orb2grid(conv_vq)

    def _perform_bwd_convolution(self, vf_gq):
        vtheta_rlmq = self._rlmq_buf
        vtheta_uq = self._uq_buf
        vconv_vq = self._vq_buf
        vconv_vq[:] = 0.0
        vtheta_uq[:] = 0.0
        self.interpolator.project_grid2orb(vf_gq, f_uq=vconv_vq)
        if self.plan.nldf_settings.nldf_type != "i":
            self.plan.get_transformed_interpolation_terms(
                vconv_vq[:, : self.plan.nalpha], i=0, fwd=True, inplace=True
            )
        self.ccl.multiply_atc_integrals(vconv_vq, output=vtheta_uq, fwd=False)
        self.plan.get_transformed_interpolation_terms(
            vtheta_uq, i=-1, fwd=False, inplace=True
        )
        self.ccl.atco_inp.convert_rad2orb_(
            vtheta_rlmq,
            vtheta_uq,
            self.grids_indexer,
            self.grids_indexer.rad_arr,
            rad2orb=False,
            offset=0,
        )
        vtheta_gq = self.grids_indexer.empty_gq(self.plan.nalpha)
        self.grids_indexer.reduce_angc_ylm_(vtheta_rlmq, vtheta_gq, a2y=False)
        return vtheta_gq

    def get_features(self, rho_in, spin=0, map_grids=True, grad_mode=False):
        # set up arrays
        self._cache[spin] = None
        ngrids_ato = self.grids_indexer.ngrids
        idx_map = self.grids_indexer.idx_map
        ngrids = self.grids_indexer.idx_map.size

        rho_tuple = self.plan.get_rho_tuple(rho_in)
        arg_in_g = self.plan.get_interpolation_arguments(rho_tuple, i=-1)
        func_in_g = self.plan.get_function_to_convolve(rho_tuple)
        arg_in_g, darg_in_g = arg_in_g[0], arg_in_g[1]
        func_in_g, dfunc_in_g = func_in_g[0], func_in_g[1]
        if map_grids:
            arg_g = np.zeros(ngrids_ato)
            func_g = np.zeros(ngrids_ato)
            arg_g[idx_map] = arg_in_g[:ngrids]
            func_g[idx_map] = func_in_g[:ngrids]
        else:
            arg_g = arg_in_g
            func_g = func_in_g
        p_gq, dp_gq = self.plan.get_interpolation_coefficients(arg_g, i=-1)
        theta_gq = p_gq * (func_g * self.grids_indexer.all_weights)[:, None]

        # compute feature convolutions
        f_gq = self._perform_fwd_convolution(theta_gq, grad_mode=grad_mode)
        if grad_mode:
            f_gq, conv_vq = f_gq

        if map_grids:
            rho_out = rho_in
        else:
            rho_out = np.zeros((rho_in.shape[0], ngrids + self.grids_indexer.padding))
            rho_out[:, :ngrids] = rho_in[:, idx_map]
        # do contractions to get features
        feat, dfeat = self.plan.eval_rho_full(
            f_gq,
            rho_out,
            apply_transformation=False,
            spin=spin,
        )
        self._cache[spin] = {
            "rho_in": rho_in,
            "rho_out": rho_out,
            "dfeat": dfeat,
            "p_gq": p_gq,
            "dp_gq": dp_gq,
            "darg_in_g": darg_in_g,
            "func_g": func_g,
            "dfunc_in_g": dfunc_in_g,
        }
        if grad_mode:
            self._cache[spin]["conv_vq"] = conv_vq
        return feat

    def get_features_and_occ_derivs(
        self, rho_in, orb_rho_in_iorb, rho_out=None, orb_rho_out_iorb=None
    ):
        if isinstance(self.interpolator, LCAOInterpolatorDirect):
            raise ValueError("Direct interpolator does not support this function.")
        if rho_out is None:
            assert orb_rho_out_iorb is None
            rho_out = rho_in
            orb_rho_out_iorb = orb_rho_in_iorb
        # set up arrays
        ngrids_ato = self.grids_indexer.ngrids
        idx_map = self.grids_indexer.idx_map
        ngrids = self.grids_indexer.idx_map.size

        rho_tuple = self.plan.get_rho_tuple(rho_in)
        arg_in_g = self.plan.get_interpolation_arguments(rho_tuple, i=-1)
        func_in_g = self.plan.get_function_to_convolve(rho_tuple)
        arg_in_g, darg_in_g = arg_in_g[0], arg_in_g[1]
        func_in_g, dfunc_in_g = func_in_g[0], func_in_g[1]
        arg_g = np.zeros(ngrids_ato)
        func_g = np.zeros(ngrids_ato)
        arg_g[idx_map] = arg_in_g[:ngrids]
        func_g[idx_map] = func_in_g[:ngrids]
        p_gq, dp_gq = self.plan.get_interpolation_coefficients(arg_g, i=-1)
        theta_gq = p_gq * (func_g * self.grids_indexer.all_weights)[:, None]

        # compute feature convolutions
        f_gq = self._perform_fwd_convolution(theta_gq)
        start = 0 if self.plan.nldf_settings.nldf_type == "i" else self.plan.nalpha
        start += len(self.plan.nldf_settings.l0_feat_specs)
        feat = self.plan.eval_rho_full(f_gq, rho_out, apply_transformation=False)[0]
        occd_feats = []

        for iorb in range(orb_rho_in_iorb.shape[0]):
            orb_rho_in = orb_rho_in_iorb[iorb]
            orb_rho_out = orb_rho_out_iorb[iorb]
            occd_sigma_in = 2 * np.einsum("xg,xg", rho_in[1:4], orb_rho_in[1:4])
            occd_arg_in_g = orb_rho_in[0] * darg_in_g[0] + occd_sigma_in * darg_in_g[1]
            occd_func_in_g = (
                orb_rho_in[0] * dfunc_in_g[0] + occd_sigma_in * dfunc_in_g[1]
            )
            if self.plan.nldf_settings.sl_level == "MGGA":
                occd_arg_in_g[:] += orb_rho_in[4] * darg_in_g[2]
                occd_func_in_g[:] += orb_rho_in[4] * dfunc_in_g[2]
            occd_arg_g = np.zeros(ngrids_ato)
            occd_func_g = np.zeros(ngrids_ato)
            occd_arg_g[idx_map] = occd_arg_in_g[:ngrids]
            occd_func_g[idx_map] = occd_func_in_g[:ngrids]
            occd_theta_gq = (
                dp_gq * (func_g * occd_arg_g)[:, None] + p_gq * occd_func_g[:, None]
            )
            occd_theta_gq[:] *= self.grids_indexer.all_weights[:, None]
            occd_f_gq = self._perform_fwd_convolution(occd_theta_gq)
            occd_feat = self.plan.eval_occd_full(
                f_gq,
                rho_out,
                occd_f_gq,
                orb_rho_out,
                apply_transformation=False,
            )
            occd_feats.append(occd_feat)
        return feat, None if len(occd_feats) == 0 else np.stack(occd_feats)

    def get_potential(self, vfeat, spin=0, map_grids=True, grad_mode=False):
        # set up arrays
        if self._cache[spin] is None:
            raise ValueError("Need to call get_features before get_potential")
        cache = self._cache[spin]
        idx_map = self.grids_indexer.idx_map
        ngrids = self.grids_indexer.idx_map.size
        vrho_out = np.zeros_like(cache["rho_out"])
        ngrid_out = vfeat.shape[1]

        # compute feature potential
        vf_gq = np.zeros((vfeat.shape[1], self.interpolator.num_out))
        vf_gq = self.plan.eval_vxc_full(
            vfeat,
            vrho_out,
            cache["dfeat"],
            cache["rho_out"],
            vf=vf_gq,
            spin=spin,
        )
        if map_grids:
            pass
        else:
            vrho_out_tmp = vrho_out
            vrho_out = np.zeros(cache["rho_in"].shape)
            vrho_out[:, idx_map] = vrho_out_tmp[:, :ngrids]
        if grad_mode:
            excsum = self.interpolator.project_orb2grid_grad(cache["conv_vq"], vf_gq)
        vtheta_gq = self._perform_bwd_convolution(vf_gq)
        varg_g = np.einsum("gq,gq->g", cache["dp_gq"], vtheta_gq) * cache["func_g"]
        vfunc_g = np.einsum("gq,gq->g", cache["p_gq"], vtheta_gq)
        varg_g[:] *= self.grids_indexer.all_weights
        if grad_mode:
            cidergg_g = vfunc_g * cache["func_g"]
        vfunc_g[:] *= self.grids_indexer.all_weights

        if map_grids:
            varg_out_g = np.zeros(ngrid_out)
            varg_out_g[:ngrids] = varg_g[idx_map]
            vfunc_out_g = np.zeros(ngrid_out)
            vfunc_out_g[:ngrids] = vfunc_g[idx_map]
        else:
            varg_out_g = varg_g
            vfunc_out_g = vfunc_g
        vrho_out[0] += (
            varg_out_g * cache["darg_in_g"][0] + vfunc_out_g * cache["dfunc_in_g"][0]
        )
        vrho_out[1:4] += (
            (varg_out_g * cache["darg_in_g"][1] + vfunc_out_g * cache["dfunc_in_g"][1])
            * 2
            * cache["rho_in"][1:4]
        )
        if self.plan.nldf_settings.sl_level == "MGGA":
            vrho_out[4] += (
                varg_out_g * cache["darg_in_g"][2]
                + vfunc_out_g * cache["dfunc_in_g"][2]
            )
        # self._cache[spin] = None
        if grad_mode:
            if map_grids:
                raise NotImplementedError("map_grids and grad mode at same time")
            return vrho_out, cidergg_g, excsum
        else:
            return vrho_out
