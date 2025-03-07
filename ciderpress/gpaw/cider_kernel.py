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
from gpaw.xc.kernel import XCKernel
from gpaw.xc.libxc import LibXC

from ciderpress.dft.plans import SemilocalPlan2
from ciderpress.dft.xc_evaluator import MappedXC
from ciderpress.dft.xc_evaluator2 import MappedXC2
from ciderpress.gpaw.config import GPAW_DEFAULT_RHO_TOL


class CiderKernel(XCKernel):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def todict(self):
        raise NotImplementedError

    def call_xc_kernel(
        self,
        e_g,
        n_sg,
        sigma_xg,
        feat_sg,
        v_sg,
        dedsigma_xg,
        vfeat_sg,
        tau_sg=None,
        dedtau_sg=None,
    ):
        # make view so we can reshape things
        e_g = e_g.view()
        n_sg = n_sg.view()
        v_sg = v_sg.view()
        feat_sg = feat_sg.view()
        vfeat_sg = vfeat_sg.view()
        if tau_sg is not None:
            tau_sg = tau_sg.view()
            dedtau_sg = dedtau_sg.view()

        nspin = n_sg.shape[0]
        ngrids = n_sg.size // nspin
        nfeat = self.mlfunc.settings.nfeat
        X0T = np.empty((nspin, nfeat, ngrids))
        has_sl = True
        has_nldf = not self.mlfunc.settings.nldf_settings.is_empty
        start = 0
        sigma_sg = sigma_xg[::2]
        new_shape = (nspin, ngrids)
        e_g.shape = (ngrids,)
        n_sg.shape = new_shape
        v_sg.shape = new_shape
        sigma_sg.shape = new_shape
        feat_sg.shape = (nspin, -1, ngrids)
        vfeat_sg.shape = (nspin, -1, ngrids)
        if has_sl:
            sl_plan = SemilocalPlan2(self.mlfunc.settings.sl_settings, nspin)
        else:
            sl_plan = None
        if tau_sg is not None:
            tau_sg.shape = new_shape
            dedtau_sg.shape = new_shape
        if has_sl:
            nfeat_tmp = self.mlfunc.settings.sl_settings.nfeat
            X0T[:, start : start + nfeat_tmp] = sl_plan.get_feat(
                n_sg, sigma_sg, tau=tau_sg
            )
            start += nfeat_tmp
        if has_nldf:
            nfeat_tmp = self.mlfunc.settings.nldf_settings.nfeat
            # X0T[:, start : start + nfeat_tmp] = nspin * feat_sg
            X0T[:, start : start + nfeat_tmp] = feat_sg
            start += nfeat_tmp
        X0TN = self.mlfunc.settings.normalizers.get_normalized_feature_vector(X0T)
        xmix = self.xmix
        if isinstance(self.mlfunc, MappedXC):
            exc_ml, dexcdX0TN_ml = self.mlfunc(X0TN, rhocut=self.rhocut)
        elif isinstance(self.mlfunc, MappedXC2):
            if tau_sg is None:
                rho_tuple = (n_sg.view(), sigma_xg.view())
            else:
                rho_tuple = (n_sg.view(), sigma_xg.view(), tau_sg.view())
            shapes_orig = []
            for arr in rho_tuple:
                shapes_orig.append(arr.shape)
                arr.shape = (arr.shape[0], -1)
            exc_ml, dexcdX0TN_ml, vrho_tuple = self.mlfunc(
                X0TN, rho_tuple, rhocut=self.rhocut
            )
            for iarr, arr in enumerate(vrho_tuple):
                arr.shape = shapes_orig[iarr]
            v_sg[:] += xmix * vrho_tuple[0]
            dedsigma_xg[:] += xmix * vrho_tuple[1]
            if tau_sg is not None:
                dedtau_sg[:] += xmix * vrho_tuple[2]
        else:
            raise TypeError("mlfunc must be MappedXC or MappedXC2")
        exc_ml *= xmix
        dexcdX0TN_ml *= xmix
        vxc_ml = self.mlfunc.settings.normalizers.get_derivative_wrt_unnormed_features(
            X0T, dexcdX0TN_ml
        )
        # vxc_ml = dexcdX0TN_ml
        e_g[:] += exc_ml

        start = 0
        dedsigma_sg = dedsigma_xg[::2].copy()
        dedsigma_sg.shape = new_shape
        if has_sl:
            nfeat_tmp = self.mlfunc.settings.sl_settings.nfeat
            sl_plan.get_vxc(
                vxc_ml[:, start : start + nfeat_tmp],
                n_sg,
                v_sg,
                sigma_sg,
                dedsigma_sg,
                tau=tau_sg,
                vtau=dedtau_sg,
            )
            start += nfeat_tmp
        if has_nldf:
            nfeat_tmp = self.mlfunc.settings.nldf_settings.nfeat
            # vfeat_sg[:] += nspin * vxc_ml[:, start : start + nfeat_tmp]
            vfeat_sg[:] += vxc_ml[:, start : start + nfeat_tmp]
            start += nfeat_tmp
        dedsigma_xg[::2] = dedsigma_sg.reshape(nspin, *dedsigma_xg.shape[1:])

    def calculate(self, *args):
        raise NotImplementedError


class CiderGGAHybridKernel(CiderKernel):
    def __init__(self, mlfunc, xmix, xstr, cstr, rhocut=GPAW_DEFAULT_RHO_TOL):
        self.type = "GGA"
        self.name = "CiderGGA"
        self.xkernel = None if xstr is None else LibXC(xstr)
        self.ckernel = None if cstr is None else LibXC(cstr)
        self.xmix = xmix
        self.mlfunc = mlfunc
        self.rhocut = rhocut

    def todict(self):
        return {
            "xmix": self.xmix,
            "xstr": None if self.xkernel is None else self.xkernel.name,
            "cstr": None if self.ckernel is None else self.ckernel.name,
        }

    def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg, feat_sg):
        """
        Evaluate CIDER 'hybrid' functional.
        """
        n_sg.shape[0]
        if self.xkernel is not None:
            self.xkernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
            e_g[:] *= 1 - self.xmix
            v_sg[:] *= 1 - self.xmix
            dedsigma_xg[:] *= 1 - self.xmix
        else:
            e_g[:] = 0.0
            v_sg[:] = 0.0
            dedsigma_xg[:] = 0.0
        if self.ckernel is not None:
            e_g_tmp, v_sg_tmp, dedsigma_xg_tmp = (
                np.zeros_like(e_g),
                np.zeros_like(v_sg),
                np.zeros_like(dedsigma_xg),
            )
            self.ckernel.calculate(e_g_tmp, n_sg, v_sg_tmp, sigma_xg, dedsigma_xg_tmp)
            e_g[:] += e_g_tmp
            v_sg[:] += v_sg_tmp
            dedsigma_xg[:] += dedsigma_xg_tmp
        vfeat_sg = np.zeros_like(feat_sg)
        self.call_xc_kernel(
            e_g,
            n_sg,
            sigma_xg,
            feat_sg,
            v_sg,
            dedsigma_xg,
            vfeat_sg,
        )
        return vfeat_sg


class CiderMGGAHybridKernel(CiderGGAHybridKernel):
    def __init__(self, mlfunc, xmix, xstr, cstr, rhocut=GPAW_DEFAULT_RHO_TOL):
        self.type = "MGGA"
        self.name = "CiderMGGA"
        self.xkernel = None if xstr is None else LibXC(xstr)
        self.ckernel = None if cstr is None else LibXC(cstr)
        self.xmix = xmix
        self.mlfunc = mlfunc
        self.rhocut = rhocut

    def calculate(
        self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg, tau_sg, dedtau_sg, feat_sg
    ):
        """
        Evaluate CIDER 'hybrid' functional.
        """
        n_sg.shape[0]
        if self.xkernel is None:
            e_g[:] = 0.0
            v_sg[:] = 0.0
            dedsigma_xg[:] = 0.0
            dedtau_sg[:] = 0.0
        elif self.xkernel.type == "GGA":
            self.xkernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
            dedtau_sg[:] = 0
        else:
            self.xkernel.calculate(
                e_g, n_sg, v_sg, sigma_xg, dedsigma_xg, tau_sg, dedtau_sg
            )
        if self.xkernel is not None:
            e_g[:] *= 1 - self.xmix
            v_sg[:] *= 1 - self.xmix
            dedsigma_xg[:] *= 1 - self.xmix
            dedtau_sg[:] *= 1 - self.xmix
        if self.ckernel is not None:
            e_g_tmp, v_sg_tmp, dedsigma_xg_tmp, dedtau_sg_tmp = (
                np.zeros_like(e_g),
                np.zeros_like(v_sg),
                np.zeros_like(dedsigma_xg),
                np.zeros_like(dedtau_sg),
            )
            if self.ckernel.type == "GGA":
                self.ckernel.calculate(
                    e_g_tmp, n_sg, v_sg_tmp, sigma_xg, dedsigma_xg_tmp
                )
            else:
                self.ckernel.calculate(
                    e_g_tmp,
                    n_sg,
                    v_sg_tmp,
                    sigma_xg,
                    dedsigma_xg_tmp,
                    tau_sg,
                    dedtau_sg_tmp,
                )
            e_g[:] += e_g_tmp
            v_sg[:] += v_sg_tmp
            dedsigma_xg[:] += dedsigma_xg_tmp
            dedtau_sg[:] += dedtau_sg_tmp
        vfeat_sg = np.zeros_like(feat_sg)
        self.call_xc_kernel(
            e_g,
            n_sg,
            sigma_xg,
            feat_sg,
            v_sg,
            dedsigma_xg,
            vfeat_sg,
            tau_sg=tau_sg,
            dedtau_sg=dedtau_sg,
        )
        return vfeat_sg
