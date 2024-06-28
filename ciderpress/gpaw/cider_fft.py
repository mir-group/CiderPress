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


import gpaw.fftw as fftw
import gpaw.mpi as mpi
import joblib
import numpy as np
import yaml
from gpaw.grid_descriptor import GridDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.xc.gga import GGA, add_gradient_correction, gga_vars
from gpaw.xc.kernel import XCKernel
from gpaw.xc.libxc import LibXC
from gpaw.xc.mgga import MGGA

from ciderpress.dft import pwutil
from ciderpress.dft.plans import NLDFSplinePlan, SemilocalPlan2
from ciderpress.gpaw.mp_cider import CiderParallelization

DEFAULT_RHO_TOL = 1e-8


class LDict(dict):
    def __init__(self, *args, **kwargs):
        super(LDict, self).__init__(*args, **kwargs)
        self._locked = False

    def lock(self):
        self._locked = True

    def unlock(self):
        self._locked = False

    def __setitem__(self, k, v):
        if self._locked:
            raise RuntimeError("Cannot set item in locked LDict")
        super(LDict, self).__setitem__(k, v)


class CiderPWDescriptor(PWDescriptor):
    def __init__(
        self, ecut, gd, dtype=None, kd=None, fftwflags=fftw.MEASURE, gammacentered=False
    ):
        """
        Same initializer as in GPAW source code EXCEPT don't
        raise an assertion error if ecut is too big (since one
        might want to include all plane-waves as opposed to just
        the ones in a sphere).
        """

        assert gd.pbc_c.all()
        pi = np.pi

        self.gd = gd
        self.fftwflags = fftwflags

        N_c = gd.N_c
        self.comm = gd.comm

        ecut0 = 0.5 * pi**2 / (self.gd.h_cv**2).sum(1).max()
        if ecut is None:
            ecut = 0.9999 * ecut0
        else:
            pass  # Can set huge encut and it's fine.

        self.ecut = ecut

        if dtype is None:
            if kd is None or kd.gamma:
                dtype = float
            else:
                dtype = complex
        self.dtype = dtype
        self.gammacentered = gammacentered

        if dtype == float:
            Nr_c = N_c.copy()
            Nr_c[2] = N_c[2] // 2 + 1
            i_Qc = np.indices(Nr_c).transpose((1, 2, 3, 0))
            i_Qc[..., :2] += N_c[:2] // 2
            i_Qc[..., :2] %= N_c[:2]
            i_Qc[..., :2] -= N_c[:2] // 2
            self.tmp_Q = fftw.empty(Nr_c, complex)
            self.tmp_R = self.tmp_Q.view(float)[:, :, : N_c[2]]
        else:
            i_Qc = np.indices(N_c).transpose((1, 2, 3, 0))
            i_Qc += N_c // 2
            i_Qc %= N_c
            i_Qc -= N_c // 2
            self.tmp_Q = fftw.empty(N_c, complex)
            self.tmp_R = self.tmp_Q

        self.fftplan = fftw.create_plan(self.tmp_R, self.tmp_Q, -1, fftwflags)
        self.ifftplan = fftw.create_plan(self.tmp_Q, self.tmp_R, 1, fftwflags)

        # Calculate reciprocal lattice vectors:
        B_cv = 2.0 * pi * gd.icell_cv
        i_Qc.shape = (-1, 3)
        self.G_Qv = np.dot(i_Qc, B_cv)

        self.kd = kd
        if kd is None:
            self.K_qv = np.zeros((1, 3))
            self.only_one_k_point = True
        else:
            self.K_qv = np.dot(kd.ibzk_qc, B_cv)
            self.only_one_k_point = kd.nbzkpts == 1

        # Map from vectors inside sphere to fft grid:
        self.Q_qG = []
        G2_qG = []
        Q_Q = np.arange(len(i_Qc), dtype=np.int32)

        self.ng_q = []
        for q, K_v in enumerate(self.K_qv):
            G2_Q = ((self.G_Qv + K_v) ** 2).sum(axis=1)
            if gammacentered:
                mask_Q = (self.G_Qv**2).sum(axis=1) <= 2 * ecut
            else:
                mask_Q = G2_Q <= 2 * ecut

            if self.dtype == float:
                mask_Q &= (
                    (i_Qc[:, 2] > 0)
                    | (i_Qc[:, 1] > 0)
                    | ((i_Qc[:, 0] >= 0) & (i_Qc[:, 1] == 0))
                )
            Q_G = Q_Q[mask_Q]
            self.Q_qG.append(Q_G)
            G2_qG.append(G2_Q[Q_G])
            ng = len(Q_G)
            self.ng_q.append(ng)

        self.ngmin = min(self.ng_q)
        self.ngmax = max(self.ng_q)

        if kd is not None:
            self.ngmin = kd.comm.min(self.ngmin)
            self.ngmax = kd.comm.max(self.ngmax)

        # Distribute things:
        S = gd.comm.size
        self.maxmyng = (self.ngmax + S - 1) // S
        ng1 = gd.comm.rank * self.maxmyng
        ng2 = ng1 + self.maxmyng

        self.G2_qG = []
        self.myQ_qG = []
        self.myng_q = []
        for q, G2_G in enumerate(G2_qG):
            G2_G = G2_G[ng1:ng2].copy()
            G2_G.flags.writeable = False
            self.G2_qG.append(G2_G)
            myQ_G = self.Q_qG[q][ng1:ng2]
            self.myQ_qG.append(myQ_G)
            self.myng_q.append(len(myQ_G))

        if S > 1:
            self.tmp_G = np.empty(self.maxmyng * S, complex)
        else:
            self.tmp_G = None


class FTGradient:
    def __init__(self, gd, v, scale=1.0, dtype=float):
        self.pd = CiderPWDescriptor(1e100, gd, dtype=dtype)
        self.Ga_G = np.ascontiguousarray(self.pd.get_reciprocal_vectors(q=0)[:, v])
        if scale != 1.0:
            self.Ga_G *= scale

    def apply(self, in_xg, out_xg):
        a_G = self.pd.fft(in_xg)
        a_G *= 1j * self.Ga_G
        out_xg[:] = self.pd.ifft(a_G)


def get_gradient_ops(gd):
    return [FTGradient(gd, v).apply for v in range(3)]


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
        dedsigma_sg = dedsigma_xg[::2].copy()
        new_shape = (nspin, ngrids)
        e_g.shape = (ngrids,)
        n_sg.shape = new_shape
        v_sg.shape = new_shape
        sigma_sg.shape = new_shape
        dedsigma_sg.shape = new_shape
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
            X0T[:, start : start + nfeat_tmp] = nspin * feat_sg
            start += nfeat_tmp

        X0TN = self.mlfunc.settings.normalizers.get_normalized_feature_vector(X0T)
        exc_ml, dexcdX0TN_ml = self.mlfunc(X0TN, rhocut=self.rhocut)
        xmix = self.xmix  # / rho.shape[0]
        exc_ml *= xmix
        dexcdX0TN_ml *= xmix
        vxc_ml = self.mlfunc.settings.normalizers.get_derivative_wrt_unnormed_features(
            X0T, dexcdX0TN_ml
        )
        # vxc_ml = dexcdX0TN_ml
        e_g[:] += exc_ml

        start = 0
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
            vfeat_sg[:] += nspin * vxc_ml[:, start : start + nfeat_tmp]
            start += nfeat_tmp
        dedsigma_xg[::2] = dedsigma_sg.reshape(nspin, *dedsigma_xg.shape[1:])

    def calculate(self, *args):
        raise NotImplementedError


class CiderGGAHybridKernel(CiderKernel):
    def __init__(self, mlfunc, xmix, xstr, cstr, rhocut=DEFAULT_RHO_TOL):
        self.type = "GGA"
        self.name = "CiderGGA"
        self.xkernel = LibXC(xstr)
        self.ckernel = LibXC(cstr)
        self.xmix = xmix
        self.mlfunc = mlfunc
        self.rhocut = rhocut

    def todict(self):
        return {
            "xmix": self.xmix,
            "xstr": self.xkernel.name,
            "cstr": self.ckernel.name,
        }

    def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg, feat_sg):
        """
        Evaluate CIDER 'hybrid' functional.
        """
        n_sg.shape[0]
        self.xkernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        e_g[:] *= 1 - self.xmix
        v_sg[:] *= 1 - self.xmix
        dedsigma_xg[:] *= 1 - self.xmix
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


class CiderGGAHybridKernelPBEPot(CiderGGAHybridKernel):
    # debug kernel that uses PBE potential and CIDER energy
    def calculate(self, e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg, feat):
        """
        Evaluate CIDER 'hybrid' functional.
        """
        nt_sg.shape[0]
        self.xkernel.calculate(e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg)
        e_g[:] *= 1 - self.xmix
        e_g_tmp, v_sg_tmp, dedsigma_xg_tmp = (
            np.zeros_like(e_g),
            np.zeros_like(v_sg),
            np.zeros_like(dedsigma_xg),
        )
        self.ckernel.calculate(e_g_tmp, nt_sg, v_sg_tmp, sigma_xg, dedsigma_xg_tmp)
        e_g[:] += e_g_tmp
        v_sg[:] += v_sg_tmp
        dedsigma_xg[:] += dedsigma_xg_tmp
        vfeat = np.zeros_like(feat)
        self.call_xc_kernel(
            e_g,
            nt_sg,
            sigma_xg,
            feat,
            np.zeros_like(v_sg),
            np.zeros_like(dedsigma_xg),
            vfeat,
        )
        return np.zeros_like(vfeat)


class CiderMGGAHybridKernel(CiderGGAHybridKernel):
    def __init__(self, mlfunc, xmix, xstr, cstr, rhocut=DEFAULT_RHO_TOL):
        self.type = "MGGA"
        self.name = "CiderMGGA"
        self.xkernel = LibXC(xstr)
        self.ckernel = LibXC(cstr)
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
        if self.xkernel.type == "GGA":
            self.xkernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
            dedtau_sg[:] = 0
        else:
            self.xkernel.calculate(
                e_g, n_sg, v_sg, sigma_xg, dedsigma_xg, tau_sg, dedtau_sg
            )
        e_g[:] *= 1 - self.xmix
        v_sg[:] *= 1 - self.xmix
        dedsigma_xg[:] *= 1 - self.xmix
        dedtau_sg[:] *= 1 - self.xmix
        e_g_tmp, v_sg_tmp, dedsigma_xg_tmp, dedtau_sg_tmp = (
            np.zeros_like(e_g),
            np.zeros_like(v_sg),
            np.zeros_like(dedsigma_xg),
            np.zeros_like(dedtau_sg),
        )
        if self.ckernel.type == "GGA":
            self.ckernel.calculate(e_g_tmp, n_sg, v_sg_tmp, sigma_xg, dedsigma_xg_tmp)
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


def get_cider_exp_gga(
    self, rho, sigma, a0=1.0, fac_mul=0.03125, amin=0.0625, return_derivs=False
):
    sr = np.sign(rho)
    rho = np.abs(rho)
    CFC = 0.3 * (3 * np.pi**2) ** (2.0 / 3)
    fac = fac_mul * 1.2 * (6 * np.pi**2) ** (2.0 / 3) / np.pi
    if self.nspin == 1:
        B = np.pi / 2 ** (2.0 / 3) * a0
    else:
        B = np.pi * a0
    C = np.pi / 2 ** (2.0 / 3) * fac / CFC
    rhod = rho + 1e-8
    tau = sigma / (8 * rhod)
    ascale = B * rho ** (2.0 / 3) + C * tau / rhod
    cond0 = ascale < amin
    ascale[cond0] = amin * np.exp(ascale[cond0] / amin - 1)
    cond1 = np.logical_or(ascale > self.maxen, np.isnan(ascale))
    cond2 = ascale < self.minen
    ascale[cond1] = self.maxen
    ascale[cond2] = self.minen
    if return_derivs:
        # note factor of 2 on second rho_deriv term
        # due to 1/rho^2 dependence
        cond3 = np.logical_or(cond1, cond2)
        dadrho = sr * (
            2 * B / (3 * rhod ** (1.0 / 3)) - 2 * sr * (C * tau / (rhod * rhod + 1e-10))
        )
        dadsigma = C / (8 * rhod * rhod + 1e-10)
        dadrho[cond0] *= ascale[cond0] * np.exp(ascale[cond0] / amin - 1)
        dadsigma[cond0] *= ascale[cond0] * np.exp(ascale[cond0] / amin - 1)
        dadrho[cond3] = 0
        dadsigma[cond3] = 0
        return ascale, dadrho, dadsigma
    else:
        return ascale


def get_cider_exp_mgga(
    self, rho, sigma, tau, a0=1.0, fac_mul=0.03125, amin=0.0625, return_derivs=False
):
    CFC = 0.3 * (3 * np.pi**2) ** (2.0 / 3)
    fac = fac_mul * 1.2 * (6 * np.pi**2) ** (2.0 / 3) / np.pi
    if self.nspin == 1:
        B = np.pi / 2 ** (2.0 / 3) * (a0 - fac)
    else:
        B = np.pi * (a0 - fac)
    C = np.pi / 2 ** (2.0 / 3) * fac / CFC
    ascale = B * rho ** (2.0 / 3) + C * tau / (rho + 1e-10)
    cond0 = ascale < amin
    ascale[cond0] = amin * np.exp(ascale[cond0] / amin - 1)
    cond1 = np.logical_or(ascale > self.maxen, np.isnan(ascale))
    cond2 = ascale < self.minen
    ascale[cond1] = self.maxen
    ascale[cond2] = self.minen
    if return_derivs:
        cond3 = np.logical_or(cond1, cond2)
        dadrho = 2 * B / (3 * rho ** (1.0 / 3) + 1e-10) - (
            C * tau / (rho * rho + 1e-10)
        )
        dadsigma = np.zeros_like(ascale)
        dadtau = C / (rho + 1e-10)

        dadrho[cond0] *= ascale[cond0] * np.exp(ascale[cond0] / amin - 1)
        dadtau[cond0] *= ascale[cond0] * np.exp(ascale[cond0] / amin - 1)
        dadrho[cond3] = 0
        dadsigma[cond3] = 0

        return ascale, dadrho, dadsigma, dadtau
    else:
        return ascale


class _CiderBase:

    has_paw = False

    RHO_TOL = DEFAULT_RHO_TOL

    def __init__(
        self,
        cider_kernel,
        nexp,
        consts,
        Nalpha=None,
        lambd=2.0,
        encut=80,
        world=None,
        **kwargs
    ):
        if world is None:
            self.world = mpi.world
        else:
            self.world = world

        self.cider_kernel = cider_kernel
        if Nalpha is None:
            amax = encut
            amin = np.min(consts[:, -1]) / np.e
            Nalpha = int(np.ceil(np.log(amax / amin) / np.log(lambd))) + 1
            lambd = np.exp(np.log(amax / amin) / (Nalpha - 1))
        # TODO remove
        self.Nalpha = Nalpha
        self.lambd = lambd
        self.consts = consts
        self.nexp = nexp
        self.encut = encut

        self.C_aip = None
        # TODO end remove
        self.phi_aajp = None
        self.verbose = False
        self.size = None
        self._plan = None

        self.setup_name = "PBE"
        self.kk = None

    def todict(self):
        kernel_params = self.cider_kernel.todict()
        xc_params = {
            "lambd": self.lambd,
            "encut": self.encut,
        }
        return {
            "kernel_params": kernel_params,
            "xc_params": xc_params,
        }

    def get_mlfunc_data(self):
        return yaml.dump(self.cider_kernel.mlfunc, Dumper=yaml.CDumper)

    def get_setup_name(self):
        return self.setup_name

    def alpha_ind(self, a):
        return a * self.a_comm.size // self.Nalpha

    def initialize(self, density, hamiltonian, wfs):
        if self.type == "GGA":
            GGA.initialize(self, density, hamiltonian, wfs)
        elif self.type == "MGGA":
            MGGA.initialize(self, density, hamiltonian, wfs)
        else:
            raise NotImplementedError
        self.timer = wfs.timer
        self.world = wfs.world
        self.get_alphas()
        self.dens = density
        self.rbuf_ag = None
        self.kbuf_ak = None
        self.theta_ak = None

    def set_grid_descriptor(self, gd):
        # XCFunctional.set_grid_descriptor(self, gd)
        # self.grad_v = get_gradient_ops(gd)
        if self.size is None:
            self.shape = gd.N_c.copy()
            for c, n in enumerate(self.shape):
                if not gd.pbc_c[c]:
                    # self.shape[c] = get_efficient_fft_size(n)
                    self.shape[c] = int(2 ** np.ceil(np.log(n) / np.log(2)))
        else:
            self.shape = np.array(self.size)
            for c, n in enumerate(self.shape):
                if gd.pbc_c[c]:
                    assert n == gd.N_c[c]
                else:
                    assert n >= gd.N_c[c]
        self.gd = gd

    def get_alphas(self):
        self.par_cider = CiderParallelization(self.world, self.Nalpha)
        self.comms = self.par_cider.build_communicators()

        self.a_comm = self.comms["a"]

        if self.a_comm is not None:
            alpha_range = self.par_cider.get_alpha_range(self.a_comm.rank)
            self.alphas = [a for a in range(alpha_range[0], alpha_range[1])]
        else:
            self.alphas = []

        self.bas_exp = self.encut / self.lambd ** np.arange(self.Nalpha)
        self.bas_exp = np.flip(self.bas_exp)
        self.maxen = 2 * self.bas_exp[-1]
        self.minen = 2 * self.bas_exp[0]
        sinv = (4 * self.bas_exp[:, None] * self.bas_exp) ** 0.75 * (
            self.bas_exp[:, None] + self.bas_exp
        ) ** -1.5
        # invert and index for this process
        # self.alpha_cl = cho_factor(sinv)
        self.sinv = np.linalg.inv(sinv)[self.alphas]

    def _setup_cd_xd_ranks(self):
        ROOT = 0
        cd = self.gdfft.comm.rank if (self.gdfft is not None) else -1
        xd = 1 if (self.gd.comm.rank == ROOT) else 0

        rank_cd = np.empty(self.world.size, dtype=np.int32)
        self.world.gather(
            np.array([cd], dtype=np.int32),
            ROOT,
            rank_cd if self.world.rank == 0 else None,
        )
        self.world.broadcast(rank_cd, ROOT)
        my_rank = self.gdfft.rank if self.gdfft is not None else -2
        cdranks = np.arange(self.world.size)[rank_cd == my_rank]
        self.r2cd_comm = self.world.new_communicator(cdranks)

        rank_xd = np.empty(self.world.size, dtype=np.int32)
        self.world.gather(
            np.array([xd], dtype=np.int32),
            ROOT,
            rank_xd if self.world.rank == 0 else None,
        )
        self.world.broadcast(rank_xd, ROOT)
        xdranks = np.arange(self.world.size)[rank_xd.astype(bool)]
        self.r2xd_comm = self.world.new_communicator(xdranks)

    def _setup_kgrid(self):
        scale_c1 = (self.shape / (1.0 * self.gd.N_c))[:, np.newaxis]
        if self.a_comm is not None:
            self.gdfft = GridDescriptor(
                self.shape, self.gd.cell_cv * scale_c1, True, comm=self.comms["d"]
            )
            self.pwfft = CiderPWDescriptor(None, self.gdfft, gammacentered=True)
            self.k2_k = self.pwfft.G2_qG[0]
            self.knorm = np.sqrt(self.k2_k)
        else:
            self.gdfft = None
            self.pwfft = None
            self.k2_k = None
            self.knorm = None

    def _setup_6d_integral_buffer(self):
        self.rbuf_ag = LDict()
        self.kbuf_ak = LDict()
        self.theta_ak = LDict()
        for a in self.alphas:
            self.rbuf_ag[a] = self.gdfft.empty()
            self.kbuf_ak[a] = self.pwfft.empty()
            self.theta_ak[a] = self.pwfft.empty()
        self.rbuf_ag.lock()
        self.kbuf_ak.lock()
        self.theta_ak.lock()

    def _setup_plan(self):
        nldf_settings = self.cider_kernel.mlfunc.settings.nldf_settings
        need_plan = self._plan is None or self._plan.nspin != self.nspin
        need_plan = need_plan and not nldf_settings.is_empty
        if need_plan:
            self._plan = NLDFSplinePlan(
                nldf_settings,
                self.nspin,
                self.encut / self.lambd ** (self.Nalpha - 1),
                self.lambd,
                self.Nalpha,
                coef_order="gq",
                alpha_formula="etb",
                proc_inds=self.alphas,
            )

    def initialize_more_things(self):
        self._setup_kgrid()
        self._setup_cd_xd_ranks()
        self._setup_6d_integral_buffer()
        self._setup_plan()

    def domain_world2cider(self, n_xg, x=None, out=None, tmp=None):
        """
        Communicate the density n_xg from the world domain to the Cider
        domain. Input and output should both be distributed.
        """
        n = n_xg.shape[:-3]
        n_xg = self.gd.collect(n_xg, broadcast=False)
        if self.world.rank != 0:
            n_xg = None
        return self.domain_root2cider(n_xg, n=n)

    def domain_cider2world(self, n_xg, out=None, tmp=None):
        n_xg = self.gdfft.collect(n_xg, broadcast=False)
        if self.gd.comm.rank != 0:
            n_xg = None
        return self.domain_root2xc(n_xg)

    def domain_root2cider(self, n_xg, n=None, out=None, tmp=None):
        """
        Communicate the density n_xg from the world root to the Cider domain.
        """
        if self.r2cd_comm is not None:
            n = n or ()
            out = self.gdfft.empty(n=n, global_array=False)
            if self.r2cd_comm.rank == 0:
                out = self.gdfft.distribute(n_xg, out=out)
            self.r2cd_comm.broadcast(out, 0)
            return out
        else:
            return None

    def domain_root2xc(self, n_xg, n=None, out=None, tmp=None):
        """
        Communicate the density n_xg from the world root to the XC domain.
        """
        n = n or ()
        if self.r2xd_comm is not None:
            if n_xg is None:
                n_xg = self.gd.empty(n=n, global_array=True)
            self.r2xd_comm.broadcast(n_xg, 0)
        out = self.gd.empty(n=n, global_array=False)
        return self.gd.distribute(n_xg, out=out)

    def calculate_6d_integral(self):
        a_comm = self.a_comm
        alphas = self._plan.proc_inds

        self.timer.start("FFT")
        for a in alphas:
            self.theta_ak[a][:] = self.pwfft.fft(self.rbuf_ag[a])
        self.timer.stop()

        self.timer.start("KSPACE")
        for a in range(self.Nalpha):
            if a_comm is not None:
                vdw_ranka = self.alpha_ind(a)
                F_k = np.zeros(self.k2_k.shape, complex)
            for b in self._plan.proc_inds:
                aexp = self._plan.alphas[a] + self._plan.alphas[b]
                fac = (
                    np.pi / aexp
                ) ** 1.5  # * self._plan.alpha_norms[a] * self._plan.alpha_norms[b]
                pwutil.mulexp(
                    F_k.ravel(),
                    self.theta_ak[b].ravel(),
                    self.k2_k.ravel(),
                    fac,
                    1.0 / (4 * aexp),
                )
            if a_comm is not None:
                self.timer.start("gather")
                a_comm.sum(F_k, vdw_ranka)
                self.timer.stop("gather")
            if a_comm is not None and a_comm.rank == vdw_ranka:
                self.kbuf_ak[a][:] = F_k
            F_k = None
        self.timer.stop()

        self.timer.start("iFFT")
        for a in alphas:
            self.rbuf_ag[a][:] = self.pwfft.ifft(self.kbuf_ak[a])
        self.timer.stop()

    def calculate_6d_stress_integral(self):
        """
        Upon calling this routine, self.theta_ak should contain
        the theta_ak functions
        (as opposed to undefined or the functional derivatives),
        and rbuf_ag should contain dEdF_bg, i.e. the functional
        derivative wrt the convolved functions.
        """
        a_comm = self.a_comm
        alphas = self.alphas

        if a_comm is not None:
            G_Qv = self.pwfft.get_reciprocal_vectors()
            dG2deps_vvQ = np.ascontiguousarray(
                (-2 * G_Qv[:, None, :] * G_Qv[:, :, None]).T.astype(np.complex128)
            )

        self.timer.start("KSPACE")
        for a in range(self.Nalpha):
            if a_comm is not None:
                vdw_ranka = self.alpha_ind(a)
                F_k = np.zeros(self.k2_k.shape, complex)
            for b in self._plan.proc_inds:
                aexp = self._plan.alphas[a] + self._plan.alphas[b]
                invexp = 1.0 / (4 * aexp)
                fac = (
                    -1 * (np.pi / aexp) ** 1.5 * invexp
                )  # * self._plan.alpha_norms[a] * self._plan.alpha_norms[b]
                pwutil.mulexp(
                    F_k.ravel(),
                    self.theta_ak[b].ravel(),
                    self.k2_k.ravel(),
                    fac,
                    invexp,
                )
            if a_comm is not None:
                self.timer.start("gather")
                a_comm.sum(F_k, vdw_ranka)
                self.timer.stop("gather")
            if a_comm is not None and a_comm.rank == vdw_ranka:
                self.kbuf_ak[a][:] = F_k
            F_k = None
        self.timer.stop()

        self.timer.start("FFT")
        for a in alphas:
            self.theta_ak[a][:] = self.pwfft.fft(self.rbuf_ag[a])
        self.timer.stop()
        stress_vv = np.zeros((3, 3))
        if self.a_comm is not None:
            P_k = np.zeros(self.k2_k.shape, dtype=np.complex128)
            for a in self.alphas:
                P_k += self.theta_ak[a].conj() * self.kbuf_ak[a]
            for v1 in range(3):
                for v2 in range(3):
                    stress_vv[v1, v2] = self.pwfft.integrate(
                        # P_k, dG2deps_Qvv[:,v1,v2], global_integral=False,
                        dG2deps_vvQ[v1, v2],
                        P_k,
                        global_integral=False,
                        hermitian=True,
                    )
        self.world.sum(stress_vv)
        # print(stress_vv)
        self._stress_vv += stress_vv

    def calculate_6d_integral_fwd(self, n_g, cider_exp, c_abi=None):
        nexp = self.nexp

        p_iag = {}
        q_ag = {}
        dq_ag = {}

        cider_exp = self.domain_world2cider(cider_exp)

        i = -1
        if self.alphas:
            di = cider_exp[i]
        else:
            di = None
        if c_abi is not None:
            self.write_augfeat_to_rbuf(c_abi)
        else:
            for a in self.alphas:
                self.rbuf_ag[a][:] = 0.0
        self.timer.start("COEFS")
        p_qg, dp_qg = self._plan.get_interpolation_coefficients(di.ravel(), i=-1)
        p_qg = p_qg.T
        dp_qg = dp_qg.T
        for ind, a in enumerate(self.alphas):
            q_ag[a] = p_qg[ind]
            dq_ag[a] = dp_qg[ind]
            q_ag[a].shape = n_g.shape
            dq_ag[a].shape = n_g.shape
            self.rbuf_ag[a][:] += n_g * q_ag[a]
        self.timer.stop()

        self.calculate_6d_integral()

        if self.has_paw:
            self.timer.start("atom interp")
            D_abi = self.interpolate_rbuf_onto_atoms(pot=False)
            self.timer.stop()

        if n_g is not None:
            feat = np.zeros([nexp - 1] + list(n_g.shape))
            dfeat = np.zeros([nexp - 1] + list(n_g.shape))
        for i in range(nexp - 1):
            if self.alphas:
                di = cider_exp[i]
            else:
                di = None
            p_qg, dp_qg = self._plan.get_interpolation_coefficients(di.ravel(), i=i)
            p_qg = p_qg.T
            dp_qg = dp_qg.T
            p_qg.shape = (len(self.alphas),) + di.shape
            dp_qg.shape = (len(self.alphas),) + di.shape
            for ind, a in enumerate(self.alphas):
                p_iag[i, a] = p_qg[ind]
                feat[i, :] += p_qg[ind] * self.rbuf_ag[a]
                dfeat[i, :] += dp_qg[ind] * self.rbuf_ag[a]

        self.timer.start("6d comm fwd")
        if n_g is not None:
            self.a_comm.sum(feat, root=0)
            self.a_comm.sum(dfeat, root=0)
            if self.a_comm is not None and self.a_comm.rank == 0:
                feat = self.gdfft.collect(feat)
                dfeat = self.gdfft.collect(dfeat)
                if self.gdfft.rank != 0:
                    feat = None
                    dfeat = None
            else:
                feat = None
                dfeat = None
        else:
            feat = None
            dfeat = None
        feat = self.domain_root2xc(feat, n=(3,))
        dfeat = self.domain_root2xc(dfeat, n=(3,))
        self.timer.stop()

        if self.has_paw:
            return feat, dfeat, p_iag, q_ag, dq_ag, D_abi
        else:
            return feat, dfeat, p_iag, q_ag, dq_ag

    def calculate_6d_integral_bwd(
        self,
        n_g,
        v_g,
        vexp,
        vfeat_g,
        p_iag,
        q_ag,
        dq_ag,
        augv_abi=None,
        compute_stress=False,
    ):
        nexp = self.nexp
        alphas = self.alphas

        if n_g is not None:
            gcond = n_g > self.RHO_TOL
            ngcond = np.logical_not(gcond)

        self.timer.start("vexp")
        if augv_abi is not None:
            self.write_augfeat_to_rbuf(augv_abi, pot=True)
        else:
            for a in self.alphas:
                self.rbuf_ag[a][:] = 0.0
        for i in range(nexp - 1):
            vfeati_g = self.domain_world2cider(vfeat_g[i])
            for a in alphas:
                self.rbuf_ag[a][:] += vfeati_g * p_iag[i, a]
        self.timer.stop()
        if compute_stress:
            self.calculate_6d_stress_integral()

        self.calculate_6d_integral()

        if self.has_paw:
            self.timer.start("atom interp")
            D_abi = self.interpolate_rbuf_onto_atoms(pot=True)
            self.timer.stop()

        self.timer.start("6d comm bwd")
        if n_g is not None:
            v_gtmp = np.zeros_like(n_g)
            v_etmp = np.zeros_like(n_g)
            for a in self.alphas:
                # FFT to real space
                v_etmp += self.rbuf_ag[a] * n_g * dq_ag[a]
                v_gtmp += self.rbuf_ag[a] * q_ag[a]
            v_etmp[ngcond] = 0
            v_gtmp[ngcond] = 0
            self.a_comm.sum(v_etmp, root=0)
            self.a_comm.sum(v_gtmp, root=0)
        if self.a_comm is not None and self.a_comm.rank == 0:
            v_etmp = self.gdfft.collect(v_etmp)
            v_gtmp = self.gdfft.collect(v_gtmp)
            if self.gdfft.rank != 0:
                v_etmp = None
                v_gtmp = None
        else:
            v_etmp = None
            v_gtmp = None
        vexp[-1] += self.domain_root2xc(v_etmp)
        v_g[:] += self.domain_root2xc(v_gtmp)
        self.timer.stop()

        if self.has_paw:
            return D_abi

    def get_ascale_and_derivs(self, nt_sg, sigma_xg, tau_sg):
        raise NotImplementedError

    def _get_conv_terms(self, nt_sg, sigma_xg, tau_sg):
        nspin = nt_sg.shape[0]
        if tau_sg is None:
            rho_tuple = (nt_sg, sigma_xg[::2])
        else:
            rho_tuple = (nt_sg, sigma_xg[::2], tau_sg)
        shape = (
            nt_sg.shape[0],
            self.nexp,
        ) + nt_sg.shape[1:]
        ascale = np.empty(shape)
        if tau_sg is None:
            ascale_derivs = (np.empty(shape), np.empty(shape))
        else:
            ascale_derivs = (np.empty(shape), np.empty(shape), np.empty(shape))
        for i in range(-1, self.nexp - 1):
            ascale[:, i], tmp = self._plan.get_interpolation_arguments(rho_tuple, i=i)
            for j in range(len(tmp)):
                ascale_derivs[j][:, i] = tmp[j]
        # TODO need to save conv function derivative in case conv function != rho
        cider_nt_sg = self.domain_world2cider(
            self._plan.get_function_to_convolve(rho_tuple)[0]
        )
        if cider_nt_sg is None:
            cider_nt_sg = {s: None for s in range(nspin)}
        return cider_nt_sg, ascale, ascale_derivs

    def calc_cider(
        self,
        e_g,
        nt_sg,
        v_sg,
        sigma_xg,
        dedsigma_xg,
        tau_sg=None,
        dedtau_sg=None,
        compute_stress=False,
    ):
        feat = {}
        dfeat = {}
        p_iag = {}
        q_ag = {}
        dq_ag = {}

        self.timer.start("initialization")
        nspin = nt_sg.shape[0]
        nexp = self.nexp
        self.nspin = nspin
        if self.rbuf_ag is None:
            self.initialize_more_things()
        self.timer.stop()

        cider_nt_sg, ascale, ascale_derivs = self._get_conv_terms(
            nt_sg, sigma_xg, tau_sg
        )

        self.timer.start("6d int")
        self.theta_sak_tmp = {s: {} for s in range(nspin)}
        for s in range(nspin):
            (
                feat[s],
                dfeat[s],
                p_iag[s],
                q_ag[s],
                dq_ag[s],
            ) = self.calculate_6d_integral_fwd(cider_nt_sg[s], ascale[s])
            if compute_stress and nspin > 1:
                for a in self.alphas:
                    self.theta_sak_tmp[s][a] = self.theta_ak[a].copy()
        self.timer.stop()

        self.timer.start("Cider XC")
        feat_sig = np.stack([feat[s] for s in range(nspin)])
        del feat

        cond0 = feat_sig < 0
        feat_sig[cond0] = 0.0

        if tau_sg is None:  # GGA
            vfeat = self.cider_kernel.calculate(
                e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg, feat_sig
            )
        else:  # MGGA
            vfeat = self.cider_kernel.calculate(
                e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg, tau_sg, dedtau_sg, feat_sig
            )

        vfeat[cond0] = 0
        vexp = np.empty([nspin, nexp] + list(nt_sg[0].shape))
        for s in range(nspin):
            vexp[s, :-1] = vfeat[s] * dfeat[s]
            vexp[s, -1] = 0.0
        self.timer.stop()

        self.timer.start("6d int")
        for s in range(nspin):
            if compute_stress and nspin > 1:
                for a in self.alphas:
                    self.theta_ak[a][:] = self.theta_sak_tmp[s][a]
            self.calculate_6d_integral_bwd(
                cider_nt_sg[s],
                v_sg[s],
                vexp[s],
                vfeat[s],
                p_iag[s],
                q_ag[s],
                dq_ag[s],
                compute_stress=compute_stress,
            )
        self.timer.stop()

        self.add_scale_derivs(vexp, ascale_derivs, v_sg, dedsigma_xg, dedtau_sg)

    def get_cider_exp(self, *args, **kwargs):
        raise NotImplementedError

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        sigma_xg, dedsigma_xg, gradn_svg = gga_vars(gd, self.grad_v, n_sg)
        self.process_cider(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        add_gradient_correction(self.grad_v, gradn_svg, sigma_xg, dedsigma_xg, v_sg)

    def process_cider(*args, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_joblib(cls, fname, **kwargs):
        mlfunc = joblib.load(fname)
        return cls.from_mlfunc(mlfunc, **kwargs)

    @staticmethod
    def from_mlfunc(
        mlfunc,
        xkernel="GGA_X_PBE",
        ckernel="GGA_C_PBE",
        Nalpha=None,
        encut=300,
        lambd=1.8,
        xmix=1.00,
        debug=False,
    ):
        if mlfunc.desc_version == "b":
            if debug:
                raise NotImplementedError
            else:
                cider_kernel = CiderMGGAHybridKernel(mlfunc, xmix, xkernel, ckernel)
            cls = CiderMGGA
        elif mlfunc.desc_version == "d":
            if debug:
                cider_kernel = CiderGGAHybridKernelPBEPot(
                    mlfunc, xmix, xkernel, ckernel
                )
            else:
                cider_kernel = CiderGGAHybridKernel(mlfunc, xmix, xkernel, ckernel)
            cls = CiderGGA
        else:
            raise ValueError(
                "Only implemented for b and d version, found {}".format(
                    mlfunc.desc_version
                )
            )

        consts = np.array([0.00, mlfunc.a0, mlfunc.fac_mul, mlfunc.amin])
        const_list = np.stack(
            [0.5 * consts, 1.0 * consts, 2.0 * consts, consts * mlfunc.vvmul]
        )
        nexp = 4

        return cls(
            cider_kernel,
            nexp,
            const_list,
            Nalpha=Nalpha,
            lambd=lambd,
            encut=encut,
            xmix=xmix,
        )


class CiderGGA(_CiderBase, GGA):
    def __init__(self, cider_kernel, nexp, consts, **kwargs):
        if cider_kernel.mlfunc.settings.sl_settings.level != "GGA":
            raise ValueError("CiderGGA only supports GGA functionals!")
        _CiderBase.__init__(self, cider_kernel, nexp, consts, **kwargs)

        GGA.__init__(self, LibXC("PBE"), stencil=2)
        self.type = "GGA"
        self.name = "CIDER_{}".format(
            self.cider_kernel.xmix
        )  # TODO more informative name

    def todict(self):
        d = super(CiderGGA, self).todict()
        d["_cider_type"] = "CiderGGA"
        return d

    def set_grid_descriptor(self, gd):
        GGA.set_grid_descriptor(self, gd)
        _CiderBase.set_grid_descriptor(self, gd)

    get_cider_exp = get_cider_exp_gga

    def process_cider(self, e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg):
        self.calc_cider(e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg)

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac

    def get_ascale_and_derivs(self, nt_sg, sigma_xg, tau_sg):
        self.timer.start("CIDER_EXP")
        nspin, nexp = self.nspin, self.nexp
        ascale = np.empty([nspin, nexp] + list(nt_sg[0].shape))
        dadn = np.empty([nspin, nexp] + list(nt_sg[0].shape))
        dadsigma = np.empty([nspin, nexp] + list(nt_sg[0].shape))
        for s in range(nspin):
            ngcond = nt_sg[s] < self.RHO_TOL
            for i in range(nexp):
                ascale[s, i], dadn[s, i], dadsigma[s, i] = self.get_cider_exp(
                    nt_sg[s],
                    sigma_xg[2 * s],
                    a0=self.consts[i, 1],
                    fac_mul=self.consts[i, 2],
                    amin=self.consts[i, 3],
                    return_derivs=True,
                )
                dadn[s, i][ngcond] = 0
                dadsigma[s, i][ngcond] = 0
        self.timer.stop()
        return ascale, (dadn, dadsigma)

    def add_scale_derivs(self, vexp, derivs, v_sg, dedsigma_xg, dedtau_sg):
        self.timer.start("contract")
        nspin, nexp = self.nspin, self.nexp
        dadn, dadsigma = derivs
        for s in range(nspin):
            for i in range(nexp):
                v_sg[s] += vexp[s, i] * dadn[s, i]
                dedsigma_xg[2 * s] += vexp[s, i] * dadsigma[s, i]
        self.timer.stop()

    def stress_tensor_contribution(self, n_sg):
        sigma_xg, dedsigma_xg, gradn_svg = gga_vars(self.gd, self.grad_v, n_sg)
        nspins = len(n_sg)
        v_sg = self.gd.zeros(nspins)
        e_g = self.gd.empty()
        self._stress_vv = np.zeros((3, 3))
        self.calc_cider(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg, compute_stress=True)

        def integrate(a1_g, a2_g=None):
            return self.gd.integrate(a1_g, a2_g, global_integral=False)

        P = integrate(e_g)
        for v_g, n_g in zip(v_sg, n_sg):
            P -= integrate(v_g, n_g)
        for sigma_g, dedsigma_g in zip(sigma_xg, dedsigma_xg):
            P -= 2 * integrate(sigma_g, dedsigma_g)

        stress_vv = P * np.eye(3)
        for v1 in range(3):
            for v2 in range(3):
                stress_vv[v1, v2] -= (
                    integrate(gradn_svg[0, v1] * gradn_svg[0, v2], dedsigma_xg[0]) * 2
                )
                if nspins == 2:
                    stress_vv[v1, v2] -= (
                        integrate(gradn_svg[0, v1] * gradn_svg[1, v2], dedsigma_xg[1])
                        * 2
                    )
                    stress_vv[v1, v2] -= (
                        integrate(gradn_svg[1, v1] * gradn_svg[1, v2], dedsigma_xg[2])
                        * 2
                    )
        self.gd.comm.sum(stress_vv)
        stress_vv[:] += self._stress_vv
        return stress_vv


class CiderMGGA(_CiderBase, MGGA):
    def __init__(self, cider_kernel, nexp, consts, **kwargs):
        if cider_kernel.mlfunc.settings.sl_settings.level != "MGGA":
            raise ValueError("CiderMGGA only supports MGGA functionals!")
        _CiderBase.__init__(self, cider_kernel, nexp, consts, **kwargs)

        MGGA.__init__(self, LibXC("PBE"), stencil=2)
        self.type = "MGGA"
        self.name = "CIDER_{}".format(self.cider_kernel.xmix)

    def todict(self):
        d = super(CiderMGGA, self).todict()
        d["_cider_type"] = "CiderMGGA"
        return d

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac

    def set_grid_descriptor(self, gd):
        MGGA.set_grid_descriptor(self, gd)
        _CiderBase.set_grid_descriptor(self, gd)

    def calculate_paw_correction(
        self, setup, D_sp, dEdD_sp=None, addcoredensity=True, a=None
    ):
        return 0

    get_cider_exp = get_cider_exp_mgga

    def process_cider(self, e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg):
        if self.fixed_ke:
            taut_sG = self._fixed_taut_sG
            if not self._taut_gradv_init:
                self._taut_gradv_init = True
                # ensure initialization for calculation potential
                self.wfs.calculate_kinetic_energy_density()
        else:
            taut_sG = self.wfs.calculate_kinetic_energy_density()

        if taut_sG is None:
            taut_sG = self.wfs.gd.zeros(len(nt_sg))

        taut_sg = np.empty_like(nt_sg)

        for taut_G, taut_g in zip(taut_sG, taut_sg):
            self.distribute_and_interpolate(taut_G, taut_g)

        dedtaut_sg = np.zeros_like(nt_sg)
        self.calc_cider(e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg, taut_sg, dedtaut_sg)

        self.dedtaut_sG = self.wfs.gd.empty(self.wfs.nspins)
        self.ekin = 0.0
        for s in range(self.wfs.nspins):
            self.restrict_and_collect(dedtaut_sg[s], self.dedtaut_sG[s])
            self.ekin -= self.wfs.gd.integrate(self.dedtaut_sG[s] * taut_sG[s])

    def get_ascale_and_derivs(self, nt_sg, sigma_xg, tau_sg):
        self.timer.start("CIDER_EXP")
        nspin, nexp = self.nspin, self.nexp
        ascale = np.empty([nspin, nexp] + list(nt_sg[0].shape))
        dadn = np.empty([nspin, nexp] + list(nt_sg[0].shape))
        dadsigma = np.empty([nspin, nexp] + list(nt_sg[0].shape))
        dadtau = np.empty([nspin, nexp] + list(nt_sg[0].shape))
        for s in range(nspin):
            ngcond = nt_sg[s] < self.RHO_TOL
            for i in range(nexp):
                (
                    ascale[s, i],
                    dadn[s, i],
                    dadsigma[s, i],
                    dadtau[s, i],
                ) = self.get_cider_exp(
                    nt_sg[s],
                    sigma_xg[2 * s],
                    tau_sg[s],
                    a0=self.consts[i, 1],
                    fac_mul=self.consts[i, 2],
                    amin=self.consts[i, 3],
                    return_derivs=True,
                )
                dadn[s, i][ngcond] = 0
                dadsigma[s, i][ngcond] = 0
                dadtau[s, i][ngcond] = 0
        self.timer.stop()
        return ascale, (dadn, dadsigma, dadtau)

    def add_scale_derivs(self, vexp, derivs, v_sg, dedsigma_xg, dedtau_sg):
        self.timer.start("contract")
        nspin, nexp = self.nspin, self.nexp
        dadn, dadsigma, dadtau = derivs
        for s in range(nspin):
            for i in range(nexp):
                v_sg[s] += vexp[s, i] * dadn[s, i]
                dedsigma_xg[2 * s] += vexp[s, i] * dadsigma[s, i]
                dedtau_sg[s] += vexp[s, i] * dadtau[s, i]
        self.timer.stop()

    def add_forces(self, F_av):
        raise NotImplementedError("MGAA Force with SG15")

    def stress_tensor_contribution(self, n_sg):
        sigma_xg, dedsigma_xg, gradn_svg = gga_vars(self.gd, self.grad_v, n_sg)
        taut_sG = self.wfs.calculate_kinetic_energy_density()
        if taut_sG is None:
            taut_sG = self.wfs.gd.zeros(len(n_sg))
        taut_sg = np.empty_like(n_sg)
        if self.tauct_G is None:
            raise NotImplementedError("MGGA Stress with SG15")
        for taut_G, taut_g in zip(taut_sG, taut_sg):
            taut_G += 1.0 / self.wfs.nspins * self.tauct_G
            self.distribute_and_interpolate(taut_G, taut_g)

        nspins = len(n_sg)
        v_sg = self.gd.zeros(nspins)
        dedtaut_sg = np.empty_like(n_sg)
        e_g = self.gd.empty()
        # self.calc_cider(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        self._stress_vv = np.zeros((3, 3))
        self.calc_cider(
            e_g,
            n_sg,
            v_sg,
            sigma_xg,
            dedsigma_xg,
            taut_sg,
            dedtaut_sg,
            compute_stress=True,
        )

        def integrate(a1_g, a2_g=None):
            return self.gd.integrate(a1_g, a2_g, global_integral=False)

        P = integrate(e_g)
        for v_g, n_g in zip(v_sg, n_sg):
            P -= integrate(v_g, n_g)
        for sigma_g, dedsigma_g in zip(sigma_xg, dedsigma_xg):
            P -= 2 * integrate(sigma_g, dedsigma_g)
        for taut_g, dedtaut_g in zip(taut_sg, dedtaut_sg):
            P -= integrate(taut_g, dedtaut_g)

        tau_svvG = self.wfs.calculate_kinetic_energy_density_crossterms()

        stress_vv = P * np.eye(3)
        for v1 in range(3):
            for v2 in range(3):
                stress_vv[v1, v2] -= (
                    integrate(gradn_svg[0, v1] * gradn_svg[0, v2], dedsigma_xg[0]) * 2
                )
                if nspins == 2:
                    stress_vv[v1, v2] -= (
                        integrate(gradn_svg[0, v1] * gradn_svg[1, v2], dedsigma_xg[1])
                        * 2
                    )
                    stress_vv[v1, v2] -= (
                        integrate(gradn_svg[1, v1] * gradn_svg[1, v2], dedsigma_xg[2])
                        * 2
                    )
        tau_cross_g = self.gd.empty()
        for s in range(nspins):
            for v1 in range(3):
                for v2 in range(3):
                    self.distribute_and_interpolate(tau_svvG[s, v1, v2], tau_cross_g)
                    stress_vv[v1, v2] -= integrate(tau_cross_g, dedtaut_sg[s])

        self.dedtaut_sG = self.wfs.gd.empty(self.wfs.nspins)
        for s in range(self.wfs.nspins):
            self.restrict_and_collect(dedtaut_sg[s], self.dedtaut_sG[s])
        self.gd.comm.sum(stress_vv)
        stress_vv[:] += self._stress_vv
        return stress_vv
