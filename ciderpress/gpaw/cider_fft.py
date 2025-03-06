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
import yaml
from gpaw import mpi
from gpaw.xc.gga import GGA
from gpaw.xc.libvdwxc import FFTDistribution
from gpaw.xc.libxc import LibXC
from gpaw.xc.mgga import MGGA

from ciderpress.dft.density_util import get_sigma
from ciderpress.dft.plans import NLDFSplinePlan
from ciderpress.gpaw.config import GPAW_DEFAULT_RHO_TOL
from ciderpress.gpaw.nldf_interface import LibCiderPW

CIDERPW_GRAD_MODE_NONE = 0
CIDERPW_GRAD_MODE_FORCE = 1
CIDERPW_GRAD_MODE_STRESS = 2


class _CiderBase:

    is_cider_functional = True

    has_paw = False

    def __init__(
        self, cider_kernel, Nalpha=None, lambd=1.8, encut=300, world=None, **kwargs
    ):
        if world is None:
            self.world = mpi.world
        else:
            self.world = world

        self.cider_kernel = cider_kernel
        nldf_settings = cider_kernel.mlfunc.settings.nldf_settings
        if nldf_settings.is_empty:
            # TODO these numbers don't matter but must be defined, which is messy
            self.Nalpha = 10
            self.lambd = 2.0
            self.encut = 100
        else:
            if Nalpha is None:
                amax = encut
                # amin = np.min(consts[:, -1]) / np.e
                # TODO better choice here? depend on feat_params too?
                amin = nldf_settings.theta_params[0]
                if hasattr(nldf_settings, "feat_params"):
                    amin = min(amin, np.min([p[0] for p in nldf_settings.feat_params]))
                    # amin = min(amin, np.min(np.array(nldf_settings.feat_params)[:, 0]))
                amin /= 64
                Nalpha = int(np.ceil(np.log(amax / amin) / np.log(lambd))) + 1
                lambd = np.exp(np.log(amax / amin) / (Nalpha - 1))
            self.Nalpha = Nalpha
            self.lambd = lambd
            self.encut = encut

        self._plan = None
        self.distribution = None
        self.fft_obj = None
        self.c_asiq = None
        self._save_c_asiq = None
        self._save_v_avsiq = None
        self._global_redistributor = None
        self.aux_gd = None

        self.setup_name = "PBE"

    def get_mlfunc_data(self):
        return yaml.dump(self.cider_kernel.mlfunc, Dumper=yaml.CDumper)

    def get_setup_name(self):
        return self.setup_name

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

    def _check_parallelization(self, wfs):
        if wfs.world.size > self.gd.comm.size:
            import warnings

            from gpaw.utilities.grid import GridRedistributor

            self.aux_gd = self.gd.new_descriptor(comm=wfs.world)
            self._global_redistributor = GridRedistributor(
                wfs.world, wfs.kptband_comm, self.gd, self.aux_gd
            )
            self.distribution = FFTDistribution(
                self.aux_gd, [self.aux_gd.comm.size, 1, 1]
            )
            warnings.warn(
                "You are using Cider with only %d out of %d available cores. "
                "The internal Cider routines will redistribute the density "
                "so that all cores can be used. Setting augment_grids=True in "
                "the parallel option is recommended and might be more efficient."
                % (self.gd.comm.size, wfs.world.size)
            )
        else:
            self.aux_gd = self.gd
            self._global_redistributor = None

    def initialize(self, density, hamiltonian, wfs):
        if self.type == "GGA":
            GGA.initialize(self, density, hamiltonian, wfs)
        elif self.type == "MGGA":
            MGGA.initialize(self, density, hamiltonian, wfs)
        else:
            raise NotImplementedError
        self.timer = wfs.timer
        self.world = wfs.world
        self.dens = density
        self._plan = None
        self.aux_gd = None
        self._check_parallelization(wfs)

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
                rhocut=GPAW_DEFAULT_RHO_TOL,
                expcut=GPAW_DEFAULT_RHO_TOL,
                use_smooth_expnt_cutoff=True,
            )
        self.fft_obj = LibCiderPW(
            self.aux_gd.N_c, self.aux_gd.cell_cv, self.aux_gd.comm, plan=self._plan
        )

    def set_grid_descriptor(self, gd):
        self.gd = gd
        self.distribution = FFTDistribution(gd, [gd.comm.size, 1, 1])

    def set_fft_work(self, s, pot=False):
        pass

    def get_fft_work(self, s, pot=False, grad_mode=CIDERPW_GRAD_MODE_NONE):
        if grad_mode in [CIDERPW_GRAD_MODE_FORCE, CIDERPW_GRAD_MODE_STRESS]:
            self._save_v_avsiq = {}

    def _set_paw_terms(self):
        pass

    def _calculate_paw_energy_and_potential(self, grad_mode=CIDERPW_GRAD_MODE_NONE):
        if grad_mode == CIDERPW_GRAD_MODE_STRESS:
            return 0

    def _get_dH_asp(self):
        pass

    def call_fwd(self, rho_tuple, grad_mode=CIDERPW_GRAD_MODE_NONE):
        p, dp = None, None
        plan = self._plan
        self._set_paw_terms()
        arg_sg, darg_sg = plan.get_interpolation_arguments(rho_tuple, i=-1)
        fun_sg, dfun_sg = plan.get_function_to_convolve(rho_tuple)
        shape = (plan.nspin, plan.nldf_settings.nfeat) + arg_sg.shape[1:]
        feat_sig = np.empty(shape)
        shape = (plan.nspin, plan.num_vj) + arg_sg.shape[1:]
        dfeat_sjg = np.empty(shape)
        if grad_mode == CIDERPW_GRAD_MODE_STRESS:
            self._theta_s_gq = {}
        for s in range(plan.nspin):
            _rho_tuple = tuple(x[s] for x in rho_tuple)
            p, dp = plan.get_interpolation_coefficients(
                arg_sg[s], i=-1, vbuf=p, dbuf=dp
            )
            p.shape = fun_sg.shape[-3:] + (p.shape[-1],)
            self.fft_obj.set_work(fun_sg[s], p)
            self.set_fft_work(s, pot=False)
            if grad_mode == CIDERPW_GRAD_MODE_STRESS:
                self._theta_s_gq[s] = self.fft_obj.copy_work_array()
            self.timer.start("FFT and kernel")
            self.fft_obj.compute_forward_convolution()
            self.timer.stop("FFT and kernel")
            self.timer.start("Features")
            self.get_fft_work(s, pot=False, grad_mode=grad_mode)
            for i in range(plan.num_vj):
                a_g = plan.get_interpolation_arguments(_rho_tuple, i=i)[0]
                p, dp = plan.get_interpolation_coefficients(a_g, i=i, vbuf=p, dbuf=dp)
                if False:  # TODO might need this later
                    coeff_multipliers = None
                    p[:] *= coeff_multipliers
                    dp[:] *= coeff_multipliers
                if False:  # TODO might need this later
                    plan.get_transformed_interpolation_terms(
                        p, i=i, fwd=True, inplace=True
                    )
                    plan.get_transformed_interpolation_terms(
                        dp, i=i, fwd=True, inplace=True
                    )
                p.shape = a_g.shape + (plan.nalpha,)
                dp.shape = a_g.shape + (plan.nalpha,)
                self.fft_obj.fill_vj_feature_(feat_sig[s, i], p)
                self.fft_obj.fill_vj_feature_(dfeat_sjg[s, i], dp)
            self.timer.stop()
        return feat_sig, dfeat_sjg, arg_sg, darg_sg, fun_sg, dfun_sg, p, dp

    def call_bwd(
        self,
        rho_tuple,
        vfeat_sig,
        dfeat_sjg,
        arg_sg,
        fun_sg,
        v_sg,
        dedsigma_xg,
        dedtau_sg,
        p_gq=None,
        dp_gq=None,
        grad_mode=CIDERPW_GRAD_MODE_NONE,
    ):
        dedarg_sg = np.empty(v_sg.shape)
        dedfun_sg = np.empty(v_sg.shape)
        plan = self._plan
        paw_res = self._calculate_paw_energy_and_potential(grad_mode=grad_mode)
        if grad_mode == CIDERPW_GRAD_MODE_FORCE:
            fs = {}
            for a, v_vsiq in self._save_v_avsiq.items():
                fs[a] = np.einsum("vsiq,siq->v", v_vsiq, self.c_asiq[a])
        elif grad_mode == CIDERPW_GRAD_MODE_STRESS:
            fs = paw_res * np.eye(3)
            for a, v_vvsiq in self._save_v_avsiq.items():
                fs += np.einsum("uvsiq,siq->uv", v_vvsiq, self.c_asiq[a])
        else:
            fs = None
        for s in range(plan.nspin):
            self.timer.start("Feature potential")
            if grad_mode == CIDERPW_GRAD_MODE_STRESS:
                self.fft_obj.fftv2(self._theta_s_gq[s])
            self.fft_obj.reset_work()
            for i in range(plan.num_vj):
                _rho_tuple = (x[s] for x in rho_tuple)
                a_g, da_g = plan.get_interpolation_arguments(_rho_tuple, i=i)
                p_gq, dp_gq = plan.get_interpolation_coefficients(
                    a_g, i=i, vbuf=p_gq, dbuf=dp_gq
                )
                p_gq.shape = a_g.shape + (plan.nalpha,)
                dp_gq.shape = a_g.shape + (plan.nalpha,)
                self.fft_obj.fill_vj_potential_(vfeat_sig[s, i], p_gq)
                tmp = dfeat_sjg[s, i] * vfeat_sig[s, i]
                v_sg[s] += da_g[0] * tmp
                dedsigma_xg[2 * s] += da_g[1] * tmp
                if len(rho_tuple) == 3:
                    dedtau_sg[s] += da_g[2] * tmp
                # dedrho_sxg[s, 0] += da_g[0] * tmp
                # dedrho_sxg[s, 1:4] += 2 * da_g[1] * rho_sxg[s, 1:4] * tmp
                # if tau_sg is not None:
                #    dedrho_sxg[s, 4] += da_g[2] * tmp
            self.timer.stop()
            self.set_fft_work(s, pot=True)
            self.timer.start("FFT and kernel")
            if grad_mode == CIDERPW_GRAD_MODE_STRESS:
                fs += self.fft_obj.compute_backward_convolution_with_stress(
                    self._theta_s_gq[s]
                )
            else:
                self.fft_obj.compute_backward_convolution()
            self.timer.stop("FFT and kernel")
            self.get_fft_work(s, pot=True, grad_mode=grad_mode)
            p_gq, dp_gq = plan.get_interpolation_coefficients(
                arg_sg[s], i=-1, vbuf=p_gq, dbuf=dp_gq
            )
            p_gq.shape = fun_sg.shape[-3:] + (p_gq.shape[-1],)
            dp_gq.shape = fun_sg.shape[-3:] + (dp_gq.shape[-1],)
            self.fft_obj.fill_vj_feature_(dedarg_sg[s], dp_gq)
            self.fft_obj.fill_vj_feature_(dedfun_sg[s], p_gq)
        if grad_mode == CIDERPW_GRAD_MODE_FORCE:
            for a, v_vsiq in self._save_v_avsiq.items():
                fs[a] += np.einsum("vsiq,siq->v", v_vsiq, self._save_c_asiq[a])
        elif grad_mode == CIDERPW_GRAD_MODE_STRESS:
            for a, v_vvsiq in self._save_v_avsiq.items():
                fs += np.einsum("uvsiq,siq->uv", v_vvsiq, self._save_c_asiq[a])
        dedarg_sg[:] *= fun_sg
        self._get_dH_asp()
        if fs is not None:
            return dedarg_sg, dedfun_sg, fs
        else:
            return dedarg_sg, dedfun_sg

    def initialize_more_things(self):
        pass

    @property
    def is_initialized(self):
        if self._plan is None or self._plan.nspin != self.nspin:
            return False
        else:
            return True

    def calc_cider(
        self,
        e_g,
        rho_sxg,
        dedrho_sxg,
        grad_mode=CIDERPW_GRAD_MODE_NONE,
    ):
        nspin = len(rho_sxg)
        self.nspin = nspin
        if not self.is_initialized:
            self._setup_plan()
            self.fft_obj.initialize_backend()
            self.initialize_more_things()
        plan = self._plan
        sigma_xg = get_sigma(rho_sxg[:, 1:4])
        dedsigma_xg = np.zeros_like(sigma_xg)
        rho_tuple = plan.get_rho_tuple(rho_sxg, with_spin=True)
        nt_sg = rho_tuple[0]
        v_sg = np.zeros_like(nt_sg)
        if len(rho_tuple) == 2:
            tau_sg = None
            dedtau_sg = None
        else:
            tau_sg = rho_tuple[2].copy()
            dedtau_sg = np.zeros_like(tau_sg)

        res = self.call_fwd(rho_tuple, grad_mode=grad_mode)
        feat_sig, dfeat_sjg, arg_sg, darg_sg, fun_sg, dfun_sg, p_gq, dp_gq = res
        # TODO version i features
        feat_sig[:] *= plan.nspin
        self.timer.start("eval xc")
        if e_g.size > 0:
            if tau_sg is None:  # GGA
                vfeat_sig = self.cider_kernel.calculate(
                    e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg, feat_sig
                )
            else:  # MGGA
                vfeat_sig = self.cider_kernel.calculate(
                    e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg, tau_sg, dedtau_sg, feat_sig
                )
        else:
            vfeat_sig = np.zeros_like(feat_sig)
        self.timer.stop()

        vfeat_sig[:] *= plan.nspin
        res = self.call_bwd(
            rho_tuple,
            vfeat_sig,
            dfeat_sjg,
            arg_sg,
            fun_sg,
            v_sg,
            dedsigma_xg,
            dedtau_sg,
            p_gq=p_gq,
            dp_gq=dp_gq,
            grad_mode=grad_mode,
        )
        if grad_mode in [CIDERPW_GRAD_MODE_FORCE, CIDERPW_GRAD_MODE_STRESS]:
            dedarg_sg, dedfun_sg, fs = res
        else:
            dedarg_sg, dedfun_sg = res
            fs = None
        self.timer.start("add potential")
        for i, term in enumerate(darg_sg):
            term[:] *= dedarg_sg
            term[:] += dedfun_sg * dfun_sg[i]
        for s in range(plan.nspin):
            v_sg[s] += darg_sg[0][s]
            dedsigma_xg[2 * s] += darg_sg[1][s]
            if tau_sg is not None:
                dedtau_sg[s] += darg_sg[2][s]

        dedrho_sxg[:, 0] += v_sg
        dedrho_sxg[:, 1:4] += 2 * rho_sxg[:, 1:4] * dedsigma_xg[::2, None, ...]
        if plan.nspin == 2:
            dedrho_sxg[:, 1:4] += rho_sxg[::-1, 1:4] * dedsigma_xg[1:2, None, ...]
        if tau_sg is not None:
            dedrho_sxg[:, 4] += dedtau_sg
        self.timer.stop()
        return fs

    def _distribute_to_cider_grid(self, n_xg):
        if self._global_redistributor is not None:
            n_xg = self._global_redistributor.distribute(n_xg)
        shape = n_xg.shape[:-3]
        ndist_xg = self.distribution.block_zeros(shape)
        self.distribution.gd2block(n_xg, ndist_xg)
        return ndist_xg

    def _add_from_cider_grid(self, d_xg, ddist_xg):
        if self._global_redistributor is not None:
            shape = d_xg.shape[:-3]
            tmp_xg = self._global_redistributor.aux_gd.zeros(shape)
            self.distribution.block2gd_add(ddist_xg, tmp_xg)
            d_xg[:] += self._global_redistributor.collect(tmp_xg)
        else:
            self.distribution.block2gd_add(ddist_xg, d_xg)

    def _get_taut(self, n_sg):
        if self.type == "MGGA":
            if self.fixed_ke:
                taut_sG = self._fixed_taut_sG
                if not self._taut_gradv_init:
                    self._taut_gradv_init = True
                    # ensure initialization for calculating potential
                    self.wfs.calculate_kinetic_energy_density()
            else:
                taut_sG = self.wfs.calculate_kinetic_energy_density()

            if taut_sG is None:
                taut_sG = self.wfs.gd.zeros(len(n_sg))

            taut_sg = np.empty_like(n_sg)

            for taut_G, taut_g in zip(taut_sG, taut_sg):
                self.distribute_and_interpolate(taut_G, taut_g)

            dedtaut_sg = np.zeros_like(n_sg)
        else:
            taut_sg = None
            dedtaut_sg = None
            taut_sG = None
        return taut_sg, dedtaut_sg, taut_sG

    def _get_cider_inputs(self, n_sg, taut_sg):
        rho_sxg = get_rho_sxg(n_sg, self.grad_v, taut_sg)
        shape = rho_sxg.shape[:2]
        rho_sxg.shape = (rho_sxg.shape[0] * rho_sxg.shape[1],) + rho_sxg.shape[-3:]
        rho_sxg = self._distribute_to_cider_grid(rho_sxg)
        rho_sxg.shape = shape + rho_sxg.shape[-3:]
        self.nspin = len(rho_sxg)
        dedrho_sxg = np.zeros_like(rho_sxg)
        _e = self._distribute_to_cider_grid(np.zeros_like(n_sg[0])[None, :])[0]
        return _e, rho_sxg, dedrho_sxg

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        taut_sg, dedtaut_sg, taut_sG = self._get_taut(n_sg)
        _e, rho_sxg, dedrho_sxg = self._get_cider_inputs(n_sg, taut_sg)
        tmp_dedrho_sxg = np.zeros((len(n_sg), rho_sxg.shape[1]) + e_g.shape)
        e_g[:] = 0.0
        self.calc_cider(_e, rho_sxg, dedrho_sxg)
        self._add_from_cider_grid(e_g[None, :], _e[None, :])
        for s in range(len(n_sg)):
            self._add_from_cider_grid(tmp_dedrho_sxg[s], dedrho_sxg[s])
        v_sg[:] = tmp_dedrho_sxg[:, 0]
        add_gradient_correction(self.grad_v, tmp_dedrho_sxg, v_sg)

        if dedtaut_sg is not None:
            dedtaut_sg[:] = tmp_dedrho_sxg[:, 4]
            self.dedtaut_sG = self.wfs.gd.empty(self.wfs.nspins)
            self.ekin = 0.0
            for s in range(self.wfs.nspins):
                self.restrict_and_collect(dedtaut_sg[s], self.dedtaut_sG[s])
                self.ekin -= self.wfs.gd.integrate(self.dedtaut_sG[s] * taut_sG[s])


class CiderGGA(_CiderBase, GGA):
    def __init__(self, cider_kernel, **kwargs):
        sl_settings = cider_kernel.mlfunc.settings.sl_settings
        if not sl_settings.is_empty and sl_settings.level != "GGA":
            raise ValueError("CiderGGA only supports GGA functionals!")
        _CiderBase.__init__(self, cider_kernel, **kwargs)

        GGA.__init__(self, LibXC("PBE"), stencil=2)
        self.type = "GGA"
        # TODO more informative name for the functional
        self.name = "CIDER_{}".format(self.cider_kernel.xmix)

    def todict(self):
        d = super(CiderGGA, self).todict()
        d["_cider_type"] = "CiderGGA"
        return d

    def set_grid_descriptor(self, gd):
        GGA.set_grid_descriptor(self, gd)
        _CiderBase.set_grid_descriptor(self, gd)

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac

    def stress_tensor_contribution(self, n_sg):
        e_g = np.zeros(n_sg.shape[1:])
        stress1_vv = np.zeros((3, 3))
        _e, rho_sxg, dedrho_sxg = self._get_cider_inputs(n_sg, None)
        tmp_dedrho_sxg = np.zeros((len(n_sg), rho_sxg.shape[1]) + e_g.shape)
        e_g[:] = 0.0
        fs = self.calc_cider(
            _e, rho_sxg, dedrho_sxg, grad_mode=CIDERPW_GRAD_MODE_STRESS
        )
        stress1_vv[:] += 0.5 * (fs + fs.T)
        self.world.sum(stress1_vv, 0)

        self._add_from_cider_grid(e_g[None, :], _e[None, :])
        for s in range(len(n_sg)):
            self._add_from_cider_grid(tmp_dedrho_sxg[s], dedrho_sxg[s])
        dedrho_sxg = tmp_dedrho_sxg
        tmp_rho_sxg = np.zeros_like(tmp_dedrho_sxg)
        for s in range(len(n_sg)):
            self._add_from_cider_grid(tmp_rho_sxg[s], rho_sxg[s])
        rho_sxg = tmp_rho_sxg

        def integrate(a1_g, a2_g=None):
            return self.gd.integrate(a1_g, a2_g, global_integral=False)

        P = integrate(e_g)
        nspin = n_sg.shape[0]
        for s in range(nspin):
            for x in range(rho_sxg.shape[1]):
                P -= integrate(dedrho_sxg[s, x], rho_sxg[s, x])

        stress_vv = P * np.eye(3)
        for s in range(nspin):
            for v1 in range(3):
                for v2 in range(3):
                    stress_vv[v1, v2] -= integrate(
                        rho_sxg[s, v1 + 1], dedrho_sxg[s, v2 + 1]
                    )

        stress_vv = 0.5 * (stress_vv + stress_vv.T)

        self.gd.comm.sum(stress_vv)

        stress_vv += stress1_vv

        self.gd.comm.broadcast(stress_vv, 0)
        return stress_vv


def get_rho_sxg(n_sg, grad_v, tau_sg=None):
    nspin = len(n_sg)
    shape = list(n_sg.shape)
    if tau_sg is None:
        shape = shape[:1] + [4] + shape[1:]
    else:
        shape = shape[:1] + [5] + shape[1:]
    rho_sxg = np.empty(shape)
    rho_sxg[:, 0] = n_sg
    for v in range(3):
        for s in range(nspin):
            grad_v[v](n_sg[s], rho_sxg[s, v + 1])
    if tau_sg is not None:
        rho_sxg[:, 4] = tau_sg
    return rho_sxg


def add_gradient_correction(grad_v, dedrho_sxg, v_sg):
    nspin = len(v_sg)
    vv_g = np.empty_like(v_sg[0])
    for v in range(3):
        for s in range(nspin):
            grad_v[v](dedrho_sxg[s, v + 1], vv_g)
            v_sg[s] -= vv_g


class CiderMGGA(_CiderBase, MGGA):
    def __init__(self, cider_kernel, **kwargs):
        sl_settings = cider_kernel.mlfunc.settings.sl_settings
        if (not sl_settings.is_empty) and sl_settings.level != "MGGA":
            raise ValueError("CiderMGGA only supports MGGA functionals!")
        _CiderBase.__init__(self, cider_kernel, **kwargs)

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

    def add_forces(self, F_av):
        raise NotImplementedError(
            "Forces not implemented for CIDER MGGA with NCPP. "
            "Please use PAW setups to get forces."
        )

    def stress_tensor_contribution(self, n_sg):
        raise NotImplementedError("MGGA stress requires PAW setups")
