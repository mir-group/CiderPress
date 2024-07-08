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
from gpaw.xc.libvdwxc import FFTDistribution

from ciderpress.dft.density_util import get_sigma
from ciderpress.dft.plans import NLDFSplinePlan
from ciderpress.gpaw.cider_fft import (
    DEFAULT_RHO_TOL,
    GGA,
    MGGA,
    CiderParallelization,
    LibXC,
    mpi,
)
from ciderpress.gpaw.nldf_interface import LibCiderPW


class _FastCiderBase:

    has_paw = False

    RHO_TOL = DEFAULT_RHO_TOL

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
                    amin = min(amin, np.min(np.array(nldf_settings.feat_params)[:, 0]))
                amin /= 64
                Nalpha = int(np.ceil(np.log(amax / amin) / np.log(lambd))) + 1
                lambd = np.exp(np.log(amax / amin) / (Nalpha - 1))
            self.Nalpha = Nalpha
            self.lambd = lambd
            self.encut = encut

        self.phi_aajp = None
        self.verbose = False
        self.size = None
        self._plan = None
        self.distribution = None
        self.fft_obj = None

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
        self.rbuf_ag = None
        self.kbuf_ak = None
        self.theta_ak = None

    def _setup_plan(self):
        nldf_settings = self.cider_kernel.mlfunc.settings.nldf_settings
        need_plan = self._plan is None or self._plan.nspin != self.nspin
        need_plan = need_plan and not nldf_settings.is_empty
        self.par_cider = CiderParallelization(self.world, self.Nalpha)
        self.comms = self.par_cider.build_communicators()
        self.a_comm = self.comms["a"]
        if need_plan:
            self._plan = NLDFSplinePlan(
                nldf_settings,
                self.nspin,
                self.encut / self.lambd ** (self.Nalpha - 1),
                self.lambd,
                self.Nalpha,
                coef_order="gq",
                alpha_formula="etb",
            )
        self.fft_obj = LibCiderPW(
            self.gd.N_c, self.gd.cell_cv, self.gd.comm, plan=self._plan
        )

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
        self.distribution = FFTDistribution(gd, [gd.comm.size, 1, 1])

    def call_fwd(self, arg_sg, fun_sg):
        f_sgq = self.fft_obj.get_real_array()
        p, dp = None, None
        for s in range(self.nspin):
            p, dp = self._plan.get_interpolation_coefficients(
                arg_sg[s], i=-1, vbuf=p, dbuf=dp
            )
            tmp = f_sgq[s]
            p.shape = fun_sg.shape[-3:] + (p.shape[-1],)
            tmp[:] = fun_sg[s][..., None] * p
        self.fft_obj.compute_forward_convolution()
        return f_sgq

    def call_bwd(self, arg_sg, fun_sg):
        f_sgq = self.fft_obj.get_real_array()
        self.fft_obj.compute_backward_convolution()
        dedarg_sg = np.empty(f_sgq.shape[:-1])
        dedfun_sg = np.empty(f_sgq.shape[:-1])
        p, dp = None, None
        for s in range(self.nspin):
            p, dp = self._plan.get_interpolation_coefficients(
                arg_sg[s], i=-1, vbuf=p, dbuf=dp
            )
            p.shape = fun_sg.shape[-3:] + (p.shape[-1],)
            dp.shape = fun_sg.shape[-3:] + (dp.shape[-1],)
            dedarg_sg[s] = np.einsum("...q,...q->...", dp, f_sgq[s]) * fun_sg[s]
            dedfun_sg[s] = np.einsum("...q,...q->...", p, f_sgq[s])
        return dedarg_sg, dedfun_sg

    def calc_cider(
        self,
        e_g,
        rho_sxg,
        dedrho_sxg,
        compute_stress=False,
    ):
        nspin = len(rho_sxg)
        self.nspin = nspin
        if self._plan is None:
            self._setup_plan()
            self.fft_obj.initialize_backend()
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
        arg_sg, darg_sg = plan.get_interpolation_arguments(rho_tuple, i=-1)
        fun_sg, dfun_sg = plan.get_function_to_convolve(rho_tuple)
        f_sgq = self.call_fwd(arg_sg, fun_sg)

        feat_sig = np.empty(
            (
                nspin,
                plan.nldf_settings.nfeat,
            )
            + e_g.shape
        )
        dfeat_sjg = np.empty(
            (
                nspin,
                plan.num_vj,
            )
            + e_g.shape
        )
        p_gq, dp_gq = None, None
        for s in range(nspin):
            for i in range(plan.num_vj):
                _rho_tuple = (x[s] for x in rho_tuple)
                a_g = plan.get_interpolation_arguments(_rho_tuple, i=i)[0]
                p_gq, dp_gq = plan.get_interpolation_coefficients(
                    a_g, i=i, vbuf=p_gq, dbuf=dp_gq
                )
                if False:  # TODO might need this later
                    coeff_multipliers = None
                    p_gq[:] *= coeff_multipliers
                    dp_gq[:] *= coeff_multipliers
                if False:  # TODO might need this later
                    plan.get_transformed_interpolation_terms(
                        p_gq, i=i, fwd=True, inplace=True
                    )
                    plan.get_transformed_interpolation_terms(
                        dp_gq, i=i, fwd=True, inplace=True
                    )
                p_gq.shape = a_g.shape + (plan.nalpha,)
                dp_gq.shape = a_g.shape + (plan.nalpha,)
                self.fft_obj.fill_vj_feature_(feat_sig[s, i], p_gq)
                self.fft_obj.fill_vj_feature_(dfeat_sjg[s, i], dp_gq)
        # TODO version i features
        feat_sig[:] *= plan.nspin

        if tau_sg is None:  # GGA
            vfeat_sig = self.cider_kernel.calculate(
                e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg, feat_sig
            )
        else:  # MGGA
            vfeat_sig = self.cider_kernel.calculate(
                e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg, tau_sg, dedtau_sg, feat_sig
            )

        vfeat_sig[:] *= plan.nspin
        f_sgq[:] = 0.0
        for s in range(nspin):
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
                if tau_sg is not None:
                    dedtau_sg[s] += da_g[2] * tmp
                # dedrho_sxg[s, 0] += da_g[0] * tmp
                # dedrho_sxg[s, 1:4] += 2 * da_g[1] * rho_sxg[s, 1:4] * tmp
                # if tau_sg is not None:
                #    dedrho_sxg[s, 4] += da_g[2] * tmp
        dedarg_sg, dedfun_sg = self.call_bwd(arg_sg, fun_sg)

        for i, term in enumerate(darg_sg):
            term[:] *= dedarg_sg
            term[:] += dedfun_sg * dfun_sg[i]
        for s in range(plan.nspin):
            v_sg[s] += darg_sg[0][s]
            dedsigma_xg[2 * s] += darg_sg[1][s]
            # if tau_sg is not None:
            #     dedtau_sg[s] += darg_sg[2][s]

        dedrho_sxg[:, 0] += v_sg
        dedrho_sxg[:, 1:4] += 2 * rho_sxg[:, 1:4] * dedsigma_xg[::2]
        if plan.nspin == 2:
            dedrho_sxg[:, 1:4] += rho_sxg[::-1, 1:4] * dedsigma_xg[1]
        # if tau_sg is not None:
        #     dedrho_sxg[:, 4] += dedtau_sg

    def _distribute_to_cider_grid(self, n_xg):
        shape = (len(n_xg),)
        ndist_xg = self.distribution.block_zeros(shape)
        self.distribution.gd2block(n_xg, ndist_xg)
        return ndist_xg

    def _add_from_cider_grid(self, d_xg, ddist_xg):
        self.distribution.block2gd_add(ddist_xg, d_xg)

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        # TODO don't calculate sigma/dedsigma here
        # instead just compute the gradient and dedgrad
        # for sake of generality
        if self.type == "MGGA":
            if self.fixed_ke:
                taut_sG = self._fixed_taut_sG
                if not self._taut_gradv_init:
                    self._taut_gradv_init = True
                    # ensure initialization for calculation potential
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
        rho_sxg = get_rho_sxg(n_sg, self.grad_v, taut_sg)
        tmp_dedrho_sxg = np.zeros_like(rho_sxg)
        shape = rho_sxg.shape[:2]
        rho_sxg.shape = (rho_sxg.shape[0] * rho_sxg.shape[1],) + rho_sxg.shape[-3:]
        rho_sxg = self._distribute_to_cider_grid(rho_sxg)
        rho_sxg.shape = shape + rho_sxg.shape[-3:]
        dedrho_sxg = np.zeros_like(rho_sxg)
        _e = self._distribute_to_cider_grid(np.zeros_like(e_g)[None, :])[0]
        e_g[:] = 0.0
        self.process_cider(_e, rho_sxg, dedrho_sxg)
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

    def process_cider(*args, **kwargs):
        raise NotImplementedError


class CiderGGA(_FastCiderBase, GGA):
    def __init__(self, cider_kernel, **kwargs):
        if cider_kernel.mlfunc.settings.sl_settings.level != "GGA":
            raise ValueError("CiderGGA only supports GGA functionals!")
        _FastCiderBase.__init__(self, cider_kernel, **kwargs)

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
        _FastCiderBase.set_grid_descriptor(self, gd)

    def process_cider(self, e_g, rho_sxg, dedrho_sxg):
        self.calc_cider(e_g, rho_sxg, dedrho_sxg)

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac


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
    return rho_sxg


def add_gradient_correction(grad_v, dedrho_sxg, v_sg):
    nspin = len(v_sg)
    vv_g = np.empty_like(v_sg[0])
    for v in range(3):
        for s in range(nspin):
            grad_v[v](dedrho_sxg[s, v + 1], vv_g)
            v_sg[s] -= vv_g


class CiderMGGA(_FastCiderBase, MGGA):
    def __init__(self, cider_kernel, **kwargs):
        if cider_kernel.mlfunc.settings.sl_settings.level != "MGGA":
            raise ValueError("CiderMGGA only supports MGGA functionals!")
        _FastCiderBase.__init__(self, cider_kernel, **kwargs)

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
        _FastCiderBase.set_grid_descriptor(self, gd)

    def calculate_paw_correction(
        self, setup, D_sp, dEdD_sp=None, addcoredensity=True, a=None
    ):
        return 0

    def process_cider(self, e_g, rho_sxg, dedrho_sxg):
        self.calc_cider(e_g, rho_sxg, dedrho_sxg)
