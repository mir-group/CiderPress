import ctypes

import numpy as np
from gpaw.xc.mgga import MGGA

from ciderpress.gpaw.cider_paw import _CiderPASDW_MPRoutines
from ciderpress.gpaw.fast_atom_utils import FastAtomPASDWSlice, FastPASDWCiderKernel
from ciderpress.gpaw.fast_fft import (
    CiderGGA,
    CiderMGGA,
    add_gradient_correction,
    get_rho_sxg,
)
from ciderpress.gpaw.nldf_interface import pwutil as libpwutil


class _FastPASDW_MPRoutines(_CiderPASDW_MPRoutines):
    def initialize_paw_kernel(self, cider_kernel_inp, Nalpha_atom, encut_atom):
        self.paw_kernel = FastPASDWCiderKernel(
            cider_kernel_inp,
            self._plan,
            self.gd,
            self.cut_xcgrid,
        )
        self.paw_kernel.initialize(
            self.dens, self.atomdist, self.atom_partition, self.setups
        )
        self.paw_kernel.initialize_more_things(self.setups)
        self.paw_kernel.interpolate_dn1 = self.interpolate_dn1
        self.atom_slices_s = None

    def _setup_atom_slices(self):
        self.timer.start("ATOM SLICE SETUP")
        self.atom_slices_s = {}
        self.atom_slices_a = {}
        self.global_slices_s = {}
        self.global_slices_a = {}
        rank_a = self.atom_partition.rank_a
        comm = self.atom_partition.comm
        for a in range(len(self.setups)):
            args = [
                self.gd,
                self.spos_ac[a],
                self.setups[a].ps_setup,
                self.fft_obj,
            ]
            kwargs = dict(
                rmax=self.setups[a].ps_setup.rcut_func,
                sphere=True,
                store_funcs=self.pasdw_store_funcs,
            )
            self.atom_slices_s[a] = FastAtomPASDWSlice.from_gd_and_setup(
                *args, **kwargs
            )
            if rank_a[a] == comm.rank and self.pasdw_ovlp_fit:
                self.global_slices_s[a] = FastAtomPASDWSlice.from_gd_and_setup(
                    *args, is_global=True, ovlp_fit=True, **kwargs
                )
            args[2] = self.setups[a].pa_setup
            kwargs["rmax"] = self.setups[a].pa_setup.rcut_func
            self.atom_slices_a[a] = FastAtomPASDWSlice.from_gd_and_setup(
                *args, **kwargs
            )
            if rank_a[a] == comm.rank and self.pasdw_ovlp_fit:
                self.global_slices_a[a] = FastAtomPASDWSlice.from_gd_and_setup(
                    *args, is_global=True, ovlp_fit=True, **kwargs
                )
        self.timer.stop()

    def write_augfeat_to_rbuf(self, c_abi, pot=False):
        self.timer.start("PAW TO GRID")
        if self.atom_slices_s is None:
            self._setup_atom_slices()

        self.timer.start("setup")
        if pot:
            atom_slices = self.atom_slices_a
            global_slices = self.global_slices_a
        else:
            atom_slices = self.atom_slices_s
            global_slices = self.global_slices_s

        shapes = [
            (self._plan.nalpha, atom_slices[a].num_funcs)
            for a in range(len(self.setups))
        ]
        sizes = np.array([s[0] * s[1] for s in shapes])
        rank_a = self.atom_partition.rank_a
        comm = self.atom_partition.comm
        atoms = [[] for r in range(comm.size)]
        for a in range(len(self.setups)):
            atoms[rank_a[a]].append(a)
        alocs = {}
        rlocs = []
        rsizes = []
        tot = 0
        for r in range(comm.size):
            rlocs.append(tot)
            for a in atoms[r]:
                alocs[a] = tot
                tot += sizes[a]
            rsizes.append(tot - rlocs[-1])
        sendbuf = np.empty(rsizes[comm.rank], dtype=np.float64)
        rtot = 0
        for a in atoms[comm.rank]:
            # assert c_abi[a].flags.c_contiguous
            if self.pasdw_ovlp_fit:
                sendbuf[rtot : rtot + sizes[a]] = np.ascontiguousarray(
                    c_abi[a].dot(global_slices[a].sinv_pf.T)
                ).ravel()
            else:
                sendbuf[rtot : rtot + sizes[a]] = np.ascontiguousarray(c_abi[a]).ravel()
            rtot += sizes[a]
        np.asarray([rtot] * comm.size)
        np.asarray([0] * comm.size)
        recvbuf = np.empty(tot, dtype=np.float64)
        self.timer.stop()
        self.timer.start("comm")
        rsizes = np.asarray(rsizes, order="C", dtype=np.int32)
        rlocs = np.asarray(rlocs, order="C", dtype=np.int32)
        libpwutil.ciderpw_all_gatherv_from_gpaw(
            ctypes.py_object(comm.get_c_object()),
            sendbuf.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(rtot),
            recvbuf.ctypes.data_as(ctypes.c_void_p),
            rsizes.ctypes.data_as(ctypes.c_void_p),
            rlocs.ctypes.data_as(ctypes.c_void_p),
        )
        # comm.alltoallv(
        #     sendbuf, scounts, sdispls, recvbuf, np.asarray(rsizes), np.asarray(rlocs)
        # )
        self.timer.stop()
        # sizes = np.array([s[0] * s[1] for s in shapes])
        # locs = np.asarray(np.append([0], np.cumsum(sizes[:-1])), order="C")
        # max_size = np.max([s[0] * s[1] for s in shapes])
        # buf = np.empty(np.sum(sizes), dtype=np.float64)
        # buf = np.empty(max_size, dtype=np.float64)

        for a in range(len(self.setups)):
            atom_slice = atom_slices[a]
            coefs_bi = np.ndarray(shapes[a], buffer=recvbuf[alocs[a] :])
            # coefs_bi = np.ndarray(shapes[a], buffer=buf)
            # coefs_bi = np.ndarray(shapes[a], buffer=buf)
            # if comm.rank == rank_a[a]:
            #     coefs_bi[:] = c_abi[a]
            # self.timer.start("comm")
            # comm.broadcast(coefs_bi, rank_a[a])
            # self.timer.stop()
            if atom_slice is not None and atom_slices[a].rad_g.size > 0:
                self.timer.start("funcs")
                funcs_ig = atom_slice.get_funcs()
                funcs_gb = np.dot(funcs_ig.T, coefs_bi.T)
                self.timer.stop()
                self.timer.start("paw2grid")
                self.fft_obj.add_paw2grid(
                    funcs_gb,
                    atom_slice.indset,
                )
                self.timer.stop()
        self.timer.stop()

    def interpolate_rbuf_onto_atoms(self, pot=False):
        self.timer.start("GRID TO PAW")
        if self.atom_slices_s is None:
            self._setup_atom_slices()

        if pot:
            atom_slices = self.atom_slices_s
            global_slices = self.global_slices_s
        else:
            atom_slices = self.atom_slices_a
            global_slices = self.global_slices_a

        shapes = [
            (self._plan.nalpha, atom_slices[a].num_funcs)
            for a in range(len(self.setups))
        ]
        # max_size = np.max([s[0] * s[1] for s in shapes])
        # buf = np.empty(max_size, dtype=np.float64)

        c_abi = {}
        rank_a = self.atom_partition.rank_a
        comm = self.atom_partition.comm
        for a in range(len(self.setups)):
            atom_slice: FastAtomPASDWSlice = atom_slices[a]
            if atom_slice.rad_g.size > 0:
                funcs_ig = atom_slice.get_funcs()
                funcs_gb = np.empty((funcs_ig.shape[1], self._plan.nalpha))
                self.fft_obj.set_grid2paw(
                    funcs_gb,
                    atom_slice.indset,
                )
                coefs_bi = np.dot(funcs_gb.T, funcs_ig.T)
                coefs_bi[:] *= atom_slice.dv
            else:
                coefs_bi = np.zeros(shapes[a])
            comm.sum(coefs_bi, rank_a[a])
            if rank_a[a] == comm.rank:
                assert coefs_bi is not None
                if self.pasdw_ovlp_fit:
                    c_abi[a] = coefs_bi.dot(global_slices[a].sinv_pf)
                else:
                    c_abi[a] = coefs_bi

        self.timer.stop()
        return c_abi

    def set_fft_work(self, s, pot=False):
        self.write_augfeat_to_rbuf(self.c_sabi[s], pot=pot)

    def get_fft_work(self, s, pot=False):
        self.c_sabi[s] = self.interpolate_rbuf_onto_atoms(pot=pot)

    def _set_paw_terms(self):
        self.timer.start("PAW TERMS")
        D_asp = self.get_D_asp()
        if not (D_asp.partition.rank_a == self.atom_partition.rank_a).all():
            raise ValueError("rank_a mismatch")
        self.c_asiq, self.df_asgLq = self.paw_kernel.calculate_paw_feat_corrections(
            self.setups, D_asp
        )
        if len(self.c_asiq.keys()) == 0:
            self.c_sabi = {s: {} for s in range(self.nspin)}
        else:
            self.c_sabi = {
                s: {a: self.c_asiq[a][s].T for a in self.c_asiq}
                for s in range(self.nspin)
            }
        self.D_asp = D_asp
        self.timer.stop()

    def _calculate_paw_energy_and_potential(self):
        self.timer.start("PAW ENERGY")
        if len(self.c_sabi.keys()) == 0:
            self.c_asiq = {}
        else:
            if not hasattr(self, "c_asiq"):
                self.c_asiq = {}
            for a in self.c_sabi[0]:
                self.c_asiq[a] = np.stack(
                    [self.c_sabi[s][a].T for s in range(self.nspin)]
                )
        self.c_asiq, deltaE, deltaV = self.paw_kernel.calculate_paw_feat_corrections(
            self.setups, self.D_asp, D_asiq=self.c_asiq, df_asgLq=self.df_asgLq
        )
        self.E_a_tmp = deltaE
        self.deltaV = deltaV
        if len(self.c_sabi.keys()) == 0:
            self.c_sabi = {s: {} for s in range(self._plan.nspin)}
        else:
            self.c_sabi = {
                s: {a: self.c_asiq[a][s].T for a in self.c_asiq}
                for s in range(self.nspin)
            }
        self.timer.stop()

    def _get_dH_asp(self):
        self.timer.start("PAW POTENTIAL")
        if len(self.c_sabi.keys()) == 0:
            self.c_asiq = {}
        else:
            if not hasattr(self, "c_asiq"):
                self.c_asiq = {}
            for a in self.c_sabi[0]:
                self.c_asiq[a] = np.stack(
                    [self.c_sabi[s][a].T for s in range(self.nspin)]
                )
        dH_asp = self.paw_kernel.calculate_paw_feat_corrections(
            self.setups, self.D_asp, vc_asiq=self.c_asiq
        )
        for a in self.D_asp.keys():
            dH_asp[a] += self.deltaV[a]
        self.dH_asp_tmp = dH_asp
        self.timer.stop()

    def calculate_paw_correction(
        self, setup, D_sp, dEdD_sp=None, addcoredensity=True, a=None
    ):
        if a is None:
            raise ValueError("Atom index a not provided.")
        if dEdD_sp is not None:
            dEdD_sp += self.dH_asp_tmp[a]
        return self.E_a_tmp[a]

    def initialize_more_things(self):
        self.setups = self._hamiltonian.setups
        self.atomdist = self._hamiltonian.atomdist
        self.atom_partition = self.get_D_asp().partition

        encut_atom = self.encut
        Nalpha_atom = self.Nalpha
        while encut_atom < self.encut_atom_min:
            encut_atom *= self.lambd
            Nalpha_atom += 1

        self.initialize_paw_kernel(self.cider_kernel, Nalpha_atom, encut_atom)


class CiderGGAPASDW(_FastPASDW_MPRoutines, CiderGGA):
    def __init__(self, cider_kernel, **kwargs):
        CiderGGA.__init__(
            self,
            cider_kernel,
            **kwargs,
        )
        defaults_list = {
            "interpolate_dn1": False,  # bool
            "encut_atom_min": 8000.0,  # float
            "cut_xcgrid": False,  # bool
            "pasdw_ovlp_fit": True,
            "pasdw_store_funcs": False,
        }
        settings = {}
        for k, v in defaults_list.items():
            settings[k] = kwargs.get(k)
            if settings[k] is None:
                settings[k] = defaults_list[k]
        self.__dict__.update(settings)
        self.k2_k = None
        self.setups = None

    def todict(self):
        d = super(CiderGGAPASDW, self).todict()
        d["_cider_type"] = "CiderGGAPASDW"
        d["xc_params"]["pasdw_store_funcs"] = self.pasdw_store_funcs
        d["xc_params"]["pasdw_ovlp_fit"] = self.pasdw_ovlp_fit
        return d

    def add_forces(self, F_av):
        raise NotImplementedError
        CiderGGA.add_forces(self, F_av)
        _CiderPASDW_MPRoutines.add_forces(self, F_av)

    def stress_tensor_contribution(self, n_sg):
        raise NotImplementedError
        nspins = len(n_sg)
        stress_vv = super(CiderGGAPASDW, self).stress_tensor_contribution(n_sg)
        tmp_vv = np.zeros((3, 3))
        for s in range(nspins):
            tmp_vv += self.construct_grad_terms(
                self.dedtheta_sag[s],
                self.c_sabi[s],
                aug=False,
                stress=True,
            )
            tmp_vv += self.construct_grad_terms(
                self.Freal_sag[s],
                self.dedq_sabi[s],
                aug=True,
                stress=True,
            )
        self.world.sum(tmp_vv)
        stress_vv[:] += tmp_vv
        self.gd.comm.broadcast(stress_vv, 0)
        return stress_vv

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac
        self.atom_slices_s = None

    def initialize(self, density, hamiltonian, wfs):
        self.dens = density
        self._hamiltonian = hamiltonian
        self.timer = wfs.timer
        self.world = wfs.world
        self.setups = None
        self.gdfft = None
        self.pwfft = None
        self.rbuf_ag = None
        self._check_parallelization(wfs)

    def calc_cider(self, *args, **kwargs):
        if (self.setups is None) or (self.atom_slices_s is None):
            self._setup_plan()
            self.fft_obj.initialize_backend()
            self.initialize_more_things()
        CiderGGA.calc_cider(self, *args, **kwargs)


class CiderMGGAPASDW(_FastPASDW_MPRoutines, CiderMGGA):
    def __init__(self, cider_kernel, **kwargs):
        CiderMGGA.__init__(
            self,
            cider_kernel,
            **kwargs,
        )
        defaults_list = {
            "interpolate_dn1": False,  # bool
            "encut_atom_min": 8000.0,  # float
            "cut_xcgrid": False,  # bool
            "pasdw_ovlp_fit": True,
            "pasdw_store_funcs": False,
        }
        settings = {}
        for k, v in defaults_list.items():
            settings[k] = kwargs.get(k)
            if settings[k] is None:
                settings[k] = defaults_list[k]
        self.__dict__.update(settings)
        self.k2_k = None
        self.setups = None

    def todict(self):
        d = super(CiderMGGAPASDW, self).todict()
        d["_cider_type"] = "CiderMGGAPASDW"
        d["xc_params"]["pasdw_store_funcs"] = self.pasdw_store_funcs
        d["xc_params"]["pasdw_ovlp_fit"] = self.pasdw_ovlp_fit
        return d

    def add_forces(self, F_av):
        raise NotImplementedError
        MGGA.add_forces(self, F_av)
        _CiderPASDW_MPRoutines.add_forces(self, F_av)

    def stress_tensor_contribution(self, n_sg):
        raise NotImplementedError
        nspins = len(n_sg)
        stress_vv = super(CiderMGGAPASDW, self).stress_tensor_contribution(n_sg)
        tmp_vv = np.zeros((3, 3))
        for s in range(nspins):
            tmp_vv += self.construct_grad_terms(
                self.dedtheta_sag[s],
                self.c_sabi[s],
                aug=False,
                stress=True,
            )
            tmp_vv += self.construct_grad_terms(
                self.Freal_sag[s],
                self.dedq_sabi[s],
                aug=True,
                stress=True,
            )
        self.world.sum(tmp_vv)
        stress_vv[:] += tmp_vv
        self.gd.comm.broadcast(stress_vv, 0)
        return stress_vv

    def set_positions(self, spos_ac):
        MGGA.set_positions(self, spos_ac)
        self.spos_ac = spos_ac
        self.atom_slices_s = None

    def initialize(self, density, hamiltonian, wfs):
        MGGA.initialize(self, density, hamiltonian, wfs)
        self.dens = density
        self._hamiltonian = hamiltonian
        self.timer = wfs.timer
        self.world = wfs.world
        self.setups = None
        self.atom_slices_s = None
        self.gdfft = None
        self.pwfft = None
        self.rbuf_ag = None
        self.kbuf_ak = None
        self.theta_ak = None
        self._check_parallelization(wfs)

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        # TODO don't calculate sigma/dedsigma here
        # instead just compute the gradient and dedgrad
        # for sake of generality
        # TODO clearn approach?
        self.timer.start("KED SETUP")
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
                taut_G += 1.0 / self.wfs.nspins * self.tauct_G
                self.distribute_and_interpolate(taut_G, taut_g)

            dedtaut_sg = np.zeros_like(n_sg)
        else:
            taut_sg = None
            dedtaut_sg = None
        self.timer.stop()
        self.timer.start("Reorganize density")
        rho_sxg = get_rho_sxg(n_sg, self.grad_v, taut_sg)
        self.nspin = len(rho_sxg)
        tmp_dedrho_sxg = np.zeros_like(rho_sxg)
        shape = rho_sxg.shape[:2]
        rho_sxg.shape = (rho_sxg.shape[0] * rho_sxg.shape[1],) + rho_sxg.shape[-3:]
        rho_sxg = self._distribute_to_cider_grid(rho_sxg)
        rho_sxg.shape = shape + rho_sxg.shape[-3:]
        dedrho_sxg = np.zeros_like(rho_sxg)
        _e = self._distribute_to_cider_grid(np.zeros_like(e_g)[None, :])[0]
        e_g[:] = 0.0
        self.timer.stop()
        self.process_cider(_e, rho_sxg, dedrho_sxg)
        self.timer.start("Reorganize potential")
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
                self.ekin -= self.wfs.gd.integrate(
                    self.dedtaut_sG[s] * (taut_sG[s] - self.tauct_G / self.wfs.nspins)
                )
        self.timer.stop()

    def calc_cider(self, *args, **kwargs):
        if (self.setups is None) or (self.atom_slices_s is None):
            self._setup_plan()
            self.fft_obj.initialize_backend()
            self.initialize_more_things()
        CiderMGGA.calc_cider(self, *args, **kwargs)
