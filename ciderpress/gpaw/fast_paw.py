import ctypes

import numpy as np
from gpaw.xc.mgga import MGGA

from ciderpress.gpaw.fast_atom_utils import FastAtomPASDWSlice, FastPASDWCiderKernel
from ciderpress.gpaw.fast_fft import CiderGGA, CiderMGGA, add_gradient_correction
from ciderpress.gpaw.nldf_interface import pwutil as libpwutil


class _FastPASDW_MPRoutines:
    has_paw = True

    def get_D_asp(self):
        return self.atomdist.to_work(self.dens.D_asp)

    def add_forces(self, F_av):
        nt_sg = self.dens.nt_xg
        redist = self._hamiltonian.xc_redistributor
        if redist is not None:
            nt_sg = redist.distribute(nt_sg)
        self.calculate_force_contribs(F_av, nt_sg)

    def _collect_paw_corrections(self):
        """
        For use in xc_tools, only after calculate_paw_correction
        is called. Collects potential using from_work
        """
        dH_asp_tmp = self.dH_asp_tmp
        E_a_tmp = self.E_a_tmp
        dH_asp_new = self.dens.setups.empty_atomic_matrix(
            self.nspin, self.atom_partition
        )
        E_a_new = self.atom_partition.arraydict([(1,)] * len(self.dens.setups), float)
        for a, E in E_a_tmp.items():
            E_a_new[a][:] = E
            dH_asp_new[a][:] = dH_asp_tmp[a]
        dist = self.atomdist
        self.E_a_tmp = dist.from_work(E_a_new)
        self.dH_asp_tmp = dist.from_work(dH_asp_new)

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

    def write_augfeat_to_rbuf(self, s, pot=False, get_force=False):
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
            (atom_slices[a].num_funcs, self._plan.nalpha)
            for a in range(len(self.setups))
        ]
        sizes = np.array([s[0] * s[1] for s in shapes])
        rank_a = self.atom_partition.rank_a
        comm = self.atom_partition.comm
        if self.fft_obj.has_mpi:
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
                c_iq = self.c_asiq[a][s]
                if self.pasdw_ovlp_fit:
                    sendbuf[rtot : rtot + sizes[a]] = np.ascontiguousarray(
                        # c_abi[a].dot(global_slices[a].sinv_pf.T)
                        global_slices[a].sinv_pf.dot(c_iq)
                    ).ravel()
                else:
                    sendbuf[rtot : rtot + sizes[a]] = np.ascontiguousarray(c_iq).ravel()
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
        else:
            max_size = np.max([s[0] * s[1] for s in shapes])
            buf = np.empty(max_size, dtype=np.float64)
        self.timer.stop()

        for a in range(len(self.setups)):
            atom_slice = atom_slices[a]
            if self.fft_obj.has_mpi:
                coefs_iq = np.ndarray(shapes[a], buffer=recvbuf[alocs[a] :])
            else:
                coefs_iq = np.ndarray(shapes[a], buffer=buf)
                coefs_iq = np.ndarray(shapes[a], buffer=buf)
                if comm.rank == rank_a[a]:
                    coefs_iq[:] = self.c_asiq[a][s]
                comm.broadcast(coefs_iq, rank_a[a])
            if atom_slice is not None and atom_slices[a].rad_g.size > 0:
                self.timer.start("funcs")
                funcs_ig = atom_slice.get_funcs()
                funcs_gq = np.dot(funcs_ig.T, coefs_iq)
                self.timer.stop()
                self.timer.start("paw2grid")
                self.fft_obj.add_paw2grid(funcs_gq, atom_slice.indset)
                self.timer.stop()
        self.timer.stop()

    def interpolate_rbuf_onto_atoms(self, s, pot=False):
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
            (atom_slices[a].num_funcs, self._plan.nalpha)
            for a in range(len(self.setups))
        ]

        rank_a = self.atom_partition.rank_a
        comm = self.atom_partition.comm
        coefs_a_iq = {}
        for a in range(len(self.setups)):
            atom_slice: FastAtomPASDWSlice = atom_slices[a]
            if atom_slice.rad_g.size > 0:
                funcs_ig = atom_slice.get_funcs()
                funcs_gq = np.empty((funcs_ig.shape[1], self._plan.nalpha))
                self.fft_obj.set_grid2paw(
                    funcs_gq,
                    atom_slice.indset,
                )
                coefs_iq = np.dot(funcs_ig, funcs_gq)
                coefs_iq[:] *= atom_slice.dv
            else:
                coefs_iq = np.zeros(shapes[a])
            coefs_a_iq[a] = coefs_iq
        for a in range(len(self.setups)):
            coefs_iq = coefs_a_iq[a]
            comm.sum(coefs_iq, rank_a[a])
            if rank_a[a] == comm.rank:
                assert coefs_iq is not None
                if self.pasdw_ovlp_fit:
                    self.c_asiq[a][s] = global_slices[a].sinv_pf.T.dot(coefs_iq)
                else:
                    self.c_asiq[a][s] = coefs_iq
        self.timer.stop()

    def interpolate_rbuf_onto_atoms_grad(self, s, pot=False):
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
            (atom_slices[a].num_funcs, self._plan.nalpha)
            for a in range(len(self.setups))
        ]

        rank_a = self.atom_partition.rank_a
        comm = self.atom_partition.comm
        coefs_a_viq = {}
        for a in range(len(self.setups)):
            atom_slice: FastAtomPASDWSlice = atom_slices[a]
            if atom_slice.rad_g.size > 0:
                funcs_vig = atom_slice.get_grads()
                funcs_gq = np.empty((funcs_vig.shape[2], self._plan.nalpha))
                self.fft_obj.set_grid2paw(
                    funcs_gq,
                    atom_slice.indset,
                )
                coefs_viq = np.dot(funcs_vig, funcs_gq)
                coefs_viq[:] *= atom_slice.dv
            else:
                coefs_viq = np.zeros((3,) + shapes[a])
            coefs_a_viq[a] = coefs_viq
        for a in range(len(self.setups)):
            coefs_viq = coefs_a_viq[a]
            comm.sum(coefs_viq, rank_a[a])
            if rank_a[a] == comm.rank:
                assert coefs_viq is not None
                for v in range(3):
                    if self.pasdw_ovlp_fit:
                        self._save_v_avsiq[a][v, s] = global_slices[a].sinv_pf.T.dot(
                            coefs_viq[v]
                        )
                    else:
                        self._save_v_avsiq[a][v, s] = coefs_viq[v]
        self.timer.stop()

    def set_fft_work(self, s, pot=False):
        self.write_augfeat_to_rbuf(s, pot=pot)

    def get_fft_work(self, s, pot=False, get_grad_terms=False):
        self.interpolate_rbuf_onto_atoms(s, pot=pot)
        if get_grad_terms:
            if s == 0:
                self._save_v_avsiq = {}
                for a, c_siq in self.c_asiq.items():
                    self._save_v_avsiq[a] = np.zeros((3,) + c_siq.shape)
            self.interpolate_rbuf_onto_atoms_grad(s, pot=pot)

    def _set_paw_terms(self):
        self.timer.start("PAW TERMS")
        D_asp = self.get_D_asp()
        if not (D_asp.partition.rank_a == self.atom_partition.rank_a).all():
            raise ValueError("rank_a mismatch")
        self.c_asiq, self.df_asgLq = self.paw_kernel.calculate_paw_feat_corrections(
            self.setups, D_asp
        )
        self._save_c_asiq = {k: v.copy() for k, v in self.c_asiq.items()}
        self.D_asp = D_asp
        self.timer.stop()

    def _calculate_paw_energy_and_potential(self):
        self.timer.start("PAW ENERGY")
        self.c_asiq, deltaE, deltaV = self.paw_kernel.calculate_paw_feat_corrections(
            self.setups, self.D_asp, D_asiq=self.c_asiq, df_asgLq=self.df_asgLq
        )
        self.E_a_tmp = deltaE
        self.deltaV = deltaV
        self.timer.stop()

    def _get_dH_asp(self):
        self.timer.start("PAW POTENTIAL")
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

    @property
    def is_initialized(self):
        if (
            (self._plan is None)
            or (self._plan.nspin != self.nspin)
            or (self.setups is None)
            or (self.atom_slices_s is None)
        ):
            return False
        else:
            return True


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
        MGGA.add_forces(self, F_av)
        _FastPASDW_MPRoutines.add_forces(self, F_av)

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
        MGGA.add_forces(self, F_av)
        _FastPASDW_MPRoutines.add_forces(self, F_av)

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

    def _get_taut(self, n_sg):
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
            taut_sG = None
        self.timer.stop()
        return taut_sg, dedtaut_sg, taut_sG

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
                self.ekin -= self.wfs.gd.integrate(
                    self.dedtaut_sG[s] * (taut_sG[s] - self.tauct_G / self.wfs.nspins)
                )

    def calculate_force_contribs(self, F_av, n_sg):
        e_g = np.zeros(n_sg.shape[1:])
        Ftmp_av = np.zeros_like(F_av)
        taut_sg, dedtaut_sg, taut_sG = self._get_taut(n_sg)
        _e, rho_sxg, dedrho_sxg = self._get_cider_inputs(n_sg, taut_sg)
        e_g[:] = 0.0
        fs = self.calc_cider(_e, rho_sxg, dedrho_sxg, compute_force=True)
        for a, f in fs.items():
            Ftmp_av[a] = f
        self.world.sum(Ftmp_av, 0)
        if self.world.rank == 0:
            F_av[:] += Ftmp_av
