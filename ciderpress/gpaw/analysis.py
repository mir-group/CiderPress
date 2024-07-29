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
from ase.dft.bandgap import bandgap
from ase.units import Bohr, Ha
from gpaw.occupations import FixedOccupationNumbers, OccupationNumberCalculator
from gpaw.pw.density import ReciprocalSpaceDensity
from gpaw.sphere.lebedev import Y_nL, weight_n
from gpaw.xc.gga import calculate_sigma

from ciderpress.gpaw.atom_analysis_utils import (
    calculate_paw_cider_features_p1,
    calculate_paw_cider_features_p2,
    calculate_paw_cider_features_p2_noderiv,
    calculate_paw_sl_features,
    calculate_paw_sl_features_deriv,
    get_features_with_sl_noderiv,
    get_features_with_sl_part,
)
from ciderpress.gpaw.cider_paw import (
    CiderGGA,
    CiderGGAHybridKernel,
    CiderGGAPASDW,
    CiderMGGA,
    CiderMGGAHybridKernel,
    CiderMGGAPASDW,
)


def calc_fixed(bd, fref_sqn, f_qn):
    if bd.nbands == fref_sqn.shape[2]:
        nspins = fref_sqn.shape[0]
        for u, f_n in enumerate(f_qn):
            s = u % len(fref_sqn)
            q = u // nspins
            bd.distribute(fref_sqn[s, q], f_n)
    else:
        raise NotImplementedError


def calculate_single_orbital_matrix(self, D_sii, kpt, a, n):
    if kpt.rho_MM is not None:
        raise NotImplementedError
    else:
        if self.collinear:
            P_i = kpt.projections[a][n]
            assert P_i.ndim == 1
            D_sii[kpt.s] += np.dot(P_i.conj(), P_i).real
        else:
            raise NotImplementedError

    if hasattr(kpt, "c_on"):
        raise NotImplementedError


def calculate_single_orbital_atomic_density_matrix(self, D_aop, p_o, nspin):
    """
    Get the atomic density matrices for the band given
    by index tuple p
    Args:
        self: GPAW Wavefunctions object
        D_asp: stores the single-orbital density matrix
        p: (s, k, n) tuple
    """
    for a, D_op in D_aop.items():
        D_op[:] = 0.0
        for o, p in enumerate(p_o):
            # Get the apprropriate k-point index on this rank
            rank, q = self.kd.who_has(p[1])
            if self.kd.comm.rank == rank:
                s = p[0]
                u = q * nspin + s
                kpt = self.kpt_u[u]
                assert kpt.s == s
                assert kpt.k == p[1]
                D_op[o] = self.get_orbital_density_matrix(a, kpt, p[2])[s]
        self.kptband_comm.sum(D_op)

    # self.symmetrize_atomic_density_matrices(D_aop)


def get_empty_atomic_matrix(dens, norb):
    return dens.setups.empty_atomic_matrix(norb, dens.atom_partition)


class FixedGOccupationNumbers(FixedOccupationNumbers):
    """
    Same as FixedOccupationNumbers but with a perturb_occupation_
    functionn for testing incremental occupation numbers.
    """

    def perturb_occupation_(self, p, delta):
        """
        Change the occupation number of a band at a k-point
        Args:
            p: (s, k, n) or (s, n). k is ignored
            delta: amount to change occupation number
        """
        if len(p) == 3:
            p = (p[0], p[2])
        elif len(p) != 2:
            raise ValueError("Length of index tuple must be 2 or 3")
        self.f_sn[p] += delta


class FixedKOccupationNumbers(OccupationNumberCalculator):
    """
    Same as FixedOccupationNumbers but the occupations are k-point specific.
    """

    extrapolate_factor = 0.0

    def __init__(self, f_skn, kpt_indices, parallel_layout=None, weight_sk=None):
        """
        Args:
            f_skn: Occupations numbers
            parallel_layout: communictors for occupations
        """
        OccupationNumberCalculator.__init__(self, parallel_layout)
        self.kpt_indices = np.array(kpt_indices)
        self.f_skn = np.array(f_skn)
        self.weight_sk = weight_sk

    def _calculate(self, nelectrons, eig_qn, weight_q, f_qn, fermi_level_guess=np.nan):
        calc_fixed(self.bd, self.f_skn[:, self.kpt_indices, :], f_qn)

        return np.inf, 0.0

    @classmethod
    def from_calculator(cls, calc):
        """
        Initialized from an existing Calculator
        Args:
            calc: GPAW Calculator

        Returns:
            Instance of cls
        """
        calc.wfs.calculate_occupation_numbers()
        if calc.wfs.collinear and calc.wfs.nspins == 1:
            degeneracy = 2
        else:
            degeneracy = 1
        f_un = np.array([kpt.f_n for kpt in calc.wfs.kpt_u])
        f_skn = calc.wfs.kd.collect(f_un, True)
        w_u = np.array([kpt.weightk for kpt in calc.wfs.kpt_u])[:, np.newaxis]
        weight_sk = calc.wfs.kd.collect(w_u, True) * degeneracy
        f_skn /= weight_sk
        weight_sk = weight_sk[..., 0]
        return cls(
            f_skn,
            calc.wfs.kd.get_indices(),
            parallel_layout=calc.wfs.occupations.parallel_layout,
            weight_sk=weight_sk,
        )

    def perturb_occupation_(self, p, delta, div_by_wt=False):
        """
        Change the occupation number of a band at a k-point
        Args:
            p: (s, k, n) tuple
            delta: amount to change occupation number
            div_by_wt: If True, self.weight_sk must not be None.
                Divide delta by the k-point weight so that
                finite derivatives with respect to occupation
                match analytical derivatives of changing electron number.
        """
        if div_by_wt:
            if self.weight_sk is None:
                raise ValueError("weight_sk must be set if div_by_wt=True")
            delta /= self.weight_sk[p[:2]]
        self.f_skn[p] += delta

    def todict(self):
        return {"name": "fixedk", "f_skn": self.f_skn}


def run_constant_occ_calculation_(calc):
    calc.set(occupations=FixedKOccupationNumbers.from_calculator(calc))
    etot = calc.get_potential_energy()
    return etot / Ha


def perturb_occ_(calc, p, delta):
    occ = FixedKOccupationNumbers.from_calculator(calc)
    occ.perturb_occupation_(p, delta, div_by_wt=True)
    calc.set(charge=calc.density.charge - delta)
    calc.calculate(properties=["energy"])
    calc.set(occupations=occ)
    calc.calculate(properties=["energy"])


def evaluate_perturbed_energy(calc, p, delta=1e-6, store_initial=False, reset=True):
    """
    def _get_energy():
        dens = calc.density
        #e_entropy = calc.wfs.calculate_occupation_numbers(dens.fixed)
        calc.wfs.calculate_occupation_numbers()
        e_entropy = 0.0
        dens.update(calc.wfs)
        dens.initialize_from_wavefunctions(calc.wfs)
        calc.hamiltonian.initialize()
        dens.calculate_pseudo_density(calc.wfs)
        dens.interpolate_pseudo_density()
        dens.calculate_pseudo_charge()
        calc.hamiltonian.update(
            dens,
            wfs=calc.wfs,
            kin_en_using_band=True,
        )
        return calc.hamiltonian.get_energy(
            e_entropy, calc.wfs, kin_en_using_band=True
        )
    """

    def _get_energy():
        calc.calculate(properties=["energy"])
        return calc.get_potential_energy() / Ha

    if store_initial:
        e_init = _get_energy()
    calc.wfs.occupations.perturb_occupation_(p, delta, div_by_wt=True)
    e_final = _get_energy()
    if reset:
        calc.wfs.occupations.perturb_occupation_(p, -1 * delta, div_by_wt=True)
    if store_initial:
        return e_final, e_init
    else:
        return e_final


def get_homo_lumo_fd_(calc, delta=1e-6):
    """
    Calculate finite difference derivatives of the
    energy wrt the VBM and CBM occupation, for comparison
    to the band gap.

    Args:
        calc: GPAW Calculator object
        delta: finite difference perturbation

    Returns:
        gap, e_vbm, e_cbm dE/df_vbm, dE/df_cbm, p_vbm, p_cbm
        All units in Ha
    """
    delta = np.abs(delta)
    gap, p_vbm, p_cbm = bandgap(calc)
    e0 = run_constant_occ_calculation_(calc)
    gap = gap / Ha
    ep_vbm, e_init = evaluate_perturbed_energy(
        calc, p_vbm, delta=-1 * delta, store_initial=True
    )
    ep_cbm = evaluate_perturbed_energy(calc, p_cbm, delta=delta, store_initial=False)
    e_vbm = calc.get_eigenvalues(kpt=p_vbm[1], spin=p_vbm[0])[p_vbm[2]] / Ha
    e_cbm = calc.get_eigenvalues(kpt=p_cbm[1], spin=p_cbm[0])[p_cbm[2]] / Ha
    assert np.abs(e_init - e0) < 1e-6, e_init - e0
    dev = (e_init - ep_vbm) / delta
    dec = (ep_cbm - e_init) / delta
    return gap, e_vbm, e_cbm, dev, dec, p_vbm, p_cbm


def get_atom_feat_wt(functional):
    ae_feat, ps_feat = functional.calculate_paw_features()
    for a in range(len(ae_feat)):
        for feat, sgn in [(ae_feat[a], 1), (ps_feat[a], -1)]:
            # feat = np.einsum('sigL,nL->sign', feat, Y_nL)
            nspin, nfeat = feat.shape[:2]
            dv_g = functional.setups[a].xc_correction.rgd.dv_g
            weight_gn = dv_g * weight_n[:, np.newaxis]
            assert dv_g.size == feat.shape[-1]
            feat = feat.reshape(nspin, nfeat, -1)
            wt = weight_gn.reshape(-1)
            yield feat, sgn * wt


def get_atom_feat_wt_deriv(functional, DD_aop, p_o):
    ae_feat, ps_feat, ae_dfeat, ps_dfeat = functional.calculate_paw_features_deriv(
        DD_aop, p_o
    )
    norb = len(p_o)
    for a in range(len(ae_feat)):
        lst = [(ae_feat[a], ae_dfeat[a], 1), (ps_feat[a], ps_dfeat[a], -1)]
        for feat, dfeat, sgn in lst:
            nspin, nfeat = feat.shape[:2]
            dv_g = functional.setups[a].xc_correction.rgd.dv_g
            weight_gn = dv_g * weight_n[:, np.newaxis]
            assert dv_g.size == feat.shape[-1]
            feat = feat.reshape(nspin, nfeat, -1)
            dfeat = dfeat.reshape(norb, nfeat, -1)
            wt = weight_gn.reshape(-1)
            yield feat, dfeat, sgn * wt


def get_features_and_weights(calc, xc, screen_dens=True):
    vol = calc.atoms.get_volume()
    ae_feat = xc.get_features_on_grid(include_density=True)
    nspin, nfeat = ae_feat.shape[:2]
    size = np.prod(ae_feat.shape[2:])
    all_wt = np.ones(size) / size * vol / Bohr**3
    all_wt = all_wt.flatten()
    ae_feat = ae_feat.reshape(nspin, nfeat, -1)
    # hopefully save disk space and avoid negative density
    if screen_dens:
        cond = ae_feat[:, 0, :].sum(axis=0) > 1e-6
        ae_feat = ae_feat[..., cond]
        all_wt = all_wt[cond]
    assert all_wt.size == ae_feat.shape[-1]

    if xc.has_paw:
        for (
            ft,
            wt,
        ) in get_atom_feat_wt(xc):
            assert wt.shape[-1] == ft.shape[-1]
            ae_feat = np.append(ae_feat, ft, axis=-1)
            all_wt = np.append(all_wt, wt, axis=-1)

    if not isinstance(xc, _SLFeatMixin):
        ae_feat *= nspin
        # TODO make this safer
        ae_feat[:, 1, :] /= nspin ** (5.0 / 3)
        if ae_feat.shape[1] == 6:
            ae_feat[:, 2, :] /= nspin ** (5.0 / 3)
    return ae_feat, all_wt


def get_features_and_weights_deriv(
    calc, xc, p_j, drhodf_jxg, DD_aop=None, screen_dens=True
):
    vol = calc.atoms.get_volume()
    feat_sig, dfeat_jig = xc.get_features_on_grid_deriv(p_j, drhodf_jxg, DD_aop=DD_aop)
    nspin, nfeat = feat_sig.shape[:2]
    size = np.prod(feat_sig.shape[2:])
    all_wt = np.ones(size) / size * vol / Bohr**3
    all_wt = all_wt.flatten()
    feat_sig = feat_sig.reshape(nspin, nfeat, -1)
    dfeat_jig = dfeat_jig.reshape(len(p_j), nfeat, -1)
    # hopefully save disk space and avoid negative density
    if screen_dens:
        cond = feat_sig[:, 0, :].sum(axis=0) > 1e-6
        feat_sig = feat_sig[..., cond]
        dfeat_jig = dfeat_jig[..., cond]
        all_wt = all_wt[cond]
    assert all_wt.size == feat_sig.shape[-1]

    if xc.has_paw:
        assert DD_aop is not None
        DD_aop = xc.atomdist.to_work(DD_aop)
        for ft, dft, wt in get_atom_feat_wt_deriv(xc, DD_aop, p_j):
            assert wt.shape[-1] == ft.shape[-1]
            feat_sig = np.append(feat_sig, ft, axis=-1)
            dfeat_jig = np.append(dfeat_jig, dft, axis=-1)
            all_wt = np.append(all_wt, wt, axis=-1)

    if not isinstance(xc, _SLFeatMixin):
        feat_sig *= nspin
        feat_sig[:, 1, :] /= nspin ** (5.0 / 3)
        dfeat_jig *= nspin
        dfeat_jig[:, 1, :] /= nspin ** (5.0 / 3)
        if feat_sig.shape[1] == 6:
            feat_sig[:, 2, :] /= nspin ** (5.0 / 3)
            dfeat_jig[:, 2, :] /= nspin ** (5.0 / 3)
    return feat_sig, dfeat_jig, all_wt


def get_drho_df(calc, p_i):
    dens = calc.density
    nspin = dens.nt_sg.shape[0]
    wfs = calc.wfs
    assert isinstance(dens, ReciprocalSpaceDensity)
    shape = (len(p_i), 4)
    real_ixR = wfs.gd.zeros(shape)
    imag_ixR = wfs.gd.zeros(shape)
    drhodf_ixR = wfs.gd.zeros((len(p_i), 5))
    for i, (s, k, n) in enumerate(p_i):
        rank, q = wfs.kd.who_has(k)
        if wfs.kd.comm.rank == rank:
            u = q * nspin + s
            kpt = wfs.kpt_u[u]
            assert kpt.s == s
            assert kpt.k == k
        else:
            continue
        G_Gv = wfs.pd.get_reciprocal_vectors(q=kpt.q)
        psi_G = kpt.psit_nG[n]
        psi_R = wfs.pd.ifft(psi_G, kpt.q)
        real_ixR[i, 0] = psi_R.real
        imag_ixR[i, 0] = psi_R.imag
        for v in range(3):
            psi_G = 1j * G_Gv[:, v] * kpt.psit_nG[n]
            psi_R = wfs.pd.ifft(psi_G, kpt.q)
            real_ixR[i, v + 1] = psi_R.real
            imag_ixR[i, v + 1] = psi_R.imag
        drhodf_ixR[i, 0] = (
            real_ixR[i, 0] * real_ixR[i, 0] + imag_ixR[i, 0] * imag_ixR[i, 0]
        )
        for v in range(1, 4):
            drhodf_ixR[i, v] = 2 * (
                real_ixR[i, 0] * real_ixR[i, v] + imag_ixR[i, 0] * imag_ixR[i, v]
            )
            drhodf_ixR[i, 4] += 0.5 * (
                real_ixR[i, v] * real_ixR[i, v] + imag_ixR[i, v] * imag_ixR[i, v]
            )
    wfs.kptband_comm.sum(drhodf_ixR)
    # for i in range(len(p_i)):
    #    for x in range(5):
    #        wfs.kd.symmetry.symmetrize(drhodf_ixR[i, x], wfs.gd)
    return drhodf_ixR


# noinspection PyUnresolvedReferences
class _FeatureMixin:
    def get_features_on_grid(self, include_density=False):
        self.nexp
        nt_sg = self.dens.nt_sg
        self.nspin = nt_sg.shape[0]
        self.gd
        if self.rbuf_ag is None:
            self.initialize_more_things()
            self.construct_cubic_splines()
        sigma_xg, _ = calculate_sigma(self.gd, self.grad_v, nt_sg)

        if self.type == "MGGA":
            taut_sG = self.wfs.calculate_kinetic_energy_density()
            if taut_sG is None:
                taut_sG = self.wfs.gd.zeros(len(nt_sg))

            taut_sg = np.empty_like(nt_sg)

            for taut_G, taut_g in zip(taut_sG, taut_sg):
                if self.has_paw:
                    taut_G += 1.0 / self.wfs.nspins * self.tauct_G
                self.distribute_and_interpolate(taut_G, taut_g)
            tau_sg = taut_sg
        else:
            tau_sg = None

        nspin = nt_sg.shape[0]
        feat, dfeat, p_iag, q_ag, dq_ag = {}, {}, {}, {}, {}
        cider_nt_sg = self.domain_world2cider(nt_sg)
        if cider_nt_sg is None:
            cider_nt_sg = {s: None for s in range(nspin)}
        ascale, _ = self.get_ascale_and_derivs(nt_sg, sigma_xg, tau_sg)

        if self.has_paw:
            c_sabi, df_asbLg = self.paw_kernel.calculate_paw_feat_corrections(
                self.setups, self.get_D_asp()
            )
            if len(c_sabi.keys()) == 0:
                c_sabi = {s: {} for s in range(nspin)}
            D_sabi = {}
            for s in range(nspin):
                (
                    feat[s],
                    dfeat[s],
                    p_iag[s],
                    q_ag[s],
                    dq_ag[s],
                    D_sabi[s],
                ) = self.calculate_6d_integral_fwd(
                    cider_nt_sg[s], ascale[s], c_abi=c_sabi[s]
                )
            self.D_sabi = D_sabi
            self.y_asbLg = df_asbLg
            self.y_asbLk = None
        else:
            for s in range(nspin):
                (
                    feat[s],
                    dfeat[s],
                    p_iag[s],
                    q_ag[s],
                    dq_ag[s],
                ) = self.calculate_6d_integral_fwd(cider_nt_sg[s], ascale[s])

        if include_density:
            if self.type == "GGA":
                tau_sg = None
            feat = get_features_with_sl_noderiv(feat, nt_sg, sigma_xg, tau_sg)

        feat_out = self.gd.empty(n=feat.shape[:2], global_array=True)
        for s in range(feat_out.shape[0]):
            for i in range(feat_out.shape[1]):
                feat_out[s, i] = self.gd.collect(feat[s, i], broadcast=True)

        return feat_out

    def communicate_paw_features(self, ae_feat, ps_feat, ncomponents, nfeat=None):
        if nfeat is None:
            nfeat = 6 if self.type == "MGGA" else 5
        assert len(ae_feat) == len(ps_feat)
        feat_shapes = [
            (ncomponents, nfeat, Y_nL.shape[0], setup.xc_correction.rgd.N)
            for setup in self.setups
        ]
        ae_feat_glob = self.atom_partition.arraydict(feat_shapes, float)
        ps_feat_glob = self.atom_partition.arraydict(feat_shapes, float)
        for a in list(ae_feat.keys()):
            ae_feat_glob[a] = ae_feat[a]
            ps_feat_glob[a] = ps_feat[a]
        dist = self.atomdist
        ae_feat_glob = dist.from_work(ae_feat_glob)
        ps_feat_glob = dist.from_work(ps_feat_glob)
        ae_feat_glob = {
            a: ae_feat_glob[a] for a in ae_feat_glob
        }  # turn to normal array
        ps_feat_glob = {
            a: ps_feat_glob[a] for a in ps_feat_glob
        }  # turn to normal array
        atom_comm = dist.comm
        rank_a = dist.partition.rank_a
        for a in range(len(self.setups)):
            if rank_a[a] == 0:
                if atom_comm.rank == 0:
                    assert a in ae_feat_glob
            else:
                src = rank_a[a]
                if atom_comm.rank == src:
                    atom_comm.send(ae_feat_glob[a], 0, tag=2 * a)
                    atom_comm.send(ps_feat_glob[a], 0, tag=2 * a + 1)
                elif atom_comm.rank == 0:
                    ae_feat_glob[a] = np.empty(feat_shapes[a])
                    ps_feat_glob[a] = np.empty(feat_shapes[a])
                    atom_comm.receive(ae_feat_glob[a], src, tag=2 * a)
                    atom_comm.receive(ps_feat_glob[a], src, tag=2 * a + 1)
            # if a not in ae_feat_glob:
            #    assert dist.atom_partition.comm.rank != 0
            #    ae_feat_glob[a] = np.empty(feat_shapes[a])
            #    ps_feat_glob[a] = np.empty(feat_shapes[a])
            if atom_comm.rank != 0:
                ae_feat_glob[a] = np.empty(feat_shapes[a])
                ps_feat_glob[a] = np.empty(feat_shapes[a])
            atom_comm.broadcast(ae_feat_glob[a], 0)
            atom_comm.broadcast(ps_feat_glob[a], 0)
        return ae_feat_glob, ps_feat_glob

    def calculate_paw_features(self):
        ae_feat, ps_feat = calculate_paw_cider_features_p2_noderiv(
            self.paw_kernel, self.setups, self.get_D_asp(), self.D_sabi, self.y_asbLg
        )
        return self.communicate_paw_features(ae_feat, ps_feat, self.nspin)

    def calculate_paw_features_deriv(self, DD_aop, p_o):
        ae_feat, ps_feat, ae_dfeat, ps_dfeat = calculate_paw_cider_features_p2(
            self.paw_kernel,
            self.setups,
            self.get_D_asp(),
            DD_aop,
            p_o,
            self.D_oabi,
            self.y_aobLg,
        )
        ae_feat, ps_feat = self.communicate_paw_features(ae_feat, ps_feat, self.nspin)
        ae_dfeat, ps_dfeat = self.communicate_paw_features(ae_dfeat, ps_dfeat, len(p_o))
        return ae_feat, ps_feat, ae_dfeat, ps_dfeat

    def calculate_6d_integral_deriv(
        self, n_g, dndf_g, dcider_exp, q_ag, dq_ag, p_iag, c_abi=None
    ):
        nexp = self.nexp

        dascaledf_ig = self.domain_world2cider(dcider_exp)

        if c_abi is not None:
            self.write_augfeat_to_rbuf(c_abi)
        else:
            for a in self.alphas:
                self.rbuf_ag[a][:] = 0.0
        self.timer.start("COEFS")
        for ind, a in enumerate(self.alphas):
            self.rbuf_ag[a][:] += dndf_g * q_ag[a]
            self.rbuf_ag[a][:] += n_g * dq_ag[a] * dascaledf_ig[-1]
        self.timer.stop()

        self.calculate_6d_integral()

        if self.has_paw:
            self.timer.start("atom interp")
            D_abi = self.interpolate_rbuf_onto_atoms(pot=False)
            self.timer.stop()

        if n_g is not None:
            dfeatdf = np.zeros([nexp - 1] + list(n_g.shape))
        for i in range(nexp - 1):
            const = ((self.consts[i, 1] + self.consts[-1, 1]) / 2) ** 1.5
            const = const + self.consts[i, 0] / (self.consts[i, 0] + const)
            for ind, a in enumerate(self.alphas):
                dfeatdf[i, :] += const * p_iag[i, a] * self.rbuf_ag[a]

        self.timer.start("6d comm fwd")
        if n_g is not None:
            self.a_comm.sum(dfeatdf, root=0)
            if self.a_comm is not None and self.a_comm.rank == 0:
                dfeatdf = self.gdfft.collect(dfeatdf)
                if self.gdfft.rank != 0:
                    dfeatdf = None
            else:
                dfeatdf = None
        else:
            dfeatdf = None
        dfeatdf = self.domain_root2xc(dfeatdf, n=(3,))
        self.timer.stop()

        if self.has_paw:
            return dfeatdf, D_abi
        else:
            return dfeatdf

    def get_features_on_grid_deriv(self, p_j, drhodf_jxg, DD_aop=None):
        """

        Args:
            self: CiderFunctional
            p_j: List of (s, k, n) band indexes
            drhodf_jxg: (norb, 5, ngrid):
                Derivatives of the density (0), gradient (1,2,3),
                and kinetic energy density (4) with respect to orbital
                occupation number.
            include_density: whether to return density or just
                feature values

        Returns:

        """
        include_density = True

        self.nexp
        nt_sg = self.dens.nt_sg
        self.nspin = nt_sg.shape[0]
        self.gd
        if self.rbuf_ag is None:
            self.initialize_more_things()
            self.construct_cubic_splines()

        if DD_aop is not None:
            DD_aop = self.atomdist.to_work(DD_aop)

        sigma_xg, grad_svg = calculate_sigma(self.gd, self.grad_v, nt_sg)
        drhodf_jg = np.ascontiguousarray(drhodf_jxg[:, 0, :])
        dsigmadf_jg = np.zeros_like(drhodf_jg)
        nj = len(drhodf_jxg)
        for j in range(nj):
            dsigmadf_jg[j] = 2 * np.einsum(
                "v...,v...->...", grad_svg[p_j[j][0]], drhodf_jxg[j, 1:4]
            )
        dtaudf_jg = np.ascontiguousarray(drhodf_jxg[:, 4, :])

        if self.type == "MGGA":
            taut_sG = self.wfs.calculate_kinetic_energy_density()
            if taut_sG is None:
                taut_sG = self.wfs.gd.zeros(len(nt_sg))

            taut_sg = np.empty_like(nt_sg)

            for taut_G, taut_g in zip(taut_sG, taut_sg):
                if self.has_paw:
                    taut_G += 1.0 / self.wfs.nspins * self.tauct_G
                self.distribute_and_interpolate(taut_G, taut_g)
            tau_sg = taut_sg
        else:
            tau_sg = None

        nspin = nt_sg.shape[0]
        feat, dfeat, p_iag, q_ag, dq_ag = {}, {}, {}, {}, {}
        dfeatdf_jig = {}
        cider_nt_sg = self.domain_world2cider(nt_sg)
        cider_drhodf_jg = self.domain_world2cider(drhodf_jg)
        if cider_nt_sg is None:
            cider_nt_sg = {s: None for s in range(nspin)}
        if cider_drhodf_jg is None:
            cider_drhodf_jg = {j: None for j in range(nj)}
        ascale, ascale_derivs = self.get_ascale_and_derivs(nt_sg, sigma_xg, tau_sg)
        dascaledf_jig = np.empty((nj,) + ascale.shape[1:], dtype=ascale.dtype)
        for j in range(nj):
            s = p_j[j][0]
            dascaledf_jig[j] = ascale_derivs[0][s] * drhodf_jg[j]
            dascaledf_jig[j] += ascale_derivs[1][s] * dsigmadf_jg[j]
            if self.type == "MGGA":
                dascaledf_jig[j] += ascale_derivs[2][s] * dtaudf_jg[j]

        if self.has_paw:
            assert DD_aop is not None
            c_oabi, df_aobLg = calculate_paw_cider_features_p1(
                self.paw_kernel, self.setups, self.get_D_asp(), DD_aop, p_j
            )
            if len(c_oabi.keys()) == 0:
                c_oabi = {o: {} for o in range(nspin + len(p_j))}
            D_oabi = {}
            for s in range(nspin):
                (
                    feat[s],
                    dfeat[s],
                    p_iag[s],
                    q_ag[s],
                    dq_ag[s],
                    D_oabi[s],
                ) = self.calculate_6d_integral_fwd(
                    cider_nt_sg[s], ascale[s], c_abi=c_oabi[s]
                )
            for j in range(nj):
                s = p_j[j][0]
                dfeatdf_jig[j], D_oabi[nspin + j] = self.calculate_6d_integral_deriv(
                    cider_nt_sg[s],
                    cider_drhodf_jg[j],
                    dascaledf_jig[j],
                    q_ag[s],
                    dq_ag[s],
                    p_iag[s],
                    c_abi=c_oabi[nspin + j],
                )
                dfeatdf_jig[j] += dascaledf_jig[j, :-1] * dfeat[s]
            self.D_oabi = D_oabi
            self.y_aobLg = df_aobLg
            self.y_aobLk = None
        else:
            for s in range(nspin):
                (
                    feat[s],
                    dfeat[s],
                    p_iag[s],
                    q_ag[s],
                    dq_ag[s],
                ) = self.calculate_6d_integral_fwd(cider_nt_sg[s], ascale[s])
            for j in range(nj):
                s = p_j[j][0]
                dfeatdf_jig[j] = self.calculate_6d_integral_deriv(
                    cider_nt_sg[s],
                    cider_drhodf_jg[j],
                    dascaledf_jig[j],
                    q_ag[s],
                    dq_ag[s],
                    p_iag[s],
                )
                dfeatdf_jig[j] += dascaledf_jig[j, :-1] * dfeat[s]

        if include_density:
            if self.type == "GGA":
                tau_sg = None
                dtaudf_jg = None
            feat, dfeatdf_jig = get_features_with_sl_part(
                p_j,
                feat,
                dfeatdf_jig,
                nt_sg,
                sigma_xg,
                drhodf_jg,
                dsigmadf_jg,
                tau_sg,
                dtaudf_jg,
            )

        # feat = self.gd.collect(feat, broadcast=True)
        feat_out = self.gd.empty(n=feat.shape[:2], global_array=True)
        for s in range(feat_out.shape[0]):
            for i in range(feat_out.shape[1]):
                feat_out[s, i] = self.gd.collect(feat[s, i], broadcast=True)

        # dfeatdf_jig = self.gd.collect(dfeatdf_jig, broadcast=True)
        dfeat_out = self.gd.empty(n=dfeatdf_jig.shape[:2], global_array=True)
        for j in range(dfeatdf_jig.shape[0]):
            for i in range(dfeatdf_jig.shape[1]):
                dfeat_out[j, i] = self.gd.collect(dfeatdf_jig[j, i], broadcast=True)

        return feat_out, dfeat_out


class _SLFeatMixin(_FeatureMixin):
    def calculate_6d_integral_deriv(
        self, n_g, dndf_g, dcider_exp, q_ag, dq_ag, p_iag, c_abi=None
    ):
        raise NotImplementedError

    def calculate_paw_features(self):
        ae_feat, ps_feat = calculate_paw_sl_features(
            self.paw_kernel, self.setups, self.get_D_asp()
        )
        return self.communicate_paw_features(ae_feat, ps_feat, self.nspin, 5)

    def calculate_paw_features_deriv(self, DD_aop, p_o):
        ae_feat, ps_feat, ae_dfeat, ps_dfeat = calculate_paw_sl_features_deriv(
            self.paw_kernel, self.setups, self.get_D_asp(), DD_aop, p_o
        )
        ae_feat, ps_feat = self.communicate_paw_features(
            ae_feat, ps_feat, self.nspin, 5
        )
        ae_dfeat, ps_dfeat = self.communicate_paw_features(
            ae_dfeat, ps_dfeat, len(p_o), 5
        )
        return ae_feat, ps_feat, ae_dfeat, ps_dfeat

    def get_features_on_grid(self, include_density=True):
        nt_sg = self.dens.nt_sg
        self.nspin = nt_sg.shape[0]
        _, gradn_svg = calculate_sigma(self.gd, self.grad_v, nt_sg)
        if self.rbuf_ag is None:
            self.initialize_more_things()
            self.construct_cubic_splines()

        taut_sG = self.wfs.calculate_kinetic_energy_density()
        if taut_sG is None:
            taut_sG = self.wfs.gd.zeros(len(nt_sg))

        taut_sg = np.empty_like(nt_sg)

        for taut_G, taut_g in zip(taut_sG, taut_sg):
            if self.has_paw:
                taut_G += 1.0 / self.wfs.nspins * self.tauct_G
            self.distribute_and_interpolate(taut_G, taut_g)
        tau_sg = taut_sg

        feat = np.concatenate(
            [nt_sg[:, np.newaxis], gradn_svg, tau_sg[:, np.newaxis]], axis=1
        )
        feat = self.gd.collect(feat, broadcast=True)
        return feat

    def get_features_on_grid_deriv(self, p_j, drhodf_jxg, DD_aop=None):
        nt_sg = self.dens.nt_sg
        self.nspin = nt_sg.shape[0]
        _, gradn_svg = calculate_sigma(self.gd, self.grad_v, nt_sg)
        if self.rbuf_ag is None:
            self.initialize_more_things()
            self.construct_cubic_splines()

        taut_sG = self.wfs.calculate_kinetic_energy_density()
        if taut_sG is None:
            taut_sG = self.wfs.gd.zeros(len(nt_sg))

        taut_sg = np.empty_like(nt_sg)

        for taut_G, taut_g in zip(taut_sG, taut_sg):
            if self.has_paw:
                taut_G += 1.0 / self.wfs.nspins * self.tauct_G
            self.distribute_and_interpolate(taut_G, taut_g)
        tau_sg = taut_sg

        feat = np.concatenate(
            [nt_sg[:, np.newaxis], gradn_svg, tau_sg[:, np.newaxis]], axis=1
        )
        feat = self.gd.collect(feat, broadcast=True)
        dfeat = self.gd.collect(drhodf_jxg, broadcast=True)
        return feat, dfeat


class FeatGGA(_FeatureMixin, CiderGGA):
    def initialize(self, density, hamiltonian, wfs):
        super(FeatGGA, self).initialize(density, hamiltonian, wfs)
        self.wfs = wfs
        if (not hasattr(hamiltonian, "xc_redistributor")) or (
            hamiltonian.xc_redistributor is None
        ):
            self.restrict_and_collect = hamiltonian.restrict_and_collect
            self.distribute_and_interpolate = density.distribute_and_interpolate
        else:

            def _distribute_and_interpolate(in_xR, out_xR=None):
                tmp_xR = density.interpolate(in_xR)
                if hamiltonian.xc_redistributor.enabled:
                    out_xR = hamiltonian.xc_redistributor.distribute(tmp_xR, out_xR)
                elif out_xR is None:
                    out_xR = tmp_xR
                else:
                    out_xR[:] = tmp_xR
                return out_xR

            def _restrict_and_collect(in_xR, out_xR=None):
                if hamiltonian.xc_redistributor.enabled:
                    in_xR = hamiltonian.xc_redistributor.collect(in_xR)
                return hamiltonian.restrict(in_xR, out_xR)

            self.restrict_and_collect = _restrict_and_collect
            self.distribute_and_interpolate = _distribute_and_interpolate


class FeatMGGA(_FeatureMixin, CiderMGGA):
    pass


class FeatGGAPAW(_FeatureMixin, CiderGGAPASDW):
    def initialize(self, density, hamiltonian, wfs):
        super(FeatGGAPAW, self).initialize(density, hamiltonian, wfs)
        self.wfs = wfs
        if (not hasattr(hamiltonian, "xc_redistributor")) or (
            hamiltonian.xc_redistributor is None
        ):
            self.restrict_and_collect = hamiltonian.restrict_and_collect
            self.distribute_and_interpolate = density.distribute_and_interpolate
        else:

            def _distribute_and_interpolate(in_xR, out_xR=None):
                tmp_xR = density.interpolate(in_xR)
                if hamiltonian.xc_redistributor.enabled:
                    out_xR = hamiltonian.xc_redistributor.distribute(tmp_xR, out_xR)
                elif out_xR is None:
                    out_xR = tmp_xR
                else:
                    out_xR[:] = tmp_xR
                return out_xR

            def _restrict_and_collect(in_xR, out_xR=None):
                if hamiltonian.xc_redistributor.enabled:
                    in_xR = hamiltonian.xc_redistributor.collect(in_xR)
                return hamiltonian.restrict(in_xR, out_xR)

            self.restrict_and_collect = _restrict_and_collect
            self.distribute_and_interpolate = _distribute_and_interpolate


class FeatMGGAPAW(_FeatureMixin, CiderMGGAPASDW):
    pass


class FeatSL(_SLFeatMixin, CiderMGGA):
    pass


class FeatSLPAW(_SLFeatMixin, CiderMGGAPASDW):
    pass


def interpolate_drhodf(gd, interp, drhodf_ixR):
    drhodf_ixg = gd.zeros(drhodf_ixR.shape[:2])
    for i in range(drhodf_ixR.shape[0]):
        for x in range(drhodf_ixR.shape[1]):
            interp(drhodf_ixR[i, x], drhodf_ixg[i, x])
    return drhodf_ixg


def get_features(
    calc,
    p_i=None,
    use_paw=True,
    version="b",
    a0=1.0,
    fac_mul=0.03125,
    vvmul=1.0,
    amin=0.015625,
    qmax=300,
    lambd=1.8,
    screen_dens=True,
):
    """
    Compute grid weights, feature vectors, and (optionally)
    feature vector derivatives with respect to orbital
    occupation.

    Args:
        calc: a converged GPAW calculation
        p_i (optional): A list of (s, k, n) indexes for orbitals
            for which to differentiate features with respect to
            occupation
        use_paw (bool, True): Whether to use PAW corrections.
        version (str, 'b'): Descriptor version b, d, or l.
            l stands for local.

    Returns:
        if p_i is None:
            weights, features
        else:
            weights, features, [list of feature derivatives]
    """
    from ciderpress.descriptors import XCShell

    mlfunc = XCShell(version if version != "l" else "b", a0, fac_mul, amin, vvmul)
    if version == "b":
        kcls = CiderMGGAHybridKernel
        cls = FeatMGGAPAW if use_paw else FeatMGGA
    elif version == "d":
        kcls = CiderGGAHybridKernel
        cls = FeatGGAPAW if use_paw else FeatGGA
    elif version == "l":
        kcls = CiderMGGAHybridKernel
        cls = FeatSLPAW if use_paw else FeatSL
    else:
        raise ValueError("Version not supported.")
    cider_kernel = kcls(mlfunc, 0, "GGA_X_PBE", "GGA_C_PBE")
    consts = np.array([0.00, mlfunc.a0, mlfunc.fac_mul, mlfunc.amin])
    const_list = np.stack(
        [0.5 * consts, 1.0 * consts, 2.0 * consts, consts * mlfunc.vvmul]
    )
    nexp = 4
    empty_xc = cls(
        cider_kernel,
        nexp,
        const_list,
        Nalpha=None,
        lambd=lambd,
        encut=qmax,
        xmix=0.0,
        pasdw_ovlp_fit=True,
        pasdw_store_funcs=False,
    )
    empty_xc.set_grid_descriptor(calc.density.finegd)
    empty_xc.initialize(calc.density, calc.hamiltonian, calc.wfs)
    empty_xc.set_positions(calc.spos_ac)
    if calc.density.nt_sg is None:
        calc.density.interpolate_pseudo_density()
    if empty_xc.orbital_dependent:
        calc.converge_wave_functions()
    if p_i is None:
        return get_features_and_weights(calc, empty_xc, screen_dens=screen_dens)
    else:
        if use_paw:
            DD_aop = get_empty_atomic_matrix(calc.density, len(p_i))
            calculate_single_orbital_atomic_density_matrix(
                calc.wfs, DD_aop, p_i, calc.density.ncomponents
            )
        else:
            DD_aop = None
        drhodf_ixR = get_drho_df(calc, p_i)
        drhodf_ixg = interpolate_drhodf(
            empty_xc.gd, empty_xc.distribute_and_interpolate, drhodf_ixR
        )
        return get_features_and_weights_deriv(
            calc,
            empty_xc,
            p_i,
            drhodf_ixg,
            DD_aop=DD_aop,
            screen_dens=screen_dens,
        )
