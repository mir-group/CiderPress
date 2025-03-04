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

from ciderpress.dft.plans import SemilocalPlan
from ciderpress.dft.settings import (
    FeatureSettings,
    FracLaplSettings,
    NLDFSettings,
    SDMXBaseSettings,
    SemilocalSettings,
)
from ciderpress.gpaw.atom_descriptor_utils import (
    PASDWCiderFeatureKernel,
    calculate_paw_sl_features,
    calculate_paw_sl_features_deriv,
)
from ciderpress.gpaw.calculator import CiderGGAHybridKernel, CiderMGGAHybridKernel
from ciderpress.gpaw.cider_paw import CiderGGA, CiderGGAPASDW, CiderMGGA, CiderMGGAPASDW
from ciderpress.gpaw.interp_paw import DiffPAWXCCorrection


class RhoVectorSettings(SemilocalSettings):
    def __init__(self):
        self.level = "MGGA"
        self.mode = "l"

    @property
    def nfeat(self):
        return 5

    @property
    def get_feat_usps(self):
        return [3, 4, 4, 4, 5]

    def ueg_vector(self, rho=1):
        tau0 = 0.3 * (3 * np.pi**2) ** (2.0 / 3) * rho ** (5.0 / 3)
        return np.array([rho, 0, 0, 0, tau0])


def get_descriptors(
    calc,
    settings,
    p_i=None,
    use_paw=True,
    screen_dens=True,
    **kwargs,
):
    """
    Compute grid weights, feature vectors, and (optionally)
    feature vector derivatives with respect to orbital
    occupation.

    Args:
        calc: a converged GPAW calculation
        settings (BaseSettings): A settings object specifying
            which features to compute.
        p_i (optional): A list of (s, k, n) indexes for orbitals
            for which to differentiate features with respect to
            occupation
        use_paw (bool, True): Whether to use PAW corrections.
        screen_dens (bool, True): Whether to remove low-density
            grids from the return feature vectors.
        kwargs: Any additional arguments to be passed to the
            Cider functional object (like qmax, lambd, etc.)
            to compute the functional.

    Returns:
        if p_i is None:
            weights, features
        else:
            weights, features, [list of feature derivatives]
    """
    if (
        hasattr(calc.hamiltonian.xc, "setups")
        and calc.hamiltonian.xc.setups is not None
    ):
        reset_setups = True
        contribs = []
        xccs = []
        for setup in calc.hamiltonian.xc.setups:
            xccs.append(setup.xc_correction)
            if hasattr(setup, "cider_contribs"):
                contribs.append(setup.cider_contribs)
            else:
                contribs.append(None)
    else:
        reset_setups = False
    if isinstance(settings, str):
        if settings != "l":
            raise ValueError("Settings must be Settings object or letter l")
        kcls = CiderMGGAHybridKernel
        cls = FeatSLMGGAPAW if use_paw else FeatSLMGGA
        cider_kernel = kcls(XCShell(RhoVectorSettings()), 0, "GGA_X_PBE", "GGA_C_PBE")
    elif isinstance(settings, SemilocalSettings):
        if settings.level == "MGGA":
            kcls = CiderMGGAHybridKernel
            cls = FeatSLMGGAPAW if use_paw else FeatSLMGGA
        else:
            kcls = CiderGGAHybridKernel
            cls = FeatSLGGAPAW if use_paw else FeatSLGGA
        cider_kernel = kcls(XCShell(settings), 0, "GGA_X_PBE", "GGA_C_PBE")
    elif isinstance(settings, NLDFSettings):
        if settings.sl_level == "MGGA":
            kcls = CiderMGGAHybridKernel
            cls = FeatMGGAPAW if use_paw else FeatMGGA
        else:
            kcls = CiderGGAHybridKernel
            cls = FeatGGAPAW if use_paw else FeatGGA
        cider_kernel = kcls(XCShell(settings), 0, "GGA_X_PBE", "GGA_C_PBE")
    elif isinstance(settings, FracLaplSettings):
        raise NotImplementedError(
            "Fractional Laplacian-based orbital descriptors have not yet been implemented for GPAW."
        )
    elif isinstance(settings, SDMXBaseSettings):
        raise NotImplementedError(
            "SDMX descriptors have not yet been implemented for GPAW."
        )
    else:
        raise ValueError("Unsupported settings")
    # TODO clean this up a bit
    kwargs["xmix"] = 0.0
    kwargs["encut"] = kwargs.get("qmax") or kwargs.get("encut") or 300
    kwargs["lambd"] = kwargs.get("lambd") or 1.8
    empty_xc = cls(
        cider_kernel,
        pasdw_ovlp_fit=True,
        pasdw_store_funcs=False,
        **kwargs,
    )
    empty_xc.set_grid_descriptor(calc.hamiltonian.xc.gd)
    empty_xc.initialize(calc.density, calc.hamiltonian, calc.wfs)
    empty_xc.set_positions(calc.spos_ac)
    if calc.density.nt_sg is None:
        calc.density.interpolate_pseudo_density()
    if empty_xc.orbital_dependent:
        calc.converge_wave_functions()
    if p_i is None:
        res = get_features_and_weights(calc, empty_xc, screen_dens=screen_dens)
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
        res = get_features_and_weights_deriv(
            calc,
            empty_xc,
            p_i,
            drhodf_ixg,
            DD_aop=DD_aop,
            screen_dens=screen_dens,
        )
    if reset_setups:
        for a, setup in enumerate(calc.hamiltonian.xc.setups):
            setup.xc_correction = xccs[a]
            if contribs[a] is None:
                setup.cider_contribs = None
            else:
                setup.cider_contribs = contribs[a]
    return res


class XCShell:
    def __init__(self, settings):
        if isinstance(settings, SemilocalSettings):
            self.settings = FeatureSettings(sl_settings=settings)
        elif isinstance(settings, NLDFSettings):
            sl_settings = SemilocalSettings(
                "npa" if settings.sl_level == "MGGA" else "np"
            )
            self.settings = FeatureSettings(
                sl_settings=sl_settings,
                nldf_settings=settings,
            )
        elif isinstance(settings, FracLaplSettings):
            self.settings = FeatureSettings(
                nlof_settings=settings,
            )
        elif isinstance(settings, SDMXBaseSettings):
            self.settings = FeatureSettings(
                sdmx_settings=settings,
            )
        else:
            raise ValueError("Unsupported settings")


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


def get_empty_atomic_matrix(dens, norb):
    return dens.setups.empty_atomic_matrix(norb, dens.atom_partition)


class FixedGOccupationNumbers(FixedOccupationNumbers):
    """
    Same as FixedOccupationNumbers but with a perturb_occupation_
    function for testing incremental occupation numbers.
    """

    def perturb_occupation_(self, p, delta):
        """
        Change the occupation number of a band at every k-point
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

    def _calculate(
        self,
        nelectrons,
        eig_qn,
        weight_q,
        f_qn,
        fermi_level_guess=np.nan,
        fix_fermi_level=False,
    ):
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


def get_atom_wt(functional, a):
    dv_g = functional.setups[a].xc_correction.rgd.dv_g
    weight_gn = dv_g[:, None] * weight_n
    return weight_gn.ravel()


def get_atom_feat_wt(functional):
    ae_feat, ps_feat = functional.calculate_paw_features()
    for a in range(len(ae_feat)):
        wt = get_atom_wt(functional, a)
        for feat, sgn in [(ae_feat[a], 1), (ps_feat[a], -1)]:
            assert wt.size == feat.shape[-1]
            yield feat, sgn * wt


def get_atom_feat_wt_deriv(functional, DD_aop, p_o):
    ae_feat, ps_feat, ae_dfeat, ps_dfeat = functional.calculate_paw_features_deriv(
        DD_aop, p_o
    )
    for a in range(len(ae_feat)):
        lst = [(ae_feat[a], ae_dfeat[a], 1), (ps_feat[a], ps_dfeat[a], -1)]
        wt = get_atom_wt(functional, a)
        for feat, dfeat, sgn in lst:
            assert wt.size == feat.shape[-1] == dfeat.shape[-1]
            yield feat, dfeat, sgn * wt


def _screen_density(calc, xc, *arrs):
    rho = calc.density.finegd.collect(calc.density.nt_sg.sum(0), broadcast=True)
    cond = rho.ravel() > 1e-6
    return tuple([a[..., cond] for a in arrs])


def get_features_and_weights(calc, xc, screen_dens=True):
    vol = calc.atoms.get_volume()
    ae_feat = xc.get_features_on_grid()
    nspin, nfeat = ae_feat.shape[:2]
    size = np.prod(ae_feat.shape[2:])
    all_wt = np.ones(size) / size * vol / Bohr**3
    all_wt = all_wt.flatten()
    ae_feat = ae_feat.reshape(nspin, nfeat, -1)
    # hopefully save disk space and avoid negative density
    if screen_dens:
        ae_feat, all_wt = _screen_density(calc, xc, ae_feat, all_wt)
        # cond = ae_feat[:, 0, :].sum(axis=0) > 1e-6
        # ae_feat = ae_feat[..., cond]
        # all_wt = all_wt[cond]
    assert all_wt.size == ae_feat.shape[-1]

    if xc.has_paw:
        for ft, wt in get_atom_feat_wt(xc):
            assert wt.shape[-1] == ft.shape[-1]
            ae_feat = np.append(ae_feat, ft, axis=-1)
            all_wt = np.append(all_wt, wt, axis=-1)
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
        feat_sig, dfeat_jig, all_wt = _screen_density(
            calc, xc, feat_sig, dfeat_jig, all_wt
        )
        # cond = feat_sig[:, 0, :].sum(axis=0) > 1e-6
        # feat_sig = feat_sig[..., cond]
        # dfeat_jig = dfeat_jig[..., cond]
        # all_wt = all_wt[cond]
    assert all_wt.size == feat_sig.shape[-1]

    if xc.has_paw:
        assert DD_aop is not None
        DD_aop = xc.atomdist.to_work(DD_aop)
        for ft, dft, wt in get_atom_feat_wt_deriv(xc, DD_aop, p_j):
            assert wt.shape[-1] == ft.shape[-1]
            feat_sig = np.append(feat_sig, ft, axis=-1)
            dfeat_jig = np.append(dfeat_jig, dft, axis=-1)
            all_wt = np.append(all_wt, wt, axis=-1)
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


def interpolate_drhodf(gd, interp, drhodf_ixR):
    drhodf_ixg = gd.zeros(drhodf_ixR.shape[:2])
    for i in range(drhodf_ixR.shape[0]):
        for x in range(drhodf_ixR.shape[1]):
            interp(drhodf_ixR[i, x], drhodf_ixg[i, x])
    return drhodf_ixg


class _FeatureMixin:
    def get_D_asp(self):
        return self.atomdist.to_work(self.dens.D_asp)

    def initialize_paw_kernel(self, cider_kernel_inp, Nalpha_atom, encut_atom):
        self.paw_kernel = PASDWCiderFeatureKernel(
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

    def _collect_feat(self, feat_xg):
        # TODO this has some redundant communication
        xshape = feat_xg.shape[:-3]
        gshape_out = tuple(self.gd.n_c)
        _feat_xg = np.zeros(xshape + gshape_out)
        self._add_from_cider_grid(_feat_xg, feat_xg)
        feat_xg = self.gd.collect(_feat_xg, broadcast=True)
        feat_xg.shape = xshape + (-1,)
        return feat_xg

    def _get_features_on_grid(self, rho_sxg):
        nspin = len(rho_sxg)
        self.nspin = nspin
        self._DD_aop = None
        if not self.is_initialized:
            self._setup_plan()
            self.fft_obj.initialize_backend()
            self.initialize_more_things()
        plan = self._plan
        rho_tuple = plan.get_rho_tuple(rho_sxg, with_spin=True)
        feat_sig = self.call_fwd(rho_tuple)[0]
        # TODO version i features
        feat_sig[:] *= plan.nspin
        return feat_sig

    def _get_pseudo_density(self):
        nt_sg = self.dens.nt_sg
        redist = self._hamiltonian.xc_redistributor
        if redist is not None:
            nt_sg = redist.distribute(nt_sg)
        return nt_sg

    def get_features_on_grid(self):
        nt_sg = self._get_pseudo_density()
        taut_sg = self._get_taut(nt_sg)[0]
        self.timer.start("Reorganize density")
        rho_sxg = self._get_cider_inputs(nt_sg, taut_sg)[1]
        self.timer.stop()
        feat = self._get_features_on_grid(rho_sxg)
        return self._collect_feat(feat)

    def _set_paw_terms(self):
        pass

    def _get_features_on_grid_deriv(self, p_j, rho_sxg, drhodf_jxg, DD_aop=None):
        nspin = len(rho_sxg)
        self.nspin = nspin
        self._p_o = p_j
        if not self.is_initialized:
            self._setup_plan()
            self.fft_obj.initialize_backend()
            self.initialize_more_things()
        if self.has_paw:
            self._DD_aop = self.atomdist.to_work(DD_aop)
        else:
            self._DD_aop = None
        plan = self._plan
        rho_tuple = plan.get_rho_tuple(rho_sxg, with_spin=True)
        drhodf_j_tuple = []
        for j, drhodf_xg in enumerate(drhodf_jxg):
            drhodf_j_tuple.append(plan.get_drhodf_tuple(rho_sxg[p_j[j][0]], drhodf_xg))
        feat_sig, dfeat_sig = self.call_fwd(rho_tuple)[:2]
        # TODO version i features
        feat_sig[:] *= plan.nspin
        dfeatdf_jig = []
        for j, drhodf_tuple in enumerate(drhodf_j_tuple):
            s = p_j[j][0]
            _rho_tuple = tuple([r[s] for r in rho_tuple])
            dfeatdf_ig = self.call_feat_deriv(j, _rho_tuple, drhodf_tuple, dfeat_sig[s])
            dfeatdf_jig.append(dfeatdf_ig)
        dfeatdf_jig = np.ascontiguousarray(dfeatdf_jig)
        dfeatdf_jig[:] *= plan.nspin

        return feat_sig, dfeatdf_jig

    def get_features_on_grid_deriv(self, p_j, drhodf_jxg, DD_aop=None):
        """
        Get the pseudo-features on the uniform FFT grid used
        to integrate the XC energy, along with the derivatives
        of the features with respect to orbital occupation.

        Args:
            self: CiderFunctional
            p_j: List of (s, k, n) band indexes
            drhodf_jxg: (norb, 5, ngrid):
                Derivatives of the density (0), gradient (1,2,3),
                and kinetic energy density (4) with respect to orbital
                occupation number.
            DD_aop (dict of numpy array): derivatives of atomic density
                matrices with respect to orbital occupation.

        Returns:
            feat_sig, dfeatdf_jig
        """
        nt_sg = self._get_pseudo_density()
        taut_sg = self._get_taut(nt_sg)[0]
        self.timer.start("Reorganize density")
        rho_sxg = self._get_cider_inputs(nt_sg, taut_sg)[1]
        drhodf_jxg = self._distribute_to_cider_grid(drhodf_jxg)
        self.timer.stop()
        nx = rho_sxg.shape[1]
        feat, dfeat = self._get_features_on_grid_deriv(
            p_j, rho_sxg, drhodf_jxg[:, :nx], DD_aop=DD_aop
        )
        return self._collect_feat(feat), self._collect_feat(dfeat)

    def communicate_paw_features(self, ae_feat, ps_feat, ncomponents, nfeat):
        assert len(ae_feat) == len(ps_feat)
        feat_shapes = [
            (ncomponents, nfeat, Y_nL.shape[0] * setup.xc_correction.rgd.N)
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
            if atom_comm.rank != 0:
                ae_feat_glob[a] = np.empty(feat_shapes[a])
                ps_feat_glob[a] = np.empty(feat_shapes[a])
            atom_comm.broadcast(ae_feat_glob[a], 0)
            atom_comm.broadcast(ps_feat_glob[a], 0)
        return ae_feat_glob, ps_feat_glob

    def call_feat_deriv(self, j, rho_tuple, drhodf_tuple, dfeat_ig):
        p, dp = None, None
        plan = self._plan
        arg_g, darg_g = plan.get_interpolation_arguments(rho_tuple, i=-1)
        fun_g, dfun_g = plan.get_function_to_convolve(rho_tuple)
        dargdf_g = 0
        dfundf_g = 0
        for darg, dfun, drho in zip(darg_g, dfun_g, drhodf_tuple):
            dargdf_g += darg * drho
            dfundf_g += dfun * drho
        p, dp = plan.get_interpolation_coefficients(arg_g, i=-1, vbuf=p, dbuf=dp)
        p.shape = arg_g.shape + (p.shape[-1],)
        dp.shape = arg_g.shape + (p.shape[-1],)
        dthetadf = dfundf_g[..., None] * p + (fun_g * dargdf_g)[..., None] * dp
        self.fft_obj.set_work(np.ones_like(fun_g), dthetadf)
        self.set_fft_work(self.nspin + j)
        self.fft_obj.compute_forward_convolution()
        self.get_fft_work(self.nspin + j)
        shape = (plan.nldf_settings.nfeat,) + arg_g.shape
        dfeatdf_ig = np.zeros(shape)
        for i in range(plan.num_vj):
            a_g, da_g = plan.get_interpolation_arguments(rho_tuple, i=i)
            p, dp = plan.get_interpolation_coefficients(a_g, i=i, vbuf=p, dbuf=dp)
            if False:  # TODO might need this later. Or can it be removed?
                coeff_multipliers = None
                p[:] *= coeff_multipliers
                dp[:] *= coeff_multipliers
            if False:  # TODO might need this later
                plan.get_transformed_interpolation_terms(p, i=i, fwd=True, inplace=True)
                plan.get_transformed_interpolation_terms(
                    dp, i=i, fwd=True, inplace=True
                )
            p.shape = a_g.shape + (plan.nalpha,)
            self.fft_obj.fill_vj_feature_(dfeatdf_ig[i], p)
            dadf_g = 0
            for darg, drho in zip(da_g, drhodf_tuple):
                dadf_g += darg * drho
            dfeatdf_ig[i] += dadf_g * dfeat_ig[i]
        return dfeatdf_ig

    def calculate_paw_features(self):
        ae_feat, ps_feat = self.paw_kernel.calculate_paw_cider_features_p2(
            self.setups, self.D_asp, self.c_asiq, None, None
        )
        nfeat = self.cider_kernel.mlfunc.settings.nldf_settings.nfeat
        return self.communicate_paw_features(ae_feat, ps_feat, self.nspin, nfeat)

    def calculate_paw_features_deriv(self, DD_aop, p_o):
        res = self.paw_kernel.calculate_paw_cider_features_p2(
            self.setups, self.D_asp, self.c_asiq, DD_aop, p_o
        )
        ae_feat, ps_feat, ae_dfeat, ps_dfeat = res
        nfeat = self.cider_kernel.mlfunc.settings.nldf_settings.nfeat
        ae_feat, ps_feat = self.communicate_paw_features(
            ae_feat, ps_feat, self.nspin, nfeat
        )
        ae_dfeat, ps_dfeat = self.communicate_paw_features(
            ae_dfeat, ps_dfeat, len(p_o), nfeat
        )
        return ae_feat, ps_feat, ae_dfeat, ps_dfeat


class _PAWFeatureMixin(_FeatureMixin):
    def _set_paw_terms(self):
        if not hasattr(self, "setups") or self.setups is None:
            return
        if self._DD_aop is None:
            D_asp = self.get_D_asp()
            if not (D_asp.partition.rank_a == self.atom_partition.rank_a).all():
                raise ValueError("rank_a mismatch")
            self.c_asiq = self.paw_kernel.calculate_paw_feat_corrections(
                self.setups, D_asp
            )
            self.D_asp = D_asp
        else:
            D_asp = self.get_D_asp()
            if not (D_asp.partition.rank_a == self.atom_partition.rank_a).all():
                raise ValueError("rank_a mismatch")
            self.c_asiq = self.paw_kernel.calculate_paw_cider_features_p1(
                self.setups, D_asp, self._DD_aop, self._p_o
            )
            self.D_asp = D_asp


class _SLFeatMixin(_FeatureMixin):
    def calculate_paw_features(self):
        settings = self.cider_kernel.mlfunc.settings.sl_settings
        ae_feat, ps_feat = calculate_paw_sl_features(
            self.setups, self.get_D_asp(), settings.level == "MGGA"
        )
        if settings.mode != "l":
            plan = SemilocalPlan(settings, self.nspin)
            for a in list(ae_feat.keys()):
                ae_feat[a] = plan.get_feat(ae_feat[a])
                ps_feat[a] = plan.get_feat(ps_feat[a])
        else:
            for a in list(ae_feat.keys()):
                ae_feat[a] *= self.nspin
                ps_feat[a] *= self.nspin
        return self.communicate_paw_features(
            ae_feat, ps_feat, self.nspin, settings.nfeat
        )

    def calculate_paw_features_deriv(self, DD_aop, p_o):
        settings = self.cider_kernel.mlfunc.settings.sl_settings
        _ae_rho, _ps_rho, _ae_drho, _ps_drho = calculate_paw_sl_features_deriv(
            self.setups, self.get_D_asp(), DD_aop, p_o, settings.level == "MGGA"
        )

        if settings.mode != "l":
            ae_feat = {}
            ae_dfeat = {}
            ps_feat = {}
            ps_dfeat = {}
            plan = SemilocalPlan(settings, self.nspin)
            for a in list(_ae_rho.keys()):
                ae_drho = _ae_drho[a]
                ae_rho = _ae_rho[a]
                ps_drho = _ps_drho[a]
                ps_rho = _ps_rho[a]
                ae_dfeat[a] = np.empty((len(p_o), settings.nfeat) + ae_drho.shape[2:])
                for j, p in enumerate(p_o):
                    ae_dfeat[a][j : j + 1] = plan.get_occd(
                        ae_rho[p[0] : p[0] + 1], ae_drho[j : j + 1]
                    )[1]
                ae_feat[a] = plan.get_feat(ae_rho)
                ps_dfeat[a] = np.empty((len(p_o), settings.nfeat) + ps_drho.shape[2:])
                for j, p in enumerate(p_o):
                    ps_dfeat[a][j : j + 1] = plan.get_occd(
                        ps_rho[p[0] : p[0] + 1], ps_drho[j : j + 1]
                    )[1]
                ps_feat[a] = plan.get_feat(ps_rho)
        else:
            ae_feat = _ae_rho
            ae_dfeat = _ae_drho
            ps_feat = _ps_rho
            ps_dfeat = _ps_drho
            for a in list(_ae_rho.keys()):
                ae_feat[a] *= self.nspin
                ae_dfeat[a] *= self.nspin
                ps_feat[a] *= self.nspin
                ps_dfeat[a] *= self.nspin

        ae_feat, ps_feat = self.communicate_paw_features(
            ae_feat, ps_feat, self.nspin, settings.nfeat
        )
        ae_dfeat, ps_dfeat = self.communicate_paw_features(
            ae_dfeat, ps_dfeat, len(p_o), settings.nfeat
        )
        return ae_feat, ps_feat, ae_dfeat, ps_dfeat

    def _collect_feat(self, feat_xg):
        # TODO this has some redundant communication
        xshape = feat_xg.shape[:-1]
        gshape_out = tuple(self.gd.n_c)
        gshape_in = tuple(self.distribution.local_output_size_c)
        feat_xg = feat_xg.view()
        feat_xg.shape = xshape + gshape_in
        _feat_xg = np.zeros(xshape + gshape_out)
        self._add_from_cider_grid(_feat_xg, feat_xg)
        feat_xg = self.gd.collect(_feat_xg, broadcast=True)
        feat_xg.shape = xshape + (-1,)
        return feat_xg

    def _check_setups(self):
        if hasattr(self, "setups") and self.setups is not None:
            for setup in self.setups:
                if (
                    not isinstance(setup.xc_correction, DiffPAWXCCorrection)
                    and setup.xc_correction is not None
                ):
                    setup.xc_correction = DiffPAWXCCorrection.from_setup(
                        setup, build_kinetic=True, ke_order_ng=False
                    )

    def get_features_on_grid(self):
        nt_sg = self._get_pseudo_density()
        taut_sg = self._get_taut(nt_sg)[0]
        rho_sxg = self._get_cider_inputs(nt_sg, taut_sg)[1]
        settings = self.cider_kernel.mlfunc.settings.sl_settings
        rho_sxg = rho_sxg.view()
        rho_sxg.shape = rho_sxg.shape[:2] + (-1,)
        if settings.mode != "l":
            plan = SemilocalPlan(settings, self.nspin)
            feat_sig = plan.get_feat(rho_sxg)
        else:
            feat_sig = self.nspin * rho_sxg
        self._check_setups()
        return self._collect_feat(feat_sig)

    def get_features_on_grid_deriv(self, p_j, drhodf_jxg, DD_aop=None):
        nt_sg = self._get_pseudo_density()
        taut_sg = self._get_taut(nt_sg)[0]
        self.timer.start("Reorganize density")
        rho_sxg = self._get_cider_inputs(nt_sg, taut_sg)[1]
        drhodf_jxg = self._distribute_to_cider_grid(drhodf_jxg)
        self.timer.stop()
        rho_sxg = rho_sxg.view()
        drhodf_jxg = drhodf_jxg.view()
        rho_sxg.shape = rho_sxg.shape[:2] + (-1,)
        drhodf_jxg.shape = drhodf_jxg.shape[:2] + (-1,)
        settings = self.cider_kernel.mlfunc.settings.sl_settings
        if settings.mode != "l":
            plan = SemilocalPlan(settings, self.nspin)
            dfeat_jig = np.empty((len(p_j), settings.nfeat) + rho_sxg.shape[2:])
            for j, p in enumerate(p_j):
                dfeat_jig[j : j + 1] = plan.get_occd(
                    rho_sxg[p[0] : p[0] + 1], drhodf_jxg[j : j + 1]
                )[1]
            feat_sig = plan.get_feat(rho_sxg)
        else:
            feat_sig = self.nspin * rho_sxg
            dfeat_jig = self.nspin * drhodf_jxg
        self._check_setups()
        return self._collect_feat(feat_sig), self._collect_feat(dfeat_jig)


def _initialize_gga_feat(self, density, hamiltonian, wfs):
    """
    GGAs do not automatically define these distribute and restrict
    functions, which we need for feature generation, so this initializer
    takes care of that.
    """
    super(self.__class__, self).initialize(density, hamiltonian, wfs)
    self.wfs = wfs
    self.atomdist = hamiltonian.atomdist
    self.setups = hamiltonian.setups
    self.atom_partition = self.get_D_asp().partition
    self._hamiltonian = hamiltonian
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


class FeatGGA(_FeatureMixin, CiderGGA):
    initialize = _initialize_gga_feat


class FeatMGGA(_FeatureMixin, CiderMGGA):
    initialize = _initialize_gga_feat


class FeatGGAPAW(_PAWFeatureMixin, CiderGGAPASDW):
    initialize = _initialize_gga_feat


class FeatMGGAPAW(_PAWFeatureMixin, CiderMGGAPASDW):
    pass


class FeatSLGGA(_SLFeatMixin, CiderGGA):
    initialize = _initialize_gga_feat


class FeatSLMGGA(_SLFeatMixin, CiderMGGA):
    initialize = _initialize_gga_feat


class FeatSLGGAPAW(_SLFeatMixin, CiderGGAPASDW):
    initialize = _initialize_gga_feat


class FeatSLMGGAPAW(_SLFeatMixin, CiderMGGAPASDW):
    initialize = _initialize_gga_feat
