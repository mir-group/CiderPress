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

"""
This module is for fast evaluation of features and their derivatives with
respect to orbital occupations and rotations. Only for b and d features
"""

import numpy as np
from pyscf import lib
from pyscf.dft.numint import NumInt

from ciderpress.dft.plans import FracLaplPlan, SemilocalPlan
from ciderpress.dft.settings import (
    FracLaplSettings,
    NLDFSettings,
    SDMXBaseSettings,
    SemilocalSettings,
)
from ciderpress.pyscf.analyzers import UHFAnalyzer
from ciderpress.pyscf.frac_lapl import FLNumInt
from ciderpress.pyscf.gen_cider_grid import CiderGrids
from ciderpress.pyscf.nldf_convolutions import DEFAULT_CIDER_LMAX, PyscfNLDFGenerator
from ciderpress.pyscf.sdmx_slow import EXXSphGenerator

NLDF_VERSION_LIST = ["i", "j", "ij", "k"]


def get_descriptors(analyzer, settings, orbs=None, **kwargs):
    """
    Get the descriptors for version b or d (l=0 only). Analyzer must
    contain mo_occ, mo_energy, and mo_coeff attributes.
    Can get feature derivatives with respect to orbital occupation using orbs.
    Spin polarization is determined automatically from the analyzer class type.

    orbs format: orbs is a dictionary of lists or (for a spin-polarized system) a
    2-tuple of dictionary of lists. Each key is either 'B', 'O', or 'U',
    with the value being a list of orbital indexes with respect to which to
    compute occupation derivatives of the features. 'B' indexes up from the
    bottom, i.e. the lowest-energy core states. 'O' indexes down from the
    highest occupied molecular orbital. 'U' indexes up from the lowest
    unoccupied molecular orbital. 0-indexing is used in all cases.
    For the spin-polarized case, if orbs is a single dictionary rather than
    a 2-tuple, orbital indexing is performed over the combined set of
    orbitals.

    Args:
        analyzer (RHFAnalyzer or UHFAnalyzer): analyzer object containing the
            desired molecule and density matrix/orbitals for which to compute
            features and their derivatives.
        version: 'b', 'd', or 'l', feature version to evaluate. 'l' stands
            for local and computes [rho, drho_x, drho_y, drho_z, tau]
        orbs: Orbital index dictionary for computing orbital derivatives.
        **kwargs: Any settings one wants to change from the default for the
                  PyscfNLDFGenerator object.

    Returns:
        tuple: The routine returns a tuple of length 3. The first item in
        the tuple is desc, an np.ndarray of shape (nspin, nfeat, ngrid),
        which contains the descriptors for the system. The second item
        is ddesc, a dict(str : dict(int : np.ndarray(nfeat, ngrds))),
        which contains derivatives of the features with respect to the orbitals given by
        orbs, stored in a nested dictionary first by indexing style
        (B, O, or U) and then by an orbital index. If orbs was a 2-tuple,
        ddesc is a 2-tuple. Index format is the same as orbs, so a
        U-indexed orbital in orbs is U-indexed in ddesc. If calculation
        was spin-polarized but orbs was not spin-separated, each value in
        the ddesc dictionary is a 2-tuple (spin, deriv_array).
        The final element in the array is eigvals, which is
        a (dict(str : dict(int : float)) or (dict, dict),
        containing the eigenvalues of the relevant orbitals.
        These are important as baseline energy derivatives, assuming
        the baseline functional is the same as the reference functional
        for the eigenvalues.
    """
    spinpol = isinstance(analyzer, UHFAnalyzer)
    mo_occ = analyzer.mo_occ
    mo_energy = analyzer.mo_energy
    mo_coeff = analyzer.mo_coeff
    if orbs is None:
        labels, en_list, sep_spins = None, None, None
        if spinpol:
            coeffs = [None, None]
        else:
            coeffs = None
    else:
        labels, coeffs, en_list, sep_spins = get_labels_and_coeffs(
            orbs, mo_coeff, mo_occ, mo_energy
        )

    if isinstance(settings, str):
        if settings != "l":
            raise ValueError("Settings must be Settings object or letter l")
        my_desc_getter = _density_getter
    elif isinstance(settings, SemilocalSettings):
        my_desc_getter = _sl_desc_getter
    elif isinstance(settings, NLDFSettings):
        my_desc_getter = _nldf_desc_getter
    elif isinstance(settings, FracLaplSettings):
        my_desc_getter = _fl_desc_getter
    elif isinstance(settings, SDMXBaseSettings):
        my_desc_getter = _sdmx_desc_getter
    else:
        raise ValueError("Unsupported settings")

    if spinpol:
        desc = []
        for s in range(2):
            desc.append(
                my_desc_getter(
                    analyzer.mol,
                    analyzer.grids,
                    2 * analyzer.rdm1[s],
                    settings,
                    coeffs=coeffs[s],
                    **kwargs
                )
            )
        desc_a, desc_b = desc
        if orbs is not None:
            desc_a, deriv_arr_a = desc_a
            desc_b, deriv_arr_b = desc_b
            deriv_arr_a *= 2.0
            deriv_arr_b *= 2.0
            if sep_spins:
                ddesc_a, eigvals_a = unpack_feature_derivs(
                    orbs[0], labels[0], deriv_arr_a, en_list[0]
                )
                ddesc_b, eigvals_b = unpack_feature_derivs(
                    orbs[1], labels[1], deriv_arr_b, en_list[1]
                )
                return (
                    np.stack([desc_a, desc_b]),
                    (ddesc_a, ddesc_b),
                    (eigvals_a, eigvals_b),
                )
            else:
                deriv_arr = [deriv_arr_a, deriv_arr_b]
                ddesc = {}
                eigvals = {}
                for k in list(orbs.keys()):
                    ddesc[k] = {}
                    eigvals[k] = {}
                for s in range(2):
                    ddesc_tmp, eigvals_tmp = unpack_feature_derivs(
                        orbs, labels[s], deriv_arr[s], en_list[s]
                    )
                    for k in list(orbs.keys()):
                        ddesc[k].update(
                            {kk: (s, vv) for kk, vv in ddesc_tmp[k].items()}
                        )
                        eigvals[k].update(
                            {kk: (s, vv) for kk, vv in eigvals_tmp[k].items()}
                        )
                return np.stack([desc_a, desc_b]), ddesc, eigvals
        else:
            return np.stack([desc_a, desc_b])
    else:
        desc = my_desc_getter(
            analyzer.mol,
            analyzer.grids,
            analyzer.rdm1,
            settings,
            coeffs=coeffs,
            **kwargs
        )
        if orbs is not None:
            desc, deriv_arr = desc
            ddesc, eigvals = unpack_feature_derivs(orbs, labels, deriv_arr, en_list)
            return desc[None, :], ddesc, eigvals
        else:
            return desc[None, :]


def unpack_feature_derivs(orbs, labels, deriv_arr, en_list):
    ddesc = {}
    eigvals = {}
    for k in list(orbs.keys()):
        ddesc[k] = {}
        eigvals[k] = {}
    for n, (k, iorb) in enumerate(labels):
        ddesc[k][iorb] = deriv_arr[n].copy()
        eigvals[k][iorb] = en_list[n]
    return ddesc, eigvals


def unpack_eigvals(orbs, labels, en_list):
    eigvals = {}
    for k in list(orbs.keys()):
        eigvals[k] = {}
    for n, (k, iorb) in enumerate(labels):
        eigvals[k][iorb] = en_list[n]
    return eigvals


def _get_labels_and_coeffs(orbs, mo_coeff, mo_occ, mo_energy, spins=None):
    index_spin = spins is not None
    if index_spin:
        labels = ([], [])
        coeff_list = ([], [])
        en_list = ([], [])
    else:
        labels = []
        coeff_list = []
        en_list = []
    for k, v in orbs.items():
        if k == "O":
            inds = (mo_occ > 0.5).nonzero()[0][::-1]
        elif k == "U":
            inds = (mo_occ <= 0.5).nonzero()[0]
        else:
            inds = np.arange(mo_occ.size)
        mo_sub = mo_coeff[:, inds]
        en_sub = mo_energy[inds]
        if index_spin:
            spin_sub = spins[inds]
        else:
            spin_sub = None
        for iorb in v:
            l = (k, iorb)
            c = mo_sub[:, iorb]
            e = en_sub[iorb]
            if index_spin:
                s = spin_sub[iorb]
                labels[s].append(l)
                coeff_list[s].append(c)
                en_list[s].append(e)
            else:
                labels.append(l)
                coeff_list.append(c)
                en_list.append(e)
    return labels, coeff_list, en_list


def get_labels_and_coeffs(orbs, mo_coeff, mo_occ, mo_energy):
    spinpol = mo_occ.ndim == 2
    norb = mo_occ.shape[-1]
    sep_spins = False
    if spinpol:
        sep_spins = (isinstance(orbs, tuple) or isinstance(orbs, list)) and len(
            orbs
        ) == 2
        if sep_spins:
            labels_a, coeffs_a, en_list_a = _get_labels_and_coeffs(
                orbs[0], mo_coeff[0], mo_occ[0], mo_energy[0]
            )
            labels_b, coeffs_b, en_list_b = _get_labels_and_coeffs(
                orbs[1], mo_coeff[1], mo_occ[1], mo_energy[1]
            )
            labels = (labels_a, labels_b)
            coeffs = (coeffs_a, coeffs_b)
            en_list = (en_list_a, en_list_b)
        else:
            assert isinstance(orbs, dict)
            spins = np.append(
                np.zeros(norb, dtype=np.int32),
                np.ones(norb, dtype=np.int32),
            )
            mo_occ = mo_occ.flatten()
            mo_energy = mo_energy.flatten()
            mo_coeff = mo_coeff.transpose(1, 0, 2).reshape(norb, 2 * norb)
            inds = np.argsort(mo_energy)
            spins = np.ascontiguousarray(spins[inds])
            mo_occ = np.ascontiguousarray(mo_occ[inds])
            mo_energy = np.ascontiguousarray(mo_energy[inds])
            mo_coeff = np.ascontiguousarray(mo_coeff[:, inds])
            labels, coeffs, en_list = _get_labels_and_coeffs(
                orbs, mo_coeff, mo_occ, mo_energy, spins=spins
            )
    else:
        labels, coeffs, en_list = _get_labels_and_coeffs(
            orbs, mo_coeff, 0.5 * mo_occ, mo_energy
        )
    return labels, coeffs, en_list, sep_spins


def get_full_rho(ni, mol, dms, grids, xctype):
    hermi = 1
    max_memory = 2000
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_deriv = 1
    Nrhofeat = 4 if xctype == "GGA" else 5
    rho_full = np.zeros(
        (nset, Nrhofeat, grids.weights.size), dtype=np.float64, order="C"
    )
    ip0 = 0
    for ao, mask, weight, coords in ni.block_loop(
        mol, grids, nao, ao_deriv, max_memory
    ):
        # NOTE using the mask causes issues for some reason.
        # We don't need the mask anyway because is not
        # performance-critical code
        mask = None
        ip1 = ip0 + weight.size
        for idm in range(nset):
            rho = make_rho(idm, ao, mask, xctype)
            rho_full[idm, :, ip0:ip1] = rho
        ip0 = ip1
    return rho_full


def get_full_rho_with_fl(ni, mol, dms, grids):
    xctype = "MGGA"
    hermi = 1
    max_memory = 2000
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_deriv = 1
    Nrhofeat = ni.plan.settings.nrho + 5
    rho_full = np.zeros(
        (nset, Nrhofeat, grids.weights.size), dtype=np.float64, order="C"
    )
    ip0 = 0
    for ao, mask, weight, coords in ni.block_loop(
        mol, grids, nao, ao_deriv, max_memory
    ):
        # NOTE using the mask causes issues for some reasons
        # We don't need the mask anyway because is not
        # performance-critical code
        mask = None
        ip1 = ip0 + weight.size
        for idm in range(nset):
            rho = make_rho(idm, ao, mask, xctype)
            rho_full[idm, :, ip0:ip1] = rho
        ip0 = ip1
    return rho_full


def get_mo_densities(ni, mol, mo_vecs, grids, xctype):
    nmo = mo_vecs.shape[0]
    mo_coeff = np.ascontiguousarray(mo_vecs[:, :, np.newaxis])
    mo_occ = np.ones((nmo, 1))
    dms = np.empty((nmo, mol.nao_nr(), 0))
    dms = lib.tag_array(dms, mo_coeff=mo_coeff, mo_occ=mo_occ)
    return get_full_rho(ni, mol, dms, grids, xctype)


def get_mo_densities_with_fl(ni, mol, mo_vecs, grids):
    nmo = mo_vecs.shape[0]
    mo_coeff = np.ascontiguousarray(mo_vecs[:, :, np.newaxis])
    mo_occ = np.ones((nmo, 1))
    dms = np.empty((nmo, mol.nao_nr(), 0))
    dms = lib.tag_array(dms, mo_coeff=mo_coeff, mo_occ=mo_occ)
    return get_full_rho_with_fl(ni, mol, dms, grids)


def _density_getter(mol, pgrids, dms, nldf_settings, coeffs=None, **kwargs):
    ni = NumInt()
    desc = get_full_rho(ni, mol, dms, pgrids, "MGGA")[0]
    if coeffs is not None:
        if len(coeffs) == 0:
            return desc, 0
        coeffs = np.array(coeffs)
        ddesc = get_mo_densities(ni, mol, coeffs, pgrids, "MGGA")
        return desc, ddesc
    else:
        return desc


def _sl_desc_getter(mol, pgrids, dms, settings, coeffs=None, **kwargs):
    ni = NumInt()
    xctype = "MGGA"
    prho = get_full_rho(ni, mol, dms, pgrids, xctype)[0]
    plan = SemilocalPlan(settings, 1)
    if coeffs is None:
        return plan.get_feat(prho[None, :])[0]
    else:
        if len(coeffs) == 0:
            return plan.get_feat(prho[None, :])[0], 0
        coeffs = np.array(coeffs)
        feat = plan.get_feat(prho[None, :])[0]
        dprho_dphi = get_mo_densities(ni, mol, coeffs, pgrids, xctype)
        occds = []
        nmo = coeffs.shape[0]
        for imo in range(nmo):
            occds.append(plan.get_occd(prho[None, :], dprho_dphi[imo : imo + 1])[1][0])
        return feat, np.stack(occds)


def _nldf_desc_getter(
    mol,
    pgrids,
    dms,
    nldf_settings,
    coeffs=None,
    inner_grids_level=3,
    inner_grids=None,
    **kwargs
):
    kwargs["lmax"] = kwargs.get("lmax") or DEFAULT_CIDER_LMAX
    kwargs["interpolator_type"] = "train_gen"
    if inner_grids is None:
        grids = CiderGrids(mol, lmax=kwargs["lmax"])
        grids.level = inner_grids_level
        grids.build(with_non0tab=False)
    else:
        grids = inner_grids
    nldf_generator = PyscfNLDFGenerator.from_mol_and_settings(
        mol, grids.grids_indexer, 1, nldf_settings, **kwargs
    )
    nldf_generator.interpolator.set_coords(pgrids.coords)
    ni = NumInt()
    xctype = "MGGA"

    rho = get_full_rho(ni, mol, dms, grids, xctype)[0]
    prho = get_full_rho(ni, mol, dms, pgrids, xctype)[0]
    if coeffs is not None and len(coeffs) > 0:
        coeffs = np.array(coeffs)
        drho_dphi = get_mo_densities(ni, mol, coeffs, grids, xctype)
        dprho_dphi = get_mo_densities(ni, mol, coeffs, pgrids, xctype)
    else:
        drho_dphi = np.empty(0)
        dprho_dphi = np.empty(0)

    res = nldf_generator.get_features_and_occ_derivs(rho, drho_dphi, prho, dprho_dphi)
    if coeffs is None:
        return res[0]
    elif len(coeffs) > 0:
        return res
    else:
        return res[0], 0


def _fl_desc_getter(mol, pgrids, dms, flsettings, coeffs=None, **kwargs):
    plan = FracLaplPlan(flsettings, 1)
    ni = FLNumInt(plan)
    desc = get_full_rho_with_fl(ni, mol, dms, pgrids)
    feat = plan.get_feat(desc, make_l1_data_copy=False)
    if coeffs is not None:
        if len(coeffs) == 0:
            return feat[0], 0
        coeffs = np.array(coeffs)
        ddesc = get_mo_densities_with_fl(ni, mol, coeffs, pgrids)
        dfeat = plan.get_occd(desc[0], ddesc)
        return feat[0], dfeat
    else:
        return feat[0]


def _sdmx_desc_getter(mol, pgrids, dm, settings, coeffs=None, **kwargs):
    exxgen = EXXSphGenerator.from_settings_and_mol(settings, 1, mol, **kwargs)
    ni = NumInt()
    ngrids = pgrids.coords.shape[0]
    nfeat = settings.nfeat
    desc = np.zeros((nfeat, ngrids))
    if coeffs is None:
        ddesc = 0
    else:
        coeffs = np.asarray(coeffs)
        ddesc = np.zeros((coeffs.shape[0], nfeat, ngrids))
    max_memory = 2000
    nao = mol.nao_nr()
    ip0 = 0
    maxmem = int(max_memory / exxgen.plan.nalpha) + 1
    if settings.n1terms > 0:
        maxmem //= 4
        maxmem += 1
    for ao, mask, weight, coords in ni.block_loop(mol, pgrids, nao, 0, maxmem):
        ip1 = ip0 + weight.size
        if coeffs is None or len(coeffs) == 0:
            desc[:, ip0:ip1] = exxgen.get_features(dm, mol, coords)
        else:
            desc[:, ip0:ip1], ddesc[:, :, ip0:ip1] = exxgen.get_feat_and_occd(
                dm, coeffs, mol, coords
            )
        ip0 = ip1
    if coeffs is None:
        return desc
    elif len(coeffs) == 0:
        return desc, 0
    return desc, ddesc
