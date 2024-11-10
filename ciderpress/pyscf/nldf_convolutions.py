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
from pyscf import gto
from pyscf.data.radii import COVALENT as COVALENT_RADII

from ciderpress.dft.lcao_convolutions import (
    DEFAULT_AUX_BETA,
    DEFAULT_CIDER_LMAX,
    ATCBasis,
    ConvolutionCollection,
    ConvolutionCollectionK,
    get_convolution_expnts_from_expnts,
    get_etb_from_expnt_range,
    get_gamma_lists_from_bas_and_env,
)
from ciderpress.dft.lcao_interpolation import LCAOInterpolator, LCAOInterpolatorDirect
from ciderpress.dft.lcao_nldf_generator import LCAONLDFGenerator
from ciderpress.dft.plans import NLDFGaussianPlan, NLDFSplinePlan, get_ccl_settings


def aug_etb_for_cider(
    mol,
    beta=DEFAULT_AUX_BETA,
    upper_fac=1.0,
    lower_fac=1.0,
    lmax=DEFAULT_CIDER_LMAX,
    def_afac_max=4.0,
    def_afac_min=0.5,
):
    """
    augment weigend basis with even-tempered gaussian basis
    exps = alpha*beta^i for i = 1..N
    For this implementation, exponent values should be on the
    same 'grid' ..., 1/beta, 1, beta, beta^2, ...

    Args:
        mol (pyscf.gto.Mole): molecule object
        beta (float): exponent parameter for ETB basis
        upper_fac (float): Max exponent should be at least
            upper_fac * max exponent in original basis
            for each l value.
        lower_fac (float): Min exponent should be at most
            lower_fac * min exponent in original basis
            for each l value.
        lmax (int): Maximum l value for expansion.
        def_afac_max (float): Max exponent relative to Bohr radius
            if no orbital products with a particular l exist
            in the original basis.
        def_afac_min (float): Min exponent relative to Bohr radius
            if not orbital products with a particular l exist
            in the original basis

    Returns:
        New basis set for expanding CIDER theta contributions.
    """
    uniq_atoms = set([a[0] for a in mol._atom])

    newbasis = {}
    for symb in uniq_atoms:
        nuc_charge = gto.charge(symb)

        def_amax = def_afac_max / COVALENT_RADII[nuc_charge] ** 2
        def_amin = def_afac_min / COVALENT_RADII[nuc_charge] ** 2

        max_shells = lmax
        emin_by_l = [1e99] * 8
        emax_by_l = [0] * 8
        for b in mol._basis[symb]:
            l = b[0]
            if l >= max_shells + 1:
                continue

            if isinstance(b[1], int):
                e_c = np.array(b[2:])
            else:
                e_c = np.array(b[1:])
            es = e_c[:, 0]
            cs = e_c[:, 1:]
            es = es[abs(cs).max(axis=1) > 1e-3]
            emax_by_l[l] = max(es.max(), emax_by_l[l])
            emin_by_l[l] = min(es.min(), emin_by_l[l])

        etb = get_etb_from_expnt_range(
            lmax,
            beta,
            emin_by_l,
            emax_by_l,
            def_amax,
            def_amin,
            lower_fac=lower_fac,
            upper_fac=upper_fac,
        )
        newbasis[symb] = gto.expand_etbs(etb)
    return newbasis


def get_gamma_lists_from_mol(mol):
    return get_gamma_lists_from_bas_and_env(mol._bas, mol._env)


class PySCFNLDFInitializer:
    def __init__(self, settings, **kwargs):
        """
        Args:
            settings (NLDFSettings): settings for features
            **kwargs: Any of the optional arguments in the
                PyscfNLDFGenerator.from_mol_and_settings
                function.
        """
        self.settings = settings
        self.kwargs = kwargs

    def initialize_nldf_generator(self, mol, grids_indexer, nspin):
        return PyscfNLDFGenerator.from_mol_and_settings(
            mol, grids_indexer, nspin, self.settings, **self.kwargs
        )


class PyscfNLDFGenerator(LCAONLDFGenerator):
    """
    A PySCF-specific wrapper for the NLDF feature generator.
    """

    @classmethod
    def from_mol_and_settings(
        cls,
        mol,
        grids_indexer,
        nspin,
        nldf_settings,
        plan_type="gaussian",
        lmax=None,
        aux_lambd=1.6,
        aug_beta=None,
        alpha_max=10000,
        alpha_min=None,
        alpha_formula=None,
        rhocut=1e-10,
        expcut=1e-10,
        gbuf=2.0,
        interpolator_type="onsite_direct",
        aparam=0.03,
        dparam=0.04,
        nrad=200,
    ):
        if lmax is None:
            lmax = grids_indexer.lmax
        if lmax > grids_indexer.lmax:
            raise ValueError("lmax cannot be larger than grids_indexer.lmax")
        if interpolator_type not in ["onsite_direct", "onsite_spline", "train_gen"]:
            raise ValueError
        if aug_beta is None:
            aug_beta = aux_lambd
        if alpha_min is None:
            alpha_min = nldf_settings.theta_params[0] / 256  # sensible default
        if plan_type not in ["gaussian", "spline"]:
            raise ValueError("plan_type must be gaussian or spline")
        if alpha_formula is None:
            alpha_formula = "etb" if plan_type == "gaussian" else "zexp"
        ratio = alpha_max / alpha_min
        if alpha_formula == "etb":
            ratio += 1
        nalpha = int(np.ceil(np.log(ratio) / np.log(aux_lambd))) + 1
        plan_class = NLDFGaussianPlan if plan_type == "gaussian" else NLDFSplinePlan
        plan = plan_class(
            nldf_settings,
            nspin,
            alpha_min,
            aux_lambd,
            nalpha,
            coef_order="gq",
            alpha_formula=alpha_formula,
            proc_inds=None,
            rhocut=rhocut,
            expcut=expcut,
        )
        basis = aug_etb_for_cider(mol, lmax=lmax, beta=aug_beta)
        mol = gto.M(
            atom=mol.atom,
            basis=basis,
            spin=mol.spin,
            charge=mol.charge,
            unit=mol.unit,
        )
        dat = get_gamma_lists_from_mol(mol)
        atco_inp = ATCBasis(*dat)
        dat = get_convolution_expnts_from_expnts(
            plan.alphas, dat[0], dat[1], dat[2], dat[4], gbuf=gbuf
        )
        atco_out = ATCBasis(*dat)
        if plan.nldf_settings.nldf_type == "k":
            ccl = ConvolutionCollectionK(
                atco_inp, atco_out, plan.alphas, plan.alpha_norms
            )
        else:
            has_vj, ifeat_ids = get_ccl_settings(plan)
            ccl = ConvolutionCollection(
                atco_inp,
                atco_out,
                plan.alphas,
                plan.alpha_norms,
                has_vj=has_vj,
                ifeat_ids=ifeat_ids,
            )
        if interpolator_type == "train_gen":
            interpolator = LCAOInterpolator.from_ccl(
                mol.atom_coords(unit="Bohr"),
                ccl,
                aparam=aparam,
                dparam=dparam,
                nrad=nrad,
            )
        else:
            onsite_direct = interpolator_type == "onsite_direct"
            interpolator = LCAOInterpolatorDirect.from_ccl(
                grids_indexer,
                mol.atom_coords(unit="Bohr"),
                ccl,
                aparam=aparam,
                dparam=dparam,
                nrad=nrad,
                onsite_direct=onsite_direct,
            )
        ccl.compute_integrals_()
        ccl.solve_projection_coefficients()
        return cls(plan, ccl, interpolator, grids_indexer)

    def get_extra_ao(self):
        return (self.interpolator.nlm + 4) * self.natm
