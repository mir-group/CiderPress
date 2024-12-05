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

import ctypes

import numpy as np
from pyscf.dft.numint import _contract_rho, _dot_ao_ao, _dot_ao_dm, _scale_ao, eval_ao
from pyscf.gto.eval_gto import (
    BLKSIZE,
    _get_intor_and_comp,
    libcgto,
    make_loc,
    make_screen_index,
)
from pyscf.gto.mole import ANG_OF, ATOM_OF, PTR_EXP

from ciderpress.dft.plans import SADMPlan, SDMXFullPlan, SDMXPlan
from ciderpress.dft.settings import SADMSettings, SDMXFullSettings, SDMXSettings
from ciderpress.dft.sph_harm_coeff import get_deriv_ylm_coeff
from ciderpress.lib import load_library as load_cider_library

libcider = load_cider_library("libmcider")


def eval_conv_gto(
    feval,
    plan,
    mol,
    coords,
    shls_slice=None,
    non0tab=None,
    ao_loc=None,
    cutoff=None,
    out=None,
):
    if isinstance(plan, tuple):
        settings, alphas, alpha_norms = plan
    else:
        settings = plan.settings
        alphas = plan.alphas
        alpha_norms = plan.alpha_norms
    if non0tab is not None:
        if (non0tab == 0).any():
            # TODO implement some sort of screening later
            raise NotImplementedError
    eval_name, comp = _get_intor_and_comp(mol, feval)
    if comp not in [1, 4]:
        raise NotImplementedError
    comp = len(alphas)
    itype = settings.integral_type
    if eval_name == "GTOval_sph_deriv0":
        cpa = 1
        if settings.mode == "exact":
            contract_fn = getattr(libcider, "GTOcontract_conv0")
        else:
            if itype == "gauss_diff":
                contract_fn = getattr(libcider, "GTOcontract_smooth0")
            elif itype == "gauss_r2":
                contract_fn = getattr(libcider, "GTOcontract_rsq0")
            else:
                raise ValueError
        eval_fn = getattr(libcgto, "GTOshell_eval_grid_cart")
    elif eval_name == "GTOval_sph_deriv1":
        cpa = 4
        if settings.mode == "exact":
            raise NotImplementedError
        else:
            if itype == "gauss_diff":
                contract_fn = getattr(libcider, "GTOcontract_smooth1")
            elif itype == "gauss_r2":
                contract_fn = getattr(libcider, "GTOcontract_rsq1")
            else:
                raise ValueError
        eval_fn = getattr(libcgto, "GTOshell_eval_grid_cart_deriv1")
    else:
        raise NotImplementedError
    comp *= cpa

    atm = np.asarray(mol._atm, dtype=np.int32, order="C")
    bas = np.asarray(mol._bas, dtype=np.int32, order="C")
    env = np.asarray(mol._env, dtype=np.double, order="C")
    coords = np.asarray(coords, dtype=np.double, order="F")
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ngrids = coords.shape[0]

    if ao_loc is None:
        ao_loc = make_loc(bas, eval_name)

    if shls_slice is None:
        shls_slice = (0, nbas)
    sh0, sh1 = shls_slice
    nao = ao_loc[sh1] - ao_loc[sh0]
    if "spinor" in eval_name:
        ao = np.ndarray(
            (2, comp, nao, ngrids), dtype=np.complex128, buffer=out
        ).transpose(0, 1, 3, 2)
    else:
        ao = np.ndarray((comp, nao, ngrids), buffer=out).transpose(0, 2, 1)

    if non0tab is None:
        if cutoff is None:
            non0tab = np.ones(((ngrids + BLKSIZE - 1) // BLKSIZE, nbas), dtype=np.uint8)
        else:
            non0tab = make_screen_index(mol, coords, shls_slice, cutoff)

    # normal call tree is GTOval_sph_deriv0 -> GTOval_sph -> GTOeval_sph_drv
    drv = getattr(libcgto, "GTOeval_sph_drv")
    for i, alpha in enumerate(alphas):
        libcider.set_global_convolution_exponent(
            ctypes.c_double(alpha),
            ctypes.c_double(alpha_norms[i]),
        )
        drv(
            eval_fn,
            contract_fn,
            ctypes.c_double(1),
            ctypes.c_int(ngrids),
            (ctypes.c_int * 2)(1, cpa),
            (ctypes.c_int * 2)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            ao[i * cpa].ctypes.data_as(ctypes.c_void_p),
            coords.ctypes.data_as(ctypes.c_void_p),
            non0tab.ctypes.data_as(ctypes.c_void_p),
            atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p),
        )
    if comp == 1:
        if "spinor" in eval_name:
            ao = ao[:, 0]
        else:
            ao = ao[0]
    return ao


def eval_conv_gto_fast(
    feval,
    plan,
    mol,
    coords,
    shls_slice=None,
    non0tab=None,
    ao_loc=None,
    cutoff=None,
    out=None,
    ylm_buf=None,
    ylm_atom_loc=None,
):
    if isinstance(plan, tuple):
        settings, alphas, alpha_norms = plan
    else:
        settings = plan.settings
        alphas = plan.alphas
        alpha_norms = plan.alpha_norms
    if non0tab is not None:
        if (non0tab == 0).any():
            # TODO implement some sort of screening later
            raise NotImplementedError
    eval_name, comp = _get_intor_and_comp(mol, feval)
    if comp not in [1, 4]:
        raise NotImplementedError
    comp = len(alphas)
    itype = plan.settings.integral_type
    if eval_name == "GTOval_sph_deriv0":
        cpa = 1
        if settings.mode == "exact":
            raise NotImplementedError
        else:
            if itype == "gauss_diff":
                contract_fn = getattr(libcider, "SDMXcontract_smooth0")
            elif itype == "gauss_r2":
                contract_fn = getattr(libcider, "SDMXcontract_rsq0")
            else:
                raise ValueError
        eval_fn = getattr(libcider, "SDMXshell_eval_grid_cart")
    elif eval_name == "GTOval_sph_deriv1":
        cpa = 4
        if settings.mode == "exact":
            raise NotImplementedError
        else:
            if itype == "gauss_diff":
                contract_fn = getattr(libcider, "SDMXcontract_smooth1")
            elif itype == "gauss_r2":
                contract_fn = getattr(libcider, "SDMXcontract_rsq1")
            else:
                raise ValueError
        eval_fn = getattr(libcider, "SDMXshell_eval_grid_cart_deriv1")
    else:
        raise NotImplementedError
    comp *= cpa
    iter_fn = getattr(libcider, "SDMXeval_sph_iter")
    drv = getattr(libcider, "SDMXeval_loop")

    atm = np.asarray(mol._atm, dtype=np.int32, order="C")
    bas = np.asarray(mol._bas, dtype=np.int32, order="C")
    env = np.asarray(mol._env, dtype=np.double, order="C")
    coords = np.asarray(coords, dtype=np.double, order="F")
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ngrids = coords.shape[0]
    gaunt_lmax = 0
    if ylm_atom_loc is None:
        ylm_atom_loc = np.zeros(mol.natm + 1, dtype=np.int32)
        for ia in range(mol.natm):
            lmax = np.max(bas[bas[:, ATOM_OF] == ia, ANG_OF]) + 1
            ylm_atom_loc[ia + 1] = ylm_atom_loc[ia] + lmax * lmax
            gaunt_lmax = max(lmax - 1, gaunt_lmax)
    ylm = np.ndarray((cpa, ylm_atom_loc[-1], ngrids), buffer=ylm_buf)
    atom_coords = np.ascontiguousarray(mol.atom_coords(unit="Bohr"))
    libcider.SDMXylm_loop(
        ctypes.c_int(coords.shape[0]),
        ylm.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        ylm_atom_loc.ctypes.data_as(ctypes.c_void_p),
        atom_coords.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(mol.natm),
    )
    if cpa > 1:
        gaunt_coeff = get_deriv_ylm_coeff(gaunt_lmax)
        libcider.SDMXylm_grad(
            ctypes.c_int(coords.shape[0]),
            ylm.ctypes.data_as(ctypes.c_void_p),
            gaunt_coeff.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(gaunt_coeff.shape[1]),
            ylm_atom_loc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mol.natm),
        )
    libcider.SDMXylm_yzx2xyz(
        ctypes.c_int(coords.shape[0]),
        ctypes.c_int(cpa),
        ylm.ctypes.data_as(ctypes.c_void_p),
        ylm_atom_loc.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(mol.natm),
    )

    if ao_loc is None:
        ao_loc = make_loc(bas, eval_name)

    if shls_slice is None:
        shls_slice = (0, nbas)
    sh0, sh1 = shls_slice
    nao = ao_loc[sh1] - ao_loc[sh0]
    if "spinor" in eval_name:
        raise NotImplementedError
        # ao = np.ndarray((2, comp, nao, ngrids), dtype=np.complex128,
        #                 buffer=out).transpose(0, 1, 3, 2)
    else:
        ao = np.ndarray((comp, nao, ngrids), buffer=out).transpose(0, 2, 1)

    if non0tab is None:
        if cutoff is None:
            non0tab = np.ones(((ngrids + BLKSIZE - 1) // BLKSIZE, nbas), dtype=np.uint8)
        else:
            non0tab = make_screen_index(mol, coords, shls_slice, cutoff)

    drv(
        iter_fn,
        eval_fn,
        contract_fn,
        ctypes.c_double(1),
        ctypes.c_int(ngrids),
        (ctypes.c_int * 2)(1, cpa),
        (ctypes.c_int * 2)(*shls_slice),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        ao.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        non0tab.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p),
        ylm.ctypes.data_as(ctypes.c_void_p),
        ylm_atom_loc.ctypes.data_as(ctypes.c_void_p),
        alphas.ctypes.data_as(ctypes.c_void_p),
        alpha_norms.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(alphas.size),
    )
    if comp == 1:
        if "spinor" in eval_name:
            ao = ao[:, 0]
        else:
            ao = ao[0]
    return ao


def eval_conv_ao(
    plan,
    mol,
    coords,
    deriv=0,
    shls_slice=None,
    non0tab=None,
    cutoff=None,
    out=None,
    verbose=None,
):
    assert deriv in [0, 1]
    if mol.cart:
        feval = "GTOval_cart_deriv%d" % deriv
    else:
        feval = "GTOval_sph_deriv%d" % deriv
    return eval_conv_gto(
        feval, plan, mol, coords, shls_slice, non0tab, cutoff=cutoff, out=out
    )


def eval_conv_ao_fast(
    plan,
    mol,
    coords,
    deriv=0,
    shls_slice=None,
    non0tab=None,
    cutoff=None,
    out=None,
    ylm_buf=None,
    ylm_atom_loc=None,
    verbose=None,
):
    assert deriv in [0, 1]
    if mol.cart:
        feval = "GTOval_cart_deriv%d" % deriv
    else:
        feval = "GTOval_sph_deriv%d" % deriv
    return eval_conv_gto_fast(
        feval,
        plan,
        mol,
        coords,
        shls_slice,
        non0tab,
        cutoff=cutoff,
        out=out,
        ylm_buf=ylm_buf,
        ylm_atom_loc=ylm_atom_loc,
    )


class PySCFSDMXInitializer:
    def __init__(self, settings, **kwargs):
        self.settings = settings
        self.kwargs = kwargs

    def initialize_sdmx_generator(self, mol, nspin):
        return EXXSphGenerator.from_settings_and_mol(
            self.settings, nspin, mol, **self.kwargs
        )


class EXXSphGenerator:
    """
    Object used to evaluate SDMX features within PySCF.
    """

    fast = False

    def __init__(self, plan, lowmem=False):
        """
        Initialize EXXSphGenerator.

        Args:
            plan (SADMPlan or SDMXBasePlan): Plan for evaluating SDMX features.
        """
        if plan.fit_metric != "ovlp":
            raise NotImplementedError
        self.plan = plan
        self.lowmem = lowmem
        self._buf = None
        self._cached_ao_data = None

    @property
    def has_l1(self):
        return self.plan.settings.n1terms > 0

    def reset_buffers(self):
        self._buf = None
        self._cached_ao_data = None

    @classmethod
    def from_settings_and_mol(
        cls,
        settings,
        nspin,
        mol,
        alpha0=None,
        lambd=None,
        nalpha=None,
        lowmem=False,
    ):
        """
        Initialize from settings object and pyscf.gto.Mole object.

        Args:
            settings (ciderpress.dft.settings.SDMXBaseSettings):
                SDMX settings object that specifies features to compute.
            nspin (int): 1 or 2, 1 for non-spin-polarized and 2 for
                spin-polarized.
            mol (pyscf.gto.Mole): molecule object
            alpha0 (float): Minimum exponent for ETB interpolation basis.
                Will automatically be set to a reasonable value if not set.
            lambd (float): Spacing parameter for ETB interpolation basis.
                Will automatically be set to 1.8 if not set.
            nalpha (int): Number of interpolating exponents.

        Returns:
            EXXSphGenerator object
        """
        if alpha0 is None:
            min_exp = np.min(mol._env[mol._bas[:, PTR_EXP]])
            max_width = 1.0 / np.sqrt(2 * min_exp)
            coords = mol.atom_coords(unit="Bohr")
            dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
            max_dist = np.max(dist)
            # max_wpd = 2 * max_dist + 4 * max_width # 2 and 4 give some buffer
            max_wpd = 4 * max_dist + 4 * max_width
            alpha0 = 1.0 / (2 * max_wpd**2)
        if lambd is None:
            lambd = 1.8
        if nalpha is None:
            max_exp = np.max(mol._env[mol._bas[:, PTR_EXP]])
            nalpha = 1 + int(1 + np.log(max_exp / alpha0) / np.log(lambd))
        if isinstance(settings, SDMXFullSettings):
            plan = SDMXFullPlan(settings, nspin, alpha0, lambd, nalpha)
        elif isinstance(settings, SDMXSettings):
            plan = SDMXPlan(settings, nspin, alpha0, lambd, nalpha)
        elif isinstance(settings, SADMSettings):
            plan = SADMPlan(settings, nspin, alpha0, lambd, nalpha, fit_metric="ovlp")
        else:
            raise ValueError("Invalid type {}".format(type(settings)))
        return cls(plan, lowmem=lowmem)

    def __call__(self, *args, **kwargs):
        """
        Evaluate the features. See get_features.
        """
        return self.get_features(*args, **kwargs)

    def _get_fit_mats(self):
        if self.plan.has_coul_list:
            fit_mats = self.plan.fit_matrices
        else:
            fit_mats = [self.plan.fit_matrix]
        return fit_mats

    def _get_buffers(self, aobuf_size, caobuf_size, save_buf):
        if self._buf is not None and self._buf.size >= aobuf_size + caobuf_size:
            buf = self._buf
        else:
            buf = np.empty(aobuf_size + caobuf_size)
        aobuf = buf[:aobuf_size]
        caobuf = buf[aobuf_size : aobuf_size + caobuf_size]
        if save_buf:
            self._buf = buf
        else:
            self._buf = None
        return aobuf, caobuf

    def get_orb_vals(self, mol, coords, non0tab=None, cutoff=None, save_buf=True):
        nalpha = self.plan.nalpha
        deriv = 1 if self.has_l1 else 0
        ncpa = 4 if self.has_l1 else 1
        nao = mol.nao_nr()
        ngrids = coords.shape[0]
        aobuf_size = nao * ngrids
        if self.lowmem:
            caobuf_size = nao * ngrids * ncpa
        else:
            caobuf_size = nao * ngrids * ncpa * nalpha
        aobuf, caobuf = self._get_buffers(aobuf_size, caobuf_size, save_buf)
        if self.lowmem:
            cao = caobuf
        else:
            cao = eval_conv_ao(
                self.plan,
                mol,
                coords,
                deriv=deriv,
                non0tab=None,
                cutoff=cutoff,
                out=caobuf,
            )
        ao = eval_ao(mol, coords, deriv=0, non0tab=non0tab, cutoff=cutoff, out=aobuf)
        return ao, cao

    def _eval_crho_terms(self, mol, coords, cao, c0, tmp):
        deriv = 1 if self.has_l1 else 0
        ncpa = 4 if self.has_l1 else 1
        nalpha = self.plan.nalpha
        if self.lowmem:
            for ialpha in range(nalpha):
                _cao = eval_conv_ao(
                    (
                        self.plan.settings,
                        self.plan.alphas[ialpha : ialpha + 1],
                        self.plan.alpha_norms[ialpha : ialpha + 1],
                    ),
                    mol,
                    coords,
                    deriv=deriv,
                    cutoff=None,
                    out=cao,
                )
                for v in range(ncpa):
                    tmp[v, ialpha] = _contract_rho(_cao[v], c0)
        else:
            for v in range(ncpa):
                for ialpha in range(nalpha):
                    tmp[v, ialpha] = _contract_rho(cao[ialpha * ncpa + v], c0)

    def _eval_crho_potential(self, mol, coords, cao, tmp2):
        if self.lowmem:
            deriv = 1 if self.has_l1 else 0
            tmp3 = 0
            for ialpha in range(self.plan.nalpha):
                _cao = eval_conv_ao(
                    (
                        self.plan.settings,
                        self.plan.alphas[ialpha : ialpha + 1],
                        self.plan.alpha_norms[ialpha : ialpha + 1],
                    ),
                    mol,
                    coords,
                    deriv=deriv,
                    cutoff=None,
                    out=cao,
                )
                tmp3 += _scale_ao(_cao, tmp2[ialpha])
            return tmp3
        else:
            ngrids = coords.shape[0]
            return _scale_ao(cao, tmp2.reshape(-1, ngrids))

    def get_features(
        self,
        dms,
        mol,
        coords,
        non0tab=None,
        cutoff=None,
        save_buf=False,
        ao=None,
        cao=None,
    ):
        """
        Compute the SDMX features, return them, and store
        quantities needed to compute vxc.

        Args:
            dms (np.ndarray): Density matrix of matrices
            mol (pyscf.gto.Mole): molecule object
            coords (np.ndarray): (ngrids, 3) array of coordinates
            non0tab: PySCF non0tab. Not current used.
            cutoff: Pyscf cutoff intput. Not currently used.
            save_buf (bool): If True, save orbital buffer for
                future use.
            buf (np.ndarray): If provided, the output will use this
                data as a buffer for calculations. Must contain
                enough space for atomic orbital evaluations.

        Returns:
            out (np.ndarray): SDMX output features.
                Shape (ndm, nfeat, ngrids)
        """
        ndim = dms.ndim
        if ndim == 2:
            dms = dms[None, :, :]
            n_out = 1
        elif ndim == 3:
            n_out = dms.shape[0]
        else:
            raise ValueError
        has_l1 = self.has_l1
        ngrids = coords.shape[0]
        ncpa = 4 if has_l1 else 1
        tmp = np.empty((ncpa, self.plan.nalpha, ngrids))
        n0 = self.plan.num_l0_feat
        nfeat = self.plan.settings.nfeat
        tmp2 = np.empty((n_out, n0, self.plan.nalpha, ngrids))
        if has_l1:
            l1tmp2 = np.empty((n_out, nfeat - n0, 3, self.plan.nalpha, ngrids))
        else:
            l1tmp2 = [None] * n_out
        out = np.empty((n_out, nfeat, ngrids))
        if ao is None and cao is None:
            ao, cao = self.get_orb_vals(
                mol, coords, non0tab=non0tab, cutoff=cutoff, save_buf=save_buf
            )
        else:
            if ao is None or cao is None:
                raise ValueError("Must provide ao and cao or neither")
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()
        for idm, dm in enumerate(dms):
            c0 = _dot_ao_dm(mol, ao, dm, None, shls_slice, ao_loc)
            self._eval_crho_terms(mol, coords, cao, c0, tmp)
            self.plan.get_features(
                tmp, out=out[idm], l0tmp=tmp2[idm], l1tmp=l1tmp2[idm]
            )

        if ndim == 2:
            out = out[0]
        if has_l1:
            self._cached_ao_data = (mol, ao, cao, tmp2, l1tmp2, coords)
        else:
            self._cached_ao_data = (mol, ao, cao, tmp2, coords)
        return out

    @property
    def deriv(self):
        return 1 if self.has_l1 else 0

    def get_extra_ao(self, mol):
        if self.lowmem:
            return (1 + 3 * self.deriv) * mol.nao_nr()
        else:
            return (1 + 3 * self.deriv) * mol.nao_nr() * self.plan.nalpha

    def get_feat_and_occd(
        self, dm, coeffs, mol, coords, non0tab=None, cutoff=None, buf=None
    ):
        """
        Get the features and their occupation derivatives. Same arguments
        as get_features except dm must be a single density matrix and
        coeffs should contain an array of coefficients, where each row
        is an orbital coefficient array. Occupation derivatives are computed
        for these orbitals.

        Returns:
            val_out, occd_out (np.ndarray, np.ndarray): Features and their
                occupation derivatives.
        """
        assert self.plan.nspin == 1
        assert dm.ndim == 2
        assert coeffs.ndim == 2
        norb = coeffs.shape[0]
        has_l1 = self.has_l1
        nalpha = self.plan.nalpha
        deriv = self.deriv
        cao = eval_conv_ao(
            self.plan, mol, coords, deriv=deriv, non0tab=None, cutoff=cutoff, out=buf
        )
        ao = eval_ao(mol, coords, deriv=0, non0tab=non0tab, cutoff=cutoff, out=buf)
        if has_l1:
            ncpa = 4
        else:
            ncpa = 1
        n0 = self.plan.num_l0_feat
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()
        c0 = _dot_ao_dm(mol, ao, dm, None, shls_slice, ao_loc)
        ngrids = coords.shape[0]
        nfeat = self.plan.settings.nfeat
        occd_out = np.zeros((norb, nfeat, ngrids))
        tmp = np.empty((self.plan.nalpha, ngrids))
        for ialpha in range(nalpha):
            tmp[ialpha] = _contract_rho(cao[ialpha * ncpa], c0)
        val_fqg = np.stack(
            [fit_matrix.dot(tmp) for fit_matrix in self._get_fit_mats()[:n0]]
        )
        val_out = np.empty((nfeat, ngrids))
        val_out[:n0] = np.einsum("fqg,fqg->fg", val_fqg, val_fqg)
        if has_l1:
            val_fvqg = np.empty((nfeat - n0, 3, nalpha, ngrids))
            l1tmp = np.empty((3, self.plan.nalpha, ngrids))
            for ialpha in range(nalpha):
                for v in range(3):
                    l1tmp[v, ialpha] = _contract_rho(cao[ialpha * ncpa + v + 1], c0)
            for ifeat, fit_matrix in enumerate(self._get_fit_mats()[n0:]):
                ifeat_shift = ifeat + n0
                for v in range(3):
                    val_fvqg[ifeat, v] = fit_matrix.dot(l1tmp[v])
                val_out[ifeat_shift] = np.einsum(
                    "vqg,vqg->g", val_fvqg[ifeat], val_fvqg[ifeat]
                )
        for iorb, coeff_vec in enumerate(coeffs):
            mo_vals = _dot_ao_dm(
                mol, ao, coeff_vec[:, None], non0tab, shls_slice, ao_loc
            )
            c0 = coeff_vec * mo_vals
            for ialpha in range(nalpha):
                tmp[ialpha] = _contract_rho(cao[ialpha * ncpa], c0)
            occd_fqg = np.stack(
                [fit_matrix.dot(tmp) for fit_matrix in self._get_fit_mats()[:n0]]
            )
            occd_out[iorb, :n0] = np.einsum("fqg,fqg->fg", val_fqg, occd_fqg)
            if has_l1:
                for ialpha in range(nalpha):
                    for v in range(3):
                        l1tmp[v, ialpha] = _contract_rho(cao[ialpha * ncpa + v + 1], c0)
                occd_fvqg = np.empty((nfeat - n0, 3, nalpha, ngrids))
                for ifeat, fit_matrix in enumerate(self._get_fit_mats()[n0:]):
                    ifeat_shift = ifeat + n0
                    for v in range(3):
                        occd_fvqg[ifeat, v] = fit_matrix.dot(l1tmp[v])
                    occd_out[iorb, ifeat_shift] = np.einsum(
                        "vqg,vqg->g", val_fvqg[ifeat], occd_fvqg[ifeat]
                    )
        occd_out[:] *= -0.5
        val_out[:] *= -0.25
        return val_out, occd_out

    def get_vxc_(self, vxc_mat, vxc_grid):
        """
        Using stored values from get_features, which must be
        called first, compute contributions to vxc matrix
        from the vxc with respect to the SDMX features on a grid.

        Args:
            vxc_mat (np.ndarray): vxc matrix in orbital space, same
                shape as density matrix
            vxc_grid (np.ndarray): (nspin, nfeat, ngrids) XC potential
                with respect to features on grid

        Returns:
            vxc_mat
        """
        if self._cached_ao_data is None:
            raise RuntimeError
        ndim = vxc_mat.ndim
        nalpha = self.plan.nalpha
        ngrids = vxc_grid.shape[-1]
        if ndim == 2:
            vxc_mat = vxc_mat[None, :, :]
            vxc_grid = vxc_grid[None, :]
            n_out = 1
        elif ndim == 3:
            n_out = vxc_mat.shape[0]
        else:
            raise ValueError
        has_l1 = self.has_l1
        if has_l1:
            mol, ao, cao, alpha_terms, l1_alpha_terms, coords = self._cached_ao_data
            ncpa = 4
        else:
            mol, ao, cao, alpha_terms, coords = self._cached_ao_data
            l1_alpha_terms = [None] * n_out
            ncpa = 1
        # n0 = self.plan.num_l0_feat
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()
        for idm in range(n_out):
            # tmp = -0.25 * vxc_grid[idm, :n0, None] * alpha_terms[idm]  # fqg
            tmp2 = np.zeros((nalpha, ncpa, ngrids))
            self.plan.get_vxc(
                vxc_grid[idm],
                alpha_terms[idm],
                l1_alpha_terms[idm],
                out=tmp2.transpose(1, 0, 2),
            )
            tmp3 = self._eval_crho_potential(mol, coords, cao, tmp2)
            vxc_mat[idm] += _dot_ao_ao(mol, ao, tmp3, None, shls_slice, ao_loc, hermi=0)
        return vxc_mat
