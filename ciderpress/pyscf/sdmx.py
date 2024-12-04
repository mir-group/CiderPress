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
from pyscf import lib
from pyscf.dft.numint import _dot_ao_ao, _dot_ao_dm, _scale_ao, eval_ao
from pyscf.gto.eval_gto import BLKSIZE, _get_intor_and_comp, make_screen_index
from pyscf.gto.mole import ANG_OF, ATOM_OF, NCTR_OF, PTR_EXP

from ciderpress.dft.plans import SADMPlan, SDMXFullPlan, SDMXIntPlan, SDMXPlan
from ciderpress.dft.settings import SADMSettings, SDMXFullSettings, SDMXSettings
from ciderpress.dft.sph_harm_coeff import get_deriv_ylm_coeff
from ciderpress.lib import load_library as load_cider_library

libcider = load_cider_library("libmcider")


def _get_ylm_atom_loc(mol):
    ylm_atom_loc = np.zeros(mol.natm + 1, dtype=np.int32)
    for ia in range(mol.natm):
        lmax = np.max(mol._bas[mol._bas[:, ATOM_OF] == ia, ANG_OF]) + 1
        ylm_atom_loc[ia + 1] = ylm_atom_loc[ia] + lmax * lmax
    return ylm_atom_loc


def _get_nrf(mol):
    """
    Get the number of unique radial functions associated with a basis
    """
    return np.sum(mol._bas[:, NCTR_OF])


def _get_rf_loc(mol):
    """
    Get an integer numpy array with that gives the location of the
    unique radial functions corresponding to each shell.
    So rf_loc[shl : shl + 1] is the range of radial functions contained
    in the shell. This is similar to ao_loc, except that it does not
    include spherical harmonics, so each shell has a factor of 2l+1
    more functions in ao_loc than rf_loc.
    """
    rf_loc = mol._bas[:, NCTR_OF]
    rf_loc = np.append([0], np.cumsum(rf_loc)).astype(np.int32)
    return rf_loc


def eval_conv_shells(
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
        alphas, alpha_norms, itype = plan
    else:
        alphas = plan.alphas
        alpha_norms = plan.alpha_norms
        itype = plan.settings.integral_type
    if non0tab is not None:
        if (non0tab == 0).any():
            # TODO implement some sort of screening later
            raise NotImplementedError
    eval_name, comp = _get_intor_and_comp(mol, feval)
    if comp not in [1, 4]:
        raise NotImplementedError
    comp = len(alphas)
    if eval_name == "GTOval_sph_deriv0":
        ncpa = 1
        if itype == "gauss_diff":
            contract_fn = getattr(libcider, "SDMXcontract_smooth0")
        elif itype == "gauss_r2":
            contract_fn = getattr(libcider, "SDMXcontract_rsq0")
        else:
            raise ValueError
        eval_fn = getattr(libcider, "SDMXrad_eval_grid")
    elif eval_name == "GTOval_sph_deriv1":
        ncpa = 2
        if itype == "gauss_diff":
            contract_fn = getattr(libcider, "SDMXcontract_smooth1")
        elif itype == "gauss_r2":
            contract_fn = getattr(libcider, "SDMXcontract_rsq1")
        else:
            raise ValueError
        eval_fn = getattr(libcider, "SDMXrad_eval_grid_deriv1")
    else:
        raise NotImplementedError
    comp *= ncpa
    drv = getattr(libcider, "SDMXeval_rad_loop")
    atm = np.asarray(mol._atm, dtype=np.int32, order="C")
    bas = np.asarray(mol._bas, dtype=np.int32, order="C")
    env = np.asarray(mol._env, dtype=np.double, order="C")
    coords = np.asarray(coords, dtype=np.double, order="F")
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ngrids = coords.shape[0]

    rf_loc = _get_rf_loc(mol)
    nrf = rf_loc[-1]

    if shls_slice is None:
        shls_slice = (0, nbas)
    if "spinor" in eval_name:
        raise NotImplementedError
        # ao = np.ndarray((2, comp, nao, ngrids), dtype=np.complex128,
        #                 buffer=out).transpose(0, 1, 3, 2)
    else:
        ao = np.ndarray((comp, nrf, ngrids), buffer=out).transpose(0, 2, 1)

    if non0tab is None:
        if cutoff is None:
            non0tab = np.ones(((ngrids + BLKSIZE - 1) // BLKSIZE, nbas), dtype=np.uint8)
        else:
            non0tab = make_screen_index(mol, coords, shls_slice, cutoff)
    drv(
        eval_fn,
        contract_fn,
        ctypes.c_double(1),
        ctypes.c_int(ngrids),
        (ctypes.c_int * 2)(1, ncpa),
        (ctypes.c_int * 2)(*shls_slice),
        rf_loc.ctypes.data_as(ctypes.c_void_p),
        ao.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        non0tab.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p),
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


def eval_conv_sh(
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
    return eval_conv_shells(
        feval, plan, mol, coords, shls_slice, non0tab, cutoff=cutoff, out=out
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
    fast = True

    def __init__(self, plan):
        if plan.fit_metric != "ovlp":
            raise NotImplementedError
        self.plan = plan
        self._ao_buf = None
        self._cao_buf = None
        self._ylm_buf = None
        self._shl_c0 = None
        self._cached_ao_data = None

    @property
    def has_l1(self):
        return self.plan.settings.n1terms > 0

    def reset_buffers(self):
        self._ao_buf = None
        self._cao_buf = None
        self._ylm_buf = None
        self._shl_c0 = None
        self._cached_ao_data = None

    @classmethod
    def from_settings_and_mol(
        cls, settings, nspin, mol, alpha0=None, lambd=None, nalpha=None, lowmem=False
    ):
        if alpha0 is None:
            min_exp = np.min(mol._env[mol._bas[:, PTR_EXP]])
            max_width = 1.0 / np.sqrt(2 * min_exp)
            coords = mol.atom_coords(unit="Bohr")
            dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
            max_dist = np.max(dist)
            max_wpd = 4 * max_dist + 4 * max_width
            # max_wpd = 2 * max_dist + 2 * max_width
            # max_wpd *= 2
            alpha0 = 1.0 / (2 * max_wpd**2)
        if lambd is None:
            lambd = 1.8
        if nalpha is None:
            max_exp = np.max(mol._env[mol._bas[:, PTR_EXP]])
            # max_exp = min(max_exp, 300)
            nalpha = 1 + int(1 + np.log(max_exp / alpha0) / np.log(lambd))
        if isinstance(settings, SDMXFullSettings):
            plan = SDMXFullPlan(settings, nspin, alpha0, lambd, nalpha)
            # plan = SDMXIntPlan(settings, nspin, alpha0, lambd, nalpha)
        elif isinstance(settings, SDMXSettings):
            plan = SDMXPlan(settings, nspin, alpha0, lambd, nalpha)
        elif isinstance(settings, SADMSettings):
            plan = SADMPlan(settings, nspin, alpha0, lambd, nalpha, fit_metric="ovlp")
        else:
            raise ValueError("Invalid type {}".format(type(settings)))
        return cls(plan)

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

    @property
    def deriv(self):
        return 1 if self.has_l1 else 0

    def get_extra_ao(self, mol):
        ymem = _get_ylm_atom_loc(mol)[-1]
        cmem = (1 + 6 * self.deriv) * _get_nrf(mol)
        bmem = (1 + self.deriv) * _get_nrf(mol) * self.plan.nalpha
        return ymem + cmem + bmem

    def get_ao(self, mol, coords, non0tab=None, cutoff=None, save_buf=True):
        # TODO use buffers
        return eval_ao(mol, coords, deriv=0, non0tab=non0tab, cutoff=cutoff, out=None)

    def get_cao(self, mol, coords, non0tab=None, cutoff=None, save_buf=True):
        caobuf_size = _get_nrf(mol) * coords.shape[0] * self.plan.nalpha
        if self.deriv > 0:
            caobuf_size *= 2
        if self._cao_buf is not None and self._cao_buf.size >= caobuf_size:
            buf = self._cao_buf
        else:
            buf = np.empty(caobuf_size)
        res = eval_conv_sh(
            self.plan,
            mol,
            coords,
            deriv=self.deriv,
            non0tab=non0tab,
            cutoff=cutoff,
            out=buf,
        )
        if save_buf:
            self._cao_buf = buf
        return res

    def _contract_ao_to_bas_single_(
        self,
        mol,
        b0,
        ylm,
        c0,
        shls_slice,
        ao_loc,
        ylm_atom_loc,
        coords=None,
        atom_coords=None,
        bwd=False,
    ):
        rf_loc = _get_rf_loc(mol)
        args = [
            ctypes.c_int(b0.shape[-1]),
            b0.ctypes.data_as(ctypes.c_void_p),
            ylm.ctypes.data_as(ctypes.c_void_p),
            c0.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int * 2)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            ylm_atom_loc.ctypes.data_as(ctypes.c_void_p),
            mol._atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mol.natm),
            mol._bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mol.nbas),
            mol._env.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(rf_loc[-1]),
            rf_loc.ctypes.data_as(ctypes.c_void_p),
        ]
        if coords is not None:
            assert coords.ndim == 1 and coords.flags.c_contiguous
            assert atom_coords.ndim == 1 and atom_coords.flags.c_contiguous
            args.append(coords.ctypes.data_as(ctypes.c_void_p))
            args.append(atom_coords.ctypes.data_as(ctypes.c_void_p))
            if bwd:
                libcider.SDMXcontract_ao_to_bas_grid_bwd(*args)
            else:
                libcider.SDMXcontract_ao_to_bas_grid(*args)
        else:
            if bwd:
                libcider.SDMXcontract_ao_to_bas_bwd(*args)
            else:
                libcider.SDMXcontract_ao_to_bas(*args)

    def _get_ylm(self, mol, coords, ylm_atom_loc=None, savebuf=True):
        ngrids = coords.shape[0]
        ncpa = 1 + 3 * self.deriv
        gaunt_lmax = np.max(mol._bas[:, ANG_OF])
        if ylm_atom_loc is None:
            ylm_atom_loc = _get_ylm_atom_loc(mol)
        ybufsize = ncpa * ylm_atom_loc[-1] * ngrids
        if self._ylm_buf is not None and self._ylm_buf.size >= ybufsize:
            buf = self._ylm_buf
        else:
            buf = np.empty(ybufsize)
        ylm = np.ndarray((ncpa, ylm_atom_loc[-1], ngrids), buffer=buf, order="C")
        atom_coords = np.ascontiguousarray(mol.atom_coords(unit="Bohr"))
        libcider.SDMXylm_loop(
            ctypes.c_int(coords.shape[0]),
            ylm.ctypes.data_as(ctypes.c_void_p),
            coords.ctypes.data_as(ctypes.c_void_p),
            ylm_atom_loc.ctypes.data_as(ctypes.c_void_p),
            atom_coords.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mol.natm),
        )
        if ncpa > 1:
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
            ctypes.c_int(ncpa),
            ylm.ctypes.data_as(ctypes.c_void_p),
            ylm_atom_loc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mol.natm),
        )
        if savebuf:
            self._ylm_buf = buf
        else:
            self._ylm_buf = None
        return ylm

    def _contract_ao_to_bas_helper(
        self, mol, b0, c0, shls_slice, ao_loc, coords, ylm=None, bwd=False
    ):
        coords = np.asfortranarray(coords)
        if ylm is None:
            ylm = self._get_ylm(mol, coords)
        ylm_atom_loc = _get_ylm_atom_loc(mol)
        if self.deriv == 1:
            atom_coords = np.asfortranarray(mol.atom_coords(unit="Bohr"))
            rf_loc = _get_rf_loc(mol)
            args = [
                ctypes.c_int(b0.shape[-1]),
                b0.ctypes.data_as(ctypes.c_void_p),
                ylm.ctypes.data_as(ctypes.c_void_p),
                c0.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int * 2)(*shls_slice),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                ylm_atom_loc.ctypes.data_as(ctypes.c_void_p),
                mol._atm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(mol.natm),
                mol._bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(mol.nbas),
                mol._env.ctypes.data_as(ctypes.c_void_p),
                coords.ctypes.data_as(ctypes.c_void_p),
                atom_coords.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(rf_loc[-1]),
                rf_loc.ctypes.data_as(ctypes.c_void_p),
            ]
            if bwd:
                fn = libcider.SDMXcontract_ao_to_bas_l1_bwd
            else:
                fn = libcider.SDMXcontract_ao_to_bas_l1
            fn(*args)
        else:
            self._contract_ao_to_bas_single_(
                mol,
                b0[0],
                ylm[0],
                c0,
                shls_slice,
                ao_loc,
                ylm_atom_loc,
                coords=None,
                bwd=bwd,
            )

    def _contract_ao_to_bas(self, mol, c0, shls_slice, ao_loc, coords, ylm=None):
        ngrids = coords.shape[0]
        b0 = np.empty((1 + 6 * self.deriv, _get_nrf(mol), ngrids))
        self._contract_ao_to_bas_helper(
            mol,
            b0,
            c0,
            shls_slice,
            ao_loc,
            coords,
            ylm=ylm,
            bwd=False,
        )
        return b0

    def _contract_ao_to_bas_bwd(self, mol, b0, shls_slice, ao_loc, coords, ylm=None):
        ngrids = coords.shape[0]
        c0 = np.zeros((mol.nao_nr(), ngrids))
        self._contract_ao_to_bas_helper(
            mol,
            b0,
            c0,
            shls_slice,
            ao_loc,
            coords,
            ylm=ylm,
            bwd=True,
        )
        return c0.T

    def _eval_crho_potential(
        self, mol, coords, cao, tmp2, shls_slice, ao_loc, ylm=None
    ):
        ngrids = coords.shape[0]
        nalpha = self.plan.nalpha
        b0 = np.empty((1 + 6 * self.deriv, _get_nrf(mol), ngrids)).transpose(0, 2, 1)
        if self.deriv == 0:
            b0[0] = _scale_ao(cao[:nalpha], tmp2[0])
        else:
            libcider.contract_shl_to_alpha_l1_bwd(
                ctypes.c_int(coords.shape[0]),
                ctypes.c_int(nalpha),
                ctypes.c_int(cao.shape[-1]),
                tmp2.ctypes.data_as(ctypes.c_void_p),
                b0.ctypes.data_as(ctypes.c_void_p),
                cao.ctypes.data_as(ctypes.c_void_p),
            )
        # if self.deriv > 0:
        #     b0[1] = _scale_ao(cao[:nalpha], tmp2[1])
        #     b0[2] = _scale_ao(cao[:nalpha], tmp2[2])
        #     b0[3] = _scale_ao(cao[:nalpha], tmp2[3])
        #     b0[4] = _scale_ao(cao[nalpha:2*nalpha], tmp2[1])
        #     b0[5] = _scale_ao(cao[nalpha:2*nalpha], tmp2[2])
        #     b0[6] = _scale_ao(cao[nalpha:2*nalpha], tmp2[3])
        return self._contract_ao_to_bas_bwd(
            mol, b0.transpose(0, 2, 1), shls_slice, ao_loc, coords, ylm=ylm
        )

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
        coords = np.asfortranarray(coords)
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
        nalpha = self.plan.nalpha
        ncpa = 4 if has_l1 else 1
        n0 = self.plan.num_l0_feat
        nfeat = self.plan.settings.nfeat
        # TODO might want to account for these when computing
        # memory footprint, though they are linear scaling in
        # size and therefore less of an issue.
        tmp = np.empty((ncpa, nalpha, ngrids))
        if isinstance(self.plan, SDMXIntPlan):
            l0tmp2 = np.empty((n_out, nalpha, ngrids))
            l1tmp2 = np.empty((n_out, 3, nalpha, ngrids))
        else:
            l0tmp2 = np.empty((n_out, n0, nalpha, ngrids))
            l1tmp2 = np.empty((n_out, nfeat - n0, 3, nalpha, ngrids))
        out = np.empty((n_out, nfeat, ngrids))
        if ao is None:
            ao = self.get_ao(
                mol, coords, non0tab=non0tab, cutoff=cutoff, save_buf=save_buf
            )
        if cao is None:
            cao = self.get_cao(
                mol, coords, non0tab=non0tab, cutoff=cutoff, save_buf=save_buf
            )
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()
        ylm = self._get_ylm(mol, coords, savebuf=True)
        for idm, dm in enumerate(dms):
            c0 = _dot_ao_dm(mol, ao, dm, None, shls_slice, ao_loc)
            b0 = self._contract_ao_to_bas(mol, c0, shls_slice, ao_loc, coords, ylm=ylm)
            if has_l1:
                assert tmp.flags.c_contiguous
                assert b0.flags.c_contiguous
                libcider.contract_shl_to_alpha_l1(
                    ctypes.c_int(coords.shape[0]),
                    ctypes.c_int(nalpha),
                    ctypes.c_int(cao.shape[-1]),
                    tmp.ctypes.data_as(ctypes.c_void_p),
                    b0.ctypes.data_as(ctypes.c_void_p),
                    cao.ctypes.data_as(ctypes.c_void_p),
                )
            else:
                for ialpha in range(nalpha):
                    tmp[0, ialpha] = lib.einsum(
                        "bg,bg->g", b0[0], cao[0 * nalpha + ialpha].T
                    )
            self.plan.get_features(
                tmp, out=out[idm], l0tmp=l0tmp2[idm], l1tmp=l1tmp2[idm]
            )
        if ndim == 2:
            out = out[0]
        self._cached_ao_data = (mol, ao, cao, l0tmp2, l1tmp2, coords, ylm)
        return out

    def get_vxc_(self, vxc_mat, vxc_grid):
        if self._cached_ao_data is None:
            raise RuntimeError("Must call get_features first")
        ndim = vxc_mat.ndim
        # nalpha = self.plan.nalpha
        # ngrids = vxc_grid.shape[-1]
        if ndim == 2:
            vxc_mat = vxc_mat[None, :, :]
            vxc_grid = vxc_grid[None, :]
            n_out = 1
        elif ndim == 3:
            n_out = vxc_mat.shape[0]
        else:
            raise ValueError
        # has_l1 = self.has_l1
        mol, ao, cao, alpha_terms, l1_alpha_terms, coords, ylm = self._cached_ao_data
        # ncpa = 1 + 3 * self.deriv
        # n0 = self.plan.num_l0_feat
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()
        for idm in range(n_out):
            tmp2 = self.plan.get_vxc(
                vxc_grid[idm], alpha_terms[idm], l1_alpha_terms[idm]
            )
            tmp3 = self._eval_crho_potential(
                mol, coords, cao, tmp2, shls_slice, ao_loc, ylm=ylm
            )
            vxc_mat[idm] += _dot_ao_ao(mol, ao, tmp3, None, shls_slice, ao_loc, hermi=0)
        return vxc_mat
