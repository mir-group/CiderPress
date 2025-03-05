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

from pyscf import lib

from ciderpress.pyscf.dft import _CiderKS as _MolCiderKS
from ciderpress.pyscf.dft import get_slxc_settings, load_cider_model
from ciderpress.pyscf.pbc.numint import CiderKNumInt, CiderNumInt, numint
from ciderpress.pyscf.pbc.sdmx_fft import PySCFSDMXInitializer


def make_cider_calc(
    ks,
    mlfunc,
    xmix=1.0,
    xc=None,
    xkernel=None,
    ckernel=None,
    mlfunc_format=None,
    nlc_coeff=None,
    nldf_init=None,
    sdmx_init=None,
    dense_mesh=None,
    rhocut=None,
):
    """
    Same as :func:`ciderpress.pyscf.dft.make_cider_calc`, but
    for periodic systems. Note that only semilocal and SDMX
    features are supported. The ks object must use a uniform XC
    integration grid and pseudopotentials.
    """
    mlfunc = load_cider_model(mlfunc, mlfunc_format)
    ks._xc = get_slxc_settings(xc, xkernel, ckernel, xmix)
    # Assign the PySCF-facing functional to be a simple SL
    # functional to avoid hybrid DFT being called.
    # NOTE this might need to be changed to some nicer
    # approach later.
    if mlfunc.settings.sl_settings.level == "MGGA":
        ks.xc = "R2SCAN"
    else:
        ks.xc = "PBE"
    new_ks = _CiderKS(
        ks,
        mlfunc,
        xmix=xmix,
        nldf_init=nldf_init,
        sdmx_init=sdmx_init,
        rhocut=rhocut,
        nlc_coeff=nlc_coeff,
        dense_mesh=dense_mesh,
    )
    return lib.set_class(new_ks, (_CiderKS, ks.__class__))


class _CiderKS(_MolCiderKS):
    def __init__(
        self,
        mf,
        mlxc,
        xmix=1.0,
        nldf_init=None,
        sdmx_init=None,
        rhocut=None,
        nlc_coeff=None,
        dense_mesh=None,
    ):
        self.dense_mesh = dense_mesh
        super().__init__(mf, mlxc, xmix, nldf_init, sdmx_init, rhocut, nlc_coeff)

    def set_mlxc(
        self,
        mlxc,
        xmix=1.0,
        nldf_init=None,
        sdmx_init=None,
        rhocut=None,
        nlc_coeff=None,
    ):
        if nldf_init is None and mlxc.settings.has_nldf:
            raise NotImplementedError
        if sdmx_init is None and mlxc.settings.has_sdmx:
            sdmx_init = PySCFSDMXInitializer(mlxc.settings.sdmx_settings, lowmem=False)
        old_grids = self.grids
        changed = False
        if mlxc.settings.has_nldf:
            raise NotImplementedError
        if changed:
            for key in (
                "atom_grid",
                "atomic_radii",
                "radii_adjust",
                "radi_method",
                "becke_scheme",
                "prune",
                "level",
            ):
                self.grids.__setattr__(key, old_grids.__getattribute__(key))
        settings = mlxc.settings
        has_nldf = not settings.nldf_settings.is_empty
        has_nlof = not settings.nlof_settings.is_empty
        has_kpts = isinstance(self._numint, numint.KNumInt)
        if has_nldf and has_nlof:
            raise NotImplementedError
        elif has_nldf:
            raise NotImplementedError
        elif has_nlof:
            raise NotImplementedError
        else:
            if has_kpts:
                cls = CiderKNumInt
            else:
                cls = CiderNumInt
        self._numint = cls(
            mlxc,
            self._xc,
            nldf_init,
            sdmx_init,
            xmix=xmix,
            rhocut=rhocut,
            nlc_coeff=nlc_coeff,
            dense_mesh=self.dense_mesh,
        )
