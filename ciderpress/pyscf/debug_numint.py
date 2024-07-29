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
from pyscf.dft.gen_grid import Grids
from pyscf.dft.numint import NumInt

from ciderpress.dft.debug_numint import get_get_exponent, get_nonlocal_features


def get_nldf_numint(
    mol, grids, dms, gg_kwargs, vv_gg_kwargs, version="i", inner_grids_level=3
):
    if dms.ndim == 2:
        return _get_nldf_numint(
            mol,
            grids,
            dms,
            gg_kwargs,
            vv_gg_kwargs,
            version=version,
            inner_grids_level=inner_grids_level,
        )[None, :]
    else:
        assert dms.shape[0] == 2
        dms = dms * 2
        res0 = _get_nldf_numint(
            mol,
            grids,
            dms[0],
            gg_kwargs,
            vv_gg_kwargs,
            version=version,
            inner_grids_level=inner_grids_level,
        )
        res1 = _get_nldf_numint(
            mol,
            grids,
            dms[1],
            gg_kwargs,
            vv_gg_kwargs,
            version=version,
            inner_grids_level=inner_grids_level,
        )
        return np.stack([res0, res1])


def _get_nldf_numint(
    mol, grids, dms, gg_kwargs, vv_gg_kwargs, version="i", inner_grids_level=3
):
    ni = NumInt()
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, 1, False)
    max_memory = 2000
    assert nset == 1

    get_exponent_r = get_get_exponent(gg_kwargs)
    get_exponent_rp = get_get_exponent(vv_gg_kwargs)

    assert nset == 1
    smgrids = Grids(mol)
    smgrids.level = inner_grids_level
    smgrids.prune = None
    smgrids.build()

    nfeat = 12 if version == "i" else 4
    desc = np.empty((0, nfeat))
    ao_deriv = 1
    vvrho = np.empty([nset, 5, 0])
    vvweight = np.empty([nset, 0])
    vvcoords = np.empty([nset, 0, 3])
    for ao, mask, weight, coords in ni.block_loop(
        mol, smgrids, nao, ao_deriv, max_memory
    ):
        rhotmp = np.empty([0, 5, weight.size])
        weighttmp = np.empty([0, weight.size])
        coordstmp = np.empty([0, weight.size, 3])
        for idm in range(nset):
            rho = make_rho(idm, ao, mask, "MGGA")
            rho = np.expand_dims(rho, axis=0)
            rhotmp = np.concatenate((rhotmp, rho), axis=0)
            weighttmp = np.concatenate(
                (weighttmp, np.expand_dims(weight, axis=0)),
                axis=0,
            )
            coordstmp = np.concatenate(
                (coordstmp, np.expand_dims(coords, axis=0)),
                axis=0,
            )
        vvrho = np.concatenate((vvrho, rhotmp), axis=2)
        vvweight = np.concatenate((vvweight, weighttmp), axis=1)
        vvcoords = np.concatenate((vvcoords, coordstmp), axis=1)
    aow = None
    for ao, mask, weight, coords in ni.block_loop(
        mol, grids, nao, ao_deriv, max_memory
    ):
        aow = np.ndarray(ao[0].shape, order="F", buffer=aow)
        for idm in range(nset):
            rho = make_rho(idm, ao, mask, "MGGA")
            feat = get_nonlocal_features(
                rho,
                coords,
                vvrho[idm],
                vvweight[idm],
                vvcoords[idm],
                get_exponent_r,
                get_exponent_rp,
                version=version,
            )
            desc = np.append(desc, feat, axis=0)
    return desc
