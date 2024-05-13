#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
# Edits by Kyle Bystrom <kylebystrom@gmail.com>
#

import time

import numpy
from pyscf.sgx.sgx_jk import _gen_batch_nuc, _gen_jk_direct, lib, logger


def get_jk_densities(sgx, dm, hermi=1, direct_scf_tol=1e-13):
    t0 = time.monotonic(), time.time()
    mol = sgx.mol
    grids = sgx.grids
    gthrd = sgx.grids_thrd

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1, nao, nao)
    nset = dms.shape[0]

    if sgx.debug:
        batch_nuc = _gen_batch_nuc(mol)
    else:
        batch_jk = _gen_jk_direct(
            mol, "s2", True, True, direct_scf_tol, sgx._opt, pjs=False
        )
    logger.timer_debug1(mol, "sgX initialziation", *t0)

    ngrids = grids.coords.shape[0]
    max_memory = sgx.max_memory - lib.current_memory()[0]
    sblk = sgx.blockdim
    blksize = min(ngrids, max(4, int(min(sblk, max_memory * 1e6 / 8 / nao**2))))
    tnuc = 0, 0
    ej, ek = numpy.zeros((nset, ngrids)), numpy.zeros((nset, ngrids))
    for i0, i1 in lib.prange(0, ngrids, blksize):
        coords = grids.coords[i0:i1]
        ao = mol.eval_gto("GTOval", coords)
        wao = ao * grids.weights[i0:i1, None]

        fg = lib.einsum("gi,xij->xgj", wao, dms)
        fgnw = lib.einsum("gi,xij->xgj", ao, dms)
        mask = numpy.zeros(i1 - i0, dtype=bool)
        for i in range(nset):
            mask |= numpy.any(fg[i] > gthrd, axis=1)
            mask |= numpy.any(fg[i] < -gthrd, axis=1)

        inds = numpy.arange(i0, i1)

        numpy.einsum("xgu,gu->xg", fg, ao)
        rho = numpy.einsum("xgu,gu->xg", fgnw, ao)

        if sgx.debug:
            tnuc = tnuc[0] - time.monotonic(), tnuc[1] - time.time()
            gbn = batch_nuc(mol, coords)
            tnuc = tnuc[0] + time.monotonic(), tnuc[1] + time.time()
            jg = numpy.einsum("gij,xij->xg", gbn, dms)
            gv = lib.einsum("gvt,xgt->xgv", gbn, fgnw)
            gbn = None
        else:
            tnuc = tnuc[0] - time.monotonic(), tnuc[1] - time.time()
            jg, gv = batch_jk(mol, coords, dms, fgnw.copy(), None)
            tnuc = tnuc[0] + time.monotonic(), tnuc[1] + time.time()

        jg = numpy.sum(jg, axis=0)
        ej[:, inds] = jg * rho / 2.0
        for i in range(nset):
            ek[i, inds] = lib.einsum("gu,gu->g", fgnw[i], gv[i])

        jg = gv = None

    ek *= -0.5
    return ej, ek
