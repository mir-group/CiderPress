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

from itertools import combinations

import numpy as np
import pyscf.lib
from interpolation.splines import UCGrid, filter_cubic

from ciderpress.models.kernels import (
    DiffAdditiveMixin,
    DiffARBF,
    DiffProduct,
    SubsetAddLLRBF,
    SubsetAddRQ,
    SubsetARBF,
    SubsetRBF,
    arbf_args,
)

"""
Script for mapping a CIDER GP to a cubic spline.
Requires an input DFTGPR object, stored in joblib format.
"""


def get_dim(x, length_scale, density=6, buff=0.0, bound=None, max_ngrid=None):
    print(length_scale, bound)
    mini = np.min(x) - buff
    maxi = np.max(x) + buff
    if bound is not None:
        mini, maxi = bound[0], bound[1]
    ran = maxi - mini
    ngrid = max(int(density * ran / length_scale) + 1, 3)
    if max_ngrid is not None and ngrid > max_ngrid:
        ngrid = max_ngrid
    return (mini, maxi, ngrid)


def project_kernel_onto_grid(alpha, k0s, dims):
    N = len(dims)
    dims = [(float(dim[0]), float(dim[1]), int(dim[2])) for dim in dims]
    if N == 0:
        raise ValueError("Kernel is constant!")
    elif N == 1:
        fps = np.dot(alpha, k0s[0])
        sgd = UCGrid(dims[0])
    elif N == 2:
        k = np.einsum("ni,nj->nij", k0s[0], k0s[1])
        sgd = UCGrid(dims[0], dims[1])
        fps = np.einsum("n,nij->ij", alpha, k)
    elif N == 3:
        fps = 0
        for p0, p1 in pyscf.lib.prange(0, len(k0s[0]), 200):
            k = np.einsum("ni,nj,nk->nijk", k0s[0][p0:p1], k0s[1][p0:p1], k0s[2][p0:p1])
            fps += np.einsum("n,nijk->ijk", alpha[p0:p1], k)
        sgd = UCGrid(dims[0], dims[1], dims[2])
    elif N == 4:
        fps = 0
        for p0, p1 in pyscf.lib.prange(0, len(k0s[0]), 20):
            k = np.einsum(
                "ni,nj,nk,nl->nijkl",
                k0s[0][p0:p1],
                k0s[1][p0:p1],
                k0s[2][p0:p1],
                k0s[3][p0:p1],
            )
            fps += np.einsum("n,nijkl->ijkl", alpha[p0:p1], k)
        # k = np.einsum('ni,nj,nk,nl->nijkl', k0s[0], k0s[1], k0s[2], k0s[3])
        # funcps.append(np.einsum('n,nijkl->ijkl', alpha, k))
        sgd = UCGrid(dims[0], dims[1], dims[2], dims[3])
    else:
        raise ValueError("Order too high!")
    return fps, sgd


def get_mapped_gp_evaluator_linear(kernel, X, alpha):
    N = X.shape[1]
    assert N == alpha.size
    XT = np.identity(N)
    coefs = kernel(XT, X).dot(alpha)
    from ciderpress.dft.xc_evaluator import GlobalLinearEvaluator

    return GlobalLinearEvaluator(coefs)


def get_mapped_gp_evaluator_simple(
    rbf,
    X,
    alpha,
    feature_list,
    rbf_density=8,
    max_ngrid=120,
):
    """
    Quick approach for pure RBF GP mapping for N <= 4
    """
    N = X.shape[1]
    length_scale = rbf.k2.length_scale
    scale = [rbf.k1.constant_value]

    inds = np.arange(N)
    if isinstance(rbf.k2, SubsetRBF):
        inds = inds[rbf.k2.indexes]
    inds = list(inds)
    D = X[:, inds]
    N = D.shape[1]

    dims = []
    for i in range(N):
        dims.append(
            get_dim(
                D[:, i],
                length_scale[i],
                density=rbf_density,
                bound=feature_list[inds[i]].bounds,
                max_ngrid=max_ngrid,
            )
        )
    grid = [np.linspace(dims[i][0], dims[i][1], dims[i][2]) for i in range(N)]

    k0s = []
    for i in range(D.shape[1]):
        diff = (D[:, i : i + 1] - grid[i][np.newaxis, :]) / length_scale[i]
        k0s.append(np.exp(-0.5 * diff**2))

    funcp, spline_gd = project_kernel_onto_grid(alpha, k0s, dims)
    ind_sets = [inds]
    funcps = [funcp]
    spline_grids = [spline_gd]
    coeff_sets = [filter_cubic(spline_grids[0], funcps[0])]
    return scale, ind_sets, spline_grids, coeff_sets


def get_mapped_gp_evaluator_additive(
    kernel,
    X,
    alpha,
    feature_list,
    srbf_density=8,
    arbf_density=8,
    max_ngrid=120,
):
    N = X.shape[1]
    dims = []
    if isinstance(kernel, (DiffARBF, DiffAdditiveMixin)):
        srbf = None
        arbf = kernel
        ndim, length_scale, scale, order = arbf_args(arbf)
        sinds = np.zeros((0,))
        ainds = np.arange(N)[arbf.indexes]
        inds = ainds.copy()
    else:
        assert isinstance(kernel, DiffProduct)
        srbf = kernel.k1
        arbf = kernel.k2
        assert isinstance(srbf, SubsetRBF)
        assert isinstance(arbf, (SubsetARBF, SubsetAddRQ, SubsetAddLLRBF))
        ndim, length_scale, scale, order = arbf_args(arbf)
        length_scale = np.append(srbf.length_scale, length_scale)
        sinds = np.arange(N)[srbf.indexes]
        ainds = np.arange(N)[arbf.indexes]
        inds = np.append(sinds, ainds)

    D = X[:, inds]
    N = D.shape[1]
    for i in range(N):
        density = srbf_density if i < len(sinds) else arbf_density
        dims.append(
            get_dim(
                D[:, i],
                length_scale[i],
                density=density,
                bound=feature_list[inds[i]].bounds,
                max_ngrid=max_ngrid,
            )
        )
    grid = [np.linspace(dims[i][0], dims[i][1], dims[i][2]) for i in range(N)]
    k0s = []
    if isinstance(arbf, DiffARBF):
        for i in range(D.shape[1]):
            diff = (D[:, i : i + 1] - grid[i][np.newaxis, :]) / length_scale[i]
            k0s.append(np.exp(-0.5 * diff**2))
    elif isinstance(arbf, DiffAdditiveMixin):
        assert srbf is None
        for i in range(D.shape[1]):
            k0s.append(arbf.get_k0_for_mapping(D[:, i], grid[i], length_scale[i]))
    else:
        raise ValueError

    funcps = []
    spline_grids = []
    ind_sets = []
    arange_s = np.arange(len(sinds)).astype(int)
    arange_a = len(sinds) + np.arange(len(ainds)).astype(int)
    const = 0
    for o in range(order + 1):
        for tmp_inds in combinations(arange_a, o):
            if len(sinds > 0):
                tmp_inds = np.append(arange_s, tmp_inds).astype(int).tolist()
            else:
                tmp_inds = np.array(tmp_inds).astype(int).tolist()
            if len(tmp_inds) == 0:
                # raise NotImplementedError
                const += scale[0] * np.sum(alpha)
            else:
                ind_sets.append(list(inds[tmp_inds]))
                funcp, spline_gd = project_kernel_onto_grid(
                    alpha,
                    [k0s[i] for i in tmp_inds],
                    [dims[i] for i in tmp_inds],
                )
                funcps.append(funcp)
                spline_grids.append(spline_gd)

    coeff_sets = []
    for i in range(len(funcps)):
        coeff_sets.append(filter_cubic(spline_grids[i], funcps[i]))

    if len(sinds) == 0:
        scale = scale[1:]
        return scale, ind_sets, spline_grids, coeff_sets, const
    return scale, ind_sets, spline_grids, coeff_sets
