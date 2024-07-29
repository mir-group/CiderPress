// CiderPress: Machine-learning based density functional theory calculations
// Copyright (C) 2024 The President and Fellows of Harvard College
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>
//
// Author: Kyle Bystrom <kylebystrom@gmail.com>
//

#include <stddef.h>
#ifndef PYSCF_GTO_H_
#define PYSCF_GTO_H_

#define CHARGE_OF 0
#define PTR_COORD 1
#define NUC_MOD_OF 2
#define PTR_ZETA 3
#define PTR_FRAC_CHARGE 4
#define RESERVE_ATMSLOT 5
#define ATM_SLOTS 6

#define ATOM_OF 0
#define ANG_OF 1
#define NPRIM_OF 2
#define NCTR_OF 3
#define KAPPA_OF 4
#define PTR_EXP 5
#define PTR_COEFF 6
#define RESERVE_BASLOT 7
#define BAS_SLOTS 8

#define POS_E1 0
#define TENSOR 1

#define LMAX ANG_MAX
#define SIMDD 8
#define NCTR_CART 128
#define BLKSIZE 56
#define NPRIMAX 40
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

#define ALIGN8_UP(buf) (void *)(((uintptr_t)buf + 7) & (-(uintptr_t)8))

typedef int (*FPtr_exp)(double *ectr, double *coord, double *alpha,
                        double *coeff, int l, int nprim, int nctr,
                        size_t ngrids, double fac);
typedef void (*FPtr_eval)(double *gto, double *ri, double *exps, double *coord,
                          double *alpha, double *coeff, double *env, int l,
                          int np, int nc, size_t nao, size_t ngrids,
                          size_t blksize);
typedef int (*FPtr_exp_sdmx)(double *ectr, double *coord, double *alpha,
                             double *coeff, int l, int nprim, int nctr,
                             size_t ngrids, double fac, double conv_alpha,
                             double conv_alpha_coeff);
typedef void (*FPtr_eval_sdmx)(double *gto, double *ri, double *exps,
                               double *coord, double *alpha, double *coeff,
                               double *env, int l, int np, int nc, size_t nao,
                               size_t ngrids, size_t blksize, double *ylm_vmg,
                               int mg_stride);
typedef void (*FPtr_eval_sdmx_rad)(double *vbas, double *exps, int nc,
                                   size_t nao, size_t ngrids, size_t blksize,
                                   int stride);

#endif
