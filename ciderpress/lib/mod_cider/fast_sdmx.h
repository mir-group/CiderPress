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

#ifndef FAST_SDMX_H
#define FAST_SDMX_H
#include "pyscf_gto.h"
#include <stdint.h>
#include <stdlib.h>

void SDMXeval_rad_iter(FPtr_eval_sdmx_rad feval, FPtr_exp_sdmx fexp, double fac,
                       size_t nao, size_t ngrids, size_t bgrids, int param[],
                       int *shls_slice, int *rf_loc, double *buf, double *vbas,
                       double *coord, uint8_t *non0table, int *atm, int natm,
                       int *bas, int nbas, double *env, double *alphas,
                       double *alpha_norms, int nalpha);

#endif
