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

#ifndef _CONV_INTERPOLATION_H
#define _CONV_INTERPOLATION_H

typedef struct {
    int *loc_i;
    int *num_i;
    double *rel_ord_coords;
    int *ind_ord_fwd;
    int *ind_ord_bwd;
    int nrad;
    double aparam;
    double dparam;
    int ngrids;
    int buffer_size;
} spline_locator;

typedef struct {
    spline_locator *sloc_list;
    int natm;
} spline_loc_list;

#endif
