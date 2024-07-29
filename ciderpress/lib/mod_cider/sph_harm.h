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

#ifndef _SPH_HARM_CIDER
#define _SPH_HARM_CIDER

#include <complex.h>

#define SQRT2 1.4142135623730950488
#define SQRT3 1.7320508075688772936
#define SPHF0 0.28209479177387814346

typedef struct {
    int nlm; // (lmax+1) * (lmax+1)
    int lmax;
    int lp1;              // (lmax+1)
    double *coef0;        // size nlm, indexed as l, m
    double *coef1;        // size nlm, indexed as l, m
    double *c0;           // size lp1, indexed as l
    double *c1;           // size lp1, indexed as l
    double complex *ylm;  // size nlm, indexed as l, m
    double complex *dylm; // size nlm, indexed as l, m
} sphbuf;

sphbuf setup_sph_harm_buffer(int nlm);

void free_sph_harm_buffer(sphbuf buf);

void recursive_sph_harm(sphbuf buf, double *r, double *res);

void recursive_sph_harm_deriv(sphbuf buf, double *r, double *res, double *dres);

void remove_radial_grad(sphbuf buf, double *r, double *dres);

void recursive_sph_harm_vec(int nlm, int n, double *r, double *res);

void recursive_sph_harm_deriv_vec(int nlm, int n, double *r, double *res,
                                  double *dres);

#endif
