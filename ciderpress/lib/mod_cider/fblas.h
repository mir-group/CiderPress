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

#ifndef _CIDER_FBLAS_H
#define _CIDER_FBLAS_H

#if defined __cplusplus
extern "C" {
#endif
#include <complex.h>

double ddot_(const int *, const double *, const int *, const double *,
             const int *);

void dgemv_(const char *, const int *, const int *, const double *,
            const double *, const int *, const double *, const int *,
            const double *, const double *, const int *);

void dtrtri_(const char *, const char *, const int *, const double *,
             const int *, const int *);

void dtrmv_(const char *, const char *, const char *, const int *,
            const double *, const int *, const double *, const int *);

void dgemm_(const char *, const char *, const int *, const int *, const int *,
            const double *, const double *, const int *, const double *,
            const int *, const double *, double *, const int *);

void dpotrs_(const char *, const int *, const int *, const double *,
             const int *, const double *, const int *, const int *);

void dpotrf_(const char *, const int *, const double *, const int *,
             const int *);

#if defined __cplusplus
} // end extern "C"
#endif

#endif
