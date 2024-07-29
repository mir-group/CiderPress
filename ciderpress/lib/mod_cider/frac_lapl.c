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

#include "spline.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLKSIZE 56
#define NPRIMAX 40
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
// #define SPLINE_SIZE 1000
// #define LMAX_1F1 7

double *SPLINE = NULL;
double *GRAD_SPLINE = NULL;
double GRID_A;
double GRID_D;
double FLAPL_S;
double CONV_ALPHA;
double CONV_ALPHA_COEFF;
int LMAX_1F1;
int SPLINE_SIZE;

int get_1f1_spline_size() { return SPLINE_SIZE; }

inline double i2r_1f1(int i) { return GRID_A * (exp(GRID_D * i) - 1); }

inline double r2i_1f1(double r) { return log(r / GRID_A + 1) / GRID_D; }

inline double deriv_r2i_1f1(double r) { return -1.0 / (r + GRID_A) / GRID_D; }

int check_1f1_initialization() {
    if (SPLINE == NULL) {
        return 0;
    } else {
        return 1;
    }
}

void initialize_spline_1f1(double *spline_buf, double *f, int size, int lmax) {
    //(double *) calloc((LMAX_1F1 + 1) * 4 * SPLINE_SIZE, sizeof(double));
    double *tmp_spline = (double *)calloc(5 * size, sizeof(double));
    double *my_spline;
    double *x = (double *)calloc(size, sizeof(double));
    int i, l;
    for (i = 0; i < size; i++) {
        x[i] = i;
    }
    for (l = 0; l <= lmax; l++) {
        get_cubic_spline_coeff(x, f + l * size, tmp_spline, size);
        my_spline = spline_buf + l * size * 4;
        for (i = 0; i < size; i++) {
            my_spline[4 * i + 0] = tmp_spline[1 * size + i];
            my_spline[4 * i + 1] = tmp_spline[2 * size + i];
            my_spline[4 * i + 2] = tmp_spline[3 * size + i];
            my_spline[4 * i + 3] = tmp_spline[4 * size + i];
        }
    }
    free(x);
    free(tmp_spline);
}

void set_spline_1f1(double *spline_buf, double a, double d, int size, int lmax,
                    double s) {
    LMAX_1F1 = lmax;
    FLAPL_S = s;
    SPLINE_SIZE = size;
    GRID_D = d;
    GRID_A = a;
    SPLINE = spline_buf;
}

void set_spline_1f1_with_grad(double *spline_buf, double *grad_buf, double a,
                              double d, int size, int lmax, double s) {
    LMAX_1F1 = lmax;
    FLAPL_S = s;
    SPLINE_SIZE = size;
    GRID_D = d;
    GRID_A = a;
    SPLINE = spline_buf;
    GRAD_SPLINE = grad_buf;
}

inline double eval_1f1(double r, double *my_spline) {
    double di = r2i_1f1(r);
    int i = (int)di;
    di -= i;
    // printf("i %d %f   %f %f %f %f\n", i, di, my_spline[i*4+0],
    // my_spline[i*4+1], my_spline[i*4+2], my_spline[i*4+3]);
    return my_spline[i * 4 + 0] +
           di * (my_spline[i * 4 + 1] +
                 di * (my_spline[i * 4 + 2] + di * my_spline[i * 4 + 3]));
}

inline double eval_1f1_grad(double r, double *my_spline) {
    double di = r2i_1f1(r);
    int i = (int)di;
    di -= i;
    // printf("i %d %f   %f %f %f %f\n", i, di, my_spline[i*4+0],
    // my_spline[i*4+1], my_spline[i*4+2], my_spline[i*4+3]);
    return (my_spline[i * 4 + 1] +
            di * (2 * my_spline[i * 4 + 2] + 3 * di * my_spline[i * 4 + 3])) *
           deriv_r2i_1f1(r);
}

void vec_eval_1f1(double *f, double *r, int n, int l) {
    double *my_spline = SPLINE + l * 4 * SPLINE_SIZE;
    int i;
    for (i = 0; i < n; i++) {
        f[i] = eval_1f1(r[i], my_spline);
    }
}

void vec_eval_1f1_grad(double *f, double *r, int n, int l) {
    double *my_spline = GRAD_SPLINE + l * 4 * SPLINE_SIZE;
    int i;
    for (i = 0; i < n; i++) {
        f[i] = eval_1f1(r[i], my_spline);
    }
}

int GTOcontract_flapl0(double *ectr, double *coord, double *alpha,
                       double *coeff, int l, int nprim, int nctr, size_t ngrids,
                       double fac) {
    size_t i, j, k;
    double arr, eprim;
    double rr[BLKSIZE];
    double *gridx = coord;
    double *gridy = coord + BLKSIZE;
    double *gridz = coord + BLKSIZE * 2;
    double sqrt_alpha;
    // fac = fac * 4 * atan(1.0) * tgamma(2 + l) / tgamma(1.5 + l);
    if (l > LMAX_1F1) {
        printf("l value too high! %d %d\n", l, LMAX_1F1);
        exit(-1);
    }
    double *my_spline = SPLINE + l * 4 * SPLINE_SIZE;
    double rmax = i2r_1f1(SPLINE_SIZE - 1);

#pragma GCC ivdep
    for (i = 0; i < ngrids; i++) {
        rr[i] = gridx[i] * gridx[i] + gridy[i] * gridy[i] + gridz[i] * gridz[i];
    }

    for (i = 0; i < nctr * BLKSIZE; i++) {
        ectr[i] = 0;
    }
    for (j = 0; j < nprim; j++) {
        sqrt_alpha = pow(alpha[j], FLAPL_S);
        // sqrt_alpha = sqrt(alpha[j]);
        for (i = 0; i < ngrids; i++) {
            arr = MIN(alpha[j] * rr[i], rmax);
            eprim = eval_1f1(arr, my_spline) * sqrt_alpha * fac;
            for (k = 0; k < nctr; k++) {
                ectr[k * BLKSIZE + i] += eprim * coeff[k * nprim + j];
            }
        }
    }
    return 1;
}

int GTOcontract_flapl1(double *ectr, double *coord, double *alpha,
                       double *coeff, int l, int nprim, int nctr, size_t ngrids,
                       double fac) {
    size_t i, j, k;
    double arr, eprim, deprim;
    double rr[BLKSIZE];
    double *gridx = coord;
    double *gridy = coord + BLKSIZE;
    double *gridz = coord + BLKSIZE * 2;
    double *ectr_2a = ectr + NPRIMAX * BLKSIZE;
    double coeff2a[NPRIMAX * NPRIMAX];
    double sqrt_alpha;
    if (l > LMAX_1F1) {
        printf("l value too high! %d %d\n", l, LMAX_1F1);
        exit(-1);
    }
    double *my_spline = SPLINE + l * 4 * SPLINE_SIZE;
    double *my_grad_spline = GRAD_SPLINE + l * 4 * SPLINE_SIZE;
    double rmax = i2r_1f1(SPLINE_SIZE - 1);

#pragma GCC ivdep
    for (i = 0; i < ngrids; i++) {
        rr[i] = gridx[i] * gridx[i] + gridy[i] * gridy[i] + gridz[i] * gridz[i];
    }

    for (i = 0; i < nctr * BLKSIZE; i++) {
        ectr[i] = 0;
        ectr_2a[i] = 0;
    }
    // -2 alpha_i C_ij exp(-alpha_i r_k^2)
    for (i = 0; i < nctr; i++) {
        for (j = 0; j < nprim; j++) {
            coeff2a[i * nprim + j] = -2. * alpha[j] * coeff[i * nprim + j];
        }
    }

    for (j = 0; j < nprim; j++) {
        sqrt_alpha = pow(alpha[j], FLAPL_S);
        for (i = 0; i < ngrids; i++) {
            arr = MIN(alpha[j] * rr[i], rmax);
            eprim = eval_1f1(arr, my_spline) * sqrt_alpha * fac;
            // deprim = eval_1f1(arr, my_grad_spline) * sqrt_alpha * fac;
            deprim = eval_1f1_grad(arr, my_spline) * sqrt_alpha * fac;
            for (k = 0; k < nctr; k++) {
                ectr[k * BLKSIZE + i] += eprim * coeff[k * nprim + j];
                ectr_2a[k * BLKSIZE + i] += deprim * coeff2a[k * nprim + j];
            }
        }
    }
    return 1;
}

void set_global_convolution_exponent(double expnt, double coeff) {
    CONV_ALPHA = expnt;
    CONV_ALPHA_COEFF = coeff;
}

int GTOcontract_conv0(double *ectr, double *coord, double *alpha, double *coeff,
                      int l, int nprim, int nctr, size_t ngrids, double fac) {
    size_t i, j, k;
    double arr, eprim;
    double rr[BLKSIZE];
    double conv_exp, conv_coeff;
    double PI = 4.0 * atan(1.0);
    double *gridx = coord;
    double *gridy = coord + BLKSIZE;
    double *gridz = coord + BLKSIZE * 2;

#pragma GCC ivdep
    for (i = 0; i < ngrids; i++) {
        rr[i] = gridx[i] * gridx[i] + gridy[i] * gridy[i] + gridz[i] * gridz[i];
    }

    for (i = 0; i < nctr * BLKSIZE; i++) {
        ectr[i] = 0;
    }
    for (j = 0; j < nprim; j++) {
        conv_exp = alpha[j] * CONV_ALPHA / (alpha[j] + CONV_ALPHA);
        conv_coeff = fac * pow(PI / CONV_ALPHA, 1.5) * CONV_ALPHA_COEFF *
                     pow(CONV_ALPHA / (alpha[j] + CONV_ALPHA), 1.5 + l);
        for (i = 0; i < ngrids; i++) {
            arr = conv_exp * rr[i];
            eprim = exp(-arr) * conv_coeff;
            for (k = 0; k < nctr; k++) {
                ectr[k * BLKSIZE + i] += eprim * coeff[k * nprim + j];
            }
        }
    }
    return 1;
}

int GTOcontract_smooth0(double *ectr, double *coord, double *alpha,
                        double *coeff, int l, int nprim, int nctr,
                        size_t ngrids, double fac) {
    size_t i, j, k;
    double arr1, arr2, eprim;
    double rr[BLKSIZE];
    // exp(-alpha * r^2) - exp(-2 * alpha * r^2)
    double conv_exp1, conv_exp2, conv_coeff1, conv_coeff2;
    double PI = 4.0 * atan(1.0);
    double *gridx = coord;
    double *gridy = coord + BLKSIZE;
    double *gridz = coord + BLKSIZE * 2;

#pragma GCC ivdep
    for (i = 0; i < ngrids; i++) {
        rr[i] = gridx[i] * gridx[i] + gridy[i] * gridy[i] + gridz[i] * gridz[i];
    }

    for (i = 0; i < nctr * BLKSIZE; i++) {
        ectr[i] = 0;
    }
    for (j = 0; j < nprim; j++) {
        conv_exp1 = alpha[j] * CONV_ALPHA / (alpha[j] + CONV_ALPHA);
        conv_exp2 = 2 * alpha[j] * CONV_ALPHA / (alpha[j] + 2 * CONV_ALPHA);
        conv_coeff1 = fac * pow(PI / CONV_ALPHA, 1.5) * CONV_ALPHA_COEFF *
                      pow(CONV_ALPHA / (alpha[j] + CONV_ALPHA), 1.5 + l);
        conv_coeff2 =
            fac * pow(PI / (2 * CONV_ALPHA), 1.5) * CONV_ALPHA_COEFF *
            pow(2 * CONV_ALPHA / (alpha[j] + 2 * CONV_ALPHA), 1.5 + l);
        for (i = 0; i < ngrids; i++) {
            arr1 = conv_exp1 * rr[i];
            arr2 = conv_exp2 * rr[i];
            eprim = exp(-arr1) * conv_coeff1 - exp(-arr2) * conv_coeff2;
            for (k = 0; k < nctr; k++) {
                ectr[k * BLKSIZE + i] += eprim * coeff[k * nprim + j];
            }
        }
    }
    return 1;
}

int GTOcontract_rsq0(double *ectr, double *coord, double *alpha, double *coeff,
                     int l, int nprim, int nctr, size_t ngrids, double fac) {
    size_t i, j, k;
    double arr, eprim;
    double rr[BLKSIZE];
    // exp(-alpha * r^2) - exp(-2 * alpha * r^2)
    double conv_exp, r0mul, r2mul;
    double PI = 4.0 * atan(1.0);
    double *gridx = coord;
    double *gridy = coord + BLKSIZE;
    double *gridz = coord + BLKSIZE * 2;
    double conv_coeff[NPRIMAX * NPRIMAX];

#pragma GCC ivdep
    for (i = 0; i < ngrids; i++) {
        rr[i] = gridx[i] * gridx[i] + gridy[i] * gridy[i] + gridz[i] * gridz[i];
    }

    for (i = 0; i < nctr * BLKSIZE; i++) {
        ectr[i] = 0;
    }
    for (i = 0; i < nctr; i++) {
        for (j = 0; j < nprim; j++) {
            conv_coeff[i * nprim + j] =
                (fac * pow(PI / CONV_ALPHA, 1.5) * CONV_ALPHA_COEFF *
                 pow(CONV_ALPHA / (alpha[j] + CONV_ALPHA), 1.5 + l)) *
                coeff[i * nprim + j];
        }
    }

    for (j = 0; j < nprim; j++) {
        conv_exp = alpha[j] * CONV_ALPHA / (alpha[j] + CONV_ALPHA);
        r0mul = (1.5 + l) / (alpha[j] + CONV_ALPHA) - l / CONV_ALPHA;
        r2mul = conv_exp * (1.0 / CONV_ALPHA - 1.0 / (alpha[j] + CONV_ALPHA));
        for (i = 0; i < ngrids; i++) {
            arr = conv_exp * rr[i];
            eprim = (r0mul + rr[i] * r2mul) * exp(-arr);
            for (k = 0; k < nctr; k++) {
                ectr[k * BLKSIZE + i] += eprim * conv_coeff[k * nprim + j];
            }
        }
    }
    return 1;
}

int GTOcontract_smooth1(double *ectr, double *coord, double *alpha,
                        double *coeff, int l, int nprim, int nctr,
                        size_t ngrids, double fac) {
    size_t i, j, k;
    double arr1, arr2, eprim, deprim;
    double rr[BLKSIZE];
    // exp(-alpha * r^2) - exp(-2 * alpha * r^2)
    double conv_exp1, conv_exp2, conv_coeff1, conv_coeff2;
    double PI = 4.0 * atan(1.0);
    double *gridx = coord;
    double *gridy = coord + BLKSIZE;
    double *gridz = coord + BLKSIZE * 2;
    double *ectr_2a = ectr + NPRIMAX * BLKSIZE;
    // double coeff2a[NPRIMAX*NPRIMAX];

#pragma GCC ivdep
    for (i = 0; i < ngrids; i++) {
        rr[i] = gridx[i] * gridx[i] + gridy[i] * gridy[i] + gridz[i] * gridz[i];
    }

    for (i = 0; i < nctr * BLKSIZE; i++) {
        ectr[i] = 0;
        ectr_2a[i] = 0;
    }

    for (j = 0; j < nprim; j++) {
        conv_exp1 = alpha[j] * CONV_ALPHA / (alpha[j] + CONV_ALPHA);
        conv_exp2 = 2 * alpha[j] * CONV_ALPHA / (alpha[j] + 2 * CONV_ALPHA);
        conv_coeff1 = fac * pow(PI / CONV_ALPHA, 1.5) * CONV_ALPHA_COEFF *
                      pow(CONV_ALPHA / (alpha[j] + CONV_ALPHA), 1.5 + l);
        conv_coeff2 =
            fac * pow(PI / (2 * CONV_ALPHA), 1.5) * CONV_ALPHA_COEFF *
            pow(2 * CONV_ALPHA / (alpha[j] + 2 * CONV_ALPHA), 1.5 + l);
        for (i = 0; i < ngrids; i++) {
            arr1 = conv_exp1 * rr[i];
            arr2 = conv_exp2 * rr[i];
            arr1 = exp(-arr1) * conv_coeff1;
            arr2 = exp(-arr2) * conv_coeff2;
            eprim = arr1 - arr2;
            deprim = -2. * conv_exp1 * arr1 + 2 * conv_exp2 * arr2;
            for (k = 0; k < nctr; k++) {
                // ectr[k*BLKSIZE+i] += eprim * coeff[k*nprim+j];
                ectr[k * BLKSIZE + i] += eprim * coeff[k * nprim + j];
                ectr_2a[k * BLKSIZE + i] += deprim * coeff[k * nprim + j];
            }
        }
    }
    return 1;
}

int GTOcontract_rsq1(double *ectr, double *coord, double *alpha, double *coeff,
                     int l, int nprim, int nctr, size_t ngrids, double fac) {
    size_t i, j, k;
    double arr, eprim, deprim, r0mul, r2mul, dr0mul, dr2mul;
    double rr[BLKSIZE];
    // exp(-alpha * r^2) - exp(-2 * alpha * r^2)
    double conv_exp;
    double PI = 4.0 * atan(1.0);
    double *gridx = coord;
    double *gridy = coord + BLKSIZE;
    double *gridz = coord + BLKSIZE * 2;
    double *ectr_2a = ectr + NPRIMAX * BLKSIZE;
    double conv_coeff[NPRIMAX * NPRIMAX];
    double conv_factor[NPRIMAX];

#pragma GCC ivdep
    for (i = 0; i < ngrids; i++) {
        rr[i] = gridx[i] * gridx[i] + gridy[i] * gridy[i] + gridz[i] * gridz[i];
    }

    for (i = 0; i < nctr * BLKSIZE; i++) {
        ectr[i] = 0;
        ectr_2a[i] = 0;
    }
    for (j = 0; j < nprim; j++) {
        conv_factor[j] = (fac * pow(PI / CONV_ALPHA, 1.5) * CONV_ALPHA_COEFF *
                          pow(CONV_ALPHA / (alpha[j] + CONV_ALPHA), 1.5 + l));
    }
    for (i = 0; i < nctr; i++) {
        for (j = 0; j < nprim; j++) {
            conv_coeff[i * nprim + j] = conv_factor[j] * coeff[i * nprim + j];
        }
    }

    for (j = 0; j < nprim; j++) {
        conv_exp = alpha[j] * CONV_ALPHA / (alpha[j] + CONV_ALPHA);
        r0mul = (1.5 + l) / (alpha[j] + CONV_ALPHA) - l / CONV_ALPHA;
        r2mul = conv_exp * (1.0 / CONV_ALPHA - 1.0 / (alpha[j] + CONV_ALPHA));
        dr0mul = -2 * conv_exp * r0mul + r2mul;
        dr2mul = -2 * conv_exp * r2mul;
        for (i = 0; i < ngrids; i++) {
            arr = exp(-conv_exp * rr[i]);
            eprim = (r0mul + rr[i] * r2mul) * arr;
            deprim = (dr0mul + rr[i] * dr2mul) * arr;
            for (k = 0; k < nctr; k++) {
                ectr[k * BLKSIZE + i] += eprim * conv_coeff[k * nprim + j];
                ectr_2a[k * BLKSIZE + i] += deprim * conv_coeff[k * nprim + j];
            }
        }
    }
    return 1;
}
