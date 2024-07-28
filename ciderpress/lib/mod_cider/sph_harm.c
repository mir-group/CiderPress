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

#include "sph_harm.h"
#include <complex.h>
#include <math.h>
#include <stdlib.h>

const double FAC_LIST[24] = {-SQRT2, SQRT2, -SQRT2, SQRT2, -SQRT2, SQRT2,
                             -SQRT2, SQRT2, -SQRT2, SQRT2, -SQRT2, SQRT2,
                             -SQRT2, SQRT2, -SQRT2, SQRT2, -SQRT2, SQRT2,
                             -SQRT2, SQRT2, -SQRT2, SQRT2, -SQRT2, SQRT2};

sphbuf setup_sph_harm_buffer(int nlm) {
    sphbuf buf;
    buf.nlm = nlm;
    buf.lmax = (int)(sqrt(nlm - 1) + 1e-7);
    buf.lp1 = buf.lmax + 1;
    buf.coef0 = calloc(buf.nlm, sizeof(double));
    buf.coef1 = calloc(buf.nlm, sizeof(double));
    buf.c0 = calloc(buf.lp1, sizeof(double));
    buf.c1 = calloc(buf.lp1, sizeof(double));
    buf.ylm = calloc(buf.nlm, sizeof(double complex));
    buf.dylm = calloc(3 * buf.nlm, sizeof(double complex));
    int l, m, ind;

    for (ind = 0; ind < nlm; ind++) {
        buf.ylm[ind] = 0;
    }
    for (ind = 0; ind < 3 * nlm; ind++) {
        buf.dylm[ind] = 0;
    }

    for (l = 0; l < buf.lp1; l++) {
        for (m = 0; m < buf.lp1; m++) {
            ind = l * buf.lp1 + m;
            if (m + 2 <= l) {
                buf.coef0[ind] =
                    sqrt((double)((2 * l + 3) * (l - m) * (l - m - 1)) /
                         ((2 * l - 1) * (l + m + 2) * (l + m + 1)));
            } else {
                buf.coef0[ind] = 0.0;
            }
            if (m <= l) {
                buf.coef1[ind] = -sqrt((double)((2 * l + 3) * (2 * l + 1)) /
                                       ((l + m + 2) * (l + m + 1)));
            } else {
                buf.coef1[ind] = 0.0;
            }
        }
        buf.c0[l] = sqrt((double)(2 * l + 3) * (2 * l + 1)) / (l + 1);
        buf.c1[l] =
            sqrt((double)(2 * l + 3) / (2 * l - 1)) * (double)l / (l + 1);
    }

    return buf;
}

void free_sph_harm_buffer(sphbuf buf) {
    free(buf.coef0);
    free(buf.coef1);
    free(buf.c0);
    free(buf.c1);
    free(buf.ylm);
    free(buf.dylm);
}

void recursive_sph_harm(sphbuf buf, double *restrict r, double *restrict res) {
    double complex xy = r[0] + I * r[1];
    double z = r[2];
    double *restrict coef0 = buf.coef0;
    double *restrict coef1 = buf.coef1;
    double *restrict c0 = buf.c0;
    double *restrict c1 = buf.c1;
    double complex *restrict ylm = buf.ylm;
    int lp1 = buf.lp1;
    int nlm = buf.nlm;

    int l, m, lm, ind;
    double fac;

    ylm[0 * lp1 + 0] = SPHF0;
    ylm[1 * lp1 + 0] = SQRT3 * SPHF0 * z;
    res[0] = SPHF0;
    res[2] = creal(ylm[1 * lp1 + 0]);
    // res[3] = creal(ylm[1*lp1+0]);

    ylm[1 * lp1 + 1] = coef1[0 * lp1 + 0] * xy * ylm[0 * lp1 + 0];
    fac = -SQRT2;
    res[1] = fac * cimag(ylm[1 * lp1 + 1]);
    res[3] = fac * creal(ylm[1 * lp1 + 1]);
    // res[1] = fac * creal(ylm[1*lp1+1]);
    // res[2] = fac * cimag(ylm[1*lp1+1]);
    for (l = 1; l < buf.lmax; l++) {
        lm = l * l + 3 * l + 2;
        ylm[(l + 1) * lp1 + 0] =
            c0[l] * z * ylm[l * lp1 + 0] - c1[l] * ylm[(l - 1) * lp1 + 0];
        res[lm] = creal(ylm[(l + 1) * lp1 + 0]);
        for (m = 0; m <= l; m++) {
            ind = l * lp1 + m;
            ylm[ind + lp1 + 1] =
                coef0[ind] * ylm[ind - lp1 + 1] + coef1[ind] * xy * ylm[ind];
            res[lm - m - 1] = FAC_LIST[m] * cimag(ylm[ind + lp1 + 1]);
            res[lm + m + 1] = FAC_LIST[m] * creal(ylm[ind + lp1 + 1]);
        }
    }
}

void recursive_sph_harm_deriv(sphbuf buf, double *r, double *res,
                              double *dres) {
    double complex xy = r[0] + I * r[1];
    double z = r[2];
    double *coef0 = buf.coef0;
    double *coef1 = buf.coef1;
    double *c0 = buf.c0;
    double *c1 = buf.c1;
    double complex *ylm = buf.ylm;
    double complex *dylmx = buf.dylm + 0 * buf.nlm;
    double complex *dylmy = buf.dylm + 1 * buf.nlm;
    double complex *dylmz = buf.dylm + 2 * buf.nlm;
    int lp1 = buf.lp1;
    int nlm = buf.nlm;
    double *dresx = dres + 0 * buf.nlm;
    double *dresy = dres + 1 * buf.nlm;
    double *dresz = dres + 2 * buf.nlm;

    int l, m, lm, ind, indp1, indm1;
    double fac = -SQRT2;

    for (lm = 0; lm < nlm; lm++) {
        ylm[lm] = 0;
        dylmx[lm] = 0;
        dylmy[lm] = 0;
        dylmz[lm] = 0;
    }

    ylm[0 * lp1 + 0] = SPHF0;
    res[0] = SPHF0;
    dresx[0] = 0.0;
    dresy[0] = 0.0;
    dresz[0] = 0.0;

    ind = 1 * lp1;
    ylm[ind] = SQRT3 * SPHF0 * z;
    dylmz[ind] = SQRT3 * SPHF0;
    res[2] = creal(ylm[ind]);
    ind++;
    ylm[ind] = coef1[0] * xy * ylm[0];
    dylmx[ind] = coef1[0] * ylm[0];
    dylmy[ind] = I * coef1[0] * ylm[0];
    res[3] = fac * creal(ylm[ind]);
    res[1] = fac * cimag(ylm[ind]);
    dresx[3] = fac * creal(dylmx[ind]);
    dresx[1] = 0.0;
    dresx[2] = 0.0;
    dresy[3] = 0.0;
    dresy[1] = fac * cimag(dylmy[ind]);
    dresy[2] = 0.0;
    dresz[3] = 0.0;
    dresz[1] = 0.0;
    dresz[2] = creal(dylmz[lp1]);

    for (l = 1; l < buf.lmax; l++) {
        lm = l * l + 3 * l + 2;
        ind = l * lp1;
        ylm[ind + lp1] = c0[l] * z * ylm[ind] - c1[l] * ylm[ind - lp1];
        // TODO if algo changes and dylmx/y becomes nonzero for m=0,
        // need to include those derivatives here.
        // dylmx[ind+lp1] = c0[l] * z * dylmx[ind] - c1[l] * dylmx[ind-lp1];
        // dylmy[ind+lp1] = c0[l] * z * dylmy[ind] - c1[l] * dylmy[ind-lp1];
        dylmz[ind + lp1] = c0[l] * ylm[ind] + c0[l] * z * dylmz[ind] -
                           c1[l] * dylmz[ind - lp1];
        res[lm] = creal(ylm[ind + lp1]);
        dresx[lm] = 0.0;
        dresy[lm] = 0.0;
        dresz[lm] = creal(dylmz[ind + lp1]);
        fac = -SQRT2;
        for (m = 0; m <= l; m++) {
            ind = l * lp1 + m;
            indp1 = ind + lp1 + 1;
            indm1 = ind - lp1 + 1;
            ylm[indp1] = coef0[ind] * ylm[indm1] + coef1[ind] * xy * ylm[ind];
            dylmx[indp1] =
                coef0[ind] * dylmx[indm1] + coef1[ind] * xy * dylmx[ind];
            dylmy[indp1] =
                coef0[ind] * dylmy[indm1] + coef1[ind] * xy * dylmy[ind];
            dylmz[indp1] =
                coef0[ind] * dylmz[indm1] + coef1[ind] * xy * dylmz[ind];
            dylmx[indp1] += coef1[ind] * ylm[ind];
            dylmy[indp1] += coef1[ind] * ylm[ind] * I;
            ind = indp1;
            indm1 = lm - m - 1;
            indp1 = lm + m + 1;
            res[indm1] = fac * cimag(ylm[ind]);
            dresx[indm1] = fac * cimag(dylmx[ind]);
            dresy[indm1] = fac * cimag(dylmy[ind]);
            dresz[indm1] = fac * cimag(dylmz[ind]);
            res[indp1] = fac * creal(ylm[ind]);
            dresx[indp1] = fac * creal(dylmx[ind]);
            dresy[indp1] = fac * creal(dylmy[ind]);
            dresz[indp1] = fac * creal(dylmz[ind]);
            fac = -fac;
        }
    }
    remove_radial_grad(buf, r, dres);
}

void remove_radial_grad(sphbuf buf, double *r, double *dres) {
    int nlm = buf.nlm;
    double *dresx = dres + 0 * nlm;
    double *dresy = dres + 1 * nlm;
    double *dresz = dres + 2 * nlm;
    int i;
    double tmp;
    for (i = 0; i < nlm; i++) {
        tmp = dresx[i] * r[0] + dresy[i] * r[1] + dresz[i] * r[2];
        dresx[i] -= tmp * r[0];
        dresy[i] -= tmp * r[1];
        dresz[i] -= tmp * r[2];
    }
}

void recursive_sph_harm_vec(int nlm, int n, double *r, double *res) {
    int i;
    sphbuf buf = setup_sph_harm_buffer(nlm);
    for (i = 0; i < n; i++) {
        recursive_sph_harm(buf, r + 3 * i, res + nlm * i);
    }
    free_sph_harm_buffer(buf);
}

void recursive_sph_harm_deriv_vec(int nlm, int n, double *r, double *res,
                                  double *dres) {
    int i;
    sphbuf buf = setup_sph_harm_buffer(nlm);
    for (i = 0; i < n; i++) {
        recursive_sph_harm_deriv(buf, r + 3 * i, res + nlm * i,
                                 dres + 3 * nlm * i);
    }
    free_sph_harm_buffer(buf);
}
