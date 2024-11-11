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

#include <complex.h>
#include <math.h>

void eval_cubic_spline(double *spline_ntp, double *funcs_ng, int *t_g,
                       double *dt_g, int nn, int nt, int ng) {
    int n, g, t;
    int np = 4;
    double dt;
    double *spline_p;
    double *spline_tp;
    double *funcs_g;
    for (n = 0; n < nn; n++) {
        funcs_g = funcs_ng + n * ng;
        spline_tp = spline_ntp + n * nt * np;
        for (g = 0; g < ng; g++) {
            t = t_g[g];
            dt = dt_g[g];
            spline_p = spline_tp + t * np;
            funcs_g[g] =
                spline_p[0] +
                dt * (spline_p[1] + dt * (spline_p[2] + dt * spline_p[3]));
        }
    }
}

void eval_cubic_spline_deriv(double *spline_ntp, double *funcs_ng, int *t_g,
                             double *dt_g, int nn, int nt, int ng) {
    int n, g, t;
    int np = 4;
    double dt;
    double *spline_p;
    double *spline_tp;
    double *funcs_g;
    for (n = 0; n < nn; n++) {
        funcs_g = funcs_ng + n * ng;
        spline_tp = spline_ntp + n * nt * np;
        for (g = 0; g < ng; g++) {
            t = t_g[g];
            dt = dt_g[g];
            spline_p = spline_tp + t * np;
            funcs_g[g] =
                spline_p[1] + dt * (2 * spline_p[2] + 3 * dt * spline_p[3]);
        }
    }
}

void eval_pasdw_funcs(double *radfuncs_ng, double *ylm_lg, double *funcs_ig,
                      int *nlist_i, int *lmlist_i, int ni, int ng, int nlm) {
    int n, i, g, lm;
    double *funcs_g;
    double *radfuncs_g;
    double *ylm_g;
    for (i = 0; i < ni; i++) {
        n = nlist_i[i];
        lm = lmlist_i[i];
        funcs_g = funcs_ig + i * ng;
        radfuncs_g = radfuncs_ng + n * ng;
        ylm_g = ylm_lg + lm * ng;
        for (g = 0; g < ng; g++) {
            funcs_g[g] = radfuncs_g[g] * ylm_g[g];
        }
    }
}

void pasdw_reduce_i(double *coefs_i, double *funcs_ig, double *augfeat_g,
                    int *indlist, int ni, int ng, int n1, int n2, int n3) {
    // NOTE n1 is the first index in row-major, i.e. long stride
    // TODO should the order of the loops be flipped to minimize jumping?
    int i, g, index, ind1, ind2, ind3;
    for (i = 0; i < ni; i++) {
        for (g = 0; g < ng; g++) {
            ind1 = indlist[3 * g + 0];
            ind2 = indlist[3 * g + 1];
            ind3 = indlist[3 * g + 2];
            index = (ind1 * n2 + ind2) * n3 + ind3;
            augfeat_g[index] += funcs_ig[i * ng + g] * coefs_i[i];
        }
    }
}

void pasdw_reduce_g(double *coefs_i, double *funcs_ig, double *augfeat_g,
                    int *indlist, int ni, int ng, int n1, int n2, int n3) {
    int i, g, index, ind1, ind2, ind3;
    for (i = 0; i < ni; i++) {
        for (g = 0; g < ng; g++) {
            ind1 = indlist[3 * g + 0];
            ind2 = indlist[3 * g + 1];
            ind3 = indlist[3 * g + 2];
            index = (ind1 * n2 + ind2) * n3 + ind3;
            coefs_i[i] += funcs_ig[i * ng + g] * augfeat_g[index];
        }
    }
}

void eval_cubic_interp(int *i_g, double *t_g, double *c_ip, double *y_g,
                       double *dy_g, int ng, int ni) {
    int i, g;
    int np = 4;
    double t;
    double *c_p;
    for (g = 0; g < ng; g++) {
        i = i_g[g];
        t = t_g[g];
        c_p = c_ip + i * np;
        y_g[g] = c_p[0] + t * (c_p[1] + t * (c_p[2] + t * c_p[3]));
        dy_g[g] = c_p[1] + t * (2.0 * c_p[2] + 3.0 * t * c_p[3]);
    }
}

void eval_cubic_interp_noderiv(int *i_g, double *t_g, double *c_ip, double *y_g,
                               int ng, int ni) {
    int i, g;
    int np = 4;
    double t;
    double *c_p;
    for (g = 0; g < ng; g++) {
        i = i_g[g];
        t = t_g[g];
        c_p = c_ip + i * np;
        y_g[g] = c_p[0] + t * (c_p[1] + t * (c_p[2] + t * c_p[3]));
    }
}

void mulexp(double complex *F_k, double complex *theta_k, double *k2_k,
            double a, double b, int nk) {
    for (int k = 0; k < nk; k++) {
        F_k[k] += a * theta_k[k] * exp(-b * k2_k[k]);
    }
}
