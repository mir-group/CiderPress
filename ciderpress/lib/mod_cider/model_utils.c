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

#include <math.h>
#include <omp.h>
#include <stdlib.h>

inline double _evaluate_se(double *x0, double *x1, double *exps, int nfeat) {
    double tot = 0;
    double tmp;
    for (int j = 0; j < nfeat; j++) {
        tmp = x0[j] - x1[j];
        tot += exps[j] * tmp * tmp;
    }
    return tot;
}

inline void _add_deriv(double *grad, double *x0, double *x1, double *exps,
                       double fac, int nfeat) {
    for (int j = 0; j < nfeat; j++) {
        grad[j] += 2 * exps[j] * (x1[j] - x0[j]) * fac;
    }
}

void evaluate_se_kernel(double *out, double *outd, double *xin, double *xctrl,
                        double *actrl, double *exps, int n, int nctrl,
                        int nfeat) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < nctrl; t++) {
            int iloc = i * nfeat;
            int cloc = t * nfeat;
            double tot = _evaluate_se(xin + iloc, xctrl + cloc, exps, nfeat);
            tot = actrl[t] * exp(-tot);
            out[i] += tot;
            _add_deriv(outd + iloc, xin + iloc, xctrl + cloc, exps, tot, nfeat);
        }
    }
}

void evaluate_se_kernel_antisym(double *out, double *outd, double *xin,
                                double *xctrl, double *actrl, double *exps,
                                int n, int nctrl, int nfeat) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < nctrl; t++) {
            int iloc = i * nfeat;
            double *xi = xin + iloc;
            double *od = outd + iloc;
            int cloc = t * nfeat;
            double *xc = xctrl + cloc;

            double fac = _evaluate_se(xi + 2, xc + 2, exps + 1, nfeat - 2);
            fac = actrl[t] * exp(-fac);

            double tmp = xi[0] - xc[0];
            tmp = exp(-exps[0] * tmp * tmp);
            double tot = tmp * fac;
            od[0] -= 2 * exps[0] * tmp * fac * (xi[0] - xc[0]);

            tmp = xi[0] - xc[1];
            tmp = exp(-exps[0] * tmp * tmp);
            tot -= tmp * fac;
            od[0] += 2 * exps[0] * tmp * fac * (xi[0] - xc[1]);

            tmp = xi[1] - xc[0];
            tmp = exp(-exps[0] * tmp * tmp);
            tot -= tmp * fac;
            od[1] += 2 * exps[0] * tmp * fac * (xi[1] - xc[0]);

            tmp = xi[1] - xc[1];
            tmp = exp(-exps[0] * tmp * tmp);
            tot += tmp * fac;
            od[1] -= 2 * exps[0] * tmp * fac * (xi[1] - xc[1]);

            out[i] += tot;
            _add_deriv(od + 2, xi + 2, xc + 2, exps + 1, tot, nfeat - 2);
        }
    }
}

void evaluate_se_kernel_spin(double *out, double *outd, double *xin,
                             double *xctrl, double *actrl, double *exps, int n,
                             int nctrl, int nfeat) {
    double *xin_a = xin;
    double *xin_b = xin + n * nfeat;
    double *xctrl_a = xctrl;
    double *xctrl_b = xctrl + nctrl * nfeat;
    double *outd_a = outd;
    double *outd_b = outd + n * nfeat;
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < nctrl; t++) {
            int iloc = i * nfeat;
            int cloc = t * nfeat;
            double aa = _evaluate_se(xin_a + iloc, xctrl_a + cloc, exps, nfeat);
            double ab = _evaluate_se(xin_a + iloc, xctrl_b + cloc, exps, nfeat);
            double ba = _evaluate_se(xin_b + iloc, xctrl_a + cloc, exps, nfeat);
            double bb = _evaluate_se(xin_b + iloc, xctrl_b + cloc, exps, nfeat);
            double aabb = actrl[t] * exp(-1 * (aa + bb));
            double abba = actrl[t] * exp(-1 * (ab + ba));
            out[i] += aabb + abba;
            _add_deriv(outd_a + iloc, xin_a + iloc, xctrl_a + cloc, exps, aabb,
                       nfeat);
            _add_deriv(outd_b + iloc, xin_b + iloc, xctrl_b + cloc, exps, aabb,
                       nfeat);
            _add_deriv(outd_a + iloc, xin_a + iloc, xctrl_b + cloc, exps, abba,
                       nfeat);
            _add_deriv(outd_b + iloc, xin_b + iloc, xctrl_a + cloc, exps, abba,
                       nfeat);
        }
    }
}

void evaluate_se_kernel_spin_v2(double *out, double *outd, double *xin,
                                double *xctrl, double *actrl, double *exps,
                                int n, int nctrl, int nfeat) {
    double *xin_a = xin;
    double *xin_b = xin + nfeat;
    double *xctrl_a = xctrl;
    double *xctrl_b = xctrl + nfeat;
    double *outd_a = outd;
    double *outd_b = outd + nfeat;
    int stride = 2 * nfeat;
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        // out[i] = 0;
        // for (int j = 0; j < nfeat; j++) {
        //     outd_a[i * stride + j] = 0;
        //     outd_b[i * stride + j] = 0;
        // }
        for (int t = 0; t < nctrl; t++) {
            int iloc = i * stride;
            int cloc = t * stride;
            double aa = _evaluate_se(xin_a + iloc, xctrl_a + cloc, exps, nfeat);
            double ab = _evaluate_se(xin_a + iloc, xctrl_b + cloc, exps, nfeat);
            double ba = _evaluate_se(xin_b + iloc, xctrl_a + cloc, exps, nfeat);
            double bb = _evaluate_se(xin_b + iloc, xctrl_b + cloc, exps, nfeat);
            double aabb = actrl[t] * exp(-1 * (aa + bb));
            double abba = actrl[t] * exp(-1 * (ab + ba));
            out[i] += aabb + abba;
            _add_deriv(outd_a + iloc, xin_a + iloc, xctrl_a + cloc, exps, aabb,
                       nfeat);
            _add_deriv(outd_b + iloc, xin_b + iloc, xctrl_a + cloc, exps, aabb,
                       nfeat);
            _add_deriv(outd_a + iloc, xin_a + iloc, xctrl_b + cloc, exps, abba,
                       nfeat);
            _add_deriv(outd_b + iloc, xin_b + iloc, xctrl_a + cloc, exps, abba,
                       nfeat);
        }
    }
}
