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
#include <stdlib.h>

void get_cubic_spline_coeff(double *x, double *y, double *spline, int N) {
    double **coeff = (double **)malloc(3 * sizeof(double *));
    coeff[0] = spline + 2 * N;
    coeff[1] = spline + 3 * N;
    coeff[2] = spline + 4 * N;

    int i;
    for (i = 0; i < N; i++) {
        spline[i] = x[i];
        spline[N + i] = y[i];
    }

    double d1p1 = (y[1] - y[0]) / (x[1] - x[0]);
    if (d1p1 > 0.99E30) {
        coeff[1][0] = 0;
        coeff[0][0] = 0;
    } else {
        coeff[1][0] = -0.5;
        coeff[0][0] =
            (3 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - d1p1);
    }

    double s = 0, r = 0;

    for (i = 1; i < N - 1; i++) {
        s = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
        r = s * coeff[1][i - 1] + 2;
        coeff[1][i] = (s - 1) / r;
        coeff[0][i] = (6 *
                           ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) -
                            (y[i] - y[i - 1]) / (x[i] - x[i - 1])) /
                           (x[i + 1] - x[i - 1]) -
                       s * coeff[0][i - 1]) /
                      r;
    }

    coeff[0][N - 1] = 0;
    coeff[1][N - 1] = 0;
    coeff[2][N - 1] = 0;

    for (i = N - 2; i >= 0; i--) {
        coeff[1][i] = coeff[1][i] * coeff[1][i + 1] + coeff[0][i];
    }

    for (i = 0; i < N - 1; i++) {
        s = x[i + 1] - x[i];
        r = (coeff[1][i + 1] - coeff[1][i]) / 6;
        coeff[2][i] = r / s;
        coeff[1][i] = coeff[1][i] / 2;
        coeff[0][i] = (y[i + 1] - y[i]) / s - (coeff[1][i] + r) * s;
    }

    free(coeff);
}

double spline_integral(double *spline, int N) {
    double *x = spline;
    double *a = spline + N;
    double *b = spline + 2 * N;
    double *c = spline + 3 * N;
    double *d = spline + 4 * N;
    double dx = 0;
    double integral = 0;
    int i;
    for (i = 0; i < N - 1; i++) {
        dx = x[i + 1] - x[i];
        integral +=
            dx * (a[i] + dx * (b[i] / 2 + dx * (c[i] / 3 + d[i] * dx / 4)));
    }

    return integral;
}
