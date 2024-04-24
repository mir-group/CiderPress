#include "fblas.h"
#include "nr_cider_numint.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CIDER_FEAT_R0_GAUSSIAN 0
#define FILL_CIDER_R0_GAUSSIAN(ind)                                            \
    tmp = 1.0 / (exp_g[g] + alphas[a]);                                        \
    tmp1 = sqrt(tmp);                                                          \
    p[ind] = pi32 * tmp * tmp1;                                                \
    dp[ind] = -1.5 * p[ind] * tmp;
#define CIDER_FEAT_R2_GAUSSIAN 1
#define FILL_CIDER_R2_GAUSSIAN(ind)                                            \
    tmp = 1.0 / (exp_g[g] + alphas[a]);                                        \
    tmp1 = sqrt(tmp);                                                          \
    tmp2 = 1.5 * pi32 * tmp * tmp * tmp1;                                      \
    p[ind] = tmp2 * exp_g[g];                                                  \
    dp[ind] = -2.5 * p[ind] * tmp + tmp2;
#define CIDER_FEAT_R4_GAUSSIAN 2
#define FILL_CIDER_R4_GAUSSIAN(ind)                                            \
    tmp = 1.0 / (exp_g[g] + alphas[a]);                                        \
    tmp1 = sqrt(tmp);                                                          \
    tmp2 = 3.75 * pi32 * tmp * tmp * tmp * tmp1 * exp_g[g];                    \
    p[ind] = tmp2 * exp_g[g];                                                  \
    dp[ind] = -3.5 * p[ind] * tmp + 2 * tmp2;
#define CIDER_FEAT_ERF_GAUSSIAN 3
#define FILL_CIDER_ERF_GAUSSIAN(ind)                                           \
    tmp = 1.0 / (exp_g[g] + alphas[a]);                                        \
    tmp2 = 1.0 / sqrt(1 + extra_args[0] * exp_g[g] * tmp);                     \
    tmp1 = sqrt(tmp);                                                          \
    p[ind] = pi32 * tmp * tmp1 * tmp2;                                         \
    dp[ind] = -1.5 * p[ind] * tmp;                                             \
    tmp2 = tmp2 * tmp;                                                         \
    dp[ind] -= 0.5 * p[ind] * tmp2 * tmp2 * extra_args[0] * alphas[a];

#define CIDER_GQ_LOOP(FEATNAME)                                                \
    for (g = 0; g < ngrids; g++) {                                             \
        p = p_ga + g * nalpha;                                                 \
        dp = dp_ga + g * nalpha;                                               \
        for (a = 0; a < nalpha; a++) {                                         \
            FILL_CIDER_##FEATNAME(a);                                          \
        }                                                                      \
    }                                                                          \
    break;
#define CIDER_QG_LOOP(FEATNAME)                                                \
    for (g = 0; g < ngrids; g++) {                                             \
        FILL_CIDER_##FEATNAME(g);                                              \
    }                                                                          \
    break;

/**
 * Fills the coefficients p_ga for auxiliary basis dcomposition.
 *
 * p_ga : coefficient array nalpha x ngrids
 * dp_ga : derivatives of coefficients wrt exponent alpha
 * exp_g : exponent array
 * alphas : control points
 * ngrids : number of realspace grids
 * nalpha : number of control points
 */
void cider_coefs_gto_gq(double *p_ga, double *dp_ga, double *exp_g,
                        double *alphas, int ngrids, int nalpha, int featid,
                        double *extra_args) {
    double pi32 = pow(4 * atan(1.0), 1.5);
#pragma omp parallel
    {
        double tmp, tmp1, tmp2;
        int g, a;
        double *p, *dp;
        switch (featid) {
        case CIDER_FEAT_R0_GAUSSIAN:
#pragma omp for
            CIDER_GQ_LOOP(R0_GAUSSIAN);
        case CIDER_FEAT_R2_GAUSSIAN:
#pragma omp for
            CIDER_GQ_LOOP(R2_GAUSSIAN);
        case CIDER_FEAT_R4_GAUSSIAN:
#pragma omp for
            CIDER_GQ_LOOP(R4_GAUSSIAN);
        case CIDER_FEAT_ERF_GAUSSIAN:
#pragma omp for
            CIDER_GQ_LOOP(ERF_GAUSSIAN);
        default:
            printf("INTERNAL CIDER ERROR\n");
        }
    }
}

void cider_coefs_gto_qg(double *p_ag, double *dp_ag, double *exp_g,
                        double *alphas, int ngrids, int nalpha, int featid,
                        double *extra_args) {
    double pi32 = pow(4 * atan(1.0), 1.5);
#pragma omp parallel
    {
        double tmp, tmp1, tmp2;
        int g, a;
        double *p, *dp;
        for (a = 0; a < nalpha; a++) {
            p = p_ag + a * ngrids;
            dp = dp_ag + a * ngrids;
            switch (featid) {
            case CIDER_FEAT_R0_GAUSSIAN:
#pragma omp for
                CIDER_QG_LOOP(R0_GAUSSIAN);
            case CIDER_FEAT_R2_GAUSSIAN:
#pragma omp for
                CIDER_QG_LOOP(R2_GAUSSIAN);
            case CIDER_FEAT_R4_GAUSSIAN:
#pragma omp for
                CIDER_QG_LOOP(R4_GAUSSIAN);
            case CIDER_FEAT_ERF_GAUSSIAN:
#pragma omp for
                CIDER_QG_LOOP(ERF_GAUSSIAN);
            default:
                printf("INTERNAL CIDER ERROR\n");
            }
        }
    }
}

/**
 * Forward pass coefs for version k descriptors
 */
void cider_coefs_vk1_gq(double *p_ga, double *dp_ga, double *exp_g,
                        double *alphas, int ngrids, int nalpha) {
#pragma omp parallel
    {
        double *tmps = (double *)malloc(nalpha * sizeof(double));
        int g, a;
        for (a = 0; a < nalpha; a++) {
            tmps[a] = -1.5 / alphas[a];
        }
#pragma omp for
        for (g = 0; g < ngrids; g++) {
            for (a = 0; a < nalpha; a++) {
                p_ga[g * nalpha + a] = exp(tmps[a] * exp_g[g]);
                dp_ga[g * nalpha + a] = tmps[a] * p_ga[g * nalpha + a];
            }
        }
    }
}

/**
 * Forward pass coefs for version k descriptors
 */
void cider_coefs_vk1_qg(double *p_ag, double *dp_ag, double *exp_g,
                        double *alphas, int ngrids, int nalpha) {
#pragma omp parallel
    {
        double tmp;
        int g, a;
        for (a = 0; a < nalpha; a++) {
            tmp = -1.5 / alphas[a];
#pragma omp for
            for (g = 0; g < ngrids; g++) {
                p_ag[a * ngrids + g] = exp(tmp * exp_g[g]);
                dp_ag[a * ngrids + g] = tmp * p_ag[a * ngrids + g];
            }
        }
    }
}

void cider_ind_etb(double *di_g, double *derivi_g, double *exp_g, int ngrids,
                   double alpha0, double lambd) {
#pragma omp parallel
    {
        int g;
        double ratio = 1.0 / log(lambd);
        double inva = 1.0 / alpha0;
#pragma omp for
        for (g = 0; g < ngrids; g++) {
            derivi_g[g] = ratio / exp_g[g];
            di_g[g] = log(exp_g[g] * inva) * ratio;
        }
    }
}

void cider_ind_zexp(double *di_g, double *derivi_g, double *exp_g, int ngrids,
                    double alpha0, double lambd) {
#pragma omp parallel
    {
        int g;
        double ratio = 1.0 / log(lambd);
        double inva = 1.0 / alpha0;
#pragma omp for
        for (g = 0; g < ngrids; g++) {
            derivi_g[g] = ratio * inva / (exp_g[g] * inva + 1);
            di_g[g] = log(exp_g[g] * inva + 1) * ratio;
        }
    }
}

void cider_coefs_spline_gq(double *p_ga, double *dp_ga, double *di_g,
                           double *w_iap, int ngrids, int nalpha, double amax,
                           double lambd) {
#pragma omp parallel
    {
        double di;
        int g, a, i;
        double *w_ap, *w_p, *p_a, *dp_a;
#pragma omp for
        for (g = 0; g < ngrids; g++) {
            p_a = p_ga + g * nalpha;
            dp_a = dp_ga + g * nalpha;
            i = (int)di_g[g];
            di = di_g[g] - i;
            w_ap = w_iap + i * 4 * nalpha;
            for (a = 0; a < nalpha; a++) {
                w_p = w_ap + 4 * a;
                p_a[a] = w_p[0] + di * (w_p[1] + di * (w_p[2] + di * w_p[3]));
                dp_a[a] = w_p[1] + di * (2 * w_p[2] + di * 3 * w_p[3]);
            }
        }
    }
}

void cider_coefs_spline_qg(double *p_ag, double *dp_ag, double *di_g,
                           double *w_iap, int ngrids, int nalpha, double amax,
                           double lambd) {
#pragma omp parallel
    {
        double di;
        int g, a, i;
        double *w_p, *p_g, *dp_g;
        for (a = 0; a < nalpha; a++) {
            p_g = p_ag + a * ngrids;
            dp_g = dp_ag + a * ngrids;
#pragma omp for
            for (g = 0; g < ngrids; g++) {
                i = (int)di_g[g];
                di = di_g[g] - i;
                w_p = w_iap + (i * nalpha + a) * 4;
                p_g[g] = w_p[0] + di * (w_p[1] + di * (w_p[2] + di * w_p[3]));
                dp_g[g] = w_p[1] + di * (2 * w_p[2] + di * 3 * w_p[3]);
            }
        }
    }
}
