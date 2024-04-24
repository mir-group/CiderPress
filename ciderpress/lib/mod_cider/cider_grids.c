#include "fblas.h"
#include <math.h>
#include <stdlib.h>

void reduce_angc_to_ylm(double *theta_rlmq, double *y_glm, double *theta_gq,
                        int *rad_loc, int *ylm_loc, int nalpha, int nrad,
                        int ngrids, int nlm, int stride, int offset) {
    theta_gq = theta_gq + offset;
#pragma omp parallel
    {
        double ALPHA = 1.0;
        double BETA = 0.0;
        char NTRANS = 'N';
        char TRANS = 'T';
        int r, nw;
        double *y_wlm, *theta_lmq, *theta_wq;
#pragma omp for schedule(dynamic, 4)
        for (r = 0; r < nrad; r++) {
            y_wlm = y_glm + ylm_loc[r] * nlm;
            nw = rad_loc[r + 1] - rad_loc[r];
            theta_lmq = theta_rlmq + r * nlm * nalpha;
            theta_wq = theta_gq + rad_loc[r] * stride;
            dgemm_(&NTRANS, &TRANS, &nalpha, &nlm, &nw, &ALPHA, theta_wq,
                   &stride, y_wlm, &nlm, &BETA, theta_lmq, &nalpha);
        }
    }
}

void reduce_ylm_to_angc(double *theta_rlmq, double *y_glm, double *theta_gq,
                        int *rad_loc, int *ylm_loc, int nalpha, int nrad,
                        int ngrids, int nlm, int stride, int offset) {
    theta_gq = theta_gq + offset;
#pragma omp parallel
    {
        double ALPHA = 1.0;
        double BETA = 0.0;
        char NTRANS = 'N';
        char TRANS = 'T';
        int r, nw;
        double *y_wlm, *theta_lmq, *theta_wq;
#pragma omp for schedule(dynamic, 4)
        for (r = 0; r < nrad; r++) {
            y_wlm = y_glm + ylm_loc[r] * nlm;
            nw = rad_loc[r + 1] - rad_loc[r];
            theta_lmq = theta_rlmq + r * nlm * nalpha;
            theta_wq = theta_gq + rad_loc[r] * stride;
            dgemm_(&NTRANS, &NTRANS, &nalpha, &nw, &nlm, &ALPHA, theta_lmq,
                   &nalpha, y_wlm, &nlm, &BETA, theta_wq, &stride);
        }
    }
}
