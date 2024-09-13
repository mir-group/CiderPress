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
}

inline void _add_deriv(double *grad, double *x0, double *x1, double *exps,
                       double fac, int nfeat) {
    for (int j = 0; j < nfeat; j++) {
        grad[j] += 2 * exps[j] * (x1[j] - x0[j]) * fac;
    }
}

void evaluate_se_kernel(double *out, double *outd, double *xin, double *xctrl,
                        double *actrl, double *exps, int n, int nctrl,
                        int nfeat, int nspin) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        out[i] = 0;
        for (int j = 0; j < nfeat; j++) {
            outd[i * nfeat + j] = 0;
        }
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

void evaluate_se_kernel_spin(double *out, double *outd, double *xin,
                             double *xctrl, double *actrl, double *exps, int n,
                             int nctrl, int nfeat, int nspin) {
    double *xin_a = xin;
    double *xin_b = xin + n * nfeat;
    double *xctrl_a = xctrl;
    double *xctrl_b = xctrl + nctrl * nfeat;
    double *outd_a = outd;
    double *outd_b = outd + n * nfeat;
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        out[i] = 0;
        for (int j = 0; j < nfeat; j++) {
            outd_a[i * nfeat + j] = 0;
            outd_b[i * nfeat + j] = 0;
        }
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
            _add_deriv(outd_b + iloc, xin_b + iloc, xctrl_a + cloc, exps, aabb,
                       nfeat);
            _add_deriv(outd_a + iloc, xin_a + iloc, xctrl_b + cloc, exps, abba,
                       nfeat);
            _add_deriv(outd_b + iloc, xin_b + iloc, xctrl_a + cloc, exps, abba,
                       nfeat);
        }
    }
}
