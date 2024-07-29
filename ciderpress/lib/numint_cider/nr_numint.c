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

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void VXC_feat_texp(double *Fvec, double *Uvec, double *Wvec, double *vva,
                   double *a, double *vvf, double *vvcoords, double *coords,
                   int vvngrids, int ngrids, double mul) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, tmp, F, U, W;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            F = 0;
            U = 0;
            W = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = vvcoords[j * 3 + 0] - coords[i * 3 + 0];
                DY = vvcoords[j * 3 + 1] - coords[i * 3 + 1];
                DZ = vvcoords[j * 3 + 2] - coords[i * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                tmp = (mul * a[i] + vva[j]) / (1 + mul);
                tmp = tmp * tmp * R2;

                // tmp = (mul*a[i]*a[i] + vva[j]*vva[j]) / (1+mul);
                // tmp = tmp * R2;

                tmp = exp(-tmp);
                // tmp = exp(-a[i] * R2);
                F += vvf[j] * tmp;
                U += tmp;
                W -= vvf[j] * R2 * tmp; // TODO check functional derivatives
            }
            Fvec[i] = F;
            Uvec[i] = U;
            Wvec[i] = W;
        }
    }
}

void VXC_feat_texp2(double *Fvec, double *Uvec, double *Wvec, double *vva,
                    double *a, double *vvf, double *vvcoords, double *coords,
                    int vvngrids, int ngrids, double mul, double *grad) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, lscale, tmp, tmp2, F1, F2, F3, FX, FY, FZ, U, W;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            F1 = 0;
            F2 = 0;
            F3 = 0;
            FX = 0;
            FY = 0;
            FZ = 0;
            U = 0;
            W = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = vvcoords[j * 3 + 0] - coords[i * 3 + 0];
                DY = vvcoords[j * 3 + 1] - coords[i * 3 + 1];
                DZ = vvcoords[j * 3 + 2] - coords[i * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                lscale = mul / (1 + mul) * a[i] * a[i];
                tmp = lscale * R2;
                tmp = exp(-tmp);
                lscale = 1 / (1 + mul) * vva[j] * vva[j];
                tmp2 = lscale * R2;
                tmp2 = exp(-tmp2);

                F1 += vvf[j] * tmp * tmp2;
                tmp = tmp * tmp;
                F2 += vvf[j] * tmp * tmp2;
                // F3 += vvf[j] * R2 * tmp * tmp2 * vva[j] * vva[j];
                // F3 += vvf[j] * R2 * tmp * tmp2;
                FX += vvf[j] * DX * tmp * tmp2;
                FY += vvf[j] * DY * tmp * tmp2;
                FZ += vvf[j] * DZ * tmp * tmp2;
                tmp = tmp * tmp;
                F3 += vvf[j] * tmp * tmp2;
            }
            Fvec[i * 5 + 0] = F1;
            Fvec[i * 5 + 1] = F2;
            Fvec[i * 5 + 2] = F3;
            // Fvec[i*5+2] = 2.0 / 3 * F3 * a[i] * a[i];
            // Fvec[i*5+2] = 2.0 / 3 * F3;
            Fvec[i * 5 + 3] = (FX * FX + FY * FY + FZ * FZ) * a[i] * a[i];
            Fvec[i * 5 + 4] = grad[i * 3 + 0] * FX + grad[i * 3 + 1] * FY +
                              grad[i * 3 + 2] * FZ;
            Uvec[i] = U;
            Wvec[i] = W;
        }
    }
}

void VXC_feat_vj(double *Fvec, double *Uvec, double *Wvec, double *vva,
                 double *a, double *vvf, double *vvcoords, double *coords,
                 int vvngrids, int ngrids, double mul, double *grad) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, lscale, tmp, tmp2, tmp3, tmp4, F1, F2, F3, F4,
            F5, U, W;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            F1 = 0;
            F2 = 0;
            F3 = 0;
            F4 = 0;
            F5 = 0;
            U = 0;
            W = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = vvcoords[j * 3 + 0] - coords[i * 3 + 0];
                DY = vvcoords[j * 3 + 1] - coords[i * 3 + 1];
                DZ = vvcoords[j * 3 + 2] - coords[i * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                lscale = mul / (1 + mul) * a[i] * a[i];
                tmp = lscale * R2;
                tmp3 = 2 * a[i] * sqrt(R2) + 1e-16;
                tmp3 = erf(tmp3) / tmp3;
                tmp4 = a[i] * sqrt(R2) + 1e-16;
                tmp4 = (erfc(tmp4) - erfc(2 * tmp4)) / tmp4;
                tmp = exp(-tmp);
                lscale = 1 / (1 + mul) * vva[j] * vva[j];
                tmp2 = lscale * R2;
                tmp2 = exp(-tmp2);

                F1 += vvf[j] * tmp * tmp2;
                F4 += vvf[j] * tmp * tmp2 * tmp3;
                tmp = tmp * tmp;
                F2 += vvf[j] * tmp * tmp2;
                F5 += vvf[j] * tmp * tmp2 * tmp3;
                // F5 += vvf[j] * tmp2 * tmp4;
                tmp = tmp * tmp;
                F3 += vvf[j] * tmp * tmp2;
            }
            Fvec[i * 5 + 0] = F1;
            Fvec[i * 5 + 1] = F2;
            Fvec[i * 5 + 2] = F3;
            Fvec[i * 5 + 3] = F4;
            Fvec[i * 5 + 4] = F5;
            Uvec[i] = U;
            Wvec[i] = W;
        }
    }
}

void VXC_feat_vk(double *Fvec, double *Uvec, double *Wvec, double *vva,
                 double *a, double *vvf, double *vvcoords, double *coords,
                 int vvngrids, int ngrids, double mul, double *grad) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, lscale, tmp, tmp2, tmp3, tmp4, F1, F2, F3, F4,
            F5, U, W;
        // double two15 = pow(2.0, 1.5);
        double two15 = pow(2.0, 0.5);
        double fac = sqrt(atan(1.0) * 4) / pow(2.0, 1.0 / 3);
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            F1 = 0;
            F2 = 0;
            F3 = 0;
            F4 = 0;
            F5 = 0;
            U = 0;
            W = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = vvcoords[j * 3 + 0] - coords[i * 3 + 0];
                DY = vvcoords[j * 3 + 1] - coords[i * 3 + 1];
                DZ = vvcoords[j * 3 + 2] - coords[i * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                lscale = 0.5 * a[i];
                tmp = lscale * R2;
                tmp2 = 2 * vva[j] / (a[i] + 1e-16);
                tmp3 = vvf[j] * exp(-1.5 * tmp2);
                tmp = exp(-tmp);

                F1 += tmp * tmp3;
                tmp2 /= 2;
                tmp3 = vvf[j] * exp(-1.5 * tmp2);
                tmp = tmp * tmp;
                F2 += tmp * tmp3;
                tmp2 /= 2;
                tmp3 = vvf[j] * exp(-1.5 * tmp2);
                tmp = tmp * tmp;
                F3 += tmp * tmp3;
                tmp2 /= 2;
                tmp3 = vvf[j] * exp(-1.5 * tmp2);
                tmp = tmp * tmp;
                F4 += tmp * tmp3;
                tmp2 /= 2;
                tmp3 = vvf[j] * exp(-1.5 * tmp2);
                tmp = tmp * tmp;
                F5 += tmp * tmp3;
            }
            Fvec[i * 5 + 0] = F1;
            Fvec[i * 5 + 1] = F2;
            Fvec[i * 5 + 2] = F3;
            Fvec[i * 5 + 3] = F4;
            Fvec[i * 5 + 4] = F5;
            Uvec[i] = U;
            Wvec[i] = W;
        }
    }
}

void VXC_feat_vg(double *Fvec, double *Uvec, double *Wvec, double *vva,
                 double *a, double *vvf, double *vvcoords, double *coords,
                 int vvngrids, int ngrids, double *grad) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, tmp, F1, F2, F3, FX, FY, FZ, U, W;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            F1 = 0;
            F2 = 0;
            F3 = 0;
            FX = 0;
            FY = 0;
            FZ = 0;
            U = 0;
            W = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = vvcoords[j * 3 + 0] - coords[i * 3 + 0];
                DY = vvcoords[j * 3 + 1] - coords[i * 3 + 1];
                DZ = vvcoords[j * 3 + 2] - coords[i * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                tmp = exp(-vva[j] * R2);

                tmp = vvf[j] * tmp;
                F1 += tmp;
                FX += tmp * DX;
                FY += tmp * DY;
                FZ += tmp * DZ;
                tmp *= R2;
                F2 += tmp;
                tmp *= vva[j];
                F3 += tmp;
            }
            Fvec[i * 5 + 0] = F1;
            Fvec[i * 5 + 1] = F2 * a[i];
            Fvec[i * 5 + 2] = F3;
            Fvec[i * 5 + 3] = (FX * FX + FY * FY + FZ * FZ) * a[i];
            Fvec[i * 5 + 4] = grad[i * 3 + 0] * FX + grad[i * 3 + 1] * FY +
                              grad[i * 3 + 2] * FZ;
            Uvec[i] = U;
            Wvec[i] = W;
        }
    }
}

void VXC_feat_vh(double *Fvec, double *Uvec, double *Wvec, double *vva,
                 double *vvt, double *vvf, double *vvcoords, double *coords,
                 int vvngrids, int ngrids, double *grad) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, tmp, ttmp, F1, F2, F1T, F2T, FX, FY, FZ, FXT,
            FYT, FZT, U, W;
        int nfeat = 7;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            F1 = 0;
            F2 = 0;
            F1T = 0;
            F2T = 0;
            FX = 0;
            FY = 0;
            FZ = 0;
            FXT = 0;
            FYT = 0;
            FZT = 0;
            U = 0;
            W = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = vvcoords[j * 3 + 0] - coords[i * 3 + 0];
                DY = vvcoords[j * 3 + 1] - coords[i * 3 + 1];
                DZ = vvcoords[j * 3 + 2] - coords[i * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                tmp = exp(-vva[j] * R2);

                ttmp = vvt[j] * tmp;
                tmp = vvf[j] * tmp;
                F1 += tmp;
                F1T += ttmp;
                FX += tmp * DX;
                FY += tmp * DY;
                FZ += tmp * DZ;
                FXT += ttmp * DX;
                FYT += ttmp * DY;
                FZT += ttmp * DZ;
                tmp *= R2;
                ttmp *= R2;
                F2 += tmp;
                F2T += ttmp;
            }
            Fvec[i * nfeat + 0] = F1;
            Fvec[i * nfeat + 1] = F2T;
            Fvec[i * nfeat + 2] = (FX * FX + FY * FY + FZ * FZ);
            Fvec[i * nfeat + 3] = (FX * FX + FY * FY + FZ * FZ) * F1T;
            Fvec[i * nfeat + 4] = (FXT * FXT + FYT * FYT + FZT * FZT);
            Fvec[i * nfeat + 5] = grad[i * 3 + 0] * FX + grad[i * 3 + 1] * FY +
                                  grad[i * 3 + 2] * FZ;
            Fvec[i * nfeat + 6] = FX * FXT + FY * FYT + FZ * FZT;
            Uvec[i] = U;
            Wvec[i] = W;
        }
    }
}

void VXC_feat_vg2(double *Fvec, double *Uvec, double *Wvec, double *vva,
                  double *vvt, double *vvf, double *vvcoords, double *coords,
                  int vvngrids, int ngrids, double *grad) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, tmp, ttmp, F1, F2, F1T, F2T, FX, FY, FZ, FXT,
            FYT, FZT, U, W;
        int nfeat = 7;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            F1 = 0;
            F2 = 0;
            F1T = 0;
            F2T = 0;
            FX = 0;
            FY = 0;
            FZ = 0;
            FXT = 0;
            FYT = 0;
            FZT = 0;
            U = 0;
            W = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = vvcoords[j * 3 + 0] - coords[i * 3 + 0];
                DY = vvcoords[j * 3 + 1] - coords[i * 3 + 1];
                DZ = vvcoords[j * 3 + 2] - coords[i * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                tmp = exp(-vva[j] * R2);

                ttmp = vvt[j] * tmp;
                tmp = vvf[j] * tmp;
                F1 += tmp;
                F1T += ttmp;
                FX += tmp * DX;
                FY += tmp * DY;
                FZ += tmp * DZ;
                FXT += ttmp * DX;
                FYT += ttmp * DY;
                FZT += ttmp * DZ;
                tmp *= R2;
                ttmp *= R2;
                F2 += tmp;
                F2T += ttmp;
            }
            Fvec[i * nfeat + 0] = F1;
            Fvec[i * nfeat + 1] = F2T;
            Fvec[i * nfeat + 2] = (FXT * FXT + FYT * FYT + FZT * FZT) / F1T;
            Fvec[i * nfeat + 3] = (FXT * FXT + FYT * FYT + FZT * FZT);
            Fvec[i * nfeat + 4] = grad[i * 3 + 0] * FXT +
                                  grad[i * 3 + 1] * FYT + grad[i * 3 + 2] * FZT;
            Uvec[i] = U;
            Wvec[i] = W;
        }
    }
}

void VXC_feat_vi(double *Fvec, double *Uvec, double *Wvec, double *vva,
                 double *vvf, double *vvcoords, double *coords, int vvngrids,
                 int ngrids, double *grad) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, tmp, ttmp, F1, F1T, F2T, FXT, FYT, FZT, U, W;
        int nfeat = 6;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            F1 = 0;
            F1T = 0;
            F2T = 0;
            FXT = 0;
            FYT = 0;
            FZT = 0;
            U = 0;
            W = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = vvcoords[j * 3 + 0] - coords[i * 3 + 0];
                DY = vvcoords[j * 3 + 1] - coords[i * 3 + 1];
                DZ = vvcoords[j * 3 + 2] - coords[i * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                tmp = exp(-vva[j] * R2);

                ttmp = vvf[j] * vva[j] * tmp;
                tmp = vvf[j] * tmp;
                F1 += tmp;
                F1T += ttmp;
                FXT += ttmp * DX;
                FYT += ttmp * DY;
                FZT += ttmp * DZ;
                ttmp *= R2;
                F2T += ttmp;
            }
            Fvec[i * nfeat + 0] = F1;
            Fvec[i * nfeat + 1] = F2T;
            Fvec[i * nfeat + 2] = F1T;
            Fvec[i * nfeat + 3] = (FXT * FXT + FYT * FYT + FZT * FZT) / F1T;
            Fvec[i * nfeat + 4] = (FXT * FXT + FYT * FYT + FZT * FZT);
            Fvec[i * nfeat + 5] = grad[i * 3 + 0] * FXT +
                                  grad[i * 3 + 1] * FYT + grad[i * 3 + 2] * FZT;
            Uvec[i] = U;
            Wvec[i] = W;
        }
    }
}

void VXC_feat_ve(double *Fvec, double *vva, double *a1, double *a2, double *vvf,
                 double *vvcoords, double *coords, int vvngrids, int ngrids) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, tmp, tmp2, tmp3, F1, F2, F3;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            F1 = 0;
            F2 = 0;
            F3 = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = vvcoords[j * 3 + 0] - coords[i * 3 + 0];
                DY = vvcoords[j * 3 + 1] - coords[i * 3 + 1];
                DZ = vvcoords[j * 3 + 2] - coords[i * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                tmp2 = exp(-a1[i] * R2);
                tmp3 = exp(-a2[i] * R2);
                tmp = exp(-vva[j] * R2);
                tmp = vvf[j] * tmp * tmp2 * tmp3;
                F1 += tmp;
                F2 += tmp * tmp2;
                F3 += tmp * tmp3;
            }
            Fvec[i * 3 + 0] = F1;
            Fvec[i * 3 + 1] = F2;
            Fvec[i * 3 + 2] = F3;
        }
    }
}

void VXC_dedrho_texp2(double *DFvec, double *DAvec, double *dedf, double *vva,
                      double *a, double *vvcoords, double *coords, int vvngrids,
                      int ngrids, double mul) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, lscale, tmp, tmp2, tmp3, DF, DA;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            DF = 0;
            DA = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = coords[i * 3 + 0] - vvcoords[j * 3 + 0];
                DY = coords[i * 3 + 1] - vvcoords[j * 3 + 1];
                DZ = coords[i * 3 + 2] - vvcoords[j * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                lscale = mul / (1 + mul) * vva[j] * vva[j];
                tmp = lscale * R2;
                tmp = exp(-tmp);
                lscale = 1 / (1 + mul) * a[i] * a[i];
                tmp2 = lscale * R2;
                tmp2 = exp(-tmp2);

                // F1 += vvf[j] * tmp * tmp2;
                tmp3 = dedf[6 * j + 0] * tmp * tmp2;
                DF += tmp3;
                DA += tmp3 * R2;

                tmp = tmp * tmp;
                // F2 += vvf[j] * tmp * tmp2;
                tmp3 = dedf[6 * j + 1] * tmp * tmp2;
                DF += tmp3;
                DA += tmp3 * R2;

                // FX += vvf[j] * DX * tmp * tmp2;
                tmp3 = dedf[6 * j + 3] * DX * tmp * tmp2;
                DF += tmp3;
                DA += tmp3 * R2;

                // FY += vvf[j] * DY * tmp * tmp2;
                tmp3 = dedf[6 * j + 4] * DY * tmp * tmp2;
                DF += tmp3;
                DA += tmp3 * R2;

                // FZ += vvf[j] * DZ * tmp * tmp2;
                tmp3 = dedf[6 * j + 5] * DZ * tmp * tmp2;
                DF += tmp3;
                DA += tmp3 * R2;

                tmp = tmp * tmp;
                // F3 += vvf[j] * tmp * tmp2;
                tmp3 = dedf[6 * j + 2] * tmp * tmp2;
                DF += tmp3;
                DA += tmp3 * R2;
            }
            DFvec[i] = DF;
            DAvec[i] = DA;
        }
    }
}

void VXC_deda1_texp2(double *Fvec, double *DFvec, double *vva, double *a,
                     double *vvf, double *vvcoords, double *coords,
                     int vvngrids, int ngrids, double mul) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, lscale, tmp, tmp2, tmp3, F1, F2, F3, FX, FY, FZ;
        double DF1, DF2, DF3, DFX, DFY, DFZ;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            F1 = 0;
            F2 = 0;
            F3 = 0;
            FX = 0;
            FY = 0;
            FZ = 0;
            DF1 = 0;
            DF2 = 0;
            DF3 = 0;
            DFX = 0;
            DFY = 0;
            DFZ = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = vvcoords[j * 3 + 0] - coords[i * 3 + 0];
                DY = vvcoords[j * 3 + 1] - coords[i * 3 + 1];
                DZ = vvcoords[j * 3 + 2] - coords[i * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                lscale = mul / (1 + mul) * a[i] * a[i];
                tmp = lscale * R2;
                tmp = exp(-tmp);
                lscale = 1 / (1 + mul) * vva[j] * vva[j];
                tmp2 = lscale * R2;
                tmp2 = exp(-tmp2);

                tmp3 = vvf[j] * tmp * tmp2;
                F1 += tmp3;
                DF1 += tmp3 * R2;
                tmp3 *= tmp;
                F2 += tmp3;
                DF2 += tmp3 * R2;
                FX += tmp3 * DX;
                FY += tmp3 * DY;
                FZ += tmp3 * DZ;
                DFX += tmp3 * DX * R2;
                DFY += tmp3 * DY * R2;
                DFZ += tmp3 * DZ * R2;
                tmp = tmp * tmp;
                tmp3 *= tmp;
                F3 += tmp3;
                DF3 += tmp3 * R2;
            }
            Fvec[i * 6 + 0] = F1;
            Fvec[i * 6 + 1] = F2;
            Fvec[i * 6 + 2] = F3;
            Fvec[i * 6 + 3] = FX;
            Fvec[i * 6 + 4] = FY;
            Fvec[i * 6 + 5] = FZ;
            // Fvec[i*6+3] = (FX*FX + FY*FY + FZ*FZ) * a[i]*a[i];
            // Fvec[i*6+4] = grad[i*3+0]*FX + grad[i*3+1]*FY + grad[i*3+2]*FZ;
            DFvec[i * 6 + 0] = DF1;
            DFvec[i * 6 + 1] = DF2;
            DFvec[i * 6 + 2] = DF3;
            DFvec[i * 6 + 3] = DFX;
            DFvec[i * 6 + 4] = DFY;
            DFvec[i * 6 + 5] = DFZ;
        }
    }
}

void VXC_deriv_l0(double *DFvec, double *DAvec, double *dedf, double *vva,
                  double *a, double *vvcoords, double *coords, int vvngrids,
                  int ngrids, double mul) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, lscale, tmp, tmp2, tmp3, DF, DA;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            DF = 0;
            DA = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = coords[i * 3 + 0] - vvcoords[j * 3 + 0];
                DY = coords[i * 3 + 1] - vvcoords[j * 3 + 1];
                DZ = coords[i * 3 + 2] - vvcoords[j * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                lscale = mul / (1 + mul) * vva[j] * vva[j];
                tmp = lscale * R2;
                tmp = exp(-tmp);
                lscale = 1 / (1 + mul) * a[i] * a[i];
                tmp2 = lscale * R2;
                tmp2 = exp(-tmp2);

                // F1 += vvf[j] * tmp * tmp2;
                tmp3 = dedf[3 * j + 0] * tmp * tmp2;
                DF += tmp3;
                DA += tmp3 * R2;

                tmp = tmp * tmp;
                // F2 += vvf[j] * tmp * tmp2;
                tmp3 = dedf[3 * j + 1] * tmp * tmp2;
                DF += tmp3;
                DA += tmp3 * R2;

                tmp = tmp * tmp;
                // F3 += vvf[j] * tmp * tmp2;
                tmp3 = dedf[3 * j + 2] * tmp * tmp2;
                DF += tmp3;
                DA += tmp3 * R2;
            }
            DFvec[i] = DF;
            DAvec[i] = DA;
        }
    }
}

void VXC_feat_l0(double *Fvec, double *DFvec, double *vva, double *a,
                 double *vvf, double *vvcoords, double *coords, int vvngrids,
                 int ngrids, double mul) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, lscale, tmp, tmp2, tmp3, F1, F2, F3;
        double DF1, DF2, DF3;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            F1 = 0;
            F2 = 0;
            F3 = 0;
            DF1 = 0;
            DF2 = 0;
            DF3 = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = vvcoords[j * 3 + 0] - coords[i * 3 + 0];
                DY = vvcoords[j * 3 + 1] - coords[i * 3 + 1];
                DZ = vvcoords[j * 3 + 2] - coords[i * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                lscale = mul / (1 + mul) * a[i] * a[i];
                tmp = lscale * R2;
                tmp = exp(-tmp);
                lscale = 1 / (1 + mul) * vva[j] * vva[j];
                tmp2 = lscale * R2;
                tmp2 = exp(-tmp2);

                tmp3 = vvf[j] * tmp * tmp2;
                F1 += tmp3;
                DF1 += tmp3 * R2;
                tmp3 *= tmp;
                F2 += tmp3;
                DF2 += tmp3 * R2;
                tmp = tmp * tmp;
                tmp3 *= tmp;
                F3 += tmp3;
                DF3 += tmp3 * R2;
            }
            Fvec[i * 3 + 0] = F1;
            Fvec[i * 3 + 1] = F2;
            Fvec[i * 3 + 2] = F3;
            DFvec[i * 3 + 0] = DF1;
            DFvec[i * 3 + 1] = DF2;
            DFvec[i * 3 + 2] = DF3;
        }
    }
}
