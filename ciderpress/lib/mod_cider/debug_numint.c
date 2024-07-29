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

void debug_numint_vi(double *Fvec, double *vva, double *vvf, double *vvcoords,
                     double *coords, int vvngrids, int ngrids) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, tmp;
        double F0, F1, F2, F3, F4, F5, F6X, F6Y, F6Z, F7X, F7Y, F7Z;
        int nfeat = 12;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            F0 = 0;
            F1 = 0;
            F2 = 0;
            F3 = 0;
            F4 = 0;
            F5 = 0;
            F6X = 0;
            F6Y = 0;
            F6Z = 0;
            F7X = 0;
            F7Y = 0;
            F7Z = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = vvcoords[j * 3 + 0] - coords[i * 3 + 0];
                DY = vvcoords[j * 3 + 1] - coords[i * 3 + 1];
                DZ = vvcoords[j * 3 + 2] - coords[i * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                tmp = exp(-vva[j] * R2) * vvf[j];

                F0 += tmp;
                F1 += R2 * tmp;
                F2 += vva[j] * R2 * tmp;
                F3 += vva[j] * tmp;
                F4 += vva[j] * vva[j] * R2 * tmp;
                F5 += (4 * vva[j] * R2 - 2) * vva[j] * tmp;
                F6X += DX * tmp;
                F6Y += DY * tmp;
                F6Z += DZ * tmp;
                F7X += DX * vva[j] * tmp;
                F7Y += DY * vva[j] * tmp;
                F7Z += DZ * vva[j] * tmp;
            }
            Fvec[i * nfeat + 0] = F0;
            Fvec[i * nfeat + 1] = F1;
            Fvec[i * nfeat + 2] = F2;
            Fvec[i * nfeat + 3] = F3;
            Fvec[i * nfeat + 4] = F4;
            Fvec[i * nfeat + 5] = F5;
            Fvec[i * nfeat + 6] = F6X;
            Fvec[i * nfeat + 7] = F6Y;
            Fvec[i * nfeat + 8] = F6Z;
            Fvec[i * nfeat + 9] = F7X;
            Fvec[i * nfeat + 10] = F7Y;
            Fvec[i * nfeat + 11] = F7Z;
        }
    }
}

void debug_numint_vj(double *Fvec, double *vva, double *a, double *vvf,
                     double *vvcoords, double *coords, int vvngrids, int ngrids,
                     double ratio) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, tmp, ttmp, F0, F1, F2, F3;
        int nfeat = 4;
        double rtpih = sqrt(4.0 * atan(1.0)) / 2.0;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            F0 = 0;
            F1 = 0;
            F2 = 0;
            F3 = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = vvcoords[j * 3 + 0] - coords[i * 3 + 0];
                DY = vvcoords[j * 3 + 1] - coords[i * 3 + 1];
                DZ = vvcoords[j * 3 + 2] - coords[i * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                tmp = exp(-(vva[j] + a[i]) * R2) * vvf[j];
                ttmp = sqrt(ratio * a[i] * R2) + 1e-16;
                ttmp = erf(ttmp) / ttmp * rtpih;

                F0 += tmp;
                F1 += R2 * tmp;
                F2 += R2 * R2 * tmp;
                F3 += ttmp * tmp;
            }
            Fvec[i * nfeat + 0] = F0;
            Fvec[i * nfeat + 1] = a[i] * F1;
            Fvec[i * nfeat + 2] = a[i] * a[i] * F2;
            Fvec[i * nfeat + 3] = F3;
        }
    }
}

void debug_numint_vk(double *Fvec, double *vva, double *a, double *vvf,
                     double *vvcoords, double *coords, int vvngrids, int ngrids,
                     double ratio) {
#pragma omp parallel
    {
        int i, j;
        double DX, DY, DZ, R2, tmp, tmp2, F0, F1, F2, F3;
        int nfeat = 4;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            F0 = 0;
            F1 = 0;
            F2 = 0;
            F3 = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = vvcoords[j * 3 + 0] - coords[i * 3 + 0];
                DY = vvcoords[j * 3 + 1] - coords[i * 3 + 1];
                DZ = vvcoords[j * 3 + 2] - coords[i * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;

                tmp = exp(-1.5 * vva[j] / (0.5 * a[i] + 1e-16)) * vvf[j];
                tmp2 = exp(-0.5 * a[i] * R2);
                F0 += tmp2 * tmp;
                tmp = exp(-1.5 * vva[j] / (1.0 * a[i] + 1e-16)) * vvf[j];
                tmp2 *= tmp2;
                F1 += tmp2 * tmp;
                tmp = exp(-1.5 * vva[j] / (2.0 * a[i] + 1e-16)) * vvf[j];
                tmp2 *= tmp2;
                F2 += tmp2 * tmp;
                tmp = exp(-1.5 * vva[j] / (4.0 * a[i] + 1e-16)) * vvf[j];
                tmp2 *= tmp2;
                F3 += tmp2 * tmp;
            }
            Fvec[i * nfeat + 0] = F0;
            Fvec[i * nfeat + 1] = F1;
            Fvec[i * nfeat + 2] = F2;
            Fvec[i * nfeat + 3] = F3;
        }
    }
}
