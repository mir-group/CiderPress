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

#include "nldf_fft_core.h"
#include "config.h"
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if HAVE_MPI
#include "nldf_fft_mpi.h"
#else
#include "nldf_fft_serial.h"
#endif

#if HAVE_MPI
#define CIDERPW_G2K ciderpw_g2k_mpi
#define CIDERPW_K2G ciderpw_k2g_mpi
#else
#define CIDERPW_G2K ciderpw_g2k_serial
#define CIDERPW_K2G ciderpw_k2g_serial
#endif
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

int ciderpw_has_mpi() { return HAVE_MPI; }

void ciderpw_allocate_buffers(ciderpw_data data) {
    data->Ng =
        data->cell.Nlocal[0] * data->cell.Nlocal[1] * data->cell.Nlocal[2];
}

int ciderpw_get_struct_size() { return sizeof(struct ciderpw_data_obj); }

void ciderpw_nullify(ciderpw_data data) {
    memset(data, 0, ciderpw_get_struct_size());
}

void ciderpw_finalize(ciderpw_data *cider) {
    ciderpw_data data = cider[0];
#if HAVE_MPI
    if (data->plan != NULL) {
        free_mpi_fft3d_plan(data->plan);
    }
#else
    if (data->plan_g2k != NULL) {
        free_fft_plan(data->plan_g2k);
    }
    if (data->plan_k2g != NULL) {
        free_fft_plan(data->plan_k2g);
    }
    if (data->work_ska != NULL) {
        free_fft_array(data->work_ska);
    }
#endif
    free(data->kernel.expnts_ab);
    free(data->kernel.expnts_ba);
    free(data->kernel.norms_ab);
    free(data->kernel.norms_ba);
    free(data->k2_G);
    free(data->kx_G);
    free(data->ky_G);
    free(data->kz_G);
    free(data->wt_G);
    ciderpw_nullify(data);
    free(data);
    cider[0] = NULL;
}

void ciderpw_set_unit_cell(ciderpw_data data, int *N_c, double *cell_cv) {
    double C00 = cell_cv[0];
    double C01 = cell_cv[3];
    double C02 = cell_cv[6];
    double C10 = cell_cv[1];
    double C11 = cell_cv[4];
    double C12 = cell_cv[7];
    double C20 = cell_cv[2];
    double C21 = cell_cv[5];
    double C22 = cell_cv[8];
    double det =
        (C00 * (C11 * C22 - C21 * C12) - C01 * (C10 * C22 - C12 * C20) +
         C02 * (C10 * C21 - C11 * C20));
    data->cell.dV = det / (N_c[0] * N_c[1] * N_c[2]);
    data->cell.vec[0] = C00;
    data->cell.vec[1] = C10;
    data->cell.vec[2] = C20;
    data->cell.vec[3] = C01;
    data->cell.vec[4] = C11;
    data->cell.vec[5] = C21;
    data->cell.vec[6] = C02;
    data->cell.vec[7] = C12;
    data->cell.vec[8] = C22;

    data->cell.Nglobal[0] = N_c[0];
    data->cell.Nglobal[1] = N_c[1];
    data->cell.Nglobal[2] = N_c[2];

    // Calculate reciprocal cell
    double idet =
        1.0 /
        (C00 * (C11 * C22 - C21 * C12) - C01 * (C10 * C22 - C12 * C20) +
         C02 * (C10 * C21 - C11 * C20)) *
        2 * CIDERPW_PI;
    data->icell.Nglobal[0] = N_c[0];
    data->icell.Nglobal[1] = N_c[1];
    data->icell.Nglobal[2] = N_c[2] / 2 + 1;
    data->icell.dV = 1e100;
    data->icell.vec[0] = (C11 * C22 - C21 * C12) * idet;
    data->icell.vec[1] = -(C01 * C22 - C02 * C21) * idet;
    data->icell.vec[2] = (C01 * C12 - C02 * C11) * idet;
    data->icell.vec[3] = -(C10 * C22 - C12 * C20) * idet;
    data->icell.vec[4] = (C00 * C22 - C02 * C20) * idet;
    data->icell.vec[5] = -(C00 * C12 - C10 * C02) * idet;
    data->icell.vec[6] = (C10 * C21 - C20 * C11) * idet;
    data->icell.vec[7] = -(C00 * C21 - C20 * C01) * idet;
    data->icell.vec[8] = (C00 * C11 - C10 * C01) * idet;

    data->Nglobal =
        data->cell.Nglobal[0] * data->cell.Nglobal[1] * data->cell.Nglobal[2];
}

ciderpw_data ciderpw_new_anyspin(int nspin) {
    ciderpw_data data = malloc(ciderpw_get_struct_size());
    ciderpw_nullify(data);
    return data;
}

void ciderpw_create(ciderpw_data *cider, double scale, int *N_c,
                    double *cell_cv) {
    ciderpw_data data = ciderpw_new_anyspin(1);
    ciderpw_set_unit_cell(data, N_c, cell_cv);
    cider[0] = data;
}

void ciderpw_setup_kernel(ciderpw_data data, int nalpha, int nbeta,
                          double *norms_ab, double *expnts_ab) {
    data->kernel.nalpha = nalpha;
    data->kernel.nbeta = nbeta;
    data->kernel.work_size = MAX(nalpha, nbeta);
    int n_ab = nalpha * nbeta;
    data->kernel.expnts_ab = (double *)malloc(sizeof(double) * n_ab);
    data->kernel.expnts_ba = (double *)malloc(sizeof(double) * n_ab);
    data->kernel.norms_ab = (double *)malloc(sizeof(double) * n_ab);
    data->kernel.norms_ba = (double *)malloc(sizeof(double) * n_ab);
    data->kernel.num_l1_feats = 0;
    int alpha, beta;
    int ab = 0;
    int ba, a, b;
    double fac = 1.0 / data->Nglobal;
    for (a = 0; a < nalpha; a++) {
        for (b = 0; b < nbeta; b++, ab++) {
            ba = b * nalpha + a;
            data->kernel.expnts_ab[ab] = -1 * expnts_ab[ab];
            data->kernel.expnts_ba[ba] = -1 * expnts_ab[ab];
            data->kernel.norms_ab[ab] = fac * norms_ab[ab];
            data->kernel.norms_ba[ba] = fac * norms_ab[ab];
        }
    }
}

void ciderpw_get_bound_inds(ciderpw_data data, int *bound_inds) {
    bound_inds[0] = data->cell.offset[0];
    bound_inds[1] = data->cell.offset[1];
    bound_inds[2] = data->cell.offset[2];
    bound_inds[3] = bound_inds[0] + data->cell.Nlocal[0];
    bound_inds[4] = bound_inds[1] + data->cell.Nlocal[1];
    bound_inds[5] = bound_inds[2] + data->cell.Nlocal[2];
}

size_t ciderpw_get_recip_size(ciderpw_data data) {
    return data->icell.Nlocal[0] * data->icell.Nlocal[1] *
           data->icell.Nlocal[2];
}

size_t ciderpw_get_real_size(ciderpw_data data) {
    return data->cell.Nlocal[0] * data->cell.Nlocal[1] * data->gLDA;
}

size_t ciderpw_get_work_size(ciderpw_data data) {
    return data->work_array_size;
}

double *ciderpw_get_work_pointer(ciderpw_data data) {
    return (double *)data->work_ska;
}

void ciderpw_get_local_size_and_lda(ciderpw_data data, int *sizes) {
    sizes[0] = data->cell.Nlocal[0];
    sizes[1] = data->cell.Nlocal[1];
    sizes[2] = data->cell.Nlocal[2];
    sizes[3] = data->gLDA;
    sizes[4] = data->icell.Nlocal[0];
    sizes[5] = data->icell.Nlocal[1];
    sizes[6] = data->icell.Nlocal[2];
    sizes[7] = data->kLDA;
}

void ciderpw_compute_kernels_helper(double k2, double *kernel_ab,
                                    double *norms_ab, double *expnts_ab,
                                    int n_ab) {
    for (int ab = 0; ab < n_ab; ab++) {
        kernel_ab[ab] = norms_ab[ab] * exp(expnts_ab[ab] * k2);
    }
}

void ciderpw_compute_kernels_sym_helper(double k2, double *kernel_ab,
                                        double *norms_ab, double *expnts_ab,
                                        int nalpha) {
    int ab, a, b;
    double kval;
    for (b = 0; b < nalpha; b++) {
        ab = b * nalpha + b;
        kernel_ab[ab] = norms_ab[ab] * exp(expnts_ab[ab] * k2);
        for (a = b + 1; a < nalpha; a++) {
            ab = b * nalpha + a;
            kernel_ab[ab] = norms_ab[ab] * exp(expnts_ab[ab] * k2);
            kernel_ab[a * nalpha + b] = kernel_ab[ab];
        }
    }
}

void ciderpw_compute_kernels_sym_stress_helper(double k2, double *kernel_ab,
                                               double *norms_ab,
                                               double *expnts_ab, int nalpha) {
    int ab, a, b;
    double kval;
    for (b = 0; b < nalpha; b++) {
        ab = b * nalpha + b;
        kernel_ab[ab] =
            2 * norms_ab[ab] * expnts_ab[ab] * exp(expnts_ab[ab] * k2);
        for (a = b + 1; a < nalpha; a++) {
            ab = b * nalpha + a;
            kernel_ab[ab] =
                2 * norms_ab[ab] * expnts_ab[ab] * exp(expnts_ab[ab] * k2);
            kernel_ab[a * nalpha + b] = kernel_ab[ab];
        }
    }
}

void ciderpw_compute_kernels(struct ciderpw_kernel *kernel, double k2,
                             double *kernel_ba) {
    ciderpw_compute_kernels_helper(k2, kernel_ba, kernel->norms_ba,
                                   kernel->expnts_ba,
                                   kernel->nbeta * kernel->nalpha);
}

void ciderpw_compute_kernels_sym(struct ciderpw_kernel *kernel, double k2,
                                 double *kernel_ba) {
    ciderpw_compute_kernels_sym_helper(k2, kernel_ba, kernel->norms_ba,
                                       kernel->expnts_ba, kernel->nalpha);
}

void ciderpw_compute_kernels_sym_stress(struct ciderpw_kernel *kernel,
                                        double k2, double *kernel_ba) {
    ciderpw_compute_kernels_sym_stress_helper(
        k2, kernel_ba, kernel->norms_ba, kernel->expnts_ba, kernel->nalpha);
}

void ciderpw_compute_kernels_t(struct ciderpw_kernel *kernel, double k2,
                               double *kernel_ab) {
    ciderpw_compute_kernels_helper(k2, kernel_ab, kernel->norms_ab,
                                   kernel->expnts_ab,
                                   kernel->nbeta * kernel->nalpha);
}

void ciderpw_multiply_l1(ciderpw_data data) {
    int n1 = data->kernel.num_l1_feats;
    double complex *work_ska = (double complex *)data->work_ska;
    double complex *work_a;
    int kindex, b;
    if (n1 > 0) {
        for (kindex = 0; kindex < data->nk; kindex++) {
            work_a = work_ska + kindex * data->kernel.work_size;
            for (b = data->kernel.nbeta - 3 * n1; b < data->kernel.nbeta;
                 b += 3) {
                work_a[b + 0] *= I * data->kx_G[kindex];
                work_a[b + 1] *= I * data->ky_G[kindex];
                work_a[b + 2] *= I * data->kz_G[kindex];
            }
        }
    }
}

void ciderpw_convolution_fwd(ciderpw_data data) {
    double *kernel_ba = (double *)malloc(sizeof(double) * data->kernel.nbeta *
                                         data->kernel.nalpha);
    double complex *work_ska = (double complex *)data->work_ska;
    int a, b;
    double complex F_b[data->kernel.nbeta];
    int kindex;
    double complex F;
    double *ikernel_ba;
    double complex *work_a;
    for (kindex = 0; kindex < data->nk; kindex++) {
        work_a = work_ska + kindex * data->kernel.work_size;
        // ciderpw_compute_kernels(&data->kernel, data->k2_G[kindex],
        //                         kernel_ba);
        ciderpw_compute_kernels_sym(&data->kernel, data->k2_G[kindex],
                                    kernel_ba);
        ikernel_ba = kernel_ba;
        for (b = 0; b < data->kernel.nbeta; b++) {
            F = 0.0;
            for (a = 0; a < data->kernel.nalpha; a++) {
                F += work_a[a] * (*ikernel_ba++);
            }
            F_b[b] = F;
        }
        for (b = 0; b < data->kernel.nbeta; b++) {
            work_a[b] = F_b[b];
        }
    }
    ciderpw_multiply_l1(data);
    free(kernel_ba);
}

void ciderpw_convolution_stress(ciderpw_data data, double *stress_vv,
                                double complex *theta_gq) {
    double *kernel_ba = (double *)malloc(sizeof(double) * data->kernel.nbeta *
                                         data->kernel.nalpha);
    double complex *work_ska = (double complex *)data->work_ska;
    int a, b;
    int kindex;
    double complex F;
    double F2;
    double *ikernel_ba;
    double complex *work_b;
    double complex *theta_a;
    for (int vv = 0; vv < 9; vv++) {
        stress_vv[vv] = 0;
    }
    for (kindex = 0; kindex < data->nk; kindex++) {
        theta_a = theta_gq + kindex * data->kernel.work_size;
        work_b = work_ska + kindex * data->kernel.work_size;
        ciderpw_compute_kernels_sym_stress(&data->kernel, data->k2_G[kindex],
                                           kernel_ba);
        ikernel_ba = kernel_ba;
        F2 = 0;
        for (b = 0; b < data->kernel.nbeta; b++) {
            F = 0.0;
            for (a = 0; a < data->kernel.nalpha; a++) {
                F += theta_a[a] * (*ikernel_ba++);
            }
            F2 += creal(work_b[b]) * creal(F) + cimag(work_b[b]) * cimag(F);
        }
        F2 *= ((double)data->wt_G[kindex]);
        stress_vv[0] += F2 * data->kx_G[kindex] * data->kx_G[kindex];
        stress_vv[1] += F2 * data->kx_G[kindex] * data->ky_G[kindex];
        stress_vv[2] += F2 * data->kx_G[kindex] * data->kz_G[kindex];
        stress_vv[4] += F2 * data->ky_G[kindex] * data->ky_G[kindex];
        stress_vv[5] += F2 * data->ky_G[kindex] * data->kz_G[kindex];
        stress_vv[8] += F2 * data->kz_G[kindex] * data->kz_G[kindex];
    }
    stress_vv[3] = stress_vv[1];
    stress_vv[6] = stress_vv[2];
    stress_vv[7] = stress_vv[5];
    for (int vv = 0; vv < 9; vv++) {
        // TODO not sure why the -1 factor is needed and why 1 / data->Nglobal
        // is not needed here, but this works using finite difference tests.
        // Would be good to figure out why this is.
        // stress_vv[vv] *= data->cell.dV / data->Nglobal;
        stress_vv[vv] *= -1 * data->cell.dV;
    }
    // TODO need to implement this
    // ciderpw_multiply_l1(data);
    free(kernel_ba);
}

void ciderpw_convolution_bwd(ciderpw_data data) {
    double *kernel_ab = (double *)malloc(sizeof(double) * data->kernel.nbeta *
                                         data->kernel.nalpha);
    double complex *work_ska = (double complex *)data->work_ska;
    int a, b;
    double complex F_a[data->kernel.nalpha];
    int kindex;
    double complex F;
    double *ikernel_ab;
    double complex *work_a;
    ciderpw_multiply_l1(data);
    for (kindex = 0; kindex < data->nk; kindex++) {
        work_a = work_ska + kindex * data->kernel.work_size;
        // ciderpw_compute_kernels_t(&data->kernel, data->k2_G[kindex],
        // kernel_ab);
        ciderpw_compute_kernels_sym(&data->kernel, data->k2_G[kindex],
                                    kernel_ab);
        ikernel_ab = kernel_ab;
        for (a = 0; a < data->kernel.nalpha; a++) {
            F = 0.0;
            for (b = 0; b < data->kernel.nbeta; b++) {
                F += work_a[b] * (*ikernel_ab++);
            }
            F_a[a] = F;
        }
        for (a = 0; a < data->kernel.nalpha; a++) {
            work_a[a] = F_a[a];
        }
    }
    free(kernel_ab);
}

void ciderpw_g2k_v2(ciderpw_data data, double complex *arr) {
    for (size_t index = 0; index < data->work_array_size; index++) {
        data->work_ska[index] = arr[index];
    }
    CIDERPW_G2K(data);
    for (size_t index = 0; index < data->work_array_size; index++) {
        arr[index] = data->work_ska[index];
    }
}

void ciderpw_compute_features(ciderpw_data data) {
    CIDERPW_G2K(data);
    ciderpw_convolution_fwd(data);
    CIDERPW_K2G(data);
}

void ciderpw_compute_potential(ciderpw_data data) {
    CIDERPW_G2K(data);
    ciderpw_convolution_bwd(data);
    CIDERPW_K2G(data);
}

void ciderpw_convolution_potential_and_stress(ciderpw_data data,
                                              double *stress_vv,
                                              double complex *theta_gq) {
    CIDERPW_G2K(data);
    ciderpw_convolution_stress(data, stress_vv, theta_gq);
    ciderpw_convolution_bwd(data);
    CIDERPW_K2G(data);
}

void ciderpw_eval_feature_vj(ciderpw_data data, double *feat_g, double *p_gq) {
    int N0, N1, N2;
    double *work_ga = (double *)data->work_ska;
    int ind, a;
    for (N0 = 0; N0 < data->cell.Nlocal[0]; N0++) {
        for (N1 = 0; N1 < data->cell.Nlocal[1]; N1++) {
            for (N2 = 0; N2 < data->cell.Nlocal[2]; N2++) {
                feat_g[N2] = 0;
                for (a = 0; a < data->kernel.nalpha; a++) {
                    ind = N2 * data->kernel.nalpha + a;
                    feat_g[N2] += work_ga[ind] * p_gq[ind];
                }
            }
            work_ga += data->gLDA * data->kernel.work_size;
            p_gq += data->cell.Nlocal[2] * data->kernel.nalpha;
            feat_g += data->cell.Nlocal[2];
        }
    }
}

void ciderpw_add_potential_vj(ciderpw_data data, double *vfeat_g,
                              double *p_gq) {
    int N0, N1, N2;
    double *work_ga = (double *)data->work_ska;
    int ind, a;
    for (N0 = 0; N0 < data->cell.Nlocal[0]; N0++) {
        for (N1 = 0; N1 < data->cell.Nlocal[1]; N1++) {
            for (N2 = 0; N2 < data->cell.Nlocal[2]; N2++) {
                for (a = 0; a < data->kernel.nalpha; a++) {
                    ind = N2 * data->kernel.nalpha + a;
                    work_ga[ind] += vfeat_g[N2] * p_gq[ind];
                }
            }
            work_ga += data->gLDA * data->kernel.work_size;
            p_gq += data->cell.Nlocal[2] * data->kernel.nalpha;
            vfeat_g += data->cell.Nlocal[2];
        }
    }
}

void ciderpw_set_work(ciderpw_data data, double *fun_g, double *p_gq) {
    int N0, N1, N2;
    double *work_ga = (double *)data->work_ska;
    int ind, a;
    for (N0 = 0; N0 < data->cell.Nlocal[0]; N0++) {
        for (N1 = 0; N1 < data->cell.Nlocal[1]; N1++) {
            for (N2 = 0; N2 < data->cell.Nlocal[2]; N2++) {
                for (a = 0; a < data->kernel.nalpha; a++) {
                    ind = N2 * data->kernel.nalpha + a;
                    work_ga[ind] = fun_g[N2] * p_gq[ind];
                }
            }
            work_ga += data->gLDA * data->kernel.work_size;
            p_gq += data->cell.Nlocal[2] * data->kernel.nalpha;
            fun_g += data->cell.Nlocal[2];
        }
    }
}

void ciderpw_zero_work(ciderpw_data data) {
    double *work_ga = (double *)data->work_ska;
    int nalpha = data->kernel.work_size;
    size_t size = nalpha * MAX(data->nk * 2, ciderpw_get_real_size(data));
    for (size_t ind = 0; ind < size; ind++) {
        work_ga[ind] = 0.0;
    }
}

void ciderpw_fill_atom_info(ciderpw_data data, int64_t *inds_c, double *disps_c,
                            int64_t *num_c, double *r_vg, int64_t *locs_g) {
    int64_t num_x = num_c[0];
    int64_t num_y = num_c[1];
    int64_t num_z = num_c[2];
    int64_t num_t = num_x * num_y * num_z;
    int64_t *indx = inds_c;
    int64_t *indy = indx + num_x;
    int64_t *indz = indy + num_y;
    double *dispx = disps_c;
    double *dispy = dispx + num_x;
    double *dispz = dispy + num_y;
    int ix, iy, iz;
    int g = 0;
    int nlocy = data->cell.Nlocal[1];
    for (ix = 0; ix < num_x; ix++) {
        for (iy = 0; iy < num_y; iy++) {
            for (iz = 0; iz < num_z; iz++, g++) {
                locs_g[g] =
                    indz[iz] + data->gLDA * (indy[iy] + nlocy * indx[ix]);
                double rx = data->cell.vec[0] * dispx[ix] +
                            data->cell.vec[3] * dispy[iy] +
                            data->cell.vec[6] * dispz[iz];
                double ry = data->cell.vec[1] * dispx[ix] +
                            data->cell.vec[4] * dispy[iy] +
                            data->cell.vec[7] * dispz[iz];
                double rz = data->cell.vec[2] * dispx[ix] +
                            data->cell.vec[5] * dispy[iy] +
                            data->cell.vec[8] * dispz[iz];
                r_vg[g] = sqrt(rx * rx + ry * ry + rz * rz + 1e-16);
                r_vg[1 * num_t + g] = rx / r_vg[g];
                r_vg[2 * num_t + g] = ry / r_vg[g];
                r_vg[3 * num_t + g] = rz / r_vg[g];
            }
        }
    }
}

void ciderpw_add_atom_info(ciderpw_data data, double *funcs_ga, int64_t *locs_g,
                           int ng) {
    double *work_ga = (double *)data->work_ska;
    double *work_a;
    double *funcs_a;
    int64_t loc;
    int a;
    const int nalpha = data->kernel.nalpha;
    for (int g = 0; g < ng; g++) {
        loc = locs_g[g];
        work_a = work_ga + loc * nalpha;
        funcs_a = funcs_ga + g * nalpha;
        for (a = 0; a < nalpha; a++) {
            work_a[a] += funcs_a[a];
        }
    }
}

void ciderpw_set_atom_info(ciderpw_data data, double *funcs_ga, int64_t *locs_g,
                           int ng) {
    double *work_ga = (double *)data->work_ska;
    double *work_a;
    double *funcs_a;
    int64_t loc;
    int a;
    const int nalpha = data->kernel.nalpha;
    for (int g = 0; g < ng; g++) {
        loc = locs_g[g];
        work_a = work_ga + loc * nalpha;
        funcs_a = funcs_ga + g * nalpha;
        for (a = 0; a < nalpha; a++) {
            funcs_a[a] = work_a[a];
        }
    }
}

void ciderpw_copy_work_array_real(ciderpw_data data, double *out) {
    int nalpha = data->kernel.work_size;
    double *work = (double *)data->work_ska;
    size_t ng = ciderpw_get_real_size(data);
    for (size_t index = 0; index < ng * nalpha; index++) {
        out[index] = work[index];
    }
}

void ciderpw_copy_work_array_recip(ciderpw_data data, double complex *out) {
    int nalpha = data->kernel.work_size;
    for (size_t index = 0; index < data->nk * nalpha; index++) {
        out[index] = data->work_ska[index];
    }
}

void ciderpw_copy_work_array(ciderpw_data data, double complex *out) {
    for (size_t index = 0; index < data->work_array_size; index++) {
        out[index] = data->work_ska[index];
    }
}
