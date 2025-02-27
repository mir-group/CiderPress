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

#ifndef NLDF_FFT_CORE_H
#define NLDF_FFT_CORE_H
#include "config.h"
#include <complex.h>
#include <stdint.h>
#if HAVE_MPI
#include "cider_mpi_fft.h"
#include <mpi.h>
#endif
#include "cider_fft.h"

#define CIDERPW_PI 3.14159265358979323846
#define CIDERPW_R2C 0
#define CIDERPW_C2C 1

struct ciderpw_unit_cell {
    double vec[9];
    int Nglobal[3];
    int Nlocal[3];
    int offset[3];
    double dV;
};

struct ciderpw_kernel {
    int kernel_type;
    int nalpha;
    int nbeta;
    int work_size;
    double *expnts_ba;
    double *norms_ba;
    double *expnts_ab;
    double *norms_ab;
    int num_l1_feats;
};

struct ciderpw_data_obj {
    int initialized;
    int nspin;
    struct ciderpw_unit_cell cell;
    struct ciderpw_unit_cell icell;
    struct ciderpw_kernel kernel;

    int fft_type;
#if HAVE_MPI
    MPI_Comm mpi_comm;
#endif
    int mpi_rank;
    int mpi_size;

    int Ng;
    int Nglobal;
    int kLDA; // Shortest dimension of k i.e. Nglobal[2] for c2c
              // and Nglobal[2] / 2 + 1 for r2c
    int gLDA; // 2 * kLDA

    // work_ska contains theta and F:
    // spin first, then k/g, then alpha
    double complex *work_ska;

    // for NLDF, this will always be r2c and c2r, but might want
    // to reuse this struct for SDMX or R3.5, which will have c2c tranforms.
#if HAVE_MPI
    mpi_fft3d_plan_t *plan;
#else
    fft_plan_t *plan_g2k;
    fft_plan_t *plan_k2g;
#endif

    int nk;
    double *k2_G;
    double *kx_G;
    double *ky_G;
    double *kz_G;
    uint8_t *wt_G;

    size_t work_array_size;

    int errorcode;
};

typedef struct ciderpw_data_obj *ciderpw_data;

int ciderpw_has_mpi();

void ciderpw_allocate_buffers(ciderpw_data data);

int ciderpw_get_struct_size();

void ciderpw_nullify(ciderpw_data data);

void ciderpw_finalize(ciderpw_data *cider);

void ciderpw_set_unit_cell(ciderpw_data data, int *N_c, double *cell_cv);

ciderpw_data ciderpw_new_anyspin(int nspin);

void ciderpw_create(ciderpw_data *cider, double scale, int *N_c,
                    double *cell_cv);

void ciderpw_setup_kernel(ciderpw_data data, int nalpha, int nbeta,
                          double *norms_ab, double *expnts_ab);

void ciderpw_get_bound_inds(ciderpw_data data, int *bound_inds);

size_t ciderpw_get_recip_size(ciderpw_data data);
size_t ciderpw_get_real_size(ciderpw_data data);
size_t ciderpw_get_work_size(ciderpw_data data);

double *ciderpw_get_work_pointer(ciderpw_data data);

void ciderpw_get_local_size_and_lda(ciderpw_data data, int *sizes);

void ciderpw_compute_kernels_helper(double k2, double *kernel_ab,
                                    double *norms_ab, double *expnts_ab,
                                    int n_ab);

void ciderpw_compute_kernels_sym_helper(double k2, double *kernel_ab,
                                        double *norms_ab, double *expnts_ab,
                                        int nalpha);

void ciderpw_compute_kernels(struct ciderpw_kernel *kernel, double k2,
                             double *kernel_ba);

void ciderpw_compute_kernels_sym(struct ciderpw_kernel *kernel, double k2,
                                 double *kernel_ba);

void ciderpw_compute_kernels_t(struct ciderpw_kernel *kernel, double k2,
                               double *kernel_ab);

void ciderpw_multiply_l1(ciderpw_data data);

void ciderpw_convolution_fwd(ciderpw_data data);

void ciderpw_convolution_bwd(ciderpw_data data);

void ciderpw_compute_features(ciderpw_data data);

void ciderpw_compute_potential(ciderpw_data data);

void ciderpw_eval_feature_vj(ciderpw_data data, double *feat_g, double *p_gq);

void ciderpw_add_potential_vj(ciderpw_data data, double *vfeat_g, double *p_gq);

void ciderpw_set_work(ciderpw_data data, double *fun_g, double *p_gq);

void ciderpw_zero_work(ciderpw_data data);

void ciderpw_fill_atom_info(ciderpw_data data, int64_t *inds_c, double *disps_c,
                            int64_t *num_c, double *r_vg, int64_t *locs_g);

void ciderpw_add_atom_info(ciderpw_data data, double *funcs_ga, int64_t *locs_g,
                           int ng);

void ciderpw_set_atom_info(ciderpw_data data, double *funcs_ga, int64_t *locs_g,
                           int ng);

void ciderpw_copy_work_array_recip(ciderpw_data data, double complex *out);
void ciderpw_copy_work_array_real(ciderpw_data data, double *out);
void ciderpw_copy_work_array(ciderpw_data data, double complex *out);

#endif
