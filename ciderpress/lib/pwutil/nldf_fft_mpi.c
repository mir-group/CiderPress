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

#include "nldf_fft_mpi.h"
#include "config.h"
#include "gpaw_interface.h"
#include "nldf_fft_core.h"
#include <assert.h>
#include <complex.h>
#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void ciderpw_set_communicator(ciderpw_data data, MPI_Comm mpi_comm) {
    assert(data->mpi_comm == NULL);
    MPI_Comm_rank(mpi_comm, &data->mpi_rank);
    MPI_Comm_size(mpi_comm, &data->mpi_size);
    data->mpi_comm = mpi_comm;
}

void ciderpw_set_communicator_from_gpaw(ciderpw_data data,
                                        PyObject *gpaw_mpi_obj) {
    MPI_Comm mpi_comm = unpack_gpaw_comm(gpaw_mpi_obj);
    ciderpw_set_communicator(data, mpi_comm);
}

void ciderpw_all_gatherv_from_gpaw(PyObject *gpaw_mpi_obj, double *sendbuf,
                                   int sendcount, double *recvbuf,
                                   const int *counts, const int *displs) {
    MPI_Comm mpi_comm = unpack_gpaw_comm(gpaw_mpi_obj);
    MPI_Allgatherv(sendbuf, sendcount, MPI_DOUBLE, recvbuf, counts, displs,
                   MPI_DOUBLE, mpi_comm);
}

void ciderpw_setup_reciprocal_vectors(ciderpw_data data) {
    int N0, N1, N2;
    int N1glob, N0glob, N2glob;
    int kindex;
    data->nk =
        data->icell.Nlocal[1] * data->icell.Nlocal[0] * data->icell.Nlocal[2];
    data->kx_G = (double *)malloc(sizeof(double) * data->nk);
    data->ky_G = (double *)malloc(sizeof(double) * data->nk);
    data->kz_G = (double *)malloc(sizeof(double) * data->nk);
    data->k2_G = (double *)malloc(sizeof(double) * data->nk);
    data->wt_G = (uint8_t *)malloc(sizeof(uint8_t) * data->nk);
    for (N1 = 0; N1 < data->icell.Nlocal[1]; N1++) {
        N1glob = N1 + data->icell.offset[1];
        for (N0 = 0; N0 < data->icell.Nlocal[0]; N0++) {
            N0glob = N0 + data->icell.offset[0];
            for (N2 = 0; N2 < data->icell.Nlocal[2]; N2++) {
                N2glob = N2 + data->icell.offset[2];
                kindex = N2 + data->icell.Nlocal[2] *
                                  (N0 + data->icell.Nlocal[0] * N1);
                int k0 = ((N0glob + data->cell.Nglobal[0] / 2) %
                          data->cell.Nglobal[0]) -
                         data->cell.Nglobal[0] / 2;
                int k1 = ((N1glob + data->cell.Nglobal[1] / 2) %
                          data->cell.Nglobal[1]) -
                         data->cell.Nglobal[1] / 2;
                int k2 = ((N2glob + data->cell.Nglobal[2] / 2) %
                          data->cell.Nglobal[2]) -
                         data->cell.Nglobal[2] / 2;
                double kx = data->icell.vec[0] * k0 + data->icell.vec[3] * k1 +
                            data->icell.vec[6] * k2;
                double ky = data->icell.vec[1] * k0 + data->icell.vec[4] * k1 +
                            data->icell.vec[7] * k2;
                double kz = data->icell.vec[2] * k0 + data->icell.vec[5] * k1 +
                            data->icell.vec[8] * k2;
                double ksq = kx * kx + ky * ky + kz * kz;
                data->kx_G[kindex] = kx;
                data->ky_G[kindex] = ky;
                data->kz_G[kindex] = kz;
                data->k2_G[kindex] = ksq;
                data->wt_G[kindex] =
                    (N2glob == 0 || N2glob == data->icell.Nglobal[2] - 1) ? 1
                                                                          : 2;
            }
        }
    }
}

/*
void ciderpw_init_mpi(ciderpw_data data, MPI_Comm mpi_comm, int nalpha,
                      int nbeta, double *norms_ab, double *expnts_ab) {
    fftw_mpi_init();
    ciderpw_set_communicator(data, mpi_comm);

    const ptrdiff_t plan_dims[3] = {
        data->cell.Nglobal[0], data->cell.Nglobal[1], data->cell.Nglobal[2]};
    ptrdiff_t local_size_dims[3];
    ptrdiff_t fftw_alloc_size;
    data->kLDA = data->icell.Nglobal[2];
    data->gLDA = 2 * data->kLDA;
    local_size_dims[0] = data->cell.Nglobal[0];
    local_size_dims[1] = data->cell.Nglobal[1];
    local_size_dims[2] = data->kLDA;
    data->fft_type = CIDERPW_R2C;

    ptrdiff_t fftw_xsize, fftw_xstart, fftw_ysize, fftw_ystart;
    ciderpw_setup_kernel(data, nalpha, nbeta, norms_ab, expnts_ab);

    fftw_alloc_size = fftw_mpi_local_size_many_transposed(
        3, local_size_dims, data->kernel.work_size, FFTW_MPI_DEFAULT_BLOCK,
        FFTW_MPI_DEFAULT_BLOCK, data->mpi_comm, &fftw_xsize, &fftw_xstart,
        &fftw_ysize, &fftw_ystart);
    data->cell.Nlocal[0] = fftw_xsize;
    data->cell.Nlocal[1] = data->cell.Nglobal[1];
    data->cell.Nlocal[2] = data->cell.Nglobal[2];
    data->cell.offset[0] = fftw_xstart;
    data->cell.offset[1] = 0;
    data->cell.offset[2] = 0;

    data->icell.Nlocal[0] = data->icell.Nglobal[0];
    data->icell.Nlocal[1] = fftw_ysize;
    data->icell.Nlocal[2] = data->icell.Nglobal[2];
    data->icell.offset[0] = 0;
    data->icell.offset[1] = fftw_ystart;
    data->icell.offset[2] = 0;

    assert(fftw_alloc_size % data->kernel.work_size == 0);
    data->work_ska = fftw_alloc_complex(fftw_alloc_size);
    data->work_array_size = fftw_alloc_size;

    if (data->fft_type == CIDERPW_R2C) {
        data->plan_g2k = fftw_mpi_plan_many_dft_r2c(
            3, plan_dims, data->kernel.work_size, FFTW_MPI_DEFAULT_BLOCK,
            FFTW_MPI_DEFAULT_BLOCK, (double *)data->work_ska, data->work_ska,
            data->mpi_comm, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT);
        data->plan_k2g = fftw_mpi_plan_many_dft_c2r(
            3, plan_dims, data->kernel.work_size, FFTW_MPI_DEFAULT_BLOCK,
            FFTW_MPI_DEFAULT_BLOCK, data->work_ska, (double *)data->work_ska,
            data->mpi_comm, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
    } else {
        data->plan_g2k = fftw_mpi_plan_many_dft(
            3, plan_dims, data->kernel.work_size, FFTW_MPI_DEFAULT_BLOCK,
            FFTW_MPI_DEFAULT_BLOCK, data->work_ska, data->work_ska,
            data->mpi_comm, FFTW_FORWARD,
            FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT);
        data->plan_k2g = fftw_mpi_plan_many_dft(
            3, plan_dims, data->kernel.work_size, FFTW_MPI_DEFAULT_BLOCK,
            FFTW_MPI_DEFAULT_BLOCK, data->work_ska, data->work_ska,
            data->mpi_comm, FFTW_BACKWARD,
            FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
    }

    assert(data->plan_g2k != NULL);
    assert(data->plan_k2g != NULL);
    ciderpw_allocate_buffers(data);
    ciderpw_setup_reciprocal_vectors(data);
}
*/

void ciderpw_init_mpi(ciderpw_data data, MPI_Comm mpi_comm, int nalpha,
                      int nbeta, double *norms_ab, double *expnts_ab) {
    // fftw_mpi_init();
    ciderpw_set_communicator(data, mpi_comm);

    const int dims[3] = {data->cell.Nglobal[0], data->cell.Nglobal[1],
                         data->cell.Nglobal[2]};
    data->kLDA = data->icell.Nglobal[2];
    data->gLDA = 2 * data->kLDA;
    data->fft_type = CIDERPW_R2C;

    ciderpw_setup_kernel(data, nalpha, nbeta, norms_ab, expnts_ab);

    if (data->fft_type == CIDERPW_R2C) {
        data->plan = allocate_mpi_fft3d_plan(data->mpi_comm, dims, 1,
                                             data->kernel.work_size);
    } else {
        data->plan = allocate_mpi_fft3d_plan(data->mpi_comm, dims, 0,
                                             data->kernel.work_size);
    }
    data->cell.Nlocal[0] = data->plan->r_Nlocal[0];
    data->cell.Nlocal[1] = data->plan->r_Nlocal[1];
    data->cell.Nlocal[2] = data->plan->r_Nlocal[2];
    data->cell.offset[0] = data->plan->r_offset[0];
    data->cell.offset[1] = data->plan->r_offset[1];
    data->cell.offset[2] = data->plan->r_offset[2];

    data->icell.Nlocal[0] = data->plan->k_Nlocal[0];
    data->icell.Nlocal[1] = data->plan->k_Nlocal[1];
    data->icell.Nlocal[2] = data->plan->k_Nlocal[2];
    data->icell.offset[0] = data->plan->k_offset[0];
    data->icell.offset[1] = data->plan->k_offset[1];
    data->icell.offset[2] = data->plan->k_offset[2];

    data->work_ska = data->plan->work;
    data->work_array_size = data->plan->work_array_size;

    ciderpw_allocate_buffers(data);
    ciderpw_setup_reciprocal_vectors(data);
}

void ciderpw_init_mpi_from_gpaw(ciderpw_data data, PyObject *gpaw_comm,
                                int nalpha, int nbeta, double *norms_ab,
                                double *expnts_ab) {
    MPI_Comm comm = unpack_gpaw_comm(gpaw_comm);
    ciderpw_init_mpi(data, comm, nalpha, nbeta, norms_ab, expnts_ab);
}

void ciderpw_g2k_mpi(ciderpw_data data) { execute_mpi_fft3d_fwd(data->plan); }

void ciderpw_k2g_mpi(ciderpw_data data) { execute_mpi_fft3d_bwd(data->plan); }

void ciderpw_g2k_mpi_gpaw(ciderpw_data data, double *in_g,
                          double complex *out_g) {
    double *buf = (double *)data->work_ska;
    for (int i = 0; i < data->cell.Nlocal[0]; i++) {
        for (int j = 0; j < data->cell.Nlocal[1]; j++) {
            for (int k = 0; k < data->cell.Nlocal[2]; k++) {
                buf[k] = in_g[k];
            }
            buf = buf + data->gLDA;
            in_g = in_g + data->cell.Nlocal[2];
        }
    }
    ciderpw_g2k_mpi(data);
    double complex *cbuf = data->work_ska;
    for (int j = 0; j < data->icell.Nlocal[1]; j++) {
        for (int i = 0; i < data->icell.Nlocal[0]; i++) {
            for (int k = 0; k < data->icell.Nlocal[2]; k++) {
                out_g[k] = cbuf[k];
            }
            cbuf = cbuf + data->icell.Nlocal[2];
            out_g = out_g + data->icell.Nlocal[2];
        }
    }
}

void ciderpw_k2g_mpi_gpaw(ciderpw_data data, double complex *in_g,
                          double *out_g) {
    double complex *cbuf = data->work_ska;
    for (int j = 0; j < data->icell.Nlocal[1]; j++) {
        for (int i = 0; i < data->icell.Nlocal[0]; i++) {
            for (int k = 0; k < data->icell.Nlocal[2]; k++) {
                cbuf[k] = in_g[k];
            }
            cbuf = cbuf + data->icell.Nlocal[2];
            in_g = in_g + data->icell.Nlocal[2];
        }
    }
    ciderpw_k2g_mpi(data);
    double *buf = (double *)data->work_ska;
    for (int i = 0; i < data->cell.Nlocal[0]; i++) {
        for (int j = 0; j < data->cell.Nlocal[1]; j++) {
            for (int k = 0; k < data->cell.Nlocal[2]; k++) {
                out_g[k] = buf[k];
            }
            buf = buf + data->gLDA;
            out_g = out_g + data->cell.Nlocal[2];
        }
    }
}
