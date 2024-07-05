#include "config.h"
#include <assert.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_MPI
#include <fftw3-mpi.h>
#include <mpi.h>
#else
#include <fftw3.h>
#endif

#include "gpaw_interface.h"
#include "nldf_fft.h"

void ciderpw_set_communicator(ciderpw_data data, MPI_Comm mpi_comm) {
    assert(data->mpi_comm == NULL);
    MPI_Comm_rank(mpi_comm, &data->mpi_rank);
    MPI_Comm_size(mpi_comm, &data->mpi_size);
    data->mpi_comm = mpi_comm;
}

void ciderpw_allocate_buffers(ciderpw_data data) {
    data->Ng =
        data->cell.Nlocal[0] * data->cell.Nlocal[1] * data->cell.Nlocal[2];
}

void ciderpw_set_communicator_from_gpaw(ciderpw_data data,
                                        PyObject *gpaw_mpi_obj) {
    MPI_Comm mpi_comm = unpack_gpaw_comm(gpaw_mpi_obj);
    ciderpw_set_communicator(data, mpi_comm);
}

int ciderpw_get_struct_size() { return sizeof(struct ciderpw_data_obj); }

void ciderpw_nullify(ciderpw_data data) {
    memset(data, 0, ciderpw_get_struct_size());
}

void ciderpw_finalize(ciderpw_data *cider) {
    ciderpw_data data = cider[0];
    if (data->plan_g2k != NULL) {
        fftw_destroy_plan(data->plan_g2k);
    }
    if (data->plan_k2g != NULL) {
        fftw_destroy_plan(data->plan_k2g);
    }
    if (data->work_ska != NULL) {
        fftw_free(data->work_ska);
    }
    ciderpw_nullify(data);
    free(data);
    cider[0] = NULL;
}

void ciderpw_set_unit_cell(ciderpw_data data, int *N_c, double *cell_cv) {
    double C00 = cell_cv[0];
    double C01 = cell_cv[1];
    double C02 = cell_cv[2];
    double C10 = cell_cv[3];
    double C11 = cell_cv[4];
    double C12 = cell_cv[5];
    double C20 = cell_cv[6];
    double C21 = cell_cv[7];
    double C22 = cell_cv[8];
    double det =
        (C00 * (C11 * C22 - C21 * C12) - C01 * (C10 * C22 - C12 * C20) +
         C02 * (C10 * C21 - C11 * C20));
    data->cell.dV = det / (N_c[0] * N_c[1] * N_c[2]);
    data->cell.vec[0] = C00;
    data->cell.vec[1] = C01;
    data->cell.vec[2] = C02;
    data->cell.vec[3] = C10;
    data->cell.vec[4] = C11;
    data->cell.vec[5] = C12;
    data->cell.vec[6] = C20;
    data->cell.vec[7] = C21;
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
    // vdwxc_set_defaults(vdw, functional, nspin);
    // vdw->kernel = vdwxc_default_kernel();
    return data;
}

void ciderpw_create(ciderpw_data *cider, double scale, int *N_c,
                    double *cell_cv) {
    ciderpw_data data = ciderpw_new_anyspin(1);
    ciderpw_set_unit_cell(data, N_c, cell_cv);
    cider[0] = data;
}

void ciderpw_init_mpi(ciderpw_data data, MPI_Comm mpi_comm) {
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

    ptrdiff_t fftw_xsize, fftw_xstart, fftw_ysize, fftw_ystart;
    data->kernel.nalpha = 1;

    fftw_alloc_size = fftw_mpi_local_size_many_transposed(
        3, local_size_dims, data->kernel.nalpha, FFTW_MPI_DEFAULT_BLOCK,
        FFTW_MPI_DEFAULT_BLOCK, data->mpi_comm, &fftw_xsize, &fftw_xstart,
        &fftw_ysize, &fftw_ystart);
    data->cell.Nlocal[0] = fftw_xsize;
    data->cell.Nlocal[1] = data->cell.Nglobal[1];
    data->cell.Nlocal[2] = data->cell.Nglobal[2];
    data->cell.offset[0] = fftw_xstart;

    data->icell.Nlocal[0] = data->cell.Nglobal[0];
    data->icell.Nlocal[1] = fftw_ysize;
    data->icell.Nlocal[2] = data->icell.Nglobal[2];
    data->icell.offset[1] = fftw_ystart;

    assert(fftw_alloc_size % data->kernel.nalpha == 0);
    data->work_ska = fftw_alloc_complex(fftw_alloc_size);

    if (data->fft_type == CIDERPW_R2C) {
        data->plan_g2k = fftw_mpi_plan_many_dft_r2c(
            3, plan_dims, data->kernel.nalpha, FFTW_MPI_DEFAULT_BLOCK,
            FFTW_MPI_DEFAULT_BLOCK, (double *)data->work_ska, data->work_ska,
            data->mpi_comm, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT);
        data->plan_k2g = fftw_mpi_plan_many_dft_c2r(
            3, plan_dims, data->kernel.nalpha, FFTW_MPI_DEFAULT_BLOCK,
            FFTW_MPI_DEFAULT_BLOCK, data->work_ska, (double *)data->work_ska,
            data->mpi_comm, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
    } else {
        data->plan_g2k = fftw_mpi_plan_many_dft(
            3, plan_dims, data->kernel.nalpha, FFTW_MPI_DEFAULT_BLOCK,
            FFTW_MPI_DEFAULT_BLOCK, data->work_ska, data->work_ska,
            data->mpi_comm, FFTW_FORWARD,
            FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT);
        data->plan_k2g = fftw_mpi_plan_many_dft(
            3, plan_dims, data->kernel.nalpha, FFTW_MPI_DEFAULT_BLOCK,
            FFTW_MPI_DEFAULT_BLOCK, data->work_ska, data->work_ska,
            data->mpi_comm, FFTW_BACKWARD,
            FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
    }

    assert(data->plan_g2k != NULL);
    assert(data->plan_k2g != NULL);
    ciderpw_allocate_buffers(data);
}

void ciderpw_init_mpi_from_gpaw(ciderpw_data data, PyObject *gpaw_comm) {
    MPI_Comm comm = unpack_gpaw_comm(gpaw_comm);
    ciderpw_init_mpi(data, comm);
}

void ciderpw_g2k_mpi(ciderpw_data data) {
    fftw_execute(data->plan_g2k);
    // if (data->fft_type == CIDERPW_R2C) {
    //     fftw_mpi_execute_dft_r2c(data->plan_g2k, (double *)data->work_ska,
    //                          data->work_ska);
    // } else {
    //     fftw_mpi_execute_dft(data->plan_g2k, data->work_ska, data->work_ska);
    // }
}

void ciderpw_k2g_mpi(ciderpw_data data) {
    fftw_execute(data->plan_k2g);
    // if (data->fft_type == CIDERPW_R2C) {
    //     fftw_mpi_execute_dft_c2r(data->plan_k2g, data->work_ska,
    //                          (double *)data->work_ska);
    // } else {
    //     fftw_mpi_execute_dft(data->plan_k2g, data->work_ska, data->work_ska);
    // }
}

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
