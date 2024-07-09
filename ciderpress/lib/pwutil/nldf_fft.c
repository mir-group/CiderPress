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

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

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

void cider_pw_setup_reciprocal_vectors(ciderpw_data data) {
    int N0, N1, N2;
    int N1glob, N0glob, N2glob;
    int kindex;
    data->nk =
        data->icell.Nlocal[1] * data->icell.Nlocal[0] * data->icell.Nlocal[2];
    data->kx_G = (double *)malloc(sizeof(double) * data->nk);
    data->ky_G = (double *)malloc(sizeof(double) * data->nk);
    data->kz_G = (double *)malloc(sizeof(double) * data->nk);
    data->k2_G = (double *)malloc(sizeof(double) * data->nk);
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
            }
        }
    }
}

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
    data->fft_type = 0;

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

    data->icell.Nlocal[0] = data->cell.Nglobal[0];
    data->icell.Nlocal[1] = fftw_ysize;
    data->icell.Nlocal[2] = data->icell.Nglobal[2];
    data->icell.offset[1] = fftw_ystart;

    assert(fftw_alloc_size % data->kernel.work_size == 0);
    data->work_ska = fftw_alloc_complex(fftw_alloc_size);

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
    cider_pw_setup_reciprocal_vectors(data);
}

void ciderpw_init_mpi_from_gpaw(ciderpw_data data, PyObject *gpaw_comm,
                                int nalpha, int nbeta, double *norms_ab,
                                double *expnts_ab) {
    MPI_Comm comm = unpack_gpaw_comm(gpaw_comm);
    ciderpw_init_mpi(data, comm, nalpha, nbeta, norms_ab, expnts_ab);
}

void ciderpw_g2k_mpi(ciderpw_data data) {
    fftw_execute(data->plan_g2k);
    // if (data->fft_type == CIDERPW_R2C) {
    //     fftw_mpi_execute_dft_r2c(data->plan_g2k, (double *)data->work_ska,
    //                              data->work_ska);
    // } else {
    //     fftw_mpi_execute_dft(data->plan_g2k, data->work_ska, data->work_ska);
    // }
}

void ciderpw_k2g_mpi(ciderpw_data data) {
    // fftw_execute(data->plan_k2g);
    if (data->fft_type == CIDERPW_R2C) {
        fftw_mpi_execute_dft_c2r(data->plan_k2g, data->work_ska,
                                 (double *)data->work_ska);
    } else {
        fftw_mpi_execute_dft(data->plan_k2g, data->work_ska, data->work_ska);
    }
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

void ciderpw_compute_kernels(struct ciderpw_kernel *kernel, double k2,
                             double *kernel_ba) {
    ciderpw_compute_kernels_helper(k2, kernel_ba, kernel->norms_ba,
                                   kernel->expnts_ba,
                                   kernel->nbeta * kernel->nalpha);
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
        ciderpw_compute_kernels(&data->kernel, data->k2_G[kindex], kernel_ba);
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
        ciderpw_compute_kernels_t(&data->kernel, data->k2_G[kindex], kernel_ab);
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
}

void ciderpw_compute_features(ciderpw_data data) {
    ciderpw_g2k_mpi(data);
    ciderpw_convolution_fwd(data);
    ciderpw_k2g_mpi(data);
}

void ciderpw_compute_potential(ciderpw_data data) {
    ciderpw_g2k_mpi(data);
    ciderpw_convolution_bwd(data);
    ciderpw_k2g_mpi(data);
}

/*
void ciderpw_eval_feature_vj(ciderpw_data data, double *feat_g, double *dfeat_g,
                             double *p_gq, double *dp_gq) {
    int N0, N1, N2;
    int N = 0;
    double *work_ga = (double *)data->work_ska;
    double tot = 0;
    int ind;
    for (N0 = 0; N0 < data->cell.Nlocal[0]; N0++) {
        for (N1 = 0; N1 < data->cell.Nlocal[1]; N1++) {
            for (N2 = 0; N2 < data->cell.Nlocal[2]; N2++, N++) {
                feat_g[N2] = 0;
                dfeat_g[N2] = 0;
                for (a = 0; a < nalpha; a++) {
                    ind = N2 * nalpha + a;
                    feat_g[N2] += work_ga[ind] * p_g[ind];
                    dfeat_g[N2] += work_ga[ind] * dp_g[ind];
                }
            }
            work_ga += data->gLDA * nalpha;
            p_qg += data->cell.Nlocal[2] * nalpha;
            dp_qg += data->cell.Nlocal[2] * nalpha;
            feat_g += data->cell.Nlocal[2];
            dfeat_g += data->cell.Nlocal[2];
        }
    }
}
*/

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
                    // if (isnan(feat_g[N2])) {
                    //     printf("%d %d %d %d %d %lf %lf\n",
                    //            N0, N1, N2, a, ind, work_ga[N2], p_gq[ind]);
                    // }
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
    int N0, N1, N2;
    double complex *work_ga = data->work_ska;
    int ind, a;
    for (N0 = 0; N0 < data->icell.Nlocal[0]; N0++) {
        for (N1 = 0; N1 < data->icell.Nlocal[1]; N1++) {
            for (N2 = 0; N2 < data->icell.Nlocal[2]; N2++) {
                for (a = 0; a < data->kernel.nalpha; a++) {
                    ind = N2 * data->kernel.nalpha + a;
                    work_ga[ind] = 0.0;
                }
            }
            work_ga += data->kLDA * data->kernel.work_size;
        }
    }
}
