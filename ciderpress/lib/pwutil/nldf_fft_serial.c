#include "nldf_fft_serial.h"
#include "config.h"
#include "nldf_fft_core.h"
#include <assert.h>
#include <complex.h>
#include <fftw3.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void ciderpw_setup_reciprocal_vectors(ciderpw_data data) {
    int N0, N1, N2;
    int N1glob, N0glob, N2glob;
    int kindex;
    data->nk =
        data->icell.Nlocal[0] * data->icell.Nlocal[1] * data->icell.Nlocal[2];
    data->kx_G = (double *)malloc(sizeof(double) * data->nk);
    data->ky_G = (double *)malloc(sizeof(double) * data->nk);
    data->kz_G = (double *)malloc(sizeof(double) * data->nk);
    data->k2_G = (double *)malloc(sizeof(double) * data->nk);
    data->wt_G = (uint8_t *)malloc(sizeof(uint8_t) * data->nk);
    for (N0 = 0; N0 < data->icell.Nlocal[0]; N0++) {
        N0glob = N0 + data->icell.offset[0];
        for (N1 = 0; N1 < data->icell.Nlocal[1]; N1++) {
            N1glob = N1 + data->icell.offset[1];
            for (N2 = 0; N2 < data->icell.Nlocal[2]; N2++) {
                N2glob = N2 + data->icell.offset[2];
                kindex = N2 + data->icell.Nlocal[2] *
                                  (N1 + data->icell.Nlocal[1] * N0);
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

void ciderpw_init_serial(ciderpw_data data, int nalpha, int nbeta,
                         double *norms_ab, double *expnts_ab) {
    const int plan_dims[3] = {data->cell.Nglobal[0], data->cell.Nglobal[1],
                              data->cell.Nglobal[2]};
    ptrdiff_t fftw_alloc_size;
    data->kLDA = data->icell.Nglobal[2];
    data->gLDA = 2 * data->kLDA;
    data->fft_type = CIDERPW_R2C;

    ciderpw_setup_kernel(data, nalpha, nbeta, norms_ab, expnts_ab);

    data->cell.Nlocal[0] = data->cell.Nglobal[0];
    data->cell.Nlocal[1] = data->cell.Nglobal[1];
    data->cell.Nlocal[2] = data->cell.Nglobal[2];
    data->cell.offset[0] = 0;
    data->cell.offset[1] = 0;
    data->cell.offset[2] = 0;

    data->icell.Nlocal[0] = data->icell.Nglobal[0];
    data->icell.Nlocal[1] = data->icell.Nglobal[1];
    data->icell.Nlocal[2] = data->icell.Nglobal[2];
    data->icell.offset[0] = 0;
    data->icell.offset[1] = 0;
    data->icell.offset[2] = 0;

    fftw_alloc_size =
        plan_dims[0] * plan_dims[1] * plan_dims[2] * data->kernel.work_size;
    assert(fftw_alloc_size % data->kernel.work_size == 0);
    data->work_ska = fftw_alloc_complex(fftw_alloc_size);

    if (data->fft_type == CIDERPW_R2C) {
        data->plan_g2k = fftw_plan_many_dft_r2c(
            3, plan_dims, data->kernel.work_size, (double *)data->work_ska,
            NULL, data->kernel.work_size, 1, data->work_ska, NULL,
            data->kernel.work_size, 1, FFTW_ESTIMATE);
        data->plan_k2g = fftw_plan_many_dft_c2r(
            3, plan_dims, data->kernel.work_size, data->work_ska, NULL,
            data->kernel.work_size, 1, (double *)data->work_ska, NULL,
            data->kernel.work_size, 1, FFTW_ESTIMATE);
    } else {
        data->plan_g2k = fftw_plan_many_dft(
            3, plan_dims, data->kernel.work_size, data->work_ska, NULL,
            data->kernel.work_size, 1, data->work_ska, NULL,
            data->kernel.work_size, 1, FFTW_FORWARD, FFTW_ESTIMATE);
        data->plan_k2g = fftw_plan_many_dft(
            3, plan_dims, data->kernel.work_size, data->work_ska, NULL,
            data->kernel.work_size, 1, data->work_ska, NULL,
            data->kernel.work_size, 1, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    assert(data->plan_g2k != NULL);
    assert(data->plan_k2g != NULL);
    ciderpw_allocate_buffers(data);
    ciderpw_setup_reciprocal_vectors(data);
}

void ciderpw_g2k_serial(ciderpw_data data) { fftw_execute(data->plan_g2k); }

void ciderpw_k2g_serial(ciderpw_data data) { fftw_execute(data->plan_k2g); }

void ciderpw_g2k_serial_gpaw(ciderpw_data data, double *in_g,
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
    ciderpw_g2k_serial(data);
    double complex *cbuf = data->work_ska;
    for (int i = 0; i < data->icell.Nlocal[0]; i++) {
        for (int j = 0; j < data->icell.Nlocal[1]; j++) {
            for (int k = 0; k < data->icell.Nlocal[2]; k++) {
                out_g[k] = cbuf[k];
            }
            cbuf = cbuf + data->icell.Nlocal[2];
            out_g = out_g + data->icell.Nlocal[2];
        }
    }
}

void ciderpw_k2g_serial_gpaw(ciderpw_data data, double complex *in_g,
                             double *out_g) {
    double complex *cbuf = data->work_ska;
    for (int i = 0; i < data->icell.Nlocal[0]; i++) {
        for (int j = 0; j < data->icell.Nlocal[1]; j++) {
            for (int k = 0; k < data->icell.Nlocal[2]; k++) {
                cbuf[k] = in_g[k];
            }
            cbuf = cbuf + data->icell.Nlocal[2];
            in_g = in_g + data->icell.Nlocal[2];
        }
    }
    ciderpw_k2g_serial(data);
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