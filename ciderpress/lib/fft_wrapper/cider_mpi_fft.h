#ifndef _CIDER_MPI_FFT_H
#define _CIDER_MPI_FFT_H
#if 1

#include "cider_fft_config.h"

#if FFT_BACKEND == FFT_MKL_BACKEND
#include <mkl.h>
#include <mkl_cdft.h>
#include <mkl_dfti.h>
#include <mkl_types.h>
#else // FFTW
#include <fftw3-mpi.h>
#include <fftw3.h>
#include <limits.h>
#endif
#include <mpi.h>

#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct mpi_fft3d_plan {
    MPI_Comm comm;
    int r_offset[3];
    int r_Nlocal[3];
    int r_Nglobal[3];
    int k_offset[3];
    int k_Nlocal[3];
    int k_Nglobal[3];
    int xpp;
    int ypp;
    int r2c;
    int ntransform;
    size_t fft_in_size;
    size_t fft_out_size;
    size_t work_array_size;
    int fwd;
    int inplace;
    void *work;
#if FFT_BACKEND == FFT_MKL_BACKEND
    DFTI_DESCRIPTOR_HANDLE xhandle;
    DFTI_DESCRIPTOR_HANDLE yhandle;
    DFTI_DESCRIPTOR_HANDLE zhandle;
#else // FFTW
    fftw_plan fwd_plan;
    fftw_plan bwd_plan;
#endif
} mpi_fft3d_plan_t;

mpi_fft3d_plan_t *allocate_mpi_fft3d_plan(MPI_Comm comm, const int *dims,
                                          int r2c, int ntransform);

void execute_mpi_fft3d_fwd(mpi_fft3d_plan_t *plan);

void execute_mpi_fft3d_bwd(mpi_fft3d_plan_t *plan);

void free_mpi_fft3d_plan(mpi_fft3d_plan_t *plan);

#endif
#endif
