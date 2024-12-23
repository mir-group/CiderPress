#ifndef _CIDER_FFT_H
#define _CIDER_FFT_H

#include "cider_fft_config.h"

#if FFT_BACKEND == FFT_MKL_BACKEND
#include <mkl.h>
#include <mkl_dfti.h>
#include <mkl_types.h>
#else // FFTW
#include <fftw3.h>
#include <limits.h>
#endif
#if HAVE_MPI
#include <mpi.h>
#endif

#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct fft_plan {
    int is_initialized;
    int ndim;
    int *dims;
    int r2c;
    int ntransform;
    size_t fft_in_size;
    size_t fft_out_size;
    int fwd;
    int batch_first;
    int inplace;
    int stride;
    int idist;
    int odist;
    void *in;
    void *out;
#if FFT_BACKEND == FFT_MKL_BACKEND
    DFTI_DESCRIPTOR_HANDLE handle;
#else // FFTW
    fftw_plan plan;
#endif
} fft_plan_t;

fft_plan_t *allocate_fftnd_plan(int ndim, int *dims, int fwd, int r2c,
                                int ntransform, int inplace, int batch_first);

int initialize_fft_plan(fft_plan_t *plan, void *in_array, void *out_array);

void execute_fft_plan(fft_plan_t *plan);

void free_fft_plan(fft_plan_t *plan);

void *malloc_fft_plan_in_array(fft_plan_t *plan);

void *malloc_fft_plan_out_array(fft_plan_t *plan);

void free_fft_array(void *arr);

#endif
