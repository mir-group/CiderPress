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
    DFTI_DESCRIPTOR_HANDLE xhandle;
    DFTI_DESCRIPTOR_HANDLE yhandle;
    DFTI_DESCRIPTOR_HANDLE zhandle;
#else // FFTW
    fftw_plan plan;
#endif
} fft_plan_t;

int cider_fft_is_initialized();

int cider_fft_is_threaded();

#if FFT_BACKEND == FFT_MKL_BACKEND
int cider_fft_get_num_mkl_threads();
#endif

void cider_fft_initialize();

void cider_fft_set_nthread(int nthread);

#if FFT_BACKEND == FFT_MKL_BACKEND
void cider_fft_init_fft3d_1d_parts(const int ntransform, const int nx,
                                   const int ny, const int nz, const int r2c,
                                   const int transpose, const int inplace,
                                   DFTI_DESCRIPTOR_HANDLE *xhandlep,
                                   DFTI_DESCRIPTOR_HANDLE *yhandlep,
                                   DFTI_DESCRIPTOR_HANDLE *zhandlep);
#endif

fft_plan_t *allocate_fftnd_plan(int ndim, int *dims, int fwd, int r2c,
                                int ntransform, int inplace, int batch_first);

int initialize_fft_plan(fft_plan_t *plan, void *in_array, void *out_array);

void execute_fft_plan(fft_plan_t *plan);

void free_fft_plan(fft_plan_t *plan);

void *malloc_fft_plan_in_array(fft_plan_t *plan);

void *malloc_fft_plan_out_array(fft_plan_t *plan);

void *alloc_fft_array(size_t objsize);

void free_fft_array(void *arr);

#endif
