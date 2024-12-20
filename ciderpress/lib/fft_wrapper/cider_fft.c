
#define HAVE_MPI 0
#define FFT_MKL_BACKEND 0
#define FFT_FFTW_BACKEND 1

#define FFT_BACKEND FFT_MKL_BACKEND
// #define FFT_BACKEND FFT_FFTW_BACKEND

#if FFT_BACKEND == FFT_MKL_BACKEND
#include <mkl.h>
#include <mkl_dfti.h>
#include <mkl_types.h>
#define FFT_DIM MKL_LONG
#define FFT_MPI_DIM MKL_LONG
#define FFT_SIZE MKL_LONG
#else // FFTW
#include <fftw3.h>
#include <limits.h>
#define FFT_DIM int
#define FFT_MPI_DIM ptrdiff_t
#define FFT_SIZE ptrdiff_t
#endif
#if HAVE_MPI
#include <mpi.h>
#endif
#include <complex.h>
#include <stdint.h>
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

/*fft_plan_t* allocate_fft1d_single_inplace_plan(int n, int r2c, int fwd) {
    fft_plan_t *plan = (fft_plan_t *)malloc(sizeof(fft_plan_t));
#if FFT_BACKEND == FFT_MKL_BACKEND
    MKL_LONG dims[1] = {n};
    if (r2c) {
        DftiCreateDescriptor(&(plan->handle), DFTI_DOUBLE, DFTI_REAL, 1, dims);
    } else {
        DftiCreateDescriptor(&(plan->handle), DFTI_DOUBLE, DFTI_COMPLEX, 1,
dims);
    }
#else
    if (r2c && fwd) {
        rfftw_create_plan(n, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);
    } else if (r2c) {
        rfftw_create_plan(n, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE);
    } else {
        fftw_create_plan(n, fwd ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE);
    }
#endif
}*/

#if FFT_BACKEND == FFT_MKL_BACKEND
void cider_fft_check_status(int status) {
    if (status != 0) {
        printf("FFT ROUTINE FAILED WITH STATUS %d:\n", status);
        char *message = DftiErrorMessage(status);
        printf("%s\n", message);
        exit(-1);
    }
}
#endif

fft_plan_t *allocate_fftnd_plan(int ndim, int *dims, int fwd, int r2c,
                                int ntransform, int inplace, int batch_first) {
    fft_plan_t *plan = (fft_plan_t *)malloc(sizeof(fft_plan_t));
    plan->is_initialized = 0;
    plan->ndim = ndim;
    plan->dims = (int *)malloc(ndim * sizeof(int));
    for (int i = 0; i < plan->ndim; i++) {
        plan->dims[i] = dims[i];
    }
    plan->r2c = r2c;
    plan->ntransform = ntransform;
    plan->fwd = fwd;
    plan->batch_first = batch_first;
    plan->inplace = inplace;
    plan->in = NULL;
    plan->out = NULL;
    printf("HERE1\n");

    if (r2c) {
        size_t real_dist, recip_dist;
        recip_dist = dims[ndim - 1] / 2 + 1;
        for (int i = 0; i < ndim - 1; i++) {
            recip_dist *= dims[i];
        }
        if (inplace) {
            real_dist = recip_dist * 2;
        } else {
            real_dist = 1;
            for (int i = 0; i < ndim; i++) {
                real_dist *= dims[i];
            }
        }
        if (fwd) {
            plan->fft_in_size = real_dist;
            plan->fft_out_size = recip_dist;
        } else {
            plan->fft_in_size = recip_dist;
            plan->fft_out_size = real_dist;
        }
    } else {
        size_t dist = 1;
        for (int i = 0; i < ndim; i++) {
            dist *= dims[i];
        }
        plan->fft_in_size = dist;
        plan->fft_out_size = dist;
    }

    int stride;
    size_t idist, odist;
    if (batch_first) {
        stride = 1;
        idist = plan->fft_in_size;
        odist = plan->fft_out_size;
    } else {
        stride = ntransform;
        idist = 1;
        odist = 1;
    }
    plan->stride = stride;
    plan->idist = idist;
    plan->odist = odist;

    printf("HERE2\n");
#if FFT_BACKEND == FFT_MKL_BACKEND
    MKL_LONG status;
    MKL_LONG ldims[ndim];
    plan->handle = NULL;
    for (int i = 0; i < ndim; i++) {
        ldims[i] = plan->dims[i];
    }
    if (r2c) {
        if (ndim > 1) {
            status = DftiCreateDescriptor(&(plan->handle), DFTI_DOUBLE,
                                          DFTI_REAL, ndim, ldims);
            cider_fft_check_status(status);
            MKL_LONG rstrides[ndim + 1];
            MKL_LONG cstrides[ndim + 1];
            rstrides[ndim] = stride;
            cstrides[ndim] = stride;
            cstrides[ndim - 1] = (ldims[ndim - 1] / 2 + 1) * cstrides[ndim];
            if (inplace) {
                rstrides[ndim - 1] = 2 * cstrides[ndim - 1];
            } else {
                rstrides[ndim - 1] = ldims[ndim - 1] * stride;
            }
            for (int d = ndim - 2; d > 0; d--) {
                rstrides[d] = rstrides[d + 1] * ldims[d];
                cstrides[d] = cstrides[d + 1] * ldims[d];
            }
            rstrides[0] = 0;
            cstrides[0] = 0;
            if (fwd) {
                status =
                    DftiSetValue(plan->handle, DFTI_INPUT_STRIDES, rstrides);
                cider_fft_check_status(status);
                status =
                    DftiSetValue(plan->handle, DFTI_OUTPUT_STRIDES, cstrides);
                cider_fft_check_status(status);
            } else {
                status =
                    DftiSetValue(plan->handle, DFTI_INPUT_STRIDES, cstrides);
                cider_fft_check_status(status);
                status =
                    DftiSetValue(plan->handle, DFTI_OUTPUT_STRIDES, rstrides);
                cider_fft_check_status(status);
            }
        } else {
            status = DftiCreateDescriptor(&(plan->handle), DFTI_DOUBLE,
                                          DFTI_REAL, ndim, ldims[0]);
            cider_fft_check_status(status);
        }
        printf("HERE3\n");
        status = DftiSetValue(plan->handle, DFTI_CONJUGATE_EVEN_STORAGE,
                              DFTI_COMPLEX_COMPLEX);
        cider_fft_check_status(status);
        status =
            DftiSetValue(plan->handle, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT);
        cider_fft_check_status(status);
        printf("HERE35\n");
    } else {
        if (ndim == 1) {
            status = DftiCreateDescriptor(&(plan->handle), DFTI_DOUBLE,
                                          DFTI_COMPLEX, ndim, ldims[0]);
        } else {
            status = DftiCreateDescriptor(&(plan->handle), DFTI_DOUBLE,
                                          DFTI_COMPLEX, ndim, ldims);
        }
        cider_fft_check_status(status);
    }
    printf("HERE36\n");
    // status = DftiSetValue(plan->handle, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)
    // plan->ntransform);
    cider_fft_check_status(status);
    // status = DftiSetValue(plan->handle, DFTI_INPUT_DISTANCE, (MKL_LONG)
    // plan->idist); status = DftiSetValue(plan->handle, DFTI_OUTPUT_DISTANCE,
    // (MKL_LONG) plan->odist);
    printf("HERE37\n");
    if (inplace) {
        status = DftiSetValue(plan->handle, DFTI_PLACEMENT, DFTI_INPLACE);
    } else {
        status = DftiSetValue(plan->handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    }
    printf("HERE38 %p\n", plan->handle);
    status = DftiCommitDescriptor(plan->handle);
    cider_fft_check_status(status);
    printf("HERE39 %d\n", status);
#else
    plan->plan = NULL;
#endif
    printf("HERE4\n");

    return plan;
}

#if FFT_BACKEND == FFT_FFTW_BACKEND
void initialize_fftw_settings(fft_plan_t *plan) {
    if (plan->plan != NULL) {
        fftw_destroy_plan(plan->plan);
        plan->plan = NULL;
    }
    if (plan->r2c) {
        if (plan->fwd) {
            plan->plan = fftw_plan_many_dft_r2c(
                plan->ndim, plan->dims, plan->ntransform, (double *)plan->in,
                NULL, plan->stride, (int)plan->idist, (fftw_complex *)plan->out,
                NULL, plan->stride, (int)plan->odist, FFTW_ESTIMATE);
        } else {
            plan->plan = fftw_plan_many_dft_c2r(
                plan->ndim, plan->dims, plan->ntransform,
                (fftw_complex *)plan->in, NULL, plan->stride, (int)plan->idist,
                (double *)plan->out, NULL, plan->stride, (int)plan->odist,
                FFTW_ESTIMATE);
        }
    } else {
        plan->plan = fftw_plan_many_dft(
            plan->ndim, plan->dims, plan->ntransform, (fftw_complex *)plan->in,
            NULL, plan->stride, (int)plan->idist, (fftw_complex *)plan->out,
            NULL, plan->stride, (int)plan->odist,
            plan->fwd ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    plan->is_initialized = 1;
}
#endif

int initialize_fft_plan(fft_plan_t *plan, void *in_array, void *out_array) {
    if (plan->inplace) {
        if (in_array == NULL) {
            return 1;
        }
        plan->in = in_array;
        plan->out = out_array;
    } else {
        if (in_array == NULL || out_array == NULL) {
            return 1;
        }
        plan->in = in_array;
        plan->out = out_array;
    }
#if FFT_BACKEND == FFT_FFTW_BACKEND
    initialize_fftw_settings(plan);
#endif
    return 0;
}

void execute_fft_plan(fft_plan_t *plan) {
#if FFT_BACKEND == FFT_MKL_BACKEND
    if (plan->inplace) {
        if (plan->fwd) {
            DftiComputeForward(plan->handle, plan->in);
        } else {
            DftiComputeBackward(plan->handle, plan->in);
        }
    } else {
        if (plan->fwd) {
            DftiComputeForward(plan->handle, plan->in, plan->out);
        } else {
            DftiComputeBackward(plan->handle, plan->in, plan->out);
        }
    }
#else
    printf("%p\n", plan->plan);
    fftw_execute(plan->plan);
#endif
}

void free_fft_plan(fft_plan_t *plan) {
    free(plan->dims);
#if FFT_BACKEND == FFT_MKL_BACKEND
    DftiFreeDescriptor(&(plan->handle));
#else
    fftw_destroy_plan(plan->plan);
#endif
    free(plan);
}

void *malloc_fft_plan_in_array(fft_plan_t *plan) {
    size_t objsize;
    if (plan->r2c && plan->fwd) {
        objsize = sizeof(double);
    } else {
        objsize = sizeof(double complex);
    }
    objsize *= plan->ntransform;
    objsize *= plan->fft_in_size;
#if FFT_BACKEND == FFT_MKL_BACKEND
    return mkl_malloc(objsize, 64);
#else
    return fftw_malloc(objsize);
#endif
}

void *malloc_fft_plan_out_array(fft_plan_t *plan) {
    size_t objsize;
    if (plan->r2c && (!plan->fwd)) {
        objsize = sizeof(double);
    } else {
        objsize = sizeof(double complex);
    }
    objsize *= plan->ntransform;
    objsize *= plan->fft_out_size;
#if FFT_BACKEND == FFT_MKL_BACKEND
    return mkl_malloc(objsize, 64);
#else
    return fftw_malloc(objsize);
#endif
}

void write_fft_input(fft_plan_t *plan, void *input) {
    size_t size = plan->ntransform * plan->fft_in_size;
    if (plan->r2c && plan->fwd) {
        double *src = (double *)input;
        double *dst = (double *)plan->in;
        for (size_t i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    } else {
        double complex *src = (double complex *)input;
        double complex *dst = (double complex *)plan->in;
        for (size_t i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    }
}

void read_fft_output(fft_plan_t *plan, void *output) {
    size_t size = plan->ntransform * plan->fft_out_size;
    if (plan->r2c && (!plan->fwd)) {
        double *src = (double *)plan->out;
        double *dst = (double *)output;
        for (size_t i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    } else {
        double complex *src = (double complex *)plan->out;
        double complex *dst = (double complex *)output;
        for (size_t i = 0; i < size; i++) {
            // printf("%lf %lf\n", creal(src[i]), cimag(src[i]));
            dst[i] = src[i];
        }
    }
}

void free_fft_array(void *arr) {
#if FFT_BACKEND == FFT_MKL_BACKEND
    mkl_free(arr);
#else
    fftw_free(arr);
#endif
}

void *get_fft_plan_in_array(fft_plan_t *plan) { return plan->in; }

void *get_fft_plan_out_array(fft_plan_t *plan) { return plan->out; }

// NOTE: Dangerous, casting to fewer bits
int get_fft_input_size(fft_plan_t *plan) { return (int)plan->fft_in_size; }

// NOTE: Dangerous, casting to fewer bits
int get_fft_output_size(fft_plan_t *plan) { return (int)plan->fft_out_size; }
