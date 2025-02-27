#include "cider_fft.h"
#if HAVE_MPI
#include <mpi.h>
#if FFT_BACKEND == FFT_FFTW_BACKEND
#include <fftw3-mpi.h>
#endif
#endif
#include <omp.h>

int CIDER_FFT_THREADED = 0;
int CIDER_FFT_INITIALIZED = 0;
#if FFT_BACKEND == FFT_MKL_BACKEND
int CIDER_FFT_MKL_NTHREAD = 1;
#endif

int cider_fft_is_initialized() { return CIDER_FFT_INITIALIZED; }

int cider_fft_is_threaded() { return CIDER_FFT_THREADED; }

#if FFT_BACKEND == FFT_MKL_BACKEND
int cider_fft_get_num_mkl_threads() { return CIDER_FFT_MKL_NTHREAD; }
#endif

void cider_fft_initialize() {
#if HAVE_MPI
    int already_initialized = 1;
    // Check whether MPI is already initialized
    MPI_Initialized(&already_initialized);
    if (already_initialized) {
        int provided;
        MPI_Query_thread(&provided);
        int threads_ok = provided >= MPI_THREAD_FUNNELED;
#if FFT_BACKEND == FFT_FFTW_BACKEND
        if (threads_ok)
            threads_ok = fftw_init_threads();
        fftw_mpi_init();
#endif
        CIDER_FFT_THREADED = threads_ok;
    } else {
#if FFT_BACKEND == FFT_FFTW_BACKEND
        fftw_init_threads();
#endif
        CIDER_FFT_THREADED = 1;
    }
#else
#if FFT_BACKEND == FFT_FFTW_BACKEND
    fftw_init_threads();
#endif
    CIDER_FFT_THREADED = 1;
#endif
    CIDER_FFT_INITIALIZED = 1;
}

/** Set the number of threads to use in FFTs.
 * This function first calls cider_fft_initialize().
 * Next, if CIDER_FFT_THREADED is 0/false, no other action
 * is performed. If CIDER_FFT_THREADED is 1/true, the number
 * of threads for FFTs is set.
 * If nthread == -1, the maximum number of available threads is used.
 * If nthread == 0, the number of threads is not changed.
 * If nthread > 0, the number of threads is set to nthread.
 * Otherwise no action is taken.
 */
void cider_fft_set_nthread(int nthread) {
    cider_fft_initialize();
    if (CIDER_FFT_THREADED) {
#if FFT_BACKEND == FFT_MKL_BACKEND
        if (nthread == -1) {
            CIDER_FFT_MKL_NTHREAD = mkl_get_max_threads();
        } else if (nthread > 0) {
            CIDER_FFT_MKL_NTHREAD = nthread;
        }
#else
        if (nthread == -1) {
            fftw_plan_with_nthreads(omp_get_max_threads());
        } else if (nthread > 0) {
            fftw_plan_with_nthreads(nthread);
        }
#endif
    }
}

#if FFT_BACKEND == FFT_MKL_BACKEND
static void cider_fft_check_status(int status) {
    if (status != 0) {
        printf("FFT ROUTINE FAILED WITH STATUS %d:\n", status);
        char *message = DftiErrorMessage(status);
        printf("%s\n", message);
        exit(-1);
    }
}
#endif

#if FFT_BACKEND == FFT_MKL_BACKEND
void cider_fft_init_fft3d_1d_parts(const int ntransform, const int nx,
                                   const int ny, const int nz, const int r2c,
                                   const int transpose, const int inplace,
                                   DFTI_DESCRIPTOR_HANDLE *xhandlep,
                                   DFTI_DESCRIPTOR_HANDLE *yhandlep,
                                   DFTI_DESCRIPTOR_HANDLE *zhandlep) {
    MKL_LONG status;
    MKL_LONG zstrides[2] = {0, ntransform};
    int stride;
    if (r2c) {
        stride = nz / 2 + 1;
    } else {
        stride = nz;
    }
    MKL_LONG ystrides[2] = {0, stride * ntransform};
    MKL_LONG xstrides[2] = {0, stride * ntransform};
    if (!transpose) {
        xstrides[1] *= ny;
    }
    if (r2c) {
        status = DftiCreateDescriptor(zhandlep, DFTI_DOUBLE, DFTI_REAL, 1, nz);
        cider_fft_check_status(status);
        status = DftiSetValue(*zhandlep, DFTI_CONJUGATE_EVEN_STORAGE,
                              DFTI_COMPLEX_COMPLEX);
        cider_fft_check_status(status);
        status = DftiSetValue(*zhandlep, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT);
        cider_fft_check_status(status);
    } else {
        status =
            DftiCreateDescriptor(zhandlep, DFTI_DOUBLE, DFTI_COMPLEX, 1, nz);
        cider_fft_check_status(status);
    }
    status = DftiSetValue(*zhandlep, DFTI_INPUT_STRIDES, zstrides);
    status = DftiSetValue(*zhandlep, DFTI_OUTPUT_STRIDES, zstrides);
    status = DftiSetValue(*zhandlep, DFTI_INPUT_DISTANCE, 1);
    status = DftiSetValue(*zhandlep, DFTI_OUTPUT_DISTANCE, 1);
    status = DftiSetValue(*zhandlep, DFTI_NUMBER_OF_TRANSFORMS, zstrides[1]);
    status = DftiSetValue(*zhandlep, DFTI_THREAD_LIMIT, 1);
    if (inplace) {
        status = DftiSetValue(*zhandlep, DFTI_PLACEMENT, DFTI_INPLACE);
    } else {
        status = DftiSetValue(*zhandlep, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    }
    cider_fft_check_status(status);
    status = DftiCommitDescriptor(*zhandlep);
    cider_fft_check_status(status);

    status = DftiCreateDescriptor(yhandlep, DFTI_DOUBLE, DFTI_COMPLEX, 1, ny);
    cider_fft_check_status(status);
    status = DftiSetValue(*yhandlep, DFTI_INPUT_STRIDES, ystrides);
    status = DftiSetValue(*yhandlep, DFTI_OUTPUT_STRIDES, ystrides);
    status = DftiSetValue(*yhandlep, DFTI_INPUT_DISTANCE, 1);
    status = DftiSetValue(*yhandlep, DFTI_OUTPUT_DISTANCE, 1);
    status = DftiSetValue(*yhandlep, DFTI_NUMBER_OF_TRANSFORMS, ystrides[1]);
    status = DftiSetValue(*yhandlep, DFTI_PLACEMENT, DFTI_INPLACE);
    if (!transpose) {
        status = DftiSetValue(*yhandlep, DFTI_THREAD_LIMIT, 1);
    }
    cider_fft_check_status(status);
    status = DftiCommitDescriptor(*yhandlep);
    cider_fft_check_status(status);

    status = DftiCreateDescriptor(xhandlep, DFTI_DOUBLE, DFTI_COMPLEX, 1, nx);
    cider_fft_check_status(status);
    status = DftiSetValue(*xhandlep, DFTI_INPUT_STRIDES, xstrides);
    status = DftiSetValue(*xhandlep, DFTI_OUTPUT_STRIDES, xstrides);
    status = DftiSetValue(*xhandlep, DFTI_INPUT_DISTANCE, 1);
    status = DftiSetValue(*xhandlep, DFTI_OUTPUT_DISTANCE, 1);
    status = DftiSetValue(*xhandlep, DFTI_NUMBER_OF_TRANSFORMS, xstrides[1]);
    status = DftiSetValue(*xhandlep, DFTI_PLACEMENT, DFTI_INPLACE);
    cider_fft_check_status(status);
    status = DftiCommitDescriptor(*xhandlep);
    cider_fft_check_status(status);
}

void cider_fft_init_mkl_handle(fft_plan_t *plan) {
    MKL_LONG status;
    int ndim = plan->ndim;
    MKL_LONG ldims[ndim];
    plan->handle = NULL;
    for (int i = 0; i < ndim; i++) {
        ldims[i] = plan->dims[i];
    }
    MKL_LONG rstrides[ndim + 1];
    MKL_LONG cstrides[ndim + 1];
    rstrides[ndim] = plan->stride;
    cstrides[ndim] = plan->stride;
    rstrides[0] = 0;
    cstrides[0] = 0;
    if (plan->r2c) {
        if (ndim > 1) {
            status = DftiCreateDescriptor(&(plan->handle), DFTI_DOUBLE,
                                          DFTI_REAL, ndim, ldims);
            cider_fft_check_status(status);
            cstrides[ndim - 1] = (ldims[ndim - 1] / 2 + 1) * cstrides[ndim];
            if (plan->inplace) {
                rstrides[ndim - 1] = 2 * cstrides[ndim - 1];
            } else {
                rstrides[ndim - 1] = ldims[ndim - 1] * plan->stride;
            }
            for (int d = ndim - 2; d > 0; d--) {
                rstrides[d] = rstrides[d + 1] * ldims[d];
                cstrides[d] = cstrides[d + 1] * ldims[d];
            }
        } else {
            status = DftiCreateDescriptor(&(plan->handle), DFTI_DOUBLE,
                                          DFTI_REAL, ndim, ldims[0]);
            cider_fft_check_status(status);
        }
        status = DftiSetValue(plan->handle, DFTI_CONJUGATE_EVEN_STORAGE,
                              DFTI_COMPLEX_COMPLEX);
        cider_fft_check_status(status);
        status =
            DftiSetValue(plan->handle, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT);
        cider_fft_check_status(status);
    } else {
        if (ndim == 1) {
            status = DftiCreateDescriptor(&(plan->handle), DFTI_DOUBLE,
                                          DFTI_COMPLEX, ndim, ldims[0]);
        } else {
            status = DftiCreateDescriptor(&(plan->handle), DFTI_DOUBLE,
                                          DFTI_COMPLEX, ndim, ldims);
        }
        cider_fft_check_status(status);
        for (int d = ndim - 1; d > 0; d--) {
            rstrides[d] = rstrides[d + 1] * ldims[d];
            cstrides[d] = cstrides[d + 1] * ldims[d];
        }
    }
    if (plan->fwd) {
        status = DftiSetValue(plan->handle, DFTI_INPUT_STRIDES, rstrides);
        cider_fft_check_status(status);
        status = DftiSetValue(plan->handle, DFTI_OUTPUT_STRIDES, cstrides);
        cider_fft_check_status(status);
    } else {
        status = DftiSetValue(plan->handle, DFTI_INPUT_STRIDES, cstrides);
        cider_fft_check_status(status);
        status = DftiSetValue(plan->handle, DFTI_OUTPUT_STRIDES, rstrides);
        cider_fft_check_status(status);
    }
    status = DftiSetValue(plan->handle, DFTI_NUMBER_OF_TRANSFORMS,
                          (MKL_LONG)plan->ntransform);
    cider_fft_check_status(status);
    status =
        DftiSetValue(plan->handle, DFTI_INPUT_DISTANCE, (MKL_LONG)plan->idist);
    status =
        DftiSetValue(plan->handle, DFTI_OUTPUT_DISTANCE, (MKL_LONG)plan->odist);
    if (plan->inplace) {
        status = DftiSetValue(plan->handle, DFTI_PLACEMENT, DFTI_INPLACE);
    } else {
        status = DftiSetValue(plan->handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    }
    status = DftiSetValue(plan->handle, DFTI_NUMBER_OF_USER_THREADS,
                          mkl_get_max_threads());
    status = DftiCommitDescriptor(plan->handle);
    cider_fft_check_status(status);
}
#endif

/** Allocate an n-dimensional fft plan.
 * This function first calls cider_fft_initialize().
 * For any of fwd, r2c, inplace, and batch_first, the parameter is essentially
 * a boolean, and any number !=0 is treated as 1/true.
 * \param ndim is the number of dimensions in the transform (e.g. 3 for 3D FFT).
 * \param dims is length-ndim integer array with the dimensions of the FFT.
 * \param fwd is 1 for a forward transform and 0 for a backward transform.
 * \param r2c is 1 for a real-to-complex transform and 0 for a
 *     complex-to-complex transform.
 * \param ntransform is the number of transforms to perform at once.
 * \param inplace is 1 for an in-place transform and 0 for out-of-place.
 * \param batch_first is 1 if the batch index (i.e. the index of the
 *     individual transform if ntransform>1) is the first index
 *     in row-major format, i.e. the outermost and slowest index.
 *     batch_first is 0 if the batch index is the last index in
 *     row-major format, i.e. the innermost and fastest index.
 * \return a pointer to an fft_plan_t object, set up with the
 *     requested parameters described above.
 */
fft_plan_t *allocate_fftnd_plan(int ndim, int *dims, int fwd, int r2c,
                                int ntransform, int inplace, int batch_first) {
    cider_fft_initialize();
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

#if FFT_BACKEND == FFT_MKL_BACKEND
    if (plan->ndim == 3 && !plan->batch_first && plan->inplace) {
        cider_fft_init_fft3d_1d_parts(
            plan->ntransform, dims[0], dims[1], dims[2], r2c, 0, plan->inplace,
            &(plan->xhandle), &(plan->yhandle), &(plan->zhandle));
        /*if (!inplace) {
            MKL_LONG rstrides[2] = {0, }
            if (fwd) {

            }
        }*/
    } else {
        cider_fft_init_mkl_handle(plan);
    }
#else
    plan->plan = NULL;
#endif

    return plan;
}

#if FFT_BACKEND == FFT_FFTW_BACKEND
static void initialize_fftw_settings(fft_plan_t *plan) {
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
        plan->out = in_array;
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

#if FFT_BACKEND == FFT_MKL_BACKEND
void execute_mkl_fft3d_fwd(fft_plan_t *plan) {
    double *work;
    double *out;
    size_t nzt_r = plan->dims[2] * plan->ntransform;
    size_t nzt_k;
    if (plan->r2c) {
        nzt_k = 2 * (plan->dims[2] / 2 + 1) * plan->ntransform;
    } else {
        nzt_r *= 2;
        nzt_k = nzt_r;
    }
    if (plan->inplace) {
        nzt_r = nzt_k;
    }
    const int ny = plan->dims[1];
    const int nx = plan->dims[0];
#pragma omp parallel for
    for (int ix = 0; ix < nx; ix++) {
        // if (plan->inplace) {
        for (int iy = 0; iy < ny; iy++) {
            work = (double *)plan->in;
            work = work + (ix * ny + iy) * nzt_r;
            DftiComputeForward(plan->zhandle, work);
        }
        //} else {
        //    for (int iy = 0; iy < ny; iy++) {
        //        work = (double *)plan->in;
        //        work = work + (ix * ny + iy) * nzt_r;
        //        out = (double *)plan->out;
        //        out = work + (ix * ny + iy) * nzt_k;
        //        DftiComputeForward(plan->zhandle, work, out);
        //    }
        //}
        work = (double *)plan->out;
        work = work + ix * ny * nzt_k;
        DftiComputeForward(plan->yhandle, work);
    }
    DftiComputeForward(plan->xhandle, plan->out);
}

void execute_mkl_fft3d_bwd(fft_plan_t *plan) {
    double *work;
    size_t nzt_r = plan->dims[2] * plan->ntransform;
    size_t nzt_k;
    if (plan->r2c) {
        nzt_k = 2 * (plan->dims[2] / 2 + 1) * plan->ntransform;
    } else {
        nzt_r *= 2;
        nzt_k = nzt_r;
    }
    const int ny = plan->dims[1];
    const int nx = plan->dims[0];
    DftiComputeBackward(plan->xhandle, plan->in);
#pragma omp parallel for
    for (int ix = 0; ix < nx; ix++) {
        work = (double *)plan->in;
        work = work + ix * ny * nzt_k;
        DftiComputeBackward(plan->yhandle, work);
        for (int iy = 0; iy < ny; iy++) {
            work = (double *)plan->in;
            work = work + (ix * ny + iy) * nzt_k;
            DftiComputeBackward(plan->zhandle, work);
        }
    }
}
#endif

void execute_fft_plan(fft_plan_t *plan) {
#if FFT_BACKEND == FFT_MKL_BACKEND
    if (plan->ndim == 3 && !plan->batch_first && plan->inplace) {
        if (plan->fwd) {
            execute_mkl_fft3d_fwd(plan);
        } else {
            execute_mkl_fft3d_bwd(plan);
        }
    } else {
        int nthread = mkl_get_max_threads();
        mkl_set_num_threads(cider_fft_get_num_mkl_threads());
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
        mkl_set_num_threads(nthread);
    }
#else
    fftw_execute(plan->plan);
#endif
}

void free_fft_plan(fft_plan_t *plan) {
    free(plan->dims);
#if FFT_BACKEND == FFT_MKL_BACKEND
    if (plan->ndim == 3 && !plan->batch_first && plan->inplace) {
        DftiFreeDescriptor(&(plan->xhandle));
        DftiFreeDescriptor(&(plan->yhandle));
        DftiFreeDescriptor(&(plan->zhandle));
    } else {
        DftiFreeDescriptor(&(plan->handle));
    }
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
    return alloc_fft_array(objsize);
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
    return alloc_fft_array(objsize);
}

void write_fft_input(fft_plan_t *plan, void *input) {
    const size_t size = plan->ntransform * plan->fft_in_size;
    if (plan->r2c && plan->fwd) {
        double *src = (double *)input;
        double *dst = (double *)plan->in;
        if (plan->inplace) {
            size_t nt = plan->batch_first ? 1 : plan->ntransform;
            size_t dm1 = plan->dims[plan->ndim - 1];
            const size_t last_dim = dm1 * nt;
            const size_t last_dim1 = 2 * (dm1 / 2 + 1) * nt;
            const size_t blksize = size / last_dim1;
#pragma omp parallel for
            for (size_t i = 0; i < blksize; i++) {
                for (size_t j = 0; j < last_dim; j++) {
                    dst[i * last_dim1 + j] = src[i * last_dim + j];
                }
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < size; i++) {
                dst[i] = src[i];
            }
        }
    } else {
        double complex *src = (double complex *)input;
        double complex *dst = (double complex *)plan->in;
#pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    }
}

void read_fft_output(fft_plan_t *plan, void *output) {
    const size_t size = plan->ntransform * plan->fft_out_size;
    if (plan->r2c && (!plan->fwd)) {
        double *src = (double *)plan->out;
        double *dst = (double *)output;
        if (plan->inplace) {
            size_t nt = plan->batch_first ? 1 : plan->ntransform;
            size_t dm1 = plan->dims[plan->ndim - 1];
            const size_t last_dim = dm1 * nt;
            const size_t last_dim1 = 2 * (dm1 / 2 + 1) * nt;
            const size_t blksize = size / last_dim1;
#pragma omp parallel for
            for (size_t i = 0; i < blksize; i++) {
                for (size_t j = 0; j < last_dim; j++) {
                    dst[i * last_dim + j] = src[i * last_dim1 + j];
                }
            }
        } else {
#pragma omp parallel for
            for (size_t i = 0; i < size; i++) {
                dst[i] = src[i];
            }
        }
    } else {
        double complex *src = (double complex *)plan->out;
        double complex *dst = (double complex *)output;
#pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    }
}

void *alloc_fft_array(size_t objsize) {
#if FFT_BACKEND == FFT_MKL_BACKEND
    return mkl_malloc(objsize, 64);
#else
    return fftw_malloc(objsize);
#endif
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
