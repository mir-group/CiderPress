#include "cider_mpi_fft.h"
#include "cider_fft.h"
#include <omp.h>

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

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

mpi_fft3d_plan_t *allocate_mpi_fft3d_plan(MPI_Comm comm, const int *dims,
                                          int r2c, int ntransform) {
    cider_fft_initialize();
#if FFT_BACKEND == FFT_MKL_BACKEND
#else
    // int provided;
    // MPI_Query_thread(&provided);
    // int threads_ok = provided >= MPI_THREAD_FUNNELED;
    // if (threads_ok) threads_ok = fftw_init_threads();
    // fftw_mpi_init();
#endif
    mpi_fft3d_plan_t *plan =
        (mpi_fft3d_plan_t *)malloc(sizeof(mpi_fft3d_plan_t));
    plan->comm = comm;
    plan->r_Nglobal[0] = dims[0];
    plan->r_Nglobal[1] = dims[1];
    plan->r_Nglobal[2] = dims[2];
    plan->r_Nlocal[1] = dims[1];
    plan->r_Nlocal[2] = dims[2];
    plan->r_offset[1] = 0;
    plan->r_offset[2] = 0;
    plan->k_Nglobal[0] = dims[0];
    plan->k_Nglobal[1] = dims[1];
    if (r2c) {
        plan->k_Nglobal[2] = dims[2] / 2 + 1;
    } else {
        plan->k_Nglobal[2] = dims[2];
    }
    plan->k_Nlocal[0] = plan->k_Nglobal[0];
    plan->k_Nlocal[2] = plan->k_Nglobal[2];
    plan->k_offset[0] = 0;
    plan->k_offset[2] = 0;
    plan->r2c = r2c;
    plan->ntransform = ntransform;
    plan->work = NULL;

#if FFT_BACKEND == FFT_MKL_BACKEND
    // MKL_LONG status;
    plan->xhandle = NULL;
    plan->yhandle = NULL;
    plan->zhandle = NULL;
    cider_fft_init_fft3d_1d_parts(plan->ntransform, plan->r_Nglobal[0],
                                  plan->r_Nglobal[1], plan->r_Nglobal[2],
                                  plan->r2c, 1, 1, &(plan->xhandle),
                                  &(plan->yhandle), &(plan->zhandle));

    int comm_size;
    MPI_Comm_size(plan->comm, &comm_size);
    int *npar_x = (int *)malloc((comm_size + 1) * sizeof(int));
    int xpp = plan->r_Nglobal[0] / comm_size;
    if (plan->r_Nglobal[0] % comm_size) {
        xpp++;
    }
    for (int i = 0; i <= comm_size; i++) {
        npar_x[i] = MIN(i * xpp, plan->r_Nglobal[0]);
    }
    int *npar_y = (int *)malloc((comm_size + 1) * sizeof(int));
    int ypp = plan->r_Nglobal[1] / comm_size;
    if (plan->r_Nglobal[1] % comm_size) {
        ypp++;
    }
    for (int i = 0; i <= comm_size; i++) {
        npar_y[i] = MIN(i * ypp, plan->r_Nglobal[1]);
    }
    int my_rank;
    MPI_Comm_rank(plan->comm, &my_rank);
    int my_nx = npar_x[my_rank + 1] - npar_x[my_rank];
    int my_ny = npar_y[my_rank + 1] - npar_y[my_rank];
    size_t size1 = xpp * plan->k_Nglobal[1] * plan->k_Nglobal[2];
    size_t size2 = ypp * plan->k_Nglobal[0] * plan->k_Nglobal[2];
    plan->xpp = xpp;
    plan->ypp = ypp;
    plan->r_Nlocal[0] = my_nx;
    plan->r_offset[0] = npar_x[my_rank];
    plan->k_Nlocal[1] = my_ny;
    plan->k_offset[1] = npar_y[my_rank];

    plan->work_array_size = MAX(size1, size2) * plan->ntransform;
    plan->work = mkl_malloc(plan->work_array_size * sizeof(double complex), 64);
#else
    ptrdiff_t local_size_dims[3];
    local_size_dims[0] = plan->k_Nglobal[0];
    local_size_dims[1] = plan->k_Nglobal[1];
    local_size_dims[2] = plan->k_Nglobal[2];
    ptrdiff_t fftw_alloc_size;
    ptrdiff_t tmp_xsize, tmp_xoff, tmp_ysize, tmp_yoff;
    fftw_alloc_size = fftw_mpi_local_size_many_transposed(
        3, local_size_dims, plan->ntransform, FFTW_MPI_DEFAULT_BLOCK,
        FFTW_MPI_DEFAULT_BLOCK, plan->comm, &tmp_xsize, &tmp_xoff, &tmp_ysize,
        &tmp_yoff);
    plan->r_Nlocal[0] = tmp_xsize;
    plan->r_offset[0] = tmp_xoff;
    plan->k_Nlocal[1] = tmp_ysize;
    plan->k_offset[1] = tmp_yoff;
    plan->work_array_size = fftw_alloc_size;
    plan->work = fftw_alloc_complex(fftw_alloc_size);
    ptrdiff_t plan_dims[3] = {dims[0], dims[1], dims[2]};
    if (r2c) {
        plan->fwd_plan = fftw_mpi_plan_many_dft_r2c(
            3, plan_dims, plan->ntransform, FFTW_MPI_DEFAULT_BLOCK,
            FFTW_MPI_DEFAULT_BLOCK, (double *)plan->work, plan->work,
            plan->comm, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT);
        plan->bwd_plan = fftw_mpi_plan_many_dft_c2r(
            3, plan_dims, plan->ntransform, FFTW_MPI_DEFAULT_BLOCK,
            FFTW_MPI_DEFAULT_BLOCK, plan->work, (double *)plan->work,
            plan->comm, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
    } else {
        plan->fwd_plan = fftw_mpi_plan_many_dft(
            3, plan_dims, plan->ntransform, FFTW_MPI_DEFAULT_BLOCK,
            FFTW_MPI_DEFAULT_BLOCK, plan->work, plan->work, plan->comm,
            FFTW_FORWARD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_OUT);
        plan->bwd_plan = fftw_mpi_plan_many_dft(
            3, plan_dims, plan->ntransform, FFTW_MPI_DEFAULT_BLOCK,
            FFTW_MPI_DEFAULT_BLOCK, plan->work, plan->work, plan->comm,
            FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_MPI_TRANSPOSED_IN);
    }
#endif

    return plan;
}

void transpose_data_helper(MPI_Comm comm, int nx, int xpp, int ypp,
                           double complex *sendbuf, double complex *recvbuf,
                           int blocksize) {
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    int count = xpp * ypp * blocksize;
    MPI_Alltoall(sendbuf, count, MPI_C_DOUBLE_COMPLEX, recvbuf, count,
                 MPI_C_DOUBLE_COMPLEX, comm);
    double complex *out;
    double complex *inp;
    for (int iy = 0; iy < ypp; iy++) {
        for (int ib = 0; ib < comm_size; ib++) {
            out = sendbuf + (iy * nx + ib * xpp) * blocksize;
            inp = recvbuf + (ib * ypp * xpp + iy * xpp) * blocksize;
            for (int ix = 0; ix < xpp * blocksize; ix++) {
                out[ix] = inp[ix];
            }
        }
    }
}

// NOTE would be nice to parallelize this, but it would require
// a higher level of threading support in MPI that FFTW3 needs.
void transpose_data_loop(MPI_Comm comm, const int nx, const int ny,
                         const int nz, const int xpp, const int ypp,
                         double complex *work) {
    const int blksize = MIN(8, nz);
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    size_t bufsize = comm_size * xpp * ypp * blksize;
    bufsize *= sizeof(double complex);
    double complex *sendbuf = (double complex *)malloc(bufsize);
    double complex *recvbuf = (double complex *)malloc(bufsize);
    const int nblk = (nz + blksize - 1) / blksize;
    int z0, dz;
    size_t ind;
    size_t ind2;
    for (int ib = 0; ib < nblk; ib++) {
        z0 = ib * blksize;
        dz = MIN(z0 + blksize, nz) - z0;
        for (int ix = 0; ix < xpp; ix++) {
            for (int iy = 0; iy < ny; iy++) {
                ind = ix * ny + iy;
                ind2 = iy * xpp + ix;
                for (int iz = 0; iz < dz; iz++) {
                    sendbuf[ind2 * blksize + iz] = work[ind * nz + z0 + iz];
                }
            }
        }
        transpose_data_helper(comm, nx, xpp, ypp, sendbuf, recvbuf, blksize);
        for (int iy = 0; iy < ypp; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                ind = iy * nx + ix;
                for (int iz = 0; iz < dz; iz++) {
                    work[ind * nz + z0 + iz] = sendbuf[ind * blksize + iz];
                }
            }
        }
    }
    free(sendbuf);
    free(recvbuf);
}

void transpose_fft3d_work_fwd(mpi_fft3d_plan_t *plan) {
    transpose_data_loop(plan->comm, plan->k_Nglobal[0], plan->k_Nglobal[1],
                        plan->ntransform * plan->k_Nglobal[2], plan->xpp,
                        plan->ypp, plan->work);
}

void transpose_fft3d_work_bwd(mpi_fft3d_plan_t *plan) {
    transpose_data_loop(plan->comm, plan->k_Nglobal[1], plan->k_Nglobal[0],
                        plan->ntransform * plan->k_Nglobal[2], plan->ypp,
                        plan->xpp, plan->work);
}

void transpose_data_loop_world(const int nx, const int ny, const int nz,
                               const int xpp, const int ypp,
                               double complex *work) {
    transpose_data_loop(MPI_COMM_WORLD, nx, ny, nz, xpp, ypp, work);
}

void execute_mpi_fft3d_fwd(mpi_fft3d_plan_t *plan) {
#if FFT_BACKEND == FFT_MKL_BACKEND
    int nthread = mkl_get_max_threads();
    mkl_set_num_threads(cider_fft_get_num_mkl_threads());
    double complex *work;
    size_t nzt = plan->k_Nglobal[2] * plan->ntransform;
    const int nxloc = plan->r_Nlocal[0];
    const int nyglob = plan->r_Nlocal[1];
    const int nyloc = plan->k_Nlocal[1];
    for (int ix = 0; ix < nxloc; ix++) {
#pragma omp parallel for
        for (int iy = 0; iy < nyglob; iy++) {
            work = (double complex *)plan->work;
            work = work + (ix * plan->r_Nlocal[1] + iy) * nzt;
            DftiComputeForward(plan->zhandle, work);
        }
        work = (double complex *)plan->work;
        work = work + ix * plan->r_Nlocal[1] * nzt;
        DftiComputeForward(plan->yhandle, work);
    }
    transpose_fft3d_work_fwd(plan);
    for (int iy = 0; iy < nyloc; iy++) {
        work = (double complex *)plan->work;
        work = work + iy * plan->k_Nlocal[0] * nzt;
        DftiComputeForward(plan->xhandle, work);
    }
    mkl_set_num_threads(nthread);
#else
    fftw_execute(plan->fwd_plan);
#endif
}

void execute_mpi_fft3d_bwd(mpi_fft3d_plan_t *plan) {
#if FFT_BACKEND == FFT_MKL_BACKEND
    double complex *work;
    size_t nzt = plan->k_Nglobal[2] * plan->ntransform;
    const int nxloc = plan->r_Nlocal[0];
    const int nyglob = plan->r_Nlocal[1];
    const int nyloc = plan->k_Nlocal[1];
    for (int iy = 0; iy < nyloc; iy++) {
        work = (double complex *)plan->work;
        work = work + iy * plan->k_Nlocal[0] * nzt;
        DftiComputeBackward(plan->xhandle, work);
    }
    transpose_fft3d_work_bwd(plan);
    for (int ix = 0; ix < nxloc; ix++) {
        work = (double complex *)plan->work;
        work = work + ix * plan->r_Nlocal[1] * nzt;
        DftiComputeBackward(plan->yhandle, work);
#pragma omp parallel for
        for (int iy = 0; iy < nyglob; iy++) {
            work = (double complex *)plan->work;
            work = work + (ix * plan->r_Nlocal[1] + iy) * nzt;
            DftiComputeBackward(plan->zhandle, work);
        }
    }
#else
    fftw_execute(plan->bwd_plan);
#endif
}

mpi_fft3d_plan_t *allocate_mpi_fft3d_plan_world(const int *dims, int r2c,
                                                int ntransform) {
    return allocate_mpi_fft3d_plan(MPI_COMM_WORLD, dims, r2c, ntransform);
}

void cider_fft_world_size_and_rank(int *size, int *rank) {
    MPI_Comm_size(MPI_COMM_WORLD, size);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
}

void free_mpi_fft3d_plan(mpi_fft3d_plan_t *plan) {
#if FFT_BACKEND == FFT_MKL_BACKEND
    DftiFreeDescriptor(&(plan->xhandle));
    DftiFreeDescriptor(&(plan->yhandle));
    DftiFreeDescriptor(&(plan->zhandle));
    mkl_free(plan->work);
#else
    fftw_destroy_plan(plan->fwd_plan);
    fftw_destroy_plan(plan->bwd_plan);
    fftw_free(plan->work);
#endif
    free(plan);
}

void write_mpi_fft3d_input(mpi_fft3d_plan_t *plan, void *input, int fwd) {
    if (plan->r2c && fwd) {
        double *src = (double *)input;
        double *dst = (double *)plan->work;
        size_t nt = plan->ntransform;
        size_t dm1 = plan->r_Nlocal[2];
        const size_t last_dim = dm1 * nt;
        const size_t last_dim1 = 2 * (dm1 / 2 + 1) * nt;
        const size_t blksize = plan->r_Nlocal[0] * plan->r_Nlocal[1];
#pragma omp parallel for
        for (size_t i = 0; i < blksize; i++) {
            for (size_t j = 0; j < last_dim; j++) {
                dst[i * last_dim1 + j] = src[i * last_dim + j];
            }
        }
    } else if (fwd) {
        const size_t size = plan->ntransform * plan->r_Nlocal[0] *
                            plan->r_Nlocal[1] * plan->r_Nlocal[2];
        double complex *src = (double complex *)input;
        double complex *dst = (double complex *)plan->work;
#pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    } else {
        const size_t size = plan->ntransform * plan->k_Nlocal[0] *
                            plan->k_Nlocal[1] * plan->k_Nlocal[2];
        double complex *src = (double complex *)input;
        double complex *dst = (double complex *)plan->work;
#pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    }
}

void read_mpi_fft3d_output(mpi_fft3d_plan_t *plan, void *output, int fwd) {
    if (plan->r2c && !fwd) {
        double *src = (double *)plan->work;
        double *dst = (double *)output;
        size_t nt = plan->ntransform;
        size_t dm1 = plan->r_Nlocal[2];
        const size_t last_dim = dm1 * nt;
        const size_t last_dim1 = 2 * (dm1 / 2 + 1) * nt;
        const size_t blksize = plan->r_Nlocal[0] * plan->r_Nlocal[1];
#pragma omp parallel for
        for (size_t i = 0; i < blksize; i++) {
            for (size_t j = 0; j < last_dim; j++) {
                dst[i * last_dim + j] = src[i * last_dim1 + j];
            }
        }
    } else if (fwd) {
        const size_t size = plan->ntransform * plan->k_Nlocal[0] *
                            plan->k_Nlocal[1] * plan->k_Nlocal[2];
        double complex *src = (double complex *)plan->work;
        double complex *dst = (double complex *)output;
#pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    } else {
        const size_t size = plan->ntransform * plan->r_Nlocal[0] *
                            plan->r_Nlocal[1] * plan->r_Nlocal[2];
        double complex *src = (double complex *)plan->work;
        double complex *dst = (double complex *)output;
#pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    }
}
