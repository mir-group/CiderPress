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

#include <complex.h>
#include <math.h>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <pyscf_gto.h>
#include <stdio.h>
#include <stdlib.h>

#define M_PI 3.14159265358979323846

void apply_orb_phases(double complex *ao, int *atom_list, int *ang_list,
                      double *gcoords, double *atom_coords, int natm, int nao,
                      int ngrids) {
#pragma omp parallel
    {
        int g, i, b;
        int blksize = 128;
        int bgrids, blk;
        int nblk = (ngrids + blksize - 1) / blksize;
        int ib, ia;
        double complex *ao_tmp;
        double *gx, *gy, *gz;
        double *acoord;
        double kdotr;
        double complex fac;
        double complex *emikr =
            (double complex *)malloc(natm * blksize * sizeof(double complex));
#pragma omp for
        for (ib = 0; ib < nblk; ib++) {
            bgrids = MIN(blksize, ngrids - ib * blksize);
            gx = gcoords + 0 * ngrids + ib * blksize;
            gy = gcoords + 1 * ngrids + ib * blksize;
            gz = gcoords + 2 * ngrids + ib * blksize;
            for (ia = 0; ia < natm; ia++) {
                acoord = atom_coords + 3 * ia;
                for (g = 0; g < bgrids; g++) {
                    kdotr = gx[g] * acoord[0] + gy[g] * acoord[1] +
                            gz[g] * acoord[2];
                    emikr[ia * bgrids + g] = cexp(-I * kdotr);
                }
            }
            for (i = 0; i < nao; i++) {
                ao_tmp = ao + i * ngrids + ib * blksize;
                ia = atom_list[i];
                fac = cexp(-I * ang_list[i] * 0.5 * M_PI);
                for (g = 0; g < bgrids; g++) {
                    // ao_tmp[g] *= emikr[ia * bgrids + g];
                    ao_tmp[g] *= emikr[ia * bgrids + g] * fac;
                }
            }
        }
        free(emikr);
    }
}

void CHECK_STATUS(int status) {
    if (status != 0) {
        printf("FFT ROUTINE FAILED WITH STATUS %d:\n", status);
        char *message = DftiErrorMessage(status);
        printf("%s\n", message);
        exit(-1);
    }
}

void parallel_mul_add_d(double *a, double *b, double *c, int dim1, int dim2) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            c[i * dim2 + j] += a[i * dim2 + j] * b[j];
        }
    }
}

void parallel_mul_add_z(double complex *a, double complex *b, double complex *c,
                        int dim1, int dim2) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            c[i * dim2 + j] += a[i * dim2 + j] * b[j];
        }
    }
}

void parallel_mul_z(double complex *a, double complex *b, double complex *c,
                    int dim1, int dim2) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            c[i * dim2 + j] = a[i * dim2 + j] * b[j];
        }
    }
}

void parallel_mul_dz(double complex *a, double *b, double complex *c, int dim1,
                     int dim2) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            c[i * dim2 + j] = a[i * dim2 + j] * b[j];
        }
    }
}

void fast_conj(double complex *a, size_t size) {
#pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        a[i] = conj(a[i]);
    }
}

void fft3d(double complex *xin, double complex *xout, int *fftg,
           DFTI_DESCRIPTOR_HANDLE handle, int fwd) {
    MKL_LONG status = 0;
    int inplace = (xout == NULL);
    if (fwd) {
        if (inplace)
            status = DftiComputeForward(handle, xin);
        else
            status = DftiComputeForward(handle, xin, xout);
    } else {
        if (inplace)
            status = DftiComputeBackward(handle, xin);
        else
            status = DftiComputeBackward(handle, xin, xout);
    }
    CHECK_STATUS(status);
}

void prune_r2c_real(double *xreal, int *fftg, int num_fft) {
    size_t real_stride = 2 * (fftg[2] / 2 + 1);
    size_t num_stride = fftg[0] * fftg[1] * num_fft;
#pragma omp parallel for
    for (size_t xy = 0; xy < num_stride; xy++) {
        xreal[xy * real_stride + fftg[2]] = 0;
    }
    if (fftg[2] % 2 == 0) {
#pragma omp parallel for
        for (size_t xy = 0; xy < num_stride; xy++) {
            xreal[xy * real_stride + fftg[2] + 1] = 0;
        }
    }
}

void prune_r2c_complex(double complex *x, int *fftg, int num_fft) {
    size_t stride = fftg[2] / 2 + 1;
    size_t num_stride = fftg[0] * fftg[1] * num_fft;
#pragma omp parallel for
    for (size_t xy = 0; xy < num_stride; xy++) { // 0 component should be real
        // x[xy * stride] = creal(x[xy * stride]);
    }
    if (fftg[2] % 2 == 0) { // N/2 component should be real for even mesh
#pragma omp parallel for
        for (size_t xy = 0; xy < num_stride; xy++) {
            x[xy * stride + stride - 1] = creal(x[xy * stride + stride - 1]);
        }
    }
}

void run_ffts(double complex *xin_list, double complex *xout_list, double scale,
              int *fftg, int fwd, int num_fft, int mklpar, int r2c) {
    MKL_LONG status = 0;
    DFTI_DESCRIPTOR_HANDLE handle = 0;
    MKL_LONG dim = 3;
    MKL_LONG length[3] = {fftg[0], fftg[1], fftg[2]};
    MKL_LONG real_distance, recip_distance;
    int nthread = mkl_get_max_threads();
    mkl_set_num_threads(nthread);
    if (r2c) {
        status =
            DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_REAL, dim, length);
        MKL_LONG sy = fftg[2] / 2 + 1;
        MKL_LONG sx = fftg[1] * sy;
        recip_distance = fftg[0] * fftg[1] * sy;
        real_distance = recip_distance * 2;
        MKL_LONG rstrides[] = {0, 2 * sx, 2 * sy, 1};
        MKL_LONG cstrides[] = {0, sx, sy, 1};
        if (xout_list != NULL) {
            rstrides[1] = fftg[1] * fftg[2];
            rstrides[2] = fftg[2];
            real_distance = fftg[0] * fftg[1] * fftg[2];
        }
        if (fwd) {
            status = DftiSetValue(handle, DFTI_INPUT_STRIDES, rstrides);
            status = DftiSetValue(handle, DFTI_OUTPUT_STRIDES, cstrides);
        } else {
            status = DftiSetValue(handle, DFTI_INPUT_STRIDES, cstrides);
            status = DftiSetValue(handle, DFTI_OUTPUT_STRIDES, rstrides);
        }
        DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(handle, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT);
        if (fwd && xout_list == NULL) {
            prune_r2c_real((double *)xin_list, fftg, num_fft);
        } else if (!fwd) {
            // prune_r2c_complex(xin_list, fftg, num_fft);
        }
    } else {
        status = DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_COMPLEX, dim,
                                      length);
        recip_distance = fftg[0] * fftg[1] * fftg[2];
        real_distance = recip_distance;
    }
    CHECK_STATUS(status);
    if (fwd) {
        status = DftiSetValue(handle, DFTI_FORWARD_SCALE, scale);
    } else {
        status = DftiSetValue(handle, DFTI_BACKWARD_SCALE, scale);
    }
    CHECK_STATUS(status);
    int i;
    if (mklpar) {
        status = DftiSetValue(handle, DFTI_NUMBER_OF_TRANSFORMS, num_fft);
        if (fwd) {
            status = DftiSetValue(handle, DFTI_INPUT_DISTANCE, real_distance);
            status = DftiSetValue(handle, DFTI_OUTPUT_DISTANCE, recip_distance);
        } else {
            status = DftiSetValue(handle, DFTI_OUTPUT_DISTANCE, real_distance);
            status = DftiSetValue(handle, DFTI_INPUT_DISTANCE, recip_distance);
        }
        if (xout_list != NULL) {
            status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        } else {
            status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_INPLACE);
        }
        status = DftiCommitDescriptor(handle);
        CHECK_STATUS(status);
        fft3d(xin_list, xout_list, fftg, handle, fwd);
    } else if (xout_list == NULL) {
        status = DftiCommitDescriptor(handle);
        CHECK_STATUS(status);
        for (i = 0; i < num_fft; i++) {
            fft3d(xin_list + i * recip_distance, NULL, fftg, handle, fwd);
        }
    } else {
        status = DftiCommitDescriptor(handle);
        CHECK_STATUS(status);
        for (i = 0; i < num_fft; i++) {
            fft3d(xin_list + i * recip_distance, xout_list + i * recip_distance,
                  fftg, handle, fwd);
        }
    }
    DftiFreeDescriptor(&handle);
    mkl_set_num_threads(nthread);
    if (r2c) {
        if (!fwd && xout_list == NULL) {
            prune_r2c_real((double *)xin_list, fftg, num_fft);
        } else if (fwd && xout_list == NULL) {
            // prune_r2c_complex(xin_list, fftg, num_fft);
        } else if (fwd) {
            // prune_r2c_complex(xout_list, fftg, num_fft);
        }
    }
    return;
    if (r2c &&
        !fwd) { // transformed into real, which gives extra indeterminate values
        double *xreal = (double *)xin_list;
        int real_stride = 2 * (fftg[2] / 2 + 1);
        int num_stride = fftg[0] * fftg[1] * num_fft;
#pragma omp parallel for
        for (int xy = 0; xy < num_stride; xy++) {
            xreal[xy * real_stride + fftg[2]] = 0;
        }
        if (fftg[2] % 2 == 0) {
#pragma omp parallel for
            for (int xy = 0; xy < num_stride; xy++) {
                xreal[xy * real_stride + fftg[2] + 1] = 0;
            }
        }
    }
    //}
}

void weight_symm_gpts(double complex *x, size_t dim1, size_t dz) {
#pragma omp parallel
    {
        size_t i, j;
        size_t dim2 = dz / 2 + 1;
#pragma omp for
        for (i = 0; i < dim1; i++) {
            x[i * dim2] *= 0.5;
        }
        if (dz % 2 == 0) {
#pragma omp for
            for (i = 0; i < dim1; i++) {
                x[i * dim2 + dim2 - 1] *= 0.5;
            }
        }
    }
}

void contract_convolution_z(double complex *p_vag, double complex *conv_ao,
                            double complex *cr, int ncpa, int nao, int ngrids,
                            int nalpha) {
#pragma omp parallel
    {
        int blksize = 128;
        int nblk = (ngrids + blksize - 1) / blksize;
        double complex *_p;
        double complex *_conv;
        double complex *_cr;
        int blk, g0, g1, v, i, g;
#pragma omp for
        for (blk = 0; blk < nblk; blk++) {
            g0 = blk * blksize;
            g1 = MIN(ngrids, g0 + blksize);
            _p = p_vag;
            _cr = cr;
            for (v = 0; v < ncpa; v++) {
                _conv = conv_ao;
                for (i = 0; i < nao; i++) {
                    for (g = g0; g < g1; g++) {
                        _p[g] += creal(_conv[g]) * creal(_cr[g]) +
                                 cimag(_conv[g]) * cimag(_cr[g]);
                    }
                    _conv += ngrids;
                    _cr += ngrids;
                }
                _p += nalpha * ngrids;
            }
        }
    }
}

void contract_convolution_d(double *p_vag, double *conv_ao, double *cr,
                            int ncpa, int nao, int ngrids, int nalpha) {
#pragma omp parallel
    {
        int blksize = 128;
        int nblk = (ngrids + blksize - 1) / blksize;
        double *_p;
        double *_conv;
        double *_cr;
        int blk, g0, g1, v, i, g;
#pragma omp for
        for (blk = 0; blk < nblk; blk++) {
            g0 = blk * blksize;
            g1 = MIN(ngrids, g0 + blksize);
            _p = p_vag;
            _cr = cr;
            for (v = 0; v < ncpa; v++) {
                _conv = conv_ao;
                for (i = 0; i < nao; i++) {
                    for (g = g0; g < g1; g++) {
                        _p[g] += _conv[g] * _cr[g];
                    }
                    _conv += ngrids;
                    _cr += ngrids;
                }
                _p += nalpha * ngrids;
            }
        }
    }
}

void recip_conv_kernel_gaussdiff(double *conv, double *G2, double alpha,
                                 double alpha_norm, int ng) {
#pragma omp parallel
    {
        int g;
        double fac, fac2, expnt;
        fac = sqrt(2.0) / 16;
        fac2 = 4 * M_PI * alpha_norm * sqrt(M_PI) / pow(alpha, 1.5);
        expnt = 0.125 / alpha;
#pragma omp for
        for (g = 0; g < ng; g++) {
            conv[g] = exp(-expnt * G2[g]);
            conv[g] *= (0.25 * conv[g] - fac);
            conv[g] *= fac2;
        }
    }
}

void recip_conv_kernel_ws(double *conv, double *vq, double *Gvec, double *lat,
                          double *maxqv, int *mesh, int ng, int nv) {
#pragma omp parallel
    {
        int g;
        double gx, gy, gz;
        int ix, iy, iz, ii;
        double *gvec;
        double itpi = 1.0 / (2 * M_PI);
        // printf("%lf %lf %lf %lf %lf %lf\n", maxqv[0], maxqv[1], maxqv[2],
        // lat[0], lat[1], lat[2]);
#pragma omp for
        for (g = 0; g < ng; g++) {
            gvec = Gvec + 3 * g;
            if (fabs(gvec[0]) <= maxqv[0] && fabs(gvec[1]) <= maxqv[1] &&
                fabs(gvec[2]) <= maxqv[2]) {
                gx = gvec[0] * lat[0] + gvec[1] * lat[1] + gvec[2] * lat[2];
                gy = gvec[0] * lat[3] + gvec[1] * lat[4] + gvec[2] * lat[5];
                gz = gvec[0] * lat[6] + gvec[1] * lat[7] + gvec[2] * lat[8];
                ix = (int)round(gx * itpi);
                iy = (int)round(gy * itpi);
                iz = (int)round(gz * itpi);
                /*if (ix > mesh[0] / 2 || ix < -(mesh[0] / 2) || iy > mesh[1] /
                2
                    || iy < -(mesh[1] / 2) || iz > mesh[2] / 2 || iz < -(mesh[2]
                / 2)) { printf("%d %d %d %lf %lf %lf\n", ix, iy, iz, gvec[0],
                           gvec[1], gvec[2]);
                    printf("%d %d %d %lf %lf %lf\n", mesh[0], mesh[1], mesh[2],
                           maxqv[0], maxqv[1], maxqv[2]);
                    exit(-1);
                }*/
                ix = (ix % mesh[0] + mesh[0]) % mesh[0];
                iy = (iy % mesh[1] + mesh[1]) % mesh[1];
                iz = (iz % mesh[2] + mesh[2]) % mesh[2];
                ii = (ix * mesh[1] + iy) * mesh[2] + iz;
                /*if (vq[ii] != vq[ii] || gx != gx || gy != gy || gz != gz) {
                    printf("%lf %lf %lf %d %d\n", gx * itpi, gy * itpi, gz *
                itpi, g, ii); printf("%d %d %d %d %d\n", ix, iy, iz, g, ii);
                    exit(-1);
                }*/
                conv[g] = vq[ii];
            } else {
                conv[g] = 0.0;
            }
        }
    }
}

void zero_even_edges_fft(double complex *x, const int num_fft, const int *fftg,
                         const int halfc) {
    // z_size = fftg[2] if halfc is false
    // z_size = (fftg[2] / 2 + 1) if halfc is true
    // yz_size = fftg[1] * z_size
    // xyz_size = fftg[0] * yz_size
    // position is fftid * xyz_size + xid * yz_size + yid * z_size + zid
    // mid[i] = fftg[i] / 2
    // only zero out midpoint if fftg[i] is even
    size_t z_size;
    size_t yz_size;
    size_t xyz_size;
    if (halfc) {
        z_size = fftg[2] / 2 + 1;
    } else {
        z_size = fftg[2];
    }
    yz_size = fftg[1] * z_size;
    xyz_size = fftg[2] * yz_size;
    const int midptx = fftg[0] / 2;
    const int midpty = fftg[1] / 2;
    const int midptz = fftg[2] / 2;
    const size_t nfxy = num_fft * fftg[0] * fftg[1];
    double complex *xshifted;
    if (fftg[0] % 2 == 0) {
        xshifted = x + midptx * yz_size;
#pragma omp parallel for collapse(2)
        for (size_t fftid = 0; fftid < num_fft; fftid++) {
            for (size_t yz = 0; yz < yz_size; yz++) {
                xshifted[fftid * xyz_size + yz] = 0.0;
            }
        }
    }
    if (fftg[1] % 2 == 0) {
        xshifted = x + midpty * z_size;
#pragma omp parallel for collapse(3)
        for (size_t fftid = 0; fftid < num_fft; fftid++) {
            for (size_t xid = 0; xid < fftg[0]; xid++) {
                for (size_t zid = 0; zid < z_size; zid++) {
                    xshifted[fftid * xyz_size + xid * yz_size + zid] = 0.0;
                }
            }
        }
    }
    if (fftg[2] % 2 == 0) {
        xshifted = x + midptz;
#pragma omp parallel for
        for (size_t fxy = 0; fxy < nfxy; fxy++) {
            xshifted[fxy * z_size] = 0.0;
        }
    }
}

void map_between_fft_meshes(double complex *x1, const int *fftg1,
                            double complex *x2, const int *fftg2, double scale,
                            const int halfc, const int num_fft) {
    int size1_z;
    int size2_z;
    if (halfc) {
        size1_z = fftg1[2] / 2 + 1;
        size2_z = fftg2[2] / 2 + 1;
    } else {
        size1_z = fftg1[2];
        size2_z = fftg2[2];
    }
    const int size1[] = {fftg1[0], fftg1[1], size1_z};
    const int size2[] = {fftg2[0], fftg2[1], size2_z};
    const size_t stride1[] = {size1[0] * size1[1] * size1[2],
                              size1[1] * size1[2], size1[2], 1};
    const size_t stride2[] = {size2[0] * size2[1] * size2[2],
                              size2[1] * size2[2], size2[2], 1};
    const int mm[] = {
        MIN(fftg1[0], fftg2[0]),
        MIN(fftg1[1], fftg2[1]),
        MIN(fftg1[2], fftg2[2]),
    };
    int lbx, lby, lbz;
    if (halfc) {
        lbx = -(mm[0] / 2);
        lby = -(mm[1] / 2);
        lbz = 0;
    } else {
        lbx = -(mm[0] / 2);
        lby = -(mm[1] / 2);
        lbz = -(mm[2] / 2);
    }
    const int lb[] = {lbx, lby, lbz};
    const int ub[] = {(mm[0] - 1) / 2, (mm[1] - 1) / 2, (mm[2] - 1) / 2};
#pragma omp parallel for
    for (int i = 0; i < num_fft * size2[0] * size2[1] * size2[2]; i++) {
        x2[i] = 0.0;
    }
    int pix1, piy1, pix2, piy2;
    double complex *x1ptr, *x2ptr;
#pragma omp parallel for collapse(3) private(pix1, piy1, pix2, piy2, x1ptr,    \
                                                 x2ptr)
    for (int i = 0; i < num_fft; i++) {
        for (int ix = lb[0]; ix <= ub[0]; ix++) {
            for (int iy = lb[1]; iy <= ub[1]; iy++) {
                // MIN(X, Y)       ((X) < (Y) ? (X) : (Y))
                pix1 = ((ix < 0) ? (ix + size1[0]) : (ix));
                piy1 = ((iy < 0) ? (iy + size1[1]) : (iy));
                pix2 = ((ix < 0) ? (ix + size2[0]) : (ix));
                piy2 = ((iy < 0) ? (iy + size2[1]) : (iy));
                x1ptr =
                    x1 + i * stride1[0] + pix1 * stride1[1] + piy1 * stride1[2];
                x2ptr =
                    x2 + i * stride2[0] + pix2 * stride2[1] + piy2 * stride2[2];
                for (int iz = 0; iz <= ub[2]; iz++) {
                    x2ptr[iz] = scale * x1ptr[iz];
                }
                for (int iz = lb[2]; iz < 0; iz++) {
                    x2ptr[stride2[2] + iz] = scale * x1ptr[stride1[2] + iz];
                }
            }
        }
    }
}

void test_fft3d(double *xr, double complex *xk, int nx, int ny, int nz,
                int fwd) {
    // taken from MKL examples:
    // https://www.smcm.iqfr.csic.es/docs/intel/mkl/mkl_manual/appendices/mkl_appC_DFT.htm
    // float x[32][100][19];
    // float _Complex y[32][100][10]; /* 10 = 19/2 + 1 */
    DFTI_DESCRIPTOR_HANDLE my_desc_handle;
    MKL_LONG status, l[3];
    MKL_LONG strides_out[4];
    int nzc = nz / 2 + 1;

    //...put input data into x[j][k][s] 0<=j<=31, 0<=k<=99, 0<=s<=18
    l[0] = nx;
    l[1] = ny;
    l[2] = nz;

    strides_out[0] = 0;
    strides_out[1] = ny * nzc;
    strides_out[2] = nzc;
    strides_out[3] = 1;

    status =
        DftiCreateDescriptor(&my_desc_handle, DFTI_DOUBLE, DFTI_REAL, 3, l);
    status = DftiSetValue(my_desc_handle, DFTI_CONJUGATE_EVEN_STORAGE,
                          DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (fwd) {
        status = DftiSetValue(my_desc_handle, DFTI_OUTPUT_STRIDES, strides_out);
    } else {
        status = DftiSetValue(my_desc_handle, DFTI_INPUT_STRIDES, strides_out);
    }
    status = DftiSetValue(my_desc_handle, DFTI_BACKWARD_SCALE, 1.0);
    status = DftiSetValue(my_desc_handle, DFTI_FORWARD_SCALE, 1.0);

    status = DftiCommitDescriptor(my_desc_handle);
    if (fwd) {
        status = DftiComputeForward(my_desc_handle, xr, xk);
    } else {
        status = DftiComputeBackward(my_desc_handle, xk, xr);
    }
    status = DftiFreeDescriptor(&my_desc_handle);
    /* result is the complex value z(j,k,s) 0<=j<=31; 0<=k<=99, 0<=s<=9
    and is stored in complex matrix y in CCE format. */
}
