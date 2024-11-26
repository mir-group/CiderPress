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

#include "fast_sdmx.h"
#include "pyscf_gto.h"
#include "sph_harm.h"
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int SDMXcontract_smooth0(double *ectr, double *coord, double *alpha,
                         double *coeff, int l, int nprim, int nctr,
                         size_t ngrids, double fac, double conv_alpha,
                         double conv_alpha_coeff) {
    size_t i, j, k;
    double arr1, arr2, eprim;
    double rr[BLKSIZE];
    // exp(-alpha * r^2) - exp(-2 * alpha * r^2)
    double conv_exp1, conv_exp2, conv_coeff1, conv_coeff2;
    double PI = 4.0 * atan(1.0);
    double *gridx = coord;
    double *gridy = coord + BLKSIZE;
    double *gridz = coord + BLKSIZE * 2;

#pragma GCC ivdep
    for (i = 0; i < ngrids; i++) {
        rr[i] = gridx[i] * gridx[i] + gridy[i] * gridy[i] + gridz[i] * gridz[i];
    }

    for (i = 0; i < nctr * BLKSIZE; i++) {
        ectr[i] = 0;
    }
    for (j = 0; j < nprim; j++) {
        conv_exp1 = alpha[j] * conv_alpha / (alpha[j] + conv_alpha);
        conv_exp2 = 2 * alpha[j] * conv_alpha / (alpha[j] + 2 * conv_alpha);
        conv_coeff1 = fac * pow(PI / conv_alpha, 1.5) * conv_alpha_coeff *
                      pow(conv_alpha / (alpha[j] + conv_alpha), 1.5 + l);
        conv_coeff2 =
            fac * pow(PI / (2 * conv_alpha), 1.5) * conv_alpha_coeff *
            pow(2 * conv_alpha / (alpha[j] + 2 * conv_alpha), 1.5 + l);
        for (i = 0; i < ngrids; i++) {
            arr1 = conv_exp1 * rr[i];
            arr2 = conv_exp2 * rr[i];
            eprim = exp(-arr1) * conv_coeff1 - exp(-arr2) * conv_coeff2;
            for (k = 0; k < nctr; k++) {
                ectr[k * BLKSIZE + i] += eprim * coeff[k * nprim + j];
            }
        }
    }
    return 1;
}

int SDMXcontract_rsq0(double *ectr, double *coord, double *alpha, double *coeff,
                      int l, int nprim, int nctr, size_t ngrids, double fac,
                      double conv_alpha, double conv_alpha_coeff) {
    size_t i, j, k;
    double arr, eprim;
    double rr[BLKSIZE];
    // exp(-alpha * r^2) - exp(-2 * alpha * r^2)
    double conv_exp, r0mul, r2mul;
    double PI = 4.0 * atan(1.0);
    double *gridx = coord;
    double *gridy = coord + BLKSIZE;
    double *gridz = coord + BLKSIZE * 2;
    double conv_coeff[NPRIMAX * NPRIMAX];

#pragma GCC ivdep
    for (i = 0; i < ngrids; i++) {
        rr[i] = gridx[i] * gridx[i] + gridy[i] * gridy[i] + gridz[i] * gridz[i];
    }

    for (i = 0; i < nctr * BLKSIZE; i++) {
        ectr[i] = 0;
    }
    for (i = 0; i < nctr; i++) {
        for (j = 0; j < nprim; j++) {
            conv_coeff[i * nprim + j] =
                (fac * pow(PI / conv_alpha, 1.5) * conv_alpha_coeff *
                 pow(conv_alpha / (alpha[j] + conv_alpha), 1.5 + l)) *
                coeff[i * nprim + j];
        }
    }

    for (j = 0; j < nprim; j++) {
        conv_exp = alpha[j] * conv_alpha / (alpha[j] + conv_alpha);
        r0mul = (1.5 + l) / (alpha[j] + conv_alpha) - l / conv_alpha;
        r2mul = conv_exp * (1.0 / conv_alpha - 1.0 / (alpha[j] + conv_alpha));
        for (i = 0; i < ngrids; i++) {
            arr = conv_exp * rr[i];
            eprim = (r0mul + rr[i] * r2mul) * exp(-arr);
            for (k = 0; k < nctr; k++) {
                ectr[k * BLKSIZE + i] += eprim * conv_coeff[k * nprim + j];
            }
        }
    }
    return 1;
}

int SDMXcontract_smooth1(double *ectr, double *coord, double *alpha,
                         double *coeff, int l, int nprim, int nctr,
                         size_t ngrids, double fac, double conv_alpha,
                         double conv_alpha_coeff) {
    size_t i, j, k;
    double arr1, arr2, eprim, deprim;
    double rr[BLKSIZE];
    // exp(-alpha * r^2) - exp(-2 * alpha * r^2)
    double conv_exp1, conv_exp2, conv_coeff1, conv_coeff2;
    double PI = 4.0 * atan(1.0);
    double *gridx = coord;
    double *gridy = coord + BLKSIZE;
    double *gridz = coord + BLKSIZE * 2;
    double *ectr_2a = ectr + NPRIMAX * BLKSIZE;
    // double coeff2a[NPRIMAX*NPRIMAX];

#pragma GCC ivdep
    for (i = 0; i < ngrids; i++) {
        rr[i] = gridx[i] * gridx[i] + gridy[i] * gridy[i] + gridz[i] * gridz[i];
    }

    for (i = 0; i < nctr * BLKSIZE; i++) {
        ectr[i] = 0;
        ectr_2a[i] = 0;
    }

    for (j = 0; j < nprim; j++) {
        conv_exp1 = alpha[j] * conv_alpha / (alpha[j] + conv_alpha);
        conv_exp2 = 2 * alpha[j] * conv_alpha / (alpha[j] + 2 * conv_alpha);
        conv_coeff1 = fac * pow(PI / conv_alpha, 1.5) * conv_alpha_coeff *
                      pow(conv_alpha / (alpha[j] + conv_alpha), 1.5 + l);
        conv_coeff2 =
            fac * pow(PI / (2 * conv_alpha), 1.5) * conv_alpha_coeff *
            pow(2 * conv_alpha / (alpha[j] + 2 * conv_alpha), 1.5 + l);
        for (i = 0; i < ngrids; i++) {
            arr1 = conv_exp1 * rr[i];
            arr2 = conv_exp2 * rr[i];
            arr1 = exp(-arr1) * conv_coeff1;
            arr2 = exp(-arr2) * conv_coeff2;
            eprim = arr1 - arr2;
            deprim = -2. * conv_exp1 * arr1 + 2 * conv_exp2 * arr2;
            for (k = 0; k < nctr; k++) {
                // ectr[k*BLKSIZE+i] += eprim * coeff[k*nprim+j];
                ectr[k * BLKSIZE + i] += eprim * coeff[k * nprim + j];
                ectr_2a[k * BLKSIZE + i] += deprim * coeff[k * nprim + j];
            }
        }
    }
    return 1;
}

int SDMXcontract_rsq1(double *ectr, double *coord, double *alpha, double *coeff,
                      int l, int nprim, int nctr, size_t ngrids, double fac,
                      double conv_alpha, double conv_alpha_coeff) {
    size_t i, j, k;
    double arr, eprim, deprim, r0mul, r2mul, dr0mul, dr2mul;
    double rr[BLKSIZE];
    // exp(-alpha * r^2) - exp(-2 * alpha * r^2)
    double conv_exp;
    double PI = 4.0 * atan(1.0);
    double *gridx = coord;
    double *gridy = coord + BLKSIZE;
    double *gridz = coord + BLKSIZE * 2;
    double *ectr_2a = ectr + NPRIMAX * BLKSIZE;
    double conv_coeff[NPRIMAX * NPRIMAX];
    double conv_factor[NPRIMAX];

#pragma GCC ivdep
    for (i = 0; i < ngrids; i++) {
        rr[i] = gridx[i] * gridx[i] + gridy[i] * gridy[i] + gridz[i] * gridz[i];
    }

    for (i = 0; i < nctr * BLKSIZE; i++) {
        ectr[i] = 0;
        ectr_2a[i] = 0;
    }
    for (j = 0; j < nprim; j++) {
        conv_factor[j] = (fac * pow(PI / conv_alpha, 1.5) * conv_alpha_coeff *
                          pow(conv_alpha / (alpha[j] + conv_alpha), 1.5 + l));
    }
    for (i = 0; i < nctr; i++) {
        for (j = 0; j < nprim; j++) {
            conv_coeff[i * nprim + j] = conv_factor[j] * coeff[i * nprim + j];
        }
    }

    for (j = 0; j < nprim; j++) {
        conv_exp = alpha[j] * conv_alpha / (alpha[j] + conv_alpha);
        r0mul = (1.5 + l) / (alpha[j] + conv_alpha) - l / conv_alpha;
        r2mul = conv_exp * (1.0 / conv_alpha - 1.0 / (alpha[j] + conv_alpha));
        dr0mul = -2 * conv_exp * r0mul + r2mul;
        dr2mul = -2 * conv_exp * r2mul;
        for (i = 0; i < ngrids; i++) {
            arr = exp(-conv_exp * rr[i]);
            eprim = (r0mul + rr[i] * r2mul) * arr;
            deprim = (dr0mul + rr[i] * dr2mul) * arr;
            for (k = 0; k < nctr; k++) {
                ectr[k * BLKSIZE + i] += eprim * conv_coeff[k * nprim + j];
                ectr_2a[k * BLKSIZE + i] += deprim * conv_coeff[k * nprim + j];
            }
        }
    }
    return 1;
}

double CINTcommon_fac_sp(int l) {
    switch (l) {
    case 0:
        return 0.282094791773878143;
    case 1:
        return 0.488602511902919921;
    default:
        return 1;
    }
}

int SDMXshloc_by_atom(int *shloc, int *shls_slice, int *atm, int *bas) {
    const int sh0 = shls_slice[0];
    const int sh1 = shls_slice[1];
    int ish, nshblk, lastatm;
    shloc[0] = sh0;
    nshblk = 1;
    lastatm = bas[BAS_SLOTS * sh0 + ATOM_OF];
    for (ish = sh0; ish < sh1; ish++) {
        if (lastatm != bas[BAS_SLOTS * ish + ATOM_OF]) {
            lastatm = bas[BAS_SLOTS * ish + ATOM_OF];
            shloc[nshblk] = ish;
            nshblk++;
        }
    }
    shloc[nshblk] = sh1;
    return nshblk;
}

// grid2atm[atm_id,xyz,grid_id]
static void _fill_grid2atm(double *grid2atm, double *coord, size_t bgrids,
                           size_t ngrids, int *atm, int natm, int *bas,
                           int nbas, double *env) {
    int atm_id;
    size_t ig;
    double *r_atm;
    for (atm_id = 0; atm_id < natm; atm_id++) {
        r_atm = env + atm[PTR_COORD + atm_id * ATM_SLOTS];
#pragma GCC ivdep
        for (ig = 0; ig < bgrids; ig++) {
            grid2atm[0 * BLKSIZE + ig] = coord[0 * ngrids + ig] - r_atm[0];
            grid2atm[1 * BLKSIZE + ig] = coord[1 * ngrids + ig] - r_atm[1];
            grid2atm[2 * BLKSIZE + ig] = coord[2 * ngrids + ig] - r_atm[2];
        }
        grid2atm += 3 * BLKSIZE;
    }
}

static void _dset0(double *out, size_t odim, size_t bgrids, int counts) {
    size_t i, j;
    for (i = 0; i < counts; i++) {
        for (j = 0; j < bgrids; j++) {
            out[i * odim + j] = 0;
        }
    }
}

void SDMXylm_loop(int ngrids, double *ylm_lg, double *coords, int *ylm_atom_loc,
                  double *atom_coords, int natm) {
#pragma omp parallel
    {
        int ia, ip, ib, blk, g;
        int l, m, lm;
        int ystart;
        int bgrids;
        int max_l2 = 0;
        sphbuf *sblist = malloc(sizeof(sphbuf) * natm);
        for (ia = 0; ia < natm; ia++) {
            lm = ylm_atom_loc[ia + 1] - ylm_atom_loc[ia];
            max_l2 = MAX(lm, max_l2);
            sblist[ia] = setup_sph_harm_buffer(lm);
        }
        double *buf = (double *)malloc(sizeof(double) * max_l2);
        double *gridx = coords;
        double *gridy = coords + ngrids;
        double *gridz = coords + 2 * ngrids;
        double norm_rvec[3];
        double rnorm;
        double rpow;
        int lmax;
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
#pragma omp for schedule(static)
        for (blk = 0; blk < natm * nblk; blk++) {
            ia = blk / nblk;
            ib = blk % nblk;
            ip = ib * BLKSIZE;
            bgrids = MIN(ngrids - ip, BLKSIZE);
            ystart = ylm_atom_loc[ia];
            if (sblist[ia].nlm > 1) {
                lmax =
                    (int)sqrt(ylm_atom_loc[ia + 1] - ylm_atom_loc[ia] + 1e-6) -
                    1;
                for (g = ip; g < ip + bgrids; g++) {
                    norm_rvec[0] = gridx[g] - atom_coords[3 * ia + 0];
                    norm_rvec[1] = gridy[g] - atom_coords[3 * ia + 1];
                    norm_rvec[2] = gridz[g] - atom_coords[3 * ia + 2];
                    rnorm = sqrt(norm_rvec[0] * norm_rvec[0] +
                                 norm_rvec[1] * norm_rvec[1] +
                                 norm_rvec[2] * norm_rvec[2]);
                    norm_rvec[0] /= rnorm;
                    norm_rvec[1] /= rnorm;
                    norm_rvec[2] /= rnorm;
                    recursive_sph_harm(sblist[ia], norm_rvec, buf);
                    rpow = 1.0;
                    lm = 0;
                    for (l = 0; l <= lmax; l++) {
                        for (m = 0; m < 2 * l + 1; m++, lm++) {
                            ylm_lg[(ystart + lm) * ngrids + g] = buf[lm] * rpow;
                        }
                        rpow *= rnorm;
                    }
                }
            } else {
                for (g = ip; g < ip + bgrids; g++) {
                    ylm_lg[(ystart + 0) * ngrids + g] = SPHF0;
                }
            }
        }
        free(buf);
        for (ia = 0; ia < natm; ia++) {
            free_sph_harm_buffer(sblist[ia]);
        }
        free(sblist);
    }
}

// assumes ylm_vlg already filled with ylm
void SDMXylm_grad(int ngrids, double *ylm_vlg, double *gaunt_vl, int gaunt_nlm,
                  int *ylm_atom_loc, int natm) {
#pragma omp parallel
    {
        int ia, ip, ib, blk, bgrids, offset;
        int nlm, lmax, ind1, ind2, ind3, nm, lm, g;
        int max_l2 = 0;
        for (ia = 0; ia < natm; ia++) {
            lm = ylm_atom_loc[ia + 1] - ylm_atom_loc[ia];
            max_l2 = MAX(lm, max_l2);
        }
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        double *ylm_lg = ylm_vlg;
        double *ylmx_lg = ylm_vlg + 1 * ylm_atom_loc[natm] * ngrids;
        double *ylmy_lg = ylm_vlg + 2 * ylm_atom_loc[natm] * ngrids;
        double *ylmz_lg = ylm_vlg + 3 * ylm_atom_loc[natm] * ngrids;
        double *gauntxm_l = gaunt_vl + 0 * gaunt_nlm;
        double *gauntxp_l = gaunt_vl + 1 * gaunt_nlm;
        double *gauntym_l = gaunt_vl + 2 * gaunt_nlm;
        double *gauntyp_l = gaunt_vl + 3 * gaunt_nlm;
        double *gauntz_l = gaunt_vl + 4 * gaunt_nlm;
        double *y, *yx, *yy, *yz;
        int l, m;
#pragma omp for schedule(static)
        for (blk = 0; blk < natm * nblk; blk++) {
            ia = blk / nblk;
            ib = blk % nblk;
            ip = ib * BLKSIZE;
            bgrids = MIN(ngrids - ip, BLKSIZE);
            offset = ylm_atom_loc[ia] * ngrids + ip;
            nlm = ylm_atom_loc[ia + 1] - ylm_atom_loc[ia];
            lmax = (int)sqrt(nlm + 1e-6) - 1;
            y = ylm_lg + offset;
            yx = ylmx_lg + offset;
            yy = ylmy_lg + offset;
            yz = ylmz_lg + offset;
            for (lm = 0; lm < nlm; lm++) {
                for (g = 0; g < bgrids; g++) {
                    yx[lm * ngrids + g] = 0;
                    yy[lm * ngrids + g] = 0;
                    yz[lm * ngrids + g] = 0;
                }
            }
            // we don't want to iterate over l=lmax because
            // l+1 gets written to
            for (l = 0; l < lmax; l++) {
                nm = 2 * l + 1;
                for (m = 0; m < nm; m++) {
                    ind1 = (l + 1) * (l + 1) + m + 1;
                    ind3 = (l + 1) * (l + 1) + 2 * l - m + 1;
                    ind2 = l * l + m;
                    for (g = 0; g < bgrids; g++) {
                        yz[ind1 * ngrids + g] +=
                            gauntz_l[ind2] * y[ind2 * ngrids + g];
                        yx[(ind1 - 1) * ngrids + g] +=
                            gauntxm_l[ind2] * y[ind2 * ngrids + g];
                        yx[(ind1 + 1) * ngrids + g] +=
                            gauntxp_l[ind2] * y[ind2 * ngrids + g];
                        yy[(ind3 - 1) * ngrids + g] +=
                            gauntym_l[ind2] * y[ind2 * ngrids + g];
                        yy[(ind3 + 1) * ngrids + g] +=
                            gauntyp_l[ind2] * y[ind2 * ngrids + g];
                    }
                }
            }
        }
    }
}

void SDMXylm_yzx2xyz(int ngrids, int nv, double *ylm_vlg, int *ylm_atom_loc,
                     int natm) {
#pragma omp parallel
    {
        double tmpx, tmpy, tmpz;
        int v, g;
        double *y;
        int blk, ia, ib, ip, bgrids, offset;
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
#pragma omp for schedule(static)
        for (blk = 0; blk < natm * nblk; blk++) {
            ia = blk / nblk;
            ib = blk % nblk;
            ip = ib * BLKSIZE;
            bgrids = MIN(ngrids - ip, BLKSIZE);
            offset = ylm_atom_loc[ia] * ngrids + ip;
            y = ylm_vlg + offset;
            if (ylm_atom_loc[ia + 1] - ylm_atom_loc[ia] > 1) {
                for (v = 0; v < nv; v++) {
                    for (g = 0; g < bgrids; g++) {
                        tmpy = y[1 * ngrids + g];
                        tmpz = y[2 * ngrids + g];
                        tmpx = y[3 * ngrids + g];
                        y[1 * ngrids + g] = tmpx;
                        y[2 * ngrids + g] = tmpy;
                        y[3 * ngrids + g] = tmpz;
                    }
                    y += ylm_atom_loc[natm] * ngrids;
                }
            }
        }
    }
}

void SDMXcontract_ao_to_bas(int ngrids, double *vbas, double *ylm_lg,
                            double *ao, int *shls_slice, int *ao_loc,
                            int *ylm_atom_loc, int *atm, int natm, int *bas,
                            int nbas, double *env, int nrf, int *rf_loc) {
#pragma omp parallel
    {
        const int nthread = omp_get_num_threads();
        const int blksize = (ngrids + nthread - 1) / nthread;
        int thread, sh, m, nm, di, g, bgrids;
        int sh0 = shls_slice[0];
        int sh1 = shls_slice[1];
        int ia, l, ip, irf;
        double *_vbas;
        double *_ylm;
        double *_ao;
#pragma omp for
        for (thread = 0; thread < nthread; thread++) {
            ip = blksize * thread;
            bgrids = MIN(ip + blksize, ngrids) - ip;
            for (sh = sh0; sh < sh1; sh++) {
                ia = bas[sh * BAS_SLOTS + ATOM_OF];
                l = bas[sh * BAS_SLOTS + ANG_OF];
                nm = 2 * l + 1;
                di = 0;
                for (irf = rf_loc[sh]; irf < rf_loc[sh + 1]; irf++) {
                    _vbas = vbas + irf * ngrids + ip;
                    for (g = 0; g < bgrids; g++) {
                        _vbas[g] = 0;
                    }
                    for (m = 0; m < nm; m++, di++) {
                        _ylm = ylm_lg +
                               (ylm_atom_loc[ia] + l * l + m) * ngrids + ip;
                        _ao = ao + (ao_loc[sh] + di) * ngrids + ip;
                        for (g = 0; g < bgrids; g++) {
                            _vbas[g] += _ylm[g] * _ao[g];
                        }
                    }
                }
            }
        }
    }
}

void SDMXcontract_ao_to_bas_bwd(int ngrids, double *vbas, double *ylm_lg,
                                double *ao, int *shls_slice, int *ao_loc,
                                int *ylm_atom_loc, int *atm, int natm, int *bas,
                                int nbas, double *env, int nrf, int *rf_loc) {
#pragma omp parallel
    {
        // NOTE: This is an in-place operation and ads to ao.
        const int nthread = omp_get_num_threads();
        const int blksize = (ngrids + nthread - 1) / nthread;
        int thread, sh, m, g, bgrids, di, nm;
        int sh0 = shls_slice[0];
        int sh1 = shls_slice[1];
        int ia, l, ip, irf;
        double *_vbas;
        double *_ylm;
        double *_ao;
#pragma omp for
        for (thread = 0; thread < nthread; thread++) {
            ip = blksize * thread;
            bgrids = MIN(ip + blksize, ngrids) - ip;
            for (sh = sh0; sh < sh1; sh++) {
                ia = bas[sh * BAS_SLOTS + ATOM_OF];
                l = bas[sh * BAS_SLOTS + ANG_OF];
                nm = 2 * l + 1;
                di = 0;
                for (irf = rf_loc[sh]; irf < rf_loc[sh + 1]; irf++) {
                    _vbas = vbas + irf * ngrids + ip;
                    for (m = 0; m < nm; m++, di++) {
                        _ylm = ylm_lg +
                               (ylm_atom_loc[ia] + l * l + m) * ngrids + ip;
                        _ao = ao + (ao_loc[sh] + di) * ngrids + ip;
                        for (g = 0; g < bgrids; g++) {
                            _ao[g] += _ylm[g] * _vbas[g];
                        }
                    }
                }
            }
        }
    }
}

void SDMXcontract_ao_to_bas_grid(int ngrids, double *vbas, double *ylm_lg,
                                 double *ao, int *shls_slice, int *ao_loc,
                                 int *ylm_atom_loc, int *atm, int natm,
                                 int *bas, int nbas, double *env, int nrf,
                                 int *rf_loc, double *gridx, double *atomx) {
#pragma omp parallel
    {
        const int nthread = omp_get_num_threads();
        const int blksize = (ngrids + nthread - 1) / nthread;
        int thread, sh, m, g, di, nm;
        int sh0 = shls_slice[0];
        int sh1 = shls_slice[1];
        int ia, l, ip, irf;
        int bgrids;
        double *_vbas;
        double *_ylm;
        double *_ao;
        double *_gridx;
        double *dx = malloc(sizeof(double) * blksize);
#pragma omp for
        for (thread = 0; thread < nthread; thread++) {
            ip = blksize * thread;
            bgrids = MIN(ip + blksize, ngrids) - ip;
            for (sh = sh0; sh < sh1; sh++) {
                ia = bas[sh * BAS_SLOTS + ATOM_OF];
                l = bas[sh * BAS_SLOTS + ANG_OF];
                nm = 2 * l + 1;
                _vbas = vbas + sh * ngrids + ip;
                _gridx = gridx + ip;
                for (g = 0; g < bgrids; g++) {
                    dx[g] = _gridx[g] - atomx[ia];
                }
                di = 0;
                for (irf = rf_loc[sh]; irf < rf_loc[sh + 1]; irf++) {
                    _vbas = vbas + irf * ngrids + ip;
                    for (g = 0; g < bgrids; g++) {
                        _vbas[g] = 0;
                    }
                    for (m = 0; m < nm; m++, di++) {
                        _ylm = ylm_lg +
                               (ylm_atom_loc[ia] + l * l + m) * ngrids + ip;
                        _ao = ao + (ao_loc[sh] + di) * ngrids + ip;
                        for (g = 0; g < bgrids; g++) {
                            _vbas[g] += _ylm[g] * _ao[g] * dx[g];
                        }
                    }
                }
            }
        }
        free(dx);
    }
}

void SDMXcontract_ao_to_bas_grid_bwd(int ngrids, double *vbas, double *ylm_lg,
                                     double *ao, int *shls_slice, int *ao_loc,
                                     int *ylm_atom_loc, int *atm, int natm,
                                     int *bas, int nbas, double *env,
                                     double *gridx, double *atomx, int nrf,
                                     int *rf_loc) {
#pragma omp parallel
    {
        // NOTE: This is an in-place operation and ads to ao.
        const int nthread = omp_get_num_threads();
        const int blksize = (ngrids + nthread - 1) / nthread;
        int thread, sh, m, g, di, nm;
        int sh0 = shls_slice[0];
        int sh1 = shls_slice[1];
        int ia, l, ip;
        int bgrids, irf;
        double *_vbas;
        double *_ylm;
        double *_ao;
        double *_gridx;
        double *dx = malloc(sizeof(double) * blksize);
#pragma omp for
        for (thread = 0; thread < nthread; thread++) {
            ip = blksize * thread;
            bgrids = MIN(ip + blksize, ngrids) - ip;
            for (sh = sh0; sh < sh1; sh++) {
                ia = bas[sh * BAS_SLOTS + ATOM_OF];
                l = bas[sh * BAS_SLOTS + ANG_OF];
                nm = 2 * l + 1;
                _gridx = gridx + ip;
                for (g = 0; g < bgrids; g++) {
                    dx[g] = _gridx[g] - atomx[ia];
                }
                di = 0;
                for (irf = rf_loc[sh]; irf < rf_loc[sh + 1]; irf++) {
                    _vbas = vbas + irf * ngrids + ip;
                    for (m = 0; m < nm; m++, di++) {
                        _ylm = ylm_lg +
                               (ylm_atom_loc[ia] + l * l + m) * ngrids + ip;
                        _ao = ao + (ao_loc[sh] + di) * ngrids + ip;
                        for (g = 0; g < bgrids; g++) {
                            _ao[g] += _ylm[g] * _vbas[g] * dx[g];
                        }
                    }
                }
            }
        }
        free(dx);
    }
}

void SDMXcontract_ao_to_bas_l1(int ngrids, double *vbas, double *ylm_vlg,
                               double *ao, int *shls_slice, int *ao_loc,
                               int *ylm_atom_loc, int *atm, int natm, int *bas,
                               int nbas, double *env, double *gridx,
                               double *atomx, int nrf, int *rf_loc) {
#pragma omp parallel
    {
        const int nthread = omp_get_num_threads();
        const int blksize = (ngrids + nthread - 1) / nthread;
        int thread, sh, m, g, di, nm;
        int sh0 = shls_slice[0];
        int sh1 = shls_slice[1];
        int ia, l, ip, irf;
        int bgrids;
        double *_ao;
        double *_gridx, *_gridy, *_gridz;
        double *atomy = atomx + natm;
        double *atomz = atomx + 2 * natm;
        double *_vbas0, *_vbas1, *_vbas2, *_vbas3;
        double *_ylm0, *_ylm1, *_ylm2, *_ylm3;
        int offset;
#pragma omp for
        for (thread = 0; thread < nthread; thread++) {
            ip = blksize * thread;
            bgrids = MIN(ip + blksize, ngrids) - ip;
            for (sh = sh0; sh < sh1; sh++) {
                ia = bas[sh * BAS_SLOTS + ATOM_OF];
                l = bas[sh * BAS_SLOTS + ANG_OF];
                nm = 2 * l + 1;
                di = 0;
                for (irf = rf_loc[sh]; irf < rf_loc[sh + 1]; irf++) {
                    offset = irf * ngrids + ip;
                    _vbas0 = vbas + offset;
                    _vbas1 = vbas + 1 * nrf * ngrids + offset;
                    _vbas2 = vbas + 2 * nrf * ngrids + offset;
                    _vbas3 = vbas + 3 * nrf * ngrids + offset;
                    _gridx = gridx + ip;
                    _gridy = gridx + ngrids + ip;
                    _gridz = gridx + 2 * ngrids + ip;
                    for (g = 0; g < bgrids; g++) {
                        _vbas0[g] = 0;
                        _vbas1[g] = 0;
                        _vbas2[g] = 0;
                        _vbas3[g] = 0;
                    }
                    for (m = 0; m < nm; m++, di++) {
                        offset = (ylm_atom_loc[ia] + l * l + m) * ngrids + ip;
                        _ylm0 = ylm_vlg + offset;
                        _ylm1 =
                            ylm_vlg + 1 * ylm_atom_loc[natm] * ngrids + offset;
                        _ylm2 =
                            ylm_vlg + 2 * ylm_atom_loc[natm] * ngrids + offset;
                        _ylm3 =
                            ylm_vlg + 3 * ylm_atom_loc[natm] * ngrids + offset;
                        _ao = ao + (ao_loc[sh] + di) * ngrids + ip;
                        for (g = 0; g < bgrids; g++) {
                            _vbas0[g] += _ylm0[g] * _ao[g];
                            _vbas1[g] += _ylm1[g] * _ao[g];
                            _vbas2[g] += _ylm2[g] * _ao[g];
                            _vbas3[g] += _ylm3[g] * _ao[g];
                        }
                    }
                    offset = 3 * nrf * ngrids;
                    for (g = 0; g < bgrids; g++) {
                        _vbas1[offset + g] =
                            _vbas0[g] * (_gridx[g] - atomx[ia]);
                        _vbas2[offset + g] =
                            _vbas0[g] * (_gridy[g] - atomy[ia]);
                        _vbas3[offset + g] =
                            _vbas0[g] * (_gridz[g] - atomz[ia]);
                    }
                }
            }
        }
    }
}

void SDMXcontract_ao_to_bas_l1_bwd(int ngrids, double *vbas, double *ylm_vlg,
                                   double *ao, int *shls_slice, int *ao_loc,
                                   int *ylm_atom_loc, int *atm, int natm,
                                   int *bas, int nbas, double *env,
                                   double *gridx, double *atomx, int nrf,
                                   int *rf_loc) {
#pragma omp parallel
    {
        // NOTE: ao must be zero upon entry
        const int nthread = omp_get_num_threads();
        const int blksize = (ngrids + nthread - 1) / nthread;
        int thread, sh, m, g, di, nm;
        int sh0 = shls_slice[0];
        int sh1 = shls_slice[1];
        int ia, l, ip, irf;
        int bgrids;
        double *_ao;
        double *_gridx, *_gridy, *_gridz;
        double *atomy = atomx + natm;
        double *atomz = atomx + 2 * natm;
        double *_vbas0, *_vbas1, *_vbas2, *_vbas3;
        double *_ylm0, *_ylm1, *_ylm2, *_ylm3;
        double *vb0tmp = malloc(sizeof(double) * blksize);
        int offset;
#pragma omp for
        for (thread = 0; thread < nthread; thread++) {
            ip = blksize * thread;
            bgrids = MIN(ip + blksize, ngrids) - ip;
            for (sh = sh0; sh < sh1; sh++) {
                ia = bas[sh * BAS_SLOTS + ATOM_OF];
                l = bas[sh * BAS_SLOTS + ANG_OF];
                nm = 2 * l + 1;
                di = 0;
                for (irf = rf_loc[sh]; irf < rf_loc[sh + 1]; irf++) {
                    offset = irf * ngrids + ip;
                    _vbas0 = vbas + offset;
                    _vbas1 = vbas + 1 * nrf * ngrids + offset;
                    _vbas2 = vbas + 2 * nrf * ngrids + offset;
                    _vbas3 = vbas + 3 * nrf * ngrids + offset;
                    _gridx = gridx + ip;
                    _gridy = gridx + ngrids + ip;
                    _gridz = gridx + 2 * ngrids + ip;
                    offset = 3 * nrf * ngrids;
                    for (g = 0; g < bgrids; g++) {
                        vb0tmp[g] = _vbas0[g];
                        vb0tmp[g] +=
                            _vbas1[offset + g] * (_gridx[g] - atomx[ia]);
                        vb0tmp[g] +=
                            _vbas2[offset + g] * (_gridy[g] - atomy[ia]);
                        vb0tmp[g] +=
                            _vbas3[offset + g] * (_gridz[g] - atomz[ia]);
                    }
                    for (m = 0; m < nm; m++, di++) {
                        offset = (ylm_atom_loc[ia] + l * l + m) * ngrids + ip;
                        _ylm0 = ylm_vlg + offset;
                        _ylm1 =
                            ylm_vlg + 1 * ylm_atom_loc[natm] * ngrids + offset;
                        _ylm2 =
                            ylm_vlg + 2 * ylm_atom_loc[natm] * ngrids + offset;
                        _ylm3 =
                            ylm_vlg + 3 * ylm_atom_loc[natm] * ngrids + offset;
                        _ao = ao + (ao_loc[sh] + di) * ngrids + ip;
                        for (g = 0; g < bgrids; g++) {
                            _ao[g] = _ylm0[g] * vb0tmp[g];
                            _ao[g] += _ylm1[g] * _vbas1[g];
                            _ao[g] += _ylm2[g] * _vbas2[g];
                            _ao[g] += _ylm3[g] * _vbas3[g];
                        }
                    }
                }
            }
        }
        free(vb0tmp);
    }
}

void contract_shl_to_alpha_l1(int ngrids, int nalpha, int nsh, double *p,
                              double *b, double *csh) {
#pragma omp parallel
    {
        int blksize = 128;
        int ablk, alpha, blk, sh, g, ip, bgrids;
        int nblk = (ngrids + blksize - 1) / blksize;
        int nablk = nblk * nalpha;
        double *b0 = b;
        double *b1 = b + 1 * ngrids * nsh;
        double *b2 = b + 2 * ngrids * nsh;
        double *b3 = b + 3 * ngrids * nsh;
        double *b4 = b + 4 * ngrids * nsh;
        double *b5 = b + 5 * ngrids * nsh;
        double *b6 = b + 6 * ngrids * nsh;
        double *p0 = p;
        double *p1 = p + 1 * ngrids * nalpha;
        double *p2 = p + 2 * ngrids * nalpha;
        double *p3 = p + 3 * ngrids * nalpha;
        double *csh0 = csh;
        double *csh1 = csh + ngrids * nsh * nalpha;
        double *p0a, *p1a, *p2a, *p3a;
        double *b0c, *b1c, *b2c, *b3c, *b4c, *b5c, *b6c;
        double *csh0ac, *csh1ac;
        int offset;
#pragma omp for
        for (ablk = 0; ablk < nablk; ablk++) {
            alpha = ablk / nblk;
            blk = ablk - alpha * nblk;
            ip = blk * blksize;
            bgrids = MIN(blksize, ngrids - ip);
            offset = alpha * ngrids + ip;
            p0a = p0 + offset;
            p1a = p1 + offset;
            p2a = p2 + offset;
            p3a = p3 + offset;
            for (g = 0; g < bgrids; g++) {
                p0a[g] = 0;
                p1a[g] = 0;
                p2a[g] = 0;
                p3a[g] = 0;
            }
            for (sh = 0; sh < nsh; sh++) {
                offset = sh * ngrids + ip;
                b0c = b0 + offset;
                b1c = b1 + offset;
                b2c = b2 + offset;
                b3c = b3 + offset;
                b4c = b4 + offset;
                b5c = b5 + offset;
                b6c = b6 + offset;
                offset = alpha * nsh * ngrids + sh * ngrids + ip;
                csh0ac = csh0 + offset;
                csh1ac = csh1 + offset;
#pragma GCC ivdep
                for (g = 0; g < bgrids; g++) {
                    p0a[g] += b0c[g] * csh0ac[g];
                    p1a[g] += b1c[g] * csh0ac[g] + b4c[g] * csh1ac[g];
                    p2a[g] += b2c[g] * csh0ac[g] + b5c[g] * csh1ac[g];
                    p3a[g] += b3c[g] * csh0ac[g] + b6c[g] * csh1ac[g];
                }
            }
        }
    }
}

void contract_shl_to_alpha_l1_bwd(int ngrids, int nalpha, int nsh, double *p,
                                  double *b, double *csh) {
#pragma omp parallel
    {
        int blksize = 128;
        int shblk, alpha, blk, sh, g, ip, bgrids;
        int nblk = (ngrids + blksize - 1) / blksize;
        int nshblk = nblk * nsh;
        double *b0 = b;
        double *b1 = b + 1 * ngrids * nsh;
        double *b2 = b + 2 * ngrids * nsh;
        double *b3 = b + 3 * ngrids * nsh;
        double *b4 = b + 4 * ngrids * nsh;
        double *b5 = b + 5 * ngrids * nsh;
        double *b6 = b + 6 * ngrids * nsh;
        double *p0 = p;
        double *p1 = p + 1 * ngrids * nalpha;
        double *p2 = p + 2 * ngrids * nalpha;
        double *p3 = p + 3 * ngrids * nalpha;
        double *csh0 = csh;
        double *csh1 = csh + ngrids * nsh * nalpha;
        double *p0a, *p1a, *p2a, *p3a;
        double *b0c, *b1c, *b2c, *b3c, *b4c, *b5c, *b6c;
        double *csh0ac, *csh1ac;
        int offset;
#pragma omp for
        for (shblk = 0; shblk < nshblk; shblk++) {
            sh = shblk / nblk;
            blk = shblk - sh * nblk;
            ip = blk * blksize;
            bgrids = MIN(blksize, ngrids - ip);
            offset = sh * ngrids + ip;
            b0c = b0 + offset;
            b1c = b1 + offset;
            b2c = b2 + offset;
            b3c = b3 + offset;
            b4c = b4 + offset;
            b5c = b5 + offset;
            b6c = b6 + offset;
            for (g = 0; g < bgrids; g++) {
                b0c[g] = 0;
                b1c[g] = 0;
                b2c[g] = 0;
                b3c[g] = 0;
                b4c[g] = 0;
                b5c[g] = 0;
                b6c[g] = 0;
            }
            for (alpha = 0; alpha < nalpha; alpha++) {
                offset = alpha * ngrids + ip;
                p0a = p0 + offset;
                p1a = p1 + offset;
                p2a = p2 + offset;
                p3a = p3 + offset;
                offset = alpha * nsh * ngrids + sh * ngrids + ip;
                csh0ac = csh0 + offset;
                csh1ac = csh1 + offset;
#pragma GCC ivdep
                for (g = 0; g < bgrids; g++) {
                    b0c[g] += p0a[g] * csh0ac[g];
                    b1c[g] += p1a[g] * csh0ac[g];
                    b2c[g] += p2a[g] * csh0ac[g];
                    b3c[g] += p3a[g] * csh0ac[g];
                    b4c[g] += p1a[g] * csh1ac[g];
                    b5c[g] += p2a[g] * csh1ac[g];
                    b6c[g] += p3a[g] * csh1ac[g];
                }
            }
        }
    }
}

// vlg : (ncomp, ylm_atom_loc[natm], ngrids)
void SDMXeval_loop(void (*fiter)(), FPtr_eval_sdmx feval, FPtr_exp_sdmx fexp,
                   double fac, int ngrids, int param[], int *shls_slice,
                   int *ao_loc, double *ao, double *coord, uint8_t *non0table,
                   int *atm, int natm, int *bas, int nbas, double *env,
                   double *ylm_vlg, int *ylm_atom_loc, double *alphas,
                   double *alpha_norms, int nalpha) {
    int shloc[shls_slice[1] - shls_slice[0] + 1];
    const int nshlblk = SDMXshloc_by_atom(shloc, shls_slice, atm, bas);
    const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
    const size_t Ngrids = ngrids;
    int _sh0 = shls_slice[0];
    int _sh1 = shls_slice[1];
    int _lmax = 0;
    int bas_id;
    for (bas_id = _sh0; bas_id < _sh1; bas_id++) {
        _lmax = MAX(bas[bas_id * BAS_SLOTS + ANG_OF], _lmax);
    }
#pragma omp parallel
    {
        const int max2lp1 = 2 * _lmax + 1;
        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];
        const size_t nao = ao_loc[sh1] - ao_loc[sh0];
        int ip, ib, k, iloc, ish;
        size_t aoff, bgrids;
        int ncart = NCTR_CART * param[TENSOR] * param[POS_E1];
        double *buf =
            malloc(sizeof(double) * BLKSIZE * (NPRIMAX * 2 + ncart + 1));
        double *ybuf =
            malloc(sizeof(double) * BLKSIZE * param[TENSOR] * max2lp1 + 8);
#pragma omp for schedule(dynamic, 4)
        for (k = 0; k < nblk * nshlblk; k++) {
            iloc = k / nblk;
            ish = shloc[iloc];
            aoff = ao_loc[ish] - ao_loc[sh0];
            ib = k - iloc * nblk;
            ip = ib * BLKSIZE;
            bgrids = MIN(ngrids - ip, BLKSIZE);
            (*fiter)(feval, fexp, fac, nao, Ngrids, bgrids, param, shloc + iloc,
                     ao_loc, buf, ao + aoff * Ngrids + ip, coord + ip,
                     non0table + ib * nbas, atm, natm, bas, nbas, env,
                     ylm_vlg + ip, ylm_atom_loc, alphas, alpha_norms, nalpha,
                     ybuf);
        }
        free(buf);
        free(ybuf);
    }
}

void SDMXeval_rad_loop(FPtr_eval_sdmx_rad feval, FPtr_exp_sdmx fexp, double fac,
                       int ngrids, int param[], int *shls_slice, int *rf_loc,
                       double *vbas, double *coord, uint8_t *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env,
                       double *alphas, double *alpha_norms, int nalpha) {
    int shloc[shls_slice[1] - shls_slice[0] + 1];
    const int nshlblk = SDMXshloc_by_atom(shloc, shls_slice, atm, bas);
    const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
    const size_t Ngrids = ngrids;
#pragma omp parallel
    {
        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];
        const size_t nao = rf_loc[sh1] - rf_loc[sh0];
        int ip, ib, k, iloc, ish;
        size_t soff, bgrids;
        int ncart = NCTR_CART * param[TENSOR] * param[POS_E1];
        double *buf =
            malloc(sizeof(double) * BLKSIZE * (NPRIMAX * 2 + ncart + 1));
#pragma omp for schedule(dynamic, 4)
        for (k = 0; k < nblk * nshlblk; k++) {
            iloc = k / nblk;
            ish = shloc[iloc];
            soff = rf_loc[ish] - rf_loc[sh0];
            ib = k - iloc * nblk;
            ip = ib * BLKSIZE;
            bgrids = MIN(ngrids - ip, BLKSIZE);
            SDMXeval_rad_iter(feval, fexp, fac, nao, Ngrids, bgrids, param,
                              shloc + iloc, rf_loc, buf,
                              vbas + soff * Ngrids + ip, coord + ip,
                              non0table + ib * nbas, atm, natm, bas, nbas, env,
                              alphas, alpha_norms, nalpha);
        }
        free(buf);
    }
}

void SDMXeval_sph_iter(FPtr_eval_sdmx feval, FPtr_exp_sdmx fexp, double fac,
                       size_t nao, size_t ngrids, size_t bgrids, int param[],
                       int *shls_slice, int *ao_loc, double *buf, double *ao,
                       double *coord, uint8_t *non0table, int *atm, int natm,
                       int *bas, int nbas, double *env, double *ylm_vlg,
                       int *ylm_atom_loc, double *alphas, double *alpha_norms,
                       int nalpha, double *ybuf) {
    const int ncomp = param[TENSOR];
    const int sh0 = shls_slice[0];
    const int sh1 = shls_slice[1];
    const int atmstart = bas[sh0 * BAS_SLOTS + ATOM_OF];
    const int atmend = bas[(sh1 - 1) * BAS_SLOTS + ATOM_OF] + 1;
    const int atmcount = atmend - atmstart;
    int i, l, np, nc, atm_id, bas_id, deg, ao_id;
    double fac1;
    double *p_exp, *pcoeff, *pcoord, *ri;
    double *grid2atm = ALIGN8_UP(buf); // [atm_id,xyz,grid]
    double *ylm_buf = ALIGN8_UP(ybuf);
    double *eprim = grid2atm + atmcount * 3 * BLKSIZE;
    int n_alm = ylm_atom_loc[natm];
    double *ylm_vmg;
    int m, g, v;

    _fill_grid2atm(grid2atm, coord, bgrids, ngrids, atm + atmstart * ATM_SLOTS,
                   atmcount, bas, nbas, env);

    for (bas_id = sh0; bas_id < sh1; bas_id++) {
        np = bas[bas_id * BAS_SLOTS + NPRIM_OF];
        nc = bas[bas_id * BAS_SLOTS + NCTR_OF];
        l = bas[bas_id * BAS_SLOTS + ANG_OF];
        deg = l * 2 + 1;
        fac1 = fac; // * CINTcommon_fac_sp(l);
        p_exp = env + bas[bas_id * BAS_SLOTS + PTR_EXP];
        pcoeff = env + bas[bas_id * BAS_SLOTS + PTR_COEFF];
        atm_id = bas[bas_id * BAS_SLOTS + ATOM_OF];
        pcoord = grid2atm + (atm_id - atmstart) * 3 * BLKSIZE;
        ao_id = ao_loc[bas_id] - ao_loc[sh0];
        ylm_vmg = ylm_vlg + ngrids * (ylm_atom_loc[atm_id] + l * l);
        for (v = 0; v < ncomp; v++) {
            for (m = 0; m < deg; m++) {
                for (g = 0; g < bgrids; g++) {
                    ylm_buf[(v * deg + m) * BLKSIZE + g] =
                        ylm_vmg[m * ngrids + g];
                }
            }
            ylm_vmg += n_alm * ngrids;
        }
        for (int ialpha = 0; ialpha < nalpha; ialpha++) {
            if (non0table[bas_id] &&
                (*fexp)(eprim, pcoord, p_exp, pcoeff, l, np, nc, bgrids, fac1,
                        alphas[ialpha], alpha_norms[ialpha])) {
                ri = env + atm[PTR_COORD + atm_id * ATM_SLOTS];
                // lbuf[i + BLKSIZE * (m + max2lp1 * v)] =
                //     ylm_vlg[i + ngrids * (m + n_alm * v)];
                (*feval)(ao + ao_id * ngrids, ri, eprim, pcoord, p_exp, pcoeff,
                         env, l, np, nc, nao, ngrids, bgrids, ylm_buf,
                         deg * BLKSIZE);
            } else {
                for (i = 0; i < ncomp; i++) {
                    // TODO this might not set everything to zero
                    _dset0(ao + (i * nao + ao_id) * ngrids, ngrids, bgrids,
                           nc * deg);
                }
            }
            ao_id += nao * ncomp;
        }
    }
}

void SDMXeval_rad_iter(FPtr_eval_sdmx_rad feval, FPtr_exp_sdmx fexp, double fac,
                       size_t nao, size_t ngrids, size_t bgrids, int param[],
                       int *shls_slice, int *rf_loc, double *buf, double *vbas,
                       double *coord, uint8_t *non0table, int *atm, int natm,
                       int *bas, int nbas, double *env, double *alphas,
                       double *alpha_norms, int nalpha) {
    const int ncomp = param[TENSOR];
    const int sh0 = shls_slice[0];
    const int sh1 = shls_slice[1];
    const int atmstart = bas[sh0 * BAS_SLOTS + ATOM_OF];
    const int atmend = bas[(sh1 - 1) * BAS_SLOTS + ATOM_OF] + 1;
    const int atmcount = atmend - atmstart;
    int i, k, l, np, nc, atm_id, bas_id;
    double fac1;
    double *p_exp, *pcoeff, *pcoord;
    double *grid2atm = ALIGN8_UP(buf); // [atm_id,xyz,grid]
    double *eprim = grid2atm + atmcount * 3 * BLKSIZE;
    int sh;

    _fill_grid2atm(grid2atm, coord, bgrids, ngrids, atm + atmstart * ATM_SLOTS,
                   atmcount, bas, nbas, env);

    for (bas_id = sh0; bas_id < sh1; bas_id++) {
        np = bas[bas_id * BAS_SLOTS + NPRIM_OF];
        nc = bas[bas_id * BAS_SLOTS + NCTR_OF];
        l = bas[bas_id * BAS_SLOTS + ANG_OF];
        fac1 = fac; // * CINTcommon_fac_sp(l);
        p_exp = env + bas[bas_id * BAS_SLOTS + PTR_EXP];
        pcoeff = env + bas[bas_id * BAS_SLOTS + PTR_COEFF];
        atm_id = bas[bas_id * BAS_SLOTS + ATOM_OF];
        pcoord = grid2atm + (atm_id - atmstart) * 3 * BLKSIZE;
        sh = rf_loc[bas_id] - rf_loc[sh0];
        for (int ialpha = 0; ialpha < nalpha; ialpha++) {
            if (non0table[bas_id] &&
                (*fexp)(eprim, pcoord, p_exp, pcoeff, l, np, nc, bgrids, fac1,
                        alphas[ialpha], alpha_norms[ialpha])) {
                (*feval)(vbas + sh * ngrids, eprim, nc, nao, ngrids, bgrids,
                         nao * ngrids * nalpha);
            } else {
                for (i = 0; i < ncomp; i++) {
                    for (k = 0; k < rf_loc[bas_id + 1] - rf_loc[bas_id]; k++) {
                        _dset0(vbas + (i * nalpha * nao + sh + k) * ngrids,
                               ngrids, bgrids, nc);
                    }
                }
            }
            sh += nao;
        }
    }
}

void SDMXrad_eval_grid(double *vbas, double *exps, int nc, size_t nao,
                       size_t ngrids, size_t blksize, int stride) {
    int k, i;
    for (k = 0; k < nc; k++) {
#pragma GCC ivdep
        for (i = 0; i < blksize; i++) {
            // gto[i] = ylm_vmg[m * ngrids + i] * exps[k * BLKSIZE + i];
            vbas[i] = exps[k * BLKSIZE + i];
        }
        vbas += ngrids;
    }
}

void SDMXrad_eval_grid_deriv1(double *vbas, double *exps, int nc, size_t nao,
                              size_t ngrids, size_t blksize, int stride) {
    int k, i;
    double *exps_2a = exps + NPRIMAX * BLKSIZE;
    double *vbas_2a = vbas + stride;
    for (k = 0; k < nc; k++) {
#pragma GCC ivdep
        for (i = 0; i < blksize; i++) {
            // gto[i] = ylm_vmg[m * ngrids + i] * exps[k * BLKSIZE + i];
            vbas[i] = exps[k * BLKSIZE + i];
        }
#pragma GCC ivdep
        for (i = 0; i < blksize; i++) {
            vbas_2a[i] = exps_2a[k * BLKSIZE + i];
        }
        vbas += ngrids;
        vbas_2a += ngrids;
    }
}

void SDMXshell_eval_grid_cart(double *gto, double *ri, double *exps,
                              double *coord, double *alpha, double *coeff,
                              double *env, int l, int np, int nc, size_t nao,
                              size_t ngrids, size_t blksize, double *ylm_vmg,
                              int mg_stride) {
    size_t l2p1 = 2 * l + 1;
    size_t i, k, m;

    for (k = 0; k < nc; k++) {
        for (m = 0; m < l2p1; m++) {
#pragma GCC ivdep
            for (i = 0; i < blksize; i++) {
                gto[i] = ylm_vmg[m * BLKSIZE + i] * exps[k * BLKSIZE + i];
            }
            gto += ngrids;
        }
    }
}

void SDMXshell_eval_grid_cart_deriv1(double *gto, double *ri, double *exps,
                                     double *coord, double *alpha,
                                     double *coeff, double *env, int l, int np,
                                     int nc, size_t nao, size_t ngrids,
                                     size_t blksize, double *ylm_vmg,
                                     int mg_stride) {
    size_t l2p1 = 2 * l + 1;
    size_t i, k, m;
    double *y0_mg = ylm_vmg;
    double *yx_mg = y0_mg + mg_stride;
    double *yy_mg = yx_mg + mg_stride;
    double *yz_mg = yy_mg + mg_stride;
    double *gridx = coord;
    double *gridy = coord + BLKSIZE;
    double *gridz = coord + BLKSIZE * 2;
    double *gtox = gto + nao * ngrids;
    double *gtoy = gto + nao * ngrids * 2;
    double *gtoz = gto + nao * ngrids * 3;
    double *exps_2a = exps + NPRIMAX * BLKSIZE;
    double tmp;

    for (k = 0; k < nc; k++) {
        for (m = 0; m < l2p1; m++) {
#pragma GCC ivdep
            for (i = 0; i < blksize; i++) {
                tmp = y0_mg[m * BLKSIZE + i] * exps_2a[k * BLKSIZE + i];
                gto[i] = y0_mg[m * BLKSIZE + i] * exps[k * BLKSIZE + i];
                gtox[i] = tmp * gridx[i];
                gtoy[i] = tmp * gridy[i];
                gtoz[i] = tmp * gridz[i];
                gtox[i] += yx_mg[m * BLKSIZE + i] * exps[k * BLKSIZE + i];
                gtoy[i] += yy_mg[m * BLKSIZE + i] * exps[k * BLKSIZE + i];
                gtoz[i] += yz_mg[m * BLKSIZE + i] * exps[k * BLKSIZE + i];
            }
            gto += ngrids;
            gtox += ngrids;
            gtoy += ngrids;
            gtoz += ngrids;
        }
    }
}
