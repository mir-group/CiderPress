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

#include "conv_interpolation.h"
#include "convolutions.h"
#include "fblas.h"
#include "sph_harm.h"
#include "spline.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void compute_spline_maps(double *w_rsp, double *Rg, int *bas, int nbas,
                         double *env, int *shls_slice, int ngrids) {
#pragma omp parallel
    {
        int g, l, ish;
        double coef, expi;
        int *ibas;
        int ish0 = 0;
        int ish1 = nbas;
        int nsh = ish1 - ish0;
        double *auxo_g = (double *)malloc(ngrids * sizeof(double));
        double *spline = (double *)malloc(5 * ngrids * sizeof(double));
        double *w_rp;
#pragma omp for schedule(dynamic, 4)
        for (ish = ish0; ish < ish1; ish++) {
            ibas = bas + ish * BAS_SLOTS;
            l = ibas[ANG_OF];
            coef = env[ibas[PTR_COEFF]];
            expi = env[ibas[PTR_EXP]];
            for (g = 0; g < ngrids; g++) {
                auxo_g[g] = coef * pow(Rg[g], l) * exp(-expi * Rg[g] * Rg[g]);
            }
            auxo_g[ngrids - 1] = 0;
            get_cubic_spline_coeff(Rg, auxo_g, spline, ngrids);
            w_rp = w_rsp + 4 * (ish - ish0);
            for (g = 0; g < ngrids; g++) {
                w_rp[g * 4 * nsh + 0] = spline[1 * ngrids + g];
                w_rp[g * 4 * nsh + 1] = spline[2 * ngrids + g];
                w_rp[g * 4 * nsh + 2] = spline[3 * ngrids + g];
                w_rp[g * 4 * nsh + 3] = spline[4 * ngrids + g];
            }
        }
        free(auxo_g);
        free(spline);
    }
}

void fill_sph_harm_deriv_coeff(double *c_xl, double *d_vxl, double *gaunt_vl,
                               int nx, int lmax) {
    // #pragma omp parallel
    {
        int l, m, lm, nlm, lmz;
        nlm = (lmax + 1) * (lmax + 1);
        int i;
        double *dx_l, *dy_l, *dz_l, *cx_l;
        double *gauntxm_l = gaunt_vl + 0 * nlm;
        double *gauntxp_l = gaunt_vl + 1 * nlm;
        double *gauntym_l = gaunt_vl + 2 * nlm;
        double *gauntyp_l = gaunt_vl + 3 * nlm;
        double *gauntz_l = gaunt_vl + 4 * nlm;
        // #pragma omp for
        for (i = 0; i < nx; i++) {
            dx_l = d_vxl + (0 * nx + i) * nlm;
            dy_l = d_vxl + (1 * nx + i) * nlm;
            dz_l = d_vxl + (2 * nx + i) * nlm;
            cx_l = c_xl + i * nlm;
            for (l = 0; l < lmax; l++) {
                for (m = 0; m < 2 * l + 1; m++) {
                    lm = l * l + m;
                    lmz = lm + 2 * l + 2;
                    dz_l[lm] += gauntz_l[lm] * cx_l[lmz];
                    dx_l[lm] += gauntxm_l[lm] * cx_l[lmz - 1];
                    dx_l[lm] += gauntxp_l[lm] * cx_l[lmz + 1];
                    lmz += 2 * (l - m);
                    dy_l[lm] += gauntym_l[lm] * cx_l[lmz - 1];
                    dy_l[lm] += gauntyp_l[lm] * cx_l[lmz + 1];
                }
            }
        }
    }
}

void compute_spline_bas(double *auxo_agi, double *coords, double *atm_coords,
                        int natm, int ngrids, int nrad, int nlm, double aparam,
                        double dparam) {
#pragma omp parallel
    {
        int g, lm, ag, at;
        int ir;
        int i;
        double dr, dr2, dr3;
        double diffr[3];
        double *auxo_i;
        double *ylm = (double *)malloc(nlm * sizeof(double));
        sphbuf buf = setup_sph_harm_buffer(nlm);
#pragma omp for
        for (ag = 0; ag < natm * ngrids; ag++) {
            at = ag / ngrids;
            g = ag % ngrids;
            auxo_i = auxo_agi + ag * 4 * nlm;
            diffr[0] = coords[3 * g + 0] - atm_coords[3 * at + 0];
            diffr[1] = coords[3 * g + 1] - atm_coords[3 * at + 1];
            diffr[2] = coords[3 * g + 2] - atm_coords[3 * at + 2];
            dr = sqrt(diffr[0] * diffr[0] + diffr[1] * diffr[1] +
                      diffr[2] * diffr[2]);
            diffr[0] /= dr;
            diffr[1] /= dr;
            diffr[2] /= dr;
            recursive_sph_harm(buf, diffr, ylm);
            ir = (int)floor(log(dr / aparam + 1) / dparam);
            ir = MIN(ir, nrad - 1);
            dr -= aparam * (exp(dparam * (double)ir) - 1);
            i = 0;
            dr2 = dr * dr;
            dr3 = dr2 * dr;
            for (lm = 0; lm < nlm; lm++) {
                auxo_i[i++] = ylm[lm];
                auxo_i[i++] = ylm[lm] * dr;
                auxo_i[i++] = ylm[lm] * dr2;
                auxo_i[i++] = ylm[lm] * dr3;
            }
        }
        free(ylm);
        free_sph_harm_buffer(buf);
    }
}

void compute_spline_bas_separate(double *auxo_agl, double *auxo_agp,
                                 double *coords, double *atm_coords, int natm,
                                 int ngrids, int nrad, int nlm, double aparam,
                                 double dparam) {
#pragma omp parallel
    {
        int g, lm, ag, at;
        int ir;
        double dr;
        double diffr[3];
        double *auxo_l;
        double *auxo_p;
        sphbuf buf = setup_sph_harm_buffer(nlm);
#pragma omp for
        for (ag = 0; ag < natm * ngrids; ag++) {
            at = ag / ngrids;
            g = ag % ngrids;
            auxo_l = auxo_agl + ag * nlm;
            auxo_p = auxo_agp + ag * 4;
            diffr[0] = coords[3 * g + 0] - atm_coords[3 * at + 0];
            diffr[1] = coords[3 * g + 1] - atm_coords[3 * at + 1];
            diffr[2] = coords[3 * g + 2] - atm_coords[3 * at + 2];
            dr = sqrt(diffr[0] * diffr[0] + diffr[1] * diffr[1] +
                      diffr[2] * diffr[2]);
            diffr[0] /= dr;
            diffr[1] /= dr;
            diffr[2] /= dr;
            recursive_sph_harm(buf, diffr, auxo_l);
            ir = (int)floor(log(dr / aparam + 1) / dparam);
            ir = MIN(ir, nrad - 1);
            dr -= aparam * (exp(dparam * (double)ir) - 1);
            auxo_p[0] = 1.0;
            auxo_p[1] = dr;
            auxo_p[2] = dr * dr;
            auxo_p[3] = auxo_p[2] * dr;
        }
        free_sph_harm_buffer(buf);
    }
}

void compute_spline_bas_separate_deriv(double *auxo_vagl, double *auxo_vagp,
                                       double *coords, double *atm_coords,
                                       int natm, int ngrids, int nrad, int nlm,
                                       double aparam, double dparam) {
#pragma omp parallel
    {
        int g, lm, ag, at;
        int ir;
        double dr;
        double diffr[3];
        double *auxo0_l;
        double *auxox_l;
        double *auxoy_l;
        double *auxoz_l;
        double *auxo0_p;
        double *auxox_p;
        double *auxoy_p;
        double *auxoz_p;
        sphbuf buf = setup_sph_harm_buffer(nlm);
        double *dylm = (double *)malloc(3 * nlm * sizeof(double));
        double ddr2;
        double ddr3;
        double invdr;
#pragma omp for
        for (ag = 0; ag < natm * ngrids; ag++) {
            at = ag / ngrids;
            g = ag % ngrids;
            auxo0_l = auxo_vagl + ag * nlm;
            auxox_l = auxo0_l + 1 * natm * ngrids * nlm;
            auxoy_l = auxo0_l + 2 * natm * ngrids * nlm;
            auxoz_l = auxo0_l + 3 * natm * ngrids * nlm;
            auxo0_p = auxo_vagp + ag * 4;
            auxox_p = auxo0_p + 1 * natm * ngrids * 4;
            auxoy_p = auxo0_p + 2 * natm * ngrids * 4;
            auxoz_p = auxo0_p + 3 * natm * ngrids * 4;
            diffr[0] = coords[3 * g + 0] - atm_coords[3 * at + 0];
            diffr[1] = coords[3 * g + 1] - atm_coords[3 * at + 1];
            diffr[2] = coords[3 * g + 2] - atm_coords[3 * at + 2];
            dr = sqrt(diffr[0] * diffr[0] + diffr[1] * diffr[1] +
                      diffr[2] * diffr[2]);
            invdr = 1.0 / (dr + 1e-10);
            diffr[0] /= dr;
            diffr[1] /= dr;
            diffr[2] /= dr;
            recursive_sph_harm_deriv(buf, diffr, auxo0_l, dylm);
            ir = (int)floor(log(dr / aparam + 1) / dparam);
            ir = MIN(ir, nrad - 1);
            dr -= aparam * (exp(dparam * (double)ir) - 1);
            auxo0_p[0] = 1.0;
            auxo0_p[1] = dr;
            auxo0_p[2] = dr * dr;
            auxo0_p[3] = dr * dr * dr;
            ddr2 = 2 * dr;
            ddr3 = 3 * dr * dr;
            auxox_p[0] = 0;
            auxox_p[1] = diffr[0];
            auxox_p[2] = ddr2 * diffr[0];
            auxox_p[3] = ddr3 * diffr[0];
            auxoy_p[0] = 0;
            auxoy_p[1] = diffr[1];
            auxoy_p[2] = ddr2 * diffr[1];
            auxoy_p[3] = ddr3 * diffr[1];
            auxoz_p[0] = 0;
            auxoz_p[1] = diffr[2];
            auxoz_p[2] = ddr2 * diffr[2];
            auxoz_p[3] = ddr3 * diffr[2];
            for (lm = 0; lm < nlm; lm++) {
                auxox_l[lm] = dylm[lm] * invdr;
                auxoy_l[lm] = dylm[lm + nlm] * invdr;
                auxoz_l[lm] = dylm[lm + 2 * nlm] * invdr;
            }
        }
        free_sph_harm_buffer(buf);
        free(dylm);
    }
}

void compute_num_spline_contribs_single(int *num_i, double *coords,
                                        double *atm_coord, double aparam,
                                        double dparam, int ngrids, int nrad,
                                        int *iatom_g, int iatom) {
    int i, ir, g, gp;
    double diffr[3];
    double dr;
    for (i = 0; i < nrad; i++) {
        num_i[i] = 0;
    }
    gp = 0;
    for (g = 0; g < ngrids; g++) {
        if (iatom_g == NULL || iatom_g[g] != iatom) {
            diffr[0] = coords[3 * g + 0] - atm_coord[0];
            diffr[1] = coords[3 * g + 1] - atm_coord[1];
            diffr[2] = coords[3 * g + 2] - atm_coord[2];
            dr = sqrt(diffr[0] * diffr[0] + diffr[1] * diffr[1] +
                      diffr[2] * diffr[2]);
            ir = (int)floor(log(dr / aparam + 1) / dparam);
            ir = MIN(ir, nrad - 1);
            num_i[ir] += 1;
        }
    }
}

void compute_num_spline_contribs_multi(spline_locator *spline_locs,
                                       double *coords, double *atm_coords,
                                       int ngrids, int natm, int *iatom_g) {
#pragma omp parallel
    {
        spline_locator my_loc;
#pragma omp for
        for (int a = 0; a < natm; a++) {
            my_loc = spline_locs[a];
            compute_num_spline_contribs_single(
                my_loc.num_i, coords, atm_coords + 3 * a, my_loc.aparam,
                my_loc.dparam, ngrids, my_loc.nrad, iatom_g, a);
        }
    }
}

/**
 * Compute the number of coordinates (coords) that fall within each
 * radial spline block. The result (num_ai) can then be used
 * in compute_spline_ind_order_new
 * to order the grids by their distance from the atom at atm_coord.
 * num_ai: The result, num_ai[ia * nrad + ir] contains the number of
 *         grid points that fall in spline index ir of atom with index ia.
 * coords: coords + 3 * g is the real-space coordinate of grid g.
 * atm_coords: atm_coords + 3 * ia is the real-space coordinate of atom ia.
 * aparam, dparam: The radial grid with index ir is
 *                 aparam * (exp(dparam * ir) - 1)
 * natm: Number of atoms
 * ngrids: Number of 3D real-space grids
 * nrad: Number of grids for the radial spline on each atom.
 * iatom_g: coords + 3 * g is part of the atomic grid belonging
 *          to the atom with index iatom_g[g]. If iatom_g is NULL,
 *          it is ignored. If it is not NULL, it is used to ignore
 *          on-site grids when constructing num_ai.
 */
void compute_num_spline_contribs_new(int *num_ai, double *coords,
                                     double *atm_coords, double aparam,
                                     double dparam, int natm, int ngrids,
                                     int nrad, int *iatom_g) {
#pragma omp parallel
    {
        spline_locator my_loc;
#pragma omp for
        for (int a = 0; a < natm; a++) {
            compute_num_spline_contribs_single(
                num_ai + a * nrad, coords, atm_coords + 3 * a, aparam, dparam,
                ngrids, nrad, iatom_g, a);
        }
    }
}

void initialize_spline_loc_list(spline_loc_list **llst_ptr, int natm,
                                int *nrads, double *aparams, double *dparams) {
    spline_loc_list *llist = malloc(sizeof(spline_loc_list));
    llist->natm = natm;
    llist->sloc_list = malloc(natm * sizeof(spline_locator));
    for (int a = 0; a < natm; a++) {
        llist->sloc_list[a].nrad = nrads[a];
        llist->sloc_list[a].aparam = aparams[a];
        llist->sloc_list[a].dparam = dparams[a];
        llist->sloc_list[a].loc_i = malloc((nrads[a] + 1) * sizeof(int));
        llist->sloc_list[a].num_i = malloc(nrads[a] * sizeof(int));
        llist->sloc_list[a].ngrids = 0;
        llist->sloc_list[a].buffer_size = 0;
        llist->sloc_list[a].rel_ord_coords = NULL;
        llist->sloc_list[a].ind_ord_fwd = NULL;
        llist->sloc_list[a].ind_ord_bwd = NULL;
    }
    llst_ptr[0] = llist;
}

void set_spline_locator_ngrids(spline_loc_list *llist, int ngrids_new) {
    for (int a = 0; a < llist->natm; a++) {
        spline_locator *sloc = llist->sloc_list + a;
        if (sloc->buffer_size < ngrids_new) {
            if (sloc->buffer_size > 0) {
                free(sloc->rel_ord_coords);
                free(sloc->ind_ord_fwd);
                free(sloc->ind_ord_bwd);
            }
            sloc->buffer_size = ngrids_new;
            sloc->ind_ord_fwd = (int *)malloc(sloc->buffer_size * sizeof(int));
            sloc->ind_ord_bwd = (int *)malloc(sloc->buffer_size * sizeof(int));
            sloc->rel_ord_coords =
                (double *)malloc(3 * sloc->buffer_size * sizeof(double));
        }
        sloc->ngrids = ngrids_new;
    }
}

void compute_num_spline_contribs(int *num_ai, double *coords,
                                 double *atm_coords, double aparam,
                                 double dparam, int natm, int ngrids, int nrad,
                                 int *ar_loc) {
#pragma omp parallel
    {
        int a, i, ir, g;
        double diffr[3];
        double dr;
        int *num_i;
#pragma omp for
        for (i = 0; i < natm * nrad; i++) {
            num_ai[i] = 0;
        }
#pragma omp for
        for (a = 0; a < natm; a++) {
            num_i = num_ai + a * nrad;
            for (g = 0; g < ngrids; g++) {
                if (ar_loc == NULL || (g < ar_loc[a]) || (g >= ar_loc[a + 1])) {
                    diffr[0] = coords[3 * g + 0] - atm_coords[3 * a + 0];
                    diffr[1] = coords[3 * g + 1] - atm_coords[3 * a + 1];
                    diffr[2] = coords[3 * g + 2] - atm_coords[3 * a + 2];
                    dr = sqrt(diffr[0] * diffr[0] + diffr[1] * diffr[1] +
                              diffr[2] * diffr[2]);
                    ir = (int)floor(log(dr / aparam + 1) / dparam);
                    ir = MIN(ir, nrad - 1);
                    num_i[ir] += 1;
                }
            }
        }
    }
}

#define ASSIGN_IND_ORDER                                                       \
    diffr[0] = coords[3 * g + 0] - atm_coord[0];                               \
    diffr[1] = coords[3 * g + 1] - atm_coord[1];                               \
    diffr[2] = coords[3 * g + 2] - atm_coord[2];                               \
    dr =                                                                       \
        sqrt(diffr[0] * diffr[0] + diffr[1] * diffr[1] + diffr[2] * diffr[2]); \
    ir = (int)floor(log(dr / aparam + 1) / dparam);                            \
    ir = MIN(ir, nrad - 1);                                                    \
    gp = loc_i[ir] + num_i_tmp[ir];                                            \
    ind_ord_fwd[gp] = g;                                                       \
    ind_ord_bwd[g] = gp;                                                       \
    coords_ord[3 * gp + 0] = coords[3 * g + 0];                                \
    coords_ord[3 * gp + 1] = coords[3 * g + 1];                                \
    coords_ord[3 * gp + 2] = coords[3 * g + 2];                                \
    num_i_tmp[ir] += 1;

void compute_spline_ind_order(int *loc_i, double *coords, double *atm_coord,
                              double *coords_ord, int *ind_ord_fwd,
                              int *ind_ord_bwd, double aparam, double dparam,
                              int ngrids, int nrad, int *ar_loc, int a) {
    int i, g, ir, gp;
    double dr;
    double diffr[3];
    int *num_i_tmp = malloc(nrad * sizeof(int));
    for (i = 0; i < nrad; i++) {
        num_i_tmp[i] = 0;
    }
    if (ar_loc == NULL) {
        for (g = 0; g < ngrids; g++) {
            ASSIGN_IND_ORDER;
        }
    } else {
        for (g = 0; g < ar_loc[a]; g++) {
            ASSIGN_IND_ORDER;
        }
        for (g = ar_loc[a + 1]; g < ngrids; g++) {
            ASSIGN_IND_ORDER;
        }
    }
    free(num_i_tmp);
}

/**
 * TODO this function should be modified to run in parallel.
 * This function sorts the grid coordinates (coords) in order of their
 * distance from the atomic coordinate (atm_coord).
 * loc_i: For each radial index ir, loc_i[ir] says where each batch
 *        corresponding to index ir of the radial spline is located
 *        in the coords_ord array. It is constructed in the
 *        _set_num_ai function of ciderpress.dft.lcao_interpolation.
 * coords: 3-D grid coordinations
 * atm_coord: Atomic coordinate
 * coords_ord: OUTPUT. The ordered coordinates are saved to this array.
 * ind_ord_fwd, ind_ord_bwd: OUTPUT. After execution, for x in 0, 1, 2,
 *         the following relationships hold:
 *         coords_ord[3 * g + x] == coords[3 * ind_ord_fwd[g] + x]
 *         coords_ord[3 * ind_ord_bwd[g] + x] == coords[3 * g + x]
 * aparam, dparam: The radial grid with index ir is
 *                 aparam * (exp(dparam * ir) - 1)
 * ngrids: Number of 3D real-space grids
 * nrad: Number of grids for the radial spline on each atom.
 * iatom_g: coords + 3 * g is part of the atomic grid belonging
 *          to the atom with index iatom_g[g]. If iatom_g is NULL,
 *          it is ignored. If it is not NULL, it is used to ignore
 *          on-site grids when constructing num_ai.
 * iatom: Index of the atom of for which the grids are being ordered.
 */
void compute_spline_ind_order_new(int *loc_i, double *coords, double *atm_coord,
                                  double *coords_ord, int *ind_ord_fwd,
                                  int *ind_ord_bwd, double aparam,
                                  double dparam, int ngrids, int nrad,
                                  int *iatom_g, int iatom) {
    int i, g, ir, gp;
    double dr;
    double diffr[3];
    int *num_i_tmp = malloc(nrad * sizeof(int));
    for (i = 0; i < nrad; i++) {
        num_i_tmp[i] = 0;
    }
    for (g = 0; g < ngrids; g++) {
        if (iatom_g == NULL || iatom_g[g] != iatom) {
            ASSIGN_IND_ORDER;
        }
    }
    free(num_i_tmp);
}

void compute_spline_ind_order_multi(int **loc_ai, double *coords,
                                    double *atm_coords, double **coords_ord,
                                    int **ind_ord_fwd, int **ind_ord_bwd) {}

/*
void compute_spline_bas_sep(
    double *auxo_agl, double *aux_agp, int *ind_ag,
    double *coords, double *atm_coords,
    int natm, int ngrids, int nrad, int nlm,
    double aparam, double dparam)
{
#pragma omp parallel
{
    int g, lm, ag, at;
    int ir;
    int i;
    double dr, dr2, dr3;
    double diffr[3];
    double *auxo_i;
    double *ylm = (double*)malloc(nlm*sizeof(double));
    sphbuf buf = setup_sph_harm_buffer(nlm);
#pragma omp for
    for (ag = 0; ag < natm*ngrids; ag++) {
        at = ag / ngrids;
        g = ag % ngrids;
        auxo_i = auxo_agi + ag*4*nlm;
        diffr[0] = coords[3*g+0] - atm_coords[3*at+0];
        diffr[1] = coords[3*g+1] - atm_coords[3*at+1];
        diffr[2] = coords[3*g+2] - atm_coords[3*at+2];
        dr = sqrt(diffr[0]*diffr[0] + diffr[1]*diffr[1] + diffr[2]*diffr[2]);
        diffr[0] /= dr;
        diffr[1] /= dr;
        diffr[2] /= dr;
        recursive_sph_harm(buf, diffr, ylm);
        ir = (int) floor(log(dr / aparam + 1) / dparam);
        ir = MIN(ir, nrad-1);
        dr -= aparam * (exp(dparam * (double)ir) - 1);
        ind_ag[ag] = ir;
        i = 0;
        dr2 = dr * dr;
        dr3 = dr2 * dr;
        for (lm = 0; lm < nlm; lm++) {
            auxo_i[i++] = ylm[lm];
            auxo_i[i++] = ylm[lm] * dr;
            auxo_i[i++] = ylm[lm] * dr2;
            auxo_i[i++] = ylm[lm] * dr3;
        }
    }
    free(ylm);
    free_sph_harm_buffer(buf);
}
}


void compute_mol_convs_sep(
    double *f_gq, double *f_rlpq,
    double *auxo_gi, int *loc_i,
    int *ind_ord_fwd,
    int nalpha, int nrad,
    int ngrids, int nlm, int maxg
)
{
#pragma omp parallel
{
    int nfpr = nalpha * nlm * 4;
    int nfpa = nlm * 4;
    int nalm = nalpha * nlm;
    int nap = nalpha * 4;
    int np = 4;
    int q, ir, g;
    double *f_qlp;
    double BETA = 0;
    double ALPHA = 1;
    char NTRANS = 'N';
    char TRANS = 'T';
    double *f_gq_buf = malloc(nalpha * maxg * sizeof(double));
    double *f_gqp_buf = malloc(nalpha * 4 * maxg * sizeof(double));
    double *f_gq_tmp, *auxo_glp;
    int gp, gq_ind, ng;
#pragma omp for schedule(dynamic, 1)
    for (ir = 0; ir < nrad-1; ir++) {
        f_qlp = f_rqlp + ir * nfpr;
        auxo_glp = auxo_gi + loc_i[ir] * nfpa;
        ng = loc_i[ir+1] - loc_i[ir];
        dgemm_(&NTRANS, &NTRANS, &nap, &ng, &np,
               &ALPHA, f_lpq, &np,
               auxo_gl, &np, &BETA,
               f_gqp_buf, &nap);
        dgemm_(&TRANS, &NTRANS, &nalpha, &ng, &nlm,
               &ALPHA, f_glq_buf, &nlm,
               auxo_gl, &nlm, &BETA,
               f_gq_buf, &nalpha);
        //dgemm_(&TRANS, &NTRANS, &nalpha, &ng, &nfpa,
        //       &ALPHA, f_qlp, &nfpa,
        //       auxo_glp, &nfpa, &BETA,
        //       f_gq_buf, &nalpha);
        gq_ind = 0;
        for (g=loc_i[ir]; g<loc_i[ir+1]; g++) {
            gp = ind_ord_fwd[g];
            f_gq_tmp = f_gq + gp*nalpha;
            for (q=0; q<nalpha; q++, gq_ind++) {
                f_gq_tmp[q] += f_gq_buf[gq_ind];
            }
        }
    }
    free(f_gq_buf);
}
}
*/

void compute_spline_bas_single_rad(double *auxo_agi, double *coords,
                                   double *atm_coords, int ir, int ngrids,
                                   int nrad, int nlm, double aparam,
                                   double dparam) {
    int g, lm;
    double r0 = aparam * (exp(dparam * (double)ir) - 1);
    int i;
    double dr, dr2, dr3;
    double diffr[3];
    double *auxo_i;
    double *ylm = (double *)malloc(nlm * sizeof(double));
    sphbuf buf = setup_sph_harm_buffer(nlm);
    for (g = 0; g < ngrids; g++) {
        auxo_i = auxo_agi + g * 4 * nlm;
        diffr[0] = coords[3 * g + 0] - atm_coords[0];
        diffr[1] = coords[3 * g + 1] - atm_coords[1];
        diffr[2] = coords[3 * g + 2] - atm_coords[2];
        dr = sqrt(diffr[0] * diffr[0] + diffr[1] * diffr[1] +
                  diffr[2] * diffr[2]);
        diffr[0] /= dr;
        diffr[1] /= dr;
        diffr[2] /= dr;
        recursive_sph_harm(buf, diffr, ylm);
        dr -= r0;
        i = 0;
        dr2 = dr * dr;
        dr3 = dr2 * dr;
        for (lm = 0; lm < nlm; lm++, i += 4) {
            auxo_i[i + 0] = ylm[lm];
            auxo_i[i + 1] = ylm[lm] * dr;
            auxo_i[i + 2] = ylm[lm] * dr2;
            auxo_i[i + 3] = ylm[lm] * dr3;
        }
    }
    free(ylm);
    free_sph_harm_buffer(buf);
}

void compute_mol_convs_single(double *f_gq, double *f_rqlp, int *loc_i,
                              int *ind_ord_fwd, double *coords,
                              double *atm_coord, int nalpha, int nrad,
                              int ngrids, int nlm, int maxg, double aparam,
                              double dparam) {
#pragma omp parallel
    {
        int nfpr = nalpha * nlm * 4;
        int nfpa = nlm * 4;
        int q, ir, g;
        double *f_qlp;
        double BETA = 0;
        double ALPHA = 1;
        char NTRANS = 'N';
        char TRANS = 'T';
        double *auxo_gi = malloc(nlm * 4 * maxg * sizeof(double));
        double *f_gq_buf = malloc(nalpha * maxg * sizeof(double));
        double *f_gq_tmp;
        int gp, gq_ind, ng;
#pragma omp for schedule(dynamic, 1)
        for (ir = 0; ir < nrad - 1; ir++) {
            f_qlp = f_rqlp + ir * nfpr;
            ng = loc_i[ir + 1] - loc_i[ir];
            compute_spline_bas_single_rad(auxo_gi, coords + 3 * loc_i[ir],
                                          atm_coord, ir, ng, nrad, nlm, aparam,
                                          dparam);
            dgemm_(&TRANS, &NTRANS, &nalpha, &ng, &nfpa, &ALPHA, f_qlp, &nfpa,
                   auxo_gi, &nfpa, &BETA, f_gq_buf, &nalpha);
            gq_ind = 0;
            for (g = loc_i[ir]; g < loc_i[ir + 1]; g++) {
                gp = ind_ord_fwd[g];
                f_gq_tmp = f_gq + gp * nalpha;
                for (q = 0; q < nalpha; q++, gq_ind++) {
                    f_gq_tmp[q] += f_gq_buf[gq_ind];
                }
            }
        }
        free(f_gq_buf);
        free(auxo_gi);
    }
}

void compute_mol_convs_single_new(double *f_gq, double *f_rlpq, double *auxo_gl,
                                  double *auxo_gp, int *loc_i, int *ind_ord_fwd,
                                  int nalpha, int nrad, int ngrids, int nlm,
                                  int maxg) {
#pragma omp parallel
    {
        int nfpr = nalpha * nlm * 4;
        int n_pq = nalpha * 4;
        int q, ir, g, p;
        double *f_lpq;
        double *auxo_tmp_gl;
        double BETA = 0;
        double ALPHA = 1;
        char NTRANS = 'N';
        double *f_gpq_buf =
            (double *)malloc(nalpha * 4 * maxg * sizeof(double));
        double *f_gq_tmp;
        double spline_contrib;
        int gp, gq_ind, ng;
#pragma omp for schedule(dynamic, 1)
        for (ir = 0; ir < nrad - 1; ir++) {
            f_lpq = f_rlpq + ir * nfpr;
            ng = loc_i[ir + 1] - loc_i[ir];
            auxo_tmp_gl = auxo_gl + loc_i[ir] * nlm;
            dgemm_(&NTRANS, &NTRANS, &n_pq, &ng, &nlm, &ALPHA, f_lpq, &n_pq,
                   auxo_tmp_gl, &nlm, &BETA, f_gpq_buf, &n_pq);
            gq_ind = 0;
            for (g = loc_i[ir]; g < loc_i[ir + 1]; g++) {
                gp = ind_ord_fwd[g];
                f_gq_tmp = f_gq + gp * nalpha;
                for (p = 0; p < 4; p++) {
                    spline_contrib = auxo_gp[g * 4 + p];
                    for (q = 0; q < nalpha; q++, gq_ind++) {
                        f_gq_tmp[q] += f_gpq_buf[gq_ind] * spline_contrib;
                    }
                }
            }
        }
        free(f_gpq_buf);
    }
}

/*void compute_mol_convs_single_new(
    double *f_gq, double *f_rlpq, double *auxo_gl, double *auxo_gp,
    int *loc_i, int *ind_ord_fwd,
    int nalpha, int nrad, int ngrids, int nlm, int maxg
)
{
#pragma omp parallel
{
    int nfpr = nalpha * nlm * 4;
    int n_pq = nalpha * 4;
    int q, ir, g, p;
    double *f_lpq;
    double *auxo_tmp_gl;
    double BETA = 0;
    double ALPHA = 1;
    char NTRANS = 'N';
    double *f_gpq_buf = (double*) malloc(nalpha * 4 * maxg * sizeof(double));
    double *f_gq_tmp;
    double spline_contrib;
    int gp, gq_ind, ng;
#pragma omp for schedule(dynamic, 1)
    for (ir = 0; ir < nrad-1; ir++) {
        f_lpq = f_rlpq + ir * nfpr;
        ng = loc_i[ir + 1] - loc_i[ir];
        auxo_tmp_gl = auxo_gl + loc_i[ir] * nlm;
        dgemm_(&NTRANS, &NTRANS, &n_pq, &ng, &nlm, &ALPHA, f_lpq, &n_pq,
               auxo_tmp_gl, &nlm, &BETA, f_gpq_buf, &n_pq);
        gq_ind = 0;
        for (g = loc_i[ir]; g < loc_i[ir + 1]; g++) {
            gp = ind_ord_fwd[g];
            f_gq_tmp = f_gq + gp * nalpha;
            for (p = 0; p < 4; p++) {
                spline_contrib = auxo_gp[g * 4 + p];
                for (q = 0; q < nalpha; q++, gq_ind++) {
                    f_gq_tmp[q] += f_gpq_buf[gq_ind] * spline_contrib;
                }
            }
        }
    }
    free(f_gpq_buf);
}
}*/

void compute_pot_convs_single_new(double *f_gq, double *f_rlpq, double *auxo_gl,
                                  double *auxo_gp, int *loc_i, int *ind_ord_fwd,
                                  int nalpha, int nrad, int ngrids, int nlm,
                                  int maxg) {
#pragma omp parallel
    {
        int nfpr = nalpha * nlm * 4;
        int n_pq = nalpha * 4;
        int q, ir, g, p;
        double *f_lpq;
        double *auxo_tmp_gl;
        // TODO beta should be 1, not 0, when doing grid batches
        double BETA = 0;
        double ALPHA = 1;
        char NTRANS = 'N';
        char TRANS = 'T';
        double *f_gpq_buf =
            (double *)malloc(nalpha * 4 * maxg * sizeof(double));
        double *f_gq_tmp;
        double spline_contrib;
        int gp, gq_ind, ng;
#pragma omp for schedule(dynamic, 1)
        for (ir = 0; ir < nrad - 1; ir++) {
            f_lpq = f_rlpq + ir * nfpr;
            ng = loc_i[ir + 1] - loc_i[ir];
            auxo_tmp_gl = auxo_gl + loc_i[ir] * nlm;
            gq_ind = 0;
            for (g = loc_i[ir]; g < loc_i[ir + 1]; g++) {
                gp = ind_ord_fwd[g];
                f_gq_tmp = f_gq + gp * nalpha;
                for (p = 0; p < 4; p++) {
                    spline_contrib = auxo_gp[g * 4 + p];
                    for (q = 0; q < nalpha; q++, gq_ind++) {
                        f_gpq_buf[gq_ind] = spline_contrib * f_gq_tmp[q];
                    }
                }
            }
            dgemm_(&NTRANS, &TRANS, &n_pq, &nlm, &ng, &ALPHA, f_gpq_buf, &n_pq,
                   auxo_tmp_gl, &nlm, &BETA, f_lpq, &n_pq);
        }
        free(f_gpq_buf);
    }
}

/*
void compute_mol_convs_(
    double *f_gq, double *f_arlpq, int **loc_a_i, int **ord_a_i,
    double *y_gl, double *d_gp,
    int natm, int nalpha, int nrad, int ngrids, int nlm, int maxg
)
{
    double *f_agq = (double*) malloc(natm * ngrids * nalpha * sizeof(double));
#pragma omp parallel
{
    double *f_gpq_buf;
    double *f_gq_buf;
    double *f_gq_tmp;
    int gq_ind;
    int g, q, gp;
    int *ind_ord_fwd;
    int *loc_i;
    double BETA = 0;
    double ALPHA = 1;
    char NTRANS = 'N';
    int n_l = nlm;
    int n_pq;
    int nfpa;
    int n_lpq;
    int ng;
    int ir;
    int ia;
    double *f_lpq;
#pragma omp for schedule(dynamic, 4)
    for (int iair = 0; iair < natm * nrad; iair++) {
        ia = iair / nrad;
        ir = iair % nrad;
        f_lpq = f_arlpq + iair * n_lpq;
        ng = loc_a_i[ia][ir+1] - loc_a_i[ia][ir];
        dgemm_(&NTRANS, &NTRANS, &nalpha, &ng, &nfpa, &ALPHA, f_lpq, &n_pq,
               y_gl, &n_l, &BETA, f_gpq_buf, &nalpha);
        for (int g = 0; g < ng; g++) {
            for (int p = 0; p < 4; p++) {
                for (int q = 0; q < nalpha; q++) {
                    f_gq_buf[g * nalpha + q] += f_gpq_buf[g * nalpha * 4 + p *
nalpha + q]
                                                * d_gp[g * 4 + p];
                }
            }
        }
        ind_ord_fwd = ord_a_i[ia];
        gq_ind = 0;
        for (g = loc_i[ir]; g < loc_i[ir + 1]; g++) {
            gp = ind_ord_fwd[g];
            f_gq_tmp = f_gq + gp * nalpha;
            for (q = 0; q < nalpha; q++, gq_ind++) {
                f_gq_tmp[q] += f_gq_buf[gq_ind];
            }
        }
    }
}
#pragma omp parallel for
    for (int g = 0; g < ngrids; g++) {
        for (int ia = 0; ia < natm; ia++) {
            for (int q = 0; q < nalpha; q++) {
                f_gq[g*nalpha + q] += f_agq[(ia * ngrids + g) * nalpha + q];
            }
        }
    }
}
*/

void project_conv_to_spline(double *f_arlpq, double *f_uq, double *w_rsp,
                            atc_basis_set *atco, int nalpha, int nrad, int nlm,
                            int orb_stride, int spline_stride,
                            int offset_spline, int offset_orb) {
    if (spline_stride < nalpha || orb_stride < nalpha) {
        printf("INTERNAL ERROR, stride < nalpha");
        exit(-1);
    }
#pragma omp parallel
    {
        int p, q, ia, r, nm, m;
        int natm = atco->natm;
        double *w_p, *inp_q, *out_q, *f_lpq;
        int *ao_loc = atco->ao_loc;
        int *atom_loc_ao = atco->atom_loc_ao;
        int l;
#pragma omp for schedule(dynamic, 4)
        for (int iar = 0; iar < natm * nrad; iar++) {
            ia = iar / nrad;
            r = iar % nrad;
            f_lpq = f_arlpq + iar * nlm * 4 * spline_stride;
            for (int ish = atom_loc_ao[ia]; ish < atom_loc_ao[ia + 1]; ish++) {
                l = atco->bas[ish * BAS_SLOTS + ANG_OF];
                w_p = w_rsp + r * 4 * atco->nbas + ish * 4;
                nm = ao_loc[ish + 1] - ao_loc[ish];
                for (m = 0; m < nm; m++) {
                    out_q =
                        f_lpq + (l * l + m) * 4 * spline_stride + offset_spline;
                    inp_q = f_uq + (ao_loc[ish] + m) * orb_stride + offset_orb;
                    for (p = 0; p < 4; p++) {
                        for (q = 0; q < nalpha; q++) {
                            out_q[q] += inp_q[q] * w_p[p];
                        }
                        out_q += spline_stride;
                    }
                }
            }
        }
    }
}

void project_spline_to_conv(double *f_arlpq, double *f_uq, double *w_rsp,
                            atc_basis_set *atco, int nalpha, int nrad, int nlm,
                            int orb_stride, int spline_stride,
                            int offset_spline, int offset_orb) {
    if (spline_stride < nalpha || orb_stride < nalpha) {
        printf("INTERNAL ERROR, stride < nalpha");
        exit(-1);
    }
#pragma omp parallel
    {
        int p, q, ia, iar, ish, nm, m;
        int natm = atco->natm;
        double *w_p, *inp_q, *out_q, *f_lpq;
        int *ao_loc = atco->ao_loc;
        int l;
#pragma omp for schedule(dynamic, 4)
        for (int ish = 0; ish < atco->nbas; ish++) {
            // ia = iash / atco->nbas;
            // ish = iash % atco->nbas;
            l = atco->bas[ish * BAS_SLOTS + ANG_OF];
            ia = atco->bas[ish * BAS_SLOTS + ATOM_OF];
            for (int r = 0; r < nrad; r++) {
                iar = ia * nrad + r;
                f_lpq = f_arlpq + iar * nlm * 4 * spline_stride;
                w_p = w_rsp + r * 4 * atco->nbas + ish * 4;
                nm = ao_loc[ish + 1] - ao_loc[ish];
                for (m = 0; m < nm; m++) {
                    inp_q =
                        f_lpq + (l * l + m) * 4 * spline_stride + offset_spline;
                    out_q = f_uq + (ao_loc[ish] + m) * orb_stride + offset_orb;
                    for (p = 0; p < 4; p++) {
                        for (q = 0; q < nalpha; q++) {
                            out_q[q] += inp_q[q] * w_p[p];
                        }
                        inp_q += spline_stride;
                    }
                }
            }
        }
    }
}

void fill_l1_coeff_fwd(double *f_u, double *d_uv, double *gaunt_vl, int nlm,
                       atc_basis_set *atco0, atc_basis_set *atco1, int stride1,
                       int offset1, int stride2, int offset2) {
    f_u = f_u + offset1;
#pragma omp parallel
    {
        double *dx_u = d_uv + offset2;
        double *dy_u = d_uv + offset2 + 1;
        double *dz_u = d_uv + offset2 + 2;
        int ind;

        int natm = atco0->natm;

        int at, l, i0, j0, lm, m, nm;
        int ish, jsh, ish0, ish1, jsh0, jsh1;
        double *gauntxm_l = gaunt_vl + 0 * nlm;
        double *gauntxp_l = gaunt_vl + 1 * nlm;
        double *gauntym_l = gaunt_vl + 2 * nlm;
        double *gauntyp_l = gaunt_vl + 3 * nlm;
        double *gauntz_l = gaunt_vl + 4 * nlm;
#pragma omp for schedule(dynamic, 4)
        for (at = 0; at < natm; at++) {
            // ish0 = atom_loc1_ao[at];
            ish1 = atco0->atom_loc_ao[at + 1];
            jsh0 = atco1->atom_loc_ao[at];
            jsh1 = atco1->atom_loc_ao[at + 1];
            ish0 = ish1 - (jsh1 - jsh0);
            jsh = jsh0;
            for (ish = ish0; ish < ish1; ish++, jsh++) {
                l = atco1->bas[jsh * BAS_SLOTS + ANG_OF];
                i0 = atco0->ao_loc[ish];
                j0 = atco1->ao_loc[jsh];
                nm = atco1->ao_loc[jsh + 1] - j0;
                for (m = 0; m < nm; m++) {
                    lm = l * l + m;
                    ind = (j0 + m) * stride2;
                    dz_u[ind] += gauntz_l[lm] * f_u[(i0 + m + 1) * stride1];
                    dx_u[ind] += gauntxm_l[lm] * f_u[(i0 + m) * stride1];
                    dx_u[ind] += gauntxp_l[lm] * f_u[(i0 + m + 2) * stride1];
                    dy_u[ind] +=
                        gauntym_l[lm] * f_u[(i0 + 2 * l - m) * stride1];
                    dy_u[ind] +=
                        gauntyp_l[lm] * f_u[(i0 + 2 * l + 2 - m) * stride1];
                }
            }
        }
    }
}

void fill_l1_coeff_bwd(double *f_u, double *d_uv, double *gaunt_vl, int nlm,
                       atc_basis_set *atco0, atc_basis_set *atco1, int stride1,
                       int offset1, int stride2, int offset2) {
    f_u = f_u + offset1;
#pragma omp parallel
    {
        double *dx_u = d_uv + offset2;
        double *dy_u = d_uv + offset2 + 1;
        double *dz_u = d_uv + offset2 + 2;
        int ind;

        int natm = atco0->natm;

        int at, l, i0, j0, lm, m, nm;
        int ish, jsh, ish0, ish1, jsh0, jsh1;
        double *gauntxm_l = gaunt_vl + 0 * nlm;
        double *gauntxp_l = gaunt_vl + 1 * nlm;
        double *gauntym_l = gaunt_vl + 2 * nlm;
        double *gauntyp_l = gaunt_vl + 3 * nlm;
        double *gauntz_l = gaunt_vl + 4 * nlm;
#pragma omp for schedule(dynamic, 4)
        for (at = 0; at < natm; at++) {
            // ish0 = atom_loc1_ao[at];
            ish1 = atco0->atom_loc_ao[at + 1];
            jsh0 = atco1->atom_loc_ao[at];
            jsh1 = atco1->atom_loc_ao[at + 1];
            ish0 = ish1 - (jsh1 - jsh0);
            jsh = jsh0;
            for (ish = ish0; ish < ish1; ish++, jsh++) {
                l = atco1->bas[jsh * BAS_SLOTS + ANG_OF];
                i0 = atco0->ao_loc[ish];
                j0 = atco1->ao_loc[jsh];
                nm = atco1->ao_loc[jsh + 1] - j0;
                for (m = 0; m < nm; m++) {
                    lm = l * l + m;
                    ind = (j0 + m) * stride2;
                    f_u[(i0 + m + 1) * stride1] += gauntz_l[lm] * dz_u[ind];
                    f_u[(i0 + m) * stride1] += gauntxm_l[lm] * dx_u[ind];
                    f_u[(i0 + m + 2) * stride1] += gauntxp_l[lm] * dx_u[ind];
                    f_u[(i0 + 2 * l - m) * stride1] +=
                        gauntym_l[lm] * dy_u[ind];
                    f_u[(i0 + 2 * l + 2 - m) * stride1] +=
                        gauntyp_l[lm] * dy_u[ind];
                }
            }
        }
    }
}

void add_lp1_term_fwd(double *f, double *coords, double *atom_coord, int n,
                      int ig, int ix, int iy, int iz, int nf) {
#pragma omp parallel
    {
        int g;
        double dx, dy, dz;
        double *f_q;
#pragma omp for
        for (g = 0; g < n; g++) {
            dx = coords[3 * g + 0] - atom_coord[0];
            dy = coords[3 * g + 1] - atom_coord[1];
            dz = coords[3 * g + 2] - atom_coord[2];
            f_q = f + nf * g;
            f_q[ix] += dx * f_q[ig];
            f_q[iy] += dy * f_q[ig];
            f_q[iz] += dz * f_q[ig];
            f_q[ig] = 0.0;
        }
    }
}

void add_lp1_term_grad(double *out, double *f0, double *f1, int *atm_g, int ia,
                       int natm, int n, int ig, int ix, int iy, int iz,
                       int nf) {
#pragma omp parallel
    {
        int g;
        double *f0_q;
        double *f1_q;
        double fac;
        int ib;
        double *tmp = (double *)malloc(natm * 3 * sizeof(double));
        for (int a = 0; a < 3 * natm; a++) {
            tmp[a] = 0;
        }
#pragma omp for
        for (g = 0; g < n; g++) {
            f0_q = f0 + nf * g;
            f1_q = f1 + nf * g;
            ib = atm_g[g];
            fac = f0_q[ig] * f1_q[ix];
            tmp[3 * ib + 0] += fac;
            tmp[3 * ia + 0] -= fac;
            fac = f0_q[ig] * f1_q[iy];
            tmp[3 * ib + 1] += fac;
            tmp[3 * ia + 1] -= fac;
            fac = f0_q[ig] * f1_q[iz];
            tmp[3 * ib + 2] += fac;
            tmp[3 * ia + 2] -= fac;
        }
#pragma omp critical
        {
            for (int a = 0; a < 3 * natm; a++) {
                out[a] += tmp[a];
            }
        }
        free(tmp);
    }
}

void add_lp1_term_onsite_fwd(double *f, double *coords, int natm,
                             double *atom_coords, int *ar_loc, int ig, int ix,
                             int iy, int iz, int nf) {
    if (ar_loc == NULL) {
        exit(-1);
    }
#pragma omp parallel
    {
        int g, a;
        double dx, dy, dz;
        double *f_q;
#pragma omp for
        for (a = 0; a < natm; a++) {
            for (g = ar_loc[a]; g < ar_loc[a + 1]; g++) {
                dx = coords[3 * g + 0] - atom_coords[3 * a + 0];
                dy = coords[3 * g + 1] - atom_coords[3 * a + 1];
                dz = coords[3 * g + 2] - atom_coords[3 * a + 2];
                f_q = f + nf * g;
                f_q[ix] += dx * f_q[ig];
                f_q[iy] += dy * f_q[ig];
                f_q[iz] += dz * f_q[ig];
                f_q[ig] = 0.0;
            }
        }
    }
}

void add_lp1_onsite_new_fwd(double *f, double *rads, int *rad_loc, int nrad,
                            double *dirs, int *dir_loc, int nf, int ig, int ix,
                            int iy, int iz) {
#pragma omp parallel
    {
        int ir, g;
        int ng;
        double radius;
        double dx, dy, dz;
        int curr_dloc;
        double *f_q;
#pragma omp for schedule(dynamic, 8)
        for (ir = 0; ir < nrad; ir++) {
            ng = rad_loc[ir + 1] - rad_loc[ir];
            for (g = 0; g < ng; g++) {
                curr_dloc = dir_loc[ir] + g;
                dx = rads[ir] * dirs[3 * curr_dloc + 0];
                dy = rads[ir] * dirs[3 * curr_dloc + 1];
                dz = rads[ir] * dirs[3 * curr_dloc + 2];
                f_q = f + nf * (rad_loc[ir] + g);
                f_q[ix] += dx * f_q[ig];
                f_q[iy] += dy * f_q[ig];
                f_q[iz] += dz * f_q[ig];
                f_q[ig] = 0.0;
            }
        }
    }
}

void add_lp1_onsite_new_bwd(double *f, double *rads, int *rad_loc, int nrad,
                            double *dirs, int *dir_loc, int nf, int ig, int ix,
                            int iy, int iz) {
#pragma omp parallel
    {
        int ir, g;
        int ng;
        double radius;
        double dx, dy, dz;
        int curr_dloc;
        double *f_q;
#pragma omp for schedule(dynamic, 8)
        for (ir = 0; ir < nrad; ir++) {
            ng = rad_loc[ir + 1] - rad_loc[ir];
            for (g = 0; g < ng; g++) {
                curr_dloc = dir_loc[ir] + g;
                dx = rads[ir] * dirs[3 * curr_dloc + 0];
                dy = rads[ir] * dirs[3 * curr_dloc + 1];
                dz = rads[ir] * dirs[3 * curr_dloc + 2];
                f_q = f + nf * (rad_loc[ir] + g);
                f_q[ig] = 0.0;
                f_q[ig] += dx * f_q[ix];
                f_q[ig] += dy * f_q[iy];
                f_q[ig] += dz * f_q[iz];
            }
        }
    }
}

void add_lp1_term_bwd(double *f, double *coords, double *atom_coord, int n,
                      int ig, int ix, int iy, int iz, int nf) {
#pragma omp parallel
    {
        int g;
        double dx, dy, dz;
        double *f_q;
#pragma omp for
        for (g = 0; g < n; g++) {
            dx = coords[3 * g + 0] - atom_coord[0];
            dy = coords[3 * g + 1] - atom_coord[1];
            dz = coords[3 * g + 2] - atom_coord[2];
            f_q = f + nf * g;
            f_q[ig] = 0.0;
            f_q[ig] += dx * f_q[ix];
            f_q[ig] += dy * f_q[iy];
            f_q[ig] += dz * f_q[iz];
        }
    }
}

void add_lp1_term_onsite_bwd(double *f, double *coords, int natm,
                             double *atom_coords, int *ar_loc, int ig, int ix,
                             int iy, int iz, int nf) {
#pragma omp parallel
    {
        int g, a;
        double dx, dy, dz;
        double *f_q;
#pragma omp for
        for (a = 0; a < natm; a++) {
            for (g = ar_loc[a]; g < ar_loc[a + 1]; g++) {
                dx = coords[3 * g + 0] - atom_coords[3 * a + 0];
                dy = coords[3 * g + 1] - atom_coords[3 * a + 1];
                dz = coords[3 * g + 2] - atom_coords[3 * a + 2];
                f_q = f + nf * g;
                f_q[ig] = 0.0;
                f_q[ig] += dx * f_q[ix];
                f_q[ig] += dy * f_q[iy];
                f_q[ig] += dz * f_q[iz];
            }
        }
    }
}

// TODO might want to move this somewhere else
void contract_grad_terms_old(double *excsum, double *f_g, int natm, int a,
                             int v, int ngrids, int *ga_loc) {
    double *tmp = (double *)calloc(natm, sizeof(double));
    int ib;
#pragma omp parallel
    {
        int ia;
        int g;
#pragma omp for
        for (ia = 0; ia < natm; ia++) {
            for (g = ga_loc[ia]; g < ga_loc[ia + 1]; g++) {
                tmp[ia] += f_g[g];
            }
        }
    }
    for (ib = 0; ib < natm; ib++) {
        excsum[ib * 3 + v] += tmp[ib];
        excsum[a * 3 + v] -= tmp[ib];
    }
    free(tmp);
}

void contract_grad_terms_serial(double *excsum, double *f_g, int natm, int a,
                                int v, int ngrids, int *atm_g) {
    double *tmp = (double *)calloc(natm, sizeof(double));
    int ib;
    int ia;
    int g;
    for (g = 0; g < ngrids; g++) {
        tmp[atm_g[g]] += f_g[g];
    }
    for (ib = 0; ib < natm; ib++) {
        excsum[ib * 3 + v] += tmp[ib];
        excsum[a * 3 + v] -= tmp[ib];
    }
    free(tmp);
}

void contract_grad_terms_parallel(double *excsum, double *f_g, int natm, int a,
                                  int v, int ngrids, int *atm_g) {
    double *tmp_priv;
    double total = 0;
#pragma omp parallel
    {
        const int nthreads = omp_get_num_threads();
        const int ithread = omp_get_thread_num();
        const int ngrids_local = (ngrids + nthreads - 1) / nthreads;
        const int ig0 = ithread * ngrids_local;
        const int ig1 = MIN(ig0 + ngrids_local, ngrids);
#pragma omp single
        { tmp_priv = (double *)calloc(nthreads * natm, sizeof(double)); }
#pragma omp barrier
        double *my_tmp = tmp_priv + ithread * natm;
        int ib;
        int it;
        int g;
        for (g = ig0; g < ig1; g++) {
            my_tmp[atm_g[g]] += f_g[g];
        }
#pragma omp barrier
#pragma omp for reduction(+ : total)
        for (ib = 0; ib < natm; ib++) {
            for (it = 0; it < nthreads; it++) {
                excsum[ib * 3 + v] += tmp_priv[it * natm + ib];
                total += tmp_priv[it * natm + ib];
            }
        }
    }
    excsum[a * 3 + v] -= total;
    free(tmp_priv);
}
