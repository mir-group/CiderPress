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

#include "convolutions.h"
#include "fblas.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/** int_0^infty dr r^(2*l+2) exp(-a*r^2) */
inline double gauss_integral(int l, double a) {
    return 0.5 * pow(a, -1.5 - l) * tgamma(1.5 + l);
}

// TODO all gauss integrals below should be inline for efficiency,
// TODO but they cannot be made inline and then assigned to function pointers.
/**
 * Computes I0 with
 * I0 = int_0_infty dr r^(2*l+2) exp(-(beta+expj) r^2)
 * with beta = expi * alpha / (expi + alpha)
 * l (int) : Angular momentum number
 * alpha (double) : convolution exponent
 * expi (double) : exponent of Gaussian to be convolved
 * expj (double) : exponent of Gaussian to project onto
 */
double gauss_i0(int l, double alpha, double expi, double expj) {
    double expi_conv = expi * alpha / (expi + alpha);
    return gauss_integral(l, expi_conv + expj);
}

/**
 * Computes alpha * I0. See gauss_i0 for details.
 */
double gauss_ai0(int l, double alpha, double expi, double expj) {
    return alpha * gauss_i0(l, alpha, expi, expj);
}

/**
 * Computes -dI0/dalpha with
 * I0 = int_0_infty dr r^(2*l+2) exp(-(beta+expj) r^2)
 * with beta = expi * alpha / (expi + alpha)
 * l (int) : Angular momentum number
 * alpha (double) : convolution exponent
 * expi (double) : exponent of Gaussian to be convolved
 * expj (double) : exponent of Gaussian to project onto
 */
double gauss_dida(int l, double alpha, double expi, double expj) {
    double expi_conv = expi * alpha / (expi + alpha);
    double coefi = (-l / alpha + (1.5 + l) / (alpha + expi));
    // TODO I think (1.5 + l) / (expi_conv + expj) can be removed
    // if l is changed to l + 1 in gauss_integral argument. This will
    // make "on-the-fly" integral generation for vi,j,k quite easy,
    // which could in turn save a lot of memory for small computational
    // overhead. (Might even speed things up due to memory accesses?)
    coefi += (1.5 + l) / (expi_conv + expj) * expi * expi /
             ((alpha + expi) * (alpha + expi));
    return coefi * gauss_integral(l, expi_conv + expj);
}

/**
 * Computes alpha * dI0/dalpha. See gauss_dida for details.
 */
double gauss_adida(int l, double alpha, double expi, double expj) {
    return alpha * gauss_dida(l, alpha, expi, expj);
}

/**
 * Computes alpha^2 * dI0/dalpha. See gauss_dida for details.
 */
double gauss_a2dida(int l, double alpha, double expi, double expj) {
    return alpha * alpha * gauss_dida(l, alpha, expi, expj);
}

double gauss_lapli0(int l, double alpha, double expi, double expj) {
    return 4 * gauss_a2dida(l, alpha, expi, expj) -
           2 * gauss_ai0(l, alpha, expi, expj);
}

/**
 * Computes -expi / (expi + alpha) * I+ with
 * I+ = int_0^infty dr r^(2*l+4) exp(-(beta+expj) r^2)
 * with beta = expi * alpha / (expi + alpha)
 * See gauss_dida for arguments.
 */
double gauss_iplus(int l, double alpha, double expi, double expj) {
    double expi_conv = expi * alpha / (expi + alpha);
    return (-1 * expi) / (expi + alpha) *
           gauss_integral(l + 1, expi_conv + expj);
}

/**
 * Computes alpha I+. See gauss_iplus for details.
 */
double gauss_alpha_iplus(int l, double alpha, double expi, double expj) {
    return alpha * gauss_iplus(l, alpha, expi, expj);
}

/**
 * Computes I- / 2 with
 * I- = int_0^infty dr r^(2*l) exp(-(beta+expj) r^2)
 * with beta = expi * alpha / (expi + alpha)
 * See gauss_dida for arguments.
 */
double gauss_iminus(int l, double alpha, double expi, double expj) {
    double expi_conv = expi * alpha / (expi + alpha);
    return 0.5 * gauss_integral(l - 1, expi_conv + expj);
}

/**
 * Computes I- / (2 * alpha). See gauss_iminus for arguments.
 */
double gauss_ainv_iminus(int l, double alpha, double expi, double expj) {
    return gauss_iminus(l, alpha, expi, expj) / alpha;
}

/**
 * Allocate memory for and initialize an atc_basis_set and assign a pointer to
 * it to *atc_ptr. In addition to constructing various index lists to help
 * efficiently perform transformations of the basis set, it also
 * computes Cholesky factorizations of each ia,l block of the overlap matrix
 * (where ia and l are atom and angular momentum indexes). This allows
 * basis projections to be performed.
 * act_ptr : On exit, *acto_ptr is a pointer to an atco_basis_set object.
 * atom2l0 : An index locator. gamma_loc[atom2l0[ia] : atom2l0[ia+1]] contains
 *           the global gamma_loc indexes for atom ia.
 * lmaxs : lmaxs[ia] is the maximum l value of the basis for atom ia.
 * gamma_loc : A shell indexing list. gamma_loc[atom2l0[ia] + l] is the
 *             index of the first shell of atom ia with angular momentum l.
 *             [gamma_loc[atom2l0[ia] + l], atom2lo[ia] + l + 1]) is the range
 *             of indexes with atom ia and angular momentum l
 * all_exps : An array with length gamma_loc[atom2l0[natm]] containing the
 *            exponent for each Gaussian in the basis.
 * all_coefs : An array with length gamma_loc[atom2l0[natm]] containing the
 *             normalization coefficients for each Gaussian in the basis.
 * natm : Number of atoms in the system
 * UPLO ('U' or 'L') : Whether to store upper or lower triangular matrix for
 *                     Cholesky factorizations of the basis set for each
 *                     ia, l combination.
 */
void generate_atc_basis_set(atc_basis_set **atco_ptr, int *atom2l0, int *lmaxs,
                            int *gamma_loc, double *all_exps, double *all_coefs,
                            int natm, char UPLO) {
    atc_basis_set *atco = (atc_basis_set *)malloc(sizeof(atc_basis_set));
    atco->atc_convs = (atc_atom *)malloc(natm * sizeof(atc_atom));
    atco->UPLO = UPLO;
    atco->natm = natm;
    int info, gsh0;
    atc_atom *atcc;
    int *gamma_loc_l;
    atco->nbas = 0;
    int ngsh;
    int ngsh2;
    int ngsh_l;
    int shl = 0;
    double exp0, coef0, exp1, coef1;
    for (int ia = 0; ia < natm; ia++) {
        atcc = atco->atc_convs + ia;
        atcc->lmax = lmaxs[ia];
        atcc->ia = ia;
        atcc->ia_loc = gamma_loc[atom2l0[ia]];
        gsh0 = atcc->ia_loc;
        atcc->l_loc = (int *)malloc((atcc->lmax + 2) * sizeof(int));
        atcc->global_l_loc = (int *)malloc((atcc->lmax + 2) * sizeof(int));
        atcc->l_loc2 = (int *)malloc((atcc->lmax + 2) * sizeof(int));
        gamma_loc_l = gamma_loc + atom2l0[ia];
        atcc->l_loc[0] = 0;
        atcc->l_loc2[0] = 0;
        atcc->global_l_loc[0] = gamma_loc_l[0];
        ngsh = 0;
        ngsh2 = 0;
        for (int l = 0; l < atcc->lmax + 1; l++) {
            ngsh_l = gamma_loc_l[l + 1] - gamma_loc_l[l];
            ngsh += ngsh_l;
            ngsh2 += ngsh_l * ngsh_l;
            atcc->l_loc[l + 1] = ngsh;
            atcc->global_l_loc[l + 1] = gamma_loc_l[l + 1];
            atcc->l_loc2[l + 1] = ngsh2;
        }
        atcc->ngsh = ngsh;
        atcc->ngsh2 = ngsh2;
        atcc->gammas = (double *)malloc(ngsh * sizeof(double));
        atcc->gcoefs = (double *)malloc(ngsh * sizeof(double));
        atcc->gtrans_0 = (double *)malloc(ngsh2 * sizeof(double));
        atcc->gtrans_m = (double *)malloc(ngsh2 * sizeof(double));
        atcc->gtrans_p = (double *)malloc(ngsh2 * sizeof(double));
        for (int g = 0; g < ngsh; g++) {
            atcc->gammas[g] = all_exps[g + gsh0];
            atcc->gcoefs[g] = all_coefs[g + gsh0];
        }
        for (int l = 0; l < atcc->lmax + 1; l++) {
            ngsh_l = atcc->l_loc[l + 1] - atcc->l_loc[l];
            for (int g0 = 0; g0 < ngsh_l; g0++) {
                coef0 = atcc->gcoefs[atcc->l_loc[l] + g0];
                exp0 = atcc->gammas[atcc->l_loc[l] + g0];
                atco->nbas++;
                for (int g1 = 0; g1 < ngsh_l; g1++) {
                    coef1 = atcc->gcoefs[atcc->l_loc[l] + g1];
                    exp1 = atcc->gammas[atcc->l_loc[l] + g1];
                    atcc->gtrans_0[atcc->l_loc2[l] + g0 * ngsh_l + g1] =
                        coef0 * coef1 * gauss_integral(l, exp0 + exp1);
                    atcc->gtrans_m[atcc->l_loc2[l] + g0 * ngsh_l + g1] =
                        coef0 * coef1 * gauss_integral(l - 1, exp0 + exp1);
                    atcc->gtrans_p[atcc->l_loc2[l] + g0 * ngsh_l + g1] =
                        coef0 * coef1 * gauss_integral(l + 1, exp0 + exp1);
                }
            }
            dpotrf_(&(atco->UPLO), &ngsh_l, atcc->gtrans_0 + atcc->l_loc2[l],
                    &ngsh_l, &info);
            dpotrf_(&(atco->UPLO), &ngsh_l, atcc->gtrans_m + atcc->l_loc2[l],
                    &ngsh_l, &info);
            dpotrf_(&(atco->UPLO), &ngsh_l, atcc->gtrans_p + atcc->l_loc2[l],
                    &ngsh_l, &info);
        }
    }
    atco->bas = (int *)malloc(BAS_SLOTS * atco->nbas * sizeof(int));
    atco->ao_loc = (int *)malloc((atco->nbas + 1) * sizeof(int));
    atco->env = (double *)malloc(atco->nbas * 2 * sizeof(double));
    atco->atom_loc_ao = (int *)malloc((atco->natm + 1) * sizeof(int));
    atco->nao = 0;
    atco->ao_loc[0] = 0;
    atco->atom_loc_ao[0] = 0;
    shl = 0;
    for (int ia = 0; ia < natm; ia++) {
        atcc = atco->atc_convs + ia;
        for (int l = 0; l < atcc->lmax + 1; l++) {
            ngsh_l = atcc->l_loc[l + 1] - atcc->l_loc[l];
            for (int g = 0; g < ngsh_l; g++) {
                coef0 = atcc->gcoefs[atcc->l_loc[l] + g];
                exp0 = atcc->gammas[atcc->l_loc[l] + g];
                atco->bas[shl * BAS_SLOTS + ATOM_OF] = ia;
                atco->bas[shl * BAS_SLOTS + ANG_OF] = l;
                atco->bas[shl * BAS_SLOTS + PTR_COEFF] = 2 * shl;
                atco->bas[shl * BAS_SLOTS + PTR_EXP] = 2 * shl + 1;
                atco->env[2 * shl] = coef0;
                atco->env[2 * shl + 1] = exp0;
                atco->nao += 2 * l + 1;
                shl++;
                atco->ao_loc[shl] = atco->nao;
            }
        }
        atco->atom_loc_ao[ia + 1] = shl;
    }
    atco_ptr[0] = atco;
}

/**
 * Free the memory associated with an atc_atom object.
 */
void free_atc_atom(atc_atom atcc) {
    free(atcc.l_loc);
    free(atcc.global_l_loc);
    free(atcc.l_loc2);
    free(atcc.gammas);
    free(atcc.gcoefs);
    free(atcc.gtrans_0);
    free(atcc.gtrans_m);
    free(atcc.gtrans_p);
}

/**
 * Free the memory associated with an atc_basis_set object.
 */
void free_atc_basis_set(atc_basis_set *atco) {
    int ia;
    if (atco->atc_convs != NULL) {
        for (ia = 0; ia < atco->natm; ia++) {
            free_atc_atom(atco->atc_convs[ia]);
        }
        free(atco->atc_convs);
    }
    if (atco->bas != NULL) {
        free(atco->bas);
        free(atco->ao_loc);
        free(atco->env);
        free(atco->atom_loc_ao);
    }
    free(atco);
}

/**
 * Get number of shells in atc_basis_set object.
 */
int get_atco_nbas(atc_basis_set *atco) { return atco->nbas; }

/**
 * Get number of orbitals in atc_basis_set object.
 */
int get_atco_nao(atc_basis_set *atco) { return atco->nao; }

/**
 * Fill bas with the bas member of atco.
 */
void get_atco_bas(int *bas, atc_basis_set *atco) {
    for (int i = 0; i < BAS_SLOTS * atco->nbas; i++) {
        bas[i] = atco->bas[i];
    }
}

/**
 * Fill env with the env member of atco.
 */
void get_atco_env(double *env, atc_basis_set *atco) {
    for (int i = 0; i < 2 * atco->nbas; i++) {
        env[i] = atco->env[i];
    }
}

/**
 * Free the memory associated with a convolution_collection object.
 * WARNING: does not free atc_basis_set objects within the
 * convolution_collection, which are considered their own independent objects
 * and must be freed separately. ccl : convolution_collection to free.
 */
void free_convolution_collection(convolution_collection *ccl) {
    int ia;
    for (ia = 0; ia < ccl->natm; ia++) {
        free(ccl->pair_loc[ia]);
    }
    free(ccl->pair_loc);
    free(ccl->ovlp_mats);
    free(ccl->alphas);
    free(ccl->alpha_norms);
    free(ccl->feat_orders);
    if (ccl->nfeat_i) {
        free(ccl->icontrib_ids);
    }
    free(ccl);
}

/**
 * Generate convolution_collection object ccl and store it in ccl_ptr.
 * ccl stores the input and output atc_basis_set objects, as well as a tensor
 * of convolution integrals ovlp_mats for convolving functions in the atco_inp
 * space into the atco_out.
 * ccl_ptr : pointer to the convolution_collection object. On exit,
 *           *ccl_ptr is a pointer to a convolution_collection.
 * atco_inp : atomic basis for input for convolutions
 * atco_out : atomic basis for output of convolutions
 * alphas : list of exponents for alpha and beta in the convolutions
 * alpha_norms : normalization factors to use for each
 *               exp(-alpha*(r-r')^2) function.
 * nalpha : Number of input functions (typically number of control points
 *          in interpolation/auxiliary expansion for CIDER feature kernel).
 * NOTE: Both len(alphas) and len(alpha_norms) must be the same and
 *       equal to ccl->nalpha
 * NOTE on nbeta: nbeta is the umber of output functions. Should be nalpha if
 *                using version j, otherwise should be nalpha + the
 *                number of intermediate version i features for
 *                the ij feature version. Should not be called
 *                for version k (only generate_atc_integrals_vi is
 *                needed for that).
 */
void generate_convolution_collection(convolution_collection **ccl_ptr,
                                     atc_basis_set *atco_inp,
                                     atc_basis_set *atco_out, double *alphas,
                                     double *alpha_norms, int *icontrib_ids,
                                     int nalpha, int nfeat_i, int has_vj) {
    int natm = atco_inp->natm;
    int **pair_loc = (int **)malloc(natm * sizeof(int *));
    int tot_shl_pairs = 0;
    atc_atom atcc_inp;
    atc_atom atcc_out;
    convolution_collection *ccl =
        (convolution_collection *)malloc(sizeof(convolution_collection));
    ccl->natm = natm;
    ccl_ptr[0] = ccl;
    int ia, l, lmax;
    int max_pair = 0;
    int num_pair;
    ccl->has_vj = has_vj;
    ccl->nfeat_i = nfeat_i;
    if (ccl->nfeat_i) {
        ccl->icontrib_ids = (int *)malloc(ccl->nfeat_i * sizeof(int));
        for (ia = 0; ia < nfeat_i; ia++) {
            ccl->icontrib_ids[ia] = icontrib_ids[ia];
        }
    } else {
        ccl->icontrib_ids = NULL;
    }
    ccl->nbeta = nfeat_i;
    if (has_vj) {
        ccl->nbeta += nalpha;
    }
    ccl->alphas = (double *)malloc(nalpha * sizeof(double));
    ccl->alpha_norms = (double *)malloc(nalpha * sizeof(double));
    for (ia = 0; ia < nalpha; ia++) {
        ccl->alphas[ia] = alphas[ia];
        ccl->alpha_norms[ia] = alpha_norms[ia];
    }
    for (ia = 0; ia < atco_inp->natm; ia++) {
        atcc_inp = atco_inp->atc_convs[ia];
        atcc_out = atco_out->atc_convs[ia];
        lmax = MIN(atcc_out.lmax, atcc_inp.lmax);
        pair_loc[ia] = (int *)malloc((lmax + 1) * sizeof(int));
        for (l = 0; l < lmax + 1; l++) {
            pair_loc[ia][l] = tot_shl_pairs;
            num_pair = (atcc_inp.l_loc[l + 1] - atcc_inp.l_loc[l]) *
                       (atcc_out.l_loc[l + 1] - atcc_out.l_loc[l]);
            tot_shl_pairs += num_pair;
            max_pair = MAX(max_pair, num_pair);
        }
    }
    ccl->max_pair = max_pair;
    ccl->tot_shl_pairs = tot_shl_pairs;
    ccl->pair_loc = pair_loc;
    ccl->nalpha = nalpha;
    ccl->atco_inp = atco_inp;
    ccl->atco_out = atco_out;
    ccl->natm = atco_inp->natm;
    ccl->ovlp_mats = (double *)malloc(ccl->nalpha * ccl->nbeta * tot_shl_pairs *
                                      sizeof(double));
    ccl->feat_orders = (int *)malloc(ccl->nbeta * sizeof(int));
    int offset = 0;
    if (has_vj) {
        for (ia = 0; ia < nalpha; ia++) {
            ccl->feat_orders[offset] = 0;
            offset++;
        }
    }
    for (ia = 0; ia < nfeat_i; ia++) {
        if (ccl->icontrib_ids[ia] == 3 || ccl->icontrib_ids[ia] == 4) {
            ccl->feat_orders[offset] = -1;
        } else if (ccl->icontrib_ids[ia] == 5 || ccl->icontrib_ids[ia] == 6) {
            ccl->feat_orders[offset] = 1;
        } else {
            ccl->feat_orders[offset] = 0;
        }
        offset++;
    }
    /* TODO might include this code later for acceleration, currently draft.
    int *pairs_ish = (int*) malloc(tot_shl_pairs * sizeof(int));
    int *pairs_jsh = (int*) malloc(tot_shl_pairs * sizeof(int));
    int *pairs_loc = (int*) malloc(tot_shl_pairs * sizeof(int));
    for (ia) {
        for(l) {
            for (ish) {
                for (jsh) {
                    pairs_ish[pair] = ish;
                    pairs_jsh[pair] = jsh;
                    pairs_loc[pair] = prod_loc[ia][l]
                                      + (jsh - jsh0) * (ish1 - ish0)
                                      + ish - ish0;;
                    pair++;
                }
            }
        }
    }
    */
}

/**
 * Compute the integrals
 * int d^3r' d^3r exp[-(alpha+beta)(r-r')^2] exp(-gamma_in * r'^2)
 * exp(-gamma_out r^2) and store them in the ovlp_mats object of ccl. Normalize
 * the integrals by multiplying by N_alpha * N_beta, where N_alpha and N_beta
 * are the elements of alpha_norms. Also normalize by the normalization
 * coefficients of the input and output bases in ccl. ccl :
 * convolution_collection in which to store the matrix elements alphas : list of
 * exponents for alpha and beta in the convolutions alpha_norms : normalization
 * factors to use for each exp(-alpha*(r-r')^2) function. NOTE: Both len(alphas)
 * and len(alpha_norms) must be the same and equal to ccl->nalpha
 */
void generate_atc_integrals_vj(convolution_collection *ccl) {
#pragma omp parallel
    {
        double PI = 4 * atan(1.0);
        double apb;
        int *bas_out, *bas_in;
        int ia, l, ish, ish0, ish1, jsh, jsh0, jsh1, loc;
        int qa, qb, q;
        atc_atom atcc_inp;
        atc_atom atcc_out;
        int nalpha = ccl->nalpha;
        int nbeta = ccl->nbeta;
        double *expi_q = (double *)malloc(nalpha * nalpha * sizeof(double));
        double *coefi_q = (double *)malloc(nalpha * nalpha * sizeof(double));
        double expi, coefi;
        double expj, coefj;
        int nalpha2 = nalpha * nalpha;
        double *ovlp;
        double *alphas = ccl->alphas;
        double *alpha_norms = ccl->alpha_norms;
#pragma omp for schedule(dynamic, 4)
        for (ish = 0; ish < ccl->atco_inp->nbas; ish++) {
            bas_in = ccl->atco_inp->bas + ish * BAS_SLOTS;
            ia = bas_in[ATOM_OF];
            l = bas_in[ANG_OF];
            expi = ccl->atco_inp->env[bas_in[PTR_EXP]];
            coefi = ccl->atco_inp->env[bas_in[PTR_COEFF]];
            atcc_inp = ccl->atco_inp->atc_convs[ia];
            atcc_out = ccl->atco_out->atc_convs[ia];
            ish0 = atcc_inp.global_l_loc[l];
            ish1 = atcc_inp.global_l_loc[l + 1];
            jsh0 = atcc_out.global_l_loc[l];
            jsh1 = atcc_out.global_l_loc[l + 1];
            q = 0;
            for (qb = 0; qb < nalpha; qb++) {
                for (qa = 0; qa < nalpha; qa++, q++) {
                    apb = alphas[qa] + alphas[qb];
                    expi_q[q] = expi * apb / (expi + apb);
                    coefi_q[q] = coefi * pow(PI / apb, 1.5) *
                                 pow(apb / (expi + apb), 1.5 + l) *
                                 alpha_norms[qa] * alpha_norms[qb];
                }
            }
            for (jsh = jsh0; jsh < jsh1; jsh++) {
                loc = ccl->pair_loc[ia][l] + (jsh - jsh0) * (ish1 - ish0) +
                      ish - ish0;
                bas_out = ccl->atco_out->bas + jsh * BAS_SLOTS;
                expj = ccl->atco_out->env[bas_out[PTR_EXP]];
                coefj = ccl->atco_out->env[bas_out[PTR_COEFF]];
                ovlp = ccl->ovlp_mats + loc * nbeta * nalpha;
                for (q = 0; q < nalpha2; q++) {
                    ovlp[q] = coefj * coefi_q[q] *
                              gauss_integral(l, expi_q[q] + expj);
                }
            }
        }
        free(expi_q);
        free(coefi_q);
    }
}

/**
 * Generate atom-centered integrals for version i or k
 * (or the i features for version ij). For these features, there is only one set
 * of control points (alphas, at r') as opposed to the alpha and beta control
 * points for vj. ccl : convolution_collection containing atom-centered bases
 * and ovlp_mats tensor featid : An integer identifier for which feature to
 * compute. See function comments for the formula for each feature id. offset :
 * Where to insert the generated integrals in ccl->ovlp_mats. NOTE: offset
 * should be at least nalpha if vj features are included and at least 0 if only
 * vi/vk features are used. offset should be no more than cc->nbeta-1.
 */
void generate_atc_integrals_vi(convolution_collection *ccl, int featid,
                               int offset) {
#pragma omp parallel
    {
        double PI = 4 * atan(1.0);
        int *bas_out, *bas_in;
        int ia, l, ish, jsh, jsh0, jsh1, loc, ish0, ish1;
        int q;
        atc_atom atcc_inp, atcc_out;
        int nalpha = ccl->nalpha;
        int nbeta = ccl->nbeta;
        double *coefi_q = (double *)malloc(nalpha * sizeof(double));
        double expi, coefi;
        double expj, coefj;
        double (*integral_func)(int, double, double, double);
        // NOTE: This messy if statement can be replaced by
        // subclassing when converting to C++.
        // I = int_0^infty d^3r' k^i_alpha(r, r') (r')^l exp(-expi * (r')^2) r^l
        // exp(-expj * (r)^2) k^0_alpha(r, r') = exp(-alpha*(r-r')^2)
        if (featid == 0)
            integral_func = &gauss_i0;
        // k^1_alpha(r, r') = (r-r')^2 * exp(-alpha*(r-r')^2)
        else if (featid == 1)
            integral_func = &gauss_dida;
        // k^2_alpha(r, r') = alpha * (r-r')^2 * exp(-alpha*(r-r')^2)
        else if (featid == 2)
            integral_func = &gauss_adida;
        // k^3_alpha(r, r') = alpha * exp(-alpha*(r-r')^2) / (r * r')
        // k^3 is used to make l-1 term of gradient feature calculation more
        // stable
        else if (featid == 3)
            integral_func = &gauss_iminus;
        // k^4_alpha(r, r') = exp(-alpha*(r-r')^2) / (r * r')
        // k^4 is used to make l-1 term of (r-r') feature calculation more
        // stable
        else if (featid == 4)
            integral_func = &gauss_ainv_iminus;
        // k^5_alpha(r, r') = exp(-alpha*(r-r')^2) * (r * r')
        // k^5 is used to make l+1 term of (r-r') feature calculation more
        // stable
        else if (featid == 5)
            integral_func = &gauss_iplus;
        // k^6_alpha(r, r') = alpha * exp(-alpha*(r-r')^2) * (r * r')
        // k^6 is used to make l+1 term of gradient feature calculation more
        // stable
        else if (featid == 6)
            integral_func = &gauss_alpha_iplus;
        // k^7_alpha(r, r') = alpha * exp(-alpha*(r-r')^2)
        else if (featid == 7)
            integral_func = &gauss_ai0;
        // k^8_alpha(r, r') = alpha^2 * (r-r')^2 * exp(-alpha*(r-r')^2)
        else if (featid == 8)
            integral_func = &gauss_a2dida;
        // NOTE: k^9 = 4 * k^8 - 2 * k^7 should given the Laplacian of the k^0
        // feature. k^9_alpha(r, r') = \nabla^2 exp(-alpha*(r-r')^2)
        else if (featid == 9)
            integral_func = &gauss_lapli0;
        else {
            printf("Unsupported featid\n");
            exit(-1);
        }
        double *ovlp;
        double *alphas = ccl->alphas;
        double *alpha_norms = ccl->alpha_norms;
#pragma omp for schedule(dynamic, 4)
        for (ish = 0; ish < ccl->atco_inp->nbas; ish++) {
            bas_in = ccl->atco_inp->bas + ish * BAS_SLOTS;
            ia = bas_in[ATOM_OF];
            l = bas_in[ANG_OF];
            expi = ccl->atco_inp->env[bas_in[PTR_EXP]];
            coefi = ccl->atco_inp->env[bas_in[PTR_COEFF]];
            atcc_inp = ccl->atco_inp->atc_convs[ia];
            atcc_out = ccl->atco_out->atc_convs[ia];
            ish0 = atcc_inp.global_l_loc[l];
            ish1 = atcc_inp.global_l_loc[l + 1];
            jsh0 = atcc_out.global_l_loc[l];
            jsh1 = atcc_out.global_l_loc[l + 1];
            for (q = 0; q < nalpha; q++) {
                coefi_q[q] = coefi * pow(PI / alphas[q], 1.5) *
                             pow(alphas[q] / (expi + alphas[q]), 1.5 + l) *
                             alpha_norms[q];
            }
            for (jsh = jsh0; jsh < jsh1; jsh++) {
                loc = ccl->pair_loc[ia][l] + (jsh - jsh0) * (ish1 - ish0) +
                      ish - ish0;
                bas_out = ccl->atco_out->bas + jsh * BAS_SLOTS;
                expj = ccl->atco_out->env[bas_out[PTR_EXP]];
                coefj = ccl->atco_out->env[bas_out[PTR_COEFF]];
                ovlp = ccl->ovlp_mats + (loc * nbeta + offset) * nalpha;
                for (q = 0; q < nalpha; q++) {
                    ovlp[q] = coefj * coefi_q[q] *
                              (*integral_func)(l, alphas[q], expi, expj);
                }
            }
        }
        free(coefi_q);
    }
}

void generate_atc_integrals_all(convolution_collection *ccl) {
    int offset = 0;
    if (ccl->has_vj) {
        generate_atc_integrals_vj(ccl);
        offset += ccl->nalpha;
    }
    for (int ii = 0; ii < ccl->nfeat_i; ii++) {
        generate_atc_integrals_vi(ccl, ccl->icontrib_ids[ii], offset);
        offset++;
    }
}

/**
 * Using numerically stable Cholesky solves, this function effectively
 * multiplies the overlap tensors between the convolved Gaussians
 * and output basis by the inverse overlap matrices of ccl->atco_inp
 * and ccl->atco_out. What this effectively means is that after
 * calling this function, multiply_atc_coefs can take the raw overlaps
 * of a given set of functions with the nonorthogonal atco_inp basis
 * and output the convolutions expanded in the space of atco_out.
 * ccl : convolution_collection with convolution matrix elements
 * While the actual tensors are sparse, the conceptual mathematical
 * operation is
 * OUTPUT_il = ((IN OVERLAP)^-1)_ij (CONVOLUTIONS^-1)_jk ((OUT OVERLAP)^-1)_kl
 * ccl : convolution_collection object. The ovlp_mats member is rewritten
 *       during the execution of this function.
 */
void solve_atc_coefs(convolution_collection *ccl) {
#pragma omp parallel
    {
        int ia, dish, djsh;
        int ish0, jsh0;
        int qq;
        atc_atom atcc_inp, atcc_out;
        int stride = ccl->nalpha * ccl->nbeta;
        double *ovlp;
        double *buf = (double *)malloc(ccl->max_pair * sizeof(double));
        double *buf2 = (double *)malloc(ccl->max_pair * sizeof(double));
        double *chomat;
        int aqq, beta;
        int info;
#pragma omp for schedule(dynamic, 4)
        for (aqq = 0; aqq < stride * ccl->natm; aqq++) {
            ia = aqq / stride;
            qq = aqq % stride;
            beta = qq / ccl->nalpha;
            atcc_inp = ccl->atco_inp->atc_convs[ia];
            atcc_out = ccl->atco_out->atc_convs[ia];
            for (int l = 0; l < atcc_inp.lmax + 1; l++) {
                ish0 = atcc_inp.global_l_loc[l];
                dish = atcc_inp.global_l_loc[l + 1] - ish0;
                jsh0 = atcc_out.global_l_loc[l];
                djsh = atcc_out.global_l_loc[l + 1] - jsh0;
                ovlp = ccl->ovlp_mats + ccl->pair_loc[ia][l] * stride + qq;
                for (int sh = 0; sh < dish * djsh; sh++) {
                    buf[sh] = ovlp[0];
                    ovlp += stride;
                }
                chomat = atcc_inp.gtrans_0 + atcc_inp.l_loc2[l];
                dpotrs_(&(ccl->atco_inp->UPLO), &dish, &djsh, chomat, &dish,
                        buf, &dish, &info);
                for (int jsh = 0; jsh < djsh; jsh++) {
                    for (int ish = 0; ish < dish; ish++) {
                        buf2[ish * djsh + jsh] = buf[jsh * dish + ish];
                    }
                }
                // TODO this will need to use gtrans_m/p for some features
                if (ccl->feat_orders[beta] == -1) { // need l-1 integrals
                    chomat = atcc_out.gtrans_m + atcc_out.l_loc2[l];
                } else if (ccl->feat_orders[beta] == 1) { // need l+1 integrals
                    chomat = atcc_out.gtrans_p + atcc_out.l_loc2[l];
                } else { // need l integrals
                    chomat = atcc_out.gtrans_0 + atcc_out.l_loc2[l];
                }
                dpotrs_(&(ccl->atco_inp->UPLO), &djsh, &dish, chomat, &djsh,
                        buf2, &djsh, &info);
                ovlp = ccl->ovlp_mats + ccl->pair_loc[ia][l] * stride + qq;
                for (int jsh = 0; jsh < djsh; jsh++) {
                    for (int ish = 0; ish < dish; ish++) {
                        ovlp[0] = buf2[ish * djsh + jsh];
                        ovlp += stride;
                    }
                }
            }
        }
        free(buf);
        free(buf2);
    }
}

void solve_atc_coefs_arr(atc_basis_set *atco, double *p_uq, int nalpha) {
#pragma omp parallel
    {
        int ia, dish;
        int ish0;
        atc_atom atcc;
        double *chomat;
        int info;
        int max_size = 0;
        int my_size;
        int n_mq, mq;
        double *p_mq;
        for (ia = 0; ia < atco->natm; ia++) {
            atcc = atco->atc_convs[ia];
            for (int l = 0; l < atcc.lmax + 1; l++) {
                my_size = (2 * l + 1) *
                          (atcc.global_l_loc[l + 1] - atcc.global_l_loc[l]);
                max_size = MAX(max_size, my_size);
            }
        }
        max_size *= nalpha;
        double *buf = (double *)malloc(max_size * sizeof(double));
#pragma omp for schedule(dynamic, 4)
        for (ia = 0; ia < atco->natm; ia++) {
            atcc = atco->atc_convs[ia];
            for (int l = 0; l < atcc.lmax + 1; l++) {
                ish0 = atcc.global_l_loc[l];
                dish = atcc.global_l_loc[l + 1] - ish0;
                n_mq = (2 * l + 1) * nalpha;
                for (int sh = 0; sh < dish; sh++) {
                    p_mq = p_uq + nalpha * atco->ao_loc[ish0 + sh];
                    for (mq = 0; mq < n_mq; mq++) {
                        buf[mq * dish + sh] = p_mq[mq];
                    }
                }
                chomat = atcc.gtrans_0 + atcc.l_loc2[l];
                dpotrs_(&(atco->UPLO), &dish, &n_mq, chomat, &dish, buf, &dish,
                        &info);
                for (int sh = 0; sh < dish; sh++) {
                    p_mq = p_uq + nalpha * atco->ao_loc[ish0 + sh];
                    for (mq = 0; mq < n_mq; mq++) {
                        p_mq[mq] = buf[mq * dish + sh];
                    }
                }
            }
        }
        free(buf);
    }
}

void convert_atomic_radial_basis(double *p_uq, double *p_vq, double *ovlps_l,
                                 int *iloc_l, int *jloc_l, int nalpha, int lmax,
                                 int fwd) {
    int ni, nj, m;
    char transa = 'N';
    char transb;
    int ldb;
    double one = 1.0;
    double zero = 0.0;
    int *kloc_l;
    if (fwd) {
        transb = 'T';
        kloc_l = jloc_l;
    } else {
        transb = 'N';
        kloc_l = iloc_l;
    }
    for (int l = 0; l <= lmax; l++) {
        ni = iloc_l[l + 1] - iloc_l[l];
        nj = jloc_l[l + 1] - jloc_l[l];
        ldb = kloc_l[l + 1] - kloc_l[l];
        m = nalpha * (2 * l + 1);
        if (ni > 0 && nj > 0) {
            dgemm_(&transa, &transb, &m, &nj, &ni, &one, p_uq, &m, ovlps_l,
                   &ldb, &zero, p_vq, &m);
        }
        p_uq += m * ni;
        p_vq += m * nj;
        ovlps_l += ni * nj;
    }
}

void solve_atc_coefs_arr_ccl(convolution_collection *ccl, double *p_uq,
                             int nalpha, int inp) {
    atc_basis_set *atco;
    if (inp) {
        atco = ccl->atco_inp;
    } else {
        atco = ccl->atco_out;
    }
    solve_atc_coefs_arr(atco, p_uq, nalpha);
}

/**
 * Convolve the input inp_uq to out_vq using the convolution_collection ccl.
 * NOTE this is for version i, j, and ij. Use multiply_atc_integrals_vk for
 * the special case of version k descriptors.
 * If fwd > 0, inp_uq is in the basis of ccl->atco_inp and
 * out_vq is in the basis of ccl->atco_out, while if fwd == 0,
 * inp_uq is in the basis of ccl->atco_out and out_vq is in the basis
 * of ccl->atco_inp.
 * inp_uq (atco_inp->nao x ccl->nalpha if fwd,
           else atco_out->nao x ccl->nbeta)
 * out_vq (atco_out->nao x ccl->nbeta if fwd,
           else atco_inp->nao x ccl->nalpha)
 * ccl : convolution_collection object containing convolution integrals
 * fwd : If 0, swap inp_uq is in the basis of atco_out and vice versa.
*/
void multiply_atc_integrals(double *inp_uq, double *out_vq,
                            convolution_collection *ccl, int fwd) {
#pragma omp parallel
    {
        double ALPHA = 1;
        double BETA = 1;
        int ish, jsh, ish0, ish1, jsh0, jsh1;
        int l, ia;
        int loc;
        int out_stride, inp_stride;
        int *bas_out;
        atc_atom atcc_inp, atcc_out;
        atc_basis_set *atco_inp, *atco_out;
        double *inp_mq, *out_mq, *ovlp_ba;
        int GEMM_M, GEMM_K, GEMM_N, GEMM_LDA, GEMM_LDB;
        char GEMM_TRANSA;
        int mat_stride = ccl->nalpha * ccl->nbeta;
        char GEMM_TRANSB = 'n';
        GEMM_LDA = ccl->nalpha;
        if (fwd) {
            atco_inp = ccl->atco_inp;
            atco_out = ccl->atco_out;
            GEMM_M = ccl->nbeta;
            GEMM_K = ccl->nalpha;
            GEMM_TRANSA = 't';
            GEMM_LDB = ccl->nalpha;
            fwd = 1;
            inp_stride = ccl->nalpha;
            out_stride = ccl->nbeta;
        } else {
            atco_inp = ccl->atco_out;
            atco_out = ccl->atco_inp;
            GEMM_M = ccl->nalpha;
            GEMM_K = ccl->nbeta;
            GEMM_TRANSA = 'n';
            GEMM_LDB = ccl->nbeta;
            fwd = 0;
            inp_stride = ccl->nbeta;
            out_stride = ccl->nalpha;
        }
#pragma omp for schedule(dynamic, 4)
        for (jsh = 0; jsh < atco_out->nbas; jsh++) {
            bas_out = atco_out->bas + jsh * BAS_SLOTS;
            ia = bas_out[ATOM_OF];
            l = bas_out[ANG_OF];
            atcc_inp = atco_inp->atc_convs[ia];
            atcc_out = atco_out->atc_convs[ia];
            ish0 = atcc_inp.global_l_loc[l];
            ish1 = atcc_inp.global_l_loc[l + 1];
            jsh0 = atcc_out.global_l_loc[l];
            jsh1 = atcc_out.global_l_loc[l + 1];
            out_mq = out_vq + out_stride * atco_out->ao_loc[jsh];
            for (ish = ish0; ish < ish1; ish++) {
                inp_mq = inp_uq + inp_stride * atco_inp->ao_loc[ish];
                loc = ccl->pair_loc[ia][l];
                GEMM_N = 2 * l + 1;
                loc += fwd * ((jsh - jsh0) * (ish1 - ish0) + ish - ish0);
                loc += (1 - fwd) * ((ish - ish0) * (jsh1 - jsh0) + jsh - jsh0);
                loc *= mat_stride;
                ovlp_ba = ccl->ovlp_mats + loc;
                dgemm_(&GEMM_TRANSA, &GEMM_TRANSB, &GEMM_M, &GEMM_N, &GEMM_K,
                       &ALPHA, ovlp_ba, &GEMM_LDA, inp_mq, &GEMM_LDB, &BETA,
                       out_mq, &GEMM_M);
                // Note: need to choose nbeta vs nalpha depending on direction
                // if the two are not equal.
                /* This is what the above matmul should do for fwd
                for (m = 0; m < 2 * l + 1; m++) {
                    for (a = 0; a < nalpha; a++) {
                        for (b = 0; b < nbeta; b++) {
                            if (fwd) {
                                out_mq[m * nbeta + b] += in_mq[m * nalpha + a]
                                                         * ovlp_ba[b * nalpha +
                a];
                            }
                        }
                    }
                }
                */
            }
        }
    }
}

void atc_reciprocal_convolution(double *in_sklmq, double *out_sklmq,
                                double *k_k, double *alphas,
                                double *alpha_norms, int nspin, int nk, int nlm,
                                int nq) {
    double *conv_facs = (double *)malloc(nq * nq * sizeof(double));
    double *conv_exps = (double *)malloc(nq * nq * sizeof(double));
    double FPI = 16 * atan(1.0);
#pragma omp parallel for
    for (int n2 = 0; n2 < nq; n2++) {
        for (int n1 = 0; n1 < nq; n1++) {
            int ind = n2 * nq + n1;
            conv_exps[ind] = 0.25 / (alphas[n1] + alphas[n2]);
            conv_facs[ind] = alpha_norms[n1] * alpha_norms[n2];
            conv_facs[ind] *= pow(FPI * conv_exps[ind], 1.5);
        }
    }
#pragma omp parallel
    {
        int k, q1, q2, lm, ind, displ, s;
        double *conv = (double *)malloc(nq * nq * sizeof(double));
        double *in_q, *out_q;
        double mk2;
        for (k = 0; k < nk; k++) {
            mk2 = -1.0 * k_k[k] * k_k[k];
            for (ind = 0; ind < nq * nq; ind++) {
                conv[ind] = conv_facs[ind] * exp(mk2 * conv_exps[ind]);
            }
            for (s = 0; s < nspin; s++) {
                for (lm = 0; lm < nlm; lm++) {
                    // TODO blas
                    ind = 0;
                    displ = nq * (lm + nlm * (k + nk * s));
                    in_q = in_sklmq + displ;
                    out_q = out_sklmq + displ;
                    for (q2 = 0; q2 < nq; q2++) {
                        out_q[q2] = 0.0;
                        for (q1 = 0; q1 < nq; q1++, ind++) {
                            out_q[q2] += conv[ind] * in_q[q1];
                        }
                    }
                }
            }
        }
        free(conv);
    }
    free(conv_exps);
    free(conv_facs);
}

/**
 * Convolve the input inp_uq to out_vq using the convolution_collection ccl.
 * This function is for version k only, where the alphas are not contracted to
 * the betas. NOTE the difference in inp_uq and out_vq compared to
 * multiply_atc_integrals, which has inner dimension nalpha regardless of other
 * settings.
 * If fwd > 0, inp_uq is in the basis of ccl->atco_inp and
 * out_vq is in the basis of ccl->atco_out, while if fwd == 0,
 * inp_uq is in the basis of ccl->atco_out and out_vq is in the basis
 * of ccl->atco_inp.
 * inp_uq (atco_inp->nao x ccl->nalpha if fwd,
           else atco_out->nao x ccl->nalpha)
 * out_vq (atco_out->nao x ccl->nalpha if fwd,
           else atco_inp->nao x ccl->nalpha)
 * ccl : convolution_collection object containing convolution integrals
 * fwd : If 0, swap inp_uq is in the basis of atco_out and vice versa.
*/
void multiply_atc_integrals_vk(double *inp_uq, double *out_vq,
                               convolution_collection *ccl, int fwd) {
#pragma omp parallel
    {
        int ish, jsh, ish0, ish1, jsh0, jsh1;
        int l, ia, m, q, mq;
        int loc;
        int *bas_out;
        atc_atom atcc_inp, atcc_out;
        atc_basis_set *atco_inp, *atco_out;
        int nalpha = ccl->nalpha;
        if (ccl->nbeta != 1) {
            printf("INTERNAL ERROR: nbeta must be 1 for vk, got %d",
                   ccl->nbeta);
        }
        double *inp_mq, *out_mq, *ovlp_ba;
        if (fwd) {
            atco_inp = ccl->atco_inp;
            atco_out = ccl->atco_out;
            fwd = 1;
        } else {
            atco_inp = ccl->atco_out;
            atco_out = ccl->atco_inp;
            fwd = 0;
        }
#pragma omp for schedule(dynamic, 4)
        for (jsh = 0; jsh < atco_out->nbas; jsh++) {
            bas_out = atco_out->bas + jsh * BAS_SLOTS;
            ia = bas_out[ATOM_OF];
            l = bas_out[ANG_OF];
            atcc_inp = atco_inp->atc_convs[ia];
            atcc_out = atco_out->atc_convs[ia];
            ish0 = atcc_inp.global_l_loc[l];
            ish1 = atcc_inp.global_l_loc[l + 1];
            jsh0 = atcc_out.global_l_loc[l];
            jsh1 = atcc_out.global_l_loc[l + 1];
            out_mq = out_vq + nalpha * atco_out->ao_loc[jsh];
            for (ish = ish0; ish < ish1; ish++) {
                inp_mq = inp_uq + nalpha * atco_inp->ao_loc[ish];
                loc = ccl->pair_loc[ia][l];
                loc += fwd * ((jsh - jsh0) * (ish1 - ish0) + ish - ish0);
                loc += (1 - fwd) * ((ish - ish0) * (jsh1 - jsh0) + jsh - jsh0);
                loc *= nalpha;
                ovlp_ba = ccl->ovlp_mats + loc;
                mq = 0;
                for (m = 0; m < 2 * l + 1; m++) {
                    for (q = 0; q < nalpha; q++, mq++) {
                        out_mq[mq] += inp_mq[mq] * ovlp_ba[q];
                    }
                }
            }
        }
    }
}

/**
 * Given a radial distribution of functions for each spherical harmonic
 * lm on each atom, with dimension nalpha, project onto the orbital
 * basis set given by atco. Note: this computes projections onto each
 * (non-orthogonal) basis function, not expansion coefficients
 * theta_rlmq (nrad x nlm x nalpha) : input functions to project onto atco basis
 * p_uq (atco->nao x nalpha) : output projections
 * ra_loc (length natm + 1) : Range of rad indices that correspond
 *                            to each atom
 * rads (length nrad) : radial coordinates for each radial index
 * nrad : number of radial coordinates over all atoms
 * nlm : number of spherical harmonics (lmax + 1)^2
 * atco : stores the atomic basis set.
 * nalpha : number of functions stored in the rlm space.
 */
void contract_rad_to_orb(double *theta_rlmq, double *p_uq, int *ra_loc,
                         double *rads, int nrad, int nlm, atc_basis_set *atco,
                         int nalpha, int stride, int offset) {
    p_uq = p_uq + offset;
#pragma omp parallel
    {
        int ish, i0, L0, nm, l, at;
        double *p_q, *theta_mq;
        int *bas = atco->bas;
        int *ao_loc = atco->ao_loc;
        double *env = atco->env;
        int nbas = atco->nbas;
        int *ibas;
        double coef, beta, val;
        int r, m, q, mq;
#pragma omp for schedule(dynamic, 4)
        for (ish = 0; ish < nbas; ish++) {
            ibas = bas + ish * BAS_SLOTS;
            at = ibas[ATOM_OF];
            l = ibas[ANG_OF];
            coef = env[ibas[PTR_COEFF]];
            beta = env[ibas[PTR_EXP]];
            i0 = ao_loc[ish];
            nm = 2 * l + 1;
            L0 = l * l;
            for (r = ra_loc[at]; r < ra_loc[at + 1]; r++) {
                val = coef * pow(rads[r], l) * exp(-beta * rads[r] * rads[r]);
                theta_mq = theta_rlmq + nalpha * (r * nlm + L0);
                p_q = p_uq + i0 * stride;
                mq = 0;
                for (m = 0; m < nm; m++) {
                    for (q = 0; q < nalpha; q++, mq++) {
                        p_q[q] += val * theta_mq[mq];
                    }
                    p_q += stride;
                }
            }
        }
    }
}

/**
 * Backwards version of contract_rad_to_orb
 * (i.e. takes p_uq as input and projects the basis
 * functions out onto the radial coordinates and spherical harmonics).
 * See contract_rad_to_orb for variable definitions and details.
 */
void contract_orb_to_rad(double *theta_rlmq, double *p_uq, int *ar_loc,
                         double *rads, int nrad, int nlm, atc_basis_set *atco,
                         int nalpha, int stride, int offset) {
    p_uq = p_uq + offset;
#pragma omp parallel
    {
        int ish, i0, L0, nm, l, at;
        double *p_q, *theta_mq;
        int *bas = atco->bas;
        int *ao_loc = atco->ao_loc;
        double *env = atco->env;
        int *ibas;
        double coef, beta, val;
        int r, m, q, mq;
#pragma omp for schedule(dynamic, 4)
        for (r = 0; r < nrad; r++) {
            at = ar_loc[r];
            for (ish = atco->atom_loc_ao[at]; ish < atco->atom_loc_ao[at + 1];
                 ish++) {
                ibas = bas + ish * BAS_SLOTS;
                l = ibas[ANG_OF];
                coef = env[ibas[PTR_COEFF]];
                beta = env[ibas[PTR_EXP]];
                i0 = ao_loc[ish];
                nm = 2 * l + 1;
                L0 = l * l;
                val = coef * pow(rads[r], l) * exp(-beta * rads[r] * rads[r]);
                theta_mq = theta_rlmq + nalpha * (r * nlm + L0);
                p_q = p_uq + i0 * stride;
                mq = 0;
                for (m = 0; m < nm; m++) {
                    for (q = 0; q < nalpha; q++, mq++) {
                        theta_mq[mq] += val * p_q[q];
                    }
                    p_q += stride;
                }
            }
        }
    }
}

void expand_to_grid(double *in_i, double *out_g, int l, int ia, double *r_g,
                    int ng, atc_basis_set *atco) {
    int *bas = atco->bas;
    double *env = atco->env;
    atc_atom *atcc = atco->atc_convs + ia;
    if (l > atcc->lmax) {
        printf("INTERNAL ERROR: l > lmax for atcc");
        exit(-1);
    }
    int ish0 = atcc->global_l_loc[l];
    int ish1 = atcc->global_l_loc[l + 1];
    int dish = ish1 - ish0;
    int ish;
    double *p_i = (double *)malloc((ish1 - ish0) * sizeof(double));
    for (ish = 0; ish < dish; ish++) {
        p_i[ish] = in_i[ish];
    }
    int info;
    double *chomat = atcc->gtrans_0 + atcc->l_loc2[l];
    int one = 1;
    dpotrs_(&(atco->UPLO), &dish, &one, chomat, &dish, p_i, &dish, &info);
    int *ibas;
    double coef, beta;
    for (int g = 0; g < ng; g++) {
        out_g[g] = 0;
        for (ish = ish0; ish < ish1; ish++) {
            ibas = bas + ish * BAS_SLOTS;
            l = ibas[ANG_OF];
            coef = env[ibas[PTR_COEFF]];
            beta = env[ibas[PTR_EXP]];
            out_g[g] += p_i[ish - ish0] * coef * pow(r_g[g], l) *
                        exp(-beta * r_g[g] * r_g[g]);
        }
    }
}

void contract_from_grid(double *in_g, double *out_i, int l, int ia, double *r_g,
                        double *dv_g, int ng, atc_basis_set *atco) {
    int *bas = atco->bas;
    double *env = atco->env;
    atc_atom *atcc = atco->atc_convs + ia;
    if (l > atcc->lmax) {
        printf("INTERNAL ERROR: l > lmax for atcc");
        exit(-1);
    }
    int ish0 = atcc->global_l_loc[l];
    int ish1 = atcc->global_l_loc[l + 1];
    int dish = ish1 - ish0;
    int ish;
    int *ibas;
    double coef, beta;
    for (ish = 0; ish < dish; ish++) {
        out_i[ish] = 0;
    }
    for (int g = 0; g < ng; g++) {
        for (ish = ish0; ish < ish1; ish++) {
            ibas = bas + ish * BAS_SLOTS;
            l = ibas[ANG_OF];
            coef = env[ibas[PTR_COEFF]];
            beta = env[ibas[PTR_EXP]];
            out_i[ish - ish0] += in_g[g] * coef * pow(r_g[g], l) *
                                 exp(-beta * r_g[g] * r_g[g]) * dv_g[g];
        }
    }
    // int info;
    // double *chomat = atcc->gtrans_0 + atcc->l_loc2[l];
    // int one = 1;
    // dpotrs_(&(atco->UPLO), &dish, &one, chomat, &dish, out_i, &dish, &info);
}

void contract_orb_to_rad_num(double *theta_rlmq, double *p_uq, double *funcs_jg,
                             int *jloc_l, int *uloc_l, int nrad, int nlm,
                             int nalpha) {
#pragma omp parallel
    {
        int j0, j1, u0, j, g, nm, l, q, m;
        double f;
        double *p_q, *theta_q;
        int lmax = sqrt(nlm + 1e-8) - 1;
#pragma omp for schedule(dynamic, 4)
        for (l = 0; l <= lmax; l++) {
            nm = 2 * l + 1;
            j0 = jloc_l[l];
            j1 = jloc_l[l + 1];
            u0 = uloc_l[l];
            for (j = j0; j < j1; j++) {
                for (g = 0; g < nrad; g++) {
                    f = funcs_jg[j * nrad + g];
                    for (m = 0; m < nm; m++) {
                        p_q = p_uq + (u0 + (j - j0) * nm + m) * nalpha;
                        theta_q = theta_rlmq + nalpha * (l * l + m + nlm * g);
                        for (q = 0; q < nalpha; q++) {
                            theta_q[q] += p_q[q] * f;
                        }
                    }
                }
            }
        }
    }
}

void contract_rad_to_orb_num(double *theta_rlmq, double *p_uq, double *funcs_jg,
                             int *jloc_l, int *uloc_l, int nrad, int nlm,
                             int nalpha) {
#pragma omp parallel
    {
        int j0, j1, u0, j, g, nm, l, q, m;
        double f;
        double *p_q, *theta_q;
        int lmax = sqrt(nlm + 1e-8) - 1;
#pragma omp for schedule(dynamic, 4)
        for (l = 0; l <= lmax; l++) {
            nm = 2 * l + 1;
            j0 = jloc_l[l];
            j1 = jloc_l[l + 1];
            u0 = uloc_l[l];
            for (j = j0; j < j1; j++) {
                for (g = 0; g < nrad; g++) {
                    f = funcs_jg[j * nrad + g];
                    for (m = 0; m < nm; m++) {
                        p_q = p_uq + (u0 + (j - j0) * nm + m) * nalpha;
                        theta_q = theta_rlmq + nalpha * (l * l + m + nlm * g);
                        for (q = 0; q < nalpha; q++) {
                            p_q[q] += theta_q[q] * f;
                        }
                    }
                }
            }
        }
    }
}

// TODO implementing these functions could be helpful for PAW implementation
/**
 * Given a radial distribution of functions for each spherical harmonic
 * lm on each atom, with dimension nalpha, project onto the orbital
 * basis set given by atco. Note: this computes projections onto each
 * (non-orthogonal) basis function, not expansion coefficients
 * theta_rlmq (nrad x nlm x nalpha) : input functions to project onto atco basis
 * p_uq (atco->nao x nalpha) : output projections
 * ra_loc (length natm + 1) : Range of rad indices that correspond
 *                            to each atom
 * rads (lenght nrad) : radial coordinates for each radial index
 * nrad : number of radial coordinates over all atoms
 * nlm : number of spherical harmonics (lmax + 1)^2
 * atco : stores the atomic basis set.
 * nalpha : number of functions stored in the rlm space.
 */
/**
void contract_rad_to_orb_with_bfvals(
    double *theta_rlmq, double *p_uq, double *func_ir, int *shl2rad_loc,
    int *ra_loc, double *rads, int nrad, int nlm,
    atc_basis_set *atco, int nalpha
)
{
#pragma omp parallel
{
    int ish, i0, L0, nm, l, at;
    double *p_mq, *theta_mq;
    int *bas = atco->bas;
    int *ao_loc = atco->ao_loc;
    double *env = atco->env;
    int nbas = atco->nbas;
    int *ibas;
    double coef, beta, val;
    int r, m, q, mq;
#pragma omp for schedule(dynamic, 4)
    for (ish = 0; ish < nbas; ish++) {
        ibas = bas + ish * BAS_SLOTS;
        at = ibas[ATOM_OF];
        l = ibas[ANG_OF];
        coef = env[ibas[PTR_COEFF]];
        beta = env[ibas[PTR_EXP]];
        i0 = ao_loc[ish];
        nm = 2*l+1;
        L0 = l*l;
        for (r = ra_loc[at]; r < ra_loc[at+1]; r++) {
            val = coef * pow(rads[r], l) * exp(-beta * rads[r] * rads[r]);
            theta_mq = theta_rlmq + nalpha * (r*nlm + L0);
            p_mq = p_uq + i0*nalpha;
            mq = 0;
            for (m = 0; m < nm; m++) {
                for (q = 0; q < nalpha; q++, mq++) {
                    p_mq[mq] += val * theta_mq[mq];
                }
            }
        }
    }
}
}
*/

/**
 * Backwards version of contract_rad_to_orb
 * (i.e. takes p_uq as input and projects the basis
 * functions out onto the radial coordinates and spherical harmonics).
 * See contract_rad_to_orb for variable definitions and details.
 */
/**
void contract_orb_to_rad_with_bfvals(
    double *theta_rlmq, double *p_uq, double *func_ir, int *shl2rad_loc,
    int *ar_loc, double *rads, int nrad, int nlm,
    atc_basis_set *atco, int nalpha
)
{
#pragma omp parallel
{
    int ish, i0, L0, nm, l, at;
    double *p_mq, *theta_mq;
    int *bas = atco->bas;
    int *ao_loc = atco->ao_loc;
    double *env = atco->env;
    int *ibas;
    double coef, beta, val;
    int r, m, q, mq;
#pragma omp for schedule(dynamic, 4)
    for (r = 0; r < nrad; r++) {
        at = ar_loc[r];
        for (ish = atco->atom_loc_ao[at]; ish < atco->atom_loc_ao[at+1]; ish++)
{ ibas = bas + ish * BAS_SLOTS; l = ibas[ANG_OF]; coef = env[ibas[PTR_COEFF]];
            beta = env[ibas[PTR_EXP]];
            i0 = ao_loc[ish];
            nm = 2*l+1;
            L0 = l*l;
            val = coef * pow(rads[r], l) * exp(-beta * rads[r] * rads[r]);
            theta_mq = theta_rlmq + nalpha * (r*nlm + L0);
            p_mq = p_uq + i0*nalpha;
            mq = 0;
            for (m = 0; m < nm; m++) {
                for (q = 0; q < nalpha; q++, mq++) {
                    theta_mq[mq] += val * p_mq[mq];
                }
            }
        }
    }
}
}
*/
