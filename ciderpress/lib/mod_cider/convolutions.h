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

#ifndef _CIDER_CONVOLUTIONS_H
#define _CIDER_CONVOLUTIONS_H

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define BAS_SLOTS 8
#define ATOM_OF 0
#define ANG_OF 1
#define PTR_EXP 5
#define PTR_COEFF 6

/**
 * Contains indexing and exponent/coefficient info for the
 * individual atomic contributions to atc_basis_set.
 * Each atc_basis_set contains a list of these with length
 * natm.
 */
typedef struct {
    int ngsh;          // number of shells in this atom
    int ngsh2;         // \sum_l (number of l shells)^2 in this atom
    int ia;            // atom index
    int ia_loc;        // first index in gamma_loc for this atom
    int lmax;          // maximum l value for this atom
    int *l_loc;        // start indexes of gammas/gcoefs at each l
    int *global_l_loc; // global_l_loc + ia+loc, for convenience
    int *l_loc2;       // start indexes of gamma ovlp (gtrans_*) at each l
    double *gammas;    // exponents of the basis
    double *gcoefs;    // coefficients of the basis
    double *gtrans_0;  // transformation (inverted overlap) matrix for gammas
    double *gtrans_m;  // transformation (inverted overlap) matrix for gammas
                      // with l-1 used for the principal angular momentum number
    double *gtrans_p; // transformation (inverted overlap) matrix for gammas
                      // with l+1 used for the principal angular momentum number
} atc_atom;

/**
 * Contains a simple, uncontracted Gaussian basis set
 * for use in expanding theta or convolutions for CIDER.
 */
typedef struct {
    atc_atom *atc_convs; // list of atomic convolution structs
    int *atom_loc_ao;    // initial bas index for each atom
    int *bas;            // pseudo-pyscf bas for calling compute_gaussians
    int *ao_loc;         // location of each shell in an array of orbital coefs
    double *env;         // pseudo-pyscf env for calling compute_gaussians
    int natm;            // number of atoms
    int nbas;            // number of bas shells (length of bas)
    int nao;             // number of orbitals (last element of ao_loc)
    char UPLO;           // upper or lower for cholesky factorizations
} atc_basis_set;

/**
 * Contains convolution matrix elements between
 * atco_inp and atco_out. Overlap values are stored
 * in ovlp_mats, and locations are stored in pair_loc
 */
typedef struct {
    int natm;                // Number of atoms in the system.
    int tot_shl_pairs;       // Total number of shell pairs
    int nalpha;              // dimension of convolution input
    int nbeta;               // dimension of convolution output
    int nfeat_i;             // number of version i intermediate features
    int has_vj;              // whether to compute vj intermediate features
    int *icontrib_ids;       // id numbers of version i intermediate features.
    int *feat_orders;        // -1, 0, or 1 for each output, indicating order
                             // of integrals relative to l for each beta index.
    int max_pair;            // maximum number of pairs for a given ia,l
    atc_basis_set *atco_inp; // Basis for input to convolutions
    atc_basis_set *atco_out; // Basis for output of convolutions
    int **pair_loc;          // Location of convolution matrix with
                             // pair_loc[ia][l] * nalpha * nbeta
                             // being the location in ovlp_mats of a
                             // nshl_l_out * nshl_l_in * nbeta * nalpha tensor.
    double *alphas;          // control point exponents
    double *alpha_norms;     // normalization factors for control points
    double *ovlp_mats;       // A ragged tensor with the convolution
                             // overlaps. See pair_loc for indexing.
} convolution_collection;

#endif
