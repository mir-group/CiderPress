#ifndef _CONV_INTERPOLATION_H
#define _CONV_INTERPOLATION_H

typedef struct {
    int *loc_i;
    int *num_i;
    double *rel_ord_coords;
    int *ind_ord_fwd;
    int *ind_ord_bwd;
    int nrad;
    double aparam;
    double dparam;
    int ngrids;
    int buffer_size;
} spline_locator;

typedef struct {
    spline_locator *sloc_list;
    int natm;
} spline_loc_list;

#endif
