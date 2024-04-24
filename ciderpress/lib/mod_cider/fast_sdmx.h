#ifndef FAST_SDMX_H
#define FAST_SDMX_H
#include "pyscf_gto.h"
#include <stdint.h>
#include <stdlib.h>

void SDMXeval_rad_iter(FPtr_eval_sdmx_rad feval, FPtr_exp_sdmx fexp, double fac,
                       size_t nao, size_t ngrids, size_t bgrids, int param[],
                       int *shls_slice, double *buf, double *vbas,
                       double *coord, uint8_t *non0table, int *atm, int natm,
                       int *bas, int nbas, double *env, double *alphas,
                       double *alpha_norms, int nalpha);

#endif
