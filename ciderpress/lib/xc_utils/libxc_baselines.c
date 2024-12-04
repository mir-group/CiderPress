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

#include <stddef.h>
#include <stdlib.h>
#include <xc.h>

void get_lda_baseline(int fn_id, int nspin, int size, double *rho, double *exc,
                      double *vrho, double dens_threshold) {
    xc_func_type func;
    xc_func_init(&func, fn_id, nspin);
    xc_func_set_dens_threshold(&func, dens_threshold);
    xc_lda_exc_vxc(&func, size, rho, exc, vrho);
}

void get_gga_baseline(int fn_id, int nspin, int size, double *rho,
                      double *sigma, double *exc, double *vrho, double *vsigma,
                      double dens_threshold) {
    xc_func_type func;
    xc_func_init(&func, fn_id, nspin);
    xc_func_set_dens_threshold(&func, dens_threshold);
    xc_gga_exc_vxc(&func, size, rho, sigma, exc, vrho, vsigma);
}

void get_mgga_baseline(int fn_id, int nspin, int size, double *rho,
                       double *sigma, double *tau, double *exc, double *vrho,
                       double *vsigma, double *vtau, double dens_threshold) {
    xc_func_type func;
    xc_func_init(&func, fn_id, nspin);
    xc_func_set_dens_threshold(&func, dens_threshold);
    xc_mgga_exc_vxc(&func, size, rho, sigma, rho, tau, exc, vrho, vsigma, NULL,
                    vtau);
}
