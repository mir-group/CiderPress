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
