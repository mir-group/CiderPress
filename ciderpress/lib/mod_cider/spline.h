#ifndef CIDER_SPLINE_H
#define CIDER_SPLINE_H

void get_cubic_spline_coeff(double *x, double *y, double *spline, int N);
double spline_integral(double *spline, int N);

#endif
