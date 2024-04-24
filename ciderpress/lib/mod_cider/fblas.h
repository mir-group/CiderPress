#ifndef _CIDER_FBLAS_H
#define _CIDER_FBLAS_H

#if defined __cplusplus
extern "C" {
#endif
#include <complex.h>

double ddot_(const int *, const double *, const int *, const double *,
             const int *);

void dgemv_(const char *, const int *, const int *, const double *,
            const double *, const int *, const double *, const int *,
            const double *, const double *, const int *);

void dtrtri_(const char *, const char *, const int *, const double *,
             const int *, const int *);

void dtrmv_(const char *, const char *, const char *, const int *,
            const double *, const int *, const double *, const int *);

void dgemm_(const char *, const char *, const int *, const int *, const int *,
            const double *, const double *, const int *, const double *,
            const int *, const double *, double *, const int *);

void dpotrs_(const char *, const int *, const int *, const double *,
             const int *, const double *, const int *, const int *);

void dpotrf_(const char *, const int *, const double *, const int *,
             const int *);

#if defined __cplusplus
} // end extern "C"
#endif

#endif
