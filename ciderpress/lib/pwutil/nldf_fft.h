#ifndef NLDF_FFT_H
#define NLDF_FFT_H

#include "config.h"
#ifdef HAVE_MPI
#include <fftw3-mpi.h>
#else
#include <fftw3.h>
#endif

#define CIDERPW_PI 3.14159265358979323846
#define CIDERPW_R2C 0
#define CIDERPW_C2C 1

struct ciderpw_unit_cell {
    double vec[9];
    int Nglobal[3];
    int Nlocal[3];
    int offset[3];
    double dV;
};

struct ciderpw_kernel {
    int kernel_type;
    int nalpha;
    int nbeta;
    int work_size;
    double *expnts_ba;
    double *norms_ba;
    double *expnts_ab;
    double *norms_ab;
    int num_l1_feats;
    int nk;
    double *k2_G;
    double *kx_G;
    double *ky_G;
    double *kz_G;
};

struct ciderpw_data_obj {
    int initialized;
    int nspin;
    struct ciderpw_unit_cell cell;
    struct ciderpw_unit_cell icell;
    struct ciderpw_kernel kernel;

    int fft_type;
#ifdef HAVE_MPI
    MPI_Comm mpi_comm;
#endif
    int mpi_rank;
    int mpi_size;

    int Ng;
    int Nglobal;
    int kLDA; // Shortest dimension of k i.e. Nglobal[2] for c2c
              // and Nglobal[2] / 2 + 1 for r2c
    int gLDA; // 2 * kLDA

    // work_ska contains theta and F:
    // spin first, then k/g, then alpha
    double complex *work_ska;

    // for NLDF, this will always be r2c and c2r, but might want
    // to reuse this struct for SDMX or R3.5, which will have c2c tranforms.
    fftw_plan plan_g2k;
    fftw_plan plan_k2g;

    int errorcode;
};

typedef struct ciderpw_data_obj *ciderpw_data;

#endif
