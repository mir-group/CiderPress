#ifndef NLDF_FFT_MPI_H
#define NLDF_FFT_MPI_H
#include "nldf_fft_core.h"

void ciderpw_setup_reciprocal_vectors(ciderpw_data data);

void ciderpw_g2k_mpi(ciderpw_data data);

void ciderpw_k2g_mpi(ciderpw_data data);

#endif
