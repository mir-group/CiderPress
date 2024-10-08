#ifndef NLDF_FFT_SERIAL_H
#define NLDF_FFT_SERIAL_H
#include "nldf_fft_core.h"

void ciderpw_setup_reciprocal_vectors(ciderpw_data data);

void ciderpw_g2k_serial(ciderpw_data data);

void ciderpw_k2g_serial(ciderpw_data data);

#endif
