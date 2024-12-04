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

#ifndef NLDF_FFT_SERIAL_H
#define NLDF_FFT_SERIAL_H
#include "nldf_fft_core.h"

void ciderpw_setup_reciprocal_vectors(ciderpw_data data);

void ciderpw_g2k_serial(ciderpw_data data);

void ciderpw_k2g_serial(ciderpw_data data);

#endif
