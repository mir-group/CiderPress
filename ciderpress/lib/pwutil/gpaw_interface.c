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

#include "gpaw_interface.h"
#include <Python.h>
#include <mpi.h>
#include <stdlib.h>

MPI_Comm unpack_gpaw_comm(PyObject *gpaw_mpi_obj) {
    MPIObject *gpaw_comm = (MPIObject *)gpaw_mpi_obj;
    return gpaw_comm->comm;
}

static void mpi_ensure_finalized(void) {
    int already_finalized = 1;
    int ierr = MPI_SUCCESS;

    MPI_Finalized(&already_finalized);
    if (!already_finalized) {
        ierr = MPI_Finalize();
    }
    if (ierr != MPI_SUCCESS)
        PyErr_SetString(PyExc_RuntimeError, "MPI_Finalize error occurred");
}

// MPI initialization
void mpi_ensure_initialized(void) {
    int already_initialized = 1;
    int ierr = MPI_SUCCESS;

    // Check whether MPI is already initialized
    MPI_Initialized(&already_initialized);
    if (!already_initialized) {
        // if not, let's initialize it
        int use_threads = 0;
// GPAW turns on threading for GPUs, but we don't have GPU support yet.
// #ifdef GPAW_GPU
//         use_threads = 1;
// #endif
#ifdef _OPENMP
        use_threads = 1;
#endif
        if (!use_threads) {
            ierr = MPI_Init(NULL, NULL);
            if (ierr == MPI_SUCCESS) {
                // No problem: register finalization when at Python exit
                Py_AtExit(*mpi_ensure_finalized);
            } else {
                // We have a problem: raise an exception
                char err[MPI_MAX_ERROR_STRING];
                int resultlen;
                MPI_Error_string(ierr, err, &resultlen);
                PyErr_SetString(PyExc_RuntimeError, err);
            }
        } else {
            int granted;
            ierr = MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &granted);
            if (ierr == MPI_SUCCESS && granted == MPI_THREAD_MULTIPLE) {
                // No problem: register finalization when at Python exit
                Py_AtExit(*mpi_ensure_finalized);
            } else if (granted != MPI_THREAD_MULTIPLE) {
                // We have a problem: raise an exception
                char err[MPI_MAX_ERROR_STRING] =
                    "MPI_THREAD_MULTIPLE is not supported";
                PyErr_SetString(PyExc_RuntimeError, err);
            } else {
                // We have a problem: raise an exception
                char err[MPI_MAX_ERROR_STRING];
                int resultlen;
                MPI_Error_string(ierr, err, &resultlen);
                PyErr_SetString(PyExc_RuntimeError, err);
            }
        }
    }
}
