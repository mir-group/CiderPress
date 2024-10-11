#include "gpaw_interface.h"
#include <Python.h>
#include <mpi.h>
#include <stdlib.h>

MPI_Comm unpack_gpaw_comm(PyObject *gpaw_mpi_obj) {
    MPIObject *gpaw_comm = (MPIObject *)gpaw_mpi_obj;
    return gpaw_comm->comm;
}
