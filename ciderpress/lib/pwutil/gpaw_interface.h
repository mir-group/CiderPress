#ifndef GPAW_INTERFACE_H
#define GPAW_INTERFACE_H

#include <Python.h>
#include <mpi.h>
#include <stdlib.h>

typedef struct {
    PyObject_HEAD int size;
    int rank;
    MPI_Comm comm;
    PyObject *parent;
    int *members;
} MPIObject;

MPI_Comm unpack_gpaw_comm(PyObject *gpaw_mpi_obj) {
    MPIObject *gpaw_comm = (MPIObject *)gpaw_mpi_obj;
    return gpaw_comm->comm;
}

#endif
