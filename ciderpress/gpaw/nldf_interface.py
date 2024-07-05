import ctypes

import numpy as np
from gpaw import cgpaw
from gpaw.xc.libvdwxc import FFTDistribution, nulltimer

from ciderpress.lib import c_null_ptr, load_library

pwutil = load_library("libpwutil")


class LibCiderPW:
    def __init__(self, N_c, cell_cv, comm):
        self.initialized = False
        ptr = c_null_ptr()
        self.shape = tuple(N_c)
        pwutil.ciderpw_create(
            ctypes.byref(ptr),
            ctypes.c_double(1),
            np.array(N_c).astype(np.int32).ctypes.data_as(ctypes.c_void_p),
            cell_cv.astype(np.float64).ctypes.data_as(ctypes.c_void_p),
        )
        self._ptr = ptr
        self.comm = comm

    def initialize_backend(self):
        assert not self.initialized

        comm = self.comm
        pwutil.ciderpw_init_mpi_from_gpaw(
            self._ptr,
            ctypes.py_object(comm.get_c_object()),
        )

        self.initialized = True

    def fft(self, in_xg, out_xg):
        for x in range(in_xg.shape[0]):
            pwutil.ciderpw_g2k_mpi_gpaw(
                self._ptr,
                in_xg[x].ctypes.data_as(ctypes.c_void_p),
                out_xg[x].ctypes.data_as(ctypes.c_void_p),
            )


class FFTWrapper:
    def __init__(self, fft_obj, dist1, dist2, pd, timer=nulltimer):
        self.fft_obj = fft_obj
        self.dist1 = dist1
        self.dist2 = dist2
        self.pd = pd
        self.timer = timer

    def run_fft(self, in_xg):
        zeros = self.dist1.block_zeros
        xshape = (len(in_xg),)
        inblock_xg = zeros(xshape)
        outshape = list(inblock_xg.shape)
        tmp_dist = FFTDistribution(self.pd.gd, [1, self.pd.gd.comm.size, 1])
        outshape = tmp_dist.local_output_size_c
        outshape = xshape + (outshape[1], outshape[0], outshape[2] // 2 + 1)
        outblock_xg = np.zeros(outshape, dtype=np.complex128)
        out_xg = np.zeros(
            [
                xshape[0],
                self.pd.gd.N_c[1],
                self.pd.gd.N_c[0],
                self.pd.gd.N_c[2] // 2 + 1,
            ],
            dtype=np.complex128,
        )
        self.dist1.gd2block(in_xg, inblock_xg)
        self.fft_obj.fft(inblock_xg, outblock_xg)
        contribs = []
        pd = self.pd
        # dummy = GridDescriptor(
        #    [gd.N_c[1], gd.N_c[0], gd.N_c[2]],
        #    comm=gd.comm,
        #    parsize_c=[gd.parsize_c[1], gd.parsize_c[0], gd.parsize_c[2]],
        # )
        # out_xg = dummy.zeros(n=xshape)
        # self.dist2.block2gd_add(outblock_xg, out_xg)
        for x in range(in_xg.shape[0]):
            if self.pd.gd.comm.rank == 0:
                self.pd.gd.comm.gather(outblock_xg[x].ravel(), 0, b=out_xg[x].ravel())
                out_g = np.ascontiguousarray(out_xg[x].transpose(1, 0, 2))
                to_scatter = out_g.ravel()[pd.Q_qG[0]]
            else:
                self.pd.gd.comm.gather(outblock_xg[x].ravel(), 0)
                to_scatter = None
            contribs.append(pd.scatter(to_scatter))
        return np.stack(contribs)

    def run_ifft(self, in_xk):
        contribs = []
        pd = self.pd
        for in_k in in_xk:
            in_k = pd.gather(in_k)
            cgpaw.pw_insert(in_k, pd.Q_qG[0], 1.0, pd.tmp_Q)
            contribs.append(pd.tmp_Q.copy())
        in_xg = np.stack(contribs)
        xshape = len(in_xg)
        zeros = self.distribution.block_zeros
        inblock_xg = zeros(xshape)
        outblock_xg = zeros(xshape)
        out_xg = np.zeros_like(in_xg)
        self.distribution.gd2block(in_xg, inblock_xg)
        self.nldf.call(inblock_xg, outblock_xg)
        self.distribution.block2gd_add(outblock_xg, out_xg)
        return out_xg


def wrap_fft_mpi(pd):
    parsize_c = [pd.gd.comm.size, 1, 1]
    dist1 = FFTDistribution(pd.gd, parsize_c)
    parsize_c = [1, pd.gd.comm.size, 1]
    dist2 = FFTDistribution(pd.gd, parsize_c)
    fft_obj = LibCiderPW(pd.gd.N_c, pd.gd.cell_cv, comm=pd.gd.comm)
    fft_obj.initialize_backend()
    wrapper = FFTWrapper(fft_obj, dist1, dist2, pd)
    return wrapper


if __name__ == "__main__":
    from gpaw.mpi import MPIComm, world

    np.random.seed(98)

    comm = MPIComm(world)
    N_c = [16, 12, 14]
    cell_cv = np.diag(np.ones(3))
    ciderpw = LibCiderPW(N_c, cell_cv, comm)
    ciderpw.initialize_backend()

    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.pw.descriptor import PWDescriptor

    gd = GridDescriptor(N_c, cell_cv, comm=MPIComm(world))
    pd = PWDescriptor(180, gd, dtype=float)

    wrapper = wrap_fft_mpi(pd)

    input = np.random.normal(size=N_c)[None, :, :, :]
    input = gd.distribute(input)

    output_ref = pd.fft(input)
    sum = np.abs(output_ref).sum()
    sum = pd.gd.comm.sum_scalar(sum)
    output_test = wrapper.run_fft(input)

    from numpy.testing import assert_allclose

    assert_allclose(output_test[0], output_ref)
