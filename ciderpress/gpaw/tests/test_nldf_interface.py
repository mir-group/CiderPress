import unittest

import numpy as np
from gpaw.mpi import MPIComm, world
from gpaw.pw.descriptor import PWDescriptor
from numpy.testing import assert_allclose

from ciderpress.gpaw.nldf_interface import GridDescriptor, LibCiderPW, wrap_fft_mpi


class TestLibCiderPW(unittest.TestCase):
    def test_pwutil(self):
        np.random.seed(98)
        comm = MPIComm(world)
        N_c = [100, 80, 110]
        cell_cv = np.diag(np.ones(3))
        ciderpw = LibCiderPW(N_c, cell_cv, comm)
        ciderpw.initialize_backend()
        gd = GridDescriptor(N_c, cell_cv, comm=MPIComm(world))
        pd = PWDescriptor(180, gd, dtype=float)
        wrapper = wrap_fft_mpi(pd)
        input = np.random.normal(size=N_c)[None, :, :, :]
        input = gd.distribute(input)
        output_ref = pd.fft(input)
        sum = np.abs(output_ref).sum()
        sum = pd.gd.comm.sum_scalar(sum)
        for i in range(20):
            output_test = wrapper.fft(input)
        input_test = wrapper.ifft(output_test)
        input_ref = pd.ifft(output_ref)
        assert_allclose(output_test[0], output_ref)
        assert_allclose(input_test[0], input_ref)


if __name__ == "__main__":
    unittest.main()
