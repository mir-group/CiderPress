#!/usr/bin/env python
# CiderPress: Machine-learning based density functional theory calculations
# Copyright (C) 2024 The President and Fellows of Harvard College
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

import ctypes

import numpy as np

try:
    from gpaw import cgpaw
except ImportError:
    import _gpaw as cgpaw
from gpaw.grid_descriptor import GridDescriptor
from gpaw.xc.libvdwxc import FFTDistribution, nulltimer

from ciderpress.lib import c_null_ptr, load_library

pwutil = load_library("libpwutil")
pwutil.ciderpw_get_work_pointer.restype = ctypes.POINTER(ctypes.c_double)
pwutil.ciderpw_has_mpi.restype = ctypes.c_int


def get_norms_and_expnts_from_plan(plan):
    nalpha = plan.nalpha
    alphas = plan.alphas
    alpha_norms = plan.alpha_norms
    settings = plan.nldf_settings
    version = settings.version
    if version in ["j", "ij", "k"]:
        nbeta = nalpha
    else:
        nbeta = 0
    norms_ab = np.empty((nalpha, nbeta))
    expnts_ab = np.empty((nalpha, nbeta))
    t = 0
    if version in ["j", "ij", "k"]:
        aexp = alphas + alphas[:, None]
        fac = (np.pi / aexp) ** 1.5
        norms_ab[:, :nalpha] = fac * alpha_norms * alpha_norms[:, None]
        expnts_ab[:, :nalpha] = 1.0 / (4 * aexp)
        t = nalpha
    if version in ["i", "ij"]:
        for spec in settings.l0_feat_specs:
            fac = alpha_norms * (np.pi / alphas) ** 1.5
            # TODO this won't be correct for r^2 and lapl ones.
            if spec == "se":
                fac *= 1
            elif spec == "se_r2":
                fac *= 1
            elif spec == "se_apr2":
                fac *= alphas
            elif spec == "se_ap":
                fac *= alphas
            elif spec == "se_ap2r2":
                fac *= alphas * alphas
            elif spec == "se_lapl":
                fac *= 1
            else:
                raise ValueError
            norms_ab[:, t] = fac
            expnts_ab[:, t] = 1.0 / (4 * alphas)
            t += 1
        for spec in settings.l1_feat_specs:
            fac = alpha_norms * (np.pi / alphas) ** 1.5
            # TODO might be rt(3) or 1/2 prefactor or something
            if spec == "se_grad":
                fac *= 1.0
            elif spec == "se_rvec":
                fac /= alphas
            else:
                raise ValueError
            norms_ab[:, t : t + 3] = fac[:, None]
            expnts_ab[:, t : t + 3] = alphas[:, None]
            t += 3
    return nalpha, nbeta, alphas, alpha_norms, norms_ab, expnts_ab


class LibCiderPW:
    def __init__(self, N_c, cell_cv, comm, plan=None):
        self.initialized = False
        ptr = c_null_ptr()
        self.shape = tuple(N_c)
        if plan is None:
            self.alpha_norms = np.ones(1)
            self.alphas = np.ones(1)
            self.nalpha = 1
            self.nbeta = 1
            self.norms_ab = np.ones((1, 1))
            self.expnts_ab = np.ones((1, 1))
        else:
            (
                self.nalpha,
                self.nbeta,
                self.alphas,
                self.alpha_norms,
                self.norms_ab,
                self.expnts_ab,
            ) = get_norms_and_expnts_from_plan(plan)
        alphas = np.asarray(self.alphas, dtype=np.float64, order="C")
        alpha_norms = np.asarray(self.alpha_norms, dtype=np.float64, order="C")
        assert len(alphas) == len(alpha_norms)
        assert cell_cv.flags.c_contiguous
        pwutil.ciderpw_create(
            ctypes.byref(ptr),
            ctypes.c_double(1),
            np.array(N_c).astype(np.int32).ctypes.data_as(ctypes.c_void_p),
            cell_cv.astype(np.float64).ctypes.data_as(ctypes.c_void_p),
        )
        self._ptr = ptr
        self.comm = comm
        self._local_shape_real = None
        self._local_shape_recip = None
        self._glda = None
        self._klda = None
        self._data_arr = None
        if not self.has_mpi and self.comm.size > 1:
            raise ValueError(
                "CIDER is only compiled for serial evaluation, but the GPAW calculation is parallel with {} processes.".format(
                    self.comm.size
                )
            )

    @property
    def has_mpi(self):
        return bool(int(pwutil.ciderpw_has_mpi()))

    def get_real_array(self):
        """
        Return a numpy array whose data is the real-space values
        of the C work array used for FFTW.
        """
        raise NotImplementedError
        return self._data_arr[:, :, :, : self._local_shape_real[2], :]

    def get_reciprocal_array(self):
        """
        Return a numpy array whose data is the reciprocal-space values
        of the C work array used for FFTW.
        """
        raise NotImplementedError
        arr = self._data_arr.view()
        nspin = 1
        nalpha = self.nalpha
        shape = self._local_shape_recip
        lda = self._klda
        shape = [nspin, shape[0], shape[1], lda, 2 * nalpha]
        arr.shape = shape
        arr.dtype = np.complex128
        shape[-1] = nalpha
        assert arr.shape == shape
        assert arr.flags.c_contiguous
        return arr

    def __del__(self):
        pwutil.ciderpw_finalize(ctypes.byref(self._ptr))
        self._ptr = None

    def initialize_backend(self):
        assert not self.initialized

        if self.has_mpi:
            comm = self.comm
            pwutil.ciderpw_init_mpi_from_gpaw(
                self._ptr,
                ctypes.py_object(comm.get_c_object()),
                ctypes.c_int(self.nalpha),
                ctypes.c_int(self.nbeta),
                self.norms_ab.ctypes.data_as(ctypes.c_void_p),
                self.expnts_ab.ctypes.data_as(ctypes.c_void_p),
            )
        else:
            pwutil.ciderpw_init_serial(
                self._ptr,
                ctypes.c_int(self.nalpha),
                ctypes.c_int(self.nbeta),
                self.norms_ab.ctypes.data_as(ctypes.c_void_p),
                self.expnts_ab.ctypes.data_as(ctypes.c_void_p),
            )

        size_data = np.zeros(8, dtype=np.int32)
        pwutil.ciderpw_get_local_size_and_lda(
            self._ptr, size_data.ctypes.data_as(ctypes.c_void_p)
        )
        self._local_shape_real = size_data[0:3]
        self._glda = size_data[3]
        self._local_shape_recip = size_data[4:7]
        self._klda = size_data[7]
        """
        obj = pwutil.ciderpw_get_work_pointer(self._ptr)
        # TODO nspin
        nspin = 1
        nalpha = self.nalpha
        lda = self._glda
        shape = self._local_shape_real
        shape = (nspin, shape[0], shape[1], lda, nalpha)
        arr = np.ctypeslib.as_array(obj, shape=shape)
        self._data_arr = arr
        """

        self.initialized = True

    def fftv2(self, arr_gq):
        pwutil.ciderpw_g2k_v2(
            self._ptr,
            arr_gq.ctypes.data_as(ctypes.c_void_p),
        )

    def fft(self, in_xg, out_xg):
        if self.has_mpi:
            fn = pwutil.ciderpw_g2k_mpi_gpaw
        else:
            fn = pwutil.ciderpw_g2k_serial_gpaw
        for x in range(in_xg.shape[0]):
            fn(
                self._ptr,
                in_xg[x].ctypes.data_as(ctypes.c_void_p),
                out_xg[x].ctypes.data_as(ctypes.c_void_p),
            )

    def ifft(self, in_xg, out_xg):
        if self.has_mpi:
            fn = pwutil.ciderpw_k2g_mpi_gpaw
        else:
            fn = pwutil.ciderpw_k2g_serial_gpaw
        for x in range(in_xg.shape[0]):
            fn(
                self._ptr,
                in_xg[x].ctypes.data_as(ctypes.c_void_p),
                out_xg[x].ctypes.data_as(ctypes.c_void_p),
            )

    def compute_forward_convolution(self):
        pwutil.ciderpw_compute_features(self._ptr)

    def compute_backward_convolution(self):
        pwutil.ciderpw_compute_potential(self._ptr)

    def compute_backward_convolution_with_stress(self, theta_gq):
        stress_vv = np.zeros((3, 3))
        pwutil.ciderpw_convolution_potential_and_stress(
            self._ptr,
            stress_vv.ctypes.data_as(ctypes.c_void_p),
            theta_gq.ctypes.data_as(ctypes.c_void_p),
        )
        return stress_vv

    def fill_vj_feature_(self, feat_g, p_gq):
        assert feat_g.flags.c_contiguous
        assert p_gq.flags.c_contiguous
        assert feat_g.shape == p_gq.shape[:-1] == tuple(self._local_shape_real)
        assert p_gq.shape[-1] == self.nalpha
        pwutil.ciderpw_eval_feature_vj(
            self._ptr,
            feat_g.ctypes.data_as(ctypes.c_void_p),
            p_gq.ctypes.data_as(ctypes.c_void_p),
        )

    def fill_vj_potential_(self, vfeat_g, p_gq):
        assert vfeat_g.flags.c_contiguous
        assert p_gq.flags.c_contiguous
        assert vfeat_g.shape == p_gq.shape[:-1] == tuple(self._local_shape_real)
        assert p_gq.shape[-1] == self.nalpha
        pwutil.ciderpw_add_potential_vj(
            self._ptr,
            vfeat_g.ctypes.data_as(ctypes.c_void_p),
            p_gq.ctypes.data_as(ctypes.c_void_p),
        )

    def set_work(self, fun_g, p_gq):
        assert fun_g.flags.c_contiguous
        assert p_gq.flags.c_contiguous
        assert fun_g.shape == p_gq.shape[:-1] == tuple(self._local_shape_real)
        assert p_gq.shape[-1] == self.nalpha
        pwutil.ciderpw_set_work(
            self._ptr,
            fun_g.ctypes.data_as(ctypes.c_void_p),
            p_gq.ctypes.data_as(ctypes.c_void_p),
        )

    def reset_work(self):
        pwutil.ciderpw_zero_work(self._ptr)

    def get_radial_info_for_atom(self, local_indices, local_fdisps):
        num_c = []
        for i in range(3):
            num_c.append(len(local_indices[i]))
            assert len(local_fdisps[i]) == num_c[i]
        ngrid = np.prod(num_c)
        r_g = np.empty((4, ngrid), order="C")
        inds = np.ascontiguousarray(np.concatenate(local_indices).astype(np.int64))
        disps = np.ascontiguousarray(np.concatenate(local_fdisps).astype(np.float64))
        locs_g = np.empty(ngrid, dtype=np.int64)
        num_c = np.asarray(num_c, order="C", dtype=np.int64)
        pwutil.ciderpw_fill_atom_info(
            self._ptr,
            inds.ctypes.data_as(ctypes.c_void_p),
            disps.ctypes.data_as(ctypes.c_void_p),
            num_c.ctypes.data_as(ctypes.c_void_p),
            r_g.ctypes.data_as(ctypes.c_void_p),
            locs_g.ctypes.data_as(ctypes.c_void_p),
        )
        return locs_g, r_g[0], r_g[1:].T

    def add_paw2grid(self, funcs_gb, indset):
        assert funcs_gb.flags.c_contiguous
        assert funcs_gb.dtype == np.float64
        assert indset.flags.c_contiguous
        assert indset.dtype == np.int64
        pwutil.ciderpw_add_atom_info(
            self._ptr,
            funcs_gb.ctypes.data_as(ctypes.c_void_p),
            indset.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(indset.size),
        )

    def set_grid2paw(self, funcs_gb, indset):
        assert funcs_gb.flags.c_contiguous
        assert indset.flags.c_contiguous
        pwutil.ciderpw_set_atom_info(
            self._ptr,
            funcs_gb.ctypes.data_as(ctypes.c_void_p),
            indset.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(indset.size),
        )

    def get_bound_inds(self):
        bound_inds = np.zeros(6, dtype=np.int32)
        pwutil.ciderpw_get_bound_inds(
            self._ptr, bound_inds.ctypes.data_as(ctypes.c_void_p)
        )
        return bound_inds[:3], bound_inds[3:]

    def get_recip_size(self):
        pwutil.ciderpw_get_recip_size.restype = ctypes.c_size_t
        return np.uintp(pwutil.ciderpw_get_recip_size(self._ptr))

    def get_real_size(self):
        pwutil.ciderpw_get_real_size.restype = ctypes.c_size_t
        return np.uintp(pwutil.ciderpw_get_real_size(self._ptr))

    def get_work_size(self):
        pwutil.ciderpw_get_work_size.restype = ctypes.c_size_t
        return np.uintp(pwutil.ciderpw_get_work_size(self._ptr))

    def copy_work_array(self):
        arr = np.empty(self.get_work_size() * self.nalpha, dtype=np.complex128)
        pwutil.ciderpw_copy_work_array(self._ptr, arr.ctypes.data_as(ctypes.c_void_p))
        return arr


class FFTWrapper:
    def __init__(self, fft_obj, distribution, pd, timer=nulltimer):
        self.fft_obj = fft_obj
        self.dist1 = distribution
        gd = pd.gd
        if fft_obj.has_mpi:
            nrecip_c = [gd.N_c[1], gd.N_c[0], gd.N_c[2] // 2 + 1]
            parsize_c = [gd.parsize_c[1], gd.parsize_c[0], gd.parsize_c[2]]
        else:
            nrecip_c = [gd.N_c[0], gd.N_c[1], gd.N_c[2] // 2 + 1]
            parsize_c = [gd.parsize_c[0], gd.parsize_c[1], gd.parsize_c[2]]
        self.dummy_gd = GridDescriptor(
            nrecip_c,
            comm=gd.comm,
            parsize_c=parsize_c,
        )
        self.dist2 = FFTDistribution(self.dummy_gd, [gd.comm.size, 1, 1])
        self.pd = pd
        self.timer = timer

    def get_reciprocal_vectors(self, q=0, add_q=True):
        return self.pd.get_reciprocal_vectors(q=q, add_q=True)

    @property
    def G2_qG(self):
        return self.pd.G2_qG

    def integrate(self, *args, **kwargs):
        return self.pd.integrate(*args, **kwargs)

    def zeros(self, x=(), dtype=None, q=None, global_array=False):
        return self.pd.zeros(x=x, dtype=dtype, q=q, global_array=global_array)

    def empty(self, x=(), dtype=None, q=None, global_array=False):
        return self.pd.empty(x=x, dtype=dtype, q=q, global_array=global_array)

    def fft(self, in_xg, q=None, Q_G=None, local=False):
        if q is not None or Q_G is not None or local:
            raise NotImplementedError
        if in_xg.ndim == 3:
            in_xg = in_xg[None, ...]
        assert in_xg.ndim == 4
        xshape = (len(in_xg),)
        inblock_xg = self.dist1.block_zeros(xshape)
        outblock_xg = self.dist2.block_zeros(xshape).astype(np.complex128)
        self.dist1.gd2block(in_xg, inblock_xg)
        self.fft_obj.fft(inblock_xg, outblock_xg)
        contribs = []
        out_xg = self.dummy_gd.zeros(n=xshape, dtype=complex, global_array=False)
        self.dist2.block2gd_add(outblock_xg, out_xg)
        pd = self.pd
        for x in range(in_xg.shape[0]):
            out_g = self.dummy_gd.collect(out_xg[x])
            if self.pd.gd.comm.rank == 0:
                if self.fft_obj.has_mpi:
                    out_g = np.ascontiguousarray(out_g.transpose(1, 0, 2))
                to_scatter = out_g.ravel()[pd.Q_qG[0]]
            else:
                to_scatter = None
            contribs.append(pd.scatter(to_scatter))
        return np.stack(contribs)

    def ifft(self, in_xk, q=None, Q_G=None, local=False):
        if q is not None or Q_G is not None or local:
            raise NotImplementedError
        if in_xk.ndim == 1:
            in_xk = in_xk[None, ...]
        assert in_xk.ndim == 2
        contribs = []
        pd = self.pd
        for in_k in in_xk:
            in_k = pd.gather(in_k)
            if in_k is not None:
                pd.tmp_Q[:] = 0
                scale = 1.0 / self.pd.tmp_R.size
                cgpaw.pw_insert(in_k, pd.Q_qG[0], scale, pd.tmp_Q)
                t = pd.tmp_Q[:, :, 0]
                n, m = pd.gd.N_c[:2] // 2 - 1
                t[0, -m:] = t[0, m:0:-1].conj()
                t[n:0:-1, -m:] = t[-n:, m:0:-1].conj()
                t[-n:, -m:] = t[n:0:-1, m:0:-1].conj()
                t[-n:, 0] = t[n:0:-1, 0].conj()
            if self.fft_obj.has_mpi:
                tmp = pd.tmp_Q.transpose(1, 0, 2).copy()
            else:
                tmp = pd.tmp_Q.copy()
            contribs.append(self.dummy_gd.distribute(tmp))
        in_xg = np.stack(contribs)
        xshape = (len(in_xg),)
        inblock_xg = self.dist2.block_zeros(xshape).astype(np.complex128)
        outblock_xg = self.dist1.block_zeros(xshape)
        self.dist2.gd2block(in_xg, inblock_xg)
        self.fft_obj.ifft(inblock_xg, outblock_xg)
        out_xg = self.pd.gd.zeros(n=xshape)
        self.dist1.block2gd_add(outblock_xg, out_xg)
        return out_xg


def wrap_fft_mpi(pd):
    parsize_c = [pd.gd.comm.size, 1, 1]
    distribution = FFTDistribution(pd.gd, parsize_c)
    parsize_c = [1, pd.gd.comm.size, 1]
    fft_obj = LibCiderPW(pd.gd.N_c, pd.gd.cell_cv, comm=pd.gd.comm)
    fft_obj.initialize_backend()
    wrapper = FFTWrapper(fft_obj, distribution, pd)
    return wrapper


if __name__ == "__main__":
    from gpaw.mpi import MPIComm, world

    np.random.seed(98)

    comm = MPIComm(world)
    N_c = [100, 80, 110]
    cell_cv = np.diag(np.ones(3))
    ciderpw = LibCiderPW(N_c, cell_cv, comm)
    ciderpw.initialize_backend()

    from gpaw.pw.descriptor import PWDescriptor

    gd = GridDescriptor(N_c, cell_cv, comm=MPIComm(world))
    pd = PWDescriptor(180, gd, dtype=float)

    wrapper = wrap_fft_mpi(pd)

    input = np.random.normal(size=N_c)[None, :, :, :]
    input = gd.distribute(input)

    output_ref = pd.fft(input)
    sum = np.abs(output_ref).sum()
    sum = pd.gd.comm.sum_scalar(sum)
    import time

    t0 = time.monotonic()
    for i in range(20):
        output_test = wrapper.fft(input)
    t1 = time.monotonic()
    print("TIME", t1 - t0)
    input_test = wrapper.ifft(output_test)
    input_ref = pd.ifft(output_ref)

    from numpy.testing import assert_allclose

    assert_allclose(output_test[0], output_ref)
    # assert_allclose(input_ref, input[0])
    assert_allclose(input_test[0], input_ref)
