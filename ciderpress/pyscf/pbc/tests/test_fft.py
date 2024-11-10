import ctypes

import numpy as np
from numpy.fft import fftn, ifftn, irfftn, rfftn

from ciderpress.pyscf.pbc.sdmx_fft import libcider


def _call_mkl_test(x, fwd, mesh):
    nx, ny, nz = mesh
    nzc = nz // 2 + 1
    if fwd:
        xr = x
        xk = np.empty((nx, ny, nzc), order="C", dtype=np.complex128)
    else:
        xk = x
        xr = np.empty((nx, ny, nz), order="C", dtype=np.float64)
    assert xr.shape == (nx, ny, nz)
    assert xk.shape == (nx, ny, nzc)
    libcider.test_fft3d(
        xr.ctypes.data_as(ctypes.c_void_p),
        xk.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nx),
        ctypes.c_int(ny),
        ctypes.c_int(nz),
        ctypes.c_int(1 if fwd else 0),
    )
    if fwd:
        return xk
    else:
        return xr


def main():
    np.random.seed(34)
    meshes = [
        [32, 100, 19],
        [32, 100, 20],
        [31, 99, 19],
        [2, 2, 4],
        [81, 81, 81],
        [80, 80, 80],
    ]

    for mesh in meshes:
        kmesh = [mesh[0], mesh[1], mesh[2] // 2 + 1]
        xrin = np.random.normal(size=mesh).astype(np.float64)
        xkin1 = np.random.normal(size=kmesh)
        xkin2 = np.random.normal(size=kmesh)
        xkin = np.empty(kmesh, dtype=np.complex128)
        xkin.real = xkin1
        xkin.imag = xkin2
        xkin[:] = rfftn(xrin, norm="backward")
        xkc = fftn(xrin.astype(np.complex128), norm="backward")
        # xkin.imag[..., 0] = 0
        # if mesh[-1] % 2 == 0:
        #     xkin.imag[..., -1] = 0
        xkin2 = xkin.copy()
        if False:  # (np.array(mesh) % 2 == 0).all():
            # xkin2[0, 0, -1] += 1j
            # xkin2[0, 0, 0] += 1j
            # xkin2[mesh[0]//2, mesh[1]//2, -1] += 50j
            cx = mesh[0] // 2
            cy = mesh[1] // 2
            const = 5j
            xkin2[cx, cy, 0] = const
            xkin2[-cx, -cy, 0] = np.conj(const)
            # print(mesh[0]//3, mesh[0])
            xkc[cx, cy, 0] = const
            xkc[-cx, -cy, 0] = np.conj(const)
            # xkin2[mesh[0]//2, mesh[1]//2, 0] += 1j
        if True:
            xkin2[0, 0, 0] = xkin2.real[0, 0, 0]
            xkin2[0, 0, -1] = xkin2.real[0, 0, -1]
            for ind in [0, -1]:
                # for i in range(1, (xkin2.shape[-2] + 1) // 2 ):
                for i in range(xkin2.shape[-3]):
                    for j in range(xkin2.shape[-2]):
                        # print(i, j, xkin2.shape[-2] - i, xkin2.shape[-1] - j)
                        tmp1 = xkin2[i, j, ind]
                        tmp2 = xkin2[-i, -j, ind]
                        xkin2[i, j, ind] = 0.5 * (tmp1 + tmp2.conj())
                        xkin2[-i, -j, ind] = 0.5 * (tmp1.conj() + tmp2)
                # xkin_tmp = xkin2[1:, 1:, ind][::-1, ::-1].conj().copy()
                # xkin2[1:, 1:, ind] += xkin_tmp
                # xkin2[1:, 1:, ind] *= 0.5
        # xkc = np.zeros(mesh, dtype=np.complex128)
        # xkc[..., :kmesh[2]] = xkin2
        # xkc[..., 0] = xkin[..., 0]
        # for i in range(1, (xkc.shape[-1] + 1) // 2):
        #     print(i, xkc.shape[-1] - i)
        #     xkc[..., i] = xkin[..., -i].conj()
        # if mesh[2] % 2 == 1:
        #     xkc[..., kmesh[2]:] = xkin2[..., -1:0:-1].conj()
        # else:
        #     xkc[..., kmesh[2]:] = xkin2[..., -2:0:-1].conj()
        if False:
            xkc[0, 0, 0] = xkc.real[0, 0, 0]
            xkc[0, 0, -1] = xkc.real[0, 0, -1]
            for ind in [0, -1]:
                xkin_tmp = xkc[1:, 1:, ind][::-1, ::-1].conj().copy()
                xkin2[1:, 1:, ind] += xkin_tmp
                xkin2[1:, 1:, ind] *= 0.5

        xk_np = rfftn(xrin)
        xk_mkl = _call_mkl_test(xrin, True, mesh)
        print(xk_np.shape, kmesh)
        assert (xk_np.shape == np.array(kmesh)).all()
        assert (xk_mkl.shape == np.array(kmesh)).all()
        print(np.linalg.norm(xk_np - xk_mkl))

        xr2_np = ifftn(xkc.copy(), s=mesh, norm="forward")
        xr_np = irfftn(xkin.copy(), s=mesh, norm="forward")
        xr3_np = irfftn(xkin2.copy(), s=mesh, norm="forward")
        xr_mkl = _call_mkl_test(xkin, False, mesh)
        xr2_mkl = _call_mkl_test(xkin2, False, mesh)
        print(xr_np.shape, mesh)
        assert (xr_np.shape == np.array(mesh)).all()
        assert (xr_mkl.shape == np.array(mesh)).all()
        print(np.linalg.norm(xr_mkl), np.linalg.norm(xr_np - xr_mkl))
        print(np.linalg.norm(xr2_mkl), np.linalg.norm(xr2_mkl - xr_mkl))
        print(np.linalg.norm(xr3_np), np.linalg.norm(xr2_mkl - xr_mkl))
        print("IMPORTANT", np.linalg.norm(xr2_np.imag), np.linalg.norm(xr2_np - xr3_np))
        if np.prod(mesh) < 100:
            print(xr_np)
            print(xr_mkl)


if __name__ == "__main__":
    main()
