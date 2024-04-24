import ctypes

import numpy as np

from ciderpress.lib import load_library as load_cider_library
from ciderpress.new_dft.plans import get_cider_exponent

libcider = load_cider_library("libmcider")


def get_get_exponent(gg_kwargs):
    def _get_exponent(rho_data):
        rho = rho_data[0]
        sigma = np.einsum("xg,xg->g", rho_data[1:4], rho_data[1:4])
        tau = rho_data[4]
        return get_cider_exponent(
            rho,
            sigma,
            tau,
            a0=gg_kwargs["a0"],
            grad_mul=0,
            tau_mul=gg_kwargs["fac_mul"],
            nspin=1,
        )[0]

    return _get_exponent


def get_nonlocal_features(
    rho, coords, vvrho, vvweight, vvcoords, get_exponent_r, get_exponent_rp, version="i"
):
    thresh = 1e-8
    nfeat = 12 if version == "i" else 4
    feat = np.zeros((rho[0].size, nfeat))

    threshind = rho[0] >= thresh
    coords = coords[threshind]
    expnt_data = get_exponent_r(rho[:, threshind])

    innerthreshind = vvrho[0] >= thresh
    vvcoords = vvcoords[innerthreshind]
    vvweight = vvweight[innerthreshind]
    vvexpnt_data = get_exponent_rp(vvrho[:, innerthreshind])

    vvf = vvrho[0, innerthreshind] * vvweight

    vvcoords = np.asarray(vvcoords, order="C")
    coords = np.asarray(coords, order="C")
    _feat = np.empty((expnt_data.shape[0], nfeat))

    if version == "i":
        libcider.debug_numint_vi(
            _feat.ctypes.data_as(ctypes.c_void_p),
            vvexpnt_data.ctypes.data_as(ctypes.c_void_p),
            vvf.ctypes.data_as(ctypes.c_void_p),
            vvcoords.ctypes.data_as(ctypes.c_void_p),
            coords.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(vvcoords.shape[0]),
            ctypes.c_int(coords.shape[0]),
        )
    else:
        if version == "j":
            fn = libcider.debug_numint_vj
        elif version == "k":
            fn = libcider.debug_numint_vk
        else:
            raise ValueError("Version must be i, j, or k")
        fn(
            _feat.ctypes.data_as(ctypes.c_void_p),
            vvexpnt_data.ctypes.data_as(ctypes.c_void_p),
            expnt_data.ctypes.data_as(ctypes.c_void_p),
            vvf.ctypes.data_as(ctypes.c_void_p),
            vvcoords.ctypes.data_as(ctypes.c_void_p),
            coords.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(vvcoords.shape[0]),
            ctypes.c_int(coords.shape[0]),
            ctypes.c_double(2.0),
        )
    feat[threshind, :] = _feat
    return feat
