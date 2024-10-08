import numpy as np


def get_sigma(grad_svg):
    nspin = grad_svg.shape[0]
    shape = grad_svg.shape[2:]
    if nspin not in [1, 2]:
        raise ValueError("Only 1 or 2 spins supported")
    sigma_xg = np.empty((2 * nspin - 1,) + shape)
    sigma_xg[::2] = np.einsum("sv...,sv...->s...", grad_svg, grad_svg)
    if nspin == 2:
        sigma_xg[1] = np.einsum("v...,v...->...", grad_svg[0], grad_svg[1])
    return sigma_xg


def get_dsigmadf(grad_svg, dgraddf_ovg, spins):
    norb = len(spins)
    assert dgraddf_ovg.shape[0] == norb
    shape = grad_svg.shape[2:]
    dsigmadf_og = np.empty((norb,) + shape)
    for o, s in enumerate(spins):
        dsigmadf_og[o] = 2 * np.einsum("vg,vg->g", grad_svg[s], dgraddf_ovg[o])
    return dsigmadf_og
