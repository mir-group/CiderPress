import unittest

import numpy as np
from ase.build import bulk
from ase.dft.bandgap import bandgap
from ase.parallel import parprint
from ase.units import Ha
from gpaw import GPAW, PW, Mixer
from gpaw.mpi import world
from gpaw.xc import XC
from numpy.testing import assert_almost_equal

from ciderpress.gpaw.calculator import get_cider_functional
from ciderpress.gpaw.xc_tools import non_self_consistent_eigenvalues as nscfeig

USE_FAST_CIDER = True

if USE_FAST_CIDER:
    from ciderpress.gpaw.fast_descriptors import (
        get_descriptors,
        get_drho_df,
        get_homo_lumo_fd_,
        interpolate_drhodf,
        run_constant_occ_calculation_,
    )
    from ciderpress.gpaw.interp_paw import DiffGGA
    from ciderpress.gpaw.interp_paw import DiffMGGA2 as DiffMGGA

    get_features = get_descriptors
else:
    from ciderpress.gpaw.descriptors import (
        get_drho_df,
        get_features,
        get_homo_lumo_fd_,
        interpolate_drhodf,
        run_constant_occ_calculation_,
    )
    from ciderpress.gpaw.interp_paw import DiffGGA, DiffMGGA


def get_xc(fname, use_paw=True, force_nl=False):
    return get_cider_functional(
        fname,
        qmax=300,
        lambd=1.8,
        xmix=0.25,
        pasdw_ovlp_fit=True,
        pasdw_store_funcs=False,
        use_paw=use_paw,
        fast=USE_FAST_CIDER,
        _force_nonlocal=force_nl,
    )


def _get_features(calc, all_settings, p_i=None, **kwargs):
    kwargs = {k: v for k, v in kwargs.items()}
    feats = []
    if p_i is not None:
        dfeats = []
    assert len(all_settings) > 0
    for settings in all_settings:
        kwargs["settings"] = settings
        if p_i is None:
            feat, wt = get_features(calc, **kwargs)
            feats.append(feat)
        else:
            feat, dfeat, wt = get_features(calc, p_i=p_i, **kwargs)
            feats.append(feat)
            dfeats.append(dfeat)
    feat = np.concatenate(feats, axis=1)
    if p_i is None:
        return feat, wt
    else:
        dfeat = np.concatenate(dfeats, axis=1)
        return feat, dfeat, wt


def get_static_xc_difference(calc, xc):
    if isinstance(xc, (str, dict)):
        xc = XC(xc)
    xc.set_grid_descriptor(calc.density.finegd)
    xc.initialize(calc.density, calc.hamiltonian, calc.wfs)
    xc.set_positions(calc.spos_ac)
    return calc.hamiltonian.get_xc_difference(xc, calc.density) * Ha


# TODO add unit tests for band, kpt, and domain parallelization


def _perturb_calc_density_(calc, p, delta):
    calc.wfs.occupations.perturb_occupation_(p, delta, div_by_wt=True)
    calc.wfs.calculate_occupation_numbers()
    calc.wfs.calculate_atomic_density_matrices(calc.density.D_asp)
    calc.density.calculate_pseudo_density(calc.wfs)
    calc.density.interpolate_pseudo_density()


def _rotate_orbs_(calc, s, k, n1, n2, delta):
    nspin = calc.density.nt_sg.shape[0]
    wfs = calc.wfs
    rank, q = wfs.kd.who_has(k)
    if wfs.kd.comm.rank == rank:
        u = q * nspin + s
        kpt = wfs.kpt_u[u]
        assert kpt.s == s
        assert kpt.k == k
    else:
        return None, None, None
    # TODO not sure if indexing is correct for band parallelization
    psi1 = kpt.psit_nG[n1].copy()
    psi2 = kpt.psit_nG[n2].copy()
    norm = np.sqrt(1 + delta * delta)
    kpt.psit_nG[n1] += delta * psi2
    kpt.psit_nG[n2] -= delta * psi1
    kpt.psit_nG[n1] /= norm
    kpt.psit_nG[n2] /= norm
    calc.wfs.calculate_atomic_density_matrices(calc.density.D_asp)
    calc.density.calculate_pseudo_density(calc.wfs)
    calc.density.interpolate_pseudo_density()
    return psi1, psi2, 2 * kpt.weight


def _reset_orbs_(calc, s, k, n1, n2, psi1, psi2):
    nspin = calc.density.nt_sg.shape[0]
    wfs = calc.wfs
    rank, q = wfs.kd.who_has(k)
    if wfs.kd.comm.rank == rank:
        u = q * nspin + s
        kpt = wfs.kpt_u[u]
        assert kpt.s == s
        assert kpt.k == k
    else:
        return
    kpt.psit_nG[n1] = psi1
    kpt.psit_nG[n2] = psi2
    calc.wfs.calculate_atomic_density_matrices(calc.density.D_asp)
    calc.density.calculate_pseudo_density(calc.wfs)
    calc.density.interpolate_pseudo_density()


def run_fd_deriv_test(xc, use_pp=False, spinpol=False):
    k = 3
    si = bulk("Si")
    si.calc = GPAW(
        mode=PW(250),
        mixer=Mixer(0.7, 5, 50.0),
        xc=xc,
        kpts=(k, k, k),
        convergence={"energy": 1e-8},
        parallel={"domain": min(2, world.size), "augment_grids": True},
        occupations={"name": "fermi-dirac", "width": 0.0},
        spinpol=spinpol,
        setups="sg15" if use_pp else "paw",
        txt="si.txt",
    )
    si.set_cell(
        np.dot(si.cell, [[1.02, 0, 0.03], [0, 0.99, -0.02], [0.2, -0.01, 1.03]]),
        scale_atoms=True,
    )

    etot = si.get_potential_energy()
    gap, vbm, cbm, dev, dec, pvbm, pcbm = get_homo_lumo_fd_(si.calc, delta=1e-6)
    parprint(dev - vbm, dec - cbm, dev, dec)
    assert_almost_equal(dev, vbm, 5)
    assert_almost_equal(dec, cbm, 5)
    return etot


def run_vxc_test(xc0, xc1, spinpol=False, use_pp=False, safe=True):
    if use_pp is False:
        if isinstance(xc0, str):
            xc0 = DiffGGA(XC(xc0).kernel)
        if isinstance(xc1, str):
            xc1 = DiffGGA(XC(xc1).kernel)

    k = 3
    si = bulk("Si")
    si.calc = GPAW(
        mode=PW(250),
        mixer=Mixer(0.7, 5, 50.0),
        xc=xc0,
        kpts=(k, k, k),
        convergence={"energy": 1e-8, "density": 1e-10},
        parallel={"domain": min(2, world.size), "augment_grids": True},
        occupations={"name": "fermi-dirac", "width": 0.0},
        spinpol=spinpol,
        setups="sg15" if use_pp else "paw",
        txt="si.txt",
    )
    # si.set_cell(
    #     np.dot(si.cell, [[1.02, 0, 0.03], [0, 0.99, -0.02], [0.2, -0.01, 1.03]]),
    #     scale_atoms=True,
    # )
    delta = 1e-5
    si.get_potential_energy()
    gap, p_vbm, p_cbm = bandgap(si.calc)
    run_constant_occ_calculation_(si.calc)

    # CIDER changes the xc_corrections and uses different grids,
    # so we need to check this.
    psi1, psi2, wt = _rotate_orbs_(
        si.calc, p_vbm[0], p_vbm[1], p_vbm[2] - 1, p_vbm[2] + 1, 0.5
    )
    ediff00 = si.calc.get_xc_difference(xc0) / Ha
    psi1, psi2 = _rotate_orbs_(
        si.calc, p_vbm[0], p_vbm[1], p_vbm[2] - 1, p_vbm[2] + 1, -delta
    )[:2]
    ediff10 = get_static_xc_difference(si.calc, xc0) / Ha
    _reset_orbs_(si.calc, p_vbm[0], p_vbm[1], p_vbm[2] - 1, p_vbm[2] + 1, psi1, psi2)
    ediff01 = si.calc.get_xc_difference(xc1) / Ha
    eig_vbm, ei_vbm, en_vbm = nscfeig(
        si.calc,
        xc1,
        n1=p_vbm[2] - 1,
        n2=p_vbm[2] + 2,
        kpt_indices=[p_vbm[1]],
        get_ham=True,
    )
    eigdiff_vbm = (en_vbm - ei_vbm)[p_vbm[0], 0, 0, 2] / Ha
    psi1, psi2 = _rotate_orbs_(
        si.calc, p_vbm[0], p_vbm[1], p_vbm[2] - 1, p_vbm[2] + 1, -delta
    )[:2]
    ediff11 = get_static_xc_difference(si.calc, xc1) / Ha
    _reset_orbs_(si.calc, p_vbm[0], p_vbm[1], p_vbm[2] - 1, p_vbm[2] + 1, psi1, psi2)
    ediff0 = ediff01 - ediff00
    ediff1 = ediff11 - ediff10

    fd_eigdiff_vbm = (ediff0 - ediff1) / delta
    parprint(eigdiff_vbm.real, fd_eigdiff_vbm / wt)
    if use_pp:
        # TODO for some reason precision is lower on pseudopp vxc,
        # is this just due to pseudos or is it some kind of bug?
        prec = 6
    else:
        prec = 7
    assert_almost_equal(eigdiff_vbm.real, fd_eigdiff_vbm / wt, prec)


def run_nscf_eigval_test(xc0, xc1, spinpol=False, use_pp=False, safe=True):
    if use_pp is False:
        if isinstance(xc0, str):
            xc0 = DiffGGA(XC(xc0).kernel)
        if isinstance(xc1, str):
            xc1 = DiffGGA(XC(xc1).kernel)

    k = 3
    si = bulk("Si")
    si.calc = GPAW(
        mode=PW(250),
        mixer=Mixer(0.7, 5, 50.0),
        xc=xc0,
        kpts=(k, k, k),
        convergence={"energy": 1e-8, "density": 1e-10},
        parallel={"domain": min(2, world.size), "augment_grids": True},
        occupations={"name": "fermi-dirac", "width": 0.0},
        spinpol=spinpol,
        setups="sg15" if use_pp else "paw",
        txt="si.txt",
    )
    delta = 1e-4
    # si.set_cell(
    #    np.dot(si.cell, [[1.02, 0, 0.03], [0, 0.99, -0.02], [0.2, -0.01, 1.03]]),
    #    scale_atoms=True,
    # )
    si.get_potential_energy()
    gap, p_vbm, p_cbm = bandgap(si.calc)
    run_constant_occ_calculation_(si.calc)

    if safe:
        # CIDER changes the xc_corrections and uses different grids,
        # so we need to check this.
        ediff00 = si.calc.get_xc_difference(xc0) / Ha
        _perturb_calc_density_(si.calc, p_vbm, -delta)
        ediff10 = get_static_xc_difference(si.calc, xc0) / Ha
        _perturb_calc_density_(si.calc, p_vbm, delta)
        _perturb_calc_density_(si.calc, p_cbm, delta)
        ediff20 = get_static_xc_difference(si.calc, xc0) / Ha
        _perturb_calc_density_(si.calc, p_cbm, -delta)
        ediff01 = si.calc.get_xc_difference(xc1) / Ha
        eig_vbm, ei_vbm, en_vbm = nscfeig(
            si.calc, xc1, n1=p_vbm[2], n2=p_vbm[2] + 1, kpt_indices=[p_vbm[1]]
        )
        eig_cbm, ei_cbm, en_cbm = nscfeig(
            si.calc, xc1, n1=p_cbm[2], n2=p_cbm[2] + 1, kpt_indices=[p_cbm[1]]
        )
        eigdiff_vbm = (en_vbm - ei_vbm)[p_vbm[0], 0, 0] / Ha
        eigdiff_cbm = (en_cbm - ei_cbm)[p_cbm[0], 0, 0] / Ha
        _perturb_calc_density_(si.calc, p_vbm, -delta)
        ediff11 = get_static_xc_difference(si.calc, xc1) / Ha
        _perturb_calc_density_(si.calc, p_vbm, delta)
        _perturb_calc_density_(si.calc, p_cbm, delta)
        ediff21 = get_static_xc_difference(si.calc, xc1) / Ha
        _perturb_calc_density_(si.calc, p_cbm, -delta)
        ediff0 = ediff01 - ediff00
        ediff1 = ediff11 - ediff10
        ediff2 = ediff21 - ediff20
    else:
        ediff0 = si.calc.get_xc_difference(xc1) / Ha
        eig_vbm, ei_vbm, en_vbm = nscfeig(
            si.calc, xc1, n1=p_vbm[2], n2=p_vbm[2] + 1, kpt_indices=[p_vbm[1]]
        )
        eig_cbm, ei_cbm, en_cbm = nscfeig(
            si.calc, xc1, n1=p_cbm[2], n2=p_cbm[2] + 1, kpt_indices=[p_cbm[1]]
        )
        eigdiff_vbm = (en_vbm - ei_vbm)[p_vbm[0], 0, 0] / Ha
        eigdiff_cbm = (en_cbm - ei_cbm)[p_cbm[0], 0, 0] / Ha
        _perturb_calc_density_(si.calc, p_vbm, -delta)
        ediff1 = (
            get_static_xc_difference(si.calc, xc1)
            - get_static_xc_difference(si.calc, xc0)
        ) / Ha
        _perturb_calc_density_(si.calc, p_vbm, delta)
        _perturb_calc_density_(si.calc, p_cbm, delta)
        ediff2 = (
            get_static_xc_difference(si.calc, xc1)
            - get_static_xc_difference(si.calc, xc0)
        ) / Ha
    fd_eigdiff_vbm = (ediff0 - ediff1) / delta
    fd_eigdiff_cbm = (ediff2 - ediff0) / delta
    parprint(eigdiff_cbm - eigdiff_vbm, fd_eigdiff_cbm - fd_eigdiff_vbm)
    parprint(eigdiff_cbm, eigdiff_vbm, fd_eigdiff_cbm, fd_eigdiff_vbm)
    assert_almost_equal(eigdiff_vbm, fd_eigdiff_vbm, 5)
    assert_almost_equal(eigdiff_cbm, fd_eigdiff_cbm, 5)


def run_drho_test(spinpol=False, use_pp=False):
    k = 3
    si = bulk("Si")
    si.calc = GPAW(
        mode=PW(250),
        mixer=Mixer(0.7, 5, 50.0),
        xc="PBE",
        kpts=(k, k, k),
        convergence={"energy": 1e-8, "density": 1e-10},
        parallel={"domain": min(2, world.size), "augment_grids": True},
        occupations={"name": "fermi-dirac", "width": 0.0},
        spinpol=spinpol,
        setups="sg15" if use_pp else "paw",
        txt="si.txt",
    )
    delta = 1e-4
    si.get_potential_energy()
    gap, p_vbm, p_cbm = bandgap(si.calc)
    run_constant_occ_calculation_(si.calc)
    p_i = [p_vbm, p_cbm]
    drhodf_ixR = get_drho_df(si.calc, p_i)
    nt0_sG = si.calc.density.nt_sG.copy()
    nt0_sg = si.calc.density.nt_sg.copy()

    _perturb_calc_density_(si.calc, p_vbm, -delta)

    nt1_sG = si.calc.density.nt_sG.copy()
    nt1_sg = si.calc.density.nt_sg.copy()
    dnt_sG = (nt1_sG - nt0_sG) / delta * -1
    dnt_sg = (nt1_sg - nt0_sg) / delta * -1
    diff = drhodf_ixR[0, 0] - dnt_sG
    idiff = si.calc.hamiltonian.xc.gd.integrate(diff * nt0_sG)
    assert_almost_equal(idiff, 0)

    drhodf_ixg = interpolate_drhodf(
        si.calc.hamiltonian.xc.gd,
        si.calc.density.distribute_and_interpolate,
        drhodf_ixR,
    )
    ind = 0
    diff = drhodf_ixg[ind, 0] - dnt_sg
    idiff = si.calc.hamiltonian.xc.gd.integrate(diff * nt0_sg)
    assert_almost_equal(idiff, 0)


def run_nl_feature_test(xc, use_pp=False, spinpol=False, baseline="PBE"):
    k = 3
    si = bulk("Si")
    if use_pp is False:
        baseline = DiffGGA(XC(baseline).kernel)

    si.calc = GPAW(
        mode=PW(250),
        mixer=Mixer(0.7, 5, 50.0),
        xc=xc,
        kpts=(k, k, k),
        convergence={"energy": 1e-8, "density": 1e-10},
        parallel={"domain": min(2, world.size), "augment_grids": True},
        occupations={"name": "fermi-dirac", "width": 0.0},
        spinpol=spinpol,
        setups="sg15" if use_pp else "paw",
        txt="si.txt",
    )

    mlfunc = xc.cider_kernel.mlfunc
    if USE_FAST_CIDER:
        all_settings = [
            mlfunc.settings.sl_settings,
            mlfunc.settings.nldf_settings,
        ]
    else:
        all_settings = [mlfunc.settings.nldf_settings]
    kwargs = dict(
        settings=mlfunc.settings.nldf_settings,
        use_paw=not use_pp,
        screen_dens=False,
        qmax=xc.encut,
        lambd=xc.lambd,
    )

    si.get_potential_energy()

    calc = si.calc
    gap, p_vbm, p_cbm = bandgap(si.calc)
    delta = 1e-4
    run_constant_occ_calculation_(si.calc)

    ediff = si.calc.get_xc_difference(baseline) / Ha
    xmix = xc.cider_kernel.xmix
    feat0, wt0 = _get_features(si.calc, all_settings, **kwargs)
    feat, dfeat_j, wt = _get_features(
        si.calc, all_settings, p_i=[p_vbm, p_cbm], **kwargs
    )
    if use_pp:
        assert feat.shape[-1] == np.prod(xc.gd.get_size_of_global_array())
    assert feat.shape[-1] == dfeat_j.shape[-1]
    assert feat.shape[-1] == wt.size
    assert_almost_equal(feat, feat0)
    assert_almost_equal(wt, wt0)
    desc = feat.copy()
    etst = 0
    X0TN = mlfunc.settings.normalizers.get_normalized_feature_vector(desc)
    eps, deps = mlfunc(X0TN)
    etst = np.sum(eps * wt)
    parprint(xmix * etst, -ediff)
    assert_almost_equal(xmix * etst, -ediff, 8)

    si.calc.hamiltonian.xc.setups = None
    si.calc.hamiltonian.xc.initialize_more_things()
    eig_vbm, ei_vbm, en_vbm = nscfeig(
        si.calc, baseline, n1=p_vbm[2], n2=p_vbm[2] + 1, kpt_indices=[p_vbm[1]]
    )
    eig_cbm, ei_cbm, en_cbm = nscfeig(
        si.calc, baseline, n1=p_cbm[2], n2=p_cbm[2] + 1, kpt_indices=[p_cbm[1]]
    )
    eigdiff_vbm = (en_vbm - ei_vbm)[p_vbm[0], 0, 0] / Ha
    eigdiff_cbm = (en_cbm - ei_cbm)[p_cbm[0], 0, 0] / Ha
    eigdiff_gap = eigdiff_vbm - eigdiff_cbm

    _perturb_calc_density_(calc, p_vbm, -delta)
    feat1, wt1 = _get_features(si.calc, all_settings, **kwargs)
    deriv1 = (feat0 - feat1) / delta
    _perturb_calc_density_(calc, p_vbm, delta)
    _perturb_calc_density_(calc, p_cbm, delta)
    feat2, wt2 = _get_features(si.calc, all_settings, **kwargs)
    _perturb_calc_density_(calc, p_cbm, -delta)
    deriv2 = (feat2 - feat0) / delta
    derivs = [deriv1, deriv2]
    # TODO features don't match but give same energy differences.
    # Not sure why this is. Definitely related to symmetry, but
    # sum(dfeat * rho) should still be consistent with FD and is not.
    # for i in range(feat0[0].shape[0]):
    #    for j, p in enumerate([p_vbm, p_cbm]):
    #        s = p[0]
    #        rho = feat0[s, 0]
    #        assert_almost_equal(
    #            np.sum(dfeat_j[j, i] * rho), np.sum(derivs[j][s, i] * rho), 5
    #        )

    etst = 0
    etst2 = 0
    p_j = [p_vbm, p_cbm]
    pwt = [-1, 1]
    X0TN = mlfunc.settings.normalizers.get_normalized_feature_vector(desc)
    eps, deps = mlfunc(X0TN)
    deps = mlfunc.settings.normalizers.get_derivative_wrt_unnormed_features(desc, deps)
    for j in range(2):
        p = p_j[j]
        s = p[0]
        dfeat_tmp = dfeat_j[j]
        dfeat_tmp2 = derivs[j][s]
        deps_tmp = np.einsum("xg,xg->g", deps[s], dfeat_tmp)
        etst += pwt[j] * np.sum(deps_tmp * wt)
        deps_tmp = np.einsum("xg,xg->g", deps[s], dfeat_tmp2)
        etst2 += pwt[j] * np.sum(deps_tmp * wt)
    # TODO why is this not consistent depending on symmetry?
    parprint(xmix * etst, xmix * etst2, eigdiff_gap)
    assert_almost_equal(xmix * etst, eigdiff_gap, 5)
    assert_almost_equal(xmix * etst2, xmix * etst, 5)
    assert_almost_equal(xmix * etst2, eigdiff_gap, 5)


def run_sl_feature_test(use_pp=False, spinpol=False):
    # TODO precision is poor for Si. Why is this?
    k = 3
    si = bulk("Ge")
    from gpaw.xc.libxc import LibXC

    if use_pp:
        xc0name = "PBE"
        xc1name = "PBEsol"
        xc0 = xc0name
        xc1 = xc1name
    else:
        xc0name = "MGGA_X_SCAN+MGGA_C_SCAN"
        xc1name = "GGA_X_PBE+GGA_C_PBE"
        xc0 = DiffMGGA(LibXC(xc0name))
        xc1 = DiffGGA(LibXC(xc1name))

    si.calc = GPAW(
        mode=PW(250),
        mixer=Mixer(0.7, 5, 50.0),
        xc=xc0,
        kpts=(k, k, k),
        convergence={"energy": 1e-8, "density": 1e-10},
        parallel={"domain": min(2, world.size), "augment_grids": True},
        occupations={"name": "fermi-dirac", "width": 0.0},
        spinpol=spinpol,
        setups="sg15" if use_pp else "paw",
        txt="si.txt",
    )

    kwargs = dict(
        settings="l",
        use_paw=not use_pp,
        screen_dens=False,
    )

    si.get_potential_energy()
    gap, p_vbm, p_cbm = bandgap(si.calc)
    # run_constant_occ_calculation_(si.calc)
    # etot = si.get_potential_energy() / Ha

    ediff = si.calc.get_xc_difference(xc1) / Ha
    feat0, wt0 = get_features(si.calc, **kwargs)
    feat, dfeat_j, wt = get_features(si.calc, p_i=[p_vbm, p_cbm], **kwargs)

    from pyscf.dft.numint import NumInt

    assert_almost_equal(feat, feat0)
    assert_almost_equal(wt, wt0)
    rho = feat.copy()
    if not spinpol:
        rho = rho.squeeze(0)

    ni = NumInt()
    if use_pp:
        exc0, vxc0 = ni.eval_xc_eff(xc0name, rho, xctype="GGA")[:2]
        exc1, vxc1 = ni.eval_xc_eff(xc1name, rho[..., :4, :], xctype="GGA")[:2]
    else:
        exc0, vxc0 = ni.eval_xc_eff(xc0name, rho, xctype="MGGA")[:2]
        exc1, vxc1 = ni.eval_xc_eff(xc1name, rho[..., :4, :], xctype="GGA")[:2]
    exc_tot_0 = np.sum(exc0 * feat[:, 0].sum(0) * wt)
    exc_tot_1 = np.sum(exc1 * feat[:, 0].sum(0) * wt)
    parprint(exc_tot_1 - exc_tot_0, ediff)
    assert_almost_equal(exc_tot_1 - exc_tot_0, ediff, 7)

    eig_vbm, ei_vbm, en_vbm = nscfeig(
        si.calc, xc1, n1=p_vbm[2], n2=p_vbm[2] + 1, kpt_indices=[p_vbm[1]]
    )
    eig_cbm, ei_cbm, en_cbm = nscfeig(
        si.calc, xc1, n1=p_cbm[2], n2=p_cbm[2] + 1, kpt_indices=[p_cbm[1]]
    )
    eigdiff_vbm = (en_vbm - ei_vbm)[p_vbm[0], 0, 0] / Ha
    eigdiff_cbm = (en_cbm - ei_cbm)[p_cbm[0], 0, 0] / Ha

    dde_list = []
    for p, dfeat in zip([p_vbm, p_cbm], dfeat_j):
        if spinpol:
            s = p[0]
            vtmp0 = vxc0[s]
            vtmp1 = vxc1[s]
        else:
            vtmp0 = vxc0
            vtmp1 = vxc1
        de0 = np.sum(dfeat[: len(vtmp0)] * vtmp0 * wt)
        de1 = np.sum(dfeat[: len(vtmp1)] * vtmp1 * wt)
        dde = de1 - de0
        dde_list.append(dde)

    parprint(dde_list, eigdiff_vbm, eigdiff_cbm)
    assert_almost_equal(dde_list[0], eigdiff_vbm, 4)
    assert_almost_equal(dde_list[1], eigdiff_cbm, 4)
    assert_almost_equal(dde_list[1] - dde_list[0], eigdiff_cbm - eigdiff_vbm, 4)


class TestDescriptors(unittest.TestCase):
    def test_occ_deriv_rho(self):
        run_drho_test(spinpol=False, use_pp=True)
        run_drho_test(spinpol=False, use_pp=False)

    def test_eigval(self):
        for use_pp in [True, False]:
            xc = get_xc(
                "functionals/CIDER23X_NL_GGA.yaml", use_paw=not use_pp, force_nl=True
            )
            run_nscf_eigval_test(
                "PBE", xc, spinpol=False, use_pp=use_pp, safe=not use_pp
            )
            run_nscf_eigval_test(
                "PBE", xc, spinpol=True, use_pp=use_pp, safe=not use_pp
            )
            xc = get_xc(
                "functionals/CIDER23X_NL_MGGA_DTR.yaml",
                use_paw=not use_pp,
                force_nl=True,
            )
            run_nscf_eigval_test(
                "PBE", xc, spinpol=False, use_pp=use_pp, safe=not use_pp
            )
            run_nscf_eigval_test(
                "PBE", xc, spinpol=True, use_pp=use_pp, safe=not use_pp
            )

    def test_vxc(self):
        for use_pp in [True, False]:
            xc = get_xc("functionals/CIDER23X_NL_GGA.yaml", use_paw=not use_pp)
            run_vxc_test("PBE", xc, spinpol=False, use_pp=use_pp, safe=not use_pp)
            run_vxc_test("PBE", xc, spinpol=True, use_pp=use_pp, safe=not use_pp)
            xc = get_xc(
                "functionals/CIDER23X_NL_MGGA_DTR.yaml",
                use_paw=not use_pp,
            )
            run_vxc_test("PBE", xc, spinpol=False, use_pp=use_pp, safe=not use_pp)
            run_vxc_test("PBE", xc, spinpol=True, use_pp=use_pp, safe=not use_pp)

    def test_vxc_quick(self):
        xc = get_xc(
            "functionals/CIDER23X_NL_MGGA_DTR.yaml",
            use_paw=True,
            force_nl=True,
        )
        run_vxc_test("PBE", xc, spinpol=False, use_pp=False, safe=True)
        run_vxc_test("PBE", xc, spinpol=True, use_pp=False, safe=True)

    def test_sl_features(self):
        # run_sl_feature_test(spinpol=False, use_pp=True)
        run_sl_feature_test(spinpol=False)
        run_sl_feature_test(spinpol=True)

    def test_nl_features(self):
        for use_pp in [True, False]:
            parprint("use_pp?", use_pp, "  GGA, spinpol=False")
            xc = get_xc("functionals/CIDER23X_NL_GGA.yaml", use_paw=not use_pp)
            baseline = "0.75_GGA_X_PBE+1.00_GGA_C_PBE"
            run_nl_feature_test(xc, use_pp=use_pp, spinpol=False, baseline=baseline)

            parprint("use_pp?", use_pp, "  GGA, spinpol=True")
            xc = get_xc("functionals/CIDER23X_NL_GGA.yaml", use_paw=not use_pp)
            baseline = "0.75_GGA_X_PBE+1.00_GGA_C_PBE"
            run_nl_feature_test(xc, use_pp=use_pp, spinpol=True, baseline=baseline)

            parprint("use_pp?", use_pp, "  MGGA, spinpol=False")
            baseline = "0.75_GGA_X_PBE+1.00_GGA_C_PBE"
            xc = get_xc("functionals/CIDER23X_NL_MGGA_DTR.yaml", use_paw=not use_pp)
            run_nl_feature_test(xc, spinpol=False, use_pp=use_pp, baseline=baseline)

            parprint("use_pp?", use_pp, "  MGGA, spinpol=True")
            baseline = "0.75_GGA_X_PBE+1.00_GGA_C_PBE"
            xc = get_xc("functionals/CIDER23X_NL_MGGA_DTR.yaml", use_paw=not use_pp)
            run_nl_feature_test(xc, spinpol=True, use_pp=use_pp, baseline=baseline)

    def test_eigval_vs_fd(self):
        run_fd_deriv_test("PBE")

        run_fd_deriv_test("MGGA_X_R2SCAN+MGGA_C_R2SCAN")

        xc = get_xc("functionals/CIDER23X_NL_GGA.yaml")
        run_fd_deriv_test(xc)

        xc = get_xc("functionals/CIDER23X_NL_MGGA_DTR.yaml")
        run_fd_deriv_test(xc)

        xc = get_xc("functionals/CIDER23X_NL_MGGA_DTR.yaml")
        run_fd_deriv_test(xc, spinpol=True)

        xc = get_xc("functionals/CIDER23X_SL_GGA.yaml")
        run_fd_deriv_test(xc)

        xc = get_xc("functionals/CIDER23X_SL_MGGA.yaml")
        run_fd_deriv_test(xc)

        xc = get_xc("functionals/CIDER23X_NL_GGA.yaml", use_paw=False)
        run_fd_deriv_test(xc, use_pp=True)

        xc = get_xc("functionals/CIDER23X_NL_MGGA.yaml", use_paw=False)
        run_fd_deriv_test(xc, use_pp=True)


if __name__ == "__main__":
    unittest.main()
