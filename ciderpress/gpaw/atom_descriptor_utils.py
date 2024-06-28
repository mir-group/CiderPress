import numpy as np
from gpaw.sphere.lebedev import R_nv, Y_nL, weight_n
from gpaw.xc.pawcorrection import rnablaY_nLv
from numpy import pi, sqrt

from ciderpress.dft.settings import dalpha, ds2, get_alpha, get_s2
from ciderpress.gpaw.atom_utils import get_atomic_convolution
from ciderpress.gpaw.fit_paw_gauss_pot import get_dv
from ciderpress.gpaw.interp_paw import (
    CiderRadialExpansion,
    calculate_cider_paw_correction,
    vec_radial_gga_vars,
)


def get_features_with_sl_noderiv(feat_sig, n_sg, sigma_xg, tau_sg=None):
    nspin = len(feat_sig)
    xctype = "GGA" if tau_sg is None else "MGGA"
    nfeat = 6 if xctype == "MGGA" else 5
    nnlfeat = feat_sig[0].shape[0]
    _feat = np.zeros((nspin, nfeat) + feat_sig[0].shape[1:])
    for s in range(nspin):
        _feat[s, nfeat - nnlfeat :] = feat_sig[s]
    feat_sig = _feat
    for s in range(nspin):
        feat_sig[s, 0] = n_sg[s]
        feat_sig[s, 1] = sigma_xg[2 * s]
        s2 = get_s2(feat_sig[s, 0], feat_sig[s, 1])
        if xctype == "MGGA":
            feat_sig[s, 2] = tau_sg[s]
            alpha = get_alpha(feat_sig[s, 0], feat_sig[s, 1], feat_sig[s, 2])
            feat_sig[s, 2] = alpha
        feat_sig[s, 1] = s2
    return feat_sig


def get_features_with_sl_part(
    p_j,
    feat_sig,
    dfeatdf_jig,
    n_sg,
    sigma_xg,
    dndf_jg,
    dsigmadf_jg,
    tau_sg=None,
    dtaudf_jg=None,
):
    nspin = len(feat_sig)
    nj = len(dfeatdf_jig)
    xctype = "GGA" if tau_sg is None else "MGGA"
    if xctype == "MGGA":
        assert dtaudf_jg is not None
    nfeat = 6 if xctype == "MGGA" else 5
    nnlfeat = feat_sig[0].shape[0]
    _feat = np.zeros((nspin, nfeat) + feat_sig[0].shape[1:])
    _dfeat = np.zeros((nj, nfeat) + feat_sig[0].shape[1:])
    for s in range(nspin):
        _feat[s, nfeat - nnlfeat :] = feat_sig[s]
    for j in range(nj):
        _dfeat[j, nfeat - nnlfeat :] = dfeatdf_jig[j]
    feat_sig = _feat
    dfeatdf_jig = _dfeat
    for s in range(nspin):
        feat_sig[s, 0] = n_sg[s]
        feat_sig[s, 1] = sigma_xg[2 * s]
        s2 = get_s2(feat_sig[s, 0], feat_sig[s, 1])
        if xctype == "MGGA":
            feat_sig[s, 2] = tau_sg[s]
            alpha = get_alpha(feat_sig[s, 0], feat_sig[s, 1], feat_sig[s, 2])
            feat_sig[s, 2] = alpha
        feat_sig[s, 1] = s2
    for j in range(nj):
        s = p_j[j][0]
        dfeatdf_jig[j, 0] = dndf_jg[j]
        dfeatdf_jig[j, 1] = dsigmadf_jg[j]
        if xctype == "MGGA":
            dfeatdf_jig[j, 2] = dtaudf_jg[j]
            dalpha_rho, dalpha_sigma, dalpha_tau = dalpha(
                n_sg[s], sigma_xg[2 * s], tau_sg[s]
            )
            dfeatdf_jig[j, 2] = (
                dndf_jg[j] * dalpha_rho
                + dsigmadf_jg[j] * dalpha_sigma
                + dtaudf_jg[j] * dalpha_tau
            )
        dprho, dpsigma = ds2(n_sg[s], sigma_xg[2 * s])
        dfeatdf_jig[j, 1] = dndf_jg[j] * dprho + dsigmadf_jg[j] * dpsigma
    return feat_sig, dfeatdf_jig


def calculate_cider_paw_correction_deriv(
    expansion, setup, D_sp, DD_op=None, addcoredensity=True
):
    xcc = setup.xc_correction
    rgd = xcc.rgd
    nspins = len(D_sp)

    setup.cider_contribs.set_D_sp(D_sp, setup.xc_correction)
    setup.cider_contribs._DD_op = DD_op

    if addcoredensity:
        nc0_sg = rgd.empty(nspins)
        nct0_sg = rgd.empty(nspins)
        dc0_sg = rgd.empty(nspins)
        dct0_sg = rgd.empty(nspins)
        nc0_sg[:] = sqrt(4 * pi) / nspins * xcc.nc_g
        nct0_sg[:] = sqrt(4 * pi) / nspins * xcc.nct_g
        dc0_sg[:] = sqrt(4 * pi) / nspins * xcc.dc_g
        dct0_sg[:] = sqrt(4 * pi) / nspins * xcc.dct_g
        if xcc.nc_corehole_g is not None and nspins == 2:
            nc0_sg[0] -= 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g
            nc0_sg[1] += 0.5 * sqrt(4 * pi) * xcc.nc_corehole_g
            dc0_sg[0] -= 0.5 * sqrt(4 * pi) * xcc.dc_corehole_g
            dc0_sg[1] += 0.5 * sqrt(4 * pi) * xcc.dc_corehole_g
    else:
        nc0_sg = 0
        nct0_sg = 0
        dc0_sg = 0
        dct0_sg = 0

    D_sLq = np.inner(D_sp, xcc.B_pqL.T)
    DD_oLq = np.inner(DD_op, xcc.B_pqL.T)

    res = expansion(rgd, D_sLq, DD_oLq, xcc.n_qg, xcc.d_qg, nc0_sg, dc0_sg, ae=True)
    rest = expansion(
        rgd, D_sLq, DD_oLq, xcc.nt_qg, xcc.dt_qg, nct0_sg, dct0_sg, ae=False
    )

    return res, rest


def calculate_paw_cider_features_p1(self, setups, D_asp, DD_aop, p_o):
    self.yref_aobLg = {}
    if len(D_asp.keys()) == 0:
        return {}, {}
    a0 = list(D_asp.keys())[0]
    nspin = D_asp[a0].shape[0]
    norb = DD_aop[a0].shape[0]
    assert (nspin == 1) or self.is_cider_functional
    y_aobLg = {}
    c_oabi = {o: {} for o in range(nspin + norb)}

    for a, D_sp in D_asp.items():
        setup = setups[a]
        psetup = setup.pasdw_setup
        Nalpha_sm = psetup.phi_jabg.shape[1]
        xcc = setup.nlxc_correction
        rgd = setup.nlxc_correction.big_rgd
        slgd = setup.xc_correction.rgd
        RCUT = np.max(setup.rcut_j)

        rcalc = CiderRadialThetaDerivCalculator(setup.cider_contribs)
        expansion = CiderRadialDerivExpansion(rcalc, p_o)
        dx_obgL, dxt_obgL = calculate_cider_paw_correction_deriv(
            expansion,
            setup,
            D_sp,
            DD_op=DD_aop[a],
        )
        dx_obgL -= dxt_obgL

        rcut = RCUT
        rmax = slgd.r_g[-1]
        fcut = 0.5 + 0.5 * np.cos(np.pi * (slgd.r_g - rcut) / (rmax - rcut))
        fcut[slgd.r_g < rcut] = 1.0
        fcut[slgd.r_g > rmax] = 0.0

        # This all should all work because none of these routines
        # depend on nspin being 1 or 2, so same functions can be
        # used as for just getting the theta components.
        dxt_obgL *= fcut[:, None]
        dx_obLk = setup.etb_projector.r2k(dx_obgL.transpose(0, 1, 3, 2))
        dxt_obLk = setup.etb_projector.r2k(dxt_obgL.transpose(0, 1, 3, 2))

        y_obLk = get_atomic_convolution(setup.cider_contribs, dx_obLk, xcc)
        yref_obLk = get_atomic_convolution(setup.cider_contribs, dxt_obLk, xcc)
        yref_obLg = 2 / np.pi * setup.etb_projector.k2r(yref_obLk)

        c_oib, df_oLpb = psetup.get_c_and_df(y_obLk[:, :Nalpha_sm], realspace=False)
        yt_obLg = psetup.get_f_realspace_contribs(c_oib, sl=True)
        df_oLpb = np.append(
            df_oLpb,
            psetup.get_df_only(y_obLk[:, Nalpha_sm:], rgd, sl=False, realspace=False),
            axis=-1,
        )
        df_obLg = psetup.get_df_realspace_contribs(df_oLpb, sl=True)
        y_aobLg[a] = df_obLg.copy()
        self.yref_aobLg[a] = yref_obLg
        self.yref_aobLg[a][:, :Nalpha_sm] += yt_obLg
        for o in range(nspin + norb):
            c_oabi[o][a] = c_oib[o].T

    return c_oabi, y_aobLg


def calculate_paw_cider_features_p2_noderiv(self, setups, D_asp, D_sabi, df_asbLg):
    if len(D_asp.keys()) == 0:
        return {}, {}
    a0 = list(D_asp.keys())[0]
    nspin = D_asp[a0].shape[0]
    assert (nspin == 1) or self.is_cider_functional
    aefeat_asig = {}
    psfeat_asig = {}

    for a, D_sp in D_asp.items():
        setup = setups[a]
        psetup = setup.paonly_setup
        ni = psetup.ni
        slgd = setup.xc_correction.rgd
        Nalpha_sm = D_sabi[0][a].shape[0]

        dv_g = get_dv(slgd)
        yref_sbLg = self.yref_asbLg[a]
        f_sbLg = np.zeros_like(df_asbLg[a])
        for i in range(ni):
            L = psetup.lmlist_i[i]
            n = psetup.nlist_i[i]
            Dref_sb = np.einsum(
                "sbg,g->sb",
                yref_sbLg[:, :Nalpha_sm, L],
                psetup.pfuncs_ng[n] * dv_g,
            )
            j = psetup.jlist_i[i]
            for s in range(nspin):
                f_sbLg[s, :Nalpha_sm, L, :] += (
                    D_sabi[s][a][:, i, None] - Dref_sb[s, :, None]
                ) * psetup.ffuncs_jg[j]
        f_sbLg[:, :] += yref_sbLg
        rcalc = CiderRadialFeatCalculator(setup.cider_contribs)
        expansion = CiderRadialExpansion(rcalc, f_sbLg, df_asbLg[a], f_in_sph_harm=True)
        aefeat_asig[a], psfeat_asig[a] = calculate_cider_paw_correction(
            expansion, setup, D_sp
        )
    return aefeat_asig, psfeat_asig


def calculate_paw_cider_features_p2(self, setups, D_asp, DD_aop, p_o, D_oabi, df_aobLg):
    if len(D_asp.keys()) == 0:
        return {}, {}, {}, {}
    a0 = list(D_asp.keys())[0]
    nspin = D_asp[a0].shape[0]
    norb = DD_aop[a0].shape[0]
    assert (nspin == 1) or self.is_cider_functional
    ae_feat_asig = {}
    ps_feat_asig = {}
    ae_dfeat_aoig = {}
    ps_dfeat_aoig = {}

    for a, D_sp in D_asp.items():
        setup = setups[a]
        psetup = setup.paonly_setup
        ni = psetup.ni
        slgd = setup.xc_correction.rgd
        Nalpha_sm = D_oabi[0][a].shape[0]

        dv_g = get_dv(slgd)
        yref_obLg = self.yref_aobLg[a]
        f_obLg = np.zeros_like(df_aobLg[a])
        for i in range(ni):
            L = psetup.lmlist_i[i]
            n = psetup.nlist_i[i]
            Dref_ob = np.einsum(
                "obg,g->ob",
                yref_obLg[:, :Nalpha_sm, L],
                psetup.pfuncs_ng[n] * dv_g,
            )
            j = psetup.jlist_i[i]
            for o in range(nspin + norb):
                f_obLg[o, :Nalpha_sm, L, :] += (
                    D_oabi[o][a][:, i, None] - Dref_ob[o, :, None]
                ) * psetup.ffuncs_jg[j]
        f_obLg[:, :] += yref_obLg
        rcalc = CiderRadialFeatDerivCalculator(setup.cider_contribs)
        expansion = CiderRadialDerivExpansion(
            rcalc, p_o, f_obLg, df_aobLg[a], f_in_sph_harm=True
        )
        (
            (ae_feat_asig[a], ae_dfeat_aoig[a]),
            (ps_feat_asig[a], ps_dfeat_aoig[a]),
        ) = calculate_cider_paw_correction_deriv(
            expansion, setup, D_sp, DD_op=DD_aop[a]
        )
    return ae_feat_asig, ps_feat_asig, ae_dfeat_aoig, ps_dfeat_aoig


def calculate_paw_sl_features(self, setups, D_asp):
    if len(D_asp.keys()) == 0:
        return {}, {}
    ae_feat_asig = {}
    ps_feat_asig = {}
    for a, D_sp in D_asp.items():
        setup = setups[a]
        rcalc = CiderRadialDensityCalculator(setup.cider_contribs)
        expansion = CiderRadialExpansion(rcalc)
        ae_feat_asig[a], ps_feat_asig[a] = calculate_cider_paw_correction(
            expansion, setup, D_sp
        )
    return ae_feat_asig, ps_feat_asig


def calculate_paw_sl_features_deriv(self, setups, D_asp, DD_aop, p_o):
    if len(D_asp.keys()) == 0:
        return {}, {}, {}, {}
    ae_feat_asig = {}
    ps_feat_asig = {}
    ae_dfeat_aoig = {}
    ps_dfeat_aoig = {}
    for a, D_sp in D_asp.items():
        setup = setups[a]
        rcalc = CiderRadialDensityDerivCalculator(setup.cider_contribs)
        expansion = CiderRadialDerivExpansion(rcalc, p_o)
        (
            (ae_feat_asig[a], ae_dfeat_aoig[a]),
            (ps_feat_asig[a], ps_dfeat_aoig[a]),
        ) = calculate_cider_paw_correction_deriv(
            expansion, setup, D_sp, DD_op=DD_aop[a]
        )
    return ae_feat_asig, ps_feat_asig, ae_dfeat_aoig, ps_dfeat_aoig


def _get_paw_helper1(
    self, p_o, n_sg, sigma_xg, dndf_og, dsigmadf_og, tau_sg=None, dtaudf_og=None
):
    nspin = n_sg.shape[0]
    norb = len(dndf_og)
    x_oag = np.zeros((norb + nspin, self.Nalpha, n_sg.shape[-1]))
    is_mgga = self._plan.nldf_settings.sl_level == "MGGA"
    if is_mgga:
        rho_tuple = (n_sg, sigma_xg[::2], tau_sg)
    else:
        rho_tuple = (n_sg, sigma_xg[::2])
    di_s, derivs = self._plan.get_interpolation_arguments(rho_tuple, i=-1)
    # TODO need to account for dfunc
    func_sg, dfunc = self._plan.get_function_to_convolve(rho_tuple)
    for s in range(nspin):
        p_qg, dp_qg = self._plan.get_interpolation_coefficients(di_s[s].ravel(), i=-1)
        for a in range(self.Nalpha):
            x_oag[s, a] = p_qg[a] * n_sg[s]
    for o in range(norb):
        s = p_o[o][0]
        dadf_g = derivs[0][s] * dndf_og[o] + derivs[1][s] * dsigmadf_og[o]
        if is_mgga:
            dadf_g += derivs[2][s] * dtaudf_og[o]
        p_qg, dp_qg = self._plan.get_interpolation_coefficients(di_s[s].ravel(), i=-1)
        for a in range(self.Nalpha):
            x_oag[nspin + o, a] = dp_qg[a] * dadf_g * n_sg[s]
            x_oag[nspin + o, a] += p_qg[a] * dndf_og[o]
    return x_oag


def _get_paw_helper2(
    self,
    p_o,
    n_sg,
    sigma_xg,
    dndf_og,
    dsigmadf_og,
    y_sbg,
    dydf_obg,
    F_sag,
    dFdf_oag,
    ae=True,
    tau_sg=None,
    dtaudf_og=None,
):
    # for getting potential and energy
    nspin = n_sg.shape[0]
    nfeat = self.nexp - 1
    ngrid = n_sg.shape[-1]
    norb = len(p_o)
    if ae:
        y_sbg = y_sbg.copy()
        y_sbg[:] += F_sag
        dydf_obg = dydf_obg.copy()
        dydf_obg += dFdf_oag
    else:
        y_sbg = F_sag
        dydf_obg = dFdf_oag
    Nalpha = self.Nalpha

    x_sig = np.zeros((nspin, nfeat, ngrid))
    dxdf_oig = np.zeros((norb, nfeat, ngrid))
    is_mgga = self._plan.nldf_settings.sl_level == "MGGA"
    if is_mgga:
        rho_tuple = (n_sg, sigma_xg[::2], tau_sg)
    else:
        rho_tuple = (n_sg, sigma_xg[::2])
    for i in range(nfeat):
        di_s, derivs = self._plan.get_interpolation_arguments(rho_tuple, i=i)
        for s in range(nspin):
            p_qg, dp_qg = self._plan.get_interpolation_coefficients(
                di_s[s].ravel(), i=i
            )
            for a in range(Nalpha):
                x_sig[s, i] += p_qg[a] * y_sbg[s, a]
    for i in range(nfeat):
        di_s, derivs = self._plan.get_interpolation_arguments(rho_tuple, i=i)
        for o in range(norb):
            s = p_o[o][0]
            dadf_g = derivs[0][s] * dndf_og[o] + derivs[1][s] * dsigmadf_og[o]
            if is_mgga:
                dadf_g += derivs[2][s] * dtaudf_og[o]
            p_qg, dp_qg = self._plan.get_interpolation_coefficients(
                di_s[s].ravel(), i=i
            )
            for a in range(Nalpha):
                dxdf_oig[o, i] += dp_qg[a] * dadf_g * y_sbg[s, a]
                dxdf_oig[o, i] += p_qg[a] * dydf_obg[o, a]

    return x_sig, dxdf_oig


def vec_radial_deriv_vars(
    rgd, p_o, n_sLg, dndf_oLg, Y_nL, dndr_sLg, dndrdf_oLg, rnablaY_nLv
):
    nspins = len(n_sLg)
    norb = len(p_o)

    n_sg = np.dot(Y_nL, n_sLg).transpose(1, 0, 2).reshape(nspins, -1)
    dndf_og = np.dot(Y_nL, dndf_oLg).transpose(1, 0, 2).reshape(norb, -1)

    a_sg = np.dot(Y_nL, dndr_sLg).transpose(1, 0, 2)
    dadf_og = np.dot(Y_nL, dndrdf_oLg).transpose(1, 0, 2)
    b_vsg = np.dot(rnablaY_nLv.transpose(0, 2, 1), n_sLg).transpose(1, 2, 0, 3)
    dbdf_vog = np.dot(rnablaY_nLv.transpose(0, 2, 1), dndf_oLg).transpose(1, 2, 0, 3)
    N = Y_nL.shape[0]
    nx = 2 * nspins - 1

    sigma_xg = rgd.empty((nx, N))
    dsigmadf_og = rgd.empty((norb, N))
    sigma_xg[::2] = (b_vsg**2).sum(0)
    if nspins == 2:
        sigma_xg[1] = (b_vsg[:, 0] * b_vsg[:, 1]).sum(0)
    for o, p in enumerate(p_o):
        s = p[0]
        dsigmadf_og[o] = 2 * (b_vsg[:, s] * dbdf_vog[:, o]).sum(0)
    sigma_xg[:, :, 1:] /= rgd.r_g[1:] ** 2
    sigma_xg[:, :, 0] = sigma_xg[:, :, 1]
    dsigmadf_og[:, :, 1:] /= rgd.r_g[1:] ** 2
    dsigmadf_og[:, :, 0] = dsigmadf_og[:, :, 1]
    sigma_xg[::2] += a_sg**2
    if nspins == 2:
        sigma_xg[1] += a_sg[0] * a_sg[1]
    for o, p in enumerate(p_o):
        s = p[0]
        dsigmadf_og[o] += 2 * a_sg[s] * dadf_og[o]

    sigma_xg = sigma_xg.reshape(nx, -1)
    dsigmadf_og = dsigmadf_og.reshape(norb, -1)
    a_sg = a_sg.reshape(nspins, -1)
    b_vsg = b_vsg.reshape(3, nspins, -1)

    return n_sg, dndf_og, sigma_xg, dsigmadf_og, a_sg, b_vsg


def vec_density_vars(xc, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, tau_sg, ae=True):
    nspins = len(n_sLg)

    n_sg = np.dot(Y_nL, n_sLg).transpose(1, 0, 2).reshape(nspins, -1)
    a_sg = np.dot(Y_nL, dndr_sLg).transpose(1, 0, 2)
    # Has shape (v, s, n, g)
    b_vsg = np.dot(rnablaY_nLv.transpose(0, 2, 1), n_sLg).transpose(1, 2, 0, 3)
    dn_svg = b_vsg.transpose(1, 0, 2, 3) + np.einsum("nv,sng->svng", R_nv, a_sg)
    dn_svg = dn_svg.reshape(nspins, 3, -1)
    rho_sig = np.concatenate(
        [n_sg[:, np.newaxis, :], dn_svg, tau_sg[:, np.newaxis, :]], axis=1
    )
    return rho_sig


def get_kinetic_energy_and_deriv(xc, ae):
    nspins = xc._D_sp.shape[0]
    xcc = xc._xcc
    if ae:
        tau_pg = xcc.tau_pg
        tauc_g = xcc.tauc_g / (np.sqrt(4 * np.pi) * nspins)
    else:
        tau_pg = xcc.taut_pg
        tauc_g = xcc.tauct_g / (np.sqrt(4 * np.pi) * nspins)
    nn = tau_pg.shape[-1] // tauc_g.shape[0]
    tau_sg = np.dot(xc._D_sp, tau_pg) + np.tile(tauc_g, nn)
    dtaudf_og = np.dot(xc._DD_op, tau_pg)
    return tau_sg, dtaudf_og


def get_paw_deriv_contribs(xc, p_o, n_sg, sigma_xg, dndf_osg, dsigmadf_oxg, ae=True):
    if xc.cider_kernel.type == "MGGA":
        tau_sg, dtaudf_og = get_kinetic_energy_and_deriv(xc, ae)
        return _get_paw_helper1(
            xc,
            p_o,
            n_sg,
            sigma_xg,
            dndf_osg,
            dsigmadf_oxg,
            tau_sg=tau_sg,
            dtaudf_og=dtaudf_og,
        )
    else:
        return _get_paw_helper1(xc, p_o, n_sg, sigma_xg, dndf_osg, dsigmadf_oxg)


def get_paw_feat_deriv_contribs(
    xc,
    p_o,
    n_sg,
    sigma_xg,
    dndf_osg,
    dsigmadf_oxg,
    y_sbg,
    dydf_obg,
    F_sbg,
    dFdf_obg,
    ae=True,
):
    if xc.cider_kernel.type == "MGGA":
        tau_sg, dtaudf_og = get_kinetic_energy_and_deriv(xc, ae)
        feat_sig, dfeat_oig = _get_paw_helper2(
            xc,
            p_o,
            n_sg,
            sigma_xg,
            dndf_osg,
            dsigmadf_oxg,
            y_sbg,
            dydf_obg,
            F_sbg,
            dFdf_obg,
            ae=ae,
            tau_sg=tau_sg,
            dtaudf_og=dtaudf_og,
        )
    else:
        feat_sig, dfeat_oig = _get_paw_helper2(
            xc,
            p_o,
            n_sg,
            sigma_xg,
            dndf_osg,
            dsigmadf_oxg,
            y_sbg,
            dydf_obg,
            F_sbg,
            dFdf_obg,
            ae=ae,
        )
        tau_sg = None
        dtaudf_og = None
    feat_sig, dfeat_oig = get_features_with_sl_part(
        p_o,
        feat_sig,
        dfeat_oig,
        n_sg,
        sigma_xg,
        dndf_osg,
        dsigmadf_oxg,
        tau_sg,
        dtaudf_og,
    )
    return feat_sig, dfeat_oig


class CiderRadialThetaDerivCalculator:
    def __init__(self, xc):
        self.xc = xc
        self.mode = "step1"

    def __call__(
        self,
        rgd,
        p_o,
        n_sLg,
        dndf_sLg,
        Y_nL,
        dndr_sLg,
        dndrdf_sLg,
        rnablaY_nLv,
        ae=True,
    ):
        (n_sg, dndf_og, sigma_xg, dsigmadf_og, a_sg, b_vsg) = vec_radial_deriv_vars(
            rgd, p_o, n_sLg, dndf_sLg, Y_nL, dndr_sLg, dndrdf_sLg, rnablaY_nLv
        )
        return get_paw_deriv_contribs(
            self.xc, p_o, n_sg, sigma_xg, dndf_og, dsigmadf_og, ae=ae
        )


class CiderRadialFeatCalculator:
    def __init__(self, xc):
        self.xc = xc
        self.mode = "debug_nosph"

    def __call__(self, rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, F_sbg, y_sbg, ae=True):
        (e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, a_sg, b_vsg) = vec_radial_gga_vars(
            rgd,
            n_sLg,
            Y_nL[:, : n_sLg.shape[1]],
            dndr_sLg,
            rnablaY_nLv[:, : n_sLg.shape[1]],
        )
        return self.xc.get_paw_atom_feat(
            n_sg, sigma_xg, y_sbg, F_sbg, ae, include_density=True
        )


class CiderRadialDensityCalculator:
    def __init__(self, xc):
        self.xc = xc
        self.mode = "debug_nosph"

    def __call__(self, rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae=True):
        tau_sg, _ = self.xc.get_kinetic_energy(ae)
        return vec_density_vars(
            self.xc, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, tau_sg, ae=ae
        )


class CiderRadialFeatDerivCalculator:
    def __init__(self, xc):
        self.xc = xc
        self.mode = "step2"

    def __call__(
        self,
        rgd,
        p_o,
        n_sLg,
        dndf_sLg,
        Y_nL,
        dndr_sLg,
        dndrdf_sLg,
        rnablaY_nLv,
        F_sbg,
        dFdf_obg,
        y_sbg,
        dydf_obg,
        ae=True,
    ):
        (n_sg, dndf_og, sigma_xg, dsigmadf_og, a_sg, b_vsg) = vec_radial_deriv_vars(
            rgd, p_o, n_sLg, dndf_sLg, Y_nL, dndr_sLg, dndrdf_sLg, rnablaY_nLv
        )
        return get_paw_feat_deriv_contribs(
            self.xc,
            p_o,
            n_sg,
            sigma_xg,
            dndf_og,
            dsigmadf_og,
            y_sbg,
            dydf_obg,
            F_sbg,
            dFdf_obg,
            ae=ae,
        )


class CiderRadialDensityDerivCalculator:
    def __init__(self, xc):
        self.xc = xc
        self.mode = "rho"

    def __call__(
        self,
        rgd,
        p_o,
        n_sLg,
        dndf_sLg,
        Y_nL,
        dndr_sLg,
        dndrdf_sLg,
        rnablaY_nLv,
        ae=True,
    ):
        tau_sg, dtaudf_og = get_kinetic_energy_and_deriv(self.xc, ae)
        rho = vec_density_vars(
            self.xc, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, tau_sg, ae=ae
        )
        drho = vec_density_vars(
            self.xc, dndf_sLg, Y_nL, dndrdf_sLg, rnablaY_nLv, dtaudf_og, ae=ae
        )
        return rho, drho


class CiderRadialDerivExpansion:
    def __init__(self, rcalc, p_o, F_obgr=None, y_obLg=None, f_in_sph_harm=False):
        self.rcalc = rcalc
        self.p_o = p_o
        if F_obgr is not None:
            nspins, nb = F_obgr.shape[:2]
            if f_in_sph_harm:
                F_obLg = F_obgr
                self.F_obg = np.einsum("sbLg,nL->sbng", F_obLg, Y_nL).reshape(
                    nspins, nb, -1
                )
            else:
                self.F_obg = F_obgr.transpose(0, 1, 3, 2).reshape(nspins, nb, -1)
            self.y_obg = np.einsum("sbLg,nL->sbng", y_obLg, Y_nL).reshape(
                nspins, nb, -1
            )
        assert self.rcalc.mode in ["step1", "step2", "rho"]

    def __call__(self, rgd, D_sLq, DD_oLq, n_qg, dndr_qg, nc0_sg, dnc0dr_sg, ae=True):
        n_sLg = np.dot(D_sLq, n_qg)
        n_sLg[:, 0] += nc0_sg
        dndf_oLg = np.dot(DD_oLq, n_qg)

        dndr_sLg = np.dot(D_sLq, dndr_qg)
        dndr_sLg[:, 0] += dnc0dr_sg
        dndrdf_oLg = np.dot(DD_oLq, dndr_qg)

        nspins, Lmax, nq = D_sLq.shape

        if self.rcalc.mode == "step1":
            x_obg = self.rcalc(
                rgd,
                self.p_o,
                n_sLg,
                dndf_oLg,
                Y_nL[:, :Lmax],
                dndr_sLg,
                dndrdf_oLg,
                rnablaY_nLv[:, :Lmax],
                ae=ae,
            )
            nb = x_obg.shape[1]
            num_o = len(x_obg)
            wx_obng = x_obg.reshape(num_o, nb, Y_nL.shape[0], -1)
            return (
                4
                * np.pi
                * np.einsum("obng,nL->obgL", wx_obng, weight_n[:, None] * Y_nL)
            )
        elif self.rcalc.mode == "step2":
            feat_xg, dfeat_xg = self.rcalc(
                rgd,
                self.p_o,
                n_sLg,
                dndf_oLg,
                Y_nL[:, :Lmax],
                dndr_sLg,
                dndrdf_oLg,
                rnablaY_nLv[:, :Lmax],
                self.F_obg[:nspins],
                self.F_obg[nspins:],
                self.y_obg[:nspins],
                self.y_obg[nspins:],
                ae=ae,
            )
            shape = feat_xg.shape[:-1]
            shape = shape + (Y_nL.shape[0], -1)
            feat_xng = feat_xg.reshape(*shape)
            shape = dfeat_xg.shape[:-1]
            shape = shape + (Y_nL.shape[0], -1)
            dfeat_xng = dfeat_xg.reshape(*shape)
            return feat_xng, dfeat_xng
        elif self.rcalc.mode == "rho":
            feat_xg, dfeat_xg = self.rcalc(
                rgd,
                self.p_o,
                n_sLg,
                dndf_oLg,
                Y_nL[:, :Lmax],
                dndr_sLg,
                dndrdf_oLg,
                rnablaY_nLv[:, :Lmax],
                ae=ae,
            )
            shape = feat_xg.shape[:-1]
            shape = shape + (Y_nL.shape[0], -1)
            feat_xng = feat_xg.reshape(*shape)
            shape = dfeat_xg.shape[:-1]
            shape = shape + (Y_nL.shape[0], -1)
            dfeat_xng = dfeat_xg.reshape(*shape)
            return feat_xng, dfeat_xng
        else:
            raise ValueError
