import numpy as np
from gpaw.sphere.lebedev import R_nv, Y_nL, weight_n
from gpaw.xc.pawcorrection import rnablaY_nLv

from ciderpress.data import get_hgbs_max_exps
from ciderpress.dft.density_util import get_sigma
from ciderpress.dft.grids_indexer import AtomicGridsIndexer
from ciderpress.dft.lcao_convolutions import (
    ATCBasis,
    ConvolutionCollection,
    get_etb_from_expnt_range,
    get_gamma_lists_from_etb_list,
)
from ciderpress.dft.plans import NLDFAuxiliaryPlan, get_ccl_settings
from ciderpress.gpaw.atom_utils import (
    AtomPASDWSlice,
    PAOnlySetup,
    PSOnlySetup,
    SBTGridContainer,
)
from ciderpress.gpaw.etb_util import ETBProjector
from ciderpress.gpaw.fit_paw_gauss_pot import get_dv
from ciderpress.gpaw.interp_paw import (
    DiffPAWXCCorrection,
    calculate_cider_paw_correction,
)


def get_ag_indices(fft_obj, gd, shape, spos_c, rmax, buffer=0):
    ng3 = np.prod(shape)
    center = np.round(spos_c * shape).astype(int)
    disp = np.empty(3, dtype=int)
    lattice = gd.cell_cv
    vol = np.abs(np.linalg.det(lattice))
    dv = vol / ng3
    for i in range(3):
        res = np.cross(lattice[(i + 1) % 3], lattice[(i + 2) % 3])
        # TODO unnecessarily conservative buffer?
        disp[i] = np.ceil(np.linalg.norm(res) * rmax / vol * shape[i]) + 1 + buffer
    indices = [
        np.arange(center[i] - disp[i], center[i] + disp[i] + 1) for i in range(3)
    ]
    fdisps = []
    for i in range(3):
        fdisps.append(indices[i].astype(np.float64) / shape[i] - spos_c[i])
    indices = [ind % s for ind, s in zip(indices, shape)]
    lbound_inds, ubound_inds = fft_obj.get_bound_inds()
    conds = []
    for i in range(3):
        conds.append(
            np.logical_and(indices[i] >= lbound_inds[i], indices[i] < ubound_inds[i])
        )
    local_indices = [indices[i][conds[i]] - lbound_inds[i] for i in range(3)]
    local_fdisps = [fdisps[i][conds[i]] for i in range(3)]
    return local_indices, local_fdisps, dv


class FastAtomPASDWSlice(AtomPASDWSlice):
    def __init__(self, *args, **kwargs):
        super(FastAtomPASDWSlice, self).__init__(*args, **kwargs)
        self.indset = np.ascontiguousarray(self.indset.astype(np.int64))

    @classmethod
    def from_gd_and_setup(
        cls,
        gd,
        spos_c,
        psetup,
        fft_obj,
        rmax=0,
        sphere=True,
        ovlp_fit=False,
        store_funcs=False,
    ):
        if ovlp_fit:
            raise NotImplementedError(
                "Overlap fitting not yet implemented for fast version"
            )
        rgd = psetup.interp_rgd
        if rmax == 0:
            rmax = psetup.rcut
        shape = gd.get_size_of_global_array()
        indices, fdisps, dv = get_ag_indices(fft_obj, gd, shape, spos_c, rmax, buffer=0)
        indset, rad_g, rhat_gv = fft_obj.get_radial_info_for_atom(indices, fdisps)
        if sphere:
            cond = rad_g <= rmax
            indset = indset[cond]
            rad_g = rad_g[cond]
            rhat_gv = rhat_gv[cond]

        # if len(rad_g) == 0:  # No need for an atom slice on this core
        #     return None

        h = rgd.r_g[1] - rgd.r_g[0]
        dg = rad_g / h  # TODO only for equidist grid
        g = np.floor(dg).astype(np.int32)
        g = np.minimum(g, rgd.r_g.size - 1)
        dg -= g

        return cls(
            indset,
            g,
            dg,
            rad_g,
            rhat_gv,
            rmax,
            psetup,
            gd.dv,
            h,
            ovlp_fit=ovlp_fit,
            store_funcs=store_funcs,
        )


class PAWCiderContribs:

    nspin: int
    ccl: ConvolutionCollection
    grids_indexer: AtomicGridsIndexer
    plan: NLDFAuxiliaryPlan

    def __init__(self, plan, cider_kernel, ccl, xcc):
        self.plan = plan
        self.cider_kernel = cider_kernel
        self.ccl = ccl
        self.ccl.compute_integrals_()
        self.ccl.solve_projection_coefficients()
        self.w_g = (xcc.rgd.dv_g[:, None] * weight_n).ravel()
        self.r_g = xcc.rgd.r_g
        self.xcc = xcc
        self.grids_indexer = AtomicGridsIndexer.make_single_atom_indexer(
            Y_nL[:, : xcc.Lmax], self.r_g
        )
        self.grids_indexer.set_weights(self.w_g)
        from gpaw.utilities.timing import Timer

        self.timer = Timer()

    @classmethod
    def from_plan(cls, plan, cider_kernel, Z, xcc, beta=1.8):
        lmax = int(np.sqrt(xcc.Lmax + 1e-8)) - 1
        rmax = np.max(xcc.rgd.r_g)
        # TODO tune this and adjust size
        # min_exps = 0.5 / (rmax * rmax) * np.ones(lp1)
        min_exps = 0.125 / (rmax * rmax) * np.ones(4)
        max_exps = get_hgbs_max_exps(Z)
        etb = get_etb_from_expnt_range(
            lmax, beta, min_exps, max_exps, 0.0, 1e99, lower_fac=1.0, upper_fac=1.0
        )
        inputs_to_atco = get_gamma_lists_from_etb_list([etb])
        atco = ATCBasis(*inputs_to_atco)
        has_vj, ifeat_ids = get_ccl_settings(plan)
        ccl = ConvolutionCollection(
            atco, atco, plan.alphas, plan.alpha_norms, has_vj, ifeat_ids
        )
        # TODO need to define w_g
        return cls(plan, cider_kernel, ccl, xcc)

    @property
    def is_mgga(self):
        return self.plan.nldf_settings.sl_level

    @property
    def nspin(self):
        return self.plan.nspin

    def get_interpolation_coefficients(self, arg_g, i=-1):
        p_gq, dp_gq = self.plan.get_interpolation_coefficients(arg_g, i=i)
        # p_gq[:] *= self.plan.alpha_norms
        # dp_gq[:] *= self.plan.alpha_norms
        # if 0 <= i < self.plan.num_vj:
        #     p_gq[:] *= self.plan.nspin
        #     dp_gq[:] *= self.plan.nspin
        return p_gq, dp_gq

    def get_kinetic_energy(self, ae):
        # TODO need to be careful about the order of g
        nspins = self._D_sp.shape[0]
        xcc = self.xcc
        if ae:
            tau_pg = xcc.tau_pg
            tauc_g = xcc.tauc_g / (np.sqrt(4 * np.pi) * nspins)
        else:
            tau_pg = xcc.taut_pg
            tauc_g = xcc.tauct_g / (np.sqrt(4 * np.pi) * nspins)
        nn = tau_pg.shape[-1] // tauc_g.shape[0]
        tau_sg = np.dot(self._D_sp, tau_pg)
        tau_sg.shape = (tau_sg.shape[0], -1, nn)
        tau_sg[:] += tauc_g[:, None]
        tau_sg.shape = (tau_sg.shape[0], -1)
        return tau_sg

    def contract_kinetic_potential(self, dedtau_sg, ae):
        xcc = self.xcc
        if ae:
            tau_pg = xcc.tau_pg
            sign = 1.0
        else:
            tau_pg = xcc.taut_pg
            sign = -1.0
        wt = self.w_g * sign
        self._dH_sp += np.dot(dedtau_sg * wt, tau_pg.T)

    def vec_radial_vars(self, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae):
        nspin = len(n_sLg)
        ngrid = n_sLg.shape[-1] * weight_n.size
        nx = 5 if self.is_mgga else 4
        rho_sxg = np.empty((nspin, nx, ngrid))
        rho_sxg[:, 0] = np.dot(Y_nL, n_sLg).transpose(1, 2, 0).reshape(nspin, -1)
        b_vsgn = np.dot(rnablaY_nLv.transpose(0, 2, 1), n_sLg).transpose(1, 2, 3, 0)
        b_vsgn[..., 1:, :] /= self.xcc.rgd.r_g[1:, None]
        b_vsgn[..., 0, :] = b_vsgn[..., 1, :]
        a_sgn = np.dot(Y_nL, dndr_sLg).transpose(1, 2, 0)
        b_vsgn += R_nv.T[:, None, None, :] * a_sgn[None, :, :]
        N = Y_nL.shape[0]
        e_g = self.xcc.rgd.empty(N).reshape(-1)
        e_g[:] = 0
        rho_sxg[:, 1:4] = b_vsgn.transpose(1, 0, 2, 3).reshape(nspin, 3, -1)
        if self.is_mgga:
            rho_sxg[:, 4] = self.get_kinetic_energy(ae)
        vrho_sxg = np.zeros_like(rho_sxg)
        return e_g, rho_sxg, vrho_sxg

    @property
    def atco_feat(self):
        # TODO this will need to be different for vi and vij
        return self.ccl.atco_out

    def convert_rad2orb_(
        self, nspin, x_sgLq, x_suq, rad2orb=True, inp=True, indexer=None
    ):
        if inp:
            atco = self.ccl.atco_inp
        else:
            atco = self.atco_feat
        if indexer is None:
            indexer = self.grids_indexer
        for s in range(nspin):
            atco.convert_rad2orb_(
                x_sgLq[s],
                x_suq[s],
                indexer,
                indexer.rad_arr,
                rad2orb=rad2orb,
                offset=0,
            )

    def set_D_sp(self, D_sp, xcc):
        self._dH_sp = np.zeros_like(D_sp)
        self._D_sp = D_sp

    def get_dH_sp(self):
        return self._dH_sp

    def perform_convolution(
        self,
        theta_sgLq,
        fwd=True,
        out_sgLq=None,
        return_orbs=False,
        indexer_in=None,
        indexer_out=None,
    ):
        theta_sgLq = np.ascontiguousarray(theta_sgLq)
        nspin = self.nspin
        nalpha = self.ccl.nalpha
        nbeta = self.ccl.nbeta
        is_not_vi = self.plan.nldf_settings.nldf_type != "i"
        inp_arr = np.zeros((nspin, self.ccl.atco_inp.nao, nalpha), order="C")
        out_arr = np.zeros((nspin, self.ccl.atco_out.nao, nbeta), order="C")
        if indexer_in is None:
            indexer_in = self.grids_indexer
        if indexer_out is None:
            indexer_out = self.grids_indexer
        if out_sgLq is None:
            if fwd:
                nout = nbeta
            else:
                nout = nalpha
            out_sgLq = np.stack(
                [indexer_out.empty_rlmq(nalpha=nout) for s in range(nspin)]
            )
            out_sgLq[:] = 0.0
        if fwd:
            i_inp = -1
            i_out = 0
            theta_suq = inp_arr
            conv_svq = out_arr
        else:
            i_inp = 0
            i_out = -1
            theta_suq = out_arr
            conv_svq = inp_arr
        self.convert_rad2orb_(
            nspin, theta_sgLq, theta_suq, rad2orb=True, inp=fwd, indexer=indexer_in
        )
        for s in range(nspin):
            if (fwd) or (is_not_vi):
                self.plan.get_transformed_interpolation_terms(
                    theta_suq[s, :, : self.plan.nalpha], i=i_inp, fwd=fwd, inplace=True
                )
            self.ccl.multiply_atc_integrals(theta_suq[s], output=conv_svq[s], fwd=fwd)
            if (not fwd) or (is_not_vi):
                self.plan.get_transformed_interpolation_terms(
                    conv_svq[s, :, : self.plan.nalpha],
                    i=i_out,
                    fwd=not fwd,
                    inplace=True,
                )
        # TODO this should be a modified atco for l1 interpolation
        self.convert_rad2orb_(
            nspin, out_sgLq, conv_svq, rad2orb=False, inp=not fwd, indexer=indexer_out
        )
        if return_orbs:
            return out_sgLq, conv_svq
        else:
            return out_sgLq

    def get_paw_atom_contribs(self, rho_sxg):
        nspin = self.nspin
        assert len(rho_sxg) == nspin
        x_gq = self.grids_indexer.empty_gq(nalpha=self.plan.nalpha)
        x_srLq = np.stack(
            [
                self.grids_indexer.empty_rlmq(nalpha=self.plan.nalpha)
                for s in range(nspin)
            ]
        )
        for s in range(nspin):
            rho_tuple = self.plan.get_rho_tuple(rho_sxg[s])
            arg_g = self.plan.get_interpolation_arguments(rho_tuple, i=-1)[0]
            fun_g = self.plan.get_function_to_convolve(rho_tuple)[0]
            fun_g[:] *= self.w_g
            p_gq = self.get_interpolation_coefficients(arg_g.ravel(), i=-1)[0]
            x_gq[:] = p_gq * fun_g[:, None]
            self.grids_indexer.reduce_angc_ylm_(x_srLq[s], x_gq, a2y=True, offset=0)
        return x_srLq

    def get_paw_atom_contribs_en(self, e_g, rho_sxg, vrho_sxg, f_srLq, feat_only=False):
        # NOTE that grid index order is different than previous implementation
        # gn instead of previous ng
        nspin = rho_sxg.shape[0]
        ngrid = rho_sxg.shape[2]
        f_srLq = np.ascontiguousarray(f_srLq)
        vf_srLq = np.empty_like(f_srLq)
        feat_sig = np.zeros((nspin, self.plan.nldf_settings.nfeat, ngrid))
        dfeat_sig = np.zeros((nspin, self.plan.num_vj, ngrid))
        f_gq = self.grids_indexer.empty_gq(nalpha=self.plan.nalpha)
        for s in range(nspin):
            self.grids_indexer.reduce_angc_ylm_(f_srLq[s], f_gq, a2y=False, offset=0)
            self.plan.eval_rho_full(
                f_gq,
                rho_sxg[s],
                spin=s,
                feat=feat_sig[s],
                dfeat=dfeat_sig[s],
                cache_p=True,
                # TODO might need to be True for gaussian interp
                apply_transformation=False,
                # coeff_multipliers=self.plan.alpha_norms,
            )
        if feat_only:
            return feat_sig
        sigma_xg = get_sigma(rho_sxg[:, 1:4])
        dedsigma_xg = np.zeros_like(sigma_xg)
        nspin = feat_sig.shape[0]
        dedn_sg = np.zeros_like(vrho_sxg[:, 0])
        e_g[:] = 0.0
        args = [
            e_g,
            rho_sxg[:, 0].copy(),
            dedn_sg,
            sigma_xg,
            dedsigma_xg,
        ]
        if rho_sxg.shape[1] == 4:
            args.append(feat_sig)
        elif rho_sxg.shape[1] == 5:
            dedtau_sg = np.zeros_like(vrho_sxg[:, 0])
            args += [
                rho_sxg[:, 4].copy(),
                dedtau_sg,
                feat_sig,
            ]
        else:
            raise ValueError
        vfeat_sig = self.cider_kernel.calculate(*args)
        vrho_sxg[:, 0] = dedn_sg
        if rho_sxg.shape[1] == 5:
            vrho_sxg[:, 4] = dedtau_sg
        vrho_sxg[:, 1:4] = 2 * dedsigma_xg[::2, None, :] * rho_sxg[:, 1:4]
        if nspin == 2:
            vrho_sxg[:, 1:4] += dedsigma_xg[1, None, :] * rho_sxg[::-1, 1:4]
        for s in range(nspin):
            f_gq[:] = self.plan.eval_vxc_full(
                vfeat_sig[s],
                vrho_sxg[s],
                dfeat_sig[s],
                rho_sxg[s],
                spin=s,
            )
            f_gq[:] *= self.w_g[:, None]
            self.grids_indexer.reduce_angc_ylm_(vf_srLq[s], f_gq, a2y=True, offset=0)
        return vf_srLq

    def get_paw_atom_contribs_pot(self, rho_sxg, vrho_sxg, vx_srLq):
        nspin = self.nspin
        vx_gq = self.grids_indexer.empty_gq(nalpha=self.plan.nalpha)
        for s in range(nspin):
            self.grids_indexer.reduce_angc_ylm_(vx_srLq[s], vx_gq, a2y=False, offset=0)
            rho_tuple = self.plan.get_rho_tuple(rho_sxg[s])
            arg_g, darg_g = self.plan.get_interpolation_arguments(rho_tuple, i=-1)
            # TODO need to account for dfunc
            fun_g, dfun_g = self.plan.get_function_to_convolve(rho_tuple)
            p_gq, dp_gq = self.get_interpolation_coefficients(arg_g, i=-1)
            dedarg_g = np.einsum("gq,gq->g", dp_gq, vx_gq) * fun_g
            dedfun_g = np.einsum("gq,gq->g", p_gq, vx_gq)
            vrho_sxg[s, 0] += dedarg_g * darg_g[0] + dedfun_g * dfun_g[0]
            tmp = dedarg_g * darg_g[1] + dedfun_g * dfun_g[1]
            vrho_sxg[s, 1:4] += 2 * rho_sxg[s, 1:4] * tmp
            if len(darg_g) == 2:
                vrho_sxg[s, 4] += dedarg_g * darg_g[2] + dedfun_g * dfun_g[2]

    # TODO remove these last two functions
    # get the y values
    def calc_y_sbLk(self, dx_sLkb, ks=None):
        # dx_sqkL = (nspin, Nq, Nk, (L+1)^2)
        # needs to be separate for CIDER and VDW
        if ks is None:
            ks = self.krad
        y_bsLk = np.zeros(np.array(dx_sLkb.shape)[[3, 0, 1, 2]])
        for b in range(y_bsLk.shape[0]):
            aexp = self.plan.alphas[b] + self.plan.alphas
            fac = (np.pi / aexp) ** 1.5
            y_bsLk[b, :, :, :] += np.einsum(
                "kq,sLkq->sLk",
                np.exp(-ks[:, None] ** 2 / (4 * aexp)) * fac,
                dx_sLkb,
            )
        return y_bsLk.transpose(1, 0, 2, 3)

    def get_atomic_convolution(self, y_sbLk, xcc):
        k_g = xcc.big_rgd.k_g
        return np.ascontiguousarray(
            self.calc_y_sbLk(y_sbLk.transpose(0, 2, 3, 1), ks=k_g)
        )


class CiderRadialFeatureCalculator:
    def __init__(self, xc):
        self.xc = xc
        self.mode = "feature"

    def __call__(self, rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae=True):
        e_g, rho_sxg, vrho_sxg = self.xc.vec_radial_vars(
            n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae
        )
        return self.xc.get_paw_atom_contribs(rho_sxg)


class CiderRadialEnergyCalculator:
    def __init__(self, xc):
        self.xc = xc
        self.mode = "energy"

    def __call__(self, rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, f_srLq, ae=True):
        e_g, rho_sxg, vrho_sxg = self.xc.vec_radial_vars(
            n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae
        )
        vf_srLq = self.xc.get_paw_atom_contribs_en(
            e_g, rho_sxg, vrho_sxg, f_srLq, feat_only=False
        )
        if rho_sxg.shape[1] == 5:
            self.xc.contract_kinetic_potential(vrho_sxg[:, 4], ae)
        return e_g, vrho_sxg[:, 0], vrho_sxg[:, 1:4], vf_srLq


class CiderRadialPotentialCalculator:
    def __init__(self, xc):
        self.xc = xc
        self.mode = "potential"

    def __call__(self, rgd, n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, vx_srLq, ae=True):
        e_g, rho_sxg, vrho_sxg = self.xc.vec_radial_vars(
            n_sLg, Y_nL, dndr_sLg, rnablaY_nLv, ae
        )
        self.xc.get_paw_atom_contribs_pot(rho_sxg, vrho_sxg, vx_srLq)
        if rho_sxg.shape[1] == 5:
            self.xc.contract_kinetic_potential(vrho_sxg[:, 4], ae)
        return vrho_sxg[:, 0], vrho_sxg[:, 1:4]


class CiderRadialExpansion:
    def __init__(self, rcalc, ft_srLq=None, df_srLq=None):
        self.rcalc = rcalc
        self.ft_srLq = ft_srLq
        self.df_srLq = df_srLq
        assert self.rcalc.mode in [
            "feature",
            "energy",
            "potential",
        ]

    def __call__(self, rgd, D_sLq, n_qg, dndr_qg, nc0_sg, dnc0dr_sg, ae=True):
        n_sLg = np.dot(D_sLq, n_qg)
        n_sLg[:, 0] += nc0_sg
        dndr_sLg = np.dot(D_sLq, dndr_qg)
        dndr_sLg[:, 0] += dnc0dr_sg
        nspins, Lmax, nq = D_sLq.shape
        if self.rcalc.mode == "feature":
            # We call it wx_srLq because it is multiplied
            # by the grid weights already
            wx_srLq = self.rcalc(
                rgd, n_sLg, Y_nL[:, :Lmax], dndr_sLg, rnablaY_nLv[:, :Lmax], ae=ae
            )
            return wx_srLq
        elif self.rcalc.mode == "energy":
            dEdD_sqL = np.zeros((nspins, nq, Lmax))
            f_srLq = self.ft_srLq + self.df_srLq if ae else self.ft_srLq
            e_g, dedn_sg, dedgrad_svg, vf_srLq = self.rcalc(
                rgd,
                n_sLg,
                Y_nL[:, :Lmax],
                dndr_sLg,
                rnablaY_nLv[:, :Lmax],
                f_srLq,
                ae=ae,
            )
            nn = Y_nL.shape[0]
            nq = f_srLq.shape[-1]
            dedn_sgn = dedn_sg.reshape(nspins, -1, nn)
            dedgrad_svgn = dedgrad_svg.reshape(nspins, 3, -1, nn)
            dedgrad_sgn = np.einsum("svgn,nv->sgn", dedgrad_svgn, R_nv)
            dEdD_sqn = np.einsum("qg,sgn->sqn", n_qg * rgd.dv_g, dedn_sgn)
            dEdD_sqn += np.einsum("qg,sgn->sqn", dndr_qg * rgd.dv_g, dedgrad_sgn)
            dEdD_sqL = np.einsum(
                "sqn,nL->sqL", dEdD_sqn, weight_n[:, None] * Y_nL[:, :Lmax]
            )
            tmp = rgd.dr_g * rgd.r_g
            B_svqn = np.einsum("qg,svgn->svqn", n_qg * tmp, dedgrad_svgn)
            dEdD_sqL += np.einsum(
                "nLv,svqn->sqL",
                (4 * np.pi) * weight_n[:, None, None] * rnablaY_nLv[:, :Lmax],
                B_svqn,
            )
            E = rgd.integrate(e_g.reshape(-1, nn).dot(weight_n))
            return E, dEdD_sqL, vf_srLq
        else:
            f_srLq = self.ft_srLq + self.df_srLq if ae else self.ft_srLq
            dedn_sg, dedgrad_svg = self.rcalc(
                rgd,
                n_sLg,
                Y_nL[:, :Lmax],
                dndr_sLg,
                rnablaY_nLv[:, :Lmax],
                f_srLq,
                ae=ae,
            )
            nn = Y_nL.shape[0]
            nq = f_srLq.shape[-1]
            dedn_sgn = dedn_sg.reshape(nspins, -1, nn)
            dedgrad_svgn = dedgrad_svg.reshape(nspins, 3, -1, nn)
            dedgrad_sgn = np.einsum("svgn,nv->sgn", dedgrad_svgn, R_nv)
            dEdD_sqn = np.einsum("qg,sgn->sqn", n_qg * rgd.dv_g, dedn_sgn)
            dEdD_sqn += np.einsum("qg,sgn->sqn", dndr_qg * rgd.dv_g, dedgrad_sgn)
            # dEdD_sqn = np.einsum("qg,sgn->sqn", n_qg, dedn_sgn)
            # dEdD_sqn += np.einsum("qg,sgn->sqn", dndr_qg, dedgrad_sgn)
            dEdD_sqL = np.einsum(
                "sqn,nL->sqL", dEdD_sqn, weight_n[:, None] * Y_nL[:, :Lmax]
            )
            tmp = rgd.dr_g * rgd.r_g
            B_svqn = np.einsum("qg,svgn->svqn", n_qg * tmp, dedgrad_svgn)
            dEdD_sqL += np.einsum(
                "nLv,svqn->sqL",
                (4 * np.pi) * weight_n[:, None, None] * rnablaY_nLv[:, :Lmax],
                B_svqn,
            )
            return dEdD_sqL


class FastPASDWCiderKernel:
    is_cider_functional = True
    """
    This class is a bit confusing because there are a lot of intermediate
    terms and contributions for the pseudo and all-electron parts of the
    feature convolutions. Here is a summary of the terms:
        xt  : The theta functions for the pseudo-density
        x   : The theta functions for the all-electron density
        dx  : x - xt
        yt  : Raw convolutions xt
        dy  : Raw convolutions of dx
        c   : Coefficients for functions to smoothly fit the (non-smooth)
              dx such that dy and dfr match outside the core radius
        dfr : The values of the convolutions of the PASDW projector functions,
              weighted by c
        fr  : dfr + yt
        ft  : fr + contributions to the convolutions from other atoms, which
              are computed based on the projections of the FFT convolutions
              onto smooth atomic functions.
        df  : Projection of (dy - dfr) onto some basis that decays outside
              the core radius. The idea is that df holds all the high-frequency
              parts of the convolutions and (approximately) vanishes
              outside the core radius.
        f   : df + ft, should accurately approximate the "true" all-electron
              features in the core radius
        v*  : Derivative of the XC energy with respect to *.
              vf, vdf, vft, vfr, etc... are all defined like this.

    After the underscore on each array, the indices are indicated:
        a : Atom
        L : l,m spherical harmonic index
        i : Projector function index
        u : Dense atomic density fitting basis set index
        q : Index for the theta functions or convolutions
        g : Radial grid index
        s : Spin index
    """

    fr_asgLq: dict
    df_asgLq: dict
    # TODO below vfr and vdf not needed, can reuse fr and df to save memory
    vfr_asgLq: dict
    vdf_asgLq: dict
    rho_asxg: dict
    vrho_asxg: dict
    rhot_asxg: dict
    vrhot_asxg: dict
    # TODO This can be reused for D_asiq and vc_asiq
    c_asiq: dict

    def __init__(self, cider_kernel, plan, gd, cut_xcgrid, timer=None):
        if timer is None:
            from gpaw.utilities.timing import Timer

            self.timer = Timer()
        self.gd = gd
        self.bas_exp_fit = plan.alphas
        self.cider_kernel = cider_kernel
        self.Nalpha_small = plan.nalpha
        self.cut_xcgrid = cut_xcgrid
        self._amin = np.min(self.bas_exp_fit)
        self.plan = plan
        self.is_mgga = plan.nldf_settings.sl_level == "MGGA"

    @property
    def lambd(self):
        return self.plan.lambd

    def get_D_asp(self):
        return self.atomdist.to_work(self.dens.D_asp)

    def initialize(self, density, atomdist, atom_partition, setups):
        self.dens = density
        self.atomdist = atomdist
        self.atom_partition = atom_partition
        self.setups = setups

    def initialize_more_things(self, setups=None):
        self.nspin = self.dens.nt_sg.shape[0]
        if setups is None:
            setups = self.dens.setups
        for setup in setups:
            if not hasattr(setup, "nlxc_correction"):
                setup.xc_correction = DiffPAWXCCorrection.from_setup(
                    setup, build_kinetic=self.is_mgga, ke_order_ng=False
                )
                # TODO it would be good to remove this
                setup.nlxc_correction = SBTGridContainer.from_setup(
                    # setup, rmin=setup.xc_correction.rgd.r_g[0]+1e-5, N=1024, encut=1e6, d=0.018
                    setup,
                    rmin=1e-4,
                    N=1024,
                    encut=5e6,
                    d=0.015,
                )
                encut = setup.Z**2 * 20
                if encut - 1e-7 <= np.max(self.bas_exp_fit):
                    encut0 = np.max(self.bas_exp_fit)
                    Nalpha = self.bas_exp_fit.size
                else:
                    Nalpha = (
                        int(np.ceil(np.log(encut / self._amin) / np.log(self.lambd)))
                        + 1
                    )
                    encut0 = self._amin * self.lambd ** (Nalpha - 1)
                    assert encut0 >= encut - 1e-6, "Math went wrong {} {}".format(
                        encut0, encut
                    )
                    assert (
                        encut0 / self.lambd < encut
                    ), "Math went wrong {} {} {} {}".format(
                        encut0, encut, self.lambd, encut0 / self.lambd
                    )
                atom_plan = self.plan.new(nalpha=Nalpha)
                setup.cider_contribs = PAWCiderContribs.from_plan(
                    atom_plan, self.cider_kernel, setup.Z, setup.xc_correction, beta=1.8
                )
                if setup.cider_contribs.plan is not None:
                    assert (
                        abs(np.max(setup.cider_contribs.plan.alphas) - encut0) < 1e-10
                    )
                setup.pasdw_setup = PSOnlySetup.from_setup(
                    setup,
                    self.bas_exp_fit,
                    self.bas_exp_fit,
                    # alpha_norms=self.plan.alpha_norms,
                    pmax=encut0,
                )
                setup.paonly_setup = PAOnlySetup.from_setup(setup)
                setup.etb_projector = ETBProjector(
                    setup.Z,
                    # setup.xc_correction.rgd,
                    setup.nlxc_correction.big_rgd,
                    setup.nlxc_correction.big_rgd,
                    alpha_min=2 * self._amin,
                )

    def calculate_paw_cider_features(self, setups, D_asp):
        """
        Computes the contributions to the convolutions F_beta arising
        from the PAW contributions. Projects them onto
        the PAW grids, and also computes the compensation function
        coefficients c_sia that reproduce the F_beta contributions
        outside the augmentation region.

        Args:
            setups: GPAW atomic setups
            D_asp: Atomic PAW density matrices
        """
        if len(D_asp.keys()) == 0:
            return {}, {}
        a0 = list(D_asp.keys())[0]
        nspin = D_asp[a0].shape[0]
        assert (nspin == 1) or self.is_cider_functional
        self.fr_asgLq = {}
        self.df_asgLq = {}
        self.c_asiq = {}

        for a, D_sp in D_asp.items():
            setup = setups[a]
            psetup = setup.pasdw_setup
            Nalpha_sm = psetup.phi_jabg.shape[1]
            rgd = setup.nlxc_correction.big_rgd
            slgd = setup.xc_correction.rgd
            RCUT = np.max(setup.rcut_j)

            self.timer.start("coefs")
            rcalc = CiderRadialFeatureCalculator(setup.cider_contribs)
            expansion = CiderRadialExpansion(rcalc)
            dx_sgLq, xt_sgLq = calculate_cider_paw_correction(
                expansion, setup, D_sp, separate_ae_ps=True
            )
            dx_sgLq -= xt_sgLq
            self.timer.stop()
            alpha_norms = setup.cider_contribs.plan.alpha_norms

            rcut = RCUT
            rmax = slgd.r_g[-1]
            fcut = 0.5 + 0.5 * np.cos(np.pi * (slgd.r_g - rcut) / (rmax - rcut))
            fcut[slgd.r_g < rcut] = 1.0
            fcut[slgd.r_g > rmax] = 0.0
            xt_sgLq *= fcut[:, None, None]

            self.timer.start("transform and convolve")
            # TODO would be faster to stay in orbital space for
            # get_f_realspace_contribs and get_df_only
            indexer = AtomicGridsIndexer.make_single_atom_indexer(
                Y_nL[:, : setup.xc_correction.Lmax], rgd.r_g
            )
            dy_sgLq = setup.cider_contribs.perform_convolution(
                dx_sgLq, fwd=True, return_orbs=False, indexer_out=indexer
            )
            yt_sgLq = setup.cider_contribs.perform_convolution(
                xt_sgLq, fwd=True, return_orbs=False
            )
            self.timer.stop()
            self.timer.start("separate long and short-range")
            # TODO between here and the block comment should be replaced
            # with the code in the block comment after everything else works.
            # dv_g = get_dv(slgd)

            # dy_sqLk = setup.etb_projector.r2k((dy_sgLq).transpose(0, 3, 2, 1))
            dy_sqLk = setup.etb_projector.r2k(
                (dy_sgLq / alpha_norms).transpose(0, 3, 2, 1)
            )

            c_siq, df_sLpq = psetup.get_c_and_df(
                dy_sqLk[:, :Nalpha_sm], realspace=False
            )
            dfr_sgLq = psetup.get_f_realspace_contribs(c_siq, sl=True).transpose(
                0, 3, 2, 1
            )
            df_sLpq = np.append(
                df_sLpq,
                psetup.get_df_only(
                    dy_sqLk[:, Nalpha_sm:], rgd, sl=False, realspace=False
                ),
                axis=-1,
            )
            df_sgLq = psetup.get_df_realspace_contribs(df_sLpq, sl=True).transpose(
                0, 3, 2, 1
            )
            """
            c_siq, df_suq = psetup.get_c_and_df(
                dy_sgLq[..., :Nalpha_sm], realspace=True
            )
            dfr_sgLq = psetup.get_f_realspace_contribs(c_siq, sl=True)
            df_suq = np.append(
                df_suq,
                psetup.get_df_only(
                    dy_sgLq[..., Nalpha_sm:], rgd, sl=False, realspace=True
                ),
                axis=-1,
            )
            df_sgLq = psetup.get_df_realspace_contribs(df_suq, sl=True)
            """
            self.df_asgLq[a] = df_sgLq
            yt_sgLq /= alpha_norms
            self.fr_asgLq[a] = yt_sgLq
            self.fr_asgLq[a][..., :Nalpha_sm] += dfr_sgLq
            self.timer.stop()
            # TODO overlap fit
            # if self.ovlp_fit:
            #     c_siq[:] = atom_slice.sinv_pf.dot(c_siq)
            self.c_asiq[a] = c_siq
        return self.c_asiq, self.df_asgLq

    def calculate_paw_cider_energy(self, setups, D_asp, D_asiq):
        """
        Taking the projections of the pseudo-convolutions F_beta onto the PASDW
        projectors (coefficients given by D_sabi), as well as the
        reference F_beta contributions from part (1), computes the PAW
        XC energy and potential, as well as the functional derivative
        with respect to D_sabi

        Args:
            setups: GPAW atomic setups
            D_asp: Atomic PAW density matrices
            D_asiq: Atomic PAW PASDW projections
        """
        deltaE = {}
        deltaV = {}
        if len(D_asp.keys()) == 0:
            return {}, {}, {}
        a0 = list(D_asp.keys())[0]
        nspin = D_asp[a0].shape[0]
        dvD_asiq = {}
        self.vfr_asgLq = {}
        self.vdf_asgLq = {}

        for a, D_sp in D_asp.items():
            setup = setups[a]
            psetup = setup.paonly_setup
            ni = psetup.ni
            slgd = setup.xc_correction.rgd
            Nalpha_sm = D_asiq[a].shape[-1]
            # TODO ovlp_fit
            # TODO replace the for loop for with acall to the setup
            dv_g = get_dv(slgd)
            df_sgLq = self.df_asgLq[a]
            ft_sgLq = np.zeros_like(df_sgLq)
            fr_sgLq = self.fr_asgLq[a]
            for i in range(ni):
                L = psetup.lmlist_i[i]
                n = psetup.nlist_i[i]
                Dref_sq = (psetup.pfuncs_ng[n] * dv_g).dot(fr_sgLq[:, :, L, :Nalpha_sm])
                j = psetup.jlist_i[i]
                for s in range(nspin):
                    ft_sgLq[s, :, L, :Nalpha_sm] += (
                        D_asiq[a][s, i] - Dref_sq[s]
                    ) * psetup.ffuncs_jg[j][:, None]
            ft_sgLq[:] += fr_sgLq
            rcalc = CiderRadialEnergyCalculator(setup.cider_contribs)
            alpha_norms = setup.cider_contribs.plan.alpha_norms
            expansion = CiderRadialExpansion(
                rcalc,
                ft_sgLq * alpha_norms,
                df_sgLq * alpha_norms,
            )
            deltaE[a], deltaV[a], vf_sgLq, vft_sgLq = calculate_cider_paw_correction(
                expansion,
                setup,
                D_sp,
                separate_ae_ps=True,
            )
            vf_sgLq[:] *= alpha_norms
            vft_sgLq[:] *= alpha_norms
            vfr_sgLq = vf_sgLq - vft_sgLq
            vft_sgLq[:] = vfr_sgLq
            dvD_asiq[a] = np.zeros((nspin, ni, Nalpha_sm))
            for i in range(ni):
                L = psetup.lmlist_i[i]
                n = psetup.nlist_i[i]
                j = psetup.jlist_i[i]
                dvD_sq = np.einsum(
                    "sgq,g->sq",
                    vft_sgLq[:, :, L, :Nalpha_sm],
                    psetup.ffuncs_jg[j],
                )
                dvD_asiq[a][:, i] = dvD_sq
                vfr_sgLq[:, :, L, :Nalpha_sm] -= (
                    dvD_sq[:, None, :] * psetup.pfuncs_ng[n][:, None]
                ) * dv_g[:, None]
            self.vfr_asgLq[a] = vfr_sgLq
            self.vdf_asgLq[a] = vf_sgLq
        return dvD_asiq, deltaE, deltaV

    def calculate_paw_cider_potential(self, setups, D_asp, vc_asiq):
        """
        Given functional derivatives with respect to the augmentation function
        coefficients (c_sabi), as well as other atom-centered functional
        derivative contributions stored from part (2), compute the XC potential
        arising from the nonlocal density dependence of the CIDER features.

        Args:
            setups: GPAW atomic setups
            D_asp: Atomic PAW density matrices
            vc_asiq: functional derivatives with respect to coefficients c_asiq
        """
        dH_asp = {}
        if len(D_asp.keys()) == 0:
            return {}
        a0 = list(D_asp.keys())[0]
        nspin = D_asp[a0].shape[0]
        assert (nspin == 1) or self.is_cider_functional

        for a, D_sp in D_asp.items():
            setup = setups[a]
            psetup = setup.pasdw_setup
            slgd = setup.xc_correction.rgd
            Nalpha_sm = psetup.phi_jabg.shape[1]
            RCUT = np.max(setup.rcut_j)
            # TODO overlap fit here
            alpha_norms = setup.cider_contribs.plan.alpha_norms
            vc_siq = vc_asiq[a]
            vdf_sgLq = self.vdf_asgLq[a]
            vdfr_sgLq = self.vfr_asgLq[a][..., :Nalpha_sm]
            vyt_sgLq = self.vfr_asgLq[a] / alpha_norms
            """
            vc_siq += psetup.get_vf_realspace_contribs(vyt_sgLq, slgd, sl=True)
            vdf_suq = psetup.get_vdf_realspace_contribs(vdf_sgLq, slgd, sl=True)
            vdy_sgLq = psetup.get_v_from_c_and_df(
                vc_siq, vdf_suq[..., :Nalpha_sm], realspace=True
            )
            vdy_sgLq = np.append(
                vdy_sgLq, psetup.get_vdf_only(vdf_sgLq[..., Nalpha_sm:], realspace=True)
            )
            """
            # TODO remove below code block and replace with comment block above
            vc_siq += psetup.get_vf_realspace_contribs(
                vdfr_sgLq.transpose(0, 3, 2, 1), None, sl=True
            )
            vdf_sLpq = psetup.get_vdf_realspace_contribs(
                vdf_sgLq.transpose(0, 3, 2, 1), None, sl=True
            )
            vdy_sqLk = psetup.get_v_from_c_and_df(
                vc_siq, vdf_sLpq[..., :Nalpha_sm], realspace=False
            )
            vdy_sqLk = np.append(
                vdy_sqLk,
                psetup.get_vdf_only(vdf_sLpq[..., Nalpha_sm:], realspace=False),
                axis=1,
            )

            vdy_sgLq = setup.etb_projector.k2r(vdy_sqLk).transpose(0, 3, 2, 1)
            vdy_sgLq[:] /= alpha_norms
            vdy_sgLq[:] *= (
                get_dv(setup.nlxc_correction.big_rgd)[:, None, None] * 2 / np.pi
            )

            self.timer.start("transform and convolve")
            # TODO would be faster to stay in orbital space for
            # get_f_realspace_contribs and get_df_only
            indexer = AtomicGridsIndexer.make_single_atom_indexer(
                Y_nL[:, : setup.xc_correction.Lmax], setup.nlxc_correction.big_rgd.r_g
            )
            vdx_sgLq = setup.cider_contribs.perform_convolution(
                vdy_sgLq, fwd=False, return_orbs=False, indexer_in=indexer
            )
            vxt_sgLq = setup.cider_contribs.perform_convolution(
                vyt_sgLq, fwd=False, return_orbs=False
            )
            self.timer.stop()

            rcut = RCUT
            rmax = slgd.r_g[-1]
            fcut = 0.5 + 0.5 * np.cos(np.pi * (slgd.r_g - rcut) / (rmax - rcut))
            fcut[slgd.r_g < rcut] = 1.0
            fcut[slgd.r_g > rmax] = 0.0
            vxt_sgLq *= fcut[:, None, None]

            F_sbLg = vdx_sgLq - vxt_sgLq
            y_sbLg = vxt_sgLq.copy()

            rcalc = CiderRadialPotentialCalculator(setup.cider_contribs)
            expansion = CiderRadialExpansion(
                rcalc,
                F_sbLg,
                y_sbLg,
            )
            dH_asp[a] = calculate_cider_paw_correction(
                expansion, setup, D_sp, separate_ae_ps=True
            )
        return dH_asp

    def calculate_paw_feat_corrections(
        self,
        setups,
        D_asp,
        D_asiq=None,
        df_asgLq=None,
        vc_asiq=None,
    ):
        """
        This function performs one of three PAW correction routines,
        depending on the inputs.
            1) If D_sabi is None and vc_sabi is None: Computes the
            contributions to the convolutions F_beta arising
            from the PAW contributions. Projects them onto
            the PAW grids, and also computes the compensation function
            coefficients c_sia that reproduce the F_beta contributions
            outside the augmentation region.
            2) If D_sabi is NOT None (df_asbLg must also be given): Taking the
            projections of the pseudo-convolutions F_beta onto the PASDW
            projectors (coefficients given by D_sabi), as well as the
            reference F_beta contributions from part (1), computes the PAW
            XC energy and potential, as well as the functional derivative
            with respect to D_sabi
            3) If D_sabi is None and vc_sabi is NOT None: Given functional
            derivatives with respect to the augmentation function coefficients
            (c_sabi), as well as other atom-centered functional derivative
            contributions stored from part (2), compute the XC potential arising
            from the nonlocal density dependence of the CIDER features.

        Args:
            setups: GPAW atomic setups
            D_asp: Atomic PAW density matrices
            D_sabi: Atomic PAW PASDW projections
            df_asbLg: delta-F_beta convolutions
            vc_sabi: functional derivatives with respect to coefficients c_sabi
        """
        if D_asiq is None and vc_asiq is None:
            return self.calculate_paw_cider_features(setups, D_asp)
        elif D_asiq is not None:
            assert df_asgLq is not None
            return self.calculate_paw_cider_energy(setups, D_asp, D_asiq)
        else:
            assert vc_asiq is not None
            return self.calculate_paw_cider_potential(setups, D_asp, vc_asiq)
