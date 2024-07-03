import gpaw.mpi as mpi
import numpy as np
from gpaw.sphere.lebedev import weight_n

from ciderpress.dft.density_util import get_sigma
from ciderpress.dft.plans import NLDFSplinePlan


class PAWCiderKernelShell:

    is_cider_functional = True

    def __init__(self, world, timer):
        if world is None:
            self.world = mpi.world
        else:
            self.world = world
        self.timer = timer

    def get_D_asp(self):
        return self.atomdist.to_work(self.dens.D_asp)

    def initialize(self, density, atomdist, atom_partition, setups):
        self.dens = density
        self.atomdist = atomdist
        self.atom_partition = atom_partition
        self.setups = setups

    def initialize_more_things(self):
        raise NotImplementedError

    def calculate_paw_feat_corrections(self, *args, **kwargs):
        raise NotImplementedError


class PAWCiderContribUtils:
    # Note: Abstract class, contains functions for
    # computing PAW contributions

    def __init__(self, cider_kernel, nspin, encut, lambd, timer, Nalpha, cut_xcgrid):

        self.cider_kernel = cider_kernel
        self.nspin = nspin
        self.encut = encut
        self.lambd = lambd
        self.timer = timer
        self.Nalpha = Nalpha
        self.cut_xcgrid = cut_xcgrid

        self.verbose = False
        nldf_settings = cider_kernel.mlfunc.settings.nldf_settings
        if nldf_settings.is_empty:
            self._plan = None
        else:
            self._plan = NLDFSplinePlan(
                nldf_settings,
                self.nspin,
                self.encut / self.lambd ** (self.Nalpha - 1),
                self.lambd,
                self.Nalpha,
                coef_order="qg",
                alpha_formula="etb",
            )

    # get the y values
    def calc_y_sbLk(self, dx_sLkb, ks=None):
        # dx_sqkL = (nspin, Nq, Nk, (L+1)^2)
        # needs to be separate for CIDER and VDW
        if ks is None:
            ks = self.krad
        y_bsLk = np.zeros(np.array(dx_sLkb.shape)[[3, 0, 1, 2]])
        for b in range(y_bsLk.shape[0]):
            aexp = self._plan.alphas[b] + self._plan.alphas
            fac = (np.pi / aexp) ** 1.5
            y_bsLk[b, :, :, :] += np.einsum(
                "kq,sLkq->sLk",
                np.exp(-ks[:, None] ** 2 / (4 * aexp)) * fac,
                dx_sLkb,
            )
        return y_bsLk.transpose(1, 0, 2, 3)

    def get_interpolation_coefficients(self, arg_g, i=-1):
        p_qg, dp_qg = self._plan.get_interpolation_coefficients(arg_g, i=i)
        p_qg[:] *= self._plan.alpha_norms[:, None]
        dp_qg[:] *= self._plan.alpha_norms[:, None]
        if 0 <= i < self._plan.num_vj:
            p_qg[:] *= self._plan.nspin
            dp_qg[:] *= self._plan.nspin
        return p_qg, dp_qg

    def get_paw_atom_contribs(self, n_sg, sigma_xg, tau_sg=None, ae=True):
        nspin = n_sg.shape[0]
        x_sag = np.zeros((nspin, self.Nalpha, n_sg.shape[-1]))
        xd_sag = np.zeros((nspin, self.Nalpha, n_sg.shape[-1]))
        if self._plan.nldf_settings.sl_level == "MGGA":
            rho_tuple = (n_sg, sigma_xg[::2], tau_sg)
        else:
            rho_tuple = (n_sg, sigma_xg[::2])
        di_s = self._plan.get_interpolation_arguments(rho_tuple, i=-1)[0]
        # TODO need to account for dfunc
        func_sg, dfunc = self._plan.get_function_to_convolve(rho_tuple)
        for s in range(nspin):
            p_qg, dp_qg = self.get_interpolation_coefficients(di_s[s].ravel(), i=-1)
            for a in range(self.Nalpha):
                x_sag[s, a] = p_qg[a] * func_sg[s]
                xd_sag[s, a] = dp_qg[a] * func_sg[s]
        return x_sag, xd_sag

    def get_paw_atom_contribs_en(
        self, n_sg, sigma_xg, y_sbg, F_sag, tau_sg=None, ae=True
    ):
        self.timer.start("get CIDER attr")
        # for getting potential and energy
        nspin = n_sg.shape[0]
        nfeat = self._plan.nldf_settings.nfeat
        ngrid = n_sg.shape[-1]
        if ae:
            y_sbg = y_sbg.copy()
            y_sbg[:] += F_sag
        else:
            y_sbg = F_sag
        Nalpha = self.Nalpha

        is_mgga = self._plan.nldf_settings.sl_level == "MGGA"
        if is_mgga:
            assert tau_sg is not None
            rho_tuple = (n_sg, sigma_xg[::2], tau_sg)
        else:
            assert tau_sg is None
            rho_tuple = (n_sg, sigma_xg[::2])

        x_sig = np.zeros((nspin, nfeat, ngrid))
        xd_sig = np.zeros((nspin, nfeat, ngrid))
        p_siag = np.zeros((nspin, nfeat, Nalpha, ngrid))
        dgdn_sig = np.zeros((nspin, nfeat, ngrid))
        dgdsigma_sig = np.zeros_like(dgdn_sig)
        if is_mgga:
            dgdtau_sig = np.zeros_like(dgdn_sig)
        else:
            dgdtau_sig = None

        for i in range(nfeat):
            di_s, derivs = self._plan.get_interpolation_arguments(rho_tuple, i=i)
            for s in range(nspin):
                p_qg, dp_qg = self.get_interpolation_coefficients(di_s[s].ravel(), i=i)
                for a in range(self.Nalpha):
                    x_sig[s, i] += p_qg[a] * y_sbg[s, a]
                    xd_sig[s, i] += dp_qg[a] * y_sbg[s, a]
                    p_siag[s, i, a] = p_qg[a]
                dgdn_sig[s, i] = xd_sig[s, i] * derivs[0][s]
                dgdsigma_sig[s, i] = xd_sig[s, i] * derivs[1][s]
                if is_mgga:
                    dgdtau_sig[s, i] = xd_sig[s, i] * derivs[2][s]
        self.timer.stop()
        if is_mgga:
            return x_sig, dgdn_sig, dgdsigma_sig, dgdtau_sig, p_siag
        else:
            return x_sig, dgdn_sig, dgdsigma_sig, p_siag

    def get_paw_atom_feat(self, n_sg, sigma_xg, y_sbg, F_sbg, ae, include_density):
        feat_sig = self.get_paw_atom_contribs_en(n_sg, sigma_xg, y_sbg, F_sbg, ae=ae)[0]
        if include_density:
            from ciderpress.gpaw.atom_descriptor_utils import (
                get_features_with_sl_noderiv,
            )

            feat_sig = get_features_with_sl_noderiv(feat_sig, n_sg, sigma_xg, None)
            # _feat_sig = feat_sig
            # feat_sig = np.zeros((feat_sig.shape[0], feat_sig.shape[1]+2) + feat_sig.shape[2:])
            # feat_sig[:,0,:] = n_sg
            # feat_sig[:,1,:] = sigma_xg[::2,:]
            # feat_sig[:,2:,:] = _feat_sig
        return feat_sig

    def get_paw_atom_feat_v2(self, n_sg, grad_svg, y_sbg, F_sbg, ae, include_density):
        rho_sxg = np.append(n_sg[:, None, :], grad_svg, axis=1)
        feat_sig = self.get_paw_atom_contribs_en_v2(
            None, rho_sxg, None, y_sbg, F_sbg, ae=ae, feat_only=True
        )
        if include_density:
            from ciderpress.gpaw.atom_descriptor_utils import (
                get_features_with_sl_noderiv,
            )

            sigma_xg = get_sigma(grad_svg)
            feat_sig = get_features_with_sl_noderiv(feat_sig, n_sg, sigma_xg, None)
        return feat_sig

    def get_paw_atom_evxc(
        self, e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, y_sbg, F_sbg, a, ae
    ):
        feat_sig, dgdn_sig, dgdsigma_sig, p_siag = self.get_paw_atom_contribs_en(
            n_sg, sigma_xg, y_sbg, F_sbg, ae=ae
        )
        nspin = feat_sig.shape[0]
        vfeat_sig = self.cider_kernel.calculate(
            e_g,
            n_sg,
            dedn_sg,
            sigma_xg,
            dedsigma_xg,
            feat_sig,
        )

        v_sag = np.einsum("sig,siag->sag", vfeat_sig, p_siag)
        vfeat_rho_sg = np.einsum("sig,sig->sg", vfeat_sig, dgdn_sig)
        vfeat_sigma_sg = np.einsum("sig,sig->sg", vfeat_sig, dgdsigma_sig)
        for s in range(nspin):
            dedn_sg[s] += vfeat_rho_sg[s]
            dedsigma_xg[2 * s] += vfeat_sigma_sg[s]

        return v_sag

    def get_paw_atom_contribs_en_v2(
        self, e_g, rho_sxg, vrho_sxg, y_sbg, F_sbg, ae, feat_only=False
    ):
        if ae:
            y_sbg = y_sbg.copy()
            y_sbg[:] += F_sbg
        else:
            y_sbg = F_sbg
        nspin = rho_sxg.shape[0]
        ngrid = rho_sxg.shape[2]
        feat_sig = np.zeros((nspin, self._plan.nldf_settings.nfeat, ngrid))
        dfeat_sig = np.zeros((nspin, self._plan.num_vj, ngrid))
        for s in range(nspin):
            self._plan.eval_rho_full(
                y_sbg[s],
                rho_sxg[s],
                spin=s,
                feat=feat_sig[s],
                dfeat=dfeat_sig[s],
                cache_p=True,
                # TODO might need to be True for gaussian interp
                apply_transformation=False,
                coeff_multipliers=self._plan.alpha_norms,
            )
        if feat_only:
            return feat_sig
        sigma_xg = get_sigma(rho_sxg[:, 1:4])
        dedsigma_xg = np.zeros_like(sigma_xg)
        nspin = feat_sig.shape[0]
        dedn_sg = np.zeros_like(vrho_sxg[:, 0])
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
        v_sbg = np.zeros((nspin, self._plan.nalpha, ngrid))
        for s in range(nspin):
            v_sbg[s] = self._plan.eval_vxc_full(
                vfeat_sig[s], vrho_sxg[s], dfeat_sig[s], rho_sxg[s], spin=s
            )

        return v_sbg

    def get_paw_atom_evxc_v2(
        self, e_g, n_sg, dedn_sg, grad_svg, dedgrad_svg, y_sbg, F_sbg, a, ae
    ):
        rho_sxg = np.append(n_sg[:, None, :], grad_svg, axis=1)
        vrho_sxg = np.zeros_like(rho_sxg)
        v_sbg = self.get_paw_atom_contribs_en_v2(
            e_g, rho_sxg, vrho_sxg, y_sbg, F_sbg, ae
        )
        dedn_sg[:] += vrho_sxg[:, 0]
        dedgrad_svg[:] += vrho_sxg[:, 1:4]
        return v_sbg

    def get_paw_atom_contribs_pot(
        self, n_sg, sigma_xg, y_sbg, F_sag, ae=True, tau_sg=None
    ):
        # for getting potential and energy
        nspin = n_sg.shape[0]
        ngrid = n_sg.shape[-1]
        if ae:
            y_sbg = y_sbg.copy()
            y_sbg[:] += F_sag
        else:
            y_sbg = F_sag

        is_mgga = self._plan.nldf_settings.sl_level == "MGGA"
        if is_mgga:
            rho_tuple = (n_sg, sigma_xg[::2], tau_sg)
        else:
            rho_tuple = (n_sg, sigma_xg[::2])
        di_s, derivs = self._plan.get_interpolation_arguments(rho_tuple, i=-1)
        # TODO need to account for dfunc
        func_sg, dfunc = self._plan.get_function_to_convolve(rho_tuple)
        dedn_sg = np.zeros((nspin, ngrid))
        dedsigma_sg = np.zeros_like(dedn_sg)
        if is_mgga:
            dedtau_sg = np.zeros_like(dedn_sg)
        else:
            dedtau_sg = None
        for s in range(nspin):
            p_qg, dp_qg = self.get_interpolation_coefficients(di_s[s].ravel(), i=-1)
            for a in range(self.Nalpha):
                dedn_sg[s] += p_qg[a] * y_sbg[s, a]
                dedsigma_sg[s] += dp_qg[a] * y_sbg[s, a]
                if is_mgga:
                    dedtau_sg[s] += dp_qg[a] * y_sbg[s, a]
            dedn_sg[s] += dedsigma_sg[s] * n_sg[s] * derivs[0][s]
            dedsigma_sg[s] *= n_sg[s] * derivs[1][s]
            if is_mgga:
                dedtau_sg[s] *= n_sg[s] * derivs[2][s]

        if is_mgga:
            return dedn_sg, dedsigma_sg, dedtau_sg
        else:
            return dedn_sg, dedsigma_sg

    def get_paw_atom_cider_pot(
        self, n_sg, dedn_sg, sigma_xg, dedsigma_xg, y_sbg, F_sbg, ae
    ):
        vrho_sg, vsigma_sg = self.get_paw_atom_contribs_pot(
            n_sg, sigma_xg, y_sbg, F_sbg, ae=ae
        )
        nspin = self.nspin
        for s in range(nspin):
            dedn_sg[s] += vrho_sg[s]
            dedsigma_xg[2 * s] += vsigma_sg[s]

    def set_D_sp(self, D_sp, xcc):
        self._dH_sp = np.zeros_like(D_sp)

    def get_dH_sp(self):
        return self._dH_sp


class MetaPAWCiderContribUtils(PAWCiderContribUtils):
    def __init__(self, *args):
        super(MetaPAWCiderContribUtils, self).__init__(*args)
        self._D_sp = None
        self._dH_sp = None
        self._xcc = None

    def set_D_sp(self, D_sp, xcc):
        self._D_sp = D_sp.copy()
        self._dH_sp = np.zeros_like(D_sp)
        self._xcc = xcc

    def get_kinetic_energy(self, ae):
        nspins = self._D_sp.shape[0]
        xcc = self._xcc
        if ae:
            tau_pg = xcc.tau_pg
            tauc_g = xcc.tauc_g / (np.sqrt(4 * np.pi) * nspins)
        else:
            tau_pg = xcc.taut_pg
            tauc_g = xcc.tauct_g / (np.sqrt(4 * np.pi) * nspins)
        nn = tau_pg.shape[-1] // tauc_g.shape[0]
        tau_sg = np.dot(self._D_sp, tau_pg) + np.tile(tauc_g, nn)
        return tau_sg, np.zeros_like(tau_sg)

    def contract_kinetic_potential(self, dedtau_sg, ae):
        xcc = self._xcc
        if ae:
            tau_pg = xcc.tau_pg
            sign = 1.0
        else:
            tau_pg = xcc.taut_pg
            sign = -1.0
        wt = weight_n[:, None] * xcc.rgd.dv_g
        wt = wt.ravel() * sign
        self._dH_sp += np.dot(dedtau_sg * wt, tau_pg.T)

    def get_paw_atom_contribs(self, n_sg, sigma_xg, ae=True):
        tau_sg, dedtau_sg = self.get_kinetic_energy(ae)
        return super(MetaPAWCiderContribUtils, self).get_paw_atom_contribs(
            n_sg, sigma_xg, tau_sg=tau_sg, ae=ae
        )

    def get_paw_atom_feat(self, n_sg, sigma_xg, y_sbg, F_sbg, ae, include_density):
        tau_sg, dedtau_sg = self.get_kinetic_energy(ae)

        feat_sig = self.get_paw_atom_contribs_en(
            n_sg, sigma_xg, y_sbg, F_sbg, ae=ae, tau_sg=tau_sg
        )[0]
        if include_density:
            from ciderpress.gpaw.atom_descriptor_utils import (
                get_features_with_sl_noderiv,
            )

            feat_sig = get_features_with_sl_noderiv(feat_sig, n_sg, sigma_xg, tau_sg)
            # _feat_sig = feat_sig
            # feat_sig = np.zeros((feat_sig.shape[0], feat_sig.shape[1]+3) + feat_sig.shape[2:])
            # feat_sig[:,0,:] = n_sg
            # feat_sig[:,1,:] = sigma_xg[::2,:]
            # feat_sig[:,2,:] = tau_sg
            # feat_sig[:,3:,:] = _feat_sig
        return feat_sig

    def get_paw_atom_feat_v2(self, n_sg, grad_svg, y_sbg, F_sbg, ae, include_density):
        tau_sg, dedtau_sg = self.get_kinetic_energy(ae)
        rho_sxg = np.concatenate(
            [n_sg[:, None, :], grad_svg, tau_sg[:, None, :]], axis=1
        )
        feat_sig = self.get_paw_atom_contribs_en_v2(
            None, rho_sxg, None, y_sbg, F_sbg, ae=ae, feat_only=True
        )
        if include_density:
            from ciderpress.gpaw.atom_descriptor_utils import (
                get_features_with_sl_noderiv,
            )

            sigma_xg = get_sigma(grad_svg)
            feat_sig = get_features_with_sl_noderiv(feat_sig, n_sg, sigma_xg, tau_sg)
        return feat_sig

    def get_paw_atom_evxc(
        self, e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, y_sbg, F_sbg, a, ae
    ):
        tau_sg, dedtau_sg = self.get_kinetic_energy(ae)

        (
            feat_sig,
            dgdn_sig,
            dgdsigma_sig,
            dgdtau_sig,
            p_siag,
        ) = self.get_paw_atom_contribs_en(
            n_sg, sigma_xg, y_sbg, F_sbg, ae=ae, tau_sg=tau_sg
        )
        nspin = feat_sig.shape[0]
        vfeat_sig = self.cider_kernel.calculate(
            e_g,
            n_sg,
            dedn_sg,
            sigma_xg,
            dedsigma_xg,
            tau_sg,
            dedtau_sg,
            feat_sig,
        )

        v_sag = np.einsum("sig,siag->sag", vfeat_sig, p_siag)
        vfeat_rho_sg = np.einsum("sig,sig->sg", vfeat_sig, dgdn_sig)
        vfeat_sigma_sg = np.einsum("sig,sig->sg", vfeat_sig, dgdsigma_sig)
        vfeat_tau_sg = np.einsum("sig,sig->sg", vfeat_sig, dgdtau_sig)
        for s in range(nspin):
            dedn_sg[s] += vfeat_rho_sg[s]
            dedsigma_xg[2 * s] += vfeat_sigma_sg[s]
            dedtau_sg[s] += vfeat_tau_sg[s]

        self.contract_kinetic_potential(dedtau_sg, ae)

        return v_sag

    def get_paw_atom_evxc_v2(
        self, e_g, n_sg, dedn_sg, grad_svg, dedgrad_svg, y_sbg, F_sbg, a, ae
    ):
        tau_sg, dedtau_sg = self.get_kinetic_energy(ae)
        rho_sxg = np.concatenate(
            [n_sg[:, None, :], grad_svg, tau_sg[:, None, :]], axis=1
        )
        vrho_sxg = np.zeros_like(rho_sxg)
        v_sbg = self.get_paw_atom_contribs_en_v2(
            e_g, rho_sxg, vrho_sxg, y_sbg, F_sbg, ae
        )
        dedn_sg[:] += vrho_sxg[:, 0]
        dedgrad_svg[:] += vrho_sxg[:, 1:4]
        dedtau_sg[:] += vrho_sxg[:, 4]

        self.contract_kinetic_potential(dedtau_sg, ae)

        return v_sbg

    def get_paw_atom_cider_pot(
        self, n_sg, dedn_sg, sigma_xg, dedsigma_xg, y_sbg, F_sbg, ae
    ):
        tau_sg, dedtau_sg = self.get_kinetic_energy(ae)

        vrho_sg, vsigma_sg, vtau_sg = self.get_paw_atom_contribs_pot(
            n_sg, sigma_xg, y_sbg, F_sbg, ae=ae, tau_sg=tau_sg
        )
        nspin = self.nspin
        for s in range(nspin):
            dedn_sg[s] += vrho_sg[s]
            dedsigma_xg[2 * s] += vsigma_sg[s]
            dedtau_sg[s] += vtau_sg[s]

        self.contract_kinetic_potential(dedtau_sg, ae)


class BasePAWCiderKernel(PAWCiderKernelShell, PAWCiderContribUtils):
    def __init__(
        self,
        cider_kernel,
        Nalpha,
        lambd,
        encut,
        world,
        timer,
        Nalpha_small,
        cut_xcgrid,
        **kwargs
    ):

        if world is None:
            self.world = mpi.world
        else:
            self.world = world

        self.Nalpha_small = Nalpha_small
        self.Nalpha = Nalpha
        self.lambd = lambd
        self.encut = encut

        self.cider_kernel = cider_kernel

        self.verbose = False
        self.timer = timer

        self.cut_xcgrid = cut_xcgrid
