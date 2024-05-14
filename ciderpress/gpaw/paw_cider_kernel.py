import numpy as np
from gpaw.sphere.lebedev import weight_n

from ciderpress.gpaw.cider_fft import (
    construct_cubic_splines,
    get_cider_coefs,
    get_cider_exp_gga,
    get_cider_exp_mgga,
)


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

    def __init__(
        self, cider_kernel, nexp, consts, encut, lambd, timer, Nalpha, cut_xcgrid
    ):

        self.cider_kernel = cider_kernel
        self.nexp = nexp
        self.consts = consts
        self.encut = encut
        self.lambd = lambd
        self.timer = timer
        self.Nalpha = Nalpha
        self.cut_xcgrid = cut_xcgrid

        self.C_aip = None
        self.verbose = False
        self.get_alphas()
        self.construct_cubic_splines()

    get_cider_coefs = get_cider_coefs

    construct_cubic_splines = construct_cubic_splines

    get_cider_exp = get_cider_exp_gga

    def get_alphas(self):
        self.bas_exp = self.encut / self.lambd ** np.arange(self.Nalpha)
        self.bas_exp = np.flip(self.bas_exp)
        self.maxen = 2 * self.bas_exp[-1]
        self.minen = 2 * self.bas_exp[0]
        self.alphas = np.arange(self.Nalpha)
        sinv = (4 * self.bas_exp[:, None] * self.bas_exp) ** 0.75 * (
            self.bas_exp[:, None] + self.bas_exp
        ) ** -1.5
        from scipy.linalg import cho_factor

        self.alpha_cl = cho_factor(sinv)
        self.sinv = np.linalg.inv(sinv)

    # get the y values
    def calc_y_sbLk(self, dx_sLkb, ks=None):
        # dx_sqkL = (nspin, Nq, Nk, (L+1)^2)
        # needs to be separate for CIDER and VDW
        if ks is None:
            ks = self.krad
        y_bsLk = np.zeros(np.array(dx_sLkb.shape)[[3, 0, 1, 2]])
        for b in range(y_bsLk.shape[0]):
            aexp = self.bas_exp[b] + self.bas_exp
            y_bsLk[b, :, :, :] += np.einsum(
                "kq,sLkq->sLk",
                np.exp(-ks[:, None] ** 2 / (4 * aexp)) * (np.pi / aexp) ** 1.5,
                dx_sLkb,
            )
        return y_bsLk.transpose(1, 0, 2, 3)

    def get_q_func(self, n_g, a2g, i=-1, return_derivs=False):
        return self.get_cider_exp(
            n_g,
            a2g,
            a0=self.consts[i, 1],
            fac_mul=self.consts[i, 2],
            amin=self.consts[i, 3],
            return_derivs=return_derivs,
        )

    def get_paw_atom_contribs(self, n_sg, sigma_xg, ae=True):
        nspin = n_sg.shape[0]
        x_sag = np.zeros((nspin, self.Nalpha, n_sg.shape[-1]))
        xd_sag = np.zeros((nspin, self.Nalpha, n_sg.shape[-1]))
        for s in range(nspin):
            # TODO factor of half for CIDER, would be ideal to remove the weird
            # factor of half issue for convenience
            # TODO if factor of 0.5 is removed and function to compute i_g is
            # added, then get_paw_atom_contribs and get_paw_atom_contribs_cider
            # will be the same and can be combined
            q0_g = 0.5 * self.get_q_func(n_sg[s], sigma_xg[2 * s])
            i_g = (
                np.log(q0_g / self.dense_bas_exp[0]) / np.log(self.dense_lambd)
            ).astype(int)
            dq0_g = q0_g - self.q_a[i_g]
            # TODO need more coefs for core region?
            for a in range(self.Nalpha):
                C_pg = self.C_aip[a, i_g].T
                pa_g = C_pg[0] + dq0_g * (C_pg[1] + dq0_g * (C_pg[2] + dq0_g * C_pg[3]))
                dpa_g = 0.5 * (C_pg[1] + dq0_g * (2 * C_pg[2] + dq0_g * 3 * C_pg[3]))
                x_sag[s, a] = pa_g * n_sg[s]
                xd_sag[s, a] = dpa_g * n_sg[s]
        return x_sag, xd_sag

    def get_paw_atom_contribs_en(self, n_sg, sigma_xg, y_sbg, F_sag, ae=True):
        self.timer.start("get CIDER attr")
        # for getting potential and energy
        nspin = n_sg.shape[0]
        nfeat = self.nexp - 1
        ngrid = n_sg.shape[-1]
        if ae:
            y_sbg = y_sbg.copy()
            y_sbg[:] += F_sag
        else:
            y_sbg = F_sag
        Nalpha = self.Nalpha

        x_sig = np.zeros((nspin, nfeat, ngrid))
        xd_sig = np.zeros((nspin, nfeat, ngrid))
        p_siag = np.zeros((nspin, nfeat, Nalpha, ngrid))
        dgdn_sig = np.zeros((nspin, nfeat, ngrid))
        dgdsigma_sig = np.zeros_like(dgdn_sig)
        for s in range(nspin):
            for i in range(nfeat):
                # TODO factor of half for CIDER, would be ideal to remove the weird
                # factor of half issue for convenience
                # TODO if factor of 0.5 is removed and function to compute i_g is
                # added, then get_paw_atom_contribs and get_paw_atom_contribs_cider
                # will be the same and can be combined
                q0_g, dadn, dadsigma = self.get_q_func(
                    n_sg[s], sigma_xg[2 * s], i=i, return_derivs=True
                )
                q0_g *= 0.5
                i_g = (
                    np.log(q0_g / self.dense_bas_exp[0]) / np.log(self.dense_lambd)
                ).astype(int)
                dq0_g = q0_g - self.q_a[i_g]
                for a in range(Nalpha):
                    C_pg = self.C_aip[a, i_g].T
                    pa_g = C_pg[0] + dq0_g * (
                        C_pg[1] + dq0_g * (C_pg[2] + dq0_g * C_pg[3])
                    )
                    dpa_g = 0.5 * (
                        C_pg[1] + dq0_g * (2 * C_pg[2] + dq0_g * 3 * C_pg[3])
                    )
                    x_sig[s, i] += pa_g * y_sbg[s, a]
                    xd_sig[s, i] += dpa_g * y_sbg[s, a]
                    p_siag[s, i, a] = pa_g
                x_sig[s, i] *= ((self.consts[i, 1] + self.consts[-1, 1]) / 2) ** 1.5
                xd_sig[s, i] *= ((self.consts[i, 1] + self.consts[-1, 1]) / 2) ** 1.5
                p_siag[s, i] *= ((self.consts[i, 1] + self.consts[-1, 1]) / 2) ** 1.5
                dgdn_sig[s, i] = xd_sig[s, i] * dadn
                dgdsigma_sig[s, i] = xd_sig[s, i] * dadsigma

        self.timer.stop()
        return x_sig, dgdn_sig, dgdsigma_sig, p_siag

    def get_paw_atom_feat(self, n_sg, sigma_xg, y_sbg, F_sbg, ae, include_density):
        feat_sig, dgdn_sig, dgdsigma_sig, p_siag = self.get_paw_atom_contribs_en(
            n_sg, sigma_xg, y_sbg, F_sbg, ae=ae
        )
        if include_density:
            from ciderpress.gpaw.atom_analysis_utils import get_features_with_sl_noderiv

            feat_sig = get_features_with_sl_noderiv(feat_sig, n_sg, sigma_xg, None)
            # _feat_sig = feat_sig
            # feat_sig = np.zeros((feat_sig.shape[0], feat_sig.shape[1]+2) + feat_sig.shape[2:])
            # feat_sig[:,0,:] = n_sg
            # feat_sig[:,1,:] = sigma_xg[::2,:]
            # feat_sig[:,2:,:] = _feat_sig
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

    def get_paw_atom_contribs_pot(self, n_sg, sigma_xg, y_sbg, F_sag, ae=True):
        # for getting potential and energy
        nspin = n_sg.shape[0]
        self.nexp - 1
        ngrid = n_sg.shape[-1]
        if ae:
            y_sbg = y_sbg.copy()
            y_sbg[:] += F_sag
        else:
            y_sbg = F_sag
        Nalpha = self.Nalpha

        dedn_sg = np.zeros((nspin, ngrid))
        dedsigma_sg = np.zeros_like(dedn_sg)
        i = -1
        for s in range(nspin):
            q0_g, dadn, dadsigma = self.get_q_func(
                n_sg[s], sigma_xg[2 * s], i=i, return_derivs=True
            )
            q0_g *= 0.5
            i_g = (
                np.log(q0_g / self.dense_bas_exp[0]) / np.log(self.dense_lambd)
            ).astype(int)
            dq0_g = q0_g - self.q_a[i_g]
            for a in range(Nalpha):
                C_pg = self.C_aip[a, i_g].T
                pa_g = C_pg[0] + dq0_g * (C_pg[1] + dq0_g * (C_pg[2] + dq0_g * C_pg[3]))
                dpa_g = 0.5 * (C_pg[1] + dq0_g * (2 * C_pg[2] + dq0_g * 3 * C_pg[3]))
                dedn_sg[s] += pa_g * y_sbg[s, a]
                dedsigma_sg[s] += dpa_g * y_sbg[s, a]
            dedn_sg[s] += dedsigma_sg[s] * n_sg[s] * dadn
            dedsigma_sg[s] *= n_sg[s] * dadsigma

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

    get_cider_exp = get_cider_exp_mgga

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

    def get_q_func(self, n_g, tau_g, i=-1, return_derivs=False):
        sigma_g = np.zeros_like(n_g)
        res = self.get_cider_exp(
            n_g,
            sigma_g,
            tau_g,
            a0=self.consts[i, 1],
            fac_mul=self.consts[i, 2],
            amin=self.consts[i, 3],
            return_derivs=return_derivs,
        )
        if return_derivs:
            return res[0], res[1], res[3]
        else:
            return res

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
        ns, ng = n_sg.shape
        nx = 2 * ns + 1
        tau_sg, dedtau_sg = self.get_kinetic_energy(ae)
        tau_xg = np.zeros((nx, ng))
        for s in range(ns):
            tau_xg[2 * s] = tau_sg[s]
        return super(MetaPAWCiderContribUtils, self).get_paw_atom_contribs(
            n_sg, tau_xg, ae=ae
        )

    def get_paw_atom_feat(self, n_sg, sigma_xg, y_sbg, F_sbg, ae, include_density):
        ns, ng = n_sg.shape
        nx = 2 * ns + 1
        tau_sg, dedtau_sg = self.get_kinetic_energy(ae)
        tau_xg = np.zeros((nx, ng))
        for s in range(ns):
            tau_xg[2 * s] = tau_sg[s]

        feat_sig, dgdn_sig, dgdsigma_sig, p_siag = self.get_paw_atom_contribs_en(
            n_sg, tau_xg, y_sbg, F_sbg, ae=ae
        )
        if include_density:
            from ciderpress.gpaw.atom_analysis_utils import get_features_with_sl_noderiv

            feat_sig = get_features_with_sl_noderiv(feat_sig, n_sg, sigma_xg, tau_sg)
            # _feat_sig = feat_sig
            # feat_sig = np.zeros((feat_sig.shape[0], feat_sig.shape[1]+3) + feat_sig.shape[2:])
            # feat_sig[:,0,:] = n_sg
            # feat_sig[:,1,:] = sigma_xg[::2,:]
            # feat_sig[:,2,:] = tau_sg
            # feat_sig[:,3:,:] = _feat_sig
        return feat_sig

    def get_paw_atom_evxc(
        self, e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg, y_sbg, F_sbg, a, ae
    ):
        ns, ng = n_sg.shape
        nx = 2 * ns + 1
        tau_sg, dedtau_sg = self.get_kinetic_energy(ae)
        tau_xg = np.zeros((nx, ng))
        for s in range(ns):
            tau_xg[2 * s] = tau_sg[s]

        feat_sig, dgdn_sig, dgdtau_sig, p_siag = self.get_paw_atom_contribs_en(
            n_sg, tau_xg, y_sbg, F_sbg, ae=ae
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
        vfeat_tau_sg = np.einsum("sig,sig->sg", vfeat_sig, dgdtau_sig)
        for s in range(nspin):
            dedn_sg[s] += vfeat_rho_sg[s]
            dedtau_sg[s] += vfeat_tau_sg[s]

        self.contract_kinetic_potential(dedtau_sg, ae)

        return v_sag

    def get_paw_atom_cider_pot(
        self, n_sg, dedn_sg, sigma_xg, dedsigma_xg, y_sbg, F_sbg, ae
    ):
        ns, ng = n_sg.shape
        nx = 2 * ns + 1
        tau_sg, dedtau_sg = self.get_kinetic_energy(ae)
        tau_xg = np.zeros((nx, ng))
        for s in range(ns):
            tau_xg[2 * s] = tau_sg[s]

        vrho_sg, dedtau_sg = self.get_paw_atom_contribs_pot(
            n_sg, tau_xg, y_sbg, F_sbg, ae=ae
        )
        nspin = self.nspin
        for s in range(nspin):
            dedn_sg[s] += vrho_sg[s]

        self.contract_kinetic_potential(dedtau_sg, ae)


class BasePAWCiderKernel(PAWCiderKernelShell, PAWCiderContribUtils):
    def __init__(
        self,
        cider_kernel,
        nexp,
        consts,
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
        self.consts = consts
        self.nexp = nexp
        self.encut = encut

        self.cider_kernel = cider_kernel

        self.C_aip = None
        self.get_alphas()
        self.verbose = False
        self.timer = timer

        self.cut_xcgrid = cut_xcgrid
