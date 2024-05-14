import warnings

import numpy as np

warnings.warn(
    "This plotting module is experimental and for debugging/developer purpose only"
)

# TODO only works for spin-unpolarized systems
def get_features(functional, a=0, ispin=0):
    all_feat = functional.get_features_on_grid(True)
    all_feat = all_feat[ispin]
    (
        check_feat,
        check_feat2,
        sphfeat,
        sphfeat2,
    ) = functional.calculate_paw_feat_corrections_test(True, False, a=a)
    ps_feat = functional.get_features_on_grid(False)
    ps_feat = ps_feat[ispin]
    return all_feat, ps_feat, check_feat, check_feat2, sphfeat, sphfeat2


def get_features_new(functional, a=0, ispin=0, get_pot=False):
    all_feat = functional.get_features_on_grid(True)
    all_feat = all_feat[ispin]
    ps_feat = functional.get_features_on_grid(False)
    ps_feat = ps_feat[ispin]
    check_feat, check_feat2 = functional.calculate_paw_feat_corrections_test(
        True, False, a=a
    )
    if get_pot:
        raise NotImplemented
        return (
            all_feat,
            ps_feat,
            check_feat,
            check_feat2,
            functional.v_sg,
            functional.vfeat[ispin],
        )
    else:
        return all_feat, ps_feat, check_feat, check_feat2


def get_density(functional):
    return functional.dens.nt_sg


def get_density_on_atom(functional, a=0):
    nt_sg = functional.dens.nt_sg
    n1_sLg, nt1_sLg, n1_sgr, nt1_sgr = functional.calculate_paw_feat_corrections_test(
        True, True, a=a
    )
    return n1_sLg, nt1_sLg, n1_sgr, nt1_sgr, nt_sg


def get_plot_data(functional, all_feat, ps_feat, nt_sg, ispin=0):
    fx, fy, fz = [np.arange(s) / s for s in functional.shape]
    print(functional.shape)
    cx, cy, cz = np.meshgrid(fx, fy, fz, indexing="ij")
    fcoords = np.stack([cx, cy, cz])
    cell_cv = functional.dens.gd.cell_cv
    shape = np.array(functional.shape)
    fcoords_c = fcoords.copy()
    for i in range(3):
        fcoords_c[i][fcoords_c[i] > 0.5] -= 1

    def get_cart_vecs(fcoords_c, center=None):
        assert fcoords_c.shape[0] == 3
        if center is not None:
            fcoords_c = fcoords_c - center
        return fcoords_c.reshape(3, -1).T.dot(cell_cv).T.reshape(3, *functional.shape)

    vecs = get_cart_vecs(fcoords_c)
    print(vecs.shape)
    rs = np.linalg.norm(vecs, axis=0)
    RMAX = 4
    which_rs = rs < RMAX
    data = []
    dens_data = []
    for pos in functional.spos_ac:
        print("pos", pos)
        for feat in np.append(all_feat, ps_feat, axis=0):
            center = np.round(pos * shape).astype(int)
            diff = pos - center / shape
            print(which_rs.shape, center.shape, center)
            which_rs_a = np.roll(which_rs, center, axis=(0, 1, 2))
            assert (
                which_rs_a[center[0], center[1], center[2]] == which_rs[0, 0, 0]
            ).all()
            fcoords_c_a = np.roll(
                fcoords_c.transpose(1, 2, 3, 0), center, axis=(0, 1, 2)
            )
            assert (fcoords_c_a[center[0], center[1], center[2]] == np.zeros(3)).all()
            fcoords_c_a -= diff
            vecs_a = get_cart_vecs(fcoords_c_a.transpose(3, 0, 1, 2))
            rs_a = np.linalg.norm(vecs_a, axis=0)
            data.append((rs_a[which_rs_a], feat[which_rs_a]))
        dens_data.append((rs_a[which_rs_a], nt_sg[ispin][which_rs_a]))
    print(cx.shape, cy.shape, cz.shape)
    return data, dens_data


def plot_thing():
    functional.dens.setups[0].xc_correction.rgd
    functional.dens.setups[0].nlxc_correction.rgd
    n_qg_int = functional.dens.setups[0].xc_correction.n_qg
    n_qg_exp = functional.dens.setups[0].nlxc_correction.n_qg
    for q, n_g_int in enumerate(n_qg_int):
        n_g_exp = n_qg_exp[q]
        plt.plot(r_int, n_g_int, label="{} int".format(q))
        # plt.plot(r_int, rgd_int.derivative(n_g_int), label='{} int'.format(q))
        plt.plot(r_exp, n_g_exp, label="{} exp".format(q))
        # plt.plot(r_exp, rgd_exp.derivative(n_g_exp), label='{} exp'.format(q))
    plt.plot([r_int[1], r_int[1]], [-20, 120])
    plt.xlim(0, 0.005)  # np.max(r_exp))
    plt.legend()
    plt.savefig("check_dens.png")
    plt.clf()
