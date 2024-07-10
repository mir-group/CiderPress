import numpy as np

from ciderpress.gpaw.atom_utils import AtomPASDWSlice


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
