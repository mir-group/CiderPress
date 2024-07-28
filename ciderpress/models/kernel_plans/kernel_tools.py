from ciderpress.models.kernels import DiffConstantKernel, SubsetARBF, SubsetRBF


def get_rbf_kernel(
    indexes, length_scale, scale=1.0, opt_hparams=False, min_lscale=None
):
    if min_lscale is None:
        min_lscale = 0.01
    length_scale_bounds = (min_lscale, 10) if opt_hparams else "fixed"
    scale_bounds = (1e-5, 1e3) if opt_hparams else "fixed"
    return DiffConstantKernel(scale, constant_value_bounds=scale_bounds) * SubsetRBF(
        indexes,
        length_scale=length_scale[indexes],
        length_scale_bounds=length_scale_bounds,
    )


def get_agpr_kernel(
    sinds,
    ainds,
    length_scale,
    scale=None,
    order=2,
    nsingle=1,
    opt_hparams=False,
    min_lscale=None,
):
    print(sinds, ainds, length_scale[sinds], length_scale[ainds])
    if min_lscale is None:
        min_lscale = 0.01
    length_scale_bounds = (min_lscale, 10) if opt_hparams else "fixed"
    scale_bounds = (1e-5, 1e5) if opt_hparams else "fixed"
    if scale is None:
        scale = [1.0] * (order + 1)
    if nsingle == 0:
        singles = None
    else:
        singles = SubsetRBF(
            sinds,
            length_scale=length_scale[sinds],
            length_scale_bounds=length_scale_bounds,
        )
    cov_kernel = SubsetARBF(
        ainds,
        order=order,
        length_scale=length_scale[ainds],
        scale=scale,
        length_scale_bounds=length_scale_bounds,
        scale_bounds=scale_bounds,
    )
    if singles is None:
        return cov_kernel
    else:
        return singles * cov_kernel
