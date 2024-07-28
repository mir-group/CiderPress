from ciderpress.dft.xc_evaluator import SplineSetEvaluator
from ciderpress.models.kernel_plans.kernel_tools import get_agpr_kernel
from ciderpress.models.kernel_plans.map_tools import get_mapped_gp_evaluator_additive


def get_kernel(
    natural_scale=None,
    natural_lscale=None,
    scale_factor=None,
    lscale_factor=None,
    **kwargs,
):
    # leave out density from feature vector
    slice(1, None, None)
    print(lscale_factor, natural_lscale)
    print(scale_factor, natural_scale)
    return get_agpr_kernel(
        slice(0, 1, None),
        slice(1, None, None),
        lscale_factor * natural_lscale,
        scale=[1e-5, 1e-5, scale_factor * natural_scale],
        order=2,
        nsingle=1,
    )


def mapping_plan(dft_kernel):
    scale, ind_sets, spline_grids, coeff_sets = get_mapped_gp_evaluator_additive(
        dft_kernel.kernel,
        dft_kernel.X1ctrl,
        dft_kernel.alpha,
        dft_kernel.feature_list,
    )
    # TODO this needs to get converted into a class
    # that evaluates a specific kernel's contribution to the
    # XC energy. This function should then be a component
    # called by DFTKernel.map().
    # Then these mapping plans need to be combined within some
    # overarching evaluator class, which should be created
    # by a function like MOLGP.map().
    return SplineSetEvaluator(scale, ind_sets, spline_grids, coeff_sets)
