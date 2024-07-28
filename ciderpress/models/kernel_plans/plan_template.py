def get_kernel(
    natural_scale=None,
    natural_lscale=None,
    scale_factor=None,
    lscale_factor=None,
):
    """
    Args:
        natural_scale (float, None): Natural scale for covariance,
            typically set based on variance of target over training data.
        natural_lscale (array, None): Natural length scale for features,
            typically set based on variance of features over the
            training data.
        scale_factor (float, None): Multiplicative factor to tune
            covariance scale
        lscale_factor (float or array, None): Multiplicative factor
            to tune length scale.

    Returns:
        An sklearn kernel
    """


def mapping_plan(model):
    """
    A function that maps a model based on the kernel above to 'fast'
    functions like splines and polynomial evaluations, then returns
    an Evaluator object based on the mapping.

    Args:
        model:

    Returns:

    """
