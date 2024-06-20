import joblib
import yaml

from ciderpress.dft.xc_evaluator import MappedXC


def load_cider_model(mlfunc, mlfunc_format):
    if isinstance(mlfunc, str):
        if mlfunc_format is None:
            if mlfunc.endswith(".yaml"):
                mlfunc_format = "yaml"
            elif mlfunc.endswith(".joblib"):
                mlfunc_format = "joblib"
            else:
                raise ValueError("Unsupported file format")
        if mlfunc_format == "yaml":
            with open(mlfunc, "r") as f:
                mlfunc = yaml.load(f, Loader=yaml.CLoader)
        elif mlfunc_format == "joblib":
            mlfunc = joblib.load(mlfunc)
        else:
            raise ValueError("Unsupported file format")
    if not isinstance(mlfunc, MappedXC):
        raise ValueError("mlfunc must be MappedXC")
    return mlfunc


def get_slxc_settings(xc, xkernel, ckernel, xmix, cider_x_only=True):
    if xc is None:
        # xc is another way to specify non-mixed part of kernel
        xc = ""
    if ckernel is not None:
        if cider_x_only:
            xc = ckernel + " + " + xc
        else:
            xc = "{} * {} + {}".format(1 - xmix, ckernel, xc)
    if xkernel is not None:
        xc = "{} * {} + {}".format(1 - xmix, xkernel, xc)
    if xc.endswith(" + "):
        xc = xc[:-3]
    return xc
