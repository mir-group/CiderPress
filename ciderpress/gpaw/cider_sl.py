import joblib
import numpy as np
import yaml

from ciderpress.gpaw.cider_kernel import CiderGGAHybridKernel, CiderMGGAHybridKernel
from ciderpress.gpaw.interp_paw import DiffGGA, DiffMGGA


class SLCiderGGAHybridWrapper(CiderGGAHybridKernel):
    def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
        super(SLCiderGGAHybridWrapper, self).calculate(
            e_g,
            n_sg,
            v_sg,
            sigma_xg,
            dedsigma_xg,
            np.zeros(
                (
                    n_sg.shape[0],
                    3,
                )
                + e_g.shape,
                dtype=e_g.dtype,
            ),
        )


class SLCiderMGGAHybridWrapper(CiderMGGAHybridKernel):
    def calculate(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg, tau_sg, dedtau_sg):
        super(SLCiderMGGAHybridWrapper, self).calculate(
            e_g,
            n_sg,
            v_sg,
            sigma_xg,
            dedsigma_xg,
            tau_sg,
            dedtau_sg,
            np.zeros(
                (
                    n_sg.shape[0],
                    3,
                )
                + e_g.shape,
                dtype=e_g.dtype,
            ),
        )


class _SLCiderBase:
    def get_setup_name(self):
        return "PBE"

    def todict(self):
        kparams = self.kernel.todict()
        return {
            "kernel_params": kparams,
        }

    def get_mlfunc_data(self):
        return yaml.dump(self.kernel.mlfunc, Dumper=yaml.CDumper)

    @classmethod
    def from_joblib(cls, fname, **kwargs):
        mlfunc = joblib.load(fname)
        return cls.from_mlfunc(mlfunc, **kwargs)

    @staticmethod
    def from_mlfunc(mlfunc, xmix=1.00, xkernel="GGA_X_PBE", ckernel="GGA_C_PBE"):
        if mlfunc.desc_version == "b":
            cider_kernel = SLCiderMGGAHybridWrapper(mlfunc, xmix, xkernel, ckernel)
            cls = SLCiderMGGA
        elif mlfunc.desc_version == "d":
            cider_kernel = SLCiderGGAHybridWrapper(mlfunc, xmix, xkernel, ckernel)
            cls = SLCiderGGA
        else:
            raise ValueError(
                "Only implemented for b and d version, found {}".format(
                    mlfunc.desc_version
                )
            )

        return cls(cider_kernel)


class SLCiderGGA(_SLCiderBase, DiffGGA):
    def todict(self):
        d = super(SLCiderGGA, self).todict()
        d["_cider_type"] = "SLCiderGGA"
        return d


class SLCiderMGGA(_SLCiderBase, DiffMGGA):
    def todict(self):
        d = super(SLCiderMGGA, self).todict()
        d["_cider_type"] = "SLCiderMGGA"
        return d
