import sys

from ase.build import bulk
from gpaw import PW

from ciderpress.gpaw.calculator import CiderGPAW, get_cider_functional

# NOTE: Run this script as follows:
# mpirun -np <NPROC> python simple_calc.py

atoms = bulk("Si")

mlfunc = "functionals/{}.yaml".format(sys.argv[1])

# This is the initializer for CIDER functionals for GPAW
xc = get_cider_functional(
    # IMPORTANT: NormGPFunctional object or a path to a joblib or yaml file
    # containing a CIDER functional.
    mlfunc,
    # IMPORTANT: xmix is the mixing parameter for exact exchange. Default=0.25
    # gives the PBE0/CIDER surrogate hybrid.
    xmix=0.25,
    # largest q for interpolating feature expansion, default=300 is usually fine
    qmax=300,
    # lambda parameter for interpolating features. default=1.8 is usually fine.
    # Lower lambd is more precise
    lambd=1.8,
    # pasdw_store_funcs=False (default) saves memory. True reduces cost
    pasdw_store_funcs=False,
    # pasdw_ovlp_fit=True (default) uses overlap fitting to improve precision
    # of PAW correction terms of features.
    pasdw_ovlp_fit=True,
    fast=True,
)

# Using CiderGPAW instead of the default GPAW calculator allows calculations
# to be restarted. GPAW calculations will run with CIDER functionals but
# cannot be saved and loaded properly.
atoms.calc = CiderGPAW(
    h=0.18,  # use a reasonably small grid spacing
    xc=xc,  # assign the CIDER functional to xc
    mode=PW(520),  # plane-wave mode with 520 eV cutoff.
    txt="-",  # output file, '-' for stdout
    occupations={"name": "fermi-dirac", "width": 0.01},
    # ^ Fermi smearing with 0.01 eV width
    kpts={"size": (4, 4, 4), "gamma": False},  # kpt mesh parameters
    convergence={"energy": 1e-5},  # convergence energy in eV/electron
    # Set augments_grids=True for CIDER functionals to parallelize
    # XC energy and potential evaluation more effectively
    parallel={"augment_grids": True},
    spinpol=True,
)
etot = atoms.get_potential_energy()  # run the calculation
