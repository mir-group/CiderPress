import sys

import numpy as np
from ase.build import bulk
from gpaw import PW, Mixer

from ciderpress.gpaw.calculator import CiderGPAW, get_cider_functional
from ciderpress.gpaw.descriptors import get_descriptors

# NOTE: Run this script as follows:
# mpirun -np <NPROC> gpaw python simple_calc.py

MODIFY_CELL = False

atoms = bulk("Si")

mlfunc = "functionals/{}.yaml".format(sys.argv[1])

# This is the initializer for CIDER functionals for GPAW
xc = get_cider_functional(
    # IMPORTANT: Path to a joblib or yaml file containing a CIDER functional.
    # Object stored in yaml/joblib must be MappedXC or MappedXC2.
    # Can also pass the object itself rather than a file name.
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
    # of PAW correction terms of features. Usually not needed.
    pasdw_ovlp_fit=True,
)

# Using CiderGPAW instead of the default GPAW calculator allows calculations
# to be restarted. Calculations using GPAW (rather than CiderGPAW)
# will run with CIDER functionals but cannot be saved and loaded properly.
atoms.calc = CiderGPAW(
    # use a reasonably small grid spacing
    h=0.18,
    # assign the CIDER functional to xc
    xc=xc,
    # plane-wave mode with 520 eV cutoff
    mode=PW(520),
    # output file, '-' for stdout
    txt="-",
    # Fermi smearing with 0.01 eV width
    occupations={"name": "fermi-dirac", "width": 0.01},
    # kpt mesh parameters
    kpts={"size": (4, 4, 4), "gamma": False},
    # convergence energy in eV/electron
    convergence={"energy": 1e-5},
    # Set augments_grids=True for CIDER functionals to parallelize
    # XC energy and potential evaluation more effectively
    parallel={"augment_grids": True},
    # Customize the mixer object if desired.
    mixer=Mixer(0.7, 8, 50),
    # Turn spin polarization on or off.
    spinpol=False,
)

# If desired, make a low-symmetry cell for testing purposes.
if MODIFY_CELL:
    atoms.set_cell(
        np.dot(atoms.cell, [[1.02, 0, 0.03], [0, 0.99, -0.02], [0.2, -0.01, 1.03]]),
        scale_atoms=True,
    )

# run the calculation
etot = atoms.get_potential_energy()

desc, wts = get_descriptors(
    atoms.calc, atoms.calc.hamiltonian.xc.cider_kernel.mlfunc.settings.nldf_settings
)

print(desc.sum())
