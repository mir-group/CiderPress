import sys
from collections import Counter

import ase
from ase import Atoms
from ase.data import chemical_symbols, ground_state_magnetic_moments
from pyscf import dft, gto, scf
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase

from ciderpress.pyscf.dft import make_cider_calc

"""
This script demonstrates running a CIDER calculation with accelerated
nonlocal feature evaluation. Example commands:

    python examples/pyscf/fast_cider.py <molecule_formula> <charge> <spin> <functional>
    python examples/pyscf/fast_cider.py H2 0 0 PBE
    python examples/pyscf/fast_cider.py O2 0 2 CIDER_NL_MGGA

<molecule_formula> is a chemical formula string like CH4, H2, etc. It must be included
in the list of molecules supported by ase.build.molecule()

<charge> is the integer charge of the system.

<spin> is the integer spin of the system 2S.

<functional> is the functional name. It can be the name of a libxc functional,
or it can be the name of a functional in the functionals/ directory, in which case
the corresponding example CIDER functional is run with the PBE0/CIDER
surrogate hybrid functional form. If a path to a joblib file is given, that
file will be read assuming it is a CIDER functional.

At the end, prints out the total energy of the molecule and its atomization energy
in Ha and eV, then saves the atomization energy in eV to aeresult.txt.
"""

name, charge, spin, functional = sys.argv[1:5]
charge = int(charge)
spin = int(spin)

spinpol = True if spin > 0 else False
BAS = "def2-qzvppd"
if name == "HF_stretch":
    BAS = "def2-svp"
    atoms = Atoms(symbols=["H", "F"], positions=[[0, 0, 0], [0, 0, 1.1]])
elif name.startswith("el-"):
    el = name[3:]
    atoms = Atoms(el)
elif name.endswith(".xyz"):
    ismol = True
    atoms = ase.io.read(name)
    atoms.center(vacuum=4)
else:
    ismol = True
    from ase.build import molecule

    atoms = molecule(name)
    atoms.center(vacuum=4)

if functional.startswith("CIDER"):
    functional = "functionals/{}.yaml".format(functional)
    is_cider = True
    mlfunc = functional
elif functional.endswith(".joblib"):
    is_cider = True
    mlfunc = functional
else:
    is_cider = False
formula = Counter(atoms.get_atomic_numbers())

mol = gto.M(
    atom=atoms_from_ase(atoms), basis=BAS, ecp=BAS, spin=spin, charge=charge, verbose=4
)


def run_calc(mol, spinpol):
    if spinpol:
        ks = dft.UKS(mol)
    else:
        ks = dft.RKS(mol)
    ks = ks.density_fit()
    ks.with_df.auxbasis = "def2-universal-jfit"
    ks = ks.apply(scf.addons.remove_linear_dep_)
    if is_cider:
        ks = make_cider_calc(
            ks,
            functional,
            xmix=0.25,
            xkernel="GGA_X_PBE",
            ckernel="GGA_C_PBE",
        )
    else:
        ks.xc = functional
    ks.grids.level = 3
    ks = ks.apply(scf.addons.remove_linear_dep_)
    etot = ks.kernel()
    return etot


if spin == 0:
    spinpol = False
else:
    spinpol = True

etot_mol = run_calc(mol, spinpol)
etot_ae = -1 * etot_mol

for Z, count in formula.items():
    atom = gto.M(
        atom=chemical_symbols[Z],
        basis=BAS,
        ecp=BAS,
        spin=int(ground_state_magnetic_moments[Z]),
        verbose=4,
    )
    etot_atom = run_calc(atom, True)
    etot_ae += count * etot_atom

print("Total and Atomization Energies, Ha")
print(etot_mol, etot_ae)
eh2ev = 27.211399
print("Total and Atomization Energies, eV")
print(etot_mol * eh2ev, etot_ae * eh2ev)
with open("aeresult.txt", "w") as f:
    f.write(str(etot_ae * eh2ev))
