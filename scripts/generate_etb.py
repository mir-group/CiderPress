import numpy as np
from pyscf.data.elements import ELEMENTS
from pyscf.gto.basis import load

ZMAX = 118
LMAX = 3
min_data = np.zeros((ZMAX + 1, LMAX + 1), dtype=np.float64)
max_data = np.zeros((ZMAX + 1, LMAX + 1), dtype=np.float64)
for Z in range(1, ZMAX + 1):
    symb = ELEMENTS[Z]
    print(symb)
    # want at least up to d shells
    if Z == 1:
        basis = "HGBSP2-5"
    else:
        basis = "HGBSP2-5"
    # basis = 'ANO-RCC'
    try:
        basis = load(basis, symb)
    except:
        basis = "HGBSP2-5"
        basis = load(basis, symb)
    try:
        basis = basis[symb]
    except TypeError:
        pass
    for lst in basis:
        l = lst[0]
        if l > 3:
            continue
        lists = lst[1:]
        expnts = []
        for lst in lists:
            expnts.append(lst[0])
        min_data[Z, l] = np.min(expnts)
        max_data[Z, l] = np.max(expnts)

np.save("ciderpress/data/expnt_mins.npy", min_data)
np.save("ciderpress/data/expnt_maxs.npy", max_data)
