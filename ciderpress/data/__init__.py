import os

import numpy as np

DATA_PATH = os.path.dirname(os.path.abspath(__file__))


def get_hgbs_min_exps(Z):
    dpath = os.path.join(DATA_PATH, "expnt_mins.py")
    return np.load(dpath)[Z]


def get_hgbs_max_exps(Z):
    dpath = os.path.join(DATA_PATH, "expnt_maxs.py")
    return np.load(dpath)[Z]
