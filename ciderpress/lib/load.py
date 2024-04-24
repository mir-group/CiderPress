import os

import numpy


def load_library(libname):
    _loaderpath = os.path.dirname(__file__)
    return numpy.ctypeslib.load_library(libname, _loaderpath)
