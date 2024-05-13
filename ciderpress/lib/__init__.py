import ctypes

from ciderpress.lib.load import load_library

__all__ = ["load_library", "c_double_p", "c_int_p", "c_null_ptr"]

c_double_p = ctypes.POINTER(ctypes.c_double)
c_int_p = ctypes.POINTER(ctypes.c_int)
c_null_ptr = ctypes.POINTER(ctypes.c_void_p)
