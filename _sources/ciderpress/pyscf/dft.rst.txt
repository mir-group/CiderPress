dft
===

The :py:mod:`ciderpress.pyscf.dft` module provides the
function :py:func:`make_cider_calc`, which takes
a Pyscf :py:class:`KohnShamDFT` object and a CIDER
functional object (:py:class:`MappedXC` or :py:class:`MappedXC2`)
and returns an instance of a :py:class:`KohnShamDFT`
subclass that uses the CIDER functional. The function
is similar to native PySCF routines like :py:func:`density_fit`,
in which the input SCF object is "decorated" with the
necessary routines to evaluate the CIDER functional.

The basic use case is::
    
    from pyscf.dft import RKS
    from pyscf import gto
    mol = gto.M(...)
    ks = dft.RKS(mol)
    ks = make_cider_calc(ks, mlfunc, ...)
    etot = ks.kernel()

For a complete example, please see :source:`examples/pyscf/simple_calc.py`
and the other examples in :source:`examples/pyscf`.

.. automodule:: ciderpress.pyscf.dft
    :members:
