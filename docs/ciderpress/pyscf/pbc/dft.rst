pbc.dft
=======

The :py:mod:`ciderpress.pyscf.pbc.dft` module serves the same
purpose as :py:mod:`ciderpress.pyscf.dft` but for periodic
systems. This module's version of :py:func:`make_cider_calc`
modifies a :py:mod:`pbc` Kohn Sham DFT object from PySCF
to evaluate a CIDER functional. Currently only semilocal
and SDMX features are supported.

**NOTE**: This module is particularly experimental. It is
provided for the purpose of reproducing previous work that
used this module (i.e. :footcite:t:`CIDER24X`) and is not as
close to production readiness as other parts of the code.

.. automodule:: ciderpress.pyscf.pbc.dft
    :members:

.. footbibliography::
