.. _GPAW Calculator Interface:

GPAW Calculator Interface
=========================

The GPAW calculator interfaces modifies GPAW to be
compatible with CIDER functionals. There are two
key components. The first is the function
:func:`ciderpress.gpaw.calculator.get_cider_functional`,
which generates a CIDER functional for use in GPAW.
The second is the :func:`ciderpress.gpaw.calculator.CiderGPAW`
class, which modifies the ``GPAW`` calculator object
to be able to read and write calculations that use CIDER
functionals. ::

    xc = get_cider_functional(...)
    atoms.calc = CiderGPAW(xc=xc, ...)
    atoms.get_potential_energy()

For a full example, see :source:`examples/gpaw/simple_calc.py`
and the other examples in :source:`examples/gpaw`

.. automodule:: ciderpress.gpaw.calculator
    :members:
