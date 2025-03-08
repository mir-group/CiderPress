GPAW Interface
==============

The GPAW interface allows CIDER functionals to be used in the GPAW code.
CIDER has only been tested with the plane-wave mode of GPAW. It is
recommended to use PAW setups (not pseudopotentials) because accurately
computing the CIDER features requires an all-electron formalism.
It is possible to run CIDER functionals with norm-conserving pseudopotentials
(except for semilocal meta-GGA models), but it is not recommended.

This documentation assumes that you are familiar with the GPAW code and
have a working installation of the software. For GPAW documentation,
see the `GPAW website <https://gpaw.readthedocs.io/>`_.

The key user-facing component of the GPAW interface is the
:ref:`calculator<GPAW Calculator Interface>` module, which provides tools
to initialize a CIDER functional object that can be used in GPAW.
It also provides a subclass of the ``GPAW`` calculator object.
See the :ref:`calculator<GPAW Calculator Interface>` module documentation
for examples and API documentation.

Note that CIDER does not support ``gpaw.new`` yet.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   calculator

