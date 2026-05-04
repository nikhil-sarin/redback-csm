Installation
============

Requirements
------------

- Python >= 3.10
- ``gfortran`` for compiling the Fortran extension
- ``redback >= 1.15.0``

Install
-------

.. code-block:: bash

   git clone https://github.com/nikhil-sarin/redback-csm
   cd redback-csm
   bash setup_fortran.sh
   pip install -e .

Optional extras:

.. code-block:: bash

   pip install -e ".[dev]"
   pip install -e ".[jax]"
   pip install -e ".[docs]"

Smoke Tests
-----------

.. code-block:: bash

   python -m pytest tests/test_smoke.py -q

The smoke tests cover simple mode, legacy diffusion, transport mode, nickel,
radio, X-rays, and the optional JAX backend when JAX is installed.
