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
   python -m pip install 'numpy>=2,<3' meson ninja
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

Continuous Integration
----------------------

GitHub Actions builds the Fortran extension from source and runs the complete
test suite on Linux (Python 3.10 and 3.12) and macOS (Python 3.12). CPU JAX is
installed in each job so the optional numerical tests run too. The installed
extension is checked outside the source tree before testing. Test reports are
attached to the workflow run, including on test failure.

The suite includes independent Radau comparisons, emergence and floating-point
perturbation regressions, and transport eigenmode/conservation/resolution checks.
It does not run inference analyses or require private data files.
