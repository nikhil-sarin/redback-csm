redback-csm documentation
=========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   models
   api

Overview
--------

**redback-csm** provides Fortran-based circumstellar matter (CSM) interaction models
for use with the `redback <https://github.com/nikhil-sarin/redback>`_ transient
inference package. Once installed, all CSM models are automatically discovered and
available in ``redback.model_library.all_models_dict``.

The package also includes :func:`redback_csm.explore.csm_lightcurve_from_density`,
a standalone utility for computing and plotting light curves from an arbitrary
user-supplied ρ(r) profile (e.g. from stellar-evolution codes) — useful for
exploration and visualisation without Bayesian inference.

Contributors
~~~~~~~~~~~~

- `nikhil-sarin <https://github.com/nikhil-sarin>`_
- `ryosuke-hirai <https://github.com/ryosuke-hirai>`_

Citation
~~~~~~~~

If you use this package please cite:

- **Sarin & Hirai (in prep)** — the paper describing the CSM interaction models
- **Sarin et al. 2024** — the redback paper (JOSS)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
