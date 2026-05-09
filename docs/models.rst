Available Models
================

The package exposes 29 base CSM-density/ejecta scenarios. These are the
unsuffixed physical configurations before adding output-specific wrappers. Each
base scenario has generated optical, nickel, radio, and X-ray wrapper variants,
so the public callable model count is larger than 29: 29 base scenarios times
six wrapper forms, plus the generic ``csm_xray`` helper, for 175 public
callables. The optical wrapper family is:

.. list-table::
   :header-rows: 1

   * - Suffix
     - Description
   * - ``{name}_bolometric``
     - Bolometric luminosity (erg/s), source-frame time in days
   * - ``{name}``
     - Multiband (flux density / magnitude / spectra), observer-frame time
   * - ``{name}_nickel_bolometric``
     - CSM + radioactive nickel/cobalt decay, bolometric
   * - ``{name}_nickel``
     - CSM + radioactive nickel/cobalt decay, multiband

The 29 base names are:
``wind_exponential``, ``wind_bpl``, ``exponential_wind``, ``bpl_wind``,
``exponential_exponential``, ``exponential_bpl``, ``bpl_bpl``,
``bpl_exponential``, ``boxwind_exponential``, ``boxwind_bpl``,
``gausswind_exponential``, ``gausswind_bpl``,
``triple_powerlaw_wind_bpl``, ``triple_powerlaw_wind_exponential``,
``exponential_triple_powerlaw_wind``, ``bpl_triple_powerlaw_wind``,
``smooth_triple_powerlaw_wind_bpl``,
``smooth_triple_powerlaw_wind_exponential``, ``generic_csm_exponential``,
``generic_csm_bpl``, ``static_powerlaw_csm_exponential``,
``static_powerlaw_csm_bpl``, ``homologous_powerlaw_csm_exponential``,
``homologous_powerlaw_csm_bpl``, ``generic_4shell_csm_bpl``,
``generic_8shell_csm_bpl``, ``static_spline_csm_bpl``,
``generic_spline_csm_bpl``, and ``generic_spline12_csm_bpl``.

Shared Runtime Options
----------------------

All optical / bolometric model wrappers also accept a common set of runtime
keywords controlling diffusion / transport:

.. list-table::
   :header-rows: 1

   * - Keyword
     - Meaning
   * - ``mode='simple'``
     - Default thin-shell calculation. If ``kappa`` is supplied, this uses the legacy post-processed diffusion light curve.
   * - ``mode='transport'``
     - Use the newer transport solver for the observed luminosity while keeping the same shell dynamics.
   * - ``kappa``
     - Opacity in ``cm^2 g^-1``. Optional in simple mode. In transport mode it defaults to ``0.34`` if omitted.
   * - ``n_rad_zones``
     - Number of transport zones used in transport mode. Python model calls promote values below 40 to 40 because the transport boundary layer is under-resolved below that. Larger values reduce numerical roughness at higher runtime cost.
   * - ``efficiency_mode``
     - Optional alternate shock-efficiency mode. Default ``0`` keeps the user-supplied constant ``eff`` for both shocks.

Output convention:

- ``lbol`` is the main observable luminosity
- ``lbol_shock`` is the shock-powered luminosity
- ``lbol_diffuse`` is the diffusion / transport luminosity when available
- ``rph`` is the historical field name, but currently stores the shell radius by convention

JAX static CSM backend
----------------------

The package also includes a JAX-native backend for the static finite power-law
CSM shell plus broken-power-law ejecta model. This backend is intended for fast
inference experiments and validation against the Fortran backend. It supports
the same ``mode='simple'`` and ``mode='transport'`` terminology, but it is not
yet a general replacement for the arbitrary-CSM Fortran implementation.

.. code-block:: python

   from jax_csm.model import get_static_powerlaw_csm_bpl_lightcurve

   lbol = get_static_powerlaw_csm_bpl_lightcurve(
       time=time_days,
       eta=-2.0,
       r_inner=5e2 * 6.957e10,
       r_outer=5e3 * 6.957e10,
       delta_sn=1.0,
       nn_sn=10.0,
       mej_sn=5.0,
       esn=1.0,
       eff=1.0,
       m_csm=1.0,
       mode='transport',
       kappa=0.2,
   )

Radio and X-ray variants
------------------------

Each base CSM scenario also has matching ``_radio`` and ``_xray``
wrappers. The radio wrappers compute synchrotron emission from the CSM
interaction shock. The X-ray wrappers compute an approximate thermal
bremsstrahlung diagnostic using the same shock trajectory and upstream CSM
density. The X-ray calculation is a fast post-processor, not a resolved cooling
layer or spectral-fitting model.

Common X-ray controls include ``logepsx``, ``e_min_kev``, ``e_max_kev``,
``output_format``, ``n_h_host``, ``n_h_mw``, ``absorb_csm``,
``shock_component``, ``normalization``, and ``max_xray_efficiency``.

Wind / Simple CSM Models
------------------------

Broken power-law ejecta cutoff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All BPL-ejecta wrappers accept the optional keyword ``vej_max_ratio``. It sets
the finite outer ejecta edge as

.. math::

   v_{\rm max} = {\tt vej\_max\_ratio}\,v_t,

where ``v_t`` is the BPL transition velocity set by the ejecta mass, kinetic
energy, and the inner/outer power-law indices. If it is omitted, wrappers that
provide BPL parameters use ``vej_max_ratio=3``. This is intentionally finite:
increasing it lets faster, lower-mass outer ejecta interact with distant CSM at
earlier times, while decreasing it delays that interaction. The aliases
``A_ratio`` and ``vej_max`` are also accepted; ``vej_max`` is supplied in km/s
and converted internally.

**wind_exponential**
    A steady, spherically symmetric wind (constant mass-loss rate) creates the CSM.
    The supernova ejecta follow an exponential density profile.
    Parameters: ``mdot`` (M☉/yr), ``vwind`` (km/s), ``mexp`` (M☉), ``eexp`` (foe), ``eff``.

**wind_bpl**
    Steady wind CSM with a broken power-law supernova ejecta profile (inner index
    ``delta``, outer index ``nn``). The most commonly used model for Type IIn supernovae.
    Parameters: ``mdot``, ``vwind``, ``delta``, ``nn``, ``mexp``, ``eexp``, ``eff``.

**exponential_wind**
    An exponential eruption (or low-energy transient) expands outward and subsequently
    interacts with an ongoing stellar wind.
    Parameters: ``mexp``, ``eexp``, ``mdot``, ``vwind``, ``eff``.

**bpl_wind**
    A broken power-law eruption expands into a surrounding wind.
    Parameters: ``delta``, ``nn``, ``mexp``, ``eexp``, ``mdot``, ``vwind``, ``eff``.

Eruption–Explosion Interaction Models
--------------------------------------

These models describe a pre-SN eruption that creates the CSM, followed some time
``interval`` (days) later by the main supernova.

**exponential_exponential**
    Both the eruption (CSM) and the supernova have exponential ejecta profiles.

**exponential_bpl**
    Exponential eruption (CSM) + broken power-law supernova.

**bpl_bpl**
    Broken power-law eruption (CSM) + broken power-law supernova.

**bpl_exponential**
    Broken power-law eruption (CSM) + exponential supernova.

Shaped Wind Profile Models
---------------------------

**boxwind_exponential** / **boxwind_bpl**
    The pre-SN mass-loss rate follows a box (step-function) profile in time:
    a baseline rate ``mdot_0`` outside the interval [t1, t2] years and an elevated
    rate ``mdot_1`` inside it. Useful for modeling discrete mass-loss episodes.

**gausswind_exponential** / **gausswind_bpl**
    The mass-loss rate follows a Gaussian profile in time, peaking at ``t_peak``
    (years before the SN) with width ``t_width``. Models a single eruptive episode
    with a smooth onset and decline.

Triple Power-Law Wind Models
-----------------------------

These models allow the pre-SN mass-loss history to follow a triple power-law in
time, with breaks at ``t_break1`` and ``t_break2`` (years). The reference rate is
``mdot_0`` at t = 1 year, with power-law indices ``alpha1``, ``alpha2``, ``alpha3``.

**triple_powerlaw_wind_bpl** / **triple_powerlaw_wind_exponential**
    Triple power-law wind CSM with BPL or exponential SN.

**exponential_triple_powerlaw_wind** / **bpl_triple_powerlaw_wind**
    SN expanding into a pre-existing triple power-law wind.

**smooth_triple_powerlaw_wind_bpl** / **smooth_triple_powerlaw_wind_exponential**
    Same as above but with smooth (tanh) transitions at the breaks rather than sharp
    discontinuities. Avoids unphysical kinks in the density profile.

Generic Phenomenological CSM Models
-------------------------------------

These models allow an arbitrary CSM density profile constructed from a power-law
base density plus up to N Gaussian shells. Set ``shell_density = 0`` to disable any
shell. This is the most flexible option, useful when the pre-SN mass-loss history is
poorly constrained.

**generic_csm_exponential** / **generic_csm_bpl**
    Power-law base + up to 3 Gaussian shells + exponential or BPL supernova.

**static_powerlaw_csm_exponential** / **static_powerlaw_csm_bpl**
    Finite-support power-law CSM shell with
    ``rho(r) = rho_in (r / r_inner)^eta`` on ``[r_inner, r_outer]``,
    normalized by the total shell mass ``m_csm``. This treats the CSM as a
    static density snapshot.

**homologous_powerlaw_csm_exponential** / **homologous_powerlaw_csm_bpl**
    The same finite-support power-law density normalization, but with a
    homologous velocity grid ``v = r / interval_sn``. Here ``interval_sn`` is
    a public parameter for the homologous variants only: it is the time between
    CSM ejection and SN explosion in days. This is the direct power-law
    analogue of the homologous generic/spline CSM constructors.

**generic_4shell_csm_bpl**
    Power-law base + up to 4 Gaussian shells + BPL supernova.

**generic_8shell_csm_bpl**
    Power-law base + up to 8 Gaussian shells + BPL supernova. High-dimensional
    (30+ parameters); consider using the 3-shell or 4-shell variant first.

Spline CSM Models
-----------------

The spline models represent the CSM density directly through log-density nodes
between ``log_r_inner`` and ``log_r_outer``. The interpolation is
shape-preserving in log-radius/log-density space, so these models are useful
when the data require more freedom than a small number of Gaussian shells but a
fully arbitrary density table would be too awkward for inference.

**static_spline_csm_bpl**
    Static finite CSM snapshot with eight log-density nodes and BPL SN ejecta.
    This is the data-driven analogue of ``static_powerlaw_csm_bpl``.

**generic_spline_csm_bpl**
    Homologous finite CSM snapshot with eight log-density nodes and BPL SN
    ejecta. The parameter ``interval_sn`` sets the homologous velocity grid
    through ``v = r / interval_sn``.

**generic_spline12_csm_bpl**
    Twelve-node homologous spline CSM. This is exposed for bolometric and
    nickel-bolometric fitting where the eight-node profile is too restrictive.
    Higher-node penalised-spline reconstructions can be run through the MLE
    utilities described below.

The public wrappers have generated priors for the exposed optical, radio, and
X-ray variants. For high-node reconstructions, use a smoothness penalty or
informative priors; the individual nodes should not be interpreted as discrete
physical shells.

Generated priors
----------------

Priors are generated programmatically from the model signatures, rather than
stored as one duplicated file per callable wrapper. This keeps the optical,
nickel, radio, X-ray, static-shell, generic-shell, and spline variants
consistent as the model list changes. The generic ``csm_xray`` convenience
function is the only exported helper without an automatically generated prior,
because its physical CSM model is selected at runtime through the ``csm_model``
argument.

CSM mass interpretation
-----------------------

The arbitrary and spline CSM models fit a density field. The direct mass
integral is therefore a spherical-equivalent mass,
``M = integral 4 pi r^2 rho(r) dr``. If the CSM is clumpy or asymmetric, this
should be scaled by an assumed covering fraction and filling factor rather than
interpreted as a unique total mass.

``redback_csm.analysis`` provides helpers for this bookkeeping, including
``csm_mass_from_density_grid``, ``generic_spline_csm_mass_from_params``, and
``sample_geometry_corrected_mass``. For example:

.. code-block:: python

   from redback_csm.analysis import (
       generic_spline_csm_mass_from_params,
       sample_geometry_corrected_mass,
   )

   m_spherical = generic_spline_csm_mass_from_params(best_fit_parameters)
   mass_samples = sample_geometry_corrected_mass(
       m_spherical,
       covering_fraction={"kind": "uniform", "min": 0.1, "max": 1.0},
       filling_factor={"kind": "loguniform", "min": 0.01, "max": 1.0},
   )

Spline MLE utilities
--------------------

For rapid iteration before running a full sampler, ``redback_csm.spline_mle``
provides lightweight least-squares utilities for static and homologous spline
CSM reconstructions. The helper handles parameter ordering, bounds, random
starts, curvature penalties on the log-density nodes, and optional CSM-mass
penalties, while the user supplies the event-specific model function. The
packaged examples use bolometric luminosities. The helper itself only assumes a
one-dimensional scalar data vector, so multiband fitting would require the
caller to flatten the observations across filters and provide a matching
flattened model vector.

.. code-block:: python

   from redback_csm.spline_mle import (
       SplineMLEProblem,
       default_spline_bounds,
       make_random_starts,
       spline_parameter_names,
   )

   names = spline_parameter_names(
       n_nodes=24,
       profile="generic",
       include_time_offset=True,
       include_nickel=True,
   )
   bounds = default_spline_bounds(
       n_nodes=24,
       profile="generic",
       include_time_offset=True,
       include_nickel=True,
   )

   problem = SplineMLEProblem(
       time=time,
       luminosity=lbol,
       error=lbol_err,
       parameter_names=names,
       bounds=bounds,
       model_function=my_model_function,
       profile="generic",
       smoothness_sigma=0.5,
       max_csm_mass=100.0,
   )
   starts = make_random_starts(start_params, bounds, names, n_random=8)
   result = problem.fit(starts, max_nfev=500)

Syntax
-------------------------------------

All models share similar syntax consistent with redback. The order of the naming is chronological: the first component
describes the CSM (e.g. ``wind``, ``exponential``, ``bpl``, ``generic``), and the second component describes the
supernova ejecta profile (``exponential`` or ``bpl``).
For example, ``wind_exponential`` means a steady wind CSM with an exponential SN ejecta profile.

Note some models are the other way round, e.g. ``exponential_wind`` means an exponential eruption
followed by a steady wind, and ``bpl_exponential`` means a BPL eruption followed by an exponential eruption.

Example with transport mode:

.. code-block:: python

   from redback_csm.models import wind_bpl_bolometric

   lbol = wind_bpl_bolometric(
       time=time,
       mdot=1e-3,
       vwind=100,
       delta=0.5,
       nn=12,
       mexp=10.0,
       eexp=1.0,
       eff=0.5,
       mode='transport',
       kappa=0.34,
       n_rad_zones=120,
   )

Exploration: Arbitrary Density Profile
---------------------------------------

For visualisation and exploration with numerically-computed density profiles
(e.g. from stellar-evolution codes), use ``csm_lightcurve_from_density`` from
``redback_csm.explore``.  This function is **not** registered as an inference
model — it is designed for interactive exploration only.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from redback_csm.explore import csm_lightcurve_from_density

   # Example: r^{-2} CSM (equivalent to a steady wind)
   r   = np.geomspace(1e13, 1e17, 500)   # cm
   rho = 5e16 / r**2                      # g/cm^3

   fig, lc = csm_lightcurve_from_density(
       radius=r,
       density=rho,
       t_ref=365.0,       # days — sets r = v × t_ref
       mexp=10.0,         # M☉
       eexp=1.0,          # foe
       eff=0.3,
       kappa=0.34,        # cm²/g — enables photon diffusion
       sn_profile='bpl',  # or 'exponential'
       delta=1.0,
       nn=12.0,
       label=r'$\rho \propto r^{-2}$',
   )
   fig.savefig('my_lightcurve.png')

The function accepts a user-supplied ``radius`` (cm) and ``density`` (g cm⁻³)
array, interpolates them onto the internal Fortran velocity grid using the
mapping r = v × ``t_ref``, runs the shock interaction model, and returns both
the figure and the raw light-curve namedtuple (``lc.time``, ``lc.lbol``,
``lc.rph``, ``lc.temperature``, …).

Multiple profiles can be overlaid by passing the same ``ax``:

.. code-block:: python

   fig, ax = plt.subplots()
   for slope in [1.5, 2.0, 2.5]:
       csm_lightcurve_from_density(
           radius=r, density=1e16 / r**slope,
           t_ref=365.0, mexp=10.0, eexp=1.0, eff=0.3,
           ax=ax, label=rf'$\rho \propto r^{{-{slope}}}$',
       )
   ax.legend()

**Parameters summary**

.. list-table::
   :header-rows: 1

   * - Parameter
     - Unit
     - Description
   * - ``radius``
     - cm
     - Radial grid of the CSM profile
   * - ``density``
     - g cm⁻³
     - Density at each radius
   * - ``t_ref``
     - days
     - Reference epoch; sets r = v × t_ref
   * - ``mexp``
     - M☉
     - Supernova ejecta mass
   * - ``eexp``
     - foe
     - Supernova explosion energy
   * - ``eff``
     - —
     - Shock-to-radiation efficiency
   * - ``kappa``
     - cm² g⁻¹
     - Opacity (optional; enables diffusion)
   * - ``sn_profile``
     - —
     - ``'bpl'`` or ``'exponential'``

Unit Conventions
-----------------

.. list-table::
   :header-rows: 1

   * - Quantity
     - Input unit
   * - Time (source frame)
     - days
   * - Mass (explosion, ejecta)
     - M☉
   * - Energy
     - foe (10⁵¹ erg)
   * - Mass-loss rate
     - M☉/yr
   * - Velocity
     - km/s
   * - Opacity
     - cm²/g
   * - Density (generic models)
     - g/cm³
   * - Radius/width (generic models)
     - cm
