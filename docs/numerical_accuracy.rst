Numerical accuracy audit
========================

Scope of the correction
-----------------------

The simple-mode static-shell solver now uses explicit free-flight contact,
the negligible-seed-mass limit, step-doubled RK4 with error control, and an
outer-edge event. Shock powers and shell properties refer to the same time.
Both sides of the forward-shock discontinuity are retained. Diffusion uses
all accepted integration steps and midpoint source coefficients, rather than
the former thinned history and left-endpoint coefficients. Static contact
rows are no longer removed by the legacy initial-row filter.

This path applies to positive-density inner boundaries with a finite BPL
ejecta edge, or the existing effective exponential edge at 20 scale velocities.
It includes the static power-law, spline, and p-spline wrappers. Static grids
starting with zero density and untruncated BPL ejecta still use the legacy
path. Moving CSM and transport dynamics are not changed by this patch.

Independent tests
-----------------

``tests/static_shell_reference.py`` integrates the dimensionless equations
with SciPy Radau, relative tolerance ``1e-11``, event location at emergence,
and dimensionless seed masses ``1e-9`` and ``1e-10``. The two seeds agree
to better than the ``2e-6`` relative test threshold. The diffusion reference
uses adaptive quadrature of the original injection-time kernel evaluated
on the Radau dense solution, not a Fortran or JAX output history.

The benchmark has CSM slope -2, radii 1e14--1e15 cm, 0.5 solar masses of
CSM, 10 solar masses of ejecta, 1e51 erg, BPL indices 1 and 10, cutoff ratio
3, efficiency 0.5, and opacity 0.34 cm2/g. Reference contact and emergence
are approximately 1.017047 and 28.455431 days.

The regression tests require relative shock-power errors below 0.1% and
diffused errors below 0.3% at specified epochs outside the immediate
emergence cusp. They also scan 301 consecutive floating-point energies
for the static BPL driver. This is a tested benchmark, not a claim of
uniform accuracy across the entire prior space.

Reproduce the wider diagnostic scan with::

    python -m tests.numerical_audit

In the audit, the original static solver shifted contact between 1.017061
and 1.029174 days and produced adjacent-energy changes of 0.427 mag in
diffused luminosity at 1.05 days. The corrected static BPL, exponential,
and spline scans had ranges below 1e-10 mag in the 31-point extended scan.
Warm calls for those examples took approximately 6 ms locally; this is
not a portable runtime guarantee.

Outstanding issues
------------------

* The homologous power-law, homologous spline, and generic four-shell
  diagnostic cases still had ranges of approximately 0.0078, 0.282, and
  0.0036 mag respectively under ULP perturbations. They retain the old
  integrator, initialization, and output-dependent diffusion. Do not infer
  their convergence from the corrected static tests.
* The gridded homologous density evaluator extends endpoint densities
  outside its grid, whereas the static evaluator returns zero. Consequently
  a finite snapshot is not currently a finite-support homologous shell in
  the dynamics. This boundary convention must be resolved explicitly before
  applying static-style contact/emergence events to those drivers.
* Transport shock power differed from Radau by approximately 0.15--0.47%
  at 2, 5, 10, and 20 days. Radiation at 10 days changed from 5.799e42 to
  5.508e42 to 5.333e42 erg/s for 40, 80, and 160 zones. These are resolution
  diagnostics, not independent verification of the radiation PDE. Relative
  changes in the almost-dark phase were substantially larger.

The emergence cusp is also a model issue
---------------------------------------

The existing kernel assigns a diffusion timescale to each injection epoch.
As the remaining optical depth vanishes, suppose
``t_diff(t') ~ a (t_emerge - t')``. At observation time ``t_emerge``, its
integrand behaves as
``L_heat exp(-1/a) / [a (t_emerge - t')]`` when the pre-emergence heating
is nonzero. Its integral is logarithmically divergent. A finite grid hides
this singularity; refinement cannot give a finite, grid-independent peak
at exactly emergence. There is no uniform numerical error bound across
that point without changing the closure.

For the corrected Fortran benchmark, comparison with adaptive quadrature
gave magnitude differences of approximately +0.00038, +0.00130, +0.0100,
-0.0316, -0.0137, and -0.00340 at offsets -0.1, -0.01, -0.001, +0.001,
+0.01, and +0.1 days from reference emergence. These are benchmark-specific
errors, not general bounds. No opacity floor or smoothing prescription has
been inserted to hide the cusp. A finite escape-time or evolving-reservoir
closure would be a separate, explicitly documented physical model change.

Combined solver validation
--------------------------

PR 6 includes the Fortran correction alongside the standalone JAX simple-mode
solver, with the independent Radau reference and
``test_jax_simple_accuracy.py`` providing shared accuracy checks.
Further convergence tests should include multiple integration horizons
and emergence-adjacent epochs, and should distinguish numerical error from
the kernel's singular continuum limit. Agreement with the legacy Fortran
curve alone is not an accuracy criterion. The outstanding issues above are
not resolved by combining the changes in one PR.
