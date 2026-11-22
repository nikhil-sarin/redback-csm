Numerical accuracy audit
========================

Dynamics and finite support
---------------------------

All Fortran simple-mode drivers use step-doubled RK4 with error control,
rather than forward Euler. Detached static and homologous shells have explicit
contact and outer-edge events and a negligible initial swept mass. A central
wind instead starts at its launch radius. Shock powers and shell properties
refer to the same time, and both sides of an emergence discontinuity are stored.
Diffusion integrates every accepted step, independently of output thinning.

Gridded homologous snapshots now have finite support: their density is zero
outside the supplied velocity grid. Previously the endpoint density was
extended beyond that grid. Profile lookup supports rejected RK steps and
non-monotone queries. Infinite analytic winds retain their wind boundary
convention. Exponential ejecta use the existing effective maximum velocity
of twenty exponential scale velocities.

Finite escape time: a model change
---------------------------------

The simple-mode diffusion kernel assigns each injection epoch an escape time.
The old ``tau R / c`` prescription goes to zero at a sharp CSM edge. With
nonzero heating there, the integral at emergence is logarithmically divergent;
a smaller numerical timestep cannot resolve it to a finite peak.

The default ``diffusion_escape='finite'`` uses ``(1 + tau) R / c``. This
retains a light-crossing time in the optically thin limit and includes ongoing
reverse-shock heating after forward-shock emergence. It is an explicit escape
closure, not a solution of optically thin radiative transfer or an opacity
floor. It removes the singularity without smoothing the density or shock event.

``diffusion_escape='legacy'`` retains the old kernel for comparison, including
its singular limit; it does not restore the old Euler dynamics. The finite
option is available in the Fortran Python wrappers and standalone JAX simple
solver. Existing fits can change, particularly near contact/emergence and for
optically thin CSM. These corrections are not bitwise-compatible performance
optimizations. Re-evaluate likelihoods before reusing previous inference results.

Independent references
----------------------

``tests/static_shell_reference.py`` independently integrates the shell equations
using SciPy Radau at relative tolerance ``1e-11``. Dimensionless seed masses
``1e-9`` and ``1e-10`` agree within ``2e-6`` in the reference regression.
Adaptive quadrature of the kernel on the Radau dense solution supplies the
diffusion reference, rather than either production solver's output rows.

The static benchmark uses slope -2, radii 1e14--1e15 cm, 0.5 solar masses of
CSM, 10 solar masses of ejecta, 1e51 erg, BPL indices 1 and 10, cutoff ratio
3, efficiency 0.5, and opacity 0.34 cm2/g. Contact and emergence occur at
approximately 1.017047 and 28.455431 days. Tests require shock-power errors
below 0.1% and diffused errors below 0.3% at specified epochs, including
emergence-adjacent tests for the finite kernel. JAX refinement is checked
against the same reference, including exactly at emergence.

Additional tests cover a homologously expanding power-law shell, an analytic
four-Gaussian homologous profile, and floating-point energy perturbations for
static exponential, static spline, homologous spline, and multiple-shell
drivers. The analytic four-shell reference is independent of the production
sampled density grid. ULP smoothness alone is not an accuracy test.

In the extended 31-point energy scan, static and homologous power-law/spline
curves vary by less than 1e-10 mag; the four-shell case retains a bounded
variation of about 3e-5 mag. No zero/finite flips occur in those tests.

Transport verification
----------------------

The transport radiation operator now uses spherical nodal control volumes,
including storage in the outer half-cell, and the corrected grid-advection
Peclet criterion. The old algebraic outer boundary omitted that storage.
Crank--Nicolson steps producing significant negative energy are retried with
backward Euler instead of clipping those modes. Shock powers are evaluated at
the output time rather than the preceding timestep.

``tests/transport_probe.f90`` exercises the actual Fortran PDE against a
spherical diffusion eigenmode with a Robin surface boundary. Tests verify
second-order spatial convergence and the energy balance of an injected flux
with a closed outer boundary. The full static benchmark's transport shock
power agrees with Radau within ``1e-4`` at 2, 5, 10, and 20 days.

A separate 80/160/320/640-zone lightcurve test checks radiation convergence
through interaction and cooling. For this benchmark, 160 versus 640 zones
agrees within 1% at the tested bright-phase epochs, and within 0.3% of the
peak luminosity across all tested epochs. Relative differences during the
nearly dark phase are larger. The default is therefore 160 zones; the minimum
40 remains a coarse option, not a general accuracy guarantee. Arbitrary sharp
profiles require their own resolution checks. These tests do not independently
validate the physical cooling-handoff assumptions or every prior combination.

The JAX transport operator also passes the analytic eigenmode test. This is
not a claim of full Fortran/JAX transport lightcurve parity: the cooling
handoffs remain different, and sizeable post-emergence differences persist.
The independent end-to-end JAX accuracy tests in this change concern its
simple-mode solver, not its transport cooling prescription.

Reproduction
------------

Build the Fortran extension, then run::

    python -m pytest tests
    python -m tests.numerical_audit

The audit reports perturbation ranges and local timings, without asserting a
portable runtime target. The radiation eigenmode test needs ``gfortran``;
JAX tests require the optional JAX dependencies.
