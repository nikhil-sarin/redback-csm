# Changelog

## Unreleased

- JAX `mode="simple"` (static power-law CSM + BPL ejecta) now uses a new
  solver, `jax_csm.simple_shock.solve_simple_static_bpl`, that follows the
  Fortran simple mode. The previous path is kept as `mode="simple_legacy"`;
  it started from a heavy seed shell (u = 0.9 v_max,
  m = 1e-4 (M_ej + M_csm) Msun), which biased the first days after collision,
  and froze the shell after it left the CSM, which made the tail too bright.
  JAX simple-mode results change accordingly. The new path uses 1.989e33 g per
  Msun like the Fortran wrapper and returns `r_shock`, `v_shock`, `lbol_fs`,
  `lbol_rs`, `tau_ahead` and `t_emerge`. The shock radius is not a
  photospheric radius.
- `solve_simple_static_bpl(..., diffusion=True)` adds the simple-mode diffusion
  through the unshocked CSM that the Fortran applies when `kappa` is passed
  (`get_diffuse_lc`), in continuous form. Close to shock emergence this
  prescription rises logarithmically, so the value right at emergence depends
  on the time resolution.
- Added `jax_csm.nickel.nickel_diffused_luminosity` (the Arnett diffusion term
  of the `*_nickel_bolometric` models) and
  `jax_csm.csm_nickel.simple_csm_nickel_bolometric` (diffused CSM + nickel).

- Fixed radio models returning the same flux for every frequency at a shared
  epoch when a per-point `frequency` array contained repeated times.
- Fixed radio and X-ray post-processing for homologously expanding CSM
  (`generic_*` shell/spline/p-spline and `homologous_powerlaw_*` models): the
  upstream density now follows the hydro's expansion,
  `rho0(r/s)/s^3` with `s = 1 + t/interval_sn`, instead of the
  explosion-time snapshot.
- Radio magnetic/electron energy densities and the X-ray shock temperature now
  use the shock velocity relative to the upstream CSM, matching the hydro.
  This also changes wind and previous-explosion CSM models slightly (by of
  order `v_csm/v_shock`).
- Variable-wind radio/X-ray density lookups now include the time since
  explosion when finding the wind emission epoch, matching the Fortran, and
  the smooth triple power-law winds now use the same tanh-smoothed mass-loss
  history as the hydro (previously a sharp-break approximation).
- Wind-driven models (`exponential_wind`, `bpl_wind`,
  `exponential_triple_powerlaw_wind`, `bpl_triple_powerlaw_wind`) now treat
  the outer explosion ejecta as the upstream medium for radio/X-ray, matching
  the Fortran (previously the inner wind's density was used).
- X-ray emission-measure normalization for homologous CSM now uses the exact
  swept-up mass of the expanding snapshot. For other moving-CSM models the
  swept mass is still the trajectory-integrated approximation.
- Fixed `exponential_wind_*` models raising `NameError: mode`.
- Split an over-long line in `fortran/csm.f90` that broke compilation under
  strict gfortran flags (`-Werror=line-truncation`), and added a test that
  keeps Fortran sources within 132 columns (#3).
- Use `scipy.integrate.trapezoid` instead of `np.trapz`/`np.trapezoid`, which
  failed on NumPy versions without `np.trapz` (#4, thanks @cgobat).
- Removed an invalid escape sequence from a `core.py` docstring.
- Added optional radio spectral-shape keywords (all off by default; output is
  unchanged when unset): synchrotron cooling break (`cooling`,
  `log_nu_c_scale`), a broadened SSA turnover (`ssa_alpha_thick`,
  `ssa_smoothing`) and external free-free absorption (`log_tau_ff`,
  `ff_time_index`, `ff_nu_ref_ghz`, `ff_t_ref_days`), with default priors.
- Added `examples/20_multifrequency_radio_fit.py`, a multi-frequency radio
  fit using a per-point frequency array.

## 0.1.0 release candidate

- Renamed the public runtime modes to `simple` and `transport`; removed stale
  hybrid wording from user-facing docs and code comments.
- Documented the legacy `simple` + `kappa` diffusion kernel separately from the
  radiation-transport solver.
- Added public static finite power-law transport and X-ray example scripts.
- Added transport validation scripts for Fig. 11-style light curves,
  resolution checks, JAX-vs-Fortran comparisons, and runtime summaries.
- Added a JAX static finite power-law CSM backend for fast inference
  experiments, scoped to the static power-law/BPL model family.
- Added thermal bremsstrahlung X-ray wrappers and an example plot script.
- Added smoke tests covering simple mode, legacy diffusion, transport,
  nickel, radio, X-rays, and the optional JAX backend.
- Removed obsolete experimental JAX modules that were no longer part of the
  public backend.
