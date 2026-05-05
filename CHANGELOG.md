# Changelog

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
