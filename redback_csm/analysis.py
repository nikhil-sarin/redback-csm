"""Analysis helpers for interpreting CSM density reconstructions.

The light-curve solvers fit density fields. The masses returned here are
therefore spherical-equivalent masses unless a covering or filling factor is
explicitly applied.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from redback_csm.core import (
    create_static_spline_csm_density,
    create_generic_spline_csm_density,
    pspline_log_rho_nodes,
    solar_mass,
)

_TRAPEZOID = getattr(np, "trapezoid", np.trapz)


def _as_sorted_density_grid(radius_cgs, density_cgs):
    radius = np.asarray(radius_cgs, dtype=float)
    density = np.asarray(density_cgs, dtype=float)
    if radius.ndim != 1 or density.ndim != 1:
        raise ValueError("radius_cgs and density_cgs must be one-dimensional arrays")
    if radius.size != density.size:
        raise ValueError("radius_cgs and density_cgs must have the same length")
    if radius.size < 2:
        raise ValueError("at least two radial grid points are required")
    if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(density)):
        raise ValueError("radius_cgs and density_cgs must be finite")
    if np.any(radius <= 0.0):
        raise ValueError("radius_cgs must be positive")

    order = np.argsort(radius)
    radius = radius[order]
    density = np.maximum(density[order], 0.0)
    if np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius_cgs must contain unique radii")
    return radius, density


def _validate_fraction(value, name):
    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    if np.any((value < 0.0) | (value > 1.0)):
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def csm_mass_from_density_grid(
    radius_cgs,
    density_cgs,
    covering_fraction=1.0,
    filling_factor=1.0,
    return_cgs=False,
):
    """
    Integrate CSM mass from an arbitrary static density grid.

    Parameters
    ----------
    radius_cgs, density_cgs : array-like
        Radius in cm and density in g cm^-3.
    covering_fraction, filling_factor : float
        Multiplicative geometry factors. The default returns the
        spherical-equivalent mass.
    return_cgs : bool
        If true, return grams. Otherwise return solar masses.
    """
    radius, density = _as_sorted_density_grid(radius_cgs, density_cgs)
    covering_fraction = _validate_fraction(covering_fraction, "covering_fraction")
    filling_factor = _validate_fraction(filling_factor, "filling_factor")
    mass_cgs = _TRAPEZOID(4.0 * np.pi * radius**2 * density, radius)
    mass_cgs = max(float(mass_cgs), 0.0) * float(covering_fraction) * float(filling_factor)
    if return_cgs:
        return mass_cgs
    return mass_cgs / solar_mass


def cumulative_csm_mass_profile(
    radius_cgs,
    density_cgs,
    covering_fraction=1.0,
    filling_factor=1.0,
    return_cgs=False,
):
    """Return cumulative mass as a function of radius for a density grid."""
    radius, density = _as_sorted_density_grid(radius_cgs, density_cgs)
    covering_fraction = float(_validate_fraction(covering_fraction, "covering_fraction"))
    filling_factor = float(_validate_fraction(filling_factor, "filling_factor"))

    integrand = 4.0 * np.pi * radius**2 * density
    cumulative = np.zeros_like(radius)
    dr = np.diff(radius)
    shell_mass = 0.5 * (integrand[1:] + integrand[:-1]) * dr
    cumulative[1:] = np.cumsum(shell_mass)
    cumulative *= covering_fraction * filling_factor
    if not return_cgs:
        cumulative /= solar_mass
    return radius, cumulative


def collect_log_rho_nodes(params: Mapping[str, float], prefix="log_rho_"):
    """Collect consecutively named log-density nodes from a parameter mapping."""
    node_indices = sorted(
        int(key.rsplit("_", 1)[1])
        for key in params
        if key.startswith(prefix) and key.rsplit("_", 1)[1].isdigit()
    )
    if not node_indices:
        raise ValueError(f"no {prefix}* nodes were found")
    return np.asarray([params[f"{prefix}{idx}"] for idx in node_indices], dtype=float)


def spline_csm_mass(
    log_r_inner,
    log_r_outer,
    log_rho_nodes,
    log_r_nodes=None,
    n_points=1000,
    covering_fraction=1.0,
    filling_factor=1.0,
    return_cgs=False,
):
    """Mass of a finite spline CSM density profile."""
    radius, density = create_static_spline_csm_density(
        log_r_inner=log_r_inner,
        log_r_outer=log_r_outer,
        log_rho_nodes=log_rho_nodes,
        log_r_nodes=log_r_nodes,
        n_points=n_points,
    )
    return csm_mass_from_density_grid(
        radius,
        density,
        covering_fraction=covering_fraction,
        filling_factor=filling_factor,
        return_cgs=return_cgs,
    )


def generic_spline_csm_mass(
    log_r_inner,
    log_r_outer,
    log_rho_nodes,
    interval_sn,
    log_r_nodes=None,
    n_points=1000,
    covering_fraction=1.0,
    filling_factor=1.0,
    return_cgs=False,
):
    """
    Mass of a homologous spline CSM density profile.

    The mass integral uses the density snapshot. ``interval_sn`` is included so
    the same inputs can be passed as the homologous spline light-curve model.
    """
    radius, _, density = create_generic_spline_csm_density(
        log_r_inner=log_r_inner,
        log_r_outer=log_r_outer,
        log_rho_nodes=log_rho_nodes,
        interval_sn=interval_sn,
        log_r_nodes=log_r_nodes,
        n_points=n_points,
    )
    return csm_mass_from_density_grid(
        radius,
        density,
        covering_fraction=covering_fraction,
        filling_factor=filling_factor,
        return_cgs=return_cgs,
    )


def generic_spline_csm_mass_from_params(
    params: Mapping[str, float],
    n_points=1000,
    covering_fraction=1.0,
    filling_factor=1.0,
    return_cgs=False,
):
    """Mass helper for dictionaries containing ``log_r_*`` and ``log_rho_*``."""
    return generic_spline_csm_mass(
        log_r_inner=params["log_r_inner"],
        log_r_outer=params["log_r_outer"],
        log_rho_nodes=collect_log_rho_nodes(params),
        interval_sn=params.get("interval_sn", 10.0 * 365.25),
        n_points=n_points,
        covering_fraction=covering_fraction,
        filling_factor=filling_factor,
        return_cgs=return_cgs,
    )


def collect_pspline_log_rho_nodes(params: Mapping[str, float]):
    """Reconstruct log-density nodes from p-spline coefficient parameters."""
    curvature_indices = sorted(
        int(key.rsplit("_", 1)[1])
        for key in params
        if key.startswith("d2_log_rho_") and key.rsplit("_", 1)[1].isdigit()
    )
    if not curvature_indices:
        raise ValueError("no d2_log_rho_* p-spline curvature parameters were found")
    d2_nodes = np.asarray(
        [params[f"d2_log_rho_{idx}"] for idx in curvature_indices], dtype=float
    )
    return pspline_log_rho_nodes(params["log_rho_0"], params["dlog_rho_0"], d2_nodes)


def pspline_csm_mass_from_params(
    params: Mapping[str, float],
    profile="generic",
    n_points=1000,
    covering_fraction=1.0,
    filling_factor=1.0,
    return_cgs=False,
):
    """Mass helper for dictionaries containing p-spline CSM parameters."""
    log_rho_nodes = collect_pspline_log_rho_nodes(params)
    if profile == "generic":
        return generic_spline_csm_mass(
            log_r_inner=params["log_r_inner"],
            log_r_outer=params["log_r_outer"],
            log_rho_nodes=log_rho_nodes,
            interval_sn=params.get("interval_sn", 10.0 * 365.25),
            n_points=n_points,
            covering_fraction=covering_fraction,
            filling_factor=filling_factor,
            return_cgs=return_cgs,
        )
    if profile == "static":
        return spline_csm_mass(
            log_r_inner=params["log_r_inner"],
            log_r_outer=params["log_r_outer"],
            log_rho_nodes=log_rho_nodes,
            n_points=n_points,
            covering_fraction=covering_fraction,
            filling_factor=filling_factor,
            return_cgs=return_cgs,
        )
    raise ValueError("profile must be 'generic' or 'static'")


def geometry_corrected_mass(
    spherical_mass,
    covering_fraction=1.0,
    filling_factor=1.0,
):
    """Apply covering and filling factors to a spherical-equivalent mass."""
    covering_fraction = _validate_fraction(covering_fraction, "covering_fraction")
    filling_factor = _validate_fraction(filling_factor, "filling_factor")
    return np.asarray(spherical_mass, dtype=float) * covering_fraction * filling_factor


def _sample_fraction_prior(prior, size, rng, name):
    if prior is None:
        return np.ones(size, dtype=float)
    if np.isscalar(prior):
        return np.full(size, float(_validate_fraction(prior, name)), dtype=float)
    if hasattr(prior, "sample"):
        samples = np.asarray(prior.sample(size), dtype=float)
        return _validate_fraction(samples, name)
    if isinstance(prior, Mapping):
        kind = str(prior.get("kind", prior.get("type", "uniform"))).lower()
        if kind == "fixed":
            samples = np.full(size, float(prior["value"]), dtype=float)
        elif kind == "uniform":
            low = float(prior.get("min", prior.get("low", 0.0)))
            high = float(prior.get("max", prior.get("high", 1.0)))
            samples = rng.uniform(low, high, size)
        elif kind in {"loguniform", "log_uniform"}:
            low = float(prior.get("min", prior.get("low")))
            high = float(prior.get("max", prior.get("high")))
            if low <= 0.0 or high <= low:
                raise ValueError(f"{name} log-uniform prior needs 0 < min < max")
            samples = np.exp(rng.uniform(np.log(low), np.log(high), size))
        elif kind == "beta":
            alpha = float(prior.get("alpha", prior.get("a")))
            beta = float(prior.get("beta", prior.get("b")))
            samples = rng.beta(alpha, beta, size)
        else:
            raise ValueError(f"unsupported {name} prior kind: {kind}")
        return _validate_fraction(samples, name)
    if isinstance(prior, (tuple, list)) and len(prior) == 2:
        low, high = map(float, prior)
        samples = rng.uniform(low, high, size)
        return _validate_fraction(samples, name)

    samples = np.asarray(prior, dtype=float)
    if samples.ndim != 1:
        raise ValueError(f"{name} prior samples must be one-dimensional")
    if samples.size == size:
        return _validate_fraction(samples, name)
    choices = rng.choice(samples, size=size, replace=True)
    return _validate_fraction(choices, name)


def summarize_samples(samples, credible_interval=0.68):
    """Return median and central credible interval for a sample array."""
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("samples must be finite and non-empty")
    alpha = 0.5 * (1.0 - float(credible_interval))
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return {
        "median": float(np.median(samples)),
        "lower": float(lo),
        "upper": float(hi),
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
    }


def sample_geometry_corrected_mass(
    spherical_mass,
    covering_fraction=1.0,
    filling_factor=1.0,
    n_samples=10000,
    random_state=None,
    credible_interval=0.68,
):
    """
    Draw geometry-corrected CSM masses from simple covering/filling priors.

    Priors may be scalars, ``(min, max)`` tuples for uniform priors, bilby-like
    objects with a ``sample`` method, one-dimensional sample arrays, or dicts:
    ``{"kind": "uniform", "min": 0.1, "max": 1}``,
    ``{"kind": "loguniform", "min": 0.01, "max": 1}``, or
    ``{"kind": "beta", "alpha": 2, "beta": 5}``.
    """
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(random_state)
    covering_samples = _sample_fraction_prior(
        covering_fraction, n_samples, rng, "covering_fraction"
    )
    filling_samples = _sample_fraction_prior(
        filling_factor, n_samples, rng, "filling_factor"
    )
    mass_samples = geometry_corrected_mass(
        spherical_mass,
        covering_fraction=covering_samples,
        filling_factor=filling_samples,
    )
    return {
        "samples": mass_samples,
        "covering_fraction": covering_samples,
        "filling_factor": filling_samples,
        "summary": summarize_samples(mass_samples, credible_interval=credible_interval),
    }
