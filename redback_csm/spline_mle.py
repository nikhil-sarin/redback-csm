"""Reusable maximum-likelihood helpers for spline CSM reconstructions.

The helper is designed for one-dimensional scalar light curves. The packaged
examples use bolometric luminosities; multiband fitting can be done by supplying
a model function and data vector that flatten all bands into one residual array.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from redback_csm.analysis import (
    generic_spline_csm_mass_from_params,
    spline_csm_mass,
)


def spline_parameter_names(
    n_nodes=8,
    profile="generic",
    include_time_offset=True,
    include_nickel=False,
):
    """Return the standard parameter ordering for spline CSM MLE fits."""
    if profile not in {"generic", "static"}:
        raise ValueError("profile must be 'generic' or 'static'")
    names = []
    if include_time_offset:
        names.append("t_explosion_offset")
    names.extend(["log_r_inner", "log_r_outer"])
    names.extend([f"log_rho_{idx}" for idx in range(int(n_nodes))])
    if profile == "generic":
        names.append("interval_sn")
    names.extend(["delta_sn", "nn_sn", "mej_sn", "esn", "eff"])
    if include_nickel:
        names.append("f_nickel")
    return names


def default_spline_bounds(
    n_nodes=8,
    profile="generic",
    include_time_offset=True,
    include_nickel=False,
):
    """
    Conservative default bounds for spline CSM optimization.

    These are intentionally broad and should usually be tightened for a specific
    transient once an approximate mode is known.
    """
    bounds = {}
    if include_time_offset:
        bounds["t_explosion_offset"] = (-200.0, 50.0)
    bounds["log_r_inner"] = (12.0, 15.5)
    bounds["log_r_outer"] = (14.5, 18.0)
    for idx in range(int(n_nodes)):
        bounds[f"log_rho_{idx}"] = (-22.0, -10.0)
    if profile == "generic":
        bounds["interval_sn"] = (1.0, 10000.0)
    elif profile != "static":
        raise ValueError("profile must be 'generic' or 'static'")
    bounds.update(
        {
            "delta_sn": (0.0, 3.0),
            "nn_sn": (6.0, 14.0),
            "mej_sn": (0.1, 50.0),
            "esn": (1.0e-3, 100.0),
            "eff": (0.01, 1.0),
        }
    )
    if include_nickel:
        bounds["f_nickel"] = (0.0, 1.0)
    return bounds


def vector_to_params(vector, parameter_names):
    """Convert an optimizer vector into a named parameter dictionary."""
    return {name: float(value) for name, value in zip(parameter_names, vector)}


def params_to_vector(params, parameter_names):
    """Convert a named parameter dictionary into an optimizer vector."""
    return np.asarray([params[name] for name in parameter_names], dtype=float)


def bounds_to_arrays(bounds, parameter_names):
    """Convert a bounds mapping into lower/upper arrays."""
    lower = np.asarray([bounds[name][0] for name in parameter_names], dtype=float)
    upper = np.asarray([bounds[name][1] for name in parameter_names], dtype=float)
    return lower, upper


def clip_to_bounds(vector, lower, upper):
    """Clip an optimizer vector strictly inside finite bounds."""
    vector = np.asarray(vector, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    eps = 1.0e-8 * np.maximum(1.0, upper - lower)
    return np.minimum(np.maximum(vector, lower + eps), upper - eps)


def spline_node_values(params: Mapping[str, float], n_nodes=None):
    """Return log-density node values from an MLE parameter dictionary."""
    if n_nodes is None:
        indices = sorted(
            int(key.rsplit("_", 1)[1])
            for key in params
            if key.startswith("log_rho_") and key.rsplit("_", 1)[1].isdigit()
        )
    else:
        indices = list(range(int(n_nodes)))
    if not indices:
        raise ValueError("no log_rho_* nodes were found")
    return np.asarray([params[f"log_rho_{idx}"] for idx in indices], dtype=float)


def spline_mass_from_params(params, profile="generic", n_points=1000):
    """Return the spherical-equivalent spline CSM mass in solar masses."""
    nodes = spline_node_values(params)
    if profile == "generic":
        return generic_spline_csm_mass_from_params(params, n_points=n_points)
    if profile == "static":
        return spline_csm_mass(
            params["log_r_inner"],
            params["log_r_outer"],
            nodes,
            n_points=n_points,
        )
    raise ValueError("profile must be 'generic' or 'static'")


def make_random_starts(
    start_params,
    bounds,
    parameter_names,
    n_random=0,
    seed=12345,
    log_rho_sigma=0.8,
):
    """Generate bounded optimizer starts around an existing parameter guess."""
    rng = np.random.default_rng(seed)
    lower, upper = bounds_to_arrays(bounds, parameter_names)
    base = clip_to_bounds(params_to_vector(start_params, parameter_names), lower, upper)
    starts = [base]
    for _ in range(int(n_random)):
        trial = base.copy()
        for idx, name in enumerate(parameter_names):
            width = upper[idx] - lower[idx]
            if name.startswith("log_rho_"):
                trial[idx] += rng.normal(0.0, log_rho_sigma)
            elif name in {"log_r_inner", "log_r_outer"}:
                trial[idx] += rng.normal(0.0, 0.08 * width)
            elif name in {"interval_sn", "mej_sn", "esn", "eff", "f_nickel"}:
                trial[idx] *= np.exp(rng.normal(0.0, 0.25))
            else:
                trial[idx] += rng.normal(0.0, 0.05 * width)
        starts.append(clip_to_bounds(trial, lower, upper))
    return starts


@dataclass
class SplineMLEResult:
    """Container for a spline MLE optimization result."""

    parameters: dict[str, float]
    chi2: float
    residual: np.ndarray
    optimizer_result: object


@dataclass
class SplineMLEProblem:
    """
    Least-squares problem for spline CSM light-curve reconstruction.

    ``model_function`` must accept ``(time, params)`` and return luminosity on
    the same grid. The intended packaged use is bolometric fitting. More general
    scalar data vectors are possible, but multiband fitting requires the caller
    to flatten the observations and return a matching flattened model vector.
    This keeps event-specific choices such as nickel components and time offsets
    outside the reusable optimizer.
    """

    time: np.ndarray
    luminosity: np.ndarray
    error: np.ndarray
    parameter_names: Sequence[str]
    bounds: Mapping[str, tuple[float, float]]
    model_function: Callable[[np.ndarray, Mapping[str, float]], np.ndarray]
    profile: str = "generic"
    smoothness_sigma: float = 0.0
    max_csm_mass: float | None = None
    n_mass_points: int = 1000

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        self.luminosity = np.asarray(self.luminosity, dtype=float)
        self.error = np.asarray(self.error, dtype=float)
        if self.time.shape != self.luminosity.shape or self.time.shape != self.error.shape:
            raise ValueError("time, luminosity, and error must have the same shape")
        if np.any(self.error <= 0.0) or not np.all(np.isfinite(self.error)):
            raise ValueError("error must be finite and positive")
        self.parameter_names = list(self.parameter_names)

    @property
    def lower_upper(self):
        return bounds_to_arrays(self.bounds, self.parameter_names)

    def vector_to_params(self, vector):
        return vector_to_params(vector, self.parameter_names)

    def params_to_vector(self, params):
        return params_to_vector(params, self.parameter_names)

    def residual(self, vector):
        params = self.vector_to_params(vector)
        model = np.asarray(self.model_function(self.time, params), dtype=float)
        extra = []
        if model.shape != self.time.shape or not np.all(np.isfinite(model)):
            return np.full(self.time.size + self._n_extra(), 1.0e6)
        if np.any(model <= 0.0):
            return np.full(self.time.size + self._n_extra(), 1.0e6)

        nodes = spline_node_values(params)
        if self.smoothness_sigma and self.smoothness_sigma > 0.0 and nodes.size > 2:
            extra.extend(np.diff(nodes, n=2) / float(self.smoothness_sigma))
        else:
            extra.extend(np.zeros(max(nodes.size - 2, 0)))

        if self.max_csm_mass is not None and self.max_csm_mass > 0.0:
            mass = spline_mass_from_params(
                params, profile=self.profile, n_points=self.n_mass_points
            )
            extra.append(max(mass / self.max_csm_mass - 1.0, 0.0) / 0.2)
        else:
            extra.append(0.0)
        return np.concatenate(((model - self.luminosity) / self.error, np.asarray(extra)))

    def _n_extra(self):
        n_nodes = sum(name.startswith("log_rho_") for name in self.parameter_names)
        return max(n_nodes - 2, 0) + 1

    def fit(self, starts, max_nfev=500, verbose=0, **least_squares_kwargs):
        """Run scipy least-squares from one or more starting vectors."""
        from scipy.optimize import least_squares

        lower, upper = self.lower_upper
        best = None
        for start in starts:
            start = clip_to_bounds(start, lower, upper)
            result = least_squares(
                self.residual,
                start,
                bounds=(lower, upper),
                max_nfev=max_nfev,
                verbose=verbose,
                **least_squares_kwargs,
            )
            chi2 = float(np.sum(self.residual(result.x) ** 2))
            if best is None or chi2 < best.chi2:
                best = SplineMLEResult(
                    parameters=self.vector_to_params(result.x),
                    chi2=chi2,
                    residual=self.residual(result.x),
                    optimizer_result=result,
                )
        return best
