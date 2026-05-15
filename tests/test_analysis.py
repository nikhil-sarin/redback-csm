import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/redback_csm_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/redback_csm_xdg_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/redback_csm_numba_cache")

import numpy as np

from redback_csm.analysis import (
    csm_mass_from_density_grid,
    cumulative_csm_mass_profile,
    pspline_csm_mass_from_params,
    sample_geometry_corrected_mass,
)
from redback_csm.core import solar_mass


def test_csm_mass_from_density_grid_constant_shell():
    radius = np.linspace(1.0e14, 2.0e14, 1000)
    density = np.ones_like(radius) * 1.0e-16
    expected_cgs = 4.0 * np.pi * density[0] * (radius[-1] ** 3 - radius[0] ** 3) / 3.0

    mass = csm_mass_from_density_grid(radius, density, return_cgs=True)
    assert np.isclose(mass, expected_cgs, rtol=1.0e-5)

    mass_msun = csm_mass_from_density_grid(
        radius,
        density,
        covering_fraction=0.5,
        filling_factor=0.2,
    )
    assert np.isclose(mass_msun, 0.1 * expected_cgs / solar_mass, rtol=1.0e-5)


def test_cumulative_csm_mass_profile_is_monotonic():
    radius = np.linspace(1.0e14, 2.0e14, 50)
    density = np.linspace(2.0e-16, 1.0e-16, radius.size)
    _, cumulative = cumulative_csm_mass_profile(radius, density)

    assert cumulative[0] == 0.0
    assert np.all(np.diff(cumulative) >= 0.0)
    assert np.isclose(cumulative[-1], csm_mass_from_density_grid(radius, density))


def test_sample_geometry_corrected_mass_summary():
    result = sample_geometry_corrected_mass(
        10.0,
        covering_fraction=(0.2, 0.4),
        filling_factor={"kind": "fixed", "value": 0.5},
        n_samples=128,
        random_state=123,
    )

    samples = result["samples"]
    assert samples.shape == (128,)
    assert np.all(samples >= 1.0)
    assert np.all(samples <= 2.0)
    assert result["summary"]["lower"] <= result["summary"]["median"]
    assert result["summary"]["median"] <= result["summary"]["upper"]


def test_pspline_mass_from_params():
    params = {
        "log_r_inner": 14.0,
        "log_r_outer": 15.0,
        "log_rho_0": -16.0,
        "dlog_rho_0": 0.0,
        "interval_sn": 100.0,
    }
    params.update({f"d2_log_rho_{idx}": 0.0 for idx in range(22)})

    mass = pspline_csm_mass_from_params(params, profile="generic", n_points=128)
    assert np.isfinite(mass)
    assert mass > 0.0
