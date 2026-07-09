import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/redback_csm_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/redback_csm_xdg_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/redback_csm_numba_cache")

import matplotlib

matplotlib.use("Agg")

import numpy as np

from redback_csm.core import YEAR, solar_mass_per_yr_to_gram_per_sec
from redback_csm.explore import (
    csm_density_profile,
    mass_loss_history,
    plot_csm_density_profile,
    plot_mass_loss_history,
)


def test_boxwind_mass_loss_history_convention():
    times = np.array([0.5, 2.0, 5.0])
    mdot = mass_loss_history(
        "boxwind_bpl",
        times,
        t1=1.0,
        t2=3.0,
        mdot_0=1.0e-4,
        mdot_1=1.0e-2,
        mdot_2=2.0e-4,
    )

    assert np.allclose(mdot, [1.0e-4, 1.0e-2, 2.0e-4])


def test_boxwind_density_profile_matches_wind_formula():
    vwind = 100.0
    vwind_cgs = vwind * 1.0e5
    times_yr = np.array([0.5, 2.0, 5.0])
    radius = vwind_cgs * times_yr * YEAR

    rho = csm_density_profile(
        "boxwind_bpl",
        radius,
        t1=1.0,
        t2=3.0,
        mdot_0=1.0e-4,
        mdot_1=1.0e-2,
        mdot_2=2.0e-4,
        vwind=vwind,
    )

    mdot_cgs = np.array([1.0e-4, 1.0e-2, 2.0e-4]) * solar_mass_per_yr_to_gram_per_sec
    expected = mdot_cgs / (4.0 * np.pi * radius**2 * vwind_cgs)
    assert np.allclose(rho, expected)


def test_plot_mass_loss_history_matches_data_function():
    times = np.geomspace(1.0e-2, 5.0, 50)
    kwargs = dict(t1=1.0, t2=3.0, mdot_0=1.0e-4, mdot_1=1.0e-2, mdot_2=2.0e-4)

    expected = mass_loss_history("boxwind_bpl", times, **kwargs)
    fig, mdot = plot_mass_loss_history("boxwind_bpl", times, **kwargs)

    assert np.allclose(mdot, expected)
    assert len(fig.axes[0].lines) == 1


def test_plot_csm_density_profile_matches_data_function():
    vwind = 100.0
    radius = np.geomspace(1.0e14, 1.0e18, 50)
    kwargs = dict(t1=1.0, t2=3.0, mdot_0=1.0e-4, mdot_1=1.0e-2, mdot_2=2.0e-4, vwind=vwind)

    expected = csm_density_profile("boxwind_bpl", radius, **kwargs)
    fig, rho = plot_csm_density_profile("boxwind_bpl", radius, **kwargs)

    assert np.allclose(rho, expected)
    assert len(fig.axes[0].lines) == 1
