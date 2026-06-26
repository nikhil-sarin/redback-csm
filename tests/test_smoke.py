import importlib.util
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/redback_csm_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/redback_csm_xdg_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/redback_csm_numba_cache")

import numpy as np
import pytest

from redback_csm.models import (
    generic_csm_bpl,
    generic_pspline24_csm_bpl_bolometric,
    generic_pspline96_csm_bpl_bolometric,
    homologous_powerlaw_csm_bpl_bolometric,
    static_powerlaw_csm_bpl_bolometric,
    wind_bpl_bolometric,
    wind_bpl_nickel_bolometric,
    wind_bpl_radio,
    wind_bpl_xray,
)


def _assert_lightcurve(values, n):
    arr = np.asarray(values, dtype=float)
    assert arr.shape == (n,)
    assert np.all(np.isfinite(arr))
    assert np.nanmax(arr) >= 0.0


def test_wind_bpl_simple_and_legacy_diffusion_smoke():
    time = np.array([5.0, 20.0, 60.0])
    kwargs = dict(
        mdot=1.0e-3,
        vwind=100.0,
        delta=0.5,
        nn=10.0,
        mexp=5.0,
        eexp=1.0,
        eff=0.5,
    )

    direct = wind_bpl_bolometric(time=time, mode="simple", **kwargs)
    diffuse = wind_bpl_bolometric(time=time, mode="simple", kappa=0.34, **kwargs)

    _assert_lightcurve(direct, time.size)
    _assert_lightcurve(diffuse, time.size)


def test_low_level_csm_output_exposes_shock_radius():
    from redback_csm.core import _call_csm

    lc = _call_csm(
        "wind_bpl",
        mdot=1.0e-3,
        vwind=100.0,
        delta=0.5,
        nn=10.0,
        mexp=5.0,
        eexp=1.0,
        eff=0.5,
        mode="simple",
    )

    assert hasattr(lc, "rshock")
    assert lc.rshock.shape == lc.rph.shape
    assert np.all(np.isfinite(lc.rshock))
    assert np.allclose(lc.rshock, lc.rph)


def test_static_powerlaw_transport_smoke():
    time = np.array([3.0, 10.0, 30.0])
    lbol = static_powerlaw_csm_bpl_bolometric(
        time=time,
        eta=-2.0,
        r_inner=500.0 * 6.957e10,
        r_outer=5000.0 * 6.957e10,
        delta_sn=1.0,
        nn_sn=10.0,
        mej_sn=5.0,
        esn=1.0,
        eff=1.0,
        m_csm=1.0,
        mode="transport",
        efficiency_mode=1,
        kappa=0.2,
        n_rad_zones=8,
    )
    _assert_lightcurve(lbol, time.size)


def test_homologous_powerlaw_simple_smoke():
    time = np.array([3.0, 10.0, 30.0])
    lbol = homologous_powerlaw_csm_bpl_bolometric(
        time=time,
        eta=-2.0,
        r_inner=500.0 * 6.957e10,
        r_outer=5000.0 * 6.957e10,
        interval_sn=100.0,
        delta_sn=1.0,
        nn_sn=10.0,
        mej_sn=5.0,
        esn=1.0,
        eff=1.0,
        m_csm=1.0,
        mode="simple",
    )
    _assert_lightcurve(lbol, time.size)


def test_generic_pspline24_simple_smoke():
    time = np.array([5.0, 20.0, 60.0])
    kwargs = dict(
        log_r_inner=13.5,
        log_r_outer=16.0,
        log_rho_0=-13.0,
        dlog_rho_0=-0.05,
        interval_sn=300.0,
        delta_sn=1.0,
        nn_sn=10.0,
        mej_sn=5.0,
        esn=1.0,
        eff=0.5,
        mode="simple",
    )
    kwargs.update({f"d2_log_rho_{idx}": 0.0 for idx in range(22)})
    lbol = generic_pspline24_csm_bpl_bolometric(time=time, **kwargs)
    _assert_lightcurve(lbol, time.size)


def test_generic_pspline96_simple_smoke():
    time = np.array([5.0, 20.0, 60.0])
    kwargs = dict(
        log_r_inner=13.5,
        log_r_outer=16.0,
        log_rho_0=-13.0,
        dlog_rho_0=-0.02,
        interval_sn=300.0,
        delta_sn=1.0,
        nn_sn=10.0,
        mej_sn=5.0,
        esn=1.0,
        eff=0.5,
        mode="simple",
    )
    kwargs.update({f"d2_log_rho_{idx}": 0.0 for idx in range(94)})
    lbol = generic_pspline96_csm_bpl_bolometric(time=time, **kwargs)
    _assert_lightcurve(lbol, time.size)


def test_generic_csm_bpl_flux_density_uses_temperature_floor():
    time = np.linspace(1.0, 300.0, 32)
    kwargs = dict(
        redshift=0.166,
        base_density=6.89097955e-05,
        base_index=-3.14752479,
        shell1_radius=0.0,
        shell1_width=0.0,
        shell1_density=0.0,
        shell2_radius=0.0,
        shell2_width=0.0,
        shell2_density=0.0,
        shell3_radius=0.0,
        shell3_width=0.0,
        shell3_density=0.0,
        interval_sn=8.36509818,
        delta_sn=1.07327481,
        nn_sn=7.57092105,
        mej_sn=79.9611478,
        esn=11.0384479,
        eff=0.780317595,
        vej_max_ratio=4.39666448,
        kappa=0.136293163,
        shell_density=0.0,
        output_format="flux_density",
        frequency=1.0e15,
        bands="ztfg",
    )

    cold = generic_csm_bpl(time, temperature_floor=168.679970, **kwargs)
    warm = generic_csm_bpl(time, temperature_floor=3000.0, **kwargs)

    _assert_lightcurve(cold, time.size)
    _assert_lightcurve(warm, time.size)
    assert warm[-1] > cold[-1]


def test_nickel_radio_and_xray_wrappers_smoke():
    time = np.array([10.0, 30.0])
    common = dict(
        mdot=1.0e-3,
        vwind=100.0,
        delta=0.5,
        nn=10.0,
        mexp=5.0,
        eexp=1.0,
        eff=0.5,
    )

    nickel = wind_bpl_nickel_bolometric(
        time=time,
        f_nickel=0.02,
        kappa=0.34,
        kappa_gamma=0.027,
        **common,
    )
    radio = wind_bpl_radio(
        time=time,
        redshift=0.02,
        logepsb=-2.0,
        logepse=-1.0,
        p=3.0,
        frequency=5.0e9,
        **common,
    )
    xray = wind_bpl_xray(
        time=time,
        redshift=0.02,
        logepsx=-1.0,
        e_min_kev=0.3,
        e_max_kev=10.0,
        output_format="luminosity",
        **common,
    )

    _assert_lightcurve(nickel, time.size)
    _assert_lightcurve(radio, time.size)
    _assert_lightcurve(xray, time.size)


def test_radio_wrapper_accepts_per_point_frequency_array():
    time = np.linspace(5.0, 100.0, 10)
    frequency = np.linspace(3.0e9, 10.0e9, time.size)
    radio = wind_bpl_radio(
        time=time,
        redshift=0.02,
        logepsb=-2.0,
        logepse=-1.0,
        p=3.0,
        frequency=frequency,
        mdot=1.0e-3,
        vwind=100.0,
        delta=0.5,
        nn=10.0,
        mexp=5.0,
        eexp=1.0,
        eff=0.5,
    )

    _assert_lightcurve(radio, time.size)


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="jax is not installed")
def test_jax_static_powerlaw_smoke():
    from jax_csm.model import get_static_powerlaw_csm_bpl_lightcurve

    time = np.array([5.0, 20.0])
    lbol = get_static_powerlaw_csm_bpl_lightcurve(
        time=time,
        eta=-2.0,
        r_inner=500.0 * 6.957e10,
        r_outer=5000.0 * 6.957e10,
        delta_sn=1.0,
        nn_sn=10.0,
        mej_sn=5.0,
        esn=1.0,
        eff=1.0,
        m_csm=1.0,
        mode="transport",
        kappa=0.2,
        n_steps=256,
        n_rad_zones=8,
    )
    _assert_lightcurve(lbol, time.size)
