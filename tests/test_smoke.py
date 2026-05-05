import importlib.util
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/redback_csm_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/redback_csm_xdg_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/redback_csm_numba_cache")

import numpy as np
import pytest

from redback_csm.models import (
    generic_powerlaw_csm_bpl_bolometric,
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


def test_static_powerlaw_transport_smoke():
    time = np.array([3.0, 10.0, 30.0])
    lbol = generic_powerlaw_csm_bpl_bolometric(
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
