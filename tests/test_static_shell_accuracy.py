"""Numerical regressions against Radau, rather than agreement of two backends."""

import numpy as np
import pytest

from redback_csm.core import (
    _get_lc_static_powerlaw_csm_bpl,
    _get_lc_static_powerlaw_csm_exponential,
    _get_lc_static_spline_csm_bpl,
)
from tests.static_shell_reference import BASE, DAY, StaticShellReference


@pytest.fixture(scope='module')
def reference():
    return StaticShellReference()


def test_reference_seed_convergence(reference):
    smaller = StaticShellReference(seed=1e-10)
    times = np.array([1.03, 1.05, 5, 20, 29, 40, 80])
    np.testing.assert_allclose(reference.shock(times), smaller.shock(times), rtol=2e-6)
    assert abs(reference.te-smaller.te)*reference.tc/DAY < 1e-5


def test_static_shell_against_radau(reference):
    lc = _get_lc_static_powerlaw_csm_bpl(**BASE, kappa=0.34)
    times = np.array([1.03, 1.05, 1.2, 2, 5, 10, 20, 27, 28, 29, 30, 40, 80])
    shock = np.interp(times, lc.time/DAY, lc.lbol_shock)
    diffuse = np.interp(times, lc.time/DAY, lc.lbol)
    np.testing.assert_allclose(shock, reference.shock(times), rtol=1e-3)
    np.testing.assert_allclose(diffuse, reference.diffuse(times), rtol=3e-3)
    assert abs(lc.time[0]-reference.tc) < 1e-7
    assert lc.lbol[0] == 0
    assert np.all(np.diff(lc.time) > 0)
    edge = np.where(lc.rshock >= BASE['r_outer'])[0][0]
    assert abs(lc.time[edge]/DAY-reference.te*reference.tc/DAY) < 3e-4


def test_static_shell_ulp_smoothness():
    times = np.array([1.018, 1.025, 1.03, 1.05, 1.2, 29, 40])
    curves = []
    for offset in range(301):
        p = dict(BASE, esn=1+offset*np.spacing(1.0))
        lc = _get_lc_static_powerlaw_csm_bpl(**p, kappa=0.34)
        curves.append(np.interp(times, lc.time/DAY, lc.lbol))
    mags = -2.5*np.log10(curves)
    assert np.max(np.ptp(mags, axis=0)) < 1e-5


@pytest.mark.parametrize('profile', ['exponential', 'spline'])
def test_other_static_drivers_ulp_smoothness(profile):
    if profile == 'exponential':
        fn = _get_lc_static_powerlaw_csm_exponential
        params = {k: v for k, v in BASE.items() if k not in ['delta_sn', 'nn_sn']}
    else:
        fn = _get_lc_static_spline_csm_bpl
        params = {k: BASE[k] for k in ['delta_sn', 'nn_sn', 'mej_sn', 'esn', 'eff']}
        params.update(log_r_inner=14., log_r_outer=15.)
        params.update({f'log_rho_{i}': -12.05-2*i/7 for i in range(8)})
    curves = []
    for offset in range(0, 301, 10):
        lc = fn(**dict(params, esn=1+offset*np.spacing(1.0)), kappa=0.34)
        curves.append(np.interp([1.03, 1.05, 5, 20, 29, 40], lc.time/DAY, lc.lbol))
    assert np.all(np.asarray(curves) > 0)
    assert np.max(np.ptp(-2.5*np.log10(curves), axis=0)) < 1e-5


def test_static_diffusion_independent_of_output_thinning(monkeypatch):
    monkeypatch.delenv('REDBACK_CSM_SIMPLE_DENSE_OUTPUT', raising=False)
    sparse = _get_lc_static_powerlaw_csm_bpl(**BASE, kappa=0.34)
    monkeypatch.setenv('REDBACK_CSM_SIMPLE_DENSE_OUTPUT', '1')
    dense = _get_lc_static_powerlaw_csm_bpl(**BASE, kappa=0.34)
    np.testing.assert_array_equal(sparse.time, dense.time)
    np.testing.assert_array_equal(sparse.lbol, dense.lbol)
