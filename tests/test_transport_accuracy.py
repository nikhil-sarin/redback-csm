"""Verify the actual Fortran PDE against a continuum eigenmode and energy balance."""

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from scipy.optimize import brentq


@pytest.fixture(scope='module')
def transport_probe(tmp_path_factory):
    compiler = shutil.which('gfortran')
    if compiler is None:
        pytest.skip('gfortran needed for standalone transport verification')
    root = Path(__file__).resolve().parents[1]
    tmp = tmp_path_factory.mktemp('transport_probe')
    sources = ['constants', 'get_vals', 'integration', 'csm_runtime', 'csm_transport']
    subprocess.run([compiler, '-O2', '-o', str(tmp/'probe'),
                    *[str(root/'fortran'/f'{name}.f90') for name in sources],
                    str(root/'tests'/'transport_probe.f90')], cwd=tmp, check=True, capture_output=True)
    return tmp/'probe'


def test_diffusion_eigenmode_and_energy_conservation(transport_probe):
    # e=f(r)exp(-k^2 y), f'(1)=0, f(2)+f'(2)=0; beta=1.
    def robin(k):
        g = np.cos(k)+np.sin(k)/k
        gp = -k*np.sin(k)+np.cos(k)
        return gp+0.5*g
    k = brentq(robin, 0.1, 2.0)
    errors = []
    for n in [21, 41, 81]:
        out = subprocess.check_output([str(transport_probe), str(n), '0.0001', '1000', str(k)], text=True)
        rows = np.array([[float(x) for x in line.split()] for line in out.splitlines()])
        errors.append(rows[0, 1])
        assert rows[0, 3] >= 0
        assert rows[1, 3] >= 0
        np.testing.assert_allclose(rows[1, 2], 0.1, rtol=2e-10)
    assert errors[-1] < 5e-5
    assert errors[0]/errors[1] > 3.5
    assert errors[1]/errors[2] > 3.5


def test_transport_dynamics_and_radiation_resolution():
    from redback_csm.core import _get_lc_static_powerlaw_csm_bpl
    from tests.static_shell_reference import BASE, DAY, StaticShellReference

    epochs = np.array([5., 10., 20., 28., 30., 40., 60., 90., 150.])
    curves = []
    reference = StaticShellReference()
    for zones in [80, 160, 320, 640]:
        lc = _get_lc_static_powerlaw_csm_bpl(
            **BASE, mode='transport', kappa=0.34, n_rad_zones=zones)
        assert np.all(np.isfinite(lc.lbol))
        assert np.all(lc.lbol >= 0)
        curves.append(np.interp(epochs, lc.time/DAY, lc.lbol))
        times = np.array([2., 5., 10., 20.])
        np.testing.assert_allclose(np.interp(times, lc.time/DAY, lc.lbol_shock),
                                   reference.shock(times), rtol=1e-4)
    curves = np.asarray(curves)
    errors = np.max(np.abs(curves[:-1]-curves[-1]), axis=1)/np.max(curves[-1])
    assert np.all(np.diff(errors) < 0)
    assert errors[1] < 0.003
    # Relative errors in the nearly dark phase are not a useful bright-phase
    # criterion. Still require convergence of its absolute luminosity above.
    np.testing.assert_allclose(curves[1, 1:7], curves[-1, 1:7], rtol=0.01)


def test_jax_radiation_operator_against_eigenmode():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    from jax_csm.static_powerlaw import (
        StaticPowerlawBPLParams, _dimless_scales, _solve_diffusion_dimless,
    )
    from tests.static_shell_reference import BASE

    params = StaticPowerlawBPLParams(**dict(BASE, eta=0.), kappa=.34)
    scales = _dimless_scales(params)._replace(tau_csm_in=4., x_out=2.)
    k = brentq(lambda k: -k*np.sin(k)+np.cos(k)
               +.5*(np.cos(k)+np.sin(k)/k), .1, 2.)
    x = jnp.linspace(1., 2., 81)
    initial = (jnp.cos(k*(x-1))+jnp.sin(k*(x-1))/k)/x

    @jax.jit
    def evolve(e):
        return jax.lax.fori_loop(0, 1000, lambda _, e: _solve_diffusion_dimless(
            params, scales, e, .0001, 1., 2., 1., 0., 0., 0., False,
            1., 2., 2., 0), e)

    np.testing.assert_allclose(evolve(initial), initial*np.exp(-k*k*.1), atol=5e-5, rtol=0.)
