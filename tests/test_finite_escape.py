import numpy as np
import pytest

from redback_csm.core import _get_lc_static_powerlaw_csm_bpl
from tests.static_shell_reference import BASE, DAY, StaticShellReference


def test_fortran_finite_escape_at_emergence():
    ref=StaticShellReference()
    te=ref.te*ref.tc/DAY
    times=te+np.array([-.1,-.01,-.001,0.,.001,.01,.1])
    lc=_get_lc_static_powerlaw_csm_bpl(**BASE,kappa=.34)
    np.testing.assert_allclose(np.interp(times,lc.time/DAY,lc.lbol),ref.diffuse(times),rtol=3e-3)


def test_jax_finite_escape_at_emergence_and_grid_convergence():
    pytest.importorskip('jax')
    from jax_csm import StaticPowerlawBPLParams,solve_simple_static_bpl
    ref=StaticShellReference()
    te=ref.te*ref.tc/DAY
    times=te+np.array([-.1,-.01,-.001,0.,.001,.01,.1])
    expected=ref.diffuse(times)
    errors=[]
    for n in [2048,4096]:
        lc=solve_simple_static_bpl(times,StaticPowerlawBPLParams(**BASE,kappa=.34),n_steps=n,
                                   t_max_days=100.,diffusion=True)
        errors.append(np.max(np.abs(np.asarray(lc.lbol)/expected-1)))
    assert errors[-1]<3e-3
    assert errors[-1]<errors[0]


def test_legacy_kernel_is_explicit_and_runtime_is_reset():
    legacy=_get_lc_static_powerlaw_csm_bpl(**BASE,kappa=.34,diffusion_escape='legacy')
    finite=_get_lc_static_powerlaw_csm_bpl(**BASE,kappa=.34)
    assert finite.lbol[-1]>legacy.lbol[-1]
    with pytest.raises(ValueError,match='diffusion_escape'):
        _get_lc_static_powerlaw_csm_bpl(**BASE,kappa=.34,diffusion_escape='invalid')
