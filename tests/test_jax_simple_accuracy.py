"""Independent accuracy contract shared with the JAX simple-mode PR.

Skipped on branches that do not yet provide that solver. Emergence itself is
excluded: the injection-time diffusion kernel has no finite continuum limit
there. See docs/numerical_accuracy.rst.
"""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
simple = pytest.importorskip('jax_csm.simple_shock')
from jax_csm import StaticPowerlawBPLParams
from tests.static_shell_reference import BASE, StaticShellReference


def test_jax_simple_against_independent_radau():
    jax.config.update('jax_enable_x64', True)
    reference = StaticShellReference()
    days = np.array([1.05, 1.2, 5, 10, 20, 40, 80])
    result = simple.solve_simple_static_bpl(
        days, StaticPowerlawBPLParams(**BASE, kappa=0.34),
        n_steps=2048, t_max_days=100., diffusion=True,
    )
    np.testing.assert_allclose(np.asarray(result.lbol_shock), reference.shock(days), rtol=5e-4)
    np.testing.assert_allclose(np.asarray(result.lbol), reference.diffuse(days), rtol=2e-3)
