"""Blackbody photosphere helpers for the JAX CSM path."""

import jax.numpy as jnp

SIGMA_SB = 5.670374419e-5  # erg cm^-2 s^-1 K^-4


def photosphere_temperature(lbol, r_photosphere, temperature_floor=3000.0):
    """Effective temperature (L / 4 pi sigma R^2)^(1/4) with a temperature floor.

    Below the floor the temperature is held at ``temperature_floor`` and the radius is
    reduced to sqrt(L / 4 pi sigma T_floor^4), so L is preserved (same convention as
    ``redback_csm.models._apply_temperature_floor``).  Returns (r_photosphere, temperature).
    Pass ``temperature_floor=0`` to disable the floor.
    """
    lbol = jnp.maximum(jnp.asarray(lbol), 0.0)
    r = jnp.asarray(r_photosphere)
    ok = (lbol > 0.0) & (r > 0.0)
    temp = jnp.where(ok, (lbol / (4.0 * jnp.pi * SIGMA_SB * jnp.where(ok, r, 1.0) ** 2)) ** 0.25, 0.0)
    floor = jnp.asarray(temperature_floor)
    use_floor = ok & (floor > 0.0) & (temp < floor)
    r_floor = jnp.sqrt(lbol / (4.0 * jnp.pi * SIGMA_SB * jnp.where(floor > 0.0, floor, 1.0) ** 4))
    return jnp.where(use_floor, r_floor, r), jnp.where(use_floor, floor, temp)
