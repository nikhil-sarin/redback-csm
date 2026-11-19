"""Simple-mode static power-law CSM + diffused radioactive nickel (JAX).

    L_bol(t) = L_CSM(t) + L_Ni,diffused(t)

* ``L_CSM`` is ``jax_csm.simple_shock.solve_simple_static_bpl``.  By default
  (``csm_diffusion=True``) it is the shock luminosity diffused through the unshocked CSM
  with ``params.kappa``, i.e. what the Fortran wrapper returns in simple mode when
  ``kappa`` is supplied (``get_diffuse_lc``) -- which the reference ``*_nickel_bolometric``
  wrappers always do.  ``csm_diffusion=False`` gives the instantaneous shock luminosity.
* ``L_Ni,diffused`` is ``jax_csm.nickel.nickel_diffused_luminosity`` with the
  redback_csm prescription: M_Ni = f_Ni * M_ej, diffusion mass M_ej + M_csm,
  v = sqrt(2E/M_ej), Arnett diffusion with partial gamma-ray trapping (kappa_gamma).
* ``params.kappa`` is used for both the CSM diffusion and the nickel diffusion time, as in
  the reference.  The nickel term is added; it does not heat or alter the CSM shell
  dynamics (same approximation as the reference).
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .nickel import nickel_diffused_luminosity
from .simple_shock import solve_simple_static_bpl


class CSMNickelResult(NamedTuple):
    time: jnp.ndarray
    lbol: jnp.ndarray
    lbol_csm: jnp.ndarray
    lbol_nickel: jnp.ndarray
    r_shock: jnp.ndarray
    v_shock: jnp.ndarray
    r_photosphere: jnp.ndarray  # max(r_shock, tau = 2/3 radius in the CSM ahead), before any floor
    t_emerge: jnp.ndarray


@partial(jax.jit, static_argnames=("n_steps", "nickel_quadrature", "csm_diffusion"))
def simple_csm_nickel_bolometric(time_days, params, f_nickel, kappa_gamma, mej_arnett=None, vej=None,
                                 n_steps=2048, t_max_days=None, nickel_quadrature="reference",
                                 csm_diffusion=True):
    time_days = jnp.asarray(time_days, jnp.float64)
    csm = solve_simple_static_bpl(time_days, params, n_steps=n_steps, t_max_days=t_max_days,
                                  diffusion=csm_diffusion)
    lni = nickel_diffused_luminosity(
        time_days, f_nickel, params.mej_sn, params.esn, params.m_csm, params.kappa, kappa_gamma,
        mej_arnett=mej_arnett, vej=vej, quadrature=nickel_quadrature,
    )
    return CSMNickelResult(time_days, csm.lbol + lni, csm.lbol, lni, csm.r_shock, csm.v_shock,
                           csm.r_photosphere, csm.t_emerge)
