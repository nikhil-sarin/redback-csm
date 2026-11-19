"""JAX radioactive-nickel component matched to ``redback_csm.models._csm_nickel_bolometric_impl``.

Reference prescription (default ``interaction_process=redback.interaction_processes.Diffusion``):

* Heating: ``redback...supernova_models._nickelcobalt_engine`` (Nadyozhin 1994)
      L_in(t) = M_Ni [6.45e43 exp(-t/8.8 d) + 1.45e43 exp(-t/111.3 d)]  erg/s,  M_Ni = f_Ni * M_ej
  where M_ej is the *SN ejecta* mass (``mej_sn``) and M_Ni is in Msun.
* Diffusion (Arnett 1982, redback ``Diffusion``), with the *diffusion mass*
      M_diff = M_ej + M_csm    (``mej_arnett`` overrides)
  and velocity v = sqrt(2 E / M_ej) [km/s] (``vej`` overrides), using
  redback.constants.solar_mass = 1.98841e33 g for this velocity only:
      tau_m^2 = 2 Msun kappa M_diff / (13.7 c v)            (days^2 after / 86400^2)
      T0^2    = 3 Msun kappa_gamma M_diff / (4 pi v^2)      (gamma-ray trapping)
      L(t) = 2/tau_m^2 * [1 - exp(-T0^2 / t^2)] * int_0^t L_in(t') t' exp((t'^2 - t^2)/tau_m^2) dt'
  i.e. *partial* gamma-ray trapping; there is no full-trapping assumption.
* The diffused nickel luminosity is simply *added* to the CSM luminosity.  It does not
  heat or otherwise modify the CSM shell dynamics.  (This JAX port keeps that
  approximation; the nickel term is independent of the shock solution.)

``quadrature="reference"`` reproduces redback's numerical quadrature bit-for-bit in
structure: the heating rate is tabulated on ``linspace(0, max(time)+100, 1000)`` and
linearly interpolated, and the integral uses redback's 100-point log-clustered grid
whose clustering depends on tau_m / (max(time)+100).  The result therefore depends
weakly on the latest requested epoch.  ``quadrature="converged"`` evaluates the
analytic heating rate on a fixed, finer grid (independent of the requested epochs);
the difference between the two quantifies redback's own quadrature error.
"""

from functools import partial

import jax
import jax.numpy as jnp

REDBACK_SOLAR_MASS = 1.988409870698051e33  # redback.constants.solar_mass (astropy M_sun)
CLIGHT = 2.99792458e10
KM_CGS = 1.0e5
DAY_S = 86400.0
NI56_LUM, CO56_LUM = 6.45e43, 1.45e43
NI56_LIFE, CO56_LIFE = 8.8, 111.3


def nickel_heating(time_days, m_ni_msun):
    """Nadyozhin (1994) Ni+Co heating rate exactly as redback's _nickelcobalt_engine."""
    t = jnp.asarray(time_days)
    return m_ni_msun * (NI56_LUM * jnp.exp(-t / NI56_LIFE) + CO56_LUM * jnp.exp(-t / CO56_LIFE))


def arnett_constants(m_diff_msun, v_ej_kms, kappa, kappa_gamma):
    """(tau_m [days], T0^2 [days^2]) as in redback Diffusion.convert_input_luminosity."""
    diffusion_constant = 2.0 * REDBACK_SOLAR_MASS / (13.7 * CLIGHT * KM_CGS)
    trapping_constant = 3.0 * REDBACK_SOLAR_MASS / (4.0 * jnp.pi * KM_CGS**2)
    tau_m = jnp.sqrt(diffusion_constant * kappa * m_diff_msun / v_ej_kms) / DAY_S
    trap = trapping_constant * kappa_gamma * m_diff_msun / v_ej_kms**2 / DAY_S**2
    return tau_m, trap


def default_nickel_velocity_kms(mej_sn, esn):
    """Reference default: v = sqrt(2 E / M_ej) with M_ej the SN ejecta mass."""
    return jnp.sqrt(2.0 * esn * 1.0e51 / (mej_sn * REDBACK_SOLAR_MASS)) / 1.0e5


def _reference_xm(tau_m, t_dense_end, timesteps=100, minimum_log_spacing=-3):
    num = int(round(timesteps / 2.0))
    lsp = jnp.logspace(jnp.log10(tau_m / t_dense_end) + minimum_log_spacing, 0.0, num)
    # redback uses np.unique(concat(lsp, 1 - lsp)); sorting is equivalent for a
    # trapezoid rule (an exact duplicate would add a zero-width interval).
    return jnp.sort(jnp.concatenate([lsp, 1.0 - lsp]))


def _diffuse(time_days, lum_at, tau_m, trap, xm, t_hi):
    t = jnp.asarray(time_days)
    tc = jnp.clip(t, 0.0, t_hi)
    int_t = jnp.clip(tc[:, None] * xm[None, :], 0.0, t_hi)
    te2 = int_t[:, -1] ** 2
    args = lum_at(int_t) * int_t * jnp.exp((int_t**2 - te2[:, None]) / tau_m**2)
    args = jnp.nan_to_num(args, nan=0.0)
    integral = jnp.trapezoid(args, int_t, axis=1)
    safe_te2 = jnp.where(te2 > 0.0, te2, 1.0)
    leak = jnp.where(te2 > 0.0, -jnp.expm1(-trap / safe_te2), 1.0)
    out = integral * 2.0 * leak / tau_m**2
    # redback: times outside [0, dense_end] are not evaluated (inputs are always inside).
    return jnp.where(t >= 0.0, out, 0.0)


@partial(jax.jit, static_argnames=("quadrature", "dense_resolution", "n_converged"))
def nickel_diffused_luminosity(time_days, f_nickel, mej_sn, esn, m_csm, kappa, kappa_gamma,
                               mej_arnett=None, vej=None, quadrature="reference",
                               dense_resolution=1000, n_converged=2000):
    """Diffused radioactive luminosity [erg/s] matching the redback_csm nickel prescription.

    ``mej_sn``/``m_csm`` in Msun, ``esn`` in 1e51 erg, ``kappa``/``kappa_gamma`` in cm^2/g,
    ``vej`` (optional override) in km/s, ``time_days`` source-frame days.
    """
    time_days = jnp.asarray(time_days, jnp.float64)
    m_ni = f_nickel * mej_sn
    m_diff = (mej_sn + m_csm) if mej_arnett is None else mej_arnett
    v = default_nickel_velocity_kms(mej_sn, esn) if vej is None else vej
    tau_m, trap = arnett_constants(m_diff, v, kappa, kappa_gamma)
    if quadrature == "reference":
        t_hi = jnp.max(time_days) + 100.0
        dense_t = jnp.linspace(0.0, t_hi, dense_resolution)
        dense_l = nickel_heating(dense_t, m_ni)
        lum_at = lambda tt: jnp.interp(tt, dense_t, dense_l)
        xm = _reference_xm(tau_m, t_hi)
        return _diffuse(time_days, lum_at, tau_m, trap, xm, t_hi)
    if quadrature == "converged":
        # Fixed log-clustered grid at both ends (independent of requested epochs),
        # analytic heating rate.  Clustering scale tied to tau_m only.
        # Uniform points plus geometric clustering towards x = t'/t = 1, where the
        # Arnett kernel exp((x^2 - 1) t^2 / tau_m^2) is concentrated at late times.
        half = n_converged // 2
        xm = jnp.sort(jnp.concatenate([jnp.linspace(0.0, 1.0, half), 1.0 - jnp.logspace(-14.0, -0.3, half)]))
        lum_at = lambda tt: nickel_heating(tt, m_ni)
        return _diffuse(time_days, lum_at, tau_m, trap, xm, jnp.inf)
    raise ValueError("quadrature must be 'reference' or 'converged'")
