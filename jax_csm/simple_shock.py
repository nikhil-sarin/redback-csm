"""JAX simple-mode (instantaneous shock emission) solver for a static power-law CSM shell.

This reproduces the Fortran reference ``lightcurve_static_bpl`` in run mode 1
*without* ``kappa`` (``redback_csm.core._get_lc_static_powerlaw_csm_bpl`` with
``mode="simple"`` and no opacity), i.e.

    L(t) = eff * [ 1/2 4 pi r^2 rho_csm u^3  +  1/2 4 pi r^2 rho_ej (r/t - u)^3 ]

with the thin-shell equations of ``fortran/integration.f90``:

    dm/dt = 4 pi r^2 [rho_ej (r/t - u) + rho_csm u]
    du/dt = 4 pi r^2 [rho_ej (r/t - u)^2 - rho_csm u^2] / m
    dr/dt = u

Conventions (same as the Fortran wrapper):

* ``r_inner``, ``r_outer`` in cm; ``mej_sn``, ``m_csm`` in Msun with
  ``SOLAR_MASS = 1.989e33`` g (``redback_csm.core.solar_mass``); ``esn`` in 1e51 erg;
  times in days on input/output.
* CSM: rho = rho_in (r / r_inner)^eta on [r_inner, r_outer], zero outside,
  normalised to ``m_csm``.  ``eta`` is the density exponent (wind: eta = -2).
* BPL ejecta: inner slope ``delta_sn`` for v <= v_t, outer slope ``nn_sn`` for
  v > v_t, zero above v_max = ``vej_max_ratio`` * v_t (``get_bpl_coeffs``, A>1 branch).
* Initial condition: the shell reaches r_inner at t_c = r_inner / v_max moving
  at v_max with negligible mass.  This is the converged limit of the Fortran start
  (1 g shell at t = 10 s on the ejecta edge), verified by shrinking the Fortran
  step factor.  Luminosity is zero before t_c.
* After the shell leaves r_outer the ODE continues with rho_csm = 0, so the
  reverse shock keeps accelerating the shell and keeps radiating (as in Fortran).

Numerics: classical RK4 in dimensionless variables
(x = r/r_inner, w = u/v_max, mu = m / (4 pi r_inner^3 rho_in), tau = t/t_c) on a grid
uniform in log(tau - 1), which resolves the stiff post-collision transient of
the near-massless seed shell.  The step that crosses r_outer is split exactly at
the boundary, and the discontinuous drop of the forward-shock term at emergence
is inserted into the output history so interpolation does not smear it.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

DAY = 86400.0
FOE = 1.0e51
SOLAR_MASS = 1.989e33  # g, matches redback_csm.core.solar_mass (Fortran wrapper)
PI = 3.141592653589793
CLIGHT = 2.99792458e10


class SimpleShockResult(NamedTuple):
    """Simple-mode output on the requested epochs.

    ``r_shock``/``v_shock`` are the thin-shell (shocked-shell) radius and velocity.
    They are *not* a photospheric radius.
    """

    time: jnp.ndarray          # days (requested epochs)
    lbol: jnp.ndarray          # erg/s: diffused luminosity if diffusion=True, else lbol_shock
    lbol_shock: jnp.ndarray    # erg/s, instantaneous eff * (L_fs + L_rs)
    lbol_fs: jnp.ndarray       # erg/s, forward-shock part (after eff)
    lbol_rs: jnp.ndarray       # erg/s, reverse-shock part (after eff)
    r_shock: jnp.ndarray       # cm
    v_shock: jnp.ndarray       # cm/s
    shell_mass: jnp.ndarray    # g
    tau_ahead: jnp.ndarray     # optical depth (params.kappa) from r_shock to r_outer
    t_collide: jnp.ndarray     # days, scalar
    t_emerge: jnp.ndarray      # days, scalar (inf if the shell never leaves r_outer)


class _Scales(NamedTuple):
    t_c: jnp.ndarray       # s
    v_max: jnp.ndarray     # cm/s
    inv_a: jnp.ndarray     # v_t / v_max
    q_ej: jnp.ndarray      # rho0_4pi v_max^3 / (4 pi r_in^3 rho_in)
    x_out: jnp.ndarray     # r_outer / r_inner
    l_scale: jnp.ndarray   # eff * 2 pi r_in^2 rho_in v_max^3   [erg/s]
    m_scale: jnp.ndarray   # 4 pi r_in^3 rho_in  [g]


def _scales(params):
    """Dimensional bookkeeping in float64 (cheap, once per model)."""
    f64 = jnp.float64
    eta = jnp.asarray(params.eta, f64)
    r_in = jnp.asarray(params.r_inner, f64)
    r_out = jnp.asarray(params.r_outer, f64)
    delta = jnp.asarray(params.delta_sn, f64)
    nn = jnp.asarray(params.nn_sn, f64)
    a = jnp.asarray(params.vej_max_ratio, f64)
    m_ej = jnp.asarray(params.mej_sn, f64) * SOLAR_MASS
    e_sn = jnp.asarray(params.esn, f64) * FOE
    m_csm = jnp.asarray(params.m_csm, f64) * SOLAR_MASS

    i_m = 1.0 / (3.0 - delta) + (1.0 - a ** (3.0 - nn)) / (nn - 3.0)
    i_e = 1.0 / (5.0 - delta) + (1.0 - a ** (5.0 - nn)) / (nn - 5.0)
    v_t = jnp.sqrt(2.0 * e_sn * i_m / (m_ej * i_e))
    v_max = a * v_t
    rho0_4pi = m_ej / (v_t**3 * i_m)

    # Mass normalisation, written via the ratio x_out = r_out/r_in for stability.
    x_out = r_out / r_in
    p3 = eta + 3.0
    near = jnp.abs(p3) < 1e-10
    shape = jnp.where(near, jnp.log(x_out), (x_out ** jnp.where(near, 1.0, p3) - 1.0) / jnp.where(near, 1.0, p3))
    m_scale = m_csm / shape  # = 4 pi r_in^3 rho_in
    rho_in = m_scale / (4.0 * PI * r_in**3)

    t_c = r_in / v_max
    q_ej = rho0_4pi * v_max**3 / m_scale  # = rho0_4pi v_max^3 / (4 pi r_in^3 rho_in)
    l_scale = jnp.asarray(params.eff, f64) * 2.0 * PI * r_in**2 * rho_in * v_max**3
    return _Scales(t_c, v_max, v_t / v_max, q_ej, x_out, l_scale, m_scale)


def _a_ej(x, tau, sc, delta, nn):
    """4 pi r^2 rho_ej in units of 4 pi r_in^2 rho_in (dimensionless)."""
    v = x / tau                                  # u_ej / v_max
    ratio = sc.inv_a / jnp.maximum(v, 1e-30)     # v_t / v
    slope = jnp.where(v > sc.inv_a, nn, delta)
    val = sc.q_ej * ratio**slope * v * v / tau
    return jnp.where(v > 1.0, 0.0, val)


def _rhs(y, tau, emerged, sc, eta, delta, nn):
    x, w, mu = y[0], y[1], y[2]
    a_ej = _a_ej(x, tau, sc, delta, nn)
    a_csm = jnp.where(emerged, 0.0, x ** (2.0 + eta))
    rel_in = jnp.maximum(x / tau - w, 0.0)
    rel_out = jnp.maximum(w, 0.0)
    dmu = a_ej * rel_in + a_csm * rel_out
    dw = (a_ej * rel_in**2 - a_csm * rel_out**2) / mu
    return jnp.stack([w, dw, dmu])


def _lum_parts(y, tau, emerged, sc, eta, delta, nn):
    x, w = y[0], y[1]
    a_ej = _a_ej(x, tau, sc, delta, nn)
    a_csm = jnp.where(emerged, 0.0, x ** (2.0 + eta))
    rel_in = jnp.maximum(x / tau - w, 0.0)
    rel_out = jnp.maximum(w, 0.0)
    return a_csm * rel_out**3, a_ej * rel_in**3


def _rk4(y, tau, h, emerged, f):
    k1 = f(y, tau, emerged)
    k2 = f(y + 0.5 * h * k1, tau + 0.5 * h, emerged)
    k3 = f(y + 0.5 * h * k2, tau + 0.5 * h, emerged)
    k4 = f(y + h * k3, tau + h, emerged)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _tau_ahead_dimless(x, x_out, eta, k_tau):
    """Optical depth from x to the outer CSM edge: k_tau * int_x^x_out x'^eta dx' (0 outside)."""
    p = eta + 1.0
    near = jnp.abs(p) < 1e-10
    xc = jnp.clip(x, 1.0, x_out)
    p_safe = jnp.where(near, 1.0, p)
    integ = jnp.where(near, jnp.log(x_out / xc), (x_out**p_safe - xc**p_safe) / p_safe)
    return jnp.where(x < x_out, k_tau * jnp.maximum(integ, 0.0), 0.0)


@partial(jax.jit, static_argnames=("n_steps", "dtype", "diffusion"))
def solve_simple_static_bpl(time_days, params, n_steps=2048, t_max_days=None, first_step=1e-7,
                            dtype=jnp.float64, diffusion=False):
    """Simple-mode static power-law CSM + BPL ejecta, matched to the Fortran reference.

    Parameters
    ----------
    time_days : array, requested source-frame epochs (days).
    params : StaticPowerlawBPLParams.  ``kappa`` is used only when ``diffusion=True``.
    n_steps : number of RK4 steps (static).  ~2048 without diffusion and ~4096 with
        diffusion give sub-percent accuracy away from shock emergence.
    t_max_days : end of the internal grid (raised to ``max(time_days)`` if smaller).
        Defaults to ``max(time_days)``.  Fix it explicitly (>= the latest epoch you will
        ever request) when results must not depend on the requested epochs.
    first_step : first step in (t - t_c)/t_c; also sets the seed mass mu0 = first_step.
    dtype : integration dtype for the dimensionless state (float64 or float32).
    diffusion : if True, ``lbol`` is the shock luminosity diffused through the unshocked
        CSM exactly as Fortran simple mode does when ``kappa`` is supplied
        (``get_diffuse_lc``, static-CSM case), in its continuous form:

            L_d(t) = int_{t_c}^{t} L_sh(t') [1/t_diff(t')] exp(-(t - t')/t_loss(t')) dt'

        with t_diff = tau_ahead r/c, tau_ahead = kappa int_r^{r_out} rho dr,
        t_adv = r/u, 1/t_loss = 1/t_diff + 1/t_adv.  A fraction t_adv/(t_diff + t_adv)
        of each parcel's energy escapes; the rest is lost adiabatically.  Shock power
        produced after the shell leaves r_outer (tau_ahead = 0) is not added, as in Fortran.
        The integral is accumulated inside the RK4 scan with L linear and t_diff, t_adv
        interval-averaged on each step, evaluated directly at the requested epochs.

        Note: as the shock approaches r_outer, t_diff ~ 2 (t_e - t), so the continuous
        prescription has a logarithmic cusp at t_e (finite energy, unbounded peak).  Any
        discretisation (Fortran's ~2% output rows included) truncates it at a
        resolution-dependent height.
    """
    time_days = jnp.asarray(time_days, jnp.float64)
    sc = _scales(params)
    eta = jnp.asarray(params.eta, jnp.float64)
    delta = jnp.asarray(params.delta_sn, jnp.float64)
    nn = jnp.asarray(params.nn_sn, jnp.float64)
    t_req = time_days * DAY
    # Never shorter than the latest requested epoch (no silent flat extrapolation).
    t_end = jnp.max(t_req) if t_max_days is None else jnp.maximum(jnp.asarray(t_max_days, jnp.float64) * DAY, jnp.max(t_req))
    tau_end = jnp.maximum(t_end / sc.t_c, 1.0 + 10.0 * first_step)

    # Cast the dimensionless problem to the working dtype.
    cast = lambda z: jnp.asarray(z, dtype)
    scd = _Scales(*[cast(v) for v in sc])
    etad, deltad, nnd = cast(eta), cast(delta), cast(nn)
    f = lambda y, tau, em: _rhs(y, tau, em, scd, etad, deltad, nnd)
    # Diffusion constants: tau_ahead = k_tau * I(x); t_diff/t_c = tau x v_max/c; t_adv/t_c = x/w.
    k_tau = cast(jnp.asarray(params.kappa, jnp.float64) * sc.m_scale / (4.0 * PI * jnp.asarray(params.r_inner, jnp.float64) ** 2))
    beta = cast(sc.v_max / CLIGHT)
    tau_q = t_req / sc.t_c
    tq = cast(tau_q)

    def diff_times(y):
        x, w = y[0], y[1]
        td = _tau_ahead_dimless(x, scd.x_out, etad, k_tau) * x * beta
        ta = x / jnp.maximum(w, cast(1e-30))
        return td, ta

    def interval_contrib(a, b, la, lb, ya, yb):
        td_a, ta_a = diff_times(ya)
        td_b, ta_b = diff_times(yb)
        td = 0.5 * (td_a + td_b)
        ta = 0.5 * (ta_a + ta_b)
        valid = (td > 0.0) & (b > a)
        td_s = jnp.where(valid, td, 1.0)
        t_loss = td_s * ta / (td_s + ta)
        escape = ta / (td_s + ta)
        bp = jnp.minimum(b, tq)
        inside = tq > a
        slope = (lb - la) / jnp.where(b > a, b - a, 1.0)
        dlt = jnp.maximum(bp - a, 0.0)
        l_bp = la + slope * dlt
        # Exact integral of a linear L(t') against exp((t' - bp)/t_loss) over [a, bp]:
        # weighted mean = L(bp) - slope * <bp - t'>, <bp - t'> = dlt * g(dlt/t_loss),
        # g(x) = 1/x - 1/expm1(x)  (-> 1/2 for x -> 0, -> 0 for x -> inf).
        xr = dlt / t_loss
        x_safe = jnp.where(xr > 1e-3, xr, 1.0)
        g = jnp.where(xr > 1e-3, 1.0 / x_safe - 1.0 / jnp.expm1(jnp.minimum(x_safe, 700.0)), 0.5 - xr / 12.0)
        lbar = l_bp - slope * dlt * g
        c = lbar * escape * jnp.exp((bp - tq) / t_loss) * (-jnp.expm1(-xr))
        return jnp.where(valid & inside, c, 0.0)

    # Grid uniform in log(tau - 1); s_0 = 0 (collision).
    s = jnp.concatenate([jnp.zeros(1, jnp.float64), jnp.geomspace(first_step, tau_end - 1.0, n_steps)])
    tau64 = 1.0 + s
    tau_grid = cast(tau64)
    h_grid = cast(jnp.diff(s))  # from the float64 (tau - 1) grid: exact even in float32
    y0 = jnp.stack([cast(1.0), cast(1.0), cast(first_step)])
    x_out = scd.x_out

    def step(carry, i):
        y, emerged, tau_e, acc = carry
        tau0 = tau_grid[i]
        h = h_grid[i]
        y_full = _rk4(y, tau0, h, emerged, f)
        crossed = (~emerged) & (y_full[0] >= x_out)
        # Split the crossing step at r_outer: in-shell part, then CSM-free part.
        frac = jnp.clip((x_out - y[0]) / jnp.maximum(y_full[0] - y[0], 1e-30), 0.0, 1.0)
        h1 = frac * h
        y_mid = _rk4(y, tau0, h1, False, f)
        y_mid = y_mid.at[0].set(x_out)
        tau_mid = tau0 + h1
        y_split = _rk4(y_mid, tau_mid, h - h1, True, f)
        y_new = jnp.where(crossed, y_split, y_full)
        emerged_new = emerged | crossed
        tau_e_new = jnp.where(crossed, tau_mid, tau_e)
        lfs, lrs = _lum_parts(y_new, tau0 + h, emerged_new, scd, etad, deltad, nnd)
        # Values just before / after emergence (for exact insertion into the history).
        lfs_m, lrs_m = _lum_parts(y_mid, tau_mid, False, scd, etad, deltad, nnd)
        lrs_p = _lum_parts(y_mid, tau_mid, True, scd, etad, deltad, nnd)[1]
        node = jnp.stack([tau0 + h, lfs, lrs, y_new[0], y_new[1], y_new[2]])
        cross_rows = jnp.stack([
            jnp.stack([tau_mid, lfs_m, lrs_m, y_mid[0], y_mid[1], y_mid[2]]),
            jnp.stack([tau_mid, cast(0.0), lrs_p, y_mid[0], y_mid[1], y_mid[2]]),
        ])
        if diffusion:
            l0 = sum(_lum_parts(y, tau0, emerged, scd, etad, deltad, nnd))
            b = jnp.where(crossed, tau_mid, tau0 + h)
            lb = jnp.where(crossed, lfs_m + lrs_m, lfs + lrs)
            yb = jnp.where(crossed, y_mid, y_new)
            add = interval_contrib(tau0, b, l0, lb, y, yb)
            acc = acc + jnp.where(emerged, 0.0, add)
        return (y_new, emerged_new, tau_e_new, acc), (node, crossed, cross_rows, frac)

    acc0 = jnp.zeros(tq.shape, dtype)
    init = (y0, jnp.array(False), cast(jnp.inf), acc0)
    (_, _, _, acc), (nodes, crossed, cross_rows, fracs) = jax.lax.scan(step, init, jnp.arange(n_steps))
    lfs0, lrs0 = _lum_parts(y0, tau_grid[0], False, scd, etad, deltad, nnd)
    first = jnp.stack([tau_grid[0], lfs0, lrs0, y0[0], y0[1], y0[2]])[None, :]
    hist = jnp.concatenate([first, nodes], axis=0).astype(jnp.float64)  # (n_steps+1, 6)
    hist = hist.at[:, 0].set(tau64)  # exact float64 abscissae

    # Insert the two emergence rows after the node preceding the crossing.
    any_cross = jnp.any(crossed)
    i_c = jnp.where(any_cross, jnp.argmax(crossed), n_steps)  # crossing step index (node i_c -> i_c+1)
    i_safe = jnp.minimum(i_c, n_steps - 1)
    rows = cross_rows[i_safe].astype(jnp.float64)
    tau_e64 = tau64[i_safe] + fracs[i_safe].astype(jnp.float64) * (tau64[i_safe + 1] - tau64[i_safe])
    rows = rows.at[:, 0].set(tau_e64)
    tau_e = jnp.where(any_cross, tau_e64, jnp.inf)
    tail = jnp.stack([hist[-1], hist[-1]])
    rows = jnp.where(any_cross, rows, tail)
    rows = rows.at[1, 0].set(rows[1, 0] * (1.0 + 1e-12))
    n_h = hist.shape[0]
    k = jnp.arange(n_h + 2)
    src = jnp.where(k <= i_c, k, k - 2)
    full = hist[jnp.clip(src, 0, n_h - 1)]
    full = jnp.where((k == i_c + 1)[:, None], rows[0], full)
    full = jnp.where((k == i_c + 2)[:, None], rows[1], full)
    full = full.at[:, 0].set(jax.lax.cummax(full[:, 0]))  # guard: non-decreasing abscissae

    interp = lambda col, left: jnp.interp(tau_q, full[:, 0], full[:, col], left=left, right=full[-1, col])
    lfs_q = interp(1, 0.0) * sc.l_scale
    lrs_q = interp(2, 0.0) * sc.l_scale
    x_q = interp(3, 1.0)
    w_q = interp(4, 1.0)
    mu_q = interp(5, 0.0)
    before = tau_q < 1.0
    r_q = jnp.where(before, sc.v_max * t_req, x_q * params.r_inner)  # pre-collision: ejecta edge
    v_q = jnp.where(before, sc.v_max, w_q * sc.v_max)
    lshock = lfs_q + lrs_q
    tau_ahead = jnp.where(before, 0.0, _tau_ahead_dimless(x_q, sc.x_out, eta, k_tau.astype(jnp.float64)))
    lbol = jnp.where(before, 0.0, acc.astype(jnp.float64) * sc.l_scale) if diffusion else lshock
    return SimpleShockResult(
        time=time_days,
        lbol=lbol,
        lbol_shock=lshock,
        lbol_fs=lfs_q,
        lbol_rs=lrs_q,
        r_shock=r_q,
        v_shock=v_q,
        shell_mass=mu_q * sc.m_scale,
        tau_ahead=tau_ahead,
        t_collide=sc.t_c / DAY,
        t_emerge=jnp.asarray(tau_e, jnp.float64) * sc.t_c / DAY,
    )
