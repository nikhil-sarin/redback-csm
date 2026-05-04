from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

DAY = 86400.0
CLIGHT = 2.99792458e10
SOLAR_MASS = 1.98847e33
FOE = 1.0e51
PI = jnp.pi


class StaticPowerlawBPLParams(NamedTuple):
    """Parameters for a static finite power-law CSM shell hit by BPL ejecta."""

    eta: float
    r_inner: float
    r_outer: float
    delta_sn: float
    nn_sn: float
    mej_sn: float
    esn: float
    eff: float
    m_csm: float
    kappa: float = 0.34
    vej_max_ratio: float = 3.0


class StaticPowerlawBPLResult(NamedTuple):
    time: jnp.ndarray
    lbol: jnp.ndarray
    lbol_shock: jnp.ndarray
    radius: jnp.ndarray
    velocity: jnp.ndarray
    shell_mass: jnp.ndarray
    internal_energy: jnp.ndarray
    optical_depth: jnp.ndarray
    t_leak: jnp.ndarray


def _solve_tridiagonal(lower, diag, upper, rhs):
    """Thomas solve for a tridiagonal system."""
    n = rhs.shape[0]
    c_star = jnp.zeros(n, dtype=rhs.dtype)
    d_star = jnp.zeros(n, dtype=rhs.dtype)
    c_star = c_star.at[0].set(jnp.where(n > 1, upper[0] / diag[0], 0.0))
    d_star = d_star.at[0].set(rhs[0] / diag[0])

    def forward(carry, i):
        c_s, d_s = carry
        denom = diag[i] - lower[i - 1] * c_s[i - 1]
        c_new = jnp.where(i < n - 1, upper[i] / denom, 0.0)
        d_new = (rhs[i] - lower[i - 1] * d_s[i - 1]) / denom
        c_s = c_s.at[i].set(c_new)
        d_s = d_s.at[i].set(d_new)
        return (c_s, d_s), None

    (c_star, d_star), _ = jax.lax.scan(forward, (c_star, d_star), jnp.arange(1, n))
    sol = jnp.zeros(n, dtype=rhs.dtype).at[-1].set(d_star[-1])

    def backward(sol, ii):
        i = n - 2 - ii
        return sol.at[i].set(d_star[i] - c_star[i] * sol[i + 1]), None

    sol, _ = jax.lax.scan(backward, sol, jnp.arange(n - 1))
    return sol


def _static_powerlaw_rho_inner(params: StaticPowerlawBPLParams):
    """Density at r_inner for rho = rho_inner * (r/r_inner)^eta."""
    m_csm = params.m_csm * SOLAR_MASS
    eta = params.eta
    log_factor = 4.0 * PI * params.r_inner**3 * jnp.log(params.r_outer / params.r_inner)
    denom_power = jnp.where(jnp.abs(eta + 3.0) < 1e-10, 1.0, eta + 3.0)
    power_factor = (
        4.0
        * PI
        * params.r_inner ** (-eta)
        * (params.r_outer ** (eta + 3.0) - params.r_inner ** (eta + 3.0))
        / denom_power
    )
    mass_factor = jnp.where(jnp.abs(eta + 3.0) < 1e-10, log_factor, power_factor)
    return m_csm / mass_factor


def _bpl_coefficients(params: StaticPowerlawBPLParams):
    """BPL coefficients using the same finite-cutoff convention as the Fortran backend."""
    m_ej = params.mej_sn * SOLAR_MASS
    e_sn = params.esn * FOE
    delta = params.delta_sn
    nn = params.nn_sn
    a_ratio = params.vej_max_ratio

    i_mass = 1.0 / (3.0 - delta) + (1.0 - a_ratio ** (3.0 - nn)) / (nn - 3.0)
    i_energy = 1.0 / (5.0 - delta) + (1.0 - a_ratio ** (5.0 - nn)) / (nn - 5.0)
    v_t = jnp.sqrt(2.0 * e_sn * i_mass / (m_ej * i_energy))
    v_max = a_ratio * v_t
    rho0_4pi = m_ej / (v_t**3 * i_mass)
    return v_t, v_max, rho0_4pi


def _rho_csm(params: StaticPowerlawBPLParams, rho_inner, r):
    in_shell = (r >= params.r_inner) & (r <= params.r_outer)
    rho = rho_inner * (r / params.r_inner) ** params.eta
    return jnp.where(in_shell, rho, 0.0)


def _static_radius_grid(params: StaticPowerlawBPLParams, n_zones):
    frac = jnp.linspace(0.0, 1.0, n_zones, dtype=jnp.float64)
    ratio = params.r_outer / params.r_inner
    return jnp.where(
        ratio > 1.05,
        params.r_inner * ratio**frac,
        params.r_inner + (params.r_outer - params.r_inner) * frac,
    )


def _zone_geometry(radius, r_inner, r_outer):
    n = radius.shape[0]
    face_l = jnp.empty(n, dtype=radius.dtype)
    face_r = jnp.empty(n, dtype=radius.dtype)
    face_l = face_l.at[0].set(r_inner)
    face_l = face_l.at[1:].set(jnp.sqrt(radius[:-1] * radius[1:]))
    face_r = face_r.at[:-1].set(jnp.sqrt(radius[:-1] * radius[1:]))
    face_r = face_r.at[-1].set(r_outer)
    vol = 4.0 * PI * jnp.maximum(face_r**3 - face_l**3, 1.0e-30) / 3.0
    return face_l, face_r, vol


def _tau_grid(params, rho, face_l, face_r):
    dtau = params.kappa * jnp.maximum(rho, 1.0e-30) * jnp.maximum(face_r - face_l, 0.0)
    return jnp.cumsum(dtau[::-1])[::-1]


def _photosphere_index_and_radius(radius, face_l, face_r, tau, r_floor):
    n = radius.shape[0]
    below = tau <= (2.0 / 3.0)
    found = jnp.any(below[1:])
    first = jnp.argmax(below[1:]) + 1
    iph_cross = jnp.maximum(first - 1, 0)
    iph = jnp.where(found, iph_cross, n - 1)

    tau_lo = tau[iph]
    tau_hi = tau[jnp.minimum(iph + 1, n - 1)]
    frac = (2.0 / 3.0 - tau_lo) / jnp.maximum(tau_hi - tau_lo, 1.0e-30)
    frac = jnp.clip(frac, 0.0, 1.0)
    r_cross = radius[iph] + frac * (radius[jnp.minimum(iph + 1, n - 1)] - radius[iph])
    r_default = face_r[-1]
    r_ph = jnp.where(found, jnp.maximum(r_cross, r_floor), r_default)
    return iph, r_ph


def _shock_zone(radius, r_shell):
    return jnp.argmin(jnp.abs(radius - r_shell))


def _transport_step(
    params,
    radius,
    rho,
    u_rad_old,
    dt,
    r_shell,
    lum_heat,
    in_cooling,
    theta,
):
    face_l, face_r, vol = _zone_geometry(radius, radius[0], radius[-1])
    tau = _tau_grid(params, rho, face_l, face_r)
    r_floor = jnp.where(in_cooling, radius[0], jnp.maximum(r_shell, radius[0]))
    iph, r_ph = _photosphere_index_and_radius(radius, face_l, face_r, tau, r_floor)
    n = radius.shape[0]
    idx = jnp.arange(n)
    ish = _shock_zone(radius, r_shell)
    ihi = jnp.array(0)
    iph = jnp.maximum(iph, ihi)
    r_ph = jnp.maximum(r_ph, radius[ihi])
    active = (idx >= ihi) & (idx <= iph)

    rho_l = 0.5 * (rho[:-1] + rho[1:])
    dr = jnp.maximum(radius[1:] - radius[:-1], 1.0e-30)
    area_faces = 4.0 * PI * jnp.sqrt(radius[:-1] * radius[1:]) ** 2
    conduct = area_faces * CLIGHT / (3.0 * params.kappa * jnp.maximum(rho_l, 1.0e-30) * dr)

    k_l = jnp.zeros(n, dtype=radius.dtype).at[1:].set(conduct)
    k_r = jnp.zeros(n, dtype=radius.dtype).at[:-1].set(conduct)
    k_l = jnp.where(idx <= ihi, 0.0, k_l)
    k_r = jnp.where(idx >= iph, 0.0, k_r)

    r_ph_eff = jnp.clip(r_ph, face_l[iph], face_r[iph])
    vol_ph = 4.0 * PI * jnp.maximum(r_ph_eff**3 - face_l[iph] ** 3, 1.0e-30) / 3.0
    vol_eff = jnp.where(idx == iph, vol_ph, vol)
    area_ph = 4.0 * PI * r_ph_eff**2
    k_escape = jnp.where(idx == iph, area_ph * CLIGHT * 0.25, 0.0)

    omt = 1.0 - theta
    diag = vol_eff / dt + theta * (k_l + k_r + k_escape)
    rhs = vol_eff * u_rad_old / dt - omt * (k_l + k_r + k_escape) * u_rad_old
    rhs = rhs.at[1:].add(omt * k_l[1:] * u_rad_old[:-1])
    rhs = rhs.at[:-1].add(omt * k_r[:-1] * u_rad_old[1:])

    source_active = (~in_cooling) & (r_shell >= radius[0]) & (r_shell <= radius[-1])
    rhs = rhs.at[ish].add(jnp.where(source_active, jnp.maximum(lum_heat, 0.0), 0.0))

    diag = jnp.where(active, diag, 1.0)
    rhs = jnp.where(active, rhs, 0.0)

    lower = -theta * k_l[1:]
    upper = -theta * k_r[:-1]
    lower = jnp.where((idx[1:] > ihi) & (idx[1:] <= iph), lower, 0.0)
    upper = jnp.where(idx[:-1] < iph, upper, 0.0)

    u_rad = jnp.nan_to_num(_solve_tridiagonal(lower, diag, upper, rhs), nan=0.0, posinf=0.0, neginf=0.0)
    u_rad = jnp.maximum(u_rad, 0.0)
    y_obs = u_rad[iph]
    lum_obs = jnp.nan_to_num(jnp.maximum(PI * r_ph_eff**2 * CLIGHT * y_obs, 0.0), nan=0.0, posinf=0.0)
    e_rad = jnp.nan_to_num(jnp.sum(u_rad * vol), nan=0.0, posinf=0.0)
    return u_rad, lum_obs, r_ph, tau[0], e_rad


def _rho_csm_4pir2(params: StaticPowerlawBPLParams, rho_inner, r):
    return 4.0 * PI * r**2 * _rho_csm(params, rho_inner, r)


def _rho_ej_4pir2(params: StaticPowerlawBPLParams, v_t, v_max, rho0_4pi, r, t):
    t = jnp.maximum(t, 1.0e-30)
    v = r / t
    v_ratio = jnp.maximum(v / v_t, 1.0e-30)
    inner = rho0_4pi * v_ratio ** (-params.delta_sn) * v**2 / t
    outer = rho0_4pi * v_ratio ** (-params.nn_sn) * v**2 / t
    rho = jnp.where(v <= v_t, inner, outer)
    return jnp.where(v <= v_max, rho, 0.0)


def _tau_static_powerlaw(params: StaticPowerlawBPLParams, rho_inner, r):
    r0 = jnp.clip(r, params.r_inner, params.r_outer)
    eta = params.eta
    log_tau = params.kappa * rho_inner * params.r_inner * jnp.log(params.r_outer / r0)
    denom_power = jnp.where(jnp.abs(eta + 1.0) < 1e-10, 1.0, eta + 1.0)
    power_tau = (
        params.kappa
        * rho_inner
        * params.r_inner ** (-eta)
        * (params.r_outer ** (eta + 1.0) - r0 ** (eta + 1.0))
        / denom_power
    )
    tau = jnp.where(jnp.abs(eta + 1.0) < 1e-10, log_tau, power_tau)
    return jnp.where(r < params.r_outer, jnp.maximum(tau, 0.0), 0.0)


def _leakage_timescale(params: StaticPowerlawBPLParams, rho_inner, r, u, m):
    tau_ahead = _tau_static_powerlaw(params, rho_inner, r)
    dr_ahead = jnp.maximum(params.r_outer - r, 0.0)
    t_interaction = (tau_ahead + 1.0) * jnp.maximum(dr_ahead, r / (tau_ahead + 1.0)) / CLIGHT

    surface_density = jnp.maximum(m, 0.0) / (4.0 * PI * jnp.maximum(r, 1.0) ** 2)
    tau_shell = params.kappa * surface_density
    t_cooling = (tau_shell + 1.0) * jnp.maximum(r / CLIGHT, params.kappa * surface_density * r / CLIGHT)
    emerged = r >= params.r_outer
    t_leak = jnp.where(emerged, t_cooling, t_interaction)
    return jnp.maximum(t_leak, 1.0)


def _rhs(params, rho_inner, v_t, v_max, rho0_4pi, state, t, transport):
    r, u, m, e_rad = state
    m_safe = jnp.maximum(m, 1.0)
    rho_in = _rho_ej_4pir2(params, v_t, v_max, rho0_4pi, r, t)
    rho_out = _rho_csm_4pir2(params, rho_inner, r)

    v_ej = r / jnp.maximum(t, 1.0e-30)
    rel_in = jnp.maximum(v_ej - u, 0.0)
    rel_out = jnp.maximum(u, 0.0)

    dm_dt = rho_in * rel_in + rho_out * rel_out
    du_dt = (rho_in * rel_in**2 - rho_out * rel_out**2) / m_safe
    dr_dt = jnp.maximum(u, 0.0)

    l_fs = 0.5 * rho_out * rel_out**3
    l_rs = 0.5 * rho_in * rel_in**3
    l_heat = params.eff * (l_fs + l_rs)

    t_leak = _leakage_timescale(params, rho_inner, r, u, m_safe)
    l_obs = e_rad / t_leak
    t_exp = jnp.maximum(r / jnp.maximum(jnp.abs(u), 1.0), 1.0)
    adiabatic = jnp.where(r >= params.r_outer, e_rad / t_exp, 0.0)
    de_dt = jnp.where(transport, l_heat - l_obs - adiabatic, 0.0)

    return jnp.array([dr_dt, du_dt, dm_dt, de_dt]), l_heat, l_obs, t_leak


def _rk4_step(params, rho_inner, v_t, v_max, rho0_4pi, state, t, dt, transport):
    k1, _, _, _ = _rhs(params, rho_inner, v_t, v_max, rho0_4pi, state, t, transport)
    k2, _, _, _ = _rhs(
        params, rho_inner, v_t, v_max, rho0_4pi, state + 0.5 * dt * k1, t + 0.5 * dt, transport
    )
    k3, _, _, _ = _rhs(
        params, rho_inner, v_t, v_max, rho0_4pi, state + 0.5 * dt * k2, t + 0.5 * dt, transport
    )
    k4, _, _, _ = _rhs(
        params, rho_inner, v_t, v_max, rho0_4pi, state + dt * k3, t + dt, transport
    )
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state = next_state.at[0].set(jnp.maximum(next_state[0], 1.0))
    next_state = next_state.at[1].set(jnp.maximum(next_state[1], 0.0))
    next_state = next_state.at[2].set(jnp.maximum(next_state[2], 1.0))
    next_state = next_state.at[3].set(jnp.maximum(next_state[3], 0.0))
    return next_state


class _DimlessScales(NamedTuple):
    rho_inner: float
    v_t: float
    v_max: float
    rho0_4pi: float
    t_in: float
    r_in: float
    x_out: float
    rho_csm_in: float
    rho_ej_in: float
    q: float
    tau_csm_in: float
    t_diff: float
    y_ratio: float
    u0: float


def _dimless_scales(params: StaticPowerlawBPLParams):
    rho_inner = _static_powerlaw_rho_inner(params)
    v_t, v_max, rho0_4pi = _bpl_coefficients(params)
    t_in = jnp.maximum(params.r_inner / jnp.maximum(v_max, 1.0), 1.0e-6)
    r_in = jnp.maximum(params.r_inner, 1.0)
    x_out = jnp.maximum(params.r_outer / r_in, 1.0001)
    rho_csm_in = jnp.maximum(rho_inner, 1.0e-30)
    rho_ej_in = jnp.maximum(_rho_ej_4pir2(params, v_t, v_max, rho0_4pi, r_in, t_in) / (4.0 * PI * r_in**2), 1.0e-30)
    q = rho_csm_in / rho_ej_in
    tau_csm_in = 3.0 * params.kappa * rho_csm_in * r_in
    t_diff = tau_csm_in * r_in / CLIGHT
    y_ratio = t_in / jnp.maximum(t_diff, 1.0e-30)
    u0 = (tau_csm_in * v_max / CLIGHT) * (0.5 * rho_csm_in * v_max**2)
    return _DimlessScales(
        rho_inner,
        v_t,
        v_max,
        rho0_4pi,
        t_in,
        r_in,
        x_out,
        rho_csm_in,
        rho_ej_in,
        q,
        tau_csm_in,
        t_diff,
        y_ratio,
        u0,
    )


def _eta_csm_x(params, scales, x, in_cooling, x_out_cool, x_sh_se):
    x = jnp.maximum(x, 1.0e-300)
    x_pre = jnp.where(
        in_cooling,
        1.0 + (x - 1.0) * (x_sh_se - 1.0) / jnp.maximum(x_out_cool - 1.0, 1.0e-12),
        x,
    )
    x_pre = jnp.where(in_cooling, jnp.clip(x_pre, 1.0, jnp.maximum(x_sh_se, 1.0)), x_pre)
    return jnp.maximum(x_pre ** params.eta, 1.0e-30)


def _tau_dimless_powerlaw(params, scales, x_from, in_cooling=False, r_in_r0=1.0, x_out_cool=1.0):
    x_end = jnp.where(in_cooling, jnp.maximum(x_out_cool, x_from), jnp.maximum(scales.x_out, x_from))
    length_scale = jnp.where(in_cooling, scales.r_in / jnp.maximum(r_in_r0, 1.0e-30) ** 2, scales.r_in)
    p = params.eta + 1.0
    integ_log = jnp.log(jnp.maximum(x_end, 1.0e-300) / jnp.maximum(x_from, 1.0e-300))
    integ_pow = (x_end**p - x_from**p) / jnp.where(jnp.abs(p) < 1.0e-12, 1.0, p)
    integ = jnp.where(jnp.abs(p) < 1.0e-12, integ_log, integ_pow)
    return jnp.maximum(params.kappa * scales.rho_csm_in * length_scale * integ, 0.0)


def _photosphere_dimless(params, scales, x_start, x_ph_prev, in_cooling=False, r_in_r0=1.0, x_out_cool=1.0):
    tau_ph = 2.0 / 3.0
    x_inner = jnp.where(in_cooling, 1.0, jnp.maximum(x_start, 1.0))
    x_outer = jnp.where(in_cooling, jnp.maximum(x_out_cool, x_inner), jnp.maximum(scales.x_out, x_inner))
    if in_cooling:
        return x_outer
    tau_cum = _tau_dimless_powerlaw(params, scales, x_inner, in_cooling, r_in_r0, x_out_cool)
    length_scale = jnp.where(in_cooling, scales.r_in / jnp.maximum(r_in_r0, 1.0e-30) ** 2, scales.r_in)
    tau_target = tau_ph / jnp.maximum(params.kappa * scales.rho_csm_in * length_scale, 1.0e-300)
    p = params.eta + 1.0
    x_log = x_outer * jnp.exp(-tau_target)
    rhs = x_outer**p - p * tau_target
    x_pow = jnp.where((rhs <= 0.0) & (p > 0.0), x_inner, jnp.maximum(rhs, 1.0e-300) ** (1.0 / p))
    x_tau = jnp.where(jnp.abs(p) < 1.0e-12, x_log, x_pow)
    thin_interaction = (~in_cooling) & (tau_cum <= tau_ph)
    shock_adjacent = jnp.minimum(x_inner + jnp.maximum(1.0e-3 * (x_outer - x_inner), 1.0e-6), x_outer)
    cached = jnp.minimum(jnp.maximum(x_ph_prev, x_inner + jnp.maximum(1.0e-3 * (x_outer - x_inner), 1.0e-6)), x_outer)
    thin_x = jnp.where(x_ph_prev > x_inner, cached, shock_adjacent)
    x_ph = jnp.where(tau_cum <= tau_ph, jnp.where(in_cooling, x_outer, thin_x), jnp.clip(x_tau, x_inner, x_outer))
    return jnp.where(x_outer <= x_inner + 1.0e-12, x_inner, x_ph)


def _eta_ej_dimless(params, scales, x, zeta):
    r = x * scales.r_in
    t = jnp.maximum(zeta * scales.t_in, 1.0e-30)
    rho = _rho_ej_4pir2(params, scales.v_t, scales.v_max, scales.rho0_4pi, r, t) / (4.0 * PI * jnp.maximum(r, 1.0) ** 2)
    return jnp.maximum(rho / jnp.maximum(scales.rho_ej_in * jnp.maximum(zeta, 1.0e-30) ** (-3.0), 1.0e-30), 0.0)


def _dimless_rhs(params, scales, x, w, phi, zeta):
    eta_csm = jnp.maximum(x, 1.0e-300) ** params.eta
    eta_ej = _eta_ej_dimless(params, scales, x, zeta)
    w_eff = jnp.maximum(w, 0.0)
    v_rel = jnp.maximum(x / jnp.maximum(zeta, 1.0e-30) - w_eff, 0.0)
    dx = w_eff
    dphi = x**2 * jnp.maximum(zeta, 1.0e-30) ** (-3.0) * v_rel * eta_ej + scales.q * x**2 * w_eff * eta_csm
    dw = (
        x**2 * jnp.maximum(zeta, 1.0e-30) ** (-3.0) * v_rel**2 * eta_ej
        - scales.q * x**2 * w_eff**2 * eta_csm
    ) / jnp.maximum(phi, 1.0e-30)
    return dx, dw, dphi


def _rk4_dimless(params, scales, x, w, phi, zeta, dzeta):
    k1 = _dimless_rhs(params, scales, x, w, phi, zeta)
    k2 = _dimless_rhs(
        params,
        scales,
        x + 0.5 * dzeta * k1[0],
        w + 0.5 * dzeta * k1[1],
        jnp.maximum(phi + 0.5 * dzeta * k1[2], 1.0e-30),
        zeta + 0.5 * dzeta,
    )
    k3 = _dimless_rhs(
        params,
        scales,
        x + 0.5 * dzeta * k2[0],
        w + 0.5 * dzeta * k2[1],
        jnp.maximum(phi + 0.5 * dzeta * k2[2], 1.0e-30),
        zeta + 0.5 * dzeta,
    )
    k4 = _dimless_rhs(
        params,
        scales,
        x + dzeta * k3[0],
        w + dzeta * k3[1],
        jnp.maximum(phi + dzeta * k3[2], 1.0e-30),
        zeta + dzeta,
    )
    x_new = jnp.maximum(x + dzeta * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0, 1.0)
    w_new = jnp.maximum(w + dzeta * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0, 1.0e-10)
    phi_new = jnp.maximum(phi + dzeta * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0, 1.0e-30)
    return x_new, w_new, phi_new, zeta + dzeta


def _shock_luminosities_dimless(params, scales, x, w, zeta, in_cooling):
    r = jnp.maximum(x * scales.r_in, 1.0)
    t = jnp.maximum(zeta * scales.t_in, 1.0e-30)
    u = jnp.maximum(w * scales.v_max, 0.0)
    rho_in_4pi = _rho_ej_4pir2(params, scales.v_t, scales.v_max, scales.rho0_4pi, r, t)
    rho_out_4pi = _rho_csm_4pir2(params, scales.rho_inner, r)
    v_ej = r / t
    rel_in = jnp.maximum(v_ej - u, 0.0)
    rel_out = jnp.maximum(u, 0.0)
    lum_fs = params.eff * 0.5 * rho_out_4pi * rel_out**3
    lum_rs = params.eff * 0.5 * rho_in_4pi * rel_in**3
    lum_fs = jnp.where(in_cooling, 0.0, jnp.maximum(lum_fs, 0.0))
    lum_rs = jnp.where(in_cooling, 0.0, jnp.maximum(lum_rs, 0.0))
    return lum_fs, lum_rs


def _solve_diffusion_dimless(params, scales, e_old, dy, x_min, x_ph, x_sh, x_sh_dot, x_ph_dot, lum_heat, in_cooling,
                             r_in_r0, x_out_cool, x_sh_se, rannacher_left):
    n = e_old.shape[0]
    xi = jnp.linspace(0.0, 1.0, n, dtype=jnp.float64)
    dxi = 1.0 / jnp.maximum(n - 1, 1)
    delta_x = x_ph - x_min
    valid = delta_x > 0.0
    theta = jnp.where((rannacher_left > 0) | in_cooling, 1.0, 0.5)
    domain_span = jnp.maximum(scales.x_out - 1.0, 1.0)
    collapse_span = jnp.maximum(2.0 * dxi * domain_span, 5.0e-2)
    theta = jnp.where((~in_cooling) & (delta_x <= collapse_span), 1.0, theta)
    omt = 1.0 - theta
    v_sh_xi = jnp.where(in_cooling, 0.0, x_sh_dot / jnp.maximum(delta_x, 1.0e-30))
    v_ph_xi = x_ph_dot / jnp.maximum(delta_x, 1.0e-30)

    idx = jnp.arange(n)
    x_i = x_min + xi * delta_x
    x_im = x_min + (xi - 0.5 * dxi) * delta_x
    x_ip = x_min + (xi + 0.5 * dxi) * delta_x
    eta_im = _eta_csm_x(params, scales, x_im, in_cooling, x_out_cool, x_sh_se)
    eta_ip = _eta_csm_x(params, scales, x_ip, in_cooling, x_out_cool, x_sh_se)
    d_im = x_im**2 / jnp.maximum(eta_im, 1.0e-30)
    d_ip = x_ip**2 / jnp.maximum(eta_ip, 1.0e-30)
    d_im = jnp.where(in_cooling, d_im * r_in_r0, d_im)
    d_ip = jnp.where(in_cooling, d_ip * r_in_r0, d_ip)
    alpha_m = d_im / jnp.maximum(x_i**2 * delta_x**2 * dxi**2, 1.0e-300)
    alpha_p = d_ip / jnp.maximum(x_i**2 * delta_x**2 * dxi**2, 1.0e-300)
    gamma = v_sh_xi * (1.0 - xi) + v_ph_xi * xi
    pe = jnp.abs(gamma) * dxi / jnp.maximum(alpha_m + alpha_p, 1.0e-60) * 2.0

    a_c = alpha_m - gamma / (2.0 * dxi)
    c_c = alpha_p + gamma / (2.0 * dxi)
    b_c = alpha_m + alpha_p
    a_up_pos = alpha_m
    c_up_pos = alpha_p + gamma / dxi
    b_up_pos = alpha_m + alpha_p + gamma / dxi
    a_up_neg = alpha_m - gamma / dxi
    c_up_neg = alpha_p
    b_up_neg = alpha_m + alpha_p - gamma / dxi
    use_up_pos = (pe > 2.0) & (gamma > 0.0)
    use_up_neg = (pe > 2.0) & (gamma < 0.0)
    aa = jnp.where(use_up_pos, a_up_pos, jnp.where(use_up_neg, a_up_neg, a_c))
    cc = jnp.where(use_up_pos, c_up_pos, jnp.where(use_up_neg, c_up_neg, c_c))
    bb = jnp.where(use_up_pos, b_up_pos, jnp.where(use_up_neg, b_up_neg, b_c))

    lower_full = -theta * dy * aa
    diag = 1.0 + theta * dy * bb
    upper_full = -theta * dy * cc
    rhs = e_old + omt * dy * aa * jnp.roll(e_old, 1) + omt * dy * cc * jnp.roll(e_old, -1) - omt * dy * bb * e_old

    interior = (idx > 0) & (idx < n - 1)
    diag = jnp.where(interior, diag, 1.0)
    rhs = jnp.where(interior, rhs, 0.0)
    lower_full = jnp.where(interior, lower_full, 0.0)
    upper_full = jnp.where(interior, upper_full, 0.0)

    x_hat_i = x_min
    x_hat_ip = x_min + 0.5 * dxi * delta_x
    eta_ip0 = _eta_csm_x(params, scales, x_hat_ip, in_cooling, x_out_cool, x_sh_se)
    d_ip0 = x_hat_ip**2 / jnp.maximum(eta_ip0, 1.0e-30)
    d_ip0 = jnp.where(in_cooling, d_ip0 * r_in_r0, d_ip0)
    alpha0 = d_ip0 / jnp.maximum(x_hat_i**2 * delta_x**2 * dxi**2, 1.0e-300)
    eta_min = _eta_csm_x(params, scales, x_min, in_cooling, x_out_cool, x_sh_se)
    f_ib = lum_heat * scales.tau_csm_in * eta_min / jnp.maximum(
        4.0 * PI * x_sh**2 * scales.r_in**2 * CLIGHT * scales.u0, 1.0e-300
    )
    f_ib = jnp.where(in_cooling, 0.0, jnp.maximum(f_ib, 0.0))
    inner_rate = 2.0 * alpha0
    inner_rate = jnp.where((~in_cooling) & (v_sh_xi > 0.0), inner_rate + v_sh_xi / dxi, inner_rate)
    diag = diag.at[0].set(1.0 + theta * dy * inner_rate)
    upper_full = upper_full.at[0].set(-theta * dy * inner_rate)
    rhs_inner_int = e_old[0] + omt * dy * inner_rate * (e_old[1] - e_old[0]) + dy * 2.0 * alpha0 * delta_x * dxi * f_ib
    rhs_inner_cool = e_old[0] + omt * dy * (2.0 * alpha0) * (e_old[1] - e_old[0])
    rhs = rhs.at[0].set(jnp.where(in_cooling, rhs_inner_cool, rhs_inner_int))

    eta_ph = _eta_csm_x(params, scales, x_ph, in_cooling, x_out_cool, x_sh_se)
    f_ob_int = -4.0 / jnp.maximum(scales.tau_csm_in * eta_ph, 1.0e-300)
    f_ob_cool = -2.0 * r_in_r0**2 / jnp.maximum(scales.tau_csm_in * eta_ph, 1.0e-300)
    f_ob = jnp.where(in_cooling, f_ob_cool, f_ob_int)
    dx_bc = delta_x * dxi
    lower_full = lower_full.at[-1].set(f_ob)
    diag = diag.at[-1].set(dx_bc - f_ob)
    rhs = rhs.at[-1].set(0.0)

    lower = lower_full[1:]
    upper = upper_full[:-1]
    sol = _solve_tridiagonal(lower, diag, upper, rhs)
    sol = jnp.where(valid, sol, e_old)
    return jnp.minimum(jnp.maximum(jnp.nan_to_num(sol, nan=0.0, posinf=0.0, neginf=0.0), 0.0), 1.0e6)


def _shell_bin_centers(x_out, n):
    return 1.0 + (jnp.arange(n, dtype=jnp.float64) + 0.5) * (jnp.maximum(x_out, 1.0001) - 1.0) / n


def _deposit_shell_energy(shell_e, x_left, x_right, energy_dim, x_out):
    n = shell_e.shape[0]
    dx_shell = (jnp.maximum(x_out, 1.0001) - 1.0) / n
    idx = jnp.arange(n, dtype=jnp.float64)
    bin_l = 1.0 + idx * dx_shell
    bin_r = bin_l + dx_shell
    x_l = jnp.minimum(jnp.maximum(jnp.minimum(x_left, x_right), 1.0), x_out)
    x_r = jnp.minimum(jnp.maximum(jnp.maximum(x_left, x_right), 1.0), x_out)
    sweep_vol = jnp.maximum((x_r**3 - x_l**3) / 3.0, 1.0e-30)
    e_add = energy_dim / sweep_vol
    overlap = (jnp.minimum(bin_r, x_r) > jnp.maximum(bin_l, x_l)) & (x_r > x_l + 1.0e-10)
    return shell_e + jnp.where(overlap, e_add, 0.0)


def _drain_shell_energy(shell_e, e_loss_dim, x_ph, x_out):
    n = shell_e.shape[0]
    dx_shell = (jnp.maximum(x_out, 1.0001) - 1.0) / n
    idx = jnp.arange(n, dtype=jnp.float64)
    bin_l = 1.0 + idx * dx_shell
    bin_r = bin_l + dx_shell
    bin_vol = jnp.maximum((bin_r**3 - bin_l**3) / 3.0, 1.0e-30)
    e_bin = jnp.maximum(shell_e, 0.0) * bin_vol
    x_drain = jnp.minimum(jnp.maximum(x_ph, 1.0), x_out)
    i_start = jnp.argmax((x_drain >= bin_l) & (x_drain <= bin_r))
    order1 = jnp.arange(n - 1, -1, -1)
    order2 = jnp.arange(n)
    priority1 = jnp.where(order1 <= i_start, 0, 1)
    priority2 = jnp.where(order2 > i_start, 0, 1)
    order = jnp.concatenate([order1[jnp.argsort(priority1, stable=True)], order2[jnp.argsort(priority2, stable=True)]])
    order = order[:n]
    e_order = e_bin[order]
    cum_before = jnp.cumsum(e_order) - e_order
    drain = jnp.minimum(e_order, jnp.maximum(e_loss_dim - cum_before, 0.0))
    drained = jnp.zeros(n, dtype=jnp.float64).at[order].add(drain)
    return jnp.maximum(shell_e - drained / bin_vol, 0.0)


def _dimless_energy_grid(e, x_min, x_ph):
    n = e.shape[0]
    xi = jnp.linspace(0.0, 1.0, n, dtype=jnp.float64)
    x = x_min + xi * (x_ph - x_min)
    dx = (x_ph - x_min) / jnp.maximum(n - 1, 1)
    integrand = x**2 * jnp.maximum(e, 0.0)
    return jnp.maximum(jnp.sum(0.5 * (integrand[:-1] + integrand[1:]) * dx), 0.0)


def _transition_to_cooling(params, scales, x, w, phi, zeta, x_ph, e_grid, shell_e, e_inj, e_rad, efs, ers, lum_pre):
    n = e_grid.shape[0]
    xi = jnp.linspace(0.0, 1.0, n, dtype=jnp.float64)
    x_sh_se = jnp.minimum(x, scales.x_out)
    x_out_cool = jnp.maximum(x_sh_se, 1.0001)
    v_se = w * scales.v_max / jnp.maximum(x_out_cool, 1.0e-30)
    x_ph_cool = _photosphere_dimless(params, scales, 1.0, x_ph, True, 1.0, x_out_cool)

    eta_out = jnp.maximum(_eta_csm_x(params, scales, x_out_cool, True, x_out_cool, x_sh_se), 1.0e-30)
    s_eff = jnp.maximum(-jnp.log(eta_out) / jnp.log(jnp.maximum(x_out_cool, 1.0001)), 0.0)
    pdv_power = 2.0 + 1.5 * jnp.minimum(s_eff, 2.0)
    tail_shape = jnp.exp(-0.8 * s_eff)
    tail_fraction = jnp.minimum(1.0, 10.0 / jnp.maximum(x_out_cool, 1.0)) * tail_shape
    e_residual = jnp.maximum(e_inj - e_rad, 0.0)
    e_tail = jnp.minimum(jnp.maximum(ers, 0.0) * tail_fraction, 0.6 * e_residual)
    e_store = jnp.maximum(e_residual - e_tail, 0.0)
    e_channel = jnp.maximum(efs + ers, 0.0)
    e_fs_resid = jnp.where(e_channel > 0.0, e_store * jnp.maximum(efs, 0.0) / e_channel, 0.0)
    e_rs_resid = jnp.maximum(e_store - e_fs_resid, 0.0)
    surface_budget = e_fs_resid / jnp.maximum(4.0 * PI * scales.u0 * scales.r_in**3, 1.0e-30)
    density_budget = e_rs_resid / jnp.maximum(4.0 * PI * scales.u0 * scales.r_in**3, 1.0e-30)

    x_cool = 1.0 + xi * (x_ph_cool - 1.0)
    x_pre = 1.0 + (x_cool - 1.0) * (x_sh_se - 1.0) / jnp.maximum(x_out_cool - 1.0, 1.0e-12)
    shell_x = _shell_bin_centers(scales.x_out, n)
    shell_profile = jnp.interp(jnp.clip(x_pre, 1.0, x_sh_se), shell_x, shell_e)
    density_profile = _eta_csm_x(params, scales, x_cool, True, x_out_cool, x_sh_se)
    dx = (x_ph_cool - 1.0) / jnp.maximum(n - 1, 1)
    surf_int = x_cool**2 * jnp.maximum(shell_profile, 0.0)
    dens_int = x_cool**2 * jnp.maximum(density_profile, 0.0)
    retain = jnp.minimum(jnp.maximum(x_cool / jnp.maximum(x_out_cool, x_cool), 0.0), 1.0) ** pdv_power
    surface_norm = jnp.sum(0.5 * (surf_int[:-1] + surf_int[1:]) * dx)
    density_norm = jnp.sum(0.5 * (dens_int[:-1] + dens_int[1:]) * dx)
    surface_ret = jnp.sum(0.5 * ((surf_int * retain)[:-1] + (surf_int * retain)[1:]) * dx)
    density_ret = jnp.sum(0.5 * ((dens_int * retain)[:-1] + (dens_int * retain)[1:]) * dx)
    surface_budget = jnp.where(surface_norm > 0.0, surface_budget * surface_ret / surface_norm, 0.0)
    density_budget = jnp.where(density_norm > 0.0, density_budget * density_ret / density_norm + jnp.where(surface_norm <= 0.0, surface_budget, 0.0), density_budget)
    e_new = jnp.where(surface_norm > 0.0, surface_budget * jnp.maximum(shell_profile, 0.0) / surface_norm, 0.0)
    e_new = e_new + jnp.where(density_norm > 0.0, density_budget * jnp.maximum(density_profile, 0.0) / density_norm, 0.0)

    m_shell = phi * 4.0 * PI * scales.r_in**3 * scales.rho_ej_in
    t_tail0 = jnp.maximum(
        params.kappa * jnp.maximum(m_shell, 0.0) / (4.0 * PI * 2.0 * CLIGHT * jnp.maximum(x_out_cool * scales.r_in, scales.r_in)),
        x_out_cool * scales.r_in / CLIGHT,
    )
    return x_sh_se, x_out_cool, v_se, x_ph_cool, jnp.maximum(e_new, 0.0), e_tail, t_tail0, lum_pre


def _cooling_tail_step(e_tail, t_tail0, dt, r_in_r0, v_se, x_out_cool, scales):
    t_leak = jnp.maximum(t_tail0 / jnp.maximum(r_in_r0, 1.0e-12), x_out_cool * scales.r_in * r_in_r0 / CLIGHT)
    t_exp = jnp.maximum(scales.r_in * r_in_r0 / jnp.maximum(v_se, 1.0e-30), 1.0e-30)
    leak_rate = 1.0 / jnp.maximum(t_leak, 1.0e-30)
    total_rate = leak_rate + 0.15 / t_exp
    decay = jnp.exp(-jnp.minimum(total_rate * dt, 80.0))
    emitted = e_tail * (1.0 - decay) * leak_rate / jnp.maximum(total_rate, 1.0e-30)
    return jnp.maximum(e_tail * decay, 0.0), jnp.maximum(emitted / jnp.maximum(dt, 1.0e-30), 0.0)


def _advance_breakout_reservoir(e_bo, lum_in, dt, t_esc):
    t_esc = jnp.maximum(t_esc, 1.0e-30)
    decay = jnp.exp(-jnp.minimum(dt / t_esc, 80.0))
    e_new = e_bo * decay + jnp.maximum(lum_in, 0.0) * t_esc * (1.0 - decay)
    lum_avg = jnp.maximum((e_bo + jnp.maximum(lum_in, 0.0) * dt - e_new) / jnp.maximum(dt, 1.0e-30), 0.0)
    return jnp.maximum(e_new, 0.0), lum_avg


@partial(jax.jit, static_argnames=("n_steps", "n_zones"))
def _solve_dimless_transport_static_bpl(time_days, params: StaticPowerlawBPLParams, n_steps: int, n_zones: int):
    scales = _dimless_scales(params)
    t_req = jnp.maximum(time_days * DAY, 1.0)
    t_min = jnp.maximum(scales.t_in, 10.0)
    t_max = jnp.maximum(jnp.max(t_req), 1.01 * t_min)
    t_grid = jnp.geomspace(t_min, t_max, n_steps)
    e0 = jnp.zeros(n_zones, dtype=jnp.float64)
    shell0 = jnp.zeros(n_zones, dtype=jnp.float64)
    x_ph0 = _photosphere_dimless(params, scales, 1.0, 0.0, False, 1.0, scales.x_out)
    carry0 = (
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0e-6),
        jnp.array(1.0),
        x_ph0,
        e0,
        shell0,
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(False),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(4),
    )

    def step(carry, i):
        (
            x,
            w,
            phi,
            zeta,
            x_ph,
            e_grid,
            shell_e,
            e_inj,
            e_rad,
            efs,
            ers,
            in_cool,
            zeta_se,
            x_sh_se,
            x_out_cool,
            v_se,
            e_tail,
            t_tail0,
            lum_emergence,
            e_bo,
            rannacher_left,
        ) = carry
        t = t_grid[i]
        dt = t_grid[i + 1] - t
        dzeta = dt / scales.t_in
        x_old, w_old, phi_old, zeta_old, x_ph_old = x, w, phi, zeta, x_ph

        x_dyn, w_dyn, phi_dyn, zeta_dyn = _rk4_dimless(params, scales, x, w, phi, zeta, dzeta)
        x_next = jnp.where(in_cool, x, x_dyn)
        w_next = jnp.where(in_cool, w, w_dyn)
        phi_next = jnp.where(in_cool, phi, phi_dyn)
        zeta_next = jnp.where(in_cool, zeta + dzeta, zeta_dyn)
        crossed = (~in_cool) & (x_next >= scales.x_out) & (x_old < scales.x_out)
        frac = jnp.clip((scales.x_out - x_old) / jnp.maximum(x_next - x_old, 1.0e-30), 0.0, 1.0)
        x_cross = x_old + frac * (x_next - x_old)
        w_cross = w_old + frac * (w_next - w_old)
        phi_cross = phi_old + frac * (phi_next - phi_old)
        zeta_cross = zeta_old + frac * dzeta

        x_solve = jnp.where(crossed, 0.5 * (x_old + x_cross), x_next)
        w_solve = jnp.where(crossed, 0.5 * (w_old + w_cross), w_next)
        zeta_solve = jnp.where(crossed, zeta_old + 0.5 * frac * dzeta, zeta_next)
        lum_fs, lum_rs = _shock_luminosities_dimless(params, scales, x_solve, w_solve, zeta_solve, in_cool)
        lum_heat = lum_fs + lum_rs
        dy = scales.y_ratio * dzeta * jnp.where(crossed, frac, 1.0)
        dt_int = dt * jnp.where(crossed, frac, 1.0)
        r_in_r0 = jnp.where(in_cool, 1.0 + v_se * (zeta_next - zeta_se) * scales.t_in / scales.r_in, 1.0)
        x_ph_new_int = _photosphere_dimless(params, scales, x_solve, x_ph, False, 1.0, scales.x_out)
        x_ph_dot = (x_ph_new_int - x_ph_old) / jnp.maximum(dy, 1.0e-30)
        x_sh_dot = jnp.where(crossed, (x_cross - x_old) / jnp.maximum(dy, 1.0e-30), w_solve / jnp.maximum(scales.y_ratio, 1.0e-30))

        e_int = _solve_diffusion_dimless(
            params,
            scales,
            e_grid,
            dy,
            x_solve,
            x_ph_new_int,
            x_solve,
            x_sh_dot,
            x_ph_dot,
            lum_heat,
            False,
            1.0,
            scales.x_out,
            x_solve,
            rannacher_left,
        )
        e_grid = jnp.where(in_cool, e_grid, e_int)
        l_factor_int = PI * CLIGHT * scales.u0 * x_ph_new_int**2 * scales.r_in**2
        lum_int = l_factor_int * jnp.maximum(e_int[-1], 0.0)
        e_rad_dim = _dimless_energy_grid(e_int, x_solve, x_ph_new_int)
        e_rad_cgs_field = 4.0 * PI * scales.u0 * scales.r_in**3 * e_rad_dim
        e_inj_new = e_inj + jnp.where(in_cool, 0.0, lum_heat * dt_int)
        efs_new = efs + jnp.where(in_cool, 0.0, lum_fs * dt_int)
        ers_new = ers + jnp.where(in_cool, 0.0, lum_rs * dt_int)
        e_rad_new = e_rad + jnp.where(in_cool, 0.0, lum_int * dt_int)
        energy_dim_fs = lum_fs * dt_int / jnp.maximum(4.0 * PI * scales.u0 * scales.r_in**3, 1.0e-30)
        shell_e_new = _deposit_shell_energy(shell_e, x_old, x_solve, energy_dim_fs, scales.x_out)
        loss_dim = lum_int * dt_int / jnp.maximum(4.0 * PI * scales.u0 * scales.r_in**3, 1.0e-30)
        shell_e_new = _drain_shell_energy(shell_e_new, loss_dim, x_ph_new_int, scales.x_out)
        e_available = jnp.maximum(e_inj_new - e_rad_new - e_rad_cgs_field, 0.0)
        shell_dim = jnp.sum(shell_e_new * ((1.0 + (jnp.arange(n_zones, dtype=jnp.float64) + 1.0) * (scales.x_out - 1.0) / n_zones) ** 3 - (1.0 + jnp.arange(n_zones, dtype=jnp.float64) * (scales.x_out - 1.0) / n_zones) ** 3) / 3.0)
        shell_target = jnp.maximum(efs_new - e_rad_new - e_rad_cgs_field, 0.0) / jnp.maximum(4.0 * PI * scales.u0 * scales.r_in**3, 1.0e-30)
        shell_e_new = jnp.where(shell_dim > 0.0, shell_e_new * shell_target / shell_dim, shell_e_new)
        tau_ahead = _tau_dimless_powerlaw(params, scales, x_solve, False, 1.0, scales.x_out)
        tau_inner = _tau_dimless_powerlaw(params, scales, 1.0, False, 1.0, scales.x_out)
        tau_breakout = CLIGHT / jnp.maximum(w_solve * scales.v_max, 1.0e5)
        e_available = jnp.maximum(e_inj_new - e_rad_new - e_bo - e_rad_cgs_field, 0.0)
        escape_weight = jnp.clip(1.0 - tau_ahead / jnp.maximum(tau_breakout, 1.0e-30), 0.0, 1.0)
        dr_ahead = jnp.maximum((scales.x_out - x_solve) * scales.r_in, 0.0)
        t_raw = jnp.maximum(jnp.maximum(tau_ahead, 1.0) * dr_ahead / CLIGHT, x_solve * scales.r_in / CLIGHT)
        t_transfer = t_raw / jnp.maximum(escape_weight, 1.0e-6)
        eta_sh = jnp.maximum(x_solve, 1.0e-300) ** params.eta
        t_bo_cross = tau_breakout / jnp.maximum(params.kappa * scales.rho_csm_in * eta_sh * w_solve * scales.v_max, 1.0e-30)
        t_emit = jnp.maximum(jnp.maximum(t_raw, t_bo_cross), x_solve * scales.r_in / CLIGHT)
        e_transfer = e_available * (1.0 - jnp.exp(-jnp.minimum(dt_int / jnp.maximum(t_transfer, 1.0e-30), 80.0)))
        e_transfer = jnp.where((tau_ahead < tau_breakout) & (tau_inner > tau_breakout) & (~in_cool), e_transfer, 0.0)
        e_bo, lum_bo_avg = _advance_breakout_reservoir(e_bo, e_transfer / jnp.maximum(dt_int, 1.0e-30), dt_int, t_emit)
        e_rad_new = e_rad_new + jnp.where(in_cool, 0.0, lum_bo_avg * dt_int)
        shell_e_new = _drain_shell_energy(
            shell_e_new,
            lum_bo_avg * dt_int / jnp.maximum(4.0 * PI * scales.u0 * scales.r_in**3, 1.0e-30),
            x_ph_new_int,
            scales.x_out,
        )
        shell_e = jnp.where(in_cool, shell_e, shell_e_new)

        trans = _transition_to_cooling(params, scales, x_cross, w_cross, phi_cross, zeta_cross, x_ph_new_int, e_int, shell_e, e_inj_new, e_rad_new, efs_new, ers_new, lum_int)
        x_sh_se_new, x_out_cool_new, v_se_new, x_ph_cool0, e_cool0, e_tail_new, t_tail0_new, lum_em_new = trans
        in_cool_new = in_cool | crossed
        zeta_se_new = jnp.where(crossed, zeta_cross, zeta_se)
        x_sh_se = jnp.where(crossed, x_sh_se_new, x_sh_se)
        x_out_cool = jnp.where(crossed, x_out_cool_new, x_out_cool)
        v_se = jnp.where(crossed, v_se_new, v_se)
        x_ph = jnp.where(crossed, x_ph_cool0, jnp.where(in_cool, x_ph, x_ph_new_int))
        e_grid = jnp.where(crossed, e_cool0, e_grid)
        e_tail = jnp.where(crossed, e_tail_new, e_tail)
        t_tail0 = jnp.where(crossed, t_tail0_new, t_tail0)
        lum_emergence = jnp.where(crossed, lum_em_new, lum_emergence)
        zeta_next = jnp.where(crossed, zeta_cross + (1.0 - frac) * dzeta, zeta_next)
        x_next = jnp.where(crossed, x_cross, x_next)
        w_next = jnp.where(crossed, w_cross, w_next)
        phi_next = jnp.where(crossed, phi_cross, phi_next)

        r_in_r0_cool = 1.0 + v_se * (zeta_next - zeta_se_new) * scales.t_in / scales.r_in
        dy_cool = scales.y_ratio * dzeta * jnp.where(crossed, 1.0 - frac, 1.0)
        dt_cool = dt * jnp.where(crossed, 1.0 - frac, 1.0)
        x_ph_cool_new = _photosphere_dimless(params, scales, 1.0, x_ph, True, r_in_r0_cool, x_out_cool)
        x_ph_dot_cool = (x_ph_cool_new - x_ph) / jnp.maximum(dy_cool, 1.0e-30)
        e_cool = _solve_diffusion_dimless(
            params,
            scales,
            e_grid,
            dy_cool,
            1.0,
            x_ph_cool_new,
            x_next,
            0.0,
            x_ph_dot_cool,
            0.0,
            True,
            r_in_r0_cool,
            x_out_cool,
            x_sh_se,
            4,
        )
        e_tail_after, l_tail = _cooling_tail_step(e_tail, t_tail0, dt_cool, r_in_r0_cool, v_se, x_out_cool, scales)
        l_factor_cool = 2.0 * PI * CLIGHT * scales.u0 * x_ph_cool_new**2 * scales.r_in**2
        lum_cool = l_factor_cool * jnp.maximum(e_cool[-1], 0.0) / jnp.maximum(r_in_r0_cool, 1.0e-30) ** 2 + l_tail
        early_cool = (zeta_next - zeta_se_new) * scales.t_in < DAY
        lum_cool = jnp.where(early_cool & (lum_emergence > 0.0), jnp.minimum(lum_cool, 1.05 * lum_emergence), lum_cool)
        e_grid = jnp.where(in_cool_new, e_cool, e_grid)
        e_tail = jnp.where(in_cool_new, e_tail_after, e_tail)
        x_ph = jnp.where(in_cool_new, x_ph_cool_new, x_ph)
        lum_interaction_emit = lum_int + lum_bo_avg
        lum_cross_emit = (lum_interaction_emit * dt_int + lum_cool * dt_cool) / jnp.maximum(dt_int + dt_cool, 1.0e-30)
        lum = jnp.where(crossed, lum_cross_emit, jnp.where(in_cool_new, lum_cool, lum_interaction_emit))
        lum_shock = jnp.where(in_cool_new, 0.0, lum_heat)
        radius = jnp.where(in_cool_new, x_ph * scales.r_in * r_in_r0_cool, x_next * scales.r_in)
        velocity = w_next * scales.v_max
        tau = jnp.where(
            in_cool_new,
            _tau_dimless_powerlaw(params, scales, 1.0, True, r_in_r0_cool, x_out_cool),
            _tau_dimless_powerlaw(params, scales, x_next, False, 1.0, scales.x_out),
        )
        sample = jnp.array([t_grid[i + 1], lum, lum_shock, radius, velocity, phi_next * 4.0 * PI * scales.r_in**3 * scales.rho_ej_in, tau], dtype=jnp.float64)
        rannacher_next = jnp.maximum(rannacher_left - 1, 0)
        return (
            x_next,
            w_next,
            phi_next,
            zeta_next,
            x_ph,
            e_grid,
            shell_e,
            e_inj_new,
            e_rad_new,
            efs_new,
            ers_new,
            in_cool_new,
            zeta_se_new,
            x_sh_se,
            x_out_cool,
            v_se,
            e_tail,
            t_tail0,
            lum_emergence,
            e_bo,
            rannacher_next,
        ), sample

    _, samples = jax.lax.scan(step, carry0, jnp.arange(n_steps - 1))
    first = jnp.array([t_grid[0], 0.0, 0.0, scales.r_in, scales.v_max, 0.0, 0.0], dtype=jnp.float64)
    history = jnp.vstack([first[None, :], samples])
    lbol = jnp.interp(t_req, history[:, 0], history[:, 1], left=0.0, right=history[-1, 1])
    lshock = jnp.interp(t_req, history[:, 0], history[:, 2], left=0.0, right=history[-1, 2])
    radius = jnp.interp(t_req, history[:, 0], history[:, 3], left=history[0, 3], right=history[-1, 3])
    velocity = jnp.interp(t_req, history[:, 0], history[:, 4], left=history[0, 4], right=history[-1, 4])
    mass = jnp.interp(t_req, history[:, 0], history[:, 5], left=history[0, 5], right=history[-1, 5])
    tau = jnp.interp(t_req, history[:, 0], history[:, 6], left=history[0, 6], right=history[-1, 6])
    return StaticPowerlawBPLResult(
        time=t_req,
        lbol=jnp.maximum(lbol, 0.0),
        lbol_shock=jnp.maximum(lshock, 0.0),
        radius=radius,
        velocity=velocity,
        shell_mass=mass,
        internal_energy=jnp.zeros_like(lbol),
        optical_depth=jnp.maximum(tau, 0.0),
        t_leak=jnp.ones_like(lbol),
    )


@partial(jax.jit, static_argnames=("mode", "n_steps", "n_zones"))
def solve_static_powerlaw_bpl(
    time_days,
    params: StaticPowerlawBPLParams,
    mode: str = "transport",
    n_steps: int = 16384,
    n_zones: int = 40,
):
    """Evaluate a JAX-native static-power-law CSM light curve.

    ``mode='simple'`` returns the instantaneous radiative shock luminosity.
    ``mode='transport'`` evolves a finite-volume radiation grid over the full
    static shell, then homologously expands that grid after shock emergence.
    The hydrodynamics and density normalizations are shared by both modes and
    match the current static finite-shell convention.
    """
    time_days = jnp.asarray(time_days, dtype=jnp.float64)
    if mode == "transport":
        return _solve_dimless_transport_static_bpl(time_days, params, n_steps=n_steps, n_zones=n_zones)

    rho_inner = _static_powerlaw_rho_inner(params)
    v_t, v_max, rho0_4pi = _bpl_coefficients(params)

    t_req = jnp.maximum(time_days * DAY, 1.0)
    t_min = jnp.maximum(params.r_inner / jnp.maximum(v_max, 1.0), 10.0)
    t_max = jnp.maximum(jnp.max(t_req), 1.01 * t_min)
    t_grid = jnp.geomspace(t_min, t_max, n_steps)

    initial_mass = jnp.maximum(1.0e-4 * (params.mej_sn + params.m_csm) * SOLAR_MASS, 1.0)
    state0 = jnp.array([params.r_inner, 0.9 * v_max, initial_mass, 0.0], dtype=jnp.float64)
    transport = mode == "transport"
    radius_ref = _static_radius_grid(params, n_zones)
    rho_ref = _rho_csm(params, rho_inner, radius_ref)
    u_rad0 = jnp.zeros(n_zones, dtype=jnp.float64)
    carry0 = (
        state0,
        u_rad0,
        jnp.array(0.0),
        jnp.array(t_grid[-1]),
        jnp.array(0.0),
        jnp.array(1.0),
        jnp.array(0.0),
    )

    def step(carry, i):
        state, u_rad, emerged, t_emerge, u_emerge, scale_prev, e_breakout = carry
        t = t_grid[i]
        dt = t_grid[i + 1] - t
        hydro_next = _rk4_step(params, rho_inner, v_t, v_max, rho0_4pi, state, t, dt, False)
        ballistic = state.at[0].set(state[0] + state[1] * dt)
        next_state = jnp.where(emerged > 0.5, ballistic, hydro_next)
        crossed = (emerged < 0.5) & (next_state[0] >= params.r_outer)
        emerged_next = jnp.where(crossed, 1.0, emerged)
        t_emerge_next = jnp.where(crossed, t_grid[i + 1], t_emerge)
        u_emerge_next = jnp.where(crossed, next_state[1], u_emerge)

        _, l_heat, l_obs, t_leak = _rhs(
            params, rho_inner, v_t, v_max, rho0_4pi, next_state, t_grid[i + 1], False
        )
        tau_ahead = _tau_static_powerlaw(params, rho_inner, jnp.minimum(next_state[0], params.r_outer))
        tau_breakout = CLIGHT / jnp.maximum(next_state[1], 1.0e5)
        breakout_weight = jnp.clip(1.0 - tau_ahead / jnp.maximum(tau_breakout, 1.0e-30), 0.0, 1.0)
        breakout_weight = jnp.where(emerged > 0.5, 0.0, breakout_weight)
        dr_ahead = jnp.maximum(params.r_outer - jnp.minimum(next_state[0], params.r_outer), 0.0)
        t_breakout = jnp.maximum(jnp.maximum(tau_ahead, 1.0) * dr_ahead / CLIGHT, next_state[0] / CLIGHT)
        e_breakout_pre = e_breakout + breakout_weight * l_heat * dt
        e_breakout_emit = e_breakout_pre * (1.0 - jnp.exp(-dt / jnp.maximum(t_breakout, 1.0)))
        l_prompt = e_breakout_emit / dt
        e_breakout_next = jnp.maximum(e_breakout_pre - e_breakout_emit, 0.0)
        l_heat_source = jnp.where(emerged > 0.5, 0.0, (1.0 - breakout_weight) * l_heat)
        l_heat_out = jnp.where(emerged_next > 0.5, 0.0, l_heat)
        tau = _tau_static_powerlaw(params, rho_inner, next_state[0])
        scale = jnp.where(
            emerged_next > 0.5,
            jnp.maximum(
                (params.r_outer + jnp.maximum(u_emerge_next, 0.0) * jnp.maximum(t_grid[i + 1] - t_emerge_next, 0.0))
                / params.r_outer,
                1.0,
            ),
            1.0,
        )
        radius = radius_ref * scale
        rho = rho_ref / scale**3
        u_rad_scaled = jnp.where(transport, u_rad * (scale_prev / scale) ** 4, u_rad)
        theta = jnp.where(i < 4, 1.0, 0.5)
        u_rad_next, l_transport, r_ph, tau_grid0, e_transport = _transport_step(
            params,
            radius,
            rho,
            u_rad_scaled,
            dt,
            jnp.minimum(next_state[0], params.r_outer),
            l_heat_source,
            emerged_next > 0.5,
            theta,
        )
        u_rad_next = jnp.where(transport, u_rad_next, u_rad)
        lbol = jnp.where(transport, l_transport + l_prompt, l_heat)
        e_out = jnp.where(transport, e_transport, next_state[3])
        tau_out = jnp.where(transport, tau_grid0, tau)
        r_ph_out = jnp.where(transport, r_ph, next_state[0])
        sample = jnp.array(
            [
                t_grid[i + 1],
                lbol,
                l_heat_out,
                next_state[0],
                next_state[1],
                next_state[2],
                e_out,
                tau_out,
                t_leak,
            ],
            dtype=jnp.float64,
        )
        return (next_state, u_rad_next, emerged_next, t_emerge_next, u_emerge_next, scale, e_breakout_next), sample

    _, samples = jax.lax.scan(step, carry0, jnp.arange(n_steps - 1))
    first = jnp.array([t_grid[0], 0.0, 0.0, state0[0], state0[1], state0[2], state0[3], 0.0, 1.0])
    history = jnp.vstack([first[None, :], samples])

    lbol = jnp.interp(t_req, history[:, 0], history[:, 1], left=0.0, right=history[-1, 1])
    lshock = jnp.interp(t_req, history[:, 0], history[:, 2], left=0.0, right=history[-1, 2])
    radius = jnp.interp(t_req, history[:, 0], history[:, 3], left=history[0, 3], right=history[-1, 3])
    velocity = jnp.interp(t_req, history[:, 0], history[:, 4], left=history[0, 4], right=history[-1, 4])
    mass = jnp.interp(t_req, history[:, 0], history[:, 5], left=history[0, 5], right=history[-1, 5])
    e_rad = jnp.interp(t_req, history[:, 0], history[:, 6], left=0.0, right=history[-1, 6])
    tau = jnp.interp(t_req, history[:, 0], history[:, 7], left=history[0, 7], right=history[-1, 7])
    t_leak = jnp.interp(t_req, history[:, 0], history[:, 8], left=history[0, 8], right=history[-1, 8])

    return StaticPowerlawBPLResult(
        time=t_req,
        lbol=jnp.maximum(lbol, 0.0),
        lbol_shock=jnp.maximum(lshock, 0.0),
        radius=radius,
        velocity=velocity,
        shell_mass=mass,
        internal_energy=e_rad,
        optical_depth=jnp.maximum(tau, 0.0),
        t_leak=jnp.maximum(t_leak, 1.0),
    )


def static_powerlaw_csm_bpl_bolometric(time, **kwargs):
    """Convenience wrapper with the same static BPL parameter names as redback_csm.models."""
    mode = str(kwargs.pop("mode", "transport")).lower()
    n_steps = int(kwargs.pop("n_steps", kwargs.pop("n_grid", 16384)))
    n_zones = int(kwargs.pop("n_zones", kwargs.pop("n_rad_zones", 40)))
    params = StaticPowerlawBPLParams(
        eta=kwargs.pop("eta"),
        r_inner=kwargs.pop("r_inner"),
        r_outer=kwargs.pop("r_outer"),
        delta_sn=kwargs.pop("delta_sn"),
        nn_sn=kwargs.pop("nn_sn"),
        mej_sn=kwargs.pop("mej_sn"),
        esn=kwargs.pop("esn"),
        eff=kwargs.pop("eff"),
        m_csm=kwargs.pop("m_csm"),
        kappa=kwargs.pop("kappa", 0.34),
        vej_max_ratio=kwargs.pop("vej_max_ratio", kwargs.pop("A_ratio", 3.0)),
    )
    result = solve_static_powerlaw_bpl(
        jnp.asarray(time, dtype=jnp.float64), params, mode=mode, n_steps=n_steps, n_zones=n_zones
    )
    return result.lbol
