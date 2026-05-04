"""Audit the Figure 11 CSM interaction setup.

This script does not modify the model.  It prints the dimensional and
nondimensional quantities that control transport mode, then samples the current
Fortran light curve around shock emergence.  It is intended to separate
initial-condition/setup issues from cooling-handoff issues.
"""

from __future__ import annotations

import math

import numpy as np

from redback_csm.core import (
    _get_lc_static_powerlaw_csm_bpl,
    _get_last_transport_diagnostics,
)


RSUN = 6.957e10
MSUN = 1.98847e33
C_LIGHT = 2.99792458e10
DAY = 86400.0
FOE = 1.0e51


def bpl_coefficients(mej_msun: float, esn_foe: float, delta: float, nn: float, cutoff_ratio: float):
    """Mirror fortran/get_vals.f90:get_bpl_coeffs for the current default."""
    mej = mej_msun * MSUN
    esn = esn_foe * FOE
    if cutoff_ratio > 1.0:
        a = cutoff_ratio
        im = 1.0 / (3.0 - delta) + (1.0 - a ** (3.0 - nn)) / (nn - 3.0)
        ie = 1.0 / (5.0 - delta) + (1.0 - a ** (5.0 - nn)) / (nn - 5.0)
        vt = math.sqrt(2.0 * esn * im / (mej * ie))
        vmax = a * vt
        rho0 = mej / (vt**3 * im)
    else:
        vt = math.sqrt(
            2.0 * (5.0 - delta) * (nn - 5.0) * esn
            / ((3.0 - delta) * (nn - 3.0) * mej)
        )
        vmax = vt
        rho0 = ((3.0 - delta) * (nn - 3.0) * mej) ** 2.5
        rho0 /= (2.0 * (5.0 - delta) * (nn - 5.0) * esn) ** 1.5
        rho0 /= nn - delta
    return vt, vmax, rho0


def csm_rho_in(r_inner: float, r_outer: float, m_csm_msun: float, s: float) -> float:
    eta = -s
    m_csm = m_csm_msun * MSUN
    if abs(eta + 3.0) < 1.0e-12:
        mass_factor = 4.0 * math.pi * r_inner**3 * math.log(r_outer / r_inner)
    else:
        mass_factor = (
            4.0
            * math.pi
            * r_inner ** (-eta)
            * (r_outer ** (eta + 3.0) - r_inner ** (eta + 3.0))
            / (eta + 3.0)
        )
    return m_csm / mass_factor


def optical_depth_integral(x_out: float, s: float) -> float:
    if abs(s - 1.0) < 1.0e-12:
        return math.log(x_out)
    return (x_out ** (1.0 - s) - 1.0) / (1.0 - s)


def print_setup_table():
    r_inner = 5.0e2 * RSUN
    vt, vmax, rho0 = bpl_coefficients(5.0, 1.0, 1.0, 10.0, 3.0)
    t_in = r_inner / vmax
    v_char = math.sqrt(2.0 * FOE / (5.0 * MSUN))
    rho_ej_in = rho0 * (vt / vmax) ** 10.0 / (4.0 * math.pi * t_in**3)

    print("BPL ejecta setup used by current Python default")
    print(f"  cutoff_ratio A         = 3")
    print(f"  v_tr                  = {vt / 1e5:9.2f} km/s")
    print(f"  v_ej_max              = {vmax / 1e5:9.2f} km/s")
    print(f"  sqrt(2E/M)            = {v_char / 1e5:9.2f} km/s")
    print(f"  t_in                  = {t_in / DAY:9.4f} d")
    print(f"  rho_ej_in             = {rho_ej_in:9.3e} g/cm^3")
    print()

    for r_outer_rsun in (5.0e3, 5.0e4):
        r_outer = r_outer_rsun * RSUN
        x_out = r_outer / r_inner
        print(f"CSM setup: R_out={r_outer_rsun:.0f} R_sun, x_out={x_out:.1f}")
        print("  s    rho_csm_in      q          tau_tot    t_diff_in(d)  t_dyn_char(d)")
        for s in (0.0, 0.5, 1.0, 1.5, 2.0):
            rho_in = csm_rho_in(r_inner, r_outer, 1.0, s)
            q = rho_in / rho_ej_in
            tau_in = 3.0 * 0.2 * rho_in * r_inner
            t_diff = tau_in * r_inner / C_LIGHT
            tau_tot = 0.2 * rho_in * r_inner * optical_depth_integral(x_out, s)
            t_dyn = r_outer / v_char
            print(
                f"  {s:3.1f}  {rho_in:11.3e}  {q:9.3e}  "
                f"{tau_tot:8.2f}  {t_diff / DAY:11.3f}  {t_dyn / DAY:11.2f}"
            )
        print()


def run_case(r_outer_rsun: float, s: float):
    r_inner = 5.0e2 * RSUN
    r_outer = r_outer_rsun * RSUN
    out = _get_lc_static_powerlaw_csm_bpl(
        eta=-s,
        r_inner=r_inner,
        r_outer=r_outer,
        delta_sn=1.0,
        nn_sn=10.0,
        mej_sn=5.0,
        esn=1.0,
        eff=1.0,
        m_csm=1.0,
        mode="transport",
        kappa=0.2,
        n_rad_zones=40,
    )
    diag = _get_last_transport_diagnostics()
    time_d = out.time / DAY
    lbol = out.lbol
    lheat = out.lbol_shock
    rfs_rsun = diag.r_forward_shock / RSUN
    emerged = np.where(rfs_rsun >= 0.999 * r_outer_rsun)[0]
    i_se = int(emerged[0]) if len(emerged) else int(np.argmax(lbol))

    print(f"Run: R_out={r_outer_rsun:.0f} R_sun, s={s:g}")
    print(f"  peak Lbol             = {np.nanmax(lbol):.3e} erg/s at {time_d[np.nanargmax(lbol)]:.3f} d")
    print(f"  approx t_se           = {time_d[i_se]:.3f} d")
    print(f"  Lbol/Lheat at t_se    = {lbol[i_se] / max(lheat[i_se], 1e-300):.3f}")
    print("  samples around emergence:")
    print("    t(d)        Lbol          Lheat       Rfs(Rsun)")
    lo = max(0, i_se - 6)
    hi = min(len(time_d), i_se + 7)
    for i in range(lo, hi):
        print(f"    {time_d[i]:8.4f}  {lbol[i]:11.4e}  {lheat[i]:11.4e}  {rfs_rsun[i]:10.2f}")
    print()


def main():
    print_setup_table()
    for r_outer_rsun in (5.0e3, 5.0e4):
        for s in (0.0, 2.0):
            run_case(r_outer_rsun, s)


if __name__ == "__main__":
    main()
