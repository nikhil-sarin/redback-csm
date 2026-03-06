"""
radio.py — Self-contained synchrotron radio emission physics for CSM interaction models.

All calculations are in CGS units. Output is in mJy.

The key scaling relation (Chevalier 1998) correctly gives:
    L_nu  ∝  epsilon_B^{(p+1)/4}
which is positive for all p > -1.  This corrects the common error of confusing the
spectral-index exponent (1-p)/2 with the epsilon_B dependence.
"""

import numpy as _np

# Physical constants (CGS)
_QE     = 4.803e-10     # esu (elementary charge)
_ME     = 9.109e-28     # g  (electron mass)
_C      = 2.998e10      # cm/s
_SIGMAT = 6.6524e-25    # cm^2 (Thomson cross-section)
_MP     = 1.6726e-24    # g  (proton mass)
_DAY    = 86400.0       # s


def synchrotron_flux_density(
    time_days,
    vshell_cgs,
    rho_csm_cgs,
    redshift,
    logepsb,
    logepse,
    p,
    frequency,
    luminosity_distance_cm,
):
    """
    Compute synchrotron radio flux density from a CSM-interaction shock.

    Uses the Chevalier (1998) formalism with self-absorption.  All input arrays
    should be on the same time grid (the Fortran output grid).

    Parameters
    ----------
    time_days : array_like
        Observer-frame time in days.
    vshell_cgs : array_like
        Shock velocity in cm/s (from lc.vshell).
    rho_csm_cgs : array_like
        Upstream CSM mass density at the shock in g/cm^3.
    redshift : float
        Source redshift z.
    logepsb : float
        log10(epsilon_B); magnetic energy fraction of shock ram pressure.
    logepse : float
        log10(epsilon_e); electron energy fraction.
    p : float
        Electron power-law index (typically 2–4).
    frequency : float or array_like
        Observing frequency in Hz (observer frame).  Can be a scalar or array of
        the same length as time_days.
    luminosity_distance_cm : float
        Luminosity distance in cm.

    Returns
    -------
    flux_mJy : ndarray
        Flux density in mJy, same shape as time_days.
    """
    time_days = _np.asarray(time_days, dtype=float)
    vshell    = _np.asarray(vshell_cgs, dtype=float)
    rho       = _np.asarray(rho_csm_cgs, dtype=float)

    eps_b = 10.0 ** logepsb
    eps_e = 10.0 ** logepse
    dl    = luminosity_distance_cm
    z     = redshift

    # K-correction: source-frame quantities
    t_src   = time_days / (1.0 + z)          # days, rest frame
    t_src_s = t_src * _DAY                   # seconds

    # Broadcast frequency to the time grid
    freq_obs = _np.asarray(frequency, dtype=float)
    if freq_obs.ndim == 0:
        freq_obs = _np.full_like(time_days, float(frequency))
    freq_src = freq_obs * (1.0 + z)          # Hz, rest frame

    # Shock radius (use directly from Fortran rph; vshell * t is a cross-check)
    # Here we re-derive from vshell * t_src for self-consistency with the
    # density calculation, which also uses rph.  The caller can pass either.
    r_s = vshell * t_src_s                   # cm  (= rph approximately)

    # Magnetic field energy density: u_B = eps_B * rho * v^2
    u_b = eps_b * rho * vshell ** 2          # erg/cm^3

    # Magnetic field strength
    B = _np.sqrt(8.0 * _np.pi * u_b)        # Gauss

    # Larmor (cyclotron) frequency
    nu_L = _QE * B / (2.0 * _np.pi * _ME * _C)  # Hz

    # Number density of radiating electrons: n_e = eps_e * (rho / m_p)
    n_e = eps_e * (rho / _MP)               # cm^-3

    # Total number of electrons in a sphere of radius r_s
    N_0 = (4.0 / 3.0) * _np.pi * r_s ** 3 * n_e  # dimensionless count

    # Synchrotron power normalisaton (erg/s/Hz at nu = nu_L)
    #   C_0 = (4/3) N_0 sigma_T c u_B
    C_0 = (4.0 / 3.0) * N_0 * _SIGMAT * _C * u_b  # erg/s

    # Optically-thin spectral luminosity (erg/s/Hz)
    #   L_nu = C_0 / (2 nu_L) * (nu / nu_L)^{(1-p)/2}
    #
    # epsilon_B dependence check:
    #   C_0   ∝  u_B  ∝  eps_B
    #   nu_L  ∝  B    ∝  eps_B^{1/2}
    #   (nu/nu_L)^{(1-p)/2}  ∝  nu_L^{(p-1)/2}  ∝  eps_B^{(p-1)/4}
    #   Overall: L_nu  ∝  eps_B * eps_B^{-1/2} * eps_B^{(p-1)/4}
    #           = eps_B^{1 - 1/2 + (p-1)/4}  =  eps_B^{(p+1)/4}  ✓
    L_nu = C_0 / (2.0 * nu_L) * (freq_src / nu_L) ** ((1.0 - p) / 2.0)  # erg/s/Hz

    # Flux density at observer (Jy)
    flux_Jy = L_nu / (4.0 * _np.pi * dl ** 2) / 1.0e-23  # Jy

    # Self-absorption frequency and suppression
    # From Chevalier (1998), the SSA flux peak at nu_SSA is F_SSA, and
    # below nu_SSA the spectrum rises as nu^{5/2}.
    #
    # Spectral peak flux density in Jy (at nu_L)
    F_peak_Jy = C_0 / (2.0 * nu_L) / (4.0 * _np.pi * dl ** 2) / 1.0e-23

    # SSA optical depth ~ 1 condition gives nu_SSA.
    # Following Chevalier (1998) Eq. 9 / Björnsson & Fransson (2004):
    #   nu_SSA = [ (3^1.5 * q_e^0.5 * B^0.5 * F_peak_Jy * 1e-23 * nu_L^{beta-1} * dl^2)
    #              / (4 pi^1.5 * r_s^2 * c^0.5 * m_e^1.5) ]^{2/(2 beta + 3)}
    # where beta = (p+4)/2 — 1 = (p+2)/2 ... actually we use beta = (p-1)/2 + 1 = (p+1)/2
    # More precisely, using the convention in the prototype code: beta = 1 - (1-p)/2 = (p+1)/2
    beta = (p + 1.0) / 2.0

    # F_peak in CGS flux units for SSA formula (erg/s/Hz/cm^2)
    F_peak_cgs = F_peak_Jy * 1.0e-23

    numerator   = dl ** 2 * 3.0 ** 1.5 * _QE ** 0.5 * B ** 0.5 * F_peak_cgs * nu_L ** (beta - 1.0)
    denominator = 4.0 * _np.pi ** 1.5 * r_s ** 2 * _C ** 0.5 * _ME ** 1.5
    nu_ssa = (numerator / denominator) ** (2.0 / (2.0 * beta + 3.0))  # Hz (rest frame)

    # SSA flux at nu_SSA
    F_ssa_Jy = F_peak_Jy * (nu_ssa / nu_L) ** (1.0 - beta)

    # Apply SSA suppression where freq_src < nu_ssa
    flux_mJy = flux_Jy * 1.0e3  # convert to mJy
    ssa_mask = freq_src < nu_ssa
    if _np.any(ssa_mask):
        flux_mJy[ssa_mask] = (
            F_ssa_Jy[ssa_mask] * (freq_src[ssa_mask] / nu_ssa[ssa_mask]) ** 2.5
            * 1.0e3   # Jy → mJy
        )

    # Guard against non-physical negative values (can occur at very early times when
    # r_s ~ 0 and numerical noise dominates)
    flux_mJy = _np.where(_np.isfinite(flux_mJy) & (flux_mJy > 0.0), flux_mJy, 0.0)

    return flux_mJy
