"""
Thermal X-ray emission from CSM-interaction shocks.

This module uses standard optically thin thermal free-free formulae:

    epsilon_nu = 6.8e-38 Z^2 n_e n_i T^{-1/2} exp(-h nu / kT) g_ff
    epsilon_ff = 1.426e-27 T^{1/2} n_e n_i g_ff

with a strong-shock temperature kT = (3/16) mu m_p v_sh^2. The model-specific
closure is the emission-measure estimate:

    EM ~= M_swept * rho_post / (mu_e * mu_i * m_p^2)

which treats the shocked CSM as a compressed thin shell. That closure is an
inference-ready approximation, not a resolved cooling-layer calculation.

References:
    Rybicki & Lightman 1979, Radiative Processes in Astrophysics
    Chevalier & Fransson 1994, ApJ, 420, 268
    Margalit, Quataert & Ho 2022, ApJ, 928, 122
"""

import numpy as _np

_C = 2.99792458e10
_MP = 1.67262192369e-24
_KEV = 1.602176634e-9
_H = 6.62607015e-27
_KEV_TO_HZ = _KEV / _H
_TRAPEZOID = getattr(_np, "trapezoid", _np.trapz)
_FF_BOL_COEFF = 1.426e-27


def shock_temperature_kev(vshell_cgs, mu=0.62, electron_temperature_fraction=1.0):
    """
    Strong-shock post-shock temperature in keV.

    kT = (3/16) mu m_p v_sh^2. The optional electron-temperature fraction allows
    a cooler electron population than the mean post-shock ion temperature.
    """
    vshell = _np.asarray(vshell_cgs, dtype=float)
    kT = (3.0 / 16.0) * mu * _MP * _np.maximum(vshell, 0.0) ** 2 / _KEV
    return _np.maximum(kT * electron_temperature_fraction, 1.0e-6)


def photoelectric_cross_section_per_h(energy_kev):
    """
    Approximate neutral-gas photoelectric cross-section per H atom.

    This is a smooth Morrison-McCammon-like approximation intended for fast
    inference, not detailed spectral fitting. It is adequate for suppressing
    soft X-rays when a host or CSM column is supplied.
    """
    energy = _np.maximum(_np.asarray(energy_kev, dtype=float), 0.03)
    return 2.4e-22 * energy ** -2.6


def local_csm_column_density(r_shock_cgs, rho_csm_cgs, mu_h=1.4, column_factor=1.0):
    """
    Crude local upstream column estimate, N_H ~ rho r / (mu_H m_p).

    Use this only as an optional CSM-absorption scale. A full column integral
    should be supplied externally when the absorbing CSM structure is known.
    """
    r_shock = _np.asarray(r_shock_cgs, dtype=float)
    rho = _np.asarray(rho_csm_cgs, dtype=float)
    column = column_factor * _np.maximum(rho, 0.0) * _np.maximum(r_shock, 0.0)
    return column / (mu_h * _MP)


def cumulative_swept_csm_mass(radius_cgs, rho_csm_cgs):
    """
    Estimate swept-up CSM mass along the shock trajectory.

    The integral is M_csm(<R) = int 4 pi r^2 rho(r) dr along the sampled shock
    radii. Negative radius steps are ignored.
    """
    radius = _np.asarray(radius_cgs, dtype=float)
    rho = _np.asarray(rho_csm_cgs, dtype=float)
    mass = _np.zeros_like(radius)
    if radius.size < 2:
        return mass

    dr = _np.maximum(_np.diff(radius), 0.0)
    integrand = 4.0 * _np.pi * _np.maximum(radius, 0.0) ** 2 * _np.maximum(rho, 0.0)
    dm = 0.5 * (integrand[1:] + integrand[:-1]) * dr
    mass[1:] = _np.cumsum(_np.where(_np.isfinite(dm) & (dm > 0.0), dm, 0.0))
    return mass


def freefree_emission_measure_luminosity(
    swept_csm_mass_cgs,
    rho_csm_cgs,
    temperature_kev,
    compression_factor=4.0,
    mu_e=1.18,
    mu_i=1.30,
    gaunt_factor=1.2,
    emission_measure_scale=1.0,
):
    """
    Bolometric thermal free-free luminosity from shocked CSM.

    Uses L_ff = 1.426e-27 g_B T^1/2 int n_e n_i dV. The emission measure is
    estimated from the swept CSM mass and compressed post-shock density:
    EM ~= M_swept rho_post / (mu_e mu_i m_p^2).
    """
    mass = _np.asarray(swept_csm_mass_cgs, dtype=float)
    rho_up = _np.asarray(rho_csm_cgs, dtype=float)
    temp_kev = _np.asarray(temperature_kev, dtype=float)

    rho_post = compression_factor * _np.maximum(rho_up, 0.0)
    temp_k = _np.maximum(temp_kev * _KEV / 1.380649e-16, 1.0)
    emission_measure = (
        _np.maximum(mass, 0.0) * rho_post / (mu_e * mu_i * _MP ** 2)
    )
    luminosity = (
        emission_measure_scale
        * _FF_BOL_COEFF
        * gaunt_factor
        * _np.sqrt(temp_k)
        * emission_measure
    )
    return _np.where(_np.isfinite(luminosity) & (luminosity > 0.0), luminosity, 0.0)


def thermal_bremsstrahlung_xray(
    time_days,
    shock_luminosity_cgs,
    vshell_cgs,
    redshift,
    logepsx,
    luminosity_distance_cm,
    e_min_kev=0.3,
    e_max_kev=10.0,
    output_format="luminosity",
    energy_kev=None,
    frequency=None,
    n_h_host=0.0,
    n_h_mw=0.0,
    n_h_csm=None,
    rho_csm_cgs=None,
    radius_cgs=None,
    swept_csm_mass_cgs=None,
    mu=0.62,
    mu_e=1.18,
    mu_i=1.30,
    electron_temperature_fraction=1.0,
    compression_factor=4.0,
    gaunt_factor=1.2,
    normalization="emission_measure",
    max_xray_efficiency=1.0,
    absorb=True,
    n_energy=128,
):
    """
    Compute thermal bremsstrahlung X-ray emission from a CSM-interaction shock.

    Parameters
    ----------
    time_days : array_like
        Observer-frame time in days. Used for shape/broadcasting.
    shock_luminosity_cgs : array_like
        Shock power in erg/s, used as an energy cap by default.
    vshell_cgs : array_like
        Shock velocity in cm/s.
    redshift : float
        Source redshift.
    logepsx : float
        In emission-measure mode, log10 multiplicative scale for the emitting
        emission measure. In shock-power mode, log10 shock-power fraction.
    luminosity_distance_cm : float
        Luminosity distance in cm.
    e_min_kev, e_max_kev : float
        Observer-frame X-ray band edges in keV.
    output_format : {"luminosity", "flux", "spectral_luminosity", "flux_density"}
        Quantity returned. Band outputs are erg/s or erg/s/cm^2. Spectral
        luminosity is dL/dE in erg/s/keV. Flux density is mJy.
    energy_kev : float, optional
        Observer-frame photon energy for spectral outputs.
    frequency : float, optional
        Observer-frame frequency in Hz; alternative to energy_kev.
    n_h_host, n_h_mw : float
        Host and Milky Way neutral hydrogen columns in cm^-2.
    n_h_csm : array_like, optional
        Additional CSM column in cm^-2 on the model time grid.
    rho_csm_cgs : array_like, optional
        Upstream CSM density at the shock in g/cm^3.
    radius_cgs : array_like, optional
        Shock radius in cm; used to estimate swept CSM mass if not supplied.
    swept_csm_mass_cgs : array_like, optional
        Swept CSM mass in g. If omitted, it is integrated from radius_cgs and
        rho_csm_cgs.
    mu : float
        Mean molecular weight for the shock temperature.
    electron_temperature_fraction : float
        Fraction of the strong-shock mean temperature reached by electrons.
    normalization : {"emission_measure", "shock_power"}
        Emission-measure mode computes physical free-free luminosity from
        density and swept mass. Shock-power mode is the old power-normalised
        approximation.
    max_xray_efficiency : float
        Cap L_ff,bol <= max_xray_efficiency * shock power in emission-measure
        mode. Set to None or a negative value to disable.
    absorb : bool
        Apply photoelectric absorption if True.
    n_energy : int
        Number of energy samples for band integration.

    Returns
    -------
    ndarray
        X-ray luminosity, flux, spectral luminosity, or flux density.
    """
    time_days = _np.asarray(time_days, dtype=float)
    shock_luminosity = _np.asarray(shock_luminosity_cgs, dtype=float)
    vshell = _np.asarray(vshell_cgs, dtype=float)

    kT_kev = shock_temperature_kev(vshell, mu=mu, electron_temperature_fraction=electron_temperature_fraction)
    norm = str(normalization).lower()
    scale = 10.0 ** float(logepsx)

    if norm in ("emission_measure", "em", "freefree"):
        if rho_csm_cgs is None:
            raise ValueError("Emission-measure X-ray mode requires rho_csm_cgs")
        if swept_csm_mass_cgs is None:
            if radius_cgs is None:
                raise ValueError(
                    "Emission-measure X-ray mode requires radius_cgs or swept_csm_mass_cgs"
                )
            swept_csm_mass_cgs = cumulative_swept_csm_mass(radius_cgs, rho_csm_cgs)
        l_ff_bol = freefree_emission_measure_luminosity(
            swept_csm_mass_cgs=swept_csm_mass_cgs,
            rho_csm_cgs=rho_csm_cgs,
            temperature_kev=kT_kev,
            compression_factor=compression_factor,
            mu_e=mu_e,
            mu_i=mu_i,
            gaunt_factor=gaunt_factor,
            emission_measure_scale=scale,
        )
        if max_xray_efficiency is not None and float(max_xray_efficiency) >= 0.0:
            cap = float(max_xray_efficiency) * _np.maximum(shock_luminosity, 0.0)
            l_ff_bol = _np.minimum(l_ff_bol, cap)
    elif norm in ("shock_power", "shock", "power"):
        l_ff_bol = scale * _np.maximum(shock_luminosity, 0.0)
    else:
        raise ValueError("normalization must be 'emission_measure' or 'shock_power'")

    n_h_total = float(n_h_host) + float(n_h_mw)
    if n_h_csm is not None:
        n_h_total = n_h_total + _np.asarray(n_h_csm, dtype=float)

    output_format = str(output_format).lower()
    if output_format in ("spectral", "spectra", "dlde"):
        output_format = "spectral_luminosity"

    if output_format in ("spectral_luminosity", "flux_density"):
        if energy_kev is None:
            if frequency is None:
                raise ValueError("X-ray spectral outputs require energy_kev or frequency")
            energy_kev = _np.asarray(frequency, dtype=float) / _KEV_TO_HZ
        energy_src = _np.asarray(energy_kev, dtype=float) * (1.0 + redshift)
        attenuation = 1.0
        if absorb:
            attenuation = _np.exp(-photoelectric_cross_section_per_h(energy_src) * n_h_total)
        dlde = l_ff_bol / kT_kev * _np.exp(-energy_src / kT_kev) * attenuation
        dlde = _np.where(_np.isfinite(dlde) & (dlde > 0.0), dlde, 0.0)
        if output_format == "spectral_luminosity":
            return dlde
        flux_density_mjy = (
            dlde / (4.0 * _np.pi * luminosity_distance_cm ** 2)
            / _KEV_TO_HZ
            * 1.0e26
            * (1.0 + redshift)
        )
        return _np.where(_np.isfinite(flux_density_mjy) & (flux_density_mjy > 0.0), flux_density_mjy, 0.0)

    e1 = float(e_min_kev) * (1.0 + redshift)
    e2 = float(e_max_kev) * (1.0 + redshift)
    if not e2 > e1 > 0.0:
        raise ValueError("Require 0 < e_min_kev < e_max_kev")

    energy_grid = _np.geomspace(e1, e2, int(max(n_energy, 8)))
    shape = _np.exp(-energy_grid[None, :] / kT_kev[:, None])
    if absorb:
        tau = photoelectric_cross_section_per_h(energy_grid)[None, :] * n_h_total[:, None] if _np.ndim(n_h_total) else (
            photoelectric_cross_section_per_h(energy_grid)[None, :] * n_h_total
        )
        shape = shape * _np.exp(-tau)
    band_fraction = _TRAPEZOID(shape, energy_grid, axis=1) / kT_kev
    band_fraction = _np.clip(_np.where(_np.isfinite(band_fraction), band_fraction, 0.0), 0.0, 1.0)
    luminosity = l_ff_bol * band_fraction
    luminosity = _np.where(_np.isfinite(luminosity) & (luminosity > 0.0), luminosity, 0.0)

    if output_format in ("luminosity", "band_luminosity"):
        return luminosity
    if output_format in ("flux", "band_flux"):
        return luminosity / (4.0 * _np.pi * luminosity_distance_cm ** 2)
    raise ValueError(
        "Unknown X-ray output_format. Use 'luminosity', 'flux', "
        "'spectral_luminosity', or 'flux_density'."
    )
