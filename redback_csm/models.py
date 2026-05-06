"""
models.py — Public redback-compatible CSM model functions.

Each CSM physical scenario is exposed as four functions:

  {name}_bolometric          bolometric luminosity (erg/s), source-frame time (days)
  {name}                     multiband, observer-frame time, full output_format dispatch
  {name}_nickel_bolometric   CSM + radioactive nickel decay, bolometric
  {name}_nickel              CSM + nickel, multiband

All functions are registered in redback.model_library.all_models_dict automatically
when the package is installed, via the redback.model.modules entry point.

Citation: Sarin & Hirai (in prep); Sarin et al. 2024 (redback)
"""

import inspect as _inspect
import os as _os
from collections import namedtuple as _namedtuple

_os.environ.setdefault("MPLCONFIGDIR", "/tmp/redback_csm_mpl_cache")
_os.environ.setdefault("XDG_CACHE_HOME", "/tmp/redback_csm_xdg_cache")
_os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/redback_csm_numba_cache")

import numpy as _np
from scipy.interpolate import interp1d as _interp1d
import astropy.units as _uu
from astropy.cosmology import Planck18 as _cosmo

from redback.utils import citation_wrapper as _citation_wrapper
from redback.utils import calc_kcorrected_properties as _calc_kcorrected_properties
from redback.utils import lambda_to_nu as _lambda_to_nu
import redback.sed as _sed
import redback.photosphere as _photosphere
import redback.interaction_processes as _interaction_processes
from redback.transient_models.supernova_models import (
    _nickelcobalt_engine as _nickelcobalt_engine,
)

from redback_csm.core import (
    _call_csm,
    _call_csm_radio,
    _call_csm_xray,
    create_generic_csm_density as _create_generic_csm_density,
)

DAY = 86400.0    # seconds per day
_AU = 1.496e13   # cm per AU
_SOLAR_MASS = 1.989e33
_TRAPEZOID = getattr(_np, "trapezoid", _np.trapz)

__all__ = [
    "wind_exponential_bolometric",
    "wind_exponential",
    "wind_exponential_nickel_bolometric",
    "wind_exponential_nickel",
    "wind_bpl_bolometric",
    "wind_bpl",
    "wind_bpl_nickel_bolometric",
    "wind_bpl_nickel",
    "exponential_wind_bolometric",
    "exponential_wind",
    "exponential_wind_nickel_bolometric",
    "exponential_wind_nickel",
    "bpl_wind_bolometric",
    "bpl_wind",
    "bpl_wind_nickel_bolometric",
    "bpl_wind_nickel",
    "exponential_exponential_bolometric",
    "exponential_exponential",
    "exponential_exponential_nickel_bolometric",
    "exponential_exponential_nickel",
    "exponential_bpl_bolometric",
    "exponential_bpl",
    "exponential_bpl_nickel_bolometric",
    "exponential_bpl_nickel",
    "bpl_bpl_bolometric",
    "bpl_bpl",
    "bpl_bpl_nickel_bolometric",
    "bpl_bpl_nickel",
    "bpl_exponential_bolometric",
    "bpl_exponential",
    "bpl_exponential_nickel_bolometric",
    "bpl_exponential_nickel",
    "boxwind_exponential_bolometric",
    "boxwind_exponential",
    "boxwind_exponential_nickel_bolometric",
    "boxwind_exponential_nickel",
    "boxwind_bpl_bolometric",
    "boxwind_bpl",
    "boxwind_bpl_nickel_bolometric",
    "boxwind_bpl_nickel",
    "gausswind_exponential_bolometric",
    "gausswind_exponential",
    "gausswind_exponential_nickel_bolometric",
    "gausswind_exponential_nickel",
    "gausswind_bpl_bolometric",
    "gausswind_bpl",
    "gausswind_bpl_nickel_bolometric",
    "gausswind_bpl_nickel",
    "triple_powerlaw_wind_bpl_bolometric",
    "triple_powerlaw_wind_bpl",
    "triple_powerlaw_wind_bpl_nickel_bolometric",
    "triple_powerlaw_wind_bpl_nickel",
    "triple_powerlaw_wind_exponential_bolometric",
    "triple_powerlaw_wind_exponential",
    "triple_powerlaw_wind_exponential_nickel_bolometric",
    "triple_powerlaw_wind_exponential_nickel",
    "exponential_triple_powerlaw_wind_bolometric",
    "exponential_triple_powerlaw_wind",
    "exponential_triple_powerlaw_wind_nickel_bolometric",
    "exponential_triple_powerlaw_wind_nickel",
    "bpl_triple_powerlaw_wind_bolometric",
    "bpl_triple_powerlaw_wind",
    "bpl_triple_powerlaw_wind_nickel_bolometric",
    "bpl_triple_powerlaw_wind_nickel",
    "smooth_triple_powerlaw_wind_bpl_bolometric",
    "smooth_triple_powerlaw_wind_bpl",
    "smooth_triple_powerlaw_wind_bpl_nickel_bolometric",
    "smooth_triple_powerlaw_wind_bpl_nickel",
    "smooth_triple_powerlaw_wind_exponential_bolometric",
    "smooth_triple_powerlaw_wind_exponential",
    "smooth_triple_powerlaw_wind_exponential_nickel_bolometric",
    "smooth_triple_powerlaw_wind_exponential_nickel",
    "generic_csm_exponential_bolometric",
    "generic_csm_exponential",
    "generic_csm_exponential_nickel_bolometric",
    "generic_csm_exponential_nickel",
    "generic_csm_bpl_bolometric",
    "generic_csm_bpl",
    "generic_csm_bpl_nickel_bolometric",
    "generic_csm_bpl_nickel",
    "generic_powerlaw_csm_exponential_bolometric",
    "generic_powerlaw_csm_exponential",
    "generic_powerlaw_csm_bpl_bolometric",
    "generic_powerlaw_csm_bpl",
    "static_spline_csm_bpl_bolometric",
    "static_spline_csm_bpl",
    "generic_powerlaw_csm_exponential_radio",
    "generic_powerlaw_csm_bpl_radio",
    "generic_4shell_csm_bpl_bolometric",
    "generic_4shell_csm_bpl",
    "generic_4shell_csm_bpl_nickel_bolometric",
    "generic_4shell_csm_bpl_nickel",
    "generic_8shell_csm_bpl_bolometric",
    "generic_8shell_csm_bpl",
    "generic_8shell_csm_bpl_nickel_bolometric",
    "generic_8shell_csm_bpl_nickel",
    # Radio synchrotron models
    "wind_exponential_radio",
    "wind_bpl_radio",
    "exponential_wind_radio",
    "bpl_wind_radio",
    "gausswind_exponential_radio",
    "gausswind_bpl_radio",
    "boxwind_exponential_radio",
    "boxwind_bpl_radio",
    "triple_powerlaw_wind_bpl_radio",
    "triple_powerlaw_wind_exponential_radio",
    "exponential_triple_powerlaw_wind_radio",
    "bpl_triple_powerlaw_wind_radio",
    "smooth_triple_powerlaw_wind_bpl_radio",
    "smooth_triple_powerlaw_wind_exponential_radio",
    "exponential_exponential_radio",
    "exponential_bpl_radio",
    "bpl_bpl_radio",
    "bpl_exponential_radio",
    "generic_csm_exponential_radio",
    "generic_csm_bpl_radio",
    "generic_4shell_csm_bpl_radio",
    "generic_8shell_csm_bpl_radio",
    # Generic X-ray helper
    "csm_xray",
]

CITATION = "Sarin & Hirai (in prep); Sarin et al. 2024"

# ---------------------------------------------------------------------------
# Private implementation helpers
# ---------------------------------------------------------------------------


def _csm_bolometric_impl(time, csm_model, **kwargs):
    """Run the Fortran CSM model and return lbol (erg/s) on the requested time array (days, source frame)."""
    lc = _call_csm(csm_model, **kwargs)
    t_days = lc.time / DAY
    return _interp1d(t_days, lc.lbol, bounds_error=False, fill_value=0.0)(time)


def _csm_rph_temp_impl(time, csm_model, **kwargs):
    """Run Fortran model; return (lbol, rph, temperature) interpolated onto the requested time array."""
    lc = _call_csm(csm_model, **kwargs)
    t_days = lc.time / DAY
    lbol = _interp1d(t_days, lc.lbol, bounds_error=False, fill_value=0.0)(time)
    rph = _interp1d(t_days, lc.rph, bounds_error=False, fill_value=0.0)(time)
    temp = _interp1d(t_days, lc.temperature, bounds_error=False, fill_value=1e3)(time)
    return lbol, rph, temp


def _nickel_ejecta_mass_energy(kwargs):
    """Return the SN ejecta mass/energy used for nickel production."""
    if "mej_sn" in kwargs and "esn" in kwargs:
        return float(kwargs["mej_sn"]), float(kwargs["esn"])
    if "mexp_out" in kwargs and "eexp_out" in kwargs:
        return float(kwargs["mexp_out"]), float(kwargs["eexp_out"])
    if "mexp" in kwargs and "eexp" in kwargs:
        return float(kwargs["mexp"]), float(kwargs["eexp"])
    raise ValueError("Nickel CSM models require ejecta mass/energy parameters")


def _finite_wind_mass_from_grid(tgrid, mdot):
    """Integrate a finite mass-loss history in Msun/yr over years."""
    tgrid = _np.asarray(tgrid, dtype=float)
    mdot = _np.asarray(mdot, dtype=float)
    if tgrid.size < 2:
        return 0.0
    order = _np.argsort(tgrid)
    return max(float(_TRAPEZOID(_np.maximum(mdot[order], 0.0), tgrid[order])), 0.0)


def _generic_density_csm_mass(kwargs, n_shells):
    """Estimate CSM mass for generic density-profile constructors."""
    r_inner = float(kwargs.get("r_inner", 1e10))
    r_outer = kwargs.get("r_outer", None)
    if not (r_inner > 0.0):
        return 0.0

    base_density = float(kwargs.get("base_density", 0.0))
    base_index = float(kwargs.get("base_index", -2.0))
    shell_radii = []
    shell_widths = []
    shell_densities = []
    for shell_idx in range(1, n_shells + 1):
        density = float(kwargs.get(f"shell{shell_idx}_density", 0.0))
        if density <= 0.0:
            continue
        radius = float(kwargs.get(f"shell{shell_idx}_radius", 0.0))
        width = float(kwargs.get(f"shell{shell_idx}_width", 0.0))
        if radius <= 0.0 or width <= 0.0:
            continue
        shell_radii.append(radius)
        shell_widths.append(width)
        shell_densities.append(density)

    r_grid, _, rho = _create_generic_csm_density(
        r_inner=r_inner,
        r_outer=r_outer,
        n_points=int(kwargs.get("n_points", 1000)),
        base_density=base_density,
        base_index=base_index,
        n_shells=len(shell_radii),
        shell_radii=shell_radii if shell_radii else None,
        shell_widths=shell_widths if shell_widths else None,
        shell_densities=shell_densities if shell_densities else None,
        shell_profiles="gaussian",
        time_ref_days=float(kwargs.get("interval_sn", 10.0 * 365.25)),
    )

    mass_cgs = _TRAPEZOID(4.0 * _np.pi * r_grid**2 * _np.maximum(rho, 0.0), r_grid)
    return max(float(mass_cgs / _SOLAR_MASS), 0.0)


def _csm_mass_for_nickel_diffusion(csm_model, kwargs):
    """Return CSM mass to add to the nickel diffusion mass, in Msun."""
    if "mej_arnett" in kwargs:
        mej_ejecta, _ = _nickel_ejecta_mass_energy(kwargs)
        return max(float(kwargs["mej_arnett"]) - mej_ejecta, 0.0)

    if "m_csm" in kwargs:
        return max(float(kwargs["m_csm"]), 0.0)
    if "csm_mass" in kwargs:
        return max(float(kwargs["csm_mass"]), 0.0)

    if "mexp_out" in kwargs and "mexp" in kwargs:
        return max(float(kwargs["mexp"]), 0.0)

    if csm_model.startswith("boxwind_"):
        return max(float(kwargs.get("mdot_1", 0.0)), 0.0) * max(
            float(kwargs.get("t2", 0.0)) - float(kwargs.get("t1", 0.0)), 0.0
        )

    if csm_model.startswith("gausswind_"):
        t_peak = float(kwargs.get("t_peak", 0.0))
        t_width = float(kwargs.get("t_width", 0.0))
        if t_width <= 0.0:
            return 0.0
        tgrid = _np.linspace(t_peak - 4.0 * t_width, t_peak + 4.0 * t_width, 200)
        profile = _np.exp(-0.5 * ((tgrid - t_peak) / t_width) ** 2)
        mdot = float(kwargs.get("mdot_baseline", 0.0)) + (
            float(kwargs.get("mdot_peak", 0.0)) - float(kwargs.get("mdot_baseline", 0.0))
        ) * profile
        return _finite_wind_mass_from_grid(tgrid, mdot)

    if "triple_powerlaw_wind" in csm_model:
        t_break1 = float(kwargs.get("t_break1", 0.0))
        t_break2 = float(kwargs.get("t_break2", 0.0))
        if t_break1 <= 0.0 or t_break2 <= t_break1:
            return 0.0
        n_points = int(kwargs.get("n_points", 50))
        tgrid = _np.logspace(_np.log10(0.1), _np.log10(max(t_break2 * 2.0, 10.0)), n_points)
        mdot_0 = float(kwargs.get("mdot_0", 0.0))
        alpha1 = float(kwargs.get("alpha1", 0.0))
        alpha2 = float(kwargs.get("alpha2", 0.0))
        alpha3 = float(kwargs.get("alpha3", 0.0))
        mdot_break1 = mdot_0 * t_break1**alpha1
        mdot_break2 = mdot_break1 * (t_break2 / t_break1) ** alpha2
        if csm_model.startswith("smooth_triple_powerlaw_wind_"):
            smooth_factor = float(kwargs.get("smooth_factor", 0.2))
            smooth_width1 = smooth_factor * _np.log10(t_break1) if t_break1 > 1.0 else smooth_factor
            smooth_width2 = smooth_factor * _np.log10(t_break2) if t_break2 > 1.0 else smooth_factor
            transition1 = 0.5 * (1.0 + _np.tanh((_np.log10(tgrid) - _np.log10(t_break1)) / smooth_width1))
            transition2 = 0.5 * (1.0 + _np.tanh((_np.log10(tgrid) - _np.log10(t_break2)) / smooth_width2))
            mdot = (
                (1.0 - transition1) * mdot_0 * tgrid**alpha1
                + transition1 * (1.0 - transition2) * mdot_break1 * (tgrid / t_break1) ** alpha2
                + transition2 * mdot_break2 * (tgrid / t_break2) ** alpha3
            )
        else:
            mdot = _np.zeros_like(tgrid)
            mask1 = tgrid < t_break1
            mask2 = (tgrid >= t_break1) & (tgrid < t_break2)
            mask3 = tgrid >= t_break2
            mdot[mask1] = mdot_0 * tgrid[mask1] ** alpha1
            mdot[mask2] = mdot_break1 * (tgrid[mask2] / t_break1) ** alpha2
            mdot[mask3] = mdot_break2 * (tgrid[mask3] / t_break2) ** alpha3
        return _finite_wind_mass_from_grid(tgrid, mdot)

    if csm_model.startswith("generic_8shell_csm_"):
        return _generic_density_csm_mass(kwargs, 8)
    if csm_model.startswith("generic_4shell_csm_"):
        return _generic_density_csm_mass(kwargs, 4)
    if csm_model.startswith("generic_csm_"):
        return _generic_density_csm_mass(kwargs, 3)

    return 0.0


def _clean_nickel_diffusion_kwargs(kwargs):
    """Strip CSM and nickel bookkeeping keys before calling redback Diffusion."""
    cleaned = dict(kwargs)
    for key in (
        "f_nickel",
        "mej",
        "vej",
        "mexp",
        "eexp",
        "mexp_out",
        "eexp_out",
        "mej_sn",
        "esn",
        "mej_arnett",
    ):
        cleaned.pop(key, None)
    return cleaned


def _csm_nickel_bolometric_impl(time, csm_model, **kwargs):
    """Combined CSM shock + radioactive nickel bolometric luminosity."""
    import redback.constants as _rc

    time = _np.asarray(time, dtype=float)
    mej_ejecta, eexp = _nickel_ejecta_mass_energy(kwargs)
    mej_arnett = float(kwargs.get(
        "mej_arnett",
        mej_ejecta + _csm_mass_for_nickel_diffusion(csm_model, kwargs),
    ))
    vej = float(kwargs.get(
        "vej",
        _np.sqrt(2.0 * eexp * 1e51 / (mej_ejecta * _rc.solar_mass)) / 1e5,
    ))

    interaction_process = kwargs.get("interaction_process", _interaction_processes.Diffusion)
    if interaction_process is None:
        nickel_lbol = _nickelcobalt_engine(
            time=time, f_nickel=kwargs["f_nickel"], mej=mej_ejecta
        )
    else:
        dense_resolution = int(kwargs.get("dense_resolution", 1000))
        dense_times = _np.linspace(0.0, float(time[-1]) + 100.0, dense_resolution)
        dense_lbol = _nickelcobalt_engine(
            time=dense_times, f_nickel=kwargs["f_nickel"], mej=mej_ejecta
        )
        diffusion_kwargs = _clean_nickel_diffusion_kwargs(kwargs)
        diffusion_kwargs.pop("interaction_process", None)
        nickel_lbol = interaction_process(
            time=time,
            dense_times=dense_times,
            luminosity=dense_lbol,
            mej=mej_arnett,
            vej=vej,
            **diffusion_kwargs,
        ).new_luminosity

    return _csm_bolometric_impl(time, csm_model, **kwargs) + nickel_lbol


def _multiband_csm_flux_density(time, redshift, csm_model, csm_kwargs, dl):
    """flux_density output for CSM-only multiband (uses Fortran rph/temp directly)."""
    frequency = csm_kwargs.pop("frequency")
    frequency, time_src = _calc_kcorrected_properties(
        frequency=frequency, redshift=redshift, time=time
    )
    _, rph, temp = _csm_rph_temp_impl(time_src, csm_model, **csm_kwargs)
    sed_1 = _sed.Blackbody(
        temperature=temp, r_photosphere=rph, frequency=frequency, luminosity_distance=dl
    )
    return sed_1.flux_density.to(_uu.mJy).value * (1 + redshift)


def _multiband_nickel_flux_density(time, redshift, csm_model, csm_kwargs, dl):
    """flux_density output for CSM+nickel multiband (uses TemperatureFloor)."""
    frequency = csm_kwargs.pop("frequency")
    frequency, time_src = _calc_kcorrected_properties(
        frequency=frequency, redshift=redshift, time=time
    )
    lbol = _csm_nickel_bolometric_impl(time_src, csm_model, **csm_kwargs)
    photo = _photosphere.TemperatureFloor(time=time_src, luminosity=lbol, **csm_kwargs)
    sed_1 = _sed.Blackbody(
        temperature=photo.photosphere_temperature,
        r_photosphere=photo.r_photosphere,
        frequency=frequency,
        luminosity_distance=dl,
    )
    return sed_1.flux_density.to(_uu.mJy).value * (1 + redshift)


def _multiband_csm(time, redshift, csm_model, csm_kwargs, dl):
    """Spectra grid for CSM-only multiband (uses Fortran rph/temp)."""
    lambda_observer_frame = csm_kwargs.pop(
        "lambda_array", _np.geomspace(100, 60000, 100)
    )
    time_observer_frame = _np.geomspace(0.1, 3000, 300) * (1.0 + redshift)
    frequency_grid, time_src = _calc_kcorrected_properties(
        frequency=_lambda_to_nu(lambda_observer_frame),
        redshift=redshift,
        time=time_observer_frame,
    )
    _, rph, temp = _csm_rph_temp_impl(time_src, csm_model, **csm_kwargs)
    sed_1 = _sed.Blackbody(
        temperature=temp,
        r_photosphere=rph,
        frequency=frequency_grid[:, None],
        luminosity_distance=dl,
    )
    spectra = _sed.flux_density_to_spectrum(
        sed_1.flux_density.T, redshift, lambda_observer_frame
    )
    return time, time_observer_frame, lambda_observer_frame, spectra


def _multiband_nickel(time, redshift, csm_model, csm_kwargs, dl):
    """Spectra grid for CSM+nickel multiband (uses TemperatureFloor)."""
    lambda_observer_frame = csm_kwargs.pop(
        "lambda_array", _np.geomspace(100, 60000, 100)
    )
    time_observer_frame = _np.geomspace(0.1, 3000, 300) * (1.0 + redshift)
    frequency_grid, time_src = _calc_kcorrected_properties(
        frequency=_lambda_to_nu(lambda_observer_frame),
        redshift=redshift,
        time=time_observer_frame,
    )
    lbol = _csm_nickel_bolometric_impl(time_src, csm_model, **csm_kwargs)
    photo = _photosphere.TemperatureFloor(time=time_src, luminosity=lbol, **csm_kwargs)
    sed_1 = _sed.Blackbody(
        temperature=photo.photosphere_temperature,
        r_photosphere=photo.r_photosphere,
        frequency=frequency_grid[:, None],
        luminosity_distance=dl,
    )
    spectra = _sed.flux_density_to_spectrum(
        sed_1.flux_density.T, redshift, lambda_observer_frame
    )
    return time, time_observer_frame, lambda_observer_frame, spectra


def _multiband_output(
    time_obs, time_observer_frame, lambda_observer_frame, spectra, **kwargs
):
    """Dispatch spectra to the requested output format."""
    if kwargs.get("output_format") == "spectra":
        return _namedtuple("output", ["time", "lambdas", "spectra"])(
            time=time_observer_frame, lambdas=lambda_observer_frame, spectra=spectra
        )
    return _sed.get_correct_output_format_from_spectra(
        time=time_obs,
        time_eval=time_observer_frame,
        spectra=spectra,
        lambda_array=lambda_observer_frame,
        **kwargs
    )


@_citation_wrapper(CITATION)
def wind_exponential_bolometric(time, mdot, vwind, mexp, eexp, eff, **kwargs):
    """Bolometric light curve for wind exponential CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "wind_exponential",
        mdot=mdot,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def wind_exponential(time, redshift, mdot, vwind, mexp, eexp, eff, **kwargs):
    """Multiband light curve for wind exponential CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(mdot=mdot, vwind=vwind, mexp=mexp, eexp=eexp, eff=eff, **kwargs)
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "wind_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "wind_exponential", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def wind_exponential_nickel_bolometric(
    time, mdot, vwind, mexp, eexp, eff, f_nickel, **kwargs
):
    """Bolometric light curve for wind exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "wind_exponential",
        mdot=mdot,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def wind_exponential_nickel(
    time, redshift, mdot, vwind, mexp, eexp, eff, f_nickel, **kwargs
):
    """Multiband light curve for wind exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        mdot=mdot,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "wind_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "wind_exponential", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def wind_bpl_bolometric(time, mdot, vwind, delta, nn, mexp, eexp, eff, **kwargs):
    """Bolometric light curve for wind bpl CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "wind_bpl",
        mdot=mdot,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def wind_bpl(time, redshift, mdot, vwind, delta, nn, mexp, eexp, eff, **kwargs):
    """Multiband light curve for wind bpl CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        mdot=mdot,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(time, redshift, "wind_bpl", csm_kwargs, dl)
    return _multiband_output(
        *_multiband_csm(time, redshift, "wind_bpl", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def wind_bpl_nickel_bolometric(
    time, mdot, vwind, delta, nn, mexp, eexp, eff, f_nickel, **kwargs
):
    """Bolometric light curve for wind bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "wind_bpl",
        mdot=mdot,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def wind_bpl_nickel(
    time, redshift, mdot, vwind, delta, nn, mexp, eexp, eff, f_nickel, **kwargs
):
    """Multiband light curve for wind bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        mdot=mdot,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "wind_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "wind_bpl", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_wind_bolometric(time, mexp, eexp, mdot, vwind, eff, **kwargs):
    """Bolometric light curve for exponential wind CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "exponential_wind",
        mexp=mexp,
        eexp=eexp,
        mdot=mdot,
        vwind=vwind,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_wind(time, redshift, mexp, eexp, mdot, vwind, eff, **kwargs):
    """Multiband light curve for exponential wind CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(mexp=mexp, eexp=eexp, mdot=mdot, vwind=vwind, eff=eff, **kwargs)
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "exponential_wind", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "exponential_wind", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_wind_nickel_bolometric(
    time, mexp, eexp, mdot, vwind, eff, f_nickel, **kwargs
):
    """Bolometric light curve for exponential wind CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "exponential_wind",
        mexp=mexp,
        eexp=eexp,
        mdot=mdot,
        vwind=vwind,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_wind_nickel(
    time, redshift, mexp, eexp, mdot, vwind, eff, f_nickel, **kwargs
):
    """Multiband light curve for exponential wind CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        mexp=mexp,
        eexp=eexp,
        mdot=mdot,
        vwind=vwind,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "exponential_wind", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "exponential_wind", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_wind_bolometric(time, delta, nn, mexp, eexp, mdot, vwind, eff, **kwargs):
    """Bolometric light curve for bpl wind CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "bpl_wind",
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        mdot=mdot,
        vwind=vwind,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_wind(time, redshift, delta, nn, mexp, eexp, mdot, vwind, eff, **kwargs):
    """Multiband light curve for bpl wind CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        mdot=mdot,
        vwind=vwind,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(time, redshift, "bpl_wind", csm_kwargs, dl)
    return _multiband_output(
        *_multiband_csm(time, redshift, "bpl_wind", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_wind_nickel_bolometric(
    time, delta, nn, mexp, eexp, mdot, vwind, eff, f_nickel, **kwargs
):
    """Bolometric light curve for bpl wind CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "bpl_wind",
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        mdot=mdot,
        vwind=vwind,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_wind_nickel(
    time, redshift, delta, nn, mexp, eexp, mdot, vwind, eff, f_nickel, **kwargs
):
    """Multiband light curve for bpl wind CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        mdot=mdot,
        vwind=vwind,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "bpl_wind", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "bpl_wind", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_exponential_bolometric(
    time, mexp, eexp, mexp_out, eexp_out, interval, eff, **kwargs
):
    """Bolometric light curve for exponential exponential CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "exponential_exponential",
        mexp=mexp,
        eexp=eexp,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_exponential(
    time, redshift, mexp, eexp, mexp_out, eexp_out, interval, eff, **kwargs
):
    """Multiband light curve for exponential exponential CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        mexp=mexp,
        eexp=eexp,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "exponential_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "exponential_exponential", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_exponential_nickel_bolometric(
    time, mexp, eexp, mexp_out, eexp_out, interval, eff, f_nickel, **kwargs
):
    """Bolometric light curve for exponential exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "exponential_exponential",
        mexp=mexp,
        eexp=eexp,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_exponential_nickel(
    time,
    redshift,
    mexp,
    eexp,
    mexp_out,
    eexp_out,
    interval,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for exponential exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        mexp=mexp,
        eexp=eexp,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "exponential_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "exponential_exponential", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_bpl_bolometric(
    time, mexp, eexp, delta_out, nn_out, mexp_out, eexp_out, interval, eff, **kwargs
):
    """Bolometric light curve for exponential bpl CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "exponential_bpl",
        mexp=mexp,
        eexp=eexp,
        delta_out=delta_out,
        nn_out=nn_out,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_bpl(
    time,
    redshift,
    mexp,
    eexp,
    delta_out,
    nn_out,
    mexp_out,
    eexp_out,
    interval,
    eff,
    **kwargs
):
    """Multiband light curve for exponential bpl CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        mexp=mexp,
        eexp=eexp,
        delta_out=delta_out,
        nn_out=nn_out,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "exponential_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "exponential_bpl", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_bpl_nickel_bolometric(
    time,
    mexp,
    eexp,
    delta_out,
    nn_out,
    mexp_out,
    eexp_out,
    interval,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for exponential bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "exponential_bpl",
        mexp=mexp,
        eexp=eexp,
        delta_out=delta_out,
        nn_out=nn_out,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_bpl_nickel(
    time,
    redshift,
    mexp,
    eexp,
    delta_out,
    nn_out,
    mexp_out,
    eexp_out,
    interval,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for exponential bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        mexp=mexp,
        eexp=eexp,
        delta_out=delta_out,
        nn_out=nn_out,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "exponential_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "exponential_bpl", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_bpl_bolometric(
    time,
    delta,
    nn,
    mexp,
    eexp,
    delta_out,
    nn_out,
    mexp_out,
    eexp_out,
    interval,
    eff,
    **kwargs
):
    """Bolometric light curve for bpl bpl CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "bpl_bpl",
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        delta_out=delta_out,
        nn_out=nn_out,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_bpl(
    time,
    redshift,
    delta,
    nn,
    mexp,
    eexp,
    delta_out,
    nn_out,
    mexp_out,
    eexp_out,
    interval,
    eff,
    **kwargs
):
    """Multiband light curve for bpl bpl CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        delta_out=delta_out,
        nn_out=nn_out,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(time, redshift, "bpl_bpl", csm_kwargs, dl)
    return _multiband_output(
        *_multiband_csm(time, redshift, "bpl_bpl", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_bpl_nickel_bolometric(
    time,
    delta,
    nn,
    mexp,
    eexp,
    delta_out,
    nn_out,
    mexp_out,
    eexp_out,
    interval,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for bpl bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "bpl_bpl",
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        delta_out=delta_out,
        nn_out=nn_out,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_bpl_nickel(
    time,
    redshift,
    delta,
    nn,
    mexp,
    eexp,
    delta_out,
    nn_out,
    mexp_out,
    eexp_out,
    interval,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for bpl bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        delta_out=delta_out,
        nn_out=nn_out,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(time, redshift, "bpl_bpl", csm_kwargs, dl)
    return _multiband_output(
        *_multiband_nickel(time, redshift, "bpl_bpl", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_exponential_bolometric(
    time, delta, nn, mexp, eexp, mexp_out, eexp_out, interval, eff, **kwargs
):
    """Bolometric light curve for bpl exponential CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "bpl_exponential",
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_exponential(
    time, redshift, delta, nn, mexp, eexp, mexp_out, eexp_out, interval, eff, **kwargs
):
    """Multiband light curve for bpl exponential CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "bpl_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "bpl_exponential", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_exponential_nickel_bolometric(
    time,
    delta,
    nn,
    mexp,
    eexp,
    mexp_out,
    eexp_out,
    interval,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for bpl exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "bpl_exponential",
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_exponential_nickel(
    time,
    redshift,
    delta,
    nn,
    mexp,
    eexp,
    mexp_out,
    eexp_out,
    interval,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for bpl exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        mexp_out=mexp_out,
        eexp_out=eexp_out,
        interval=interval,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "bpl_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "bpl_exponential", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def boxwind_exponential_bolometric(
    time, t1, t2, mdot_0, mdot_1, mdot_2, vwind, mexp, eexp, eff, **kwargs
):
    """Bolometric light curve for boxwind exponential CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "boxwind_exponential",
        t1=t1,
        t2=t2,
        mdot_0=mdot_0,
        mdot_1=mdot_1,
        mdot_2=mdot_2,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def boxwind_exponential(
    time, redshift, t1, t2, mdot_0, mdot_1, mdot_2, vwind, mexp, eexp, eff, **kwargs
):
    """Multiband light curve for boxwind exponential CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t1=t1,
        t2=t2,
        mdot_0=mdot_0,
        mdot_1=mdot_1,
        mdot_2=mdot_2,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "boxwind_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "boxwind_exponential", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def boxwind_exponential_nickel_bolometric(
    time,
    t1,
    t2,
    mdot_0,
    mdot_1,
    mdot_2,
    vwind,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for boxwind exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "boxwind_exponential",
        t1=t1,
        t2=t2,
        mdot_0=mdot_0,
        mdot_1=mdot_1,
        mdot_2=mdot_2,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def boxwind_exponential_nickel(
    time,
    redshift,
    t1,
    t2,
    mdot_0,
    mdot_1,
    mdot_2,
    vwind,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for boxwind exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t1=t1,
        t2=t2,
        mdot_0=mdot_0,
        mdot_1=mdot_1,
        mdot_2=mdot_2,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "boxwind_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "boxwind_exponential", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def boxwind_bpl_bolometric(
    time, t1, t2, mdot_0, mdot_1, mdot_2, vwind, delta, nn, mexp, eexp, eff, **kwargs
):
    """Bolometric light curve for boxwind bpl CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "boxwind_bpl",
        t1=t1,
        t2=t2,
        mdot_0=mdot_0,
        mdot_1=mdot_1,
        mdot_2=mdot_2,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def boxwind_bpl(
    time,
    redshift,
    t1,
    t2,
    mdot_0,
    mdot_1,
    mdot_2,
    vwind,
    delta,
    nn,
    mexp,
    eexp,
    eff,
    **kwargs
):
    """Multiband light curve for boxwind bpl CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t1=t1,
        t2=t2,
        mdot_0=mdot_0,
        mdot_1=mdot_1,
        mdot_2=mdot_2,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "boxwind_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "boxwind_bpl", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def boxwind_bpl_nickel_bolometric(
    time,
    t1,
    t2,
    mdot_0,
    mdot_1,
    mdot_2,
    vwind,
    delta,
    nn,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for boxwind bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "boxwind_bpl",
        t1=t1,
        t2=t2,
        mdot_0=mdot_0,
        mdot_1=mdot_1,
        mdot_2=mdot_2,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def boxwind_bpl_nickel(
    time,
    redshift,
    t1,
    t2,
    mdot_0,
    mdot_1,
    mdot_2,
    vwind,
    delta,
    nn,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for boxwind bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t1=t1,
        t2=t2,
        mdot_0=mdot_0,
        mdot_1=mdot_1,
        mdot_2=mdot_2,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "boxwind_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "boxwind_bpl", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def gausswind_exponential_bolometric(
    time, t_peak, t_width, mdot_baseline, mdot_peak, vwind, mexp, eexp, eff, **kwargs
):
    """Bolometric light curve for gausswind exponential CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "gausswind_exponential",
        t_peak=t_peak,
        t_width=t_width,
        mdot_baseline=mdot_baseline,
        mdot_peak=mdot_peak,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def gausswind_exponential(
    time,
    redshift,
    t_peak,
    t_width,
    mdot_baseline,
    mdot_peak,
    vwind,
    mexp,
    eexp,
    eff,
    **kwargs
):
    """Multiband light curve for gausswind exponential CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t_peak=t_peak,
        t_width=t_width,
        mdot_baseline=mdot_baseline,
        mdot_peak=mdot_peak,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "gausswind_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "gausswind_exponential", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def gausswind_exponential_nickel_bolometric(
    time,
    t_peak,
    t_width,
    mdot_baseline,
    mdot_peak,
    vwind,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for gausswind exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "gausswind_exponential",
        t_peak=t_peak,
        t_width=t_width,
        mdot_baseline=mdot_baseline,
        mdot_peak=mdot_peak,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def gausswind_exponential_nickel(
    time,
    redshift,
    t_peak,
    t_width,
    mdot_baseline,
    mdot_peak,
    vwind,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for gausswind exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t_peak=t_peak,
        t_width=t_width,
        mdot_baseline=mdot_baseline,
        mdot_peak=mdot_peak,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "gausswind_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "gausswind_exponential", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def gausswind_bpl_bolometric(
    time,
    t_peak,
    t_width,
    mdot_baseline,
    mdot_peak,
    vwind,
    delta,
    nn,
    mexp,
    eexp,
    eff,
    **kwargs
):
    """Bolometric light curve for gausswind bpl CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "gausswind_bpl",
        t_peak=t_peak,
        t_width=t_width,
        mdot_baseline=mdot_baseline,
        mdot_peak=mdot_peak,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def gausswind_bpl(
    time,
    redshift,
    t_peak,
    t_width,
    mdot_baseline,
    mdot_peak,
    vwind,
    delta,
    nn,
    mexp,
    eexp,
    eff,
    **kwargs
):
    """Multiband light curve for gausswind bpl CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t_peak=t_peak,
        t_width=t_width,
        mdot_baseline=mdot_baseline,
        mdot_peak=mdot_peak,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "gausswind_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "gausswind_bpl", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def gausswind_bpl_nickel_bolometric(
    time,
    t_peak,
    t_width,
    mdot_baseline,
    mdot_peak,
    vwind,
    delta,
    nn,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for gausswind bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "gausswind_bpl",
        t_peak=t_peak,
        t_width=t_width,
        mdot_baseline=mdot_baseline,
        mdot_peak=mdot_peak,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def gausswind_bpl_nickel(
    time,
    redshift,
    t_peak,
    t_width,
    mdot_baseline,
    mdot_peak,
    vwind,
    delta,
    nn,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for gausswind bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t_peak=t_peak,
        t_width=t_width,
        mdot_baseline=mdot_baseline,
        mdot_peak=mdot_peak,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "gausswind_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "gausswind_bpl", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def triple_powerlaw_wind_bpl_bolometric(
    time,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    delta,
    nn,
    mexp,
    eexp,
    eff,
    **kwargs
):
    """Bolometric light curve for triple powerlaw wind bpl CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "triple_powerlaw_wind_bpl",
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def triple_powerlaw_wind_bpl(
    time,
    redshift,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    delta,
    nn,
    mexp,
    eexp,
    eff,
    **kwargs
):
    """Multiband light curve for triple powerlaw wind bpl CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "triple_powerlaw_wind_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "triple_powerlaw_wind_bpl", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def triple_powerlaw_wind_bpl_nickel_bolometric(
    time,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    delta,
    nn,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for triple powerlaw wind bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "triple_powerlaw_wind_bpl",
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def triple_powerlaw_wind_bpl_nickel(
    time,
    redshift,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    delta,
    nn,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for triple powerlaw wind bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "triple_powerlaw_wind_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "triple_powerlaw_wind_bpl", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def triple_powerlaw_wind_exponential_bolometric(
    time,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    mexp,
    eexp,
    eff,
    **kwargs
):
    """Bolometric light curve for triple powerlaw wind exponential CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "triple_powerlaw_wind_exponential",
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def triple_powerlaw_wind_exponential(
    time,
    redshift,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    mexp,
    eexp,
    eff,
    **kwargs
):
    """Multiband light curve for triple powerlaw wind exponential CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "triple_powerlaw_wind_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(
            time, redshift, "triple_powerlaw_wind_exponential", csm_kwargs, dl
        ),
        **kwargs
    )


@_citation_wrapper(CITATION)
def triple_powerlaw_wind_exponential_nickel_bolometric(
    time,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for triple powerlaw wind exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "triple_powerlaw_wind_exponential",
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def triple_powerlaw_wind_exponential_nickel(
    time,
    redshift,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for triple powerlaw wind exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "triple_powerlaw_wind_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(
            time, redshift, "triple_powerlaw_wind_exponential", csm_kwargs, dl
        ),
        **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_triple_powerlaw_wind_bolometric(
    time,
    mexp,
    eexp,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    eff,
    **kwargs
):
    """Bolometric light curve for exponential triple powerlaw wind CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "exponential_triple_powerlaw_wind",
        mexp=mexp,
        eexp=eexp,
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_triple_powerlaw_wind(
    time,
    redshift,
    mexp,
    eexp,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    eff,
    **kwargs
):
    """Multiband light curve for exponential triple powerlaw wind CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        mexp=mexp,
        eexp=eexp,
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "exponential_triple_powerlaw_wind", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(
            time, redshift, "exponential_triple_powerlaw_wind", csm_kwargs, dl
        ),
        **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_triple_powerlaw_wind_nickel_bolometric(
    time,
    mexp,
    eexp,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for exponential triple powerlaw wind CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "exponential_triple_powerlaw_wind",
        mexp=mexp,
        eexp=eexp,
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_triple_powerlaw_wind_nickel(
    time,
    redshift,
    mexp,
    eexp,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for exponential triple powerlaw wind CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        mexp=mexp,
        eexp=eexp,
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "exponential_triple_powerlaw_wind", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(
            time, redshift, "exponential_triple_powerlaw_wind", csm_kwargs, dl
        ),
        **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_triple_powerlaw_wind_bolometric(
    time,
    delta,
    nn,
    mexp,
    eexp,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    eff,
    **kwargs
):
    """Bolometric light curve for bpl triple powerlaw wind CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "bpl_triple_powerlaw_wind",
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_triple_powerlaw_wind(
    time,
    redshift,
    delta,
    nn,
    mexp,
    eexp,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    eff,
    **kwargs
):
    """Multiband light curve for bpl triple powerlaw wind CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "bpl_triple_powerlaw_wind", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "bpl_triple_powerlaw_wind", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_triple_powerlaw_wind_nickel_bolometric(
    time,
    delta,
    nn,
    mexp,
    eexp,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for bpl triple powerlaw wind CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "bpl_triple_powerlaw_wind",
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_triple_powerlaw_wind_nickel(
    time,
    redshift,
    delta,
    nn,
    mexp,
    eexp,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for bpl triple powerlaw wind CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "bpl_triple_powerlaw_wind", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "bpl_triple_powerlaw_wind", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def smooth_triple_powerlaw_wind_bpl_bolometric(
    time,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    delta,
    nn,
    mexp,
    eexp,
    eff,
    **kwargs
):
    """Bolometric light curve for smooth triple powerlaw wind bpl CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "smooth_triple_powerlaw_wind_bpl",
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def smooth_triple_powerlaw_wind_bpl(
    time,
    redshift,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    delta,
    nn,
    mexp,
    eexp,
    eff,
    **kwargs
):
    """Multiband light curve for smooth triple powerlaw wind bpl CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "smooth_triple_powerlaw_wind_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(
            time, redshift, "smooth_triple_powerlaw_wind_bpl", csm_kwargs, dl
        ),
        **kwargs
    )


@_citation_wrapper(CITATION)
def smooth_triple_powerlaw_wind_bpl_nickel_bolometric(
    time,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    delta,
    nn,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for smooth triple powerlaw wind bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "smooth_triple_powerlaw_wind_bpl",
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def smooth_triple_powerlaw_wind_bpl_nickel(
    time,
    redshift,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    delta,
    nn,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for smooth triple powerlaw wind bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "smooth_triple_powerlaw_wind_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(
            time, redshift, "smooth_triple_powerlaw_wind_bpl", csm_kwargs, dl
        ),
        **kwargs
    )


@_citation_wrapper(CITATION)
def smooth_triple_powerlaw_wind_exponential_bolometric(
    time,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    mexp,
    eexp,
    eff,
    **kwargs
):
    """Bolometric light curve for smooth triple powerlaw wind exponential CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "smooth_triple_powerlaw_wind_exponential",
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def smooth_triple_powerlaw_wind_exponential(
    time,
    redshift,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    mexp,
    eexp,
    eff,
    **kwargs
):
    """Multiband light curve for smooth triple powerlaw wind exponential CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "smooth_triple_powerlaw_wind_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(
            time, redshift, "smooth_triple_powerlaw_wind_exponential", csm_kwargs, dl
        ),
        **kwargs
    )


@_citation_wrapper(CITATION)
def smooth_triple_powerlaw_wind_exponential_nickel_bolometric(
    time,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for smooth triple powerlaw wind exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "smooth_triple_powerlaw_wind_exponential",
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def smooth_triple_powerlaw_wind_exponential_nickel(
    time,
    redshift,
    t_break1,
    t_break2,
    mdot_0,
    alpha1,
    alpha2,
    alpha3,
    vwind,
    mexp,
    eexp,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for smooth triple powerlaw wind exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        t_break1=t_break1,
        t_break2=t_break2,
        mdot_0=mdot_0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        vwind=vwind,
        mexp=mexp,
        eexp=eexp,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "smooth_triple_powerlaw_wind_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(
            time, redshift, "smooth_triple_powerlaw_wind_exponential", csm_kwargs, dl
        ),
        **kwargs
    )


def _generic_powerlaw_csm(eta, r_inner, r_outer, m_csm):
    """Return normalized finite-shell power-law CSM kwargs for the generic CSM backend."""
    if not (r_outer > r_inner > 0):
        raise ValueError("Require r_outer > r_inner > 0")
    m_csm_cgs = m_csm * _SOLAR_MASS
    if abs(eta + 3.0) < 1e-12:
        mass_factor = 4.0 * _np.pi * r_inner**3 * _np.log(r_outer / r_inner)
    else:
        mass_factor = (
            4.0
            * _np.pi
            * r_inner ** (-eta)
            * (r_outer ** (eta + 3.0) - r_inner ** (eta + 3.0))
            / (eta + 3.0)
        )
    rho_in = m_csm_cgs / mass_factor

    return dict(
        base_density=rho_in,
        base_index=eta,
        shell1_radius=0.0,
        shell1_width=0.0,
        shell1_density=0.0,
        shell2_radius=0.0,
        shell2_width=0.0,
        shell2_density=0.0,
        shell3_radius=0.0,
        shell3_width=0.0,
        shell3_density=0.0,
        r_inner=r_inner,
        r_outer=r_outer,
    )


@_citation_wrapper(CITATION)
def generic_powerlaw_csm_exponential_bolometric(
    time,
    eta,
    r_inner,
    r_outer,
    mej_sn,
    esn,
    eff,
    m_csm,
    **kwargs
):
    """Bolometric light curve for a static finite power-law CSM shell plus exponential ejecta."""
    return _csm_bolometric_impl(
        time,
        "static_powerlaw_csm_exponential",
        eta=eta,
        r_inner=r_inner,
        r_outer=r_outer,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        m_csm=m_csm,
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_powerlaw_csm_exponential(
    time,
    redshift,
    eta,
    r_inner,
    r_outer,
    mej_sn,
    esn,
    eff,
    m_csm,
    **kwargs
):
    """Multiband light curve for a static finite power-law CSM shell plus exponential ejecta."""
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        eta=eta,
        r_inner=r_inner,
        r_outer=r_outer,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        m_csm=m_csm,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "static_powerlaw_csm_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "static_powerlaw_csm_exponential", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_powerlaw_csm_bpl_bolometric(
    time,
    eta,
    r_inner,
    r_outer,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    m_csm,
    **kwargs
):
    """Bolometric light curve for a static finite power-law CSM shell plus BPL ejecta."""
    return _csm_bolometric_impl(
        time,
        "static_powerlaw_csm_bpl",
        eta=eta,
        r_inner=r_inner,
        r_outer=r_outer,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        m_csm=m_csm,
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_powerlaw_csm_bpl(
    time,
    redshift,
    eta,
    r_inner,
    r_outer,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    m_csm,
    **kwargs
):
    """Multiband light curve for a static finite power-law CSM shell plus BPL ejecta."""
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        eta=eta,
        r_inner=r_inner,
        r_outer=r_outer,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        m_csm=m_csm,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "static_powerlaw_csm_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "static_powerlaw_csm_bpl", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def static_spline_csm_bpl_bolometric(
    time,
    log_r_inner,
    log_r_outer,
    log_rho_0,
    log_rho_1,
    log_rho_2,
    log_rho_3,
    log_rho_4,
    log_rho_5,
    log_rho_6,
    log_rho_7,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    **kwargs
):
    """Bolometric light curve for a finite static CSM snapshot parameterised by density nodes."""
    return _csm_bolometric_impl(
        time,
        "static_spline_csm_bpl",
        log_r_inner=log_r_inner,
        log_r_outer=log_r_outer,
        log_rho_0=log_rho_0,
        log_rho_1=log_rho_1,
        log_rho_2=log_rho_2,
        log_rho_3=log_rho_3,
        log_rho_4=log_rho_4,
        log_rho_5=log_rho_5,
        log_rho_6=log_rho_6,
        log_rho_7=log_rho_7,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def static_spline_csm_bpl(
    time,
    redshift,
    log_r_inner,
    log_r_outer,
    log_rho_0,
    log_rho_1,
    log_rho_2,
    log_rho_3,
    log_rho_4,
    log_rho_5,
    log_rho_6,
    log_rho_7,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    **kwargs
):
    """Multiband light curve for a finite static CSM snapshot parameterised by density nodes."""
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        log_r_inner=log_r_inner,
        log_r_outer=log_r_outer,
        log_rho_0=log_rho_0,
        log_rho_1=log_rho_1,
        log_rho_2=log_rho_2,
        log_rho_3=log_rho_3,
        log_rho_4=log_rho_4,
        log_rho_5=log_rho_5,
        log_rho_6=log_rho_6,
        log_rho_7=log_rho_7,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "static_spline_csm_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "static_spline_csm_bpl", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_powerlaw_csm_exponential_radio(
    time, redshift, eta, r_inner, r_outer, mej_sn, esn, eff, m_csm,
    logepsb, logepse, p, **kwargs
):
    """Radio flux density (mJy) for a static finite power-law CSM shell plus exponential ejecta."""
    return _csm_radio_impl(
        time, redshift, "static_powerlaw_csm_exponential",
        dict(
            eta=eta, r_inner=r_inner, r_outer=r_outer,
            mej_sn=mej_sn, esn=esn, eff=eff, m_csm=m_csm,
            logepsb=logepsb, logepse=logepse, p=p,
            **kwargs
        ),
    )


@_citation_wrapper(CITATION)
def generic_powerlaw_csm_bpl_radio(
    time, redshift, eta, r_inner, r_outer, delta_sn, nn_sn, mej_sn, esn, eff, m_csm,
    logepsb, logepse, p, **kwargs
):
    """Radio flux density (mJy) for a static finite power-law CSM shell plus BPL ejecta."""
    return _csm_radio_impl(
        time, redshift, "static_powerlaw_csm_bpl",
        dict(
            eta=eta, r_inner=r_inner, r_outer=r_outer,
            delta_sn=delta_sn, nn_sn=nn_sn,
            mej_sn=mej_sn, esn=esn, eff=eff, m_csm=m_csm,
            logepsb=logepsb, logepse=logepse, p=p,
            **kwargs
        ),
    )


@_citation_wrapper(CITATION)
def generic_csm_exponential_bolometric(
    time,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    interval_sn,
    mej_sn,
    esn,
    eff,
    **kwargs
):
    """Bolometric light curve for generic csm exponential CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "generic_csm_exponential",
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        interval_sn=interval_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_csm_exponential(
    time,
    redshift,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    interval_sn,
    mej_sn,
    esn,
    eff,
    **kwargs
):
    """Multiband light curve for generic csm exponential CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        interval_sn=interval_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "generic_csm_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "generic_csm_exponential", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_csm_exponential_nickel_bolometric(
    time,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    interval_sn,
    mej_sn,
    esn,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for generic csm exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "generic_csm_exponential",
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        interval_sn=interval_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_csm_exponential_nickel(
    time,
    redshift,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    interval_sn,
    mej_sn,
    esn,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for generic csm exponential CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        interval_sn=interval_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "generic_csm_exponential", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "generic_csm_exponential", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_csm_bpl_bolometric(
    time,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    interval_sn,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    **kwargs
):
    """Bolometric light curve for generic csm bpl CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "generic_csm_bpl",
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_csm_bpl(
    time,
    redshift,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    interval_sn,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    **kwargs
):
    """Multiband light curve for generic csm bpl CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "generic_csm_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "generic_csm_bpl", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def generic_csm_bpl_nickel_bolometric(
    time,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    interval_sn,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for generic csm bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "generic_csm_bpl",
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_csm_bpl_nickel(
    time,
    redshift,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    interval_sn,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for generic csm bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "generic_csm_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "generic_csm_bpl", csm_kwargs, dl), **kwargs
    )


@_citation_wrapper(CITATION)
def generic_4shell_csm_bpl_bolometric(
    time,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    shell4_radius,
    shell4_width,
    shell4_density,
    interval_sn,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    **kwargs
):
    """Bolometric light curve for generic 4shell csm bpl CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "generic_4shell_csm_bpl",
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius * _AU,
        shell4_width=shell4_width * _AU,
        shell4_density=shell4_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_4shell_csm_bpl(
    time,
    redshift,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    shell4_radius,
    shell4_width,
    shell4_density,
    interval_sn,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    **kwargs
):
    """Multiband light curve for generic 4shell csm bpl CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius * _AU,
        shell4_width=shell4_width * _AU,
        shell4_density=shell4_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "generic_4shell_csm_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "generic_4shell_csm_bpl", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_4shell_csm_bpl_nickel_bolometric(
    time,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    shell4_radius,
    shell4_width,
    shell4_density,
    interval_sn,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for generic 4shell csm bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "generic_4shell_csm_bpl",
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius * _AU,
        shell4_width=shell4_width * _AU,
        shell4_density=shell4_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_4shell_csm_bpl_nickel(
    time,
    redshift,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    shell4_radius,
    shell4_width,
    shell4_density,
    interval_sn,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for generic 4shell csm bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius * _AU,
        shell4_width=shell4_width * _AU,
        shell4_density=shell4_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "generic_4shell_csm_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "generic_4shell_csm_bpl", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_8shell_csm_bpl_bolometric(
    time,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    shell4_radius,
    shell4_width,
    shell4_density,
    shell5_radius,
    shell5_width,
    shell5_density,
    shell6_radius,
    shell6_width,
    shell6_density,
    shell7_radius,
    shell7_width,
    shell7_density,
    shell8_radius,
    shell8_width,
    shell8_density,
    interval_sn,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    **kwargs
):
    """Bolometric light curve for generic 8shell csm bpl CSM interaction.

    :param time: time in days (source frame)
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: bolometric luminosity array in erg/s
    """
    return _csm_bolometric_impl(
        time,
        "generic_8shell_csm_bpl",
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius * _AU,
        shell4_width=shell4_width * _AU,
        shell4_density=shell4_density,
        shell5_radius=shell5_radius * _AU,
        shell5_width=shell5_width * _AU,
        shell5_density=shell5_density,
        shell6_radius=shell6_radius * _AU,
        shell6_width=shell6_width * _AU,
        shell6_density=shell6_density,
        shell7_radius=shell7_radius * _AU,
        shell7_width=shell7_width * _AU,
        shell7_density=shell7_density,
        shell8_radius=shell8_radius * _AU,
        shell8_width=shell8_width * _AU,
        shell8_density=shell8_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_8shell_csm_bpl(
    time,
    redshift,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    shell4_radius,
    shell4_width,
    shell4_density,
    shell5_radius,
    shell5_width,
    shell5_density,
    shell6_radius,
    shell6_width,
    shell6_density,
    shell7_radius,
    shell7_width,
    shell7_density,
    shell8_radius,
    shell8_width,
    shell8_density,
    interval_sn,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    **kwargs
):
    """Multiband light curve for generic 8shell csm bpl CSM interaction.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :param kappa: (optional) opacity in cm2/g — enables photon diffusion
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius * _AU,
        shell4_width=shell4_width * _AU,
        shell4_density=shell4_density,
        shell5_radius=shell5_radius * _AU,
        shell5_width=shell5_width * _AU,
        shell5_density=shell5_density,
        shell6_radius=shell6_radius * _AU,
        shell6_width=shell6_width * _AU,
        shell6_density=shell6_density,
        shell7_radius=shell7_radius * _AU,
        shell7_width=shell7_width * _AU,
        shell7_density=shell7_density,
        shell8_radius=shell8_radius * _AU,
        shell8_width=shell8_width * _AU,
        shell8_density=shell8_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_csm_flux_density(
            time, redshift, "generic_8shell_csm_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_csm(time, redshift, "generic_8shell_csm_bpl", csm_kwargs, dl),
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_8shell_csm_bpl_nickel_bolometric(
    time,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    shell4_radius,
    shell4_width,
    shell4_density,
    shell5_radius,
    shell5_width,
    shell5_density,
    shell6_radius,
    shell6_width,
    shell6_density,
    shell7_radius,
    shell7_width,
    shell7_density,
    shell8_radius,
    shell8_width,
    shell8_density,
    interval_sn,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    f_nickel,
    **kwargs
):
    """Bolometric light curve for generic 8shell csm bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (source frame)
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required for nickel diffusion)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :return: bolometric luminosity array in erg/s
    """
    return _csm_nickel_bolometric_impl(
        time,
        "generic_8shell_csm_bpl",
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius * _AU,
        shell4_width=shell4_width * _AU,
        shell4_density=shell4_density,
        shell5_radius=shell5_radius * _AU,
        shell5_width=shell5_width * _AU,
        shell5_density=shell5_density,
        shell6_radius=shell6_radius * _AU,
        shell6_width=shell6_width * _AU,
        shell6_density=shell6_density,
        shell7_radius=shell7_radius * _AU,
        shell7_width=shell7_width * _AU,
        shell7_density=shell7_density,
        shell8_radius=shell8_radius * _AU,
        shell8_width=shell8_width * _AU,
        shell8_density=shell8_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )


@_citation_wrapper(CITATION)
def generic_8shell_csm_bpl_nickel(
    time,
    redshift,
    base_density,
    base_index,
    shell1_radius,
    shell1_width,
    shell1_density,
    shell2_radius,
    shell2_width,
    shell2_density,
    shell3_radius,
    shell3_width,
    shell3_density,
    shell4_radius,
    shell4_width,
    shell4_density,
    shell5_radius,
    shell5_width,
    shell5_density,
    shell6_radius,
    shell6_width,
    shell6_density,
    shell7_radius,
    shell7_width,
    shell7_density,
    shell8_radius,
    shell8_width,
    shell8_density,
    interval_sn,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    f_nickel,
    **kwargs
):
    """Multiband light curve for generic 8shell csm bpl CSM interaction with radioactive nickel/cobalt decay.

    :param time: time in days (observer frame)
    :param redshift: source redshift
    :param f_nickel: nickel mass fraction
    :param mej: ejecta mass in M_sun
    :param kappa: opacity in cm2/g (required)
    :param kappa_gamma: gamma-ray opacity in cm2/g
    :param temperature_floor: minimum photosphere temperature in K
    :param output_format: flux_density, magnitude, flux, or spectra
    :param frequency: required if output_format=flux_density
    :param bands: required if output_format=magnitude or flux
    :return: set by output_format
    """
    dl = kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
    csm_kwargs = dict(
        base_density=base_density,
        base_index=base_index,
        shell1_radius=shell1_radius * _AU,
        shell1_width=shell1_width * _AU,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius * _AU,
        shell2_width=shell2_width * _AU,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius * _AU,
        shell3_width=shell3_width * _AU,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius * _AU,
        shell4_width=shell4_width * _AU,
        shell4_density=shell4_density,
        shell5_radius=shell5_radius * _AU,
        shell5_width=shell5_width * _AU,
        shell5_density=shell5_density,
        shell6_radius=shell6_radius * _AU,
        shell6_width=shell6_width * _AU,
        shell6_density=shell6_density,
        shell7_radius=shell7_radius * _AU,
        shell7_width=shell7_width * _AU,
        shell7_density=shell7_density,
        shell8_radius=shell8_radius * _AU,
        shell8_width=shell8_width * _AU,
        shell8_density=shell8_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        **kwargs
    )
    if kwargs.get("output_format") == "flux_density":
        return _multiband_nickel_flux_density(
            time, redshift, "generic_8shell_csm_bpl", csm_kwargs, dl
        )
    return _multiband_output(
        *_multiband_nickel(time, redshift, "generic_8shell_csm_bpl", csm_kwargs, dl),
        **kwargs
    )


# ===========================================================================
# Radio synchrotron models — one per base CSM model
# ===========================================================================

def _csm_radio_impl(time, redshift, csm_model, csm_kwargs):
    """
    Interpolate synchrotron radio flux density (mJy) onto the requested
    observer-frame time array.

    The Fortran runs on its own internal time grid; we interpolate the result
    onto the caller's time array, returning zero outside the grid.
    """
    dl = _cosmo.luminosity_distance(redshift).cgs.value
    logepsb   = csm_kwargs.pop('logepsb')
    logepse   = csm_kwargs.pop('logepse')
    p         = csm_kwargs.pop('p')
    frequency = csm_kwargs.pop('frequency')
    t_grid, flux_grid = _call_csm_radio(
        csm_model,
        redshift=redshift,
        logepsb=logepsb,
        logepse=logepse,
        p=p,
        frequency=frequency,
        luminosity_distance_cm=dl,
        **csm_kwargs,
    )
    return _interp1d(t_grid, flux_grid, bounds_error=False, fill_value=0.0)(time)


def _pop_xray_kwarg(kwargs, *names, default=None):
    for name in names:
        if name in kwargs:
            return kwargs.pop(name)
    return default


def _csm_xray_impl(time, redshift, csm_model, csm_kwargs):
    """
    Interpolate thermal bremsstrahlung X-ray output onto the requested
    observer-frame time array.
    """
    dl = csm_kwargs.pop("luminosity_distance_cm", None)
    if dl is None:
        dl = csm_kwargs.pop("luminosity_distance", None)
    if dl is None:
        dl = csm_kwargs.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value

    logepsx = csm_kwargs.pop("logepsx")
    e_min_kev = _pop_xray_kwarg(csm_kwargs, "e_min_kev", "e_min_keV", default=0.3)
    e_max_kev = _pop_xray_kwarg(csm_kwargs, "e_max_kev", "e_max_keV", default=10.0)
    output_format = _pop_xray_kwarg(
        csm_kwargs, "xray_output_format", "output_format", default="luminosity"
    )
    energy_kev = _pop_xray_kwarg(csm_kwargs, "energy_kev", "energy_keV", default=None)
    n_h_host = _pop_xray_kwarg(csm_kwargs, "n_h_host", "nh_host", default=0.0)
    n_h_mw = _pop_xray_kwarg(csm_kwargs, "n_h_mw", "nh_mw", default=0.0)
    absorb = csm_kwargs.pop("absorb", csm_kwargs.pop("xray_absorb", True))
    absorb_csm = csm_kwargs.pop("absorb_csm", False)
    csm_column_factor = csm_kwargs.pop("csm_column_factor", 1.0)
    shock_component = csm_kwargs.pop("shock_component", "total")
    mu = csm_kwargs.pop("mu", 0.62)
    mu_e = csm_kwargs.pop("mu_e", 1.18)
    mu_i = csm_kwargs.pop("mu_i", 1.30)
    electron_temperature_fraction = csm_kwargs.pop("electron_temperature_fraction", 1.0)
    compression_factor = csm_kwargs.pop("compression_factor", 4.0)
    gaunt_factor = csm_kwargs.pop("gaunt_factor", 1.2)
    normalization = csm_kwargs.pop("normalization", "emission_measure")
    max_xray_efficiency = csm_kwargs.pop("max_xray_efficiency", 1.0)
    n_energy = csm_kwargs.pop("n_energy", 128)
    frequency = csm_kwargs.pop("frequency", None)

    t_grid, xray_grid = _call_csm_xray(
        csm_model,
        redshift=redshift,
        logepsx=logepsx,
        luminosity_distance_cm=dl,
        e_min_kev=e_min_kev,
        e_max_kev=e_max_kev,
        output_format=output_format,
        energy_kev=energy_kev,
        frequency=frequency,
        n_h_host=n_h_host,
        n_h_mw=n_h_mw,
        absorb=absorb,
        absorb_csm=absorb_csm,
        csm_column_factor=csm_column_factor,
        shock_component=shock_component,
        mu=mu,
        mu_e=mu_e,
        mu_i=mu_i,
        electron_temperature_fraction=electron_temperature_fraction,
        compression_factor=compression_factor,
        gaunt_factor=gaunt_factor,
        normalization=normalization,
        max_xray_efficiency=max_xray_efficiency,
        n_energy=n_energy,
        **csm_kwargs,
    )
    return _interp1d(t_grid, xray_grid, bounds_error=False, fill_value=0.0)(time)


@_citation_wrapper(CITATION)
def csm_xray(time, redshift, csm_model, logepsx, **kwargs):
    """
    Generic thermal-bremsstrahlung X-ray wrapper for any supported CSM model.

    Pass the usual parameters for ``csm_model`` as keyword arguments. Additional
    X-ray keywords include ``e_min_kev``, ``e_max_kev``, ``output_format``
    ('luminosity', 'flux', 'spectral_luminosity', or 'flux_density'),
    ``energy_kev``/``frequency``, ``n_h_host``, ``n_h_mw``, ``absorb_csm``, and
    ``shock_component`` ('total', 'forward', or 'reverse'). By default,
    ``logepsx`` scales the free-free emission measure; set
    ``normalization='shock_power'`` for the older shock-power-fraction model.
    """
    csm_kwargs = dict(logepsx=logepsx, **kwargs)
    return _csm_xray_impl(time, redshift, csm_model, csm_kwargs)


# ---------------------------------------------------------------------------
# Wind + exponential ejecta
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def wind_exponential_radio(time, redshift, mdot, vwind, mexp, eexp, eff,
                           logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for wind + exponential ejecta CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param mdot: mass-loss rate (M_sun/yr)
    :param vwind: wind velocity (km/s)
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "wind_exponential",
        dict(mdot=mdot, vwind=vwind, mexp=mexp, eexp=eexp, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Wind + BPL ejecta
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def wind_bpl_radio(time, redshift, mdot, vwind, delta, nn, mexp, eexp, eff,
                   logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for wind + BPL ejecta CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param mdot: mass-loss rate (M_sun/yr)
    :param vwind: wind velocity (km/s)
    :param delta: inner ejecta power-law index
    :param nn: outer ejecta power-law index
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "wind_bpl",
        dict(mdot=mdot, vwind=vwind, delta=delta, nn=nn,
             mexp=mexp, eexp=eexp, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Exponential ejecta + wind CSM
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def exponential_wind_radio(time, redshift, mexp, eexp, mdot, vwind, eff,
                           logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for exponential ejecta + wind CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param mdot: mass-loss rate (M_sun/yr)
    :param vwind: wind velocity (km/s)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "exponential_wind",
        dict(mexp=mexp, eexp=eexp, mdot=mdot, vwind=vwind, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# BPL ejecta + wind CSM
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def bpl_wind_radio(time, redshift, delta, nn, mexp, eexp, mdot, vwind, eff,
                   logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for BPL ejecta + wind CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param delta: inner ejecta power-law index
    :param nn: outer ejecta power-law index
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param mdot: mass-loss rate (M_sun/yr)
    :param vwind: wind velocity (km/s)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "bpl_wind",
        dict(delta=delta, nn=nn, mexp=mexp, eexp=eexp, mdot=mdot, vwind=vwind, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Gaussian wind + exponential ejecta
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def gausswind_exponential_radio(time, redshift, t_peak, t_width, mdot_baseline,
                                mdot_peak, vwind, mexp, eexp, eff,
                                logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for Gaussian wind + exponential ejecta CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param t_peak: Gaussian wind peak time (years before explosion)
    :param t_width: Gaussian wind width sigma (years)
    :param mdot_baseline: baseline mass-loss rate (M_sun/yr)
    :param mdot_peak: peak mass-loss rate (M_sun/yr)
    :param vwind: wind velocity (km/s)
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "gausswind_exponential",
        dict(t_peak=t_peak, t_width=t_width, mdot_baseline=mdot_baseline,
             mdot_peak=mdot_peak, vwind=vwind, mexp=mexp, eexp=eexp, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Gaussian wind + BPL ejecta
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def gausswind_bpl_radio(time, redshift, t_peak, t_width, mdot_baseline,
                        mdot_peak, vwind, delta, nn, mexp, eexp, eff,
                        logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for Gaussian wind + BPL ejecta CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param t_peak: Gaussian wind peak time (years before explosion)
    :param t_width: Gaussian wind width sigma (years)
    :param mdot_baseline: baseline mass-loss rate (M_sun/yr)
    :param mdot_peak: peak mass-loss rate (M_sun/yr)
    :param vwind: wind velocity (km/s)
    :param delta: inner ejecta power-law index
    :param nn: outer ejecta power-law index
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "gausswind_bpl",
        dict(t_peak=t_peak, t_width=t_width, mdot_baseline=mdot_baseline,
             mdot_peak=mdot_peak, vwind=vwind, delta=delta, nn=nn,
             mexp=mexp, eexp=eexp, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Box wind + exponential ejecta
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def boxwind_exponential_radio(time, redshift, t1, t2, mdot_0, mdot_1, mdot_2,
                              vwind, mexp, eexp, eff,
                              logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for box (step-function) wind + exponential ejecta CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param t1: start time of enhanced wind (years)
    :param t2: end time of enhanced wind (years)
    :param mdot_0: mass-loss rate before t1 (M_sun/yr)
    :param mdot_1: mass-loss rate during box (M_sun/yr)
    :param mdot_2: mass-loss rate after t2 (M_sun/yr)
    :param vwind: wind velocity (km/s)
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "boxwind_exponential",
        dict(t1=t1, t2=t2, mdot_0=mdot_0, mdot_1=mdot_1, mdot_2=mdot_2,
             vwind=vwind, mexp=mexp, eexp=eexp, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Box wind + BPL ejecta
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def boxwind_bpl_radio(time, redshift, t1, t2, mdot_0, mdot_1, mdot_2,
                      vwind, delta, nn, mexp, eexp, eff,
                      logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for box wind + BPL ejecta CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param t1: start time of enhanced wind (years)
    :param t2: end time of enhanced wind (years)
    :param mdot_0: mass-loss rate before t1 (M_sun/yr)
    :param mdot_1: mass-loss rate during box (M_sun/yr)
    :param mdot_2: mass-loss rate after t2 (M_sun/yr)
    :param vwind: wind velocity (km/s)
    :param delta: inner ejecta power-law index
    :param nn: outer ejecta power-law index
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "boxwind_bpl",
        dict(t1=t1, t2=t2, mdot_0=mdot_0, mdot_1=mdot_1, mdot_2=mdot_2,
             vwind=vwind, delta=delta, nn=nn, mexp=mexp, eexp=eexp, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Triple power-law wind + BPL ejecta
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def triple_powerlaw_wind_bpl_radio(time, redshift, t_break1, t_break2, mdot_0,
                                   alpha1, alpha2, alpha3, vwind,
                                   delta, nn, mexp, eexp, eff,
                                   logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for triple power-law wind + BPL ejecta CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param t_break1: first break time (years)
    :param t_break2: second break time (years)
    :param mdot_0: reference mass-loss rate at t=1 yr (M_sun/yr)
    :param alpha1: power-law index t < t_break1
    :param alpha2: power-law index t_break1 < t < t_break2
    :param alpha3: power-law index t > t_break2
    :param vwind: wind velocity (km/s)
    :param delta: inner ejecta power-law index
    :param nn: outer ejecta power-law index
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "triple_powerlaw_wind_bpl",
        dict(t_break1=t_break1, t_break2=t_break2, mdot_0=mdot_0,
             alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, vwind=vwind,
             delta=delta, nn=nn, mexp=mexp, eexp=eexp, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Triple power-law wind + exponential ejecta
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def triple_powerlaw_wind_exponential_radio(time, redshift, t_break1, t_break2, mdot_0,
                                           alpha1, alpha2, alpha3, vwind,
                                           mexp, eexp, eff,
                                           logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for triple power-law wind + exponential ejecta CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param t_break1: first break time (years)
    :param t_break2: second break time (years)
    :param mdot_0: reference mass-loss rate at t=1 yr (M_sun/yr)
    :param alpha1: power-law index t < t_break1
    :param alpha2: power-law index t_break1 < t < t_break2
    :param alpha3: power-law index t > t_break2
    :param vwind: wind velocity (km/s)
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "triple_powerlaw_wind_exponential",
        dict(t_break1=t_break1, t_break2=t_break2, mdot_0=mdot_0,
             alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, vwind=vwind,
             mexp=mexp, eexp=eexp, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Exponential ejecta + triple power-law wind CSM
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def exponential_triple_powerlaw_wind_radio(time, redshift, mexp, eexp,
                                           t_break1, t_break2, mdot_0,
                                           alpha1, alpha2, alpha3, vwind, eff,
                                           logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for exponential ejecta + triple power-law wind CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param t_break1: first break time (years)
    :param t_break2: second break time (years)
    :param mdot_0: reference mass-loss rate at t=1 yr (M_sun/yr)
    :param alpha1: power-law index t < t_break1
    :param alpha2: power-law index t_break1 < t < t_break2
    :param alpha3: power-law index t > t_break2
    :param vwind: wind velocity (km/s)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "exponential_triple_powerlaw_wind",
        dict(mexp=mexp, eexp=eexp, t_break1=t_break1, t_break2=t_break2,
             mdot_0=mdot_0, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3,
             vwind=vwind, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# BPL ejecta + triple power-law wind CSM
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def bpl_triple_powerlaw_wind_radio(time, redshift, delta, nn, mexp, eexp,
                                   t_break1, t_break2, mdot_0,
                                   alpha1, alpha2, alpha3, vwind, eff,
                                   logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for BPL ejecta + triple power-law wind CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param delta: inner ejecta power-law index
    :param nn: outer ejecta power-law index
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param t_break1: first break time (years)
    :param t_break2: second break time (years)
    :param mdot_0: reference mass-loss rate at t=1 yr (M_sun/yr)
    :param alpha1: power-law index t < t_break1
    :param alpha2: power-law index t_break1 < t < t_break2
    :param alpha3: power-law index t > t_break2
    :param vwind: wind velocity (km/s)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "bpl_triple_powerlaw_wind",
        dict(delta=delta, nn=nn, mexp=mexp, eexp=eexp,
             t_break1=t_break1, t_break2=t_break2, mdot_0=mdot_0,
             alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, vwind=vwind, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Smooth triple power-law wind + BPL ejecta
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def smooth_triple_powerlaw_wind_bpl_radio(time, redshift, t_break1, t_break2, mdot_0,
                                          alpha1, alpha2, alpha3, vwind,
                                          delta, nn, mexp, eexp, eff,
                                          logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for smooth triple power-law wind + BPL ejecta CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param t_break1: first break time (years)
    :param t_break2: second break time (years)
    :param mdot_0: reference mass-loss rate at t=1 yr (M_sun/yr)
    :param alpha1: power-law index t < t_break1
    :param alpha2: power-law index t_break1 < t < t_break2
    :param alpha3: power-law index t > t_break2
    :param vwind: wind velocity (km/s)
    :param delta: inner ejecta power-law index
    :param nn: outer ejecta power-law index
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "smooth_triple_powerlaw_wind_bpl",
        dict(t_break1=t_break1, t_break2=t_break2, mdot_0=mdot_0,
             alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, vwind=vwind,
             delta=delta, nn=nn, mexp=mexp, eexp=eexp, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Smooth triple power-law wind + exponential ejecta
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def smooth_triple_powerlaw_wind_exponential_radio(time, redshift,
                                                  t_break1, t_break2, mdot_0,
                                                  alpha1, alpha2, alpha3, vwind,
                                                  mexp, eexp, eff,
                                                  logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for smooth triple power-law wind + exponential ejecta CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param t_break1: first break time (years)
    :param t_break2: second break time (years)
    :param mdot_0: reference mass-loss rate at t=1 yr (M_sun/yr)
    :param alpha1: power-law index t < t_break1
    :param alpha2: power-law index t_break1 < t < t_break2
    :param alpha3: power-law index t > t_break2
    :param vwind: wind velocity (km/s)
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "smooth_triple_powerlaw_wind_exponential",
        dict(t_break1=t_break1, t_break2=t_break2, mdot_0=mdot_0,
             alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, vwind=vwind,
             mexp=mexp, eexp=eexp, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Exponential ejecta + exponential outer CSM
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def exponential_exponential_radio(time, redshift, mexp, eexp, mexp_out, eexp_out,
                                  interval, eff,
                                  logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for exponential ejecta + exponential outer CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param mexp: inner ejecta mass (M_sun)
    :param eexp: inner explosion energy (foe)
    :param mexp_out: outer CSM mass (M_sun)
    :param eexp_out: outer CSM kinetic energy (foe)
    :param interval: time between eruptions (days)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "exponential_exponential",
        dict(mexp=mexp, eexp=eexp, mexp_out=mexp_out, eexp_out=eexp_out,
             interval=interval, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Exponential ejecta + BPL outer CSM
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def exponential_bpl_radio(time, redshift, mexp, eexp, delta_out, nn_out,
                          mexp_out, eexp_out, interval, eff,
                          logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for exponential ejecta + BPL outer CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param mexp: inner ejecta mass (M_sun)
    :param eexp: inner explosion energy (foe)
    :param delta_out: outer CSM inner power-law index
    :param nn_out: outer CSM outer power-law index
    :param mexp_out: outer CSM mass (M_sun)
    :param eexp_out: outer CSM kinetic energy (foe)
    :param interval: time between eruptions (days)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "exponential_bpl",
        dict(mexp=mexp, eexp=eexp, delta_out=delta_out, nn_out=nn_out,
             mexp_out=mexp_out, eexp_out=eexp_out, interval=interval, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# BPL ejecta + BPL outer CSM
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def bpl_bpl_radio(time, redshift, delta, nn, mexp, eexp,
                  delta_out, nn_out, mexp_out, eexp_out, interval, eff,
                  logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for BPL ejecta + BPL outer CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param delta: inner ejecta power-law index
    :param nn: outer ejecta power-law index
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param delta_out: outer CSM inner power-law index
    :param nn_out: outer CSM outer power-law index
    :param mexp_out: outer CSM mass (M_sun)
    :param eexp_out: outer CSM kinetic energy (foe)
    :param interval: time between eruptions (days)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "bpl_bpl",
        dict(delta=delta, nn=nn, mexp=mexp, eexp=eexp,
             delta_out=delta_out, nn_out=nn_out, mexp_out=mexp_out,
             eexp_out=eexp_out, interval=interval, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# BPL ejecta + exponential outer CSM
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def bpl_exponential_radio(time, redshift, delta, nn, mexp, eexp,
                          mexp_out, eexp_out, interval, eff,
                          logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for BPL ejecta + exponential outer CSM interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param delta: inner ejecta power-law index
    :param nn: outer ejecta power-law index
    :param mexp: ejecta mass (M_sun)
    :param eexp: explosion energy (foe)
    :param mexp_out: outer CSM mass (M_sun)
    :param eexp_out: outer CSM kinetic energy (foe)
    :param interval: time between eruptions (days)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "bpl_exponential",
        dict(delta=delta, nn=nn, mexp=mexp, eexp=eexp,
             mexp_out=mexp_out, eexp_out=eexp_out, interval=interval, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Generic 3-shell CSM + exponential ejecta
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def generic_csm_exponential_radio(time, redshift,
                                   base_density, base_index,
                                   shell1_radius, shell1_width, shell1_density,
                                   shell2_radius, shell2_width, shell2_density,
                                   shell3_radius, shell3_width, shell3_density,
                                   interval_sn, mej_sn, esn, eff,
                                   logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for generic 3-shell CSM + exponential ejecta interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param base_density: base power-law density at r_inner (g/cm^3)
    :param base_index: base power-law index
    :param shell{1,2,3}_radius: shell centre radius (cm)
    :param shell{1,2,3}_width: shell FWHM (cm)
    :param shell{1,2,3}_density: shell peak density (g/cm^3; 0 to disable)
    :param interval_sn: time interval between CSM construction and explosion (days)
    :param mej_sn: ejecta mass (M_sun)
    :param esn: explosion energy (foe)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "generic_csm_exponential",
        dict(base_density=base_density, base_index=base_index,
             shell1_radius=shell1_radius * _AU, shell1_width=shell1_width * _AU, shell1_density=shell1_density,
             shell2_radius=shell2_radius * _AU, shell2_width=shell2_width * _AU, shell2_density=shell2_density,
             shell3_radius=shell3_radius * _AU, shell3_width=shell3_width * _AU, shell3_density=shell3_density,
             interval_sn=interval_sn, mej_sn=mej_sn, esn=esn, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Generic 3-shell CSM + BPL ejecta
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def generic_csm_bpl_radio(time, redshift,
                           base_density, base_index,
                           shell1_radius, shell1_width, shell1_density,
                           shell2_radius, shell2_width, shell2_density,
                           shell3_radius, shell3_width, shell3_density,
                           interval_sn, delta_sn, nn_sn, mej_sn, esn, eff,
                           logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for generic 3-shell CSM + BPL ejecta interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param base_density: base power-law density at r_inner (g/cm^3)
    :param base_index: base power-law index
    :param shell{1,2,3}_radius: shell centre radius (cm)
    :param shell{1,2,3}_width: shell FWHM (cm)
    :param shell{1,2,3}_density: shell peak density (g/cm^3; 0 to disable)
    :param interval_sn: time interval between CSM construction and explosion (days)
    :param delta_sn: inner ejecta power-law index
    :param nn_sn: outer ejecta power-law index
    :param mej_sn: ejecta mass (M_sun)
    :param esn: explosion energy (foe)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "generic_csm_bpl",
        dict(base_density=base_density, base_index=base_index,
             shell1_radius=shell1_radius * _AU, shell1_width=shell1_width * _AU, shell1_density=shell1_density,
             shell2_radius=shell2_radius * _AU, shell2_width=shell2_width * _AU, shell2_density=shell2_density,
             shell3_radius=shell3_radius * _AU, shell3_width=shell3_width * _AU, shell3_density=shell3_density,
             interval_sn=interval_sn, delta_sn=delta_sn, nn_sn=nn_sn,
             mej_sn=mej_sn, esn=esn, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Generic 4-shell CSM + BPL ejecta
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def generic_4shell_csm_bpl_radio(time, redshift,
                                  base_density, base_index,
                                  shell1_radius, shell1_width, shell1_density,
                                  shell2_radius, shell2_width, shell2_density,
                                  shell3_radius, shell3_width, shell3_density,
                                  shell4_radius, shell4_width, shell4_density,
                                  interval_sn, delta_sn, nn_sn, mej_sn, esn, eff,
                                  logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for generic 4-shell CSM + BPL ejecta interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param base_density: base power-law density at r_inner (g/cm^3)
    :param base_index: base power-law index
    :param shell{1..4}_radius/width/density: shell parameters
    :param interval_sn: time interval (days)
    :param delta_sn: inner ejecta power-law index
    :param nn_sn: outer ejecta power-law index
    :param mej_sn: ejecta mass (M_sun)
    :param esn: explosion energy (foe)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "generic_4shell_csm_bpl",
        dict(base_density=base_density, base_index=base_index,
             shell1_radius=shell1_radius * _AU, shell1_width=shell1_width * _AU, shell1_density=shell1_density,
             shell2_radius=shell2_radius * _AU, shell2_width=shell2_width * _AU, shell2_density=shell2_density,
             shell3_radius=shell3_radius * _AU, shell3_width=shell3_width * _AU, shell3_density=shell3_density,
             shell4_radius=shell4_radius * _AU, shell4_width=shell4_width * _AU, shell4_density=shell4_density,
             interval_sn=interval_sn, delta_sn=delta_sn, nn_sn=nn_sn,
             mej_sn=mej_sn, esn=esn, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


# ---------------------------------------------------------------------------
# Generic 8-shell CSM + BPL ejecta
# ---------------------------------------------------------------------------

@_citation_wrapper(CITATION)
def generic_8shell_csm_bpl_radio(time, redshift,
                                  base_density, base_index,
                                  shell1_radius, shell1_width, shell1_density,
                                  shell2_radius, shell2_width, shell2_density,
                                  shell3_radius, shell3_width, shell3_density,
                                  shell4_radius, shell4_width, shell4_density,
                                  shell5_radius, shell5_width, shell5_density,
                                  shell6_radius, shell6_width, shell6_density,
                                  shell7_radius, shell7_width, shell7_density,
                                  shell8_radius, shell8_width, shell8_density,
                                  interval_sn, delta_sn, nn_sn, mej_sn, esn, eff,
                                  logepsb, logepse, p, **kwargs):
    """Radio flux density (mJy) for generic 8-shell CSM + BPL ejecta interaction.

    :param time: observer-frame time in days
    :param redshift: source redshift
    :param base_density: base power-law density at r_inner (g/cm^3)
    :param base_index: base power-law index
    :param shell{1..8}_radius/width/density: shell parameters
    :param interval_sn: time interval (days)
    :param delta_sn: inner ejecta power-law index
    :param nn_sn: outer ejecta power-law index
    :param mej_sn: ejecta mass (M_sun)
    :param esn: explosion energy (foe)
    :param eff: radiative efficiency
    :param logepsb: log10(epsilon_B)
    :param logepse: log10(epsilon_e)
    :param p: electron power-law index
    :param frequency: observing frequency in Hz (passed via kwargs)
    :return: flux density in mJy
    """
    return _csm_radio_impl(
        time, redshift, "generic_8shell_csm_bpl",
        dict(base_density=base_density, base_index=base_index,
             shell1_radius=shell1_radius * _AU, shell1_width=shell1_width * _AU, shell1_density=shell1_density,
             shell2_radius=shell2_radius * _AU, shell2_width=shell2_width * _AU, shell2_density=shell2_density,
             shell3_radius=shell3_radius * _AU, shell3_width=shell3_width * _AU, shell3_density=shell3_density,
             shell4_radius=shell4_radius * _AU, shell4_width=shell4_width * _AU, shell4_density=shell4_density,
             shell5_radius=shell5_radius * _AU, shell5_width=shell5_width * _AU, shell5_density=shell5_density,
             shell6_radius=shell6_radius * _AU, shell6_width=shell6_width * _AU, shell6_density=shell6_density,
             shell7_radius=shell7_radius * _AU, shell7_width=shell7_width * _AU, shell7_density=shell7_density,
             shell8_radius=shell8_radius * _AU, shell8_width=shell8_width * _AU, shell8_density=shell8_density,
             interval_sn=interval_sn, delta_sn=delta_sn, nn_sn=nn_sn,
             mej_sn=mej_sn, esn=esn, eff=eff,
             logepsb=logepsb, logepse=logepse, p=p, **kwargs),
    )


def _make_xray_wrapper(base_name, radio_func):
    """Create a radio-signature-matched X-ray wrapper for a base CSM model."""
    csm_model = {
        "generic_powerlaw_csm_exponential": "static_powerlaw_csm_exponential",
        "generic_powerlaw_csm_bpl": "static_powerlaw_csm_bpl",
    }.get(base_name, base_name)
    radio_sig = _inspect.signature(radio_func)
    new_params = []
    inserted = False
    for param in radio_sig.parameters.values():
        if param.name in ("logepsb", "logepse", "p"):
            if not inserted:
                new_params.append(
                    _inspect.Parameter(
                        "logepsx",
                        kind=param.kind,
                        default=_inspect.Parameter.empty,
                    )
                )
                inserted = True
            continue
        new_params.append(param)
    new_sig = radio_sig.replace(parameters=new_params)

    def wrapper(*args, **kwargs):
        bound = new_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        extra_kwargs = values.pop("kwargs", {})
        values.update(extra_kwargs)
        time = values.pop("time")
        redshift = values.pop("redshift")
        logepsx = values.pop("logepsx")
        values["logepsx"] = logepsx
        return _csm_xray_impl(time, redshift, csm_model, values)

    wrapper.__name__ = f"{base_name}_xray"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__signature__ = new_sig
    wrapper.__doc__ = (
        f"Thermal bremsstrahlung X-ray emission for ``{base_name}``.\n\n"
        "Uses the same physical CSM parameters as the corresponding radio "
        "wrapper, replacing ``logepsb, logepse, p`` with ``logepsx``. "
        "X-ray controls are supplied through kwargs: ``e_min_kev``, "
        "``e_max_kev``, ``output_format``, ``energy_kev``/``frequency``, "
        "``n_h_host``, ``n_h_mw``, ``absorb_csm``, ``shock_component``, "
        "``normalization``, and ``max_xray_efficiency``."
    )
    wrapper = _citation_wrapper(CITATION)(wrapper)
    wrapper.__signature__ = new_sig
    return wrapper


_XRAY_MODEL_NAMES = []
for _radio_name, _radio_func in list(globals().items()):
    if not (_radio_name.endswith("_radio") and callable(_radio_func)):
        continue
    _base_name = _radio_name[:-6]
    _xray_name = f"{_base_name}_xray"
    if _xray_name in globals():
        continue
    globals()[_xray_name] = _make_xray_wrapper(_base_name, _radio_func)
    _XRAY_MODEL_NAMES.append(_xray_name)

__all__.extend(name for name in _XRAY_MODEL_NAMES if name not in __all__)
