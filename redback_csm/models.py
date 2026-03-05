"""
models.py — Public redback-compatible CSM model functions.

Each CSM physical scenario is exposed as four functions:

  {name}_bolometric          bolometric luminosity (erg/s), source-frame time (days)
  {name}                     multiband, observer-frame time, full output_format dispatch
  {name}_nickel_bolometric   CSM + radioactive nickel decay, bolometric
  {name}_nickel              CSM + nickel, multiband

All functions are registered in redback.model_library.all_models_dict automatically
when the package is installed, via the redback.model.modules entry point.

Citation: Sarin & Hirai (in prep); Sarin et al. 2024 (redback, JOSS)
"""

import numpy as _np
from collections import namedtuple as _namedtuple
from scipy.interpolate import interp1d as _interp1d
import astropy.units as _uu
from astropy.cosmology import Planck18 as _cosmo

from redback.utils import citation_wrapper as _citation_wrapper
from redback.utils import calc_kcorrected_properties as _calc_kcorrected_properties
from redback.utils import lambda_to_nu as _lambda_to_nu
import redback.sed as _sed
import redback.photosphere as _photosphere
from redback.transient_models.supernova_models import (
    arnett_bolometric as _arnett_bolometric,
)

from redback_csm.core import _call_csm

DAY = 86400.0  # seconds per day
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
    "generic_4shell_csm_bpl_bolometric",
    "generic_4shell_csm_bpl",
    "generic_4shell_csm_bpl_nickel_bolometric",
    "generic_4shell_csm_bpl_nickel",
    "generic_8shell_csm_bpl_bolometric",
    "generic_8shell_csm_bpl",
    "generic_8shell_csm_bpl_nickel_bolometric",
    "generic_8shell_csm_bpl_nickel",
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


def _csm_nickel_bolometric_impl(time, csm_model, **kwargs):
    """Combined CSM shock + radioactive nickel bolometric luminosity."""
    return _csm_bolometric_impl(time, csm_model, **kwargs) + _arnett_bolometric(
        time=time, **kwargs
    )


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
    return sed_1.flux_density.to(_uu.mJy).value / (1 + redshift)


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
    return sed_1.flux_density.to(_uu.mJy).value / (1 + redshift)


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
    time, mdot, vwind, mexp, eexp, eff, f_nickel, mej, **kwargs
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
        mej=mej,
        **kwargs
    )


@_citation_wrapper(CITATION)
def wind_exponential_nickel(
    time, redshift, mdot, vwind, mexp, eexp, eff, f_nickel, mej, **kwargs
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
        mej=mej,
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
    time, mdot, vwind, delta, nn, mexp, eexp, eff, f_nickel, mej, **kwargs
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
        mej=mej,
        **kwargs
    )


@_citation_wrapper(CITATION)
def wind_bpl_nickel(
    time, redshift, mdot, vwind, delta, nn, mexp, eexp, eff, f_nickel, mej, **kwargs
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
        mej=mej,
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
    time, mexp, eexp, mdot, vwind, eff, f_nickel, mej, **kwargs
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
        mej=mej,
        **kwargs
    )


@_citation_wrapper(CITATION)
def exponential_wind_nickel(
    time, redshift, mexp, eexp, mdot, vwind, eff, f_nickel, mej, **kwargs
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
        mej=mej,
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
    time, delta, nn, mexp, eexp, mdot, vwind, eff, f_nickel, mej, **kwargs
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
        mej=mej,
        **kwargs
    )


@_citation_wrapper(CITATION)
def bpl_wind_nickel(
    time, redshift, delta, nn, mexp, eexp, mdot, vwind, eff, f_nickel, mej, **kwargs
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
        mej=mej,
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
    time, mexp, eexp, mexp_out, eexp_out, interval, eff, f_nickel, mej, **kwargs
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
    mej,
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
        mej=mej,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
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
    mej,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
        shell3_density=shell3_density,
        interval_sn=interval_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        mej=mej,
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
    mej,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
        shell3_density=shell3_density,
        interval_sn=interval_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        mej=mej,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
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
    mej,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
        shell3_density=shell3_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        mej=mej,
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
    mej,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
        shell3_density=shell3_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        mej=mej,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius,
        shell4_width=shell4_width,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius,
        shell4_width=shell4_width,
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
    mej,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius,
        shell4_width=shell4_width,
        shell4_density=shell4_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        mej=mej,
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
    mej,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius,
        shell4_width=shell4_width,
        shell4_density=shell4_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        mej=mej,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius,
        shell4_width=shell4_width,
        shell4_density=shell4_density,
        shell5_radius=shell5_radius,
        shell5_width=shell5_width,
        shell5_density=shell5_density,
        shell6_radius=shell6_radius,
        shell6_width=shell6_width,
        shell6_density=shell6_density,
        shell7_radius=shell7_radius,
        shell7_width=shell7_width,
        shell7_density=shell7_density,
        shell8_radius=shell8_radius,
        shell8_width=shell8_width,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius,
        shell4_width=shell4_width,
        shell4_density=shell4_density,
        shell5_radius=shell5_radius,
        shell5_width=shell5_width,
        shell5_density=shell5_density,
        shell6_radius=shell6_radius,
        shell6_width=shell6_width,
        shell6_density=shell6_density,
        shell7_radius=shell7_radius,
        shell7_width=shell7_width,
        shell7_density=shell7_density,
        shell8_radius=shell8_radius,
        shell8_width=shell8_width,
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
    mej,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius,
        shell4_width=shell4_width,
        shell4_density=shell4_density,
        shell5_radius=shell5_radius,
        shell5_width=shell5_width,
        shell5_density=shell5_density,
        shell6_radius=shell6_radius,
        shell6_width=shell6_width,
        shell6_density=shell6_density,
        shell7_radius=shell7_radius,
        shell7_width=shell7_width,
        shell7_density=shell7_density,
        shell8_radius=shell8_radius,
        shell8_width=shell8_width,
        shell8_density=shell8_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        mej=mej,
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
    mej,
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
        shell1_radius=shell1_radius,
        shell1_width=shell1_width,
        shell1_density=shell1_density,
        shell2_radius=shell2_radius,
        shell2_width=shell2_width,
        shell2_density=shell2_density,
        shell3_radius=shell3_radius,
        shell3_width=shell3_width,
        shell3_density=shell3_density,
        shell4_radius=shell4_radius,
        shell4_width=shell4_width,
        shell4_density=shell4_density,
        shell5_radius=shell5_radius,
        shell5_width=shell5_width,
        shell5_density=shell5_density,
        shell6_radius=shell6_radius,
        shell6_width=shell6_width,
        shell6_density=shell6_density,
        shell7_radius=shell7_radius,
        shell7_width=shell7_width,
        shell7_density=shell7_density,
        shell8_radius=shell8_radius,
        shell8_width=shell8_width,
        shell8_density=shell8_density,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        f_nickel=f_nickel,
        mej=mej,
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
