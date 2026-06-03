"""
models.py — Public redback-compatible CSM model functions.

Each CSM physical scenario is exposed as a small wrapper family:

  {name}_bolometric          bolometric luminosity (erg/s), source-frame time (days)
  {name}                     multiband, observer-frame time, full output_format dispatch
  {name}_nickel_bolometric   CSM + radioactive nickel decay, bolometric
  {name}_nickel              CSM + nickel, multiband
  {name}_radio               synchrotron radio flux density
  {name}_xray                thermal bremsstrahlung X-ray output

All functions are registered in redback.model_library.all_models_dict automatically
when the package is installed, via the redback.model.modules entry point.

Citation: Sarin & Hirai 2026 (arXiv:2605.19571); Sarin et al. 2024 (redback)
"""

redback_model_type = "supernova"

import inspect as _inspect
import os as _os
import collections as _collections

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
    GENERIC_CSM_DEFAULT_N_POINTS,
    create_static_spline_csm_density as _create_static_spline_csm_density,
    create_generic_csm_density as _create_generic_csm_density,
    create_generic_spline_csm_density as _create_generic_spline_csm_density,
    pspline_log_rho_nodes as _pspline_log_rho_nodes,
)

DAY = 86400.0    # seconds per day
_AU = 1.496e13   # cm per AU
_SOLAR_MASS = 1.989e33
_TRAPEZOID = getattr(_np, "trapezoid", _np.trapz)

BASE_MODEL_NAMES = (
    "wind_exponential",
    "wind_bpl",
    "exponential_wind",
    "bpl_wind",
    "exponential_exponential",
    "exponential_bpl",
    "bpl_bpl",
    "bpl_exponential",
    "boxwind_exponential",
    "boxwind_bpl",
    "gausswind_exponential",
    "gausswind_bpl",
    "triple_powerlaw_wind_bpl",
    "triple_powerlaw_wind_exponential",
    "exponential_triple_powerlaw_wind",
    "bpl_triple_powerlaw_wind",
    "smooth_triple_powerlaw_wind_bpl",
    "smooth_triple_powerlaw_wind_exponential",
    "generic_csm_exponential",
    "generic_csm_bpl",
    "static_powerlaw_csm_exponential",
    "static_powerlaw_csm_bpl",
    "homologous_powerlaw_csm_exponential",
    "homologous_powerlaw_csm_bpl",
    "generic_4shell_csm_bpl",
    "generic_8shell_csm_bpl",
    "static_spline_csm_bpl",
    "generic_spline_csm_bpl",
    "generic_spline12_csm_bpl",
    "static_pspline24_csm_bpl",
    "generic_pspline24_csm_bpl",
    "static_pspline48_csm_bpl",
    "generic_pspline48_csm_bpl",
    "static_pspline96_csm_bpl",
    "generic_pspline96_csm_bpl",
)

__all__ = []

CITATION = "Sarin & Hirai 2026 (arXiv:2605.19571); Sarin et al. 2024"

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
        n_points=int(kwargs.get("n_points", GENERIC_CSM_DEFAULT_N_POINTS)),
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


def _generic_spline_csm_mass(kwargs):
    """Estimate CSM mass for homologous spline-profile constructors."""
    node_indices = sorted(
        int(key.rsplit("_", 1)[1])
        for key in kwargs
        if key.startswith("log_rho_") and key.rsplit("_", 1)[1].isdigit()
    )
    log_rho_nodes = _np.array([kwargs[f"log_rho_{i}"] for i in node_indices], dtype=float)
    r_grid, _, rho = _create_generic_spline_csm_density(
        log_r_inner=float(kwargs["log_r_inner"]),
        log_r_outer=float(kwargs["log_r_outer"]),
        log_rho_nodes=log_rho_nodes,
        interval_sn=float(kwargs.get("interval_sn", 10.0 * 365.25)),
        n_points=int(kwargs.get("n_points", GENERIC_CSM_DEFAULT_N_POINTS)),
    )
    mass_cgs = _TRAPEZOID(4.0 * _np.pi * r_grid**2 * _np.maximum(rho, 0.0), r_grid)
    return max(float(mass_cgs / _SOLAR_MASS), 0.0)


def _pspline_node_values(kwargs):
    """Return reconstructed log-density nodes for p-spline constructors."""
    indices = sorted(
        int(key.rsplit("_", 1)[1])
        for key in kwargs
        if key.startswith("d2_log_rho_") and key.rsplit("_", 1)[1].isdigit()
    )
    if not indices:
        raise ValueError("p-spline CSM models require d2_log_rho_* parameters")
    d2_nodes = _np.array([kwargs[f"d2_log_rho_{idx}"] for idx in indices], dtype=float)
    return _pspline_log_rho_nodes(kwargs["log_rho_0"], kwargs["dlog_rho_0"], d2_nodes)


def _pspline_csm_mass(kwargs, profile):
    """Estimate CSM mass for static or homologous p-spline constructors."""
    log_rho_nodes = _pspline_node_values(kwargs)
    n_points = int(kwargs.get("n_points", GENERIC_CSM_DEFAULT_N_POINTS))
    if profile == "generic":
        r_grid, _, rho = _create_generic_spline_csm_density(
            log_r_inner=float(kwargs["log_r_inner"]),
            log_r_outer=float(kwargs["log_r_outer"]),
            log_rho_nodes=log_rho_nodes,
            interval_sn=float(kwargs.get("interval_sn", 10.0 * 365.25)),
            n_points=n_points,
        )
    elif profile == "static":
        r_grid, rho = _create_static_spline_csm_density(
            log_r_inner=float(kwargs["log_r_inner"]),
            log_r_outer=float(kwargs["log_r_outer"]),
            log_rho_nodes=log_rho_nodes,
            n_points=n_points,
        )
    else:
        raise ValueError("profile must be 'generic' or 'static'")
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
    if csm_model.startswith("generic_spline_csm_") or csm_model.startswith("generic_spline12_csm_"):
        return _generic_spline_csm_mass(kwargs)
    if csm_model.startswith("generic_pspline"):
        return _pspline_csm_mass(kwargs, profile="generic")
    if csm_model.startswith("static_pspline"):
        return _pspline_csm_mass(kwargs, profile="static")
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
        dense_times = _np.linspace(0.0, float(_np.max(time)) + 100.0, dense_resolution)
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
    if "vej" not in csm_kwargs:
        mej_ejecta, eexp = _nickel_ejecta_mass_energy(csm_kwargs)
        csm_kwargs["vej"] = _np.sqrt(2.0 * eexp * 1e51 / (mej_ejecta * _SOLAR_MASS)) / 1e5
    csm_kwargs.setdefault("temperature_floor", 3000.0)
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
    if "vej" not in csm_kwargs:
        mej_ejecta, eexp = _nickel_ejecta_mass_energy(csm_kwargs)
        csm_kwargs["vej"] = _np.sqrt(2.0 * eexp * 1e51 / (mej_ejecta * _SOLAR_MASS)) / 1e5
    csm_kwargs.setdefault("temperature_floor", 3000.0)
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
        return _collections.namedtuple("output", ["time", "lambdas", "spectra"])(
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
def static_powerlaw_csm_exponential_bolometric(
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
def homologous_powerlaw_csm_exponential_bolometric(
    time,
    eta,
    r_inner,
    r_outer,
    interval_sn,
    mej_sn,
    esn,
    eff,
    m_csm,
    **kwargs
):
    """Bolometric light curve for a homologous finite power-law CSM shell plus exponential ejecta."""
    return _csm_bolometric_impl(
        time,
        "homologous_powerlaw_csm_exponential",
        eta=eta,
        r_inner=r_inner,
        r_outer=r_outer,
        interval_sn=interval_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        m_csm=m_csm,
        **kwargs
    )


@_citation_wrapper(CITATION)
def homologous_powerlaw_csm_bpl_bolometric(
    time,
    eta,
    r_inner,
    r_outer,
    interval_sn,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    m_csm,
    **kwargs
):
    """Bolometric light curve for a homologous finite power-law CSM shell plus BPL ejecta."""
    return _csm_bolometric_impl(
        time,
        "homologous_powerlaw_csm_bpl",
        eta=eta,
        r_inner=r_inner,
        r_outer=r_outer,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        m_csm=m_csm,
        **kwargs
    )








@_citation_wrapper(CITATION)
def static_powerlaw_csm_bpl_bolometric(
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
def generic_spline_csm_bpl_bolometric(
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
    interval_sn,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    **kwargs
):
    """Bolometric light curve for a homologous finite spline CSM plus BPL ejecta."""
    return _csm_bolometric_impl(
        time,
        "generic_spline_csm_bpl",
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
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
        **kwargs
    )








@_citation_wrapper(CITATION)
def generic_spline12_csm_bpl_bolometric(
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
    log_rho_8,
    log_rho_9,
    log_rho_10,
    log_rho_11,
    interval_sn,
    delta_sn,
    nn_sn,
    mej_sn,
    esn,
    eff,
    **kwargs
):
    """Bolometric light curve for a homologous finite 12-node spline CSM plus BPL ejecta."""
    return _csm_bolometric_impl(
        time,
        "generic_spline12_csm_bpl",
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
        log_rho_8=log_rho_8,
        log_rho_9=log_rho_9,
        log_rho_10=log_rho_10,
        log_rho_11=log_rho_11,
        interval_sn=interval_sn,
        delta_sn=delta_sn,
        nn_sn=nn_sn,
        mej_sn=mej_sn,
        esn=esn,
        eff=eff,
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
    :param vej_max_ratio: (optional) finite BPL ejecta cutoff, v_max / v_t.
        Defaults to 3 when omitted. Larger values allow faster outer ejecta.
        The aliases A_ratio and vej_max (km/s) are also accepted via kwargs.
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



# ---------------------------------------------------------------------------
# Wind + BPL ejecta
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Exponential ejecta + wind CSM
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# BPL ejecta + wind CSM
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Gaussian wind + exponential ejecta
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Gaussian wind + BPL ejecta
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Box wind + exponential ejecta
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Box wind + BPL ejecta
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Triple power-law wind + BPL ejecta
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Triple power-law wind + exponential ejecta
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Exponential ejecta + triple power-law wind CSM
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# BPL ejecta + triple power-law wind CSM
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Smooth triple power-law wind + BPL ejecta
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Smooth triple power-law wind + exponential ejecta
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Exponential ejecta + exponential outer CSM
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Exponential ejecta + BPL outer CSM
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# BPL ejecta + BPL outer CSM
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# BPL ejecta + exponential outer CSM
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Generic 3-shell CSM + exponential ejecta
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Generic 3-shell CSM + BPL ejecta
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Generic 4-shell CSM + BPL ejecta
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Generic 8-shell CSM + BPL ejecta
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Spline CSM + BPL ejecta
# ---------------------------------------------------------------------------







def _core_model_name(base_name):
    return base_name


def _pspline_parameter_names(n_nodes=24, homologous=False):
    names = ["log_r_inner", "log_r_outer", "log_rho_0", "dlog_rho_0"]
    names.extend(f"d2_log_rho_{idx}" for idx in range(int(n_nodes) - 2))
    if homologous:
        names.append("interval_sn")
    names.extend(["delta_sn", "nn_sn", "mej_sn", "esn", "eff"])
    return tuple(names)


def _make_pspline_bolometric_wrapper(base_name, n_nodes=24, homologous=False):
    parameters = [
        _inspect.Parameter(
            "time",
            kind=_inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=_inspect.Parameter.empty,
        )
    ]
    parameters.extend(
        _inspect.Parameter(
            name,
            kind=_inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=_inspect.Parameter.empty,
        )
        for name in _pspline_parameter_names(n_nodes=n_nodes, homologous=homologous)
    )
    parameters.append(
        _inspect.Parameter(
            "kwargs",
            kind=_inspect.Parameter.VAR_KEYWORD,
            default=_inspect.Parameter.empty,
        )
    )
    sig = _inspect.Signature(parameters)

    def wrapper(*args, **kwargs):
        values = _bound_public_values(sig, args, kwargs)
        time = values.pop("time")
        return _csm_bolometric_impl(time, base_name, **values)

    wrapper.__name__ = f"{base_name}_bolometric"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__signature__ = sig
    wrapper.__doc__ = (
        f"Bolometric light curve for ``{base_name}``, using a {n_nodes}-node "
        "p-spline density parameterisation."
    )
    wrapper = _citation_wrapper(CITATION)(wrapper)
    wrapper.__signature__ = sig
    return wrapper


for _pspline_n_nodes in (24, 48, 96):
    globals()[f"static_pspline{_pspline_n_nodes}_csm_bpl_bolometric"] = (
        _make_pspline_bolometric_wrapper(
            f"static_pspline{_pspline_n_nodes}_csm_bpl",
            n_nodes=_pspline_n_nodes,
            homologous=False,
        )
    )
    globals()[f"generic_pspline{_pspline_n_nodes}_csm_bpl_bolometric"] = (
        _make_pspline_bolometric_wrapper(
            f"generic_pspline{_pspline_n_nodes}_csm_bpl",
            n_nodes=_pspline_n_nodes,
            homologous=True,
        )
    )


def _insert_signature_parameters(sig, parameters):
    params = list(sig.parameters.values())
    insert_at = len(params)
    for i, param in enumerate(params):
        if param.kind == param.VAR_KEYWORD:
            insert_at = i
            break
    new_params = (
        params[:insert_at]
        + [
            _inspect.Parameter(
                name,
                kind=_inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=_inspect.Parameter.empty,
            )
            for name in parameters
        ]
        + params[insert_at:]
    )
    return sig.replace(parameters=new_params)


def _insert_signature_parameters_after(sig, after_name, parameters):
    params = list(sig.parameters.values())
    insert_at = 1
    for i, param in enumerate(params):
        if param.name == after_name:
            insert_at = i + 1
            break
    new_params = (
        params[:insert_at]
        + [
            _inspect.Parameter(
                name,
                kind=_inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=_inspect.Parameter.empty,
            )
            for name in parameters
        ]
        + params[insert_at:]
    )
    return sig.replace(parameters=new_params)


def _bound_public_values(sig, args, kwargs):
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    values = dict(bound.arguments)
    extra_kwargs = values.pop("kwargs", {})
    values.update(extra_kwargs)
    return values


def _public_values_to_core_kwargs(values):
    core_kwargs = {}
    for key, value in values.items():
        if key in {"time", "redshift"}:
            continue
        if _re_shell_radius_width(key):
            value = value * _AU
        core_kwargs[key] = value
    return core_kwargs


def _re_shell_radius_width(key):
    return (
        key.startswith("shell")
        and (key.endswith("_radius") or key.endswith("_width"))
        and key[5:key.find("_")].isdigit()
    )


def _make_multiband_wrapper(base_name):
    core_model = _core_model_name(base_name)
    base_sig = _inspect.signature(globals()[f"{base_name}_bolometric"])
    new_sig = _insert_signature_parameters_after(base_sig, "time", ("redshift",))

    def wrapper(*args, **kwargs):
        values = _bound_public_values(new_sig, args, kwargs)
        time = values.pop("time")
        redshift = values.pop("redshift")
        dl = values.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
        csm_kwargs = _public_values_to_core_kwargs(values)
        if csm_kwargs.get("output_format") == "flux_density":
            return _multiband_csm_flux_density(time, redshift, core_model, csm_kwargs, dl)
        return _multiband_output(
            *_multiband_csm(time, redshift, core_model, csm_kwargs, dl),
            **csm_kwargs
        )

    wrapper.__name__ = base_name
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__signature__ = new_sig
    wrapper.__doc__ = f"Multiband light curve for ``{base_name}``."
    wrapper = _citation_wrapper(CITATION)(wrapper)
    wrapper.__signature__ = new_sig
    return wrapper


def _make_nickel_bolometric_wrapper(base_name):
    core_model = _core_model_name(base_name)
    base_sig = _inspect.signature(globals()[f"{base_name}_bolometric"])
    new_sig = _insert_signature_parameters(base_sig, ("f_nickel",))

    def wrapper(*args, **kwargs):
        values = _bound_public_values(new_sig, args, kwargs)
        time = values.pop("time")
        csm_kwargs = _public_values_to_core_kwargs(values)
        return _csm_nickel_bolometric_impl(time, core_model, **csm_kwargs)

    wrapper.__name__ = f"{base_name}_nickel_bolometric"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__signature__ = new_sig
    wrapper.__doc__ = (
        f"Bolometric light curve for ``{base_name}`` plus radioactive "
        "nickel/cobalt decay."
    )
    wrapper = _citation_wrapper(CITATION)(wrapper)
    wrapper.__signature__ = new_sig
    return wrapper


def _make_nickel_wrapper(base_name):
    core_model = _core_model_name(base_name)
    base_sig = _inspect.signature(globals()[base_name])
    new_sig = _insert_signature_parameters(base_sig, ("f_nickel",))

    def wrapper(*args, **kwargs):
        values = _bound_public_values(new_sig, args, kwargs)
        time = values.pop("time")
        redshift = values.pop("redshift")
        dl = values.get("cosmology", _cosmo).luminosity_distance(redshift).cgs.value
        csm_kwargs = _public_values_to_core_kwargs(values)
        if csm_kwargs.get("output_format") == "flux_density":
            return _multiband_nickel_flux_density(
                time, redshift, core_model, csm_kwargs, dl
            )
        return _multiband_output(
            *_multiband_nickel(time, redshift, core_model, csm_kwargs, dl),
            **csm_kwargs
        )

    wrapper.__name__ = f"{base_name}_nickel"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__signature__ = new_sig
    wrapper.__doc__ = (
        f"Multiband light curve for ``{base_name}`` plus radioactive "
        "nickel/cobalt decay."
    )
    wrapper = _citation_wrapper(CITATION)(wrapper)
    wrapper.__signature__ = new_sig
    return wrapper


def _make_radio_wrapper(base_name):
    core_model = _core_model_name(base_name)
    base_sig = _inspect.signature(globals()[base_name])
    new_sig = _insert_signature_parameters(base_sig, ("logepsb", "logepse", "p"))

    def wrapper(*args, **kwargs):
        values = _bound_public_values(new_sig, args, kwargs)
        time = values.pop("time")
        redshift = values.pop("redshift")
        csm_kwargs = _public_values_to_core_kwargs(values)
        return _csm_radio_impl(time, redshift, core_model, csm_kwargs)

    wrapper.__name__ = f"{base_name}_radio"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__signature__ = new_sig
    wrapper.__doc__ = f"Radio synchrotron flux density for ``{base_name}``."
    wrapper = _citation_wrapper(CITATION)(wrapper)
    wrapper.__signature__ = new_sig
    return wrapper


_BASE_MODEL_NAMES = list(BASE_MODEL_NAMES)
for _base_name in _BASE_MODEL_NAMES:
    globals()[_base_name] = _make_multiband_wrapper(_base_name)
    globals()[f"{_base_name}_nickel_bolometric"] = _make_nickel_bolometric_wrapper(_base_name)
    globals()[f"{_base_name}_nickel"] = _make_nickel_wrapper(_base_name)
    globals()[f"{_base_name}_radio"] = _make_radio_wrapper(_base_name)

__all__ = []
for _base_name in _BASE_MODEL_NAMES:
    __all__.extend([
        f"{_base_name}_bolometric",
        _base_name,
        f"{_base_name}_nickel_bolometric",
        f"{_base_name}_nickel",
        f"{_base_name}_radio",
    ])


def _make_xray_wrapper(base_name, radio_func):
    """Create a radio-signature-matched X-ray wrapper for a base CSM model."""
    csm_model = _core_model_name(base_name)
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
if "csm_xray" not in __all__:
    __all__.append("csm_xray")
