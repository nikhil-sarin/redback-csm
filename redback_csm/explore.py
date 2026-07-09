"""
explore.py — Utility for computing and plotting CSM interaction light curves
from an arbitrary user-supplied density profile.

Intended for exploration and visualisation (e.g. stellar-evolution density
profiles), not for Bayesian inference.  The supernova ejecta can be either a
broken power law (BPL) or an exponential profile.

Example
-------
>>> import numpy as np
>>> from redback_csm.explore import csm_lightcurve_from_density

>>> # Simple r^{-2} wind profile
>>> r = np.geomspace(1e13, 1e17, 500)          # cm
>>> rho = 5e16 / r**2                           # g/cm^3

>>> fig = csm_lightcurve_from_density(
...     radius=r, density=rho,
...     t_ref=365.0,            # days — sets r = v * t_ref
...     sn_profile='bpl',
...     delta=1.0, nn=12.0,
...     mexp=10.0, eexp=1.0,
...     eff=0.3, kappa=0.34,
... )
>>> fig.savefig('my_lightcurve.png')
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from redback_csm.core import (
    _get_csm,
    _configure_runtime_from_kwargs,
    DAY,
    YEAR,
    foe,
    solar_mass,
    solar_mass_per_yr_to_gram_per_sec,
)

__all__ = [
    "csm_lightcurve_from_density",
    "mass_loss_history",
    "csm_density_profile",
    "plot_mass_loss_history",
    "plot_csm_density_profile",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_fortran(
    v_grid, density, t_ref_sec, sn_profile, delta, nn, mexp_g, eexp_erg, eff, kappa
):
    """Call the appropriate Fortran subroutine and return the lc namedtuple."""
    csm = _get_csm()

    if sn_profile == "bpl":
        if kappa is not None:
            csm.lc_mod.lightcurve_explosion_bpl(
                density,
                v_grid,
                t_ref_sec,
                delta,
                nn,
                mexp_g,
                eexp_erg,
                t_ref_sec,
                eff,
                kappa,
            )
        else:
            csm.lc_mod.lightcurve_explosion_bpl(
                density, v_grid, t_ref_sec, delta, nn, mexp_g, eexp_erg, t_ref_sec, eff
            )
    else:  # exponential
        if kappa is not None:
            csm.lc_mod.lightcurve_explosion_exponential(
                density, v_grid, t_ref_sec, mexp_g, eexp_erg, t_ref_sec, eff, kappa
            )
        else:
            csm.lc_mod.lightcurve_explosion_exponential(
                density, v_grid, t_ref_sec, mexp_g, eexp_erg, t_ref_sec, eff
            )

    time = csm.lc_mod.tarray.copy()
    lbol_shock = csm.lc_mod.larray.copy()
    rph = csm.lc_mod.rpharray.copy()
    temperature = csm.lc_mod.temparray.copy()
    lbol_diffuse = csm.lc_mod.ldiff.copy() if kappa is not None else None
    lbol = lbol_diffuse if kappa is not None else lbol_shock

    LC = namedtuple(
        "LC", ["time", "lbol", "lbol_shock", "lbol_diffuse", "rph", "temperature"]
    )
    return LC(
        time=time,
        lbol=lbol,
        lbol_shock=lbol_shock,
        lbol_diffuse=lbol_diffuse,
        rph=rph,
        temperature=temperature,
    )


def _radius_density_to_vgrid(radius, density, t_ref_sec):
    """
    Interpolate a user-supplied ρ(r) profile onto the internal velocity grid
    expected by Fortran: v_grid = 1…100000 km/s (in cm/s steps of 1 km/s),
    with r = v * t_ref.

    Values outside the supplied radius range are extrapolated as zero density.
    """
    v_grid = np.arange(1, 100001, dtype=np.float64) * 1e5  # cm/s
    r_grid = v_grid * t_ref_sec  # cm

    # Log-space interpolation is more accurate for power-law profiles
    log_r = np.log10(radius)
    log_rho = np.log10(np.clip(density, 1e-100, None))

    log_rho_interp = np.interp(
        np.log10(r_grid),
        log_r,
        log_rho,
        left=-100.0,  # zero outside range
        right=-100.0,
    )
    density_interp = 10.0**log_rho_interp

    return v_grid, density_interp


def _canonical_model_name(model_name):
    name = str(model_name).lower()
    for suffix in (
        "_nickel_bolometric",
        "_bolometric",
        "_nickel",
        "_radio",
        "_xray",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def mass_loss_history(model_name, time_before_sn, **kwargs):
    """
    Return the wind mass-loss history used by a redback-csm wind model.

    This helper is intended for visualisation and sanity checks. The time
    coordinate is the wind look-back time before explosion in **years**. For a
    constant-velocity wind, material at radius ``r`` at explosion corresponds
    to ``time_before_sn = r / vwind``.

    Supported models are the wind-history families:
    ``wind_*``, ``*_wind``, ``boxwind_*``, ``gausswind_*``,
    ``triple_powerlaw_wind_*``, ``*_triple_powerlaw_wind`` and
    ``smooth_triple_powerlaw_wind_*``.

    For ``boxwind_*`` with ``t1 < t2``:

    - ``mdot_0`` applies for ``time_before_sn <= t1``; this is the most recent
      wind material and maps to the smallest radii.
    - ``mdot_1`` applies for ``t1 < time_before_sn < t2``; this is the box
      enhancement.
    - ``mdot_2`` applies for ``time_before_sn >= t2``; this is older material
      and maps to larger radii.

    Parameters
    ----------
    model_name : str
        Base model name, e.g. ``"boxwind_bpl"``. Suffixes such as
        ``"_bolometric"`` and ``"_radio"`` are accepted and stripped.
    time_before_sn : array-like
        Look-back time before explosion in years.
    **kwargs
        Model parameters, including the relevant ``mdot`` values.

    Returns
    -------
    numpy.ndarray
        Mass-loss rate in ``M_sun yr^-1`` sampled at ``time_before_sn``.
    """
    name = _canonical_model_name(model_name)
    t = np.asarray(time_before_sn, dtype=np.float64)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    if np.any(t < 0.0):
        raise ValueError("time_before_sn must be non-negative")

    if name in (
        "wind_exponential",
        "wind_bpl",
        "exponential_wind",
        "bpl_wind",
    ):
        mdot = np.full_like(t, float(kwargs["mdot"]), dtype=np.float64)

    elif name in ("boxwind_exponential", "boxwind_bpl"):
        t1 = float(kwargs["t1"])
        t2 = float(kwargs["t2"])
        if t2 < t1:
            raise ValueError("boxwind requires t2 >= t1")
        mdot = np.full_like(t, float(kwargs["mdot_2"]), dtype=np.float64)
        mdot[t <= t1] = float(kwargs["mdot_0"])
        mdot[(t > t1) & (t < t2)] = float(kwargs["mdot_1"])

    elif name in ("gausswind_exponential", "gausswind_bpl"):
        t_peak = float(kwargs["t_peak"])
        t_width = float(kwargs["t_width"])
        if t_width <= 0.0:
            raise ValueError("gausswind requires t_width > 0")
        mdot_baseline = float(kwargs["mdot_baseline"])
        mdot_peak = float(kwargs["mdot_peak"])
        profile = np.exp(-0.5 * ((t - t_peak) / t_width) ** 2)
        mdot = mdot_baseline + (mdot_peak - mdot_baseline) * profile

    elif name in (
        "triple_powerlaw_wind_bpl",
        "triple_powerlaw_wind_exponential",
        "exponential_triple_powerlaw_wind",
        "bpl_triple_powerlaw_wind",
        "smooth_triple_powerlaw_wind_bpl",
        "smooth_triple_powerlaw_wind_exponential",
    ):
        if np.any(t <= 0.0):
            raise ValueError("triple-power-law wind histories require positive times")
        t_break1 = float(kwargs["t_break1"])
        t_break2 = float(kwargs["t_break2"])
        if t_break1 <= 0.0 or t_break2 <= t_break1:
            raise ValueError("triple-power-law wind requires 0 < t_break1 < t_break2")
        mdot_0 = float(kwargs["mdot_0"])
        alpha1 = float(kwargs["alpha1"])
        alpha2 = float(kwargs["alpha2"])
        alpha3 = float(kwargs["alpha3"])
        mdot_break1 = mdot_0 * (t_break1 / 1.0) ** alpha1
        mdot_break2 = mdot_break1 * (t_break2 / t_break1) ** alpha2

        if name.startswith("smooth_triple_powerlaw_wind_"):
            smooth_factor = float(kwargs.get("smooth_factor", 0.2))
            smooth_width1 = smooth_factor * np.log10(t_break1) if t_break1 > 1.0 else smooth_factor
            smooth_width2 = smooth_factor * np.log10(t_break2) if t_break2 > 1.0 else smooth_factor
            transition1 = 0.5 * (1.0 + np.tanh((np.log10(t) - np.log10(t_break1)) / smooth_width1))
            transition2 = 0.5 * (1.0 + np.tanh((np.log10(t) - np.log10(t_break2)) / smooth_width2))
            mdot = (
                (1.0 - transition1) * mdot_0 * (t / 1.0) ** alpha1
                + transition1 * (1.0 - transition2) * mdot_break1 * (t / t_break1) ** alpha2
                + transition2 * mdot_break2 * (t / t_break2) ** alpha3
            )
        else:
            mdot = np.empty_like(t, dtype=np.float64)
            mask1 = t < t_break1
            mask2 = (t >= t_break1) & (t < t_break2)
            mask3 = t >= t_break2
            mdot[mask1] = mdot_0 * (t[mask1] / 1.0) ** alpha1
            mdot[mask2] = mdot_break1 * (t[mask2] / t_break1) ** alpha2
            mdot[mask3] = mdot_break2 * (t[mask3] / t_break2) ** alpha3

    else:
        raise ValueError(f"mass_loss_history does not support model '{model_name}'")

    return float(mdot[0]) if scalar else mdot


def csm_density_profile(model_name, radius, time_since_explosion=0.0, **kwargs):
    """
    Return the CSM density profile for wind-history models.

    The density is computed from the same wind convention used by the solver,

    ``rho(r, t) = mdot(|r / vwind - t|) / (4 pi r^2 vwind)``.

    Parameters
    ----------
    model_name : str
        Wind-history model name, e.g. ``"boxwind_bpl"``.
    radius : array-like
        Radius in cm.
    time_since_explosion : float, optional
        Epoch at which to evaluate the profile, in days. Default 0 evaluates
        the pre-SN profile at explosion.
    **kwargs
        Model parameters. Must include ``vwind`` in km/s and the relevant
        mass-loss history parameters.

    Returns
    -------
    numpy.ndarray
        CSM density in ``g cm^-3``.
    """
    r = np.asarray(radius, dtype=np.float64)
    scalar = r.ndim == 0
    r = np.atleast_1d(r)
    if np.any(r <= 0.0):
        raise ValueError("radius must be positive")
    vwind_cgs = float(kwargs["vwind"]) * 1e5
    if vwind_cgs <= 0.0:
        raise ValueError("vwind must be positive")

    t_wind_yr = np.abs(r / vwind_cgs - float(time_since_explosion) * DAY) / YEAR
    mdot_msun_yr = mass_loss_history(model_name, t_wind_yr, **kwargs)
    mdot_cgs = np.asarray(mdot_msun_yr, dtype=np.float64) * solar_mass_per_yr_to_gram_per_sec
    rho = mdot_cgs / (4.0 * np.pi * r**2 * vwind_cgs)
    return float(rho[0]) if scalar else rho


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_mass_loss_history(
    model_name,
    time_before_sn,
    ax=None,
    label=None,
    color=None,
    xlabel="Look-back time before explosion (yr)",
    ylabel=r"$\dot{M}$ ($M_\odot\,{\rm yr}^{-1}$)",
    title=None,
    **kwargs,
):
    """
    Plot the wind mass-loss history Ṁ(t) used by a redback-csm wind model.

    This is a thin plotting wrapper around :func:`mass_loss_history`, intended
    to help sanity-check what a model's parameters mean before fitting.

    Parameters
    ----------
    model_name : str
        Wind-history model name, e.g. ``"boxwind_bpl"``.
    time_before_sn : array-like
        Look-back time before explosion in years, over which to evaluate and
        plot Ṁ(t).
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, a new figure is created.
    label : str or None, optional
        Line label for the legend.
    color : str or None, optional
        Line colour.
    xlabel, ylabel : str, optional
        Axis labels.
    title : str or None, optional
        Axes title.
    **kwargs
        Model parameters, forwarded to :func:`mass_loss_history` (e.g. for
        ``boxwind_bpl``: ``t1``, ``t2``, ``mdot_0``, ``mdot_1``, ``mdot_2``).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    mdot : numpy.ndarray
        Mass-loss rate in ``M_sun yr^-1`` sampled at ``time_before_sn``.

    Examples
    --------
    >>> import numpy as np
    >>> from redback_csm.explore import plot_mass_loss_history
    >>> t = np.geomspace(1e-2, 20.0, 500)
    >>> fig, mdot = plot_mass_loss_history(
    ...     "boxwind_bpl", t,
    ...     t1=1.0, t2=3.0, mdot_0=1e-4, mdot_1=1e-2, mdot_2=2e-4,
    ... )
    """
    t = np.asarray(time_before_sn, dtype=np.float64)
    mdot = mass_loss_history(model_name, t, **kwargs)
    mdot = np.atleast_1d(mdot)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.get_figure()

    ax.plot(t, mdot, color=color, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.legend()
    fig.tight_layout()

    return fig, mdot


def plot_csm_density_profile(
    model_name,
    radius,
    time_since_explosion=0.0,
    ax=None,
    label=None,
    color=None,
    xlabel="Radius (cm)",
    ylabel=r"CSM density (g cm$^{-3}$)",
    title=None,
    **kwargs,
):
    """
    Plot the CSM density profile ρ(r) used by a redback-csm wind model.

    This is a thin plotting wrapper around :func:`csm_density_profile`,
    intended to help sanity-check what a model's parameters mean before
    fitting, e.g. to see exactly what CSM structure ``mdot_0``, ``mdot_1``,
    ``mdot_2`` (for ``boxwind_*``) or other wind-history parameters produce,
    without running a light-curve fit.

    Parameters
    ----------
    model_name : str
        Wind-history model name, e.g. ``"boxwind_bpl"``.
    radius : array-like
        Radius in cm, over which to evaluate and plot ρ(r).
    time_since_explosion : float, optional
        Epoch at which to evaluate the profile, in days. Default 0 evaluates
        the pre-SN profile at explosion.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, a new figure is created.
    label : str or None, optional
        Line label for the legend.
    color : str or None, optional
        Line colour.
    xlabel, ylabel : str, optional
        Axis labels.
    title : str or None, optional
        Axes title.
    **kwargs
        Model parameters, forwarded to :func:`csm_density_profile` (must
        include ``vwind`` in km/s and the relevant mass-loss history
        parameters).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    rho : numpy.ndarray
        CSM density in ``g cm^-3`` sampled at ``radius``.

    Examples
    --------
    >>> import numpy as np
    >>> from redback_csm.explore import plot_csm_density_profile
    >>> r = np.geomspace(1e14, 1e18, 500)
    >>> fig, rho = plot_csm_density_profile(
    ...     "boxwind_bpl", r,
    ...     t1=1.0, t2=3.0, mdot_0=1e-4, mdot_1=1e-2, mdot_2=2e-4, vwind=100.0,
    ... )
    """
    r = np.asarray(radius, dtype=np.float64)
    rho = csm_density_profile(model_name, r, time_since_explosion=time_since_explosion, **kwargs)
    rho = np.atleast_1d(rho)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.get_figure()

    ax.plot(r, rho, color=color, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.legend()
    fig.tight_layout()

    return fig, rho


def csm_lightcurve_from_density(
    radius,
    density,
    t_ref,
    mexp,
    eexp,
    eff,
    sn_profile="bpl",
    delta=1.0,
    nn=12.0,
    kappa=None,
    mode="simple",
    n_rad_zones=40,
    efficiency_mode=0,
    ax=None,
    label=None,
    color=None,
    show_shock=False,
    xlabel="Time (days)",
    ylabel=r"Bolometric luminosity (erg s$^{-1}$)",
    title=None,
):
    """
    Compute and plot a CSM interaction light curve from an arbitrary density profile.

    The CSM density profile ρ(r) is interpolated onto the internal Fortran
    velocity grid using r = v × t_ref (the reference epoch at which the
    profile is evaluated).  This is the same convention used throughout
    redback-csm: t_ref is the time between CSM formation and the supernova.

    Parameters
    ----------
    radius : array-like
        Radial grid of the CSM density profile, in **cm**.
        Should span at least the region of interest (typically 1e13–1e17 cm).
    density : array-like
        Density at each radius, in **g cm⁻³**.
    t_ref : float
        Reference epoch at which the density profile is evaluated, in **days**.
        Sets the mapping r = v × t_ref between the velocity grid and radius.
        For a pre-SN wind, this is typically the time between the end of
        mass loss and the supernova.
    mexp : float
        Supernova ejecta mass in **M☉**.
    eexp : float
        Supernova explosion energy in **foe** (10⁵¹ erg).
    eff : float
        Shock-to-radiation efficiency (0–1).
    sn_profile : {'bpl', 'exponential'}
        Supernova ejecta density profile.
        ``'bpl'`` — broken power law (inner index ``delta``, outer ``nn``).
        ``'exponential'`` — exponential profile.
    delta : float, optional
        Inner power-law index for BPL ejecta (ignored for exponential).
        Default 1.0.
    nn : float, optional
        Outer power-law index for BPL ejecta (ignored for exponential).
        Default 12.0.
    kappa : float or None, optional
        Opacity in cm² g⁻¹ for photon diffusion.  If provided, the returned
        (and plotted) luminosity is the diffuse (observed) luminosity; otherwise
        it is the raw shock luminosity.  Common values: 0.34 (electron
        scattering), 0.1 (lower bound).
    mode : {'simple', 'transport'}, optional
        Runtime mode. ``'simple'`` uses the thin-shell luminosity/post-processed
        diffusion path. ``'transport'`` uses the radiation transport solver with
        the sampled CSM density grid. Default ``'simple'``.
    n_rad_zones : int, optional
        Number of radiation zones for ``mode='transport'``. Default 40.
    efficiency_mode : int, optional
        Fortran shock-efficiency mode. Default 0.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on.  If None, a new figure is created.
    label : str or None, optional
        Line label for the legend.
    color : str or None, optional
        Line colour.
    show_shock : bool, optional
        If True and ``kappa`` is provided, also plot the underlying shock
        luminosity as a dashed line.  Default False.
    xlabel, ylabel : str, optional
        Axis labels.
    title : str or None, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the light curve.
    lc : namedtuple
        Raw Fortran output with fields ``time`` (s), ``lbol``, ``lbol_shock``,
        ``lbol_diffuse`` (erg/s), ``rph`` (cm), ``temperature`` (K).

    Examples
    --------
    Plot a simple r⁻² wind profile interacting with a BPL supernova:

    >>> import numpy as np
    >>> from redback_csm.explore import csm_lightcurve_from_density
    >>> r = np.geomspace(1e13, 1e17, 500)
    >>> rho = 5e16 / r**2
    >>> fig, lc = csm_lightcurve_from_density(
    ...     radius=r, density=rho,
    ...     t_ref=365.0, mexp=10.0, eexp=1.0, eff=0.3, kappa=0.34,
    ... )

    Overlay multiple profiles on the same axes:

    >>> fig, ax = plt.subplots()
    >>> for A in [1e15, 5e15, 1e16]:
    ...     csm_lightcurve_from_density(
    ...         radius=r, density=A / r**2,
    ...         t_ref=365.0, mexp=10.0, eexp=1.0, eff=0.3,
    ...         ax=ax, label=f'A={A:.0e}',
    ...     )
    >>> ax.legend()
    """
    radius = np.asarray(radius, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)

    if sn_profile not in ("bpl", "exponential"):
        raise ValueError("sn_profile must be 'bpl' or 'exponential'")
    if radius.shape != density.shape:
        raise ValueError("radius and density must have the same shape")
    if not np.all(np.diff(radius) > 0):
        raise ValueError("radius must be strictly increasing")

    t_ref_sec = t_ref * DAY
    mexp_g = mexp * solar_mass
    eexp_erg = eexp * foe

    runtime_kwargs = dict(
        mode=mode,
        kappa=kappa,
        n_rad_zones=n_rad_zones,
        efficiency_mode=efficiency_mode,
        delta=delta,
        nn=nn,
        mexp=mexp,
        eexp=eexp,
    )
    mode, kappa = _configure_runtime_from_kwargs(runtime_kwargs, default_n_rad_zones=n_rad_zones)

    v_grid, density_interp = _radius_density_to_vgrid(radius, density, t_ref_sec)

    lc = _run_fortran(
        v_grid,
        density_interp,
        t_ref_sec,
        sn_profile,
        delta,
        nn,
        mexp_g,
        eexp_erg,
        eff,
        kappa,
    )

    # --- Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.get_figure()

    t_days = lc.time / DAY
    mask = lc.lbol > 0
    ax.plot(t_days[mask], lc.lbol[mask], color=color, label=label)

    if show_shock and kappa is not None and lc.lbol_shock is not None:
        ax.plot(
            t_days[mask],
            lc.lbol_shock[mask],
            color=color,
            linestyle="--",
            alpha=0.5,
            label=(label + " (shock)") if label else "shock",
        )

    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if title is not None:
        ax.set_title(title)
    if label is not None:
        ax.legend()
    fig.tight_layout()

    return fig, lc
