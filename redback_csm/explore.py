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

from redback_csm.core import _get_csm, _configure_runtime_from_kwargs, DAY, foe, solar_mass

__all__ = ["csm_lightcurve_from_density"]

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
