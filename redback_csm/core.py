"""
core.py — Fortran CSM model wrappers and dispatch registry.

All _get_lc_* functions are direct wrappers around the compiled Fortran extension.
The _DISPATCH dict and _call_csm() helper are used by models.py to route model
name strings to the correct underlying function.
"""

csm = None  # Lazy import — loaded on first use by _get_csm()


def _get_csm():
    """Return the compiled Fortran extension, importing it lazily on first call."""
    global csm
    if csm is None:
        import importlib.util
        import glob
        import os

        # The .so lives next to this file inside the redback_csm package directory
        _pkg_dir = os.path.dirname(__file__)
        _matches = glob.glob(os.path.join(_pkg_dir, "csm*.so")) + glob.glob(
            os.path.join(_pkg_dir, "csm*.pyd")
        )
        if not _matches:
            raise ImportError(
                "redback_csm Fortran extension not found. "
                "Run `bash setup_fortran.sh && pip install -e .` to compile."
            )
        _spec = importlib.util.spec_from_file_location("csm", _matches[0])
        _csm_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_csm_module)
        csm = _csm_module
    return csm


import numpy as np
from collections import namedtuple
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Unit conversion constants (internal use for Fortran interface)
foe = 1e51  # 1 foe in erg
solar_mass = 1.989e33  # 1 solar mass in grams
solar_mass_per_yr_to_gram_per_sec = solar_mass / (365.25 * 24 * 3600)
YEAR = 365.25 * 24 * 3600  # seconds in a year
DAY = 86400  # seconds in a day
YEAR_DAYS = YEAR / DAY  # 365.25 days in a year


def _get_lc_wind_exponential(mdot, vwind, mexp, eexp, eff=None, mode='simple', 
                             n_rad_zones=40, hydrogen_fraction=0.7, **kwargs):
    """
    Calculate the light curve for a fixed mass-loss wind with an exponential profile explosion.
    i.e., windy CSM with a supernova.

    :param mdot: Mass loss rate in M☉/yr
    :param vwind: Wind velocity in km/s
    :param mexp: Explosion mass in M☉
    :param eexp: Explosion energy in foe
    :param eff: Efficiency factor (0-1). 
                For simple mode: fraction of kinetic energy that radiates
                For hybrid mode: ignored (efficiency emerges from radiation physics)
    :param mode: 'simple' (thin shell only) or 'hybrid' (thin shell + radiation diffusion)
    :param n_rad_zones: Number of radiation grid zones (for hybrid mode only, default 40)
    :param hydrogen_fraction: Hydrogen mass fraction (for hybrid mode only, default 0.7)
    :param kappa: (optional) Opacity in cm²/g for photon diffusion calculation.
                  Common values: 0.34 (electron scattering), 0.1 (lower bound)
    :return: Named tuple with fields:
             - time: Time array in seconds
             - lbol: Bolometric luminosity (photospheric for hybrid, shock×eff for simple)
             - lbol_shock: Shock luminosity (instantaneous kinetic energy deposition)
             - lbol_diffuse: Diffuse luminosity (photospheric for hybrid, None otherwise)
             - rph: Photospheric radius
             - temperature: Temperature
             - vshell: Shell velocity
             - shell_mass: Shell mass

    Example::

        # Simple mode (fast) - requires efficiency parameter
        lc = _get_lc_wind_exponential(mdot=1e-3, vwind=100, mexp=10.0, eexp=1.0, eff=0.5)
        plt.plot(lc.time, lc.lbol)  # Plots eff × shock luminosity

        # Hybrid mode (accurate) - efficiency emerges naturally from diffusion
        lc = _get_lc_wind_exponential(mdot=1e-3, vwind=100, mexp=10.0, eexp=1.0,
                                      mode='hybrid', kappa=0.34, n_rad_zones=40)
        plt.plot(lc.time, lc.lbol)         # Plots photospheric luminosity
        plt.plot(lc.time, lc.lbol_shock)   # Plots shock luminosity (always higher early on)
    """
    kappa = kwargs.get("kappa", None)

    mdot = mdot * solar_mass_per_yr_to_gram_per_sec
    vwind = vwind * 1e5  # Convert km/s to cm/s
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert erg to ergs

    mdot = np.array([mdot], dtype=np.float64)
    tgrid = np.array([1.0], dtype=np.float64)
    
    # Set run mode and handle efficiency
    if mode == 'simple':
        _get_csm().lc_mod.set_run_mode(1)
        if eff is None:
            raise ValueError("Simple mode requires 'eff' parameter (0-1)")
    elif mode == 'hybrid':
        _get_csm().lc_mod.set_run_mode(2)
        
        # Hybrid mode: efficiency emerges from physics, ignore eff parameter
        if eff is not None and eff != 1.0:
            import warnings
            warnings.warn(
                f"Hybrid mode: ignoring eff={eff}. Efficiency emerges from radiation physics. "
                "Setting internal eff=1.0 to deposit full kinetic luminosity.",
                UserWarning
            )
        eff = 1.0  # Always use full kinetic luminosity internally
        
        # Set hybrid parameters
        if kappa is None:
            kappa = 0.34  # Default opacity for hybrid mode
        
        _get_csm().lc_mod.set_hybrid_parameters(
            n_zones=n_rad_zones,
            h_frac=hydrogen_fraction,
            kappa_val=kappa
        )
    else:
        raise ValueError(f"mode must be 'simple' or 'hybrid', got {mode}")

    # Call Fortran with or without kappa
    if kappa is not None:
        _get_csm().lc_mod.lightcurve_wind_exponential(
            mdot, tgrid, vwind, mexp, eexp, eff, kappa
        )
    else:
        _get_csm().lc_mod.lightcurve_wind_exponential(
            mdot, tgrid, vwind, mexp, eexp, eff
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_fs = _get_csm().lc_mod.lfs.copy()
    lbol_rs = _get_csm().lc_mod.lrs.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()
    
    # larray contains different things depending on mode:
    # - Simple mode: larray = eff * (lfs + lrs) 
    # - Hybrid mode: larray = surface luminosity from grid
    larray_raw = _get_csm().lc_mod.larray.copy()
    
    # Shock luminosity is always lfs + lrs (total kinetic luminosity deposited)
    lbol_shock_total = lbol_fs + lbol_rs

    # Get diffuse/observed luminosity
    if mode == 'hybrid':
        # Hybrid: observed luminosity comes from radiation solver (stored in larray and ldiff)
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse  # Main output is surface luminosity
        lbol_shock = lbol_shock_total  # Store actual shock luminosity for comparison
    elif kappa is not None:
        # Simple with kappa: post-processed diffusion
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
        lbol_shock = larray_raw  # This is eff * (lfs + lrs)
    else:
        # Simple without kappa: no diffusion
        lbol_diffuse = None
        lbol = larray_raw  # This is eff * (lfs + lrs)
        lbol_shock = larray_raw

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_fs",
            "lbol_rs",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_fs = lbol_fs
    outs.lbol_rs = lbol_rs
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_wind_bpl(mdot, vwind, delta, nn, mexp, eexp, eff, **kwargs):
    """
    Calculate the light curve for a fixed mass-loss wind with a broken power law profile explosion.
    i.e., windy CSM with a broken power law supernova.

    :param mdot: Mass loss rate in M☉/yr
    :param vwind: Wind velocity in km/s
    :param delta: Delta parameter for broken power law
    :param nn: Outer ejecta power law index
    :param mexp: Explosion mass in M☉
    :param eexp: Explosion energy in foe
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kappa: (optional) Opacity in cm²/g for photon diffusion calculation.
                  Common values: 0.34 (electron scattering), 0.1 (lower bound)
    :return: Named tuple with fields:
             - time: Time array in seconds
             - lbol: Bolometric luminosity (diffuse if kappa provided, else shock)
             - lbol_shock: Shock luminosity (instantaneous energy deposition)
             - lbol_diffuse: Diffuse luminosity (None if kappa not provided)
             - rph: Photospheric radius
             - temperature: Temperature
             - vshell: Shell velocity
             - shell_mass: Shell mass

    Example::

        # Without diffusion - lbol = lbol_shock
        lc = _get_lc_wind_bpl(mdot=1e-5, vwind=500, delta=0.5, nn=12, mexp=2.0, eexp=1.0, eff=0.5)
        plt.plot(lc.time, lc.lbol)  # Plots shock luminosity

        # With diffusion - lbol = lbol_diffuse (observed luminosity)
        lc = _get_lc_wind_bpl(mdot=1e-5, vwind=500, delta=0.5, nn=12, mexp=2.0, eexp=1.0, eff=0.5, kappa=0.34)
        plt.plot(lc.time, lc.lbol)         # Plots diffuse (observed) luminosity
        plt.plot(lc.time, lc.lbol_shock)   # Plots shock luminosity
        plt.plot(lc.time, lc.lbol_diffuse) # Same as lc.lbol when kappa provided
    """
    kappa = kwargs.get("kappa", None)

    mdot = mdot * solar_mass_per_yr_to_gram_per_sec
    vwind = vwind * 1e5  # Convert km/s to cm/s
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert foe to ergs

    mdot = np.array([mdot], dtype=np.float64)
    tgrid = np.array([1.0], dtype=np.float64)

    # Call Fortran with or without kappa
    if kappa is not None:
        _get_csm().lc_mod.lightcurve_wind_bpl(
            mdot, tgrid, vwind, delta, nn, mexp, eexp, eff, kappa
        )
    else:
        _get_csm().lc_mod.lightcurve_wind_bpl(
            mdot, tgrid, vwind, delta, nn, mexp, eexp, eff
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    lbol_fs = _get_csm().lc_mod.lfs.copy()
    lbol_rs = _get_csm().lc_mod.lrs.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    # Get diffuse luminosity if kappa was provided
    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse  # Main output is diffuse when kappa provided
    else:
        lbol_diffuse = None
        lbol = lbol_shock  # Main output is shock when no kappa

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_fs",
            "lbol_rs",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_fs = lbol_fs
    outs.lbol_rs = lbol_rs
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_exponential_wind(mexp, eexp, mdot, vwind, eff, **kwargs):
    """
    Calculate the light curve for an exponential explosion with wind afterwards.

    :param mexp: Explosion mass in M☉
    :param eexp: Explosion energy in foe
    :param mdot: Mass loss rate in M☉/yr
    :param vwind: Wind velocity in km/s
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kappa: (optional) Opacity in cm²/g for photon diffusion calculation.
                  Common values: 0.34 (electron scattering), 0.1 (lower bound)
    :return: Named tuple with fields:
             - time: Time array in seconds
             - lbol: Bolometric luminosity (diffuse if kappa provided, else shock)
             - lbol_shock: Shock luminosity (instantaneous energy deposition)
             - lbol_diffuse: Diffuse luminosity (None if kappa not provided)
             - rph: Photospheric radius
             - temperature: Temperature
             - vshell: Shell velocity
             - shell_mass: Shell mass

    Example::

        # Without diffusion - lbol = lbol_shock
        lc = _get_lc_exponential_wind(mexp=10.0, eexp=1.0, mdot=1e-3, vwind=100, eff=0.5)
        plt.plot(lc.time, lc.lbol)  # Plots shock luminosity

        # With diffusion - lbol = lbol_diffuse (observed luminosity)
        lc = _get_lc_exponential_wind(mexp=10.0, eexp=1.0, mdot=1e-3, vwind=100, eff=0.5, kappa=0.34)
        plt.plot(lc.time, lc.lbol)         # Plots diffuse (observed) luminosity
        plt.plot(lc.time, lc.lbol_shock)   # Plots shock luminosity
        plt.plot(lc.time, lc.lbol_diffuse) # Same as lc.lbol when kappa provided
    """
    kappa = kwargs.get("kappa", None)

    mdot = mdot * solar_mass_per_yr_to_gram_per_sec
    vwind = vwind * 1e5  # Convert km/s to cm/s
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert foe to ergs

    mdot = np.array([mdot], dtype=np.float64)
    tgrid = np.array([1.0], dtype=np.float64)

    # Call Fortran with or without kappa
    if kappa is not None:
        _get_csm().lc_mod.lightcurve_exponential_wind(
            mexp, eexp, mdot, tgrid, vwind, eff, kappa
        )
    else:
        _get_csm().lc_mod.lightcurve_exponential_wind(
            mexp, eexp, mdot, tgrid, vwind, eff
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    lbol_fs = _get_csm().lc_mod.lfs.copy()
    lbol_rs = _get_csm().lc_mod.lrs.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    # Get diffuse luminosity if kappa was provided
    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse  # Main output is diffuse when kappa provided
    else:
        lbol_diffuse = None
        lbol = lbol_shock  # Main output is shock when no kappa

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_fs",
            "lbol_rs",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_fs = lbol_fs
    outs.lbol_rs = lbol_rs
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_bpl_wind(delta, nn, mexp, eexp, mdot, vwind, eff, **kwargs):
    """
    Calculate the light curve for a broken power law explosion with wind interaction afterwards.

    :param delta: Delta parameter for broken power law
    :param nn: Power law index
    :param mexp: Explosion mass in M☉
    :param eexp: Explosion energy in foe
    :param mdot: Mass loss rate in M☉/yr
    :param vwind: Wind velocity in km/s
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kappa: (optional) Opacity in cm²/g for photon diffusion calculation.
                  Common values: 0.34 (electron scattering), 0.1 (lower bound)
    :return: Named tuple with fields:
             - time: Time array in seconds
             - lbol: Bolometric luminosity (diffuse if kappa provided, else shock)
             - lbol_shock: Shock luminosity (instantaneous energy deposition)
             - lbol_diffuse: Diffuse luminosity (None if kappa not provided)
             - rph: Photospheric radius
             - temperature: Temperature
             - vshell: Shell velocity
             - shell_mass: Shell mass

    Example::

        # Without diffusion - lbol = lbol_shock
        lc = _get_lc_bpl_wind(delta=0.5, nn=12, mexp=2.0, eexp=1.0, mdot=1e-3, vwind=100, eff=0.5)
        plt.plot(lc.time, lc.lbol)  # Plots shock luminosity

        # With diffusion - lbol = lbol_diffuse (observed luminosity)
        lc = _get_lc_bpl_wind(delta=0.5, nn=12, mexp=2.0, eexp=1.0, mdot=1e-3, vwind=100, eff=0.5, kappa=0.34)
        plt.plot(lc.time, lc.lbol)         # Plots diffuse (observed) luminosity
        plt.plot(lc.time, lc.lbol_shock)   # Plots shock luminosity
        plt.plot(lc.time, lc.lbol_diffuse) # Same as lc.lbol when kappa provided
    """
    kappa = kwargs.get("kappa", None)

    mdot = mdot * solar_mass_per_yr_to_gram_per_sec
    vwind = vwind * 1e5  # Convert km/s to cm/s
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert foe to ergs

    tgrid = np.array([1.0], dtype=np.float64)
    mdot = np.array([mdot], dtype=np.float64)

    # Call Fortran with or without kappa
    if kappa is not None:
        _get_csm().lc_mod.lightcurve_bpl_wind(
            delta, nn, mexp, eexp, mdot, tgrid, vwind, eff, kappa
        )
    else:
        _get_csm().lc_mod.lightcurve_bpl_wind(
            delta, nn, mexp, eexp, mdot, tgrid, vwind, eff
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    # Get diffuse luminosity if kappa was provided
    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse  # Main output is diffuse when kappa provided
    else:
        lbol_diffuse = None
        lbol = lbol_shock  # Main output is shock when no kappa

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_exponential_exponential(
    mexp, eexp, mexp_out, eexp_out, interval, eff, **kwargs
):
    """
    Calculate the light curve for double exponential explosion interaction.

    :param mexp: Inner explosion mass in M☉
    :param eexp: Inner explosion energy in foe
    :param mexp_out: Outer explosion mass in M☉
    :param eexp_out: Outer explosion energy in foe
    :param interval: Time interval between explosions in days
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
    :param kappa: (optional) Opacity in cm²/g for photon diffusion calculation.
                  Common values: 0.34 (electron scattering), 0.1 (lower bound)
    :return: Named tuple with fields:
             - time: Time array in seconds
             - lbol: Bolometric luminosity (diffuse if kappa provided, else shock)
             - lbol_shock: Shock luminosity (instantaneous energy deposition)
             - lbol_diffuse: Diffuse luminosity (None if kappa not provided)
             - rph: Photospheric radius
             - temperature: Temperature
             - vshell: Shell velocity
             - shell_mass: Shell mass

    Example::

        # Without diffusion - lbol = lbol_shock
        lc = _get_lc_exponential_exponential(mexp=0.1, eexp=0.01, mexp_out=10.0, eexp_out=1.0, interval=100, eff=0.5)
        plt.plot(lc.time, lc.lbol)  # Plots shock luminosity

        # With diffusion - lbol = lbol_diffuse (observed luminosity)
        lc = _get_lc_exponential_exponential(mexp=0.1, eexp=0.01, mexp_out=10.0, eexp_out=1.0, interval=100, eff=0.5, kappa=0.34)
        plt.plot(lc.time, lc.lbol)         # Plots diffuse (observed) luminosity
        plt.plot(lc.time, lc.lbol_shock)   # Plots shock luminosity
        plt.plot(lc.time, lc.lbol_diffuse) # Same as lc.lbol when kappa provided
    """
    kappa = kwargs.get("kappa", None)

    # Convert to CGS
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert foe to ergs
    mexp_out = mexp_out * solar_mass  # Convert solar masses to grams
    eexp_out = eexp_out * foe  # Convert foe to ergs
    interval_sec = interval * DAY  # Convert days to seconds

    # Call Fortran with or without kappa
    if kappa is not None:
        _get_csm().lc_mod.lightcurve_exponential_exponential(
            mexp, eexp, mexp_out, eexp_out, interval_sec, eff, kappa
        )
    else:
        _get_csm().lc_mod.lightcurve_exponential_exponential(
            mexp, eexp, mexp_out, eexp_out, interval_sec, eff
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    # Get diffuse luminosity if kappa was provided
    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse  # Main output is diffuse when kappa provided
    else:
        lbol_diffuse = None
        lbol = lbol_shock  # Main output is shock when no kappa

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_exponential_bpl(
    mexp, eexp, delta_out, nn_out, mexp_out, eexp_out, interval, eff, **kwargs
):
    """
    Calculate the light curve for an exponential explosion (CSM) interacting with a broken power law explosion (SN).

    :param mexp: Inner explosion mass in M☉
    :param eexp: Inner explosion energy in foe
    :param delta_out: Delta parameter for outer broken power law
    :param nn_out: Outer power law index
    :param mexp_out: Outer explosion mass in M☉
    :param eexp_out: Outer explosion energy in foe
    :param interval: Time interval between explosions in days
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
        - time_ref: Reference time for CSM setup in days (default: interval), i.e., \rho_csm(t = time_ref)
        - kappa: (optional) Opacity in cm²/g for photon diffusion calculation.
                 Common values: 0.34 (electron scattering), 0.1 (lower bound)
    :return: Named tuple with fields:
             - time: Time array in seconds
             - lbol: Bolometric luminosity (diffuse if kappa provided, else shock)
             - lbol_shock: Shock luminosity (instantaneous energy deposition)
             - lbol_diffuse: Diffuse luminosity (None if kappa not provided)
             - rph: Photospheric radius
             - temperature: Temperature
             - vshell: Shell velocity
             - shell_mass: Shell mass

    Example::

        # Without diffusion - lbol = lbol_shock
        lc = _get_lc_exponential_bpl(mexp=0.05, eexp=0.01, delta_out=0.5, nn_out=12,
                                     mexp_out=2.0, eexp_out=1.0, interval=365, eff=0.5)
        plt.plot(lc.time, lc.lbol)  # Plots shock luminosity

        # With diffusion - lbol = lbol_diffuse (observed luminosity)
        lc = _get_lc_exponential_bpl(mexp=0.05, eexp=0.01, delta_out=0.5, nn_out=12,
                                     mexp_out=2.0, eexp_out=1.0, interval=365, eff=0.5, kappa=0.34)
        plt.plot(lc.time, lc.lbol)         # Plots diffuse (observed) luminosity
        plt.plot(lc.time, lc.lbol_shock)   # Plots shock luminosity
        plt.plot(lc.time, lc.lbol_diffuse) # Same as lc.lbol when kappa provided
    """
    kappa = kwargs.get("kappa", None)
    time_ref_days = kwargs.get("time_ref", interval)

    csmclass = SequentialCSMModel(verbose=False, time_ref=time_ref_days)
    v_grid, _, density, _ = csmclass.create_exponential_profile(mexp, eexp)

    # Convert to CGS
    mexp_out = mexp_out * solar_mass  # Convert solar masses to grams
    eexp_out = eexp_out * foe  # Convert foe to ergs
    interval_sec = interval * DAY  # Convert days to seconds
    time_ref_sec = time_ref_days * DAY  # Convert days to seconds

    # Call Fortran with or without kappa
    if kappa is not None:
        _get_csm().lc_mod.lightcurve_explosion_bpl(
            density,
            v_grid,
            time_ref_sec,
            delta_out,
            nn_out,
            mexp_out,
            eexp_out,
            interval_sec,
            eff,
            kappa,
        )
    else:
        _get_csm().lc_mod.lightcurve_explosion_bpl(
            density,
            v_grid,
            time_ref_sec,
            delta_out,
            nn_out,
            mexp_out,
            eexp_out,
            interval_sec,
            eff,
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    # Get diffuse luminosity if kappa was provided
    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse  # Main output is diffuse when kappa provided
    else:
        lbol_diffuse = None
        lbol = lbol_shock  # Main output is shock when no kappa

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_bpl_bpl(
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
    **kwargs,
):
    """
    Calculate the light curve for double broken power law explosion interaction.
    First set of parameters correspond to the first explosion (CSM), second set to the supernova.

    :param delta: Inner delta parameter for broken power law
    :param nn: Inner power law index
    :param mexp: Inner explosion mass in M☉
    :param eexp: Inner explosion energy in foe
    :param delta_out: Outer delta parameter for broken power law
    :param nn_out: Outer power law index
    :param mexp_out: Outer explosion mass in M☉
    :param eexp_out: Outer explosion energy in foe
    :param interval: Time interval between explosions in days
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
    :param kappa: (optional) Opacity in cm²/g for photon diffusion calculation.
                  Common values: 0.34 (electron scattering), 0.1 (lower bound)
    :return: Named tuple with fields:
             - time: Time array in seconds
             - lbol: Bolometric luminosity (diffuse if kappa provided, else shock)
             - lbol_shock: Shock luminosity (instantaneous energy deposition)
             - lbol_diffuse: Diffuse luminosity (None if kappa not provided)
             - rph: Photospheric radius
             - temperature: Temperature
             - vshell: Shell velocity
             - shell_mass: Shell mass

    Example::

        # Without diffusion - lbol = lbol_shock
        lc = _get_lc_bpl_bpl(delta=0.5, nn=10, mexp=0.1, eexp=0.01, delta_out=0.5, nn_out=12,
                             mexp_out=2.0, eexp_out=1.0, interval=100, eff=0.5)
        plt.plot(lc.time, lc.lbol)  # Plots shock luminosity

        # With diffusion - lbol = lbol_diffuse (observed luminosity)
        lc = _get_lc_bpl_bpl(delta=0.5, nn=10, mexp=0.1, eexp=0.01, delta_out=0.5, nn_out=12,
                             mexp_out=2.0, eexp_out=1.0, interval=100, eff=0.5, kappa=0.34)
        plt.plot(lc.time, lc.lbol)         # Plots diffuse (observed) luminosity
        plt.plot(lc.time, lc.lbol_shock)   # Plots shock luminosity
        plt.plot(lc.time, lc.lbol_diffuse) # Same as lc.lbol when kappa provided
    """
    kappa = kwargs.get("kappa", None)

    # Convert to CGS
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert foe to ergs
    mexp_out = mexp_out * solar_mass  # Convert solar masses to grams
    eexp_out = eexp_out * foe  # Convert foe to ergs
    interval_sec = interval * DAY  # Convert days to seconds

    # Call Fortran with or without kappa
    if kappa is not None:
        _get_csm().lc_mod.lightcurve_bpl_bpl(
            delta,
            nn,
            mexp,
            eexp,
            delta_out,
            nn_out,
            mexp_out,
            eexp_out,
            interval_sec,
            eff,
            kappa,
        )
    else:
        _get_csm().lc_mod.lightcurve_bpl_bpl(
            delta,
            nn,
            mexp,
            eexp,
            delta_out,
            nn_out,
            mexp_out,
            eexp_out,
            interval_sec,
            eff,
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    # Get diffuse luminosity if kappa was provided
    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse  # Main output is diffuse when kappa provided
    else:
        lbol_diffuse = None
        lbol = lbol_shock  # Main output is shock when no kappa

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_bpl_exponential(
    delta, nn, mexp, eexp, mexp_out, eexp_out, interval, eff, **kwargs
):
    """
    Calculate the light curve for a broken power law CSM interacting with an exponential SN.

    Uses the new optimized lightcurve_bpl_exponential Fortran routine with analytical exponential formula.
    This is ~500x faster than the old method that created density grids for both components.

    :param delta: Inner power-law index for BPL CSM
    :param nn: Outer power-law index for BPL CSM
    :param mexp: BPL CSM mass in M☉
    :param eexp: BPL CSM energy in foe
    :param mexp_out: Exponential SN mass in M☉
    :param eexp_out: Exponential SN energy in foe
    :param interval: Time interval between CSM creation and SN explosion in days
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
    :param kappa: (optional) Opacity in cm²/g for photon diffusion calculation.
                  Common values: 0.34 (electron scattering), 0.1 (lower bound)
    :return: Named tuple with fields:
             - time: Time array in seconds
             - lbol: Bolometric luminosity (diffuse if kappa provided, else shock)
             - lbol_shock: Shock luminosity (instantaneous energy deposition)
             - lbol_diffuse: Diffuse luminosity (None if kappa not provided)
             - rph: Photospheric radius
             - temperature: Temperature
             - vshell: Shell velocity
             - shell_mass: Shell mass

    Example::

        # Without diffusion - lbol = lbol_shock
        lc = _get_lc_bpl_exponential(delta=0.5, nn=10, mexp=0.1, eexp=0.01,
                                     mexp_out=10.0, eexp_out=1.0, interval=100, eff=0.5)
        plt.plot(lc.time, lc.lbol)  # Plots shock luminosity

        # With diffusion - lbol = lbol_diffuse (observed luminosity)
        lc = _get_lc_bpl_exponential(delta=0.5, nn=10, mexp=0.1, eexp=0.01,
                                     mexp_out=10.0, eexp_out=1.0, interval=100, eff=0.5, kappa=0.34)
        plt.plot(lc.time, lc.lbol)         # Plots diffuse (observed) luminosity
        plt.plot(lc.time, lc.lbol_shock)   # Plots shock luminosity
        plt.plot(lc.time, lc.lbol_diffuse) # Same as lc.lbol when kappa provided
    """
    kappa = kwargs.get("kappa", None)

    # Convert to CGS
    mexp_cgs = mexp * solar_mass
    eexp_cgs = eexp * foe
    mexp_out_cgs = mexp_out * solar_mass
    eexp_out_cgs = eexp_out * foe
    interval_sec = interval * DAY

    # Use new optimized Fortran routine with analytical exponential
    # Naming: bpl_exponential = BPL CSM + exponential SN
    # Arguments: (delta, nn, M_csm, E_csm, M_sn, E_sn, interval, eff)
    # Call Fortran with or without kappa
    if kappa is not None:
        _get_csm().lc_mod.lightcurve_bpl_exponential(
            delta,
            nn,
            mexp_cgs,
            eexp_cgs,
            mexp_out_cgs,
            eexp_out_cgs,
            interval_sec,
            eff,
            kappa,
        )
    else:
        _get_csm().lc_mod.lightcurve_bpl_exponential(
            delta, nn, mexp_cgs, eexp_cgs, mexp_out_cgs, eexp_out_cgs, interval_sec, eff
        )

    # Extract results
    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    # Get diffuse luminosity if kappa was provided
    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse  # Main output is diffuse when kappa provided
    else:
        lbol_diffuse = None
        lbol = lbol_shock  # Main output is shock when no kappa

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_boxwind_exponential(
    t1, t2, mdot_0, mdot_1, mdot_2, vwind, mexp, eexp, eff, **kwargs
):
    """
    Calculate the light curve for a boxy (step function) mass-loss wind followed by an exponential explosion.

    :param t1: Start time of the box wind in years
    :param t2: End time of the box wind in years
    :param mdot_0: Mass loss rate before t1 in M☉/yr
    :param mdot_1: Mass loss rate during the box wind in M☉/yr
    :param mdot_2: Mass loss rate after t2 in M☉/yr (for box profile: mdot_2 = mdot_0)
    :param vwind: Wind velocity in km/s
    :param mexp: Explosion mass in M☉
    :param eexp: Explosion energy in foe
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kappa: (optional) Opacity in cm²/g for photon diffusion calculation
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)
    """
    kappa = kwargs.get("kappa", None)

    mdot = np.array([mdot_0, mdot_1, mdot_1, mdot_2])
    mdot = mdot * solar_mass_per_yr_to_gram_per_sec  # Convert to g/s
    tgrid = np.array([t1, t1, t2, t2], dtype=np.float64)
    tgrid = tgrid * 365.25 * 24 * 3600  # Convert years to seconds
    vwind = vwind * 1e5  # Convert km/s to cm/s
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert erg to ergs

    if kappa is not None:
        _get_csm().lc_mod.lightcurve_wind_exponential(
            mdot, tgrid, vwind, mexp, eexp, eff, kappa
        )
    else:
        _get_csm().lc_mod.lightcurve_wind_exponential(
            mdot, tgrid, vwind, mexp, eexp, eff
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_boxwind_bpl(
    t1, t2, mdot_0, mdot_1, mdot_2, vwind, delta, nn, mexp, eexp, eff, **kwargs
):
    """
    Calculate the light curve for a boxy (step function) mass-loss wind followed by a broken power law explosion.

    :param t1: Start time of the box wind in years
    :param t2: End time of the box wind in years
    :param mdot_0: Mass loss rate before t1 in M☉/yr
    :param mdot_1: Mass loss rate during the box wind in M☉/yr
    :param mdot_2: Mass loss rate after t2 in M☉/yr (for box profile: mdot_2 = mdot_0)
    :param vwind: Wind velocity in km/s
    :param delta: Delta parameter for broken power law
    :param nn: Power law index
    :param mexp: Explosion mass in M☉
    :param eexp: Explosion energy in foe
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kappa: (optional) Opacity in cm²/g for photon diffusion calculation
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)
    """
    kappa = kwargs.get("kappa", None)
    efficiency_mode = kwargs.get("efficiency_mode", None)

    mdot = np.array([mdot_0, mdot_1, mdot_1, mdot_2])
    mdot = mdot * solar_mass_per_yr_to_gram_per_sec  # Convert to g/s
    tgrid = np.array([t1, t1, t2, t2], dtype=np.float64)
    tgrid = tgrid * 365.25 * 24 * 3600  # Convert years to seconds
    vwind = vwind * 1e5  # Convert km/s to cm/s
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert foe to ergs

    if efficiency_mode is not None:
        _get_csm().lc_mod.set_efficiency_mode(int(efficiency_mode))

    if kappa is not None:
        _get_csm().lc_mod.lightcurve_wind_bpl(
            mdot, tgrid, vwind, delta, nn, mexp, eexp, eff, kappa
        )
    else:
        _get_csm().lc_mod.lightcurve_wind_bpl(
            mdot, tgrid, vwind, delta, nn, mexp, eexp, eff
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    lbol_fs = _get_csm().lc_mod.lfs.copy()
    lbol_rs = _get_csm().lc_mod.lrs.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_fs",
            "lbol_rs",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_fs = lbol_fs
    outs.lbol_rs = lbol_rs
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_gausswind_exponential(
    t_peak, t_width, mdot_baseline, mdot_peak, vwind, mexp, eexp, eff, **kwargs
):
    """
    A light curve for a Gaussian-like mass-loss profile followed by an exponential explosion.

    :param t_peak: Peak time of Gaussian wind in years
    :param t_width: Width (sigma) of Gaussian wind in years
    :param mdot_baseline: Baseline mass loss rate in M☉/yr
    :param mdot_peak: Peak mass loss rate in M☉/yr
    :param vwind: Wind velocity in km/s
    :param mexp: Explosion mass in M☉
    :param eexp: Explosion energy in foe
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
        - n_points: Number of points to discretize Gaussian profile (default: 50)
        - kappa: Opacity in cm²/g for photon diffusion calculation
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)

    Example::

        # Enhanced mass loss 5 years before explosion
        lc = _get_lc_gausswind_exponential(t_peak=5.0, t_width=1.0, mdot_baseline=1e-6,
                                           mdot_peak=1e-3, vwind=100, mexp=10.0, eexp=1.0, eff=0.5)
    """
    n_points = kwargs.get("n_points", 50)
    kappa = kwargs.get("kappa", None)

    # Create time grid for Gaussian profile
    t_start = t_peak - 4 * t_width  # Start 4 sigma before peak
    t_end = t_peak + 4 * t_width  # End 4 sigma after peak
    tgrid = np.linspace(t_start, t_end, n_points)

    # Create Gaussian mass loss profile
    gaussian_profile = np.exp(-0.5 * ((tgrid - t_peak) / t_width) ** 2)
    mdot = mdot_baseline + (mdot_peak - mdot_baseline) * gaussian_profile

    # Convert units
    mdot = mdot * solar_mass_per_yr_to_gram_per_sec  # Convert to g/s
    tgrid = tgrid * 365.25 * 24 * 3600  # Convert years to seconds
    vwind = vwind * 1e5  # Convert km/s to cm/s
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert foe to ergs

    if kappa is not None:
        _get_csm().lc_mod.lightcurve_wind_exponential(
            mdot, tgrid, vwind, mexp, eexp, eff, kappa
        )
    else:
        _get_csm().lc_mod.lightcurve_wind_exponential(
            mdot, tgrid, vwind, mexp, eexp, eff
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_gausswind_bpl(
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
    **kwargs,
):
    """
    A light curve for a Gaussian-like mass-loss profile followed by a broken power law explosion.

    :param t_peak: Peak time of Gaussian wind in years
    :param t_width: Width (sigma) of Gaussian wind in years
    :param mdot_baseline: Baseline mass loss rate in M☉/yr
    :param mdot_peak: Peak mass loss rate in M☉/yr
    :param vwind: Wind velocity in km/s
    :param delta: Delta parameter for broken power law
    :param nn: Power law index
    :param mexp: Explosion mass in M☉
    :param eexp: Explosion energy in foe
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
        - n_points: Number of points to discretize Gaussian profile (default: 50)
        - kappa: Opacity in cm²/g for photon diffusion calculation
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)
    """
    n_points = kwargs.get("n_points", 50)
    kappa = kwargs.get("kappa", None)

    # Create time grid for Gaussian profile
    t_start = t_peak - 4 * t_width  # Start 4 sigma before peak
    t_end = t_peak + 4 * t_width  # End 4 sigma after peak
    tgrid = np.linspace(t_start, t_end, n_points)

    # Create Gaussian mass loss profile
    gaussian_profile = np.exp(-0.5 * ((tgrid - t_peak) / t_width) ** 2)
    mdot = mdot_baseline + (mdot_peak - mdot_baseline) * gaussian_profile

    # Convert units
    mdot = mdot * solar_mass_per_yr_to_gram_per_sec  # Convert to g/s
    tgrid = tgrid * 365.25 * 24 * 3600  # Convert years to seconds
    vwind = vwind * 1e5  # Convert km/s to cm/s
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert foe to ergs

    if kappa is not None:
        _get_csm().lc_mod.lightcurve_wind_bpl(
            mdot, tgrid, vwind, delta, nn, mexp, eexp, eff, kappa
        )
    else:
        _get_csm().lc_mod.lightcurve_wind_bpl(
            mdot, tgrid, vwind, delta, nn, mexp, eexp, eff
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_triple_powerlaw_wind_bpl(
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
    **kwargs,
):
    """
    A light curve for a triple power law mass-loss profile followed by a broken power law explosion.

    Mass loss rate follows: mdot(t) = mdot_0 * (t/t_ref)^alpha_i
    where alpha_i changes at t_break1 and t_break2

    :param t_break1: First break time in years
    :param t_break2: Second break time in years (must be > t_break1)
    :param mdot_0: Reference mass loss rate at t=1 year in M☉/yr
    :param alpha1: Power law index for t < t_break1
    :param alpha2: Power law index for t_break1 < t < t_break2
    :param alpha3: Power law index for t > t_break2
    :param vwind: Wind velocity in km/s
    :param delta: Delta parameter for broken power law
    :param nn: Power law index
    :param mexp: Explosion mass in M☉
    :param eexp: Explosion energy in foe
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
        - n_points: Number of points to discretize profile (default: 50)
        - kappa: Opacity in cm²/g for photon diffusion calculation
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)
    """
    n_points = kwargs.get("n_points", 50)
    kappa = kwargs.get("kappa", None)

    # Create time grid spanning the three power law regimes
    t_start = 0.1  # Start at 0.1 years to avoid t=0
    t_end = max(t_break2 * 2, 10)  # End well beyond second break
    tgrid = np.logspace(np.log10(t_start), np.log10(t_end), n_points)

    # Create triple power law mass loss profile
    mdot = np.zeros_like(tgrid)

    # First regime: t < t_break1
    mask1 = tgrid < t_break1
    mdot[mask1] = mdot_0 * (tgrid[mask1] / 1.0) ** alpha1

    # Calculate normalization for continuity at t_break1
    mdot_break1 = mdot_0 * (t_break1 / 1.0) ** alpha1

    # Second regime: t_break1 <= t < t_break2
    mask2 = (tgrid >= t_break1) & (tgrid < t_break2)
    mdot[mask2] = mdot_break1 * (tgrid[mask2] / t_break1) ** alpha2

    # Calculate normalization for continuity at t_break2
    mdot_break2 = mdot_break1 * (t_break2 / t_break1) ** alpha2

    # Third regime: t >= t_break2
    mask3 = tgrid >= t_break2
    mdot[mask3] = mdot_break2 * (tgrid[mask3] / t_break2) ** alpha3

    # Convert units
    mdot = mdot * solar_mass_per_yr_to_gram_per_sec  # Convert to g/s
    tgrid = tgrid * 365.25 * 24 * 3600  # Convert years to seconds
    vwind = vwind * 1e5  # Convert km/s to cm/s
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert foe to ergs

    if kappa is not None:
        _get_csm().lc_mod.lightcurve_wind_bpl(
            mdot, tgrid, vwind, delta, nn, mexp, eexp, eff, kappa
        )
    else:
        _get_csm().lc_mod.lightcurve_wind_bpl(
            mdot, tgrid, vwind, delta, nn, mexp, eexp, eff
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_triple_powerlaw_wind_exponential(
    t_break1, t_break2, mdot_0, alpha1, alpha2, alpha3, vwind, mexp, eexp, eff, **kwargs
):
    """
    A light curve for a triple power law mass-loss profile followed by an exponential explosion.

    Mass loss rate follows: mdot(t) = mdot_0 * (t/t_ref)^alpha_i
    where alpha_i changes at t_break1 and t_break2

    :param t_break1: First break time in years
    :param t_break2: Second break time in years (must be > t_break1)
    :param mdot_0: Reference mass loss rate at t=1 year in M☉/yr
    :param alpha1: Power law index for t < t_break1
    :param alpha2: Power law index for t_break1 < t < t_break2
    :param alpha3: Power law index for t > t_break2
    :param vwind: Wind velocity in km/s
    :param mexp: Explosion mass in M☉
    :param eexp: Explosion energy in foe
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
        - n_points: Number of points to discretize profile (default: 50)
        - kappa: Opacity in cm²/g for photon diffusion calculation
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)
    """
    n_points = kwargs.get("n_points", 50)
    kappa = kwargs.get("kappa", None)

    # Create time grid spanning the three power law regimes
    t_start = 0.1  # Start at 0.1 years to avoid t=0
    t_end = max(t_break2 * 2, 10)  # End well beyond second break
    tgrid = np.logspace(np.log10(t_start), np.log10(t_end), n_points)

    # Create triple power law mass loss profile
    mdot = np.zeros_like(tgrid)

    # First regime: t < t_break1
    mask1 = tgrid < t_break1
    mdot[mask1] = mdot_0 * (tgrid[mask1] / 1.0) ** alpha1

    # Calculate normalization for continuity at t_break1
    mdot_break1 = mdot_0 * (t_break1 / 1.0) ** alpha1

    # Second regime: t_break1 <= t < t_break2
    mask2 = (tgrid >= t_break1) & (tgrid < t_break2)
    mdot[mask2] = mdot_break1 * (tgrid[mask2] / t_break1) ** alpha2

    # Calculate normalization for continuity at t_break2
    mdot_break2 = mdot_break1 * (t_break2 / t_break1) ** alpha2

    # Third regime: t >= t_break2
    mask3 = tgrid >= t_break2
    mdot[mask3] = mdot_break2 * (tgrid[mask3] / t_break2) ** alpha3

    # Convert units
    mdot = mdot * solar_mass_per_yr_to_gram_per_sec  # Convert to g/s
    tgrid = tgrid * 365.25 * 24 * 3600  # Convert years to seconds
    vwind = vwind * 1e5  # Convert km/s to cm/s
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert foe to ergs

    if kappa is not None:
        _get_csm().lc_mod.lightcurve_wind_exponential(
            mdot, tgrid, vwind, mexp, eexp, eff, kappa
        )
    else:
        _get_csm().lc_mod.lightcurve_wind_exponential(
            mdot, tgrid, vwind, mexp, eexp, eff
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_exponential_triple_powerlaw_wind(
    mexp, eexp, t_break1, t_break2, mdot_0, alpha1, alpha2, alpha3, vwind, eff, **kwargs
):
    """
    Calculate the light curve for an exponential explosion followed with triple power law wind interaction.

    :param mexp: Explosion mass in M☉
    :param eexp: Explosion energy in foe
    :param t_break1: First break time in years
    :param t_break2: Second break time in years (must be > t_break1)
    :param mdot_0: Reference mass loss rate at t=1 year in M☉/yr
    :param alpha1: Power law index for t < t_break1
    :param alpha2: Power law index for t_break1 < t < t_break2
    :param alpha3: Power law index for t > t_break2
    :param vwind: Wind velocity in km/s
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
        - n_points: Number of points to discretize profile (default: 50)
        - kappa: Opacity in cm²/g for photon diffusion calculation
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)
    """
    n_points = kwargs.get("n_points", 50)
    kappa = kwargs.get("kappa", None)

    # Create time grid spanning the three power law regimes
    t_start = 0.1  # Start at 0.1 years to avoid t=0
    t_end = max(t_break2 * 2, 10)  # End well beyond second break
    tgrid = np.logspace(np.log10(t_start), np.log10(t_end), n_points)

    # Create triple power law mass loss profile
    mdot = np.zeros_like(tgrid)

    # First regime: t < t_break1
    mask1 = tgrid < t_break1
    mdot[mask1] = mdot_0 * (tgrid[mask1] / 1.0) ** alpha1

    # Calculate normalization for continuity at t_break1
    mdot_break1 = mdot_0 * (t_break1 / 1.0) ** alpha1

    # Second regime: t_break1 <= t < t_break2
    mask2 = (tgrid >= t_break1) & (tgrid < t_break2)
    mdot[mask2] = mdot_break1 * (tgrid[mask2] / t_break1) ** alpha2

    # Calculate normalization for continuity at t_break2
    mdot_break2 = mdot_break1 * (t_break2 / t_break1) ** alpha2

    # Third regime: t >= t_break2
    mask3 = tgrid >= t_break2
    mdot[mask3] = mdot_break2 * (tgrid[mask3] / t_break2) ** alpha3

    # Convert units
    mdot = mdot * solar_mass_per_yr_to_gram_per_sec  # Convert to g/s
    tgrid = tgrid * 365.25 * 24 * 3600  # Convert years to seconds
    vwind = vwind * 1e5  # Convert km/s to cm/s
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert foe to ergs

    if kappa is not None:
        _get_csm().lc_mod.lightcurve_exponential_wind(
            mexp, eexp, mdot, tgrid, vwind, eff, kappa
        )
    else:
        _get_csm().lc_mod.lightcurve_exponential_wind(
            mexp, eexp, mdot, tgrid, vwind, eff
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_bpl_triple_powerlaw_wind(
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
    **kwargs,
):
    """
    Calculate the light curve for a broken power law explosion followed by triple power law wind interaction.

    :param delta: Delta parameter for broken power law
    :param nn: Power law index
    :param mexp: Explosion mass in M☉
    :param eexp: Explosion energy in foe
    :param t_break1: First break time in years
    :param t_break2: Second break time in years (must be > t_break1)
    :param mdot_0: Reference mass loss rate at t=1 year in M☉/yr
    :param alpha1: Power law index for t < t_break1
    :param alpha2: Power law index for t_break1 < t < t_break2
    :param alpha3: Power law index for t > t_break2
    :param vwind: Wind velocity in km/s
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
        - n_points: Number of points to discretize profile (default: 50)
        - kappa: Opacity in cm²/g for photon diffusion calculation
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)
    """
    n_points = kwargs.get("n_points", 50)
    kappa = kwargs.get("kappa", None)

    # Create time grid spanning the three power law regimes
    t_start = 0.1  # Start at 0.1 years to avoid t=0
    t_end = max(t_break2 * 2, 10)  # End well beyond second break
    tgrid = np.logspace(np.log10(t_start), np.log10(t_end), n_points)

    # Create triple power law mass loss profile
    mdot = np.zeros_like(tgrid)

    # First regime: t < t_break1
    mask1 = tgrid < t_break1
    mdot[mask1] = mdot_0 * (tgrid[mask1] / 1.0) ** alpha1

    # Calculate normalization for continuity at t_break1
    mdot_break1 = mdot_0 * (t_break1 / 1.0) ** alpha1

    # Second regime: t_break1 <= t < t_break2
    mask2 = (tgrid >= t_break1) & (tgrid < t_break2)
    mdot[mask2] = mdot_break1 * (tgrid[mask2] / t_break1) ** alpha2

    # Calculate normalization for continuity at t_break2
    mdot_break2 = mdot_break1 * (t_break2 / t_break1) ** alpha2

    # Third regime: t >= t_break2
    mask3 = tgrid >= t_break2
    mdot[mask3] = mdot_break2 * (tgrid[mask3] / t_break2) ** alpha3

    # Convert units
    mdot = mdot * solar_mass_per_yr_to_gram_per_sec  # Convert to g/s
    tgrid = tgrid * 365.25 * 24 * 3600  # Convert years to seconds
    vwind = vwind * 1e5  # Convert km/s to cm/s
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert foe to ergs

    if kappa is not None:
        _get_csm().lc_mod.lightcurve_bpl_wind(
            delta, nn, mexp, eexp, mdot, tgrid, vwind, eff, kappa
        )
    else:
        _get_csm().lc_mod.lightcurve_bpl_wind(
            delta, nn, mexp, eexp, mdot, tgrid, vwind, eff
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_smooth_triple_powerlaw_wind_bpl(
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
    **kwargs,
):
    """
    Calculate the light curve for a smooth triple power law mass-loss wind followed by a broken power law explosion.
    Uses tanh transitions for smooth breaks instead of sharp discontinuities.

    :param t_break1: First break time in years
    :param t_break2: Second break time in years (must be > t_break1)
    :param mdot_0: Reference mass loss rate at t=1 year in M☉/yr
    :param alpha1: Power law index for t < t_break1
    :param alpha2: Power law index for t_break1 < t < t_break2
    :param alpha3: Power law index for t > t_break2
    :param vwind: Wind velocity in km/s
    :param delta: Delta parameter for broken power law
    :param nn: Power law index
    :param mexp: Explosion mass in M☉
    :param eexp: Explosion energy in foe
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
        - n_points: Number of points to discretize profile (default: 50)
        - smooth_factor: Controls smoothness of transitions, 0.1-0.5 (default: 0.2)
        - kappa: (optional) Opacity in cm²/g for photon diffusion
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)
    """
    n_points = kwargs.get("n_points", 50)
    smooth_factor = kwargs.get("smooth_factor", 0.2)
    kappa = kwargs.get("kappa", None)

    # Create time grid spanning the three power law regimes
    t_start = 0.1  # Start at 0.1 years to avoid t=0
    t_end = max(t_break2 * 2, 10)  # End well beyond second break
    tgrid = np.logspace(np.log10(t_start), np.log10(t_end), n_points)

    # Create smooth triple power law mass loss profile using tanh transitions

    # Calculate the three power law components
    mdot_1 = mdot_0 * (tgrid / 1.0) ** alpha1

    # Calculate normalization for continuity at t_break1
    mdot_break1 = mdot_0 * (t_break1 / 1.0) ** alpha1
    mdot_2 = mdot_break1 * (tgrid / t_break1) ** alpha2

    # Calculate normalization for continuity at t_break2
    mdot_break2 = mdot_break1 * (t_break2 / t_break1) ** alpha2
    mdot_3 = mdot_break2 * (tgrid / t_break2) ** alpha3

    # Smooth transition widths (in log space for better scaling)
    smooth_width1 = (
        smooth_factor * np.log10(t_break1) if t_break1 > 1 else smooth_factor
    )
    smooth_width2 = (
        smooth_factor * np.log10(t_break2) if t_break2 > 1 else smooth_factor
    )

    # Create smooth transitions using tanh
    # Transition 1: from regime 1 to regime 2
    transition1 = 0.5 * (
        1 + np.tanh((np.log10(tgrid) - np.log10(t_break1)) / smooth_width1)
    )

    # Transition 2: from regime 2 to regime 3
    transition2 = 0.5 * (
        1 + np.tanh((np.log10(tgrid) - np.log10(t_break2)) / smooth_width2)
    )

    # Combine the three regimes with smooth transitions
    mdot = (
        (1 - transition1) * mdot_1
        + transition1 * (1 - transition2) * mdot_2
        + transition2 * mdot_3
    )

    # Convert units
    mdot = mdot * solar_mass_per_yr_to_gram_per_sec  # Convert to g/s
    tgrid = tgrid * 365.25 * 24 * 3600  # Convert years to seconds
    vwind = vwind * 1e5  # Convert km/s to cm/s
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert foe to ergs

    if kappa is not None:
        _get_csm().lc_mod.lightcurve_wind_bpl(
            mdot, tgrid, vwind, delta, nn, mexp, eexp, eff, kappa
        )
    else:
        _get_csm().lc_mod.lightcurve_wind_bpl(
            mdot, tgrid, vwind, delta, nn, mexp, eexp, eff
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_smooth_triple_powerlaw_wind_exponential(
    t_break1, t_break2, mdot_0, alpha1, alpha2, alpha3, vwind, mexp, eexp, eff, **kwargs
):
    """
    Calculate the light curve for a smooth triple power law mass-loss wind followed by an exponential explosion.
    Uses tanh transitions for smooth breaks instead of sharp discontinuities.

    :param t_break1: First break time in years
    :param t_break2: Second break time in years (must be > t_break1)
    :param mdot_0: Reference mass loss rate at t=1 year in M☉/yr
    :param alpha1: Power law index for t < t_break1
    :param alpha2: Power law index for t_break1 < t < t_break2
    :param alpha3: Power law index for t > t_break2
    :param vwind: Wind velocity in km/s
    :param mexp: Explosion mass in M☉
    :param eexp: Explosion energy in foe
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
        - n_points: Number of points to discretize profile (default: 50)
        - smooth_factor: Controls smoothness of transitions, 0.1-0.5 (default: 0.2)
        - kappa: (optional) Opacity in cm²/g for photon diffusion
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)
    """
    n_points = kwargs.get("n_points", 50)
    smooth_factor = kwargs.get("smooth_factor", 0.2)
    kappa = kwargs.get("kappa", None)

    # Create time grid spanning the three power law regimes
    t_start = 0.1  # Start at 0.1 years to avoid t=0
    t_end = max(t_break2 * 2, 10)  # End well beyond second break
    tgrid = np.logspace(np.log10(t_start), np.log10(t_end), n_points)

    # Create smooth triple power law mass loss profile using tanh transitions

    # Calculate the three power law components
    mdot_1 = mdot_0 * (tgrid / 1.0) ** alpha1

    # Calculate normalization for continuity at t_break1
    mdot_break1 = mdot_0 * (t_break1 / 1.0) ** alpha1
    mdot_2 = mdot_break1 * (tgrid / t_break1) ** alpha2

    # Calculate normalization for continuity at t_break2
    mdot_break2 = mdot_break1 * (t_break2 / t_break1) ** alpha2
    mdot_3 = mdot_break2 * (tgrid / t_break2) ** alpha3

    # Smooth transition widths (in log space for better scaling)
    smooth_width1 = (
        smooth_factor * np.log10(t_break1) if t_break1 > 1 else smooth_factor
    )
    smooth_width2 = (
        smooth_factor * np.log10(t_break2) if t_break2 > 1 else smooth_factor
    )

    # Create smooth transitions using tanh
    # Transition 1: from regime 1 to regime 2
    transition1 = 0.5 * (
        1 + np.tanh((np.log10(tgrid) - np.log10(t_break1)) / smooth_width1)
    )

    # Transition 2: from regime 2 to regime 3
    transition2 = 0.5 * (
        1 + np.tanh((np.log10(tgrid) - np.log10(t_break2)) / smooth_width2)
    )

    # Combine the three regimes with smooth transitions
    mdot = (
        (1 - transition1) * mdot_1
        + transition1 * (1 - transition2) * mdot_2
        + transition2 * mdot_3
    )

    # Convert units
    mdot = mdot * solar_mass_per_yr_to_gram_per_sec  # Convert to g/s
    tgrid = tgrid * 365.25 * 24 * 3600  # Convert years to seconds
    vwind = vwind * 1e5  # Convert km/s to cm/s
    mexp = mexp * solar_mass  # Convert solar masses to grams
    eexp = eexp * foe  # Convert foe to ergs

    if kappa is not None:
        _get_csm().lc_mod.lightcurve_wind_exponential(
            mdot, tgrid, vwind, mexp, eexp, eff, kappa
        )
    else:
        _get_csm().lc_mod.lightcurve_wind_exponential(
            mdot, tgrid, vwind, mexp, eexp, eff
        )

    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def create_wind_density_profile(mdot, vwind, time_ref):
    """
    Create a steady wind density profile: ρ = Mdot / (4πr²v)

    :param mdot: Mass loss rate in solar masses per year
    :param vwind: Wind velocity in km/s
    :param time_ref: Reference time for CSM setup in days (default: interval), i.e., \rho_csm(t = time_ref)
    :return: v_grid (velocity grid in cm/s), r_grid (radius grid in cm/s), density (density profile in g/cm³)
    """
    time_ref = time_ref * DAY  # Convert to seconds

    # Create velocity and radius grids
    v_grid = np.arange(1, 100001, dtype=np.float64) * 1e5  # 1e5 to 1e10 cm/s
    r_grid = v_grid * time_ref

    # Convert units
    mdot_cgs = mdot * solar_mass_per_yr_to_gram_per_sec  # g/s
    vwind_cgs = vwind * 1e5  # cm/s

    # Wind density: rho = Mdot / (4*pi*r^2*v)
    density = mdot_cgs / (4 * np.pi * r_grid**2 * vwind_cgs)

    return v_grid, r_grid, density


def _get_lc_multi_eruption_bpl_sn(
    eruption_list, interval, delta_sn, nn_sn, mej, esn, **kwargs
):
    """
    Calculate the light curve for multiple eruptions creating CSM, followed by a broken power law supernova.

    :param eruption_list: List of eruption dictionaries, each containing 'mass_msun', 'energy_foe', 'profile_type', etc.
    :param interval: Time interval between CSM construction and the supernova in DAYS
    :param delta_sn: Inner power-law index for supernova
    :param nn_sn: Outer power-law index for supernova
    :param mej: Supernova ejecta mass in solar masses
    :param esn: Supernova energy in foe (10^51 ergs)
    :param kwargs: Optional parameters
        - time_ref: Reference time for CSM setup in days (default: interval), i.e., \rho_csm(t = time_ref)
        - eff: Efficiency of kinetic to radiation energy conversion (default: 0.5)
        - kappa: (optional) Opacity in cm²/g for photon diffusion
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)

    Example::

        # Three eruptions creating complex CSM structure
        eruptions = [
            {'mass_msun': 0.1, 'energy_foe': 0.01, 'profile_type': 'exponential'},
            {'mass_msun': 0.05, 'energy_foe': 0.005, 'profile_type': 'exponential'},
            {'mass_msun': 0.2, 'energy_foe': 0.02, 'profile_type': 'exponential'}
        ]
        lc = _get_lc_multi_eruption_bpl_sn(eruptions, interval=350.0, delta_sn=0.5, nn_sn=12,
                                           mej_sn_grams=2.0, esn_sn_ergs=1)
    """
    mej = mej * solar_mass  # Convert to grams
    esn = esn * foe  # Convert to ergs
    time_ref_days = kwargs.get("time_ref", interval)
    time_ref_seconds = time_ref_days * DAY
    interval_seconds = interval * DAY
    eff = kwargs.get("eff", 0.5)
    kappa = kwargs.get("kappa", None)

    # Create CSM model from eruption list
    csm_model = SequentialCSMModel(time_ref=time_ref_days, verbose=False)

    for eruption in eruption_list:
        csm_model.add_eruption(
            mass_msun=eruption["mass_msun"],
            energy_foe=eruption["energy_foe"],
            label=eruption.get("label", None),
            profile_type=eruption.get("profile_type", "exponential"),
            profile_params=eruption.get("profile_params", None),
            shell_config=eruption.get("shell_config", None),
        )

    # Get CSM density profile
    result = csm_model.finalize_model()
    csm_density = result["final_csm_density"]
    v_grid = result["v_grid"]

    # Use the explosion_bpl Fortran interface
    # Signature: lightcurve_explosion_bpl(rho_input, vinput, t_ref, inner_slope, outer_slope, Mexp, Eexp, interval, eff)
    if kappa is not None:
        _get_csm().lc_mod.lightcurve_explosion_bpl(
            csm_density,
            v_grid,
            time_ref_seconds,
            delta_sn,
            nn_sn,
            mej,
            esn,
            interval_seconds,
            eff,
            kappa,
        )
    else:
        _get_csm().lc_mod.lightcurve_explosion_bpl(
            csm_density,
            v_grid,
            time_ref_seconds,
            delta_sn,
            nn_sn,
            mej,
            esn,
            interval_seconds,
            eff,
        )

    # Extract results
    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass

    return outs


def _get_lc_multi_eruption_exponential_sn(eruption_list, interval, mej, esn, **kwargs):
    """
    Calculate the light curve for multiple eruptions creating CSM, followed by an exponential supernova.

    :param eruption_list: List of eruption dictionaries, each containing 'mass_msun', 'energy_foe', 'profile_type', etc.
    :param interval: Time interval between CSM construction and the supernova in DAYS
    :param mej: Supernova ejecta mass in solar masses
    :param esn: Supernova energy in foe (10^51 ergs)
    :param kwargs: Optional parameters
        - time_ref: Reference time for CSM setup in days (default: interval), i.e., \rho_csm(t = time_ref)
        - eff: Efficiency of kinetic to radiation energy conversion (default: 0.5)
        - kappa: (optional) Opacity in cm²/g for photon diffusion
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)
    """
    # Store input values before conversion
    mej_msun = mej  # Keep solar mass version for create_exponential_profile
    esn_foe = esn  # Keep foe version for create_exponential_profile

    mej_cgs = mej * solar_mass  # Convert to grams for Fortran
    esn_cgs = esn * foe  # Convert to ergs for Fortran
    time_ref_days = kwargs.get("time_ref", interval)
    time_ref_seconds = time_ref_days * DAY
    interval_seconds = interval * DAY
    eff = kwargs.get("eff", 0.5)
    kappa = kwargs.get("kappa", None)

    # Create CSM model from eruption list
    csm_model = SequentialCSMModel(time_ref=time_ref_days, verbose=False)

    for eruption in eruption_list:
        csm_model.add_eruption(
            mass_msun=eruption["mass_msun"],
            energy_foe=eruption["energy_foe"],
            label=eruption.get("label", None),
            profile_type=eruption.get("profile_type", "exponential"),
            profile_params=eruption.get("profile_params", None),
            shell_config=eruption.get("shell_config", None),
        )

    # Get CSM density profile
    result = csm_model.finalize_model()
    csm_density = result["final_csm_density"]
    v_grid = result["v_grid"]

    # Use the new faster lightcurve_explosion_exponential interface
    # This uses analytical exponential formula for SN (inner) instead of creating a density grid
    # The CSM (outer) uses the arbitrary density profile from eruptions
    # This is much faster than lightcurve_explosion_explosion
    # Naming: explosion_exponential = arbitrary CSM + exponential SN
    # Arguments: (rho_csm, v_csm, t_ref_csm, Mexp_sn, Eexp_sn, interval, eff)
    if kappa is not None:
        _get_csm().lc_mod.lightcurve_explosion_exponential(
            csm_density,
            v_grid,
            time_ref_seconds,
            mej_cgs,
            esn_cgs,
            interval_seconds,
            eff,
            kappa,
        )
    else:
        _get_csm().lc_mod.lightcurve_explosion_exponential(
            csm_density,
            v_grid,
            time_ref_seconds,
            mej_cgs,
            esn_cgs,
            interval_seconds,
            eff,
        )

    # Extract results
    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass

    return outs


def _get_lc_multi_eruption_arbitrary_sn(
    eruption_list, interval, sn_eruption_dict, **kwargs
):
    """
    Calculate the light curve for multiple eruptions creating CSM, followed by an arbitrary supernova profile.

    :param eruption_list: List of eruption dictionaries for CSM creation with 'mass_msun', 'energy_foe', 'profile_type', etc.
    :param interval: Time interval between CSM construction and the supernova in DAYS
    :param sn_eruption_dict: Supernova eruption dictionary with 'mass_msun', 'energy_foe', 'profile_type', etc.
    :param kwargs: Optional parameters
        - time_ref: Reference time for CSM setup in days (default: interval), i.e., \rho_csm(t = time_ref)
        - eff: Efficiency of kinetic to radiation energy conversion (default: 0.5)
        - kappa: (optional) Opacity in cm²/g for photon diffusion
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)
    """
    time_ref_days = kwargs.get("time_ref", interval)
    time_ref_seconds = time_ref_days * DAY
    interval_seconds = interval * DAY
    eff = kwargs.get("eff", 0.5)
    kappa = kwargs.get("kappa", None)

    # Create CSM model from eruption list
    csm_model = SequentialCSMModel(time_ref=time_ref_days, verbose=False)

    for eruption in eruption_list:
        csm_model.add_eruption(
            mass_msun=eruption["mass_msun"],
            energy_foe=eruption["energy_foe"],
            label=eruption.get("label", None),
            profile_type=eruption.get("profile_type", "exponential"),
            profile_params=eruption.get("profile_params", None),
            shell_config=eruption.get("shell_config", None),
        )

    # Get CSM density profile
    result = csm_model.finalize_model()
    csm_density = result["final_csm_density"]
    v_grid = result["v_grid"]

    # Create supernova profile
    # The SN explodes at t=0, so use a small but non-zero time_ref
    # The density will scale as (t_ref/t)^3 in the Fortran code
    sn_time_ref = 1.0  # 1 day (reasonable for SN to establish profile)
    sn_model = SequentialCSMModel(verbose=False, time_ref=sn_time_ref)
    if sn_eruption_dict["profile_type"] == "exponential":
        v_grid_sn, _, sn_density, _ = sn_model.create_exponential_profile(
            sn_eruption_dict["mass_msun"], sn_eruption_dict["energy_foe"]
        )
    else:  # broken_powerlaw
        v_grid_sn, _, sn_density, _ = sn_model.create_broken_powerlaw_profile(
            sn_eruption_dict["mass_msun"],
            sn_eruption_dict["energy_foe"],
            sn_eruption_dict["profile_params"]["delta"],
            sn_eruption_dict["profile_params"]["n"],
        )

    # Use explosion_explosion interface
    # op(1) = SN (inner): t_ref = sn_time_ref days, delay = 0
    # op(2) = CSM (outer): t_ref = time_ref_days, delay = interval
    # The SN density will scale from t_ref=1day properly in Fortran via (t_ref/t)^3
    if kappa is not None:
        _get_csm().lc_mod.lightcurve_explosion_explosion(
            csm_density,
            v_grid,
            time_ref_seconds,
            sn_density,
            v_grid_sn,
            sn_time_ref * DAY,
            interval_seconds,
            eff,
            kappa,
        )
    else:
        _get_csm().lc_mod.lightcurve_explosion_explosion(
            csm_density,
            v_grid,
            time_ref_seconds,
            sn_density,
            v_grid_sn,
            sn_time_ref * DAY,
            interval_seconds,
            eff,
        )

    # Extract results
    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass

    return outs


def create_generic_csm_density(
    r_inner=1e10,
    r_outer=1e20,
    n_points=1000,
    base_density=1e-14,
    base_index=-2.0,
    n_shells=0,
    shell_radii=None,
    shell_widths=None,
    shell_densities=None,
    shell_profiles="gaussian",
    base_profile="powerlaw",
    base_bpl_params=None,
    time_ref_days=10 * YEAR_DAYS,
):
    """
    Create a generic phenomenological CSM density profile for fitting observations.

    This function constructs an arbitrary CSM density profile with:
    - A base component: simple power-law OR broken power-law (BPL)
    - Optional multiple shell/bump features (Gaussian or top-hat profiles)

    Parameters
    ----------
    r_inner : float
        Inner radius in cm (default: 1e10 cm)
    r_outer : float
        Outer radius in cm (default: 1e20 cm)
    n_points : int
        Number of grid points (default: 1000)
    base_density : float
        Normalization density at r_inner in g/cm^3 (default: 1e-14)
    base_index : float
        Power-law index for base CSM (default: -2.0 for wind)
        rho_base = base_density * (r/r_inner)^base_index
        Only used if base_profile='powerlaw'
    n_shells : int
        Number of shell/bump features to add (default: 0)
    shell_radii : array-like or None
        Radii of shell centers in cm. If None and n_shells > 0, will be evenly spaced in log
    shell_widths : array-like or None
        Widths (FWHM) of shells in cm. If None, uses 0.1 * r_shell
    shell_densities : array-like or None
        Peak density enhancements for each shell in g/cm^3.
        If None, uses 10 * base_density at shell location
    shell_profiles : str or list
        Profile type for shells: 'gaussian', 'tophat', or 'exponential'
        Can be a single string (applied to all) or list of strings (one per shell)
    base_profile : str
        Type of base density profile: 'powerlaw' or 'bpl' (broken power-law)
        Default: 'powerlaw'
    base_bpl_params : dict or None
        Parameters for broken power-law base profile (only used if base_profile='bpl'):
        - 'r_break': Break radius in cm (required)
        - 'index_inner': Power-law index for r < r_break (required)
        - 'index_outer': Power-law index for r > r_break (required)
        Example: {'r_break': 1e15, 'index_inner': -1.5, 'index_outer': -2.5}
    time_ref_days : float
        Expansion age in days used to derive v_grid (defaults to 10 years for standalone use).

    Returns
    -------
    r_grid : array
        Radius grid in cm
    v_grid : array
        Velocity grid in cm/s (assuming expansion from t=time_ref_days).
    csm_density : array
        Total CSM density profile in g/cm^3
    """
    # Create radius grid (log spacing for better resolution)
    r_grid = np.logspace(np.log10(r_inner), np.log10(r_outer), n_points)

    # Base density profile
    if base_profile == "powerlaw":
        # Simple power-law density profile
        csm_density = base_density * (r_grid / 1e14) ** base_index

    elif base_profile == "bpl":
        # Broken power-law density profile
        if base_bpl_params is None:
            raise ValueError("base_bpl_params required when base_profile='bpl'")

        r_break = base_bpl_params.get("r_break")
        index_inner = base_bpl_params.get("index_inner")
        index_outer = base_bpl_params.get("index_outer")

        if r_break is None or index_inner is None or index_outer is None:
            raise ValueError(
                "base_bpl_params must contain 'r_break', 'index_inner', 'index_outer'"
            )

        # Ensure continuity at break radius
        # At r_break: rho_inner(r_break) = rho_outer(r_break)
        # base_density * (r_break/r_inner)^index_inner = rho_break
        # For r > r_break: rho = rho_break * (r/r_break)^index_outer

        csm_density = np.zeros_like(r_grid)
        inner_mask = r_grid < r_break
        outer_mask = r_grid >= r_break

        # Inner region
        csm_density[inner_mask] = (
            base_density * (r_grid[inner_mask] / 1e14) ** index_inner
        )

        # Density at break (for continuity)
        rho_break = base_density * (r_break / 1e14) ** index_inner

        # Outer region
        csm_density[outer_mask] = (
            rho_break * (r_grid[outer_mask] / r_break) ** index_outer
        )

    else:
        raise ValueError(
            f"Unknown base_profile type: {base_profile}. Must be 'powerlaw' or 'bpl'"
        )

    # Add shells if requested
    if n_shells > 0:
        # Handle default shell parameters
        if shell_radii is None:
            # Evenly space shells in log space
            shell_radii = np.logspace(
                np.log10(r_inner * 2), np.log10(r_outer / 2), n_shells
            )
        else:
            shell_radii = np.atleast_1d(shell_radii)

        if shell_widths is None:
            # Default: width = 10% of shell radius
            shell_widths = 0.1 * shell_radii
        else:
            shell_widths = np.atleast_1d(shell_widths)

        if shell_densities is None:
            # Default: 10x the base density at shell location
            shell_densities = np.array(
                [base_density * (r / r_inner) ** base_index * 10.0 for r in shell_radii]
            )
        else:
            shell_densities = np.atleast_1d(shell_densities)

        # Handle shell profile types
        if isinstance(shell_profiles, str):
            shell_profiles = [shell_profiles] * n_shells

        # Add each shell
        for i in range(n_shells):
            r_shell = shell_radii[i]
            width = shell_widths[i]
            peak_density = shell_densities[i]
            profile_type = shell_profiles[i]

            if profile_type == "gaussian":
                # Gaussian shell: sigma = width / 2.355 (FWHM)
                sigma = width / 2.355
                shell_component = peak_density * np.exp(
                    -0.5 * ((r_grid - r_shell) / sigma) ** 2
                )

            elif profile_type == "tophat":
                # Top-hat shell: constant density within width
                shell_component = np.zeros_like(r_grid)
                mask = np.abs(r_grid - r_shell) < width / 2
                shell_component[mask] = peak_density

            elif profile_type == "exponential":
                # Exponential decay from shell center (symmetric)
                scale_length = width / 2.0
                shell_component = peak_density * np.exp(
                    -np.abs(r_grid - r_shell) / scale_length
                )

            else:
                raise ValueError(f"Unknown shell profile type: {profile_type}")

            # Add shell to total density
            csm_density += shell_component

    # Create velocity grid assuming material expanded for time_ref_days
    t_expansion = time_ref_days * DAY
    v_grid = r_grid / t_expansion

    return r_grid, v_grid, csm_density


def _get_lc_generic_csm_exponential(
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
    **kwargs,
):
    """
    Calculate the light curve for generic phenomenological CSM interacting with an EXPONENTIAL supernova.
    Fixed signature for redback integration with up to 3 shells (set density=0 to disable).

    Similar to _get_lc_generic_csm_bpl but uses an exponential final explosion instead of BPL.
    This is more appropriate for red novae, LRN, and merger events.

    :param base_density: Base CSM density normalization at r_inner in g/cm³
    :param base_index: Power-law index for base CSM (e.g., -2.0 for wind)
    :param shell1_radius: Radius of first shell center in cm (ignored if shell1_density=0)
    :param shell1_width: Width (FWHM) of first shell in cm
    :param shell1_density: Peak density of first shell in g/cm³ (set to 0 to disable)
    :param shell2_radius: Radius of second shell center in cm (ignored if shell2_density=0)
    :param shell2_width: Width (FWHM) of second shell in cm
    :param shell2_density: Peak density of second shell in g/cm³ (set to 0 to disable)
    :param shell3_radius: Radius of third shell center in cm (ignored if shell3_density=0)
    :param shell3_width: Width (FWHM) of third shell in cm
    :param shell3_density: Peak density of third shell in g/cm³ (set to 0 to disable)
    :param interval_sn: Time interval between CSM construction and SN explosion in DAYS
    :param mej_sn: SN ejecta mass in M☉
    :param esn: SN explosion energy in foe
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
        - r_inner: Inner CSM radius in cm (default: 1e13)
        - r_outer: Outer CSM radius in cm (default: 1e17)
        - kappa: Opacity in cm²/g for photon diffusion calculation
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)

    Example::

        # V838 Mon-like: multi-shell CSM + exponential explosion
        lc = _get_lc_generic_csm_exponential(
            base_density=1e-16, base_index=-2.0,
            shell1_radius=1e15, shell1_width=5e14, shell1_density=1e-14,
            shell2_radius=3e15, shell2_width=1e15, shell2_density=5e-15,
            shell3_radius=0, shell3_width=0, shell3_density=0,  # Disabled
            interval_sn=30, mej_sn=0.2, esn=0.01, eff=0.5
        )
    """
    r_inner = kwargs.get("r_inner", 1e10)
    r_outer = kwargs.get("r_outer", 1e20)
    kappa = kwargs.get("kappa", None)

    # Build shell parameters based on which shells are enabled
    shell_radii = []
    shell_widths = []
    shell_densities = []

    if shell1_density > 0:
        shell_radii.append(shell1_radius)
        shell_widths.append(shell1_width)
        shell_densities.append(shell1_density)

    if shell2_density > 0:
        shell_radii.append(shell2_radius)
        shell_widths.append(shell2_width)
        shell_densities.append(shell2_density)

    if shell3_density > 0:
        shell_radii.append(shell3_radius)
        shell_widths.append(shell3_width)
        shell_densities.append(shell3_density)

    n_shells = len(shell_radii)

    # Create generic CSM density profile
    r_grid, v_grid, csm_density = create_generic_csm_density(
        r_inner=r_inner,
        r_outer=r_outer,
        n_points=1000,
        base_density=base_density,
        base_index=base_index,
        n_shells=n_shells,
        shell_radii=shell_radii if n_shells > 0 else None,
        shell_widths=shell_widths if n_shells > 0 else None,
        shell_densities=shell_densities if n_shells > 0 else None,
        shell_profiles="gaussian",
        time_ref_days=interval_sn,
    )

    # Convert SN parameters to CGS
    mej_sn_grams = mej_sn * solar_mass
    esn_ergs = esn * foe
    interval_sec = interval_sn * DAY  # Convert to seconds

    # Call Fortran lightcurve function with EXPONENTIAL explosion
    if kappa is not None:
        _get_csm().lc_mod.lightcurve_explosion_exponential(
            csm_density,
            v_grid,
            interval_sec,
            mej_sn_grams,
            esn_ergs,
            interval_sec,
            eff,
            kappa,
        )
    else:
        _get_csm().lc_mod.lightcurve_explosion_exponential(
            csm_density, v_grid, interval_sec, mej_sn_grams, esn_ergs, interval_sec, eff
        )

    # Extract results
    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass
    return outs


def _get_lc_generic_csm_bpl(
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
    **kwargs,
):
    """
    Calculate the light curve for generic phenomenological CSM interacting with a broken power law supernova.
    Fixed signature for redback integration with up to 3 shells (set density=0 to disable).

    :param base_density: Base CSM density normalization at r_inner in g/cm³
    :param base_index: Power-law index for base CSM (e.g., -2.0 for wind)
    :param shell1_radius: Radius of first shell center in cm (ignored if shell1_density=0)
    :param shell1_width: Width (FWHM) of first shell in cm
    :param shell1_density: Peak density of first shell in g/cm³ (set to 0 to disable)
    :param shell2_radius: Radius of second shell center in cm (ignored if shell2_density=0)
    :param shell2_width: Width (FWHM) of second shell in cm
    :param shell2_density: Peak density of second shell in g/cm³ (set to 0 to disable)
    :param shell3_radius: Radius of third shell center in cm (ignored if shell3_density=0)
    :param shell3_width: Width (FWHM) of third shell in cm
    :param shell3_density: Peak density of third shell in g/cm³ (set to 0 to disable)
    :param interval_sn: Time interval between CSM construction and SN explosion in DAYS
    :param delta_sn: Inner power-law index for SN ejecta
    :param nn_sn: Outer power-law index for SN ejecta
    :param mej_sn: SN ejecta mass in M☉
    :param esn: SN explosion energy in foe
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
        - r_inner: Inner CSM radius in cm (default: 1e10)
        - r_outer: Outer CSM radius in cm (default: 1e20)
        - kappa: (optional) Opacity in cm²/g for photon diffusion
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)
    """
    r_inner = kwargs.get("r_inner", 1e10)
    r_outer = kwargs.get("r_outer", 1e20)
    kappa = kwargs.get("kappa", None)
    # Build shell parameters based on which shells are enabled
    shell_radii = []
    shell_widths = []
    shell_densities = []

    if shell1_density > 0:
        shell_radii.append(shell1_radius)
        shell_widths.append(shell1_width)
        shell_densities.append(shell1_density)

    if shell2_density > 0:
        shell_radii.append(shell2_radius)
        shell_widths.append(shell2_width)
        shell_densities.append(shell2_density)

    if shell3_density > 0:
        shell_radii.append(shell3_radius)
        shell_widths.append(shell3_width)
        shell_densities.append(shell3_density)

    n_shells = len(shell_radii)

    # Create generic CSM density profile
    r_grid, v_grid, csm_density = create_generic_csm_density(
        r_inner=r_inner,
        r_outer=r_outer,
        n_points=1000,
        base_density=base_density,
        base_index=base_index,
        n_shells=n_shells,
        shell_radii=shell_radii if n_shells > 0 else None,
        shell_widths=shell_widths if n_shells > 0 else None,
        shell_densities=shell_densities if n_shells > 0 else None,
        shell_profiles="gaussian",
        time_ref_days=interval_sn,
    )

    # Convert SN parameters to CGS
    mej_sn_grams = mej_sn * solar_mass
    esn_ergs = esn * foe
    interval_sec = interval_sn * DAY  # Convert to seconds

    # Call Fortran lightcurve function
    if kappa is not None:
        _get_csm().lc_mod.lightcurve_explosion_bpl(
            csm_density,
            v_grid,
            interval_sec,
            delta_sn,
            nn_sn,
            mej_sn_grams,
            esn_ergs,
            interval_sec,
            eff,
            kappa,
        )
    else:
        _get_csm().lc_mod.lightcurve_explosion_bpl(
            csm_density,
            v_grid,
            interval_sec,
            delta_sn,
            nn_sn,
            mej_sn_grams,
            esn_ergs,
            interval_sec,
            eff,
        )

    # Extract results
    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass

    return outs


def _get_lc_generic_4shell_csm_bpl(
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
    **kwargs,
):
    """
    Calculate the light curve for generic phenomenological CSM interacting with a broken power law supernova.
    Fixed signature for redback integration with up to 4 shells (set density=0 to disable).

    :param base_density: Base CSM density normalization at r_inner in g/cm³
    :param base_index: Power-law index for base CSM (e.g., -2.0 for wind)
    :param shell1_radius: Radius of first shell center in cm (ignored if shell1_density=0)
    :param shell1_width: Width (FWHM) of first shell in cm
    :param shell1_density: Peak density of first shell in g/cm³ (set to 0 to disable)
    :param shell2_radius: Radius of second shell center in cm (ignored if shell2_density=0)
    :param shell2_width: Width (FWHM) of second shell in cm
    :param shell2_density: Peak density of second shell in g/cm³ (set to 0 to disable)
    :param shell3_radius: Radius of third shell center in cm (ignored if shell3_density=0)
    :param shell3_width: Width (FWHM) of third shell in cm
    :param shell3_density: Peak density of third shell in g/cm³ (set to 0 to disable)
    :param interval_sn: Time interval between CSM construction and SN explosion in DAYS
    :param delta_sn: Inner power-law index for SN ejecta
    :param nn_sn: Outer power-law index for SN ejecta
    :param mej_sn: SN ejecta mass in M☉
    :param esn: SN explosion energy in foe
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
        - r_inner: Inner CSM radius in cm (default: 1e10)
        - r_outer: Outer CSM radius in cm (default: 1e20)
        - kappa: (optional) Opacity in cm²/g for photon diffusion
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)
    """
    r_inner = kwargs.get("r_inner", 1e10)
    r_outer = kwargs.get("r_outer", 1e20)
    kappa = kwargs.get("kappa", None)
    # Build shell parameters based on which shells are enabled
    shell_radii = []
    shell_widths = []
    shell_densities = []

    if shell1_density > 0:
        shell_radii.append(shell1_radius)
        shell_widths.append(shell1_width)
        shell_densities.append(shell1_density)

    if shell2_density > 0:
        shell_radii.append(shell2_radius)
        shell_widths.append(shell2_width)
        shell_densities.append(shell2_density)

    if shell3_density > 0:
        shell_radii.append(shell3_radius)
        shell_widths.append(shell3_width)
        shell_densities.append(shell3_density)

    if shell4_density > 0:
        shell_radii.append(shell4_radius)
        shell_widths.append(shell4_width)
        shell_densities.append(shell4_density)

    n_shells = len(shell_radii)

    # Create generic CSM density profile
    r_grid, v_grid, csm_density = create_generic_csm_density(
        r_inner=r_inner,
        r_outer=r_outer,
        n_points=1000,
        base_density=base_density,
        base_index=base_index,
        n_shells=n_shells,
        shell_radii=shell_radii if n_shells > 0 else None,
        shell_widths=shell_widths if n_shells > 0 else None,
        shell_densities=shell_densities if n_shells > 0 else None,
        shell_profiles="gaussian",
        time_ref_days=interval_sn,
    )

    # Convert SN parameters to CGS
    mej_sn_grams = mej_sn * solar_mass
    esn_ergs = esn * foe
    interval_sec = interval_sn * DAY  # Convert to seconds

    # Call Fortran lightcurve function
    if kappa is not None:
        _get_csm().lc_mod.lightcurve_explosion_bpl(
            csm_density,
            v_grid,
            interval_sec,
            delta_sn,
            nn_sn,
            mej_sn_grams,
            esn_ergs,
            interval_sec,
            eff,
            kappa,
        )
    else:
        _get_csm().lc_mod.lightcurve_explosion_bpl(
            csm_density,
            v_grid,
            interval_sec,
            delta_sn,
            nn_sn,
            mej_sn_grams,
            esn_ergs,
            interval_sec,
            eff,
        )

    # Extract results
    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass

    return outs


def _get_lc_generic_8shell_csm_bpl(
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
    **kwargs,
):
    """
    Calculate the light curve for generic phenomenological CSM interacting with a broken power law supernova.
    Fixed signature for redback integration with up to 8 shells (set density=0 to disable).

    This is an extension of _get_lc_generic_4shell_csm_bpl to support more complex CSM structures
    with up to 8 distinct shell features, useful for modeling highly variable mass-loss histories
    or multiple pre-SN eruptions.

    :param base_density: Base CSM density normalization at r_inner in g/cm³
    :param base_index: Power-law index for base CSM (e.g., -2.0 for wind)
    :param shell1_radius: Radius of first shell center in cm (ignored if shell1_density=0)
    :param shell1_width: Width (FWHM) of first shell in cm
    :param shell1_density: Peak density of first shell in g/cm³ (set to 0 to disable)
    :param shell2_radius: Radius of second shell center in cm (ignored if shell2_density=0)
    :param shell2_width: Width (FWHM) of second shell in cm
    :param shell2_density: Peak density of second shell in g/cm³ (set to 0 to disable)
    :param shell3_radius: Radius of third shell center in cm (ignored if shell3_density=0)
    :param shell3_width: Width (FWHM) of third shell in cm
    :param shell3_density: Peak density of third shell in g/cm³ (set to 0 to disable)
    :param shell4_radius: Radius of fourth shell center in cm (ignored if shell4_density=0)
    :param shell4_width: Width (FWHM) of fourth shell in cm
    :param shell4_density: Peak density of fourth shell in g/cm³ (set to 0 to disable)
    :param shell5_radius: Radius of fifth shell center in cm (ignored if shell5_density=0)
    :param shell5_width: Width (FWHM) of fifth shell in cm
    :param shell5_density: Peak density of fifth shell in g/cm³ (set to 0 to disable)
    :param shell6_radius: Radius of sixth shell center in cm (ignored if shell6_density=0)
    :param shell6_width: Width (FWHM) of sixth shell in cm
    :param shell6_density: Peak density of sixth shell in g/cm³ (set to 0 to disable)
    :param shell7_radius: Radius of seventh shell center in cm (ignored if shell7_density=0)
    :param shell7_width: Width (FWHM) of seventh shell in cm
    :param shell7_density: Peak density of seventh shell in g/cm³ (set to 0 to disable)
    :param shell8_radius: Radius of eighth shell center in cm (ignored if shell8_density=0)
    :param shell8_width: Width (FWHM) of eighth shell in cm
    :param shell8_density: Peak density of eighth shell in g/cm³ (set to 0 to disable)
    :param interval_sn: Time interval between CSM construction and SN explosion in DAYS
    :param delta_sn: Inner power-law index for SN ejecta
    :param nn_sn: Outer power-law index for SN ejecta
    :param mej_sn: SN ejecta mass in M☉
    :param esn: SN explosion energy in foe
    :param eff: Efficiency of kinetic to radiation energy conversion (0-1)
    :param kwargs: Optional parameters
        - r_inner: Inner CSM radius in cm (default: 1e10)
        - r_outer: Outer CSM radius in cm (default: 1e20)
        - base_profile: 'powerlaw' or 'bpl' for base density (default: 'powerlaw')
        - base_bpl_params: dict with 'r_break', 'index_inner', 'index_outer' if base_profile='bpl'
        - kappa: (optional) Opacity in cm²/g for photon diffusion
    :return: Named tuple (time, lbol, lbol_shock, lbol_diffuse, rph, temperature, vshell, shell_mass)

    Example::

        # Model iPTF14hls with 8 shell features from multiple eruptions
        lc = _get_lc_generic_8shell_csm_bpl(
            base_density=1e-15, base_index=-2.0,
            shell1_radius=1e14, shell1_width=5e13, shell1_density=1e-13,
            shell2_radius=3e14, shell2_width=1e14, shell2_density=5e-14,
            shell3_radius=8e14, shell3_width=2e14, shell3_density=3e-14,
            shell4_radius=2e15, shell4_width=5e14, shell4_density=2e-14,
            shell5_radius=5e15, shell5_width=1e15, shell5_density=1e-14,
            shell6_radius=0, shell6_width=0, shell6_density=0,  # Disabled
            shell7_radius=0, shell7_width=0, shell7_density=0,  # Disabled
            shell8_radius=0, shell8_width=0, shell8_density=0,  # Disabled
            interval_sn=100, delta_sn=0.5, nn_sn=10, mej_sn=10.0, esn=1.0, eff=0.5
        )
    """
    r_inner = kwargs.get("r_inner", 1e10)
    r_outer = kwargs.get("r_outer", 1e20)
    base_profile = kwargs.get("base_profile", "powerlaw")
    base_bpl_params = kwargs.get("base_bpl_params", None)
    kappa = kwargs.get("kappa", None)

    # Build shell parameters based on which shells are enabled
    shell_radii = []
    shell_widths = []
    shell_densities = []

    if shell1_density > 0:
        shell_radii.append(shell1_radius)
        shell_widths.append(shell1_width)
        shell_densities.append(shell1_density)

    if shell2_density > 0:
        shell_radii.append(shell2_radius)
        shell_widths.append(shell2_width)
        shell_densities.append(shell2_density)

    if shell3_density > 0:
        shell_radii.append(shell3_radius)
        shell_widths.append(shell3_width)
        shell_densities.append(shell3_density)

    if shell4_density > 0:
        shell_radii.append(shell4_radius)
        shell_widths.append(shell4_width)
        shell_densities.append(shell4_density)

    if shell5_density > 0:
        shell_radii.append(shell5_radius)
        shell_widths.append(shell5_width)
        shell_densities.append(shell5_density)

    if shell6_density > 0:
        shell_radii.append(shell6_radius)
        shell_widths.append(shell6_width)
        shell_densities.append(shell6_density)

    if shell7_density > 0:
        shell_radii.append(shell7_radius)
        shell_widths.append(shell7_width)
        shell_densities.append(shell7_density)

    if shell8_density > 0:
        shell_radii.append(shell8_radius)
        shell_widths.append(shell8_width)
        shell_densities.append(shell8_density)

    n_shells = len(shell_radii)

    # Create generic CSM density profile
    r_grid, v_grid, csm_density = create_generic_csm_density(
        r_inner=r_inner,
        r_outer=r_outer,
        n_points=1000,
        base_density=base_density,
        base_index=base_index,
        n_shells=n_shells,
        shell_radii=shell_radii if n_shells > 0 else None,
        shell_widths=shell_widths if n_shells > 0 else None,
        shell_densities=shell_densities if n_shells > 0 else None,
        shell_profiles="gaussian",
        base_profile=base_profile,
        base_bpl_params=base_bpl_params,
        time_ref_days=interval_sn,
    )
    # Convert SN parameters to CGS
    mej_sn_grams = mej_sn * solar_mass
    esn_ergs = esn * foe
    interval_sec = interval_sn * DAY  # Convert to seconds

    # Call Fortran lightcurve function
    if kappa is not None:
        _get_csm().lc_mod.lightcurve_explosion_bpl(
            csm_density,
            v_grid,
            interval_sec,
            delta_sn,
            nn_sn,
            mej_sn_grams,
            esn_ergs,
            interval_sec,
            eff,
            kappa,
        )
    else:
        _get_csm().lc_mod.lightcurve_explosion_bpl(
            csm_density,
            v_grid,
            interval_sec,
            delta_sn,
            nn_sn,
            mej_sn_grams,
            esn_ergs,
            interval_sec,
            eff,
        )

    # Extract results
    time_array = _get_csm().lc_mod.tarray.copy()
    lbol_shock = _get_csm().lc_mod.larray.copy()
    rph = _get_csm().lc_mod.rarray.copy()
    vshell = _get_csm().lc_mod.varray.copy()
    shell_mass = _get_csm().lc_mod.marray.copy()
    temperature = _get_csm().lc_mod.temparray.copy()

    if kappa is not None:
        lbol_diffuse = _get_csm().lc_mod.ldiff.copy()
        lbol = lbol_diffuse
    else:
        lbol_diffuse = None
        lbol = lbol_shock

    outs = namedtuple(
        "output",
        [
            "time",
            "lbol",
            "lbol_shock",
            "lbol_diffuse",
            "rph",
            "temperature",
            "vshell",
            "shell_mass",
        ],
    )
    outs.time = time_array
    outs.lbol = lbol
    outs.lbol_shock = lbol_shock
    outs.lbol_diffuse = lbol_diffuse
    outs.rph = rph
    outs.temperature = temperature
    outs.vshell = vshell
    outs.shell_mass = shell_mass

    return outs


def combine_lightcurves(lc1, lc2, theta_polar):
    """
    Combine two light curves weighted by solid angle

    :param lc1: light curve tuple for polar explosion
    :param lc2: light curve tuple for equatorial explosion
    :param theta_polar: half opening angle of the polar explosion
    :return: Named tuple (time, lbol)
    """
    LCOutput = namedtuple("output", ["time", "lbol"])

    weight1 = 1 - np.cos(theta_polar / 180 * np.pi)
    weight2 = 1 - weight1

    # Determine overlapping time interval
    tmin = max(lc1.time.min(), lc2.time.min())
    tmax = min(lc1.time.max(), lc2.time.max())
    if tmax <= tmin:
        raise ValueError("No overlapping time region between light curves.")

    # Build adaptive time grid within the overlap
    t_all = np.unique(
        np.concatenate(
            (
                lc1.time[(lc1.time >= tmin) & (lc1.time <= tmax)],
                lc2.time[(lc2.time >= tmin) & (lc2.time <= tmax)],
            )
        )
    )
    t_all.sort()

    # Define helper for interpolating a field (no extrapolation)
    def interp_field(field):
        f1 = interp1d(
            lc1.time, field(lc1), bounds_error=False, fill_value="extrapolate"
        )
        f2 = interp1d(
            lc2.time, field(lc2), bounds_error=False, fill_value="extrapolate"
        )
        # Evaluate only within overlap; beyond that the time grid is already truncated
        return weight1 * f1(t_all) + weight2 * f2(t_all)

    return LCOutput(time=t_all, lbol=interp_field(lambda lc: lc.lbol))


class SequentialCSMModel:
    """
    Class for creating sequential CSM models with shell interactions.

    Supports both exponential and broken power-law density profiles with
    smooth Gaussian shells and proper mass conservation.

    Units: Mass in solar masses, Energy in 10^51 ergs (foe),
    time_ref specified in days.
    """

    def __init__(self, time_ref, verbose=True):
        """
        Initialize the sequential CSM model.

        Parameters
        ----------
        time_ref : float
            Reference time in days that sets the expansion age used to convert
            velocities to radii when building the CSM snapshots.
        verbose : bool, optional
            Whether to print detailed output (default: True)
        """
        time_ref_days = time_ref
        self.time_ref_days = time_ref_days
        self.time_ref_seconds = self.time_ref_days * DAY
        self.verbose = verbose
        self.step_results = []
        self.current_csm_density = None
        self.v_grid = None
        self.r_grid = None
        self.final_result = None

    def _print(self, message):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def set_wind(self, mdot, vwind, duration_years=None):
        """
        Set an initial wind component to establish baseline CSM.

        The wind creates a steady-state density profile: ρ = Mdot / (4πr²v_wind)
        This acts as the "first eruption" baseline that subsequent eruptions interact with.

        Parameters
        ----------
        mdot : float
            Mass-loss rate in M_sun/year
        vwind : float
            Wind velocity in km/s (constant)
        duration_years : float, optional
            Duration of wind mass loss in years. If None, wind fills the grid.

        Returns
        -------
        wind_result : dict
            Results including wind density profile and mass
        """
        self._print(f"\nSetting wind baseline:")
        self._print(f"  Mdot: {mdot:.2e} M_sun/yr")
        self._print(f"  v_wind: {vwind:.0f} km/s")

        # Convert to CGS
        mdot_cgs = mdot * solar_mass_per_yr_to_gram_per_sec  # g/s
        vwind_cgs = vwind * 1e5  # cm/s

        # Create grids if not yet created
        if self.v_grid is None:
            self.v_grid = np.arange(1, 100001, dtype=np.float64) * 1e5  # cm/s
            self.r_grid = self.v_grid * self.time_ref_seconds

        # Wind extent
        if duration_years is None:
            # Default: wind fills entire grid
            r_outer_wind = self.r_grid.max()
            duration_years = (r_outer_wind / vwind_cgs) / (365.25 * DAY)

        duration_seconds = duration_years * 365.25 * DAY

        # Total wind mass
        total_wind_mass_cgs = mdot_cgs * duration_seconds
        total_wind_mass_msun = total_wind_mass_cgs / solar_mass

        # Wind density profile: rho = Mdot / (4*pi*r^2*v)
        wind_density = mdot_cgs / (4 * np.pi * self.r_grid**2 * vwind_cgs)

        # Wind radial extent (inner: 1 day old, outer: duration_years old)
        r_wind_outer = vwind_cgs * duration_seconds
        r_wind_inner = vwind_cgs * (1.0 * DAY)

        # Mask wind to physical extent
        wind_mask = (self.r_grid >= r_wind_inner) & (self.r_grid <= r_wind_outer)
        wind_density_physical = np.zeros_like(self.r_grid)
        wind_density_physical[wind_mask] = wind_density[wind_mask]

        # Verify mass conservation
        actual_wind_mass_cgs = np.trapz(
            wind_density_physical * 4 * np.pi * self.r_grid**2, self.r_grid
        )
        actual_wind_mass_msun = actual_wind_mass_cgs / solar_mass

        self._print(f"  Duration: {duration_years:.2f} years")
        self._print(
            f"  Wind radial extent: {r_wind_inner/1e15:.2f} - {r_wind_outer/1e15:.2f} × 10¹⁵ cm"
        )
        self._print(f"  Total wind mass: {actual_wind_mass_msun:.4f} M_sun")

        # Set as baseline CSM (like first eruption)
        self.current_csm_density = wind_density_physical

        # Store wind info as step 0
        wind_result = {
            "step_number": 0,
            "label": "Wind_Baseline",
            "profile_type": "wind",
            "mdot": mdot,
            "vwind": vwind,
            "duration_years": duration_years,
            "wind_density": wind_density_physical.copy(),
            "csm_density": self.current_csm_density.copy(),
            "mass_msun": actual_wind_mass_msun,
            "calculated_mass_cgs": actual_wind_mass_cgs,
            "calculated_mass_msun": actual_wind_mass_msun,
            "interaction_type": "wind_baseline",
        }

        self.step_results.append(wind_result)

        return wind_result

    def create_exponential_profile(self, mass_msun, energy_foe):
        """
        Create exponential density profile for a single eruption.

        Parameters
        ----------
        mass_msun : float
            Explosion mass in solar masses
        energy_foe : float
            Explosion energy in foe (10^51 erg)

        Returns
        -------
        v_grid : ndarray
            Velocity grid in cm/s
        r_grid : ndarray
            Radius grid in cm
        density : ndarray
            Density profile in g/cm³
        v0 : float
            Characteristic velocity in cm/s
        """
        # Convert to CGS units
        mass = mass_msun * solar_mass
        energy = energy_foe * foe

        # Characteristic velocity: v₀ = √(E/(6M))
        v0 = np.sqrt(energy / (6.0 * mass))

        # Velocity grid
        v_grid = np.arange(1, 100001, dtype=np.float64) * 1e5  # 1e5 to 1e10 cm/s

        # Radius grid: r = v * t
        r_grid = v_grid * self.time_ref_seconds

        # Exponential density profile: ρ(v,t) = M/(8π v₀³ t³) * exp(-v/v₀)
        density = (mass / (8.0 * np.pi * (v0 * self.time_ref_seconds) ** 3)) * np.exp(
            -v_grid / v0
        )

        return v_grid, r_grid, density, v0

    def create_broken_powerlaw_profile(self, mass_msun, energy_foe, delta, n):
        """
        Create broken power-law density profile using analytical formula.

        Uses analytical formula from Matzner & McKee (1999):
        ρ_csm(v_ej, t) = [2(5-δ)(n-5)E_ej] / [4π(n-δ)[(3-δ)(n-3)M_ej]^((δ-3)/2)] * t^-3 * v_ej^-δ  (v_ej > v_i)
        ρ_csm(v_ej, t) = [2(5-δ)(n-5)E_ej]^((δ-3)/2) / [4π(n-δ)[(3-δ)(n-3)M_ej]^((n-3)/2)] * t^-3 * v_ej^-n  (v_ej < v_i)

        Parameters
        ----------
        mass_msun : float
            Explosion mass in solar masses
        energy_foe : float
            Explosion energy in foe (10^51 erg)
        delta : float
            Inner power-law index (for v < v*)
        n : float
            Outer power-law index (for v > v*)

        Returns
        -------
        v_grid : ndarray
            Velocity grid in cm/s
        r_grid : ndarray
            Radius grid in cm
        density : ndarray
            Density profile in g/cm³
        v_star : float
            Break velocity in cm/s
        """
        # Convert to CGS units
        mass = mass_msun * solar_mass
        energy = energy_foe * foe

        # Calculate break velocity: v_i = √[(2(5-δ)(n-5)E)/((3-δ)(n-3)M)]
        numerator = 2 * (5 - delta) * (n - 5) * energy
        denominator = (3 - delta) * (n - 3) * mass

        if denominator <= 0 or numerator <= 0:
            raise ValueError(
                f"Invalid parameters for broken power law: delta={delta}, n={n}"
            )

        v_star = np.sqrt(numerator / denominator)

        # Velocity grid
        v_grid = np.arange(1, 100001, dtype=np.float64) * 1e5  # 1e5 to 1e10 cm/s

        # Radius grid: r = v * t
        r_grid = v_grid * self.time_ref_seconds

        # Use analytical formula from Matzner & McKee (1999)
        # This provides the exact analytical coefficients for the BPL density profile
        #
        # The Fortran code calculates coefficients that work with 4πr²ρ:
        # 4πr²ρ = bpl_co × (r/t)^(2-n) / t  (outer)
        # 4πr²ρ = bpl_ci × (r/t)^(2-δ) / t  (inner)
        #
        # To get ρ, we divide by 4πr² where r = v × t_ref:
        # ρ = [bpl_co × (r/t)^(2-n) / t] / (4πr²)
        #   = [bpl_co × v^(2-n) / t] / (4π(vt)²)
        #   = bpl_co / (4π) × v^(2-n) / (t × v² × t²)
        #   = bpl_co / (4π) × v^(-n) / t³

        E_term = 2 * (5 - delta) * (n - 5) * energy
        M_term = (3 - delta) * (n - 3) * mass

        # Calculate Fortran coefficients (for 4πr²ρ) using analytical formula
        bpl_co = (E_term ** (0.5 * (n - 3))) / (M_term ** (0.5 * (n - 5))) / (n - delta)
        bpl_ci = (
            (E_term ** (0.5 * (delta - 3)))
            / (M_term ** (0.5 * (delta - 5)))
            / (n - delta)
        )

        density = np.zeros_like(v_grid)

        # Outer region (v >= v*):
        # Fortran: 4πr²ρ = bpl_co × v^(2-n) / t
        # So: ρ = bpl_co × v^(2-n) / t / (4πr²)
        #       = bpl_co × v^(2-n) / t / (4π × v² × t²)
        #       = bpl_co / (4π) × v^(-n) / t³
        outer_mask = v_grid >= v_star
        density[outer_mask] = (
            (bpl_co / (4 * np.pi))
            * (v_grid[outer_mask] ** (-n))
            / (self.time_ref_seconds**3)
        )

        # Inner region (v < v*):
        # Fortran: 4πr²ρ = bpl_ci × v^(2-δ) / t
        # So: ρ = bpl_ci × v^(2-δ) / t / (4πr²)
        #       = bpl_ci × v^(2-δ) / t / (4π × v² × t²)
        #       = bpl_ci / (4π) × v^(-δ) / t³
        inner_mask = v_grid < v_star
        density[inner_mask] = (
            (bpl_ci / (4 * np.pi))
            * (v_grid[inner_mask] ** (-delta))
            / (self.time_ref_seconds**3)
        )

        return v_grid, r_grid, density, v_star

    def create_density_profile(
        self, mass_msun, energy_foe, profile_type="exponential", profile_params=None
    ):
        """
        Create density profile with support for multiple profile types.

        Parameters
        ----------
        mass_msun : float
            Explosion mass in solar masses
        energy_foe : float
            Explosion energy in foe (10^51 erg)
        profile_type : str
            'exponential' or 'broken_powerlaw'
        profile_params : dict
            Additional parameters for broken power law:
            - 'delta': inner power-law index
            - 'n': outer power-law index

        Returns
        -------
        v_grid : ndarray
            Velocity grid in cm/s
        r_grid : ndarray
            Radius grid in cm
        density : ndarray
            Density profile in g/cm³
        characteristic_velocity : float
            v0 for exponential or v* for broken power law
        """
        if profile_type == "exponential":
            return self.create_exponential_profile(mass_msun, energy_foe)

        elif profile_type == "broken_powerlaw":
            if profile_params is None:
                raise ValueError("profile_params required for broken_powerlaw")

            delta = profile_params.get("delta", 0.5)
            n = profile_params.get("n", 10.0)

            return self.create_broken_powerlaw_profile(mass_msun, energy_foe, delta, n)

        else:
            raise ValueError(f"Unknown profile_type: {profile_type}")

    def calculate_swept_mass(self, density, r_inner, r_outer):
        """
        Calculate swept mass from a density profile between two radii.

        Parameters
        ----------
        density : ndarray
            Density profile in g/cm³
        r_inner : float
            Inner radius in cm
        r_outer : float
            Outer radius in cm

        Returns
        -------
        swept_mass : float
            Mass swept into shell in grams
        """
        # Create mask for the region
        mask = (self.r_grid >= r_inner) & (self.r_grid <= r_outer)

        if not np.any(mask):
            return 0.0

        # Extract region data
        r_region = self.r_grid[mask]
        density_region = density[mask]

        # Integrate mass: M = ∫ ρ(r) * 4πr² dr
        integrand = density_region * 4 * np.pi * r_region**2
        swept_mass = np.trapz(integrand, r_region)

        return swept_mass

    def create_gaussian_shell(self, shell_mass, r_sh, delta_r):
        """
        Create smooth Gaussian shell that will be ADDED to underlying density.

        Parameters
        ----------
        shell_mass : float
            Total mass in shell in grams
        r_sh : float
            Shell center radius in cm
        delta_r : float
            Shell width (approximately 4σ for Gaussian)

        Returns
        -------
        shell_enhancement : ndarray
            Gaussian enhancement to be added to base density in g/cm³
        sigma : float
            Gaussian width parameter in cm
        """
        # Gaussian width parameter (broader shell for mass conservation)
        sigma = delta_r / 2.0

        # Create Gaussian profile centered at r_sh
        gaussian_profile = np.exp(-(((self.r_grid - r_sh) / sigma) ** 2))

        # Normalize to conserve mass
        gaussian_mass_integral = np.trapz(
            gaussian_profile * 4 * np.pi * self.r_grid**2, self.r_grid
        )

        if gaussian_mass_integral > 0:
            # Scale to achieve desired shell mass
            normalization_factor = shell_mass / gaussian_mass_integral
            shell_enhancement = normalization_factor * gaussian_profile
        else:
            shell_enhancement = np.zeros_like(self.r_grid)

        return shell_enhancement, sigma

    def add_eruption(
        self,
        mass_msun,
        energy_foe,
        label=None,
        profile_type="exponential",
        profile_params=None,
        shell_config=None,
    ):
        """
        Add a new eruption to the sequential CSM model.

        Parameters
        ----------
        mass_msun : float
            Explosion mass in solar masses
        energy_foe : float
            Explosion energy in foe (10^51 erg)
        label : str, optional
            Descriptive label for the eruption
        profile_type : str, optional
            'exponential' or 'broken_powerlaw' (default: 'exponential')
        profile_params : dict, optional
            Additional parameters for broken power law:
            - 'delta': inner power-law index
            - 'n': outer power-law index
        shell_config : dict, optional
            Shell interaction configuration:
            - 'r_shell': shell center radius in cm
            - 'shell_width': shell width in cm

        Returns
        -------
        step_result : dict
            Results from processing this eruption
        """
        step_number = len(self.step_results) + 1
        if label is None:
            label = f"Eruption_{step_number}"

        self._print(f"\nStep {step_number}: Processing {label}")
        self._print(f"  Mass: {mass_msun:.3f} M☉")
        self._print(f"  Energy: {energy_foe:.3f} foe")
        self._print(f"  Profile type: {profile_type}")

        if profile_type == "broken_powerlaw" and profile_params:
            self._print(
                f"  Power-law indices: δ={profile_params.get('delta', 0.0):.1f}, n={profile_params.get('n', 7.0):.1f}"
            )

        # Create explosion profile
        self.v_grid, self.r_grid, explosion_density, char_velocity = (
            self.create_density_profile(
                mass_msun, energy_foe, profile_type, profile_params
            )
        )

        if profile_type == "exponential":
            self._print(f"  Characteristic velocity v₀: {char_velocity / 1e5:.0f} km/s")
        else:
            self._print(f"  Break velocity v*: {char_velocity / 1e5:.0f} km/s")

        # Verify mass conservation
        calculated_mass_cgs = np.trapz(
            explosion_density * 4 * np.pi * self.r_grid**2, self.r_grid
        )
        calculated_mass_msun = calculated_mass_cgs / solar_mass
        self._print(
            f"  Mass conservation check: {calculated_mass_msun / mass_msun:.4f} (should be ~1.0)"
        )

        # Initialize step result
        step_result = {
            "step_number": step_number,
            "label": label,
            "profile_type": profile_type,
            "explosion_density": explosion_density.copy(),
            "characteristic_velocity": char_velocity,
            "mass_msun": mass_msun,
            "energy_foe": energy_foe,
            "calculated_mass_cgs": calculated_mass_cgs,
            "calculated_mass_msun": calculated_mass_msun,
        }

        if profile_params:
            step_result["profile_params"] = profile_params.copy()

        if step_number == 1:
            # First eruption: establishes baseline CSM
            self.current_csm_density = explosion_density.copy()
            step_result["csm_density"] = self.current_csm_density.copy()
            step_result["interaction_type"] = "first_eruption"
            self._print(f"  First eruption - establishes baseline CSM")

        else:
            # Subsequent eruptions: interact with current CSM
            if shell_config is not None:
                # Shell interaction occurs
                r_sh = shell_config["r_shell"]
                delta_r = shell_config["shell_width"]
                r_inner = r_sh - delta_r / 2
                r_outer = r_sh + delta_r / 2
                sigma = delta_r / 4.0

                self._print(f"  Shell interaction with correct piecewise physics:")
                self._print(f"    Shell center: {r_sh / 1e15:.2f} × 10¹⁵ cm")
                self._print(
                    f"    Shell boundaries: {r_inner / 1e15:.2f} - {r_outer / 1e15:.2f} × 10¹⁵ cm"
                )

                # Calculate swept masses
                mass_csm_swept = self.calculate_swept_mass(
                    self.current_csm_density, 0.0, r_outer
                )
                mass_new_swept = self.calculate_swept_mass(
                    explosion_density, r_inner, self.r_grid.max()
                )
                total_shell_mass = mass_csm_swept + mass_new_swept

                self._print(
                    f"    Total shell mass: {total_shell_mass / solar_mass:.4f} M☉"
                )

                # Create smooth Gaussian shell enhancement
                shell_enhancement, sigma = self.create_gaussian_shell(
                    total_shell_mass, r_sh, delta_r
                )

                # CORRECT PIECEWISE CONSTRUCTION:
                new_csm_density = np.zeros_like(self.r_grid)

                # Region 1: r < r_inner → NEW EXPLOSION (faster, inner material)
                inner_mask = self.r_grid < r_inner
                new_csm_density[inner_mask] = explosion_density[inner_mask]

                # Region 2: r_inner ≤ r ≤ r_outer → SHELL (Gaussian enhancement on base)
                shell_mask = (self.r_grid >= r_inner) & (self.r_grid <= r_outer)

                # Base for shell region: blend between new explosion (inner) and current CSM (outer)
                base_inner = explosion_density[shell_mask]
                base_outer = self.current_csm_density[shell_mask]

                # Linear blend within shell region
                r_shell_region = self.r_grid[shell_mask]
                blend_factor = (r_shell_region - r_inner) / delta_r
                shell_base = (1 - blend_factor) * base_inner + blend_factor * base_outer

                # Add Gaussian enhancement to base
                new_csm_density[shell_mask] = shell_base + shell_enhancement[shell_mask]

                # Region 3: r > r_outer → CURRENT CSM (previous material preserved)
                outer_mask = self.r_grid > r_outer
                new_csm_density[outer_mask] = self.current_csm_density[outer_mask]

                self._print(
                    f"    Piecewise: [New {profile_type} | Smooth shell | Previous CSM]"
                )
                self._print(
                    f"    Outer region properly steps down to previous material"
                )

                # Update current CSM
                self.current_csm_density = new_csm_density

                # Store results
                step_result.update(
                    {
                        "interaction_type": "correct_piecewise_gaussian",
                        "r_shell": r_sh,
                        "shell_width": delta_r,
                        "sigma": sigma,
                        "mass_csm_swept": mass_csm_swept,
                        "mass_new_swept": mass_new_swept,
                        "total_shell_mass": total_shell_mass,
                        "mass_csm_swept_msun": mass_csm_swept / solar_mass,
                        "mass_new_swept_msun": mass_new_swept / solar_mass,
                        "total_shell_mass_msun": total_shell_mass / solar_mass,
                        "shell_enhancement": shell_enhancement.copy(),
                        "csm_density": self.current_csm_density.copy(),
                        "previous_csm_density": (
                            self.step_results[-1]["csm_density"].copy()
                            if self.step_results
                            else None
                        ),
                    }
                )

            else:
                # No shell interaction
                self._print(
                    f"  No shell interaction - new explosion dominates where denser"
                )

                new_csm_density = np.maximum(
                    self.current_csm_density, explosion_density
                )
                self.current_csm_density = new_csm_density

                step_result.update(
                    {
                        "interaction_type": "no_shell",
                        "csm_density": self.current_csm_density.copy(),
                        "previous_csm_density": (
                            self.step_results[-1]["csm_density"].copy()
                            if self.step_results
                            else None
                        ),
                    }
                )

        self.step_results.append(step_result)
        return step_result

    def finalize_model(self):
        """
        Finalize the sequential CSM model and return complete results.

        Returns
        -------
        result : dict
            Complete sequential model results, including:
            - 'time_ref_days': expansion age supplied in days
            - 'time_ref_seconds': same expansion age in seconds (for Fortran coupling)
            - 'final_csm_density': final density profile sampled on the stored grids
        """
        if not self.step_results:
            raise ValueError("No eruptions added to model")

        # Calculate final mass conservation
        final_csm_mass_cgs = np.trapz(
            self.current_csm_density * 4 * np.pi * self.r_grid**2, self.r_grid
        )
        final_csm_mass_msun = final_csm_mass_cgs / solar_mass
        total_input_mass_msun = sum(step["mass_msun"] for step in self.step_results)

        self._print(f"\nFinal Results:")
        self._print(f"  Total input mass: {total_input_mass_msun:.3f} M☉")
        self._print(f"  Final CSM mass: {final_csm_mass_msun:.3f} M☉")
        self._print(
            f"  Mass conservation: {final_csm_mass_msun / total_input_mass_msun:.4f}"
        )

        # Count shells created
        n_shells = sum(1 for step in self.step_results if "r_shell" in step)
        self._print(f"  Shells created: {n_shells}")

        # Package results
        self.final_result = {
            "v_grid": self.v_grid,
            "r_grid": self.r_grid,
            "step_results": self.step_results,
            "final_csm_density": self.current_csm_density,
            "time_ref_days": self.time_ref_days,
            "time_ref_seconds": self.time_ref_seconds,
            "final_csm_mass_cgs": final_csm_mass_cgs,
            "final_csm_mass_msun": final_csm_mass_msun,
            "total_input_mass_msun": total_input_mass_msun,
            "n_shells": n_shells,
        }

        return self.final_result

    def create_model_from_sequence(self, eruption_sequence):
        """
        Create complete model from eruption sequence.

        Parameters
        ----------
        eruption_sequence : list of dict
            Each dict contains eruption parameters:
            - 'mass': explosion mass in solar masses
            - 'energy': explosion energy in foe (10^51 erg)
            - other parameters as needed

        Returns
        -------
        result : dict
            Complete sequential model results
        """
        self._print(
            "Sequential Multiple Eruption Shell Interaction Model (Extended Profile Support)"
        )
        self._print("Units: Mass in M☉, Energy in foe (10⁵¹ erg)")
        self._print("=" * 85)

        for eruption in eruption_sequence:
            mass_msun = eruption["mass"]
            energy_foe = eruption["energy"]
            label = eruption.get("label", None)
            profile_type = eruption.get("profile_type", "exponential")
            profile_params = eruption.get("profile_params", None)
            shell_config = eruption.get("shell_config", None)

            self.add_eruption(
                mass_msun, energy_foe, label, profile_type, profile_params, shell_config
            )

        return self.finalize_model()

    def plot_model(self, figsize=(20, 12), show_summary=None):
        """
        Plot the sequential CSM model development.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size (default: (20, 12))
        show_summary : bool, optional
            Whether to print plot summary. If None, uses self.verbose setting.
        """
        if self.final_result is None:
            raise ValueError("Model not finalized. Call finalize_model() first.")

        if show_summary is None:
            show_summary = self.verbose

        n_steps = len(self.step_results)

        # Create subplot layout
        if n_steps <= 3:
            n_cols = n_steps
            n_rows = 2
        else:
            n_cols = 3
            n_rows = (n_steps + 2) // 3 + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Convert radius to convenient units
        r_1e15 = self.r_grid / 1e15

        # Color scheme
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        # Plot each step
        for i, step_result in enumerate(self.step_results):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            label = step_result["label"]
            profile_type = step_result["profile_type"]
            interaction_type = step_result["interaction_type"]

            # Plot the explosion density for this step
            profile_label = f"{label} ({profile_type})"
            if profile_type == "broken_powerlaw" and "profile_params" in step_result:
                params = step_result["profile_params"]
                profile_label += (
                    f' δ={params.get("delta", 0):.1f},n={params.get("n", 7):.1f}'
                )

            ax.loglog(
                r_1e15,
                step_result["explosion_density"],
                color=colors[i],
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=profile_label,
            )

            # Plot previous CSM if it exists
            if (
                "previous_csm_density" in step_result
                and step_result["previous_csm_density"] is not None
            ):
                ax.loglog(
                    r_1e15,
                    step_result["previous_csm_density"],
                    color="gray",
                    linestyle=":",
                    linewidth=2,
                    alpha=0.5,
                    label="Previous CSM",
                )

            # Plot shell enhancement if it exists
            if "shell_enhancement" in step_result:
                shell_enhancement = step_result["shell_enhancement"]
                shell_mask = shell_enhancement > shell_enhancement.max() * 0.01
                if np.any(shell_mask):
                    ax.loglog(
                        r_1e15[shell_mask],
                        shell_enhancement[shell_mask],
                        color="green",
                        linewidth=3,
                        alpha=0.8,
                        label="Shell enhancement",
                    )

                    # Mark shell center and σ boundaries
                    r_shell = step_result["r_shell"] / 1e15
                    sigma = step_result["sigma"] / 1e15

                    ax.axvline(
                        r_shell, color="green", linestyle="-", alpha=0.8, linewidth=2
                    )
                    ax.axvline(
                        r_shell - sigma, color="green", linestyle="--", alpha=0.5
                    )
                    ax.axvline(
                        r_shell + sigma, color="green", linestyle="--", alpha=0.5
                    )

            # Plot resulting CSM
            ax.loglog(
                r_1e15,
                step_result["csm_density"],
                color="black",
                linewidth=3,
                label="Final CSM",
            )

            # Formatting
            ax.set_xlabel("Radius (10¹⁵ cm)")
            ax.set_ylabel("Density (g/cm³)")
            ax.set_title(
                f'Step {step_result["step_number"]}: {label}\n({profile_type}, {interaction_type.replace("_", " ")})'
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Add mass and profile information (now in convenient units)
            info_text = f'Mass: {step_result["calculated_mass_msun"]:.3f} M☉'
            if "total_shell_mass_msun" in step_result:
                info_text += f'\nShell: {step_result["total_shell_mass_msun"]:.4f} M☉'
                info_text += f'\nσ: {step_result["sigma"] / 1e15:.2f}×10¹⁵cm'

            ax.text(
                0.05,
                0.05,
                info_text,
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        # Plot final comparison in remaining subplot
        if n_steps < n_rows * n_cols:
            ax_final = axes[-1, -1]

            # Plot all explosion profiles
            for i, step_result in enumerate(self.step_results):
                profile_type = step_result["profile_type"]
                profile_label = f'{step_result["label"]} ({profile_type})'

                ax_final.loglog(
                    r_1e15,
                    step_result["explosion_density"],
                    color=colors[i],
                    linestyle="--",
                    alpha=0.4,
                    linewidth=1,
                    label=profile_label,
                )

            # Plot final CSM
            ax_final.loglog(
                r_1e15,
                self.final_result["final_csm_density"],
                color="black",
                linewidth=4,
                label="Final CSM",
            )

            # Mark all shell centers
            for step_result in self.step_results:
                if "r_shell" in step_result:
                    r_shell = step_result["r_shell"] / 1e15
                    ax_final.axvline(r_shell, color="red", linestyle=":", alpha=0.6)

            ax_final.set_xlabel("Radius (10¹⁵ cm)")
            ax_final.set_ylabel("Density (g/cm³)")
            ax_final.set_title(
                f'Final CSM Profile\n({self.final_result["n_shells"]} shells, mixed profiles)'
            )
            ax_final.legend(fontsize=8)
            ax_final.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_steps, n_rows * n_cols - 1):
            row = i // n_cols
            col = i % n_cols
            if row < n_rows and col < n_cols:
                axes[row, col].set_visible(False)

        plt.tight_layout()
        plt.show()

        # Print summary only if requested
        if show_summary:
            print(f"\nPlot Summary:")
            print(f"Total eruptions processed: {n_steps}")
            print(
                f"Profile types used: {[step['profile_type'] for step in self.step_results]}"
            )
            print(f"Gaussian shell interactions: {self.final_result['n_shells']}")
            print(
                f"Mass conservation: {self.final_result['final_csm_mass_msun'] / self.final_result['total_input_mass_msun']:.4f}"
            )

    def set_verbose(self, verbose):
        """
        Set verbose mode on or off.

        Parameters
        ----------
        verbose : bool
            Whether to print detailed output
        """
        self.verbose = verbose

    def get_final_csm_density(self):
        """Get the final CSM density profile."""
        if self.current_csm_density is None:
            raise ValueError("No CSM density available. Add eruptions first.")
        return self.current_csm_density.copy()

    def get_radius_grid(self):
        """Get the radius grid."""
        if self.r_grid is None:
            raise ValueError("No radius grid available. Add eruptions first.")
        return self.r_grid.copy()

    def get_velocity_grid(self):
        """Get the velocity grid."""
        if self.v_grid is None:
            raise ValueError("No velocity grid available. Add eruptions first.")
        return self.v_grid.copy()

    def get_step_results(self):
        """Get all step results."""
        return [step.copy() for step in self.step_results]

    def get_final_mass_msun(self):
        """Get final CSM mass in solar masses."""
        if self.final_result is None:
            raise ValueError("Model not finalized. Call finalize_model() first.")
        return self.final_result["final_csm_mass_msun"]

    def get_total_input_mass_msun(self):
        """Get total input mass in solar masses."""
        if not self.step_results:
            raise ValueError("No eruptions added to model")
        return sum(step["mass_msun"] for step in self.step_results)


# ---------------------------------------------------------------------------
# Dispatch registry: maps model name string -> (_get_lc_* function, [param names])
# param_names are extracted from kwargs in order by _call_csm.
# ---------------------------------------------------------------------------
_DISPATCH = {
    "wind_exponential": (
        _get_lc_wind_exponential,
        ["mdot", "vwind", "mexp", "eexp", "eff"],
    ),
    "wind_bpl": (
        _get_lc_wind_bpl,
        ["mdot", "vwind", "delta", "nn", "mexp", "eexp", "eff"],
    ),
    "exponential_wind": (
        _get_lc_exponential_wind,
        ["mexp", "eexp", "mdot", "vwind", "eff"],
    ),
    "bpl_wind": (
        _get_lc_bpl_wind,
        ["delta", "nn", "mexp", "eexp", "mdot", "vwind", "eff"],
    ),
    "exponential_exponential": (
        _get_lc_exponential_exponential,
        ["mexp", "eexp", "mexp_out", "eexp_out", "interval", "eff"],
    ),
    "exponential_bpl": (
        _get_lc_exponential_bpl,
        [
            "mexp",
            "eexp",
            "delta_out",
            "nn_out",
            "mexp_out",
            "eexp_out",
            "interval",
            "eff",
        ],
    ),
    "bpl_bpl": (
        _get_lc_bpl_bpl,
        [
            "delta",
            "nn",
            "mexp",
            "eexp",
            "delta_out",
            "nn_out",
            "mexp_out",
            "eexp_out",
            "interval",
            "eff",
        ],
    ),
    "bpl_exponential": (
        _get_lc_bpl_exponential,
        ["delta", "nn", "mexp", "eexp", "mexp_out", "eexp_out", "interval", "eff"],
    ),
    "boxwind_exponential": (
        _get_lc_boxwind_exponential,
        ["t1", "t2", "mdot_0", "mdot_1", "mdot_2", "vwind", "mexp", "eexp", "eff"],
    ),
    "boxwind_bpl": (
        _get_lc_boxwind_bpl,
        [
            "t1",
            "t2",
            "mdot_0",
            "mdot_1",
            "mdot_2",
            "vwind",
            "delta",
            "nn",
            "mexp",
            "eexp",
            "eff",
        ],
    ),
    "gausswind_exponential": (
        _get_lc_gausswind_exponential,
        [
            "t_peak",
            "t_width",
            "mdot_baseline",
            "mdot_peak",
            "vwind",
            "mexp",
            "eexp",
            "eff",
        ],
    ),
    "gausswind_bpl": (
        _get_lc_gausswind_bpl,
        [
            "t_peak",
            "t_width",
            "mdot_baseline",
            "mdot_peak",
            "vwind",
            "delta",
            "nn",
            "mexp",
            "eexp",
            "eff",
        ],
    ),
    "triple_powerlaw_wind_bpl": (
        _get_lc_triple_powerlaw_wind_bpl,
        [
            "t_break1",
            "t_break2",
            "mdot_0",
            "alpha1",
            "alpha2",
            "alpha3",
            "vwind",
            "delta",
            "nn",
            "mexp",
            "eexp",
            "eff",
        ],
    ),
    "triple_powerlaw_wind_exponential": (
        _get_lc_triple_powerlaw_wind_exponential,
        [
            "t_break1",
            "t_break2",
            "mdot_0",
            "alpha1",
            "alpha2",
            "alpha3",
            "vwind",
            "mexp",
            "eexp",
            "eff",
        ],
    ),
    "exponential_triple_powerlaw_wind": (
        _get_lc_exponential_triple_powerlaw_wind,
        [
            "mexp",
            "eexp",
            "t_break1",
            "t_break2",
            "mdot_0",
            "alpha1",
            "alpha2",
            "alpha3",
            "vwind",
            "eff",
        ],
    ),
    "bpl_triple_powerlaw_wind": (
        _get_lc_bpl_triple_powerlaw_wind,
        [
            "delta",
            "nn",
            "mexp",
            "eexp",
            "t_break1",
            "t_break2",
            "mdot_0",
            "alpha1",
            "alpha2",
            "alpha3",
            "vwind",
            "eff",
        ],
    ),
    "smooth_triple_powerlaw_wind_bpl": (
        _get_lc_smooth_triple_powerlaw_wind_bpl,
        [
            "t_break1",
            "t_break2",
            "mdot_0",
            "alpha1",
            "alpha2",
            "alpha3",
            "vwind",
            "delta",
            "nn",
            "mexp",
            "eexp",
            "eff",
        ],
    ),
    "smooth_triple_powerlaw_wind_exponential": (
        _get_lc_smooth_triple_powerlaw_wind_exponential,
        [
            "t_break1",
            "t_break2",
            "mdot_0",
            "alpha1",
            "alpha2",
            "alpha3",
            "vwind",
            "mexp",
            "eexp",
            "eff",
        ],
    ),
    "generic_csm_exponential": (
        _get_lc_generic_csm_exponential,
        [
            "base_density",
            "base_index",
            "shell1_radius",
            "shell1_width",
            "shell1_density",
            "shell2_radius",
            "shell2_width",
            "shell2_density",
            "shell3_radius",
            "shell3_width",
            "shell3_density",
            "interval_sn",
            "mej_sn",
            "esn",
            "eff",
        ],
    ),
    "generic_csm_bpl": (
        _get_lc_generic_csm_bpl,
        [
            "base_density",
            "base_index",
            "shell1_radius",
            "shell1_width",
            "shell1_density",
            "shell2_radius",
            "shell2_width",
            "shell2_density",
            "shell3_radius",
            "shell3_width",
            "shell3_density",
            "interval_sn",
            "delta_sn",
            "nn_sn",
            "mej_sn",
            "esn",
            "eff",
        ],
    ),
    "generic_4shell_csm_bpl": (
        _get_lc_generic_4shell_csm_bpl,
        [
            "base_density",
            "base_index",
            "shell1_radius",
            "shell1_width",
            "shell1_density",
            "shell2_radius",
            "shell2_width",
            "shell2_density",
            "shell3_radius",
            "shell3_width",
            "shell3_density",
            "shell4_radius",
            "shell4_width",
            "shell4_density",
            "interval_sn",
            "delta_sn",
            "nn_sn",
            "mej_sn",
            "esn",
            "eff",
        ],
    ),
    "generic_8shell_csm_bpl": (
        _get_lc_generic_8shell_csm_bpl,
        [
            "base_density",
            "base_index",
            "shell1_radius",
            "shell1_width",
            "shell1_density",
            "shell2_radius",
            "shell2_width",
            "shell2_density",
            "shell3_radius",
            "shell3_width",
            "shell3_density",
            "shell4_radius",
            "shell4_width",
            "shell4_density",
            "shell5_radius",
            "shell5_width",
            "shell5_density",
            "shell6_radius",
            "shell6_width",
            "shell6_density",
            "shell7_radius",
            "shell7_width",
            "shell7_density",
            "shell8_radius",
            "shell8_width",
            "shell8_density",
            "interval_sn",
            "delta_sn",
            "nn_sn",
            "mej_sn",
            "esn",
            "eff",
        ],
    ),
}


def _call_csm(csm_model, **kwargs):
    """
    Call the appropriate _get_lc_* function for the given model name.
    Positional arguments are popped from kwargs in the order defined in _DISPATCH.
    Remaining kwargs are passed through (e.g. kappa, time_ref).
    """
    if csm_model not in _DISPATCH:
        raise ValueError(
            f"Unknown CSM model '{csm_model}'. "
            f"Available models: {sorted(_DISPATCH.keys())}"
        )
    paper_mode = bool(kwargs.pop("paper_mode", False))
    efficiency_mode = kwargs.pop("efficiency_mode", None)
    _get_csm().lc_mod.set_model_mode(1 if paper_mode else 0)
    if efficiency_mode is not None:
        _get_csm().lc_mod.set_efficiency_mode(int(efficiency_mode))
    func, param_names = _DISPATCH[csm_model]
    args = [kwargs.pop(p) for p in param_names]
    return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Radio / synchrotron support
# ---------------------------------------------------------------------------

# Unit conversion shorthands used by density evaluators
_MSUN_CGS = solar_mass                                   # g
_MSUN_PER_YR_CGS = solar_mass / (365.25 * 24 * 3600)    # g/s per (Msun/yr)
_FOE = 1e51                                              # erg
_MP  = 1.6726e-24                                        # g


def _rho_wind_at_r(r_array, **kwargs):
    """CSM density for a simple steady wind: rho = mdot / (4 pi r^2 vwind)."""
    mdot_cgs  = kwargs['mdot'] * _MSUN_PER_YR_CGS        # g/s
    vwind_cgs = kwargs['vwind'] * 1e5                     # cm/s
    return mdot_cgs / (4.0 * np.pi * r_array ** 2 * vwind_cgs)


def _rho_exponential_at_r(r_array, lc, **kwargs):
    """
    CSM density for homologously expanding exponential ejecta evaluated at radii r_array.

    rho(r, t) = M / (8 pi (v0 t_csm)^3) * exp(-r / (v0 t_csm))
    v0 = sqrt(E / (6 M))

    t_csm = lc.time + interval * DAY  — the CSM eruption started 'interval' days
    before the SN, so by the time t has elapsed since the SN, the ejecta age is
    t + interval.
    """
    mexp_cgs = kwargs['mexp'] * _MSUN_CGS
    eexp_cgs = kwargs['eexp'] * _FOE
    v0 = np.sqrt(eexp_cgs / (6.0 * mexp_cgs))
    interval_s = kwargs.get('interval', 0.0) * DAY       # seconds
    t_csm = lc.time + interval_s                         # total CSM age in seconds
    return mexp_cgs / (8.0 * np.pi * (v0 * t_csm) ** 3) * np.exp(-r_array / (v0 * t_csm))


def _rho_bpl_at_r(r_array, lc, **kwargs):
    """
    CSM density for BPL ejecta at radii r_array.

    The BPL ejecta profile in velocity space is a broken power law with
    indices delta (inner) and nn (outer), normalised to (mexp, eexp).
    We evaluate the density at homologous coordinates v = r/t.
    """
    mexp_cgs = kwargs['mexp'] * _MSUN_CGS
    eexp_cgs = kwargs['eexp'] * _FOE
    delta = kwargs['delta']
    nn    = kwargs['nn']
    interval_s = kwargs.get('interval', 0.0) * DAY       # seconds
    t_csm = lc.time + interval_s                         # total CSM age in seconds

    # Transition (break) velocity — same formula as the Fortran get_bpl_coeffs:
    #   v_t = sqrt( 2*(5-delta)*(nn-5)*E / ((3-delta)*(nn-3)*M) )
    num = 2.0 * (5.0 - delta) * (nn - 5.0) * eexp_cgs
    den = (3.0 - delta) * (nn - 3.0) * mexp_cgs
    if nn <= 5.0 or delta >= 3.0 or num <= 0.0 or den <= 0.0:
        return _rho_exponential_at_r(r_array, lc, **kwargs)
    v_t = np.sqrt(num / den)

    # Normalisation: rho_0 from mass integral 4pi int rho v^2 dv = mexp
    rho_0 = mexp_cgs / (4.0 * np.pi * v_t ** 3 * (1.0 / (3.0 - delta) + 1.0 / (nn - 3.0)))

    v_r = r_array / t_csm                               # homologous velocity at shock
    rho = np.where(
        v_r < v_t,
        rho_0 / t_csm ** 3 * (v_r / v_t) ** (-delta),
        rho_0 / t_csm ** 3 * (v_r / v_t) ** (-nn),
    )
    return rho


def _rho_variable_wind_at_r(model_name, r_array, lc, **kwargs):
    """
    CSM density for variable-wind models (gausswind, boxwind, triple_powerlaw_wind).

    The density profile is rho = mdot(t_wind) / (4 pi r^2 vwind), but mdot is
    a function of the wind emission time t_wind = r / vwind (i.e. the time the
    wind parcel was emitted in order to be at radius r today).

    We reconstruct the same mdot(t) array that was passed to the Fortran, then
    evaluate it at t_wind = r / vwind for each shock radius.
    """
    vwind_cgs = kwargs['vwind'] * 1e5    # cm/s
    n_points  = kwargs.get('n_points', 50)

    if model_name in ('gausswind_exponential', 'gausswind_bpl'):
        t_peak        = kwargs['t_peak']
        t_width       = kwargs['t_width']
        mdot_baseline = kwargs['mdot_baseline']
        mdot_peak     = kwargs['mdot_peak']
        t_start = t_peak - 4 * t_width
        t_end   = t_peak + 4 * t_width
        tgrid_yr = np.linspace(t_start, t_end, n_points)
        gauss = np.exp(-0.5 * ((tgrid_yr - t_peak) / t_width) ** 2)
        mdot_yr = mdot_baseline + (mdot_peak - mdot_baseline) * gauss

    elif model_name in ('boxwind_exponential', 'boxwind_bpl'):
        t1    = kwargs['t1']
        t2    = kwargs['t2']
        mdot_0 = kwargs['mdot_0']
        mdot_1 = kwargs['mdot_1']
        mdot_2 = kwargs['mdot_2']
        tgrid_yr = np.array([t1, t1, t2, t2])
        mdot_yr  = np.array([mdot_0, mdot_1, mdot_1, mdot_2])

    elif model_name in ('triple_powerlaw_wind_bpl', 'triple_powerlaw_wind_exponential',
                        'exponential_triple_powerlaw_wind', 'bpl_triple_powerlaw_wind',
                        'smooth_triple_powerlaw_wind_bpl', 'smooth_triple_powerlaw_wind_exponential'):
        t_break1 = kwargs['t_break1']
        t_break2 = kwargs['t_break2']
        mdot_0   = kwargs['mdot_0']
        alpha1   = kwargs['alpha1']
        alpha2   = kwargs['alpha2']
        alpha3   = kwargs['alpha3']
        t_start = 0.1
        t_end   = max(t_break2 * 2, 10)
        tgrid_yr = np.logspace(np.log10(t_start), np.log10(t_end), n_points)
        mdot_yr = np.zeros_like(tgrid_yr)
        m1 = tgrid_yr < t_break1
        mdot_yr[m1] = mdot_0 * (tgrid_yr[m1] / 1.0) ** alpha1
        mdot_b1 = mdot_0 * (t_break1 / 1.0) ** alpha1
        m2 = (tgrid_yr >= t_break1) & (tgrid_yr < t_break2)
        mdot_yr[m2] = mdot_b1 * (tgrid_yr[m2] / t_break1) ** alpha2
        mdot_b2 = mdot_b1 * (t_break2 / t_break1) ** alpha2
        m3 = tgrid_yr >= t_break2
        mdot_yr[m3] = mdot_b2 * (tgrid_yr[m3] / t_break2) ** alpha3
    else:
        raise ValueError(f"_rho_variable_wind_at_r: unknown model '{model_name}'")

    # Convert to CGS
    tgrid_s  = tgrid_yr * 365.25 * 24 * 3600   # yr → s
    mdot_cgs = mdot_yr * _MSUN_PER_YR_CGS      # Msun/yr → g/s

    # Build an interpolator: mdot as a function of time (seconds)
    from scipy.interpolate import interp1d as _interp1d_loc
    mdot_interp = _interp1d_loc(
        tgrid_s, mdot_cgs,
        bounds_error=False,
        fill_value=(mdot_cgs[0], mdot_cgs[-1]),
    )

    # Wind emission time for a parcel at radius r: t_wind = r / vwind
    t_wind = r_array / vwind_cgs              # seconds
    mdot_at_r = mdot_interp(t_wind)
    rho = mdot_at_r / (4.0 * np.pi * r_array ** 2 * vwind_cgs)
    return rho


def _rho_generic_at_r(r_array, **kwargs):
    """
    CSM density for generic-shell models, evaluated by re-building the same
    density grid that was passed to the Fortran and interpolating at r_array.
    """
    r_inner = kwargs.get('r_inner', 1e10)
    r_outer = kwargs.get('r_outer', 1e20)
    base_density = kwargs['base_density']
    base_index   = kwargs['base_index']

    # Collect shell parameters (up to 8 shells)
    shell_radii     = []
    shell_widths    = []
    shell_densities = []
    for i in range(1, 9):
        d = kwargs.get(f'shell{i}_density', 0.0)
        if d > 0:
            shell_radii.append(kwargs[f'shell{i}_radius'])
            shell_widths.append(kwargs[f'shell{i}_width'])
            shell_densities.append(d)

    n_shells = len(shell_radii)
    r_grid, _, csm_density = create_generic_csm_density(
        r_inner=r_inner,
        r_outer=r_outer,
        n_points=1000,
        base_density=base_density,
        base_index=base_index,
        n_shells=n_shells,
        shell_radii=shell_radii if n_shells > 0 else None,
        shell_widths=shell_widths if n_shells > 0 else None,
        shell_densities=shell_densities if n_shells > 0 else None,
        shell_profiles='gaussian',
        time_ref_days=kwargs.get('interval_sn', 10 * YEAR_DAYS),
    )

    from scipy.interpolate import interp1d as _interp1d_loc
    rho_interp = _interp1d_loc(
        r_grid, csm_density,
        bounds_error=False,
        fill_value=(csm_density[0], csm_density[-1]),
    )
    return rho_interp(r_array)


# Maps each model name to the type of upstream density estimator it needs.
# 'wind'            — simple steady wind: rho = mdot/(4 pi r^2 vwind)
# 'exponential'     — exponential ejecta outer CSM
# 'bpl'             — broken power-law ejecta outer CSM
# 'variable_wind'   — time-varying wind (gausswind / boxwind / triple_powerlaw)
# 'generic'         — generic shell model (density grid interpolation)
_CSM_DENSITY_TYPE = {
    'wind_exponential':                    'wind',
    'wind_bpl':                            'wind',
    'exponential_wind':                    'wind',
    'bpl_wind':                            'wind',
    'gausswind_exponential':               'variable_wind',
    'gausswind_bpl':                       'variable_wind',
    'boxwind_exponential':                 'variable_wind',
    'boxwind_bpl':                         'variable_wind',
    'triple_powerlaw_wind_bpl':            'variable_wind',
    'triple_powerlaw_wind_exponential':    'variable_wind',
    'exponential_triple_powerlaw_wind':    'variable_wind',
    'bpl_triple_powerlaw_wind':            'variable_wind',
    'smooth_triple_powerlaw_wind_bpl':     'variable_wind',
    'smooth_triple_powerlaw_wind_exponential': 'variable_wind',
    'exponential_exponential':             'exponential',
    'exponential_bpl':                     'exponential',
    'bpl_bpl':                             'bpl',
    'bpl_exponential':                     'bpl',
    'generic_csm_exponential':             'generic',
    'generic_csm_bpl':                     'generic',
    'generic_4shell_csm_bpl':              'generic',
    'generic_8shell_csm_bpl':              'generic',
}


def _get_rho_csm_at_shock(csm_model, lc, **kwargs):
    """
    Return the upstream CSM mass density (g/cm^3) at the shock position for each
    time step in lc, using the analytic or reconstructed density profile that
    corresponds to csm_model.

    Parameters
    ----------
    csm_model : str
        CSM model name (key in _DISPATCH).
    lc : namedtuple
        Output from _call_csm — must have lc.rph (shock radius in cm) and lc.time (s).
    **kwargs
        All model parameters (same dict that was passed to _call_csm, before popping).

    Returns
    -------
    rho : ndarray, shape (len(lc.time),)
        Upstream density in g/cm^3.
    """
    r_sh = lc.rph                     # shock radius, cm, same grid as lc.time

    density_type = _CSM_DENSITY_TYPE.get(csm_model)
    if density_type is None:
        raise ValueError(
            f"_get_rho_csm_at_shock: unknown model '{csm_model}'. "
            f"Expected one of: {sorted(_CSM_DENSITY_TYPE.keys())}"
        )

    if density_type == 'wind':
        return _rho_wind_at_r(r_sh, **kwargs)

    elif density_type == 'exponential':
        return _rho_exponential_at_r(r_sh, lc, **kwargs)

    elif density_type == 'bpl':
        return _rho_bpl_at_r(r_sh, lc, **kwargs)

    elif density_type == 'variable_wind':
        return _rho_variable_wind_at_r(csm_model, r_sh, lc, **kwargs)

    elif density_type == 'generic':
        return _rho_generic_at_r(r_sh, **kwargs)

    else:
        raise ValueError(f"Unhandled density type '{density_type}'")


def _call_csm_radio(csm_model, redshift, logepsb, logepse, p, frequency,
                    luminosity_distance_cm, **kwargs):
    """
    Run the CSM Fortran model, compute the upstream density at the shock, and
    return synchrotron flux density in mJy on the Fortran time grid.

    Parameters
    ----------
    csm_model : str
    redshift : float
    logepsb : float
        log10(epsilon_B)
    logepse : float
        log10(epsilon_e)
    p : float
        Electron power-law index
    frequency : float or array
        Observer-frame frequency in Hz
    luminosity_distance_cm : float
    **kwargs
        All physical parameters for the CSM model (passed through to _call_csm
        and also used by the density evaluators).

    Returns
    -------
    time_days : ndarray
        Time in observer-frame days (Fortran grid).
    flux_mJy : ndarray
        Radio flux density in mJy.
    """
    from redback_csm.radio import synchrotron_flux_density

    # Keep a copy of kwargs for density evaluation (before _call_csm pops them)
    kwargs_density = dict(kwargs)

    lc = _call_csm(csm_model, **kwargs)

    rho = _get_rho_csm_at_shock(csm_model, lc, **kwargs_density)

    flux_mJy = synchrotron_flux_density(
        time_days=lc.time / DAY,
        vshell_cgs=lc.vshell,
        rho_csm_cgs=rho,
        redshift=redshift,
        logepsb=logepsb,
        logepse=logepse,
        p=p,
        frequency=frequency,
        luminosity_distance_cm=luminosity_distance_cm,
    )
    return lc.time / DAY, flux_mJy
