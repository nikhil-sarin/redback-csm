"""
04_radio.py — Radio synchrotron lightcurves from CSM interaction.

Demonstrates:
  - Multi-frequency radio lightcurves for a steady wind
  - The effect of CSM geometry on the radio light curve shape:
      * Steady wind (smooth, featureless: rho ~ r^-2)
      * Single eruption (bump when shock crosses the shell)
      * Two eruptions (two bumps at different times)
  - Effect of microphysical parameters epsilon_B and p

Run:
    python examples/04_radio.py
"""

import numpy as np
import matplotlib.pyplot as plt

from redback_csm.models import wind_bpl_radio, gausswind_bpl_radio

# Observer-frame time (days)
time = np.geomspace(5, 2000, 400)
redshift = 0.01

# Shared ejecta and efficiency parameters
ejecta_kw = dict(delta=0.5, nn=12.0, mexp=5.0, eexp=1.0, eff=0.5)

# Shared synchrotron microphysics
sync_kw = dict(logepsb=-2.0, logepse=-1.0, p=3.0)

# ---------------------------------------------------------------------------
# Panel 1: Multiple radio frequencies — steady wind
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

ax = axes[0, 0]
freqs  = [1.4e9, 5e9, 10e9, 33e9]
labels = ["1.4 GHz", "5 GHz", "10 GHz", "33 GHz"]
colors = ["purple", "royalblue", "darkorange", "firebrick"]

for freq, label, color in zip(freqs, labels, colors):
    flux = wind_bpl_radio(
        time=time, redshift=redshift,
        mdot=1e-3, vwind=100.0,
        frequency=freq,
        **ejecta_kw, **sync_kw,
    )
    ax.plot(time, flux, color=color, label=label)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time (observer days)", fontsize=11)
ax.set_ylabel("Flux density (mJy)", fontsize=11)
ax.set_title(r"Steady wind ($\dot{M}=10^{-3}$), multi-frequency", fontsize=10)
ax.legend(fontsize=9)
ax.set_xlim(5, 2000)

# ---------------------------------------------------------------------------
# Panel 2: CSM geometry — smooth wind vs single eruption vs two eruptions
#
# The key: gausswind places a dense shell at r = v_wind * t_peak.
# The shock hits it at t_encounter ~ r_shell / v_avg_shock.
# With vwind=100 km/s and t_peak=5 yr:  r_shell ~ 1.5e15 cm, encountered ~25 d
# With t_peak=30 yr: r_shell ~ 9e15 cm, encountered ~300 d
# ---------------------------------------------------------------------------
ax = axes[0, 1]

# Steady wind baseline
flux_wind = wind_bpl_radio(
    time=time, redshift=redshift,
    mdot=1e-5, vwind=100.0, frequency=5e9,
    **ejecta_kw, **sync_kw,
)

# Single eruption: sharp mass-loss event 5 years before explosion
# Creates a shell at r = 100 km/s * 5 yr ~ 1.6e15 cm
flux_1erupt = gausswind_bpl_radio(
    time=time, redshift=redshift,
    t_peak=5.0, t_width=0.5,
    mdot_baseline=1e-6, mdot_peak=0.1,
    vwind=100.0, frequency=5e9,
    **ejecta_kw, **sync_kw,
)

# Two eruptions: earlier event at 30 yr, later at 5 yr
# Use two gausswind calls and sum (superposition is approximate but illustrative)
flux_early = gausswind_bpl_radio(
    time=time, redshift=redshift,
    t_peak=30.0, t_width=2.0,
    mdot_baseline=1e-6, mdot_peak=0.1,
    vwind=100.0, frequency=5e9,
    **ejecta_kw, **sync_kw,
)
flux_late = gausswind_bpl_radio(
    time=time, redshift=redshift,
    t_peak=5.0, t_width=0.5,
    mdot_baseline=1e-6, mdot_peak=0.05,
    vwind=100.0, frequency=5e9,
    **ejecta_kw, **sync_kw,
)

ax.plot(time, flux_wind,   "k",          ls="--", lw=1.2, label="Steady wind (baseline)")
ax.plot(time, flux_1erupt, "darkorange",           label="Single eruption (5 yr ago)")
ax.plot(time, flux_early,  "steelblue",  ls=":",  label="Two eruptions (5 and 30 yr ago)")
ax.plot(time, flux_late,   "steelblue",  ls=":",  alpha=0.4)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time (observer days)", fontsize=11)
ax.set_ylabel("Flux density (mJy)", fontsize=11)
ax.set_title("CSM geometry — bumps from eruption shells (5 GHz)", fontsize=10)
ax.legend(fontsize=9)
ax.set_xlim(5, 2000)

# ---------------------------------------------------------------------------
# Panel 3: Effect of epsilon_B at 33 GHz (optically thin — clean scaling)
# ---------------------------------------------------------------------------
ax = axes[1, 0]

for logepsb, color in [(-1.0, "firebrick"), (-2.0, "darkorange"), (-3.0, "steelblue")]:
    flux = wind_bpl_radio(
        time=time, redshift=redshift,
        mdot=1e-3, vwind=100.0, frequency=3.3e10,
        logepsb=logepsb, logepse=-1.0, p=3.0,
        **ejecta_kw,
    )
    ax.plot(time, flux, color=color,
            label=rf"$\log\epsilon_B={logepsb:.0f}$")

ax.text(0.97, 0.05,
        r"$F_\nu \propto \epsilon_B^{(p+1)/4}$" + "\n" + r"$= \epsilon_B^1$ for $p=3$",
        ha="right", va="bottom", transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time (observer days)", fontsize=11)
ax.set_ylabel("Flux density (mJy)", fontsize=11)
ax.set_title(r"$\epsilon_B$ dependence at 33 GHz (optically thin)", fontsize=10)
ax.legend(fontsize=9)
ax.set_xlim(5, 2000)

# ---------------------------------------------------------------------------
# Panel 4: Effect of electron power-law index p
# ---------------------------------------------------------------------------
ax = axes[1, 1]

for p, color in [(2.2, "firebrick"), (3.0, "darkorange"), (3.5, "steelblue")]:
    flux = wind_bpl_radio(
        time=time, redshift=redshift,
        mdot=1e-3, vwind=100.0, frequency=5e9,
        logepsb=-2.0, logepse=-1.0, p=p,
        **ejecta_kw,
    )
    ax.plot(time, flux, color=color, label=rf"$p={p}$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time (observer days)", fontsize=11)
ax.set_ylabel("Flux density (mJy)", fontsize=11)
ax.set_title(r"Electron spectral index $p$ at 5 GHz", fontsize=10)
ax.legend(fontsize=9)
ax.set_xlim(5, 2000)

fig.tight_layout()
fig.savefig("examples/radio_lightcurves.png", dpi=150)
print("Saved: examples/radio_lightcurves.png")
