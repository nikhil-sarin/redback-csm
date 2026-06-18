"""
02_multiband.py — Multiband (per-filter) lightcurves from CSM interaction.

Demonstrates:
  - wind_bpl with output_format='flux_density' for a specific frequency
  - wind_bpl with output_format='magnitude' for photometric bands
  - Plotting g, r, i band lightcurves

Requires redback to be installed (for the band/filter infrastructure).

Run:
    python examples/02_multiband.py
"""

import numpy as np
import matplotlib.pyplot as plt

from redback_csm.models import wind_bpl

# Observer-frame time array
time = np.geomspace(1, 500, 300)
redshift = 0.02

# ---------------------------------------------------------------------------
# Panel 1: Flux density at a single optical frequency (r-band ~ 4.8e14 Hz)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
freq_r = 4.8e14   # Hz, roughly r-band

for mdot, color in [(1e-4, "royalblue"), (1e-3, "darkorange"), (1e-2, "crimson")]:
    flux = wind_bpl(
        time=time,
        redshift=redshift,
        mdot=mdot, vwind=100.0,
        delta=0.5, nn=12.0, mexp=5.0, eexp=1.0, eff=0.5,
        temperature_floor=3000.0, kappa=0.1,
        frequency=np.full_like(time, freq_r),
        output_format="flux_density",
    )
    label = rf"$\dot{{M}}=10^{{{int(np.log10(mdot))}}}\ M_\odot\,\mathrm{{yr}}^{{-1}}$"
    ax.plot(time, flux, color=color, label=label)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time (observer-frame days)", fontsize=12)
ax.set_ylabel("Flux density (mJy)", fontsize=12)
ax.set_title(r"$r$-band flux density ($z=0.02$)", fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(1, 500)

# ---------------------------------------------------------------------------
# Panel 2: g, r, i magnitudes for a single model
# ---------------------------------------------------------------------------
ax = axes[1]

# Broadband filters available through redback's sncosmo bandpass registry
bands = ["sdssg", "sdssr", "sdssi"]
colors = ["steelblue", "darkorange", "firebrick"]
labels = ["g", "r", "i"]

# Call once per band — redback expects a single band name per call
for band, color, label in zip(bands, colors, labels):
    mag = wind_bpl(
        time=time,
        redshift=redshift,
        mdot=1e-3, vwind=100.0,
        delta=0.5, nn=12.0, mexp=5.0, eexp=1.0, eff=0.5,
        temperature_floor=3000.0, kappa=0.1,
        bands=band,
        output_format="magnitude",
    )
    good = np.isfinite(mag) & (mag < 40)
    ax.plot(time[good], mag[good], color=color, label=label)

ax.invert_yaxis()
ax.set_xlabel("Time (observer-frame days)", fontsize=12)
ax.set_ylabel("AB magnitude", fontsize=12)
ax.set_title(r"$gri$ magnitudes, wind BPL ($\dot{M}=10^{-3}$, $z=0.02$)", fontsize=11)
ax.legend(fontsize=9)
ax.set_xscale("log")
ax.set_xlim(1, 500)

fig.tight_layout()
fig.savefig("examples/multiband_lightcurves.png", dpi=150)
print("Saved: examples/multiband_lightcurves.png")
