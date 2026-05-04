"""
01_bolometric.py — Bolometric lightcurves from CSM interaction.

Demonstrates:
  - wind_bpl_bolometric: plain shock luminosity vs diffuse (photon-diffusion) luminosity
  - Comparing three wind mass-loss rates on the same axes
  - Using gausswind_bpl_bolometric for a time-variable wind

Run:
    python examples/01_bolometric.py
"""

import numpy as np
import matplotlib.pyplot as plt

from redback_csm.models import wind_bpl_bolometric, gausswind_bpl_bolometric

# Time grid in source-frame days
time = np.geomspace(1, 1000, 300)

# ---------------------------------------------------------------------------
# Panel 1: Steady wind — shock vs diffuse luminosity
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
for mdot, color in [(1e-4, "royalblue"), (1e-3, "darkorange"), (1e-2, "crimson")]:
    # Without kappa: returns instantaneous shock luminosity
    lbol_shock = wind_bpl_bolometric(
        time=time, mdot=mdot, vwind=100.0,
        delta=0.5, nn=12.0, mexp=5.0, eexp=1.0, eff=0.5,
    )
    # With kappa: returns photon-diffusion (observed) luminosity
    lbol_diff = wind_bpl_bolometric(
        time=time, mdot=mdot, vwind=100.0,
        delta=0.5, nn=12.0, mexp=5.0, eexp=1.0, eff=0.5,
        kappa=0.34,
    )
    label = rf"$\dot{{M}}=10^{{{int(np.log10(mdot))}}}\ M_\odot\,\mathrm{{yr}}^{{-1}}$"
    ax.plot(time, lbol_shock, color=color, ls="--", alpha=0.5)
    ax.plot(time, lbol_diff,  color=color, ls="-",  label=label)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time (days)", fontsize=12)
ax.set_ylabel(r"$L_{\rm bol}$ (erg s$^{-1}$)", fontsize=12)
ax.set_title("Steady wind CSM (solid = diffuse, dashed = shock)", fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(1, 1000)

# ---------------------------------------------------------------------------
# Panel 2: Gaussian wind — variable mass-loss history
# ---------------------------------------------------------------------------
ax = axes[1]
for t_peak, color, label in [
    (5.0,  "teal",      r"$t_{\rm peak}=5$ yr"),
    (20.0, "purple",    r"$t_{\rm peak}=20$ yr"),
    (50.0, "saddlebrown", r"$t_{\rm peak}=50$ yr"),
]:
    lbol = gausswind_bpl_bolometric(
        time=time,
        t_peak=t_peak, t_width=2.0,
        mdot_baseline=1e-5, mdot_peak=1e-2,
        vwind=100.0, delta=0.5, nn=12.0,
        mexp=5.0, eexp=1.0, eff=0.5,
        kappa=0.34,
    )
    ax.plot(time, lbol, color=color, label=label)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time (days)", fontsize=12)
ax.set_ylabel(r"$L_{\rm bol}$ (erg s$^{-1}$)", fontsize=12)
ax.set_title(r"Gaussian wind CSM ($\dot{M}_{\rm peak}=10^{-2}$, $\sigma=2$ yr)", fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(1, 1000)

fig.tight_layout()
fig.savefig("examples/bolometric_lightcurves.png", dpi=150)
print("Saved: examples/bolometric_lightcurves.png")
