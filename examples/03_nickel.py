"""
03_nickel.py — CSM interaction + radioactive nickel/cobalt decay.

Demonstrates:
  - wind_bpl_nickel_bolometric vs wind_bpl_bolometric (CSM only)
  - How the nickel tail dominates at late times
  - Varying the nickel mass fraction

The nickel component uses the Arnett (1982) diffusion process. The nickel mass
is f_nickel times the SN ejecta mass; finite CSM constructors add the CSM shell
mass to the Arnett diffusion mass.

Run:
    python examples/03_nickel.py
"""

import numpy as np
import matplotlib.pyplot as plt

from redback_csm.models import wind_bpl_bolometric, wind_bpl_nickel_bolometric

time = np.geomspace(1, 500, 300)

# Shared CSM + ejecta parameters
csm_kw = dict(
    mdot=5e-4, vwind=100.0,
    delta=0.5, nn=12.0,
    mexp=5.0, eexp=1.0, eff=0.5,
)
# Arnett nickel parameters (kappa shared with CSM diffusion)
nickel_kw = dict(
    kappa=0.34,
    kappa_gamma=10.0,
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ---------------------------------------------------------------------------
# Panel 1: CSM-only vs CSM + nickel for a range of f_nickel
# ---------------------------------------------------------------------------
ax = axes[0]

lbol_csm = wind_bpl_bolometric(time=time, kappa=0.34, **csm_kw)
ax.plot(time, lbol_csm, "k--", lw=2, label="CSM only")

for f_ni, color in [(0.01, "steelblue"), (0.05, "darkorange"), (0.15, "crimson")]:
    lbol = wind_bpl_nickel_bolometric(
        time=time,
        f_nickel=f_ni,
        **csm_kw, **nickel_kw,
    )
    ax.plot(time, lbol, color=color,
            label=rf"$f_{{\rm Ni}}={f_ni:.2f}$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time (days, source frame)", fontsize=12)
ax.set_ylabel(r"$L_{\rm bol}$ (erg s$^{-1}$)", fontsize=12)
ax.set_title("CSM interaction + nickel decay", fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(1, 500)

# ---------------------------------------------------------------------------
# Panel 2: Decomposing the two contributions
#   Total = CSM + Ni; isolate Ni as Total - CSM
# ---------------------------------------------------------------------------
ax = axes[1]

f_ni = 0.05
lbol_csm  = wind_bpl_bolometric(time=time, kappa=0.34, **csm_kw)
lbol_both = wind_bpl_nickel_bolometric(
    time=time, f_nickel=f_ni,
    **csm_kw, **nickel_kw,
)
lbol_ni = np.clip(lbol_both - lbol_csm, 1e38, None)  # nickel contribution

ax.plot(time, lbol_csm,  "royalblue",  ls="--", label="CSM shock")
ax.plot(time, lbol_ni,   "firebrick",  ls=":",  label=r"Ni/Co decay ($f_{\rm Ni}=0.05$)")
ax.plot(time, lbol_both, "k",          ls="-",  lw=2, label="Total")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time (days, source frame)", fontsize=12)
ax.set_ylabel(r"$L_{\rm bol}$ (erg s$^{-1}$)", fontsize=12)
ax.set_title("CSM vs nickel contribution", fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(1, 500)

fig.tight_layout()
fig.savefig("examples/nickel_lightcurves.png", dpi=150)
print("Saved: examples/nickel_lightcurves.png")
plt.show()
