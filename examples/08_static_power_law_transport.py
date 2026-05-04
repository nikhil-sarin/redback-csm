"""Static finite power-law CSM transport light curves.

This example reproduces the compact/extended finite power-law CSM setup used
for transport-mode validation, but with a release-facing name and output.

Run:
    MPLBACKEND=Agg python examples/08_static_power_law_transport.py
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

from redback_csm.models import generic_powerlaw_csm_bpl_bolometric


R_SUN = 6.957e10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="examples/static_power_law_transport.png")
    p.add_argument("--efficiency-mode", type=int, default=0)
    p.add_argument("--n-rad-zones", type=int, default=40)
    return p.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    time = np.geomspace(1.0, 300.0, 500)
    slopes = [0.0, 0.5, 1.0, 1.5, 2.0]
    colors = ["#3b0f70", "#7f1d8d", "#c2417a", "#f07a5a", "#f3bf74"]
    panels = [
        {
            "title": r"Compact CSM: $R_{\rm out}=5\times10^3\,R_\odot$",
            "r_outer": 5.0e3 * R_SUN,
            "ylim": (1e42, 6e45),
        },
        {
            "title": r"Extended CSM: $R_{\rm out}=5\times10^4\,R_\odot$",
            "r_outer": 5.0e4 * R_SUN,
            "ylim": (1e41, 1.3e45),
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8))

    for ax, panel in zip(axes, panels):
        for s, color in zip(slopes, colors):
            lbol = generic_powerlaw_csm_bpl_bolometric(
                time=time,
                eta=-s,
                r_inner=5.0e2 * R_SUN,
                r_outer=panel["r_outer"],
                delta_sn=1.0,
                nn_sn=10.0,
                mej_sn=5.0,
                esn=1.0,
                eff=1.0,
                m_csm=1.0,
                mode="transport",
                efficiency_mode=args.efficiency_mode,
                kappa=0.2,
                n_rad_zones=args.n_rad_zones,
            )
            ax.plot(time, lbol, color=color, lw=2.7, label=fr"$s={s:g}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.0, 300.0)
        ax.set_ylim(panel["ylim"])
        ax.set_xlabel("Time (days)")
        ax.set_ylabel(r"$L_{\rm bol}$ (erg s$^{-1}$)")
        ax.set_title(panel["title"])
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.18, which="both")

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
