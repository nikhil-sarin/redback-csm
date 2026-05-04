from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

from redback_csm.models import generic_powerlaw_csm_bpl_bolometric

R_SUN = 6.957e10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="fig_from_paper_static.png")
    p.add_argument("--efficiency-mode", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    time = np.geomspace(1.0, 300.0, 500)
    slopes = [0.0, 0.5, 1.0, 1.5, 2.0]
    colors = ["#3b0f70", "#7f1d8d", "#c2417a", "#f07a5a", "#f3bf74"]
    panels = [
        {
            "title": r"Compact CSM: $R_{\rm csm,out}=5\times10^3\,R_\odot$",
            "r_outer": 5.0e3 * R_SUN,
            "xlim": (1.0, 300.0),
            "ylim": (1e42, 6e45),
            "n_rad_zones": 40,
        },
        {
            "title": r"Extended CSM: $R_{\rm csm,out}=5\times10^4\,R_\odot$",
            "r_outer": 5.0e4 * R_SUN,
            "xlim": (1.0, 300.0),
            "ylim": (1e41, 1.3e45),
            "n_rad_zones": 40,
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14.14, 4.72))

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
                n_rad_zones=panel["n_rad_zones"],
            )
            ax.plot(time, lbol, color=color, lw=3.0, label=fr"$s={s:g}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(panel["xlim"])
        ax.set_ylim(panel["ylim"])
        ax.set_xlabel("Time (days)", fontsize=18)
        ax.set_title(panel["title"], fontsize=16)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax.tick_params(axis="both", which="major", labelsize=12, length=12, width=2)
        ax.tick_params(axis="both", which="minor", length=6, width=1.8)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.legend(loc="upper right", fontsize=12, frameon=True, edgecolor="0.6")

    axes[0].set_ylabel(r"Bolometric Luminosity (erg s$^{-1}$)", fontsize=20)
    axes[1].set_ylabel(r"Bolometric Luminosity (erg s$^{-1}$)", fontsize=20)
    fig.tight_layout(w_pad=2.8)
    fig.savefig(output, dpi=200)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
