from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

from redback_csm.models import generic_powerlaw_csm_bpl_bolometric

R_SUN = 6.957e10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="examples/fig11_resolution_convergence.png")
    p.add_argument("--efficiency-mode", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)

    time = np.geomspace(1.0, 300.0, 500)
    slopes = [0.0, 0.5, 1.0, 1.5, 2.0]
    colors = ["#3b0f70", "#7f1d8d", "#c2417a", "#f07a5a", "#f3bf74"]
    resolutions = [
        ("low", {"compact": 30, "extended": 30}, ":"),
        ("base", {"compact": 120, "extended": 80}, "-"),
        ("high", {"compact": 240, "extended": 160}, "--"),
    ]
    panels = [
        {
            "name": "compact",
            "title": r"Compact: $R_{\rm csm,out}=5\times10^3\,R_\odot$",
            "r_outer": 5.0e3 * R_SUN,
            "ylim": (1e41, 6e45),
        },
        {
            "name": "extended",
            "title": r"Extended: $R_{\rm csm,out}=5\times10^4\,R_\odot$",
            "r_outer": 5.0e4 * R_SUN,
            "ylim": (1e41, 1.3e45),
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))

    for ax, panel in zip(axes, panels):
        for s, color in zip(slopes, colors):
            for label, zone_map, linestyle in resolutions:
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
                    n_rad_zones=zone_map[panel["name"]],
                )
                ax.plot(
                    time,
                    lbol,
                    color=color,
                    lw=2.2 if label == "base" else 1.5,
                    ls=linestyle,
                    alpha=1.0 if label == "base" else 0.8,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.0, 300.0)
        ax.set_ylim(*panel["ylim"])
        ax.set_xlabel("Time (days)", fontsize=16)
        ax.set_title(panel["title"], fontsize=15)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax.tick_params(axis="both", which="major", labelsize=11, length=10, width=1.8)
        ax.tick_params(axis="both", which="minor", length=5, width=1.4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)

    color_handles = [
        plt.Line2D([0], [0], color=color, lw=2.5, ls="-", label=fr"$s={s:g}$")
        for s, color in zip(slopes, colors)
    ]
    style_handles = [
        plt.Line2D([0], [0], color="black", lw=2.0 if label == "base" else 1.5, ls=linestyle, label=label)
        for label, _, linestyle in resolutions
    ]

    axes[0].set_ylabel(r"Bolometric Luminosity (erg s$^{-1}$)", fontsize=18)
    axes[0].legend(handles=color_handles, loc="upper left", fontsize=11, frameon=True)
    axes[1].legend(handles=style_handles, loc="upper right", fontsize=11, frameon=True, title="Resolution")

    fig.suptitle(
        rf"Fig. 11 Resolution Check, transport mode, efficiency_mode={args.efficiency_mode}",
        fontsize=14,
    )
    fig.tight_layout(w_pad=2.4)
    fig.savefig(output, dpi=200)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
