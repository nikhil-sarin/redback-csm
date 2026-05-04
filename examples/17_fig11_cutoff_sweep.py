from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

from redback_csm.models import generic_powerlaw_csm_bpl_bolometric

R_SUN = 6.957e10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="examples/fig11_A_cutoff_sweep.png")
    p.add_argument("--efficiency-mode", type=int, default=0)
    p.add_argument("--a-values", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0])
    p.add_argument("--time-points", type=int, default=500)
    p.add_argument("--compact-ymax", type=float, default=8e45)
    p.add_argument("--extended-ymax", type=float, default=1.3e45)
    return p.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    time = np.geomspace(1.0, 300.0, args.time_points)
    slopes = [0.0, 0.5, 1.0, 1.5, 2.0]
    colors = ["#3b0f70", "#7f1d8d", "#c2417a", "#f07a5a", "#f3bf74"]
    panels = [
        {
            "title": r"Compact CSM: $R_{\rm csm,out}=5\times10^3\,R_\odot$",
            "r_outer": 5.0e3 * R_SUN,
            "ylim": (1e42, args.compact_ymax),
        },
        {
            "title": r"Extended CSM: $R_{\rm csm,out}=5\times10^4\,R_\odot$",
            "r_outer": 5.0e4 * R_SUN,
            "ylim": (1e41, args.extended_ymax),
        },
    ]

    nrows = len(args.a_values)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(14.2, 3.1 * nrows),
        sharex=True,
        squeeze=False,
    )

    for row, a_ratio in enumerate(args.a_values):
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            for s, color in zip(slopes, colors):
                print(
                    f"A={a_ratio:g} {panel['title'].split(':')[0]} s={s:g}",
                    flush=True,
                )
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
                    n_rad_zones=40,
                    vej_max_ratio=a_ratio,
                )
                ax.plot(time, lbol, color=color, lw=2.4, label=fr"$s={s:g}$")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1.0, 300.0)
            ax.set_ylim(panel["ylim"])
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
            ax.tick_params(axis="both", which="major", labelsize=10, length=9, width=1.8)
            ax.tick_params(axis="both", which="minor", length=4.5, width=1.4)
            for spine in ax.spines.values():
                spine.set_linewidth(1.8)
            if row == 0:
                ax.set_title(panel["title"], fontsize=15)
            if col == 0:
                ax.set_ylabel(
                    fr"$A=v_{{\rm ej,max}}/v_{{\rm tr}}={a_ratio:g}$"
                    "\n"
                    r"$L_{\rm bol}$ (erg s$^{-1}$)",
                    fontsize=13,
                )
            if row == nrows - 1:
                ax.set_xlabel("Time (days)", fontsize=14)
            if row == 0 and col == 1:
                ax.legend(loc="upper right", fontsize=10, frameon=True, edgecolor="0.6")

    fig.tight_layout(w_pad=2.6, h_pad=1.0)
    fig.savefig(output, dpi=200)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
