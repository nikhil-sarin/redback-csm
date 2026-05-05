"""Compare JAX and Fortran transport for static power-law CSM shells.

Run:
    MPLBACKEND=Agg python examples/18_jax_vs_fortran_transport.py
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from jax_csm.model import get_static_powerlaw_csm_bpl_lightcurve
from redback_csm.models import generic_powerlaw_csm_bpl_bolometric


R_SUN = 6.957e10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="jax_vs_fortran_transport.png")
    p.add_argument("--n-steps", type=int, default=16384)
    p.add_argument("--n-rad-zones", type=int, default=40)
    p.add_argument("--time-points", type=int, default=400)
    return p.parse_args()


def _run_fortran(time, s, r_outer, n_rad_zones):
    return generic_powerlaw_csm_bpl_bolometric(
        time=time,
        eta=-s,
        r_inner=5.0e2 * R_SUN,
        r_outer=r_outer,
        delta_sn=1.0,
        nn_sn=10.0,
        mej_sn=5.0,
        esn=1.0,
        eff=1.0,
        m_csm=1.0,
        mode="transport",
        kappa=0.2,
        n_rad_zones=n_rad_zones,
    )


def _run_jax(time, s, r_outer, n_steps, n_rad_zones):
    return np.asarray(
        get_static_powerlaw_csm_bpl_lightcurve(
            time=time,
            eta=-s,
            r_inner=5.0e2 * R_SUN,
            r_outer=r_outer,
            delta_sn=1.0,
            nn_sn=10.0,
            mej_sn=5.0,
            esn=1.0,
            eff=1.0,
            m_csm=1.0,
            mode="transport",
            kappa=0.2,
            n_steps=n_steps,
            n_rad_zones=n_rad_zones,
        )
    )


def main():
    args = parse_args()
    output = Path(args.output)
    time = np.geomspace(1.0, 300.0, args.time_points)
    slopes = [0.0, 1.0, 2.0]
    colors = ["#3b0f70", "#c2417a", "#f3bf74"]
    panels = [
        ("Compact", 5.0e3 * R_SUN, (1e42, 6e45)),
        ("Extended", 5.0e4 * R_SUN, (1e41, 1.3e45)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.0), sharex="col")
    rows = []

    for col, (name, r_outer, ylim) in enumerate(panels):
        ax = axes[0, col]
        ax_res = axes[1, col]
        for s, color in zip(slopes, colors):
            lf = np.asarray(_run_fortran(time, s, r_outer, args.n_rad_zones))
            lj = _run_jax(time, s, r_outer, args.n_steps, args.n_rad_zones)
            mask = (lf > 1.0e38) & (lj > 1.0e38) & np.isfinite(lf) & np.isfinite(lj)
            dex = np.full_like(time, np.nan, dtype=float)
            dex[mask] = np.log10(lj[mask]) - np.log10(lf[mask])

            ax.plot(time, lf, color=color, lw=2.4, label=fr"Fortran $s={s:g}$")
            ax.plot(time, lj, color=color, lw=1.8, ls="--", label=fr"JAX $s={s:g}$")
            ax_res.plot(time, dex, color=color, lw=1.8, label=fr"$s={s:g}$")

            if np.any(mask):
                rows.append(
                    (
                        name,
                        s,
                        np.nanmedian(np.abs(dex[mask])),
                        np.nanmax(np.abs(dex[mask])),
                        time[np.nanargmax(lf)],
                        np.nanmax(lf),
                        time[np.nanargmax(lj)],
                        np.nanmax(lj),
                    )
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_title(f"{name} CSM")
        ax.set_ylabel(r"$L_{\rm bol}$ (erg s$^{-1}$)")
        ax.grid(alpha=0.18, which="both")
        ax_res.axhline(0.0, color="0.3", lw=1.0)
        ax_res.set_xscale("log")
        ax_res.set_ylim(-0.35, 0.35)
        ax_res.set_xlabel("Time (days)")
        ax_res.set_ylabel(r"$\Delta\log_{10} L$")
        ax_res.grid(alpha=0.18, which="both")

    axes[0, 1].legend(fontsize=8, ncol=2)
    fig.suptitle(
        rf"JAX vs Fortran transport, n_steps={args.n_steps}, n_rad_zones={args.n_rad_zones}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)

    print(f"Saved: {output}")
    print("case,s,median_abs_dex,max_abs_dex,tpeak_fortran,Lpeak_fortran,tpeak_jax,Lpeak_jax")
    for row in rows:
        print(
            f"{row[0]},{row[1]:g},{row[2]:.4f},{row[3]:.4f},"
            f"{row[4]:.3f},{row[5]:.4e},{row[6]:.3f},{row[7]:.4e}"
        )


if __name__ == "__main__":
    main()
