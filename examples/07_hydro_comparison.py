"""
07_hydro_comparison.py — Side-by-side comparison of semi-analytic model vs hydro.

Left:  R_out = 5e4 R_sun (hir001)
Right: R_out = 5e3 R_sun (hir002)

Both: M_ej = 5 Msun, E_ej = 1 foe, n = 7, delta = 0.5,
      v_csm = 100 km/s, M_csm = 1 Msun, R_in = 500 R_sun

Run:
    MPLBACKEND=Agg python examples/07_hydro_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from redback_csm.core import _get_lc_boxwind_bpl

R_SUN = 6.957e10
YEAR = 365.25 * 24.0 * 3600.0
SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"


def shell_parameters(r_in_rsun, r_out_rsun, v_csm_kms, m_csm_msun):
    v_csm_cgs = v_csm_kms * 1e5
    r_in = r_in_rsun * R_SUN
    r_out = r_out_rsun * R_SUN
    t1 = r_in / v_csm_cgs / YEAR
    t2 = r_out / v_csm_cgs / YEAR
    mdot = m_csm_msun * v_csm_cgs * YEAR / (r_out - r_in)
    return dict(t1=t1, t2=t2, mdot_0=0.0, mdot_1=mdot, mdot_2=0.0)


def load_hydro(path):
    if not path.exists():
        return None
    arr = np.loadtxt(path, skiprows=1)
    mask = arr[:, 0] > 0.0
    if not np.any(mask):
        return None
    return arr[mask, 0], 10.0 ** arr[mask, 2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eff", type=float, default=0.9)
    p.add_argument("--kappa", type=float, default=0.3)
    p.add_argument("--output", default="examples/hydro_comparison.png")
    return p.parse_args()


def main():
    args = parse_args()

    panels = [
        {
            "title": r"$R_{\rm out}=5\times10^4\,R_\odot$",
            "hydro": SCRIPT_DIR / "hir001.lbol",
            "r_out_rsun": 50000.0,
            "xlim": (0, 100),
        },
        {
            "title": r"$R_{\rm out}=5\times10^3\,R_\odot$",
            "hydro": SCRIPT_DIR / "hir002.lbol",
            "r_out_rsun": 5000.0,
            "xlim": (0, 30),
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, panel in zip(axes, panels):
        sp = shell_parameters(
            r_in_rsun=500.0,
            r_out_rsun=panel["r_out_rsun"],
            v_csm_kms=100.0,
            m_csm_msun=1.0,
        )

        lc_transport_const = _get_lc_boxwind_bpl(
            **sp, vwind=100.0, delta=0.5, nn=7.0,
            mexp=5.0, eexp=1.0, eff=args.eff, kappa=args.kappa,
            mode="transport", efficiency_mode=0, n_rad_zones=32,
        )

        lc_transport_timed = _get_lc_boxwind_bpl(
            **sp, vwind=100.0, delta=0.5, nn=7.0,
            mexp=5.0, eexp=1.0, eff=args.eff, kappa=args.kappa,
            mode="transport", efficiency_mode=1, n_rad_zones=32,
        )

        lc_simple = _get_lc_boxwind_bpl(
            **sp, vwind=100.0, delta=0.5, nn=7.0,
            mexp=5.0, eexp=1.0, eff=args.eff,
            mode="simple",
        )

        t_transport_const = lc_transport_const.time / 86400.0
        t_transport_timed = lc_transport_timed.time / 86400.0
        t_simple = lc_simple.time / 86400.0

        ax.plot(
            t_transport_const,
            lc_transport_const.lbol,
            color="teal",
            ls="--",
            lw=1.8,
            label="Transport (constant shock eff.)",
        )
        ax.plot(
            t_transport_timed,
            lc_transport_timed.lbol,
            color="royalblue",
            ls="-",
            lw=2.0,
            label="Transport (time-dependent shock eff.)",
        )
        ax.plot(
            t_simple,
            lc_simple.lbol,
            color="darkorange",
            ls=":",
            lw=2.0,
            label="Simple mode",
        )

        # Hydro
        hydro = load_hydro(panel["hydro"])
        if hydro is not None:
            ax.plot(hydro[0], hydro[1], color="purple", lw=2.0, label="Hydro")

        ax.set_yscale("log")
        ax.set_xlim(panel["xlim"])
        ax.set_ylim(1e42, 1e45)
        ax.set_xlabel("Time since explosion (days)")
        ax.set_title(panel["title"])
        ax.grid(alpha=0.2, which="both")

    axes[0].set_ylabel(r"$L_{\rm bol}$ (erg s$^{-1}$)")
    axes[1].legend(loc="upper right", fontsize=9)

    fig.suptitle(
        r"$M_{\rm ej}=5\,M_\odot$, $E_{\rm ej}=10^{51}$ erg, "
        r"$n=7$, $\delta=0.5$, "
        rf"$v_{{\rm csm}}=100$ km s$^{{-1}}$, $M_{{\rm csm}}=1\,M_\odot$"
        f"\n"
        rf"$\epsilon={args.eff}$, $\kappa={args.kappa}$ cm$^2$ g$^{{-1}}$",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
