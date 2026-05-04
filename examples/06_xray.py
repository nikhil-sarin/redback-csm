"""Thermal bremsstrahlung X-ray light curves from CSM interaction.

This mirrors the radio example, but computes a free-free X-ray band luminosity
from the shock evolution.

Run:
    MPLBACKEND=Agg python examples/06_xray.py
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

from redback_csm.models import wind_bpl_xray


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="examples/xray_lightcurves.png")
    return p.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)

    time = np.geomspace(1.0, 300.0, 160)
    redshift = 0.02

    base = dict(
        mdot=1e-2,
        vwind=100.0,
        delta=0.5,
        nn=12.0,
        mexp=5.0,
        eexp=1.0,
        eff=0.5,
        mode="simple",
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for logepsx, label in [(-2.0, "1%"), (-1.0, "10%"), (-0.5, "32%")]:
        lx = wind_bpl_xray(
            time=time,
            redshift=redshift,
            logepsx=logepsx,
            e_min_kev=0.3,
            e_max_kev=10.0,
            output_format="luminosity",
            **base,
        )
        ax.loglog(time, lx, label=label)

    absorbed = wind_bpl_xray(
        time=time,
        redshift=redshift,
        logepsx=-1.0,
        e_min_kev=0.3,
        e_max_kev=10.0,
        output_format="luminosity",
        n_h_host=1e22,
        **base,
    )
    ax.loglog(time, absorbed, "--", color="0.25", label=r"10%, $N_H=10^{22}$ cm$^{-2}$")

    ax.set_xlabel("Observer-frame time [days]")
    ax.set_ylabel(r"$L_X(0.3-10\,\mathrm{keV})$ [erg s$^{-1}$]")
    ax.set_title("CSM thermal bremsstrahlung X-rays")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
