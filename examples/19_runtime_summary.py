"""Runtime summary for release validation.

This script is deliberately small enough to run on a laptop. It reports wall
times for the main inference-relevant public paths.

Run:
    python examples/19_runtime_summary.py
"""

from __future__ import annotations

import argparse
import time as _time
from pathlib import Path

import numpy as np

from redback_csm.models import generic_powerlaw_csm_bpl_bolometric, wind_bpl_bolometric


R_SUN = 6.957e10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="examples/runtime_summary.csv")
    p.add_argument("--repeat", type=int, default=3)
    return p.parse_args()


def _time_call(label, repeat, func):
    times = []
    value = None
    for _ in range(repeat):
        t0 = _time.perf_counter()
        value = func()
        times.append(_time.perf_counter() - t0)
    arr = np.asarray(value, dtype=float)
    return {
        "label": label,
        "median_ms": 1.0e3 * float(np.median(times)),
        "min_ms": 1.0e3 * float(np.min(times)),
        "max_ms": 1.0e3 * float(np.max(times)),
        "n_time": arr.size,
        "finite": bool(np.all(np.isfinite(arr))),
        "l_peak": float(np.nanmax(arr)),
    }


def main():
    args = parse_args()
    output = Path(args.output)
    time_grid = np.geomspace(1.0, 300.0, 220)

    wind_kwargs = dict(
        time=time_grid,
        mdot=1.0e-3,
        vwind=100.0,
        delta=0.5,
        nn=10.0,
        mexp=5.0,
        eexp=1.0,
        eff=0.5,
    )
    static_kwargs = dict(
        time=time_grid,
        eta=-2.0,
        r_inner=500.0 * R_SUN,
        r_outer=5000.0 * R_SUN,
        delta_sn=1.0,
        nn_sn=10.0,
        mej_sn=5.0,
        esn=1.0,
        eff=1.0,
        m_csm=1.0,
        kappa=0.2,
    )

    rows = [
        _time_call(
            "wind_bpl_simple",
            args.repeat,
            lambda: wind_bpl_bolometric(mode="simple", **wind_kwargs),
        ),
        _time_call(
            "wind_bpl_simple_kappa",
            args.repeat,
            lambda: wind_bpl_bolometric(mode="simple", kappa=0.34, **wind_kwargs),
        ),
        _time_call(
            "static_powerlaw_transport_n20",
            args.repeat,
            lambda: generic_powerlaw_csm_bpl_bolometric(
                mode="transport", n_rad_zones=20, **static_kwargs
            ),
        ),
        _time_call(
            "static_powerlaw_transport_n40",
            args.repeat,
            lambda: generic_powerlaw_csm_bpl_bolometric(
                mode="transport", n_rad_zones=40, **static_kwargs
            ),
        ),
    ]

    if output.parent:
        output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write("label,median_ms,min_ms,max_ms,n_time,finite,l_peak\n")
        for row in rows:
            f.write(
                f"{row['label']},{row['median_ms']:.3f},{row['min_ms']:.3f},"
                f"{row['max_ms']:.3f},{row['n_time']},{row['finite']},{row['l_peak']:.6e}\n"
            )
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
