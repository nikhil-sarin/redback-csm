import sys
from pathlib import Path
import argparse

# Add project root to sys.path to find jax_csm
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from jax_csm.model import get_csm_lightcurve
from jax_csm.dynamics import DynamicsParams

R_SUN = 6.957e10

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="fig_from_paper_jax.png")
    return p.parse_args()

def main():
    args = parse_args()
    output = Path(args.output)
    
    slopes = [0.0, 0.5, 1.0, 1.5, 2.0]
    colors = ["#3b0f70", "#7f1d8d", "#c2417a", "#f07a5a", "#f3bf74"]
    panels = [
        {
            "title": r"Compact CSM: $R_{\rm csm,out}=5\times10^3\,R_\odot$",
            "r_outer": 5.0e3 * R_SUN,
            "xlim": (1.0, 300.0),
        },
        {
            "title": r"Extended CSM: $R_{\rm csm,out}=5\times10^4\,R_\odot$",
            "r_outer": 5.0e4 * R_SUN,
            "xlim": (1.0, 300.0),
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14.14, 4.72))

    for ax, panel in zip(axes, panels):
        for s, color in zip(slopes, colors):
            # Prepare parameters for the JAX model
            # v_tr and v_sc should be solved from the shock-CSM interaction
            # For now, use reasonable estimates:
            # For BPL ejecta with delta=1, n=10, M_ej=5, E_sn=1:
            # v_ej_max ~ sqrt(2*E/M) * sqrt(n/(n-3)) ~ sqrt(2*1e51 / (5*1.989e33)) * sqrt(10/7)
            import numpy as np
            m_sun = 1.989e33
            e_foe = 1e51
            v_ej_max = np.sqrt(2 * 1.0 * e_foe / (5.0 * m_sun)) * np.sqrt(10.0 / 7.0)
            # Set v_tr to a fraction of v_ej_max, and v_sc similarly
            v_tr_est = 0.5 * v_ej_max
            v_sc_est = 0.5 * v_ej_max
            
            params = {
                'dyn': {
                    'm_ej': 5.0,
                    'e_sn': 1.0,
                    'm_csm': 1.0,
                    'r_csm_in': 5.0e2 * R_SUN,
                    'r_csm_out': panel["r_outer"],
                    's': s,
                    'delta': 1.0,
                    'n': 10.0,
                    'v_tr': float(v_tr_est),
                    'v_sc': float(v_sc_est),
                    'profile_type': 0, # BPL
                },
                'kappa': 0.2,
                'eff_int': 1.0,
                'n_grid': 100,
                'dt_zeta': 0.01
            }
            
            try:
                time_days, lbol = get_csm_lightcurve(params)
                ax.plot(time_days, lbol, color=color, lw=3.0, label=fr"$s={s:g}$")
            except Exception as e:
                print(f"Error calculating s={s}: {e}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(panel["xlim"])
        ax.set_ylim(1e41, 1.3e45 if panel["r_outer"] > 1e4 * R_SUN else 6e45)
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
