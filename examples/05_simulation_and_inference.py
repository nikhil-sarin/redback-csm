"""
05_simulation_and_inference.py — Simulate synthetic data and fit with redback.

Demonstrates the full workflow:
  1. Display the three variable wind CSM density models and their lightcurves
  2. Simulate synthetic bolometric observations from a Gaussian wind model
  3. Fit the data with redback using the gausswind_exponential_bolometric model
  4. Plot posterior predictions: mass-loss history and lightcurve fit

Based on the notebook variable_wind_fit.ipynb but using the redback-csm
public API throughout — no custom model wrappers needed.

Run:
    python examples/05_simulation_and_inference.py
"""

import numpy as np
import matplotlib.pyplot as plt
import bilby
import redback
from redback_csm.models import (
    wind_exponential_bolometric,
    gausswind_exponential_bolometric,
    smooth_triple_powerlaw_wind_exponential_bolometric,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------------------------------
# Shared ejecta parameters
# ---------------------------------------------------------------------------
MEXP  = 3.0   # ejecta mass (M_sun)
EEXP  = 1.5   # explosion energy (foe)
EFF   = 0.3   # CSM conversion efficiency
VWIND = 80.0  # wind velocity (km/s)
KAPPA = 0.1

# True parameters used to generate synthetic data
TRUE_PARAMS = dict(
    t_peak        = 60.0,   # yr
    t_width       = 25.0,   # yr
    mdot_baseline = 1e-5,   # M_sun/yr
    mdot_peak     = 8e-5,   # M_sun/yr
    vwind         = VWIND,
    mexp          = MEXP,
    eexp          = EEXP,
    eff           = EFF,
    kappa         = KAPPA,
)

# ---------------------------------------------------------------------------
# Figure 1: Wind model overview — density histories + lightcurves
# ---------------------------------------------------------------------------
time_yr  = np.logspace(-1, 3, 300)   # yr before explosion (for density panels)
time_days = np.geomspace(1, 1000, 300)  # days after explosion (for lightcurves)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ---- Left: mass-loss histories ----
# Constant wind
ax1.loglog(time_yr, np.full_like(time_yr, 1e-5),
           color="steelblue", lw=2, label="Constant wind")

# Gaussian wind
gauss = np.exp(-0.5 * ((time_yr - 50) / 20) ** 2)
ax1.loglog(time_yr, 1e-5 + (1e-4 - 1e-5) * gauss,
           color="darkorange", lw=2, label="Gaussian outburst")

# Smooth triple power-law wind
t1, t2 = 5.0, 50.0
sf = 0.2
a1 = np.log10(t1) * sf if t1 > 1 else sf
a2 = np.log10(t2) * sf if t2 > 1 else sf
w1 = 0.5 * (1 + np.tanh((np.log10(time_yr) - np.log10(t1)) / a1))
w2 = 0.5 * (1 + np.tanh((np.log10(time_yr) - np.log10(t2)) / a2))
m0 = 2e-5
s1 = m0 * (time_yr / 1.0) ** (-0.5)
s2 = m0 * (t1 / 1.0) ** (-0.5) * (time_yr / t1) ** 1.0
s3 = m0 * (t1 / 1.0) ** (-0.5) * (t2 / t1) ** 1.0 * (time_yr / t2) ** (-1.5)
ax1.loglog(time_yr, (1 - w1) * s1 + w1 * (1 - w2) * s2 + w1 * w2 * s3,
           color="firebrick", lw=2, label="Smooth triple power-law")

ax1.set_xlabel("Time before explosion (yr)", fontsize=12)
ax1.set_ylabel(r"$\dot{M}$ ($M_\odot$ yr$^{-1}$)", fontsize=12)
ax1.set_title("Mass-loss histories", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# ---- Right: corresponding bolometric lightcurves ----
ax2.loglog(time_days,
           wind_exponential_bolometric(time_days, mdot=1e-5, vwind=30,
                                       mexp=MEXP, eexp=EEXP, eff=EFF, kappa=KAPPA),
           color="steelblue", lw=2, label="Constant wind")

ax2.loglog(time_days,
           gausswind_exponential_bolometric(time_days,
               t_peak=50, t_width=20, mdot_baseline=1e-5, mdot_peak=1e-4,
               vwind=30, mexp=MEXP, eexp=EEXP, eff=EFF, kappa=KAPPA),
           color="darkorange", lw=2, label="Gaussian outburst")

ax2.loglog(time_days,
           smooth_triple_powerlaw_wind_exponential_bolometric(time_days,
               t_break1=5, t_break2=50, mdot_0=2e-5,
               alpha1=-0.5, alpha2=1.0, alpha3=-1.5,
               vwind=30, mexp=MEXP, eexp=EEXP, eff=EFF, kappa=KAPPA),
           color="firebrick", lw=2, label="Smooth triple power-law")

ax2.set_xlabel("Time (days)", fontsize=12)
ax2.set_ylabel(r"$L_{\rm bol}$ (erg s$^{-1}$)", fontsize=12)
ax2.set_title("CSM interaction lightcurves", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 1000)

fig.tight_layout()
fig.savefig("variable_wind_overview.png", dpi=150)
print("Saved: variable_wind_overview.png")
plt.show()

# ---------------------------------------------------------------------------
# Figure 2: Synthetic data from Gaussian wind
# ---------------------------------------------------------------------------

# Generate the noiseless truth on a fine grid
t_truth = np.geomspace(0.5, 2000, 500)
lbol_truth = gausswind_exponential_bolometric(t_truth, **TRUE_PARAMS)

# Sample at observation times and add noise
t_obs = np.geomspace(1, 500, 35)
lbol_noiseless = gausswind_exponential_bolometric(t_obs, **TRUE_PARAMS)
noise_frac = 0.12
lbol_obs = lbol_noiseless * np.random.normal(1.0, noise_frac, len(t_obs))
lbol_err = noise_frac * lbol_noiseless

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_truth, lbol_truth, "k--", lw=1.5, label="True model")
ax.errorbar(t_obs, lbol_obs, yerr=lbol_err, fmt="o", color="firebrick",
            capsize=3, ms=5, label="Synthetic observations")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time (days)", fontsize=12)
ax.set_ylabel(r"$L_{\rm bol}$ (erg s$^{-1}$)", fontsize=12)
ax.set_title("Synthetic Gaussian wind CSM data", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.8, 600)
fig.tight_layout()
fig.savefig("variable_wind_synthetic_data.png", dpi=150)
print("Saved: variable_wind_synthetic_data.png")
plt.show()

# ---------------------------------------------------------------------------
# Bayesian inference with redback
# ---------------------------------------------------------------------------

# Build the redback transient object (bolometric luminosity mode)
transient = redback.transient.Supernova(
    name="SN_gausswind_demo",
    data_mode="luminosity",
    time_rest_frame=t_obs,
    Lum50=lbol_obs / 1e50,      # redback stores luminosity in units of 1e50 erg/s
    Lum50_err=lbol_err / 1e50,
)

# Load default priors for this model (from redback_csm/priors/)
priors = redback.priors.get_priors(model="gausswind_exponential_bolometric")

# Tighten a few ranges to speed up this demo (optional — remove for real fits)
priors["t_peak"]  = bilby.core.prior.LogUniform(10, 200,  name="t_peak",
    latex_label=r"$t_{\rm peak}$ (yr)")
priors["t_width"] = bilby.core.prior.LogUniform(5,  100,  name="t_width",
    latex_label=r"$\sigma_t$ (yr)")
priors["vwind"]   = bilby.core.prior.LogUniform(10, 300,  name="vwind",
    latex_label=r"$v_{\rm wind}$ (km s$^{-1}$)")
priors["mdot_baseline"].maximum = 1e-4
priors['kappa'] = KAPPA

# The model function redback will call: gausswind_exponential_bolometric(time, **params)
# redback passes Lum50 so the model must return luminosity in the same units.
# We wrap to handle the 1e50 scaling transparently.
def model_for_redback(time, **kwargs):
    return gausswind_exponential_bolometric(time, **kwargs) / 1e50

print("\nStarting Bayesian inference with redback ...")
print(f"  Data points : {len(t_obs)}")
print(f"  Free params : {len(priors)}")
print(f"  Model       : gausswind_exponential_bolometric\n")

result = redback.fit_model(
    transient,
    model=model_for_redback,
    prior=priors,
    sampler="pymultinest", # change to different sampler if you prefer
    nlive=500,
    plot=False,
    injection_parameters=TRUE_PARAMS,
    clean=True
)

print(f"\nlog Z = {result.log_evidence:.2f} ± {result.log_evidence_err:.2f}")

# ---------------------------------------------------------------------------
# Figure 3: Corner plot
# ---------------------------------------------------------------------------
result.plot_corner(filename="variable_wind_corner.png", show=False)
print("Saved: variable_wind_corner.png")

# ---------------------------------------------------------------------------
# Figure 4: Redback lightcurve fit
# ---------------------------------------------------------------------------
result.plot_lightcurve(filename="variable_wind_lightcurve.png", model=model_for_redback,
                       show=False)
print("Saved: variable_wind_corner.png")

# ---------------------------------------------------------------------------
# Figure 5: Posterior predictions — mass-loss history + lightcurve fit
# ---------------------------------------------------------------------------
fig, (ax_lc, ax_mdot) = plt.subplots(2, 1, figsize=(7, 10))

# ---- Bottom panel: mass-loss history reconstruction ----
# The Gaussian CSM profile is fully determined by t_peak, t_width,
# mdot_baseline, mdot_peak — reconstruct the same grid used by the Fortran code.
t_common = np.linspace(0.1, 200, 1000)   # years before explosion

all_mdot = []
for _, row in result.posterior.iterrows():
    gauss_i = np.exp(-0.5 * ((t_common - row.t_peak) / row.t_width) ** 2)
    mdot_i  = row.mdot_baseline + (row.mdot_peak - row.mdot_baseline) * gauss_i
    all_mdot.append(mdot_i)

all_mdot = np.array(all_mdot)
lo95, med, hi95 = np.percentile(all_mdot, [2.5, 50, 97.5], axis=0)
lo68, hi68      = np.percentile(all_mdot, [16, 84], axis=0)

ax_mdot.fill_between(t_common, lo95, hi95, color="steelblue", alpha=0.25, label=r"$95\%$ CI")

# True mass-loss history (evaluated on the same t_common grid)
gauss_true = np.exp(-0.5 * ((t_common - TRUE_PARAMS["t_peak"]) / TRUE_PARAMS["t_width"]) ** 2)
mdot_true  = (TRUE_PARAMS["mdot_baseline"]
              + (TRUE_PARAMS["mdot_peak"] - TRUE_PARAMS["mdot_baseline"]) * gauss_true)
ax_mdot.plot(t_common, mdot_true, "r-", lw=2.5, label="True model", zorder=5)

ax_mdot.set_xlabel("Time before explosion (yr)", fontsize=12)
ax_mdot.set_ylabel(r"$\dot{M}$ ($M_\odot$ yr$^{-1}$)", fontsize=12)
ax_mdot.set_title("Inferred mass-loss history", fontsize=12)
ax_mdot.set_yscale("log")
ax_mdot.set_xlim(0, 150)
ax_mdot.legend(fontsize=9)
ax_mdot.grid(True, alpha=0.3)

# ---- Top panel: lightcurve fit ----
t_plot = np.geomspace(1, 600, 200)

# Posterior predictive envelope
all_lc = []
# Build fixed-parameter overrides for any prior entries that are not free (e.g. kappa)
fixed_params = {k: v for k, v in priors.items() if not hasattr(v, 'sample')}
for _, row in result.posterior.iterrows():
    kwargs = {k: row[k] for k in result.posterior.columns if k in priors}
    kwargs.update(fixed_params)
    lc_i = gausswind_exponential_bolometric(t_plot, **kwargs)
    all_lc.append(lc_i)

all_lc = np.array(all_lc)
lc_lo95, lc_med, lc_hi95 = np.percentile(all_lc, [2.5, 50, 97.5], axis=0)
lc_lo68, lc_hi68          = np.percentile(all_lc, [16, 84], axis=0)

ax_lc.fill_between(t_plot, lc_lo95, lc_hi95, color="steelblue", alpha=0.25)
ax_lc.plot(t_plot, gausswind_exponential_bolometric(t_plot, **TRUE_PARAMS),
           "r-", lw=2.5, label="True model")
ax_lc.errorbar(t_obs, lbol_obs, yerr=lbol_err, fmt="o", color="k",
               capsize=3, ms=4, label="Observations", zorder=5)

ax_lc.set_xscale("log")
ax_lc.set_yscale("log")
ax_lc.set_xlabel("Time (days)", fontsize=12)
ax_lc.set_ylabel(r"$L_{\rm bol}$ (erg s$^{-1}$)", fontsize=12)
ax_lc.set_title("Lightcurve posterior predictions", fontsize=12)
ax_lc.set_xlim(0.8, 600)
ax_lc.legend(fontsize=9)
ax_lc.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("variable_wind_posterior_predictions.png", dpi=150)
print("Saved: variable_wind_posterior_predictions.png")
plt.show()