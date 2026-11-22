[![arXiv](https://img.shields.io/badge/arXiv-2605.19571-00ff00.svg)](https://arxiv.org/abs/2605.19571)
[![Documentation Status](https://readthedocs.org/projects/redback-csm/badge/?version=latest)](https://redback-csm.readthedocs.io/en/latest/?badge=latest)
# redback-csm

Fortran-based circumstellar matter (CSM) interaction models for the
[redback](https://github.com/nikhil-sarin/redback) transient modelling and inference package.

Once installed, all CSM models are automatically available in redback's model
library and can be used for Bayesian inference in the same way as built-in models.

## Contributors

- [nikhil-sarin](https://github.com/nikhil-sarin)
- [ryosuke-hirai](https://github.com/ryosuke-hirai)

## Citation

If you use this package please cite:

- **Sarin & Hirai 2026** — the paper describing the CSM interaction models (https://arxiv.org/abs/2605.19571)
- **Sarin et al. 2024** — the redback paper (https://arxiv.org/abs/2308.12806)
- Other relevant papers describing the underlying physics and numerical methods will be cited in the main paper.

## Models

### Naming convention

Model names follow the convention **`{outer_CSM}_{inner_ejecta}`** — the first part
is the older, outer CSM density profile (laid down first by the progenitor), and the
second part is the more recent, inner ejecta profile (the transient event that sweeps
outward into the CSM).

For example, `wind_exponential` means a steady progenitor wind (outer CSM) being
swept up by an exponential SN explosion (inner ejecta).

Note that the inner ejecta profile is not necessarily a SN explosion —  for example,`exponential_wind` 
means an exponential outer shell (e.g. from a non-SN eruption) being swept up by a later steady wind (e.g. from a surviving star). 
The naming convention is purely descriptive of the density profiles and does not imply any particular physical scenario.

### Optical / bolometric model variants

The package currently exposes 35 base CSM-density/ejecta scenarios. These are
the physical configurations listed below, before adding output-specific wrapper
suffixes. Each base scenario has the same optical wrapper family:

| Suffix | Output | Time frame |
|---|---|---|
| `{name}_bolometric` | Bolometric luminosity (erg/s) | Source frame |
| `{name}` | Flux density / magnitude / spectra | Observer frame |
| `{name}_nickel_bolometric` | CSM + radioactive ⁵⁶Ni decay, bolometric | Source frame |
| `{name}_nickel` | CSM + radioactive ⁵⁶Ni decay, multiband | Observer frame |

The multiband functions accept an `output_format` keyword:
`'flux_density'` (mJy), `'magnitude'` (AB), `'flux'`, or `'spectra'`.

The `_nickel` variants add `f_nickel` (nickel mass fraction) and `kappa_gamma`
(gamma-ray opacity) to model radioactive ⁵⁶Ni/⁵⁶Co heating alongside CSM shock
emission. The nickel mass is always computed from the SN ejecta mass
(`M_Ni = f_nickel * M_ej`), while the Arnett diffusion mass includes the
finite CSM shell mass when that mass is defined by the CSM constructor. There
is no separate CSM-mass parameter for the nickel component; for unusual cases
you can override only the total Arnett diffusion mass with `mej_arnett`.
Note that redback's own `csm_nickel` uses only the optically thick part of the
CSM (tau > 2/3) in the diffusion mass; for an optically thick shell the two
agree closely.

Priors are generated programmatically from the model signatures, so the package
does not need a separate `.prior` file for every callable wrapper. This covers
all exported wrappers except the generic `csm_xray` convenience function.

### Shared Runtime Options

All optical / bolometric CSM wrappers also accept a common set of runtime
keywords controlling the transport treatment:

| Keyword | Meaning |
|---|---|
| `mode='simple'` | Default thin-shell calculation. If `kappa` is provided, this uses the legacy post-processed diffusion light curve. |
| `mode='transport'` | Use the newer transport solver for the observed luminosity while keeping the same shell dynamics. |
| `kappa` | Opacity in `cm^2 g^-1`. Optional in simple mode. In transport mode, defaults to `0.34` if not supplied. |
| `n_rad_zones` | Transport resolution (default 160, minimum 40). Check convergence for your parameters; the dark phase can require substantially more zones than the bright phase. Higher resolution increases runtime. |
| `diffusion_escape` | Simple-mode diffusion prescription: `"finite"` (default) uses `(1 + tau) R / c`; `"legacy"` uses `tau R / c`, which has an emergence singularity for a sharply truncated shell. |
| `transport_wind_inner_age` | Effective inner wind age in years for wind-history CSMs in transport mode. If omitted, generated wind histories use the larger of their first tabulated age and `1 yr`; one-point steady winds use `1 yr`. This creates the finite inner cavity needed by the nondimensional transport setup and is ignored in simple mode. |
| `transport_wind_age` | Effective outer wind age in years for wind-history CSMs in transport mode. If omitted, generated wind histories use their last tabulated age; one-point steady winds use `100 yr`. This supplies the finite outer edge required by the transport solver and is ignored in simple mode. |
| `transport_r_inner` / `r_inner` | Alternative direct inner cutoff radius in cm for wind-history CSMs in transport mode. `transport_r_inner` takes precedence. |
| `transport_r_outer` / `r_outer` | Alternative direct outer cutoff radius in cm for wind-history CSMs in transport mode. `transport_r_outer` takes precedence. |
| `photosphere_mode` | Multiband wrappers only. `'legacy'` (default) or `'csm'`; see [Photosphere for multiband output](#photosphere-for-multiband-output). |
| `temperature_floor` | Multiband wrappers only. Blackbody temperature floor in K; below it the photosphere shrinks so that L is preserved. Defaults: none for CSM-only legacy, 3000 K for `_nickel` and for `photosphere_mode='csm'`. |
| `efficiency_mode` | Optional shock-efficiency mode. Default is `0`, which applies the user-supplied constant `eff` to both shocks; `1` applies the free-free-limited time-dependent efficiency to both forward and reverse shocks. |

Transport mode requires a finite CSM support. Static CSM constructors already
have this through their radial grid or explicit `r_outer`. In transport mode,
wind-history constructors are first sampled into a finite static radial CSM grid
using `transport_wind_inner_age` / `transport_wind_age` or direct
`transport_r_inner` / `transport_r_outer` cutoffs. Simple mode keeps the older
effectively steady wind extrapolation.

Current output convention:

- `lbol` is the main observable luminosity
- `lbol_shock` is the shock-powered luminosity
- `lbol_diffuse` is the diffusion / transport luminosity when available
- `rshock` is the forward-shock / shell radius
- `rph` is retained for backwards compatibility; in one-zone mode it is the
  same as `rshock`, while in transport mode it is the radiation photosphere

### Photosphere for multiband output

The multiband (`{name}`, `{name}_nickel`) wrappers turn the bolometric
luminosity into a blackbody, which needs a photospheric radius.

- `photosphere_mode='legacy'` (default, unchanged): CSM-only wrappers use the
  shell radius `rph` and the Fortran blackbody temperature; `_nickel` wrappers
  use redback's `TemperatureFloor`, i.e. `R = vej * t` with
  `vej = sqrt(2 E / M_ej)` until the temperature reaches the floor. Neither
  follows the photosphere while the shock is still buried in optically thick
  CSM, and `vej * t` ignores the deceleration of the shell.
- `photosphere_mode='csm'`: the photosphere is the radius where the optical
  depth through the CSM ahead of the shell reaches 2/3 (the fixed CSM
  photosphere of Chatzopoulos, Wheeler & Vinko 2012), and becomes the shell
  radius once the CSM ahead is optically thin (e.g. after the shock leaves a
  finite shell). It is computed in the Fortran for every CSM density model,
  with the same `kappa` and density profile as the diffusion, and is the same
  for the CSM-only and `_nickel` wrappers. The temperature is
  `(L / 4 pi sigma R^2)^(1/4)` with the total luminosity, and the
  `temperature_floor` (default 3000 K) handles late-time recombination while
  preserving L. Simple mode requires `kappa`; transport mode uses the transport
  photosphere.

```python
mag = static_powerlaw_csm_bpl_nickel(
    time=time_obs, redshift=0.05, eta=-2.0, r_inner=1e14, r_outer=1e15,
    delta_sn=1.0, nn_sn=10.0, mej_sn=10.0, esn=2.0, eff=0.5, m_csm=3.0,
    f_nickel=0.05, kappa=0.34, kappa_gamma=0.1,
    photosphere_mode='csm', bands='lsstg', output_format='magnitude',
)
```

### Broken power-law ejecta cutoff

BPL ejecta models use a finite maximum ejecta velocity by default. The optional
`vej_max_ratio` parameter sets

```text
v_max = vej_max_ratio * v_t
```

where `v_t` is the BPL transition velocity implied by the ejecta mass, kinetic
energy, and the `delta`/`nn` power-law indices. If `vej_max_ratio` is not given,
the default is `3` for Python wrappers that provide the BPL parameters. Larger
values allow faster outer ejecta to reach distant CSM earlier; smaller values
delay or suppress early interaction with large-radius shells. The old alias
`A_ratio` is also accepted. You can instead pass `vej_max` in km/s, in which
case the wrapper converts it to `vej_max_ratio`.

### JAX static CSM prototype

The `jax_csm` package provides JAX-native (JIT- and `vmap`-compatible) versions
of the static finite power-law CSM shell with BPL ejecta.

```python
import numpy as np
from jax_csm import StaticPowerlawBPLParams, solve_simple_static_bpl, simple_csm_nickel_bolometric

params = StaticPowerlawBPLParams(
    eta=-2.0, r_inner=1e14, r_outer=1e15, delta_sn=1.0, nn_sn=10.0,
    mej_sn=10.0, esn=2.0, eff=0.5, m_csm=3.0, kappa=0.34,
)
time_days = np.geomspace(1.0, 300.0, 200)
res = solve_simple_static_bpl(time_days, params, n_steps=4096, t_max_days=300.0, diffusion=True)
res.lbol, res.lbol_shock, res.r_shock, res.v_shock, res.r_photosphere, res.t_emerge

total = simple_csm_nickel_bolometric(time_days, params, f_nickel=0.05, kappa_gamma=0.1, n_steps=4096)
```

- `solve_simple_static_bpl` integrates the same thin-shell equations as the
  Fortran simple mode (RK4 in dimensionless variables). With `diffusion=False`
  `lbol` is the shock luminosity `eff * (L_fs + L_rs)`; with `diffusion=True` it
  is the diffused luminosity that the Fortran simple mode returns when `kappa`
  is supplied. Near shock emergence that prescription rises logarithmically, so
  the value right at emergence depends on the time resolution in any
  implementation. About 2048 steps (no diffusion) or 4096 steps (diffusion)
  give sub-percent accuracy elsewhere. Fix `t_max_days` when results must not
  depend on the latest requested epoch.
- `nickel_diffused_luminosity` / `simple_csm_nickel_bolometric` reproduce the
  `_nickel_bolometric` prescription (Arnett diffusion with gamma-ray leakage).
- `r_photosphere` is the `photosphere_mode='csm'` radius;
  `jax_csm.photosphere_temperature` applies the same temperature floor.
- `solve_static_powerlaw_bpl(..., mode="simple")` uses the new solver;
  `mode="simple_legacy"` keeps the previous implementation; `mode="transport"`
  is the dimensionless transport solver for static power-law shells.

### Radio synchrotron model variants

Each base CSM scenario also has a `{name}_radio` function that computes
synchrotron radio emission from the CSM-interaction shock using the explicit
shock radius `rshock`, the shock velocity, and the upstream CSM density. The
radio model follows the Chevalier (1998) formalism with self-absorption. These
take additional parameters:

| Parameter | Description |
|---|---|
| `logepsb` | log₁₀(ε_B), magnetic energy fraction of shock ram pressure |
| `logepse` | log₁₀(ε_e), electron energy fraction |
| `p` | Electron power-law index (typically 2–4) |
| `frequency` | Observing frequency in Hz: a scalar, or one value per `time` point |

Output is flux density in mJy. For multi-frequency data, pass `time` and
`frequency` as matching per-point arrays (repeated times are fine): one call
then evaluates every data point with a single hydro run. See
`examples/20_multifrequency_radio_fit.py`.

The shock velocity used is relative to the upstream CSM, and for homologously
expanding CSM (`generic_*`, `homologous_powerlaw_*`) the upstream density
follows the same expansion as the hydro.

Optional spectral-shape keywords (all off by default):

| Parameter | Description |
|---|---|
| `cooling`, `log_nu_c_scale` | Synchrotron cooling break: thin spectrum steepens by 1/2 above `nu_c` (computed from B and t), scaled by `10**log_nu_c_scale` |
| `ssa_alpha_thick` | Optically thick spectral index (default 5/2); lower values mimic an inhomogeneous absorber |
| `ssa_smoothing` | Smooth join of the thick and thin asymptotes, `(F_thick^-s + F_thin^-s)^(-1/s)`; default is a sharp break |
| `log_tau_ff`, `ff_time_index`, `ff_nu_ref_ghz`, `ff_t_ref_days` | External free-free absorption `exp(-tau)`, `tau = 10**log_tau_ff (nu/nu_ref)^-2.1 (t/t_ref)^-ff_time_index` |

### X-ray thermal bremsstrahlung model variants

Each base CSM scenario also has a matching `{name}_xray` function. The
X-ray layer is a fast post-processor: it uses the same shock radius `rshock`,
shock velocity, and upstream density as the radio layer, sets the post-shock
plasma temperature from the strong-shock velocity, estimates the shocked-CSM
emission measure, and emits a thermal free-free spectrum. The free-free
luminosity is capped by the shock power by default.

| Parameter | Description |
|---|---|
| `logepsx` | log10 scale factor for the emitting free-free emission measure |
| `e_min_kev`, `e_max_kev` | Observer-frame X-ray band edges in keV; defaults to 0.3--10 keV |
| `output_format` | `luminosity`, `flux`, `spectral_luminosity`, or `flux_density` |
| `energy_kev` / `frequency` | Photon energy or frequency for spectral outputs |
| `n_h_host`, `n_h_mw` | Optional absorbing columns in cm^-2 |
| `absorb_csm` | Optionally include a local CSM column estimate |
| `shock_component` | `total`, `forward`, or `reverse` shock power |
| `normalization` | `emission_measure` by default; `shock_power` for a simple shock-power-fraction approximation |
| `max_xray_efficiency` | Cap on bolometric free-free luminosity relative to shock power; defaults to 1 |

For dense CSM, soft X-rays can be strongly absorbed, so `n_h_host` and
`absorb_csm=True` are often more important than the intrinsic free-free bandpass.

The standard physics used is optically thin thermal free-free emission:

```text
epsilon_nu = 6.8e-38 Z^2 n_e n_i T^{-1/2} exp(-h nu / kT) g_ff
epsilon_ff = 1.426e-27 T^{1/2} n_e n_i g_ff
kT_shock = (3/16) mu m_p v_shock^2
```

The inference approximation is the shocked-shell emission measure:

```text
EM ~= M_swept,CSM * rho_post / (mu_e * mu_i * m_p^2)
rho_post = compression_factor * rho_CSM
```

This is a thin-shell closure, not a resolved X-ray cooling-layer calculation.
The normalization parameter `logepsx` therefore scales the emitting emission
measure by default. Set `normalization='shock_power'` to recover a simple
shock-power-fraction approximation.

References for the underlying physics include Rybicki & Lightman (1979),
Chevalier & Fransson (1994, ApJ, 420, 268), and Margalit, Quataert & Ho
(2022, ApJ, 928, 122).

### Available base CSM-density/ejecta scenarios (35 total)

These are the unsuffixed physical configurations. Each base scenario has
generated optical, nickel, radio, and X-ray wrapper variants, so the public
model function count is larger than 35: 35 base scenarios times six wrapper
forms (`{name}`, `{name}_bolometric`, `{name}_nickel`,
`{name}_nickel_bolometric`, `{name}_radio`, `{name}_xray`), plus the generic
`csm_xray` helper, for 211 public callables.

| Base model | CSM / outer material | Inner explosion/ejecta |
|---|---|---|
| `wind_exponential` | steady wind | exponential SN ejecta |
| `wind_bpl` | steady wind | broken power-law SN ejecta |
| `exponential_wind` | steady wind | exponential eruption/ejecta |
| `bpl_wind` | steady wind | broken power-law eruption/ejecta |
| `exponential_exponential` | exponential outer shell | exponential SN ejecta |
| `exponential_bpl` | exponential outer shell | broken power-law SN ejecta |
| `bpl_bpl` | broken power-law outer shell | broken power-law SN ejecta |
| `bpl_exponential` | broken power-law outer shell | exponential SN ejecta |
| `boxwind_exponential` | box-shaped wind history | exponential SN ejecta |
| `boxwind_bpl` | box-shaped wind history | broken power-law SN ejecta |
| `gausswind_exponential` | Gaussian wind history | exponential SN ejecta |
| `gausswind_bpl` | Gaussian wind history | broken power-law SN ejecta |
| `triple_powerlaw_wind_bpl` | triple power-law wind history | broken power-law SN ejecta |
| `triple_powerlaw_wind_exponential` | triple power-law wind history | exponential SN ejecta |
| `exponential_triple_powerlaw_wind` | triple power-law wind | exponential inner ejecta |
| `bpl_triple_powerlaw_wind` | triple power-law wind | broken power-law inner ejecta |
| `smooth_triple_powerlaw_wind_bpl` | smoothed triple power-law wind history | broken power-law SN ejecta |
| `smooth_triple_powerlaw_wind_exponential` | smoothed triple power-law wind history | exponential SN ejecta |
| `generic_csm_exponential` | power-law base + 3 Gaussian shells | exponential SN ejecta |
| `generic_csm_bpl` | power-law base + 3 Gaussian shells | broken power-law SN ejecta |
| `static_powerlaw_csm_exponential` | finite static power-law shell | exponential SN ejecta |
| `static_powerlaw_csm_bpl` | finite static power-law shell | broken power-law SN ejecta |
| `homologous_powerlaw_csm_exponential` | finite homologous power-law shell | exponential SN ejecta |
| `homologous_powerlaw_csm_bpl` | finite homologous power-law shell | broken power-law SN ejecta |
| `generic_4shell_csm_bpl` | power-law base + 4 Gaussian shells | broken power-law SN ejecta |
| `generic_8shell_csm_bpl` | power-law base + 8 Gaussian shells | broken power-law SN ejecta |
| `static_spline_csm_bpl` | finite static spline density profile | broken power-law SN ejecta |
| `generic_spline_csm_bpl` | homologous eight-node spline density profile | broken power-law SN ejecta |
| `generic_spline12_csm_bpl` | homologous twelve-node spline density profile | broken power-law SN ejecta |
| `static_pspline24_csm_bpl` | finite static 24-node p-spline density profile | broken power-law SN ejecta |
| `generic_pspline24_csm_bpl` | homologous 24-node p-spline density profile | broken power-law SN ejecta |
| `static_pspline48_csm_bpl` | finite static 48-node p-spline density profile | broken power-law SN ejecta |
| `generic_pspline48_csm_bpl` | homologous 48-node p-spline density profile | broken power-law SN ejecta |
| `static_pspline96_csm_bpl` | finite static 96-node p-spline density profile | broken power-law SN ejecta |
| `generic_pspline96_csm_bpl` | homologous 96-node p-spline density profile | broken power-law SN ejecta |

### Finite power-law CSM shell models

These are the classic finite-support power-law CSM density profiles,
consistent with e.g., Chatzopoulos+2012. Two variants are exposed:
`static_powerlaw_csm_*` treats the CSM as a static density snapshot, while
`homologous_powerlaw_csm_*` uses the same mass-normalized radial density but
assigns a homologous velocity grid through `v = r / interval_sn`.

- `static_powerlaw_csm_exponential`
- `static_powerlaw_csm_bpl`
- `homologous_powerlaw_csm_exponential`
- `homologous_powerlaw_csm_bpl`
- and the matching optical, radio, and X-ray variants

These have a density profile parameterized as 

\[
\rho(r) = \rho_{\rm in}\left(\frac{r}{r_{\rm inner}}\right)^\eta
\]

for `r_inner <= r <= r_outer`, and zero outside `r_outer`.

Public parameters are:

- `eta` — power-law slope
- `r_inner` — inner shell radius in `cm`
- `r_outer` — outer shell radius in `cm`
- `m_csm` — total CSM mass in `Msun`

For the homologous variants only:

- `interval_sn` — time between CSM ejection and SN explosion in days; this sets
  the homologous CSM velocity field through `v = r / interval_sn`.

The density normalization `rho_in` is derived internally from `m_csm`, so the
public API does not ask for a redundant density parameter.

## Exploration utility

For exploring arbitrary (e.g. stellar-evolution) density profiles without
Bayesian fitting, use `csm_lightcurve_from_density`:

```python
import numpy as np
from redback_csm.explore import csm_lightcurve_from_density

r   = np.geomspace(1e13, 1e17, 500)  # cm
rho = 5e16 / r**2                    # g/cm^3  (r^-2 wind)

fig, lc = csm_lightcurve_from_density(
    radius=r, density=rho,
    t_ref=365.0,       # days — sets r = v * t_ref
    sn_profile='bpl',
    delta=1.0, nn=12.0,
    mexp=10.0, eexp=1.0,
    eff=0.3, kappa=0.34,
)
fig.savefig('my_lightcurve.png')
```

## CSM mass analysis

Arbitrary and spline CSM fits return density fields. The direct mass integral is
a spherical-equivalent value unless you explicitly assume a covering fraction or
volume filling factor:

```python
from redback_csm.analysis import (
    generic_spline_csm_mass_from_params,
    pspline_csm_mass_from_params,
    sample_geometry_corrected_mass,
)

m_spherical = generic_spline_csm_mass_from_params(best_fit_parameters)
# For p-spline fits, use:
# m_spherical = pspline_csm_mass_from_params(best_fit_parameters, profile="generic")
mass_samples = sample_geometry_corrected_mass(
    m_spherical,
    covering_fraction={"kind": "uniform", "min": 0.1, "max": 1.0},
    filling_factor={"kind": "loguniform", "min": 0.01, "max": 1.0},
)
```

## Spline CSM MLE helpers

For fast data-driven CSM reconstruction, the package includes finite spline CSM
models and a small least-squares helper. The public model wrappers are
`static_spline_csm_bpl`, `generic_spline_csm_bpl`, and
`generic_spline12_csm_bpl`. For redback-native p-spline inference, use the
`static_pspline{24,48,96}_csm_bpl` or `generic_pspline{24,48,96}_csm_bpl`
families. These sample an anchor density, an initial log-density slope, and
bounded second differences (`d2_log_rho_*`) instead of independent density
nodes. The smoothness is therefore part of the prior transform used by nested
samplers, not a post-hoc penalty. The generated p-spline priors are
deliberately conservative: `dlog_rho_0` is limited to `[-1, 1]` and each
`d2_log_rho_*` term uses a truncated Gaussian with `sigma=0.1` on `[-0.4, 0.4]`.
Reconstructed log-density nodes are also clipped to the broad physical support
`-30 <= log10(rho) <= -5` in cgs units. The MLE helper can still be used with
more nodes in custom scripts. The included MLE examples are
bolometric. The helper only requires a one-dimensional scalar data vector, so
multiband fitting would need a custom wrapper that flattens all bands into one
residual vector.

```python
from redback_csm.spline_mle import (
    SplineMLEProblem,
    default_spline_bounds,
    make_random_starts,
    spline_parameter_names,
)

names = spline_parameter_names(n_nodes=24, profile="generic", include_nickel=True)
bounds = default_spline_bounds(n_nodes=24, profile="generic", include_nickel=True)
problem = SplineMLEProblem(
    time=time,
    luminosity=lbol,
    error=lbol_err,
    parameter_names=names,
    bounds=bounds,
    model_function=my_model_function,
    profile="generic",
    smoothness_sigma=0.5,
)
starts = make_random_starts(start_params, bounds, names, n_random=8)
result = problem.fit(starts, max_nfev=500)
```

## Installation

### Requirements

- Python >= 3.11
- `gfortran` (for Fortran compilation)
- `redback >= 1.15.0`

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/nikhil-sarin/redback-csm
cd redback-csm

# 2. Compile the Fortran extension
bash setup_fortran.sh

# 3. Install
pip install -e .
```

Optional development extras:

```bash
pip install -e ".[dev]"
pip install -e ".[jax]"   # optional static-CSM JAX backend
```

## Validation and smoke tests

Run the fast smoke tests with:

```bash
python -m pytest tests/test_smoke.py -q
```

Useful release-validation scripts are:

```bash
MPLBACKEND=Agg python examples/08_static_power_law_transport.py
MPLBACKEND=Agg python examples/06_xray.py
```

These regenerate the static finite power-law CSM transport example and the
thermal bremsstrahlung X-ray example.

## Quick start

```python
import numpy as np
import redback

# All CSM models are automatically in redback's model library after install
from redback.model_library import all_models_dict
print('wind_bpl_bolometric' in all_models_dict)  # True

# Use a bolometric model directly
from redback_csm.models import wind_bpl_bolometric

time = np.linspace(1, 300, 200)  # days, source frame
lbol = wind_bpl_bolometric(
    time=time,
    mdot=1e-3,    # M_sun/yr
    vwind=100,    # km/s
    delta=0.5,
    nn=12,
    mexp=10.0,    # M_sun
    eexp=1.0,     # foe
    eff=0.5,
    kappa=0.34,   # cm^2/g — enables photon diffusion (optional)
)

# Transport mode
lbol_transport = wind_bpl_bolometric(
    time=time,
    mdot=1e-3,
    vwind=100,
    delta=0.5,
    nn=12,
    mexp=10.0,
    eexp=1.0,
    eff=0.5,
    mode='transport',
    kappa=0.34,
    n_rad_zones=160,
)

# Multiband (per-band magnitudes)
from redback_csm.models import wind_bpl

time_obs = np.geomspace(1, 500, 200)  # days, observer frame
mag = wind_bpl(
    time=time_obs, redshift=0.02,
    mdot=1e-3, vwind=100, delta=0.5, nn=12,
    mexp=5.0, eexp=1.0, eff=0.5,
    temperature_floor=3000.0,
    bands='sdssg',
    output_format='magnitude',
)

# Radio synchrotron
from redback_csm.models import wind_bpl_radio

flux_mJy = wind_bpl_radio(
    time=time_obs, redshift=0.02,
    mdot=1e-3, vwind=100, delta=0.5, nn=12,
    mexp=5.0, eexp=1.0, eff=0.5,
    logepsb=-2, logepse=-1, p=3.0,
    frequency=8.4e9,  # Hz
    output_format='flux_density',
)

# Thermal bremsstrahlung X-rays
from redback_csm.models import wind_bpl_xray

lx = wind_bpl_xray(
    time=time_obs, redshift=0.02,
    mdot=1e-3, vwind=100, delta=0.5, nn=12,
    mexp=5.0, eexp=1.0, eff=0.5,
    logepsx=-1.0,
    e_min_kev=0.3, e_max_kev=10.0,
    output_format='luminosity',
)
```

## Fitting with redback

After installation, the CSM wrappers are registered with redback's model
library and their priors are generated automatically from the model signatures.
For a multiband fit, use the unsuffixed model name and the matching unsuffixed
prior. The example below generates a tiny synthetic transient in memory, so it
does not require any catalogue downloads or local data files:

```python
import numpy as np
import redback

from redback_csm.models import wind_bpl

time = np.geomspace(2.0, 120.0, 20)
true_parameters = dict(
    redshift=0.02,
    mdot=1e-3,
    vwind=100.0,
    delta=0.5,
    nn=12.0,
    mexp=5.0,
    eexp=1.0,
    eff=0.5,
    temperature_floor=3000.0,
    bands="sdssr",
    output_format="magnitude",
)
mag = wind_bpl(time=time, **true_parameters)
mag_err = np.full_like(mag, 0.1)

transient = redback.transient.Supernova(
    name="synthetic_csm_demo",
    data_mode="magnitude",
    time=time,
    magnitude=mag,
    magnitude_err=mag_err,
    bands=np.array(["sdssr"] * len(time)),
)
priors = redback.priors.get_priors("wind_bpl")

result = redback.fit_model(
    transient=transient,
    prior=priors,
    model="wind_bpl",
    model_kwargs={
        "output_format": "magnitude",
        "bands": transient.bands,
    },
    sampler="dynesty",
    nlive=500,
)
```

For observed multiband photometry in MJD, use Redback's phase/extinction
wrapper rather than adding phase or dust parameters by hand. The CSM model is
passed as the plugin base model:

```python
result = redback.fit_model(
    transient=transient,  # use_phase_model=True and time_mjd populated
    prior=priors,         # include t0 and av_host
    model="t0_supernova_extinction",
    model_kwargs={
        "base_model": "wind_bpl",
        "output_format": "magnitude",
        "bands": transient.bands,
        "kappa": 0.1,
        "temperature_floor": 3500.0,
        "av_mw": 0.02,
    },
)
```

See `examples/09_redback_photometry_phase_extinction.py` for a complete
`ztfg`/`ztfr`/`sdssu` simulation and recovery that samples both `t0` and
`av_host`.

For bolometric luminosity data, use the bolometric wrapper and matching
bolometric prior instead. Redback's luminosity transient stores luminosity in
units of `1e50 erg/s`, so wrap the model by the same factor:

```python
import numpy as np
import redback

from redback_csm.models import wind_bpl_bolometric

time = np.geomspace(2.0, 120.0, 20)
lbol = wind_bpl_bolometric(
    time=time,
    mdot=1e-3,
    vwind=100.0,
    delta=0.5,
    nn=12.0,
    mexp=5.0,
    eexp=1.0,
    eff=0.5,
    kappa=0.34,
)
transient = redback.transient.Supernova(
    name="synthetic_csm_bolometric_demo",
    data_mode="luminosity",
    time_rest_frame=time,
    Lum50=lbol / 1e50,
    Lum50_err=0.1 * lbol / 1e50,
)
priors = redback.priors.get_priors("wind_bpl_bolometric")

def model_lum50(time, **kwargs):
    return wind_bpl_bolometric(time=time, **kwargs) / 1e50

result = redback.fit_model(
    transient=transient,
    prior=priors,
    model=model_lum50,
    sampler="dynesty",
    nlive=500,
)
```

For p-spline CSM inference with redback, use one of the packaged p-spline
wrappers. These are usually a better nested-sampling target than fitting
independent `log_rho_*` nodes:

```python
# Reuse the synthetic luminosity transient from the bolometric example above,
# or replace this with your own redback.transient.Supernova luminosity object.
bolometric_transient = transient

from redback_csm.models import generic_pspline24_csm_bpl_bolometric

priors = redback.priors.get_priors("generic_pspline24_csm_bpl_bolometric")

# Optional event-specific tightening. The d2_log_rho_* priors control
# roughness; the generated defaults are already fairly smooth, but these
# bounds are usually worth tightening for a specific transient.
priors["log_r_inner"].minimum = 13.0
priors["log_r_inner"].maximum = 15.0
priors["log_r_outer"].minimum = 15.0
priors["log_r_outer"].maximum = 17.5
priors["interval_sn"].minimum = 30.0
priors["interval_sn"].maximum = 3000.0

def pspline_model_lum50(time, **kwargs):
    return generic_pspline24_csm_bpl_bolometric(time=time, **kwargs) / 1e50

result = redback.fit_model(
    transient=bolometric_transient,
    prior=priors,
    model=pspline_model_lum50,
    sampler="dynesty",
    nlive=1000,
)
```

## Unit conventions

All model functions use astronomy-friendly input units:

| Quantity | Unit |
|---|---|
| Time | days |
| Mass | M☉ |
| Energy | foe (10⁵¹ erg) |
| Mass-loss rate | M☉ yr⁻¹ |
| Velocity | km/s |
| Opacity | cm²/g |
| Radio frequency | Hz |
| X-ray energy | keV |

Outputs follow redback conventions: flux density in mJy, magnitudes in AB system.
