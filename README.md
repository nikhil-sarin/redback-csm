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

- **Sarin & Hirai (in prep)** — the paper describing the CSM interaction models
- **Sarin et al. 2024** — the redback paper
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

The package currently exposes 29 base CSM-density/ejecta scenarios. These are
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
| `n_rad_zones` | Number of radiation/transport zones in transport mode. Python model calls promote values below 40 to 40 because the transport boundary layer is under-resolved below that. Higher values reduce numerical roughness but increase runtime. |
| `transport_wind_inner_age` | Effective inner wind age in years for wind-history CSMs in transport mode. If omitted, generated wind histories use the larger of their first tabulated age and `1 yr`; one-point steady winds use `1 yr`. This creates the finite inner cavity needed by the nondimensional transport setup and is ignored in simple mode. |
| `transport_wind_age` | Effective outer wind age in years for wind-history CSMs in transport mode. If omitted, generated wind histories use their last tabulated age; one-point steady winds use `100 yr`. This supplies the finite outer edge required by the transport solver and is ignored in simple mode. |
| `transport_r_inner` / `r_inner` | Alternative direct inner cutoff radius in cm for wind-history CSMs in transport mode. `transport_r_inner` takes precedence. |
| `transport_r_outer` / `r_outer` | Alternative direct outer cutoff radius in cm for wind-history CSMs in transport mode. `transport_r_outer` takes precedence. |
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
- `rph` is the historical output field name, but currently stores the shell radius by convention

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

This release includes a JAX-native prototype for fast inference with static
finite power-law CSM shells and BPL ejecta:

```python
from jax_csm.model import get_static_powerlaw_csm_bpl_lightcurve

lbol = get_static_powerlaw_csm_bpl_lightcurve(
    time=time_days,
    eta=-2.0,
    r_inner=5e2 * 6.957e10,
    r_outer=5e3 * 6.957e10,
    delta_sn=1.0,
    nn_sn=10.0,
    mej_sn=5.0,
    esn=1.0,
    eff=1.0,
    m_csm=1.0,
    mode="simple",
)
```

`mode="simple"` shares the current static-shell mass normalization and finite
BPL ejecta cutoff convention with the Fortran backend. `mode="transport"` uses
the same dimensionless diffusion setup as the Fortran static power-law/BPL
transport path, including breakout-reservoir leakage and homologous
post-emergence cooling. It is still scoped to static finite power-law CSMs,
not every arbitrary CSM constructor supported by the Fortran backend.

### Radio synchrotron model variants

Each base CSM scenario also has a `{name}_radio` function that computes
synchrotron radio emission from the CSM-interaction shock using the
Chevalier (1998) formalism with self-absorption. These take additional parameters:

| Parameter | Description |
|---|---|
| `logepsb` | log₁₀(ε_B), magnetic energy fraction of shock ram pressure |
| `logepse` | log₁₀(ε_e), electron energy fraction |
| `p` | Electron power-law index (typically 2–4) |
| `frequency` | Observing frequency in Hz |

Output is flux density in mJy.

### X-ray thermal bremsstrahlung model variants

Each base CSM scenario also has a matching `{name}_xray` function. The
X-ray layer is a fast post-processor: it uses the CSM shock evolution and
upstream density, sets the post-shock plasma temperature from the strong-shock
velocity, estimates the shocked-CSM emission measure, and emits a thermal
free-free spectrum. The free-free luminosity is capped by the shock power by
default.

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

### Available base CSM-density/ejecta scenarios (29 total)

These are the unsuffixed physical configurations. Each base scenario has
generated optical, nickel, radio, and X-ray wrapper variants, so the public
model function count is larger than 29: 29 base scenarios times six wrapper
forms (`{name}`, `{name}_bolometric`, `{name}_nickel`,
`{name}_nickel_bolometric`, `{name}_radio`, `{name}_xray`), plus the generic
`csm_xray` helper, for 175 public callables.

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

fig = csm_lightcurve_from_density(
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
    sample_geometry_corrected_mass,
)

m_spherical = generic_spline_csm_mass_from_params(best_fit_parameters)
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
`generic_spline12_csm_bpl`; the MLE helper can also be used with more nodes in
custom scripts. The included MLE examples are bolometric. The helper only
requires a one-dimensional scalar data vector, so multiband fitting would need a
custom wrapper that flattens all bands into one residual vector.

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
    n_rad_zones=20,
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
prior:

```python
import redback

transient = redback.transient.Supernova.from_open_access_catalogue("SN2010jl")
priors = redback.priors.get_priors("wind_bpl")

result = redback.fit_model(
    transient=transient,
    prior=priors,
    model="wind_bpl",
    sampler="dynesty",
    nlive=500,
)
```

For bolometric luminosity data, use the bolometric wrapper and matching
bolometric prior instead:

```python
priors = redback.priors.get_priors("wind_bpl_bolometric")
result = redback.fit_model(
    transient=bolometric_transient,
    prior=priors,
    model="wind_bpl_bolometric",
    sampler="dynesty",
    nlive=500,
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
