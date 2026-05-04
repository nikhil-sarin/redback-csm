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

Each of the 24 CSM scenarios is exposed as four functions:

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

### Shared Runtime Options

All optical / bolometric CSM wrappers also accept a common set of runtime
keywords controlling the transport treatment:

| Keyword | Meaning |
|---|---|
| `mode='simple'` | Default thin-shell calculation. If `kappa` is provided, this uses the legacy post-processed diffusion light curve. |
| `mode='transport'` | Use the newer transport solver for the observed luminosity while keeping the same shell dynamics. |
| `kappa` | Opacity in `cm^2 g^-1`. Optional in simple mode. In transport mode, defaults to `0.34` if not supplied. |
| `n_rad_zones` | Number of radiation/transport zones in transport mode. Higher values reduce numerical roughness but increase runtime. |
| `efficiency_mode` | Optional shock-efficiency mode. Default is `0`, which applies the user-supplied constant `eff` to both shocks; `1` applies the free-free-limited time-dependent efficiency to both forward and reverse shocks. |

Current output convention:

- `lbol` is the main observable luminosity
- `lbol_shock` is the shock-powered luminosity
- `lbol_diffuse` is the diffusion / transport luminosity when available
- `rph` is the historical output field name, but currently stores the shell radius by convention

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

Each of the 24 CSM scenarios also has a `{name}_radio` function that computes
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

Each radio-enabled CSM scenario also has a matching `{name}_xray` function. The
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

### Available CSM scenarios (24 total)

**Steady wind CSM**
- `wind_exponential` — steady wind CSM + exponential SN ejecta
- `wind_bpl` — steady wind CSM + broken power-law SN ejecta
- `exponential_wind` — exponential outer shell + inner wind ejecta (non-SN transients)
- `bpl_wind` — broken power-law outer shell + inner wind ejecta (non-SN transients)

**Two-component eruption + explosion**
- `exponential_exponential` — outer exponential shell + inner exponential SN
- `exponential_bpl` — outer exponential shell + inner BPL SN
- `bpl_bpl` — outer BPL shell + inner BPL SN
- `bpl_exponential` — outer BPL shell + inner exponential SN

**Shaped wind profiles**
- `boxwind_exponential` — box-shaped (constant interval) wind CSM + exponential SN
- `boxwind_bpl` — box-shaped wind CSM + BPL SN
- `gausswind_exponential` — Gaussian-varying mass-loss rate wind CSM + exponential SN
- `gausswind_bpl` — Gaussian-varying mass-loss rate wind CSM + BPL SN

**Triple power-law winds**
- `triple_powerlaw_wind_bpl` — piecewise triple power-law wind CSM + BPL SN
- `triple_powerlaw_wind_exponential` — piecewise triple power-law wind CSM + exponential SN
- `exponential_triple_powerlaw_wind` — outer exponential shell + inner triple power-law wind
- `bpl_triple_powerlaw_wind` — outer BPL shell + inner triple power-law wind
- `smooth_triple_powerlaw_wind_bpl` — smooth (tanh-connected) triple power-law wind + BPL SN
- `smooth_triple_powerlaw_wind_exponential` — smooth triple power-law wind + exponential SN

**Generic phenomenological CSM**
- `generic_csm_exponential` — power-law base density + 3 shells + exponential SN
- `generic_csm_bpl` — power-law base density + 3 shells + BPL SN
- `generic_4shell_csm_bpl` — power-law base density + 4 shells + BPL SN
- `generic_8shell_csm_bpl` — power-law base density + 8 shells + BPL SN

### New finite power-law CSM shell models

This is the classic power-law finite-support density profiles, 
consistent with e.g., Chatzopoulos+2012 

- `generic_powerlaw_csm_exponential`
- `generic_powerlaw_csm_bpl`
- and the matching `_bolometric` / `_radio` variants

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
    n_rad_zones=120,
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

# Get priors for Bayesian fitting
priors = redback.priors.get_priors('wind_bpl_bolometric')

# Fit to data using redback
transient = redback.transient.Supernova.from_open_access_catalogue('SN2010jl')
result = redback.fit_model(
    transient=transient,
    prior=priors,
    model='wind_bpl',
    sampler='dynesty',
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
