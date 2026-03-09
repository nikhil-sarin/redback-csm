# redback-csm

Fortran-based circumstellar matter (CSM) interaction models for the
[redback](https://github.com/nikhil-sarin/redback) transient inference package.

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

Each of the 22 CSM scenarios is exposed as four functions:

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
emission. The ejecta mass and velocity for nickel diffusion are derived internally
from `mexp` and `eexp`, so no additional mass parameter is needed.

### Radio synchrotron model variants

Each of the 22 CSM scenarios also has a `{name}_radio` function that computes
synchrotron radio emission from the CSM-interaction shock using the
Chevalier (1998) formalism with self-absorption. These take additional parameters:

| Parameter | Description |
|---|---|
| `logepsb` | log₁₀(ε_B), magnetic energy fraction of shock ram pressure |
| `logepse` | log₁₀(ε_e), electron energy fraction |
| `p` | Electron power-law index (typically 2–4) |
| `frequency` | Observing frequency in Hz |

Output is flux density in mJy.

### Available CSM scenarios (22 total)

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

- Python >= 3.10
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

# Get priors for Bayesian fitting
priors = redback.priors.get_priors('wind_bpl_bolometric')

# Fit to data using redback
transient = redback.transient.Supernova.from_open_access_catalogue('SN2010jl')
result = redback.fit_model(
    transient=transient,
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

Outputs follow redback conventions: flux density in mJy, magnitudes in AB system.
