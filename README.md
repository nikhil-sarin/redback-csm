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

The package provides 22 CSM interaction scenarios, each exposed as four functions:

| Suffix | Description |
|---|---|
| `{name}_bolometric` | Bolometric luminosity, source-frame time |
| `{name}` | Multiband (flux density / magnitude / spectra), observer-frame time |
| `{name}_nickel_bolometric` | CSM + radioactive nickel, bolometric |
| `{name}_nickel` | CSM + radioactive nickel, multiband |

### Available models

**Wind / simple CSM**
- `wind_exponential` — steady wind CSM + exponential SN ejecta
- `wind_bpl` — steady wind CSM + broken power-law SN ejecta
- `exponential_wind` — exponential explosion into surrounding wind
- `bpl_wind` — broken power-law explosion into surrounding wind

**Eruption–explosion interactions**
- `exponential_exponential` — eruption (exponential) + SN (exponential)
- `exponential_bpl` — eruption (exponential) + SN (broken power-law)
- `bpl_bpl` — eruption (BPL) + SN (broken power-law)
- `bpl_exponential` — eruption (BPL) + SN (exponential)

**Shaped wind profiles**
- `boxwind_exponential` — box-shaped wind eruption + exponential SN
- `boxwind_bpl` — box-shaped wind eruption + BPL SN
- `gausswind_exponential` — Gaussian mass-loss eruption + exponential SN
- `gausswind_bpl` — Gaussian mass-loss eruption + BPL SN

**Triple power-law winds**
- `triple_powerlaw_wind_bpl` — triple power-law wind + BPL SN
- `triple_powerlaw_wind_exponential` — triple power-law wind + exponential SN
- `exponential_triple_powerlaw_wind` — exponential SN into triple power-law wind
- `bpl_triple_powerlaw_wind` — BPL SN into triple power-law wind
- `smooth_triple_powerlaw_wind_bpl` — smooth (tanh) triple power-law wind + BPL SN
- `smooth_triple_powerlaw_wind_exponential` — smooth triple power-law wind + exponential SN

**Generic phenomenological CSM**
- `generic_csm_exponential` — arbitrary density profile (base + 3 shells) + exponential SN
- `generic_csm_bpl` — arbitrary density profile (base + 3 shells) + BPL SN
- `generic_4shell_csm_bpl` — arbitrary density profile (base + 4 shells) + BPL SN
- `generic_8shell_csm_bpl` — arbitrary density profile (base + 8 shells) + BPL SN

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

# All CSM models are now in redback's model library
from redback.model_library import all_models_dict
print('wind_bpl_bolometric' in all_models_dict)  # True

# Use a model directly
from redback_csm.models import wind_bpl_bolometric

time = np.linspace(1, 300, 200)  # days
lbol = wind_bpl_bolometric(
    time=time,
    mdot=1e-3,    # M_sun/yr
    vwind=100,    # km/s
    delta=0.5,
    nn=12,
    mexp=10.0,    # M_sun
    eexp=1.0,     # foe
    eff=0.5,
    kappa=0.34,   # cm^2/g (enables photon diffusion)
)

# Get priors for Bayesian fitting
priors = redback.priors.get_priors('wind_bpl_bolometric')

# Fit to data using redback
transient = redback.transient.Supernova.from_open_transient_catalog_data('SN2010jl')
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
| Mass-loss rate | M☉/yr |
| Velocity | km/s |
| Opacity | cm²/g |

Outputs follow redback conventions (flux in mJy, magnitudes in AB system, etc.).
