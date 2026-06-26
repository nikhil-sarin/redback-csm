"""
Simulate and recover multiband CSM photometry with redback.

This example uses Redback's built-in phase/extinction wrapper,
``t0_supernova_extinction``, with the redback-csm plugin base model
``wind_bpl``. This is the recommended path when fitting observed
photometry in MJD while sampling both the explosion epoch ``t0`` and host
extinction ``av_host``.

Run a fast synthetic-data check only:

    python examples/09_redback_photometry_phase_extinction.py --no-fit

Run a small recovery:

    python examples/09_redback_photometry_phase_extinction.py --nlive 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import bilby
import matplotlib.pyplot as plt
import numpy as np
import redback
from redback.transient_models.phase_models import t0_supernova_extinction


OUTDIR = Path("examples/photometry_phase_extinction_demo")
LABEL = "wind_bpl_phase_extinction"

TRUE = dict(
    t0=60000.0,
    av_host=0.12,
    redshift=0.03,
    mdot=1.5e-3,
    vwind=120.0,
    delta=0.5,
    nn=12.0,
    mexp=7.0,
    eexp=2.0,
    eff=0.4,
    vej_max_ratio=5.0,
)

SAMPLED_PARAMETERS = ("mdot", "vwind", "mexp", "eexp", "eff", "t0", "av_host")

FIXED_MODEL_KWARGS = dict(
    base_model="wind_bpl",
    output_format="magnitude",
    kappa=0.10,
    temperature_floor=3500.0,
    av_mw=0.02,
    rv_host=3.1,
    rv_mw=3.1,
    host_law="fitzpatrick99",
    mw_law="fitzpatrick99",
)


def build_injection_parameters():
    """Return the true values for parameters shown in the recovery corner plot."""
    return {key: TRUE[key] for key in SAMPLED_PARAMETERS}


def make_synthetic_photometry(seed: int = 12):
    """Generate toy ztfg/ztfr/sdssu photometry in observed MJD."""
    rng = np.random.default_rng(seed)
    # Keep every synthetic detection safely after any allowed t0 value. The
    # Redback phase/extinction wrapper masks pre-explosion times but currently
    # does not apply the same mask to per-point band arrays.
    phases = np.geomspace(8.0, 120.0, 12)
    bands = np.array(["ztfg", "ztfr", "sdssu"])

    time_mjd = []
    band_array = []
    for index, band in enumerate(bands):
        # Small offsets avoid perfectly simultaneous synthetic points.
        time_mjd.append(TRUE["t0"] + phases + 0.7 * index)
        band_array.append(np.full_like(phases, band, dtype=object))
    time_mjd = np.concatenate(time_mjd).astype(float)
    band_array = np.concatenate(band_array).astype(str)

    order = np.argsort(time_mjd)
    time_mjd = time_mjd[order]
    band_array = band_array[order]

    true_params = dict(TRUE)
    true_params.update(FIXED_MODEL_KWARGS)
    true_params["bands"] = band_array
    magnitude = t0_supernova_extinction(time_mjd, **true_params)

    mag_err = np.full_like(magnitude, 0.08, dtype=float)
    observed = magnitude + rng.normal(0.0, mag_err)
    return time_mjd, band_array, observed, mag_err, magnitude


def build_transient(time_mjd, bands, magnitude, magnitude_err):
    return redback.transient.Supernova(
        name="synthetic_wind_bpl_phase_extinction",
        data_mode="magnitude",
        time_mjd=time_mjd,
        magnitude=magnitude,
        magnitude_err=magnitude_err,
        bands=bands,
        active_bands="all",
        plotting_order=np.array(["sdssu", "ztfg", "ztfr"]),
        use_phase_model=True,
    )


def build_priors():
    priors = redback.priors.get_priors("wind_bpl")

    priors["t0"] = bilby.core.prior.Uniform(
        TRUE["t0"] - 4.0, TRUE["t0"] + 4.0, name="t0", latex_label=r"$t_0$"
    )
    priors["av_host"] = bilby.core.prior.Uniform(
        0.0, 0.4, name="av_host", latex_label=r"$A_{V,\rm host}$"
    )

    # Keep this toy recovery low-dimensional. For real data, broaden these
    # priors only after checking that random prior draws are numerically sane.
    priors["redshift"] = bilby.core.prior.DeltaFunction(TRUE["redshift"], name="redshift")
    priors["delta"] = bilby.core.prior.DeltaFunction(TRUE["delta"], name="delta")
    priors["nn"] = bilby.core.prior.DeltaFunction(TRUE["nn"], name="nn")
    priors["vej_max_ratio"] = bilby.core.prior.DeltaFunction(
        TRUE["vej_max_ratio"], name="vej_max_ratio"
    )

    priors["mdot"] = bilby.core.prior.LogUniform(5e-4, 8e-3, name="mdot")
    priors["vwind"] = bilby.core.prior.Uniform(50.0, 250.0, name="vwind")
    priors["mexp"] = bilby.core.prior.Uniform(2.0, 8.0, name="mexp")
    priors["eexp"] = bilby.core.prior.Uniform(0.5, 4.0, name="eexp")
    priors["eff"] = bilby.core.prior.Uniform(0.1, 0.8, name="eff")
    return priors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-fit", action="store_true", help="only generate and plot synthetic data")
    parser.add_argument("--nlive", type=int, default=300, help="dynesty live points for the toy recovery")
    parser.add_argument("--seed", type=int, default=12, help="random seed for synthetic photometry")
    args = parser.parse_args()

    time_mjd, bands, observed, mag_err, noiseless = make_synthetic_photometry(seed=args.seed)

    transient = build_transient(time_mjd, bands, observed, mag_err)
    transient.plot_data(save=False)
    if args.no_fit:
        return

    model_kwargs = dict(FIXED_MODEL_KWARGS)
    model_kwargs["bands"] = transient.bands
    injection_parameters = build_injection_parameters()

    result = redback.fit_model(
        transient=transient,
        model="t0_supernova_extinction",
        prior=build_priors(),
        model_kwargs=model_kwargs,
        sampler="pymultinest",
        nlive=args.nlive,
        outdir=str(OUTDIR),
        label=LABEL,
        resume=False,
        plot=False,
        clean=False,
        injection_parameters=injection_parameters,
    )
    print(result)
    lightcurve_path = OUTDIR / f"{LABEL}_lightcurve.png"
    corner_path = OUTDIR / f"{LABEL}_corner.png"
    result.plot_lightcurve(outdir=str(OUTDIR), label=str(LABEL))
    result.plot_corner(filename=str(corner_path))
    print(f"Saved {lightcurve_path}")
    print(f"Saved {corner_path}")


if __name__ == "__main__":
    main()
