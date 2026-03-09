"""
Basic smoke tests for redback-csm.

Run with: pytest tests/

Note: tests that call model functions require the Fortran extension to be compiled.
Run `bash setup_fortran.sh && pip install -e .` before running those tests.
"""

import pytest


def test_package_imports():
    import redback_csm

    assert redback_csm.__version__ == "0.1.0"


def test_prior_provider_known_model():
    from redback_csm.prior_provider import get_prior

    prior = get_prior("wind_bpl_bolometric")
    assert prior is not None
    assert "mdot" in prior
    assert "vwind" in prior
    assert "mexp" in prior
    assert "eexp" in prior
    assert "eff" in prior
    # bolometric prior should NOT contain redshift
    assert "redshift" not in prior


def test_prior_provider_multiband():
    from redback_csm.prior_provider import get_prior

    prior = get_prior("wind_bpl")
    assert prior is not None
    assert "redshift" in prior
    assert "temperature_floor" in prior


def test_prior_provider_nickel():
    from redback_csm.prior_provider import get_prior

    prior = get_prior("wind_bpl_nickel_bolometric")
    assert prior is not None
    assert "f_nickel" in prior
    assert "mej" in prior
    assert "kappa" in prior


def test_prior_provider_unknown():
    from redback_csm.prior_provider import get_prior

    assert get_prior("not_a_real_model_xyz") is None


def test_all_prior_files_exist():
    """Check that all 88 expected prior files exist."""
    from redback_csm.prior_provider import PRIOR_DIR
    import os

    models = [
        "wind_exponential",
        "wind_bpl",
        "exponential_wind",
        "bpl_wind",
        "exponential_exponential",
        "exponential_bpl",
        "bpl_bpl",
        "bpl_exponential",
        "boxwind_exponential",
        "boxwind_bpl",
        "gausswind_exponential",
        "gausswind_bpl",
        "triple_powerlaw_wind_bpl",
        "triple_powerlaw_wind_exponential",
        "exponential_triple_powerlaw_wind",
        "bpl_triple_powerlaw_wind",
        "smooth_triple_powerlaw_wind_bpl",
        "smooth_triple_powerlaw_wind_exponential",
        "generic_csm_exponential",
        "generic_csm_bpl",
        "generic_4shell_csm_bpl",
        "generic_8shell_csm_bpl",
    ]
    suffixes = ["_bolometric", "", "_nickel_bolometric", "_nickel"]
    missing = []
    for m in models:
        for s in suffixes:
            fname = f"{m}{s}.prior"
            if not os.path.exists(os.path.join(PRIOR_DIR, fname)):
                missing.append(fname)
    assert not missing, f"Missing prior files: {missing}"


def test_models_registered_in_redback():
    """After install, CSM models should appear in redback's model library."""
    try:
        import redback
        from redback.model_library import all_models_dict

        expected = [
            "wind_bpl_bolometric",
            "wind_bpl",
            "wind_bpl_nickel_bolometric",
            "wind_bpl_nickel",
            "gausswind_bpl_bolometric",
            "gausswind_bpl",
            "smooth_triple_powerlaw_wind_bpl_bolometric",
        ]
        missing = [m for m in expected if m not in all_models_dict]
        assert not missing, f"Models missing from redback: {missing}"
    except ImportError:
        pytest.skip("redback not installed")


def test_bolometric_model_callable():
    """wind_bpl_bolometric should run and return an array of the right length."""
    try:
        from redback_csm.models import wind_bpl_bolometric
        import numpy as np

        time = np.linspace(1, 300, 50)
        result = wind_bpl_bolometric(
            time=time,
            mdot=1e-3,
            vwind=100.0,
            delta=0.5,
            nn=12.0,
            mexp=10.0,
            eexp=1.0,
            eff=0.5,
            kappa=0.34,
        )
        assert len(result) == 50
        assert np.all(result >= 0)
    except ImportError:
        pytest.skip("Fortran extension not compiled")


def test_prior_latex_labels_render():
    """All latex_labels in every prior file must render without error in matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from redback_csm.prior_provider import PRIOR_DIR
    import os
    from bilby.core.prior import PriorDict

    prior_files = sorted(f for f in os.listdir(PRIOR_DIR) if f.endswith(".prior"))
    assert prior_files, "No prior files found"

    fig, ax = plt.subplots()
    failures = []
    for fname in prior_files:
        priors = PriorDict()
        priors.from_file(os.path.join(PRIOR_DIR, fname))
        for param, prior in priors.items():
            try:
                ax.set_xlabel(prior.latex_label)
                fig.canvas.draw()
            except Exception as e:
                failures.append(f"{fname}::{param} — {prior.latex_label!r} → {e}")
    plt.close(fig)
    assert not failures, "Latex label rendering failures:\n" + "\n".join(failures)


def test_dispatch_registry():
    """_DISPATCH should contain all 22 expected model keys."""
    from redback_csm.core import _DISPATCH

    expected_keys = [
        "wind_exponential",
        "wind_bpl",
        "exponential_wind",
        "bpl_wind",
        "exponential_exponential",
        "exponential_bpl",
        "bpl_bpl",
        "bpl_exponential",
        "boxwind_exponential",
        "boxwind_bpl",
        "gausswind_exponential",
        "gausswind_bpl",
        "triple_powerlaw_wind_bpl",
        "triple_powerlaw_wind_exponential",
        "exponential_triple_powerlaw_wind",
        "bpl_triple_powerlaw_wind",
        "smooth_triple_powerlaw_wind_bpl",
        "smooth_triple_powerlaw_wind_exponential",
        "generic_csm_exponential",
        "generic_csm_bpl",
        "generic_4shell_csm_bpl",
        "generic_8shell_csm_bpl",
    ]
    missing = [k for k in expected_keys if k not in _DISPATCH]
    assert not missing, f"Missing dispatch keys: {missing}"
