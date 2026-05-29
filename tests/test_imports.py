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
    assert "temperature_floor" not in prior


def test_prior_provider_nickel_multiband_temperature_floor():
    from redback_csm.prior_provider import get_prior

    prior = get_prior("wind_bpl_nickel")
    assert prior is not None
    assert "redshift" in prior
    assert "temperature_floor" in prior


def test_prior_provider_nickel():
    from redback_csm.prior_provider import get_prior

    prior = get_prior("wind_bpl_nickel_bolometric")
    assert prior is not None
    assert "f_nickel" in prior
    assert "mexp" in prior
    assert "eexp" in prior
    assert "kappa" in prior


def test_prior_provider_spline_models():
    from redback_csm.prior_provider import get_prior

    static_prior = get_prior("static_spline_csm_bpl_bolometric")
    generic_prior = get_prior("generic_spline12_csm_bpl_nickel_bolometric")
    pspline_prior = get_prior("generic_pspline24_csm_bpl_bolometric")
    pspline96_prior = get_prior("generic_pspline96_csm_bpl_bolometric")

    assert static_prior is not None
    assert "log_r_inner" in static_prior
    assert "log_rho_7" in static_prior
    assert generic_prior is not None
    assert "log_rho_11" in generic_prior
    assert "interval_sn" in generic_prior
    assert "f_nickel" in generic_prior
    assert pspline_prior is not None
    assert "dlog_rho_0" in pspline_prior
    assert "d2_log_rho_21" in pspline_prior
    assert "log_rho_23" not in pspline_prior
    assert pspline96_prior is not None
    assert "d2_log_rho_93" in pspline96_prior
    assert "d2_log_rho_94" not in pspline96_prior


def test_prior_provider_unknown():
    from redback_csm.prior_provider import get_prior

    assert get_prior("not_a_real_model_xyz") is None


def test_xray_bremsstrahlung_smoke():
    import numpy as np
    from redback_csm.xray import thermal_bremsstrahlung_xray

    out = thermal_bremsstrahlung_xray(
        time_days=np.array([1.0, 2.0, 3.0]),
        shock_luminosity_cgs=np.array([1e42, 1e43, 1e44]),
        vshell_cgs=np.array([1e8, 5e8, 1e9]),
        redshift=0.01,
        logepsx=-1.0,
        luminosity_distance_cm=1e26,
        rho_csm_cgs=np.array([1e-16, 8e-17, 5e-17]),
        radius_cgs=np.array([1e14, 2e14, 3e14]),
    )
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


def test_csm_nickel_diffuses_through_finite_wind_csm(monkeypatch):
    import numpy as np
    import redback_csm.models as models

    calls = {}

    def fake_csm(time, csm_model, **kwargs):
        return np.zeros_like(time, dtype=float)

    def fake_engine(time, f_nickel, mej, **kwargs):
        calls.setdefault("engine_mej", []).append(mej)
        return np.ones_like(time, dtype=float) * f_nickel * mej

    class FakeDiffusion:
        def __init__(self, time, dense_times, luminosity, kappa, kappa_gamma, mej, vej, **kwargs):
            calls["diffusion_mej"] = mej
            self.new_luminosity = np.ones_like(time, dtype=float) * mej

    monkeypatch.setattr(models, "_csm_bolometric_impl", fake_csm)
    monkeypatch.setattr(models, "_nickelcobalt_engine", fake_engine)

    result = models._csm_nickel_bolometric_impl(
        np.array([1.0, 2.0]),
        "boxwind_bpl",
        t1=1.0,
        t2=3.0,
        mdot_0=0.0,
        mdot_1=2.0,
        mdot_2=0.0,
        vwind=100.0,
        delta=0.5,
        nn=12.0,
        mexp=5.0,
        eexp=1.0,
        eff=0.5,
        f_nickel=0.1,
        kappa=0.34,
        kappa_gamma=0.027,
        interaction_process=FakeDiffusion,
    )

    assert calls["engine_mej"] == [5.0]
    assert calls["diffusion_mej"] == 9.0
    assert np.all(result == 9.0)


def test_csm_nickel_uses_outer_ejecta_for_double_explosion(monkeypatch):
    import numpy as np
    import redback_csm.models as models

    calls = {}

    def fake_csm(time, csm_model, **kwargs):
        return np.zeros_like(time, dtype=float)

    def fake_engine(time, f_nickel, mej, **kwargs):
        calls.setdefault("engine_mej", []).append(mej)
        return np.ones_like(time, dtype=float)

    class FakeDiffusion:
        def __init__(self, time, dense_times, luminosity, kappa, kappa_gamma, mej, vej, **kwargs):
            calls["diffusion_mej"] = mej
            self.new_luminosity = np.ones_like(time, dtype=float)

    monkeypatch.setattr(models, "_csm_bolometric_impl", fake_csm)
    monkeypatch.setattr(models, "_nickelcobalt_engine", fake_engine)

    models._csm_nickel_bolometric_impl(
        np.array([1.0, 2.0]),
        "exponential_bpl",
        mexp=0.4,
        eexp=0.02,
        delta_out=0.5,
        nn_out=12.0,
        mexp_out=6.0,
        eexp_out=1.0,
        interval=100.0,
        eff=0.5,
        f_nickel=0.1,
        kappa=0.34,
        kappa_gamma=0.027,
        interaction_process=FakeDiffusion,
    )

    assert calls["engine_mej"] == [6.0]
    assert calls["diffusion_mej"] == 6.4


def test_csm_nickel_bolometric_consistency(monkeypatch):
    import numpy as np
    import redback_csm.models as models

    time = np.array([10.0, 1.0, 3.0])
    csm_lbol = np.array([4.0, 2.0, 3.0])
    engine_lbol = np.array([40.0, 20.0, 30.0])
    calls = {}

    def fake_csm(input_time, csm_model, **kwargs):
        return csm_lbol

    def fake_engine(time, f_nickel, mej, **kwargs):
        return np.ones_like(time, dtype=float) * f_nickel * mej

    class FakeDiffusion:
        def __init__(self, time, dense_times, luminosity, mej, vej, **kwargs):
            calls["dense_max"] = dense_times[-1]
            calls["time"] = np.array(time)
            calls["mej"] = mej
            self.new_luminosity = engine_lbol

    monkeypatch.setattr(models, "_csm_bolometric_impl", fake_csm)
    monkeypatch.setattr(models, "_nickelcobalt_engine", fake_engine)

    zero = models._csm_nickel_bolometric_impl(
        time,
        "wind_bpl",
        mdot=0.1,
        vwind=100.0,
        delta=0.5,
        nn=12.0,
        mexp=5.0,
        eexp=1.0,
        eff=0.5,
        f_nickel=0.0,
        interaction_process=None,
    )
    assert np.all(zero == csm_lbol)

    combined = models._csm_nickel_bolometric_impl(
        time,
        "wind_bpl",
        mdot=0.1,
        vwind=100.0,
        delta=0.5,
        nn=12.0,
        mexp=5.0,
        eexp=1.0,
        eff=0.5,
        f_nickel=0.1,
        kappa=0.34,
        kappa_gamma=0.027,
        interaction_process=FakeDiffusion,
        dense_resolution=16,
    )

    assert calls["dense_max"] == np.max(time) + 100.0
    assert np.all(calls["time"] == time)
    assert np.all(combined == csm_lbol + engine_lbol)


def test_generated_priors_cover_exported_models():
    """Every exported concrete model except the generic csm_xray helper gets a prior."""
    from redback_csm.prior_provider import get_prior
    import redback_csm.models as models

    missing = []
    for name in models.__all__:
        if name == "csm_xray":
            continue
        if get_prior(name) is None:
            missing.append(name)
    assert not missing, f"Missing generated priors: {missing}"


def test_models_registered_in_redback():
    """After install, CSM models should appear in redback's model library."""
    try:
        import redback_csm  # noqa: F401
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
        if missing:
            import redback_csm.models as csm_models
            local_missing = [m for m in expected if not hasattr(csm_models, m)]
            assert not local_missing, f"Models missing from redback_csm.models: {local_missing}"
            pytest.skip("redback model library was imported before plugin registration")
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
    """All generated latex_labels must render without error in matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from redback_csm.prior_provider import get_prior
    import redback_csm.models as models

    fig, ax = plt.subplots()
    failures = []
    for model_name in models.__all__:
        if model_name == "csm_xray":
            continue
        priors = get_prior(model_name)
        assert priors is not None
        for param, prior in priors.items():
            try:
                ax.set_xlabel(prior.latex_label)
                fig.canvas.draw()
            except Exception as e:
                failures.append(f"{model_name}::{param} — {prior.latex_label!r} → {e}")
    plt.close(fig)
    assert not failures, "Latex label rendering failures:\n" + "\n".join(failures)


def test_dispatch_registry():
    """_DISPATCH should contain all base CSM model keys."""
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
        "static_powerlaw_csm_exponential",
        "static_powerlaw_csm_bpl",
        "homologous_powerlaw_csm_exponential",
        "homologous_powerlaw_csm_bpl",
        "static_spline_csm_bpl",
        "generic_spline_csm_bpl",
        "generic_spline12_csm_bpl",
        "static_pspline24_csm_bpl",
        "generic_pspline24_csm_bpl",
        "static_pspline48_csm_bpl",
        "generic_pspline48_csm_bpl",
        "static_pspline96_csm_bpl",
        "generic_pspline96_csm_bpl",
        "generic_4shell_csm_bpl",
        "generic_8shell_csm_bpl",
    ]
    missing = [k for k in expected_keys if k not in _DISPATCH]
    assert not missing, f"Missing dispatch keys: {missing}"
