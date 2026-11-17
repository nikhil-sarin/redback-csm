"""Tests for the optional CSM photosphere (photosphere_mode="csm")."""
import numpy as np
import pytest

DAY = 86400.0
STATIC = dict(eta=-2.0, r_inner=1e14, r_outer=1e15, delta_sn=1.0, nn_sn=10.0, mej_sn=10.0, esn=2.0,
              eff=0.5, m_csm=3.0)


def _lc_and_rph(**extra):
    core = pytest.importorskip("redback_csm.core")
    try:
        return core._call_csm_with_csm_photosphere("static_powerlaw_csm_bpl", **{**STATIC, **extra})
    except ImportError:
        pytest.skip("Fortran extension not built")


def test_csm_photosphere_rows_simple_mode_with_kappa():
    lc, rph = _lc_and_rph(mode="simple", kappa=0.34)
    assert rph.shape == lc.time.shape
    assert np.all(rph >= lc.rshock * (1 - 1e-12))
    t = lc.time / DAY
    # buried shock: photosphere near the outer CSM edge, far outside the shell
    early = t < 2.0
    assert np.all(rph[early] > 3.0 * lc.rshock[early])
    assert np.all(rph[early] <= STATIC["r_outer"])
    # after the shell leaves the CSM the photosphere is the shell
    late = lc.rshock > 1.01 * STATIC["r_outer"]
    assert late.any()
    np.testing.assert_array_equal(rph[late], lc.rshock[late])


def test_csm_photosphere_rows_without_kappa_and_in_transport():
    lc, rph = _lc_and_rph(mode="simple")
    np.testing.assert_array_equal(rph, lc.rph)
    lc, rph = _lc_and_rph(mode="transport", kappa=0.2, n_rad_zones=40)
    np.testing.assert_array_equal(rph, lc.rph)


def test_photosphere_mode_validation():
    models = pytest.importorskip("redback_csm.models")
    with pytest.raises(ValueError):
        models._pop_photosphere_mode({"photosphere_mode": "nope"})
    with pytest.raises(ValueError):
        models._pop_photosphere_mode({"photosphere_mode": "csm", "mode": "simple"})
    kw = {"photosphere_mode": "csm", "mode": "simple", "kappa": 0.34}
    assert models._pop_photosphere_mode(kw) == "csm" and "photosphere_mode" not in kw
    assert models._pop_photosphere_mode({}) == "legacy"


def test_csm_photosphere_temperature_preserves_luminosity():
    models = pytest.importorskip("redback_csm.models")
    lc, rph = _lc_and_rph(mode="simple", kappa=0.34)
    t = np.geomspace(1.0, 300.0, 50)
    lbol = np.interp(t, lc.time / DAY, lc.lbol)
    for floor in (None, 3000.0):
        r, temp = models._csm_photosphere_rph_temp(t, lbol, lc, rph, floor)
        ok = lbol > 0
        np.testing.assert_allclose(4 * np.pi * models._SIGMA_SB * r[ok] ** 2 * temp[ok] ** 4, lbol[ok], rtol=1e-10)
        if floor:
            assert np.all(temp[ok] >= floor * (1 - 1e-12))


@pytest.mark.parametrize("nickel", [False, True])
def test_photosphere_mode_csm_multiband_runs_and_differs(nickel):
    models = pytest.importorskip("redback_csm.models")
    t = np.geomspace(2.0, 150.0, 12)
    kw = dict(STATIC, mode="simple", kappa=0.34, redshift=0.05, frequency=np.full(t.size, 5e14),
              output_format="flux_density")
    func = models.static_powerlaw_csm_bpl
    if nickel:
        func = models.static_powerlaw_csm_bpl_nickel
        kw.update(f_nickel=0.05, kappa_gamma=0.1)
    try:
        legacy = func(t, **kw)
    except ImportError:
        pytest.skip("Fortran extension not built")
    csm = func(t, photosphere_mode="csm", **kw)
    assert np.all(np.isfinite(csm)) and np.all(csm >= 0)
    assert not np.allclose(csm, legacy)
