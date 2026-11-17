"""Tests for the JAX simple-mode static power-law CSM solver, its diffusion option and nickel."""
import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="jax is not installed")

DAY = 86400.0
MSUN = 1.989e33

CASES = {
    "wind_compact": dict(eta=-2.0, r_inner=1e14, r_outer=1e15, delta_sn=1.0, nn_sn=10.0, mej_sn=10.0, esn=2.0, eff=0.5, m_csm=3.0),
    "flat_compact": dict(eta=0.0, r_inner=1e14, r_outer=5e14, delta_sn=1.0, nn_sn=10.0, mej_sn=10.0, esn=2.0, eff=0.5, m_csm=3.0),
    "mid_extended": dict(eta=-1.0, r_inner=1e14, r_outer=2e15, delta_sn=1.0, nn_sn=10.0, mej_sn=20.0, esn=5.0, eff=0.5, m_csm=5.0),
}


def _params(p, **kw):
    from jax_csm import StaticPowerlawBPLParams

    return StaticPowerlawBPLParams(**{**p, **kw})


def _jax(p, t_days, n_steps=2048, **kw):
    from jax_csm import solve_simple_static_bpl

    return solve_simple_static_bpl(np.asarray(t_days), _params(p, **kw), n_steps=n_steps, t_max_days=300.0)



def bpl_coefficients(mej_sn, esn, delta_sn, nn_sn, vej_max_ratio=3.0):
    m_ej = mej_sn * MSUN
    e_sn = esn * 1e51
    a = vej_max_ratio
    i_m = 1.0 / (3.0 - delta_sn) + (1.0 - a ** (3.0 - nn_sn)) / (nn_sn - 3.0)
    i_e = 1.0 / (5.0 - delta_sn) + (1.0 - a ** (5.0 - nn_sn)) / (nn_sn - 5.0)
    v_t = np.sqrt(2.0 * e_sn * i_m / (m_ej * i_e))
    return v_t, a * v_t, m_ej / (v_t**3 * i_m)


def test_swept_mass_invariant():
    """m(t) = seed + M_csm(r_in..r) + M_ej(v > r/t): exact for the thin-shell ODE."""
    p = CASES["wind_compact"]
    t = np.geomspace(1.0, 25.0, 50)
    res = _jax(p, t)
    r = np.asarray(res.r_shock)
    m = np.asarray(res.shell_mass)
    rho_in = p["m_csm"] * MSUN / (4 * np.pi * p["r_inner"] ** 2 * (p["r_outer"] - p["r_inner"]))  # eta = -2
    m_csm = 4 * np.pi * rho_in * p["r_inner"] ** 2 * (r - p["r_inner"])
    v_t, v_max, rho0 = bpl_coefficients(p["mej_sn"], p["esn"], p["delta_sn"], p["nn_sn"])
    v = r / (t * DAY)
    n, d = p["nn_sn"], p["delta_sn"]
    assert v.min() < v_t < v.max()  # covers both ejecta regimes
    outer = lambda lo: rho0 * v_t**n * (lo ** (3 - n) - v_max ** (3 - n)) / (n - 3)
    m_ej = np.where(v > v_t, outer(v), outer(v_t) + rho0 * v_t**d * (v_t ** (3 - d) - v ** (3 - d)) / (3 - d))
    assert np.max(np.abs(m / (m_csm + m_ej) - 1)) < 1e-4


def test_zero_before_collision_and_fs_off_after_emergence():
    p = CASES["flat_compact"]
    res = _jax(p, np.array([0.1, 0.5, 5.0, 20.0, 100.0]))
    t_c = float(res.t_collide)
    assert 0.5 < t_c < 1.0
    assert float(res.lbol[0]) == 0.0 and float(res.lbol[1]) == 0.0
    te = float(res.t_emerge)
    assert 5.0 < te < 20.0
    assert float(res.lbol_fs[3]) == 0.0 and float(res.lbol_fs[4]) == 0.0
    assert float(res.lbol_rs[3]) > 0.0
    assert float(res.r_shock[3]) > p["r_outer"]


def test_emergence_drop_is_sharp():
    """The FS term must drop at t_e itself, not be smeared across a grid step."""
    p = CASES["flat_compact"]
    te = float(_jax(p, np.array([20.0])).t_emerge)
    res = _jax(p, np.array([te * (1 - 1e-6), te * (1 + 1e-6)]))
    assert float(res.lbol_fs[0]) > 0.0 and float(res.lbol_fs[1]) == 0.0


def test_shell_keeps_accelerating_after_emergence_unlike_legacy():
    """Legacy simple mode froze u after emergence (ballistic); the reference keeps
    integrating with rho_csm = 0, so the reverse shock accelerates the shell."""
    from jax_csm import solve_static_powerlaw_bpl

    p = CASES["flat_compact"]
    t = np.array([15.0, 30.0, 60.0])
    new = _jax(p, t)
    v_new = np.asarray(new.v_shock)
    assert np.all(np.diff(v_new) > 0)
    leg = solve_static_powerlaw_bpl(t, _params(p), mode="simple_legacy", n_steps=16384)
    v_leg = np.asarray(leg.velocity)
    assert np.allclose(v_leg, v_leg[0], rtol=1e-12)
    # and the legacy tail is too bright (factor ~2 for this case)
    assert float(leg.lbol[-1]) / float(new.lbol[-1]) > 1.5


def test_eff_scales_luminosity_linearly():
    p = CASES["mid_extended"]
    t = np.geomspace(1, 200, 30)
    a = np.asarray(_jax(p, t, eff=0.2).lbol)
    b = np.asarray(_jax(p, t, eff=0.8).lbol)
    assert np.allclose(b, 4 * a, rtol=1e-12, atol=0)


def test_negligible_csm_limit():
    """m_csm -> 0: no FS power, shell rides the ejecta edge and L_rs -> 0."""
    p = CASES["wind_compact"]
    res = _jax(p, np.array([2.0, 20.0]), m_csm=1e-9)
    big = _jax(p, np.array([2.0, 20.0]))
    assert np.all(np.asarray(res.lbol) < 1e-6 * np.asarray(big.lbol))


def test_mode_simple_dispatches_to_new_solver():
    from jax_csm import solve_static_powerlaw_bpl

    p = CASES["wind_compact"]
    t = np.geomspace(1, 300, 40)
    a = solve_static_powerlaw_bpl(t, _params(p), mode="simple", n_steps=2048)
    b = _jax(p, t)
    assert np.allclose(np.asarray(a.lbol), np.asarray(b.lbol), rtol=1e-12)
    assert np.allclose(np.asarray(a.radius), np.asarray(b.r_shock), rtol=1e-12)


def test_vmap_matches_loop():
    import jax
    import jax.numpy as jnp
    from jax_csm import StaticPowerlawBPLParams, solve_simple_static_bpl

    t = jnp.geomspace(1.0, 200.0, 25)
    base = CASES["wind_compact"]
    m_csm = jnp.array([1.0, 3.0, 8.0])
    full = {**base, "kappa": 0.34, "vej_max_ratio": 3.0}  # every field must be batched
    batch = StaticPowerlawBPLParams(**{**{k: jnp.full(3, v) for k, v in full.items()}, "m_csm": m_csm})
    out = jax.vmap(lambda q: solve_simple_static_bpl(t, q, n_steps=1024, t_max_days=200.0).lbol)(batch)
    for i in range(3):
        one = solve_simple_static_bpl(t, _params(base, m_csm=float(m_csm[i])), n_steps=1024, t_max_days=200.0).lbol
        assert np.allclose(np.asarray(out[i]), np.asarray(one), rtol=1e-10)


NI = dict(f_nickel=0.05, kappa=0.1, kappa_gamma=0.03)


def _redback_nickel(t, f_ni, mej, esn, mcsm, kap, kg):
    import redback.constants as rc
    import redback.interaction_processes as ip
    from redback.transient_models.supernova_models import _nickelcobalt_engine

    dense = np.linspace(0.0, t.max() + 100.0, 1000)
    vej = np.sqrt(2 * esn * 1e51 / (mej * rc.solar_mass)) / 1e5
    return ip.Diffusion(time=t, dense_times=dense, luminosity=_nickelcobalt_engine(time=dense, f_nickel=f_ni, mej=mej),
                        kappa=kap, kappa_gamma=kg, mej=mej + mcsm, vej=vej).new_luminosity


def test_nickel_component_matches_reference_quadrature():
    pytest.importorskip("redback")
    from jax_csm import nickel_diffused_luminosity

    p = CASES["wind_compact"]
    t = np.concatenate([np.linspace(0.5, 40, 30), np.geomspace(41, 300, 30)])
    ref = _redback_nickel(t, NI["f_nickel"], p["mej_sn"], p["esn"], p["m_csm"], NI["kappa"], NI["kappa_gamma"])
    jx = np.asarray(nickel_diffused_luminosity(t, NI["f_nickel"], p["mej_sn"], p["esn"], p["m_csm"], NI["kappa"], NI["kappa_gamma"]))
    assert np.max(np.abs(jx / ref - 1)) < 1e-12
    conv = np.asarray(nickel_diffused_luminosity(t, NI["f_nickel"], p["mej_sn"], p["esn"], p["m_csm"], NI["kappa"],
                                                 NI["kappa_gamma"], quadrature="converged"))
    # redback's 100-point quadrature is itself only good to ~1e-2
    assert np.max(np.abs(ref / conv - 1)) < 1e-2


def test_nickel_uses_ejecta_mass_for_nickel_and_ejecta_plus_csm_for_diffusion():
    from jax_csm import nickel_diffused_luminosity

    t = np.array([5.0, 30.0, 150.0])
    a = nickel_diffused_luminosity(t, 0.1, 10.0, 2.0, 5.0, 0.1, 0.05)
    b = nickel_diffused_luminosity(t, 0.1, 10.0, 2.0, 0.0, 0.1, 0.05, mej_arnett=15.0)
    assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-14)
    # M_Ni = f_Ni * M_ej does not depend on m_csm: in the instant-diffusion (tiny kappa),
    # full-trapping limit L equals the heating rate of f_Ni * mej_sn for any m_csm.
    from jax_csm import nickel_heating

    t2 = np.geomspace(20.0, 300.0, 8)
    heat = np.asarray(nickel_heating(t2, 1.0))
    for m_csm in (0.0, 5.0, 50.0):
        lim = np.asarray(nickel_diffused_luminosity(t2, 0.1, 10.0, 2.0, m_csm, 1e-6, 1e8, quadrature="converged"))
        assert np.max(np.abs(lim / heat - 1)) < 1e-3


def test_nickel_limits():
    from jax_csm import nickel_diffused_luminosity, nickel_heating

    t = np.geomspace(50.0, 400.0, 10)
    heat = np.asarray(nickel_heating(t, 0.5))
    # kappa_gamma -> large, kappa -> small: full trapping and instant diffusion => L = heating
    inst = np.asarray(nickel_diffused_luminosity(t, 0.1, 5.0, 2.0, 0.0, 1e-6, 1e8, quadrature="converged"))
    assert np.max(np.abs(inst / heat - 1)) < 1e-3
    # finite diffusion time: full-trapping L exceeds the (decaying) heating rate by a lag
    # term ~ tau_m^2 / (2 t tau_Co) that shrinks with time
    full = np.asarray(nickel_diffused_luminosity(t, 0.1, 5.0, 2.0, 0.0, 0.05, 1e8, quadrature="converged"))
    dev = full / heat - 1
    assert np.all(dev > 0) and np.all(np.diff(dev) < 0) and dev[-1] < 1e-2
    # kappa_gamma -> 0: no deposition
    none = np.asarray(nickel_diffused_luminosity(t, 0.1, 5.0, 2.0, 0.0, 0.05, 1e-12))
    assert np.all(none < 1e-6 * full)
    # partial trapping is strictly below full trapping
    part = np.asarray(nickel_diffused_luminosity(t, 0.1, 5.0, 2.0, 0.0, 0.05, 0.03, quadrature="converged"))
    assert np.all(part < full)


def test_csm_nickel_zero_nickel_recovers_csm_only_and_sums():
    from jax_csm import nickel_diffused_luminosity, simple_csm_nickel_bolometric

    p = CASES["wind_compact"]
    t = np.geomspace(1, 300, 40)
    prm = _params(p, kappa=NI["kappa"])
    zero = simple_csm_nickel_bolometric(t, prm, 0.0, NI["kappa_gamma"], t_max_days=300.0, csm_diffusion=False)
    csm = _jax(p, t)
    assert np.array_equal(np.asarray(zero.lbol), np.asarray(csm.lbol))
    assert np.all(np.asarray(zero.lbol_nickel) == 0.0)
    both = simple_csm_nickel_bolometric(t, prm, NI["f_nickel"], NI["kappa_gamma"], t_max_days=300.0, csm_diffusion=False)
    ni = nickel_diffused_luminosity(t, NI["f_nickel"], p["mej_sn"], p["esn"], p["m_csm"], NI["kappa"], NI["kappa_gamma"])
    assert np.allclose(np.asarray(both.lbol), np.asarray(csm.lbol) + np.asarray(ni), rtol=1e-14)
    # nickel does not alter the shock solution
    assert np.array_equal(np.asarray(both.r_shock), np.asarray(csm.r_shock))


def _jaxd(p, t_days, n_steps=2048, kappa=0.34, t_max=300.0):
    from jax_csm import solve_simple_static_bpl

    return solve_simple_static_bpl(np.asarray(t_days), _params(p, kappa=kappa), n_steps=n_steps,
                                   t_max_days=t_max, diffusion=True)


def test_diffusion_thin_limit_recovers_shock_luminosity():
    p = CASES["wind_compact"]
    te = float(_jax(p, np.array([300.0])).t_emerge)
    t = np.concatenate([np.geomspace(1.0, 0.98 * te, 30), [1.05 * te, 2 * te]])
    d = _jaxd(p, t, kappa=1e-7)
    s = _jax(p, t)
    assert np.max(np.abs(np.asarray(d.lbol[:30]) / np.asarray(s.lbol[:30]) - 1)) < 1e-3
    # post-emergence shock power is not added (Fortran: tau_ahead = 0 -> skipped)
    assert np.all(np.asarray(d.lbol[30:]) < 1e-6 * float(d.lbol[29]))


def test_diffused_csm_nickel_matches_public_wrapper_outside_cusp():
    """Full reference: redback_csm static_powerlaw_csm_bpl_nickel_bolometric (kappa passed)."""
    pytest.importorskip("redback_csm.core")
    import redback_csm.models as RM
    from jax_csm import simple_csm_nickel_bolometric

    p = CASES["mid_extended"]
    te = float(_jax(p, np.array([300.0])).t_emerge)
    t = np.geomspace(2.0, 300.0, 120)
    t = t[np.abs(t / te - 1) > 0.10]
    try:
        ref = RM.static_powerlaw_csm_bpl_nickel_bolometric(t, f_nickel=0.1, kappa=0.1, kappa_gamma=0.03, mode="simple", **p)
    except ImportError:
        pytest.skip("Fortran extension not built")
    jx = np.asarray(simple_csm_nickel_bolometric(t, _params(p, kappa=0.1), 0.1, 0.03).lbol)
    # bounded by the published Fortran discretisation (2% output rows, 1% Euler steps)
    assert np.max(np.abs(jx - ref)) / ref.max() < 3e-2
