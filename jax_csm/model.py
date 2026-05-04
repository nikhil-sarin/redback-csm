import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from .static_powerlaw import (
    StaticPowerlawBPLParams,
    solve_static_powerlaw_bpl,
    static_powerlaw_csm_bpl_bolometric,
)


def get_csm_lightcurve(params_dict):
    """High-level compatibility wrapper for the JAX static CSM model.

    The preferred path is ``static_powerlaw_csm_bpl_bolometric`` with the same
    parameter names as ``redback_csm.models.generic_powerlaw_csm_bpl_bolometric``.
    This wrapper also accepts the older ``{"dyn": ...}`` dictionary used by the
    original branch and maps it onto the current static finite-shell convention.
    """
    mode = str(params_dict.get("mode", "transport")).lower()
    n_steps = int(params_dict.get("n_steps", params_dict.get("n_grid", 16384)))
    n_zones = int(params_dict.get("n_zones", params_dict.get("n_rad_zones", 40)))

    if "dyn" in params_dict:
        dyn = dict(params_dict["dyn"])
        params = StaticPowerlawBPLParams(
            eta=-float(dyn["s"]),
            r_inner=float(dyn["r_csm_in"]),
            r_outer=float(dyn["r_csm_out"]),
            delta_sn=float(dyn["delta"]),
            nn_sn=float(dyn["n"]),
            mej_sn=float(dyn["m_ej"]),
            esn=float(dyn["e_sn"]),
            eff=float(params_dict.get("eff_int", 1.0)),
            m_csm=float(dyn["m_csm"]),
            kappa=float(params_dict.get("kappa", 0.34)),
            vej_max_ratio=float(params_dict.get("vej_max_ratio", params_dict.get("A_ratio", 3.0))),
        )
        time_days = jnp.asarray(params_dict.get("time", jnp.geomspace(1.0, 300.0, 500)), dtype=jnp.float64)
    else:
        time_days = jnp.asarray(params_dict.pop("time"), dtype=jnp.float64)
        params = StaticPowerlawBPLParams(
            eta=params_dict.pop("eta"),
            r_inner=params_dict.pop("r_inner"),
            r_outer=params_dict.pop("r_outer"),
            delta_sn=params_dict.pop("delta_sn"),
            nn_sn=params_dict.pop("nn_sn"),
            mej_sn=params_dict.pop("mej_sn"),
            esn=params_dict.pop("esn"),
            eff=params_dict.pop("eff"),
            m_csm=params_dict.pop("m_csm"),
            kappa=params_dict.pop("kappa", 0.34),
            vej_max_ratio=params_dict.pop("vej_max_ratio", params_dict.pop("A_ratio", 3.0)),
        )

    result = solve_static_powerlaw_bpl(time_days, params, mode=mode, n_steps=n_steps, n_zones=n_zones)
    return time_days, result.lbol


def get_static_powerlaw_csm_bpl_lightcurve(time, return_diagnostics=False, **kwargs):
    """Evaluate the JAX static finite-power-law CSM + BPL-ejecta model."""
    mode = str(kwargs.pop("mode", "transport")).lower()
    n_steps = int(kwargs.pop("n_steps", kwargs.pop("n_grid", 16384)))
    n_zones = int(kwargs.pop("n_zones", kwargs.pop("n_rad_zones", 40)))
    params = StaticPowerlawBPLParams(
        eta=kwargs.pop("eta"),
        r_inner=kwargs.pop("r_inner"),
        r_outer=kwargs.pop("r_outer"),
        delta_sn=kwargs.pop("delta_sn"),
        nn_sn=kwargs.pop("nn_sn"),
        mej_sn=kwargs.pop("mej_sn"),
        esn=kwargs.pop("esn"),
        eff=kwargs.pop("eff"),
        m_csm=kwargs.pop("m_csm"),
        kappa=kwargs.pop("kappa", 0.34),
        vej_max_ratio=kwargs.pop("vej_max_ratio", kwargs.pop("A_ratio", 3.0)),
    )
    result = solve_static_powerlaw_bpl(
        jnp.asarray(time, dtype=jnp.float64), params, mode=mode, n_steps=n_steps, n_zones=n_zones
    )
    if return_diagnostics:
        return result
    return result.lbol


def vmapped_static_powerlaw_csm_bpl(time, params_batch, mode="transport", n_steps=16384, n_zones=40):
    """Vectorized static CSM evaluation for batches of StaticPowerlawBPLParams."""
    time = jnp.asarray(time, dtype=jnp.float64)
    return jax.vmap(lambda p: solve_static_powerlaw_bpl(time, p, mode=mode, n_steps=n_steps, n_zones=n_zones).lbol)(
        params_batch
    )


__all__ = [
    "StaticPowerlawBPLParams",
    "solve_static_powerlaw_bpl",
    "static_powerlaw_csm_bpl_bolometric",
    "get_csm_lightcurve",
    "get_static_powerlaw_csm_bpl_lightcurve",
    "vmapped_static_powerlaw_csm_bpl",
]
