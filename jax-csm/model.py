import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple
from .dynamics import DynamicsParams
from .transport import TransportParams, solve_full_model

def get_csm_lightcurve(params_dict):
    """
    High-level wrapper for the TransFit-CSM model.
    Input: dictionary of parameters
    Output: (time_days, l_bol_erg_s)
    """
    # Convert dict to NamedTuples
    dyn_params = DynamicsParams(**params_dict['dyn'])
    trans_params = TransportParams(
        dyn_params=dyn_params,
        kappa=params_dict['kappa'],
        eff_int=params_dict['eff_int'],
        n_grid=params_dict.get('n_grid', 100),
        dt_zeta=params_dict.get('dt_zeta', 0.01)
    )
    
    zeta_grid, l_bol = solve_full_model(trans_params)
    
    # Convert zeta (t/t_in) to days
    # Need t_in from scales
    from .dynamics import get_characteristic_scales
    v_ej_max, t_in, _, _ = get_characteristic_scales(dyn_params)
    
    time_days = (zeta_grid * t_in) / (24 * 3600)
    
    return time_days, l_bol

def vmapped_csm_model(params_batch):
    """
    Vectorized version of the model for MCMC/Bayesian inference.
    params_batch: array of parameters
    """
    # This would involve creating a batch of TransportParams
    # and using jax.vmap(solve_full_model)
    return jax.vmap(lambda p: solve_full_model(p))(params_batch)
