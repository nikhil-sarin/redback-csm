import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple

class DynamicsParams(NamedTuple):
    m_ej: float        # Solar masses
    e_sn: float        # 10^51 erg
    m_csm: float       # Solar masses
    r_csm_in: float    # cm
    r_csm_out: float   # cm
    s: float           # CSM density slope
    delta: float       # Ejecta inner slope
    n: float           # Ejecta outer slope
    v_tr: float        # Transition velocity (cm/s)
    v_sc: float        # Scale velocity (cm/s)
    profile_type: int  # 0 for BPL, 1 for Exponential

def get_characteristic_scales(params: DynamicsParams) -> Tuple[float, float, float, float]:
    """
    Compute characteristic scales for dimensionless transformation.
    Returns: v_ej_max, t_in, rho_ej_in, q
    """
    m_sun = 1.989e33
    e_foe = 1e51
    
    m_ej_phys = params.m_ej * m_sun
    e_sn_phys = params.e_sn * e_foe
    
    # Correct v_ej_max for BPL based on E_sn and M_ej
    # For n >> 3 and delta small, v_ej_max is roughly sqrt(2*E/M)
    # More precisely: E = 0.5 * M * v_avg^2. For power law, v_avg is a fraction of v_max.
    # We'll use the approximation v_ej_max = sqrt(2*E/M) * sqrt(n/(n-3))
    v_ej_max = jnp.sqrt(2 * e_sn_phys / m_ej_phys) * jnp.sqrt(params.n / (params.n - 3.0))
    t_in = params.r_csm_in / v_ej_max
    
    # M_ej = 4*pi * rho_ej_in * t_in^3 * integral_{0}^{v_ej_max} v^2 * eta(v) dv
    # Let v_norm = v / v_ej_max. M_ej = 4*pi * rho_ej_in * (v_ej_max * t_in)^3 * integral_{0}^{1} v_norm^2 * eta(v_norm) dv_norm
    # Since v_ej_max * t_in = r_csm_in:
    # rho_ej_in = M_ej / (4 * pi * r_csm_in^3 * integral)
    
    if params.profile_type == 0: # BPL
        v_tr_norm = params.v_tr / v_ej_max
        # Integral of v^2 * eta(v) from 0 to 1
        # Part 1: 0 to v_tr_norm: v^2 * (v/v_tr)^-delta = v_tr^delta * v^(2-delta)
        # Integral = [v_tr^delta * v^(3-delta) / (3-delta)] from 0 to v_tr_norm
        # = v_tr^delta * v_tr_norm^(3-delta) / (3-delta) = v_tr_norm^3 / (3-delta)
        # Part 2: v_tr_norm to 1: v^2 * (v/v_tr)^-n = v_tr^n * v^(2-n)
        # Integral = [v_tr^n * v^(3-n) / (3-n)] from v_tr_norm to 1
        # = (v_tr^n / (3-n)) * (1 - v_tr_norm^(3-n))
        # This is slightly messy. Let's simplify: 
        # Normalized integral I = integral_{0}^{1} v^2 * eta_norm(v) dv
        # where eta_norm(v) = (v/v_tr_norm)^-delta for v < v_tr_norm and (v/v_tr_norm)^-n for v > v_tr_norm
        
        # Let's do the integral numerically or use a simpler formula.
        # I = v_tr_norm^3/(3-delta) + (v_tr_norm^3/(3-n)) * (1 - v_tr_norm^(3-n))
        # Note: n > 3, so 3-n is negative.
        i_val = (v_tr_norm**3 / (3.0 - params.delta)) + (v_tr_norm**3 / (3.0 - params.n)) * (1.0 - v_tr_norm**(3.0 - params.n))
    else: # Exponential
        v_sc_norm = params.v_sc / v_ej_max
        # Integral of v^2 * exp(-v/v_sc) from 0 to 1
        # Let u = v/v_sc, dv = v_sc du. Integral = v_sc^3 * integral(u^2 * exp(-u) du)
        # Integral(u^2 exp(-u)) = -exp(-u)(u^2 + 2u + 2)
        # I = v_sc_norm^3 * [ -exp(-u)(u^2 + 2u + 2) ] from 0 to 1/v_sc_norm
        u_max = 1.0 / v_sc_norm
        i_val = v_sc_norm**3 * (2.0 - jnp.exp(-u_max) * (u_max**2 + 2*u_max + 2))

    rho_ej_in = m_ej_phys / (4.0 * jnp.pi * params.r_csm_in**3 * i_val)
    
    # rho_csm_in = m_csm / (4 * pi * integral_{r_in}^{r_out} r^2 * r^-s dr)
    # integral r^(2-s) = [ r^(3-s) / (3-s) ] from r_in to r_out
    m_csm_phys = params.m_csm * m_sun
    int_csm = (params.r_csm_out**(3.0 - params.s) - params.r_csm_in**(3.0 - params.s)) / (3.0 - params.s)
    rho_csm_in = m_csm_phys / (4.0 * jnp.pi * int_csm)
    
    q = rho_csm_in / rho_ej_in
    
    return v_ej_max, t_in, rho_ej_in, q

def eta_ej_norm(v_norm, params: DynamicsParams):
    """Normalized ejecta profile where v_norm = v / v_ej_max."""
    if params.profile_type == 0: # BPL
        v_tr_norm = params.v_tr / (params.v_tr if params.v_tr > 0 else 1.0) # Logic handled in ODE
        # In the ODE, we pass v_tr_norm separately.
        return 0.0 # Placeholder
    else:
        return 0.0 # Placeholder

def dynamics_ode(state, zeta, params: DynamicsParams, q: float, v_tr_norm: float, v_sc_norm: float):
    x, phi, w = state
    # x/zeta is the velocity in the ejecta frame (normalized)
    # v_ej_norm = v_ej / v_ej_max = x / zeta (in normalized coordinates)
    v_ej_norm = x / zeta
    
    if params.profile_type == 0:
        # BPL profile: eta(v) where v is in units of v_ej_max
        eta_val = jnp.where(v_ej_norm < v_tr_norm, 
                           (v_ej_norm / v_tr_norm)**(-params.delta), 
                           (v_ej_norm / v_tr_norm)**(-params.n))
    else:
        # Exponential profile
        eta_val = jnp.exp(-v_ej_norm / v_sc_norm)

    eta_csm_val = x**(-params.s)
    
    dxdzeta = w
    dphidzeta = (x**2 * zeta**(-3) * (x/zeta - w) * eta_val) + (q * x**2 * w * eta_csm_val)
    dwdzeta = (1.0 / phi) * ( (x**2 * zeta**(-3) * (x/zeta - w)**2 * eta_val) - (q * x**2 * w**2 * eta_csm_val) )
    
    return jnp.array([dxdzeta, dphidzeta, dwdzeta])
