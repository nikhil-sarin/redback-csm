import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple
from .dynamics import DynamicsParams, get_characteristic_scales, dynamics_ode
from .pde_core import crank_nicolson_step

class TransportParams(NamedTuple):
    dyn_params: DynamicsParams
    kappa: float       # cm^2/g
    eff_int: float     # efficiency
    n_grid: int        # number of spatial zones
    dt_zeta: float     # time step in dimensionless zeta

def solve_full_model(params: TransportParams):
    """
    Coordinate the original experimental CSM transport model:
    Dynamics -> Interaction Diffusion -> Shock Cooling
    """
    # 1. Characteristic Scales
    v_ej_max, t_in, rho_ej_in, q = get_characteristic_scales(params.dyn_params)
    v_tr_norm = params.dyn_params.v_tr / v_ej_max
    v_sc_norm = params.dyn_params.v_sc / v_ej_max
    
    # Physical Constants
    c_light = 2.99792458e10
    
    # 2. Dynamics Integration (Interaction Phase)
    # We use a fixed-step RK4 integration loop in JAX
    def rk4_step(state, zeta):
        k1 = dynamics_ode(state, zeta, params.dyn_params, q, v_tr_norm, v_sc_norm)
        k2 = dynamics_ode(state + 0.5 * params.dt_zeta * k1, zeta + 0.5 * params.dt_zeta, params.dyn_params, q, v_tr_norm, v_sc_norm)
        k3 = dynamics_ode(state + 0.5 * params.dt_zeta * k2, zeta + 0.5 * params.dt_zeta, params.dyn_params, q, v_tr_norm, v_sc_norm)
        k4 = dynamics_ode(state + params.dt_zeta * k3, zeta + params.dt_zeta, params.dyn_params, q, v_tr_norm, v_sc_norm)
        return state + (params.dt_zeta / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Initial state [x, phi, w]
    state_0 = jnp.array([1.0, 1e-6, 1.0])
    x_se = params.dyn_params.r_csm_out / params.dyn_params.r_csm_in
    
    # Integrate until x = x_se with history tracking
    # Use scan to get full trajectory
    max_steps = 2000
    
    def scan_body(state, _):
        x_prev, phi_prev, w_prev, zeta = state
        next_state = rk4_step(jnp.array([x_prev, phi_prev, w_prev]), zeta)
        new_zeta = zeta + params.dt_zeta
        # Condition: continue if x < x_se
        should_continue = (next_state[0] < x_se)
        return (next_state[0], next_state[1], next_state[2], new_zeta), (next_state, should_continue)
    
    init_scan = (1.0, 1e-6, 1.0, 1.0)
    (final_x, final_phi, final_w, final_zeta), (traj_info, should_continue_array) = jax.lax.scan(
        scan_body, init_scan, jnp.arange(max_steps)
    )
    
    # Find where integration should stop
    num_steps = jnp.argmax(~should_continue_array)
    num_steps = jnp.where(num_steps == 0, max_steps, num_steps)
    
    # Get the trajectory up to num_steps
    state_history = traj_info[0][0]  # This won't work with scan directly
    # Better approach: redo the integration with explicit stops
    
    def scan_dyn(state, _):
        x, phi, w = state
        next_state = rk4_step(jnp.array([x, phi, w]), 1.0 + _ * params.dt_zeta)
        return next_state, next_state

    # Use a fixed number of steps (reduced for faster computation)
    num_steps = 100
    _, state_history = jax.lax.scan(scan_dyn, state_0, jnp.arange(num_steps))
    
    # state_history has shape (num_steps, 3) with outputs from steps 0..num_steps-1
    # Each step i produces state at zeta = 1.0 + i * dt_zeta
    x_traj = state_history[:, 0]
    w_traj = state_history[:, 2]
    zeta_grid = 1.0 + jnp.arange(num_steps) * params.dt_zeta

    # 3. Diffusion Integration
    # rho_csm_in calculation
    m_sun = 1.989e33
    m_csm_phys = params.dyn_params.m_csm * m_sun
    int_csm = (params.dyn_params.r_csm_out**(3.0 - params.dyn_params.s) - params.dyn_params.r_csm_in**(3.0 - params.dyn_params.s)) / (3.0 - params.dyn_params.s)
    rho_csm_in = m_csm_phys / (4.0 * jnp.pi * int_csm)
    
    tau_csm_in = 3.0 * params.kappa * rho_csm_in * params.dyn_params.r_csm_in
    t_diff = (tau_csm_in * params.dyn_params.r_csm_in) / c_light
    
    #Grid: x from x_sh to x_ph. 
    # This is tricky because x_sh moves. 
    # Paper says: "Luminosity is set by emergent flux at CSM photosphere".
    # We'll use a fixed grid from x=1 to x=x_se and handle the inner boundary.
    x_grid = jnp.linspace(1.0, x_se, params.n_grid)
    eta_csm = x_grid**(-params.dyn_params.s)
    
    e_0 = jnp.zeros(params.n_grid)
    
    # Pre-calculate f_ob (outer boundary coefficient - constant for this run)
    eta_ph = x_se**(-params.dyn_params.s)
    f_ob = -4.0 / (tau_csm_in * eta_ph)
    
    def diffusion_step(e, i):
        zeta = zeta_grid[i]
        x_sh = x_traj[i]
        w = w_traj[i]
        
        # Inner Boundary Condition f_ib = epsilon * eta_csm^2 * w^3
        # We evaluate eta_csm at the current shock position
        eta_sh = x_sh**(-params.dyn_params.s)
        f_ib = params.eff_int * (eta_sh**2) * (w**3)
        
        dt_y = (params.dt_zeta * t_in) / t_diff
        
        e_next = crank_nicolson_step(e, x_grid, eta_csm, params.kappa, params.dyn_params.r_csm_in, dt_y, f_ib, f_ob)
        return e_next, e_next

    final_e, e_history = jax.lax.scan(diffusion_step, e_0, jnp.arange(num_steps))
    
    # 4. Bolometric Luminosity
    # L_bol = 4 * pi * sigma * R_ph^2 * T^4
    # e(x_ph) is dimensionless energy density.
    # u_0 = (tau_csm_in * v_ej_max / c) * (0.5 * rho_csm_in * v_ej_max^2)
    u_0 = (tau_csm_in * v_ej_max / c_light) * (0.5 * rho_csm_in * v_ej_max**2)
    
    # L_bol = 4 * pi * R_ph^2 * (c / 3*kappa*rho_csm) * (de/dx)
    # Or use the surface e value and the Eddington BC.
    # L_bol = 4 * pi * R_ph^2 * flux = 4 * pi * R_ph^2 * (c / 3*kappa*rho_csm(x_ph)) * (e[N] * (1/f_ob))
    
    r_ph = params.dyn_params.r_csm_out
    rho_ph = rho_csm_in * (x_se**(-params.dyn_params.s))
    
    # flux = (c / (3 * kappa * rho_ph)) * (e_history[:, -1] * u_0 / f_ob) 
    # Actually, use the boundary condition e_N = f_ob * de/dx
    # flux = (c / (3 * kappa * rho_ph)) * (e_history[:, -1] * u_0 / f_ob)
    
    flux = (c_light / (3.0 * params.kappa * rho_ph)) * (e_history[:, -1] * u_0 / f_ob)
    l_bol = 4.0 * jnp.pi * r_ph**2 * flux
    
    return zeta_grid, l_bol
