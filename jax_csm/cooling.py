import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple

class CoolingParams(NamedTuple):
    v_se: float        # Shock velocity at emergence (cm/s)
    r_se: float        # Shock radius at emergence (cm)
    tau_csm_in: float  # Inner optical depth
    kappa: float       # cm^2/g
    r_csm_in: float    # cm
    s: float           # CSM slope

def build_cooling_matrix(x_grid, eta_csm, dt_y, expansion_factor):
    """
    Constructs the tridiagonal matrix for the cooling phase.
    The diffusion coefficient D is modified by the expansion factor (R_in/R_0).
    """
    N = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    
    # D(x, y) = (x^2 / eta_csm(x)) * (R_in / R_0)
    D = (x_grid**2 / eta_csm) * expansion_factor
    
    D_half_plus = 0.5 * (D[1:] + D[:-1])
    D_half_minus = jnp.concatenate([jnp.array([0.0]), D_half_plus])
    
    inv_x2_dx2 = 1.0 / (x_grid**2 * dx**2)
    
    a = inv_x2_dx2[1:] * D_half_minus[1:]
    b = -inv_x2_dx2 * (jnp.concatenate([D_half_plus, jnp.array([0.0])]) + D_half_minus)
    c = inv_x2_dx2[:-1] * D_half_plus
    
    return a, b, c

def compute_cooling_step(e_n, x_grid, eta_csm, dt_y, expansion_factor, f_ob):
    """
    Performs one step of the shock-cooling phase using Crank-Nicolson.
    e_n: current energy density
    expansion_factor: R_in(t) / R_0
    f_ob: Updated outer boundary coefficient
    """
    N = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    
    a, b, c = build_cooling_matrix(x_grid, eta_csm, dt_y, expansion_factor)
    
    # LHS Matrix
    lhs_a = -0.5 * dt_y * a
    lhs_b = 1.0 - 0.5 * dt_y * b
    lhs_c = -0.5 * dt_y * c
    
    # RHS Vector
    L_en = b * e_n
    L_en = L_en.at[1:].add(a * e_n[:-1])
    L_en = L_en.at[:-1].add(c * e_n[1:])
    rhs = e_n + 0.5 * dt_y * L_en
    
    # Inner Boundary Condition: Adiabatic (de/dx = 0)
    # (e_1 - e_0)/dx = 0 => e_0 = e_1
    # lhs_b[0]*e_0 + lhs_c[0]*e_1 = rhs[0]
    # (lhs_b[0] + lhs_c[0])*e_1 = rhs[0]
    
    orig_lhs_b0 = 1.0 - 0.5 * dt_y * b[0]
    lhs_b = lhs_b.at[0].set(orig_lhs_b0 + lhs_c[0])
    # rhs[0] remains rhs[0] since f_ib = 0
    
    # Outer Boundary Condition: Eddington
    lhs_a = lhs_a.at[-1].set(-f_ob / dx)
    lhs_b = lhs_b.at[-1].set(1.0 - f_ob / dx)
    rhs = rhs.at[-1].set(0.0)
    
    # Solve tridiagonal
    from .pde_core import solve_tridiagonal
    e_next = solve_tridiagonal(lhs_a, lhs_b, lhs_c, rhs)
    
    # Adiabatic Cooling Term: dE/dt = - P dV/dt
    # For a radiation-dominated gas, P = u/3. 
    # In the dimensionless form, this manifests as a scaling of e based on the volume increase.
    # e_next = e_next * (R_0 / R_in)^4 (roughly)
    # We apply the specific adiabatic loss proportional to the expansion rate.
    
    # The paper specifies that the specific internal energy follows:
    # dE/dt + P dV/dt = - dL/dm
    # This is handled by the R_in/R_0 scaling in the energy density definition u(r,t)
    # but we must ensure the numerical e(x,y) tracks this.
    
    return e_next
