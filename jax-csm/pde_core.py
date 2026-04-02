import jax
import jax.numpy as jnp
from typing import Tuple

def solve_tridiagonal(a, b, c, d):
    """
    Solve Ax = d where A is a tridiagonal matrix.
    a: lower diagonal (size N-1)
    b: main diagonal (size N)
    c: upper diagonal (size N-1)
    d: right hand side (size N)
    """
    return jax.lax.linalg.tridiagonal_solve(a, b, c, d)

def build_diffusion_matrix(x_grid, eta_csm, kappa, r_csm_in, dt_y):
    """
    Constructs the tridiagonal matrix for the Crank-Nicolson operator.
    D(x) = x^2 / eta_csm(x)
    Operator: L[e] = (1/x^2) * d/dx [ D(x) * de/dx ]
    """
    N = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    
    # D(x) at grid points
    D = x_grid**2 / eta_csm
    
    # We use a central difference for d/dx [ D(x) de/dx ]
    # (D_{i+1/2} * (e_{i+1}-e_i)/dx - D_{i-1/2} * (e_i-e_{i-1})/dx) / dx
    # divided by x_i^2
    
    D_plus = 0.5 * (D[1:] + D[:-1])
    D_minus = 0.5 * (D[:-1] + D[0:N-1]) # This is just D_plus shifted
    
    # Note: D_minus[i] should be D_{i-1/2}
    # Correct indexing for D_{i+1/2} and D_{i-1/2}
    D_half_plus = 0.5 * (D[1:] + D[:-1])
    D_half_minus = jnp.concatenate([jnp.array([0.0]), D_half_plus])
    
    # Main diagonal b, lower a, upper c
    # L[e]_i = (1/(x_i^2 * dx^2)) * [ D_{i+1/2}(e_{i+1}-e_i) - D_{i-1/2}(e_i-e_{i-1}) ]
    
    inv_x2_dx2 = 1.0 / (x_grid**2 * dx**2)
    
    # Coefficients for e_{i-1}, e_i, e_{i+1}
    a = inv_x2_dx2[1:] * D_half_minus[1:]
    b = -inv_x2_dx2 * (D_half_plus + D_half_minus) # D_half_plus needs padding
    # Fix b padding
    b = -inv_x2_dx2 * (jnp.concatenate([D_half_plus, jnp.array([0.0])]) + D_half_minus)
    c = inv_x2_dx2[:-1] * D_half_plus
    
    return a, b, c

def crank_nicolson_step(e_n, x_grid, eta_csm, kappa, r_csm_in, dt_y, f_ib, f_ob):
    """
    Performs one step of the Crank-Nicolson diffusion solver.
    e_n: current energy density
    f_ib: inner boundary gradient (Neumann)
    f_ob: outer boundary coefficient (Eddington)
    """
    N = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    
    a, b, c = build_diffusion_matrix(x_grid, eta_csm, kappa, r_csm_in, dt_y)
    
    # Matrix operator A (the L operator)
    # (I - dt/2 * A) e_{n+1} = (I + dt/2 * A) e_n
    
    # LHS Matrix
    lhs_a = -0.5 * dt_y * a
    lhs_b = 1.0 - 0.5 * dt_y * b
    lhs_c = -0.5 * dt_y * c
    
    # RHS Vector
    # RHS = e_n + 0.5 * dt_y * L[e_n]
    # Calculate L[e_n] using tridiagonal multiply
    # L[e]_i = a_i*e_{i-1} + b_i*e_i + c_i*e_{i+1}
    
    L_en = b * e_n
    L_en = L_en + a * e_n[1:] # Shifted
    # Need to be careful with indices for L_en
    # L_en[i] = a[i-1]*e[i-1] + b[i]*e[i] + c[i]*e[i+1]
    
    # Correct L_en computation
    L_en = b * e_n
    L_en = L_en.at[1:].add(a * e_n[:-1])
    L_en = L_en.at[:-1].add(c * e_n[1:])
    
    rhs = e_n + 0.5 * dt_y * L_en
    
    # Apply Inner Boundary Condition: de/dx = f_ib
    # (e_1 - e_0)/dx = f_ib  => e_0 = e_1 - dx * f_ib
    # This modifies the first equation of the system
    # lhs_b[0]*e_0 + lhs_c[0]*e_1 = rhs[0]
    # lhs_b[0]*(e_1 - dx*f_ib) + lhs_c[0]*e_1 = rhs[0]
    # (lhs_b[0] + lhs_c[0])*e_1 = rhs[0] + lhs_b[0]*dx*f_ib
    
    lhs_b = lhs_b.at[0].set(lhs_b[0] + lhs_c[0])
    rhs = rhs.at[0].set(rhs[0] + lhs_b[0] * dx * f_ib) # Wait, used modified lhs_b
    # Correct:
    # rhs[0] = rhs[0] + (-0.5 * dt_y * b[0]) * dx * f_ib
    # No, the original lhs_b[0] was 1 - 0.5*dt_y*b[0].
    # Let's be precise:
    # The eqn is: (1 - 0.5*dt_y*b[0])*e_0 + (-0.5*dt_y*c[0])*e_1 = rhs[0]
    # Substitute e_0 = e_1 - dx*f_ib:
    # (1 - 0.5*dt_y*b[0])*(e_1 - dx*f_ib) + (-0.5*dt_y*c[0])*e_1 = rhs[0]
    # (1 - 0.5*dt_y*b[0] - 0.5*dt_y*c[0])*e_1 = rhs[0] + (1 - 0.5*dt_y*b[0])*dx*f_ib
    
    # Let's recalculate the first term:
    orig_lhs_b0 = 1.0 - 0.5 * dt_y * b[0]
    lhs_b = lhs_b.at[0].set(orig_lhs_b0 + lhs_c[0])
    rhs = rhs.at[0].set(rhs[0] + orig_lhs_b0 * dx * f_ib)
    
    # Apply Outer Boundary Condition: e_N = f_ob * (e_N - e_{N-1})/dx
    # e_N (1 - f_ob/dx) = - f_ob/dx * e_{N-1}
    # This replaces the last equation.
    # lhs_a[N-2]*e_{N-1} + lhs_b[N-1]*e_N = rhs[N-1]
    # Actually, let's just set the last row of the matrix to the BC.
    
    # New last row:
    # -f_ob/dx * e_{N-1} + (1 - f_ob/dx) * e_N = 0
    # But the RHS for the last row is 0 in this case? 
    # No, we should probably maintain the time evolution but enforce the BC.
    # Better: Use the BC to eliminate e_N.
    # e_N = e_{N-1} / (1 - dx/f_ob) ... no.
    # From e_N = f_ob * (e_N - e_{N-1})/dx:
    # e_N (1 - f_ob/dx) = - f_ob/dx * e_{N-1}
    # e_N = (-f_ob/dx) / (1 - f_ob/dx) * e_{N-1} = e_{N-1} / (1 - dx/f_ob)
    
    # Let's just set the last row to enforce the BC:
    # lhs_a[N-2]*e_{N-1} + lhs_b[N-1]*e_N = rhs[N-1]
    # We replace it with: -f_ob/dx * e_{N-1} + (1 - f_ob/dx) * e_N = 0
    
    lhs_a = lhs_a.at[-1].set(-f_ob / dx)
    lhs_b = lhs_b.at[-1].set(1.0 - f_ob / dx)
    rhs = rhs.at[-1].set(0.0)
    
    e_next = solve_tridiagonal(lhs_a, lhs_b, lhs_c, rhs)
    return e_next
