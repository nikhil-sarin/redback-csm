module integration

 use get_vals
 use constants, only: pi, clight

 implicit none

 real(8), parameter :: gamma_shock = 5d0 / 3d0
 real(8), parameter :: ff_emissivity_norm = 4.7d16

contains

function dmdt(u,r,t,op)
  ! Continuity equation
  type(outflow_parameters),intent(inout):: op(1:2)
  real(8),intent(in):: u,r,t
  real(8):: dmdt, rel_in, rel_out
  rel_in = max(v_in(r,t,op(1)) - u, 0d0)
  rel_out = max(u - v_out(r,t,op(2)), 0d0)
  dmdt = rho4pir2_in (r,t,op(1))*rel_in &
       + rho4pir2_out(r,t,op(2))*rel_out
 end function dmdt

function dudt(u,r,m,t,op) result(du_dt)
  ! Equation of motion
  type(outflow_parameters),intent(inout):: op(1:2)
  real(8),intent(in):: u,r,m,t
  real(8):: du_dt, aram, rel_in, rel_out

  rel_in = max(v_in(r,t,op(1)) - u, 0d0)
  rel_out = max(u - v_out(r,t,op(2)), 0d0)
  aram = ( rho4pir2_in (r,t,op(1))*rel_in**2 &
         - rho4pir2_out(r,t,op(2))*rel_out**2 ) / m
  du_dt = aram
 end function dudt

function forward_shock_luminosity(r,t,u,op) result(lum)
  ! Forward-shock contribution to the dissipated luminosity
  type(outflow_parameters),intent(inout):: op(1:2)
  real(8),intent(in):: r,t,u
  real(8):: lum, rel_out
  rel_out = max(u - v_out(r,t,op(2)), 0d0)
  lum = 0.5d0*rho4pir2_out(r,t,op(2))*rel_out**3
 end function forward_shock_luminosity

function reverse_shock_luminosity(r,t,u,op) result(lum)
  ! Reverse-shock contribution to the dissipated luminosity
  type(outflow_parameters),intent(inout):: op(1:2)
  real(8),intent(in):: r,t,u
  real(8):: lum, rel_in
  rel_in = max(v_in(r,t,op(1)) - u, 0d0)
  lum = 0.5d0*rho4pir2_in(r,t,op(1))*rel_in**3
 end function reverse_shock_luminosity

 function shock_luminosity(r,t,u,op) result(lum)
  ! Compute shock luminosity
  type(outflow_parameters),intent(inout):: op(1:2)
  real(8),intent(in):: r,t,u
  real(8):: lum
  lum = reverse_shock_luminosity(r,t,u,op) + forward_shock_luminosity(r,t,u,op)
 end function shock_luminosity

function forward_shock_radiative_efficiency(r,t,u,op,eff_max) result(eta_fs)
  ! Forward-shock radiative efficiency limited by local free-free emission.
  ! This follows the idea in Tsuna et al. (2019): once free-free emission
  ! cannot sustain thermal equilibrium, the forward shock becomes less radiative.
  type(outflow_parameters),intent(inout):: op(1:2)
  real(8),intent(in):: r,t,u,eff_max
  real(8):: eta_fs
  real(8):: rho_up, rho_down, v_rel, p_down, epsilon_ff, thermal_factor

  v_rel = u - v_out(r,t,op(2))
  if (v_rel <= 0d0 .or. r <= 0d0) then
   eta_fs = 0d0
   return
  end if

  rho_up = rho4pir2_out(r,t,op(2)) / (4d0 * pi * r**2)
  if (rho_up <= 0d0) then
   eta_fs = 0d0
   return
  end if

  rho_down = (gamma_shock + 1d0) / (gamma_shock - 1d0) * rho_up
  p_down = 2d0 / (gamma_shock + 1d0) * rho_up * v_rel**2
  if (p_down <= 0d0) then
   eta_fs = 0d0
   return
  end if

  epsilon_ff = ff_emissivity_norm * sqrt(p_down) * rho_down**1.5d0
  thermal_factor = epsilon_ff * r / (3d0 * p_down * clight)
  eta_fs = eff_max * min(1d0, max(0d0, thermal_factor))
 end function forward_shock_radiative_efficiency

 function reverse_shock_radiative_efficiency(r,t,u,op,eff_max) result(eta_rs)
  ! Reverse-shock analogue of the free-free-limited radiative efficiency.
  type(outflow_parameters),intent(inout):: op(1:2)
  real(8),intent(in):: r,t,u,eff_max
  real(8):: eta_rs
  real(8):: rho_up, rho_down, v_rel, p_down, epsilon_ff, thermal_factor

  v_rel = v_in(r,t,op(1)) - u
  if (v_rel <= 0d0 .or. r <= 0d0) then
   eta_rs = 0d0
   return
  end if

  rho_up = rho4pir2_in(r,t,op(1)) / (4d0 * pi * r**2)
  if (rho_up <= 0d0) then
   eta_rs = 0d0
   return
  end if

  rho_down = (gamma_shock + 1d0) / (gamma_shock - 1d0) * rho_up
  p_down = 2d0 / (gamma_shock + 1d0) * rho_up * v_rel**2
  if (p_down <= 0d0) then
   eta_rs = 0d0
   return
  end if

  epsilon_ff = ff_emissivity_norm * sqrt(p_down) * rho_down**1.5d0
  thermal_factor = epsilon_ff * r / (3d0 * p_down * clight)
  eta_rs = eff_max * min(1d0, max(0d0, thermal_factor))
 end function reverse_shock_radiative_efficiency

end module integration
