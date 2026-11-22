module integration

 use get_vals
 use constants, only: pi, clight

 implicit none

 real(8), parameter :: gamma_shock = 5d0 / 3d0
 real(8), parameter :: ff_emissivity_norm = 4.7d16

contains

 subroutine initialize_simple_shell(t,r,u,m,h,t_end)
  real(8), intent(inout) :: t,r,u,m
  real(8), intent(out) :: h
  real(8), intent(in) :: t_end
  real(8) :: vmax, vmin, rmin, a, b, vi, vo, seed_dt, rsample, tsample
  integer :: i
  vmax=r/t
  if(op(1)%bpl_vmax>0d0)vmax=op(1)%bpl_vmax
  if(op(1)%exp_v0>0d0)vmax=20d0*op(1)%exp_v0
  if(associated(op(1)%v_grid))vmax=maxval(op(1)%v_grid)
  if(op(1)%vwind>0d0)vmax=op(1)%vwind
  rmin=0d0
  vmin=0d0
  if(associated(op(2)%r_grid_static))then
   i=1
   do while(i<size(op(2)%rho_static))
    if(op(2)%rho_static(i)>0d0)exit
    i=i+1
   end do
   if(i>1)i=i-1
   rmin=op(2)%r_grid_static(i)
   t=rmin/vmax
  elseif(associated(op(2)%v_grid))then
   i=1
   do while(i<size(op(2)%rho_expl))
    if(op(2)%rho_expl(i)>0d0)exit
    i=i+1
   end do
   if(i>1)i=i-1
   vmin=op(2)%v_grid(i)
   if(vmax<=vmin)then
    t=t_end+1d0
   else
    t=max(t,vmin*op(2)%delay/(vmax-vmin))
   end if
  end if
  ! A continuously supplied central wind starts at the launch radius, not
  ! at the fastest homologous ejecta front (where both velocities coincide).
  if(op(1)%vwind<=0d0.or.rmin>0d0.or.vmin>0d0)r=vmax*t
  ! Sample just inside contact, also handling a density ramp from zero.
  seed_dt=1d-8*t
  tsample=t+seed_dt
  rsample=nearest(vmax*tsample,-1d0)
  if(op(1)%vwind>0d0)rsample=r
  a=rho4pir2_in(rsample,tsample,op(1))
  b=rho4pir2_out(rsample,tsample,op(2))
  vi=v_in(rsample,tsample,op(1))
  vo=v_out(rsample,tsample,op(2))
  u=vmax
  if(a+b>0d0)u=(sqrt(a)*vi+sqrt(b)*vo)/(sqrt(a)+sqrt(b))
  m=max((a*max(vi-u,0d0)+b*max(u-vo,0d0))*seed_dt,1d0)
  h=seed_dt
 end subroutine initialize_simple_shell

 function simple_outer_edge(t) result(edge)
  real(8), intent(in) :: t
  real(8) :: edge
  edge=huge(1d0)
  if(associated(op(2)%r_grid_static).or.associated(op(2)%v_grid).or.op(2)%bpl_vmax>0d0) &
     edge=query_csm_outer_edge(t,op(2))
 end function simple_outer_edge

 subroutine simple_shell_rhs(t, y, dy, exterior)
  ! No scan-index side effects: rejected RK stages may visit earlier radii.
  real(8), intent(in) :: t, y(3)
  real(8), intent(out) :: dy(3)
  logical, intent(in) :: exterior
  real(8) :: a, b, vi, ri, ro, radius, rcsm, vo
  radius = y(1)
  vi = v_in(radius,t,op(1))
  a = rho4pir2_in(radius,t,op(1))
  b = 0d0
  if (.not.exterior) then
   ! Continue the interior density to locate the outer-edge event without
   ! integrating across its discontinuity.
   rcsm=min(radius,simple_outer_edge(t))
   b=4d0*pi*radius**2*query_csm_density(rcsm,t,op(2))
  end if
  ri = max(vi-y(2),0d0)
  vo=v_out(radius,t,op(2))
  ro = max(y(2)-vo,0d0)
  dy = [y(2), (a*ri**2-b*ro**2)/y(3), a*ri+b*ro]
 end subroutine simple_shell_rhs

 subroutine simple_shell_rk4(t, y, h, z, exterior)
  real(8), intent(in) :: t, y(3), h
  real(8), intent(out) :: z(3)
  logical, intent(in) :: exterior
  real(8) :: k1(3), k2(3), k3(3), k4(3)
  call simple_shell_rhs(t,y,k1,exterior)
  call simple_shell_rhs(t+h/2d0,y+h*k1/2d0,k2,exterior)
  call simple_shell_rhs(t+h/2d0,y+h*k2/2d0,k3,exterior)
  call simple_shell_rhs(t+h,y+h*k3,k4,exterior)
  z = y+h*(k1+2d0*k2+2d0*k3+k4)/6d0
 end subroutine simple_shell_rk4

 subroutine advance_simple_shell(t, r, u, m, h_next, t_end, at_edge, exterior)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  real(8), intent(inout) :: t, r, u, m, h_next
  real(8), intent(in) :: t_end
  logical, intent(out) :: at_edge
  logical, intent(in) :: exterior
  real(8) :: y(3), full(3), half(3), z(3), scale(3), h, err, edge, lo, hi, trial, rel, width
  integer :: j, cell
  y = [r,u,m]
  h = min(h_next,0.005d0*t,t_end-t)
  edge=simple_outer_edge(t)
  rel=max(u-v_out(r,t,op(2)),1d0)
  if (.not.exterior) h=min(h,0.05d0*max(edge-r,0d0)/rel+1d-6*t)
  if (exterior .and. r<1.2d0*edge) h=min(h,0.001d0*t)
  ! Do not jump over a narrow feature between RK stage evaluations.
  width=huge(1d0)
  if (.not.exterior) then
   if(associated(op(2)%r_grid_static))then
    cell=profile_interval(r,op(2)%r_grid_static)
    width=op(2)%r_grid_static(cell+1)-op(2)%r_grid_static(cell)
   elseif(associated(op(2)%v_grid))then
    cell=profile_interval(r/(t+op(2)%delay),op(2)%v_grid)
    width=(op(2)%v_grid(cell+1)-op(2)%v_grid(cell))*(t+op(2)%delay)
   end if
  end if
  h=min(h,0.5d0*width/rel)
  at_edge = .false.
  do
   call simple_shell_rk4(t,y,h,full,exterior)
   call simple_shell_rk4(t,y,h/2d0,half,exterior)
   call simple_shell_rk4(t+h/2d0,half,h/2d0,z,exterior)
   scale = 1d-8*max(abs(y),abs(z)) + [1d0,1d-3,1d0]
   err = maxval(abs(z-full)/scale)/15d0
   if (all(ieee_is_finite(z)) .and. ieee_is_finite(err)) then
    if (err <= 1d0 .and. z(3)>0d0 .and. z(2)>0d0) exit
   else
    err = 1d10
   end if
   h = h*max(0.1d0,0.8d0*max(err,1d0)**(-0.2d0))
   if (t+h==t) then
    h_next=0d0
    return
   end if
  end do
  if (.not.exterior .and. z(1)>=simple_outer_edge(t+h)) then
   lo = 0d0
   hi = h
   do j=1,50
    trial = (lo+hi)/2d0
    call simple_shell_rk4(t,y,trial/2d0,half,exterior)
    call simple_shell_rk4(t+trial/2d0,half,trial/2d0,z,exterior)
    if (z(1)<simple_outer_edge(t+trial)) then
     lo=trial
    else
     hi=trial
    end if
   end do
   h=(lo+hi)/2d0
   call simple_shell_rk4(t,y,h/2d0,half,exterior)
   call simple_shell_rk4(t+h/2d0,half,h/2d0,z,exterior)
   z(1)=simple_outer_edge(t+h)
   at_edge=.true.
  end if
  t=t+h
  r=z(1)
  u=z(2)
  m=z(3)
  h_next=h*min(2d0,max(0.5d0,0.9d0*max(err,1d-12)**(-0.2d0)))
 end subroutine advance_simple_shell

 subroutine shell_rhs_and_luminosity(u, r, m, t, op, du_dt, dr_dt, dm_dt, lum_fs, lum_rs)
  ! Evaluate shell dynamics and raw shock powers with one set of density/velocity calls.
  type(outflow_parameters),intent(inout):: op(1:2)
  real(8),intent(in):: u,r,m,t
  real(8),intent(out):: du_dt, dr_dt, dm_dt, lum_fs, lum_rs
  real(8):: v_inner, v_outer, rho_inner_4pir2, rho_outer_4pir2
  real(8):: rel_in, rel_out

  v_inner = v_in(r,t,op(1))
  v_outer = v_out(r,t,op(2))
  rho_inner_4pir2 = rho4pir2_in(r,t,op(1))
  rho_outer_4pir2 = rho4pir2_out(r,t,op(2))

  rel_in = max(v_inner - u, 0d0)
  rel_out = max(u - v_outer, 0d0)

  dm_dt = rho_inner_4pir2*rel_in + rho_outer_4pir2*rel_out
  du_dt = (rho_inner_4pir2*rel_in**2 - rho_outer_4pir2*rel_out**2) / m
  dr_dt = u
  lum_fs = 0.5d0*rho_outer_4pir2*rel_out**3
  lum_rs = 0.5d0*rho_inner_4pir2*rel_in**3
 end subroutine shell_rhs_and_luminosity

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
  ! Radiative-shock condition: compare the post-shock cooling time
  ! t_cool ~= 3p/q_ff to the flow time r/v_rel, not to the light-crossing
  ! time.  Photon escape is handled separately by the transport solver.
  thermal_factor = epsilon_ff * r / (3d0 * p_down * max(v_rel, 1d5))
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
  ! Radiative-shock condition: compare the post-shock cooling time
  ! t_cool ~= 3p/q_ff to the flow time r/v_rel, not to the light-crossing
  ! time.  Photon escape is handled separately by the transport solver.
  thermal_factor = epsilon_ff * r / (3d0 * p_down * max(v_rel, 1d5))
  eta_rs = eff_max * min(1d0, max(0d0, thermal_factor))
 end function reverse_shock_radiative_efficiency

end module integration
