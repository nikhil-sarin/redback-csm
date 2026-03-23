module get_vals

 implicit none

 type outflow_parameters
  ! Efficiency parameter
  real(8):: eff
  ! Parameters for wind-like outflow
  real(8),pointer:: mdot(:),t_grid(:)
  real(8):: vwind
  ! Parameters for explosion-like outflow
  real(8):: Mej, Eej
  ! Additional parameters for exponential density distribution
  real(8):: exp_v0
  ! Additional parameters for broken power-law density distribution
  real(8):: bpl_n, bpl_d, bpl_vt, bpl_rho0
  ! Arrays for setting arbitrary density distribution in an explosion
  real(8),pointer:: rho_expl(:),v_grid(:)
  integer:: scan_i
  real(8):: t_ref
  ! Interval between two explosions
  real(8):: delay
 end type outflow_parameters

 interface
  function get_val(r,t,op) result(val)
   import outflow_parameters
   real(8),intent(in):: r,t
   type(outflow_parameters),intent(inout):: op
   real(8):: val
  end function get_val
 end interface

 type(outflow_parameters):: op(1:2)
 procedure(get_val),pointer:: rho4pir2_in, rho4pir2_out, v_in, v_out

contains

 subroutine reset_outflow(op)
  type(outflow_parameters),intent(inout):: op

  op%eff = 0d0
  if(associated(op%mdot))nullify(op%mdot)
  if(associated(op%t_grid))nullify(op%t_grid)
  op%vwind = 0d0
  op%Mej = 0d0
  op%Eej = 0d0
  op%exp_v0 = 0d0
  op%bpl_n = 0d0
  op%bpl_d = 0d0
  op%bpl_vt = 0d0
  op%bpl_rho0 = 0d0
  if(associated(op%rho_expl))nullify(op%rho_expl)
  if(associated(op%v_grid))nullify(op%v_grid)
  op%scan_i = 0
  op%t_ref = 0d0
  op%delay = 0d0
 end subroutine reset_outflow

 subroutine get_tauprep_explosion(tauprep)
  ! Compute integral needed to compute optical depth for explosions
  ! tauprep = tau * t^2 / kappa
  real(8),allocatable,intent(out):: tauprep(:)
  integer:: i, n

  if(allocated(tauprep))deallocate(tauprep)
  allocate(tauprep,mold=op(2)%rho_expl)
  n = size(op(2)%rho_expl)
  tauprep(n) = 0d0
  do i = n-1, 1, -1
   tauprep(i) = tauprep(i+1) &
              + op(2)%rho_expl(i+1)*(op(2)%v_grid(i+1)-op(2)%v_grid(i))
  end do
  tauprep = op(2)%t_ref**3*tauprep

 end subroutine get_tauprep_explosion

 subroutine get_exp_v0(op)
! Computes frequently used quantities for exponential ejecta
  type(outflow_parameters),intent(inout):: op
  ! Velocity scale of ejecta as defined in Owocki+19
  op%exp_v0 = sqrt(op%Eej/(6d0*op%Mej))
 end subroutine get_exp_v0

 function rho_exponential(r,t,op)
! Computes 4*pi*r^2*rho at a given location and time for exponential ejecta
  type(outflow_parameters),intent(inout):: op
  real(8),intent(in):: r,t
  real(8):: rho_exponential, tp
  tp = t+op%delay
  rho_exponential = op%Mej*r*r/(2d0*(op%exp_v0*tp)**3)*exp(-r/(op%exp_v0*tp))
 end function rho_exponential

 subroutine get_bpl_coeffs(op)
! Computes frequently used quantities for broken power-law ejecta
  type(outflow_parameters),intent(inout):: op
  real(8):: n,d,Mej,Eej

  n = op%bpl_n
  d = op%bpl_d
  Mej = op%Mej
  Eej = op%Eej

  ! Transition velocity
  op%bpl_vt = sqrt( (2d0*(5d0-d)*(n-5d0)*Eej) &
                  /     ((3d0-d)*(n-3d0)*Mej) )

  ! Density scale
  op%bpl_rho0 = (    (3d0-d)*(n-3d0)*Mej)**2.5d0 &
              / (2d0*(5d0-d)*(n-5d0)*Eej)**1.5d0 &
              / (n-d)
 end subroutine get_bpl_coeffs

 function rho_bpl(r,t,op)
! Computes 4*pi*r^2*rho at a given location and time for broken power-law ejecta
  type(outflow_parameters),intent(inout):: op
  real(8),intent(in):: r,t
  real(8):: rho_bpl, tp, v

  tp = t+op%delay
  v  = r/tp
  if(r/tp>op%bpl_vt)then ! Outer ejecta
   rho_bpl = op%bpl_rho0 * (op%bpl_vt/v)**op%bpl_n * v**2/tp
  else                  ! Inner ejecta
   rho_bpl = op%bpl_rho0 * (op%bpl_vt/v)**op%bpl_d * v**2/tp
  end if

 end function rho_bpl

 function rho_explosion(r,t,op)
! Computes 4*pi*r^2*rho at a given location and time for explosion-like ejecta
  use constants,only:pi,intpol
  type(outflow_parameters),intent(inout):: op
  real(8),intent(in):: r,t
  real(8):: rho_explosion, tp
  integer:: i,n, is,ie,increment
  ! op%v_grid : grid of velocity to describe the ejecta density distribution
  ! op%rho_expl : density distribution as a function of v at t_ref
  ! op%t_ref : reference time when density distribution was recorded
  tp = t+op%delay
  n = size(op%v_grid)

  ! Specify scan range
  if(op%scan_i>0)then
   is = op%scan_i  ! start from previous cell
   ie = n-1
   increment = 1   ! shell only moves outwards into outer outflow
  else
   is = -op%scan_i ! start from previous cell
   ie = 1
   increment = -1  ! shell only retreates inwards into inner outflow
  end if

  if(r<=op%v_grid(1)*tp)then
   op%scan_i = increment
   rho_explosion = op%rho_expl(1)
  elseif(r>=op%v_grid(n)*tp)then
   op%scan_i = n*increment
   rho_explosion = op%rho_expl(n)
  else
   do i = is, ie, increment
    if(op%v_grid(i)*tp<=r.and.op%v_grid(i+1)*tp>r)exit
   end do
   op%scan_i = i*increment
   rho_explosion = intpol(r/tp,op%v_grid(i:i+1),op%rho_expl(i:i+1))
  end if
  rho_explosion = 4d0*pi*rho_explosion*(op%t_ref/tp)**3*r*r
 end function rho_explosion

 function rho_wind(r,t,op)
! Computes 4*pi*r^2*rho at a given location and time for wind-like ejecta
  use constants,only:intpol
  type(outflow_parameters),intent(inout):: op
  real(8),intent(in):: r,t
  real(8):: rho_wind, t_wind, Mdot_wind
  integer:: i,n, is,ie,increment
  ! op%t_grid : grid of time before/since explosion
  ! op%mdot   : mass loss rate at corresponding times on t_grid
  t_wind = abs(r/op%vwind-t)
  n = size(op%t_grid)

  ! Specify scan range
  if(op%scan_i>0)then
   is = op%scan_i  ! start from previous cell
   ie = n-1
   increment = 1   ! shell only moves outwards into outer outflow
  else
   is = -op%scan_i ! start from previous cell
   ie = 1
   increment = -1  ! shell only retreates inwards into inner outflow
  end if

  if(t_wind<=op%t_grid(1))then
   op%scan_i = increment
   Mdot_wind = op%mdot(1)
  elseif(t_wind>=op%t_grid(n))then
   op%scan_i = n*increment
   Mdot_wind = op%mdot(n)
  else
   do i = is, ie, increment
    if(op%t_grid(i)<=t_wind.and.op%t_grid(i+1)>t_wind)exit
   end do
   op%scan_i = i*increment
   Mdot_wind = intpol(t_wind,op%t_grid(i:i+1),op%mdot(i:i+1))
  end if
  rho_wind = Mdot_wind/op%vwind
 end function rho_wind

 function v_wind(r,t,op)
  type(outflow_parameters),intent(inout):: op
  real(8),intent(in):: r,t
  real(8):: v_wind
  v_wind = op%vwind
 end function v_wind

function v_explosion(r,t,op)
  type(outflow_parameters),intent(inout):: op
  real(8),intent(in):: r,t
  real(8):: v_explosion, tp
  tp = t+op%delay
  v_explosion = r/tp
 end function v_explosion

 function query_csm_density(r,t,op_in) result(rho_csm)
! Side-effect-safe CSM density query for transport modules.
  use constants,only:pi
  type(outflow_parameters),intent(in):: op_in
  real(8),intent(in):: r,t
  real(8):: rho_csm
  type(outflow_parameters):: op_local

  op_local = op_in
 rho_csm = rho4pir2_out(r,t,op_local)/(4d0*pi*max(r,1d0)**2)
 end function query_csm_density

 function query_ejecta_density(r,t,op_in) result(rho_ej)
! Side-effect-safe ejecta density query for transport modules.
  use constants,only:pi
  type(outflow_parameters),intent(in):: op_in
  real(8),intent(in):: r,t
  real(8):: rho_ej
  type(outflow_parameters):: op_local

  op_local = op_in
  rho_ej = rho4pir2_in(r,t,op_local)/(4d0*pi*max(r,1d0)**2)
 end function query_ejecta_density

 function query_csm_velocity(r,t,op_in) result(v_csm)
! Side-effect-safe CSM velocity query for transport modules.
  type(outflow_parameters),intent(in):: op_in
  real(8),intent(in):: r,t
  real(8):: v_csm
  type(outflow_parameters):: op_local

  op_local = op_in
 v_csm = v_out(r,t,op_local)
 end function query_csm_velocity

 function query_ejecta_velocity(r,t,op_in) result(v_ej)
! Side-effect-safe ejecta velocity query for transport modules.
  type(outflow_parameters),intent(in):: op_in
  real(8),intent(in):: r,t
  real(8):: v_ej
  type(outflow_parameters):: op_local

  op_local = op_in
  v_ej = v_in(r,t,op_local)
 end function query_ejecta_velocity

 function query_csm_inner_edge(t,op_in) result(r_in)
! Estimate the inner support of the supplied CSM profile.
  real(8),intent(in):: t
  type(outflow_parameters),intent(in):: op_in
  real(8):: r_in
  integer:: n

  r_in = 1d0

  if(associated(op_in%t_grid).and.associated(op_in%mdot))then
   n = size(op_in%t_grid)
   if(n>0.and.op_in%vwind>0d0)then
    r_in = max(op_in%vwind*(t + minval(op_in%t_grid(1:n))), 1d0)
   end if
  elseif(associated(op_in%v_grid).and.associated(op_in%rho_expl))then
   n = size(op_in%v_grid)
   if(n>0)r_in = max(op_in%v_grid(1)*(t + op_in%delay), 1d0)
  elseif(op_in%exp_v0>0d0)then
   r_in = max(0.05d0*op_in%exp_v0*(t + op_in%delay), 1d0)
  elseif(op_in%bpl_vt>0d0)then
   r_in = max(0.05d0*op_in%bpl_vt*(t + op_in%delay), 1d0)
  end if
 end function query_csm_inner_edge

 function query_csm_outer_edge(t,op_in) result(r_out)
! Estimate the finite outer support of the supplied CSM profile.
! For effectively steady winds this returns a very large radius so the
! transport layer can treat them as non-emergent / interaction-phase only.
  real(8),intent(in):: t
  type(outflow_parameters),intent(in):: op_in
  real(8):: r_out
  integer:: n

  r_out = huge(1d0)

  if(associated(op_in%t_grid).and.associated(op_in%mdot))then
   n = size(op_in%t_grid)
   if(n>0.and.op_in%vwind>0d0)then
    r_out = op_in%vwind*(t + maxval(op_in%t_grid(1:n)))
   end if
  elseif(associated(op_in%v_grid).and.associated(op_in%rho_expl))then
   n = size(op_in%v_grid)
   if(n>0)r_out = op_in%v_grid(n)*(t + op_in%delay)
  elseif(op_in%exp_v0>0d0)then
   r_out = 20d0*op_in%exp_v0*(t + op_in%delay)
  elseif(op_in%bpl_vt>0d0)then
   r_out = 20d0*op_in%bpl_vt*(t + op_in%delay)
  end if
 end function query_csm_outer_edge

 function query_tau_to_edge(r,t,op_in,kappa) result(tau_out)
! Side-effect-safe optical depth from radius r to the outer edge of the CSM.
  type(outflow_parameters),intent(in):: op_in
  real(8),intent(in):: r,t,kappa
  real(8):: tau_out
  real(8):: r0, r1, rout, dr, rho0, rho1, ratio
  integer:: i, nseg

  tau_out = 0d0
  if(kappa<=0d0)return

  r0 = max(r,1d0)
  rout = query_csm_outer_edge(t,op_in)
  if(.not.(rout>r0))return

  ! For effectively steady winds, do not integrate to an absurd outer radius.
  if(rout>r0*1d4)rout = r0*1d4

  nseg = 128
  ratio = (rout/r0)**(1d0/dble(nseg))
  r1 = r0
  rho1 = query_csm_density(r1,t,op_in)

  do i = 1, nseg
   r0 = r1
   rho0 = rho1
   r1 = min(rout, r1*ratio)
   rho1 = query_csm_density(r1,t,op_in)
   dr = r1-r0
   tau_out = tau_out + 0.5d0*kappa*(rho0+rho1)*dr
   if(r1>=rout)exit
  end do
 end function query_tau_to_edge

 function query_csm_photosphere_radius(r_inner,t,op_in,kappa) result(r_ph)
! Side-effect-safe estimate of the radius where tau(r)=2/3.
  type(outflow_parameters),intent(in):: op_in
  real(8),intent(in):: r_inner,t,kappa
  real(8):: r_ph
  real(8):: rlo, rhi, rmid, tau_mid, rout
  integer:: iter

  rlo = max(r_inner,1d0)
  rout = query_csm_outer_edge(t,op_in)
  if(.not.(rout>rlo))then
   r_ph = rlo
   return
  end if

  if(query_tau_to_edge(rlo,t,op_in,kappa)<=2d0/3d0)then
   r_ph = rlo
   return
  end if

  rhi = rout
  if(rhi>rlo*1d4)rhi = rlo*1d4

  do iter = 1, 60
   rmid = sqrt(rlo*rhi)
   tau_mid = query_tau_to_edge(rmid,t,op_in,kappa)
   if(tau_mid>2d0/3d0)then
    rlo = rmid
   else
    rhi = rmid
   end if
  end do

  r_ph = rhi
 end function query_csm_photosphere_radius

end module get_vals
