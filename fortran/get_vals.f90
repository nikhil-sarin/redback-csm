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

end module get_vals
