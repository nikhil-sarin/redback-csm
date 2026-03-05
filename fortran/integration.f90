module integration

 use get_vals

 implicit none

contains

 function dmdt(u,r,m,t,op)
  ! Continuity equation
  type(outflow_parameters),intent(inout):: op(1:2)
  real(8),intent(in):: u,r,m,t
  real(8):: dmdt
  dmdt = rho4pir2_in (r,t,op(1))*(v_in (r,t,op(1))-u) &
       + rho4pir2_out(r,t,op(2))*(u-v_out(r,t,op(2)))
 end function dmdt

 function dudt(u,r,m,t,op)
  ! Equation of motion
  type(outflow_parameters),intent(inout):: op(1:2)
  real(8),intent(in):: u,r,m,t
  real(8):: dudt
  dudt = ( rho4pir2_in (r,t,op(1))*(v_in (r,t,op(1))-u)**2 &
         - rho4pir2_out(r,t,op(2))*(u-v_out(r,t,op(2)))**2 ) / m
 end function dudt

 function drdt(u,r,m,t,op)
  ! Velocity
  type(outflow_parameters),intent(in):: op(1:2)
  real(8),intent(in):: u,r,m,t
  real(8):: drdt
  drdt = u
 end function drdt

 function shock_luminosity(r,t,u,op) result(lum)
  ! Compute shock luminosity
  type(outflow_parameters),intent(inout):: op(1:2)
  real(8),intent(in):: r,t,u
  real(8):: lum
  lum = ( rho4pir2_in (r,t,op(1))*(v_in (r,t,op(1))-u)**3 &
        + rho4pir2_out(r,t,op(2))*(u-v_out(r,t,op(2)))**3 ) * 0.5d0
 end function shock_luminosity

end module integration
