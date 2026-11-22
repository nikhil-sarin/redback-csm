program transport_probe
 use csm_transport
 use constants, only: pi, clight
 implicit none
 type(dimless_state_type) :: state
 integer :: n, steps, i, j, mode
 real(8) :: dt, k, x, exact, err, volume, xl, xr, energy
 character(len=64) :: arg
 call get_command_argument(1,arg); read(arg,*)n
 call get_command_argument(2,arg); read(arg,*)dt
 call get_command_argument(3,arg); read(arg,*)steps
 call get_command_argument(4,arg); read(arg,*)k
 do mode=1,2
  call reset_dimless_state(state)
  state%n_zones=n
  state%x_sh=1d0
  state%x_min=1d0
  state%x_ph=2d0
  state%x_csm_out=2d0
  state%csm_powerlaw_fast=.true.
  state%csm_eta_pow=0d0
  state%tau_csm_in=4d0
  state%R_csm_in=1d0
  state%u0=1d0
  state%x_sh_dot=0d0
  state%x_ph_dot=0d0
  state%lum_heat_cgs=0d0
  state%rannacher_left=0
  call setup_xi_grid(state)
  do i=1,n
   x=1d0+state%xi_grid(i)
   state%e_grid(i)=(cos(k*(x-1d0))+sin(k*(x-1d0))/k)/x
  end do
  if(mode==2)then
   state%e_grid=0d0
   state%tau_csm_in=1d-20
   state%lum_heat_cgs=4d0*pi*clight/state%tau_csm_in
  end if
  do j=1,steps
   call solve_diffusion_dimless(state,dt)
  end do
  err=0d0
  energy=0d0
  do i=1,n
   x=1d0+state%xi_grid(i)
   exact=(cos(k*(x-1d0))+sin(k*(x-1d0))/k)/x*exp(-k*k*dt*steps)
   err=max(err,abs(state%e_grid(i)-exact))
   xl=max(1d0,x-0.5d0/dble(n-1))
   xr=min(2d0,x+0.5d0/dble(n-1))
   volume=(xr**3-xl**3)/3d0
   energy=energy+state%e_grid(i)*volume
  end do
  write(*,*)mode,err,energy,minval(state%e_grid)
 end do
end program transport_probe
