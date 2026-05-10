module lc_mod

 use constants,only:pi,year,clight,intpol,temperature
 use csm_runtime, only: diffusion_enabled, eff_global, erad, &
                        shock_efficiency_mode, &
                        configure_runtime, reset_diffusion_state, &
                        shell_optical_depth, photosphere_radius, &
                        solve_diffusion_step, &
                        set_shock_efficiency_mode_runtime => set_shock_efficiency_mode
use csm_transport, only: dimless_state_type, dimless_comoving_transport_step, &
                                   reset_dimless_state, initialize_dimless_state, dimless_to_cgs, &
                                   dimless_dynamics_timescale_cgs, update_dimless_shock_luminosities

 implicit none

 real(8),allocatable,dimension(:),public:: tarray, Larray, temparray, rarray, varray, marray, ldiff, lfs, lrs
 real(8),allocatable,dimension(:),public:: rfsarray, rpharray, etraparray, tleakarray, tauarray_transport
 public:: lightcurve_wind_exponential, lightcurve_wind_bpl, &
          lightcurve_bpl_wind, lightcurve_exponential_wind, &
          lightcurve_wind_explosion, lightcurve_explosion_wind, &
          lightcurve_wind_wind, lightcurve_explosion_explosion, &
          lightcurve_bpl_bpl, lightcurve_exponential_exponential, &
          lightcurve_explosion_bpl, lightcurve_bpl_exponential, &
          lightcurve_exponential_explosion, lightcurve_explosion_exponential, &
          lightcurve_static_bpl, lightcurve_static_exponential, &
          set_efficiency_mode, set_run_mode, set_transport_parameters, &
          set_bpl_cutoff_ratio, get_dimless_state_debug

 private:: finalize_outputs, do_main_loop
 private:: get_diffuse_lc
 private
integer,parameter:: ll=200000, n_skip_initial_outputs=10
 real(8),parameter:: simple_store_min_dlogt=2d-2
 real(8),parameter:: simple_store_min_rell=2d-2
 real(8),parameter:: simple_store_min_relr=1d-2
 real(8),dimension(ll):: t_array, L_array, ld_array, r_array, v_array, m_array, fs_array, rs_array
 real(8),dimension(ll):: rfs_array, rph_array, etrap_array, tleak_array, tau_array_transport
 integer,dimension(ll):: i_array
 real(8):: t_start=1d1, t_end=10d0*year
 real(8):: u,r,m,t

 ! Run mode: 1=simple, 3=transport
 integer :: run_mode = 1

 ! Transport mode parameters
 integer :: n_rad_zones_global = 20
 real(8) :: opacity_const_global = 0.34d0

 ! Mode 3 persistent state (survives across dimless_comoving_transport_step calls)
 type(dimless_state_type) :: dl_state_global

contains

 subroutine set_efficiency_mode(mode)
  integer,intent(in):: mode

  call set_shock_efficiency_mode_runtime(mode)
 end subroutine set_efficiency_mode

 subroutine set_run_mode(mode)
  integer, intent(in) :: mode
  run_mode = mode
  if (mode /= 1 .and. mode /= 3) then
   print *, 'WARNING: Invalid run_mode', mode, 'using simple (1)'
   run_mode = 1
  endif
 end subroutine set_run_mode

 subroutine set_transport_parameters(n_zones, kappa_val)
  integer, intent(in), optional :: n_zones
  real(8), intent(in), optional :: kappa_val
  
  if (present(n_zones)) n_rad_zones_global = n_zones
  if (present(kappa_val)) opacity_const_global = kappa_val
 end subroutine set_transport_parameters

 subroutine set_bpl_cutoff_ratio(ratio)
  use get_vals, only: set_global_bpl_vmax_ratio
  real(8), intent(in) :: ratio
  call set_global_bpl_vmax_ratio(ratio)
 end subroutine set_bpl_cutoff_ratio

 function initial_bpl_shell_velocity(op_bpl) result(v_init)
  use get_vals, only: outflow_parameters
  type(outflow_parameters),intent(in):: op_bpl
  real(8):: v_init

  if(op_bpl%bpl_vmax>0d0)then
   ! The shell is initialized just inside the finite outer ejecta edge; r=1.2*u*t
   ! below then places the initial interaction at r ~= v_max*t.
   v_init = op_bpl%bpl_vmax/1.2d0
  else
   v_init = 20d0*op_bpl%bpl_vt/1.2d0
  end if
  v_init = min(v_init,0.9d0*clight)
 end function initial_bpl_shell_velocity

 function initial_exp_shell_velocity(op_exp) result(v_init)
  use get_vals, only: outflow_parameters
  type(outflow_parameters),intent(in):: op_exp
  real(8):: v_init

  ! Exponential profiles have infinite formal support.  Twenty scale velocities
  ! is already beyond the effective mass-bearing edge and avoids superluminal
  ! startup artifacts from the old 100*v0 convention.
  v_init = min(20d0*op_exp%exp_v0/1.2d0,0.9d0*clight)
 end function initial_exp_shell_velocity

 subroutine finalize_outputs(csm_type, kappa)
  integer,intent(in):: csm_type
  real(8),intent(in),optional:: kappa

  if(run_mode == 1)then
   if(present(kappa))then
    call get_diffuse_lc(csm_type,kappa)
    temparray = temperature(ldiff,rarray)
    call drop_initial_output_rows(n_skip_initial_outputs)
   elseif(allocated(ldiff))then
    deallocate(ldiff)
   end if
  end if
 end subroutine finalize_outputs

 subroutine drop_initial_output_rows(n_drop)
  integer,intent(in):: n_drop
  integer:: n_old, n_new
  real(8),allocatable,dimension(:):: tmp

  if(.not.allocated(tarray))return
  n_old = size(tarray)
  if(n_drop <= 0 .or. n_old <= n_drop)return
  n_new = n_old - n_drop

  allocate(tmp(n_new)); tmp = tarray(n_drop+1:n_old); call move_alloc(tmp,tarray)
  allocate(tmp(n_new)); tmp = larray(n_drop+1:n_old); call move_alloc(tmp,larray)
  allocate(tmp(n_new)); tmp = lfs(n_drop+1:n_old); call move_alloc(tmp,lfs)
  allocate(tmp(n_new)); tmp = lrs(n_drop+1:n_old); call move_alloc(tmp,lrs)
  allocate(tmp(n_new)); tmp = temparray(n_drop+1:n_old); call move_alloc(tmp,temparray)
  allocate(tmp(n_new)); tmp = rarray(n_drop+1:n_old); call move_alloc(tmp,rarray)
  allocate(tmp(n_new)); tmp = varray(n_drop+1:n_old); call move_alloc(tmp,varray)
  allocate(tmp(n_new)); tmp = marray(n_drop+1:n_old); call move_alloc(tmp,marray)
  allocate(tmp(n_new)); tmp = ldiff(n_drop+1:n_old); call move_alloc(tmp,ldiff)
  allocate(tmp(n_new)); tmp = rfsarray(n_drop+1:n_old); call move_alloc(tmp,rfsarray)
  allocate(tmp(n_new)); tmp = rpharray(n_drop+1:n_old); call move_alloc(tmp,rpharray)
  allocate(tmp(n_new)); tmp = etraparray(n_drop+1:n_old); call move_alloc(tmp,etraparray)
  allocate(tmp(n_new)); tmp = tleakarray(n_drop+1:n_old); call move_alloc(tmp,tleakarray)
  allocate(tmp(n_new)); tmp = tauarray_transport(n_drop+1:n_old); call move_alloc(tmp,tauarray_transport)
 end subroutine drop_initial_output_rows

 subroutine lightcurve_wind_exponential(Mdotinput, tinput, vwindinput, Mexp, Eexp, eff, kappa)

! purpose: To compute LC for exponential explosion interacting with wind

  use get_vals

  real(8),intent(in):: Eexp, Mexp, vwindinput
  real(8),intent(in),target:: Mdotinput(:), tinput(:)
  real(8),intent(in),optional:: eff, kappa

  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_explosion
  v_out => v_wind
  rho4pir2_in  => rho_exponential
  rho4pir2_out => rho_wind

  op(1)%scan_i = -1
  op(2)%scan_i = 1

  op(1)%Mej = Mexp
  op(1)%Eej = Eexp
  op(1)%delay = 0d0
  op(2)%mdot => Mdotinput
  op(2)%t_grid => tinput
  op(2)%vwind = vwindinput

  call get_exp_v0(op(1))

  t = t_start   ! Use non-zero initial time to avoid singularity
  u = max(initial_exp_shell_velocity(op(1)),op(2)%vwind)
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(2,eff,kappa)

  call do_main_loop
  call finalize_outputs(2,kappa)

 end subroutine lightcurve_wind_exponential

 subroutine lightcurve_wind_explosion(Mdotinput, tinput, vwindinput, rho_input, vinput, t_ref, eff, kappa)

! purpose: To compute LC for arbitrary explosion interacting with wind

  use get_vals

  real(8),intent(in):: vwindinput, t_ref
  real(8),intent(in),target:: Mdotinput(:), tinput(:), rho_input(:), vinput(:)
  real(8),intent(in),optional:: eff, kappa

  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_explosion
  v_out => v_wind
  rho4pir2_in  => rho_explosion
  rho4pir2_out => rho_wind

  op(1)%scan_i = -size(vinput)
  op(2)%scan_i = 1

  op(1)%rho_expl => rho_input
  op(1)%v_grid => vinput
  op(1)%t_ref = t_ref
  op(1)%delay = 0d0
  op(2)%mdot => Mdotinput
  op(2)%t_grid => tinput
  op(2)%vwind = vwindinput

  t = t_start   ! Use non-zero initial time to avoid singularity
  u = max(maxval(vinput),op(2)%vwind)
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(2,eff,kappa)

  call do_main_loop
  call finalize_outputs(2,kappa)

 end subroutine lightcurve_wind_explosion

 subroutine lightcurve_wind_bpl(Mdotinput, tinput, vwindinput, inner_slope, outer_slope, Mexp, Eexp, eff, kappa)

! purpose: To compute LC for broken power-law explosion interacting with wind

  use get_vals
  use integration, only: forward_shock_radiative_efficiency, reverse_shock_radiative_efficiency

  real(8),intent(in):: Eexp, Mexp, inner_slope, outer_slope, vwindinput
  real(8),intent(in),target:: Mdotinput(:), tinput(:)
  real(8),intent(in),optional:: eff, kappa
  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_explosion
  v_out => v_wind
  rho4pir2_in  => rho_bpl
  rho4pir2_out => rho_wind

  op(1)%scan_i = -1
  op(2)%scan_i = 1

  op(1)%Mej = Mexp
  op(1)%Eej = Eexp
  op(1)%bpl_d = inner_slope
  op(1)%bpl_n = outer_slope
  op(1)%delay = 0d0
  op(2)%mdot => Mdotinput
  op(2)%t_grid => tinput
  op(2)%vwind = vwindinput

  call get_bpl_coeffs(op(1))

  t = t_start   ! Use non-zero initial time to avoid singularity
  u = max(initial_bpl_shell_velocity(op(1)),op(2)%vwind)
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(2,eff,kappa)

  call do_main_loop
  call finalize_outputs(2,kappa)

end subroutine lightcurve_wind_bpl


 subroutine lightcurve_bpl_wind(inner_slope, outer_slope, Mexp, Eexp, Mdotinput, tinput, vwindinput, eff, kappa)

! purpose: To compute LC for wind interacting with broken power-law explosion

  use get_vals

  real(8),intent(in):: Eexp, Mexp, inner_slope, outer_slope, vwindinput
  real(8),intent(in),target:: Mdotinput(:), tinput(:)
  real(8),intent(in),optional:: eff, kappa

  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_wind
  v_out => v_explosion
  rho4pir2_in  => rho_wind
  rho4pir2_out => rho_bpl

  op(1)%scan_i = -size(tinput)
  op(2)%scan_i = 1

  op(1)%mdot => Mdotinput
  op(1)%t_grid => tinput
  op(1)%vwind = vwindinput
  op(2)%Mej = Mexp
  op(2)%Eej = Eexp
  op(2)%bpl_d = inner_slope
  op(2)%bpl_n = outer_slope
  op(2)%delay = 0d0

  call get_bpl_coeffs(op(2))

  t = t_start   ! Use non-zero initial time to avoid singularity
  r = 1d0
  u = r/t
  m = 1d0
  erad = 0d0
  call configure_runtime(3,eff,kappa)

  call do_main_loop
  call finalize_outputs(3,kappa)

 end subroutine lightcurve_bpl_wind

 subroutine lightcurve_exponential_wind(Mexp, Eexp, Mdotinput, tinput, vwindinput, eff, kappa)

! purpose: To compute LC for wind interacting with exponential explosion

  use get_vals

  real(8),intent(in):: Eexp, Mexp, vwindinput
  real(8),intent(in),target:: Mdotinput(:), tinput(:)
  real(8),intent(in),optional:: eff, kappa

  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_wind
  v_out => v_explosion
  rho4pir2_in  => rho_wind
  rho4pir2_out => rho_exponential

  op(1)%scan_i = -size(tinput)
  op(2)%scan_i = 1

  op(1)%mdot => Mdotinput
  op(1)%t_grid => tinput
  op(1)%vwind = vwindinput
  op(2)%Mej = Mexp
  op(2)%Eej = Eexp
  op(2)%delay = 0d0

  call get_exp_v0(op(2))

  t = t_start   ! Use non-zero initial time to avoid singularity
  r = 1d0
  u = r/t
  m = 1d0
  erad = 0d0
  call configure_runtime(4,eff,kappa)

  call do_main_loop
  call finalize_outputs(4,kappa)

 end subroutine lightcurve_exponential_wind

 subroutine lightcurve_explosion_wind(rho_input, vinput, t_ref, Mdotinput, tinput, vwindinput, eff, kappa)

! purpose: To compute LC for wind interacting with an arbitrary explosion

  use get_vals

  real(8),intent(in):: t_ref, vwindinput
  real(8),intent(in),target:: rho_input(:), vinput(:), Mdotinput(:), tinput(:)
  real(8),intent(in),optional:: eff, kappa

  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_wind
  v_out => v_explosion
  rho4pir2_in  => rho_wind
  rho4pir2_out => rho_explosion

  op(1)%scan_i = -size(tinput)
  op(2)%scan_i = 1

  op(1)%mdot => Mdotinput
  op(1)%t_grid => tinput
  op(1)%vwind = vwindinput
  op(2)%rho_expl => rho_input
  op(2)%v_grid => vinput
  op(2)%t_ref = t_ref
  op(2)%delay = 0d0

  t = t_start   ! Use non-zero initial time to avoid singularity
  r = 1d0
  u = r/t
  m = 1d0
  erad = 0d0
  call configure_runtime(5,eff,kappa)

  call do_main_loop
  call finalize_outputs(5,kappa)

 end subroutine lightcurve_explosion_wind

 subroutine lightcurve_wind_wind(Mdoto, tinputo, vwindo, Mdoti, tinputi, vwindi, eff, kappa)

! purpose: To compute LC for wind interacting with another wind

  use get_vals

  real(8),intent(in):: vwindo, vwindi
  real(8),intent(in),target:: Mdoto(:), tinputo(:), Mdoti(:), tinputi(:)
  real(8),intent(in),optional:: eff, kappa

  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_wind
  v_out => v_wind
  rho4pir2_in  => rho_wind
  rho4pir2_out => rho_wind

  op(1)%scan_i = -size(tinputi)
  op(2)%scan_i = 1

  op(1)%mdot => Mdoti
  op(1)%t_grid => tinputi
  op(1)%vwind = vwindi
  op(2)%mdot => Mdoto
  op(2)%t_grid => tinputo
  op(2)%vwind = vwindo

  t = t_start   ! Use non-zero initial time to avoid singularity
  r = 1d0
  u = 0.5d0*(vwindi+vwindo)
  m = 1d0
  erad = 0d0
  call configure_runtime(2,eff,kappa)

  call do_main_loop
  call finalize_outputs(2,kappa)

 end subroutine lightcurve_wind_wind

 subroutine lightcurve_bpl_bpl(inner_slopeo, outer_slopeo, Mexpo, Eexpo, &
                               inner_slopei, outer_slopei, Mexpi, Eexpi, &
                               interval, eff, kappa)

! purpose: To compute LC for interaction between two explosions (bpl-bpl)

  use get_vals

  real(8),intent(in):: inner_slopeo, outer_slopeo, Mexpo, Eexpo, inner_slopei, outer_slopei, Mexpi, Eexpi, interval
  real(8),intent(in),optional:: eff, kappa

  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_explosion
  v_out => v_explosion
  rho4pir2_in  => rho_bpl
  rho4pir2_out => rho_bpl

  op(1)%Mej = Mexpi
  op(1)%Eej = Eexpi
  op(1)%bpl_d = inner_slopei
  op(1)%bpl_n = outer_slopei
  op(1)%delay = 0d0
  op(2)%Mej = Mexpo
  op(2)%Eej = Eexpo
  op(2)%bpl_d = inner_slopeo
  op(2)%bpl_n = outer_slopeo
  op(2)%delay = interval

  call get_bpl_coeffs(op(1))
  call get_bpl_coeffs(op(2))

  t = t_start   ! Use non-zero initial time to avoid singularity
  u = initial_bpl_shell_velocity(op(1))
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(3,eff,kappa)

  call do_main_loop
  call finalize_outputs(3,kappa)

 end subroutine lightcurve_bpl_bpl

 subroutine lightcurve_exponential_exponential(Mexpo, Eexpo, Mexpi, Eexpi, interval, eff, kappa)

! purpose: To compute LC for interaction between two explosions (exponential-exponential)

  use get_vals

  real(8),intent(in):: Eexpo, Mexpo, Eexpi, Mexpi, interval
  real(8),intent(in),optional:: eff, kappa

  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_explosion
  v_out => v_explosion
  rho4pir2_in  => rho_exponential
  rho4pir2_out => rho_exponential

  op(1)%Mej = Mexpi
  op(1)%Eej = Eexpi
  op(1)%delay = 0d0
  op(2)%Mej = Mexpo
  op(2)%Eej = Eexpo
  op(2)%delay = interval

  call get_exp_v0(op(1))
  call get_exp_v0(op(2))

  t = t_start   ! Use non-zero initial time to avoid singularity
  u = initial_exp_shell_velocity(op(1))
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(4,eff,kappa)

  call do_main_loop
  call finalize_outputs(4,kappa)

 end subroutine lightcurve_exponential_exponential

 subroutine lightcurve_explosion_explosion(rho_o,vgrido,trefo,rho_i,vgridi,trefi,interval, eff, kappa)

! purpose: To compute LC for interaction between two arbitrary explosions

  use get_vals

  real(8),intent(in):: trefo,trefi,interval
  real(8),intent(in),target:: rho_o(:),vgrido(:),rho_i(:),vgridi(:)
  real(8),intent(in),optional:: eff, kappa

  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_explosion
  v_out => v_explosion
  rho4pir2_in  => rho_explosion
  rho4pir2_out => rho_explosion

  op(1)%scan_i = -size(vgridi)
  op(2)%scan_i = 1

  op(1)%rho_expl => rho_i
  op(1)%v_grid => vgridi
  op(1)%t_ref = trefi
  op(1)%delay = 0d0
  op(2)%rho_expl => rho_o
  op(2)%v_grid => vgrido
  op(2)%t_ref = trefo
  op(2)%delay = interval

  t = t_start   ! Use non-zero initial time to avoid singularity
  u = maxval(vgridi)
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(5,eff,kappa)

  call do_main_loop
  call finalize_outputs(5,kappa)

 end subroutine lightcurve_explosion_explosion

 subroutine lightcurve_explosion_bpl(rho_input, vinput, t_ref, inner_slope, outer_slope, Mexp, Eexp, interval, eff, kappa)

! purpose: To compute LC for arbitrary CSM (outer) interacting with BPL SN (inner)
! Naming: explosion_bpl means CSM is arbitrary, SN is bpl
! op(1) = BPL SN (delay=0, explodes at t=0)
! op(2) = arbitrary CSM (delay=interval, pre-existing)

  use get_vals
  use constants,only:intpol

  real(8),intent(in):: t_ref, inner_slope, outer_slope, Mexp, Eexp, interval
  real(8),intent(in),target:: rho_input(:), vinput(:)
  real(8),intent(in),optional:: eff, kappa

  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_explosion
  v_out => v_explosion
  rho4pir2_in  => rho_bpl
  rho4pir2_out => rho_explosion

  op(1)%scan_i = -1
  op(2)%scan_i =  1

  op(1)%Mej = Mexp
  op(1)%Eej = Eexp
  op(1)%bpl_d = inner_slope
  op(1)%bpl_n = outer_slope
  op(1)%delay = 0d0
  op(2)%rho_expl => rho_input
  op(2)%v_grid => vinput
  op(2)%t_ref = t_ref
  op(2)%delay = interval

  call get_bpl_coeffs(op(1))

  t = t_start   ! Use non-zero initial time to avoid singularity
  u = initial_bpl_shell_velocity(op(1))
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(5,eff,kappa)

  call do_main_loop
  call finalize_outputs(5,kappa)

 end subroutine lightcurve_explosion_bpl

 subroutine lightcurve_static_bpl(rho_input, rinput, inner_slope, outer_slope, Mexp, Eexp, eff, kappa)

! purpose: To compute LC for static arbitrary CSM (outer) interacting with BPL SN (inner)

  use get_vals

  real(8),intent(in):: inner_slope, outer_slope, Mexp, Eexp
  real(8),intent(in),target:: rho_input(:), rinput(:)
  real(8),intent(in),optional:: eff, kappa

  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_explosion
  v_out => v_static
  rho4pir2_in  => rho_bpl
  rho4pir2_out => rho_static_profile

  op(1)%scan_i = -1
  op(2)%scan_i =  1

  op(1)%Mej = Mexp
  op(1)%Eej = Eexp
  op(1)%bpl_d = inner_slope
  op(1)%bpl_n = outer_slope
  op(1)%delay = 0d0
  op(2)%rho_static => rho_input
  op(2)%r_grid_static => rinput

  call get_bpl_coeffs(op(1))

  t = t_start
  u = initial_bpl_shell_velocity(op(1))
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(5,eff,kappa)

  call do_main_loop
  call finalize_outputs(5,kappa)

 end subroutine lightcurve_static_bpl

! ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 subroutine lightcurve_bpl_exponential(inner_slope, outer_slope, Mexp, Eexp, Mexp_out, Eexp_out, interval, eff, kappa)

! purpose: To compute LC for BPL CSM (outer) interacting with exponential SN (inner)
 ! Naming: bpl_exponential means CSM is BPL, SN is exponential
! op(1) = exponential SN (delay=0, explodes at t=0)
! op(2) = BPL CSM (delay=interval, pre-existing)

  use get_vals

  real(8),intent(in):: inner_slope, outer_slope, Mexp, Eexp, Mexp_out, Eexp_out, interval
  real(8),intent(in),optional:: eff, kappa

  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_explosion
  v_out => v_explosion
  rho4pir2_in  => rho_exponential
  rho4pir2_out => rho_bpl

  op(1)%Mej = Mexp_out
  op(1)%Eej = Eexp_out
  op(1)%delay = 0d0
  op(2)%Mej = Mexp
  op(2)%Eej = Eexp
  op(2)%bpl_d = inner_slope
  op(2)%bpl_n = outer_slope
  op(2)%delay = interval

  call get_exp_v0(op(1))
  call get_bpl_coeffs(op(2))

  t = t_start   ! Use non-zero initial time to avoid singularity
  u = initial_exp_shell_velocity(op(1))
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(3,eff,kappa)

  call do_main_loop
  call finalize_outputs(3,kappa)

 end subroutine lightcurve_bpl_exponential

! ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 subroutine lightcurve_exponential_explosion(Mexp, Eexp, rho_input, vinput, t_ref, interval, eff, kappa)

! purpose: To compute LC for exponential CSM (outer) interacting with arbitrary explosion SN (inner)
! Naming: exponential_explosion means CSM is exponential, SN is arbitrary
! op(1) = arbitrary explosion SN (delay=0, explodes at t=0)
! op(2) = exponential CSM (delay=interval, pre-existing)

  use get_vals

  real(8),intent(in):: Mexp, Eexp, t_ref, interval
  real(8),intent(in),target:: rho_input(:), vinput(:)
  real(8),intent(in),optional:: eff, kappa

  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_explosion
  v_out => v_explosion
  rho4pir2_in  => rho_explosion
  rho4pir2_out => rho_exponential

  op(1)%scan_i = -size(vinput)
  op(2)%scan_i = 1

  op(1)%rho_expl => rho_input
  op(1)%v_grid => vinput
  op(1)%t_ref = t_ref
  op(1)%delay = 0d0
  op(2)%Mej = Mexp
  op(2)%Eej = Eexp
  op(2)%delay = interval

  call get_exp_v0(op(2))

  t = t_start   ! Use non-zero initial time to avoid singularity
  u = maxval(vinput)
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(4,eff,kappa)

  call do_main_loop
  call finalize_outputs(4,kappa)

 end subroutine lightcurve_exponential_explosion

! ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 subroutine lightcurve_explosion_exponential(rho_input, vinput, t_ref, Mexp, Eexp, interval, eff, kappa)

! purpose: To compute LC for arbitrary CSM (outer) interacting with exponential SN (inner)
! Naming: explosion_exponential means CSM is arbitrary, SN is exponential
! op(1) = exponential SN (delay=0, explodes at t=0)
! op(2) = arbitrary CSM (delay=interval, pre-existing)

  use get_vals

  real(8),intent(in):: Mexp, Eexp, t_ref, interval
  real(8),intent(in),target:: rho_input(:), vinput(:)
  real(8),intent(in),optional:: eff, kappa

  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_explosion
  v_out => v_explosion
  rho4pir2_in  => rho_exponential
  rho4pir2_out => rho_explosion

  op(1)%scan_i = -1
  op(2)%scan_i = 1

  op(1)%Mej = Mexp
  op(1)%Eej = Eexp
  op(1)%delay = 0d0
  op(2)%rho_expl => rho_input
  op(2)%v_grid => vinput
  op(2)%t_ref = t_ref
  op(2)%delay = interval

  call get_exp_v0(op(1))

  t = t_start   ! Use non-zero initial time to avoid singularity
  u = initial_exp_shell_velocity(op(1))
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(1,eff,kappa)

  call do_main_loop
  call finalize_outputs(1,kappa)

 end subroutine lightcurve_explosion_exponential

 subroutine lightcurve_static_exponential(rho_input, rinput, Mexp, Eexp, eff, kappa)

! purpose: To compute LC for static arbitrary CSM (outer) interacting with exponential SN (inner)

  use get_vals

  real(8),intent(in):: Mexp, Eexp
  real(8),intent(in),target:: rho_input(:), rinput(:)
  real(8),intent(in),optional:: eff, kappa

  call reset_outflow(op(1))
  call reset_outflow(op(2))

  v_in  => v_explosion
  v_out => v_static
  rho4pir2_in  => rho_exponential
  rho4pir2_out => rho_static_profile

  op(1)%scan_i = -1
  op(2)%scan_i = 1

  op(1)%Mej = Mexp
  op(1)%Eej = Eexp
  op(1)%delay = 0d0
  op(2)%rho_static => rho_input
  op(2)%r_grid_static => rinput

  call get_exp_v0(op(1))

  t = t_start
  u = initial_exp_shell_velocity(op(1))
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(1,eff,kappa)

  call do_main_loop
  call finalize_outputs(1,kappa)

 end subroutine lightcurve_static_exponential

! ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 subroutine do_main_loop

  use integration
  use get_vals

   integer:: n, m3_log_counter, n_store_seen
   real(8):: dt,ku,kr,km, t_end_run
   real(8):: lum_fs, lum_rs, lum_heat, lum_store, r_store, eta_fs, eta_rs
   real(8):: tau_ahead, tau_therm, therm_floor, therm_factor, escape_factor
   real(8):: simple_t_last_store, simple_l_last_store, simple_r_last_store
   real(8):: simple_dlogt, simple_rell, simple_relr
   real(8) :: r_ph, L_ph
   character(len=32) :: m3_debug_env
   integer :: m3_debug_status
   logical :: mode3_debug, simple_dense_output

  call reset_dimless_state(dl_state_global)
   if(run_mode == 3)then
    t_end_run = min(t_end, 300d0*86400d0)
   else
    t_end_run = t_end
   end if
  
  L_ph = 0.0d0
  r_ph = 0.0d0
  m3_log_counter = 0
  mode3_debug = .false.
  m3_debug_env = ''
  m3_debug_status = 1
  call get_environment_variable('REDBACK_CSM_TRANSPORT_DEBUG', m3_debug_env, status=m3_debug_status)
  if (m3_debug_status == 0) then
   mode3_debug = len_trim(m3_debug_env) > 0 .and. m3_debug_env(1:1) /= '0' .and. m3_debug_env(1:1) /= 'f' .and. &
                 m3_debug_env(1:1) /= 'F' .and. m3_debug_env(1:1) /= 'n' .and. m3_debug_env(1:1) /= 'N'
  end if
  simple_dense_output = .false.
  m3_debug_env = ''
  m3_debug_status = 1
  call get_environment_variable('REDBACK_CSM_SIMPLE_DENSE_OUTPUT', m3_debug_env, status=m3_debug_status)
  if (m3_debug_status == 0) then
   simple_dense_output = len_trim(m3_debug_env) > 0 .and. m3_debug_env(1:1) /= '0' .and. &
                         m3_debug_env(1:1) /= 'f' .and. m3_debug_env(1:1) /= 'F' .and. &
                         m3_debug_env(1:1) /= 'n' .and. m3_debug_env(1:1) /= 'N'
  end if

  if (run_mode == 3) then
   ! Run mode 3 uses the Appendix-A dimensionless interaction solve so the
   ! active diffusion column is the paper's shock-to-photosphere domain.  The
   ! same state also tracks the swept shocked-shell e_int(x) used at cooling
   ! handoff.
   call initialize_dimless_state(dl_state_global, opacity_const_global, eff_global, n_rad_zones_global)
   call dimless_to_cgs(dl_state_global)
   r = dl_state_global%r_sh_cgs
   u = dl_state_global%v_sh_cgs
   m = max(dl_state_global%m_sh_cgs, 1d-30)
   t = dl_state_global%t_cgs
  end if

  t_array = -1d0
  ld_array = 0d0
  fs_array = 0d0
  rs_array = 0d0
  rfs_array = 0d0
  rph_array = 0d0
  etrap_array = 0d0
  tleak_array = 0d0
  tau_array_transport = 0d0
  i_array = 0
  n = 0
  n_store_seen = 0
  simple_t_last_store = -1d0
  simple_l_last_store = 0d0
  simple_r_last_store = 0d0
  do while (t<=t_end_run)

! Evolve shell properties
   if (run_mode == 3) then
    call update_dimless_shock_luminosities(dl_state_global)
    ku = 0d0
    kr = max(dl_state_global%v_sh_cgs, 0d0)
    km = 0d0
    lum_fs = dl_state_global%lum_heat_fs_cgs
    lum_rs = dl_state_global%lum_heat_rs_cgs
    lum_heat = dl_state_global%lum_heat_total_cgs
   else
    call shell_rhs_and_luminosity(u, r, m, t, op, ku, kr, km, lum_fs, lum_rs)
    select case (shock_efficiency_mode)
    case (1)
     eta_fs = forward_shock_radiative_efficiency(r,t,u,op,eff_global)
     eta_rs = reverse_shock_radiative_efficiency(r,t,u,op,eff_global)
     if (diffusion_enabled) then
      tau_ahead = shell_optical_depth(r, t)
      tau_therm = 7d0
      therm_floor = 0.25d0
      therm_factor = therm_floor + (1d0 - therm_floor) / &
                    (1d0 + (tau_therm / max(tau_ahead, 1d-30))**12)
      escape_factor = 1d0 - exp(-min((tau_ahead / 0.10d0)**2, 80d0))
      therm_factor = therm_factor * escape_factor
      eta_fs = eta_fs * therm_factor
      eta_rs = eta_rs * therm_factor
     end if
     lum_fs = eta_fs*lum_fs
     lum_rs = eta_rs*lum_rs
    case default
     lum_fs = eff_global*lum_fs
     lum_rs = eff_global*lum_rs
    end select
    lum_heat = lum_fs + lum_rs
   end if

! Adaptively adjust time stepping so that shell changes are resolved to ~1%
   if(run_mode == 3)then
    ! Use adaptive global cadence for output/driver stepping.
    ! The transport solver subcycles internally, but if this outer step is too
    ! coarse (e.g. fixed 1 day) fast dark-phase/rise features are lost by
    ! interpolation onto user times.
    dt = dimless_dynamics_timescale_cgs(dl_state_global)
    dt = 0.02d0*dt

    ! Keep the compact-CSM peak finely sampled, but avoid writing thousands of
    ! redundant dark-phase and late-tail points.  The transport step below
    ! still subcycles internally when the diffusion/handoff problem requires it.
    if (t < 4d0*86400d0) then
     dt = min(dt, 0.05d0*86400d0)
    else if (t < 10d0*86400d0) then
     dt = min(dt, 0.05d0*86400d0)
    else if (t < 30d0*86400d0) then
     dt = min(dt, 0.15d0*86400d0)
    else if (t < 120d0*86400d0) then
     dt = min(dt, 0.50d0*86400d0)
    else
     dt = min(dt, 1.00d0*86400d0)
    end if

    dt = max(dt, 10d0)              ! avoid zero/underflow
    if (t + dt > t_end_run) dt = t_end_run - t
   if (.not.(dt>0d0.and.dt<huge(1d0))) exit
   else if(run_mode == 1)then
    dt = 0.01d0*min(abs(u/ku),abs(r/kr),abs(m/km))
   end if

   if (run_mode /= 3) then
    u = u + dt*ku
    r = r + dt*kr
    m = m + dt*km
   end if

    if(run_mode == 1 .or. .not.diffusion_enabled)then
     erad = 0d0
    end if

   if (run_mode /= 3) t = t + dt

   if (run_mode == 3 .and. diffusion_enabled) then
    m3_log_counter = m3_log_counter + 1
    if (mode3_debug .and. (m3_log_counter <= 25 .or. mod(m3_log_counter, 200) == 0)) then
     write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
          'M3_LOOP_PRE t=', t, 'dt=', dt, 'r=', r, 'u=', u, 'Lheat=', lum_heat
    end if
    call dimless_comoving_transport_step(dl_state_global, dt, lum_heat, L_ph, r_ph)
    r = dl_state_global%r_sh_cgs
    u = dl_state_global%v_sh_cgs
    m = max(dl_state_global%m_sh_cgs, 1d-30)
    t = dl_state_global%t_cgs
    if (mode3_debug .and. (m3_log_counter <= 25 .or. mod(m3_log_counter, 200) == 0)) then
     write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,L1)') &
          'M3_LOOP_POST t=', t, 'L=', L_ph, 'r=', r, 'm=', m, 'cool=', dl_state_global%in_cooling_phase
    end if
    if (.not.(L_ph == L_ph) .or. abs(L_ph) > 1d250) L_ph = 0d0
   end if

! Store output arrays
   if((run_mode == 1 .and. t>1d4) .or. (run_mode == 3 .and. t>t_start))then
    if (n >= 1) then
     if (run_mode == 3) then
      if (t <= t_array(n) + 1d-6) cycle
     else
      if (t <= t_array(n) * (1.0d0 + 1.0d-10)) cycle
     end if
    end if
    if (run_mode == 1 .and. diffusion_enabled .and. .not.simple_dense_output .and. n >= 1 .and. t < t_end_run) then
     simple_dlogt = abs(log(max(t, 1d-99) / max(simple_t_last_store, 1d-99)))
     simple_rell = abs(lum_heat - simple_l_last_store) / max(max(abs(lum_heat), abs(simple_l_last_store)), 1d-99)
     simple_relr = abs(r - simple_r_last_store) / max(max(abs(r), abs(simple_r_last_store)), 1d-99)
     if (simple_dlogt < simple_store_min_dlogt .and. &
         simple_rell < simple_store_min_rell .and. &
         simple_relr < simple_store_min_relr) cycle
    end if
    if (run_mode == 3) then
     n_store_seen = n_store_seen + 1
     if (n_store_seen <= n_skip_initial_outputs) cycle
    end if
    if (n >= ll) exit
    n = n+1
    
    ! Determine what to output based on mode
     if(run_mode == 3)then
     ! TRANSPORT MODE: store shock power and emergent diffuse luminosity separately.
     lum_store = lum_heat
     ld_array(n) = L_ph
     r_store = r
     if (run_mode == 3 .and. dl_state_global%in_cooling_phase) then
      r_store = dl_state_global%x_out_cool * dl_state_global%R0 * dl_state_global%R_in_R0
     end if
     if(diffusion_enabled)then
       rfs_array(n) = r_store
       rph_array(n) = r_ph
       etrap_array(n) = 0d0
       tleak_array(n) = 0d0
      tau_array_transport(n) = dl_state_global%tau_ahead_csm
     else
      rfs_array(n) = r
      rph_array(n) = r_ph
      etrap_array(n) = erad
      tleak_array(n) = 0d0
      tau_array_transport(n) = 0d0
     end if
    else
     ! SIMPLE MODE: store shock luminosity; diffusion is applied later in finalize_outputs
     lum_store = lum_heat
     r_store = r
     ld_array(n) = 0d0
     rfs_array(n) = r
     rph_array(n) = r
     etrap_array(n) = 0d0
     tleak_array(n) = 0d0
      tau_array_transport(n) = 0d0
    end if
    
    t_array(n) = t
    if(run_mode == 1)i_array(n) = op(2)%scan_i
    l_array(n) = lum_store
    fs_array(n) = lum_fs
    rs_array(n) = lum_rs
    r_array(n) = r_store
    m_array(n) = m
    v_array(n) = u
    if (run_mode == 1 .and. diffusion_enabled) then
     simple_t_last_store = t
     simple_l_last_store = lum_heat
     simple_r_last_store = r
    end if
   end if

  end do

  if(allocated(tarray))deallocate(tarray)
  if(allocated(larray))deallocate(larray)
  if(allocated(temparray))deallocate(temparray)
  if(allocated(rarray))deallocate(rarray)
  if(allocated(varray))deallocate(varray)
  if(allocated(marray))deallocate(marray)
  if(allocated(ldiff))deallocate(ldiff)
  if(allocated(lfs))deallocate(lfs)
  if(allocated(lrs))deallocate(lrs)
  if(allocated(rfsarray))deallocate(rfsarray)
  if(allocated(rpharray))deallocate(rpharray)
  if(allocated(etraparray))deallocate(etraparray)
  if(allocated(tleakarray))deallocate(tleakarray)
  if(allocated(tauarray_transport))deallocate(tauarray_transport)
  allocate(tarray(n))
  allocate(larray,temparray,rarray,varray,marray,ldiff,lfs,lrs,rfsarray,rpharray,etraparray,tleakarray,tauarray_transport,mold=tarray)

  tarray(1:n) = t_array(1:n)
  larray(1:n) = l_array(1:n)
  lfs(1:n) = fs_array(1:n)
  lrs(1:n) = rs_array(1:n)
  rarray(1:n) = r_array(1:n)
  varray(1:n) = v_array(1:n)
  marray(1:n) = m_array(1:n)
  ldiff(1:n) = ld_array(1:n)
  rfsarray(1:n) = rfs_array(1:n)
  rpharray(1:n) = rph_array(1:n)
  etraparray(1:n) = etrap_array(1:n)
  tleakarray(1:n) = tleak_array(1:n)
  tauarray_transport(1:n) = tau_array_transport(1:n)
  if (run_mode == 3) then
   temparray(1:n) = temperature(ldiff(1:n), max(rpharray(1:n), rarray(1:n)))
  else
   temparray(1:n) = temperature(larray(1:n), rarray(1:n))
  end if

!!$  do n = 1, size(tarray)
!!$   print*,tarray(n),larray(n)
!!$  end do
  return
 end subroutine do_main_loop

 subroutine get_diffuse_lc(csm_type,kappa)
! purpose: To compute the light curve with diffusion effects in simplified form

  use constants,only:intpol
  use get_vals

  integer,intent(in):: csm_type
  real(8),intent(in):: kappa
  integer:: i,j,k,n
  real(8):: dldiff, tp, tadv, tloss, tdiff, mdj, mdjp,tj,tjp
  real(8),allocatable,dimension(:):: tauprep, tau

  select case(csm_type)
  case(1)
   call get_tauprep_explosion(tauprep)
  end select

  if(allocated(ldiff))deallocate(ldiff)
  allocate(tau,ldiff,mold=larray)
  ldiff = 0d0

  do i = 1, size(tarray)-1
   select case(csm_type)
  case(1)
    j = i_array(i)
    tp = tarray(i) + op(2)%delay
    tau(i) = intpol(rarray(i)/tp,op(2)%v_grid(j:j+1),tauprep(j:j+1))/tp**2
   case(2)
    j = i_array(i)
    n = size(op(2)%t_grid)
    if(j==n)then
     tau(i) = op(2)%mdot(n)*op(2)%vwind/rarray(i)
    else
     tau(i) = 0d0
     do k = n-1, j+1, -1
      if(op(2)%t_grid(k+1)-op(2)%t_grid(k)>1d-99)then
       tau(i) = tau(i) &
              + (op(2)%mdot(k+1)-op(2)%mdot(k))&
               /(op(2)%t_grid(k+1)-op(2)%t_grid(k))&
               *log((op(2)%t_grid(k+1)+tarray(i))&
                   /(op(2)%t_grid(k  )+tarray(i)))
      else
       tau(i) = tau(i) &
              + (op(2)%mdot(k+1)-op(2)%mdot(k))&
               /(op(2)%t_grid(k)+tarray(i))
      end if
     end do

     if(j==1.and.rarray(i)/op(2)%vwind-tarray(i)<op(2)%t_grid(1))then
      tj   = op(2)%t_grid(1)
      tjp  = op(2)%t_grid(2)
      mdj  = op(2)%mdot(1)
      mdjp = op(2)%mdot(2)
      if(tjp-tj>1d-99)then
       tau(i) = tau(i) &
              + (mdjp-mdj)/(tjp-tj)&
               *log((tjp+tarray(i))/(tj+tarray(i)))
      else
       tau(i) = tau(i) + (mdjp-mdj)/(tj+tarray(i))
      end if
      tau(i) = tau(i) &
             + mdj/(tj+tarray(i)) &
             + mdj*(op(2)%vwind/rarray(i)-1d0/(tj+tarray(i)))
     else
     tj   = op(2)%t_grid(j  )
     tjp  = op(2)%t_grid(j+1)
     mdj  = op(2)%mdot(j  )
     mdjp = op(2)%mdot(j+1)
      if(tjp-tj>1d-99)then
       tau(i) = tau(i) &
              + mdjp/(tjp+tarray(i))&
              + (mdj*(tjp+tarray(i))-mdjp*(tj+tarray(i)))&
               /(tjp-tj)*(op(2)%vwind/rarray(i)-1d0/(tjp+tarray(i)))&
              +(mdjp-mdj)/(tjp-tj)&
               *log((tjp+tarray(i))*op(2)%vwind/rarray(i))
      else
       ! A duplicate wind-history point represents a discontinuous jump rather
       ! than a finite radial interval.  It contributes no optical-depth width
       ! by itself; finite neighbouring intervals are handled by the loop above
       ! or by the scan index on the next timestep.
       tau(i) = tau(i)
      end if
     end if
    end if
    tau(i) = tau(i)/(4d0*pi*op(2)%vwind**2)
   case(3)
    tau(i) = bpl_tau_to_edge(rarray(i), tarray(i), op(2))
    if(tau(i)<0d0)then
     tp = tarray(i) + op(2)%delay
     if(rarray(i)/tp>=op(2)%bpl_vt)then
      tau(i) = op(2)%bpl_rho0*op(2)%bpl_vt/(4d0*pi*(op(2)%bpl_n-1d0)*tp**2) &
              *(tp*op(2)%bpl_vt/rarray(i))**(op(2)%bpl_n-1d0)
     else
      tau(i) = op(2)%bpl_rho0*op(2)%bpl_vt/(4d0*pi*tp**2) &
             * ( (1d0-(rarray(i)/(tp*op(2)%bpl_vt))**(1d0-op(2)%bpl_d)) &
                 /(1d0-op(2)%bpl_d) &
               + 1d0/(op(2)%bpl_n-1d0) )
     end if
    end if
   case(4)
    tp = tarray(i) + op(2)%delay
    tau(i) = op(2)%Mej/(8d0*pi*(op(2)%exp_v0*tp)**2)&
             *exp(-rarray(i)/(op(2)%exp_v0*tp))
   case(5)
    tau(i) = query_tau_to_edge(rarray(i), tarray(i), op(2), 1d0)
   end select
   tau(i) = kappa*tau(i)
   if(.not.(tau(i)==tau(i)) .or. abs(tau(i)) > 1d250)cycle

   tdiff = tau(i)*rarray(i)/clight
   tadv  = rarray(i)/varray(i)
   if(.not.(tdiff==tdiff) .or. .not.(tadv==tadv))cycle
   if(tdiff<=0d0.or.tadv<=0d0)cycle
   tloss = 1d0/(1d0/tdiff+1d0/tadv)
   if(.not.(tloss==tloss) .or. tloss<=0d0)cycle
   do j = i+1, size(tarray)
    dldiff = exp((tarray(i+1)-tarray(j))/tloss)&
           - exp((tarray(i  )-tarray(j))/tloss)
    ldiff(j) = ldiff(j)+larray(i)*dldiff*tloss/tdiff
    if(tarray(j)>tarray(i+1)+max(tdiff,tadv)*50d0)exit
   end do
  end do

  return
 end subroutine get_diffuse_lc

 subroutine get_dimless_state_debug(x_sh_out, x_csm_out_val, in_cooling, nsub_val)
  real(8), intent(out) :: x_sh_out, x_csm_out_val
  integer, intent(out) :: in_cooling, nsub_val
  x_sh_out = dl_state_global%x_sh
  x_csm_out_val = dl_state_global%x_csm_out
  if (dl_state_global%in_cooling_phase) then
   in_cooling = 1
  else
   in_cooling = 0
  end if
  nsub_val = dl_state_global%nsub_last
 end subroutine get_dimless_state_debug

end module lc_mod
