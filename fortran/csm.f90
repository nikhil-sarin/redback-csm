module lc_mod

 use constants,only:pi,year,clight,intpol,temperature
 use csm_runtime, only: diffusion_enabled, eff_global, erad, &
                        shock_efficiency_mode, &
                        configure_runtime, reset_diffusion_state, &
                        shell_optical_depth, photosphere_radius, &
                        solve_diffusion_step, &
                        set_shock_efficiency_mode_runtime => set_shock_efficiency_mode
use csm_transport, only: transport_state_type, reset_transport_state, &
                                   interaction_transport_step, shock_has_emerged, &
                                   initialize_cooling_state_from_interaction, &
                                   cooling_transport_step, transport_timestep_limit, &
                                   shock_motion_timestep_limit, find_transport_photosphere, &
                                   forward_shock_radius, shell_leakage_timescale, &
                                   total_radiation_energy

 implicit none

 real(8),allocatable,dimension(:),public:: tarray, Larray, temparray, rarray, varray, marray, ldiff, lfs, lrs
 real(8),allocatable,dimension(:),public:: rfsarray, rpharray, etraparray, tleakarray, tauarray_hybrid
 public:: lightcurve_wind_exponential, lightcurve_wind_bpl, &
          lightcurve_bpl_wind, lightcurve_exponential_wind, &
          lightcurve_wind_explosion, lightcurve_explosion_wind, &
          lightcurve_wind_wind, lightcurve_explosion_explosion, &
          lightcurve_bpl_bpl, lightcurve_exponential_exponential, &
          lightcurve_explosion_bpl, lightcurve_bpl_exponential, &
          lightcurve_exponential_explosion, lightcurve_explosion_exponential, &
          lightcurve_static_bpl, lightcurve_static_exponential, &
          set_model_mode, set_efficiency_mode, set_run_mode, set_hybrid_parameters, &
          set_bpl_cutoff_ratio

 private:: finalize_outputs, do_main_loop
 private:: get_diffuse_lc
 private
integer,parameter:: ll=200000
 real(8),dimension(ll):: t_array, L_array, ld_array, r_array, v_array, m_array, fs_array, rs_array
 real(8),dimension(ll):: rfs_array, rph_array, etrap_array, tleak_array, tauhyb_array
 integer,dimension(ll):: i_array
 real(8):: t_start=1d1, t_end=10d0*year
 real(8):: u,r,m,t

 ! Run mode: 1=simple, 2=hybrid
 integer :: run_mode = 1
 
 ! Hybrid mode parameters
 integer :: n_rad_zones_global = 40
 real(8) :: opacity_const_global = 0.34d0

contains

 subroutine set_model_mode(mode)
  integer,intent(in):: mode

  if(mode==1)then
   run_mode = 1
  else
   run_mode = 2
  end if
 end subroutine set_model_mode

 subroutine set_efficiency_mode(mode)
  integer,intent(in):: mode

  call set_shock_efficiency_mode_runtime(mode)
 end subroutine set_efficiency_mode

 subroutine set_run_mode(mode)
  integer, intent(in) :: mode
  run_mode = mode
  if (mode /= 1 .and. mode /= 2) then
   print *, 'WARNING: Invalid run_mode', mode, 'using simple (1)'
   run_mode = 1
  endif
 end subroutine set_run_mode

 subroutine set_hybrid_parameters(n_zones, kappa_val)
  integer, intent(in), optional :: n_zones
  real(8), intent(in), optional :: kappa_val
  
  if (present(n_zones)) n_rad_zones_global = n_zones
  if (present(kappa_val)) opacity_const_global = kappa_val
 end subroutine set_hybrid_parameters

 subroutine set_bpl_cutoff_ratio(ratio)
  use get_vals, only: set_global_bpl_vmax_ratio
  real(8), intent(in) :: ratio
  call set_global_bpl_vmax_ratio(ratio)
 end subroutine set_bpl_cutoff_ratio

 subroutine finalize_outputs(csm_type, eff, kappa)
  integer,intent(in):: csm_type
  real(8),intent(in),optional:: eff, kappa

  if(run_mode == 1)then
   if(present(kappa))then
    call get_diffuse_lc(csm_type,kappa)
    temparray = temperature(ldiff,rarray)
   elseif(allocated(ldiff))then
    deallocate(ldiff)
   end if
  end if
 end subroutine finalize_outputs

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
  u = max(1.d2*op(1)%exp_v0,op(2)%vwind)
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(2,eff,kappa)

  call do_main_loop
  call finalize_outputs(2,eff,kappa)

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
  call finalize_outputs(2,eff,kappa)

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
  u = max(1.d2*op(1)%bpl_vt,op(2)%vwind)
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(2,eff,kappa)

  call do_main_loop
  call finalize_outputs(2,eff,kappa)

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
  call finalize_outputs(3,eff,kappa)

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
  call finalize_outputs(4,eff,kappa)

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
  call finalize_outputs(5,eff,kappa)

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
  call finalize_outputs(2,eff,kappa)

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
  u = 1.d2*op(1)%bpl_vt
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(3,eff,kappa)

  call do_main_loop
  call finalize_outputs(3,eff,kappa)

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
  u = 1.d2*op(1)%exp_v0
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(4,eff,kappa)

  call do_main_loop
  call finalize_outputs(4,eff,kappa)

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
  call finalize_outputs(5,eff,kappa)

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
  u = 1.d2*op(1)%bpl_vt
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(5,eff,kappa)

  call do_main_loop
  call finalize_outputs(5,eff,kappa)

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
  u = 1.d2*op(1)%bpl_vt
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(5,eff,kappa)

  call do_main_loop
  call finalize_outputs(5,eff,kappa)

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
  u = 1.d2*op(1)%exp_v0
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(3,eff,kappa)

  call do_main_loop
  call finalize_outputs(3,eff,kappa)

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
  call finalize_outputs(4,eff,kappa)

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
  u = 1.d2*op(1)%exp_v0
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(1,eff,kappa)

  call do_main_loop
  call finalize_outputs(1,eff,kappa)

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
  u = 1.d2*op(1)%exp_v0
  r = u*t*1.2d0
  m = 1d0
  erad = 0d0
  call configure_runtime(1,eff,kappa)

  call do_main_loop
  call finalize_outputs(1,eff,kappa)

 end subroutine lightcurve_static_exponential

! ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 subroutine do_main_loop

  use integration
  use get_vals

	  integer:: n, nsub, isub, nsub_cool, jsub
	  real(8):: dt,ku,kr,km, t_end_run
	  real(8):: lum_fs, lum_rs, lum_heat, lum_store, r_store, eta_fs, eta_rs, lum_heat_sub
	  real(8) :: r_ph, L_ph, dt_rad, dt_sub, dt_remain, dt_cool, dt_cool_cap, dt_int_cap, dt_move
	  real(8) :: t_old, r_old, u_old, m_old, lum_heat_old, r_sub, m_sub, t_sub
	  real(8) :: t_sub_prev, r_sub_prev, m_sub_prev, r_out_prev, r_out_sub, f_emerge
	  real(8) :: r_in_sub, shell_span_sub, gap_to_edge, u_sub, u_emerge, lum_heat_emerge
   real(8) :: shell_ratio_run, lum_heat_cool
   real(8) :: compact_breakout_lum, compact_blend_duration, compact_blend_weight
   logical :: compact_breakout_mode
   logical :: compact_breakout_ready
   real(8) :: r_fs_prev, r_fs_sub
   type(transport_state_type) :: tr_state

  call reset_transport_state(tr_state)
  if(run_mode == 2)then
   t_end_run = min(t_end, 300d0*86400d0)
  else
   t_end_run = t_end
  end if
  
  L_ph = 0.0d0
  r_ph = 0.0d0
  compact_breakout_mode = .false.
  compact_breakout_ready = .false.
  compact_breakout_lum = 0d0
  compact_blend_duration = 0.75d0 * 86400d0
  shell_ratio_run = huge(1d0)
  if (run_mode == 2) then
   shell_ratio_run = query_csm_outer_edge(t, op(2)) / max(query_csm_inner_edge(t, op(2)), 1d-30)
   compact_breakout_mode = (shell_ratio_run <= 20d0)
  end if

  t_array = -1d0
  ld_array = 0d0
  fs_array = 0d0
  rs_array = 0d0
  rfs_array = 0d0
  rph_array = 0d0
  etrap_array = 0d0
  tleak_array = 0d0
  tauhyb_array = 0d0
  i_array = 0
  n = 0
  do while (t<=t_end_run)

! Evolve shell properties
	   ku = dudt(u,r,m,t,op)
	   kr = drdt(u,r,m,t,op)
	   km = dmdt(u,r,m,t,op)
	   lum_fs = forward_shock_luminosity(r,t,u,op)
	   lum_rs = reverse_shock_luminosity(r,t,u,op)
	   select case (shock_efficiency_mode)
	   case (1)
	    eta_fs = forward_shock_radiative_efficiency(r,t,u,op,eff_global)
	    eta_rs = reverse_shock_radiative_efficiency(r,t,u,op,eff_global)
	    lum_fs = eta_fs*lum_fs
	    lum_rs = eta_rs*lum_rs
	   case default
	    lum_fs = eff_global*lum_fs
	    lum_rs = eff_global*lum_rs
	   end select
	   lum_heat = lum_fs + lum_rs

! Adaptively adjust time stepping so that shell changes are resolved to ~1%
   if(run_mode == 1)then
    dt = 0.01d0*min(abs(u/ku),abs(r/kr),abs(m/km))
   else
    dt = huge(1d0)
    if(abs(ku)>1d-30)dt = min(dt,abs(u/ku))
    if(abs(kr)>1d-30)dt = min(dt,abs(r/kr))
    if(abs(km)>1d-30)dt = min(dt,abs(m/km))
   dt = 0.01d0*dt
    if(.not.(dt>0d0.and.dt<huge(1d0)))exit
   end if

	   t_old = t
	   r_old = r
	   u_old = u
	   m_old = m
	   lum_heat_old = lum_heat

   u = u + dt*ku
   r = r + dt*kr
   if(run_mode == 1)then
    m = m + dt*km
   else
    m = max(m + dt*km,1d-30)
   end if

	   if(run_mode == 1 .or. .not.diffusion_enabled)then
	    erad = 0d0
	   end if

   t = t + dt

   if (run_mode == 2) then
     if(diffusion_enabled)then
      if(.not.tr_state%initialized)then
       tr_state%kappa = opacity_const_global
       tr_state%n_zones = n_rad_zones_global
       nsub = 1
      else
       dt_rad = transport_timestep_limit(tr_state)
       if(dt_rad > 0d0 .and. dt_rad < huge(1d0))then
        nsub = min(128, max(1, ceiling(dt / max(dt_rad, 1d-30))))
       else
        nsub = 1
       end if
       dt_move = shock_motion_timestep_limit(tr_state, u_old)
       if(dt_move > 0d0 .and. dt_move < huge(1d0))then
        nsub = min(256, max(nsub, ceiling(dt / max(dt_move, 1d-30))))
       end if
       dt_int_cap = huge(1d0)
       if(.not.tr_state%in_cooling_phase)then
        if(tr_state%initialized .and. tr_state%r_outer_support > tr_state%r_inner)then
         r_out_sub = tr_state%r_outer_support
         r_in_sub = tr_state%r_inner
        else
         r_out_sub = query_csm_outer_edge(t, op(2))
         r_in_sub = query_csm_inner_edge(t, op(2))
        end if
        shell_span_sub = max(r_out_sub - r_in_sub, 1d-30)
        gap_to_edge = r_out_sub - r
        if(gap_to_edge > 0d0 .and. gap_to_edge < 0.1d0*shell_span_sub)then
         dt_int_cap = 0.01d0*86400d0
         if(gap_to_edge < 0.05d0*shell_span_sub) dt_int_cap = 0.004d0*86400d0
        end if
       end if
       if(dt_int_cap < huge(1d0))then
        nsub = min(256, max(nsub, ceiling(dt / dt_int_cap)))
       end if
      end if
      dt_sub = dt / dble(max(nsub,1))
      do isub = 1, nsub
       t_sub_prev = t_old + dble(isub-1) * dt_sub
       r_sub_prev = r_old + (r - r_old) * dble(isub-1) / dble(max(nsub,1))
       m_sub_prev = m_old + (m - m_old) * dble(isub-1) / dble(max(nsub,1))
       t_sub = t_old + dble(isub) * dt_sub
       r_sub = r_old + (r - r_old) * dble(isub) / dble(max(nsub,1))
       u_sub = u_old + (u - u_old) * (dble(isub) - 0.5d0) / dble(max(nsub,1))
       m_sub = m_old + (m - m_old) * dble(isub) / dble(max(nsub,1))
       r_fs_sub = r_sub
       lum_heat_sub = lum_heat_old + (lum_heat - lum_heat_old) * &
            (dble(isub) - 0.5d0) / dble(max(nsub,1))
       lum_heat_sub = max(lum_heat_sub, 0d0)
       if(.not.tr_state%in_cooling_phase)then
	        if(tr_state%initialized .and. tr_state%r_outer_support > tr_state%r_inner)then
	         r_out_prev = tr_state%r_outer_support
	         r_out_sub = tr_state%r_outer_support
	        else
	         r_out_prev = query_csm_outer_edge(t_sub_prev, op(2))
	         r_out_sub = query_csm_outer_edge(t_sub, op(2))
	        end if
            r_fs_prev = forward_shock_radius(tr_state, r_sub_prev, t_sub_prev, m_sub_prev)
            r_fs_sub = forward_shock_radius(tr_state, r_sub, t_sub, m_sub)
	        if(r_fs_sub >= r_out_sub)then
	         if(r_fs_prev < r_out_prev)then
	         f_emerge = (r_out_prev - r_fs_prev) / &
	              max((r_fs_sub - r_fs_prev) - &
	                   (r_out_sub - r_out_prev), 1d-30)
	         f_emerge = min(max(f_emerge, 0d0), 1d0)
         u_emerge = u_old + (u - u_old) * (dble(isub-1) + f_emerge) / dble(max(nsub,1))
         lum_heat_emerge = lum_heat_old + (lum_heat - lum_heat_old) * &
              (dble(isub-1) + f_emerge) / dble(max(nsub,1))
         lum_heat_emerge = max(lum_heat_emerge, 0d0)
          if(f_emerge > 0d0)then
           call interaction_transport_step(tr_state, f_emerge*dt_sub, &
                r_sub_prev + f_emerge*(r_sub-r_sub_prev), u_emerge, &
                t_sub_prev + f_emerge*(t_sub-t_sub_prev), &
                m_sub_prev + f_emerge*(m_sub-m_sub_prev), lum_heat_emerge, L_ph, r_ph)
          end if
          call initialize_cooling_state_from_interaction(tr_state, &
               r_sub_prev + f_emerge*(r_sub-r_sub_prev), u_emerge, &
               m_sub_prev + f_emerge*(m_sub-m_sub_prev), &
               t_sub_prev + f_emerge*(t_sub-t_sub_prev), L_ph)
          dt_remain = (1d0-f_emerge)*dt_sub
          if(dt_remain > 0d0)then
           dt_rad = transport_timestep_limit(tr_state)
           if(dt_rad > 0d0 .and. dt_rad < huge(1d0))then
            nsub_cool = min(32, max(1, ceiling(dt_remain / max(0.5d0*dt_rad, 1d-30))))
           else
            nsub_cool = 1
           end if
           dt_cool_cap = huge(1d0)
           if (tr_state%t_emerge > 0d0) then
           if ((t_sub_prev + f_emerge*(t_sub-t_sub_prev)) - tr_state%t_emerge < 0.05d0*86400d0) then
             dt_cool_cap = 0.004d0*86400d0
            else if ((t_sub_prev + f_emerge*(t_sub-t_sub_prev)) - tr_state%t_emerge < 0.1d0*86400d0) then
             dt_cool_cap = 0.01d0*86400d0
            end if
           end if
           if (dt_cool_cap < huge(1d0)) then
            nsub_cool = min(128, max(nsub_cool, ceiling(dt_remain / dt_cool_cap)))
           end if
           dt_cool = dt_remain / dble(max(nsub_cool,1))
           do jsub = 1, nsub_cool
            call cooling_transport_step(tr_state, dt_cool, &
                 t_sub_prev + f_emerge*(t_sub-t_sub_prev) + dble(jsub)*dt_cool, L_ph, r_ph)
            if (compact_breakout_ready .and. compact_breakout_lum > 0d0) then
             compact_blend_weight = min(max(((t_sub_prev + f_emerge*(t_sub-t_sub_prev) + dble(jsub)*dt_cool) - &
                                   tr_state%t_emerge) / max(compact_blend_duration, 1d-30), 0d0), 1d0)
             L_ph = (1d0 - compact_blend_weight) * compact_breakout_lum + compact_blend_weight * L_ph
            end if
           end do
          else
           call find_transport_photosphere(tr_state, r_ph, L_ph)
          end if
       else
        dt_cool_cap = huge(1d0)
        if (tr_state%t_emerge > 0d0) then
         if (t_sub - tr_state%t_emerge < 0.05d0*86400d0) then
          dt_cool_cap = 0.004d0*86400d0
         else if (t_sub - tr_state%t_emerge < 0.1d0*86400d0) then
          dt_cool_cap = 0.01d0*86400d0
         end if
        end if
        if (dt_cool_cap < huge(1d0) .and. dt_sub > dt_cool_cap) then
         nsub_cool = min(128, max(1, ceiling(dt_sub / dt_cool_cap)))
         dt_cool = dt_sub / dble(max(nsub_cool,1))
         do jsub = 1, nsub_cool
          lum_heat_cool = 0d0
          if (compact_breakout_ready) then
           r_fs_sub = forward_shock_radius(tr_state, r_sub, t_sub_prev + dble(jsub)*dt_cool, m_sub)
           if (tr_state%r_outer_support > tr_state%r_inner) then
            r_out_sub = tr_state%r_outer_support
           else
            r_out_sub = query_csm_outer_edge(t_sub_prev + dble(jsub)*dt_cool, op(2))
           end if
           if (r_fs_sub < r_out_sub) lum_heat_cool = lum_heat_sub
          end if
          call cooling_transport_step(tr_state, dt_cool, t_sub_prev + dble(jsub)*dt_cool, L_ph, r_ph, lum_heat_cool)
          if (compact_breakout_ready .and. compact_breakout_lum > 0d0) then
           compact_blend_weight = min(max(((t_sub_prev + dble(jsub)*dt_cool) - tr_state%t_emerge) / &
                                 max(compact_blend_duration, 1d-30), 0d0), 1d0)
           L_ph = (1d0 - compact_blend_weight) * compact_breakout_lum + compact_blend_weight * L_ph
          end if
         end do
        else
         lum_heat_cool = 0d0
         if (compact_breakout_ready) then
          r_fs_sub = forward_shock_radius(tr_state, r_sub, t_sub, m_sub)
          if (tr_state%r_outer_support > tr_state%r_inner) then
           r_out_sub = tr_state%r_outer_support
          else
           r_out_sub = query_csm_outer_edge(t_sub, op(2))
          end if
          if (r_fs_sub < r_out_sub) lum_heat_cool = lum_heat_sub
         end if
         call cooling_transport_step(tr_state, dt_sub, t_sub, L_ph, r_ph, lum_heat_cool)
         if (compact_breakout_ready .and. compact_breakout_lum > 0d0) then
          compact_blend_weight = min(max((t_sub - tr_state%t_emerge) / max(compact_blend_duration, 1d-30), 0d0), 1d0)
          L_ph = (1d0 - compact_blend_weight) * compact_breakout_lum + compact_blend_weight * L_ph
         end if
        end if
       end if
       else
         r_fs_prev = forward_shock_radius(tr_state, r_sub_prev, t_sub_prev, m_sub_prev)
         call interaction_transport_step(tr_state, dt_sub, r_sub, u_sub, t_sub, m_sub, lum_heat_sub, L_ph, r_ph)
         if (compact_breakout_mode .and. .not.compact_breakout_ready) then
          r_fs_sub = forward_shock_radius(tr_state, r_sub, t_sub, m_sub)
          if (r_ph > 0d0 .and. &
              r_fs_prev < (1d0 - 1d0/dble(max(tr_state%n_zones, 8))) * r_ph .and. &
              r_fs_sub >= (1d0 - 1d0/dble(max(tr_state%n_zones, 8))) * r_ph) then
           compact_breakout_ready = .true.
           compact_breakout_lum = max(L_ph, 0d0)
           call initialize_cooling_state_from_interaction(tr_state, r_sub, u_sub, m_sub, t_sub, L_ph)
           call find_transport_photosphere(tr_state, r_ph, L_ph)
           if (compact_breakout_lum > 0d0) L_ph = compact_breakout_lum
          end if
         end if
        end if
       else
        call cooling_transport_step(tr_state, dt_sub, t_sub, L_ph, r_ph)
       end if

       if(t_sub > t_start)then
        if (n < 1) then
         if (n >= ll) exit
         n = n + 1
         ld_array(n) = L_ph
         t_array(n) = t_sub
         l_array(n) = lum_heat_sub
         fs_array(n) = lum_fs
         rs_array(n) = lum_rs
         r_array(n) = r_sub
         m_array(n) = m_sub
         v_array(n) = u
         rfs_array(n) = r_fs_sub
         rph_array(n) = r_ph
         etrap_array(n) = total_radiation_energy(tr_state)
         tleak_array(n) = shell_leakage_timescale(tr_state, r_sub, t_sub, m_sub)
         tauhyb_array(n) = shell_optical_depth(r_sub, t_sub)
        else if (t_sub > t_array(n) * (1.0d0 + 1.0d-3)) then
         if (n >= ll) exit
         n = n + 1
         ld_array(n) = L_ph
         t_array(n) = t_sub
         l_array(n) = lum_heat_sub
         fs_array(n) = lum_fs
         rs_array(n) = lum_rs
         r_array(n) = r_sub
         m_array(n) = m_sub
         v_array(n) = u
         rfs_array(n) = r_fs_sub
         rph_array(n) = r_ph
         etrap_array(n) = total_radiation_energy(tr_state)
         tleak_array(n) = shell_leakage_timescale(tr_state, r_sub, t_sub, m_sub)
         tauhyb_array(n) = shell_optical_depth(r_sub, t_sub)
        end if
       end if
      end do
      if (n >= ll) exit
     else
      L_ph = lum_heat
      r_ph = r
     end if
   endif

! Store output arrays
   if((run_mode == 1 .and. t>1d4) .or. (run_mode == 2 .and. t>t_start .and. .not.diffusion_enabled))then
    if (n >= 1) then
      if (run_mode == 2) then
        if (t <= t_array(n) * (1.0d0 + 1.0d-3)) cycle
      else
        if (t <= t_array(n) * (1.0d0 + 1.0d-10)) cycle
      end if
    end if
   if(run_mode == 1)then
    if(n>=1)i_array(n) = op(2)%scan_i
   end if
   if (n >= ll) exit
    n = n+1
    
    ! Determine what to output based on mode
     if(run_mode == 2)then
     ! HYBRID MODE: store shock power and emergent diffuse luminosity separately.
     lum_store = lum_heat
     ld_array(n) = L_ph
     r_store = r
     if(diffusion_enabled)then
      rfs_array(n) = forward_shock_radius(tr_state, r, t, m)
      rph_array(n) = r_ph
      etrap_array(n) = total_radiation_energy(tr_state)
      tleak_array(n) = shell_leakage_timescale(tr_state, r, t, m)
      tauhyb_array(n) = shell_optical_depth(r, t)
     else
      rfs_array(n) = r
      rph_array(n) = r_ph
      etrap_array(n) = erad
      tleak_array(n) = 0d0
      tauhyb_array(n) = 0d0
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
     tauhyb_array(n) = 0d0
    end if
    
    t_array(n) = t
    l_array(n) = lum_store
    fs_array(n) = lum_fs
    rs_array(n) = lum_rs
    r_array(n) = r_store
    m_array(n) = m
    v_array(n) = u
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
  if(allocated(tauarray_hybrid))deallocate(tauarray_hybrid)
  allocate(tarray(n))
  allocate(larray,temparray,rarray,varray,marray,ldiff,lfs,lrs,rfsarray,rpharray,etraparray,tleakarray,tauarray_hybrid,mold=tarray)

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
  tauarray_hybrid(1:n) = tauhyb_array(1:n)
  temparray(1:n) = temperature(larray(1:n),rarray(1:n))

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
      tau(i) = tau(i) &
             + mdjp/(tjp+tarray(i))&
             + (mdj*(tjp+tarray(i))-mdjp*(tj+tarray(i)))&
              /(tjp-tj)*(op(2)%vwind/rarray(i)-1d0/(tjp+tarray(i)))&
             +(mdjp-mdj)/(tjp-tj)&
              *log((tjp+tarray(i))*op(2)%vwind/rarray(i))
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

   tdiff = tau(i)*rarray(i)/clight
   tadv  = rarray(i)/varray(i)
   if(tdiff<=0d0.or.tadv<=0d0)cycle
   tloss = 1d0/(1d0/tdiff+1d0/tadv)
   if(tloss<=0d0)cycle
   do j = i+1, size(tarray)
    dldiff = exp((tarray(i+1)-tarray(j))/tloss)&
           - exp((tarray(i  )-tarray(j))/tloss)
    ldiff(j) = ldiff(j)+larray(i)*dldiff*tloss/tdiff
    if(tarray(j)>tarray(i+1)+max(tdiff,tadv)*50d0)exit
   end do
  end do

  return
 end subroutine get_diffuse_lc

end module lc_mod
