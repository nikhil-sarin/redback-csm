module lc_mod

 use constants,only:pi,year,clight,intpol,temperature
 use diffusion_runtime, only: paper_mode, diffusion_enabled, eff_global, erad, &
                              shock_efficiency_mode, &
                              configure_runtime, reset_diffusion_state, &
                              shell_optical_depth, photosphere_radius, &
                              solve_diffusion_step, set_runtime_mode, &
                              set_shock_efficiency_mode_runtime => set_shock_efficiency_mode

 implicit none

 real(8),allocatable,dimension(:),public:: tarray, Larray, temparray, rarray, varray, marray, ldiff, lfs, lrs
 public:: lightcurve_wind_exponential, lightcurve_wind_bpl, &
          lightcurve_bpl_wind, lightcurve_exponential_wind, &
          lightcurve_wind_explosion, lightcurve_explosion_wind, &
          lightcurve_wind_wind, lightcurve_explosion_explosion, &
          lightcurve_bpl_bpl, lightcurve_exponential_exponential, &
          lightcurve_explosion_bpl, lightcurve_bpl_exponential, &
          lightcurve_exponential_explosion, lightcurve_explosion_exponential, &
          set_model_mode, set_efficiency_mode

 private:: finalize_outputs, do_main_loop
 private:: get_diffuse_lc
 private
integer,parameter:: ll=10000
 real(8),dimension(ll):: t_array, L_array, ld_array, r_array, v_array, m_array, fs_array, rs_array
 integer,dimension(ll):: i_array
 real(8):: t_start=1d1, t_end=10d0*year
 real(8):: u,r,m,t

contains

 subroutine set_model_mode(mode)
  integer,intent(in):: mode

 call set_runtime_mode(mode)
 end subroutine set_model_mode

 subroutine set_efficiency_mode(mode)
  integer,intent(in):: mode

  call set_shock_efficiency_mode_runtime(mode)
 end subroutine set_efficiency_mode

 subroutine finalize_outputs(csm_type, eff, kappa)
  integer,intent(in):: csm_type
  real(8),intent(in),optional:: eff, kappa

  if(paper_mode)then
   if(present(eff))then
    larray = eff*larray
    lfs = eff*lfs
    lrs = eff*lrs
    temparray = eff**0.25d0*temparray
   end if

   if(present(kappa))then
    call get_diffuse_lc(csm_type,kappa)
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
  call configure_runtime(1,eff,kappa)

  call do_main_loop
  call finalize_outputs(1,eff,kappa)

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
  call configure_runtime(1,eff,kappa)

  call do_main_loop
  call finalize_outputs(1,eff,kappa)

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
  call configure_runtime(1,eff,kappa)

  call do_main_loop
  call finalize_outputs(1,eff,kappa)

 end subroutine lightcurve_explosion_bpl

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

! ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 subroutine do_main_loop

  use integration

	  integer:: n, scan_save
	  real(8):: dt,ku,kr,km
	  real(8):: lum_fs, lum_rs, lum_heat, lum_store, tau_now, r_ph_now, r_store, eta_fs

  t_array = -1d0
  ld_array = 0d0
  fs_array = 0d0
  rs_array = 0d0
  i_array = 0
  n = 0
  do while (t<=t_end)

! Evolve shell properties
	   ku = dudt(u,r,m,t,op)
	   kr = drdt(u,r,m,t,op)
	   km = dmdt(u,r,m,t,op)
	   lum_fs = forward_shock_luminosity(r,t,u,op)
	   lum_rs = reverse_shock_luminosity(r,t,u,op)
	   if(paper_mode)then
	    lum_heat = lum_fs + lum_rs
	   else
	    select case (shock_efficiency_mode)
	    case (1)
	     eta_fs = forward_shock_radiative_efficiency(r,t,u,op,eff_global)
	     lum_fs = eta_fs*lum_fs
	     lum_rs = eff_global*lum_rs
	    case default
	     lum_fs = eff_global*lum_fs
	     lum_rs = eff_global*lum_rs
	    end select
	    lum_heat = lum_fs + lum_rs
	   end if

! Adaptively adjust time stepping so that shell changes are resolved to ~1%
   if(paper_mode)then
    dt = 0.01d0*min(abs(u/ku),abs(r/kr),abs(m/km))
   else
    dt = huge(1d0)
    if(abs(ku)>1d-30)dt = min(dt,abs(u/ku))
    if(abs(kr)>1d-30)dt = min(dt,abs(r/kr))
    if(abs(km)>1d-30)dt = min(dt,abs(m/km))
    dt = 0.01d0*dt
    if(.not.(dt>0d0.and.dt<huge(1d0)))exit
   end if

   u = u + dt*ku
   r = r + dt*kr
   if(paper_mode)then
    m = m + dt*km
   else
    m = max(m + dt*km,1d-30)
   end if

	   if(paper_mode .or. .not.diffusion_enabled)then
	    erad = 0d0
	   end if

   t = t + dt
! Store output arrays
   if(t>1d4)then ! avoid first few steps that contain large errors
    if(paper_mode)then
     if(n>=1)i_array(n) = op(2)%scan_i
    end if
    n = n+1
	    if(paper_mode)then
	     lum_store = lum_heat
	     ld_array(n) = 0d0
	     r_store = r
	    else
	     lum_store = lum_heat
	     if(diffusion_enabled)then
	      scan_save = op(2)%scan_i
	      tau_now = shell_optical_depth(r,t)
	      r_ph_now = photosphere_radius(r,t,tau_now)
	      call solve_diffusion_step(dt,r,r_ph_now,t,lum_store,ld_array(n))
	      op(2)%scan_i = scan_save
	      r_store = r
	     else
	      ld_array(n) = 0d0
	      r_store = r
	     end if
	    end if
    t_array(n) = t
    l_array(n) = lum_store
    fs_array(n) = lum_fs
    rs_array(n) = lum_rs
    r_array(n) = r_store
    m_array(n) = m
    v_array(n) = u
    if(n>=ll)exit
   end if

  end do

  if(allocated(tarray))deallocate(tarray,larray,temparray,rarray,varray,marray,ldiff,lfs,lrs)
  allocate(tarray(n))
  allocate(larray,temparray,rarray,varray,marray,ldiff,lfs,lrs,mold=tarray)

  tarray(1:n) = t_array(1:n)
  larray(1:n) = l_array(1:n)
  lfs(1:n) = fs_array(1:n)
  lrs(1:n) = rs_array(1:n)
  rarray(1:n) = r_array(1:n)
  varray(1:n) = v_array(1:n)
  marray(1:n) = m_array(1:n)
  ldiff(1:n) = ld_array(1:n)
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
    j = max(1,min(i_array(i),size(op(2)%v_grid)-1))
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
      if(tjp-tj<=1d-99 .and. j>1)then
       tj   = op(2)%t_grid(j-1)
       mdj  = op(2)%mdot(j-1)
      end if
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
   case(4)
    tp = tarray(i) + op(2)%delay
    tau(i) = op(2)%Mej/(8d0*pi*(op(2)%exp_v0*tp)**2)&
             *exp(-rarray(i)/(op(2)%exp_v0*tp))
   end select
   tau(i) = kappa*tau(i)

   tdiff = max(tau(i)*rarray(i)/clight,1d-30)
   tadv  = max(rarray(i)/max(varray(i),1d-30),1d-30)
   tloss = 1d0/(1d0/tdiff+1d0/tadv)
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
