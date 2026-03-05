module lc_mod

 use constants,only:pi,year,clight

 implicit none

 real(8),allocatable,dimension(:),public:: tarray, Larray, temparray, rarray, varray, marray, ldiff
 public:: lightcurve_wind_exponential, lightcurve_wind_bpl, &
          lightcurve_bpl_wind, lightcurve_exponential_wind, &
          lightcurve_wind_explosion, lightcurve_explosion_wind, &
          lightcurve_wind_wind, lightcurve_explosion_explosion, &
          lightcurve_bpl_bpl, lightcurve_exponential_exponential, &
          lightcurve_explosion_bpl, lightcurve_bpl_exponential, &
          lightcurve_exponential_explosion, lightcurve_explosion_exponential

 private:: do_main_loop,get_diffuse_lc
 private
 integer,parameter:: ll=10000
 real(8),dimension(ll):: t_array, L_array, temp_array, r_array, v_array, m_array
 integer,dimension(ll):: i_array
 real(8):: t_start=1d1, t_end=10d0*year
 real(8):: u,r,m,t,lum

contains

 subroutine lightcurve_wind_exponential(Mdotinput, tinput, vwindinput, Mexp, Eexp, eff, kappa)

! purpose: To compute LC for exponential explosion interacting with wind

  use get_vals

  real(8),intent(in):: Eexp, Mexp, vwindinput
  real(8),intent(in),target:: Mdotinput(:), tinput(:)
  real(8),intent(in),optional:: eff, kappa

  v_in  => v_explosion
  v_out => v_wind
  rho4pir2_in  => rho_exponential
  rho4pir2_out => rho_wind

  op(1)%scan_i = -1
  op(2)%scan_i = 1

  op(1)%Mej = Mexp
  op(1)%Eej = Eexp
  op(2)%mdot => Mdotinput
  op(2)%t_grid => tinput
  op(2)%vwind = vwindinput

  call get_exp_v0(op(1))

  t = t_start   ! Use non-zero initial time to avoid singularity
  u = max(1.d2*op(1)%exp_v0,op(2)%vwind)
  r = u*t*1.2d0
  m = 1d0

  call do_main_loop

  if(present(eff))then
   larray = eff*larray
   temparray = eff**0.25d0*temparray
  end if

  if(present(kappa))then
   call get_diffuse_lc(2,kappa)
  end if

 end subroutine lightcurve_wind_exponential

 subroutine lightcurve_wind_explosion(Mdotinput, tinput, vwindinput, rho_input, vinput, t_ref, eff, kappa)

! purpose: To compute LC for arbitrary explosion interacting with wind

  use get_vals

  real(8),intent(in):: vwindinput, t_ref
  real(8),intent(in),target:: Mdotinput(:), tinput(:), rho_input(:), vinput(:)
  real(8),intent(in),optional:: eff, kappa

  v_in  => v_explosion
  v_out => v_wind
  rho4pir2_in  => rho_explosion
  rho4pir2_out => rho_wind

  op(1)%scan_i = -size(vinput)
  op(2)%scan_i = 1

  op(1)%rho_expl => rho_input
  op(1)%v_grid => vinput
  op(1)%t_ref = t_ref
  op(2)%mdot => Mdotinput
  op(2)%t_grid => tinput
  op(2)%vwind = vwindinput

  t = t_start   ! Use non-zero initial time to avoid singularity
  u = max(maxval(vinput),op(2)%vwind)
  r = u*t*1.2d0
  m = 1d0

  call do_main_loop

  if(present(eff))then
   larray = eff*larray
   temparray = eff**0.25d0*temparray
  end if

  if(present(kappa))then
   call get_diffuse_lc(2,kappa)
  end if

 end subroutine lightcurve_wind_explosion

 subroutine lightcurve_wind_bpl(Mdotinput, tinput, vwindinput, inner_slope, outer_slope, Mexp, Eexp, eff, kappa)

! purpose: To compute LC for broken power-law explosion interacting with wind

  use get_vals

  real(8),intent(in):: Eexp, Mexp, inner_slope, outer_slope, vwindinput
  real(8),intent(in),target:: Mdotinput(:), tinput(:)
  real(8),intent(in),optional:: eff, kappa

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

  call do_main_loop

  if(present(eff))then
   larray = eff*larray
   temparray = eff**0.25d0*temparray
  end if

  if(present(kappa))then
   call get_diffuse_lc(2,kappa)
  end if

 end subroutine lightcurve_wind_bpl


 subroutine lightcurve_bpl_wind(inner_slope, outer_slope, Mexp, Eexp, Mdotinput, tinput, vwindinput, eff, kappa)

! purpose: To compute LC for wind interacting with broken power-law explosion

  use get_vals

  real(8),intent(in):: Eexp, Mexp, inner_slope, outer_slope, vwindinput
  real(8),intent(in),target:: Mdotinput(:), tinput(:)
  real(8),intent(in),optional:: eff, kappa

  v_in  => v_wind
  v_out => v_explosion
  rho4pir2_in  => rho_wind
  rho4pir2_out => rho_bpl

  op(1)%scan_i = -size(tinput)
  op(2)%scan_i = 1

  op(1)%mdot = Mdotinput
  op(1)%t_grid = tinput
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

  call do_main_loop

  if(present(eff))then
   larray = eff*larray
   temparray = eff**0.25d0*temparray
  end if

  if(present(kappa))then
   call get_diffuse_lc(3,kappa)
  end if

 end subroutine lightcurve_bpl_wind

 subroutine lightcurve_exponential_wind(Mexp, Eexp, Mdotinput, tinput, vwindinput, eff, kappa)

! purpose: To compute LC for wind interacting with exponential explosion

  use get_vals

  real(8),intent(in):: Eexp, Mexp, vwindinput
  real(8),intent(in),target:: Mdotinput(:), tinput(:)
  real(8),intent(in),optional:: eff, kappa

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

  call get_exp_v0(op(2))

  t = t_start   ! Use non-zero initial time to avoid singularity
  r = 1d0
  u = r/t
  m = 1d0

  call do_main_loop

  if(present(eff))then
   larray = eff*larray
   temparray = eff**0.25d0*temparray
  end if

  if(present(kappa))then
   call get_diffuse_lc(4,kappa)
  end if

 end subroutine lightcurve_exponential_wind

 subroutine lightcurve_explosion_wind(rho_input, vinput, t_ref, Mdotinput, tinput, vwindinput, eff, kappa)

! purpose: To compute LC for wind interacting with an arbitrary explosion

  use get_vals

  real(8),intent(in):: t_ref, vwindinput
  real(8),intent(in),target:: rho_input(:), vinput(:), Mdotinput(:), tinput(:)
  real(8),intent(in),optional:: eff, kappa

  v_in  => v_wind
  v_out => v_explosion
  rho4pir2_in  => rho_wind
  rho4pir2_out => rho_explosion

  op(1)%scan_i = -size(tinput)
  op(2)%scan_i = 1

  op(1)%mdot = Mdotinput
  op(1)%t_grid = tinput
  op(1)%vwind = vwindinput
  op(2)%rho_expl => rho_input
  op(2)%v_grid => vinput
  op(2)%t_ref = t_ref
  op(2)%delay = 0d0

  call get_bpl_coeffs(op(2))

  t = t_start   ! Use non-zero initial time to avoid singularity
  r = 1d0
  u = r/t
  m = 1d0

  call do_main_loop

  if(present(eff))then
   larray = eff*larray
   temparray = eff**0.25d0*temparray
  end if

  if(present(kappa))then
   call get_diffuse_lc(1,kappa)
  end if

 end subroutine lightcurve_explosion_wind

 subroutine lightcurve_wind_wind(Mdoto, tinputo, vwindo, Mdoti, tinputi, vwindi, eff, kappa)

! purpose: To compute LC for wind interacting with another wind

  use get_vals

  real(8),intent(in):: vwindo, vwindi
  real(8),intent(in),target:: Mdoto(:), tinputo(:), Mdoti(:), tinputi(:)
  real(8),intent(in),optional:: eff, kappa

  v_in  => v_wind
  v_out => v_wind
  rho4pir2_in  => rho_wind
  rho4pir2_out => rho_wind

  op(1)%scan_i = -size(tinputi)
  op(2)%scan_i = 1

  op(1)%mdot => Mdoti
  op(1)%t_grid => tinputi
  op(1)%vwind = vwindi
  op(2)%mdot = Mdoto
  op(2)%t_grid = tinputo
  op(2)%vwind = vwindo

  t = t_start   ! Use non-zero initial time to avoid singularity
  r = 1d0
  u = 0.5d0*(vwindi+vwindo)
  m = 1d0

  call do_main_loop

  if(present(eff))then
   larray = eff*larray
   temparray = eff**0.25d0*temparray
  end if

  if(present(kappa))then
   call get_diffuse_lc(2,kappa)
  end if

 end subroutine lightcurve_wind_wind

 subroutine lightcurve_bpl_bpl(inner_slopeo, outer_slopeo, Mexpo, Eexpo, &
                               inner_slopei, outer_slopei, Mexpi, Eexpi, &
                               interval, eff, kappa)

! purpose: To compute LC for interaction between two explosions (bpl-bpl)

  use get_vals

  real(8),intent(in):: inner_slopeo, outer_slopeo, Mexpo, Eexpo, inner_slopei, outer_slopei, Mexpi, Eexpi, interval
  real(8),intent(in),optional:: eff, kappa

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

  call do_main_loop

  if(present(eff))then
   larray = eff*larray
   temparray = eff**0.25d0*temparray
  end if

  if(present(kappa))then
   call get_diffuse_lc(3,kappa)
  end if

 end subroutine lightcurve_bpl_bpl

 subroutine lightcurve_exponential_exponential(Mexpo, Eexpo, Mexpi, Eexpi, interval, eff, kappa)

! purpose: To compute LC for interaction between two explosions (exponential-exponential)

  use get_vals

  real(8),intent(in):: Eexpo, Mexpo, Eexpi, Mexpi, interval
  real(8),intent(in),optional:: eff, kappa

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

  call do_main_loop

  if(present(eff))then
   larray = eff*larray
   temparray = eff**0.25d0*temparray
  end if

  if(present(kappa))then
   call get_diffuse_lc(4,kappa)
  end if

 end subroutine lightcurve_exponential_exponential

 subroutine lightcurve_explosion_explosion(rho_o,vgrido,trefo,rho_i,vgridi,trefi,interval, eff, kappa)

! purpose: To compute LC for interaction between two arbitrary explosions

  use get_vals

  real(8),intent(in):: trefo,trefi,interval
  real(8),intent(in),target:: rho_o(:),vgrido(:),rho_i(:),vgridi(:)
  real(8),intent(in),optional:: eff, kappa

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

  call do_main_loop

  if(present(eff))then
   larray = eff*larray
   temparray = eff**0.25d0*temparray
  end if

  if(present(kappa))then
   call get_diffuse_lc(1,kappa)
  end if

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

  call do_main_loop

  if(present(eff))then
   larray = eff*larray
   temparray = eff**0.25d0*temparray
  end if

  if(present(kappa))then
   call get_diffuse_lc(1,kappa)
  end if

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

  call do_main_loop

  if(present(eff))then
   larray = eff*larray
   temparray = eff**0.25d0*temparray
  end if

  if(present(kappa))then
   call get_diffuse_lc(3,kappa)
  end if

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

  call do_main_loop

  if(present(eff))then
   larray = eff*larray
   temparray = eff**0.25d0*temparray
  end if

  if(present(kappa))then
   call get_diffuse_lc(4,kappa)
  end if

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

  call do_main_loop

  if(present(eff))then
   larray = eff*larray
   temparray = eff**0.25d0*temparray
  end if

  if(present(kappa))then
   call get_diffuse_lc(1,kappa)
  end if

 end subroutine lightcurve_explosion_exponential

! ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 subroutine do_main_loop

  use constants,only:temperature
  use integration

  integer:: n
  real(8):: dt,ku,kr,km,lum

  t_array = -1d0
  
  n = 0
  do while (t<=t_end)

! Evolve shell properties
   ku = dudt(u,r,m,t,op)
   kr = drdt(u,r,m,t,op)
   km = dmdt(u,r,m,t,op)

! Adaptively adjust time stepping so that shell changes are resolved to ~1%
   dt = 0.01d0*min(abs(u/ku),abs(r/kr),abs(m/km))

   lum = shock_luminosity(r,t,u,op)

   u = u + dt*ku
   r = r + dt*kr
   m = m + dt*km

   t = t + dt

! Store output arrays
   if(t>1d4)then ! avoid first few steps that contain large errors
    if(n>=1)i_array(n) = op(2)%scan_i
    n = n+1
    t_array(n) = t
    l_array(n) = lum
    r_array(n) = r
    m_array(n) = m
    v_array(n) = u
   end if

  end do

  if(allocated(tarray))deallocate(tarray,larray,temparray,rarray,varray,marray)
  allocate(tarray(n))
  allocate(larray,temparray,rarray,varray,marray,mold=tarray)

  tarray(1:n) = t_array(1:n)
  larray(1:n) = l_array(1:n)
  rarray(1:n) = r_array(1:n)
  varray(1:n) = v_array(1:n)
  marray(1:n) = m_array(1:n)
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

  ! Prepare quantities for optical depth calculation
  select case(csm_type)
  case(1) ! explosion case
   call get_tauprep_explosion(tauprep)
  end select

  if(allocated(ldiff))deallocate(ldiff)
  allocate(tau,ldiff,mold=larray)
  ldiff = 0d0

  do i = 1, size(tarray)-1
   select case(csm_type)
   case(1) ! arbitrary explosion
    j = i_array(i)
    tp = tarray(i) + op(2)%delay
    tau(i) = intpol(rarray(i)/tp,op(2)%v_grid(j:j+1),tauprep(j:j+1)) &
           / tp**2
   case(2) ! arbitrary wind
    j = i_array(i)
    n = size(op(2)%t_grid)
    if(j==n)then
     tau(i) = op(2)%mdot(n)*op(2)%vwind/rarray(i)
    else
     tau(i) = 0d0!op(2)%mdot(j+1)/(op(2)%t_grid(j+1)+tarray(i))
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
   case(3) ! broken power-law explosion
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
   case(4) ! exponential explosion
    tp = tarray(i) + op(2)%delay
    tau(i) = op(2)%Mej/(8d0*pi*(op(2)%exp_v0*tp)**2)&
             *exp(-rarray(i)/(op(2)%exp_v0*tp))
   end select
   tau(i) = kappa*tau(i)

   tdiff = tau(i)*rarray(i)/clight
   tadv  = rarray(i)/varray(i)
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
