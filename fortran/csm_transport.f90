module csm_transport

 use constants, only: pi, clight
 use physical_constants, only: a_rad
use get_vals, only: op, query_csm_density, query_csm_inner_edge, query_csm_outer_edge, &
                     query_tau_to_edge, query_csm_photosphere_radius, query_csm_velocity, &
                     query_ejecta_density, query_ejecta_velocity
use integration, only: dudt, drdt, dmdt

 implicit none

 type transport_state_type
 logical :: initialized = .false.
 logical :: comoving_initialized = .false.
 logical :: in_cooling_phase = .false.
 logical :: cooling_initialized = .false.
  integer :: n_zones = 0
  real(8) :: kappa = 0d0
  real(8) :: t_shell = 0d0
  real(8) :: r_shell_current = 0d0
  real(8) :: r_shell_prev = 0d0
 real(8) :: m_shocked_csm = 0d0
 real(8) :: m_shocked_ej = 0d0
 real(8) :: e_residual = 0d0
 real(8) :: t_emerge = 0d0
 real(8) :: u_shell_current = 0d0
 real(8) :: lum_heat_last = 0d0
  real(8) :: r_inner_support = 0d0
  real(8) :: r_inner = 0d0
 real(8) :: r_outer = 0d0
 real(8) :: r_outer_support = 0d0
 real(8) :: r_emerge_inner = 0d0
 real(8) :: r_emerge_outer = 0d0
 real(8) :: r_emerge_shell = 0d0
 real(8) :: u_emerge_shell = 0d0
 real(8) :: cooling_scale = 1d0
 real(8) :: cooling_scale_prev = 1d0
 real(8) :: cooling_expansion_ratio = 1d0
 real(8) :: compact_tail_drain = 0d0
 real(8) :: t_gap_end = 0d0
 real(8) :: lum_heat_gap = 0d0
 real(8) :: y_surface_target = 0d0
 integer :: interaction_rannacher_left = 0
 integer :: cooling_rannacher_left = 0
  real(8) :: m_emerge_csm = 0d0
  real(8) :: m_emerge_ej = 0d0
  real(8) :: m_emerge_shell = 0d0
  real(8), allocatable :: radius(:)
  real(8), allocatable :: radius_ref(:)
  real(8), allocatable :: rho(:)
  real(8), allocatable :: rho_ref(:)
  real(8), allocatable :: y(:)
  real(8), allocatable :: e_swept(:)
  real(8), allocatable :: tau(:)
  real(8), allocatable :: work_a(:)
  real(8), allocatable :: work_b(:)
  real(8), allocatable :: work_c(:)
  real(8), allocatable :: work_rhs(:)
  real(8), allocatable :: work_sol(:)
  real(8), allocatable :: work_old_y(:)
  real(8), allocatable :: work_kl(:)
  real(8), allocatable :: work_kr(:)
  real(8), allocatable :: work_kesc(:)
  real(8), allocatable :: work_vol(:)
  real(8), allocatable :: work_vol_old(:)
  real(8), allocatable :: work_gam(:)
 end type transport_state_type

 public :: transport_state_type, reset_transport_state, &
           interaction_transport_step, find_transport_photosphere, &
           shock_has_emerged, initialize_cooling_state_from_interaction, &
           cooling_transport_step, transport_timestep_limit, shock_motion_timestep_limit, &
           shock_motion_timestep_limit_at, forward_shock_radius, shell_leakage_timescale, total_radiation_energy, &
           comoving_transport_step

 logical, save :: mode3_debug_initialized = .false.
 logical, save :: mode3_debug_enabled_cached = .false.

! ==========================================================================
! Dimensionless formulation types (paper Appendix A)
! ==========================================================================

 type dimless_state_type
  logical :: initialized = .false.
  logical :: in_cooling_phase = .false.
  integer :: n_zones = 0

  ! Characteristic scales (set once from op(1), op(2))
  real(8) :: R_csm_in = 0d0     ! R_{csm,in} [cm]
  real(8) :: v_ej_max = 0d0     ! v_{ej,max} [cm/s]
  real(8) :: t_in = 0d0         ! t_in = R_csm_in / v_ej_max [s]
  real(8) :: rho_ej_in = 0d0    ! rho_{ej,in} [g/cm^3]
  real(8) :: rho_csm_in = 0d0   ! rho_{csm,in} [g/cm^3]
  real(8) :: q = 0d0            ! q = rho_csm_in / rho_ej_in
  real(8) :: tau_csm_in = 0d0   ! tau_{csm,in} = 3*kappa*rho_csm_in*R_csm_in
  real(8) :: t_diff = 0d0       ! t_diff = tau_csm_in * R_csm_in / c [s]
  real(8) :: u0 = 0d0           ! paper Eq. 836
  real(8) :: kappa = 0d0
  real(8) :: eff = 1d0          ! radiative efficiency epsilon
  real(8) :: R_csm_out = 0d0    ! outer CSM edge [cm]
  real(8) :: x_csm_out = 0d0    ! R_csm_out / R_csm_in (dimensionless)
  real(8) :: y_ratio = 0d0      ! t_in / t_diff (constant)

  ! Dynamics state (dimensionless)
  real(8) :: x_sh = 1d0         ! x = R_sh / R_csm_in
  real(8) :: w_sh = 1d0         ! w = v_sh / v_ej_max
  real(8) :: phi_sh = 1d-6      ! phi = M_sh / (4*pi*R_csm_in^3*rho_ej_in)
  real(8) :: zeta = 1d0         ! zeta = t / t_in
  real(8) :: y_diff = 0d0       ! y = t / t_diff

  ! Cooling phase
  real(8) :: zeta_se = 0d0      ! zeta at shock emergence
  real(8) :: y_se = 0d0         ! y at shock emergence
  real(8) :: R0 = 0d0           ! R_in(t_se) [cm]
  real(8) :: v_se = 0d0         ! v_se [cm/s]
  real(8) :: R_in_R0 = 1d0      ! R_in(t) / R_0 (starts at 1)

  ! Diffusion grid (fixed ξ-space, ξ ∈ [0,1])
  real(8) :: x_ph = 0d0         ! photosphere position in x
  real(8) :: x_min = 1d0        ! inner boundary of diffusion domain in x
  real(8) :: x_sh_dot = 0d0     ! dx_sh/dy (shock velocity in y-units)
  integer :: rannacher_left = 0
  real(8) :: x_ph_cached_xsh = -1d0  ! x_sh when x_ph was last computed
  real(8) :: x_ph_cached_xmin = -1d0 ! x_min when x_ph was last computed
  real(8), allocatable :: xi_grid(:)   ! fixed uniform ξ = (i-1)/(n-1)
  real(8), allocatable :: e_grid(:)    ! dimensionless energy density e(ξ)
  ! Tridiagonal work arrays
  real(8), allocatable :: work_a(:), work_b(:), work_c(:)
  real(8), allocatable :: work_rhs(:), work_sol(:), work_gam(:)
  real(8), allocatable :: work_old_e(:)

  ! Cached CGS outputs
  real(8) :: lum_obs_cgs = 0d0
  real(8) :: r_ph_cgs = 0d0
  real(8) :: r_sh_cgs = 0d0
  real(8) :: v_sh_cgs = 0d0
  real(8) :: m_sh_cgs = 0d0
  real(8) :: t_cgs = 0d0
  integer :: nsub_last = 0
 end type dimless_state_type

 public :: dimless_state_type, dimless_comoving_transport_step, &
           reset_dimless_state, initialize_dimless_state, dimless_to_cgs

contains

logical function mode3_debug_enabled()
 character(len=32) :: env
 integer :: stat

 if (.not. mode3_debug_initialized) then
  env = ''
  stat = 1
  call get_environment_variable('TRANSFIT_MODE3_DEBUG', env, status=stat)
  if (stat == 0) then
   mode3_debug_enabled_cached = len_trim(env) > 0 .and. env(1:1) /= '0' .and. env(1:1) /= 'f' .and. &
                                env(1:1) /= 'F' .and. env(1:1) /= 'n' .and. env(1:1) /= 'N'
  else
   mode3_debug_enabled_cached = .false.
  end if
  mode3_debug_initialized = .true.
 end if

 mode3_debug_enabled = mode3_debug_enabled_cached
end function mode3_debug_enabled

 subroutine reset_transport_state(state)
  type(transport_state_type), intent(inout) :: state

  if (allocated(state%radius)) deallocate(state%radius)
  if (allocated(state%radius_ref)) deallocate(state%radius_ref)
  if (allocated(state%rho)) deallocate(state%rho)
  if (allocated(state%rho_ref)) deallocate(state%rho_ref)
  if (allocated(state%y)) deallocate(state%y)
  if (allocated(state%e_swept)) deallocate(state%e_swept)
  if (allocated(state%tau)) deallocate(state%tau)
  if (allocated(state%work_a)) deallocate(state%work_a)
  if (allocated(state%work_b)) deallocate(state%work_b)
  if (allocated(state%work_c)) deallocate(state%work_c)
  if (allocated(state%work_rhs)) deallocate(state%work_rhs)
  if (allocated(state%work_sol)) deallocate(state%work_sol)
  if (allocated(state%work_old_y)) deallocate(state%work_old_y)
  if (allocated(state%work_kl)) deallocate(state%work_kl)
  if (allocated(state%work_kr)) deallocate(state%work_kr)
  if (allocated(state%work_kesc)) deallocate(state%work_kesc)
  if (allocated(state%work_vol)) deallocate(state%work_vol)
  if (allocated(state%work_vol_old)) deallocate(state%work_vol_old)
 if (allocated(state%work_gam)) deallocate(state%work_gam)
 state%y_surface_target = 0d0

  state%initialized = .false.
  state%comoving_initialized = .false.
  state%in_cooling_phase = .false.
  state%cooling_initialized = .false.
  ! reset local saved flags handled in caller by reinitialization
  state%n_zones = 0
  state%kappa = 0d0
  state%t_shell = 0d0
  state%r_shell_current = 0d0
  state%r_shell_prev = 0d0
  state%m_shocked_csm = 0d0
  state%m_shocked_ej = 0d0
  state%e_residual = 0d0
  state%t_emerge = 0d0
  state%u_shell_current = 0d0
  state%lum_heat_last = 0d0
  state%r_inner_support = 0d0
  state%r_inner = 0d0
  state%r_outer = 0d0
  state%r_outer_support = 0d0
  state%r_emerge_inner = 0d0
  state%r_emerge_outer = 0d0
  state%r_emerge_shell = 0d0
  state%u_emerge_shell = 0d0
  state%cooling_scale = 1d0
  state%cooling_scale_prev = 1d0
  state%cooling_expansion_ratio = 1d0
 state%compact_tail_drain = 0d0
 state%t_gap_end = 0d0
 state%lum_heat_gap = 0d0
 state%interaction_rannacher_left = 0
 state%cooling_rannacher_left = 0
  state%m_emerge_csm = 0d0
  state%m_emerge_ej = 0d0
  state%m_emerge_shell = 0d0
 end subroutine reset_transport_state

 subroutine build_radial_grid(r_inner, r_outer, radius)
  real(8), intent(in) :: r_inner, r_outer
  real(8), intent(out) :: radius(:)
  integer :: i, n
  real(8) :: x

  n = size(radius)
  do i = 1, n
   x = dble(i-1) / dble(max(n-1, 1))
   if (r_outer / r_inner > 1.05d0) then
    radius(i) = r_inner * (r_outer / r_inner)**x
   else
    radius(i) = r_inner + (r_outer - r_inner) * x
   end if
  end do
 end subroutine build_radial_grid

 subroutine fill_stationary_reference_profile(state, t_shell)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: t_shell
  integer :: i

  do i = 1, state%n_zones
   state%rho_ref(i) = max(query_csm_density(state%radius_ref(i), t_shell, op(2)), 1d-30)
  end do
 end subroutine fill_stationary_reference_profile

 subroutine fill_active_profile_from_reference(state)
  type(transport_state_type), intent(inout) :: state
  integer :: i

  do i = 1, state%n_zones
   state%rho(i) = max(interp_linear_monotonic(state%radius(i), state%radius_ref, state%rho_ref, state%n_zones), 1d-30)
  end do
 end subroutine fill_active_profile_from_reference

subroutine deposit_energy_interval_to_reference(src_r, src_inner, src_outer, src_y, n_src, &
                                                 cut_l, cut_r, ref_r, ref_inner, ref_outer, n_ref, e_ref)
  integer, intent(in) :: n_src, n_ref
  real(8), intent(in) :: src_r(n_src), src_inner, src_outer, src_y(n_src)
  real(8), intent(in) :: cut_l, cut_r
  real(8), intent(in) :: ref_r(n_ref), ref_inner, ref_outer
  real(8), intent(inout) :: e_ref(n_ref)
  real(8), allocatable :: src_face_l(:), src_face_r(:), src_face_vol(:)
  real(8), allocatable :: ref_face_l(:), ref_face_r(:), ref_face_vol(:)
  integer :: i, j, i_start
  real(8) :: src_l, src_rface, ref_l, ref_rface
  real(8) :: rhoe_src, rlo, rhi, overlap_vol, rlo3, rhi3, ref_l3, ref_r3
  real(8), parameter :: four_pi_over_3 = 4.18879020478639098461685784438d0

  if (cut_r <= cut_l) return

  allocate(src_face_l(n_src), src_face_r(n_src), src_face_vol(n_src))
  allocate(ref_face_l(n_ref), ref_face_r(n_ref), ref_face_vol(n_ref))
  call prepare_zone_geometry_arrays(src_r, src_inner, src_outer, n_src, src_face_l, src_face_r, src_face_vol)
  call prepare_zone_geometry_arrays(ref_r, ref_inner, ref_outer, n_ref, ref_face_l, ref_face_r, ref_face_vol)

  i_start = 1
  do j = 1, n_src
   src_l = src_face_l(j)
   src_rface = src_face_r(j)
   rlo = max(src_l, cut_l)
   rhi = min(src_rface, cut_r)
   if (rhi <= rlo) cycle
   rhoe_src = a_rad * max(src_y(j), 0d0)
   if (rhoe_src <= 0d0) cycle

   rlo3 = rlo**3
   rhi3 = rhi**3
   do i = i_start, n_ref
    ref_l = ref_face_l(i)
    if (ref_l >= rhi) exit
    ref_rface = ref_face_r(i)
    if (ref_rface <= rlo) then
     i_start = i + 1
     cycle
    end if
    ref_l3 = max(rlo3, ref_l**3)
    ref_r3 = min(rhi3, ref_rface**3)
    if (ref_r3 > ref_l3) then
     overlap_vol = four_pi_over_3 * (ref_r3 - ref_l3)
     e_ref(i) = e_ref(i) + rhoe_src * overlap_vol
    end if
   end do
  end do

  deallocate(src_face_l, src_face_r, src_face_vol)
  deallocate(ref_face_l, ref_face_r, ref_face_vol)
end subroutine deposit_energy_interval_to_reference

subroutine deposit_scalar_energy_to_reference(cut_l, cut_r, ref_r, ref_inner, ref_outer, rho_ref, n_ref, e_dep, dE)
  integer, intent(in) :: n_ref
  real(8), intent(in) :: cut_l, cut_r
  real(8), intent(in) :: ref_r(n_ref), ref_inner, ref_outer, rho_ref(n_ref)
  real(8), intent(inout) :: e_dep(n_ref)
  real(8), intent(in) :: dE
  real(8), allocatable :: ref_face_l(:), ref_face_r(:), ref_face_vol(:)
  integer :: i, iclose
  real(8) :: ref_l, ref_rface, overlap_vol, weight, weight_sum, rmid, dmin
  real(8), allocatable :: weights(:)

  if (dE <= 0d0) return
  allocate(weights(n_ref))
  weights = 0d0
  weight_sum = 0d0

  allocate(ref_face_l(n_ref), ref_face_r(n_ref), ref_face_vol(n_ref))
  call prepare_zone_geometry_arrays(ref_r, ref_inner, ref_outer, n_ref, ref_face_l, ref_face_r, ref_face_vol)

  do i = 1, n_ref
   ref_l = ref_face_l(i)
   ref_rface = ref_face_r(i)
   overlap_vol = 4d0 * pi * max(min(cut_r, ref_rface)**3 - max(cut_l, ref_l)**3, 0d0) / 3d0
   if (overlap_vol > 0d0) then
    weight = max(rho_ref(i), 1d-30) * overlap_vol
    weights(i) = weight
    weight_sum = weight_sum + weight
   end if
  end do

  if (weight_sum <= 0d0) then
   iclose = 1
   dmin = huge(1d0)
   rmid = 0.5d0 * (cut_l + cut_r)
   do i = 1, n_ref
    if (abs(ref_r(i) - rmid) < dmin) then
     dmin = abs(ref_r(i) - rmid)
     iclose = i
    end if
   end do
   e_dep(iclose) = e_dep(iclose) + dE
  else
   do i = 1, n_ref
    if (weights(i) > 0d0) e_dep(i) = e_dep(i) + dE * weights(i) / weight_sum
   end do
  end if

  deallocate(weights)
  deallocate(ref_face_l, ref_face_r, ref_face_vol)
end subroutine deposit_scalar_energy_to_reference

subroutine initialize_interaction_grid(state, r_shell, t_shell, m_shell, kappa, n_zones)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: r_shell, t_shell, m_shell, kappa
  integer, intent(in) :: n_zones

  call reset_transport_state(state)
  state%n_zones = max(8, n_zones)
  state%kappa = max(kappa, 1d-30)
  state%initialized = .true.
  state%t_shell = t_shell
  state%r_shell_current = max(r_shell, 1d0)
  state%r_shell_prev = state%r_shell_current
  state%r_inner_support = max(query_csm_inner_edge(t_shell, op(2)), 1d0)
  state%r_outer_support = query_csm_outer_edge(t_shell, op(2))
  if (.not.(state%r_outer_support > state%r_inner_support)) then
   state%r_outer_support = state%r_inner_support * (1d0 + 1d-3)
  end if
  state%r_inner = min(max(state%r_shell_current, state%r_inner_support), state%r_outer_support * (1d0 - 1d-10))
  state%r_outer = state%r_outer_support

  allocate(state%radius(state%n_zones), state%radius_ref(state%n_zones), &
           state%rho(state%n_zones), state%rho_ref(state%n_zones), state%y(state%n_zones), &
           state%e_swept(state%n_zones), state%tau(state%n_zones))
  allocate(state%work_a(state%n_zones), state%work_b(state%n_zones), state%work_c(state%n_zones), &
           state%work_rhs(state%n_zones), state%work_sol(state%n_zones), state%work_old_y(state%n_zones), &
           state%work_kl(state%n_zones), state%work_kr(state%n_zones), state%work_kesc(state%n_zones), &
           state%work_vol(state%n_zones), state%work_vol_old(state%n_zones), state%work_gam(state%n_zones))

  call build_radial_grid(state%r_inner_support, state%r_outer_support, state%radius_ref)
  call fill_stationary_reference_profile(state, t_shell)
  call build_radial_grid(state%r_inner, state%r_outer, state%radius)
  call fill_active_profile_from_reference(state)

  state%y = 0d0
  state%e_swept = 0d0
  state%interaction_rannacher_left = 4
  call update_tau_and_luminosity(state)
  if (mode3_debug_enabled()) then
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
        'M3_INIT_GRID rin=', state%r_inner_support, 'rout=', state%r_outer_support, &
        'rho_in=', state%rho_ref(1), 'rho_out=', state%rho_ref(state%n_zones), 'tau0=', state%tau(1), 'kappa=', state%kappa
  end if
 end subroutine initialize_interaction_grid

integer function first_active_zone_at(state, r_shell) result(ihi)
  type(transport_state_type), intent(in) :: state
  real(8), intent(in) :: r_shell
  integer :: i
  real(8) :: r_face_l, r_face_r, vol_i, r_target
  real(8) :: tol

  ihi = 1
  r_target = max(r_shell, state%r_inner_support)
  tol = 1d-6
  if (state%n_zones <= 0) return

  if (r_target <= state%r_inner_support) then
    ihi = 1
    return
  end if

  if (r_target >= state%r_outer_support) then
    ihi = state%n_zones
    return
  end if

  do i = 1, state%n_zones
    call zone_geometry_from_array(state%radius_ref, state%r_inner_support, state%r_outer_support, state%n_zones, i, &
         r_face_l, r_face_r, vol_i)
    if (r_target >= r_face_l .and. r_target <= r_face_r) then
      if (r_target >= r_face_r - tol * max(r_face_r - r_face_l, 1d-30) .and. i < state%n_zones) then
        ihi = i + 1
      else
        ihi = i
      end if
      return
    end if
  end do
end function first_active_zone_at

integer function first_active_zone_at_arr(radius_ref, r_inner, r_outer, n, r_shell) result(ihi)
  integer, intent(in) :: n
  real(8), intent(in) :: radius_ref(n), r_inner, r_outer, r_shell
  integer :: i
  real(8) :: r_face_l, r_face_r, vol_i, r_target, tol

  ihi = 1
  r_target = max(r_shell, r_inner)
  tol = 1d-6
  if (n <= 0) return
  if (r_target <= r_inner) then; ihi = 1; return; end if
  if (r_target >= r_outer) then; ihi = n; return; end if
  do i = 1, n
    call zone_geometry_from_array(radius_ref, r_inner, r_outer, n, i, r_face_l, r_face_r, vol_i)
    if (r_target >= r_face_l .and. r_target <= r_face_r) then
      if (r_target >= r_face_r - tol * max(r_face_r - r_face_l, 1d-30) .and. i < n) then
        ihi = i + 1
      else
        ihi = i
      end if
      return
    end if
  end do
end function first_active_zone_at_arr

subroutine interaction_zone_geometry_at(state, i, r_shell, r_face_l, r_face_r, vol_i)
  type(transport_state_type), intent(in) :: state
  integer, intent(in) :: i
  real(8), intent(in) :: r_shell
  real(8), intent(out) :: r_face_l, r_face_r, vol_i
  integer :: ihi

  call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
  if (state%r_inner >= min(max(r_shell, state%r_inner_support), state%r_outer_support * (1d0 - 1d-10))) then
   ihi = 1
  else
   ihi = first_active_zone_at(state, r_shell)
  end if
  if (i == ihi) then
    r_face_l = min(max(r_shell, r_face_l), r_face_r)
    vol_i = 4d0 * pi * max(r_face_r**3 - r_face_l**3, 1d-30) / 3d0
  end if
end subroutine interaction_zone_geometry_at

subroutine update_interaction_geometry(state, r_shell, t_shell, m_shell)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: r_shell, t_shell, m_shell
  real(8), allocatable :: old_radius(:), old_y(:)
  real(8) :: old_inner, old_outer, new_inner, e_removed

  state%t_shell = t_shell
  state%r_shell_prev = state%r_shell_current
  state%r_shell_current = max(r_shell, 1d0)
  old_inner = state%r_inner
  old_outer = state%r_outer
  allocate(old_radius(state%n_zones), old_y(state%n_zones))
  old_radius = state%radius
  old_y = state%y

  new_inner = max(state%r_shell_current, state%r_inner_support)
  if (state%r_outer_support > state%r_inner_support) then
   new_inner = min(new_inner, state%r_outer_support * (1d0 - 1d-10))
  end if
  state%r_inner = new_inner
  state%r_outer = state%r_outer_support
  call build_radial_grid(state%r_inner, state%r_outer, state%radius)
  call fill_active_profile_from_reference(state)

  ! e_swept is now accumulated in solve_transport_step (lum_heat*dt per zone).
  ! e_residual is updated via energy balance in interaction_transport_step.
  call remap_y_conservative(old_radius, old_inner, old_outer, old_y, state%n_zones, &
       state%radius, state%r_inner, state%r_outer, state%y, state%n_zones, e_removed)

  deallocate(old_radius, old_y)
end subroutine update_interaction_geometry

subroutine remap_y_conservative(old_r, old_inner, old_outer, old_y, n_old, &
                                 new_r, new_inner, new_outer, new_y, n_new, e_removed)
  integer, intent(in) :: n_old, n_new
  real(8), intent(in) :: old_r(n_old), old_inner, old_outer, old_y(n_old)
  real(8), intent(in) :: new_r(n_new), new_inner, new_outer
  real(8), intent(out) :: new_y(n_new), e_removed
  integer :: i, j, i_start
  real(8), allocatable :: e_new(:)
  real(8) :: old_l, old_rface, old_vol, new_l, new_rface, new_vol
  real(8) :: rhoe_old, rlo, rhi, overlap_vol, rlo3, rhi3, new_l3, new_r3
  real(8), allocatable :: old_face_l(:), old_face_r(:), old_face_vol(:)
  real(8), allocatable :: new_face_l(:), new_face_r(:), new_face_vol(:)
  real(8), parameter :: four_pi_over_3 = 4.18879020478639098461685784438d0

  allocate(e_new(n_new))
  e_new = 0d0
  e_removed = 0d0

  allocate(old_face_l(n_old), old_face_r(n_old), old_face_vol(n_old))
  allocate(new_face_l(n_new), new_face_r(n_new), new_face_vol(n_new))
  call prepare_zone_geometry_arrays(old_r, old_inner, old_outer, n_old, old_face_l, old_face_r, old_face_vol)
  call prepare_zone_geometry_arrays(new_r, new_inner, new_outer, n_new, new_face_l, new_face_r, new_face_vol)

  i_start = 1
  do j = 1, n_old
   old_l = old_face_l(j)
   old_rface = old_face_r(j)
   old_vol = old_face_vol(j)
   rhoe_old = a_rad * max(old_y(j), 0d0)
   if (rhoe_old <= 0d0) cycle

   if (old_l < new_inner) then
    rlo = old_l
    rhi = min(old_rface, new_inner)
    if (rhi > rlo) then
     e_removed = e_removed + rhoe_old * four_pi_over_3 * (rhi**3 - rlo**3)
    end if
   end if

   rlo = max(old_l, new_inner)
   rhi = min(old_rface, new_inner + (new_face_r(n_new) - new_inner))
   if (rhi <= rlo) cycle
   rlo3 = rlo**3
   rhi3 = rhi**3

   do i = i_start, n_new
    new_l = new_face_l(i)
    if (new_l >= rhi) exit
    new_rface = new_face_r(i)
    if (new_rface <= rlo) then
     i_start = i + 1
     cycle
    end if
    new_vol = new_face_vol(i)
    new_l3 = max(rlo3, new_l**3)
    new_r3 = min(rhi3, new_rface**3)
    if (new_r3 > new_l3) then
     overlap_vol = four_pi_over_3 * (new_r3 - new_l3)
     e_new(i) = e_new(i) + rhoe_old * overlap_vol
    end if
   end do
  end do

  do i = 1, n_new
   new_vol = new_face_vol(i)
   new_y(i) = max(e_new(i) / max(a_rad * new_vol, 1d-30), 0d0)
  end do

  deallocate(old_face_l, old_face_r, old_face_vol)
  deallocate(new_face_l, new_face_r, new_face_vol)
  deallocate(e_new)
end subroutine remap_y_conservative

subroutine update_interaction_grid(state, r_shell, t_shell, m_shell)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: r_shell, t_shell, m_shell

  if (.not. state%initialized) return
  call update_interaction_geometry(state, r_shell, t_shell, m_shell)
  call update_tau_and_luminosity(state)
 end subroutine update_interaction_grid

subroutine shell_structure_estimate(state, r_shell, t_shell, m_shell, delta_r, tau_sh, tleak, m_csm_shell, m_ej_shell, delta_r_csm_out, delta_r_ej_out)
  type(transport_state_type), intent(in) :: state
  real(8), intent(in) :: r_shell, t_shell, m_shell
  real(8), intent(in), optional :: m_csm_shell, m_ej_shell
  real(8), intent(out), optional :: delta_r_csm_out, delta_r_ej_out
  real(8), intent(out) :: delta_r, tau_sh, tleak
  real(8) :: rho_csm, rho_ej, m_csm_eff, m_ej_eff
  real(8) :: delta_r_csm, delta_r_ej
  real(8) :: v_csm, v_ej, u_eff, comp_csm, comp_ej

  rho_csm = max(query_csm_density(r_shell, t_shell, op(2)), 1d-30)
  rho_ej = max(query_ejecta_density(r_shell, t_shell, op(1)), 1d-30)
  v_csm = query_csm_velocity(r_shell, t_shell, op(2))
  v_ej = query_ejecta_velocity(r_shell, t_shell, op(1))
  u_eff = max(state%u_emerge_shell, 1d-30)

  if (present(m_csm_shell)) then
   m_csm_eff = max(m_csm_shell, 0d0)
  else
   m_csm_eff = max(state%m_shocked_csm, 0d0)
  end if
  if (present(m_ej_shell)) then
   m_ej_eff = max(m_ej_shell, 0d0)
  else
   m_ej_eff = max(state%m_shocked_ej, 0d0)
  end if

  if (m_csm_eff + m_ej_eff <= 0d0) then
   m_csm_eff = 0.5d0 * max(m_shell, 0d0)
   m_ej_eff = 0.5d0 * max(m_shell, 0d0)
  end if

  ! Use the canonical strong-shock compression for gamma = 5/3. Higher
  ! compression over-thins the unresolved shell and releases the stored
  ! breakout energy too early.
  comp_csm = 4d0
  comp_ej = 4d0

  delta_r_csm = m_csm_eff / (4d0*pi*max(r_shell,1d0)**2 * comp_csm*rho_csm)
  delta_r_ej = m_ej_eff / (4d0*pi*max(r_shell,1d0)**2 * comp_ej*rho_ej)
  if (present(delta_r_csm_out)) delta_r_csm_out = delta_r_csm
  if (present(delta_r_ej_out)) delta_r_ej_out = delta_r_ej
  delta_r = max(delta_r_csm + delta_r_ej, 1d-6*max(r_shell,1d0))
  tau_sh = state%kappa * (comp_csm*rho_csm*delta_r_csm + comp_ej*rho_ej*delta_r_ej)
  tleak = max(delta_r/clight, tau_sh*delta_r/clight)
end subroutine shell_structure_estimate

real(8) function forward_shock_radius(state, r_shell, t_shell, m_shell) result(r_fs)
  type(transport_state_type), intent(in) :: state
  real(8), intent(in) :: r_shell, t_shell, m_shell
  integer :: i
  real(8) :: target_mass, cum_mass, cell_mass
  real(8) :: r_face_l, r_face_r, vol_i, rho_i, r3

  if (allocated(state%radius_ref) .and. allocated(state%rho_ref) .and. state%r_outer_support > state%r_inner_support) then
   target_mass = max(state%m_shocked_csm, 0d0)
   cum_mass = 0d0
   r_fs = state%r_inner_support

   if (target_mass <= 0d0) then
    r_fs = max(state%r_inner_support, min(max(r_shell, 1d0), state%r_outer_support))
    return
   end if

   do i = 1, state%n_zones
    call zone_geometry_from_array(state%radius_ref, state%r_inner_support, state%r_outer_support, state%n_zones, i, &
         r_face_l, r_face_r, vol_i)
    rho_i = max(state%rho_ref(i), 1d-30)
    cell_mass = rho_i * vol_i
    if (cum_mass + cell_mass >= target_mass) then
     r3 = r_face_l**3 + 3d0 * (target_mass - cum_mass) / (4d0 * pi * rho_i)
     r_fs = min(max(r3, r_face_l**3), r_face_r**3) ** (1d0 / 3d0)
     return
    end if
    cum_mass = cum_mass + cell_mass
   end do
   r_fs = state%r_outer_support
  else
   r_fs = max(r_shell, 1d0)
  end if
end function forward_shock_radius

real(8) function shell_leakage_timescale(state, r_shell, t_shell, m_shell) result(tleak)
  type(transport_state_type), intent(in) :: state
  real(8), intent(in) :: r_shell, t_shell, m_shell
  real(8) :: delta_r, tau_sh

  call shell_structure_estimate(state, r_shell, t_shell, m_shell, delta_r, tau_sh, tleak)
end function shell_leakage_timescale

real(8) function total_radiation_energy(state) result(e_tot)
  type(transport_state_type), intent(in) :: state
  integer :: i
  real(8) :: r_face_l, r_face_r, vol_i

  e_tot = 0d0
  if (.not. state%initialized) return

  do i = 1, state%n_zones
   call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   e_tot = e_tot + a_rad * max(state%y(i), 0d0) * vol_i
  end do
end function total_radiation_energy

real(8) function interaction_shell_leakage_timescale(state, r_shell, t_shell, m_shell) result(tleak)
  type(transport_state_type), intent(in) :: state
  real(8), intent(in) :: r_shell, t_shell, m_shell
  real(8) :: tleak_shell, tau_ext, r_ph_ext, delta_ext

  tleak_shell = shell_leakage_timescale(state, r_shell, t_shell, m_shell)
  tau_ext = max(query_tau_to_edge(max(r_shell, state%r_inner), t_shell, op(2), state%kappa) - 2d0/3d0, 0d0)
  r_ph_ext = query_csm_photosphere_radius(max(r_shell, state%r_inner), t_shell, op(2), state%kappa)
  delta_ext = max(r_ph_ext - max(r_shell, state%r_inner), 0d0)

  if (delta_ext > 0d0 .and. tau_ext > 0d0) then
   tleak = tleak_shell + max(delta_ext / clight, tau_ext * delta_ext / clight)
  else
   tleak = tleak_shell
  end if
 end function interaction_shell_leakage_timescale

subroutine interaction_transport_step(state, dt, r_shell, u_shell, t_shell, m_shell, lum_heat, lum_obs, r_ph)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: dt, r_shell, u_shell, t_shell, m_shell, lum_heat
  real(8), intent(out) :: lum_obs, r_ph
  real(8) :: lum_input
  real(8) :: u_grid, r_face_l, r_face_r, vol_i
  real(8) :: e_tot_before, e_tot_after_target, e_res_before
  integer :: i

  e_res_before = max(state%e_residual, 0d0)
  e_tot_before = e_res_before
  if (state%initialized) e_tot_before = e_tot_before + total_radiation_energy(state)
  state%u_shell_current = u_shell
  state%lum_heat_last = lum_heat
  if (.not. state%initialized) then
   call initialize_interaction_grid(state, max(r_shell, 1d0), t_shell, m_shell, max(state%kappa, 1d-30), max(state%n_zones,48))
  else
   call update_interaction_grid(state, max(r_shell, 1d0), t_shell, m_shell)
  end if

  lum_input = max(lum_heat, 0d0)

  if (state%interaction_rannacher_left > 0) then
   call solve_transport_step(state, dt, lum_input, 1d0)
   state%interaction_rannacher_left = state%interaction_rannacher_left - 1
  else
   call solve_transport_step(state, dt, lum_input, 0.5d0)
  end if
  call find_transport_photosphere(state, r_ph, lum_obs)

  if (state%initialized) then
   u_grid = 0d0
   do i = 1, state%n_zones
    call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
    u_grid = u_grid + a_rad * max(state%y(i), 0d0) * vol_i
   end do
   e_tot_after_target = max(e_tot_before + (lum_input - max(lum_obs, 0d0)) * max(dt, 0d0), 0d0)
   state%e_residual = max(e_tot_after_target - u_grid, 0d0)

   ! e_swept is accumulated in solve_transport_step (dt*lum_heat at ihi).
   ! No additional accumulation needed here; e_residual is updated above.
  end if
end subroutine interaction_transport_step

subroutine initialize_cooling_state_from_interaction(state, r_shell, u_shell, m_shell, t_shell, lum_target, lum_heat, u_reservoir, &
     pre_y_in, pre_radius_in, pre_r_inner_in, pre_r_outer_in)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: r_shell, u_shell, m_shell, t_shell
  real(8), intent(in), optional :: lum_target, lum_heat
  real(8), intent(in), optional :: u_reservoir
  ! Optional: pre-collapse radiation field from before interaction_transport_step.
  ! When provided, these are used as the IC source rather than the (collapsed) state%y.
  real(8), intent(in), optional :: pre_y_in(:), pre_radius_in(:)
  real(8), intent(in), optional :: pre_r_inner_in, pre_r_outer_in

  integer :: i
  real(8), allocatable :: old_y(:), old_radius(:), y_map(:)
  real(8) :: u_grid, tau_shell, r_face_l, r_face_r, vol_i
  real(8) :: u_map, e_removed_map
  real(8) :: old_inner_active, old_outer_active
  logical :: debug_mode3

  debug_mode3 = mode3_debug_enabled()

  if (.not. state%initialized) return

  state%in_cooling_phase = .true.
  state%cooling_initialized = .true.
  state%t_emerge = t_shell
  state%t_gap_end = t_shell
  state%lum_heat_gap = 0d0
  ! R_0 = shell position at emergence = r_csm,out (per paper §3.2)
  state%r_emerge_shell = max(r_shell, 1d0)
  state%u_emerge_shell = max(u_shell, 1d-30)
  state%m_emerge_csm = max(state%m_shocked_csm, 0d0)
  state%m_emerge_ej = max(state%m_shocked_ej, 0d0)
  state%m_emerge_shell = max(m_shell, state%m_shocked_csm + state%m_shocked_ej, 1d-30)

  ! Use pre-collapse radiation field if provided (avoids grid-collapse energy loss).
  ! When r_shell >= r_outer_support, interaction_transport_step collapses the grid
  ! to near-zero extent and the conservative remap discards all radiation energy.
  ! By saving state%y BEFORE interaction_transport_step, we preserve the IC.
  allocate(old_y(state%n_zones))
  allocate(old_radius(state%n_zones))
  allocate(y_map(state%n_zones))
  if (present(pre_y_in) .and. present(pre_radius_in) .and. &
      present(pre_r_inner_in) .and. present(pre_r_outer_in)) then
   old_y = pre_y_in
   old_radius = pre_radius_in
   old_inner_active = pre_r_inner_in
   old_outer_active = pre_r_outer_in
  else
   old_y = state%y
   old_radius = state%radius
   old_inner_active = state%r_inner
   old_outer_active = state%r_outer
  end if

  ! Measure energy in interaction grid (paper IC: e_int(x) = e(x, y_se))
  u_grid = 0d0
  do i = 1, state%n_zones
   call zone_geometry_from_array(old_radius, old_inner_active, old_outer_active, state%n_zones, i, &
        r_face_l, r_face_r, vol_i)
   u_grid = u_grid + a_rad * max(old_y(i), 0d0) * vol_i
  end do

  ! Cooling domain: full CSM from r_inner_support to r_outer_support.
  ! The comoving coordinate x = r/R_in(t) scales the domain homologously.
  ! Reference grid is the unscaled (t=t_emerge) CSM.
  state%r_emerge_outer = state%r_outer_support
  state%r_emerge_inner = state%r_inner_support
  if (.not.(state%r_emerge_outer > state%r_emerge_inner)) then
   state%r_emerge_inner = state%r_emerge_outer * (1d0 - 1d-6)
  end if

  state%cooling_scale = 1d0
  state%cooling_scale_prev = 1d0
  state%compact_tail_drain = 0d0
  state%cooling_rannacher_left = 4
  state%r_inner = state%r_emerge_inner
  state%r_outer = state%r_emerge_outer

  ! Build reference grid and fill density profile from the CSM at emergence.
  call build_radial_grid(state%r_emerge_inner, state%r_emerge_outer, state%radius_ref)
  do i = 1, state%n_zones
   state%radius(i) = state%radius_ref(i)
   state%rho_ref(i) = max(query_csm_density(state%radius_ref(i), t_shell, op(2)), 1d-30)
   state%rho(i) = state%rho_ref(i)
   state%y(i) = 0d0
  end do

  ! IC (paper eq. 948): e(x, y_se) = e_int(x) — remap the interaction-phase
  ! radiation field onto the cooling grid conservatively. No ad-hoc corrections.
  call remap_y_conservative(old_radius, old_inner_active, old_outer_active, old_y, state%n_zones, &
       state%radius_ref, state%r_emerge_inner, state%r_emerge_outer, y_map, state%n_zones, e_removed_map)
  state%y = max(y_map, 0d0)

  call update_tau_and_luminosity(state)
  tau_shell = state%tau(1)

  if (debug_mode3) then
   u_map = 0d0
   do i = 1, state%n_zones
    call zone_geometry_from_array(state%radius_ref, state%r_emerge_inner, state%r_emerge_outer, state%n_zones, i, &
         r_face_l, r_face_r, vol_i)
    u_map = u_map + a_rad * max(state%y(i), 0d0) * vol_i
   end do
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
        'M3_HANDOFF_GEOM rin=', state%r_emerge_inner, 'rsh=', state%r_emerge_shell, 'rout=', state%r_emerge_outer, 'tau0=', tau_shell
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
        'M3_HANDOFF_M Mcsm=', state%m_emerge_csm, 'Mej=', state%m_emerge_ej, 'Mtot=', state%m_emerge_csm+state%m_emerge_ej
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
        'M3_HANDOFF_EN Ugrid=', u_grid, 'Umap=', u_map, 'Erem=', e_removed_map
  end if

  state%e_swept = 0d0
  state%e_residual = 0d0

  deallocate(old_y)
  deallocate(old_radius)
  deallocate(y_map)
end subroutine initialize_cooling_state_from_interaction

subroutine update_cooling_grid(state, t_shell)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: t_shell
  real(8) :: prev_scale, next_scale
  integer :: i

  if (.not. state%initialized) return

  state%t_shell = t_shell
  prev_scale = max(state%cooling_scale, 1d-30)
  state%cooling_scale_prev = prev_scale
  next_scale = max((state%r_emerge_shell + state%u_emerge_shell * max(t_shell - state%t_emerge, 0d0)) &
                   / max(state%r_emerge_shell, 1d0), 1d0)
  state%cooling_scale = next_scale

  state%r_inner = state%r_emerge_inner * next_scale
  state%r_outer = state%r_emerge_outer * next_scale
  do i = 1, state%n_zones
   state%radius(i) = state%radius_ref(i) * next_scale
   state%rho(i) = max(state%rho_ref(i) / next_scale**3, 1d-30)
  end do
  state%y = max(state%y * (prev_scale / next_scale)**4, 0d0)
  call update_tau_and_luminosity(state)
 end subroutine update_cooling_grid

subroutine zone_geometry_from_array(radius, r_inner, r_outer, n, i, r_face_l, r_face_r, vol_i)
 integer, intent(in) :: n, i
 real(8), intent(in) :: radius(n), r_inner, r_outer
 real(8), intent(out) :: r_face_l, r_face_r, vol_i

  if (i == 1) then
   r_face_l = r_inner
  else
   r_face_l = sqrt(radius(i-1) * radius(i))
  end if
  if (i == n) then
   r_face_r = r_outer
  else
   r_face_r = sqrt(radius(i) * radius(i+1))
  end if
  vol_i = 4d0 * pi * max(r_face_r**3 - r_face_l**3, 1d-30) / 3d0
end subroutine zone_geometry_from_array

subroutine prepare_zone_geometry_arrays(radius, r_inner, r_outer, n, r_face_l, r_face_r, vol)
  integer, intent(in) :: n
  real(8), intent(in) :: radius(n), r_inner, r_outer
  real(8), intent(out) :: r_face_l(n), r_face_r(n), vol(n)
  integer :: i
  real(8) :: r3_l, r3_r
  real(8), parameter :: four_pi_over_3 = 4.18879020478639098461685784438d0

  if (n <= 0) return
  r_face_l(1) = r_inner
  if (n > 1) then
   do i = 2, n
    r_face_l(i) = sqrt(radius(i-1) * radius(i))
   end do
  end if
  if (n > 1) then
   do i = 1, n-1
    r_face_r(i) = sqrt(radius(i) * radius(i+1))
   end do
  end if
  r_face_r(n) = r_outer
  do i = 1, n
   r3_l = r_face_l(i)**3
   r3_r = r_face_r(i)**3
   vol(i) = four_pi_over_3 * max(r3_r - r3_l, 1d-30)
  end do
end subroutine prepare_zone_geometry_arrays

subroutine cooling_transport_step(state, dt, t_shell, lum_obs, r_ph, lum_heat)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: dt, t_shell
  real(8), intent(out) :: lum_obs, r_ph
  real(8), intent(in), optional :: lum_heat
  real(8) :: lum_heat_local
  real(8) :: theta_use
  if (.not. state%initialized) then
   lum_obs = 0d0
   r_ph = 0d0
   return
  end if
  if (.not. state%cooling_initialized) then
   call initialize_cooling_state_from_interaction(state, state%r_shell_current, &
        max(state%u_shell_current, 1d-30), max(state%m_shocked_csm + state%m_shocked_ej, 1d-30), t_shell, &
        lum_heat=state%lum_heat_last)
  end if

  lum_heat_local = 0d0
  if (present(lum_heat)) lum_heat_local = max(lum_heat, 0d0)
  call update_cooling_grid(state, t_shell)
  if (state%cooling_rannacher_left > 0) then
   theta_use = 1d0
   state%cooling_rannacher_left = state%cooling_rannacher_left - 1
  else
   theta_use = 0.5d0
  end if
  call solve_transport_step(state, dt, lum_heat_local, theta_use)
  call find_transport_photosphere(state, r_ph, lum_obs)
  ! debug print disabled
end subroutine cooling_transport_step

subroutine comoving_transport_step(state, dt, t_shell, r_sh, v_sh, m_sh, lum_heat, lum_obs, r_ph_out)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: dt, t_shell, r_sh, v_sh, m_sh, lum_heat
  real(8), intent(out) :: lum_obs
  real(8), intent(out), optional :: r_ph_out

  integer :: nsub, isub
  integer :: nz_init
  real(8) :: t_loc, t_next, dt_step, dt_move, dt_rad, subratio
  real(8) :: r_loc, u_loc, m_loc
  real(8) :: m_csm_loc, m_ej_loc, dm_csm_dt, dm_ej_dt
  real(8) :: ku, kr, km
  real(8) :: ku_mid, kr_mid, t_mid
  real(8) :: u_mid, r_mid, m_mid, m_ej_mid, m_csm_mid
  real(8) :: dm_ej_mid, dm_csm_mid
  real(8) :: r_out, r_ph, lum_tmp, r_fs_now, r_fs_prev
  real(8) :: heat_sub
  real(8) :: e_rad
  real(8) :: du_res, e_prev, e_res_prev, ebal
  real(8) :: r_init, kappa_init
  ! Pre-collapse radiation field saved before interaction_transport_step
  real(8), allocatable :: pre_y(:), pre_radius(:)
  real(8) :: pre_r_inner, pre_r_outer
  integer :: dbg_stride, call_count
  logical :: do_log_step, do_log_sub
  logical :: stop_after_sub
  integer, save :: mode3_call_count = 0
  logical :: debug_mode3

  debug_mode3 = mode3_debug_enabled()

  lum_obs = 0d0
  r_ph = 0d0

  if (.not. state%initialized) then
   m_loc = max(m_sh, 1d-30)
   nz_init = max(state%n_zones, 48)
   kappa_init = max(state%kappa, 1d-30)
   state%n_zones = nz_init
   state%kappa = kappa_init
   state%comoving_initialized = .true.
   r_init = max(r_sh, 1d0)
   state%r_shell_current = r_init
   state%u_shell_current = max(v_sh, 1d5)
   state%m_shocked_csm = m_loc
   state%m_shocked_ej = 0d0
   state%e_residual = 0d0
   state%t_shell = t_shell
   ! Initialize transport geometry before timestep selection so the very first
   ! step respects the diffusion timescale (critical for the early dark phase).
   call initialize_interaction_grid(state, r_init, t_shell, &
        m_loc, kappa_init, nz_init)
   state%u_shell_current = max(v_sh, 1d5)
   state%m_shocked_csm = m_loc
   state%m_shocked_ej = 0d0
   state%t_shell = t_shell
  end if

  t_loc = max(state%t_shell, t_shell)
  t_next = t_shell + max(dt, 0d0)
  if (t_next <= t_loc + 1d-20) return

  r_loc = max(state%r_shell_current, 1d0)
  u_loc = max(state%u_shell_current, 1d5)
  m_csm_loc = max(state%m_shocked_csm, 0d0)
  m_ej_loc = max(state%m_shocked_ej, 0d0)
  m_loc = max(m_sh, m_csm_loc + m_ej_loc, 1d-30)
  if (m_csm_loc + m_ej_loc <= 0d0) then
   m_csm_loc = m_loc
   m_ej_loc = 0d0
  end if
  state%m_shocked_csm = max(m_csm_loc, 0d0)
  state%m_shocked_ej = max(m_ej_loc, 0d0)

  dt_move = shock_motion_timestep_limit_at(state, r_loc, u_loc)
  dt_rad = transport_timestep_limit(state)
  dt_step = t_next - t_loc
  if (dt_move > 0d0 .and. dt_move < huge(1d0)) dt_step = min(dt_step, 0d0 + dt_move)
  if (dt_rad > 0d0 .and. dt_rad < huge(1d0)) dt_step = min(dt_step, 0.5d0*dt_rad)
  if (state%in_cooling_phase .and. state%t_emerge > 0d0) then
   if (t_loc - state%t_emerge < 0.05d0*86400d0) then
    dt_step = min(dt_step, 0.0005d0*86400d0)
   else if (t_loc - state%t_emerge < 0.2d0*86400d0) then
    dt_step = min(dt_step, 0.002d0*86400d0)
   end if
  end if
  dt_step = max(min(dt_step, 0.2d0*86400d0), 1d-6)
  subratio = (t_next - t_loc) / max(dt_step, 1d-30)
  if (subratio >= 256d0) then
   nsub = 256
  else
   nsub = max(1, int(ceiling(subratio)))
  end if
  dt_step = (t_next - t_loc) / dble(nsub)
  dbg_stride = max(1, nsub / 4)
  mode3_call_count = mode3_call_count + 1
  call_count = mode3_call_count
  do_log_step = (call_count <= 25 .or. mod(call_count, 200) == 0)

  if (debug_mode3 .and. do_log_step) then
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,I0,1X,A,1X,L1)') &
        'M3_ENTER t=', t_shell, 'dt=', dt, 'r=', r_loc, 'u=', u_loc, 'nsub=', nsub, 'cool=', state%in_cooling_phase
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
        'M3_DTLIM move=', dt_move, 'rad=', dt_rad, 'heat=', lum_heat
  end if

  do isub = 1, nsub
   stop_after_sub = .false.
   do_log_sub = (isub == 1 .or. isub == nsub .or. mod(isub, dbg_stride) == 0)
   e_prev = total_radiation_energy(state)
   if (.not. state%in_cooling_phase) then
    if (debug_mode3 .and. do_log_step .and. do_log_sub) then
     write(*,'(A,1X,I0,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
          'M3_INT_PRE isub=', isub, 't=', t_loc, 'r=', r_loc, 'u=', u_loc, 'm=', m_loc
    end if

    e_res_prev = state%e_residual
    state%m_shocked_csm = max(m_csm_loc, 0d0)
    state%m_shocked_ej = max(m_ej_loc, 0d0)
    r_fs_prev = r_loc  ! emergence criterion: shell reaches r_csm,out (paper §3.1)
    ku = dudt(u_loc, r_loc, m_loc, t_loc, op)
    kr = drdt(u_loc, r_loc, m_loc, t_loc, op)
    dm_ej_dt = 4d0*pi*r_loc*r_loc*query_ejecta_density(r_loc, t_loc, op(1)) * &
               (query_ejecta_velocity(r_loc, t_loc, op(1)) - u_loc)
    dm_csm_dt = 4d0*pi*r_loc*r_loc*query_csm_density(r_loc, t_loc, op(2)) * &
                (u_loc - query_csm_velocity(r_loc, t_loc, op(2)))
    dm_ej_dt = max(dm_ej_dt, 0d0)
    dm_csm_dt = max(dm_csm_dt, 0d0)

    ! Midpoint integration damps the early-time Euler overshoot in w and Msh,
    ! which otherwise makes the breakout transition too impulsive.
    t_mid = t_loc + 0.5d0*dt_step
    u_mid = max(u_loc + 0.5d0*ku*dt_step, 1d5)
    r_mid = max(r_loc + 0.5d0*kr*dt_step, 1d0)
    m_ej_mid = max(m_ej_loc + 0.5d0*dm_ej_dt*dt_step, 0d0)
    m_csm_mid = max(m_csm_loc + 0.5d0*dm_csm_dt*dt_step, 0d0)
    m_mid = max(m_ej_mid + m_csm_mid, 1d-30)

    ku_mid = dudt(u_mid, r_mid, m_mid, t_mid, op)
    kr_mid = drdt(u_mid, r_mid, m_mid, t_mid, op)
    dm_ej_mid = 4d0*pi*r_mid*r_mid*query_ejecta_density(r_mid, t_mid, op(1)) * &
                (query_ejecta_velocity(r_mid, t_mid, op(1)) - u_mid)
    dm_csm_mid = 4d0*pi*r_mid*r_mid*query_csm_density(r_mid, t_mid, op(2)) * &
                 (u_mid - query_csm_velocity(r_mid, t_mid, op(2)))
    dm_ej_mid = max(dm_ej_mid, 0d0)
    dm_csm_mid = max(dm_csm_mid, 0d0)

    u_loc = max(u_loc + ku_mid*dt_step, 1d5)
    r_loc = max(r_loc + kr_mid*dt_step, 1d0)
    m_ej_loc = max(m_ej_loc + dm_ej_mid*dt_step, 0d0)
    m_csm_loc = max(m_csm_loc + dm_csm_mid*dt_step, 0d0)
    m_loc = max(m_ej_loc + m_csm_loc, 1d-30)
    state%m_shocked_csm = max(m_csm_loc, 0d0)
    state%m_shocked_ej = max(m_ej_loc, 0d0)

    heat_sub = max(lum_heat, 0d0)
    ! Save radiation field and grid BEFORE interaction_transport_step collapses
    ! the grid when r_shell >= r_outer_support (Fix 8: grid collapse at emergence).
    if (.not. allocated(pre_y)) then
     allocate(pre_y(state%n_zones))
     allocate(pre_radius(state%n_zones))
    end if
    if (debug_mode3 .and. r_loc >= state%r_outer_support * 0.98d0) then
     write(*,'(A,1X,I0,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
          'M3_PRESAVE isub=', isub, 'r=', r_loc, 'rout=', state%r_outer_support, &
          'ysum=', sum(state%y), 'rin=', state%r_inner
    end if
    pre_y = state%y
    pre_radius = state%radius
    pre_r_inner = state%r_inner
    pre_r_outer = state%r_outer
    call interaction_transport_step(state, dt_step, r_loc, u_loc, t_loc + dt_step, m_loc, heat_sub, lum_tmp, r_ph)
    e_rad = total_radiation_energy(state)
    du_res = state%e_residual - e_res_prev
    ebal = (heat_sub - max(lum_tmp, 0d0)) * dt_step - ((e_rad + state%e_residual) - (e_prev + e_res_prev))

    if (debug_mode3 .and. do_log_step .and. do_log_sub) then
     write(*,'(A,1X,I0,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
          'M3_INT_POST isub=', isub, 't=', t_loc+dt_step, 'L=', lum_tmp, 'rph=', r_ph, 'erad=', e_rad, 'tau0=', state%tau(1)
     write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5)') 'M3_RES dE=', du_res, 'Eres=', state%e_residual
     write(*,'(A,1X,ES12.5)') 'M3_EBAL=', ebal
    end if

    r_out = query_csm_outer_edge(t_loc + dt_step, op(2))
    ! Paper §3.1: emergence when the thin-shell position reaches r_csm,out,
    ! i.e. x(ζ_se) = R_csm,out / R_csm,in. Use r_loc (shell radius) directly.
    r_fs_now = r_loc  ! retained for debug print compatibility
    if (.not. state%in_cooling_phase .and. r_fs_prev < r_out .and. r_loc >= r_out) then
     if (debug_mode3) then
      write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
           'M3_EMERGE t=', t_loc+dt_step, 'r=', r_loc, 'rfs=', r_fs_now, 'rout=', r_out, 'L=', lum_tmp
      write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5)') 'M3_EMERGE_URES=', state%e_residual, 'Erad=', e_rad
      write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5)') 'M3_PRE_Y_SUM=', sum(pre_y), 'STATE_Y_SUM=', sum(state%y)
      write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
           'M3_PRE_GEOM rin=', pre_r_inner, 'rout=', pre_r_outer, 'state_rin=', state%r_inner, 'state_rout=', state%r_outer
     end if
      call initialize_cooling_state_from_interaction(state, r_loc, u_loc, m_loc, t_loc + dt_step, lum_tmp, heat_sub, &
           pre_y_in=pre_y, pre_radius_in=pre_radius, &
           pre_r_inner_in=pre_r_inner, pre_r_outer_in=pre_r_outer)
      stop_after_sub = .true.
    end if
   else
    call cooling_transport_step(state, dt_step, t_loc + dt_step, lum_tmp, r_ph, 0d0)
    r_loc = max(r_loc + u_loc*dt_step, 1d0)
    e_rad = total_radiation_energy(state)
    ebal = e_rad - e_prev + max(lum_tmp, 0d0) * dt_step
    if (debug_mode3 .and. do_log_step .and. do_log_sub) then
     write(*,'(A,1X,I0,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
          'M3_COOL isub=', isub, 't=', t_loc+dt_step, 'L=', lum_tmp, 'rph=', r_ph, 'erad=', e_rad
     write(*,'(A,1X,ES12.5)') 'M3_EBAL=', ebal
    end if
   end if

   if (.not.(lum_tmp == lum_tmp) .or. abs(lum_tmp) > 1d60) then
    write(*,'(A,1X,I0,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,L1)') &
         'M3_WARN_L isub=', isub, 't=', t_loc+dt_step, 'L=', lum_tmp, 'cool=', state%in_cooling_phase
   end if
   if (maxval(abs(state%y)) > 1d40 .or. any(state%y /= state%y)) then
    write(*,'(A,1X,I0,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
         'M3_WARN_Y isub=', isub, 't=', t_loc+dt_step, 'ymax=', maxval(abs(state%y)), 'tau0=', state%tau(1)
   end if

   t_loc = t_loc + dt_step
   lum_obs = max(lum_tmp, 0d0)
   if (stop_after_sub) exit
  end do

  state%r_shell_current = r_loc
  state%u_shell_current = max(u_loc, 0d0)
  state%m_shocked_csm = max(m_csm_loc, 0d0)
  state%m_shocked_ej = max(m_ej_loc, 0d0)
  state%t_shell = t_loc
  if (present(r_ph_out)) then
   call find_transport_photosphere(state, r_ph_out, lum_obs)
  end if
  if (debug_mode3 .and. do_log_step) then
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,L1)') &
        'M3_EXIT t=', state%t_shell, 'L=', lum_obs, 'r=', state%r_shell_current, 'cool=', state%in_cooling_phase
  end if
  if (allocated(pre_y)) deallocate(pre_y)
  if (allocated(pre_radius)) deallocate(pre_radius)
end subroutine comoving_transport_step

logical function shock_has_emerged(r_shell, t_shell)
  real(8), intent(in) :: r_shell, t_shell
  real(8) :: r_out

  r_out = query_csm_outer_edge(t_shell, op(2))
  shock_has_emerged = (r_out < huge(1d0) .and. r_shell >= r_out)
 end function shock_has_emerged

subroutine photosphere_boundary_from_tau(state, istart, iph, r_ph)
  type(transport_state_type), intent(in) :: state
  integer, intent(in) :: istart
  integer, intent(out) :: iph
  real(8), intent(out) :: r_ph
  integer :: i, jstart
  real(8) :: tau_lo, tau_hi, frac, r_floor

  jstart = min(max(istart, 1), state%n_zones)
  if (state%in_cooling_phase) then
   r_floor = max(state%r_inner, state%radius(jstart))
  else
   r_floor = max(state%r_shell_current, state%radius(jstart))
  end if
  ! Default: photosphere at the outer CSM boundary
  iph = state%n_zones
  r_ph = state%r_outer

  do i = jstart + 1, state%n_zones
   if (state%tau(i) <= 2d0/3d0) then
    iph = i - 1
    tau_lo = state%tau(i-1)
    tau_hi = state%tau(i)
    frac = (2d0/3d0 - tau_lo) / max(tau_hi - tau_lo, 1d-30)
    frac = min(max(frac, 0d0), 1d0)
    r_ph = state%radius(i-1) + frac * (state%radius(i) - state%radius(i-1))
    r_ph = min(max(r_ph, state%radius(i-1)), state%radius(i))
    r_ph = max(r_ph, r_floor)
    return
   end if
  end do
end subroutine photosphere_boundary_from_tau

real(8) function outer_escape_conductance(state, iph, r_ph) result(k_escape)
  type(transport_state_type), intent(in) :: state
  integer, intent(in) :: iph
  real(8), intent(in) :: r_ph
  real(8) :: r_face_l, r_face_r, vol_i, area_ph, stream_factor
  real(8) :: dr_half, dtau_edge, escape_factor

  if (iph < 1 .or. iph > state%n_zones) then
   k_escape = 0d0
   return
  end if

  call zone_geometry(state, iph, r_face_l, r_face_r, vol_i)
  r_face_r = min(max(r_ph, r_face_l), r_face_r)
  area_ph = 4d0 * pi * r_face_r**2
  if (state%in_cooling_phase) then
   ! Cooling phase outer Robin BC (paper §3.2, eq. 917-923):
   ! The f_ob(y) factor differs from interaction by (R_in/R_0)^2/2 relative to
   ! the interaction factor of 4. The conductance is:
   !   stream_factor = (1/4) * (1/2) * (R_in/R_0)^2 = cooling_scale^2 / 8
   ! where cooling_scale = R_in(t)/R_0 = 1 at emergence and grows with time.
   stream_factor = 0.125d0 * max(state%cooling_scale, 1d0)**2
  else
   ! Interaction phase: Eddington closure at τ=2/3 photosphere: F = (c/4) u.
   stream_factor = 0.25d0
  end if
  k_escape = area_ph * clight * a_rad * stream_factor
end function outer_escape_conductance

subroutine find_transport_photosphere(state, r_ph, lum_obs)
  type(transport_state_type), intent(in) :: state
  real(8), intent(out) :: r_ph, lum_obs
  integer :: iph, ihi
  real(8) :: y_ph, tau_lo, tau_hi, frac, y_obs, r_obs
  real(8) :: t_relax, ramp
  real(8) :: r_face_l, r_face_r, vol_i, dr, rho_face, dcoef, e_n, e_nm1, flux_out

  ihi = 1
  if (.not. state%in_cooling_phase) ihi = first_active_zone(state)
  call photosphere_boundary_from_tau(state, ihi, iph, r_ph)
  call zone_geometry(state, iph, r_face_l, r_face_r, vol_i)
  y_obs = max(state%y(iph), 0d0)
  r_obs = min(max(r_ph, r_face_l), r_face_r)
  if (iph < state%n_zones) then
   tau_lo = state%tau(iph)
   tau_hi = state%tau(iph+1)
   if (tau_lo >= 2d0/3d0 .and. tau_hi <= 2d0/3d0) then
    frac = (2d0/3d0 - tau_lo) / max(tau_hi - tau_lo, 1d-30)
    frac = min(max(frac, 0d0), 1d0)
    r_obs = state%radius(iph) + frac * (state%radius(iph+1) - state%radius(iph))
    y_obs = state%y(iph) + frac * (state%y(iph+1) - state%y(iph))
    y_obs = max(y_obs, 0d0)
   end if
  end if
  e_n = a_rad * y_obs
  if (state%in_cooling_phase) then
   ! Cooling phase: same stream_factor as outer_escape_conductance
   flux_out = 0.125d0 * max(state%cooling_scale, 1d0)**2 * clight * e_n
  else
   flux_out = 0.25d0 * clight * e_n
  end if
  lum_obs = max(4d0 * pi * r_obs**2 * flux_out, 0d0)
  ! No luminosity override during the gap: let the boundary condition set the flux.
end subroutine find_transport_photosphere

real(8) function transport_timestep_limit(state)
  type(transport_state_type), intent(in) :: state
  integer :: i, iph, ihi, ilo
  real(8) :: dr, rho_face, dcoef, r_ph
  real(8) :: r_face_l, r_face_r, vol_i, area_l, area_r
  real(8) :: rho_face_l, rho_face_r, dcoef_l, dcoef_r, k_l, k_r, k_escape
  real(8) :: t_relax

  transport_timestep_limit = huge(1d0)
  if (.not. state%initialized) return
  if (state%n_zones < 2) then
   transport_timestep_limit = huge(1d0)
   return
  end if
  ihi = 1
  if (.not. state%in_cooling_phase) ihi = first_active_zone(state)
  ilo = ihi
  call photosphere_boundary_from_tau(state, ilo, iph, r_ph)

  if (iph > ilo) then
   do i = ilo, iph-1
    dr = max(state%radius(i+1) - state%radius(i), 1d-30)
    rho_face = 0.5d0 * (state%rho(i) + state%rho(i+1))
    dcoef = clight * a_rad / (3d0 * state%kappa * max(rho_face, 1d-30))
    transport_timestep_limit = min(transport_timestep_limit, dr*dr / max(dcoef, 1d-30))
   end do
  end if

  do i = ilo, iph
   if (.not. state%in_cooling_phase) then
    call interaction_zone_geometry_at(state, i, state%r_shell_current, r_face_l, r_face_r, vol_i)
   else
    call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   end if
   if (i == iph) then
    r_face_r = min(max(r_ph, r_face_l), r_face_r)
    vol_i = 4d0 * pi * max(r_face_r**3 - r_face_l**3, 1d-30) / 3d0
   end if

   if (i == ilo) then
    k_l = 0d0
   else
    area_l = 4d0 * pi * r_face_l**2
    rho_face_l = 0.5d0 * (state%rho(i-1) + state%rho(i))
    dcoef_l = clight * a_rad / (3d0 * state%kappa * max(rho_face_l, 1d-30))
    k_l = area_l * dcoef_l / max(state%radius(i) - state%radius(i-1), 1d-30)
   end if

   if (i == iph) then
    k_r = 0d0
    k_escape = outer_escape_conductance(state, i, r_face_r)
   else
    area_r = 4d0 * pi * r_face_r**2
    rho_face_r = 0.5d0 * (state%rho(i) + state%rho(i+1))
    dcoef_r = clight * a_rad / (3d0 * state%kappa * max(rho_face_r, 1d-30))
    k_r = area_r * dcoef_r / max(state%radius(i+1) - state%radius(i), 1d-30)
    k_escape = 0d0
   end if

   t_relax = a_rad * vol_i / max(k_l + k_r + k_escape, 1d-30)
   transport_timestep_limit = min(transport_timestep_limit, t_relax)
  end do

 if (.not.(transport_timestep_limit > 0d0 .and. transport_timestep_limit < huge(1d0))) then
   transport_timestep_limit = huge(1d0)
  end if
 end function transport_timestep_limit

real(8) function shock_motion_timestep_limit(state, u_shell)
  type(transport_state_type), intent(in) :: state
  real(8), intent(in) :: u_shell
  integer :: ihi, i
  real(8) :: dr_nom, r_face_l, r_face_r, vol_i

  shock_motion_timestep_limit = huge(1d0)
  if (.not. state%initialized) return
  if (state%in_cooling_phase) return
  if (state%n_zones < 2) return
  if (abs(u_shell) <= 1d-30) return

  ihi = first_active_zone(state)

  call zone_geometry_from_array(state%radius_ref, state%r_inner_support, state%r_outer_support, state%n_zones, ihi, &
       r_face_l, r_face_r, vol_i)
  dr_nom = max(r_face_r - r_face_l, 1d-30)
  shock_motion_timestep_limit = 0.4d0 * dr_nom / abs(u_shell)
end function shock_motion_timestep_limit

real(8) function shock_motion_timestep_limit_at(state, r_shell, u_shell) result(dt_cap)
  type(transport_state_type), intent(in) :: state
  real(8), intent(in) :: r_shell, u_shell
  integer :: ihi
  real(8) :: r_face_l, r_face_r, vol_i, dr_nom

  dt_cap = huge(1d0)
  if (.not. state%initialized) return
  if (state%in_cooling_phase) return
  if (state%n_zones < 2) return
  if (abs(u_shell) <= 1d-30) return

  ihi = first_active_zone_at(state, r_shell)
  call zone_geometry_from_array(state%radius_ref, state%r_inner_support, state%r_outer_support, state%n_zones, ihi, &
       r_face_l, r_face_r, vol_i)
  dr_nom = max(r_face_r - r_face_l, 1d-30)
  dt_cap = 0.4d0 * dr_nom / abs(u_shell)
end function shock_motion_timestep_limit_at

subroutine solve_transport_step(state, dt, lum_heat, theta_in)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: dt, lum_heat
  real(8), intent(in), optional :: theta_in

  integer :: i, n, ihi, iph
  real(8) :: dr_l, dr_r, r_face_l, r_face_r, area_l, area_r, vol_i
  real(8) :: vol_old
  real(8) :: rho_face_l, rho_face_r, dcoef_l, dcoef_r, k_l, k_r, k_escape, r_ph
  real(8) :: theta, omt
  real(8) :: expansion_lambda
  if (.not. state%initialized) return
  n = state%n_zones
  if (n < 2) return

  state%work_old_y = state%y
  theta = 0.5d0
  if (present(theta_in)) theta = min(max(theta_in, 0d0), 1d0)
  omt = 1d0 - theta
  state%work_a = 0d0
  state%work_b = 0d0
  state%work_c = 0d0
  state%work_rhs = 0d0
  state%work_kl = 0d0
  state%work_kr = 0d0
  state%work_kesc = 0d0
  state%work_vol = 0d0
  state%work_vol_old = 0d0
  ihi = 1
  if (.not. state%in_cooling_phase) then
   ihi = first_active_zone(state)
  end if
  call photosphere_boundary_from_tau(state, ihi, iph, r_ph)

  expansion_lambda = 0d0
  do i = 1, n
   if (.not. state%in_cooling_phase .and. i < ihi) then
    state%work_vol(i) = 0d0
    state%work_vol_old(i) = 0d0
    state%work_kl(i) = 0d0
    state%work_kr(i) = 0d0
    state%work_kesc(i) = 0d0
    cycle
   end if
   if (.not. state%in_cooling_phase) then
    call interaction_zone_geometry_at(state, i, state%r_shell_current, r_face_l, r_face_r, vol_i)
   else
    call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   end if
   k_escape = 0d0

   if (i > iph) then
    state%work_vol(i) = 0d0
    state%work_vol_old(i) = 0d0
    state%work_kl(i) = 0d0
    state%work_kr(i) = 0d0
    state%work_kesc(i) = 0d0
    cycle
   end if

   if (i == ihi) then
    k_l = 0d0
   else
    dr_l = state%radius(i) - state%radius(i-1)
    area_l = 4d0 * pi * r_face_l**2
    rho_face_l = 0.5d0 * (state%rho(i-1) + state%rho(i))
    dcoef_l = clight * a_rad / (3d0 * state%kappa * max(rho_face_l, 1d-30))
    k_l = area_l * dcoef_l / max(dr_l, 1d-30)
   end if

   if (i == iph) then
    r_face_r = min(max(r_ph, r_face_l), r_face_r)
    vol_i = 4d0 * pi * max(r_face_r**3 - r_face_l**3, 1d-30) / 3d0
    area_r = 4d0 * pi * r_face_r**2
    k_r = 0d0
    k_escape = outer_escape_conductance(state, i, r_face_r)
   else
    dr_r = state%radius(i+1) - state%radius(i)
    area_r = 4d0 * pi * r_face_r**2
    rho_face_r = 0.5d0 * (state%rho(i) + state%rho(i+1))
    dcoef_r = clight * a_rad / (3d0 * state%kappa * max(rho_face_r, 1d-30))
    k_r = area_r * dcoef_r / max(dr_r, 1d-30)
   end if

   if (state%in_cooling_phase) then
    ! update_cooling_grid already applies the homologous R^-4 cooling to the
    ! transported radiation field. The implicit solve should therefore only
    ! advance diffusion on the new comoving grid geometry.
    vol_old = vol_i
    state%work_vol(i) = vol_i
    state%work_vol_old(i) = vol_old
   else
    vol_old = vol_i
    state%work_vol(i) = vol_i
    state%work_vol_old(i) = vol_old
   end if
   state%work_kl(i) = k_l
   state%work_kr(i) = k_r
   state%work_kesc(i) = k_escape
  end do

  do i = 1, n
   if (.not. state%in_cooling_phase .and. i < ihi) then
    state%work_a(i) = 0d0
    state%work_b(i) = 1d0
    state%work_c(i) = 0d0
    state%work_rhs(i) = 0d0
    cycle
   end if
   if (.not. state%in_cooling_phase) then
    call interaction_zone_geometry_at(state, i, state%r_shell_current, r_face_l, r_face_r, vol_i)
   else
    call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   end if
   if (i > iph) then
    state%work_a(i) = 0d0
    state%work_b(i) = 1d0
    state%work_c(i) = 0d0
    state%work_rhs(i) = 0d0
    cycle
   end if

   state%work_b(i) = a_rad * state%work_vol(i) / dt + &
                     theta * (state%work_kl(i) + state%work_kr(i) + state%work_kesc(i))
   state%work_rhs(i) = a_rad * state%work_vol_old(i) * state%work_old_y(i) / dt - &
            omt * (state%work_kl(i) + state%work_kr(i) + state%work_kesc(i)) * state%work_old_y(i)

   if (i > ihi) then
    state%work_a(i) = -theta * state%work_kl(i)
    state%work_rhs(i) = state%work_rhs(i) + omt * state%work_kl(i) * state%work_old_y(i-1)
   end if
   if (i < iph) then
    state%work_c(i) = -theta * state%work_kr(i)
    state%work_rhs(i) = state%work_rhs(i) + omt * state%work_kr(i) * state%work_old_y(i+1)
   end if

   if (max(lum_heat, 0d0) > 0d0 .and. .not. state%in_cooling_phase) then
    if (i == ihi) then
     state%work_rhs(i) = state%work_rhs(i) + max(lum_heat, 0d0)
    end if
   end if
  end do

  if (iph >= 1 .and. iph <= n) then
   k_l = state%work_kl(iph)
   k_escape = state%work_kesc(iph)
   state%work_a(iph) = 0d0
   if (iph > 1) then
    state%work_a(iph) = -theta * k_l
   end if
   state%work_b(iph) = a_rad * state%work_vol(iph) / dt + theta * (k_l + k_escape)
   state%work_c(iph) = 0d0
   state%work_rhs(iph) = a_rad * state%work_vol_old(iph) * state%work_old_y(iph) / dt - &
        omt * (k_l + k_escape) * state%work_old_y(iph)
   if (iph > 1) then
    state%work_rhs(iph) = state%work_rhs(iph) + omt * k_l * state%work_old_y(iph-1)
   end if
  end if

  if (iph >= 1 .and. iph <= n) then
   if (state%work_b(iph) > 0d0) then
    if (a_rad * state%work_rhs(iph) / state%work_b(iph) > 1d30) then
     write(*,'(A,1X,ES12.5,1X,A,1X,I0)') 'ROW_SPIKE iph=', dble(iph), 'n=', n
     write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5)') '  vol=', state%work_vol(iph), 'vol_old=', state%work_vol_old(iph)
     write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
          '  k_l=', state%work_kl(iph), 'k_esc=', state%work_kesc(iph), 'rhs=', state%work_rhs(iph)
     write(*,'(A,1X,ES12.5)') '  b=', state%work_b(iph)
    end if
   end if
  end if

  ! Do not force a surface target during the gap; rely on volumetric heating + diffusion.

  call tridag(state%work_a, state%work_b, state%work_c, state%work_rhs, state%work_sol, state%work_gam, n)
  state%y = max(state%work_sol, 0d0)
  if (sum(state%y) <= 0d0 .and. sum(max(state%work_old_y, 0d0)) > 0d0) then
   if (mode3_debug_enabled()) then
    write(*,'(A,1X,ES12.5,1X,A,1X,I0,1X,A,1X,L1)') 'M3_WARN_FALLBACK dt=', dt, 'iph=', iph, 'cool=', state%in_cooling_phase
   end if
   state%y = max(state%work_old_y, 0d0)
  end if
  call update_tau_and_luminosity(state)
end subroutine solve_transport_step

subroutine locate_source_cells(state, ilo, ihi, w_lo, w_hi)
  type(transport_state_type), intent(in) :: state
  integer, intent(out) :: ilo, ihi
  real(8), intent(out) :: w_lo, w_hi
  integer :: i

  if (state%r_shell_current <= state%radius(1)) then
   ilo = 1
   ihi = 1
   w_lo = 1d0
   w_hi = 0d0
   return
  end if
  do i = 1, state%n_zones - 1
   if (state%r_shell_current >= state%radius(i) .and. state%r_shell_current <= state%radius(i+1)) then
    ilo = i + 1
    ihi = i + 1
    w_lo = 1d0
    w_hi = 0d0
    return
   end if
  end do
  ilo = state%n_zones
  ihi = state%n_zones
  w_lo = 1d0
  w_hi = 0d0
end subroutine locate_source_cells

integer function locate_shock_zone(state) result(ish)
  type(transport_state_type), intent(in) :: state
  integer :: i
  real(8) :: dmin, dcur

  ish = 1
  dmin = abs(state%radius(1) - state%r_shell_current)
  do i = 2, state%n_zones
   dcur = abs(state%radius(i) - state%r_shell_current)
   if (dcur < dmin) then
    dmin = dcur
    ish = i
   end if
  end do
end function locate_shock_zone

integer function locate_shock_zone_ref(state) result(ish)
  ! Find the reference-grid zone closest to the current shock position.
  ! radius_ref spans the full static CSM support, so this tracks where
  ! the shock is in the comoving reference frame used for e_swept.
  type(transport_state_type), intent(in) :: state
  integer :: i
  real(8) :: dmin, dcur

  ish = 1
  dmin = abs(state%radius_ref(1) - state%r_shell_current)
  do i = 2, state%n_zones
   dcur = abs(state%radius_ref(i) - state%r_shell_current)
   if (dcur < dmin) then
    dmin = dcur
    ish = i
   end if
  end do
end function locate_shock_zone_ref

integer function first_active_zone(state) result(ihi)
  type(transport_state_type), intent(in) :: state
  if (.not. state%in_cooling_phase .and. state%r_inner >= state%r_inner_support) then
   ihi = 1
  else
   ihi = first_active_zone_at(state, state%r_shell_current)
  end if
end function first_active_zone

 subroutine zone_geometry(state, i, r_face_l, r_face_r, vol_i)
  type(transport_state_type), intent(in) :: state
  integer, intent(in) :: i
  real(8), intent(out) :: r_face_l, r_face_r, vol_i

  if (i == 1) then
   r_face_l = state%r_inner
  else
   r_face_l = sqrt(state%radius(i-1) * state%radius(i))
  end if
  if (i == state%n_zones) then
   r_face_r = state%r_outer
  else
   r_face_r = sqrt(state%radius(i) * state%radius(i+1))
  end if
  vol_i = 4d0 * pi * max(r_face_r**3 - r_face_l**3, 1d-30) / 3d0
 end subroutine zone_geometry

subroutine update_tau_and_luminosity(state)
  type(transport_state_type), intent(inout) :: state
  integer :: i
  real(8) :: r_face_l, r_face_r, vol_i, dr
  real(8) :: rho_cell

  state%tau(state%n_zones) = 0d0
  do i = state%n_zones, 1, -1
   call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   dr = max(r_face_r - r_face_l, 0d0)
   rho_cell = max(state%rho(i), 1d-30)
   if (i == state%n_zones) then
    state%tau(i) = state%kappa * rho_cell * dr
   else
    state%tau(i) = state%tau(i+1) + state%kappa * rho_cell * dr
   end if
  end do
end subroutine update_tau_and_luminosity

 subroutine tridag(a, b, c, r, u, gam, n)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n), b(n), c(n), r(n)
  real(8), intent(inout) :: gam(n)
  real(8), intent(out) :: u(n)
  real(8) :: bet
  integer :: j

  gam = 0d0
  bet = b(1)
  if (abs(bet) < 1d-60) bet = sign(1d-60, b(1) + 1d-60)
  u(1) = r(1) / bet
  do j = 2, n
   gam(j) = c(j-1) / bet
   bet = b(j) - a(j) * gam(j)
   if (abs(bet) < 1d-60) bet = sign(1d-60, b(j) + 1d-60)
   u(j) = (r(j) - a(j) * u(j-1)) / bet
  end do
  do j = n-1, 1, -1
   u(j) = u(j) - gam(j+1) * u(j+1)
  end do
 end subroutine tridag

 pure function interp_linear_monotonic(x, xp, fp, n) result(fx)
  integer, intent(in) :: n
  real(8), intent(in) :: x
  real(8), dimension(n), intent(in) :: xp, fp
  real(8) :: fx
  integer :: j

  if (n <= 1) then
   fx = fp(1)
   return
  end if
  if (x <= xp(1)) then
   fx = fp(1)
   return
  end if
  if (x >= xp(n)) then
   fx = fp(n)
   return
  end if

  do j = 1, n-1
   if (x >= xp(j) .and. x <= xp(j+1)) then
    if (xp(j+1) - xp(j) > 1d-30) then
     fx = fp(j) + (fp(j+1)-fp(j)) * (x-xp(j)) / (xp(j+1)-xp(j))
    else
     fx = fp(j)
    end if
    return
   end if
  end do

  fx = fp(n)
 end function interp_linear_monotonic

! ==========================================================================
! Dimensionless formulation subroutines (paper Appendix A)
! Run mode 3 uses these exclusively.
! ==========================================================================

! ------------------------------------------------------------------
! Reset dimensionless state
! ------------------------------------------------------------------
 subroutine reset_dimless_state(state)
  type(dimless_state_type), intent(inout) :: state
  if (allocated(state%xi_grid)) deallocate(state%xi_grid)
  if (allocated(state%e_grid)) deallocate(state%e_grid)
  if (allocated(state%work_a)) deallocate(state%work_a)
  if (allocated(state%work_b)) deallocate(state%work_b)
  if (allocated(state%work_c)) deallocate(state%work_c)
  if (allocated(state%work_rhs)) deallocate(state%work_rhs)
  if (allocated(state%work_sol)) deallocate(state%work_sol)
  if (allocated(state%work_gam)) deallocate(state%work_gam)
  if (allocated(state%work_old_e)) deallocate(state%work_old_e)
  state%initialized = .false.
  state%in_cooling_phase = .false.
  state%n_zones = 0
  state%x_sh = 1d0
  state%w_sh = 1d0
  state%phi_sh = 1d-2
  state%zeta = 1d0
  state%y_diff = 0d0
  state%R_in_R0 = 1d0
  state%x_sh_dot = 0d0
  state%rannacher_left = 0
  state%x_ph_cached_xsh = -1d0
  state%x_ph_cached_xmin = -1d0
  state%lum_obs_cgs = 0d0
 end subroutine reset_dimless_state

! ------------------------------------------------------------------
! Compute characteristic scales from op(1), op(2)
! Paper Eq. 728-742, 796-800, 836-838
! ------------------------------------------------------------------
 subroutine initialize_dimless_state(state, kappa_in, eff_in, n_zones_in)
  type(dimless_state_type), intent(inout) :: state
  real(8), intent(in) :: kappa_in, eff_in
  integer, intent(in) :: n_zones_in

  real(8) :: t_dim

  call reset_dimless_state(state)

  state%kappa = max(kappa_in, 1d-30)
  state%eff = eff_in
  state%n_zones = max(n_zones_in, 8)

  ! R_csm_in: inner edge of CSM (paper uses t_in = R_csm_in / v_ej_max)
  state%R_csm_in = max(query_csm_inner_edge(1d1, op(2)), 1d0)

  ! v_ej_max: maximum ejecta velocity
  if (op(1)%bpl_vmax > 0d0) then
   state%v_ej_max = op(1)%bpl_vmax
  elseif (op(1)%bpl_vt > 0d0) then
   state%v_ej_max = op(1)%bpl_vt
  elseif (op(1)%exp_v0 > 0d0) then
   state%v_ej_max = 3d0 * op(1)%exp_v0
  elseif (associated(op(1)%v_grid)) then
   state%v_ej_max = maxval(op(1)%v_grid)
  else
   state%v_ej_max = 1d9
  end if

  ! t_in = R_csm_in / v_ej_max (paper Eq. 728)
  state%t_in = state%R_csm_in / state%v_ej_max

  ! Reference densities at t = t_in, r = R_csm_in
  t_dim = state%t_in
  state%rho_csm_in = max(query_csm_density(state%R_csm_in, t_dim, op(2)), 1d-30)
  state%rho_ej_in = max(query_ejecta_density(state%R_csm_in, t_dim, op(1)), 1d-30)

  ! q = rho_csm_in / rho_ej_in (paper Eq. 751)
  state%q = state%rho_csm_in / state%rho_ej_in

  ! tau_csm_in = 3 * kappa * rho_csm_in * R_csm_in (paper Eq. 800)
  state%tau_csm_in = 3d0 * state%kappa * state%rho_csm_in * state%R_csm_in

  ! t_diff = tau_csm_in * R_csm_in / c (paper Eq. 797)
  state%t_diff = state%tau_csm_in * state%R_csm_in / clight

  ! y_ratio = t_in / t_diff (paper Eq. 806)
  state%y_ratio = state%t_in / max(state%t_diff, 1d-30)

  ! u_0 (paper Eq. 836)
  state%u0 = (state%tau_csm_in * state%v_ej_max / clight) * &
              (0.5d0 * state%rho_csm_in * state%v_ej_max**2)

  ! CSM outer edge
  state%R_csm_out = query_csm_outer_edge(t_dim, op(2))
  state%x_csm_out = state%R_csm_out / state%R_csm_in

  ! Initial conditions: paper Eq. 756
  state%x_sh = 1d0
  state%w_sh = 1d0
  state%phi_sh = 1d-2
  state%zeta = 1d0
  state%y_diff = state%y_ratio * state%zeta
  state%x_sh_dot = state%w_sh * state%y_ratio

  ! Compute initial photosphere
  call estimate_photosphere_x(state%x_sh, state)

  ! Set up fixed ξ-grid (only allocates once, never remaps)
  call setup_xi_grid(state)

  state%initialized = .true.
  state%rannacher_left = 4

  state%initialized = .true.
  state%rannacher_left = 4
 end subroutine initialize_dimless_state

! ------------------------------------------------------------------
! Set up fixed ξ-grid (called once at initialization)
! ξ ∈ [0,1], uniformly spaced: ξ_i = (i-1)/(N-1) for i = 1,...,N
! The grid never changes — all physics handled by coordinate transform
! ------------------------------------------------------------------
 subroutine setup_xi_grid(state)
  type(dimless_state_type), intent(inout) :: state
  integer :: i, n

  n = state%n_zones

  allocate(state%xi_grid(n))
  allocate(state%e_grid(n))
  allocate(state%work_a(n), state%work_b(n), state%work_c(n))
  allocate(state%work_rhs(n), state%work_sol(n), state%work_gam(n))
  allocate(state%work_old_e(n))

  ! Fixed uniform ξ ∈ [0,1]
  do i = 1, n
   state%xi_grid(i) = dble(i - 1) / dble(max(n - 1, 1))
  end do

  state%e_grid = 0d0
 end subroutine setup_xi_grid

! ------------------------------------------------------------------
! Compute eta_csm(x):
!   Interaction: η_csm(x) = ρ_csm(x·R_csm_in) / ρ_csm_in  [paper Eq. 738]
!   Cooling: η_csm(x) is time-independent (frozen at t_se)
!            Evaluate the CSM profile at t_se in interaction coordinates
!            Paper Eq. 897: ρ = ρ₀·(R0/R_in)³·η_csm(x)
!            The (R0/R_in)³ factor is handled in the PDE/D scaling,
!            not here. This function returns the comoving profile only.
! ------------------------------------------------------------------
 function compute_eta_csm(x, state) result(eta)
  type(dimless_state_type), intent(in) :: state
  real(8), intent(in) :: x
  real(8) :: eta, r_dim, t_dim, x_int

  if (state%in_cooling_phase) then
   ! Cooling coordinate: x = r/R_in(t)
   ! Convert to interaction coordinate: x_int = r/R_csm_in = x * R_in/R_csm_in
   x_int = x * state%R0 * state%R_in_R0 / state%R_csm_in
   r_dim = x_int * state%R_csm_in
   ! Evaluate at t_se (frozen CSM profile, paper Eq. 897)
   t_dim = state%zeta_se * state%t_in
   eta = query_csm_density(r_dim, t_dim, op(2)) / max(state%rho_csm_in, 1d-30)
  else
   ! Interaction coordinate: x = r/R_csm_in
   r_dim = x * state%R_csm_in
   t_dim = state%zeta * state%t_in
   eta = query_csm_density(r_dim, t_dim, op(2)) / max(state%rho_csm_in, 1d-30)
  end if
 end function compute_eta_csm

! ------------------------------------------------------------------
! Compute eta_ej(x, zeta): rho_ej(r,t) / (rho_ej_in * zeta^{-3})
! Paper Eq. 736
! ------------------------------------------------------------------
 function compute_eta_ej(x, zeta_in, state) result(eta)
  type(dimless_state_type), intent(in) :: state
  real(8), intent(in) :: x, zeta_in
  real(8) :: eta, r_dim, t_dim

  r_dim = x * state%R_csm_in
  t_dim = zeta_in * state%t_in
  eta = query_ejecta_density(r_dim, t_dim, op(1)) / &
        max(state%rho_ej_in * max(zeta_in, 1d-30)**(-3), 1d-30)
 end function compute_eta_ej

! ------------------------------------------------------------------
! Estimate photosphere x from the full CSM density profile
! Works in both interaction and cooling coordinates.
! Interaction: x = r/R_csm_in, integrate τ from x_sh to x_csm_out
! Cooling: x = r/R_in(t), integrate τ from x_min to x_csm_out_cool
! ------------------------------------------------------------------
 subroutine estimate_photosphere_x(x_start, state)
  real(8), intent(in) :: x_start
  type(dimless_state_type), intent(inout) :: state

  integer, parameter :: n_scan = 100
  integer :: i
  real(8) :: x_lo, x_hi, dx_scan, tau_cum, eta_val, x_mid, tau_mid
  real(8) :: x_inner, x_outer, length_scale

  ! Determine integration bounds and length scale
  if (state%in_cooling_phase) then
   x_inner = 1d0  ! x_min = 1 in cooling coordinate x = r/R_in
   ! x_csm_out in cooling coordinate: R_csm_out / R_in(t)
   x_outer = max(state%R_csm_out / (state%R0 * state%R_in_R0), x_inner * 1.01d0)
   ! τ = κ·ρ_csm_in·(R0/R_in)³·R_in·∫ η_csm dx = κ·ρ_csm_in·R0³/R_in²·∫ η_csm dx
   length_scale = state%R0**3 / (state%R0 * state%R_in_R0)**2
  else
   x_inner = max(x_start, 1d0)
   x_outer = max(state%x_csm_out, x_inner * 1.01d0)
   length_scale = state%R_csm_in
  end if

  ! Scan from x_inner to x_outer, integrating τ
  dx_scan = (x_outer - x_inner) / dble(n_scan)

  tau_cum = 0d0
  do i = n_scan, 1, -1
   x_mid = x_inner + (dble(i) - 0.5d0) * dx_scan
   eta_val = compute_eta_csm(x_mid, state)
   tau_cum = tau_cum + state%kappa * state%rho_csm_in * length_scale * eta_val * dx_scan
  end do

  if (tau_cum <= 2d0 / 3d0) then
   ! Entire CSM is optically thin from x_inner
   state%x_ph = x_inner
   return
  end if

  ! Bisection: find x_ph where tau from x_ph to x_outer = 2/3
  x_lo = x_inner
  x_hi = x_outer
  do i = 1, 30
   x_mid = 0.5d0 * (x_lo + x_hi)
   call integrate_tau_from_x(x_mid, state, tau_mid)
   if (tau_mid > 2d0 / 3d0) then
    x_lo = x_mid
   else
    x_hi = x_mid
   end if
  end do
  state%x_ph = 0.5d0 * (x_lo + x_hi)
 end subroutine estimate_photosphere_x

 subroutine integrate_tau_from_x(x_from, state, tau_out)
  real(8), intent(in) :: x_from
  type(dimless_state_type), intent(in) :: state
  real(8), intent(out) :: tau_out

  integer, parameter :: n_int = 50
  integer :: i
  real(8) :: x_end, dx, x_mid, eta_val, length_scale

  if (state%in_cooling_phase) then
   x_end = max(state%R_csm_out / (state%R0 * state%R_in_R0), x_from * 1.01d0)
   length_scale = state%R0**3 / (state%R0 * state%R_in_R0)**2
  else
   x_end = max(state%x_csm_out, x_from * 1.01d0)
   length_scale = state%R_csm_in
  end if
  dx = (x_end - x_from) / dble(n_int)
  tau_out = 0d0
  do i = 1, n_int
   x_mid = x_from + (dble(i) - 0.5d0) * dx
   eta_val = compute_eta_csm(x_mid, state)
   tau_out = tau_out + state%kappa * state%rho_csm_in * length_scale * eta_val * dx
  end do
 end subroutine integrate_tau_from_x

! ------------------------------------------------------------------
! Dynamics RHS: paper Eq. 746-750
! ------------------------------------------------------------------
 subroutine dynamics_rhs(x_in, w_in, phi_in, zeta_in, state, dx_dz, dw_dz, dphi_dz)
  type(dimless_state_type), intent(in) :: state
  real(8), intent(in) :: x_in, w_in, phi_in, zeta_in
  real(8), intent(out) :: dx_dz, dw_dz, dphi_dz
  real(8) :: eta_ej_val, eta_csm_val, v_rel_ej, q_loc

  ! Save current zeta for eta evaluation
  ! (compute_eta_ej uses state%zeta, but we want the passed zeta_in)

  eta_csm_val = compute_eta_csm(x_in, state)
  eta_ej_val = compute_eta_ej(x_in, zeta_in, state)

  ! v_ej at shell = R_sh/t = (x*R_csm_in)/(zeta*t_in) = x/zeta * v_ej_max
  ! Relative velocity: (v_ej - v_sh)/v_ej_max = x/zeta - w
  v_rel_ej = x_in / max(zeta_in, 1d-30) - w_in

  q_loc = state%q

  ! Paper Eq. 746
  dx_dz = w_in

  ! Paper Eq. 748: dphi/dz = x^2 * zeta^{-3} * (x/zeta - w) * eta_ej + q * x^2 * w * eta_csm
  dphi_dz = x_in**2 * max(zeta_in, 1d-30)**(-3) * max(v_rel_ej, 0d0) * eta_ej_val &
           + q_loc * x_in**2 * max(w_in, 0d0) * eta_csm_val

  ! Paper Eq. 749: dw/dz = (1/phi)[x^2*zeta^{-3}*(x/zeta-w)^2*eta_ej - q*x^2*w^2*eta_csm]
  if (phi_in > 1d-30) then
   dw_dz = (x_in**2 * max(zeta_in, 1d-30)**(-3) * max(v_rel_ej, 0d0)**2 * eta_ej_val &
           - q_loc * x_in**2 * w_in**2 * eta_csm_val) / phi_in
  else
   dw_dz = 0d0
  end if
 end subroutine dynamics_rhs

! ------------------------------------------------------------------
! RK4 dynamics step (paper Sec. A3)
! ------------------------------------------------------------------
 subroutine rk4_step_dynamics(state, dzeta)
  type(dimless_state_type), intent(inout) :: state
  real(8), intent(in) :: dzeta

  real(8) :: x, w, phi, zeta
  real(8) :: k1_x, k1_w, k1_phi
  real(8) :: k2_x, k2_w, k2_phi
  real(8) :: k3_x, k3_w, k3_phi
  real(8) :: k4_x, k4_w, k4_phi

  x = state%x_sh
  w = state%w_sh
  phi = max(state%phi_sh, 1d-30)
  zeta = state%zeta

  ! k1
  call dynamics_rhs(x, w, phi, zeta, state, k1_x, k1_w, k1_phi)

  ! k2 (midpoint)
  call dynamics_rhs(x + 0.5d0*dzeta*k1_x, w + 0.5d0*dzeta*k1_w, &
                     max(phi + 0.5d0*dzeta*k1_phi, 1d-30), zeta + 0.5d0*dzeta, &
                     state, k2_x, k2_w, k2_phi)

  ! k3 (midpoint with k2 slopes)
  call dynamics_rhs(x + 0.5d0*dzeta*k2_x, w + 0.5d0*dzeta*k2_w, &
                     max(phi + 0.5d0*dzeta*k2_phi, 1d-30), zeta + 0.5d0*dzeta, &
                     state, k3_x, k3_w, k3_phi)

  ! k4 (endpoint with k3 slopes)
  call dynamics_rhs(x + dzeta*k3_x, w + dzeta*k3_w, &
                     max(phi + dzeta*k3_phi, 1d-30), zeta + dzeta, &
                     state, k4_x, k4_w, k4_phi)

  ! Update
  state%x_sh = x + dzeta * (k1_x + 2d0*k2_x + 2d0*k3_x + k4_x) / 6d0
  state%w_sh = w + dzeta * (k1_w + 2d0*k2_w + 2d0*k3_w + k4_w) / 6d0
  state%phi_sh = max(phi + dzeta * (k1_phi + 2d0*k2_phi + 2d0*k3_phi + k4_phi) / 6d0, 1d-30)
  state%zeta = zeta + dzeta
  state%y_diff = state%y_ratio * state%zeta
 end subroutine rk4_step_dynamics

! ------------------------------------------------------------------
! Diffusion solver in ξ-coordinates: Crank-Nicolson
! PDE: ∂e/∂y = (1/Δx²)·(1/x̂²)·∂/∂ξ[D(ξ)·∂e/∂ξ] + V(ξ)·∂e/∂ξ
! where x̂(ξ) = x_min + ξ·Δx, Δx = x_ph - x_min
! Interaction: x_min = x_sh, V(ξ) = ẋ_sh·(1-ξ)/Δx
! Cooling: x_min = 1, V(ξ) = 0 (comoving frame)
! Spec Section 3.3
! ------------------------------------------------------------------
 subroutine solve_diffusion_dimless(state, dy)
  type(dimless_state_type), intent(inout) :: state
  real(8), intent(in) :: dy

  integer :: i, n
  real(8) :: theta, omt
  real(8) :: Delta_x, dxi, x_hat_i, x_hat_im, x_hat_ip
  real(8) :: eta_im, eta_ip, D_im, D_ip
  real(8) :: alpha_m, alpha_p, gamma_v, pe_cell
  real(8) :: f_ib, f_ob, G, V_i
  real(8) :: adiabatic_rate, robin_coeff, robin_ratio
  real(8) :: a_n_std, b_n_std, c_n_std, a_n_rhs, b_n_rhs, c_n_rhs

  n = state%n_zones
  if (n < 3) return

  ! Domain: interaction → [x_sh, x_ph], cooling → [x_min, x_ph]
  if (.not. state%in_cooling_phase) then
   state%x_min = state%x_sh
  else
   state%x_min = 1d0  ! Paper Eq. 945: x_min = 1 in comoving x = r/R_in
  end if

  Delta_x = state%x_ph - state%x_min
  if (Delta_x <= 0d0) then
   return
  end if
  dxi = 1d0 / dble(n - 1)  ! Uniform ξ spacing

  state%work_old_e = state%e_grid

  ! Theta: Rannacher start-up (fully implicit), then Crank-Nicolson
  if (state%rannacher_left > 0) then
   theta = 1d0
   state%rannacher_left = state%rannacher_left - 1
  else
   theta = 0.5d0
  end if
  omt = 1d0 - theta

  ! Advection velocity in ξ-coordinates
  if (.not. state%in_cooling_phase) then
   ! Interaction: V(ξ) = ẋ_sh·(1-ξ)/Δx
   V_i = state%x_sh_dot / Delta_x
  else
   ! Cooling: grid is stationary in comoving frame
   V_i = 0d0
  end if

  ! Zero out work arrays
  state%work_a = 0d0
  state%work_b = 0d0
  state%work_c = 0d0
  state%work_rhs = 0d0

  ! --- Interior zones i = 2, ..., N-1 ---
  do i = 2, n - 1
   ! Physical position: x̂(ξ_i) = x_min + ξ_i · Δx
   x_hat_i = state%x_min + state%xi_grid(i) * Delta_x
   x_hat_im = state%x_min + (state%xi_grid(i) - 0.5d0 * dxi) * Delta_x  ! x̂_{i-1/2}
   x_hat_ip = state%x_min + (state%xi_grid(i) + 0.5d0 * dxi) * Delta_x  ! x̂_{i+1/2}

   ! Evaluate η_csm at face positions
   eta_im = compute_eta_csm(x_hat_im, state)
   eta_ip = compute_eta_csm(x_hat_ip, state)

   ! D(ξ) at faces: D = x̂²/η_csm  [×R_in/R0 in cooling]
   D_im = x_hat_im**2 / max(eta_im, 1d-30)
   D_ip = x_hat_ip**2 / max(eta_ip, 1d-30)
   if (state%in_cooling_phase) then
    D_im = D_im * state%R_in_R0
    D_ip = D_ip * state%R_in_R0
   end if

   ! Stencil coefficients (spec Section 3.3)
   alpha_m = D_im / (x_hat_i**2 * Delta_x**2 * dxi**2)
   alpha_p = D_ip / (x_hat_i**2 * Delta_x**2 * dxi**2)

   ! Advection: V(ξ_i)·∂e/∂ξ
   gamma_v = V_i * (1d0 - state%xi_grid(i))  ! V_i·(1-ξ_i)

   ! Check Péclet number for upwinding
   pe_cell = abs(gamma_v) * dxi / max(alpha_m + alpha_p, 1d-60) * 2d0

   if (pe_cell > 2d0 .and. gamma_v > 0d0) then
    ! Upwind: V > 0, advect from left
    ! a_i includes both α⁻ and -γ/Δξ
    state%work_a(i) = -theta * dy * (alpha_m + gamma_v / dxi)
    state%work_c(i) = -theta * dy * alpha_p
    state%work_b(i) = 1d0 + theta * dy * (alpha_m + alpha_p + gamma_v / dxi)

    ! RHS explicit part
    state%work_rhs(i) = state%work_old_e(i) &
        + omt * dy * (alpha_m + gamma_v / dxi) * state%work_old_e(i-1) &
        + omt * dy * alpha_p * state%work_old_e(i+1) &
        - omt * dy * (alpha_m + alpha_p + gamma_v / dxi) * state%work_old_e(i)
   else if (pe_cell > 2d0 .and. gamma_v < 0d0) then
    ! Upwind: V < 0, advect from right
    state%work_a(i) = -theta * dy * alpha_m
    state%work_c(i) = -theta * dy * (alpha_p - gamma_v / dxi)
    state%work_b(i) = 1d0 + theta * dy * (alpha_m + alpha_p - gamma_v / dxi)

    state%work_rhs(i) = state%work_old_e(i) &
        + omt * dy * alpha_m * state%work_old_e(i-1) &
        + omt * dy * (alpha_p - gamma_v / dxi) * state%work_old_e(i+1) &
        - omt * dy * (alpha_m + alpha_p - gamma_v / dxi) * state%work_old_e(i)
   else
    ! Central difference for advection
    state%work_a(i) = -theta * dy * (alpha_m - gamma_v / (2d0 * dxi))
    state%work_c(i) = -theta * dy * (alpha_p + gamma_v / (2d0 * dxi))
    state%work_b(i) = 1d0 + theta * dy * (alpha_m + alpha_p)

    state%work_rhs(i) = state%work_old_e(i) &
        + omt * dy * (alpha_m - gamma_v / (2d0 * dxi)) * state%work_old_e(i-1) &
        + omt * dy * (alpha_p + gamma_v / (2d0 * dxi)) * state%work_old_e(i+1) &
        - omt * dy * (alpha_m + alpha_p) * state%work_old_e(i)
   end if

  end do

  ! --- Inner BC (ξ = 0, i = 1) ---
  if (.not. state%in_cooling_phase) then
   ! Neumann BC: shock injects flux OUTWARD into the CSM.
   ! Physical convention: outward flux = -D·∂e/∂x > 0, so ∂e/∂x < 0 at shock.
   ! In ξ-coordinates: ∂e/∂ξ|_{ξ=0} = -Δx·f_ib  (negative gradient = energy decreasing outward)
   ! Ghost point: e_0 = e_2 + 2·Δξ·G  where G = Δx·f_ib > 0
   ! This gives source term +α_m·2Δξ·G (always positive, survives for all θ)
   f_ib = state%eff * compute_eta_csm(state%x_min, state)**2 * state%w_sh**3
   G = Delta_x * f_ib

   x_hat_i = state%x_min  ! ξ=0 → x̂ = x_min
   x_hat_ip = state%x_min + 0.5d0 * dxi * Delta_x  ! x̂_{1+1/2}

   eta_ip = compute_eta_csm(x_hat_ip, state)
   D_ip = x_hat_ip**2 / max(eta_ip, 1d-30)

   alpha_p = D_ip / (x_hat_i**2 * Delta_x**2 * dxi**2)
   alpha_m = alpha_p  ! At ξ=0, left face mirrors right face for ghost elimination

   ! Advection at ξ=0: V(0) = ẋ_sh·(1-0)/Δx = x_sh_dot/Delta_x
   gamma_v = V_i  ! = x_sh_dot / Delta_x

   ! Ghost point: e_0 = e_2 + 2Δξ·G
   ! After ghost elimination into row 1:
   ! LHS: (a_1+c_1)·e_2 + b_1·e_1 = RHS - a_1·2Δξ·G
   ! Since a_1 = -θ·dy·α_m < 0, the source -a_1·2Δξ·G = +θ·dy·α_m·2Δξ·G > 0

   state%work_a(1) = 0d0
   state%work_c(1) = -theta * dy * (alpha_m + alpha_p)
   state%work_b(1) = 1d0 + theta * dy * (alpha_m + alpha_p)

   ! RHS with ghost elimination: e_0^n = e_2^n + 2Δξ·G^n
   ! Explicit G source: +omt·dy·α_m·2Δξ·G (positive, from e_0^n ghost)
   ! Implicit G source: +θ·dy·α_m·2Δξ·G (positive, from -a_1·2Δξ·G = +θ·dy·α_m·2Δξ·G)
   ! Total G source = dy·α_m·2Δξ·G > 0 for all θ
   state%work_rhs(1) = state%work_old_e(1) &
       + omt * dy * alpha_m * (state%work_old_e(2) + 2d0 * dxi * G) &
       + omt * dy * alpha_p * state%work_old_e(2) &
       - omt * dy * (alpha_m + alpha_p) * state%work_old_e(1) &
       - omt * dy * gamma_v / (2d0 * dxi) * (state%work_old_e(2) - state%work_old_e(1)) &
       + theta * dy * alpha_m * 2d0 * dxi * G  ! implicit ghost: -a_1·2Δξ·G
  else
   ! Cooling phase: zero flux at inner boundary (spec Section 4.1)
   ! Ghost point: e_0 = e_2 (symmetry, G = 0)
   x_hat_i = state%x_min
   x_hat_ip = state%x_min + 0.5d0 * dxi * Delta_x
   eta_ip = compute_eta_csm(x_hat_ip, state)
   D_ip = x_hat_ip**2 / max(eta_ip, 1d-30) * state%R_in_R0

   alpha_p = D_ip / (x_hat_i**2 * Delta_x**2 * dxi**2)
   alpha_m = alpha_p

   ! V(0) = ẋ_sh/Δx; in cooling, x_sh is typically stationary in comoving frame
   gamma_v = V_i

   ! Ghost: e_0 = e_2 (G = 0)
   state%work_a(1) = 0d0
   state%work_c(1) = -theta * dy * (alpha_m + alpha_p)
   state%work_b(1) = 1d0 + theta * dy * (alpha_m + alpha_p)

   state%work_rhs(1) = state%work_old_e(1) &
       + omt * dy * (alpha_m + alpha_p) * state%work_old_e(2) &
       - omt * dy * (alpha_m + alpha_p) * state%work_old_e(1) &
       - omt * dy * gamma_v / (2d0 * dxi) * (state%work_old_e(2) - state%work_old_e(1))

  end if

  ! --- Outer BC (ξ = 1, i = N) ---
  ! Robin/Eddington BC: e(x_ph) = f_ob·∂e/∂x|_{x_ph}
  ! Interaction: f_ob = -4/(τ_csm_in·η_csm(x_ph))  [paper Eq. 852]
  ! Cooling:     f_ob = -2/(τ_csm_in·η_csm(x_ph))·(R_in/R0)²  [paper Eq. 919-922]
  if (.not. state%in_cooling_phase) then
   f_ob = -4d0 / (state%tau_csm_in * max(compute_eta_csm(state%x_ph, state), 1d-30))
  else
   f_ob = -2d0 / (state%tau_csm_in * max(compute_eta_csm(state%x_ph, state), 1d-30)) &
          * state%R_in_R0**2
  end if

  x_hat_i = state%x_ph  ! ξ=1 → x̂ = x_ph
  x_hat_im = state%x_ph - 0.5d0 * dxi * Delta_x  ! x̂_{N-1/2}

  eta_im = compute_eta_csm(x_hat_im, state)
  D_im = x_hat_im**2 / max(eta_im, 1d-30)
  if (state%in_cooling_phase) D_im = D_im * state%R_in_R0

  alpha_m = D_im / (x_hat_i**2 * Delta_x**2 * dxi**2)
  alpha_p = alpha_m  ! At ξ=1, right face mirrors left face for ghost elimination

  ! Advection at ξ=1: V(1) = ẋ_sh·(1-1)/Δx = 0
  gamma_v = 0d0

  ! Ghost-point Robin BC: e_{N+1} = e_{N-1} + r*e_N
  ! where r = 2Δξ·Δx/f_ob (from centered-difference Robin condition)
  ! This preserves diffusion coupling at the boundary.
  robin_coeff = 2d0 * dxi * Delta_x / f_ob  ! < 0 since f_ob < 0

  ! Build standard row N before ghost elimination:
  a_n_std = -theta * dy * (alpha_m - gamma_v / (2d0 * dxi))
  c_n_std = -theta * dy * (alpha_p + gamma_v / (2d0 * dxi))
  b_n_std = 1d0 + theta * dy * (alpha_m + alpha_p)
  a_n_rhs = omt * dy * (alpha_m - gamma_v / (2d0 * dxi))
  c_n_rhs = omt * dy * (alpha_p + gamma_v / (2d0 * dxi))
  b_n_rhs = 1d0 - omt * dy * (alpha_m + alpha_p)

  if (abs(robin_coeff) > 10d0) then
   ! Thick CSM: ghost point makes |r| too large, fall back to one-sided
   robin_ratio = -f_ob / (Delta_x * dxi - f_ob)
   state%work_a(n) = -robin_ratio
   state%work_b(n) = 1d0
   state%work_c(n) = 0d0
   state%work_rhs(n) = 0d0
  else
   ! Ghost elimination: e_{N+1} = e_{N-1} + robin_coeff*e_N
   ! Fold c_N into a_N and b_N:
   state%work_a(n) = a_n_std + c_n_std
   state%work_b(n) = b_n_std + c_n_std * robin_coeff
   state%work_c(n) = 0d0
   state%work_rhs(n) = (a_n_rhs + c_n_rhs) * state%work_old_e(n-1) &
                      + (b_n_rhs + c_n_rhs * robin_coeff) * state%work_old_e(n)
  end if

  ! Solve tridiagonal system
  call tridag(state%work_a, state%work_b, state%work_c, state%work_rhs, &
              state%work_sol, state%work_gam, n)
  state%e_grid = max(state%work_sol, 0d0)

  ! Fallback if solve collapses
  if (sum(state%e_grid(1:n)) <= 0d0 .and. sum(max(state%work_old_e(1:n), 0d0)) > 0d0) then
   state%e_grid = max(state%work_old_e, 0d0)
  end if
 end subroutine solve_diffusion_dimless

! ------------------------------------------------------------------
! Transition from interaction to cooling phase
! Transition criterion: x_sh >= x_csm_out (spec Section 6)
! Remap e(ξ) from interaction coordinates [x_sh, x_ph] to
! cooling coordinates [1, x_ph_cool] where x = r/R_in.
! Paper Eq. 948: e(x, y_se) = e_int(x)
! ------------------------------------------------------------------
 subroutine transition_to_cooling(state)
  type(dimless_state_type), intent(inout) :: state

  integer :: i, n
  real(8) :: x_sh_old, x_ph_old, dxi, Delta_x_old
  real(8) :: x_cool, x_int, xi_src, e_shock
  real(8) :: E_before, E_after, dx, eta_lo, eta_hi

  ! Save pre-transition state
  x_sh_old = state%x_sh
  x_ph_old = state%x_ph
  n = state%n_zones
  dxi = 1d0 / dble(n - 1)

  ! Compute total energy before transition (for diagnostics)
  E_before = 0d0
  Delta_x_old = x_ph_old - x_sh_old
  if (Delta_x_old > 0d0) then
   do i = 1, n - 1
    eta_lo = compute_eta_csm(x_sh_old + state%xi_grid(i) * Delta_x_old, state)
    eta_hi = compute_eta_csm(x_sh_old + state%xi_grid(i+1) * Delta_x_old, state)
    dx = Delta_x_old * dxi
    E_before = E_before + 0.5d0 * (eta_lo * state%e_grid(i) + eta_hi * state%e_grid(i+1)) * dx
   end do
  end if

  ! Shock value (energy at inner boundary = shock)
  e_shock = state%e_grid(1)

  ! Set cooling parameters
  state%zeta_se = state%zeta
  state%y_se = state%y_diff
  state%R0 = state%R_csm_in  ! Paper Eq. 889: R0 = R_in(t_se) = R_csm_in
  state%v_se = state%w_sh * state%v_ej_max
  state%R_in_R0 = 1d0
  state%in_cooling_phase = .true.
  state%rannacher_left = 4

  ! Remap e(ξ): interaction x ∈ [x_sh, x_ph] → cooling x ∈ [1, x_ph_cool]
  ! In cooling coordinates: x_cool = r/R_in = r/R_csm_in = x_int (at t=t_se, R_in=R_csm_in)
  ! So the interaction coordinate and cooling coordinate are the same at t=t_se!
  ! x_cool(ξ) = x_int(ξ) since R_in(t_se) = R_csm_in = R0
  !
  ! The interaction grid spanned [x_sh, x_ph]. The cooling grid spans [1, x_ph_cool].
  ! Region [1, x_sh] was the shocked CSM (not in interaction diffusion grid).
  ! Region [x_sh, x_ph] maps directly since x_cool = x_int at t=t_se.
  !
  ! Fill [1, x_sh] with shock energy e_shock (uniform in shocked region).

  ! After transition, x_min = 1. Recompute x_ph in cooling coords.
  call estimate_photosphere_x(1d0, state)

  ! Remap: for each ξ-point, x_cool = 1 + ξ*(x_ph_new - 1)
  ! Map to interaction x: x_int = x_cool (same at t_se)
  ! If x_int < x_sh_old: use e_shock
  ! If x_int >= x_sh_old: interpolate from old profile
  Delta_x_old = x_ph_old - x_sh_old
  do i = 1, n
   x_cool = 1d0 + state%xi_grid(i) * (state%x_ph - 1d0)
   if (x_cool < x_sh_old) then
    ! Below the old shock position: fill with shock energy
    state%e_grid(i) = e_shock
   else if (Delta_x_old > 0d0) then
    ! Above old shock: map back to old ξ and interpolate
    xi_src = (x_cool - x_sh_old) / Delta_x_old
    if (xi_src >= 1d0) then
     state%e_grid(i) = state%e_grid(n)  ! clamp to outer value
    else
     ! Linear interpolation in old grid
     state%e_grid(i) = state%e_grid(1) * (1d0 - xi_src) + state%e_grid(n) * xi_src
    end if
   end if
  end do

  ! Compute total energy after remap (diagnostics)
  E_after = 0d0
  do i = 1, n - 1
   eta_lo = compute_eta_csm(1d0 + state%xi_grid(i) * (state%x_ph - 1d0), state)
   eta_hi = compute_eta_csm(1d0 + state%xi_grid(i+1) * (state%x_ph - 1d0), state)
   dx = (state%x_ph - 1d0) * dxi
   E_after = E_after + 0.5d0 * (eta_lo * state%e_grid(i) + eta_hi * state%e_grid(i+1)) * dx
  end do

  write(*,'(A)') '=== EMERGENCE HANDOFF ==='
  write(*,'(A,ES12.4,A,ES12.4)') '  t_se(d)=', state%zeta*state%t_in/86400d0, &
    ' zeta_se=', state%zeta_se
  write(*,'(A,ES12.4,A,ES12.4)') '  x_sh=', x_sh_old, ' x_csm_out=', state%x_csm_out
  write(*,'(A,ES12.4,A,ES12.4)') '  E_before=', E_before, ' E_after=', E_after
  write(*,'(A,ES12.4,A,ES12.4)') '  x_ph_new=', state%x_ph, ' Delta_x_cool=', state%x_ph - 1d0
  write(*,'(A,ES12.4,A,ES12.4)') '  R0=', state%R0, ' R_in_R0=', state%R_in_R0
  if (state%x_ph - 1d0 < 0.01d0) then
   write(*,'(A)') '  *** ABORT: cooling domain collapsed! ***'
  end if
  write(*,'(A)') '========================='

 end subroutine transition_to_cooling

! ------------------------------------------------------------------
! Convert dimensionless state to CGS outputs
! Luminosity from e_N (Robin BC value at photosphere, ξ=1)
! L = 4π·c·x_ph²·R_csm_in·u₀ / (3·κ·ρ_csm_in·η_csm(x_ph)) · e_N / β_Ed
! ------------------------------------------------------------------
 subroutine dimless_to_cgs(state)
  type(dimless_state_type), intent(inout) :: state
  real(8) :: e_N, f_ob_abs, L_factor

  ! Basic conversions
  state%r_sh_cgs = state%x_sh * state%R_csm_in
  state%v_sh_cgs = state%w_sh * state%v_ej_max
  state%m_sh_cgs = state%phi_sh * 4d0 * pi * state%R_csm_in**3 * state%rho_ej_in
  state%t_cgs = state%zeta * state%t_in

  ! Photosphere radius: interaction uses R_csm_in, cooling uses R_in(t)
  if (.not. state%in_cooling_phase) then
   state%r_ph_cgs = state%x_ph * state%R_csm_in
  else
   state%r_ph_cgs = state%x_ph * state%R0 * state%R_in_R0
  end if

  ! Luminosity from Robin BC: e(x_ph) = f_ob · ∂e/∂x
  ! ∂e/∂x|_{x_ph} = e_N / f_ob → |∂e/∂x| = e_N / |f_ob|
  !
  ! Interaction: u = u0·e, R_ph = x_ph·R_csm_in, ρ(R_ph) = ρ_csm_in·η_csm
  ! |f_ob| = 4/(τ·η)  →  L = π·c·u₀·τ·x_ph²·R_csm_in/(3κρ_csm_in)·e_N
  !
  ! Cooling: u = u₀·e·(R0/R_in)⁴, R_ph = x_ph·R_in, ρ = ρ_csm_in·(R0/R_in)³·η_csm
  ! |f_ob| = 2/(τ·η)·(R_in/R0)²
  ! L = 2π·c·u₀·τ·x_ph²·R0·(R0/R_in)²/(3κρ_csm_in)·e_N
  ! With R0 = R_csm_in:  L = 2π·c·u₀·τ·x_ph²·R_csm_in·(1/R_in_R0)²/(3κρ_csm_in)·e_N

  e_N = max(state%e_grid(state%n_zones), 0d0)

  if (.not. state%in_cooling_phase) then
   ! Interaction: L = π·c·u₀·τ_csm_in·x_ph²·R_csm_in / (3·κ·ρ_csm_in) · e_N
   L_factor = pi * clight * state%u0 * state%tau_csm_in * state%x_ph**2 &
             * state%R_csm_in / (3d0 * state%kappa * state%rho_csm_in)
   state%lum_obs_cgs = L_factor * e_N
  else
   ! Cooling: L = 2π·c·u₀·τ·x_ph²·R_csm_in·(1/R_in_R0)²/(3κρ_csm_in)·e_N
   L_factor = 2d0 * pi * clight * state%u0 * state%tau_csm_in * state%x_ph**2 &
             * state%R_csm_in / (3d0 * state%kappa * state%rho_csm_in)
   state%lum_obs_cgs = L_factor * e_N / max(state%R_in_R0, 1d-30)**2
  end if

  state%lum_obs_cgs = max(state%lum_obs_cgs, 0d0)

  ! Cap luminosity at unphysical levels (safety)
  if (state%lum_obs_cgs > 1d50) state%lum_obs_cgs = 0d0

 end subroutine dimless_to_cgs

! ------------------------------------------------------------------
! Main driver: dimensionless comoving transport step
! Replaces old comoving_transport_step for run_mode 3
! Paper Sec. A3: operator splitting (dynamics first, then diffusion)
! ------------------------------------------------------------------
 subroutine dimless_comoving_transport_step(state, dt, t_shell, r_sh, v_sh, m_sh, &
                                             lum_heat, lum_obs, r_ph_out)
  type(dimless_state_type), intent(inout) :: state
  real(8), intent(in) :: dt, t_shell, r_sh, v_sh, m_sh, lum_heat
  real(8), intent(out) :: lum_obs
  real(8), intent(out), optional :: r_ph_out

  integer :: nsub
  real(8) :: dzeta_step, dzeta_total, dzeta_target
  real(8) :: dy_step
  real(8) :: dt_dyn_cfl, dt_diff_cfl, advection_cfl
  real(8) :: Delta_x, x_sh_old, frac, dy_pre, dy_post

  lum_obs = 0d0

  ! Initialize if first call
  if (.not. state%initialized) then
   call initialize_dimless_state(state, state%kappa, state%eff, state%n_zones)
  end if

  ! Determine how far to advance in zeta
  dzeta_target = dt / state%t_in

  ! Sub-stepping with CFL constraints
  nsub = 1
  dzeta_total = 0d0
  dzeta_step = dzeta_target  ! Initialize for first iteration
  do while (dzeta_total < dzeta_target - 1d-15)
   ! --- CFL constraints ---
   ! Velocity CFL: dzeta < C * phi / (q * w^2) to resolve deceleration
   dt_dyn_cfl = huge(1d0)
   if (state%w_sh > 1d-30 .and. state%q > 1d-30 .and. state%phi_sh > 1d-30) then
    dt_dyn_cfl = 0.3d0 * state%phi_sh / (state%q * state%w_sh**2)
   end if

   ! Diffusion CFL: dy < 0.5 * t_diff (generous for implicit)
   dt_diff_cfl = huge(1d0)
   if (state%t_diff > 0d0) then
    dt_diff_cfl = 0.5d0 * state%t_diff
   end if

   ! Advection CFL: x_sh_dot should not move more than ~0.3*Delta_x per step
   ! This is for accuracy (upwinding handles stability), so relax it
   advection_cfl = huge(1d0)
   Delta_x = state%x_ph - state%x_min
   if (Delta_x > 0d0 .and. state%x_sh_dot > 1d-30) then
    advection_cfl = max(0.3d0 * Delta_x / state%x_sh_dot, 1d-6)
   end if

   dzeta_step = dzeta_target - dzeta_total
   if (dt_dyn_cfl < huge(1d0)) dzeta_step = min(dzeta_step, dt_dyn_cfl)
   if (dt_diff_cfl < huge(1d0)) dzeta_step = min(dzeta_step, dt_diff_cfl / state%t_in)
   if (advection_cfl < huge(1d0)) dzeta_step = min(dzeta_step, advection_cfl)
   dzeta_step = max(dzeta_step, 1d-10)

   ! Cap at 0.2 day in zeta units
   dzeta_step = min(dzeta_step, 0.2d0 * 86400d0 / state%t_in)

   ! Post-emergence refinement
   if (state%in_cooling_phase .and. state%zeta_se > 0d0) then
    if ((state%zeta - state%zeta_se) * state%t_in < 0.05d0 * 86400d0) then
     dzeta_step = min(dzeta_step, 0.0005d0 * 86400d0 / state%t_in)
    else if ((state%zeta - state%zeta_se) * state%t_in < 0.2d0 * 86400d0) then
     dzeta_step = min(dzeta_step, 0.002d0 * 86400d0 / state%t_in)
    end if
   end if

   ! --- Step 1: Advance dynamics ---
   if (.not. state%in_cooling_phase) then
    ! Save pre-step shock position for crossover detection
    x_sh_old = state%x_sh
    ! Interaction: advance shock ODEs with RK4
    call rk4_step_dynamics(state, dzeta_step)
    ! Compute dx_sh/dy for the advection term
    state%x_sh_dot = state%w_sh * state%y_ratio
    ! Debug: store sub-step count for diagnostics
    state%nsub_last = nsub
   else
    ! Cooling: shock has emerged, advance zeta but x_sh/w_sh/phi_sh are frozen
    ! R_in(t) = R0 + v_se*(t-t_se), tracked via R_in_R0 [paper Eq. 891]
    state%zeta = state%zeta + dzeta_step
    state%y_diff = state%y_ratio * state%zeta
    state%R_in_R0 = 1d0 + state%v_se * (state%zeta - state%zeta_se) * state%t_in / state%R0
    state%x_sh_dot = 0d0  ! No advection in comoving frame
   end if

   ! --- Step 2: Advance diffusion ---
   dy_step = state%y_ratio * dzeta_step

   if (.not. state%in_cooling_phase) then
    ! Interaction phase: update photosphere (with caching), solve diffusion
    if (state%x_ph_cached_xsh < 0d0 .or. &
        abs(state%x_sh - state%x_ph_cached_xsh) > 0.01d0 * max(state%x_sh, 1d0)) then
     call estimate_photosphere_x(state%x_sh, state)
     state%x_ph_cached_xsh = state%x_sh
     state%x_ph_cached_xmin = state%x_min
    end if

    ! Check for transition (x_sh >= x_csm_out, spec Section 6)
    if (state%x_sh >= state%x_csm_out .and. x_sh_old < state%x_csm_out) then
     ! Shock crossed x_csm_out this substep — split at crossover
     frac = (state%x_csm_out - x_sh_old) / max(state%x_sh - x_sh_old, 1d-30)
     frac = max(min(frac, 1d0), 0d0)
     dy_pre = dy_step * frac
     dy_post = dy_step * (1d0 - frac)

     ! Solve diffusion for pre-emergence portion
     if (dy_pre > 0d0) call solve_diffusion_dimless(state, dy_pre)

     ! Transition to cooling
     call transition_to_cooling(state)

     ! Solve diffusion for post-emergence portion
     if (dy_post > 0d0) call solve_diffusion_dimless(state, dy_post)

     ! Update photosphere caching
     state%x_ph_cached_xsh = state%x_sh
     state%x_ph_cached_xmin = state%x_min
    else if (state%x_sh >= state%x_csm_out) then
     ! Already past crossover (transition already happened or x_sh was already >= x_csm_out)
     call transition_to_cooling(state)
     call solve_diffusion_dimless(state, dy_step)
     state%x_ph_cached_xsh = state%x_sh
     state%x_ph_cached_xmin = state%x_min
    else
     ! Normal interaction step (no crossover)
     call solve_diffusion_dimless(state, dy_step)
    end if
   else
    ! Cooling phase: R_in/R0 already updated in Step 1
    ! Photosphere caching: recompute when R_in_R0 changes >1%
    if (state%x_ph_cached_xmin < 0d0 .or. &
        abs(state%R_in_R0 - state%x_ph_cached_xmin) > 0.01d0 * max(state%R_in_R0, 1d0)) then
     call estimate_photosphere_x(state%x_sh, state)
     state%x_ph_cached_xsh = state%x_sh
     state%x_ph_cached_xmin = state%R_in_R0
    end if
    call solve_diffusion_dimless(state, dy_step)
   end if

   dzeta_total = dzeta_total + dzeta_step
   nsub = nsub + 1
   if (nsub > 50000) then
    if (mode3_debug_enabled()) then
     write(6,'(A,I6,A,ES12.4,A,ES12.4)') 'DEBUG dimless: sub-step limit hit, nsub=', nsub, &
      ' dzeta_total=', dzeta_total, ' dzeta_target=', dzeta_target
    end if
    exit
   end if
  end do

  ! Compute CGS outputs
  call dimless_to_cgs(state)
  lum_obs = state%lum_obs_cgs
  if (present(r_ph_out)) r_ph_out = state%r_ph_cgs

  ! Debug diagnostics (gated by environment variable)
  if (mode3_debug_enabled()) then
   write(6,'(A,ES12.4)') 'DEBUG dimless: L_cgs=', state%lum_obs_cgs
   write(6,'(A,ES12.4)') 'DEBUG dimless: x_sh=', state%x_sh
   write(6,'(A,ES12.4)') 'DEBUG dimless: w_sh=', state%w_sh
   write(6,'(A,ES12.4)') 'DEBUG dimless: x_ph=', state%x_ph
   write(6,'(A,ES12.4)') 'DEBUG dimless: zeta=', state%zeta
   write(6,'(A,ES12.4)') 'DEBUG dimless: e_N=', state%e_grid(state%n_zones)
   write(6,'(A,ES12.4)') 'DEBUG dimless: Delta_x=', state%x_ph - state%x_min
  end if

 end subroutine dimless_comoving_transport_step

end module csm_transport
