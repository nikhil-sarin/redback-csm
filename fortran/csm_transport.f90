module csm_transport

 use constants, only: pi, clight
 use physical_constants, only: a_rad
use get_vals, only: op, query_csm_density, query_csm_inner_edge, query_csm_outer_edge, &
                     query_tau_to_edge, query_csm_photosphere_radius, query_csm_velocity, &
                     query_ejecta_density, query_ejecta_velocity
use integration, only: dudt, drdt, dmdt, forward_shock_luminosity, reverse_shock_luminosity, &
                       forward_shock_radiative_efficiency, reverse_shock_radiative_efficiency
use csm_runtime, only: shock_efficiency_mode

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
 integer, parameter :: max_dimless_history = 20000

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
  real(8) :: E_breakout_cgs = 0d0 ! rapidly escaping breakout-depth energy [erg]
  real(8) :: t_breakout_cgs = 0d0 ! breakout release timescale [s]
  real(8) :: lum_breakout_cgs = 0d0     ! instantaneous breakout-depth leakage [erg/s]
  real(8) :: lum_breakout_avg_cgs = 0d0 ! substep-averaged breakout leakage [erg/s]
  real(8) :: E_breakout_output_cgs = 0d0 ! breakout energy emitted during caller step [erg]
  real(8) :: dt_breakout_output_cgs = 0d0 ! caller-step time represented by breakout output [s]
  real(8) :: E_cooling_tail_cgs = 0d0    ! trapped shocked-ejecta/shell reservoir [erg]
  real(8) :: t_cooling_tail0_cgs = 0d0   ! leakage time at shock emergence [s]
  real(8) :: lum_cooling_tail_cgs = 0d0  ! substep-averaged tail leakage [erg/s]
  real(8) :: E_surface_handoff_cgs = 0d0 ! photospheric radiation carried across handoff [erg]
  real(8) :: t_surface_handoff_cgs = 0d0 ! surface leakage time after handoff [s]
  real(8) :: lum_surface_handoff_cgs = 0d0 ! substep-averaged surface leakage [erg/s]
  real(8) :: tau_ahead_csm = 0d0        ! optical depth from shock to CSM edge
  logical :: breakout_active = .false.  ! true when tau_ahead <= c/v_sh

  ! Cooling geometry (constant after handoff)
  real(8) :: x_min_cool = 1d0   ! inner support in cooling x (usually 1)
  real(8) :: x_out_cool = 0d0   ! outer edge in comoving x = R_csm_out/R0 (frozen at handoff)
  real(8) :: x_sh_se = 1d0      ! interaction shock coordinate at emergence
  real(8) :: eta_cool_scale = 1d0 ! density normalization for cooling shell mass
  ! Shock heating powers [erg/s].  The diffusion inner boundary uses
  ! lum_heat_cgs as the paper's L_heat boundary condition.
  real(8) :: lum_heat_cgs = 0d0
  real(8) :: lum_heat_total_cgs = 0d0
  real(8) :: lum_heat_fs_cgs = 0d0
  real(8) :: lum_heat_rs_cgs = 0d0

  ! Diffusion grid (fixed ξ-space, ξ ∈ [0,1])
  real(8) :: x_ph = 0d0         ! photosphere position in x
  real(8) :: x_min = 1d0        ! inner boundary of diffusion domain in x
  real(8) :: x_sh_dot = 0d0     ! dx_sh/dy (shock velocity in y-units)
  real(8) :: x_ph_dot = 0d0     ! dx_ph/dy for the moving photospheric boundary
  integer :: rannacher_left = 0
  real(8) :: x_ph_cached_xsh = -1d0  ! x_sh when x_ph was last computed
  real(8) :: x_ph_cached_xmin = -1d0 ! x_min when x_ph was last computed
  logical :: x_ph_cache_valid = .false.
  real(8) :: x_ph_cache_start = -1d0
  real(8) :: x_ph_cache_outer = -1d0
  real(8) :: x_ph_cache_scale = -1d0
  logical :: csm_powerlaw_fast = .false.
  real(8) :: csm_eta_pow = 0d0
  logical :: csm_cache_valid = .false.
  logical :: csm_cache_cooling = .false.
  real(8) :: csm_cache_time = -1d0
  real(8) :: csm_cache_x_inner = -1d0
  real(8) :: csm_cache_x_outer = -1d0
  integer :: csm_cache_n = 0
  real(8), allocatable :: csm_cache_x(:), csm_cache_eta(:), csm_cache_int(:)
  real(8), allocatable :: xi_grid(:)   ! fixed uniform ξ = (i-1)/(n-1)
  real(8), allocatable :: e_grid(:)    ! dimensionless energy density e(ξ)
  ! Frozen η_csm on the cooling ξ-grid (set at handoff, used throughout cooling)
  real(8), allocatable :: eta_cool_grid(:)
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
  integer :: diag_step_counter = 0

  ! Cumulative energy tracking for cooling handoff.
  real(8) :: E_injected_cum = 0d0   ! cumulative total shock heating [erg]
  real(8) :: E_radiated_cum = 0d0   ! cumulative diffusive emission [erg]
  real(8) :: E_injected_fs_cum = 0d0 ! cumulative forward-shock heating [erg]
  real(8) :: E_injected_rs_cum = 0d0 ! cumulative reverse-shock heating [erg]
  ! Shock-deposition history for the post-emergence cooling IC.
  integer :: n_history = 0
  real(8) :: hist_x(max_dimless_history) = 0d0
  real(8) :: hist_e(max_dimless_history) = 0d0
  ! Explicit shocked-shell internal-energy field.  This is the e_int(x)
  ! named by the cooling appendix: it is built as the shock sweeps CSM layers
  ! and drained by luminosity that escapes during the interaction phase.
  real(8), allocatable :: shell_x(:)
  real(8), allocatable :: shell_e(:)
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
  ! Full-grid transport keeps the complete CSM support active during the
  ! interaction phase. The shock is an internal heating front, not a moving
  ! computational boundary, so the radiation field behind the shock is the
  ! cooling initial condition at emergence.
  state%r_inner = state%r_inner_support
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
  state%radius = state%radius_ref
  state%rho = state%rho_ref

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

  ! Full-grid interaction: cell geometry is fixed over the complete CSM.
  ! r_shell is retained in the interface for compatibility with older callers.
  call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
end subroutine interaction_zone_geometry_at

subroutine update_interaction_geometry(state, r_shell, t_shell, m_shell)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: r_shell, t_shell, m_shell

  state%t_shell = t_shell
  state%r_shell_prev = state%r_shell_current
  state%r_shell_current = max(r_shell, 1d0)

  ! The full diffusion solve keeps a fixed CSM grid until shock emergence.
  ! This is the key difference from the Appendix-A moving-boundary solver:
  ! swept material stays on-grid, so no internal energy is discarded or
  ! reconstructed at t_se.
  state%r_inner = state%r_inner_support
  state%r_outer = state%r_outer_support
  state%radius = state%radius_ref
  state%rho = state%rho_ref
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

  ! Full-grid transport keeps swept material on the grid, so there is no
  ! unresolved behind-shock reservoir to reconstruct at emergence.
  state%e_residual = 0d0
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
  real(8) :: tau_lo, tau_hi, frac, r_floor, r_cross

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
    r_cross = state%radius(i-1) + frac * (state%radius(i) - state%radius(i-1))
    r_cross = min(max(r_cross, state%radius(i-1)), state%radius(i))
    r_ph = max(r_cross, r_floor)
    if (r_ph > r_cross * (1d0 + 1d-12)) then
     ! If the optical-depth crossing is behind the shock/inner cooling edge,
     ! the boundary is physically at r_floor.  Keep iph consistent with that
     ! clamped radius; otherwise the matrix truncates before the shock-heating
     ! cell and drops the breakout source term.
     iph = zone_index_at_current_radius(state, r_ph, jstart)
    end if
    return
   end if
  end do
end subroutine photosphere_boundary_from_tau

integer function zone_index_at_current_radius(state, r_target, istart) result(idx)
  type(transport_state_type), intent(in) :: state
  real(8), intent(in) :: r_target
  integer, intent(in) :: istart

  integer :: i, ilo
  real(8) :: r_face_l, r_face_r, vol_i, r_loc

  ilo = min(max(istart, 1), state%n_zones)
  idx = ilo
  if (state%n_zones <= 0) return

  r_loc = min(max(r_target, state%r_inner), state%r_outer)
  do i = ilo, state%n_zones
   call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   if (r_loc >= r_face_l .and. r_loc <= r_face_r) then
    idx = i
    return
   end if
  end do
  idx = state%n_zones
end function zone_index_at_current_radius

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
  ! This finite-volume solver evolves physical radii, densities, and radiation
  ! energy density directly. Expansion factors that appear in the dimensionless
  ! Appendix-A Robin coefficient are therefore already represented by geometry,
  ! rho ∝ R^-3, and y ∝ R^-4. The boundary conductance should remain the
  ! physical Eddington/free-streaming closure F = c u / 4 in both phases.
  stream_factor = 0.25d0
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
  flux_out = 0.25d0 * clight * e_n
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

  integer :: i, n, ihi, iph, ish
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
  ish = 1
  if (.not. state%in_cooling_phase) ish = first_active_zone_at(state, state%r_shell_current)
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
    if (i == ish .and. state%r_shell_current >= state%r_inner_support .and. &
        state%r_shell_current <= state%r_outer_support) then
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
   if (max(lum_heat, 0d0) > 0d0 .and. .not. state%in_cooling_phase) then
    if (iph == ish .and. state%r_shell_current >= state%r_inner_support .and. &
        state%r_shell_current <= state%r_outer_support) then
     state%work_rhs(iph) = state%work_rhs(iph) + max(lum_heat, 0d0)
    end if
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
  if (allocated(state%eta_cool_grid)) deallocate(state%eta_cool_grid)
  if (allocated(state%csm_cache_x)) deallocate(state%csm_cache_x)
  if (allocated(state%csm_cache_eta)) deallocate(state%csm_cache_eta)
  if (allocated(state%csm_cache_int)) deallocate(state%csm_cache_int)
  if (allocated(state%shell_x)) deallocate(state%shell_x)
  if (allocated(state%shell_e)) deallocate(state%shell_e)
  state%initialized = .false.
  state%in_cooling_phase = .false.
  state%n_zones = 0
  state%x_sh = 1d0
  state%w_sh = 1d0
   state%phi_sh = 1d-6
  state%zeta = 1d0
  state%y_diff = 0d0
   state%R_in_R0 = 1d0
   state%x_sh_dot = 0d0
   state%x_ph_dot = 0d0
   state%rannacher_left = 0
  state%x_ph_cached_xsh = -1d0
  state%x_ph_cached_xmin = -1d0
  state%x_ph_cache_valid = .false.
  state%x_ph_cache_start = -1d0
  state%x_ph_cache_outer = -1d0
  state%x_ph_cache_scale = -1d0
  state%csm_powerlaw_fast = .false.
  state%csm_eta_pow = 0d0
  state%csm_cache_valid = .false.
  state%csm_cache_cooling = .false.
  state%csm_cache_time = -1d0
  state%csm_cache_x_inner = -1d0
  state%csm_cache_x_outer = -1d0
  state%csm_cache_n = 0
  state%lum_heat_cgs = 0d0
  state%lum_heat_total_cgs = 0d0
  state%lum_heat_fs_cgs = 0d0
  state%lum_heat_rs_cgs = 0d0
  state%lum_obs_cgs = 0d0
  state%x_min_cool = 1d0
  state%x_out_cool = 0d0
  state%x_sh_se = 1d0
  state%eta_cool_scale = 1d0
  state%diag_step_counter = 0
  state%E_injected_cum = 0d0
  state%E_radiated_cum = 0d0
  state%E_injected_fs_cum = 0d0
  state%E_injected_rs_cum = 0d0
  state%E_breakout_cgs = 0d0
  state%t_breakout_cgs = 0d0
  state%lum_breakout_cgs = 0d0
  state%lum_breakout_avg_cgs = 0d0
  state%E_breakout_output_cgs = 0d0
  state%dt_breakout_output_cgs = 0d0
  state%E_cooling_tail_cgs = 0d0
  state%t_cooling_tail0_cgs = 0d0
  state%lum_cooling_tail_cgs = 0d0
  state%E_surface_handoff_cgs = 0d0
  state%t_surface_handoff_cgs = 0d0
  state%lum_surface_handoff_cgs = 0d0
  state%tau_ahead_csm = 0d0
  state%breakout_active = .false.
  state%n_history = 0
  state%hist_x = 0d0
  state%hist_e = 0d0
 end subroutine reset_dimless_state

! ------------------------------------------------------------------
! Compute characteristic scales from op(1), op(2)
! Paper Eq. 728-742, 796-800, 836-838
! ------------------------------------------------------------------
 subroutine initialize_dimless_state(state, kappa_in, eff_in, n_zones_in)
  type(dimless_state_type), intent(inout) :: state
  real(8), intent(in) :: kappa_in, eff_in
  integer, intent(in) :: n_zones_in

  real(8) :: t_dim, rho_edge, rho_inside
  real(8) :: t_lo, t_hi, t_mid, f_hi, f_mid
  integer :: iter_contact

  call reset_dimless_state(state)

  state%kappa = max(kappa_in, 1d-30)
  state%eff = eff_in
  state%n_zones = max(n_zones_in, 3)

  ! v_ej_max: maximum ejecta velocity
  if (op(1)%bpl_vmax > 0d0) then
   state%v_ej_max = op(1)%bpl_vmax
  elseif (op(1)%bpl_vt > 0d0) then
   ! The untruncated BPL branch has no formal finite outer edge.  The legacy
   ! TransFit dynamics initialize the fastest ejecta at 100*v_tr, so use the
   ! same effective v_ej,max for the Appendix-A nondimensionalization.
   state%v_ej_max = 1d2 * op(1)%bpl_vt
  elseif (op(1)%exp_v0 > 0d0) then
   state%v_ej_max = 3d0 * op(1)%exp_v0
  elseif (associated(op(1)%v_grid)) then
   state%v_ej_max = maxval(op(1)%v_grid)
  else
   state%v_ej_max = 1d9
  end if

  ! First contact with the CSM.  For static CSM this reduces to
  ! t_in=R_csm_in/v_ej,max (paper Eq. 728).  For wind-like arbitrary CSM the
  ! inner support moves outward, so solve v_ej,max*t = R_csm,in(t) instead of
  ! sampling the edge at an arbitrary early time.
  t_lo = 0d0
  t_hi = max(query_csm_inner_edge(1d1, op(2)) / max(state%v_ej_max, 1d-30), 1d-6)
  f_hi = state%v_ej_max * t_hi - query_csm_inner_edge(t_hi, op(2))
  do while (f_hi < 0d0 .and. t_hi < 1d12)
   t_hi = 2d0 * t_hi
   f_hi = state%v_ej_max * t_hi - query_csm_inner_edge(t_hi, op(2))
  end do
  if (f_hi < 0d0) then
   state%t_in = max(query_csm_inner_edge(1d1, op(2)) / max(state%v_ej_max, 1d-30), 1d-6)
  else
   do iter_contact = 1, 80
    t_mid = 0.5d0 * (t_lo + t_hi)
    f_mid = state%v_ej_max * t_mid - query_csm_inner_edge(t_mid, op(2))
    if (f_mid >= 0d0) then
     t_hi = t_mid
    else
     t_lo = t_mid
    end if
   end do
   state%t_in = max(t_hi, 1d-6)
  end if

  ! R_csm_in at first contact.
  state%R_csm_in = max(state%v_ej_max * state%t_in, query_csm_inner_edge(state%t_in, op(2)), 1d0)

  ! Reference densities at t = t_in, r = R_csm_in
  t_dim = state%t_in
  rho_edge = query_csm_density(state%R_csm_in, t_dim, op(2))
  rho_inside = query_csm_density(state%R_csm_in * (1d0 + 1d-8), t_dim, op(2))
  state%rho_csm_in = max(max(rho_edge, rho_inside), 1d-30)
  state%rho_ej_in = max(query_ejecta_density(state%R_csm_in, t_dim, op(1)), 1d-30)
  state%csm_powerlaw_fast = static_powerlaw_csm_slope(state, state%csm_eta_pow)

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
   state%phi_sh = 1d-6
   state%zeta = 1d0
   state%y_diff = state%y_ratio * state%zeta
   state%x_sh_dot = state%w_sh / max(state%y_ratio, 1d-30)
   state%x_ph_dot = 0d0
  ! Compute initial photosphere
  call estimate_photosphere_x(state%x_sh, state)

  ! Set up fixed ξ-grid (only allocates once, never remaps)
  call setup_xi_grid(state)

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
  allocate(state%shell_x(n), state%shell_e(n))

  ! Fixed uniform ξ ∈ [0,1]
  do i = 1, n
   state%xi_grid(i) = dble(i - 1) / dble(max(n - 1, 1))
   state%shell_x(i) = 1d0 + (dble(i) - 0.5d0) * &
                      (max(state%x_csm_out, 1.0001d0) - 1d0) / dble(max(n, 1))
  end do

  state%e_grid = 0d0
  state%shell_e = 0d0
 end subroutine setup_xi_grid

! ------------------------------------------------------------------
 function raw_wind_density_cgs(r_dim, t_dim) result(rho_csm)
  real(8), intent(in) :: r_dim, t_dim
  real(8) :: rho_csm, t_wind, mdot_wind, frac
  integer :: lo, hi, mid, n

  rho_csm = -1d0
  if (.not. associated(op(2)%t_grid) .or. .not. associated(op(2)%mdot)) return
  if (op(2)%vwind <= 0d0) return

  n = size(op(2)%t_grid)
  if (n <= 0) return
  t_wind = abs(r_dim / op(2)%vwind - t_dim)

  if (n == 1 .or. t_wind <= op(2)%t_grid(1)) then
   mdot_wind = op(2)%mdot(1)
  else if (t_wind >= op(2)%t_grid(n)) then
   mdot_wind = op(2)%mdot(n)
  else
   lo = 1
   hi = n
   do while (hi - lo > 1)
    mid = (lo + hi) / 2
    if (op(2)%t_grid(mid) <= t_wind) then
     lo = mid
    else
     hi = mid
    end if
   end do
   frac = (t_wind - op(2)%t_grid(lo)) / &
          max(op(2)%t_grid(lo+1) - op(2)%t_grid(lo), 1d-30)
   frac = min(max(frac, 0d0), 1d0)
   mdot_wind = (1d0 - frac) * op(2)%mdot(lo) + frac * op(2)%mdot(lo+1)
  end if

  rho_csm = mdot_wind / (4d0 * pi * op(2)%vwind * max(r_dim, 1d0)**2)
 end function raw_wind_density_cgs

! ------------------------------------------------------------------
! Compute eta_csm(x):
!   Interaction: η_csm(x) = ρ_csm(x·R_csm_in) / ρ_csm_in  [paper Eq. 738]
!   Cooling: η_csm(x) is time-independent (frozen at t_se)
!            Evaluate the CSM profile at t_se in interaction coordinates
!            Paper Eq. 897: ρ = ρ₀·(R0/R_in)³·η_csm(x)
!            The (R0/R_in)³ factor is handled in the PDE/D scaling,
!            not here. This function returns the comoving profile only.
! ------------------------------------------------------------------
 function raw_eta_csm(x, state) result(eta)
  type(dimless_state_type), intent(in) :: state
  real(8), intent(in) :: x
  real(8) :: eta, r_dim, t_dim, x_pre, rho_fast

  if (state%in_cooling_phase) then
   ! Cooling phase uses the frozen density profile in the homologous
   ! coordinate x=r/R_in(t) [paper Eq. 896-907].  This profile must not be
   ! remapped to the current photosphere as x_ph recedes; eta_csm(x) is a
   ! fixed function over the comoving shell support [x_min_cool,x_out_cool].
   if (state%x_out_cool > state%x_min_cool + 1d-12) then
    x_pre = 1d0 + (x - state%x_min_cool) * (state%x_sh_se - 1d0) / &
            (state%x_out_cool - state%x_min_cool)
   else
    x_pre = state%x_sh_se
   end if
   x_pre = min(max(x_pre, 1d0), max(state%x_sh_se, 1d0))
   if (state%csm_powerlaw_fast) then
    eta = max(x_pre, 1d-300)**state%csm_eta_pow
   else
    r_dim = x_pre * state%R_csm_in
    t_dim = state%zeta_se * state%t_in
    rho_fast = raw_wind_density_cgs(r_dim, t_dim)
    if (rho_fast >= 0d0) then
     eta = rho_fast / max(state%rho_csm_in, 1d-30)
    else
     eta = query_csm_density(r_dim, t_dim, op(2)) / max(state%rho_csm_in, 1d-30)
    end if
   end if
   eta = max(eta, 1d-30)
  else
   ! Interaction coordinate: x = r/R_csm_in
   if (state%csm_powerlaw_fast) then
    eta = max(x, 1d-300)**state%csm_eta_pow
   else
    r_dim = x * state%R_csm_in
    t_dim = state%zeta * state%t_in
    rho_fast = raw_wind_density_cgs(r_dim, t_dim)
    if (rho_fast >= 0d0) then
     eta = rho_fast / max(state%rho_csm_in, 1d-30)
    else
     eta = query_csm_density(r_dim, t_dim, op(2)) / max(state%rho_csm_in, 1d-30)
    end if
   end if
  end if
 end function raw_eta_csm

 subroutine ensure_csm_cache(state, x_inner_in, x_outer_in)
  type(dimless_state_type), intent(inout) :: state
  real(8), intent(in) :: x_inner_in, x_outer_in

  integer :: i, n_cache
  real(8) :: x_inner, x_outer, t_cache, rel_t, rel_x0, rel_x1, dx, x_eval

  if (state%csm_powerlaw_fast) return

  x_inner = max(min(x_inner_in, x_outer_in), 1d-12)
  x_outer = max(x_outer_in, x_inner * (1d0 + 1d-12))
  if (state%in_cooling_phase) then
   t_cache = state%zeta_se * state%t_in
  else
   t_cache = state%zeta * state%t_in
  end if

  if (state%csm_cache_valid) then
   rel_t = abs(t_cache - state%csm_cache_time) / max(abs(state%csm_cache_time), 1d0)
   rel_x0 = abs(x_inner - state%csm_cache_x_inner) / max(abs(state%csm_cache_x_inner), 1d0)
   rel_x1 = abs(x_outer - state%csm_cache_x_outer) / max(abs(state%csm_cache_x_outer), 1d0)
   if (state%csm_cache_cooling .eqv. state%in_cooling_phase) then
    if (rel_t < 1d-10 .and. rel_x0 < 1d-10 .and. rel_x1 < 1d-10) return
   end if
  end if

  n_cache = max(96, min(512, 4 * max(state%n_zones, 24)))
  if (allocated(state%csm_cache_x)) then
   if (size(state%csm_cache_x) /= n_cache) then
    deallocate(state%csm_cache_x, state%csm_cache_eta, state%csm_cache_int)
   end if
  end if
  if (.not. allocated(state%csm_cache_x)) then
   allocate(state%csm_cache_x(n_cache), state%csm_cache_eta(n_cache), state%csm_cache_int(n_cache))
  end if

  do i = 1, n_cache
   state%csm_cache_x(i) = x_inner + (x_outer - x_inner) * dble(i - 1) / dble(max(n_cache - 1, 1))
  end do

  dx = (x_outer - x_inner) / dble(max(n_cache - 1, 1))
  do i = 1, n_cache
   ! Sample one-sided at the support edges.  Tabulated winds can intentionally
   ! jump at the inner/outer edge; transport needs the cell value just ahead of
   ! the shock, not the constructor's exact-boundary convention.
   x_eval = state%csm_cache_x(i)
   if (i == 1) x_eval = min(x_outer, x_inner + 0.5d0 * dx)
   if (i == n_cache) x_eval = max(x_inner, x_outer - 0.5d0 * dx)
   state%csm_cache_eta(i) = max(raw_eta_csm(x_eval, state), 0d0)
  end do

  state%csm_cache_int(n_cache) = 0d0
  do i = n_cache - 1, 1, -1
   dx = state%csm_cache_x(i+1) - state%csm_cache_x(i)
   state%csm_cache_int(i) = state%csm_cache_int(i+1) + &
       0.5d0 * (state%csm_cache_eta(i) + state%csm_cache_eta(i+1)) * dx
  end do

  state%csm_cache_valid = .true.
  state%csm_cache_cooling = state%in_cooling_phase
  state%csm_cache_time = t_cache
  state%csm_cache_x_inner = x_inner
  state%csm_cache_x_outer = x_outer
  state%csm_cache_n = n_cache
 end subroutine ensure_csm_cache

 function interp_csm_cache_eta(x, state) result(eta)
  type(dimless_state_type), intent(in) :: state
  real(8), intent(in) :: x
  real(8) :: eta, pos, frac
  integer :: i, n

  eta = 0d0
  if (.not. state%csm_cache_valid) return
  n = state%csm_cache_n
  if (n < 2) return
  if (x <= state%csm_cache_x(1)) then
   eta = state%csm_cache_eta(1)
   return
  end if
  if (x >= state%csm_cache_x(n)) then
   eta = state%csm_cache_eta(n)
   return
  end if
  pos = (x - state%csm_cache_x(1)) / max(state%csm_cache_x(n) - state%csm_cache_x(1), 1d-30) * dble(n - 1)
  i = min(max(int(pos) + 1, 1), n - 1)
  frac = min(max(pos - dble(i - 1), 0d0), 1d0)
  eta = (1d0 - frac) * state%csm_cache_eta(i) + frac * state%csm_cache_eta(i+1)
 end function interp_csm_cache_eta

 function interp_csm_cache_int(x, state) result(integ)
  type(dimless_state_type), intent(in) :: state
  real(8), intent(in) :: x
  real(8) :: integ, pos, frac
  integer :: i, n

  integ = 0d0
  if (.not. state%csm_cache_valid) return
  n = state%csm_cache_n
  if (n < 2) return
  if (x <= state%csm_cache_x(1)) then
   integ = state%csm_cache_int(1)
   return
  end if
  if (x >= state%csm_cache_x(n)) then
   integ = 0d0
   return
  end if
  pos = (x - state%csm_cache_x(1)) / max(state%csm_cache_x(n) - state%csm_cache_x(1), 1d-30) * dble(n - 1)
  i = min(max(int(pos) + 1, 1), n - 1)
  frac = min(max(pos - dble(i - 1), 0d0), 1d0)
  integ = (1d0 - frac) * state%csm_cache_int(i) + frac * state%csm_cache_int(i+1)
 end function interp_csm_cache_int

 function compute_eta_csm(x, state) result(eta)
  type(dimless_state_type), intent(in) :: state
  real(8), intent(in) :: x
  real(8) :: eta

  if (state%csm_cache_valid .and. x >= state%csm_cache_x_inner - 1d-12 .and. &
      x <= state%csm_cache_x_outer + 1d-12 .and. &
      (state%csm_cache_cooling .eqv. state%in_cooling_phase)) then
   eta = interp_csm_cache_eta(x, state)
  else
   eta = raw_eta_csm(x, state)
  end if
 end function compute_eta_csm

 ! Linear interpolation from a ξ-gridded cooling η profile.
 ! Used by compute_eta_csm in the cooling phase to look up the frozen
 ! η_csm(ξ) stored at handoff.
 subroutine interp_eta_cool(xi, xi_grid, eta_grid, n, eta)
  real(8), intent(in) :: xi
  real(8), intent(in) :: xi_grid(:)
  real(8), intent(in) :: eta_grid(:)
  integer, intent(in) :: n
  real(8), intent(out) :: eta
  integer :: j
  real(8) :: frac

  if (n < 2) then
   eta = eta_grid(1)
   return
  end if

  if (xi <= xi_grid(1)) then
   eta = eta_grid(1)
   return
  end if
  if (xi >= xi_grid(n)) then
   eta = eta_grid(n)
   return
  end if

  do j = 1, n - 1
   if (xi >= xi_grid(j) .and. xi <= xi_grid(j+1)) then
    frac = (xi - xi_grid(j)) / max(xi_grid(j+1) - xi_grid(j), 1d-30)
    eta = (1d0 - frac) * eta_grid(j) + frac * eta_grid(j+1)
    return
   end if
  end do
  eta = eta_grid(n)
 end subroutine interp_eta_cool

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
 logical function static_powerlaw_csm_slope(state, eta_pow) result(ok)
  type(dimless_state_type), intent(in) :: state
  real(8), intent(out) :: eta_pow

  integer :: n, im
  real(8) :: r1, r2, rm, rho1, rho2, rhom, rho_pred

  ok = .false.
  eta_pow = 0d0
  if (.not. associated(op(2)%r_grid_static)) return
  if (.not. associated(op(2)%rho_static)) return
  n = size(op(2)%r_grid_static)
  if (n < 3) return

  r1 = op(2)%r_grid_static(1)
  r2 = op(2)%r_grid_static(n)
  rho1 = op(2)%rho_static(1)
  rho2 = op(2)%rho_static(n)
  if (r2 <= r1 .or. r1 <= 0d0 .or. rho1 <= 0d0 .or. rho2 <= 0d0) return

  eta_pow = log(rho2 / rho1) / log(r2 / r1)

  ! Guard the analytic path so arbitrary tabulated CSM profiles still use the
  ! general quadrature/inversion code.
  im = max(2, min(n - 1, n / 2))
  rm = op(2)%r_grid_static(im)
  rhom = op(2)%rho_static(im)
  if (rm <= 0d0 .or. rhom <= 0d0) return
  rho_pred = rho1 * (rm / r1)**eta_pow
  if (abs(log(max(rhom, 1d-300) / max(rho_pred, 1d-300))) > 1d-3) return

  ok = .true.
 end function static_powerlaw_csm_slope

 subroutine analytic_powerlaw_tau_from_x(x_from, state, tau_out, ok)
  real(8), intent(in) :: x_from
  type(dimless_state_type), intent(in) :: state
  real(8), intent(out) :: tau_out
  logical, intent(out) :: ok

  real(8) :: eta_pow, x_end, length_scale, p, integ

  tau_out = 0d0
  if (state%csm_powerlaw_fast) then
   ok = .true.
   eta_pow = state%csm_eta_pow
  else
   ok = static_powerlaw_csm_slope(state, eta_pow)
  end if
  if (.not. ok) return

  if (state%in_cooling_phase) then
   x_end = max(state%x_out_cool, x_from)
   length_scale = state%R0 / state%R_in_R0**2
  else
   x_end = max(state%x_csm_out, x_from)
   length_scale = state%R_csm_in
  end if
  if (x_end <= x_from + 1d-12) then
   tau_out = 0d0
   return
  end if

  p = eta_pow + 1d0
  if (abs(p) < 1d-12) then
   integ = log(max(x_end, 1d-300) / max(x_from, 1d-300))
  else
   integ = (x_end**p - x_from**p) / p
  end if
  tau_out = max(state%kappa * state%rho_csm_in * length_scale * integ, 0d0)
 end subroutine analytic_powerlaw_tau_from_x

 subroutine analytic_powerlaw_photosphere_x(x_start, state, found)
  real(8), intent(in) :: x_start
  type(dimless_state_type), intent(inout) :: state
  logical, intent(out) :: found

  real(8) :: eta_pow, x_inner, x_outer, length_scale, tau_cum, tau_target
  real(8) :: p, rhs, x_ph_analytic
  real(8), parameter :: tau_ph = 2d0 / 3d0

  if (state%csm_powerlaw_fast) then
   found = .true.
   eta_pow = state%csm_eta_pow
  else
   found = static_powerlaw_csm_slope(state, eta_pow)
  end if
  if (.not. found) return

  if (state%in_cooling_phase) then
   x_inner = max(state%x_min_cool, 1d-12)
   x_outer = max(state%x_out_cool, x_inner)
   length_scale = state%R0 / state%R_in_R0**2
  else
   x_inner = max(x_start, 1d0)
   x_outer = max(state%x_csm_out, x_inner)
   length_scale = state%R_csm_in
  end if

  if (x_outer <= x_inner + 1d-12) then
   state%x_ph = x_inner
   return
  end if

  call analytic_powerlaw_tau_from_x(x_inner, state, tau_cum, found)
  if (.not. found) return
  if (tau_cum <= tau_ph) then
   if (.not. state%in_cooling_phase) then
    ! Once the remaining shock-to-edge layer is optically thinner than 2/3,
    ! no tau=2/3 surface exists ahead of the shock.  The emitting surface is
    ! then shock-adjacent/free-streaming; jumping the transport boundary to
    ! x_outer artificially re-expands the diffusion column and creates a notch
    ! just before shock emergence.
    if (state%x_ph_cache_valid .and. state%x_ph > x_inner) then
     state%x_ph = min(max(state%x_ph, x_inner + max(1d-3 * (x_outer - x_inner), 1d-6)), x_outer)
    else
     state%x_ph = min(x_inner + max(1d-3 * (x_outer - x_inner), 1d-6), x_outer)
    end if
   else
    state%x_ph = x_outer
   end if
  else
   tau_target = tau_ph / max(state%kappa * state%rho_csm_in * length_scale, 1d-300)
   p = eta_pow + 1d0
   if (abs(p) < 1d-12) then
    x_ph_analytic = x_outer * exp(-tau_target)
   else
    rhs = x_outer**p - p * tau_target
    if (rhs <= 0d0 .and. p > 0d0) then
     x_ph_analytic = x_inner
    else
     x_ph_analytic = max(rhs, 1d-300)**(1d0 / p)
    end if
   end if
   state%x_ph = min(max(x_ph_analytic, x_inner), x_outer)
  end if

  state%x_ph_cache_valid = .true.
  state%x_ph_cache_start = x_inner
  state%x_ph_cache_outer = x_outer
  state%x_ph_cache_scale = length_scale
 end subroutine analytic_powerlaw_photosphere_x

 subroutine estimate_photosphere_x(x_start, state)
  real(8), intent(in) :: x_start
  type(dimless_state_type), intent(inout) :: state

  integer :: i
  real(8) :: x_lo, x_hi, tau_cum, x_mid, tau_mid, tau_guess
  real(8) :: x_inner, x_outer, length_scale
  real(8) :: rel_start, rel_outer, rel_scale, x_guess
  logical :: analytic_found
  real(8), parameter :: tau_ph = 2d0 / 3d0

  ! Determine integration bounds and length scale
	  if (state%in_cooling_phase) then
	   x_inner = max(state%x_min_cool, 1d-12)
	   ! Cooling uses a comoving coordinate tied to the expanding shell support.
	   ! x_min_cool and x_out_cool are frozen at handoff and remain fixed in x.
	   x_outer = max(state%x_out_cool, x_inner)
	   ! τ = κ·ρ_csm_in·(R0/R_in)³·R_in·∫ η_csm dx = κ·ρ_csm_in·R0³/R_in²·∫ η_csm dx
	   length_scale = state%R0**3 / (state%R0 * state%R_in_R0)**2
	   ! Keep the cooling diffusion solve on the full homologous shell support.
	   ! The previous moving-photosphere boundary shrank the computational
	   ! domain without conservatively remapping the radiation field, which
	   ! numerically removed the post-breakout reservoir.  The photosphere
	   ! recession should be handled as an emission/diagnostic surface, not by
	   ! deleting the outer shell from the diffusion state.
	   state%x_ph = x_outer
	   state%x_ph_cache_valid = .true.
	   state%x_ph_cache_start = x_inner
	   state%x_ph_cache_outer = x_outer
	   state%x_ph_cache_scale = length_scale
	   return
	  else
   x_inner = max(x_start, 1d0)
   x_outer = max(state%x_csm_out, x_inner)
   length_scale = state%R_csm_in
  end if

  call analytic_powerlaw_photosphere_x(x_start, state, analytic_found)
  if (analytic_found) return

  if (x_outer <= x_inner + 1d-12) then
   state%x_ph = x_inner
   state%x_ph_cache_valid = .false.
   return
  end if

  if (state%x_ph_cache_valid .and. state%x_ph > 0d0) then
   rel_start = abs(x_inner - state%x_ph_cache_start) / max(abs(state%x_ph_cache_start), 1d0)
   rel_outer = abs(x_outer - state%x_ph_cache_outer) / max(abs(state%x_ph_cache_outer), 1d0)
   rel_scale = abs(length_scale - state%x_ph_cache_scale) / max(abs(state%x_ph_cache_scale), 1d-30)
   if (state%in_cooling_phase) then
    ! During cooling the density scale changes smoothly as R_in^-2.  Reusing
    ! the last photosphere over sub-percent scale changes avoids thousands of
    ! identical optical-depth inversions during the early cooling refinement.
    if (rel_start < 1d-12 .and. rel_outer < 1d-12 .and. rel_scale < 2d-3) then
     state%x_ph = min(max(state%x_ph, x_inner), x_outer)
     return
    end if
   else
    ! In interaction the tau=2/3 surface is independent of the inner boundary
    ! until the shock approaches it.  Cache only when the boundary moved very
    ! little, so breakout timing remains controlled by explicit inversions.
    if (rel_start < 5d-4 .and. rel_outer < 1d-12 .and. state%x_ph > x_inner) then
     state%x_ph = min(max(state%x_ph, x_inner), x_outer)
     return
    end if
   end if
  end if

  call integrate_tau_from_x(x_inner, state, tau_cum)

  if (tau_cum <= tau_ph) then
   ! Optically thin over the active interaction layer: no tau=2/3 surface lies
   ! between the shock and the outer CSM edge.  Keep the boundary adjacent to
   ! the shock/free-streaming surface instead of jumping to x_outer, which
   ! would discontinuously create a new diffusion column.
   if (.not. state%in_cooling_phase) then
    if (state%x_ph_cache_valid .and. state%x_ph > x_inner) then
     state%x_ph = min(max(state%x_ph, x_inner + max(1d-3 * (x_outer - x_inner), 1d-6)), x_outer)
    else
     state%x_ph = min(x_inner + max(1d-3 * (x_outer - x_inner), 1d-6), x_outer)
    end if
   else
    state%x_ph = x_outer
   end if
   state%x_ph_cache_valid = .true.
   state%x_ph_cache_start = x_inner
   state%x_ph_cache_outer = x_outer
   state%x_ph_cache_scale = length_scale
   return
  end if

  ! Bisection: find x_ph where tau from x_ph to x_outer = 2/3
  x_lo = x_inner
  x_hi = x_outer
  if (state%x_ph_cache_valid) then
   x_guess = min(max(state%x_ph, x_inner), x_outer)
   if (x_guess > x_inner + 1d-12 .and. x_guess < x_outer - 1d-12) then
    call integrate_tau_from_x(x_guess, state, tau_guess)
    if (tau_guess > tau_ph) then
     x_lo = x_guess
    else
     x_hi = x_guess
    end if
   end if
  end if

  do i = 1, 22
   x_mid = 0.5d0 * (x_lo + x_hi)
   call integrate_tau_from_x(x_mid, state, tau_mid)
   if (tau_mid > tau_ph) then
    x_lo = x_mid
   else
    x_hi = x_mid
   end if
  end do
  state%x_ph = 0.5d0 * (x_lo + x_hi)
  state%x_ph_cache_valid = .true.
  state%x_ph_cache_start = x_inner
  state%x_ph_cache_outer = x_outer
  state%x_ph_cache_scale = length_scale
 end subroutine estimate_photosphere_x

 subroutine integrate_tau_from_x(x_from, state, tau_out)
  real(8), intent(in) :: x_from
  type(dimless_state_type), intent(inout) :: state
  real(8), intent(out) :: tau_out

 integer, parameter :: n_int = 32
  integer :: i
  real(8) :: x_end, dx, x_mid, eta_val, length_scale, x_cache_start
  logical :: analytic_ok

  call analytic_powerlaw_tau_from_x(x_from, state, tau_out, analytic_ok)
  if (analytic_ok) return

  if (state%in_cooling_phase) then
   ! Cooling phase uses the frozen η_csm profile (set at handoff) and the
   ! (R0/R_in)³ expansion factor.  The optical depth integral is:
   !   τ = ∫κ·ρ·dr = κ·ρ_csm_in·(R0/R_in)³·∫η_frozen·R_in·dx
   !     = κ·ρ_csm_in·R0³/R_in²·∫η_frozen·dx
   !   length_scale = R0³/(R0·R_in_R0)² = R0/R_in_R0²
   ! The cooling coordinate is comoving with the homologous shell, so the
   ! shell support is fixed in x.  The physical radius grows through the
   ! R_in/R0 factors, not by shrinking x_out_cool.
   x_end = max(state%x_out_cool, x_from)
   length_scale = state%R0 / state%R_in_R0**2
  else
   x_end = max(state%x_csm_out, x_from)
   length_scale = state%R_csm_in
  end if
  if (x_end <= x_from + 1d-12) then
   tau_out = 0d0
   return
  end if
  if (state%in_cooling_phase) then
   x_cache_start = min(x_from, state%x_min_cool)
  else
   x_cache_start = min(x_from, max(state%x_sh, 1d-12))
  end if
  call ensure_csm_cache(state, x_cache_start, x_end)
  if (state%csm_cache_valid) then
   tau_out = max(state%kappa * state%rho_csm_in * length_scale * &
                 interp_csm_cache_int(x_from, state), 0d0)
   return
  end if
  dx = (x_end - x_from) / dble(n_int)
  tau_out = 0d0
  do i = 1, n_int
   x_mid = x_from + (dble(i) - 0.5d0) * dx
   eta_val = compute_eta_csm(x_mid, state)
   tau_out = tau_out + state%kappa * state%rho_csm_in * length_scale * eta_val * dx
 end do
 end subroutine integrate_tau_from_x

 real(8) function dimless_total_radiation_energy(state) result(e_tot)
  type(dimless_state_type), intent(in) :: state
  integer :: i, n
  real(8) :: x_lo, Delta_x, dxi, x_i, x_ip1

  e_tot = 0d0
  n = state%n_zones
  if (n < 2) return

  if (state%in_cooling_phase) then
   x_lo = max(state%x_min_cool, 1d-12)
  else
   x_lo = max(state%x_sh, 1d-12)
  end if
  Delta_x = state%x_ph - x_lo
  if (Delta_x <= 0d0) return

  dxi = 1d0 / dble(n - 1)
  do i = 1, n - 1
   x_i = x_lo + state%xi_grid(i) * Delta_x
   x_ip1 = x_lo + state%xi_grid(i+1) * Delta_x
   e_tot = e_tot + 0.5d0 * (x_i**2 * max(state%e_grid(i), 0d0) + &
                            x_ip1**2 * max(state%e_grid(i+1), 0d0)) * Delta_x * dxi
  end do
 end function dimless_total_radiation_energy

 real(8) function dimless_boundary_heating_luminosity(state) result(lum)
  type(dimless_state_type), intent(in) :: state
  real(8) :: eta_sh, f_ib

  if (state%in_cooling_phase) then
   lum = 0d0
   return
  end if

  eta_sh = max(compute_eta_csm(state%x_sh, state), 1d-30)
  f_ib = max(state%eff * eta_sh**2 * state%w_sh**3, 0d0)
  lum = 4d0 * pi * state%x_sh**2 * state%R_csm_in**2 * clight * state%u0 * f_ib / &
        max(state%tau_csm_in * eta_sh, 1d-30)
 end function dimless_boundary_heating_luminosity

real(8) function current_dimless_shock_luminosity(state) result(lum)
  type(dimless_state_type), intent(in) :: state
  real(8) :: lum_fs, lum_rs

  lum = 0d0
  if (state%in_cooling_phase) return

  call current_dimless_shock_components(state, lum_fs, lum_rs)
  lum = max(lum_fs + lum_rs, 0d0)
end function current_dimless_shock_luminosity

subroutine update_dimless_shock_luminosities(state)
  type(dimless_state_type), intent(inout) :: state

  if (state%in_cooling_phase) then
   state%lum_heat_fs_cgs = 0d0
   state%lum_heat_rs_cgs = 0d0
   state%lum_heat_total_cgs = 0d0
   state%lum_heat_cgs = 0d0
   return
  end if

  call current_dimless_shock_components(state, state%lum_heat_fs_cgs, state%lum_heat_rs_cgs)
  state%lum_heat_total_cgs = max(state%lum_heat_fs_cgs + state%lum_heat_rs_cgs, 0d0)
  ! The observable diffusion column is sourced by the total shock power in the
  ! thin-shell approximation; the component bookkeeping is retained for
  ! diagnostics and possible future resolved e_int closures.
  state%lum_heat_cgs = state%lum_heat_total_cgs
end subroutine update_dimless_shock_luminosities

subroutine current_dimless_shock_components(state, lum_fs, lum_rs)
  type(dimless_state_type), intent(in) :: state
  real(8), intent(out) :: lum_fs, lum_rs
  real(8) :: r_dim, t_dim, u_dim
  real(8) :: eta_fs, eta_rs

  lum_fs = 0d0
  lum_rs = 0d0
  if (state%in_cooling_phase) return

  r_dim = max(state%x_sh * state%R_csm_in, 1d0)
  t_dim = max(state%zeta * state%t_in, 1d-30)
  u_dim = max(state%w_sh * state%v_ej_max, 0d0)

  lum_fs = forward_shock_luminosity(r_dim, t_dim, u_dim, op)
  lum_rs = reverse_shock_luminosity(r_dim, t_dim, u_dim, op)
  select case (shock_efficiency_mode)
  case (1)
   eta_fs = forward_shock_radiative_efficiency(r_dim, t_dim, u_dim, op, state%eff)
   eta_rs = reverse_shock_radiative_efficiency(r_dim, t_dim, u_dim, op, state%eff)
   lum_fs = eta_fs * lum_fs
   lum_rs = eta_rs * lum_rs
  case default
   lum_fs = state%eff * lum_fs
   lum_rs = state%eff * lum_rs
  end select

  lum_fs = max(lum_fs, 0d0)
  lum_rs = max(lum_rs, 0d0)
end subroutine current_dimless_shock_components

real(8) function interaction_breakout_timescale(state, tau_ahead) result(t_esc)
 type(dimless_state_type), intent(in) :: state
 real(8), intent(in) :: tau_ahead
 real(8) :: dr_ahead, t_ang

 dr_ahead = max((state%x_csm_out - state%x_sh) * state%R_csm_in, 0d0)
 t_ang = max(state%x_sh * state%R_csm_in / clight, 1d-30)
 if (dr_ahead <= 0d0) then
  t_esc = t_ang
 else
  ! Diffusion time through the material ahead of the shock.  The light-crossing
  ! term keeps the breakout reservoir causal once the remaining optical depth
  ! is below unity; the angular time sets the observed spherical breakout
  ! smoothing scale as the shock approaches the outer edge.
  t_esc = max(max(tau_ahead, 1d0) * dr_ahead / clight, t_ang, 1d-30)
 end if
end function interaction_breakout_timescale

subroutine advance_breakout_reservoir(state, dt_cgs, lum_in_cgs, t_esc_cgs)
 type(dimless_state_type), intent(inout) :: state
 real(8), intent(in) :: dt_cgs, lum_in_cgs, t_esc_cgs

 real(8) :: e_old, e_new, decay, t_esc, lum_in

 state%lum_breakout_avg_cgs = 0d0
 if (dt_cgs <= 0d0) return

 e_old = max(state%E_breakout_cgs, 0d0)
 lum_in = max(lum_in_cgs, 0d0)
 if (e_old <= 0d0 .and. lum_in <= 0d0) then
  state%E_breakout_cgs = 0d0
  state%lum_breakout_cgs = 0d0
  return
 end if

 t_esc = max(t_esc_cgs, 1d-30)
 decay = exp(-min(dt_cgs / t_esc, 80d0))
 e_new = e_old * decay + lum_in * t_esc * (1d0 - decay)

 state%lum_breakout_avg_cgs = max((e_old + lum_in * dt_cgs - e_new) / dt_cgs, 0d0)
 state%E_breakout_output_cgs = state%E_breakout_output_cgs + state%lum_breakout_avg_cgs * dt_cgs
 state%dt_breakout_output_cgs = state%dt_breakout_output_cgs + dt_cgs
 state%E_breakout_cgs = max(e_new, 0d0)
 state%t_breakout_cgs = t_esc
 state%lum_breakout_cgs = state%E_breakout_cgs / t_esc
end subroutine advance_breakout_reservoir

subroutine advance_cooling_tail_reservoir(state, dt_cgs)
 type(dimless_state_type), intent(inout) :: state
 real(8), intent(in) :: dt_cgs

 real(8) :: e_old, e_new, emitted, decay, r_ratio
 real(8) :: t_leak, t_exp, leak_rate, ad_rate, total_rate

 state%lum_cooling_tail_cgs = 0d0
 if (dt_cgs <= 0d0) return
 if (.not. state%in_cooling_phase) return

 e_old = max(state%E_cooling_tail_cgs, 0d0)
 if (e_old <= 0d0) then
  state%E_cooling_tail_cgs = 0d0
  return
 end if

 r_ratio = max(state%R_in_R0, 1d-12)
 ! Homologous expansion gives tau ∝ R^-2 and path length ∝ R, so the
 ! diffusion/leakage time of an unresolved trapped shell scales as R^-1.
 t_leak = max(state%t_cooling_tail0_cgs / r_ratio, &
              state%x_out_cool * state%R0 * r_ratio / clight, 1d-30)
 ! The unresolved reverse-shocked ejecta reservoir is a coarse closure for the
 ! broken-power-law ejecta side of the shell, not the resolved CSM radiation
 ! field.  Use a softened PdV sink so this component behaves like an ejecta
 ! shock-cooling reservoir rather than losing all bolometric power on one
 ! expansion time.
 t_exp = max(state%R0 * r_ratio / max(state%v_se, 1d-30), 1d-30)
 leak_rate = 1d0 / t_leak
 ad_rate = 0.15d0 / t_exp
 total_rate = leak_rate + ad_rate
 decay = exp(-min(total_rate * dt_cgs, 80d0))
 emitted = e_old * (1d0 - decay) * leak_rate / max(total_rate, 1d-30)
 e_new = e_old * decay

 state%E_cooling_tail_cgs = max(e_new, 0d0)
 state%lum_cooling_tail_cgs = max(emitted / dt_cgs, 0d0)
end subroutine advance_cooling_tail_reservoir

subroutine advance_surface_handoff_reservoir(state, dt_cgs)
 type(dimless_state_type), intent(inout) :: state
 real(8), intent(in) :: dt_cgs

 real(8) :: e_old, decay, emitted, t_emit

 state%lum_surface_handoff_cgs = 0d0
 if (dt_cgs <= 0d0) return
 if (.not. state%in_cooling_phase) return

 e_old = max(state%E_surface_handoff_cgs, 0d0)
 if (e_old <= 0d0) then
  state%E_surface_handoff_cgs = 0d0
  return
 end if

 t_emit = max(state%t_surface_handoff_cgs, 1d-30)
 decay = exp(-min(dt_cgs / t_emit, 80d0))
 emitted = e_old * (1d0 - decay)
 state%E_surface_handoff_cgs = max(e_old * decay, 0d0)
 state%lum_surface_handoff_cgs = max(emitted / dt_cgs, 0d0)
end subroutine advance_surface_handoff_reservoir

subroutine prepare_interaction_heating(state, dt_cgs)
 type(dimless_state_type), intent(inout) :: state
 real(8), intent(in) :: dt_cgs

 real(8) :: tau_ahead, tau_breakout, t_esc, v_sh
 real(8) :: lum_trapped
 integer, save :: breakout_log_counter = 0

 state%lum_breakout_avg_cgs = 0d0
 state%breakout_active = .false.
 if (state%E_breakout_cgs <= 0d0) state%lum_breakout_cgs = 0d0
 if (state%in_cooling_phase) then
  state%lum_heat_cgs = 0d0
  return
 end if

 call integrate_tau_from_x(state%x_sh, state, tau_ahead)
 state%tau_ahead_csm = max(tau_ahead, 0d0)
 v_sh = max(state%w_sh * state%v_ej_max, 1d5)
 tau_breakout = clight / v_sh

  lum_trapped = max(state%lum_heat_total_cgs, 0d0)

 if (state%tau_ahead_csm <= tau_breakout) then
  state%breakout_active = .true.
  ! Breakout-depth status is diagnostic during the interaction solve.  The
  ! appendix transport equation already responds to the shrinking optical depth
  ! through the moving inner boundary; diverting all newly generated shock heat
  ! into a separate reservoir before emergence creates an artificial shock-power
  ! plateau and a discontinuous jump at t_se.
 end if
 if (state%E_breakout_cgs > 0d0) then
  t_esc = max(state%t_breakout_cgs, interaction_breakout_timescale(state, state%tau_ahead_csm))
  call advance_breakout_reservoir(state, dt_cgs, 0d0, t_esc)
 end if
 state%lum_heat_cgs = lum_trapped

 if (mode3_debug_enabled()) then
  breakout_log_counter = breakout_log_counter + 1
  if (breakout_log_counter <= 80 .or. mod(breakout_log_counter, 200) == 0) then
   write(6,'(A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,L1)') &
    'M3_BO tau=', state%tau_ahead_csm, 'taubo=', tau_breakout, 'Ltot=', state%lum_heat_total_cgs, &
    'Ltrap=', state%lum_heat_cgs, 'Lbo=', state%lum_breakout_cgs, 'tbo=', state%t_breakout_cgs, &
    'active=', state%breakout_active
  end if
 end if
end subroutine prepare_interaction_heating

subroutine release_interaction_breakout_reservoir(state, dt_cgs)
 type(dimless_state_type), intent(inout) :: state
 real(8), intent(in) :: dt_cgs

 real(8) :: v_sh, tau_breakout, escape_weight, tau_inner
 real(8) :: t_transfer, t_emit, t_raw, t_bo_cross, eta_sh
 real(8) :: E_field_dim, E_field_cgs, E_available, E_transfer, lum_in

 if (state%in_cooling_phase) return
 if (dt_cgs <= 0d0) return
 if (state%tau_ahead_csm <= 0d0) return

 v_sh = max(state%w_sh * state%v_ej_max, 1d5)
 tau_breakout = clight / v_sh
 if (state%tau_ahead_csm >= tau_breakout) return
 call integrate_tau_from_x(1d0, state, tau_inner)
 if (tau_inner <= tau_breakout) return

 ! The diffusion grid contains radiation still in the unshocked transport
 ! layer.  The remainder of the global energy budget is shocked-shell energy
 ! left behind the moving boundary.  When tau_ahead <= c/v_sh, that reservoir
 ! can leak through the remaining CSM during the interaction phase.
 E_field_dim = dimless_total_radiation_energy(state)
 E_field_cgs = 4d0 * pi * state%u0 * state%R_csm_in**3 * E_field_dim
 E_available = state%E_injected_cum - state%E_radiated_cum - &
               state%E_breakout_cgs - E_field_cgs
 E_available = max(E_available, 0d0)
 if (E_available <= 0d0) return

 escape_weight = 1d0 - state%tau_ahead_csm / max(tau_breakout, 1d-30)
 escape_weight = min(max(escape_weight, 0d0), 1d0)
 if (escape_weight <= 0d0) return

 t_raw = interaction_breakout_timescale(state, state%tau_ahead_csm)
 t_transfer = max(t_raw / max(escape_weight, 1d-6), 1d-30)
 eta_sh = max(compute_eta_csm(state%x_sh, state), 1d-30)
 t_bo_cross = tau_breakout / max(state%kappa * state%rho_csm_in * eta_sh * v_sh, 1d-30)
 t_emit = max(t_raw, t_bo_cross, state%x_sh * state%R_csm_in / clight, 1d-30)

 E_transfer = E_available * (1d0 - exp(-min(dt_cgs / t_transfer, 80d0)))
 E_transfer = min(max(E_transfer, 0d0), E_available)
 if (E_transfer <= 0d0) return

 lum_in = E_transfer / dt_cgs
 call advance_breakout_reservoir(state, dt_cgs, lum_in, t_emit)
end subroutine release_interaction_breakout_reservoir

subroutine dimless_surface_flux_diag(state, dedx_surf, lum_grad_cgs)
  type(dimless_state_type), intent(in) :: state
  real(8), intent(out) :: dedx_surf, lum_grad_cgs

  integer :: n
  real(8) :: x_lo, Delta_x, dx
  real(8) :: eta_ph, rho_ph, u_prefac, r_scale, du_dr, flux_cgs, r_ph

  dedx_surf = 0d0
  lum_grad_cgs = 0d0

  n = state%n_zones
  if (n < 2) return
  if (state%kappa <= 0d0 .or. state%rho_csm_in <= 0d0 .or. state%u0 <= 0d0) return

  if (state%in_cooling_phase) then
   x_lo = max(state%x_min_cool, 1d-12)
  else
   x_lo = max(state%x_sh, 1d-12)
  end if
  Delta_x = state%x_ph - x_lo
  if (Delta_x <= 0d0) return

  dx = Delta_x / dble(n - 1)
  if (dx <= 0d0) return
  dedx_surf = (state%e_grid(n) - state%e_grid(n-1)) / dx

  eta_ph = max(compute_eta_csm(state%x_ph, state), 1d-30)
  if (state%in_cooling_phase) then
   r_scale = state%R0 * max(state%R_in_R0, 1d-30)
   u_prefac = state%u0 / max(state%R_in_R0, 1d-30)**4
   rho_ph = state%rho_csm_in * eta_ph / max(state%R_in_R0, 1d-30)**3
  else
   r_scale = state%R_csm_in
   u_prefac = state%u0
   rho_ph = state%rho_csm_in * eta_ph
  end if
  if (r_scale <= 0d0 .or. rho_ph <= 0d0) return

  du_dr = u_prefac * dedx_surf / r_scale
  flux_cgs = -clight * du_dr / (3d0 * state%kappa * rho_ph)
  r_ph = state%x_ph * r_scale
  lum_grad_cgs = 4d0 * pi * r_ph**2 * max(flux_cgs, 0d0)
 end subroutine dimless_surface_flux_diag

 subroutine record_deposition_profile(state, x_left_in, x_right_in, lum_heat, dt_cgs)
  type(dimless_state_type), intent(inout) :: state
  real(8), intent(in) :: x_left_in, x_right_in, lum_heat, dt_cgs

  real(8) :: x_left, x_right, x_mid, dvol_dimless, e_dep
  integer :: n

  if (state%in_cooling_phase) return
  if (lum_heat <= 0d0 .or. dt_cgs <= 0d0) return
  if (state%u0 <= 0d0 .or. state%R_csm_in <= 0d0) return

  x_left = min(max(min(x_left_in, x_right_in), 1d0), state%x_csm_out)
  x_right = min(max(max(x_left_in, x_right_in), 1d0), state%x_csm_out)
  if (x_right <= x_left + 1d-10) return

  dvol_dimless = (x_right**3 - x_left**3) / 3d0
  if (dvol_dimless <= 0d0) return

  x_mid = 0.5d0 * (x_left + x_right)

  ! The deposited internal-energy density follows directly from the swept
  ! shell volume: dE = 4*pi*u0*R_csm_in^3*e_dep*int x^2 dx.
  e_dep = lum_heat * dt_cgs / max(4d0*pi*state%u0*state%R_csm_in**3*dvol_dimless, 1d-30)
  if (.not.(e_dep > 0d0 .and. e_dep < 1d30)) return

  if (state%n_history <= 0) then
   state%n_history = 1
   state%hist_x(1) = x_mid
   state%hist_e(1) = e_dep
   return
  end if

  n = state%n_history
  if (x_mid <= state%hist_x(n) + 1d-10) then
   state%hist_x(n) = max(state%hist_x(n), x_mid)
   state%hist_e(n) = state%hist_e(n) + e_dep
   return
  end if

  if (n < max_dimless_history) then
   state%n_history = n + 1
   state%hist_x(state%n_history) = x_mid
   state%hist_e(state%n_history) = e_dep
  else
   ! Preserve monotonic coverage if an unusually fine run exhausts storage.
   state%hist_x(n) = x_mid
   state%hist_e(n) = state%hist_e(n) + e_dep
  end if
 end subroutine record_deposition_profile

 real(8) function deposition_profile_at(state, x_cool) result(e_dep)
  type(dimless_state_type), intent(in) :: state
  real(8), intent(in) :: x_cool

  integer :: j
  real(8) :: x_pre, frac

  e_dep = 0d0
  if (state%n_history <= 0) return

  if (state%x_out_cool > state%x_min_cool + 1d-12) then
   x_pre = 1d0 + (x_cool - state%x_min_cool) * (state%x_sh_se - 1d0) / &
           (state%x_out_cool - state%x_min_cool)
  else
   x_pre = state%x_sh_se
  end if
  x_pre = min(max(x_pre, 1d0), max(state%x_sh_se, 1d0))

  if (x_pre <= state%hist_x(1)) then
   e_dep = max(state%hist_e(1), 0d0)
   return
  end if
  if (x_pre >= state%hist_x(state%n_history)) then
   e_dep = max(state%hist_e(state%n_history), 0d0)
   return
  end if

  do j = 1, state%n_history - 1
   if (x_pre >= state%hist_x(j) .and. x_pre <= state%hist_x(j+1)) then
    frac = (x_pre - state%hist_x(j)) / max(state%hist_x(j+1) - state%hist_x(j), 1d-30)
    e_dep = max((1d0 - frac) * state%hist_e(j) + frac * state%hist_e(j+1), 0d0)
    return
   end if
  end do
  end function deposition_profile_at

subroutine shell_bin_bounds(state, i, x_l, x_r)
  type(dimless_state_type), intent(in) :: state
  integer, intent(in) :: i
  real(8), intent(out) :: x_l, x_r
  real(8) :: dx_shell

  dx_shell = (max(state%x_csm_out, 1.0001d0) - 1d0) / dble(max(state%n_zones, 1))
  x_l = 1d0 + dble(i - 1) * dx_shell
  x_r = 1d0 + dble(i) * dx_shell
end subroutine shell_bin_bounds

subroutine advance_shocked_shell_energy(state, x_left_in, x_right_in, lum_heat, dt_cgs)
  type(dimless_state_type), intent(inout) :: state
  real(8), intent(in) :: x_left_in, x_right_in, lum_heat, dt_cgs

  integer :: i
  real(8) :: x_left, x_right, bin_l, bin_r, ov_l, ov_r
  real(8) :: sweep_vol, e_add, E_dim

  if (.not. allocated(state%shell_e)) return
  if (state%in_cooling_phase) return
  if (lum_heat <= 0d0 .or. dt_cgs <= 0d0) return
  if (state%u0 <= 0d0 .or. state%R_csm_in <= 0d0) return

  x_left = min(max(min(x_left_in, x_right_in), 1d0), state%x_csm_out)
  x_right = min(max(max(x_left_in, x_right_in), 1d0), state%x_csm_out)
  if (x_right <= x_left + 1d-10) return

  sweep_vol = (x_right**3 - x_left**3) / 3d0
  if (sweep_vol <= 0d0) return

  E_dim = lum_heat * dt_cgs / max(4d0 * pi * state%u0 * state%R_csm_in**3, 1d-30)
  e_add = E_dim / sweep_vol
  if (.not. (e_add > 0d0 .and. e_add < 1d30)) return

  do i = 1, state%n_zones
   call shell_bin_bounds(state, i, bin_l, bin_r)
   ov_l = max(bin_l, x_left)
   ov_r = min(bin_r, x_right)
   if (ov_r > ov_l) state%shell_e(i) = state%shell_e(i) + e_add
  end do
end subroutine advance_shocked_shell_energy

subroutine drain_shocked_shell_energy(state, E_loss_cgs)
  type(dimless_state_type), intent(inout) :: state
  real(8), intent(in) :: E_loss_cgs

  integer :: i, i_start
  real(8) :: remaining, bin_l, bin_r, bin_vol, E_bin, dE, x_drain

  if (.not. allocated(state%shell_e)) return
  if (E_loss_cgs <= 0d0) return
  if (state%u0 <= 0d0 .or. state%R_csm_in <= 0d0) return

  remaining = E_loss_cgs / max(4d0 * pi * state%u0 * state%R_csm_in**3, 1d-30)
  if (remaining <= 0d0) return

  ! Escaping photons drain the photospheric shocked layers first.  Material
  ! outside x_ph is already optically thin and does not set the post-emergence
  ! diffusion boundary, so spending the loss budget there leaves an artificial
  ! hot photospheric surface at handoff.
  x_drain = min(max(state%x_ph, 1d0), max(state%x_csm_out, 1d0))
  i_start = state%n_zones
  do i = 1, state%n_zones
   call shell_bin_bounds(state, i, bin_l, bin_r)
   if (x_drain >= bin_l .and. x_drain <= bin_r) then
    i_start = i
    exit
   end if
  end do

  do i = i_start, 1, -1
   call shell_bin_bounds(state, i, bin_l, bin_r)
   bin_vol = (bin_r**3 - bin_l**3) / 3d0
   if (bin_vol <= 0d0) cycle
   E_bin = max(state%shell_e(i), 0d0) * bin_vol
   if (E_bin <= 0d0) cycle
   dE = min(E_bin, remaining)
   state%shell_e(i) = max(state%shell_e(i) - dE / bin_vol, 0d0)
   remaining = remaining - dE
   if (remaining <= 0d0) exit
  end do

  if (remaining <= 0d0) return

  do i = i_start + 1, state%n_zones
   call shell_bin_bounds(state, i, bin_l, bin_r)
   bin_vol = (bin_r**3 - bin_l**3) / 3d0
   if (bin_vol <= 0d0) cycle
   E_bin = max(state%shell_e(i), 0d0) * bin_vol
   if (E_bin <= 0d0) cycle
   dE = min(E_bin, remaining)
   state%shell_e(i) = max(state%shell_e(i) - dE / bin_vol, 0d0)
   remaining = remaining - dE
   if (remaining <= 0d0) exit
  end do
end subroutine drain_shocked_shell_energy

real(8) function shocked_shell_dimless_energy(state) result(E_dim)
  type(dimless_state_type), intent(in) :: state

  integer :: i
  real(8) :: bin_l, bin_r, bin_vol

  E_dim = 0d0
  if (.not. allocated(state%shell_e)) return
  do i = 1, state%n_zones
   call shell_bin_bounds(state, i, bin_l, bin_r)
   bin_vol = (bin_r**3 - bin_l**3) / 3d0
   E_dim = E_dim + max(state%shell_e(i), 0d0) * max(bin_vol, 0d0)
  end do
end function shocked_shell_dimless_energy

subroutine enforce_shocked_shell_energy_budget(state)
  type(dimless_state_type), intent(inout) :: state

  real(8) :: E_field_dim, E_field_cgs, E_target_cgs, E_target_dim
  real(8) :: E_shell_dim, scale

  if (.not. allocated(state%shell_e)) return
  if (state%in_cooling_phase) return
  if (state%u0 <= 0d0 .or. state%R_csm_in <= 0d0) return

  E_field_dim = dimless_total_radiation_energy(state)
  E_field_cgs = 4d0 * pi * state%u0 * state%R_csm_in**3 * E_field_dim
  ! shell_e tracks the forward-shock/surface reservoir only.  Diffusive
  ! luminosity escapes through the surface, so remove it from this component
  ! before the reverse-shock/mixed reservoir is constructed at handoff.
  E_target_cgs = state%E_injected_fs_cum - state%E_radiated_cum - &
                 state%E_breakout_cgs - E_field_cgs
  E_target_cgs = max(E_target_cgs, 0d0)
  E_target_dim = E_target_cgs / max(4d0 * pi * state%u0 * state%R_csm_in**3, 1d-30)

  E_shell_dim = shocked_shell_dimless_energy(state)
  if (E_target_dim <= 0d0) then
   state%shell_e = 0d0
  else if (E_shell_dim > 0d0) then
   scale = E_target_dim / E_shell_dim
   state%shell_e = max(state%shell_e * scale, 0d0)
  end if
end subroutine enforce_shocked_shell_energy_budget

real(8) function shocked_shell_profile_at(state, x_cool) result(e_shell)
  type(dimless_state_type), intent(in) :: state
  real(8), intent(in) :: x_cool

  integer :: j
  real(8) :: x_pre, frac

  e_shell = 0d0
  if (.not. allocated(state%shell_e)) return
  if (state%n_zones <= 0) return

  if (state%x_out_cool > state%x_min_cool + 1d-12) then
   x_pre = 1d0 + (x_cool - state%x_min_cool) * (state%x_sh_se - 1d0) / &
           (state%x_out_cool - state%x_min_cool)
  else
   x_pre = state%x_sh_se
  end if
  x_pre = min(max(x_pre, 1d0), max(state%x_sh_se, 1d0))

  if (x_pre <= state%shell_x(1)) then
   e_shell = max(state%shell_e(1), 0d0)
   return
  end if
  if (x_pre >= state%shell_x(state%n_zones)) then
   e_shell = max(state%shell_e(state%n_zones), 0d0)
   return
  end if

  do j = 1, state%n_zones - 1
   if (x_pre >= state%shell_x(j) .and. x_pre <= state%shell_x(j+1)) then
    frac = (x_pre - state%shell_x(j)) / max(state%shell_x(j+1) - state%shell_x(j), 1d-30)
    e_shell = max((1d0 - frac) * state%shell_e(j) + frac * state%shell_e(j+1), 0d0)
    return
   end if
  end do
end function shocked_shell_profile_at

  ! ------------------------------------------------------------------
  ! Dynamics RHS: paper Eq. 746-750
! ------------------------------------------------------------------
 subroutine dynamics_rhs(x_in, w_in, phi_in, zeta_in, state, dx_dz, dw_dz, dphi_dz)
  type(dimless_state_type), intent(in) :: state
  real(8), intent(in) :: x_in, w_in, phi_in, zeta_in
  real(8), intent(out) :: dx_dz, dw_dz, dphi_dz
  real(8) :: eta_ej_val, eta_csm_val, v_rel_ej, q_loc, w_eff

  ! Save current zeta for eta evaluation
  ! (compute_eta_ej uses state%zeta, but we want the passed zeta_in)

  eta_csm_val = compute_eta_csm(x_in, state)
  eta_ej_val = compute_eta_ej(x_in, zeta_in, state)

  ! The thin-shell equations assume an outward-moving shock.  RK stage
  ! extrapolations can briefly try w<0 in dense finite CSM; clamp the stage
  ! velocity to the physical branch instead of feeding an invalid negative
  ! shock speed into the ram-pressure terms.
  w_eff = max(w_in, 0d0)

  ! v_ej at shell = R_sh/t = (x*R_csm_in)/(zeta*t_in) = x/zeta * v_ej_max
  ! Relative velocity: (v_ej - v_sh)/v_ej_max = x/zeta - w
  v_rel_ej = x_in / max(zeta_in, 1d-30) - w_eff

  q_loc = state%q

  ! Paper Eq. 746
  dx_dz = w_eff

  ! Paper Eq. 748: dphi/dz = x^2 * zeta^{-3} * (x/zeta - w) * eta_ej + q * x^2 * w * eta_csm
  dphi_dz = x_in**2 * max(zeta_in, 1d-30)**(-3) * max(v_rel_ej, 0d0) * eta_ej_val &
           + q_loc * x_in**2 * w_eff * eta_csm_val

  ! Paper Eq. 749: dw/dz = (1/phi)[x^2*zeta^{-3}*(x/zeta-w)^2*eta_ej - q*x^2*w^2*eta_csm]
  if (phi_in > 1d-30) then
   dw_dz = (x_in**2 * max(zeta_in, 1d-30)**(-3) * max(v_rel_ej, 0d0)**2 * eta_ej_val &
           - q_loc * x_in**2 * w_eff**2 * eta_csm_val) / phi_in
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
  state%x_sh = max(x + dzeta * (k1_x + 2d0*k2_x + 2d0*k3_x + k4_x) / 6d0, 1d0)
  state%w_sh = max(w + dzeta * (k1_w + 2d0*k2_w + 2d0*k3_w + k4_w) / 6d0, 1d-10)
  state%phi_sh = max(phi + dzeta * (k1_phi + 2d0*k2_phi + 2d0*k3_phi + k4_phi) / 6d0, 1d-30)
  if (.not.(state%x_sh == state%x_sh) .or. .not.(state%w_sh == state%w_sh) .or. &
      .not.(state%phi_sh == state%phi_sh)) then
   state%x_sh = max(x, 1d0)
   state%w_sh = max(w, 1d-10)
   state%phi_sh = max(phi, 1d-30)
  end if
  state%zeta = zeta + dzeta
  state%y_diff = state%y_ratio * state%zeta
 end subroutine rk4_step_dynamics

! ------------------------------------------------------------------
! Diffusion solver in ξ-coordinates: Crank-Nicolson
! PDE: ∂e/∂y = (1/Δx²)·(1/x̂²)·∂/∂ξ[D(ξ)·∂e/∂ξ] + V(ξ)·∂e/∂ξ
! where x̂(ξ) = x_min + ξ·Δx, Δx = x_ph - x_min
! Interaction: x_min = x_sh, V(ξ) = [ẋ_sh·(1-ξ)+ẋ_ph·ξ]/Δx
! Cooling: x_min = x_min_cool, V(ξ) = ẋ_ph·ξ/Δx in the comoving frame
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
   real(8) :: f_ib, f_ob, V_sh, V_ph
  real(8) :: eta_min_val, f_ib_lum
   real(8) :: L_from_f_ib, f_ib_ratio
  real(8) :: dx_bc
  real(8) :: inner_rate
  integer, save :: ibc_log_counter = 0

  n = state%n_zones
  if (n < 3) return

  ! Domain: interaction → [x_sh, x_ph], cooling → [x_min, x_ph]
  if (.not. state%in_cooling_phase) then
   state%x_min = state%x_sh
  else
   state%x_min = max(state%x_min_cool, 1d-12)
  end if

  Delta_x = state%x_ph - state%x_min
  if (Delta_x <= 0d0) then
   return
  end if
  dxi = 1d0 / dble(n - 1)  ! Uniform ξ spacing

  state%work_old_e = state%e_grid

  ! Crank-Nicolson by default; use backward-Euler only during Rannacher startup.
  theta = 0.5d0
  if (state%rannacher_left > 0) then
   theta = 1d0
   state%rannacher_left = state%rannacher_left - 1
  end if
  omt = 1d0 - theta

  ! Advection velocity in ξ-coordinates.  The interaction grid is fixed in
  ! ξ but both endpoints can move: the shock advances and the photosphere
  ! recedes as the remaining optical depth changes.
   if (.not. state%in_cooling_phase) then
   ! Interaction: V(ξ) = [ẋ_sh·(1-ξ)+ẋ_ph·ξ]/Δx
    V_sh = state%x_sh_dot / Delta_x
    V_ph = state%x_ph_dot / Delta_x
   else
    ! Cooling: the homologous coordinate is comoving, but the numerical
    ! diffusion interval ends at the photosphere, which recedes through that
    ! coordinate as the optical depth drops.
    V_sh = 0d0
    V_ph = state%x_ph_dot / Delta_x
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
    gamma_v = V_sh * (1d0 - state%xi_grid(i)) + V_ph * state%xi_grid(i)

   ! Check Péclet number for upwinding
   pe_cell = abs(gamma_v) * dxi / max(alpha_m + alpha_p, 1d-60) * 2d0

    if (pe_cell > 2d0 .and. gamma_v > 0d0) then
     ! PDE term is +gamma*dE/dxi, equivalent to advection speed -gamma.
     ! Positive gamma therefore upwinds from the right.
     state%work_a(i) = -theta * dy * alpha_m
     state%work_c(i) = -theta * dy * (alpha_p + gamma_v / dxi)
     state%work_b(i) = 1d0 + theta * dy * (alpha_m + alpha_p + gamma_v / dxi)

     state%work_rhs(i) = state%work_old_e(i) &
         + omt * dy * alpha_m * state%work_old_e(i-1) &
         + omt * dy * (alpha_p + gamma_v / dxi) * state%work_old_e(i+1) &
         - omt * dy * (alpha_m + alpha_p + gamma_v / dxi) * state%work_old_e(i)
    else if (pe_cell > 2d0 .and. gamma_v < 0d0) then
     ! Negative gamma upwinds from the left.
     state%work_a(i) = -theta * dy * (alpha_m - gamma_v / dxi)
     state%work_c(i) = -theta * dy * alpha_p
     state%work_b(i) = 1d0 + theta * dy * (alpha_m + alpha_p - gamma_v / dxi)

     state%work_rhs(i) = state%work_old_e(i) &
         + omt * dy * (alpha_m - gamma_v / dxi) * state%work_old_e(i-1) &
         + omt * dy * alpha_p * state%work_old_e(i+1) &
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
   ! Neumann BC: inject the shock power specified by the main text,
   ! F = L_heat/(4*pi*R_sh^2).  In the appendix scaling this is
   ! f_ib = L_heat*tau_csm_in*eta/(4*pi*x_sh^2*R_csm_in^2*c*u0).
   ! With outward x, positive outward flux implies de/dx < 0.
   eta_min_val = compute_eta_csm(state%x_min, state)
   if (state%u0 > 0d0 .and. state%x_sh > 0d0) then
    f_ib_lum = state%lum_heat_cgs * state%tau_csm_in * max(eta_min_val, 1d-30) / &
                (4d0 * pi * state%x_sh**2 * state%R_csm_in**2 * clight * state%u0)
    f_ib = max(f_ib_lum, 0d0)
   else
    f_ib = 0d0
   end if
   L_from_f_ib = 4d0 * pi * state%x_sh**2 * state%R_csm_in**2 * clight * state%u0 * f_ib &
                 / max(state%tau_csm_in * max(eta_min_val, 1d-30), 1d-30)
   if (state%lum_heat_cgs > 0d0) then
    f_ib_ratio = L_from_f_ib / state%lum_heat_cgs
   else
    f_ib_ratio = 0d0
   end if
   if (mode3_debug_enabled()) then
    ibc_log_counter = ibc_log_counter + 1
    if (ibc_log_counter <= 80 .or. mod(ibc_log_counter, 200) == 0) then
     write(6,'(A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4)') &
      'M3_IBC Lheat=', state%lum_heat_cgs, 'f_ib=', f_ib, 'L_from_f_ib=', L_from_f_ib, &
      'ratio=', f_ib_ratio, 'x_sh=', state%x_sh, 'x_ph=', state%x_ph, 'eta_min=', eta_min_val
    end if
   end if
    x_hat_i = state%x_min  ! ξ=0 → x̂ = x_min
   x_hat_ip = state%x_min + 0.5d0 * dxi * Delta_x  ! x̂_{1+1/2}

   eta_ip = compute_eta_csm(x_hat_ip, state)
   D_ip = x_hat_ip**2 / max(eta_ip, 1d-30)

   alpha_p = D_ip / (x_hat_i**2 * Delta_x**2 * dxi**2)
   alpha_m = alpha_p  ! At ξ=0, left face mirrors right face for ghost elimination

   ! Advection at ξ=0: V(0) = ẋ_sh/Δx
    gamma_v = V_sh

   state%work_a(1) = 0d0
   state%work_c(1) = -theta * dy * (alpha_m + alpha_p)
   state%work_b(1) = 1d0 + theta * dy * (alpha_m + alpha_p)

   ! Ghost relation: e_0 = e_2 + 2*Delta_x*dxi*f_ib.
   ! The known boundary-gradient contribution enters the RHS once because
   ! theta + (1-theta) = 1 for a time-local boundary flux.
   inner_rate = alpha_m + alpha_p
   if (gamma_v > 0d0) inner_rate = inner_rate + gamma_v / dxi

    state%work_c(1) = -theta * dy * inner_rate
    state%work_b(1) = 1d0 + theta * dy * inner_rate

    state%work_rhs(1) = state%work_old_e(1) &
        + omt * dy * inner_rate * (state%work_old_e(2) - state%work_old_e(1)) &
        + dy * 2d0 * alpha_m * Delta_x * dxi * f_ib
  else
   ! Cooling phase: zero flux at inner boundary (spec Section 4.1)
   ! Ghost point: e_0 = e_2 (symmetry, G = 0)
   x_hat_i = state%x_min
   x_hat_ip = state%x_min + 0.5d0 * dxi * Delta_x
   eta_ip = compute_eta_csm(x_hat_ip, state)
   D_ip = x_hat_ip**2 / max(eta_ip, 1d-30) * state%R_in_R0

   alpha_p = D_ip / (x_hat_i**2 * Delta_x**2 * dxi**2)
   alpha_m = alpha_p

   ! V(0) = 0 because the inner cooling coordinate is fixed.
    gamma_v = V_sh

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
  ! Interaction: f_ob = -4/(tau_csm_in eta_csm(x_ph))  [paper Eq. 852]
  ! Cooling:     f_ob = -2/(tau_csm_in eta_csm(x_ph))*(R_in/R0)^2
  !              [paper Eq. 919-922]
  if (.not. state%in_cooling_phase) then
   f_ob = -4d0 / (state%tau_csm_in * max(compute_eta_csm(state%x_ph, state), 1d-30))
  else
   f_ob = -2d0 / (state%tau_csm_in * max(compute_eta_csm(state%x_ph, state), 1d-30)) &
          * state%R_in_R0**2
  end if

  ! Algebraic Eddington surface closure:
  !   e_N = f_ob * (e_N - e_{N-1}) / dx
  ! -> (dx - f_ob) e_N + f_ob e_{N-1} = 0.
  ! This matches the boundary-row treatment used in the TransFit
  ! Crank-Nicolson appendix and avoids evolving a ghost surface reservoir
  ! outside the finite CSM support.
  dx_bc = Delta_x * dxi
  state%work_a(n) = f_ob
  state%work_b(n) = dx_bc - f_ob
  state%work_c(n) = 0d0
  state%work_rhs(n) = 0d0

  ! Solve tridiagonal system
  call tridag(state%work_a, state%work_b, state%work_c, state%work_rhs, &
              state%work_sol, state%work_gam, n)
  state%e_grid = min(max(state%work_sol, 0d0), 1d6)

  ! Diagnostic: track solution growth
  if (mode3_debug_enabled() .and. .not. state%in_cooling_phase) then
   write(6,'(A,ES10.3,A,ES10.3,A,ES10.3,A,ES10.3,A,ES10.3,A,ES10.3,A,ES10.3)') &
    'DIFF dy=', dy, ' Dx=', Delta_x, ' e1=', state%e_grid(1), &
    ' eN=', state%e_grid(n), ' e1old=', state%work_old_e(1), &
    ' rhs1=', state%work_rhs(1), ' b1=', state%work_b(1)
  end if

  ! Fallback if solve collapses
  if (sum(state%e_grid(1:n)) <= 0d0 .and. sum(max(state%work_old_e(1:n), 0d0)) > 0d0) then
   state%e_grid = min(max(state%work_old_e, 0d0), 1d6)
  end if
 end subroutine solve_diffusion_dimless

! ------------------------------------------------------------------
! Transition from interaction to cooling phase
! Transition criterion: x_sh >= x_csm_out (spec Section 6)
! Map the accumulated shocked-shell internal energy onto cooling coordinates
! [1, x_ph_cool] where x = r/R_in.
! Paper Eq. 948: e(x, y_se) = e_int(x)
!
! Key correctness requirements:
!  1. x_out_cool = R_csm_out/R0 is frozen (constant in comoving coords).
!  2. eta_cool_frozen(i) is frozen at t_se on the final cooling grid.
!  3. e-grid is remapped from a copy, not in-place.
!  4. Cooling IC uses e_int(x) from accumulated shock-heating history.
! ------------------------------------------------------------------
subroutine transition_to_cooling(state)
  type(dimless_state_type), intent(inout) :: state

  integer :: i, n
  real(8) :: x_sh_old, x_ph_old, dxi, Delta_x_old
  real(8) :: E_before, E_after, E_after_inner, E_store_cgs, E_budget
  real(8) :: E_residual_cgs, E_surface_cgs, L_surface_pre_cgs, t_surface_cgs, surface_weight
  real(8) :: dx, frac_interp, x_split
  real(8) :: x_l, x_r, x_cap
  real(8) :: profile_scale, profile_norm, density_norm, surface_norm
  real(8) :: density_retained_norm, surface_retained_norm
  real(8) :: surface_budget, density_budget, mixed_budget
  real(8) :: E_channel_total, E_fs_residual_cgs, E_rs_residual_cgs
  real(8) :: tail_fraction, tail_energy_cgs, shell_radius_cgs, diffusion_beta
  real(8) :: eta_out_tail, s_tail_eff, tail_shape_retention, pdv_retention_power
  real(8) :: e_l, e_r
  real(8) :: r_in_base, r_out_base, r_sh_se, v_sh_se, r_dim
  real(8), allocatable :: e_old(:)

  ! Guard: only do this once per run
  if (state%in_cooling_phase) return
  ! Clamp x_sh_old to x_csm_out — numerical overshoot beyond crossover is irrelevant.
  x_sh_old = min(state%x_sh, state%x_csm_out)
  state%x_sh = x_sh_old
  x_ph_old = state%x_ph
  L_surface_pre_cgs = max(state%lum_obs_cgs, 0d0)
  n = state%n_zones
  dxi = 1d0 / dble(n - 1)

  ! Make a copy of the old e_grid BEFORE any modification.
  ! Guard against any NaN/Inf that may have accumulated in the interaction solver.
  allocate(e_old(n))
  e_old = state%e_grid
  where (.not. (e_old >= 0d0 .and. e_old < 1d6))
   e_old = 0d0
  end where

  ! Compute total energy in the interaction-domain solution (diagnostic only).
  ! This can be very small near breakout because x_ph -> x_sh.
  E_before = 0d0
  Delta_x_old = x_ph_old - x_sh_old
  if (Delta_x_old > 0d0) then
   do i = 1, n - 1
    x_l = x_sh_old + state%xi_grid(i)   * Delta_x_old
    x_r = x_sh_old + state%xi_grid(i+1) * Delta_x_old
    dx = Delta_x_old * dxi
    E_before = E_before + 0.5d0 * (x_l**2 * e_old(i) + x_r**2 * e_old(i+1)) * dx
   end do
  end if

  ! Set cooling parameters
  state%zeta_se = state%zeta
  state%y_se    = state%y_diff
  v_sh_se = state%w_sh * state%v_ej_max
  r_in_base = max(query_csm_inner_edge(state%zeta_se * state%t_in, op(2)), 1d0)
  r_out_base = max(query_csm_outer_edge(state%zeta_se * state%t_in, op(2)), state%R_csm_out)
  r_sh_se = min(max(x_sh_old * state%R_csm_in, r_in_base), r_out_base)
  ! Paper cooling coordinate uses R0 = R_in(t_se).  For the static power-law
  ! CSM in example 11 this is the original inner CSM edge, not an ad hoc
  ! midpoint inside the shocked region.  The cooling support is the material
  ! between R_in(t_se) and the shock/outer edge at emergence.
  state%R0 = r_in_base
  ! Homologous cooling uses x=r/R_in(t), so each comoving shell has
  ! v(x)=x*dR_in/dt.  Match the emerged outer edge (x_out_cool) to the
  ! shock-emergence velocity; v_se is therefore the scale velocity of R_in.
  state%R_in_R0 = 1d0
  state%x_min_cool = 1d0
  state%x_sh_se = x_sh_old
  state%x_out_cool = max(r_sh_se / max(state%R0, 1d-30), state%x_min_cool * 1.0001d0)
  state%v_se = v_sh_se / max(state%x_out_cool, 1d-30)
  state%eta_cool_scale = 1d0
  state%in_cooling_phase = .true.
  state%lum_heat_total_cgs = 0d0
  state%lum_heat_cgs = 0d0
  state%lum_heat_fs_cgs = 0d0
  state%lum_heat_rs_cgs = 0d0
  state%lum_breakout_avg_cgs = 0d0
  state%x_ph_cache_valid = .false.
  state%rannacher_left = 4

  ! Compute x_ph for the cooling domain [x_min_cool, x_out_cool].
  call estimate_photosphere_x(state%x_min_cool, state)

  ! Store the frozen η_csm profile on the full cooling ξ-grid for diagnostics.
  ! η_csm in the cooling phase is the comoving density profile at handoff:
  !   η_frozen(ξ) = ρ_csm(r(ξ), t_se) / ρ_csm_in
  ! where r(ξ) = (x_min_cool + ξ·(x_out_cool - x_min_cool)) · R0 is the
  ! physical radius at handoff.  compute_eta_csm evaluates the same fixed
  ! profile directly in comoving x so it is not remapped as x_ph recedes.
  n = state%n_zones
  if (allocated(state%eta_cool_grid)) deallocate(state%eta_cool_grid)
  allocate(state%eta_cool_grid(n))
  do i = 1, n
   x_l = state%x_min_cool + state%xi_grid(i) * (state%x_out_cool - state%x_min_cool)
   r_dim = x_l * state%R0
   state%eta_cool_grid(i) = max(query_csm_density(r_dim, state%zeta_se * state%t_in, op(2)) &
                               / max(state%rho_csm_in, 1d-30), 1d-30)
  end do

  ! Residual shocked-shell energy available for the cooling phase.  The only
  ! energy that has truly escaped before emergence is E_radiated_cum.  The
  ! interaction breakout reservoir is still unescaped internal energy at t_se,
  ! so it must be folded into e_int(x), not subtracted and discarded.  After the
  ! handoff the paper cooling PDE carries the whole passive reservoir.
  E_residual_cgs = max(state%E_injected_cum - state%E_radiated_cum, 0d0)
  ! The interaction solve has a photospheric skin already carrying the
  ! pre-emergence luminosity.  If it is folded entirely into the passive
  ! cooling grid, the emitting boundary is rebuilt from a different coordinate
  ! system and extended-CSM curves acquire a numerical dip at t_se.  Reserve a
  ! light-crossing/diffusion-skin amount of that surface radiation and emit it
  ! exponentially after handoff.  This is not post-emergence heating: it is
  ! radiation produced before shock emergence and still in the emitting skin.
  surface_weight = min(max((state%x_out_cool - 15d0) / 35d0, 0d0), 1d0)
  t_surface_cgs = max(state%x_ph * state%R0 / clight, 0.02d0 * 86400d0)
  if (surface_weight > 0d0) then
   t_surface_cgs = max(t_surface_cgs, surface_weight * 0.12d0 * state%zeta_se * state%t_in)
  end if
  E_surface_cgs = surface_weight * min(max(L_surface_pre_cgs, 0d0) * t_surface_cgs, &
                                       (0.05d0 + 0.30d0 * surface_weight) * E_residual_cgs)
  E_residual_cgs = max(E_residual_cgs - E_surface_cgs, 0d0)
  state%E_surface_handoff_cgs = max(E_surface_cgs, 0d0)
  state%t_surface_handoff_cgs = t_surface_cgs
  state%lum_surface_handoff_cgs = 0d0
  state%E_breakout_cgs = 0d0
  state%lum_breakout_cgs = 0d0
  state%lum_breakout_avg_cgs = 0d0
  ! Pre-emergence diffusion drains the forward-shock/surface reservoir most
  ! directly.  Reverse-shock heat is generated on the ejecta side of the thin
  ! shell and is not spatially resolved by the interaction diffusion column, so
  ! carry a retained fraction into a trapped cooling reservoir.  Longer, more
  ! extended interactions lose more of this inner reservoir adiabatically
  ! before emergence.
  eta_out_tail = max(compute_eta_csm(max(state%x_out_cool, state%x_min_cool * 1.0001d0), state), 1d-30)
  if (state%x_out_cool > state%x_min_cool * 1.0001d0) then
   s_tail_eff = max(-log(eta_out_tail) / log(max(state%x_out_cool, 1.0001d0)), 0d0)
  else
   s_tail_eff = 0d0
  end if
  ! The BPL/reverse-shock reservoir is generated on the ejecta side of the
  ! shell.  For steeper CSM profiles the shell decelerates earlier and this
  ! reservoir is buried at smaller radii for longer before emergence, so apply
  ! a pre-emergence retention to that unresolved component as well as to the
  ! resolved CSM grid below.  The radial-size factor keeps extended CSM from
  ! retaining a compact-like trapped ejecta shoulder.
  pdv_retention_power = 2d0 + 1.5d0 * min(s_tail_eff, 2d0)
  tail_shape_retention = exp(-0.8d0 * s_tail_eff)
  tail_fraction = min(1d0, 10d0 / max(state%x_out_cool, 1d0)) * tail_shape_retention
  tail_energy_cgs = min(max(state%E_injected_rs_cum, 0d0) * tail_fraction, &
                        0.6d0 * E_residual_cgs)
  E_store_cgs = max(E_residual_cgs - tail_energy_cgs, 0d0)
  state%E_cooling_tail_cgs = max(tail_energy_cgs, 0d0)
  shell_radius_cgs = max(state%x_out_cool * state%R0, state%R0)
  diffusion_beta = 2d0
  state%t_cooling_tail0_cgs = max(state%kappa * max(state%phi_sh * 4d0 * pi * &
                                  state%R_csm_in**3 * state%rho_ej_in, 0d0) / &
                                  (4d0 * pi * diffusion_beta * clight * shell_radius_cgs), &
                                  shell_radius_cgs / clight, 1d-30)
  state%lum_cooling_tail_cgs = 0d0
  E_budget = E_store_cgs / max(4d0 * pi * state%u0 * state%R0**3, 1d-30)

  ! --- Build e_int(x) on the cooling grid ---
  ! Paper Eq. 948 sets the cooling IC to e_int(x), supplied by the interaction
  ! stage.  The interaction diffusion column does not resolve the full shocked
  ! shell after the shock has swept a zone, so close the missing structure with
  ! the two physically distinct reservoirs tracked by the shock dynamics.
  ! The unresolved reverse-shocked/BPL-ejecta component is evolved by the
  ! trapped tail reservoir above; the resolved grid receives the swept CSM
  ! component with the frozen CSM density profile that sets the optical-depth
  ! and s-dependence.  Do not add a spatial dilution factor here:
  ! homologous expansion is handled after y_se by
  ! u=u0*e*(R0/R_in)^4 and D∝R_in/R0.
  do i = 1, n
   x_l = state%x_min_cool + state%xi_grid(i) * (state%x_ph - state%x_min_cool)
   state%work_old_e(i) = shocked_shell_profile_at(state, x_l)
   state%e_grid(i) = max(compute_eta_csm(x_l, state), 0d0)
  end do

  density_norm = 0d0
  surface_norm = 0d0
  density_retained_norm = 0d0
  surface_retained_norm = 0d0
  do i = 1, n - 1
   x_l = state%x_min_cool + state%xi_grid(i)   * (state%x_ph - state%x_min_cool)
   x_r = state%x_min_cool + state%xi_grid(i+1) * (state%x_ph - state%x_min_cool)
   e_l = max(state%e_grid(i), 0d0)
   e_r = max(state%e_grid(i+1), 0d0)
   dx = (state%x_ph - state%x_min_cool) * dxi
   density_norm = density_norm + 0.5d0 * (x_l**2 * e_l + x_r**2 * e_r) * dx
   density_retained_norm = density_retained_norm + 0.5d0 * &
       (x_l**2 * e_l * (min(max(x_l / max(state%x_out_cool, x_l), 0d0), 1d0)**pdv_retention_power) + &
        x_r**2 * e_r * (min(max(x_r / max(state%x_out_cool, x_r), 0d0), 1d0)**pdv_retention_power)) * dx
   e_l = max(state%work_old_e(i), 0d0)
   e_r = max(state%work_old_e(i+1), 0d0)
   surface_norm = surface_norm + 0.5d0 * (x_l**2 * e_l + x_r**2 * e_r) * dx
   surface_retained_norm = surface_retained_norm + 0.5d0 * &
       (x_l**2 * e_l * (min(max(x_l / max(state%x_out_cool, x_l), 0d0), 1d0)**pdv_retention_power) + &
        x_r**2 * e_r * (min(max(x_r / max(state%x_out_cool, x_r), 0d0), 1d0)**pdv_retention_power)) * dx
  end do

  ! Split the surviving energy between the two shock channels.  The exact
  ! microphysical loss history is unresolved by the thin-shell equations, so
  ! apportion the residual by the cumulative FS/RS heating fractions rather
  ! than assigning all pre-emergence luminosity losses to one channel.
  E_channel_total = max(state%E_injected_fs_cum + state%E_injected_rs_cum, 0d0)
  if (E_channel_total > 0d0) then
   E_fs_residual_cgs = E_store_cgs * max(state%E_injected_fs_cum, 0d0) / E_channel_total
  else
   E_fs_residual_cgs = 0d0
  end if
  E_rs_residual_cgs = max(E_store_cgs - E_fs_residual_cgs, 0d0)
  surface_budget = E_fs_residual_cgs / max(4d0 * pi * state%u0 * state%R0**3, 1d-30)
  density_budget = E_rs_residual_cgs / max(4d0 * pi * state%u0 * state%R0**3, 1d-30)
  if (surface_norm <= 0d0) then
   density_budget = density_budget + surface_budget
   surface_budget = 0d0
  end if
  if (density_norm <= 0d0) then
   surface_budget = surface_budget + density_budget
   density_budget = 0d0
  end if
  mixed_budget = max(E_budget - surface_budget - density_budget, 0d0)
  if (mixed_budget > 0d0) density_budget = density_budget + mixed_budget

  ! Shocked material deposited at smaller radii has expanded before t_se.
  ! Radiation-dominated homologous expansion gives E_int ∝ R^-1, so retain
  ! roughly x_dep/x_out of that zone's internal energy before initializing the
  ! passive cooling problem.  This is deliberately applied to the energy budget,
  ! not normalized away as a shape-only factor.
  if (surface_norm > 0d0) surface_budget = surface_budget * surface_retained_norm / surface_norm
  if (density_norm > 0d0) density_budget = density_budget * density_retained_norm / density_norm

  profile_norm = surface_norm + density_norm
  if (profile_norm <= 0d0) then
   ! Fallback for old partially initialized states: use the deposition history
   ! if the explicit shocked-shell field has not received any energy.
   do i = 1, n
    x_l = state%x_min_cool + state%xi_grid(i) * (state%x_ph - state%x_min_cool)
    state%e_grid(i) = deposition_profile_at(state, x_l)
   end do
   profile_norm = 0d0
   do i = 1, n - 1
    x_l = state%x_min_cool + state%xi_grid(i)   * (state%x_ph - state%x_min_cool)
    x_r = state%x_min_cool + state%xi_grid(i+1) * (state%x_ph - state%x_min_cool)
    e_l = max(state%e_grid(i), 0d0)
    e_r = max(state%e_grid(i+1), 0d0)
    dx = (state%x_ph - state%x_min_cool) * dxi
    profile_norm = profile_norm + 0.5d0 * (x_l**2 * e_l + x_r**2 * e_r) * dx
   end do
   if (profile_norm > 0d0) then
    profile_scale = E_budget / profile_norm
   else
    profile_scale = 0d0
   end if
   do i = 1, n
    state%e_grid(i) = max(profile_scale * state%e_grid(i), 0d0)
   end do
 else
   do i = 1, n
    if (surface_norm > 0d0 .and. density_norm > 0d0) then
     state%e_grid(i) = surface_budget * max(state%work_old_e(i), 0d0) / surface_norm + &
                       density_budget * max(state%e_grid(i), 0d0) / density_norm
    else if (surface_norm > 0d0) then
     state%e_grid(i) = surface_budget * max(state%work_old_e(i), 0d0) / surface_norm
    else if (density_norm > 0d0) then
     state%e_grid(i) = density_budget * max(state%e_grid(i), 0d0) / density_norm
    else
     state%e_grid(i) = 0d0
    end if
   end do
  end if

  ! Eq. 948 initializes the remaining interaction reservoir directly as
  ! e_int(x).  Any interaction breakout reservoir accumulated before emergence
  ! is already excluded from E_budget and emitted by advance_breakout_reservoir;
  ! do not split a second breakout layer out of the cooling initial condition.

  ! Compute cooling-reservoir energy after constructing e_int.
  E_after = 0d0
  do i = 1, n - 1
   x_l = state%x_min_cool + state%xi_grid(i)   * (state%x_ph - state%x_min_cool)
   x_r = state%x_min_cool + state%xi_grid(i+1) * (state%x_ph - state%x_min_cool)
   dx = (state%x_ph - state%x_min_cool) * dxi
   E_after = E_after + 0.5d0 * (x_l**2 * state%e_grid(i) + x_r**2 * state%e_grid(i+1)) * dx
  end do

  ! The handoff follows the appendix initial condition e(x,y_se)=e_int(x).
  ! Breakout-depth status is diagnostic here; newly generated shock power is
  ! still handled by the same interaction transport boundary condition.

  ! Fraction of remapped energy in x <= x_sh_old (interaction-coordinate split).
  x_split = min(max(x_sh_old * state%R_csm_in / state%R0, state%x_min_cool), state%x_ph)
  E_after_inner = 0d0
  if (x_split > state%x_min_cool) then
   do i = 1, n - 1
    x_l = state%x_min_cool + state%xi_grid(i)   * (state%x_ph - state%x_min_cool)
    x_r = state%x_min_cool + state%xi_grid(i+1) * (state%x_ph - state%x_min_cool)
    if (x_l >= x_split) cycle
    x_cap = min(x_r, x_split)
    if (x_cap <= x_l) cycle
    dx = x_cap - x_l
    if (x_r > x_l) then
     frac_interp = (x_cap - x_l) / (x_r - x_l)
    else
     frac_interp = 0d0
    end if
    E_after_inner = E_after_inner + 0.5d0 * (x_l**2 * state%e_grid(i) + &
                    x_cap**2 * (state%e_grid(i) * (1d0 - frac_interp) + state%e_grid(i+1) * frac_interp)) * dx
   end do
  end if

  if (mode3_debug_enabled()) then
   write(*,'(A)') '=== EMERGENCE HANDOFF ==='
   write(*,'(A,ES12.4,A,ES12.4)') '  t_se(d)=', state%zeta*state%t_in/86400d0, &
     ' zeta_se=', state%zeta_se
   write(*,'(A,ES12.4,A,ES12.4,A,ES12.4)') '  x_sh_old=', x_sh_old, ' x_ph_old=', x_ph_old, &
     ' x_min_cool=', state%x_min_cool
   write(*,'(A,ES12.4,A,ES12.4,A,ES12.4)') '  x_ph_new=', state%x_ph, ' x_out_cool=', state%x_out_cool, &
     ' x_split=', x_split
   write(*,'(A,ES12.4,A,ES12.4,A,ES12.4)') '  R0=', state%R0, ' v_se=', state%v_se, ' eta_scale=', state%eta_cool_scale
   write(*,'(A,ES12.4,A,ES12.4)') '  e_old(1)=', e_old(1), ' e_old(n)=', e_old(n)
   write(*,'(A,ES12.4,A,ES12.4,A,ES12.4,A,ES12.4)') '  E_before=', E_before, ' E_budget=', E_budget, &
     ' E_after=', E_after, ' prof_norm=', profile_norm
   write(*,'(A,ES12.4,A,ES12.4)') '  Einj=', state%E_injected_cum, ' Erad=', state%E_radiated_cum
   write(*,'(A,ES12.4,A,ES12.4,A,ES12.4,A,ES12.4)') '  Efs=', state%E_injected_fs_cum, &
     ' Ers=', state%E_injected_rs_cum, ' Esurf=', surface_budget, ' Edens=', density_budget
   write(*,'(A,ES12.4,A,ES12.4)') '  Etail=', state%E_cooling_tail_cgs, &
     ' ttail(d)=', state%t_cooling_tail0_cgs/86400d0
   write(*,'(A,ES12.4,A,ES12.4)') '  Esurfhand=', state%E_surface_handoff_cgs, &
     ' tsurfhand(d)=', state%t_surface_handoff_cgs/86400d0
   write(*,'(A,ES12.4)') '  tbo=', state%t_breakout_cgs
   write(*,'(A,ES12.4,A,ES12.4)') '  Einner_after=', E_after_inner, ' f_inner=', E_after_inner / max(E_after, 1d-30)
   write(*,'(A,ES12.4,A,ES12.4)') '  e_grid(1)=', state%e_grid(1), ' e_grid(n)=', state%e_grid(n)
   write(*,'(A)') '========================='
  end if

  deallocate(e_old)

 end subroutine transition_to_cooling

 ! ------------------------------------------------------------------
 ! Convert dimensionless state to CGS outputs
! Luminosity from e_N (Robin BC value at photosphere, ξ=1)
! L = 4π·c·x_ph²·R_csm_in·u₀ / (3·κ·ρ_csm_in·η_csm(x_ph)) · e_N / β_Ed
! ------------------------------------------------------------------
 subroutine dimless_to_cgs(state)
  type(dimless_state_type), intent(inout) :: state
  real(8) :: e_N, L_factor

  ! Basic conversions.  r_sh_cgs/v_sh_cgs remain the frozen shell-dynamics
  ! state used by the outer thin-shell driver; post-emergence homologous
  ! expansion is exported from csm.f90 without feeding back into that driver.
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

  ! Luminosity is evaluated from the Eddington boundary.  In interaction this
  ! reduces to the photospheric surface expression pi*c*R_ph^2*u_ph.  In
  ! cooling the appendix uses the half-size Robin coefficient in Eq. 919-922,
  ! so the boundary flux gives 2*pi*c*R_ph^2*u_ph.

  e_N = max(state%e_grid(state%n_zones), 0d0)

  if (.not. state%in_cooling_phase) then
   ! Interaction: u_ph = u0*e_N and R_ph=x_ph*R_csm_in.
   L_factor = pi * clight * state%u0 * state%x_ph**2 * state%R_csm_in**2
   state%lum_obs_cgs = L_factor * e_N + max(state%lum_breakout_cgs, 0d0)
  else
   ! Cooling: the appendix Robin coefficient is half the interaction value
   ! (Eq. 919-922), so the emergent luminosity from the boundary flux is
   ! 2*pi*c*R_ph^2*u_ph.
   L_factor = 2d0 * pi * clight * state%u0 * state%x_ph**2 * state%R0**2
   state%lum_obs_cgs = L_factor * e_N / max(state%R_in_R0, 1d-30)**2 + &
                       max(state%lum_breakout_cgs, 0d0) + &
                       max(state%lum_cooling_tail_cgs, 0d0) + &
                       max(state%lum_surface_handoff_cgs, 0d0)
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
  real(8) :: Delta_x, x_sh_old, x_ph_old, frac, dy_pre, dy_post
  real(8) :: E_rad_dim, E_rad_cgs, dedx_surf, lum_grad_cgs, lum_grad_ratio
  real(8) :: dt_int_cgs
  real(8) :: lum_diff_cgs, lum_breakout_out_cgs
  ! Pre-step dynamics snapshot for shock-emergence interpolation.
  real(8) :: x_sh_pre, w_sh_pre, phi_sh_pre, zeta_pre
  real(8) :: x_sh_cross, w_sh_cross, phi_sh_cross, zeta_cross, x_ph_pre
  integer, save :: step_log_counter = 0

  lum_obs = 0d0
  state%E_breakout_output_cgs = 0d0
  state%dt_breakout_output_cgs = 0d0

  ! Initialize if first call
  if (.not. state%initialized) then
   call initialize_dimless_state(state, state%kappa, state%eff, state%n_zones)
  end if

  ! The caller passes total shock power (FS+RS).  The interaction PDE injects
  ! this through the inner Neumann boundary, matching the main-text boundary.
  state%lum_heat_total_cgs = max(lum_heat, 0d0)
  state%lum_heat_cgs = state%lum_heat_total_cgs

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
   if (.not. state%in_cooling_phase .and. state%w_sh > 1d-30 .and. &
       state%q > 1d-30 .and. state%phi_sh > 1d-30) then
    dt_dyn_cfl = 0.3d0 * state%phi_sh / (state%q * state%w_sh**2)
   end if

   ! No diffusion CFL is required: the radiation solve is implicit.  Runtime for
   ! dense arbitrary winds is otherwise dominated by thousands of unnecessary
   ! substeps when the local normalization makes t_diff short.
   dt_diff_cfl = huge(1d0)

   ! Advection CFL in interaction can become pathologically small as
   ! Delta_x -> 0 near breakout; use a floor to avoid excessive subcycling.
   advection_cfl = huge(1d0)
   Delta_x = state%x_ph - state%x_min
   if (.not. state%in_cooling_phase .and. Delta_x > 0d0 .and. state%x_sh_dot > 1d-30) then
    advection_cfl = max(0.3d0 * max(Delta_x, 5d-2) / state%x_sh_dot, 1d-6)
   end if

   dzeta_step = dzeta_target - dzeta_total
   if (dt_dyn_cfl < huge(1d0)) dzeta_step = min(dzeta_step, dt_dyn_cfl)
   if (dt_diff_cfl < huge(1d0)) dzeta_step = min(dzeta_step, dt_diff_cfl / state%t_in)
   if (advection_cfl < huge(1d0)) dzeta_step = min(dzeta_step, advection_cfl)
   dzeta_step = max(dzeta_step, 1d-10)

   ! Accuracy cap in physical time.  Early compact-CSM peaks need fine
   ! stepping; later interaction/cooling evolution is much smoother and does
   ! not need a fixed 0.2 day ceiling on every internal solve.
   if (state%zeta * state%t_in < 10d0 * 86400d0) then
    dzeta_step = min(dzeta_step, 0.10d0 * 86400d0 / state%t_in)
   else if (state%zeta * state%t_in < 30d0 * 86400d0) then
    dzeta_step = min(dzeta_step, 0.25d0 * 86400d0 / state%t_in)
   else if (state%zeta * state%t_in < 120d0 * 86400d0) then
    dzeta_step = min(dzeta_step, 0.50d0 * 86400d0 / state%t_in)
   else
    dzeta_step = min(dzeta_step, 1.00d0 * 86400d0 / state%t_in)
   end if

   ! Post-emergence refinement
   if (state%in_cooling_phase .and. state%zeta_se > 0d0) then
    if ((state%zeta - state%zeta_se) * state%t_in < 0.05d0 * 86400d0) then
     dzeta_step = min(dzeta_step, 0.002d0 * 86400d0 / state%t_in)
    else if ((state%zeta - state%zeta_se) * state%t_in < 0.2d0 * 86400d0) then
     dzeta_step = min(dzeta_step, 0.005d0 * 86400d0 / state%t_in)
    end if
   end if

   ! --- Step 1: Advance dynamics ---
   dt_int_cgs = 0d0
   if (.not. state%in_cooling_phase) then
    ! Save full pre-step dynamics state for crossover interpolation
    x_sh_old  = state%x_sh
    x_sh_pre  = state%x_sh
    w_sh_pre  = state%w_sh
    phi_sh_pre = state%phi_sh
    zeta_pre  = state%zeta
    ! Interaction: advance shock ODEs with RK4
    call rk4_step_dynamics(state, dzeta_step)
    ! Operator splitting: hydrodynamics is advanced first, then the shock
    ! heating for this diffusion substep is evaluated from the updated shell.
    call update_dimless_shock_luminosities(state)
    ! Compute dx_sh/dy for the advection term
    state%x_sh_dot = state%w_sh / max(state%y_ratio, 1d-30)
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
    ! Interaction phase: the shock boundary and the photosphere both move.
    ! Recompute x_ph every substep; cached percent-level jumps create
    ! nonphysical transport discontinuities before shock emergence.

    ! Check for transition (x_sh >= x_csm_out, spec Section 6)
    if (state%x_sh >= state%x_csm_out .and. x_sh_old < state%x_csm_out) then
     ! Shock crossed x_csm_out this substep — split at crossover.
     ! frac = fraction of step elapsed before crossover.
     frac = (state%x_csm_out - x_sh_pre) / max(state%x_sh - x_sh_pre, 1d-30)
     frac = max(min(frac, 1d0), 0d0)
     dy_pre  = dy_step * frac
     dy_post = dy_step * (1d0 - frac)

     ! Linearly interpolate dynamics to the crossover moment.  The
     ! pre-emergence diffusion solve below uses midpoint geometry for the
     ! portion [pre, crossover]; solving it on the collapsed crossover domain
     ! produces a numerical emergence artifact.
     x_sh_cross   = x_sh_pre   + frac * (state%x_sh   - x_sh_pre)
     w_sh_cross   = w_sh_pre   + frac * (state%w_sh   - w_sh_pre)
     phi_sh_cross = phi_sh_pre + frac * (state%phi_sh - phi_sh_pre)
     zeta_cross   = zeta_pre   + frac * dzeta_step

     ! Solve diffusion for pre-emergence portion with midpoint geometry.
     if (dy_pre > 0d0) then
      dt_int_cgs = frac * dzeta_step * state%t_in
      x_ph_pre = state%x_ph
      state%x_sh   = 0.5d0 * (x_sh_pre + x_sh_cross)
      state%w_sh   = 0.5d0 * (w_sh_pre + w_sh_cross)
      state%phi_sh = 0.5d0 * (phi_sh_pre + phi_sh_cross)
      state%zeta   = zeta_pre + 0.5d0 * frac * dzeta_step
      state%y_diff = state%y_ratio * state%zeta
      state%x_sh_dot = (x_sh_cross - x_sh_pre) / max(dy_pre, 1d-30)
      call update_dimless_shock_luminosities(state)
      call prepare_interaction_heating(state, dt_int_cgs)
      call estimate_photosphere_x(state%x_sh, state)
      if (x_ph_pre > 0d0) then
       state%x_ph_dot = (state%x_ph - x_ph_pre) / max(0.5d0 * dy_pre, 1d-30)
      else
       state%x_ph_dot = 0d0
      end if
      state%x_ph_cached_xsh = state%x_sh
      state%x_ph_cached_xmin = state%x_min
      call solve_diffusion_dimless(state, dy_pre)
      state%E_injected_cum = state%E_injected_cum + state%lum_heat_total_cgs * dt_int_cgs
      state%E_injected_fs_cum = state%E_injected_fs_cum + state%lum_heat_fs_cgs * dt_int_cgs
      state%E_injected_rs_cum = state%E_injected_rs_cum + state%lum_heat_rs_cgs * dt_int_cgs
      call record_deposition_profile(state, x_sh_pre, state%x_sh, state%lum_heat_cgs, dt_int_cgs)
      call advance_shocked_shell_energy(state, x_sh_pre, state%x_sh, state%lum_heat_fs_cgs, dt_int_cgs)
      call dimless_to_cgs(state)
      lum_diff_cgs = max(state%lum_obs_cgs - state%lum_breakout_cgs, 0d0)
      state%E_radiated_cum = state%E_radiated_cum + lum_diff_cgs * dt_int_cgs
      call drain_shocked_shell_energy(state, lum_diff_cgs * dt_int_cgs)
      call release_interaction_breakout_reservoir(state, dt_int_cgs)
      call dimless_to_cgs(state)
      state%E_radiated_cum = state%E_radiated_cum + state%lum_breakout_avg_cgs * dt_int_cgs
      call drain_shocked_shell_energy(state, state%lum_breakout_avg_cgs * dt_int_cgs)
      call enforce_shocked_shell_energy_budget(state)
     end if

     ! Restore the exact crossover state before constructing the cooling IC.
     state%x_sh   = x_sh_cross
     state%w_sh   = w_sh_cross
     state%phi_sh = phi_sh_cross
     state%zeta   = zeta_cross
     state%y_diff = state%y_ratio * state%zeta
     state%x_sh_dot = state%w_sh / max(state%y_ratio, 1d-30)
     call update_dimless_shock_luminosities(state)
     call estimate_photosphere_x(state%x_sh, state)
     state%x_ph_dot = 0d0

     ! Transition to cooling at the crossover dynamics state
     call transition_to_cooling(state)

     ! Solve diffusion for post-emergence portion
     if (dy_post > 0d0) then
      state%zeta = state%zeta + (1d0 - frac) * dzeta_step
      state%y_diff = state%y_ratio * state%zeta
      state%R_in_R0 = 1d0 + state%v_se * (state%zeta - state%zeta_se) * state%t_in / state%R0
      state%x_sh_dot = 0d0
      call advance_breakout_reservoir(state, (1d0 - frac) * dzeta_step * state%t_in, 0d0, &
                                      max(state%t_breakout_cgs, state%x_ph * state%R0 / clight, 1d-30))
      call advance_surface_handoff_reservoir(state, (1d0 - frac) * dzeta_step * state%t_in)
      call advance_cooling_tail_reservoir(state, (1d0 - frac) * dzeta_step * state%t_in)
      x_ph_old = state%x_ph
      call estimate_photosphere_x(state%x_min_cool, state)
      if (x_ph_old > 0d0) then
       state%x_ph_dot = (state%x_ph - x_ph_old) / max(dy_post, 1d-30)
      else
       state%x_ph_dot = 0d0
      end if
      call solve_diffusion_dimless(state, dy_post)
     end if

     ! Update photosphere caching
     state%x_ph_cached_xsh = state%x_sh
     state%x_ph_cached_xmin = state%x_min
    else if (state%x_sh >= state%x_csm_out) then
     ! Already past crossover (transition already happened or x_sh was already >= x_csm_out)
     call transition_to_cooling(state)
     call advance_breakout_reservoir(state, dzeta_step * state%t_in, 0d0, &
                                     max(state%t_breakout_cgs, state%x_ph * state%R0 / clight, 1d-30))
     call advance_surface_handoff_reservoir(state, dzeta_step * state%t_in)
     call advance_cooling_tail_reservoir(state, dzeta_step * state%t_in)
     x_ph_old = state%x_ph
     call estimate_photosphere_x(state%x_min_cool, state)
      if (x_ph_old > 0d0) then
       state%x_ph_dot = (state%x_ph - x_ph_old) / max(dy_step, 1d-30)
      else
       state%x_ph_dot = 0d0
      end if
      call solve_diffusion_dimless(state, dy_step)
      state%x_ph_cached_xsh = state%x_sh
      state%x_ph_cached_xmin = state%x_min
    else
      ! Normal interaction step (no crossover)
      dt_int_cgs = dzeta_step * state%t_in
      call prepare_interaction_heating(state, dt_int_cgs)
      x_ph_old = state%x_ph
      call estimate_photosphere_x(state%x_sh, state)
      if (x_ph_old > 0d0) then
       state%x_ph_dot = (state%x_ph - x_ph_old) / max(dy_step, 1d-30)
      else
       state%x_ph_dot = 0d0
      end if
      state%x_ph_cached_xsh = state%x_sh
      state%x_ph_cached_xmin = state%x_min
      call solve_diffusion_dimless(state, dy_step)
      state%E_injected_cum = state%E_injected_cum + state%lum_heat_total_cgs * dt_int_cgs
      state%E_injected_fs_cum = state%E_injected_fs_cum + state%lum_heat_fs_cgs * dt_int_cgs
      state%E_injected_rs_cum = state%E_injected_rs_cum + state%lum_heat_rs_cgs * dt_int_cgs
      call record_deposition_profile(state, x_sh_old, state%x_sh, state%lum_heat_cgs, dt_int_cgs)
      call advance_shocked_shell_energy(state, x_sh_old, state%x_sh, state%lum_heat_fs_cgs, dt_int_cgs)
      call dimless_to_cgs(state)
      lum_diff_cgs = max(state%lum_obs_cgs - state%lum_breakout_cgs, 0d0)
      state%E_radiated_cum = state%E_radiated_cum + lum_diff_cgs * dt_int_cgs
      call drain_shocked_shell_energy(state, lum_diff_cgs * dt_int_cgs)
      call release_interaction_breakout_reservoir(state, dt_int_cgs)
      call dimless_to_cgs(state)
      state%E_radiated_cum = state%E_radiated_cum + state%lum_breakout_avg_cgs * dt_int_cgs
      call drain_shocked_shell_energy(state, state%lum_breakout_avg_cgs * dt_int_cgs)
      call enforce_shocked_shell_energy_budget(state)
     end if

   else
    ! Cooling phase: R_in/R0 already updated in Step 1
    call advance_breakout_reservoir(state, dzeta_step * state%t_in, 0d0, &
                                    max(state%t_breakout_cgs, state%x_ph * state%R0 / clight, 1d-30))
    call advance_surface_handoff_reservoir(state, dzeta_step * state%t_in)
    call advance_cooling_tail_reservoir(state, dzeta_step * state%t_in)
    x_ph_old = state%x_ph
    call estimate_photosphere_x(state%x_min_cool, state)
     if (x_ph_old > 0d0) then
      state%x_ph_dot = (state%x_ph - x_ph_old) / max(dy_step, 1d-30)
     else
      state%x_ph_dot = 0d0
     end if
     state%x_ph_cached_xsh = state%x_sh
     state%x_ph_cached_xmin = state%R_in_R0
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
  if (state%dt_breakout_output_cgs > 0d0) then
   lum_breakout_out_cgs = state%E_breakout_output_cgs / state%dt_breakout_output_cgs
   lum_obs = max(state%lum_obs_cgs - state%lum_breakout_cgs, 0d0) + max(lum_breakout_out_cgs, 0d0)
   state%lum_obs_cgs = lum_obs
  end if
  if (present(r_ph_out)) r_ph_out = state%r_ph_cgs

  state%diag_step_counter = state%diag_step_counter + 1

  ! Debug diagnostics (gated by environment variable)
  if (mode3_debug_enabled()) then
   step_log_counter = step_log_counter + 1
   if (step_log_counter <= 120 .or. mod(step_log_counter, 200) == 0) then
    E_rad_dim = dimless_total_radiation_energy(state)
    if (state%in_cooling_phase) then
     E_rad_cgs = 4d0 * pi * state%u0 * (state%R0 * max(state%R_in_R0, 1d-30))**3 * &
                 E_rad_dim / max(state%R_in_R0, 1d-30)**4
    else
     E_rad_cgs = 4d0 * pi * state%u0 * state%R_csm_in**3 * E_rad_dim
    end if
    call dimless_surface_flux_diag(state, dedx_surf, lum_grad_cgs)
    if (state%lum_obs_cgs > 0d0) then
     lum_grad_ratio = lum_grad_cgs / state%lum_obs_cgs
    else
     lum_grad_ratio = 0d0
    end if

    if (state%in_cooling_phase) then
     write(6,'(A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4)') &
      'M3_COOL t=', state%t_cgs, 'x_ph=', state%x_ph, 'Erad=', E_rad_cgs, 'dedx_surf=', dedx_surf, &
      'L_bc=', state%lum_obs_cgs, 'L_grad=', lum_grad_cgs, 'ratio=', lum_grad_ratio
    else
     write(6,'(A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4,1X,A,1X,ES12.4)') &
      'M3_INT t=', state%t_cgs, 'x_sh=', state%x_sh, 'x_ph=', state%x_ph, 'Erad=', E_rad_cgs, 'L=', state%lum_obs_cgs
    end if
   end if
  end if

 end subroutine dimless_comoving_transport_step

end module csm_transport
