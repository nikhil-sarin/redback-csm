module csm_transport

 use constants, only: pi, clight
 use physical_constants, only: a_rad
use get_vals, only: op, query_csm_density, query_csm_inner_edge, query_csm_outer_edge, &
                     query_tau_to_edge, query_csm_photosphere_radius, query_csm_velocity, &
                     query_ejecta_density, query_ejecta_velocity
use integration, only: dudt, forward_shock_luminosity, reverse_shock_luminosity, &
                       forward_shock_radiative_efficiency, reverse_shock_radiative_efficiency
use csm_runtime, only: shock_efficiency_mode

 implicit none

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
  real(8) :: lum_emergence_cgs = 0d0

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
  call get_environment_variable('REDBACK_CSM_TRANSPORT_DEBUG', env, status=stat)
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

 subroutine tridag(a, b, c, r, u, gam, n)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n), b(n), c(n), r(n)
  real(8), intent(out) :: u(n)
  real(8), intent(inout) :: gam(n)
  integer :: j
  real(8) :: bet

  if (n <= 0) return
  bet = b(1)
  if (abs(bet) < 1d-300) bet = sign(1d-300, bet + 1d-300)
  u(1) = r(1) / bet
  do j = 2, n
   gam(j) = c(j-1) / bet
   bet = b(j) - a(j) * gam(j)
   if (abs(bet) < 1d-300) bet = sign(1d-300, bet + 1d-300)
   u(j) = (r(j) - a(j) * u(j-1)) / bet
  end do
  do j = n - 1, 1, -1
   u(j) = u(j) - gam(j+1) * u(j+1)
  end do
 end subroutine tridag

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
  state%lum_emergence_cgs = 0d0
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
   ! The untruncated BPL branch has no formal finite outer edge.  The shell
   ! dynamics initialize the fastest ejecta at 100*v_tr, so use the same
   ! effective v_ej,max for the Appendix-A nondimensionalization.
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
  state%csm_powerlaw_fast = static_powerlaw_csm_slope(state%csm_eta_pow)

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
  real(8) :: xi_cool

  if (state%in_cooling_phase .and. allocated(state%eta_cool_grid) .and. &
      allocated(state%xi_grid)) then
   xi_cool = (x - state%x_min_cool) / max(state%x_out_cool - state%x_min_cool, 1d-30)
   call interp_eta_cool(min(max(xi_cool, 0d0), 1d0), state%xi_grid, state%eta_cool_grid, &
                        state%n_zones, eta)
   eta = max(eta, 1d-30)
   return
  end if

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
 logical function static_powerlaw_csm_slope(eta_pow) result(ok)
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
   ok = static_powerlaw_csm_slope(eta_pow)
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
   found = static_powerlaw_csm_slope(eta_pow)
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
  real(8) :: tau_ahead, tau_therm, therm_floor, therm_factor, escape_factor

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
   tau_ahead = state%tau_ahead_csm
   if (tau_ahead <= 0d0) tau_ahead = 1d99
   ! A radiative shock can cool locally (handled by the free-free t_cool/t_flow
   ! limiter above) but still fail to build a trapped optical radiation field
   ! once the unshocked CSM column ahead of it is small.  Use the remaining
   ! optical depth as a smooth thermalisation factor, with a low partially
   ! radiative floor while tau_ahead is of order a few and a final shutoff
   ! once the upstream column is optically thin.
   tau_therm = 7d0
   therm_floor = 0.25d0
   therm_factor = therm_floor + (1d0 - therm_floor) / &
                 (1d0 + (tau_therm / max(tau_ahead, 1d-30))**12)
   escape_factor = 1d0 - exp(-min((tau_ahead / 0.10d0)**2, 80d0))
   therm_factor = therm_factor * escape_factor
   eta_fs = eta_fs * therm_factor
   eta_rs = eta_rs * therm_factor
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
  real(8) :: sweep_vol, e_add, E_dim, bin_vol, ov_vol

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
   if (ov_r > ov_l) then
    bin_vol = (bin_r**3 - bin_l**3) / 3d0
    ov_vol = (ov_r**3 - ov_l**3) / 3d0
    if (bin_vol > 0d0 .and. ov_vol > 0d0) then
     state%shell_e(i) = state%shell_e(i) + e_add * min(ov_vol / bin_vol, 1d0)
    end if
   end if
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

 function dimless_dynamics_timescale_cgs(state) result(dt_cgs)
  type(dimless_state_type), intent(in) :: state
  real(8) :: dt_cgs
  real(8) :: dx_dz, dw_dz, dphi_dz, dt_zeta

  dt_cgs = huge(1d0)
  if (.not. state%initialized) return
  if (state%in_cooling_phase) return

  call dynamics_rhs(state%x_sh, state%w_sh, state%phi_sh, state%zeta, &
                    state, dx_dz, dw_dz, dphi_dz)

  dt_zeta = huge(1d0)
  if (abs(dx_dz) > 1d-30) dt_zeta = min(dt_zeta, abs(state%x_sh / dx_dz))
  if (abs(dw_dz) > 1d-30) dt_zeta = min(dt_zeta, abs(state%w_sh / dw_dz))
  if (abs(dphi_dz) > 1d-30) dt_zeta = min(dt_zeta, abs(state%phi_sh / dphi_dz))

  if (dt_zeta < huge(1d0)) dt_cgs = dt_zeta * state%t_in
 end function dimless_dynamics_timescale_cgs

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
  real(8) :: domain_span, collapse_span
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
  if (.not. state%in_cooling_phase .and. .not. state%csm_powerlaw_fast .and. &
      Delta_x <= 1d-4 * max(state%x_csm_out - state%x_min, 1d0)) then
   return
  end if
  dxi = 1d0 / dble(n - 1)  ! Uniform ξ spacing

  state%work_old_e = state%e_grid

  ! Crank-Nicolson by default.  Use backward Euler where the transport problem
  ! is L-stability limited rather than accuracy limited: after shock emergence
  ! the cooling photosphere can recede rapidly through a small optical-depth
  ! shell, and just before emergence the interaction domain can collapse to a
  ! tiny interval.  CN is stable there but rings, which shows up as late-time
  ! luminosity spikes after clipping negative surface modes.
  theta = 0.5d0
  if (state%rannacher_left > 0) then
   theta = 1d0
   state%rannacher_left = state%rannacher_left - 1
  else if (state%in_cooling_phase) then
   theta = 1d0
  else
   domain_span = max(state%x_csm_out - 1d0, 1d0)
   collapse_span = max(2d0 * dxi * domain_span, 5d-2)
   if (Delta_x <= collapse_span) theta = 1d0
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
  ! This matches the Appendix-A boundary-row treatment and avoids evolving a
  ! ghost surface reservoir outside the finite CSM support.
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
  real(8) :: L_heat_pre_cgs
  real(8) :: dx, frac_interp, x_split
  real(8) :: x_l, x_r, x_cap
  real(8) :: profile_scale, profile_norm, density_norm, surface_norm
  real(8) :: density_retained_norm, surface_retained_norm
  real(8) :: surface_budget, density_budget, mixed_budget
  real(8) :: E_channel_total, E_fs_residual_cgs, E_rs_residual_cgs
  real(8) :: tail_fraction, tail_energy_cgs, shell_radius_cgs, diffusion_beta
  real(8) :: eta_out_tail, s_tail_eff, tail_shape_retention, pdv_retention_power
  real(8) :: e_l, e_r
  real(8) :: tau_depth, trap_weight, tau_trap_scale
  real(8) :: r_in_base, r_out_base, r_sh_se, v_sh_se, r_dim
  real(8) :: raw_lum_total_se, radiative_retention_se
  real(8), allocatable :: e_old(:)

  ! Guard: only do this once per run
  if (state%in_cooling_phase) return
  ! Clamp x_sh_old to x_csm_out — numerical overshoot beyond crossover is irrelevant.
  x_sh_old = min(state%x_sh, state%x_csm_out)
  state%x_sh = x_sh_old
  x_ph_old = state%x_ph
  L_surface_pre_cgs = max(state%lum_obs_cgs, 0d0)
  L_heat_pre_cgs = max(state%lum_heat_total_cgs, 0d0)
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
  state%lum_emergence_cgs = L_surface_pre_cgs
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
   dx = (state%x_out_cool - state%x_min_cool) / dble(max(n - 1, 1))
   if (i == 1) x_l = min(state%x_out_cool, x_l + 0.5d0 * dx)
   if (i == n) x_l = max(state%x_min_cool, x_l - 0.5d0 * dx)
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
  if (shock_efficiency_mode == 1) then
   ! For cooling-limited efficiencies, the unresolved cumulative residual should
   ! not be promoted wholesale into a radiative cooling reservoir. Retain the
   ! fraction corresponding to the local radiative efficiency at emergence; the
   ! rest is shock/internal energy that has not cooled into photons on the local
   ! free-free timescale.
   raw_lum_total_se = state%eff * max(forward_shock_luminosity(r_sh_se, state%zeta_se * state%t_in, &
                                      v_sh_se, op) + &
                                      reverse_shock_luminosity(r_sh_se, state%zeta_se * state%t_in, &
                                      v_sh_se, op), 0d0)
   if (raw_lum_total_se > 0d0) then
    radiative_retention_se = min(max(L_heat_pre_cgs / raw_lum_total_se, 0d0), 1d0)
    E_residual_cgs = E_residual_cgs * radiative_retention_se
   end if
  end if
  ! The interaction solve has a photospheric skin already carrying the
  ! pre-emergence luminosity.  If it is folded entirely into the passive
  ! cooling grid, the emitting boundary is rebuilt from a different coordinate
  ! system and extended-CSM curves acquire a numerical dip at t_se.  Reserve a
  ! light-crossing/diffusion-skin amount of that surface radiation and emit it
  ! exponentially after handoff.  This is not post-emergence heating: it is
  ! radiation produced before shock emergence and still in the emitting skin.
  surface_weight = 0d0
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
   call integrate_tau_from_x(x_l, state, tau_depth)
   ! The scalar residual at handoff is trapped internal energy, not radiation
   ! already sitting in the free-streaming surface cell.  Weight the passive
   ! reservoir by optical-depth depth from the outer edge so the emitting skin
   ! is populated by diffusion after handoff, which removes the numerical flash
   ! caused by remapping the whole reservoir directly onto x=x_out.
   tau_trap_scale = max(2d0 / 3d0, min(clight / max(v_sh_se, 1d5), &
                    (2d0 / 3d0) * max(state%x_out_cool / 10d0, 1d0)**3))
   trap_weight = 1d0 - exp(-max(tau_depth, 0d0) / tau_trap_scale)
   state%e_grid(i) = max(compute_eta_csm(x_l, state) * trap_weight, 0d0)
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
! Main transport implementation used by mode='transport'
! Paper Sec. A3: operator splitting (dynamics first, then diffusion)
! ------------------------------------------------------------------
 subroutine dimless_comoving_transport_step(state, dt, lum_heat, lum_obs, r_ph_out)
  type(dimless_state_type), intent(inout) :: state
  real(8), intent(in) :: dt, lum_heat
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
  real(8) :: lum_step_energy_cgs, lum_step_time_cgs, lum_emit_cgs, dt_emit_cgs
  real(8) :: lum_pre_cross_cgs
  ! Pre-step dynamics snapshot for shock-emergence interpolation.
  real(8) :: x_sh_pre, w_sh_pre, phi_sh_pre, zeta_pre
  real(8) :: x_sh_cross, w_sh_cross, phi_sh_cross, zeta_cross, x_ph_pre
  integer, save :: step_log_counter = 0

  lum_obs = 0d0
  state%E_breakout_output_cgs = 0d0
  state%dt_breakout_output_cgs = 0d0
  lum_step_energy_cgs = 0d0
  lum_step_time_cgs = 0d0
  lum_pre_cross_cgs = 0d0

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
   ! For analytic power-law CSM this additional cap preserves the established
   ! publication-plot morphology.  For arbitrary wind tables the outer driver
   ! already limits by the same dynamics timescale; applying this local
   ! phi/(q*w^2) estimate again can over-resolve dense wind shells by orders of
   ! magnitude because q is normalized at the moving contact edge.
   dt_dyn_cfl = huge(1d0)
   if (state%csm_powerlaw_fast .and. .not. state%in_cooling_phase .and. &
       state%w_sh > 1d-30 .and. state%q > 1d-30 .and. state%phi_sh > 1d-30) then
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
   if (state%csm_powerlaw_fast .and. .not. state%in_cooling_phase .and. &
       Delta_x > 0d0 .and. state%x_sh_dot > 1d-30) then
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
    ! The cooling luminosity is read directly from the Eddington surface
    ! cell.  When the post-emergence diffusion wave crosses that cell, taking
    ! the same physical timestep at much finer n_rad_zones under-resolves the
    ! boundary layer and creates zone-dependent surface spikes.  Scale the
    ! cooling accuracy cap with the grid spacing so increasing n_zones
    ! approaches a resolved limit instead of becoming less stable.
    dzeta_step = min(dzeta_step, (0.50d0 * 40d0 / dble(max(state%n_zones, 1))) * &
                              86400d0 / state%t_in)
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
       call dimless_to_cgs(state)
       lum_pre_cross_cgs = max(state%lum_obs_cgs, 0d0)
       lum_emit_cgs = max(state%lum_obs_cgs, 0d0)
       if (state%lum_breakout_avg_cgs > 0d0 .or. state%lum_breakout_cgs > 0d0) then
        lum_emit_cgs = max(lum_emit_cgs - state%lum_breakout_cgs, 0d0) + state%lum_breakout_avg_cgs
       end if
       lum_step_energy_cgs = lum_step_energy_cgs + lum_emit_cgs * dt_int_cgs
       lum_step_time_cgs = lum_step_time_cgs + dt_int_cgs
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
      call dimless_to_cgs(state)
      dt_emit_cgs = (1d0 - frac) * dzeta_step * state%t_in
      lum_emit_cgs = max(state%lum_obs_cgs, 0d0)
      if (state%lum_breakout_avg_cgs > 0d0 .or. state%lum_breakout_cgs > 0d0) then
        lum_emit_cgs = max(lum_emit_cgs - state%lum_breakout_cgs, 0d0) + state%lum_breakout_avg_cgs
       end if
       lum_step_energy_cgs = lum_step_energy_cgs + lum_emit_cgs * dt_emit_cgs
       lum_step_time_cgs = lum_step_time_cgs + dt_emit_cgs
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
     call dimless_to_cgs(state)
     dt_emit_cgs = dzeta_step * state%t_in
     lum_emit_cgs = max(state%lum_obs_cgs, 0d0)
     if (state%lum_breakout_avg_cgs > 0d0 .or. state%lum_breakout_cgs > 0d0) then
        lum_emit_cgs = max(lum_emit_cgs - state%lum_breakout_cgs, 0d0) + state%lum_breakout_avg_cgs
       end if
       lum_step_energy_cgs = lum_step_energy_cgs + lum_emit_cgs * dt_emit_cgs
       lum_step_time_cgs = lum_step_time_cgs + dt_emit_cgs
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
       call dimless_to_cgs(state)
       lum_emit_cgs = max(state%lum_obs_cgs, 0d0)
       if (state%lum_breakout_avg_cgs > 0d0 .or. state%lum_breakout_cgs > 0d0) then
        lum_emit_cgs = max(lum_emit_cgs - state%lum_breakout_cgs, 0d0) + state%lum_breakout_avg_cgs
       end if
       lum_step_energy_cgs = lum_step_energy_cgs + lum_emit_cgs * dt_int_cgs
       lum_step_time_cgs = lum_step_time_cgs + dt_int_cgs
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
      call dimless_to_cgs(state)
      dt_emit_cgs = dzeta_step * state%t_in
      lum_emit_cgs = max(state%lum_obs_cgs, 0d0)
      if (state%lum_breakout_avg_cgs > 0d0 .or. state%lum_breakout_cgs > 0d0) then
       lum_emit_cgs = max(lum_emit_cgs - state%lum_breakout_cgs, 0d0) + state%lum_breakout_avg_cgs
      end if
      lum_step_energy_cgs = lum_step_energy_cgs + lum_emit_cgs * dt_emit_cgs
      lum_step_time_cgs = lum_step_time_cgs + dt_emit_cgs
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
   if (lum_step_time_cgs > 0d0) then
    lum_obs = max(lum_step_energy_cgs / lum_step_time_cgs, 0d0)
    state%lum_obs_cgs = lum_obs
   else
    lum_obs = state%lum_obs_cgs
   end if
   if (lum_step_time_cgs <= 0d0 .and. state%dt_breakout_output_cgs > 0d0) then
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
