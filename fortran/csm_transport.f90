module csm_transport

 use constants, only: pi, clight
 use physical_constants, only: a_rad
 use get_vals, only: op, query_csm_density, query_csm_inner_edge, query_csm_outer_edge, &
                     query_tau_to_edge, query_csm_photosphere_radius, query_csm_velocity, &
                     query_ejecta_density, query_ejecta_velocity

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

contains

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
  call build_radial_grid(state%r_inner_support, state%r_outer_support, state%radius)
  call fill_active_profile_from_reference(state)

  state%y = 1d-40
  state%e_swept = 0d0
  call update_tau_and_luminosity(state)
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
  ihi = first_active_zone_at(state, r_shell)
  if (i == ihi) then
    r_face_l = min(max(r_shell, r_face_l), r_face_r)
    vol_i = 4d0 * pi * max(r_face_r**3 - r_face_l**3, 1d-30) / 3d0
  end if
end subroutine interaction_zone_geometry_at

subroutine update_interaction_geometry(state, r_shell, t_shell, m_shell)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: r_shell, t_shell, m_shell
  integer :: i, ihi
  real(8), parameter :: compression = 7d0   ! strong-shock compression ratio

  state%t_shell = t_shell
  state%r_shell_prev = state%r_shell_current
  state%r_shell_current = max(r_shell, 1d0)
  state%r_inner = state%r_inner_support   ! inner boundary fixed at CSM inner edge
  state%r_outer = state%r_outer_support
  state%radius = state%radius_ref

  ! Cells behind the shock use compressed (post-shock) density so that
  ! radiation is trapped in the shell reservoir.  Cells ahead are unshocked.
  ihi = first_active_zone_at(state, state%r_shell_current)
  do i = 1, state%n_zones
   if (i < ihi) then
    state%rho(i) = compression * state%rho_ref(i)
   else
    state%rho(i) = state%rho_ref(i)
   end if
  end do
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
   new_y(i) = max(e_new(i) / max(a_rad * new_vol, 1d-30), 1d-40)
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
   if (state%in_cooling_phase) then
    call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   else
    call interaction_zone_geometry_at(state, i, state%r_shell_current, r_face_l, r_face_r, vol_i)
   end if
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
  real(8) :: mdot_csm, mdot_ej, v_csm, v_ej, rho_csm, rho_ej
  real(8) :: lum_input, r_fs
  real(8) :: u_grid, r_face_l, r_face_r, vol_i, dUdt, dt_eff
  integer :: i, istart
  real(8), save :: u_prev = 0d0
  logical, save :: have_prev = .false.
  integer, save :: energy_step = 0

  r_fs = forward_shock_radius(state, r_shell, t_shell, m_shell)
  state%u_shell_current = u_shell
  state%lum_heat_last = lum_heat
  if (.not. state%initialized) then
   call initialize_interaction_grid(state, r_fs, t_shell, m_shell, max(state%kappa, 1d-30), max(state%n_zones,48))
  else
   call update_interaction_grid(state, r_fs, t_shell, m_shell)
  end if

  rho_csm = max(query_csm_density(r_shell, t_shell, op(2)), 0d0)
  rho_ej = max(query_ejecta_density(r_shell, t_shell, op(1)), 0d0)
  v_csm = query_csm_velocity(r_shell, t_shell, op(2))
  v_ej = query_ejecta_velocity(r_shell, t_shell, op(1))
  mdot_csm = 4d0*pi*r_shell**2 * rho_csm * max(u_shell - v_csm, 0d0)
  mdot_ej = 4d0*pi*r_shell**2 * rho_ej * max(v_ej - u_shell, 0d0)
  state%m_shocked_csm = state%m_shocked_csm + dt * mdot_csm
  state%m_shocked_ej = state%m_shocked_ej + dt * mdot_ej

  lum_input = max(lum_heat, 0d0)
  state%e_residual = 0d0

  call solve_transport_step(state, dt, lum_input)
  call find_transport_photosphere(state, r_ph, lum_obs)

  if (state%initialized) then
   u_grid = 0d0
   do i = 1, state%n_zones
    call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
    u_grid = u_grid + a_rad * max(state%y(i), 0d0) * vol_i
   end do
   if (have_prev) then
    dt_eff = max(dt, 1d-30)
    dUdt = (u_grid - u_prev) / dt_eff
    energy_step = energy_step + 1
    if (mod(energy_step, 50) == 0) then
     write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
          'ENERGY_BUDGET t=', t_shell, 'U_grid=', u_grid, 'dUdt=', dUdt, 'L_heat=', lum_heat, 'L_obs=', lum_obs
    end if
   end if
   u_prev = u_grid
   have_prev = .true.
  end if
end subroutine interaction_transport_step

subroutine initialize_cooling_state_from_interaction(state, r_shell, u_shell, m_shell, t_shell, lum_target, lum_heat, u_reservoir)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: r_shell, u_shell, m_shell, t_shell
  real(8), intent(in), optional :: lum_target, lum_heat
  real(8), intent(in), optional :: u_reservoir
  integer :: i
  real(8), allocatable :: u_old(:), u_new(:)
  real(8), allocatable :: old_face_l(:), old_face_r(:), old_face_vol(:)
  real(8), allocatable :: new_face_l(:), new_face_r(:), new_face_vol(:)
  real(8), allocatable :: old_radius_ref(:)
  real(8) :: r_inner_old, r_outer_old
  real(8) :: r_face_l, r_face_r, vol_full, mass_shape, rho_norm
  real(8) :: rho_out, eta
  real(8) :: tau_shell
  real(8) :: vol_i, dr
  real(8) :: rhoe_old, rlo, rhi, overlap_vol, rlo3, rhi3, new_l3, new_r3
  real(8) :: m_residual, m_shell_eff, dt_gap
  real(8) :: r0, r1, rho0, rho1, dr_seg, ratio
  real(8) :: m_test, r_lo, r_hi, r_mid
  integer :: nseg, iter
  real(8), parameter :: four_pi_over_3 = 4.18879020478639098461685784438d0
  integer :: i_start, j, ihi
  integer :: nbridge, i0
  real(8) :: u_int, u_target, u_surf, u_interior, scale_int, y_target
  real(8) :: y0, frac, scale_bridge, u_bridge
  real(8) :: u_res_add, total_vol
  if (.not. state%initialized) return

  state%in_cooling_phase = .true.
  state%cooling_initialized = .true.
  state%t_emerge = t_shell
  state%r_emerge_shell = max(r_shell, 1d0)
  state%u_emerge_shell = max(u_shell, 1d-30)
  state%m_emerge_csm = max(state%m_shocked_csm, 0d0)
  state%m_emerge_ej = max(state%m_shocked_ej, 0d0)
  dt_gap = 0d0
  state%t_gap_end = t_shell
  state%lum_heat_gap = 0d0
  m_residual = 0d0
  if (state%r_outer_support > max(r_shell, 1d0)) then
   r0 = max(r_shell, 1d0)
   rhi = state%r_outer_support
   nseg = 128
   ratio = (rhi / r0)**(1d0 / dble(nseg))
   r1 = r0
   rho1 = query_csm_density(r1, t_shell, op(2))
   do i = 1, nseg
    r0 = r1
    rho0 = rho1
    r1 = min(rhi, r1 * ratio)
    rho1 = query_csm_density(r1, t_shell, op(2))
    dr_seg = r1 - r0
    m_residual = m_residual + 4d0 * pi * 0.5d0 * (rho0 * r0**2 + rho1 * r1**2) * dr_seg
    if (r1 >= rhi) exit
   end do
  end if
  m_shell_eff = max(state%m_shocked_csm + state%m_emerge_ej + m_residual, 1d-30)
  state%m_emerge_shell = m_shell_eff
  state%r_emerge_outer = state%r_outer_support
  rho_out = max(query_csm_density(state%r_emerge_shell, t_shell, op(2)), 1d-30)
  eta = 7d0
  ! Solve for r_emerge_inner by integrating the ejecta density inward from r_shell.
  ! The shell includes both shocked CSM (outside r_inner_support) and shocked ejecta
  ! (inside r_shell). We search inward through ejecta until the enclosed mass
  ! (eta * CSM mass from r_shell outward + ejecta mass from r_shell inward) equals
  ! m_shell_eff.
  r_lo = state%r_inner_support * 1d-3   ! allow deep into ejecta
  r_hi = state%r_emerge_shell
  do iter = 1, 80
   r_mid = sqrt(r_lo * r_hi)
   m_test = 0d0
   ! CSM contribution: eta * integral from r_mid to r_outer (if r_mid < r_inner_support)
   if (r_mid < state%r_inner_support) then
    r0 = state%r_inner_support
   else
    r0 = r_mid
   end if
   r1 = r0
   nseg = 128
   if (state%r_emerge_outer > r0) then
    ratio = (state%r_emerge_outer / max(r0, 1d0))**(1d0 / dble(nseg))
    rho1 = query_csm_density(r1, t_shell, op(2))
    do i = 1, nseg
     r0 = r1
     rho0 = rho1
     r1 = min(state%r_emerge_outer, r1 * ratio)
     rho1 = query_csm_density(r1, t_shell, op(2))
     dr_seg = r1 - r0
     m_test = m_test + eta * 4d0 * pi * 0.5d0 * (rho0 * r0**2 + rho1 * r1**2) * dr_seg
     if (r1 >= state%r_emerge_outer) exit
    end do
   end if
   ! Ejecta contribution: integral from r_mid to r_shell
   r0 = r_mid
   r1 = r0
   nseg = 128
   ratio = (state%r_emerge_shell / max(r0, 1d0))**(1d0 / dble(nseg))
   rho1 = query_ejecta_density(r1, t_shell, op(1))
   do i = 1, nseg
    r0 = r1
    rho0 = rho1
    r1 = min(state%r_emerge_shell, r1 * ratio)
    rho1 = query_ejecta_density(r1, t_shell, op(1))
    dr_seg = r1 - r0
    m_test = m_test + 4d0 * pi * 0.5d0 * (rho0 * r0**2 + rho1 * r1**2) * dr_seg
    if (r1 >= state%r_emerge_shell) exit
   end do
   if (m_test > m_shell_eff) then
    r_lo = r_mid
   else
    r_hi = r_mid
   end if
  end do
  state%r_emerge_inner = min(max(r_lo, state%r_inner_support * 1d-3), state%r_emerge_shell)
  if (state%r_emerge_outer <= state%r_emerge_inner) then
   state%r_emerge_inner = max(state%r_emerge_outer * (1d0 - 1d-6), 1d0)
  end if
  state%cooling_scale = 1d0
  state%compact_tail_drain = 0d0

  state%r_inner = state%r_emerge_inner
  state%r_outer = state%r_emerge_outer

  ! Save old interaction grid before rebuilding, so energy handoff uses the
  ! correct old face geometry.
  allocate(old_radius_ref(state%n_zones))
  old_radius_ref = state%radius_ref
  r_inner_old = state%r_inner_support
  r_outer_old = state%r_outer_support

  ! Rebuild radius_ref to span [r_emerge_inner, r_emerge_outer] so the cooling
  ! grid is properly distributed over the shell, and update_cooling_grid scales
  ! from this baseline.
  call build_radial_grid(state%r_emerge_inner, state%r_emerge_outer, state%radius_ref)
  state%radius = state%radius_ref
  do i = 1, state%n_zones
   if (state%radius_ref(i) < state%r_inner_support) then
    ! Inside CSM inner edge: use ejecta density
    state%rho_ref(i) = max(query_ejecta_density(state%radius_ref(i), t_shell, op(1)), 1d-30)
   else
    ! In the CSM/shell region: use eta-compressed CSM density
    state%rho_ref(i) = max(eta * query_csm_density(state%radius_ref(i), t_shell, op(2)), 1d-30)
   end if
   state%rho(i) = state%rho_ref(i)
  end do

  allocate(u_old(state%n_zones), u_new(state%n_zones))
  allocate(old_face_l(state%n_zones), old_face_r(state%n_zones), old_face_vol(state%n_zones))
  allocate(new_face_l(state%n_zones), new_face_r(state%n_zones), new_face_vol(state%n_zones))
  u_old = 0d0
  u_new = 0d0

  ! Compute ihi and old face geometry using the SAVED old interaction grid,
  ! not the new cooling grid (which radius_ref now points to).
  ihi = first_active_zone_at_arr(old_radius_ref, r_inner_old, r_outer_old, state%n_zones, state%r_shell_current)
  do i = 1, state%n_zones
   call zone_geometry_from_array(old_radius_ref, r_inner_old, r_outer_old, state%n_zones, i, &
        r_face_l, r_face_r, vol_full)
   if (i < ihi) then
    r_face_l = r_face_r
   else if (i == ihi) then
    r_face_l = min(max(state%r_shell_current, r_face_l), r_face_r)
   end if
   old_face_l(i) = r_face_l
   old_face_r(i) = r_face_r
   old_face_vol(i) = four_pi_over_3 * max(r_face_r**3 - r_face_l**3, 1d-30)
   u_old(i) = a_rad * max(state%y(i), 0d0) * old_face_vol(i)
  end do

  call prepare_zone_geometry_arrays(state%radius_ref, state%r_emerge_inner, state%r_emerge_outer, state%n_zones, &
       new_face_l, new_face_r, new_face_vol)

  i_start = 1
  do j = 1, state%n_zones
   if (old_face_vol(j) <= 0d0) cycle
   rhoe_old = u_old(j) / max(old_face_vol(j), 1d-30)
   if (rhoe_old <= 0d0) cycle
   rlo = max(old_face_l(j), state%r_emerge_inner)
   rhi = min(old_face_r(j), state%r_emerge_outer)
   if (rhi <= rlo) cycle
   rlo3 = rlo**3
   rhi3 = rhi**3
   do i = i_start, state%n_zones
    if (new_face_l(i) >= rhi) exit
    if (new_face_r(i) <= rlo) then
     i_start = i + 1
     cycle
    end if
    new_l3 = max(rlo3, new_face_l(i)**3)
    new_r3 = min(rhi3, new_face_r(i)**3)
    if (new_r3 > new_l3) then
     overlap_vol = four_pi_over_3 * (new_r3 - new_l3)
     u_new(i) = u_new(i) + rhoe_old * overlap_vol
    end if
   end do
  end do

  ! Add cumulative shock heating reservoir energy, distributed uniformly by mass.
  u_res_add = 0d0
  if (present(u_reservoir)) u_res_add = max(u_reservoir, 0d0)
  if (u_res_add > 0d0) then
   ! Distribute by optical depth (mass) so reservoir energy stays inside photosphere.
   rlo = 0d0
   do i = 1, state%n_zones
    call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
    dr = max(r_face_r - r_face_l, 1d-30)
    old_face_l(i) = state%kappa * max(state%rho(i), 1d-30) * dr  ! weight = optical depth
    rlo = rlo + old_face_l(i)
   end do
   if (rlo > 0d0) then
    do i = 1, state%n_zones
     u_new(i) = u_new(i) + u_res_add * (old_face_l(i) / rlo)
    end do
   else
    total_vol = sum(new_face_vol)
    if (total_vol > 0d0) then
     do i = 1, state%n_zones
      u_new(i) = u_new(i) + u_res_add * (new_face_vol(i) / total_vol)
     end do
    end if
   end if
  end if

  do i = 1, state%n_zones
   state%y(i) = max(u_new(i) / max(a_rad * new_face_vol(i), 1d-30), 1d-40)
  end do

  ! Report handoff: vol_full=U_int, rlo=U_cool
  vol_full = sum(u_old)
  rlo = sum(u_new)
  write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
       'HANDOFF_ENERGY U_int=', vol_full, 'U_cool=', rlo, 'R_in=', state%r_emerge_inner, &
       'R_out=', state%r_emerge_outer, 'dR=', state%r_emerge_outer - state%r_emerge_inner
  call flush(6)
  state%e_swept = 0d0
  state%e_residual = 0d0
  deallocate(u_old, u_new)
  deallocate(new_face_l, new_face_r, new_face_vol)
  deallocate(old_radius_ref)
  tau_shell = state%tau(1)
  write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
       'HANDOFF_SHELL m_shell=', state%m_emerge_shell, 'rho_sh=', rho_out, 'tau_shell=', tau_shell, &
       'r_inner=', state%r_emerge_inner
end subroutine initialize_cooling_state_from_interaction

subroutine update_cooling_grid(state, t_shell)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: t_shell
  real(8) :: prev_scale, next_scale, scale_ratio4
  real(8) :: t_relax, ramp
  integer :: i

  if (.not. state%initialized) return

  state%t_shell = t_shell
  prev_scale = max(state%cooling_scale, 1d-30)
  state%cooling_scale_prev = prev_scale
  ! Homologous expansion: each zone expands as r(t) = r_ref * t/t_emerge.
  next_scale = max(t_shell / max(state%t_emerge, 1d-30), 1d0)
  state%cooling_scale = next_scale

  state%r_inner = state%r_emerge_inner * next_scale
  state%r_outer = state%r_emerge_outer * next_scale
  do i = 1, state%n_zones
   state%radius(i) = state%radius_ref(i) * next_scale
   state%rho(i) = max(state%rho_ref(i) / next_scale**3, 1d-30)
  end do
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
  if (.not. state%initialized) then
   lum_obs = 0d0
   r_ph = 0d0
   return
  end if
  if (.not. state%cooling_initialized) then
   write(*,'(A,1X,ES12.5)') 'WARNING: forced cooling init at t_shell=', t_shell
   call initialize_cooling_state_from_interaction(state, state%r_shell_current, &
        max(state%u_shell_current, 1d-30), max(state%m_shocked_csm + state%m_shocked_ej, 1d-30), t_shell, &
        lum_heat=state%lum_heat_last)
  end if

  lum_heat_local = 0d0
  if (present(lum_heat)) lum_heat_local = max(lum_heat, 0d0)
  call update_cooling_grid(state, t_shell)
  call solve_transport_step(state, dt, lum_heat_local)
  call find_transport_photosphere(state, r_ph, lum_obs)
  ! Debug: track energy in cooling grid
  if (mod(int(t_shell/86400d0), 5) == 0 .and. t_shell > state%t_emerge + 0.5d0*86400d0) then
   lum_heat_local = total_radiation_energy(state)
   write(*,'(A,F8.2,A,ES12.5,A,ES12.5,A,ES12.5,A,ES12.5)') &
    'COOL_DBG t=', t_shell/86400d0, 'd E=', lum_heat_local, ' L=', lum_obs, &
    ' tau1=', state%tau(1), ' scale=', state%cooling_scale
   call flush(6)
  end if
end subroutine cooling_transport_step

subroutine comoving_transport_step(state, dt, t_shell, r_sh, v_sh, lum_heat, lum_obs)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: dt, t_shell, r_sh, v_sh, lum_heat
  real(8), intent(out) :: lum_obs
  integer, save :: dbg_count = 0
  logical, save :: dbg_enabled = .false.
  integer, save :: last_ish = 1
  logical, save :: emerge_logged = .false.
  integer :: i, n, ish
  integer :: n_active
  real(8) :: r_in, r_out, x_out, x_sh, dx
  real(8) :: x_i, x_m, x_p
  real(8) :: rho_in, rho_i, rho_m, rho_p, eta_i, eta_m, eta_p
  real(8) :: tau_in, t_diff, dy, dy_total, dy_max
  real(8) :: x_sh_loc, dx_step, dt_step, t_loc, t_target
  real(8) :: r_sh_loc, v_sh_loc
  real(8) :: m_sh_csm, m_sh_ej, m_sh_tot
  real(8) :: rho_csm, rho_ej, v_csm, v_ej
  real(8) :: lum_sh, scale
  real(8) :: rho_out_eff
  real(8) :: t_in, zeta, dzeta, phi_dim
  real(8) :: zeta_loc, zeta_target, dzeta_step, dzeta_max
  real(8) :: rho_ej_in, q_ratio, eta_ej, eta_csm
  real(8) :: dx_dzeta, dphi_dzeta, dw_dzeta
  real(8) :: dm_csm, dm_ej, p_dot, dv_sh
  real(8) :: Dm, Dp, coef_m, coef_p
  real(8) :: u0, w, f_ib, f_ob
  real(8) :: v_ej_max, rho_sh, rho_out, eta_sh
  real(8) :: du_dx, u_out
  logical :: emerged
  real(8) :: y_min, y_max
  logical :: y_bad

  if (.not. state%initialized) then
   n = max(16, state%n_zones)
   if (state%kappa <= 0d0) state%kappa = 0.34d0
   call reset_transport_state(state)
   state%n_zones = n
   state%initialized = .true.
   allocate(state%radius(state%n_zones), state%radius_ref(state%n_zones), &
            state%rho(state%n_zones), state%rho_ref(state%n_zones), state%y(state%n_zones), &
            state%e_swept(state%n_zones), state%tau(state%n_zones))
   allocate(state%work_a(state%n_zones), state%work_b(state%n_zones), state%work_c(state%n_zones), &
            state%work_rhs(state%n_zones), state%work_sol(state%n_zones), state%work_old_y(state%n_zones), &
            state%work_gam(state%n_zones))
   state%y = 0d0
   state%e_swept = 0d0
   last_ish = 1
   emerge_logged = .false.
  end if
  n = state%n_zones
  if (n < 3) return

  r_in = max(query_csm_inner_edge(t_shell, op(2)), 1d0)
  r_out = query_csm_outer_edge(t_shell, op(2))
   if (state%in_cooling_phase .and. state%r_emerge_inner > 0d0) then
    r_in = state%r_emerge_inner + state%u_emerge_shell * max(t_shell - state%t_emerge, 0d0)
    r_out = state%r_emerge_outer * (r_in / state%r_emerge_inner)
   end if
  x_out = max(r_out / r_in, 1d0)
  dx = (x_out - 1d0) / dble(n-1)
  if (state%radius_ref(1) <= 0d0 .or. abs(state%radius_ref(n) - x_out) > 1d-6) then
   do i = 1, n
    state%radius_ref(i) = 1d0 + dx * dble(i-1)
   end do
  end if

  if (state%kappa <= 0d0) state%kappa = 0.34d0
  ! Use the appropriate max ejecta velocity for the ejecta profile type.
  ! For exponential profiles use exp_v0; for BPL profiles use bpl_vt.
  v_ej_max = max(op(1)%exp_v0, op(1)%bpl_vt, 1d-30)
  if (state%in_cooling_phase .and. state%cooling_initialized .and. state%r_emerge_inner > 0d0) then
   ! Cooling phase: freeze all normalization quantities at emergence values.
   ! R_0 = r_emerge_inner (= R_csm_in), use it for tau_in, t_diff, u0.
   ! The expanding r_in is only used for the scale = R_in/R_0 factor in D.
   rho_in = max(state%rho_ref(1), 1d-30)
   tau_in = 3d0 * state%kappa * rho_in * state%r_emerge_inner
   t_diff = max(tau_in * state%r_emerge_inner / clight, 1d-30)
   t_in = state%r_emerge_inner / v_ej_max
   rho_ej_in = max(query_ejecta_density(state%r_emerge_inner, t_in, op(1)), 1d-30)
   q_ratio = max(rho_in / rho_ej_in, 1d-30)
   u0 = (tau_in * v_ej_max / clight) * (0.5d0 * rho_in * v_ej_max * v_ej_max)
  else
   rho_in = max(query_csm_density(state%radius_ref(1) * r_in, t_shell, op(2)), 1d-30)
   tau_in = 3d0 * state%kappa * rho_in * r_in
   t_diff = max(tau_in * r_in / clight, 1d-30)
   t_in = r_in / v_ej_max
   rho_ej_in = max(query_ejecta_density(r_in, t_in, op(1)), 1d-30)
   q_ratio = max(rho_in / rho_ej_in, 1d-30)
   ! Appendix A: u0 scaled to CSM density at R_in
   u0 = (tau_in * v_ej_max / clight) * (0.5d0 * rho_in * v_ej_max * v_ej_max)
  end if

  x_sh = max(r_sh / r_in, 1d0)
  if (.not. state%comoving_initialized) then
   r_sh_loc = r_in
   v_sh_loc = v_ej_max
   x_sh_loc = 1d0
   w = 1d0
   phi_dim = 1d-1
   m_sh_tot = max(phi_dim * (4d0 * pi * r_in**3 * rho_ej_in), 1d-30)
   m_sh_csm = m_sh_tot
   m_sh_ej = 0d0
   t_loc = t_in
   state%comoving_initialized = .true.
  else
   if (state%r_shell_current > 0d0 .and. state%u_shell_current > 1d-6 * v_ej_max) then
    r_sh_loc = max(state%r_shell_current, r_in)
    v_sh_loc = max(state%u_shell_current, 1d-30)
    m_sh_csm = max(state%m_shocked_csm, 0d0)
    m_sh_ej = max(state%m_shocked_ej, 0d0)
    m_sh_tot = max(m_sh_csm + m_sh_ej, 1d-30)
   else
    r_sh_loc = max(r_sh, r_in)
    v_sh_loc = max(v_sh, v_ej_max)
    m_sh_tot = 1d-10 * (4d0 * pi * r_in**3 * rho_ej_in)
    m_sh_csm = m_sh_tot
    m_sh_ej = 0d0
   end if
   x_sh_loc = r_sh_loc / r_in
   t_loc = t_shell
  end if
  zeta_loc = t_loc / t_in
  zeta_target = zeta_loc + max(dt / t_in, 0d0)
  dzeta_max = 1d-2

  ! Precompute CSM density on fixed x-grid for this t_shell.
  ! During cooling phase, the shell has left the CSM so we must NOT query the
  ! CSM module (which returns zero outside R_csm_out). Instead, freeze the
  ! reference density at emergence (querying using the original CSM inner radius)
  ! and apply the (R_0/R_in)^3 adiabatic expansion factor separately in D.
  ! rho_ref stores the UNSCALED reference profile (as at emergence).
  if (state%in_cooling_phase .and. state%cooling_initialized) then
   ! During cooling: use frozen profile at original x*R_0 positions.
   ! Only re-fill rho_ref if this is the first cooling call (rho_ref still
   ! holds the interaction-phase densities at R_csm_in scale — reuse them).
   ! rho_out: use the n-th zone reference density (no expansion scaling needed
   ! here; scaling is applied in D_m/D_p computation below via 'scale').
   rho_out = state%rho_ref(n)
  else
   do i = 1, n
    state%rho_ref(i) = max(query_csm_density(state%radius_ref(i) * r_in, t_shell, op(2)), 1d-30)
   end do
   rho_out = max(query_csm_density(r_out, t_shell, op(2)), 1d-30)
  end if

  ! Operator-split integration: couple Euler ODE + CN diffusion each sub-step.
  ! ODE accuracy requires dzeta ~ 0.01; CN is unconditionally stable so one
  ! diffusion solve per ODE step is fine. Typically ~100 steps per 1-day call.
  dzeta_max = 1d-2
  do while (zeta_loc < zeta_target - 1d-12)
   dzeta_step = min(dzeta_max, zeta_target - zeta_loc)
   dt_step = dzeta_step * t_in
   dy = dt_step / t_diff
   if (dy <= 0d0) exit

    emerged = state%in_cooling_phase
    if (emerged .and. state%r_emerge_inner > 0d0) then
     r_in = state%r_emerge_inner + state%u_emerge_shell * max(t_loc - state%t_emerge, 0d0)
     r_out = state%r_emerge_outer * (r_in / state%r_emerge_inner)
    end if
    ish = int((x_sh_loc - 1d0) / dx) + 1
    ish = max(2, min(n-1, ish))

    rho_csm = max(query_csm_density(r_sh_loc, t_shell, op(2)), 1d-30)
    v_csm = query_csm_velocity(r_sh_loc, t_shell, op(2))
    rho_ej = max(query_ejecta_density(r_sh_loc, t_loc, op(1)), 1d-30)
    v_ej = query_ejecta_velocity(r_sh_loc, t_loc, op(1))
    eta_csm = max(rho_csm / rho_in, 1d-30)
    zeta = max(t_loc / t_in, 1d-30)
    eta_ej = max(rho_ej * zeta**3 / rho_ej_in, 1d-30)
    w = max(v_sh_loc / v_ej_max, 1d-30)
    f_ib = 0.5d0 * eta_csm * eta_csm * w**3
    lum_sh = 2d0 * pi * r_sh_loc**2 * rho_csm * max(v_sh_loc - v_csm, 0d0)**3
    if (.not. state%in_cooling_phase) state%e_residual = state%e_residual + lum_sh * dt_step

    if (.not. emerged) then
     state%e_swept(1:ish) = state%e_swept(1:ish) + f_ib * dy
    end if
   state%work_old_y = state%y
   state%work_a = 0d0
   state%work_b = 0d0
   state%work_c = 0d0
   state%work_rhs = 0d0

   do i = 2, n-1
    x_i = state%radius_ref(i)
    x_m = x_i - 0.5d0 * dx
    x_p = x_i + 0.5d0 * dx
    rho_i = state%rho_ref(i)
    rho_m = 0.5d0 * (state%rho_ref(i-1) + state%rho_ref(i))
    rho_p = 0.5d0 * (state%rho_ref(i) + state%rho_ref(i+1))
    eta_m = rho_m / rho_in
    eta_p = rho_p / rho_in
    scale = 1d0
    if (state%in_cooling_phase .and. state%r_emerge_inner > 0d0) then
     scale = r_in / state%r_emerge_inner
    end if
    Dm = (x_m * x_m) / max(eta_m, 1d-30) * scale
    Dp = (x_p * x_p) / max(eta_p, 1d-30) * scale
    coef_m = 0.5d0 * dy * Dm / (x_i * x_i * dx * dx)
    coef_p = 0.5d0 * dy * Dp / (x_i * x_i * dx * dx)
    if (.not. emerged .and. i == ish) then
     ! Shock cell: inject Neumann flux f_ib from the left (shock → CSM direction).
     ! Standard CN interior stencil but left flux replaced by f_ib source.
     state%work_a(i) = 0d0
     state%work_b(i) = 1d0 + coef_p
     state%work_c(i) = -coef_p
     state%work_rhs(i) = state%work_old_y(i) + coef_p * (state%work_old_y(i+1) - state%work_old_y(i)) &
                         + coef_m * f_ib
    else
     state%work_a(i) = -coef_m
     state%work_b(i) = 1d0 + coef_m + coef_p
     state%work_c(i) = -coef_p
     state%work_rhs(i) = state%work_old_y(i) + coef_p * (state%work_old_y(i+1) - state%work_old_y(i)) &
                         - coef_m * (state%work_old_y(i) - state%work_old_y(i-1))
    end if
   end do

   ! Inner boundary at x=1: always reflective (zero flux) — energy is trapped
   ! between x=1 and x_sh, diffusing outward.  When the shock is at cell 1,
   ! inject the Neumann source here instead.
   if (.not. emerged .and. ish == 1) then
    ! Shock at first cell: inject flux as Neumann source on i=1
    x_i = state%radius_ref(1)
    x_p = x_i + 0.5d0 * dx
    rho_p = 0.5d0 * (state%rho_ref(1) + state%rho_ref(2))
    eta_p = rho_p / rho_in
    Dp = (x_p * x_p) / max(eta_p, 1d-30)
    coef_p = 0.5d0 * dy * Dp / (x_i * x_i * dx * dx)
    state%work_a(1) = 0d0
    state%work_b(1) = 1d0 + coef_p
    state%work_c(1) = -coef_p
    state%work_rhs(1) = state%work_old_y(1) + coef_p * (state%work_old_y(2) - state%work_old_y(1))
   else
    ! Reflective (zero-gradient) BC at x=1: y(1) = y(2)
    state%work_a(1) = 0d0
    state%work_b(1) = 1d0
    state%work_c(1) = -1d0
    state%work_rhs(1) = 0d0
   end if

   ! Outer boundary Robin at x_out:  e(n) = f_ob * de/dx
   ! Encoded as: a(n)*e(n-1) + b(n)*e(n) = 0
   !   where de/dx ≈ (e(n)-e(n-1))/dx  →  a = f_ob/dx, b = 1 - f_ob/dx.
   ! This is stable for the Thomas algorithm for any value of f_ob.
   eta_i = rho_out / rho_in
   if (.not. emerged) then
    f_ob = -4d0 / max(tau_in * eta_i, 1d-30)
   else
    scale = 1d0
    if (state%r_emerge_inner > 0d0) scale = r_in / state%r_emerge_inner
    f_ob = -2d0 / max(tau_in * eta_i, 1d-30) * scale * scale
   end if
   state%work_a(n) = f_ob / dx
   state%work_b(n) = 1d0 - f_ob / dx
   state%work_c(n) = 0d0
   state%work_rhs(n) = 0d0

    ! Always solve the full n-cell system.  Interior cells behind the shock
    ! (i < ish) are part of the domain — energy trapped there can diffuse
    ! outward and contribute to the observed luminosity.
    call tridag(state%work_a, state%work_b, state%work_c, state%work_rhs, state%work_sol, state%work_gam, n)
    state%y = max(state%work_sol, 0d0)

    ! Update shell dynamics using dimensionless ODEs (Appendix A)
    if (.not. state%in_cooling_phase) then
     dzeta = dt_step / t_in
     phi_dim = max(m_sh_tot / (4d0 * pi * r_in**3 * rho_ej_in), 1d-30)
     dx_dzeta = w
     dphi_dzeta = x_sh_loc**2 * zeta**(-3d0) * (x_sh_loc / zeta - w) * eta_ej + &
                  q_ratio * x_sh_loc**2 * w * eta_csm
     dw_dzeta = (x_sh_loc**2 * zeta**(-3d0) * (x_sh_loc / zeta - w)**2 * eta_ej - &
                 q_ratio * x_sh_loc**2 * w**2 * eta_csm) / max(phi_dim,1d-30)

     x_sh_loc = max(x_sh_loc + dx_dzeta * dzeta, 1d0)
     w = max(w + dw_dzeta * dzeta, 1d-20)
     phi_dim = max(phi_dim + dphi_dzeta * dzeta, 1d-30)
     r_sh_loc = x_sh_loc * r_in
     v_sh_loc = w * v_ej_max
     m_sh_tot = phi_dim * (4d0 * pi * r_in**3 * rho_ej_in)
     m_sh_csm = m_sh_tot
     m_sh_ej = 0d0
     if ((.not. state%in_cooling_phase) .and. (x_sh_loc >= x_out)) then
      state%in_cooling_phase = .true.
      state%t_emerge = t_loc
      state%r_emerge_inner = r_in
      state%r_emerge_outer = r_out
      state%u_emerge_shell = v_sh_loc
      state%cooling_initialized = .true.
      ! Keep state%y as-is: the CN solve already produced the correct radiation
      ! profile at emergence. Do NOT overwrite with e_swept.
      emerge_logged = .true.
      x_sh_loc = x_out
      r_sh_loc = x_out * r_in
      w = 0d0
      v_sh_loc = 0d0
     end if
    else
     ! After breakout: freeze shock at outer boundary
     x_sh_loc = x_out
     r_sh_loc = x_out * r_in
     w = 0d0
     v_sh_loc = 0d0
    end if
    t_loc = t_loc + dt_step
    zeta_loc = zeta_loc + dt_step / t_in
  end do

  eta_i = rho_out / rho_in
  ! Compute luminosity from the diffusion flux at the outer boundary:
  !   L = 4π R_ph^2 * F_ph,  F_ph = -(c / 3κρ_ph) * ∂u_phys/∂r
  ! In dimensionless form:
  !   Interaction:  u_phys = u0 * e,  ∂u/∂r = u0/R_csm_in * ∂e/∂x
  !     F = (c*u0)/(3*τ_in*η_ph*R_csm_in) * |de/dx|
  !     L = 4π*R_ph^2 * (c*u0)/(3*τ_in*η_ph*R_csm_in) * |de/dx|
  !   Cooling:  u_phys = u0 * e * (R_0/R_in)^4,  ∂u/∂r = u0*(R_0/R_in)^4/R_in * ∂e/∂x
  !     F = (c*u0*(R_0/R_in)^4)/(3*τ_in_frozen*η_ph*(R_0/R_in)^2*R_in) * |de/dx|
  !       = (c*u0*(R_0/R_in)^2)/(3*τ_in_frozen*η_ph*R_in) * |de/dx|
  !       = (c*u0*(R_0/R_in)^2)/(3*κ*ρ_in*η_ph*R_in) * |de/dx|
  ! Simplified: in both phases  L = 4π*R_ph^2 * D_n_phys * u0 * phys_scale * |de/dx| / R_ref
  ! where D_n = x_n^2/η_ph (dimensionless diffusion at outer cell face) and the
  ! physical factor is c/(3*τ_in*R_csm_in^2) * R_ph^2 = cR_ph^2/(3τ_in*R0^2).
  ! Practical formula (same sign convention as Eddington):
  !   L = 4π * (c / (3*τ_in*η_ph)) * u0 * |de/dx| * R_ph^2 / R_csm_in
  ! where de/dx = (y(n) - y(n-1))/dx  and R_ph = r_out.
  scale = 1d0
  if (state%in_cooling_phase .and. state%r_emerge_inner > 0d0) then
   r_in = state%r_emerge_inner + state%u_emerge_shell * max(t_loc - state%t_emerge, 0d0)
   r_out = state%r_emerge_outer * (r_in / state%r_emerge_inner)
   scale = (state%r_emerge_inner / r_in)**4
  end if
  ! Use the Eddington surface formula: L = 4πR^2 * (c/4) * u_surface
  ! = 4π * r_out^2 * (c/4) * u0 * scale_phys * y(n)
  ! This is equivalent to the flux formula when the Robin BC is active.
  ! It also gives a smooth result when the Dirichlet cap is active (y(n)→0 → L→0
  ! gracefully) and remains finite as long as y(n) is finite.
  lum_obs = 4d0 * pi * r_out**2 * (0.25d0 * clight) * (u0 * state%y(n) * scale)
  if (.not.(lum_obs == lum_obs) .or. abs(lum_obs) > 1d300) lum_obs = 0d0
  lum_obs = max(lum_obs, 0d0)

  dbg_count = dbg_count + 1
  if (.false.) then  ! debug prints disabled
   write(*,'(A,I4,A,ES10.3,A,ES10.3,A,ES10.3,A,ES10.3,A,ES10.3,A,L1)') &
        'COMOV #', dbg_count, ' t=', t_shell, ' L=', lum_obs, ' y(n)=', state%y(n), &
        ' scale=', scale, ' r_in/R0=', r_in/(state%r_emerge_inner+1d0), ' cool=', state%in_cooling_phase
   call flush(6)
  end if

  if (dbg_enabled .and. dbg_count < 20) then
   dbg_count = dbg_count + 1
   y_min = minval(state%y)
   y_max = maxval(state%y)
   y_bad = .false.
   if (y_min /= y_min .or. y_max /= y_max) y_bad = .true.
   if (abs(y_min) > 1d250 .or. abs(y_max) > 1d250) y_bad = .true.
   write(*,'(A,1X,I3,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
        'COMOV dbg', dbg_count, 't_shell=', t_shell, 't_target=', t_loc, 'x_sh=', x_sh_loc
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
        '  r_in=', r_in, 'r_out=', r_out, 'lum_heat=', lum_heat
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
        '  rho_in=', rho_in, 'rho_out=', rho_out, 'tau_in=', tau_in
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
        '  f_ib=', f_ib, 'f_ob=', f_ob, 'lum_obs=', lum_obs
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
        '  v_sh_loc=', v_sh_loc, 'v_ej_max=', v_ej_max, 'w=', w
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
        '  yN=', state%y(n), 'yNm1=', state%y(n-1), 'dy_total=', dy_total
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,L1)') &
        '  y_min=', y_min, 'y_max=', y_max, 'y_bad=', y_bad
   call flush(6)
  end if

  state%r_shell_current = r_sh_loc
  state%u_shell_current = v_sh_loc
  state%m_shocked_csm = max(m_sh_csm, 0d0)
  state%m_shocked_ej = max(m_sh_ej, 0d0)
  state%t_shell = t_loc
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
  real(8) :: tau_lo, tau_hi, frac

  jstart = min(max(istart, 1), state%n_zones)
  iph = state%n_zones
  r_ph = state%r_outer

  if (state%tau(jstart) <= 2d0/3d0) then
   iph = jstart
   if (.not. state%in_cooling_phase) then
    r_ph = min(max(state%r_shell_current, state%r_inner), state%r_outer)
   else
    r_ph = min(max(state%r_inner, state%r_inner), state%r_outer)
   end if
   return
  end if

  do i = jstart + 1, state%n_zones
   if (state%tau(i) <= 2d0/3d0) then
    iph = i - 1
    tau_lo = state%tau(i-1)
    tau_hi = state%tau(i)
    frac = (2d0/3d0 - tau_lo) / max(tau_hi - tau_lo, 1d-30)
    frac = min(max(frac, 0d0), 1d0)
    r_ph = state%radius(i-1) + frac * (state%radius(i) - state%radius(i-1))
    r_ph = min(max(r_ph, state%radius(i-1)), state%radius(i))
    return
   end if
  end do
end subroutine photosphere_boundary_from_tau

real(8) function outer_escape_conductance(state, iph, r_ph) result(k_escape)
  type(transport_state_type), intent(in) :: state
  integer, intent(in) :: iph
  real(8), intent(in) :: r_ph
  real(8) :: r_face_l, r_face_r, vol_i, area_ph

  if (iph < 1 .or. iph > state%n_zones) then
   k_escape = 0d0
   return
  end if

  call zone_geometry(state, iph, r_face_l, r_face_r, vol_i)
  r_face_r = min(max(r_ph, r_face_l), r_face_r)
  area_ph = 4d0 * pi * r_face_r**2
  k_escape = area_ph * clight * a_rad / 4d0
end function outer_escape_conductance

subroutine find_transport_photosphere(state, r_ph, lum_obs)
  type(transport_state_type), intent(in) :: state
  real(8), intent(out) :: r_ph, lum_obs
  integer :: iph, istart
  real(8) :: y_ph, tau_lo, tau_hi, frac, y_obs, r_obs
  real(8) :: t_relax, ramp
  real(8) :: r_face_l, r_face_r, vol_i, dr, rho_face, dcoef, e_n, e_nm1, flux_out
  call photosphere_boundary_from_tau(state, 1, iph, r_ph)
  if (state%in_cooling_phase) then
   if (state%t_shell - state%t_emerge < 0.1d0 * 86400d0) then
    iph = state%n_zones
    r_ph = state%r_outer
   end if
  end if
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
  ! Eddington boundary: F = (c/4) u
  flux_out = 0.25d0 * clight * e_n
  lum_obs = max(4d0 * pi * r_obs**2 * flux_out, 0d0)
  ! No luminosity override during the gap: let the boundary condition set the flux.
end subroutine find_transport_photosphere

real(8) function transport_timestep_limit(state)
  type(transport_state_type), intent(in) :: state
  integer :: i, ihi, iph
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
  call photosphere_boundary_from_tau(state, ihi, iph, r_ph)

  do i = max(ihi,1), max(ihi, iph-1)
   dr = max(state%radius(i+1) - state%radius(i), 1d-30)
   rho_face = 0.5d0 * (state%rho(i) + state%rho(i+1))
   dcoef = clight * a_rad / (3d0 * state%kappa * max(rho_face, 1d-30))
   transport_timestep_limit = min(transport_timestep_limit, dr*dr / max(dcoef, 1d-30))
  end do

  do i = max(ihi,1), iph
   if (.not. state%in_cooling_phase) then
    call interaction_zone_geometry_at(state, i, state%r_shell_current, r_face_l, r_face_r, vol_i)
   else
    call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   end if
   if (i == iph) then
    r_face_r = min(max(r_ph, r_face_l), r_face_r)
    vol_i = 4d0 * pi * max(r_face_r**3 - r_face_l**3, 1d-30) / 3d0
   end if

   if (i == 1 .or. (.not. state%in_cooling_phase .and. i == ihi)) then
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

subroutine solve_transport_step(state, dt, lum_heat)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: dt, lum_heat

  integer :: i, n, ihi, iph, ihi_old, k
  real(8) :: dr_l, dr_r, r_face_l, r_face_r, area_l, area_r, vol_i
  real(8) :: vol_old, r_face_l_old, r_face_l_base, r_face_r_base
  real(8) :: e_carry, r_face_l_prev, r_face_r_prev, vol_residual
  real(8) :: rho_face_l, rho_face_r, dcoef_l, dcoef_r, k_l, k_r, k_escape, r_ph
  real(8) :: theta, omt
  real(8) :: dr_cut, dr_nom, v_merged_new, u_merged_old
  real(8) :: total_vol, t_relax, ramp
  real(8) :: expansion_lambda
  logical :: merge_cut
  logical :: use_gap_flux

  if (.not. state%initialized) return
  n = state%n_zones
  if (n < 2) return

  state%work_old_y = state%y
  theta = 0.5d0
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
   ihi = first_active_zone(state)   ! zone containing the shock (heating source)
  end if
  ihi_old = ihi
  merge_cut = .false.
  call photosphere_boundary_from_tau(state, 1, iph, r_ph)

  expansion_lambda = 0d0
  ! adiab_lhs/rhs: Crank-Nicolson adiabatic P*dV factors.
  ! For homologous expansion H = 1/t, the energy equation is
  ! d(a_rad*y*V)/dt = -(a_rad*y*V)*H + diffusion
  ! => diagonal multiplied by (1 + 0.5*H*dt), RHS by (1 - 0.5*H*dt).
  ! We store 0.5*H*dt in t_relax (reused as scratch) and ramp.
  ramp = 0d0
  if (state%in_cooling_phase .and. state%t_shell > 1d0) then
   t_relax = state%t_shell + 0.5d0 * dt   ! midpoint time
   ramp = 0.5d0 * dt / max(t_relax, 1d-30)  ! = 0.5*H_mid*dt
  end if

  do i = 1, n
   call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   k_escape = 0d0

   if (i > iph) then
    state%work_vol(i) = 0d0
    state%work_vol_old(i) = 0d0
    state%work_kl(i) = 0d0
    state%work_kr(i) = 0d0
    state%work_kesc(i) = 0d0
    cycle
   end if

   if (i == 1) then
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
    ! Cooling phase: vol at old timestep was vol_ref * prev_scale^3.
    ! vol_i is already vol_ref * cooling_scale^3 (new timestep).
    vol_old = vol_i * (state%cooling_scale_prev / max(state%cooling_scale, 1d-30))**3
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

  use_gap_flux = .false.

  total_vol = 0d0
  ! no gap heating in cooling phase (Appendix A)

  do i = 1, n
   if (i > iph) then
    state%work_a(i) = 0d0
    state%work_b(i) = 1d0
    state%work_c(i) = 0d0
    state%work_rhs(i) = state%work_old_y(i)
    cycle
   end if

   ! Apply adiabatic P*dV factors for cooling phase (ramp = 0.5*H*dt = dt/(2*t_mid)).
   ! Multiplying the vol/dt terms by (1 + ramp) on LHS and (1 - ramp) on RHS
   ! correctly discretizes d(a_rad*y*V)/dt = -(a_rad*y*V)/t + diffusion in CN form.
   state%work_b(i) = a_rad * state%work_vol(i) / dt * (1d0 + ramp) + &
                     theta * (state%work_kl(i) + state%work_kr(i) + state%work_kesc(i))
   state%work_rhs(i) = a_rad * state%work_vol_old(i) * state%work_old_y(i) / dt * (1d0 - ramp) - &
            omt * (state%work_kl(i) + state%work_kr(i) + state%work_kesc(i)) * state%work_old_y(i)

   if (i > 1) then
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

  if (.not. state%in_cooling_phase .and. iph >= 1 .and. iph <= n) then
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

  if (merge_cut .and. ihi + 1 <= iph) then
   state%work_a(ihi) = 0d0
   state%work_b(ihi) = 1d0
   state%work_c(ihi) = -1d0
   state%work_rhs(ihi) = 0d0

   v_merged_new = state%work_vol(ihi) + state%work_vol(ihi+1)
   u_merged_old = a_rad * (state%work_vol_old(ihi) * state%work_old_y(ihi) + &
                           state%work_vol_old(ihi+1) * state%work_old_y(ihi+1))

   k_r = state%work_kr(ihi+1)
   k_escape = state%work_kesc(ihi+1)
   state%work_a(ihi+1) = 0d0
   state%work_b(ihi+1) = a_rad * v_merged_new / dt + theta * (k_r + k_escape)
   state%work_c(ihi+1) = 0d0
   state%work_rhs(ihi+1) = u_merged_old / dt - omt * (k_r + k_escape) * state%work_old_y(ihi+1)

   if (ihi + 1 < iph) then
    state%work_c(ihi+1) = -theta * k_r
    state%work_rhs(ihi+1) = state%work_rhs(ihi+1) + omt * k_r * state%work_old_y(ihi+2)
   end if
   if (max(lum_heat, 0d0) > 0d0) then
    state%work_rhs(ihi+1) = state%work_rhs(ihi+1) + max(lum_heat, 0d0)
   end if
  end if

  if (.not. state%in_cooling_phase .and. ihi_old < ihi) then
   e_carry = 0d0
   do k = ihi_old, ihi - 1
    call zone_geometry_from_array(state%radius_ref, state%r_inner_support, state%r_outer_support, state%n_zones, k, &
         r_face_l_prev, r_face_r_prev, vol_i)
    r_face_l_prev = min(max(state%r_shell_prev, r_face_l_prev), r_face_r_prev)
    vol_residual = 4d0 * pi * max(r_face_r_prev**3 - r_face_l_prev**3, 1d-30) / 3d0
    e_carry = e_carry + a_rad * max(state%work_old_y(k), 0d0) * vol_residual
   end do
   if (e_carry > 0d0 .and. ihi <= n) then
    state%work_rhs(ihi) = state%work_rhs(ihi) + e_carry / max(dt, 1d-30)
   end if
  end if

  if (.not. state%in_cooling_phase .and. iph >= 1 .and. iph <= n) then
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
  state%y = max(state%work_sol, 1d-40)
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

integer function first_active_zone(state) result(ihi)
  type(transport_state_type), intent(in) :: state
  ihi = first_active_zone_at(state, state%r_shell_current)
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
  integer :: i, istart
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

end module csm_transport
