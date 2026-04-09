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

  state%y = 1d-40
  state%e_swept = 0d0
  call update_tau_and_luminosity(state)
  write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
       'M3_INIT_GRID rin=', state%r_inner_support, 'rout=', state%r_outer_support, &
       'rho_in=', state%rho_ref(1), 'rho_out=', state%rho_ref(state%n_zones), 'tau0=', state%tau(1), 'kappa=', state%kappa
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

  if (state%r_inner > old_inner) then
   call deposit_energy_interval_to_reference(old_radius, old_inner, old_outer, old_y, state%n_zones, &
        old_inner, state%r_inner, state%radius_ref, state%r_inner_support, state%r_outer_support, state%n_zones, state%e_swept)
  end if
  call remap_y_conservative(old_radius, old_inner, old_outer, old_y, state%n_zones, &
       state%radius, state%r_inner, state%r_outer, state%y, state%n_zones, e_removed)
  state%e_residual = max(sum(state%e_swept), 0d0)

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
  real(8) :: u_grid, r_face_l, r_face_r, vol_i, dUdt, dt_eff
  integer :: i, istart
  real(8), save :: u_prev = 0d0
  logical, save :: have_prev = .false.
  integer, save :: energy_step = 0

  state%u_shell_current = u_shell
  state%lum_heat_last = lum_heat
  if (.not. state%initialized) then
   call initialize_interaction_grid(state, max(r_shell, 1d0), t_shell, m_shell, max(state%kappa, 1d-30), max(state%n_zones,48))
  else
   call update_interaction_grid(state, max(r_shell, 1d0), t_shell, m_shell)
  end if

  ! Shell mass is evolved by the coupled dynamics (m_shell input).  Keep the
  ! transport state's mass synchronized to avoid double-counting swept-up mass.
  state%m_shocked_csm = max(m_shell, 1d-30)
  state%m_shocked_ej = 0d0

  lum_input = max(lum_heat, 0d0)

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
    ! debug print disabled
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
  real(8), allocatable :: old_y(:), e_breakout(:), old_radius_ref(:), old_rho_ref(:), old_dm(:)
  real(8) :: u_grid, tau_shell, r_face_l, r_face_r, vol_i
  real(8) :: delta_r_shell, tau_dummy, tleak_dummy, m_old_total
  logical, parameter :: debug_mode3 = .true.

  if (.not. state%initialized) return

  state%in_cooling_phase = .true.
  state%cooling_initialized = .true.
  state%t_emerge = t_shell
  state%t_gap_end = t_shell
  state%lum_heat_gap = 0d0
  state%r_emerge_shell = max(r_shell, 1d0)
  state%u_emerge_shell = max(u_shell, 1d-30)
  state%m_emerge_csm = max(state%m_shocked_csm, 0d0)
  state%m_emerge_ej = max(state%m_shocked_ej, 0d0)
  state%m_emerge_shell = max(m_shell, state%m_shocked_csm + state%m_shocked_ej, 1d-30)

  allocate(old_y(state%n_zones))
  allocate(e_breakout(state%n_zones))
  allocate(old_radius_ref(state%n_zones))
  allocate(old_rho_ref(state%n_zones))
  allocate(old_dm(state%n_zones))
  old_y = state%y
  old_radius_ref = state%radius_ref
  old_rho_ref = state%rho_ref
  e_breakout = max(state%e_swept, 0d0)

  ! Total stored breakout energy = swept dark-zone reservoir + radiation still
  ! present on the interaction grid at emergence.
  call deposit_energy_interval_to_reference(state%radius, state%r_inner, state%r_outer, old_y, state%n_zones, &
       state%r_inner, state%r_outer, old_radius_ref, state%r_inner_support, state%r_outer_support, state%n_zones, e_breakout)
  u_grid = sum(e_breakout)

  ! Appendix A switches to diffusion in the freely expanding shocked shell.
  ! Compress the swept-up CSM mass profile into a thin post-shock shell at
  ! breakout, then carry the stored breakout-energy profile onto that shell.
  call shell_structure_estimate(state, max(r_shell, 1d0), t_shell, state%m_emerge_shell, delta_r_shell, tau_dummy, tleak_dummy)
  delta_r_shell = min(max(delta_r_shell, 1d-3 * max(r_shell, 1d0)), 0.25d0 * max(r_shell, 1d0))
  state%r_emerge_outer = max(r_shell, 1d0)
  state%r_emerge_inner = max(state%r_emerge_outer - delta_r_shell, 1d0)
  if (.not.(state%r_emerge_outer > state%r_emerge_inner)) then
   state%r_emerge_inner = state%r_emerge_outer * (1d0 - 1d-6)
  end if

  state%cooling_scale = 1d0
  state%cooling_scale_prev = 1d0
  state%compact_tail_drain = 0d0
  state%r_inner = state%r_emerge_inner
  state%r_outer = state%r_emerge_outer

  call build_radial_grid(state%r_emerge_inner, state%r_emerge_outer, state%radius_ref)

  m_old_total = 0d0
  do i = 1, state%n_zones
   call zone_geometry_from_array(old_radius_ref, state%r_inner_support, state%r_outer_support, state%n_zones, i, &
        r_face_l, r_face_r, vol_i)
   old_dm(i) = max(old_rho_ref(i), 1d-30) * vol_i
   m_old_total = m_old_total + old_dm(i)
  end do
  if (m_old_total <= 0d0) m_old_total = 1d0

  do i = 1, state%n_zones
   call zone_geometry_from_array(state%radius_ref, state%r_emerge_inner, state%r_emerge_outer, state%n_zones, i, &
        r_face_l, r_face_r, vol_i)
   state%radius(i) = state%radius_ref(i)
   state%rho_ref(i) = max(state%m_emerge_shell * old_dm(i) / m_old_total / max(vol_i, 1d-30), 1d-30)
   state%rho(i) = state%rho_ref(i)
   state%y(i) = max(e_breakout(i) / max(a_rad * vol_i, 1d-30), 1d-40)
  end do

  call update_tau_and_luminosity(state)
  tau_shell = state%tau(1)
  state%e_swept = 0d0
  state%e_residual = 0d0

  if (debug_mode3) then
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
        'M3_HANDOFF_GEOM rin=', state%r_emerge_inner, 'rsh=', state%r_emerge_shell, 'rout=', state%r_emerge_outer, 'tau0=', tau_shell
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
        'M3_HANDOFF_EN Ugrid=', u_grid, 'Utarget=', u_grid, 'Usum=', u_grid
  end if

  deallocate(e_breakout)
  deallocate(old_y)
  deallocate(old_radius_ref)
  deallocate(old_rho_ref)
  deallocate(old_dm)
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
  next_scale = max((state%r_emerge_inner + state%u_emerge_shell * max(t_shell - state%t_emerge, 0d0)) &
                   / max(state%r_emerge_inner, 1d0), 1d0)
  state%cooling_scale = next_scale

  state%r_inner = state%r_emerge_inner * next_scale
  state%r_outer = state%r_emerge_outer * next_scale
  do i = 1, state%n_zones
   state%radius(i) = state%radius_ref(i) * next_scale
   state%rho(i) = max(state%rho_ref(i) / next_scale**3, 1d-30)
  end do
  state%y = max(state%y * (prev_scale / next_scale)**4, 1d-40)
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
   call initialize_cooling_state_from_interaction(state, state%r_shell_current, &
        max(state%u_shell_current, 1d-30), max(state%m_shocked_csm + state%m_shocked_ej, 1d-30), t_shell, &
        lum_heat=state%lum_heat_last)
  end if

  lum_heat_local = 0d0
  if (present(lum_heat)) lum_heat_local = max(lum_heat, 0d0)
  call update_cooling_grid(state, t_shell)
  call solve_transport_step(state, dt, lum_heat_local)
  call find_transport_photosphere(state, r_ph, lum_obs)
  ! debug print disabled
end subroutine cooling_transport_step

subroutine comoving_transport_step(state, dt, t_shell, r_sh, v_sh, m_sh, lum_heat, lum_obs)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: dt, t_shell, r_sh, v_sh, m_sh, lum_heat
  real(8), intent(out) :: lum_obs

  integer :: nsub, isub
  integer :: nz_init
  real(8) :: t_loc, t_next, dt_step, dt_move, dt_rad, subratio
  real(8) :: r_loc, u_loc, m_loc
  real(8) :: ku, kr, km
  real(8) :: r_out, r_ph, lum_tmp
  real(8) :: heat_sub
  real(8) :: e_rad
  real(8) :: du_res, e_prev, e_res_prev, ebal
  real(8) :: r_init, kappa_init
  integer :: dbg_stride, call_count
  logical :: do_log_step, do_log_sub
  integer, save :: mode3_call_count = 0
  logical, parameter :: debug_mode3 = .true.

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
  m_loc = max(m_sh, state%m_shocked_csm + state%m_shocked_ej, 1d-30)

  dt_move = shock_motion_timestep_limit_at(state, r_loc, u_loc)
  dt_rad = transport_timestep_limit(state)
  dt_step = t_next - t_loc
  if (dt_move > 0d0 .and. dt_move < huge(1d0)) dt_step = min(dt_step, 0d0 + dt_move)
  if (dt_rad > 0d0 .and. dt_rad < huge(1d0)) dt_step = min(dt_step, 0.5d0*dt_rad)
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
   do_log_sub = (isub == 1 .or. isub == nsub .or. mod(isub, dbg_stride) == 0)
   e_prev = total_radiation_energy(state)
   if (.not. state%in_cooling_phase) then
    if (debug_mode3 .and. do_log_step .and. do_log_sub) then
     write(*,'(A,1X,I0,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
          'M3_INT_PRE isub=', isub, 't=', t_loc, 'r=', r_loc, 'u=', u_loc, 'm=', m_loc
    end if

    e_res_prev = state%e_residual
    ku = dudt(u_loc, r_loc, m_loc, t_loc, op)
    kr = drdt(u_loc, r_loc, m_loc, t_loc, op)
    km = dmdt(u_loc, r_loc, m_loc, t_loc, op)

    u_loc = max(u_loc + ku*dt_step, 1d5)
    r_loc = max(r_loc + kr*dt_step, 1d0)
    m_loc = max(m_loc + km*dt_step, 1d-30)

    heat_sub = max(lum_heat, 0d0)
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
    if (.not. state%in_cooling_phase .and. (r_loc >= r_out .or. shock_has_emerged(r_loc, t_loc + dt_step))) then
     if (debug_mode3) then
      write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5)') &
           'M3_EMERGE t=', t_loc+dt_step, 'r=', r_loc, 'rout=', r_out, 'L=', lum_tmp
      write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5)') 'M3_EMERGE_URES=', state%e_residual, 'Erad=', e_rad
     end if
     call initialize_cooling_state_from_interaction(state, r_loc, u_loc, m_loc, t_loc + dt_step, lum_tmp, heat_sub)
    end if
   else
    call cooling_transport_step(state, dt_step, t_loc + dt_step, lum_tmp, r_ph, 0d0)
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
  end do

  state%r_shell_current = r_loc
  state%u_shell_current = max(u_loc, 0d0)
  state%m_shocked_csm = max(m_loc, 1d-30)
  state%m_shocked_ej = 0d0
  state%t_shell = t_loc
  if (debug_mode3 .and. do_log_step) then
   write(*,'(A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,ES12.5,1X,A,1X,L1)') &
        'M3_EXIT t=', state%t_shell, 'L=', lum_obs, 'r=', state%r_shell_current, 'cool=', state%in_cooling_phase
  end if
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
   ! Once the optical depth ahead of the shock/shell drops below 2/3, the
   ! diffusion domain is effectively optically thin. The emitting surface is
   ! then the outer edge rather than the inner moving boundary.
   iph = state%n_zones
   r_ph = state%r_outer
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
  real(8) :: r_face_l, r_face_r, vol_i, area_ph, stream_factor

  if (iph < 1 .or. iph > state%n_zones) then
   k_escape = 0d0
   return
  end if

  call zone_geometry(state, iph, r_face_l, r_face_r, vol_i)
  r_face_r = min(max(r_ph, r_face_l), r_face_r)
  area_ph = 4d0 * pi * r_face_r**2
  if (state%in_cooling_phase) then
   ! Appendix A changes the cooling-phase outer boundary to
   ! e = f_ob(y) * de/dx with f_ob = -2/(tau_in eta) (Rin/R0)^2,
   ! which corresponds to F = (c/2) u at the emitting surface.
   stream_factor = 0.5d0
  else
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
   ! Cooling phase uses the Appendix A outer boundary, which gives F=(c/2)u.
   flux_out = 0.5d0 * clight * e_n
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

subroutine solve_transport_step(state, dt, lum_heat)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: dt, lum_heat

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
   state%e_residual = max(sum(state%e_swept), 0d0)
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
    state%work_rhs(i) = 1d-40
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
    state%work_rhs(i) = 1d-40
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

end module csm_transport
