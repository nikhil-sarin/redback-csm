module csm_transport

 use constants, only: pi, clight
 use physical_constants, only: a_rad
 use get_vals, only: op, query_csm_density, query_csm_inner_edge, query_csm_outer_edge, &
                     query_tau_to_edge, query_csm_photosphere_radius, query_csm_velocity, &
                     query_ejecta_density, query_ejecta_velocity

 implicit none

 type transport_state_type
  logical :: initialized = .false.
 logical :: in_cooling_phase = .false.
  integer :: n_zones = 0
  real(8) :: kappa = 0d0
  real(8) :: t_shell = 0d0
  real(8) :: r_shell_current = 0d0
  real(8) :: m_shocked_csm = 0d0
  real(8) :: m_shocked_ej = 0d0
  real(8) :: e_residual = 0d0
  real(8) :: t_emerge = 0d0
  real(8) :: r_inner_support = 0d0
  real(8) :: r_inner = 0d0
 real(8) :: r_outer = 0d0
 real(8) :: r_outer_support = 0d0
 real(8) :: r_emerge_inner = 0d0
 real(8) :: r_emerge_outer = 0d0
 real(8) :: r_emerge_shell = 0d0
 real(8) :: u_emerge_shell = 0d0
 real(8) :: cooling_scale = 1d0
 real(8) :: compact_tail_drain = 0d0
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
  real(8), allocatable :: work_gam(:)
 end type transport_state_type

 public :: transport_state_type, reset_transport_state, &
           interaction_transport_step, find_transport_photosphere, &
           shock_has_emerged, initialize_cooling_state_from_interaction, &
           cooling_transport_step, transport_timestep_limit, shock_motion_timestep_limit, &
           forward_shock_radius, shell_leakage_timescale, total_radiation_energy

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
  if (allocated(state%work_gam)) deallocate(state%work_gam)

  state%initialized = .false.
  state%in_cooling_phase = .false.
  state%n_zones = 0
  state%kappa = 0d0
  state%t_shell = 0d0
  state%r_shell_current = 0d0
  state%m_shocked_csm = 0d0
  state%m_shocked_ej = 0d0
  state%e_residual = 0d0
  state%t_emerge = 0d0
  state%r_inner_support = 0d0
  state%r_inner = 0d0
  state%r_outer = 0d0
  state%r_outer_support = 0d0
  state%r_emerge_inner = 0d0
  state%r_emerge_outer = 0d0
  state%r_emerge_shell = 0d0
  state%u_emerge_shell = 0d0
  state%cooling_scale = 1d0
  state%compact_tail_drain = 0d0
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
  integer :: i, j
  real(8) :: src_l, src_rface, src_vol, ref_l, ref_rface, ref_vol
  real(8) :: rhoe_src, rlo, rhi, overlap_vol

  if (cut_r <= cut_l) return

  do j = 1, n_src
   call zone_geometry_from_array(src_r, src_inner, src_outer, n_src, j, src_l, src_rface, src_vol)
   rlo = max(src_l, cut_l)
   rhi = min(src_rface, cut_r)
   if (rhi <= rlo) cycle
   rhoe_src = a_rad * max(src_y(j), 0d0)

   do i = 1, n_ref
    call zone_geometry_from_array(ref_r, ref_inner, ref_outer, n_ref, i, ref_l, ref_rface, ref_vol)
    overlap_vol = 4d0 * pi * max(min(rhi, ref_rface)**3 - max(rlo, ref_l)**3, 0d0) / 3d0
    if (overlap_vol > 0d0) e_ref(i) = e_ref(i) + rhoe_src * overlap_vol
   end do
 end do
 end subroutine deposit_energy_interval_to_reference

 subroutine deposit_scalar_energy_to_reference(cut_l, cut_r, ref_r, ref_inner, ref_outer, rho_ref, n_ref, e_dep, dE)
  integer, intent(in) :: n_ref
  real(8), intent(in) :: cut_l, cut_r
  real(8), intent(in) :: ref_r(n_ref), ref_inner, ref_outer, rho_ref(n_ref)
  real(8), intent(inout) :: e_dep(n_ref)
  real(8), intent(in) :: dE
  integer :: i, iclose
  real(8) :: ref_l, ref_rface, ref_vol, overlap_vol, weight, weight_sum, rmid, dmin
  real(8), allocatable :: weights(:)

  if (dE <= 0d0) return
  allocate(weights(n_ref))
  weights = 0d0
  weight_sum = 0d0

  do i = 1, n_ref
   call zone_geometry_from_array(ref_r, ref_inner, ref_outer, n_ref, i, ref_l, ref_rface, ref_vol)
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
           state%work_vol(state%n_zones), state%work_gam(state%n_zones))

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
  ihi = 1
 end function first_active_zone_at

 subroutine interaction_zone_geometry_at(state, i, r_shell, r_face_l, r_face_r, vol_i)
  type(transport_state_type), intent(in) :: state
  integer, intent(in) :: i
  real(8), intent(in) :: r_shell
  real(8), intent(out) :: r_face_l, r_face_r, vol_i

  call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
 end subroutine interaction_zone_geometry_at

subroutine update_interaction_geometry(state, r_shell, t_shell, m_shell)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: r_shell, t_shell, m_shell

  state%t_shell = t_shell
  state%r_shell_current = max(r_shell, 1d0)
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
  integer :: i, j
  real(8), allocatable :: e_new(:)
  real(8) :: old_l, old_rface, old_vol, new_l, new_rface, new_vol
  real(8) :: rhoe_old, rlo, rhi, overlap_vol

  allocate(e_new(n_new))
  e_new = 0d0
  e_removed = 0d0

  do j = 1, n_old
   call zone_geometry_from_array(old_r, old_inner, old_outer, n_old, j, old_l, old_rface, old_vol)
   rhoe_old = a_rad * max(old_y(j), 0d0)

   if (old_l < new_inner) then
    rlo = old_l
    rhi = min(old_rface, new_inner)
    if (rhi > rlo) then
     e_removed = e_removed + rhoe_old * 4d0 * pi * (rhi**3 - rlo**3) / 3d0
    end if
   end if

   do i = 1, n_new
    call zone_geometry_from_array(new_r, new_inner, new_outer, n_new, i, new_l, new_rface, new_vol)
    rlo = max(old_l, new_l)
    rhi = min(old_rface, new_rface)
    if (rhi > rlo) then
     overlap_vol = 4d0 * pi * (rhi**3 - rlo**3) / 3d0
     e_new(i) = e_new(i) + rhoe_old * overlap_vol
    end if
   end do
  end do

  do i = 1, n_new
   call zone_geometry_from_array(new_r, new_inner, new_outer, n_new, i, new_l, new_rface, new_vol)
   new_y(i) = max(e_new(i) / max(a_rad * new_vol, 1d-30), 1d-40)
  end do

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

  r_fs = forward_shock_radius(state, r_shell, t_shell, m_shell)
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

  call solve_transport_step(state, dt, lum_input)
  call find_transport_photosphere(state, r_ph, lum_obs)
 end subroutine interaction_transport_step

subroutine initialize_cooling_state_from_interaction(state, r_shell, u_shell, m_shell, t_shell, lum_target)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: r_shell, u_shell, m_shell, t_shell
  real(8), intent(in), optional :: lum_target
  integer :: i, ismooth
  real(8), allocatable :: e_active(:), e_work(:), e_remap(:)
  real(8) :: r_face_l, r_face_r, vol_full, mass_shape, rho_norm
  real(8) :: e_total_scalar, e_dep_total, target_mass, cum_mass, cell_mass, r_dep_outer, r3
  real(8) :: src_inner, src_outer, blend_weight, peak_fraction, taper_strength, xfrac, compactness
  if (.not. state%initialized) return

  state%in_cooling_phase = .true.
  state%t_emerge = t_shell
  state%r_emerge_shell = max(r_shell, 1d0)
  state%u_emerge_shell = max(u_shell, 1d-30)
  state%m_emerge_csm = max(state%m_shocked_csm, 0d0)
  state%m_emerge_ej = max(state%m_shocked_ej, 0d0)
  state%m_emerge_shell = max(m_shell, 1d-30)
  state%r_emerge_inner = state%r_inner_support
  state%r_emerge_outer = state%r_outer_support
  if (state%r_emerge_outer <= state%r_emerge_inner) then
   state%r_emerge_outer = max(state%r_emerge_inner * (1d0 + 1d-6), state%r_outer)
  end if
  state%cooling_scale = 1d0
  state%compact_tail_drain = 0d0

  allocate(e_active(state%n_zones), e_work(state%n_zones), e_remap(state%n_zones))
  e_active = 0d0
  e_work = 0d0
  e_remap = 0d0

  src_inner = state%r_inner
  src_outer = state%r_outer
  state%r_inner = state%r_emerge_inner
  state%r_outer = state%r_emerge_outer
  mass_shape = 0d0
  do i = 1, state%n_zones
   call zone_geometry_from_array(state%radius_ref, state%r_inner_support, state%r_outer_support, state%n_zones, i, &
        r_face_l, r_face_r, vol_full)
   mass_shape = mass_shape + max(state%rho_ref(i), 1d-30) * vol_full
  end do
  if (mass_shape > 0d0) then
   rho_norm = state%m_emerge_shell / mass_shape
  else
   rho_norm = 1d0
  end if
  do i = 1, state%n_zones
   state%radius(i) = state%radius_ref(i)
   state%rho_ref(i) = max(state%rho_ref(i) * rho_norm, 1d-30)
   state%rho(i) = state%rho_ref(i)
  end do

  e_total_scalar = 0d0
  do i = 1, state%n_zones
   call interaction_zone_geometry_at(state, i, state%r_shell_current, r_face_l, r_face_r, vol_full)
   e_total_scalar = e_total_scalar + a_rad * max(state%y(i), 0d0) * vol_full
  end do

  target_mass = min(max(state%m_emerge_csm, 0d0), state%m_emerge_shell)
  r_dep_outer = min(max(r_shell, state%r_emerge_inner), state%r_emerge_outer)
  if (target_mass > 0d0) then
   cum_mass = 0d0
   r_dep_outer = state%r_emerge_inner
   do i = 1, state%n_zones
    call zone_geometry_from_array(state%radius_ref, state%r_inner_support, state%r_outer_support, state%n_zones, i, &
         r_face_l, r_face_r, vol_full)
    cell_mass = max(state%rho_ref(i), 1d-30) * vol_full
    if (cum_mass + cell_mass >= target_mass) then
     r3 = r_face_l**3 + 3d0 * (target_mass - cum_mass) / &
          (4d0 * pi * max(state%rho_ref(i), 1d-30))
     r_dep_outer = min(max(r3, r_face_l**3), r_face_r**3) ** (1d0 / 3d0)
     exit
    end if
    cum_mass = cum_mass + cell_mass
    r_dep_outer = r_face_r
   end do
  end if
  r_dep_outer = min(max(r_dep_outer, state%r_emerge_inner * (1d0 + 1d-6)), state%r_emerge_outer)
  compactness = 1d0 - (r_dep_outer - state%r_emerge_inner) / &
       max(state%r_emerge_outer - state%r_emerge_inner, state%r_emerge_inner * 1d-6)
  compactness = min(1d0, max(0d0, compactness))
  e_active = 0d0
  call deposit_energy_interval_to_reference(state%radius, src_inner, src_outer, state%y, state%n_zones, &
       state%r_emerge_inner, r_dep_outer, state%radius_ref, state%r_inner_support, state%r_outer_support, &
       state%n_zones, e_remap)
  e_dep_total = sum(e_remap)
  if (e_dep_total > 0d0 .and. e_total_scalar > 0d0) then
   e_remap = e_remap * (e_total_scalar / e_dep_total)
  else if (e_total_scalar > 0d0) then
   call deposit_scalar_energy_to_reference(state%r_emerge_inner, r_dep_outer, state%radius_ref, &
        state%r_inner_support, state%r_outer_support, state%rho_ref, state%n_zones, e_remap, e_total_scalar)
  end if

  if (sum(max(state%e_swept, 0d0)) > 0d0) then
   e_active = max(state%e_swept, 0d0)
   do ismooth = 1, 8
    e_work = e_active
    e_active(1) = 0.75d0 * e_work(1) + 0.25d0 * e_work(min(2, state%n_zones))
    do i = 2, state%n_zones - 1
     e_active(i) = 0.25d0 * e_work(i-1) + 0.5d0 * e_work(i) + 0.25d0 * e_work(i+1)
    end do
    if (state%n_zones > 1) then
     e_active(state%n_zones) = 0.25d0 * e_work(state%n_zones-1) + 0.75d0 * e_work(state%n_zones)
    end if
   end do
   e_dep_total = sum(e_active)
   if (e_dep_total > 0d0 .and. e_total_scalar > 0d0) then
    e_active = e_active * (e_total_scalar / e_dep_total)
   end if
   if (sum(e_active) > 0d0 .and. sum(e_remap) > 0d0) then
    peak_fraction = maxval(e_active) / max(sum(e_active), 1d-30)
    blend_weight = min(0.65d0, max(0.10d0, 0.45d0 - 4.0d0 * max(peak_fraction - 0.08d0, 0d0)))
    e_active = blend_weight * e_active + (1d0 - blend_weight) * e_remap
    taper_strength = min(0.55d0, max(0d0, 6.0d0 * (peak_fraction - 0.11d0)))
    if (taper_strength > 0d0 .and. state%n_zones > 1) then
     do i = 1, state%n_zones
      xfrac = dble(i - 1) / dble(state%n_zones - 1)
      e_active(i) = e_active(i) * max(0.15d0, 1d0 - taper_strength * xfrac**2)
     end do
    end if
    state%compact_tail_drain = max(min(1.0d0, max(0d0, 6.0d0 * (peak_fraction - 0.10d0))), &
         min(1.0d0, max(0d0, 2.5d0 * (compactness - 0.55d0))))
   else if (sum(e_remap) > 0d0) then
    e_active = e_remap
    state%compact_tail_drain = 0d0
   end if
   e_dep_total = sum(e_active)
   if (e_dep_total > 0d0 .and. e_total_scalar > 0d0) then
    e_active = e_active * (e_total_scalar / e_dep_total)
   end if
  else
   e_active = e_remap
   state%compact_tail_drain = 0d0
  end if

  do i = 1, state%n_zones
   call zone_geometry_from_array(state%radius_ref, state%r_inner_support, state%r_outer_support, state%n_zones, i, &
        r_face_l, r_face_r, vol_full)
   state%y(i) = max(e_active(i) / max(a_rad * vol_full, 1d-30), 1d-40)
  end do
  state%e_swept = 0d0
  state%e_residual = 0d0
  deallocate(e_active, e_work, e_remap)

  call update_tau_and_luminosity(state)
end subroutine initialize_cooling_state_from_interaction

subroutine update_cooling_grid(state, t_shell)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: t_shell
  real(8) :: prev_scale, next_scale
  integer :: i

  if (.not. state%initialized) return

  state%t_shell = t_shell
  prev_scale = max(state%cooling_scale, 1d-30)
  next_scale = max((state%r_emerge_shell + state%u_emerge_shell * max(t_shell - state%t_emerge, 0d0)) &
                   / max(state%r_emerge_shell, 1d0), 1d0)
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

  lum_heat_local = 0d0
  if (present(lum_heat)) lum_heat_local = max(lum_heat, 0d0)
  call update_cooling_grid(state, t_shell)
  call solve_transport_step(state, dt, lum_heat_local)
  call find_transport_photosphere(state, r_ph, lum_obs)
end subroutine cooling_transport_step

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
   r_ph = min(max(state%radius(jstart), state%r_inner), state%r_outer)
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
  real(8) :: y_ph
  istart = 1
  if (.not. state%in_cooling_phase) istart = first_active_zone(state)
  if (state%tau(istart) <= 2d0/3d0) then
   r_ph = state%r_outer
   y_ph = max(state%y(state%n_zones), 1d-40)
   lum_obs = max(pi * clight * r_ph**2 * a_rad * y_ph, 0d0)
   return
  end if
  call photosphere_boundary_from_tau(state, istart, iph, r_ph)
  y_ph = max(state%y(iph), 1d-40)
  if (iph < state%n_zones .and. state%radius(iph+1) > state%radius(iph)) then
   y_ph = max(interp_linear_monotonic(r_ph, state%radius, state%y, state%n_zones), 1d-40)
  end if
  lum_obs = max(pi * clight * r_ph**2 * a_rad * y_ph, 0d0)
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
  real(8) :: dr_local

  shock_motion_timestep_limit = huge(1d0)
  if (.not. state%initialized) return
  if (state%in_cooling_phase) return
  if (state%n_zones < 2) return
  if (abs(u_shell) <= 1d-30) return

  ihi = first_active_zone(state)

  if (ihi == 1) then
   dr_local = max(state%radius(1) - state%r_inner, 1d-30)
  else
   dr_local = max(state%radius(ihi) - state%radius(ihi-1), 1d-30)
  end if
  shock_motion_timestep_limit = 0.5d0 * dr_local / abs(u_shell)
 end function shock_motion_timestep_limit

subroutine solve_transport_step(state, dt, lum_heat)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: dt, lum_heat

 integer :: i, n, ihi, iph, ish_lo, ish_hi
  real(8) :: dr_l, dr_r, r_face_l, r_face_r, area_l, area_r, vol_i
  real(8) :: rho_face_l, rho_face_r, dcoef_l, dcoef_r, k_l, k_r, k_escape, r_ph
  real(8) :: theta, omt, src_w_lo, src_w_hi, k_tail, xfrac, t_drain

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
  ihi = 1
  if (.not. state%in_cooling_phase) then
   ihi = first_active_zone(state)
  end if
  call locate_source_cells(state, ish_lo, ish_hi, src_w_lo, src_w_hi)
  call photosphere_boundary_from_tau(state, ihi, iph, r_ph)

  do i = 1, n
   if (.not. state%in_cooling_phase) then
    call interaction_zone_geometry_at(state, i, state%r_shell_current, r_face_l, r_face_r, vol_i)
   else
    call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   end if
   k_escape = 0d0

   if ((.not. state%in_cooling_phase .and. i < ihi) .or. i > iph) then
    state%work_vol(i) = 0d0
    state%work_kl(i) = 0d0
    state%work_kr(i) = 0d0
    state%work_kesc(i) = 0d0
    cycle
   end if

   if (i == 1 .or. (.not. state%in_cooling_phase .and. i == ihi)) then
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

   if (state%in_cooling_phase .and. state%compact_tail_drain > 0d0 .and. n > 1) then
    xfrac = dble(i - 1) / dble(n - 1)
    t_drain = max(0.5d0 * state%r_emerge_shell / max(state%u_emerge_shell, 1d-30), 1d0)
    k_tail = state%compact_tail_drain * xfrac**2 * a_rad * vol_i / t_drain
    k_escape = k_escape + k_tail
   end if

   state%work_vol(i) = vol_i
   state%work_kl(i) = k_l
   state%work_kr(i) = k_r
   state%work_kesc(i) = k_escape
  end do

  do i = 1, n
   if ((.not. state%in_cooling_phase .and. i < ihi) .or. i > iph) then
    state%work_a(i) = 0d0
    state%work_b(i) = 1d0
    state%work_c(i) = 0d0
    state%work_rhs(i) = state%work_old_y(i)
    cycle
   end if

   state%work_b(i) = a_rad * state%work_vol(i) / dt + theta * (state%work_kl(i) + state%work_kr(i) + state%work_kesc(i))
   state%work_rhs(i) = a_rad * state%work_vol(i) * state%work_old_y(i) / dt - &
            omt * (state%work_kl(i) + state%work_kr(i) + state%work_kesc(i)) * state%work_old_y(i)

   if (i > 1 .and. (.not. state%in_cooling_phase .and. i > ihi .or. state%in_cooling_phase)) then
    state%work_a(i) = -theta * state%work_kl(i)
    state%work_rhs(i) = state%work_rhs(i) + omt * state%work_kl(i) * state%work_old_y(i-1)
   end if
   if (i < iph) then
    state%work_c(i) = -theta * state%work_kr(i)
    state%work_rhs(i) = state%work_rhs(i) + omt * state%work_kr(i) * state%work_old_y(i+1)
   end if

   if (max(lum_heat, 0d0) > 0d0) then
    if (.not. state%in_cooling_phase) then
     if (i == ish_lo) then
      state%work_rhs(i) = state%work_rhs(i) + max(lum_heat, 0d0) * src_w_lo
      state%e_swept(i) = state%e_swept(i) + dt * max(lum_heat, 0d0) * src_w_lo
     end if
     if (i == ish_hi) then
      state%work_rhs(i) = state%work_rhs(i) + max(lum_heat, 0d0) * src_w_hi
      state%e_swept(i) = state%e_swept(i) + dt * max(lum_heat, 0d0) * src_w_hi
     end if
    else
     if (i == 1) state%work_rhs(i) = state%work_rhs(i) + max(lum_heat, 0d0)
    end if
   end if
  end do

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
  real(8) :: dr, rho_avg

  istart = 1
  if (.not. state%in_cooling_phase) istart = first_active_zone(state)

  state%tau(state%n_zones) = 0d0
  do i = state%n_zones-1, istart, -1
   dr = state%radius(i+1) - state%radius(i)
   rho_avg = 0.5d0 * (state%rho(i) + state%rho(i+1))
   state%tau(i) = state%tau(i+1) + state%kappa * rho_avg * max(dr, 0d0)
  end do
  if (istart > 1) state%tau(1:istart-1) = state%tau(istart)
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
