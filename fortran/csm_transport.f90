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
  real(8) :: e_shocked_store = 0d0
  real(8) :: m_shocked_csm = 0d0
  real(8) :: m_shocked_ej = 0d0
  real(8) :: t_emerge = 0d0
  real(8) :: r_inner = 0d0
 real(8) :: r_outer = 0d0
 real(8) :: r_outer_support = 0d0
 real(8) :: r_emerge_inner = 0d0
 real(8) :: r_emerge_outer = 0d0
 real(8) :: r_emerge_shell = 0d0
 real(8) :: u_emerge_shell = 0d0
 real(8) :: cooling_scale = 1d0
  real(8) :: m_emerge_csm = 0d0
  real(8) :: m_emerge_ej = 0d0
  real(8) :: m_emerge_shell = 0d0
  real(8), allocatable :: radius(:)
  real(8), allocatable :: radius_ref(:)
  real(8), allocatable :: rho(:)
  real(8), allocatable :: rho_ref(:)
  real(8), allocatable :: y(:)
  real(8), allocatable :: tau(:)
  real(8), allocatable :: lum(:)
 end type transport_state_type

 public :: transport_state_type, reset_transport_state, &
           interaction_transport_step, find_transport_photosphere, &
           shock_has_emerged, initialize_cooling_state_from_interaction, &
           cooling_transport_step, transport_timestep_limit, shock_motion_timestep_limit, &
           forward_shock_radius

contains

 subroutine reset_transport_state(state)
  type(transport_state_type), intent(inout) :: state

  if (allocated(state%radius)) deallocate(state%radius)
  if (allocated(state%radius_ref)) deallocate(state%radius_ref)
  if (allocated(state%rho)) deallocate(state%rho)
  if (allocated(state%rho_ref)) deallocate(state%rho_ref)
  if (allocated(state%y)) deallocate(state%y)
  if (allocated(state%tau)) deallocate(state%tau)
  if (allocated(state%lum)) deallocate(state%lum)

  state%initialized = .false.
  state%in_cooling_phase = .false.
  state%n_zones = 0
  state%kappa = 0d0
  state%t_shell = 0d0
  state%r_shell_current = 0d0
  state%e_shocked_store = 0d0
  state%m_shocked_csm = 0d0
  state%m_shocked_ej = 0d0
  state%t_emerge = 0d0
  state%r_inner = 0d0
  state%r_outer = 0d0
  state%r_outer_support = 0d0
  state%r_emerge_inner = 0d0
  state%r_emerge_outer = 0d0
  state%r_emerge_shell = 0d0
  state%u_emerge_shell = 0d0
  state%cooling_scale = 1d0
  state%m_emerge_csm = 0d0
  state%m_emerge_ej = 0d0
  state%m_emerge_shell = 0d0
 end subroutine reset_transport_state

 subroutine initialize_interaction_grid(state, r_shell, t_shell, m_shell, kappa, n_zones)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: r_shell, t_shell, m_shell, kappa
  integer, intent(in) :: n_zones

  call reset_transport_state(state)
  state%n_zones = max(8, n_zones)
  state%kappa = max(kappa, 1d-30)
  state%initialized = .true.
  state%t_shell = t_shell

  call update_interaction_geometry(state, r_shell, t_shell, m_shell)
  call fill_profile_arrays(state, t_shell)

  ! Cold start: transport field should brighten only through injected shock power.
 state%y = 1d-40
  call update_tau_and_luminosity(state)
 end subroutine initialize_interaction_grid

 integer function first_active_zone_at(state, r_shell) result(ihi)
  type(transport_state_type), intent(in) :: state
  real(8), intent(in) :: r_shell
  integer :: i
  real(8) :: r_face_l, r_face_r, vol_i

  ihi = state%n_zones
  if (r_shell <= state%r_inner) then
   ihi = 1
   return
  end if

  do i = 1, state%n_zones
   call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   if (r_shell < r_face_r) then
    ihi = i
    return
   end if
  end do
 end function first_active_zone_at

 subroutine interaction_zone_geometry_at(state, i, r_shell, r_face_l, r_face_r, vol_i)
  type(transport_state_type), intent(in) :: state
  integer, intent(in) :: i
  real(8), intent(in) :: r_shell
  real(8), intent(out) :: r_face_l, r_face_r, vol_i
  integer :: ihi

  call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
  ihi = first_active_zone_at(state, r_shell)

  if (i < ihi) then
   r_face_r = r_face_l
   vol_i = 0d0
  else if (i == ihi) then
   r_face_l = max(r_face_l, min(r_shell, r_face_r))
   vol_i = 4d0 * pi * max(r_face_r**3 - r_face_l**3, 1d-30) / 3d0
  end if
 end subroutine interaction_zone_geometry_at

 subroutine update_interaction_geometry(state, r_shell, t_shell, m_shell)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: r_shell, t_shell, m_shell

  integer :: i, istart
  real(8) :: x

  state%t_shell = t_shell
  state%r_shell_current = max(r_shell, 1d0)
  state%r_inner = max(query_csm_inner_edge(t_shell, op(2)), 1d0)
  state%r_outer_support = query_csm_outer_edge(t_shell, op(2))
  if (.not.(state%r_outer_support > state%r_inner)) then
   state%r_outer_support = state%r_inner * (1d0 + 1d-3)
  end if

  state%r_outer = max(state%r_outer_support, state%r_inner * (1d0 + 1d-3))

  if (.not. allocated(state%radius)) then
   allocate(state%radius(state%n_zones), state%radius_ref(state%n_zones), &
            state%rho(state%n_zones), state%rho_ref(state%n_zones), state%y(state%n_zones), &
            state%tau(state%n_zones), state%lum(state%n_zones))
  end if

  do i = 1, state%n_zones
   x = dble(i-1) / dble(max(state%n_zones-1, 1))
   if (state%r_outer / state%r_inner > 1.05d0) then
    state%radius(i) = state%r_inner * (state%r_outer / state%r_inner)**x
   else
    state%radius(i) = state%r_inner + (state%r_outer - state%r_inner) * x
   end if
  end do
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

 subroutine fill_profile_arrays(state, t_shell)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: t_shell
  integer :: i, istart

  do i = 1, state%n_zones
   state%rho(i) = max(query_csm_density(state%radius(i), t_shell, op(2)), 1d-30)
  end do
 end subroutine fill_profile_arrays

 subroutine update_interaction_grid(state, r_shell, t_shell, m_shell)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: r_shell, t_shell, m_shell
  integer :: i
  real(8) :: old_shell, new_shell
  real(8) :: r_face_l_old, r_face_r_old, vol_old
  real(8) :: r_face_l_new, r_face_r_new, vol_new
  real(8) :: e_removed

  if (.not. state%initialized) return
  ! During the interaction phase the paper treats the unshocked CSM as
  ! effectively stationary. Keep the diffusion grid and density profile fixed
  ! after initialization; only the moving shock/source position evolves.
  old_shell = max(state%r_shell_current, state%r_inner)
  new_shell = max(r_shell, state%r_inner)
  e_removed = 0d0

  if (new_shell > old_shell) then
   do i = 1, state%n_zones
    call interaction_zone_geometry_at(state, i, old_shell, r_face_l_old, r_face_r_old, vol_old)
    call interaction_zone_geometry_at(state, i, new_shell, r_face_l_new, r_face_r_new, vol_new)
    if (vol_old > vol_new) then
     e_removed = e_removed + a_rad * max(state%y(i), 1d-40) * (vol_old - vol_new)
     if (vol_new <= 1d-30) state%y(i) = 1d-40
    end if
   end do
  end if

  state%e_shocked_store = state%e_shocked_store + max(e_removed, 0d0)
  state%t_shell = t_shell
  state%r_shell_current = max(new_shell, 1d0)
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
  real(8) :: delta_r, tau_sh, tleak, delta_r_csm

  call shell_structure_estimate(state, r_shell, t_shell, m_shell, delta_r, tau_sh, tleak, &
       delta_r_csm_out=delta_r_csm)
  r_fs = max(r_shell, 1d0) + max(delta_r_csm, 0d0)
end function forward_shock_radius

real(8) function shell_leakage_timescale(state, r_shell, t_shell, m_shell) result(tleak)
  type(transport_state_type), intent(in) :: state
  real(8), intent(in) :: r_shell, t_shell, m_shell
  real(8) :: delta_r, tau_sh

  call shell_structure_estimate(state, r_shell, t_shell, m_shell, delta_r, tau_sh, tleak)
end function shell_leakage_timescale

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
  real(8) :: tleak, e_release, lum_input

  if (.not. state%initialized) then
   call initialize_interaction_grid(state, r_shell, t_shell, m_shell, max(state%kappa, 1d-30), max(state%n_zones,48))
  else
   call update_interaction_grid(state, r_shell, t_shell, m_shell)
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
  if (state%e_shocked_store > 0d0) then
   tleak = interaction_shell_leakage_timescale(state, r_shell, t_shell, max(m_shell, 1d-30))
   tleak = max(tleak, 1d-30)
   e_release = state%e_shocked_store * (1d0 - exp(-dt / tleak))
   e_release = min(max(e_release, 0d0), state%e_shocked_store)
   state%e_shocked_store = state%e_shocked_store - e_release
   lum_input = lum_input + e_release / max(dt, 1d-30)
  end if

  call solve_transport_step(state, dt, lum_input)
  call find_transport_photosphere(state, r_ph, lum_obs)
 end subroutine interaction_transport_step

 subroutine deposit_shell_store(state, e_add)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: e_add
  integer :: i
  real(8) :: r_face_l, r_face_r, vol_i, w_i, w_sum

  if (e_add <= 0d0) return
  if (.not. state%initialized) return

  w_sum = 0d0
  do i = 1, state%n_zones
   call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   w_i = max(state%rho(i), 1d-30) * vol_i
   w_sum = w_sum + w_i
  end do
  if (w_sum <= 0d0) return

  do i = 1, state%n_zones
   call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   w_i = max(state%rho(i), 1d-30) * vol_i
   state%y(i) = max(state%y(i) + e_add * w_i / max(a_rad * vol_i * w_sum, 1d-30), 1d-40)
  end do
 end subroutine deposit_shell_store

subroutine initialize_cooling_state_from_interaction(state, r_shell, u_shell, m_shell, t_shell)
 type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: r_shell, u_shell, m_shell, t_shell
  integer :: i, istart, nprof, j, n_active
  real(8) :: e_total, e_now, renorm, span_old, span_new, x_new
  real(8) :: r_face_l, r_face_r, vol_i, delta_r, tau_sh, tleak
  real(8) :: shell_vol, rho_shell, r_old_inner, r_old_outer
  real(8), allocatable :: old_x(:), old_y(:), old_r_active(:), old_y_active(:)

  if (.not. state%initialized) return

  state%in_cooling_phase = .true.
  state%t_emerge = t_shell
  state%r_emerge_shell = max(r_shell, 1d0)
  state%u_emerge_shell = max(u_shell, 1d-30)
  state%m_emerge_csm = max(state%m_shocked_csm, 0d0)
  state%m_emerge_ej = max(state%m_shocked_ej, 0d0)
  state%m_emerge_shell = max(m_shell, 1d-30)
  istart = first_active_zone_at(state, r_shell)
  n_active = max(1, state%n_zones - istart + 1)
  allocate(old_r_active(n_active), old_y_active(n_active))
  do i = 1, n_active
   old_r_active(i) = state%radius(istart + i - 1)
   old_y_active(i) = max(state%y(istart + i - 1), 1d-40)
  end do
  e_total = max(state%e_shocked_store, 0d0)
  do i = istart, state%n_zones
   call interaction_zone_geometry_at(state, i, r_shell, r_face_l, r_face_r, vol_i)
   e_total = e_total + a_rad * max(state%y(i), 1d-40) * vol_i
  end do

  call shell_structure_estimate(state, r_shell, t_shell, state%m_emerge_shell, delta_r, tau_sh, tleak, &
       state%m_emerge_csm, state%m_emerge_ej)
  state%r_emerge_outer = max(state%r_outer_support, max(r_shell, 1d0))
  state%r_emerge_inner = max(state%r_emerge_outer - delta_r, 1d0)
  if (state%r_emerge_outer <= state%r_emerge_inner) then
   state%r_emerge_inner = max(0.99d0 * state%r_emerge_outer, 1d0)
  end if
  state%cooling_scale = 1d0
  if (.not. allocated(state%radius_ref)) allocate(state%radius_ref(state%n_zones))
  if (.not. allocated(state%rho_ref)) allocate(state%rho_ref(state%n_zones))

  r_old_inner = max(r_shell, state%r_inner)
  r_old_outer = state%r_outer_support
  span_old = max(r_old_outer - r_old_inner, 1d-30)
  span_new = max(state%r_emerge_outer - state%r_emerge_inner, 1d-30)

  do i = 1, state%n_zones
   if (state%r_emerge_outer / state%r_emerge_inner > 1.05d0) then
    state%radius(i) = state%r_emerge_inner * (state%r_emerge_outer / state%r_emerge_inner) ** &
         (dble(i-1) / dble(max(state%n_zones-1, 1)))
   else
    state%radius(i) = state%r_emerge_inner + span_new * dble(i-1) / dble(max(state%n_zones-1, 1))
   end if
  end do

  shell_vol = 4d0 * pi * max(state%r_emerge_outer**3 - state%r_emerge_inner**3, 1d-30) / 3d0
  rho_shell = state%m_emerge_shell / max(shell_vol, 1d-30)
  state%rho = max(rho_shell, 1d-30)

  nprof = max(2, n_active + 2)
  allocate(old_x(nprof), old_y(nprof))
  old_x(1) = 0d0
  old_y(1) = old_y_active(1)
  j = 2
  do i = 1, n_active
   old_x(j) = min(max((old_r_active(i) - r_old_inner) / span_old, 0d0), 1d0)
   if (j > 2) old_x(j) = max(old_x(j), old_x(j-1))
   old_y(j) = old_y_active(i)
   j = j + 1
  end do
  old_x(j) = 1d0
  old_y(j) = old_y_active(n_active)

  do i = 1, state%n_zones
   x_new = min(max((state%radius(i) - state%r_emerge_inner) / span_new, 0d0), 1d0)
   state%y(i) = max(interp_linear_monotonic(x_new, old_x, old_y, j), 1d-40)
  end do

  e_now = 0d0
  do i = 1, state%n_zones
   call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   e_now = e_now + a_rad * max(state%y(i), 1d-40) * vol_i
  end do
  if (e_now > 0d0 .and. e_total > 0d0) then
   renorm = e_total / e_now
   state%y = max(state%y * renorm, 1d-40)
  end if

  state%radius_ref = state%radius
  state%rho_ref = max(state%rho, 1d-30)
  state%e_shocked_store = 0d0
  deallocate(old_x, old_y, old_r_active, old_y_active)
  call update_tau_and_luminosity(state)
end subroutine initialize_cooling_state_from_interaction

subroutine initialize_cooling_csm_from_total_energy(state, t_shell)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: t_shell
  integer :: i, istart
  real(8), allocatable :: old_r(:), old_y(:)
  real(8) :: e_total
  real(8) :: r_face_l, r_face_r, vol_i, x, weight_sum
  real(8) :: y_norm, w_i, e_now, renorm
  real(8) :: old_inner, old_outer

  allocate(old_r(state%n_zones), old_y(state%n_zones))
  old_r = state%radius
  old_y = state%y
  old_inner = state%r_inner
  old_outer = state%r_outer

  e_total = state%e_shocked_store
  do i = 1, state%n_zones
   call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   e_total = e_total + a_rad * state%y(i) * vol_i
  end do

  state%r_emerge_inner = max(query_csm_inner_edge(t_shell, op(2)), 1d0)
  state%r_emerge_outer = max(query_csm_outer_edge(t_shell, op(2)), state%r_emerge_inner*(1d0 + 1d-6))
  state%r_inner = state%r_emerge_inner
  state%r_outer = state%r_emerge_outer
  state%cooling_scale = 1d0

  do i = 1, state%n_zones
   x = dble(i-1) / dble(max(state%n_zones-1, 1))
   if (state%r_outer / state%r_inner > 1.05d0) then
    state%radius(i) = state%r_inner * (state%r_outer / state%r_inner)**x
   else
    state%radius(i) = state%r_inner + (state%r_outer - state%r_inner) * x
   end if
   state%rho(i) = max(query_csm_density(state%radius(i), t_shell, op(2)), 1d-30)
  end do

  state%radius_ref = state%radius
  state%rho_ref = state%rho

  weight_sum = 0d0
  do i = 1, state%n_zones
   call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   w_i = state%rho(i) * max((state%radius(i) / max(state%r_outer, 1d0))**4, 1d-12)
   weight_sum = weight_sum + w_i * vol_i
  end do
  if (weight_sum > 0d0) then
   y_norm = e_total / max(a_rad * weight_sum, 1d-30)
   do i = 1, state%n_zones
    w_i = state%rho(i) * max((state%radius(i) / max(state%r_outer, 1d0))**4, 1d-12)
    state%y(i) = max(y_norm * w_i, 1d-40)
   end do
  else
   state%y = 1d-40
  end if

  do i = 1, state%n_zones
   if (state%radius(i) >= old_inner .and. state%radius(i) <= old_outer) then
    state%y(i) = max(state%y(i), interp_linear_monotonic(state%radius(i), old_r, old_y, state%n_zones))
   end if
  end do

  e_now = 0d0
  do i = 1, state%n_zones
   call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   e_now = e_now + a_rad * state%y(i) * vol_i
  end do
  if (e_now > 0d0) then
   renorm = e_total / e_now
   do i = 1, state%n_zones
    state%y(i) = max(state%y(i) * renorm, 1d-40)
   end do
  end if

  state%e_shocked_store = 0d0
  call update_tau_and_luminosity(state)
  deallocate(old_r, old_y)
end subroutine initialize_cooling_csm_from_total_energy

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

subroutine cooling_transport_step(state, dt, t_shell, lum_obs, r_ph)
  type(transport_state_type), intent(inout) :: state
  real(8), intent(in) :: dt, t_shell
  real(8), intent(out) :: lum_obs, r_ph
  real(8) :: tleak, e_release, r_shell_now

  if (.not. state%initialized) then
   lum_obs = 0d0
   r_ph = 0d0
   return
  end if

  call update_cooling_grid(state, t_shell)
  if (state%e_shocked_store > 0d0) then
   r_shell_now = state%r_emerge_shell * max(state%cooling_scale, 1d0)
   tleak = shell_leakage_timescale(state, r_shell_now, t_shell, state%m_emerge_shell)
   tleak = max(tleak, 1d-30)
   e_release = state%e_shocked_store * (1d0 - exp(-dt / tleak))
   e_release = min(max(e_release, 0d0), state%e_shocked_store)
   call deposit_shell_store(state, e_release)
   state%e_shocked_store = state%e_shocked_store - e_release
  end if
  call solve_transport_step(state, dt, 0d0)
  call find_transport_photosphere(state, r_ph, lum_obs)
 end subroutine cooling_transport_step

 logical function shock_has_emerged(r_shell, t_shell)
  real(8), intent(in) :: r_shell, t_shell
  real(8) :: r_out

  r_out = query_csm_outer_edge(t_shell, op(2))
  shock_has_emerged = (r_out < huge(1d0) .and. r_shell >= r_out)
 end function shock_has_emerged

subroutine find_transport_photosphere(state, r_ph, lum_obs)
  type(transport_state_type), intent(in) :: state
  real(8), intent(out) :: r_ph, lum_obs
  integer :: i, iph, istart
  real(8) :: tau_lo, tau_hi, frac, lum_lo, lum_hi

  r_ph = state%r_outer
  iph = state%n_zones
  istart = 1
  if (.not. state%in_cooling_phase) istart = first_active_zone(state)

  if (state%tau(istart) <= 2d0/3d0) then
   r_ph = state%r_outer
   lum_obs = max(state%lum(state%n_zones), 0d0)
   return
  end if
  do i = istart, state%n_zones
   if (state%tau(i) <= 2d0/3d0) then
    iph = i
    if (i == istart) then
     r_ph = state%radius(i)
     lum_obs = max(state%lum(i), 0d0)
    else
     tau_lo = state%tau(i-1)
     tau_hi = state%tau(i)
     frac = (2d0/3d0 - tau_lo) / max(tau_hi - tau_lo, 1d-30)
     frac = min(max(frac, 0d0), 1d0)
     r_ph = state%radius(i-1) + frac * (state%radius(i) - state%radius(i-1))
     lum_lo = state%lum(max(i-1,1))
     lum_hi = state%lum(i)
     lum_obs = max(lum_lo + frac * (lum_hi - lum_lo), 0d0)
    end if
    exit
   end if
  end do
  lum_obs = max(lum_obs, 0d0)
end subroutine find_transport_photosphere

real(8) function transport_timestep_limit(state)
  type(transport_state_type), intent(in) :: state
  integer :: i, ihi
  real(8) :: dr, rho_face, dcoef
  real(8) :: r_face_l, r_face_r, vol_i, area_l, area_r
  real(8) :: rho_face_l, rho_face_r, dcoef_l, dcoef_r, k_l, k_r, k_escape
  real(8) :: tau_edge, tau_escape, t_relax

  transport_timestep_limit = huge(1d0)
  if (.not. state%initialized) return
  if (state%n_zones < 2) then
   transport_timestep_limit = huge(1d0)
   return
  end if
  ihi = 1
  if (.not. state%in_cooling_phase) ihi = first_active_zone(state)

  do i = max(ihi,1), state%n_zones-1
   dr = max(state%radius(i+1) - state%radius(i), 1d-30)
   rho_face = 0.5d0 * (state%rho(i) + state%rho(i+1))
   dcoef = clight * a_rad / (3d0 * state%kappa * max(rho_face, 1d-30))
   transport_timestep_limit = min(transport_timestep_limit, dr*dr / max(dcoef, 1d-30))
  end do

  do i = max(ihi,1), state%n_zones
   if (.not. state%in_cooling_phase) then
    call interaction_zone_geometry_at(state, i, state%r_shell_current, r_face_l, r_face_r, vol_i)
   else
    call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   end if

   if (i == 1 .or. (.not. state%in_cooling_phase .and. i == ihi)) then
    k_l = 0d0
   else
    area_l = 4d0 * pi * r_face_l**2
    rho_face_l = 0.5d0 * (state%rho(i-1) + state%rho(i))
    dcoef_l = clight * a_rad / (3d0 * state%kappa * max(rho_face_l, 1d-30))
    k_l = area_l * dcoef_l / max(state%radius(i) - state%radius(i-1), 1d-30)
   end if

   if (i == state%n_zones) then
    area_r = 4d0 * pi * r_face_r**2
    tau_edge = state%kappa * state%rho(i) * max(r_face_r - r_face_l, 0d0)
    tau_escape = max(2d0/3d0, tau_edge)
    k_r = 0d0
    k_escape = area_r * clight * a_rad / max(2d0 + 3d0*tau_escape, 1d0)
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

 integer :: i, n, ish, ilo, ihi
  real(8), allocatable :: a(:), b(:), c(:), rhs(:), sol(:), old_y(:), kl_arr(:), kr_arr(:), kesc_arr(:), vol_arr(:)
  real(8) :: dr_l, dr_r, r_face_l, r_face_r, area_l, area_r, vol_i
  real(8) :: rho_face_l, rho_face_r, dcoef_l, dcoef_r, k_l, k_r, k_escape, tau_edge, tau_escape
  real(8) :: theta, omt

  if (.not. state%initialized) return
  n = state%n_zones
  if (n < 2) return

  allocate(a(n), b(n), c(n), rhs(n), sol(n), old_y(n), kl_arr(n), kr_arr(n), kesc_arr(n), vol_arr(n))
  old_y = state%y
  theta = 0.55d0
  omt = 1d0 - theta
  a = 0d0
  b = 0d0
  c = 0d0
  rhs = 0d0
  kl_arr = 0d0
  kr_arr = 0d0
  kesc_arr = 0d0
  vol_arr = 0d0
  ish = locate_shock_zone(state)
  ilo = ish
  ihi = ish
  if (.not. state%in_cooling_phase) then
   ihi = first_active_zone(state)
  end if

  do i = 1, n
   if (.not. state%in_cooling_phase) then
    call interaction_zone_geometry_at(state, i, state%r_shell_current, r_face_l, r_face_r, vol_i)
   else
    call zone_geometry(state, i, r_face_l, r_face_r, vol_i)
   end if
   k_escape = 0d0

   if (.not. state%in_cooling_phase .and. i < ihi) then
    vol_arr(i) = 0d0
    kl_arr(i) = 0d0
    kr_arr(i) = 0d0
    kesc_arr(i) = 0d0
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

   if (i == n) then
    area_r = 4d0 * pi * r_face_r**2
    k_r = 0d0
    tau_edge = state%kappa * state%rho(i) * max(r_face_r - r_face_l, 0d0)
    tau_escape = max(2d0/3d0, tau_edge)
    k_escape = area_r * clight * a_rad / max(2d0 + 3d0*tau_escape, 1d0)
   else
    dr_r = state%radius(i+1) - state%radius(i)
    area_r = 4d0 * pi * r_face_r**2
    rho_face_r = 0.5d0 * (state%rho(i) + state%rho(i+1))
    dcoef_r = clight * a_rad / (3d0 * state%kappa * max(rho_face_r, 1d-30))
    k_r = area_r * dcoef_r / max(dr_r, 1d-30)
   end if

   vol_arr(i) = vol_i
   kl_arr(i) = k_l
   kr_arr(i) = k_r
   kesc_arr(i) = k_escape
  end do

  do i = 1, n
   if (.not. state%in_cooling_phase .and. i < ihi) then
    a(i) = 0d0
    b(i) = 1d0
    c(i) = 0d0
    rhs(i) = 1d-40
    cycle
   end if

   b(i) = a_rad * vol_arr(i) / dt + theta * (kl_arr(i) + kr_arr(i) + kesc_arr(i))
   rhs(i) = a_rad * vol_arr(i) * old_y(i) / dt - &
            omt * (kl_arr(i) + kr_arr(i) + kesc_arr(i)) * old_y(i)

   if (i > 1 .and. (.not. state%in_cooling_phase .and. i > ihi .or. state%in_cooling_phase)) then
    a(i) = -theta * kl_arr(i)
    rhs(i) = rhs(i) + omt * kl_arr(i) * old_y(i-1)
   end if
   if (i < n) then
    c(i) = -theta * kr_arr(i)
    rhs(i) = rhs(i) + omt * kr_arr(i) * old_y(i+1)
   end if

   if (.not. state%in_cooling_phase .and. i == ihi) then
    ! Paper-consistent interaction source: impose shock heating as the flux
    ! injected through the moving inner boundary of the active unshocked CSM.
    rhs(i) = rhs(i) + max(lum_heat, 0d0)
   end if
  end do

  call tridag(a, b, c, rhs, sol, n)
  state%y = max(sol, 1d-40)
  call update_tau_and_luminosity(state)

  deallocate(a, b, c, rhs, sol, old_y, kl_arr, kr_arr, kesc_arr, vol_arr)
end subroutine solve_transport_step

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
  real(8) :: dr, rho_avg, r_face_l, r_face_r, vol_i, tau_edge, tau_escape

  istart = 1
  if (.not. state%in_cooling_phase) istart = first_active_zone(state)

  state%tau(state%n_zones) = 0d0
  do i = state%n_zones-1, istart, -1
   dr = state%radius(i+1) - state%radius(i)
   rho_avg = 0.5d0 * (state%rho(i) + state%rho(i+1))
   state%tau(i) = state%tau(i+1) + state%kappa * rho_avg * max(dr, 0d0)
  end do
  if (istart > 1) state%tau(1:istart-1) = state%tau(istart)

  state%lum = 0d0
  do i = istart, state%n_zones-1
   state%lum(i) = -4d0*pi*(0.5d0*(state%radius(i)+state%radius(i+1)))**2 * &
                  (clight*a_rad/(3d0*state%kappa*max(0.5d0*(state%rho(i)+state%rho(i+1)),1d-30))) * &
                  (state%y(i+1)-state%y(i))/max(state%radius(i+1)-state%radius(i),1d-30)
  end do
  call zone_geometry(state, state%n_zones, r_face_l, r_face_r, vol_i)
  tau_edge = state%kappa * state%rho(state%n_zones) * max(r_face_r - r_face_l, 0d0)
  tau_escape = max(2d0/3d0, tau_edge)
  state%lum(state%n_zones) = 4d0*pi*state%r_outer**2 * clight * a_rad * state%y(state%n_zones) &
                           / max(2d0 + 3d0*tau_escape, 1d0)
end subroutine update_tau_and_luminosity

 subroutine tridag(a, b, c, r, u, n)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n), b(n), c(n), r(n)
  real(8), intent(out) :: u(n)
  real(8), allocatable :: gam(:)
  real(8) :: bet
  integer :: j

  allocate(gam(n))
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
  deallocate(gam)
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
