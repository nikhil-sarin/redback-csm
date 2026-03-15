module diffusion_runtime

 use constants, only: pi, clight, intpol
 use get_vals, only: op, get_tauprep_explosion, rho4pir2_out

 implicit none

 integer, parameter, public :: ndiff = 48

 real(8), public :: kappa_global = 0d0, eff_global = 1d0, erad = 0d0
 integer, public :: diffusion_type = 0
 logical, public :: diffusion_enabled = .false.
 logical, public :: paper_mode = .false.

 ! Swept-up shell reservoir for post-CSM-exit diffusion
 real(8), public :: e_shell = 0d0, m_swept = 0d0
 real(8), public :: r_shell_exit = 0d0, v_shell_exit = 0d0, t_shell_exit = 0d0
 real(8), public :: v_csm_global = 0d0
 logical, public :: shell_exit_set = .false.

 real(8), allocatable, dimension(:), public :: tauprep_global
 real(8), allocatable, dimension(:) :: wind_tau_cache
 real(8) :: wind_tau_cache_t = -1d300
 logical :: wind_tau_cache_ready = .false.
 real(8), dimension(0:ndiff), public :: diff_r = 0d0, diff_e = 0d0
 logical, public :: diffusion_state_ready = .false.

 private :: prepare_wind_tau_cache, wind_interval_integral
 private :: estimate_outer_radius, diffusion_total_energy
 private :: csm_density, remap_diffusion_state, build_diffusion_grid
 private :: solve_tridiagonal

contains

 subroutine set_runtime_mode(mode)
  integer, intent(in) :: mode

  paper_mode = (mode /= 0)
 end subroutine set_runtime_mode

 subroutine configure_runtime(csm_type, eff, kappa)
  integer, intent(in) :: csm_type
  real(8), intent(in), optional :: eff, kappa

  if (present(eff)) then
   eff_global = eff
  else
   eff_global = 1d0
  end if

  if (allocated(tauprep_global)) deallocate(tauprep_global)
  if (allocated(wind_tau_cache)) deallocate(wind_tau_cache)
  diffusion_enabled = .false.
  diffusion_type = 0
  kappa_global = 0d0
  wind_tau_cache_ready = .false.
  wind_tau_cache_t = -1d300

  if (.not. paper_mode .and. present(kappa)) then
   diffusion_enabled = .true.
   diffusion_type = csm_type
   kappa_global = kappa
   select case (diffusion_type)
   case (1)
    call get_tauprep_explosion(tauprep_global)
   case (2)
    allocate(wind_tau_cache(size(op(2)%t_grid)))
   end select
  end if

  ! Store CSM velocity for shell expansion timescale
  v_csm_global = op(2)%vwind

  call reset_diffusion_state
 end subroutine configure_runtime

 function shell_optical_depth(r_shell, t_shell) result(tau)
  real(8), intent(in) :: r_shell, t_shell
  real(8) :: tau, tp, t_wind
  real(8) :: mdj, mdjp, tj, tjp
  integer :: j, n

  tau = 0d0
  if (.not. diffusion_enabled) return

  select case (diffusion_type)
  case (1)
   tp = t_shell + op(2)%delay
   n = size(op(2)%v_grid)
   if (r_shell >= op(2)%v_grid(n) * tp) then
    tau = 0d0
   elseif (r_shell <= op(2)%v_grid(1) * tp) then
    tau = tauprep_global(1) / tp**2
   else
    j = max(1, min(abs(op(2)%scan_i), n - 1))
    do while (j < n - 1 .and. r_shell >= op(2)%v_grid(j + 1) * tp)
     j = j + 1
    end do
    do while (j > 1 .and. r_shell < op(2)%v_grid(j) * tp)
     j = j - 1
    end do
    op(2)%scan_i = j
    tau = intpol(r_shell / tp, op(2)%v_grid(j:j + 1), tauprep_global(j:j + 1)) / tp**2
   end if

  case (2)
   n = size(op(2)%t_grid)
   t_wind = abs(r_shell / op(2)%vwind - t_shell)
   if (n == 1) then
    tau = op(2)%mdot(1) / (4d0 * pi * op(2)%vwind * r_shell)
   else
    call prepare_wind_tau_cache(t_shell)
    if (t_wind <= op(2)%t_grid(1)) then
     j = 1
    elseif (t_wind >= op(2)%t_grid(n)) then
     j = n
    else
     j = max(1, min(abs(op(2)%scan_i), n - 1))
     do while (j < n - 1 .and. t_wind >= op(2)%t_grid(j + 1))
      j = j + 1
     end do
     do while (j > 1 .and. t_wind < op(2)%t_grid(j))
      j = j - 1
     end do
     op(2)%scan_i = j
    end if

    if (j == n) then
      tau = op(2)%mdot(n) * op(2)%vwind / max(r_shell, 1d-30)
    else
     if (j == 1 .and. r_shell / op(2)%vwind - t_shell < op(2)%t_grid(1)) then
      tj = op(2)%t_grid(1)
      mdj = op(2)%mdot(1)
      tau = wind_tau_cache(1) + mdj * (1d0 / (t_wind + t_shell) - 1d0 / (tj + t_shell))
     else
      tj = op(2)%t_grid(j)
      tjp = op(2)%t_grid(j + 1)
      mdj = op(2)%mdot(j)
      mdjp = op(2)%mdot(j + 1)
      tau = wind_tau_cache(j + 1) + wind_interval_integral(t_wind, tjp, t_shell, mdj, mdjp, tj, tjp)
     end if
     tau = tau / (4d0 * pi * op(2)%vwind**2)
    end if
   end if

  case (3)
   tp = t_shell + op(2)%delay
   if (r_shell / tp >= op(2)%bpl_vt) then
    tau = op(2)%bpl_rho0 * op(2)%bpl_vt / (4d0 * pi * (op(2)%bpl_n - 1d0) * tp**2) &
         * (tp * op(2)%bpl_vt / r_shell)**(op(2)%bpl_n - 1d0)
   else
    tau = op(2)%bpl_rho0 * op(2)%bpl_vt / (4d0 * pi * tp**2) &
         * ((1d0 - (r_shell / (tp * op(2)%bpl_vt))**(1d0 - op(2)%bpl_d)) / (1d0 - op(2)%bpl_d) &
         + 1d0 / (op(2)%bpl_n - 1d0))
   end if

  case (4)
   tp = t_shell + op(2)%delay
   tau = op(2)%Mej / (8d0 * pi * (op(2)%exp_v0 * tp)**2) &
        * exp(-r_shell / (op(2)%exp_v0 * tp))
  end select

  tau = max(kappa_global * tau, 0d0)
 end function shell_optical_depth

 subroutine prepare_wind_tau_cache(t_shell)
  real(8), intent(in) :: t_shell
  integer :: n, k

  if (.not. allocated(wind_tau_cache)) return
  if (wind_tau_cache_ready .and. abs(t_shell - wind_tau_cache_t) <= 1d-12 * max(1d0, abs(t_shell))) return

  n = size(op(2)%t_grid)
  if (n < 1) return

  wind_tau_cache(n) = op(2)%mdot(n) / (op(2)%t_grid(n) + t_shell)
  do k = n - 1, 1, -1
   wind_tau_cache(k) = wind_tau_cache(k + 1) &
                     + wind_interval_integral(op(2)%t_grid(k), op(2)%t_grid(k + 1), t_shell, &
                                              op(2)%mdot(k), op(2)%mdot(k + 1), &
                                              op(2)%t_grid(k), op(2)%t_grid(k + 1))
  end do

  wind_tau_cache_t = t_shell
  wind_tau_cache_ready = .true.
 end subroutine prepare_wind_tau_cache

 function wind_interval_integral(t_lo, t_hi, t_shell, md_lo, md_hi, t_grid_lo, t_grid_hi) result(val)
  real(8), intent(in) :: t_lo, t_hi, t_shell, md_lo, md_hi, t_grid_lo, t_grid_hi
  real(8) :: val, slope, intercept, x_lo, x_hi

  if (t_hi <= t_lo) then
   val = 0d0
   return
  end if

  if (abs(t_grid_hi - t_grid_lo) <= 1d-99) then
   val = 0d0
   return
  end if

  slope = (md_hi - md_lo) / (t_grid_hi - t_grid_lo)
  intercept = md_lo - slope * t_grid_lo
  x_lo = t_lo + t_shell
  x_hi = t_hi + t_shell
  val = slope * log(x_hi / x_lo) + (intercept - slope * t_shell) * (1d0 / x_lo - 1d0 / x_hi)
 end function wind_interval_integral

 function photosphere_radius(r_shell, t_shell, tau_shell) result(r_ph)
  real(8), intent(in) :: r_shell, t_shell, tau_shell
  real(8) :: r_ph, lo, hi, mid, tau_mid, tau_hi, r_outer
  integer :: iter, scan_save

  if (.not. diffusion_enabled .or. tau_shell <= 2d0 / 3d0) then
   r_ph = r_shell
   return
  end if

  scan_save = op(2)%scan_i
  lo = r_shell
  r_outer = estimate_outer_radius(t_shell)
  hi = max(r_shell * (1d0 + 1d-6), r_outer)
  tau_hi = shell_optical_depth(hi, t_shell)

  if (diffusion_type /= 2) then
   do while (tau_hi > 2d0 / 3d0 .and. hi < 1d6 * max(r_shell, 1d0))
    hi = 2d0 * hi
    tau_hi = shell_optical_depth(hi, t_shell)
   end do
  end if

  if (tau_hi > 2d0 / 3d0) then
   r_ph = hi
   op(2)%scan_i = scan_save
   return
  end if

  do iter = 1, 60
   mid = 0.5d0 * (lo + hi)
   tau_mid = shell_optical_depth(mid, t_shell)
   if (tau_mid > 2d0 / 3d0) then
    lo = mid
   else
    hi = mid
   end if
  end do

  r_ph = 0.5d0 * (lo + hi)
  op(2)%scan_i = scan_save
 end function photosphere_radius

 function estimate_outer_radius(t_shell) result(r_outer)
  real(8), intent(in) :: t_shell
  real(8) :: r_outer, tp

  select case (diffusion_type)
  case (1)
   tp = t_shell + op(2)%delay
   r_outer = max(op(2)%v_grid(size(op(2)%v_grid)) * tp, 1d0)
  case (2)
   r_outer = max(op(2)%vwind * (t_shell + op(2)%t_grid(size(op(2)%t_grid))), 1d0)
  case (3)
   tp = t_shell + op(2)%delay
   r_outer = max(10d0 * op(2)%bpl_vt * tp, 1d0)
  case (4)
   tp = t_shell + op(2)%delay
   r_outer = max(20d0 * op(2)%exp_v0 * tp, 1d0)
  case default
   r_outer = 1d0
  end select
 end function estimate_outer_radius

 subroutine reset_diffusion_state
  diff_r = 0d0
  diff_e = 0d0
  diffusion_state_ready = .false.
  erad = 0d0
  e_shell = 0d0
  m_swept = 0d0
  r_shell_exit = 0d0
  v_shell_exit = 0d0
  t_shell_exit = 0d0
  v_csm_global = 0d0
  shell_exit_set = .false.
 end subroutine reset_diffusion_state

 subroutine build_diffusion_grid(r_shell, r_ph, r_grid)
  real(8), intent(in) :: r_shell, r_ph
  real(8), intent(out) :: r_grid(0:ndiff)
  integer :: i

  do i = 0, ndiff
   r_grid(i) = r_shell + (r_ph - r_shell) * real(i, 8) / real(ndiff, 8)
  end do
 end subroutine build_diffusion_grid

 function csm_density(r_eval, t_shell) result(rho)
  real(8), intent(in) :: r_eval, t_shell
  real(8) :: rho

  rho = max(rho4pir2_out(r_eval, t_shell, op(2)) / (4d0 * pi * max(r_eval, 1d-30)**2), 0d0)
 end function csm_density

 subroutine remap_diffusion_state(r_grid_new, e_grid_new)
  real(8), intent(in) :: r_grid_new(0:ndiff)
  real(8), intent(out) :: e_grid_new(0:ndiff)
  integer :: i, j
  real(8) :: x0, x1, w

  e_grid_new = 0d0
  if (.not. diffusion_state_ready) return

  do i = 0, ndiff - 1
   if (r_grid_new(i) <= diff_r(0) .or. r_grid_new(i) >= diff_r(ndiff)) cycle
   do j = 0, ndiff - 1
    x0 = diff_r(j)
    x1 = diff_r(j + 1)
    if (r_grid_new(i) >= x0 .and. r_grid_new(i) <= x1) then
     if (x1 - x0 <= 1d-30) then
      e_grid_new(i) = diff_e(j)
     else
      w = (r_grid_new(i) - x0) / (x1 - x0)
      e_grid_new(i) = (1d0 - w) * diff_e(j) + w * diff_e(j + 1)
     end if
     exit
    end if
   end do
  end do
 end subroutine remap_diffusion_state

 subroutine solve_tridiagonal(n, a, b, c, rhs, x)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n), b(n), c(n), rhs(n)
  real(8), intent(out) :: x(n)
  real(8) :: cprime(n), dprime(n), denom
  integer :: i

  cprime(1) = c(1) / b(1)
  dprime(1) = rhs(1) / b(1)
  do i = 2, n
   denom = b(i) - a(i) * cprime(i - 1)
   cprime(i) = c(i) / denom
   dprime(i) = (rhs(i) - a(i) * dprime(i - 1)) / denom
  end do

  x(n) = dprime(n)
  do i = n - 1, 1, -1
   x(i) = dprime(i) - cprime(i) * x(i + 1)
  end do
 end subroutine solve_tridiagonal

 function diffusion_total_energy() result(e_tot)
  real(8) :: e_tot, dr_seg
  integer :: i

  e_tot = 0d0
  if (.not. diffusion_state_ready) return

  do i = 0, ndiff - 1
   dr_seg = diff_r(i + 1) - diff_r(i)
   e_tot = e_tot + 2d0 * pi * dr_seg &
        * (diff_r(i)**2 * diff_e(i) + diff_r(i + 1)**2 * diff_e(i + 1))
  end do
 end function diffusion_total_energy

 subroutine solve_diffusion_step(dt, r_shell, r_ph, t_shell, lum_heat, lum_obs)
  real(8), intent(in) :: dt, r_shell, r_ph, t_shell, lum_heat
  real(8), intent(out) :: lum_obs
  real(8) :: r_grid(0:ndiff), e_old(0:ndiff)
  real(8) :: a(ndiff - 1), b(ndiff - 1), c(ndiff - 1), rhs(ndiff - 1), sol(ndiff - 1)
  real(8) :: dr, f_in, ri, rim, rip, rhoip, rhoim, dip, dim
  real(8) :: aip, aim, source, d_in, d_out
  real(8) :: escape_time, e_store, leak_frac, e_prev, lum_cap, lum_raw, lum_scale
  integer :: i

  e_prev = diffusion_total_energy()

  if (r_ph <= r_shell * (1d0 + 1d-12)) then
   e_store = e_prev
   if (e_store <= 0d0 .or. .not. diffusion_state_ready) then
    lum_obs = max(lum_heat, 0d0)
    erad = 0d0
   else
    escape_time = max(5d0 * dt, max(r_shell, 1d0) / clight)
    leak_frac = 1d0 - exp(-dt / escape_time)
    lum_obs = lum_heat + e_store * leak_frac / max(dt, 1d-30)
    diff_e = diff_e * max(1d0 - leak_frac, 0d0)
    erad = diffusion_total_energy()
    if (erad <= 1d-60) then
     diff_r = 0d0
     diff_e = 0d0
     diffusion_state_ready = .false.
     erad = 0d0
    end if
   end if
   return
  end if

  call build_diffusion_grid(r_shell, r_ph, r_grid)
  call remap_diffusion_state(r_grid, e_old)
  dr = (r_ph - r_shell) / real(ndiff, 8)
  f_in = lum_heat / (4d0 * pi * max(r_shell, 1d-30)**2)

  a = 0d0
  b = 0d0
  c = 0d0
  rhs = 0d0

  do i = 1, ndiff - 1
   ri = r_grid(i)
   rip = 0.5d0 * (r_grid(i) + r_grid(i + 1))
   rhoip = csm_density(rip, t_shell)
   dip = clight / (3d0 * kappa_global * max(rhoip, 1d-60))
   aip = dip * rip**2 / (max(ri, 1d-30)**2 * dr**2)

   if (i == 1) then
    source = (r_shell**2 * f_in) / (max(ri, 1d-30)**2 * dr)
    b(i) = 1d0 + dt * aip
    c(i) = -dt * aip
    rhs(i) = e_old(i) + dt * source
   else
    rim = 0.5d0 * (r_grid(i - 1) + r_grid(i))
    rhoim = csm_density(rim, t_shell)
    dim = clight / (3d0 * kappa_global * max(rhoim, 1d-60))
    aim = dim * rim**2 / (max(ri, 1d-30)**2 * dr**2)
    a(i) = -dt * aim
    b(i) = 1d0 + dt * (aim + aip)
    rhs(i) = e_old(i)
    if (i < ndiff - 1) c(i) = -dt * aip
   end if
  end do

  call solve_tridiagonal(ndiff - 1, a, b, c, rhs, sol)

  diff_r = r_grid
  diff_e = 0d0
  diff_e(1:ndiff - 1) = max(sol, 0d0)
  d_in = clight / (3d0 * kappa_global * max(csm_density(0.5d0 * (r_grid(0) + r_grid(1)), t_shell), 1d-60))
  diff_e(0) = diff_e(1) + dr * f_in / max(d_in, 1d-30)
  d_out = clight / (3d0 * kappa_global * max(csm_density(0.5d0 * (r_grid(ndiff - 1) + r_grid(ndiff)), t_shell), 1d-60))
  lum_raw = max(4d0 * pi * r_ph**2 * d_out * diff_e(ndiff - 1) / dr, 0d0)
  lum_cap = lum_heat + e_prev / max(dt, 1d-30)
  if (lum_raw > lum_cap .and. lum_raw > 0d0) then
   lum_scale = lum_cap / lum_raw
   diff_e = diff_e * lum_scale
   lum_obs = lum_cap
  else
   lum_obs = lum_raw
  end if
  diffusion_state_ready = .true.
  erad = diffusion_total_energy()
 end subroutine solve_diffusion_step

 subroutine update_shell_reservoir(dt, r_shell, u_shell, tau_unshocked, &
                                   lum_heat, lum_shell_leak, in_escape)
  real(8), intent(in) :: dt, r_shell, u_shell, tau_unshocked, lum_heat
  logical, intent(in) :: in_escape
  real(8), intent(out) :: lum_shell_leak
  real(8) :: tau_sh, t_diff_sh, r_now, t_exp

  if (.not. diffusion_enabled .or. m_swept <= 0d0 .or. r_shell <= 0d0) then
   lum_shell_leak = 0d0
   return
  end if

  ! Record shell state at CSM exit (first time escape triggers)
  if (in_escape .and. .not. shell_exit_set) then
   r_shell_exit = r_shell
   v_shell_exit = u_shell
   t_shell_exit = 0d0
   shell_exit_set = .true.
  end if

  ! Shell diffusion timescale: t_diff = tau * R / c
  ! tau from swept mass, R is current shell radius
  tau_sh = kappa_global * m_swept / (4d0 * pi * r_shell**2)
  t_diff_sh = max(tau_sh * r_shell / clight, dt)

  if (shell_exit_set) then
   ! Post-exit: expanding shell with adiabatic + diffusion losses
   ! Shell radius grows: R(t) = R_exit + v_exit * dt_since_exit
   t_shell_exit = t_shell_exit + dt
   r_now = r_shell_exit + v_shell_exit * t_shell_exit

   ! Update t_diff for expanding shell: tau decreases as r^-2
   tau_sh = kappa_global * m_swept / (4d0 * pi * r_now**2)
   t_diff_sh = max(tau_sh * r_now / clight, dt)

   ! Adiabatic expansion timescale for radiation: t_exp = R / (4*v)
   ! Factor 4 for radiation-dominated PdV work (P = E/3V)
   t_exp = r_now / (4d0 * v_shell_exit)

   ! Implicit Euler with both diffusion and adiabatic losses, no heating
   e_shell = e_shell / (1d0 + dt / t_diff_sh + dt / t_exp)
  else
   ! During interaction: fill with residual, leak via diffusion
   ! (keeps reservoir in quasi-steady state for smooth transition)
   e_shell = (e_shell + dt * lum_heat) / (1d0 + dt / t_diff_sh)
  end if

  ! Luminosity leaking from shell
  lum_shell_leak = e_shell / t_diff_sh

 end subroutine update_shell_reservoir

end module diffusion_runtime
