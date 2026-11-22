"""Independent dimensionless thin-shell reference, not a production backend.

SciPy Radau integrates the equations with an explicit outer-edge event. No
Fortran/JAX dynamics, density helpers, or output grids are used here.
"""

import numpy as np
from scipy.integrate import quad, solve_ivp

DAY = 86400.0
MSUN = 1.989e33
BASE = dict(eta=-2.0, r_inner=1e14, r_outer=1e15, delta_sn=1.0,
            nn_sn=10.0, mej_sn=10.0, esn=1.0, eff=0.5, m_csm=0.5)


class StaticShellReference:
    def __init__(self, seed=1e-9, rtol=1e-11, end_days=100.0):
        p = BASE
        a, d, n = 3.0, p['delta_sn'], p['nn_sn']
        im = 1 / (3-d) + (1-a**(3-n)) / (n-3)
        ie = 1 / (5-d) + (1-a**(5-n)) / (n-5)
        self.vt = np.sqrt(2*p['esn']*1e51*im/(p['mej_sn']*MSUN*ie))
        self.vmax = a*self.vt
        self.tc = p['r_inner']/self.vmax
        self.xout = p['r_outer']/p['r_inner']
        self.rho = p['m_csm']*MSUN/(4*np.pi*p['r_inner']**3*(self.xout-1))
        mass_scale = 4*np.pi*p['r_inner']**3*self.rho
        self.q = p['mej_sn']*MSUN/(self.vt**3*im)*self.vmax**3/mass_scale
        self.lscale = p['eff']*2*np.pi*p['r_inner']**2*self.rho*self.vmax**3

        def event(t, y):
            return y[0]-self.xout
        event.terminal = True
        event.direction = 1
        opts = dict(method='Radau', rtol=rtol, atol=[1e-14, 1e-14, seed*1e-6], dense_output=True)
        end = end_days*DAY/self.tc
        self.inside = solve_ivp(lambda t, y: self.rhs(t, y, False), (1, end),
                                [1, 1, seed], events=event, **opts)
        assert self.inside.success and len(self.inside.t_events[0]) == 1
        self.te = self.inside.t_events[0][0]
        self.outside = solve_ivp(lambda t, y: self.rhs(t, y, True), (self.te, end),
                                 self.inside.y_events[0][0], **opts)
        assert self.outside.success

    def coefficients(self, t, y, exterior):
        x, w, _ = y
        v = x/t
        slope = 10 if v > 1/3 else 1
        ae = self.q*(1/(3*v))**slope*v*v/t if v <= 1 else 0.0
        ac = 0.0 if exterior else 1.0  # eta = -2
        return ae, ac, max(v-w, 0.0), max(w, 0.0)

    def rhs(self, t, y, exterior):
        ae, ac, vi, vo = self.coefficients(t, y, exterior)
        return [y[1], (ae*vi**2-ac*vo**2)/y[2], ae*vi+ac*vo]

    def luminosity(self, t, y, exterior):
        ae, ac, vi, vo = self.coefficients(t, y, exterior)
        return self.lscale*(ae*vi**3+ac*vo**3)

    def shock(self, days):
        result = []
        for day in np.atleast_1d(days):
            t = day*DAY/self.tc
            exterior = t > self.te
            sol = self.outside if exterior else self.inside
            result.append(0 if t < 1 else self.luminosity(t, sol.sol(t), exterior))
        return np.array(result)

    def diffuse(self, days, kappa=0.34):
        # Evaluate the existing injection-time kernel, not a different
        # reservoir ODE. Its value at exactly emergence is singular.
        result = []
        for day in np.atleast_1d(days):
            tq = day*DAY/self.tc
            def integrand(t):
                y = self.inside.sol(t)
                x, w, _ = y
                tau = kappa*self.rho*BASE['r_inner']*(1/x-1/self.xout)
                td = tau*x*BASE['r_inner']/2.99792458e10/self.tc
                ta = x/w
                if td <= 0:
                    return 0.0
                return self.luminosity(t, y, False)/self.lscale/td*np.exp(-(tq-t)*(1/td+1/ta))
            val, _ = quad(integrand, 1, min(tq, self.te), epsabs=1e-9, epsrel=1e-8, limit=300)
            result.append(val*self.lscale)
        return np.array(result)
