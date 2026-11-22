"""Run extended ULP and transport checks: python -m tests.numerical_audit.

Diagnostic only: small ULP sensitivity does not establish integration accuracy.
"""

import json
import time

import numpy as np

from redback_csm import core
from tests.static_shell_reference import BASE, DAY, StaticShellReference


def scan(function, params, energy='esn', count=31):
    epochs = np.array([1.018, 1.03, 1.05, 1.2, 5, 20, 29, 40, 80])
    values = []
    start = time.perf_counter()
    for i in range(count):
        p = dict(params, kappa=0.34)
        p[energy] = 1.0+i*10*np.spacing(1.0)
        lc = function(**p)
        values.append(np.interp(epochs, lc.time/DAY, lc.lbol, left=0, right=0))
    values = np.asarray(values)
    positive = np.all(values > 0, axis=0)
    delta = np.ptp(-2.5*np.log10(values[:, positive]), axis=0)
    return dict(seconds_per_call=(time.perf_counter()-start)/count,
                max_magnitude_range=float(np.max(delta)),
                zero_finite_flips=bool(np.any(np.any(values > 0, axis=0) != positive)))


def main():
    common = {k: BASE[k] for k in ['delta_sn', 'nn_sn', 'mej_sn', 'esn', 'eff']}
    nodes = dict(log_r_inner=14.0, log_r_outer=15.0,
                 **{f'log_rho_{i}': -12.05-2*i/7 for i in range(8)})
    generic = dict(common, base_density=1e-12, base_index=-2.0,
                   r_inner=1e14, r_outer=1e15, interval_sn=1000.0)
    for i in range(1, 5):
        generic.update({f'shell{i}_radius': (1+i)*1e14,
                        f'shell{i}_width': 5e13, f'shell{i}_density': 1e-12})
    cases = {
        'static_bpl': (core._get_lc_static_powerlaw_csm_bpl, BASE),
        'static_exponential': (core._get_lc_static_powerlaw_csm_exponential,
                               {k: v for k, v in BASE.items() if k not in ['delta_sn', 'nn_sn']}),
        'static_spline': (core._get_lc_static_spline_csm_bpl, dict(common, **nodes)),
        'homologous_powerlaw': (core._get_lc_homologous_powerlaw_csm_bpl, dict(BASE, interval_sn=1000.0)),
        'homologous_spline': (core._get_lc_generic_spline_csm_bpl, dict(common, **nodes, interval_sn=1000.0)),
        'generic_four_shell': (core._get_lc_generic_4shell_csm_bpl, generic),
    }
    for name, (fn, params) in cases.items():
        print(json.dumps({name: scan(fn, params)}), flush=True)
    print(json.dumps({'wind_bpl': scan(core._get_lc_wind_bpl,
          dict(mdot=1e-3, vwind=100, delta=1, nn=10, mexp=10, eff=0.5), energy='eexp')}), flush=True)
    reference = StaticShellReference()
    epochs = np.array([2., 5., 10., 20.])
    for zones in [40, 80, 160]:
        start = time.perf_counter()
        lc = core._get_lc_static_powerlaw_csm_bpl(**BASE, mode='transport', kappa=0.34, n_rad_zones=zones)
        print(json.dumps({'transport_zones': zones, 'seconds': time.perf_counter()-start,
                          'epochs_days': epochs.tolist(),
                          'shock_relative_to_radau': (np.interp(epochs, lc.time/DAY, lc.lbol_shock)
                                                     /reference.shock(epochs)-1).tolist(),
                          'diffused_luminosity': np.interp(epochs, lc.time/DAY, lc.lbol).tolist()}), flush=True)


if __name__ == '__main__':
    main()
