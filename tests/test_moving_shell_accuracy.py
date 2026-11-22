import numpy as np
import pytest

from redback_csm import core
from tests.static_shell_reference import BASE, DAY, StaticShellReference


def test_homologous_powerlaw_against_radau():
    ref=StaticShellReference(interval_days=1000.)
    lc=core._get_lc_homologous_powerlaw_csm_bpl(**BASE,interval_sn=1000.,kappa=.34)
    t=np.array([1.03,1.05,1.2,2,5,10,20,28,30,40,80])
    np.testing.assert_allclose(np.interp(t,lc.time/DAY,lc.lbol_shock),ref.shock(t),rtol=1e-3)
    np.testing.assert_allclose(np.interp(t,lc.time/DAY,lc.lbol),ref.diffuse(t),rtol=3e-3)


def four_shell_params():
    p={k:BASE[k] for k in ['delta_sn','nn_sn','mej_sn','esn','eff']}
    p.update(base_density=1e-12,base_index=-2.,r_inner=1e14,r_outer=1e15,interval_sn=1000.)
    for i in range(1,5):
        p.update({f'shell{i}_radius':(1+i)*1e14,f'shell{i}_width':5e13,f'shell{i}_density':1e-12})
    return p


def test_four_shell_dynamics_against_independent_radau():
    def rho(r):
        return 1e-12*(r/1e14)**-2 + sum(1e-12*np.exp(-.5*((r-(1+i)*1e14)/(5e13/2.355))**2) for i in range(1,5))
    ref=StaticShellReference(interval_days=1000.,density_function=rho)
    lc=core._get_lc_generic_4shell_csm_bpl(**four_shell_params())
    t=np.array([1.05,2,3,5,8,12,20,40,60,80])
    np.testing.assert_allclose(np.interp(t,lc.time/DAY,lc.lbol_shock),ref.shock(t),rtol=3e-3)


@pytest.mark.parametrize('profile',['homologous_spline','four_shell','eight_shell'])
def test_moving_profile_ulp_smoothness(profile):
    if profile in ('four_shell','eight_shell'):
        fn=core._get_lc_generic_4shell_csm_bpl if profile=='four_shell' else core._get_lc_generic_8shell_csm_bpl
        p=four_shell_params()
        if profile=='eight_shell':
            for i in range(5,9):
                p.update({f'shell{i}_radius':(1+i)*1e14,f'shell{i}_width':5e13,f'shell{i}_density':1e-12})
    else:
        fn=core._get_lc_generic_spline_csm_bpl
        p={k:BASE[k] for k in ['delta_sn','nn_sn','mej_sn','esn','eff']}
        p.update(log_r_inner=14.,log_r_outer=15.,interval_sn=1000.)
        p.update({f'log_rho_{i}':-12.05-2*i/7 for i in range(8)})
    curves=[]
    for offset in range(0,301,10):
        lc=fn(**dict(p,esn=1+offset*np.spacing(1.)),kappa=.34)
        curves.append(np.interp([1.03,1.05,5,20,29,40],lc.time/DAY,lc.lbol))
    assert np.all(np.asarray(curves)>0)
    assert np.max(np.ptp(-2.5*np.log10(curves),axis=0))<1e-4
