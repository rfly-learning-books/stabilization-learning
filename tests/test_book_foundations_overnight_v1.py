"""Numerical references independent of the experiment report machinery."""
import numpy as np
from extensions.book_foundations_overnight_v1.env import exact_affine,scalar_integrator,zoh
from extensions.book_foundations_overnight_v1.algo import pi_action,virtual_output_action


def test_zoh_integrator_and_book_motor_rounding():
    Ad,Bd=zoh([[0.]],[[1.]],.2)
    np.testing.assert_allclose(Ad,[[1.]],atol=1e-14)
    np.testing.assert_allclose(Bd,[[.2]],atol=1e-14)
    Ad,Bd=zoh([[-100.,-10.],[10.,-10.]],[100.,0.],.01)
    np.testing.assert_allclose(Ad,[[.365340,-.059564],[.059564,.901415]],rtol=0,atol=5.01e-7)
    np.testing.assert_allclose(Bd,[[.631113],[.035473]],rtol=0,atol=5.01e-7)


def test_sampling_changes_stability_and_limit_changes_orbit():
    t,x,u=scalar_integrator(.4,.2,30)
    np.testing.assert_allclose(x,.4*.6**np.arange(len(t)),rtol=1e-12,atol=1e-14)
    _,unstable,ua=scalar_integrator(.4,1.1,30)
    _,bounded,ub=scalar_integrator(.4,1.1,30,1.)
    assert abs(unstable[-1])>10
    assert max(abs(ub))<=1
    assert np.max(abs(bounded))<=1
    assert abs(bounded[-1])>.1


def test_virtual_output_removes_internal_unstable_mode():
    A=np.array([[0.,1.],[-1.,1.]]);B=np.array([0.,1.])
    for route,expected in [('original_output',[-2.,1.]),('virtual',[-2.,-1.])]:
        baseline=virtual_output_action([0.,0.],route=route)
        gains=np.array([virtual_output_action(np.eye(2)[i],route=route)-baseline for i in range(2)])
        np.testing.assert_allclose(np.sort(np.linalg.eigvals(A+np.outer(B,gains))),expected,atol=1e-12)
    assert virtual_output_action([-1.,0.],route='virtual')==-1.


def test_pi_constant_disturbance_equilibrium_and_certificate():
    A=np.array([[-2.,-1.],[1.,0.]]);P=np.array([[.5,.5],[.5,1.5]])
    np.testing.assert_allclose(A.T@P+P@A,-np.eye(2),atol=1e-14)
    assert min(np.linalg.eigvalsh(P))>0
    actual,requested=pi_action([0.,.3],.3,'PI')
    assert actual==requested==-.3
    np.testing.assert_allclose(exact_affine(A,[.3,0.],[0.,.3],[0.,1.,10.]),[[0.,.3]]*3,atol=1e-14)
    actual,_=pi_action([0.,1.2],1.2,'PI',.5)
    assert actual+1.2>0
