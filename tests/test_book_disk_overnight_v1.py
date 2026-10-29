import numpy as np
from extensions.book_disk_overnight_v1.env import DiskConfig,DiskEnv,transitions,value
from extensions.book_disk_overnight_v1.algo import fit,calibrate,choose_action


def test_printed_relation_and_safe_domain():
    env=DiskEnv();x=np.linspace(-.05,0.,101);a=-x
    following=env.step(x,a)
    np.testing.assert_allclose(following,.9*x,rtol=1e-14,atol=1e-16)
    np.testing.assert_allclose(value(following),.81*value(x),rtol=1e-14,atol=1e-16)
    varied=DiskEnv(DiskConfig(response='varying'))
    assert varied.in_domain(varied.step(x,varied.safe_upper(x))).all()


def test_direct_rank_deficiency_and_excited_identification():
    env=DiskEnv()
    feedback=transitions(env,128,np.random.default_rng(41),'feedback')
    _,bad=fit('direct_D',feedback,env.config)
    assert bad['information']['rank']==1
    random=transitions(env,512,np.random.default_rng(41),'random')
    model,good=fit('direct_D',random,env.config)
    assert good['information']['rank']==2
    np.testing.assert_allclose(model.parameters['beta'],[.1,.005],atol=1e-14)


def test_all_routes_zero_input_and_complete_D():
    env=DiskEnv();data=transitions(env,128,np.random.default_rng(41))
    x=np.linspace(-.05,0.,25);a=env.safe_upper(x)*.7
    for route in ('known_model','direct_D','identified_model','GMM'):
        model,_=fit(route,data,env.config)
        np.testing.assert_allclose(model.predict_D(x,np.zeros_like(x),env),0.,atol=0.)
        np.testing.assert_allclose(model.predict_D(x,a,env),value(env.step(x,a))-value(x),atol=1e-10)
        cal=calibrate(model,data,env)
        choice=choose_action(model,np.r_[x,0.],env,cal,'margin')
        assert choice['action'][-1]==0.
        assert env.in_domain(env.step(np.r_[x,0.],choice['action'])).all()


def test_infeasible_estimate_recorded_not_hidden():
    env=DiskEnv();data=transitions(env,64,np.random.default_rng(41))
    model,_=fit('direct_D',data,env.config)
    cal={'coefficient':100.}
    choice=choose_action(model,np.array([-.04]),env,cal,'margin')
    assert not choice['feasible'][0]
    assert choice['action'][0]>=0.


def test_decreasing_candidate_does_not_imply_one_sided_domain():
    env=DiskEnv();x=-.004;a=.05;following=env.step(x,a)
    assert value(following)-value(x)<0.
    assert not env.in_domain(following)
    assert a>env.safe_upper(x)
