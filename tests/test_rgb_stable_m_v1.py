import math
import torch
from stabilization_learning.rgb_stable.objectives import component_energy, progress_risk, priority_gradient, trust_statistics, group_risks
from extensions.rgb_stable_m_v1.learner import ComponentCritic


def test_components_use_actual_success_tolerances():
    q=torch.tensor([[.02,0,0,math.radians(1)]],dtype=torch.float64)
    torch.testing.assert_close(component_energy(q),torch.ones(1,2,dtype=torch.float64))


def test_progress_prefers_reaching_over_outside_tolerance_stall():
    q=torch.tensor([[.025,0,0,0.]])
    stall=progress_risk(q,q)[0]
    progress=q.clone(); progress[:,0]=.024
    assert progress_risk(q,progress)[0]<stall


def test_hold_penalizes_departure_from_success_region():
    q=torch.tensor([[.018,0,0,0.]])
    leaves=q.clone(); leaves[:,0]=.023
    enters=q.clone(); enters[:,0]=.015
    assert progress_risk(q,leaves)[0]>progress_risk(q,enters)[0]


def test_auxiliary_cannot_reverse_primary_gradient():
    p=torch.tensor([1.,0.,0.]); a=torch.tensor([-100.,30.,0.])
    combined,stats=priority_gradient(p,a,.2)
    assert torch.dot(p,combined)>=torch.dot(p,p)-1e-7
    assert (combined-p).norm()<=.200001
    assert stats['conflict']


def test_zero_primary_does_not_allow_auxiliary_to_take_over():
    output,_=priority_gradient(torch.zeros(3),torch.ones(3),.2)
    assert torch.equal(output,torch.zeros(3))


def test_action_and_matrix_change_are_distinct():
    matrix=torch.eye(16)[None].repeat(2,1,1)
    stats=trust_statistics(torch.zeros(2,4),torch.ones(2,4)*.1,matrix,matrix)
    assert stats['metric_relative_change_mean']==0
    assert stats['action_change_max']>.19


def test_component_critic_is_action_linear_and_goal_anchored():
    critic=ComponentCritic(); z=torch.randn(8,16); goal=torch.randn(8,16)
    a,b=torch.randn(8,4),torch.randn(8,4)
    torch.testing.assert_close(critic(z,goal,a+b),critic(z,goal,a)+critic(z,goal,b),atol=1e-4,rtol=1e-5)
    assert torch.equal(critic(goal,goal,a),torch.zeros(8,2))
    c=critic.coefficients(z,goal)
    assert torch.equal(c[:,0,3],torch.zeros(8))
    assert torch.equal(c[:,1,:3],torch.zeros(8,3))


def test_group_checks_do_not_mix_position_and_yaw():
    q=torch.tensor([[.01,0,0,0],[.03,0,0,0],[.2,0,0,0]])
    components=torch.arange(6.).reshape(3,2)
    torch.testing.assert_close(group_risks(components,q),components.flatten())
