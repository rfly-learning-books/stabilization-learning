import torch
from stabilization_learning import CholeskyMetric, make
from stabilization_learning.losses import model_certificate_loss


def test_full_model_certificate_loss_has_correct_M_to_bounded_action_to_state_gradient():
    env = make("point_mass")
    transform = CholeskyMetric(env.metric_spec).double()
    raw = (transform.identity_raw(dtype=torch.float64) + .06).requires_grad_()
    states = torch.tensor([[.2, -.1, .3, -.2], [-.1, .2, -.1, .3]], dtype=torch.float64)

    def objective(x):
        return model_certificate_loss(env, states, transform(x))["loss"]

    loss = objective(raw)
    gradient, = torch.autograd.grad(loss, raw)
    assert gradient.norm() > 1e-6
    assert torch.autograd.gradcheck(objective, (raw,), eps=1e-6, atol=2e-5)
