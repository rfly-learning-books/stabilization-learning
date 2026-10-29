"""Optional direct-gradient objective for differentiable state-model tasks."""
import torch
from .metrics import quadratic_certificate


def model_certificate_loss(env, states, M, *, decay=.05, certificate_weight=1.):
    """Hold M over one transition; use the environment's bounded final action.

    No detach/NumPy is used. Returns gradients through M -> u -> model -> e+.
    This is empirical model-based training, not an analytic feasibility proof.
    Hard candidate controllers should use the episode PPO learner instead.
    """
    e = env.tensor_error(states)
    goal = torch.as_tensor(env.goal, dtype=states.dtype, device=states.device)
    action = env.calculate_action_from_M(M, e, context={"state": states, "goal": goal})
    following = env.torch_step(states, action)
    next_e = env.tensor_error(following)
    V, next_V = quadratic_certificate(e, M), quadratic_certificate(next_e, M)
    residual = torch.relu(next_V - V + decay * env.dt * e.square().sum(-1))
    performance = next_e.square().sum(-1)
    return {"loss": (performance + certificate_weight * residual).mean(),
            "performance": performance.mean(), "descent_residual": residual.mean()}
