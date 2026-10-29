"""Same caller for an analytic task, calibrated pushing and RGB servo."""
import torch
from stabilization_learning import make, CholeskyMetric


def run(name, **kwargs):
    env = make(name, **kwargs)
    obs, info = env.reset(seed=41)
    mapping = CholeskyMetric(env.metric_spec)
    M = mapping(mapping.identity_raw())
    for _ in range(env.max_steps):
        e, context = env.build_error(obs)
        u = env.calculate_action_from_M(M.to(e), e, context=context)
        obs, cost, terminated, truncated, info = env.step(u.detach().cpu().numpy())
        if terminated or truncated:
            break
    env.close()
    print(name, {k: info.get(k) for k in ("success", "failure", "physical_error")})


if __name__ == "__main__":
    torch.set_num_threads(1)
    for task in ("point_mass", "push_any_shape", "visual_servo"):
        run(task, max_steps=20)
