"""Independent positive-cost Gymnasium task library for metric controllers."""
from gymnasium.envs.registration import register, registry
from .common import MetricSpec
from .envs import make, available_envs

__version__ = "0.2.0"

GYM_ENVIRONMENTS = {
    "Stabilization/CoupledLinear-v0": "stabilization_gym.envs.classic_control:CoupledLinearEnv",
    "Stabilization/PointMass-v0": "stabilization_gym.envs.classic_control:PointMassEnv",
    "Stabilization/Pendulum-v0": "stabilization_gym.envs.classic_control:PendulumEnv",
    "Stabilization/CartPole-v0": "stabilization_gym.envs.classic_control:CartPoleEnv",
    "Stabilization/TwoLinkArm-v0": "stabilization_gym.envs.robotics:TwoLinkArmEnv",
    "Stabilization/UnicycleTracking-v0": "stabilization_gym.envs.robotics:UnicycleTrackingEnv",
    "Stabilization/MultirotorTracking-v0": "stabilization_gym.envs.robotics:MultirotorTrackingEnv",
    "Stabilization/PushAnyShape-v0": "stabilization_gym.envs.tasks.push_any_shape:PushAnyShapeEnv",
    "Stabilization/VisualServo-v0": "stabilization_gym.envs.tasks.visual_servo:VisualServoEnv",
}

for env_id, entry_point in GYM_ENVIRONMENTS.items():
    if env_id not in registry:
        register(id=env_id, entry_point=entry_point)

__all__ = ["make", "available_envs", "MetricSpec", "GYM_ENVIRONMENTS"]
