"""Standalone MuJoCo tasks; environment implementations are loaded on demand."""
from gymnasium.envs.registration import register, registry

__version__ = "0.1.0"
TASKS = (
    "PushTCost", "PushDiskCost", "PushDiskGDCLFCost", "PushTSO101Cost",
    "PushDiskSO101Cost", "PushMultiDisksSO101Cost",
)
ENV_IDS = tuple(f"DLPushContact/{task}-v1" for task in TASKS)

def register_envs():
    for task, env_id in zip(TASKS, ENV_IDS):
        if env_id not in registry:
            register(id=env_id, entry_point="dl_push_contact.factory:make_registered",
                     kwargs={"task": task}, max_episode_steps=300)

def make(task="PushTCost", **kwargs):
    """Create a registered task, with its reward/cost adapter and time limit."""
    import gymnasium as gym
    env_id = task if task in ENV_IDS else f"DLPushContact/{task.removesuffix('-v1')}-v1"
    if env_id not in ENV_IDS:
        raise ValueError(f"Unknown task {task!r}; choose one of {TASKS}")
    return gym.make(env_id, **kwargs)

register_envs()

