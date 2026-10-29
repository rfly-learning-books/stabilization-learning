from dataclasses import dataclass, field, asdict, replace
import math
from stabilization_learning.joint_v2.visual import JointVisualConfig


@dataclass(frozen=True)
class RGBConfig:
    seed: int = 81
    device: str = "cuda:0"
    implementation: str = "joint_rgb_v4_gpu_coverage"
    interaction_backend: str = "forward"
    scope: str = "formal_training"
    warmup_updates: int = 1000
    joint_pairs: int = 6500
    batch_size: int = 16
    parallel_envs: int = 16
    collection_horizon: int = 64
    collection_pairs: tuple = (0, 1000, 2500, 4000, 5500)
    transitions_per_stratum: tuple = (80, 64, 64, 64, 64)
    random_episode_fraction_after_bootstrap: float = 0.5
    random_action_hold_steps: int = 8
    validation_pairs: tuple = (1300, 2600, 3900, 5200, 6500)
    evaluation_max_steps: int = 200
    calibration_samples: int = 384
    probe_count: int = 8
    train_scene_limit: int = 0
    validation_case_limit: int = 0
    test_case_limit: int = 0
    mmap_episodes: int = 64
    max_hours: float = 3.0
    minimum_free_gib: float = 30.0
    samples_per_axis: int = 2
    evaluation_starts: str = "runs/scratch_apps_so101_full_v1_20261026_184356/rgb_seed71/evaluation_starts.json"
    learner: JointVisualConfig = field(default_factory=JointVisualConfig)

    def __post_init__(self):
        if self.implementation != "joint_rgb_v4_gpu_coverage" or self.interaction_backend != "forward":
            raise ValueError("Explicit v4 GPU runner with unchanged forward-JVP learner required")
        if self.scope not in {"formal_training", "engineering_smoke", "performance_profile"}:
            raise ValueError("Unknown scope")
        for key in ("warmup_updates", "joint_pairs", "batch_size", "parallel_envs", "collection_horizon",
                    "random_action_hold_steps", "evaluation_max_steps", "calibration_samples", "probe_count", "mmap_episodes"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(key + " must be a positive integer")
        if not self.collection_pairs or self.collection_pairs[0] != 0 or len(self.collection_pairs) != len(self.transitions_per_stratum):
            raise ValueError("Declare bootstrap and matching per-stratum quotas")
        if sorted(set(self.collection_pairs)) != list(self.collection_pairs) or any(p >= self.joint_pairs for p in self.collection_pairs):
            raise ValueError("Collection must precede remaining updates")
        if any(type(n) is not int or n < 1 for n in self.transitions_per_stratum):
            raise ValueError("Positive integer collection quotas required")
        if sorted(set(self.validation_pairs)) != list(self.validation_pairs) or self.validation_pairs[-1] != self.joint_pairs:
            raise ValueError("Final update must be validated; order is immutable")
        if any(p < 1 for p in self.validation_pairs):
            raise ValueError("Select only actually updated checkpoints")
        if not 0 <= self.random_episode_fraction_after_bootstrap <= 1:
            raise ValueError("Invalid random episode fraction")
        if not math.isfinite(self.max_hours) or self.max_hours <= 0:
            raise ValueError("Finite positive total budget required")
        if min(self.train_scene_limit, self.validation_case_limit, self.test_case_limit, self.minimum_free_gib) < 0:
            raise ValueError("Negative scope/reserve")
        if self.scope == "formal_training" and (self.train_scene_limit or self.validation_case_limit or self.test_case_limit or self.evaluation_max_steps != 200):
            raise ValueError("Formal run must retain every declared scene/start and 200 steps")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ("collection_pairs", "transitions_per_stratum", "validation_pairs"):
            if key in value:
                value[key] = tuple(value[key])
        if "learner" in value:
            value["learner"] = JointVisualConfig(**value["learner"])
        return cls(**value)

    @classmethod
    def smoke(cls):
        return cls(scope="engineering_smoke", warmup_updates=3, joint_pairs=4, batch_size=4,
                   parallel_envs=8, collection_horizon=4, collection_pairs=(0, 2),
                   transitions_per_stratum=(4, 2), validation_pairs=(2, 4),
                   evaluation_max_steps=4, calibration_samples=16, probe_count=2,
                   train_scene_limit=2, validation_case_limit=2, test_case_limit=2, max_hours=.5)

    @classmethod
    def profile(cls):
        return cls(scope="performance_profile", warmup_updates=30, joint_pairs=80,
                   collection_pairs=(0, 40), transitions_per_stratum=(64, 32),
                   validation_pairs=(80,), validation_case_limit=12, test_case_limit=12, max_hours=1.)
