from copy import deepcopy
from pathlib import Path
import torch

from stabilization_gym.envs.classic_control import PointMassEnv
from stabilization_learning.envs.classic import PointMassEnv as LegacyPointMassEnv
from stabilization_learning.common.compat import canonical_protocol
from stabilization_learning.trainer import StabilizationLearner, TrainConfig


def test_old_import_is_only_an_alias_to_independent_environment():
    assert LegacyPointMassEnv is PointMassEnv
    assert PointMassEnv.__module__.startswith("stabilization_gym.")


def test_module_compatibility_does_not_ignore_parameters_or_hashes():
    before = {"class": "stabilization_learning.envs.classic.PointMassEnv", "dt": .04, "hash": "abc"}
    after = {"class": PointMassEnv.__module__ + ".PointMassEnv", "dt": .04, "hash": "abc"}
    assert canonical_protocol(before) == canonical_protocol(after)
    wrong = deepcopy(after)
    wrong["dt"] = .08
    assert canonical_protocol(before) != canonical_protocol(wrong)
    wrong = deepcopy(after)
    wrong["hash"] = "changed"
    assert canonical_protocol(before) != canonical_protocol(wrong)


def test_original_checkpoint_can_restore_after_directory_and_module_move():
    root = Path(__file__).resolve().parents[2]
    path = root / "runs/point_mass_demo/last.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    protocol = checkpoint["protocol"]
    learner = StabilizationLearner(PointMassEnv(**protocol["env_kwargs"]),
        config=TrainConfig(**protocol["config"]), env_id=protocol["env_id"], env_kwargs=protocol["env_kwargs"])
    learner.restore(path)
    assert learner.episodes == checkpoint["episodes"] == 96
    for key, value in learner.actor.state_dict().items():
        torch.testing.assert_close(value, checkpoint["actor"][key], rtol=0, atol=0)


def test_known_float32_physical_bound_metadata_migrates_without_accepting_new_limits():
    before = {"class": "stabilization_learning.envs.visual_servo.VisualServoEnv",
              "action_high": [0.4000000059604645]*3 + [0.3499999940395355]}
    after = {"class": "stabilization_gym.envs.tasks.visual_servo.VisualServoEnv",
             "action_high": [.4]*3 + [.35]}
    assert canonical_protocol(before) == canonical_protocol(after)
    after["action_high"][0] = .5
    assert canonical_protocol(before) != canonical_protocol(after)
