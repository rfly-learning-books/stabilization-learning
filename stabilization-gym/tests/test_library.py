"""CPU contract checks for public factories; never launches learning or CUDA."""
from pathlib import Path
import sys
import unittest
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from stabilization_gym.library import (
    STATE_TASKS, OBSERVER_TASKS, make_state, make_observer, make_push, make_rgb,
)
from stabilization_gym.tasks_v2.contracts import ActionPlan


class LibraryContracts(unittest.TestCase):
    def test_all_state_cpu_plants(self):
        for name in STATE_TASKS:
            with self.subTest(name=name):
                bundle = make_state(name, tensor_observations=True)
                try:
                    observation, _ = bundle.reset(seed=81)
                    self.assertTrue(torch.is_tensor(observation))
                    action = np.zeros(bundle.env.action_space.shape)
                    following, reward, _, _, info = bundle.step(action)
                    self.assertTrue(torch.is_tensor(following))
                    self.assertAlmostEqual(reward, -info["physical_cost"])
                    self.assertEqual(bundle.capabilities.physical_device, "cpu")
                finally:
                    bundle.close()

    def test_all_observers_preserve_physical_boundary(self):
        for name in OBSERVER_TASKS:
            with self.subTest(name=name):
                bundle = make_observer(name, tensor_observations=True)
                try:
                    observation, _ = bundle.reset(options={"split": "engineering", "index": 81, "scenario": "nominal"})
                    self.assertNotIn("truth", observation)
                    self.assertNotIn("estimate", observation)
                    self.assertTrue(torch.is_tensor(observation["measurement"]))
                    memory = bundle.adapter.initial_memory()
                    context = bundle.adapter.make_context(observation, memory=memory)
                    self.assertIsNone(context.error)
                    if observation["known_input"].numel():
                        with self.assertRaises(ValueError):
                            bundle.step(observation["known_input"]+100.)
                    else:
                        with self.assertRaises(ValueError):
                            bundle.step(torch.ones(1))
                    correction = ActionPlan(torch.ones(100)*999., memory={})
                    with self.assertRaises(TypeError):
                        bundle.step(correction)
                    following, reward, _, _, _ = bundle.advance(correction, observation)
                    self.assertEqual(reward, 0.)
                    self.assertGreater(float(following["time"][0]), float(observation["time"][0]))
                finally:
                    bundle.close()

    def test_push_explicit_bank_cpu(self):
        bank_path = PACKAGE.parent / "runs/scratch_apps_so101_full_v1_20261026_184356/so101_environment/time_bank.json"
        if not bank_path.is_file():
            self.skipTest("Explicit local calibration fixture is not installed")
        bundle = make_push(shapes=["T"], time_bank=bank_path, tensor_observations=True)
        try:
            observation, _ = bundle.reset(seed=81)
            self.assertTrue(torch.is_tensor(observation["q"]))
            context = bundle.adapter.make_context(observation)
            self.assertEqual(tuple(context.error.shape), (1, 3))
            following, reward, terminated, truncated, info = bundle.step(0)
            self.assertTrue(torch.is_tensor(following["q"]))
            self.assertTrue(np.isfinite(reward))
            self.assertEqual(bundle.capabilities.physical_device, "cpu")
            self.assertFalse(truncated)
        finally:
            bundle.close()

    def test_honest_device_and_explicit_resources(self):
        with self.assertRaises(ValueError):
            make_state(device="cuda")
        with self.assertRaises(ValueError):
            make_observer(device="cuda")
        with self.assertRaises(ValueError):
            make_push(shapes=["T"])
        adapter = SimpleNamespace(make_context=lambda *a: None, construct_action=lambda *a: None)
        with self.assertRaises(ValueError):
            make_rgb(cfg={}, scenes=[], adapter=adapter, device="cpu")
        with self.assertRaises(TypeError):
            make_rgb(cfg={}, scenes=[], adapter=None)

    def test_z_no_algo_package_import(self):
        # Other test modules may legitimately import the learner before this
        # test is collected. Exercise the dependency boundary in a fresh process.
        code = f"""import runpy, sys
namespace = runpy.run_path({str(Path(__file__).resolve())!r})
probe = namespace['LibraryContracts']()
probe.test_all_state_cpu_plants()
probe.test_all_observers_preserve_physical_boundary()
probe.test_push_explicit_bank_cpu()
assert not any(name == 'stabilization_learning' or name.startswith('stabilization_learning.') or name.startswith('extensions.') for name in sys.modules)
"""
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
