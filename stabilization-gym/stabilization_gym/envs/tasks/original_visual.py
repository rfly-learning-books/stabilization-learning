"""Declared numerical migration adapter for the retained visual experiment.

Vendored computational sources remain byte-identical. Only paths and complete
checkpoint initialization are adapted here. Room construction is not replaced.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class OriginalVisualAdapter:
    """Shared phi, goal-only M and the exact normalized-command controller.

    ``calculate_action_from_M`` preserves gradients through M and e. Its output
    is physical body velocity [m/s, m/s, m/s, rad/s]. The supplied reference J is
    ALWAYS with respect to the original normalized command. Scaling J and then
    recomputing an isotropic ridge would change the controller.
    """
    physical_scales = (.4, .4, .4, .35)

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.source = self.root / 'third_party/visual_original'
        self.bundle = self.root / 'assets/original_visual_initialization'
        self.legacy_root = self.root / 'assets/recovered_visual_legacy'
        self.original_protocol = json.loads((self.bundle / 'protocol.json').read_text(encoding='utf-8'))
        self.cfg = copy.deepcopy(self.original_protocol['config'])
        self.cfg['paths']['legacy_root'] = str(self.legacy_root)
        # This branch is immediately overwritten by a complete retained model
        # state, never used as a random replacement of the historical AE.
        self.cfg['paths']['initial_autoencoder'] = None
        self.verify_numerical_assets()
        self._load_sources()

    def verify_numerical_assets(self):
        """Fail before import on changed vendored code, weights or metadata."""
        source_manifest = json.loads((self.root / 'migration/visual_source_bundle.json').read_text(encoding='utf-8'))
        for row in source_manifest['source_records']:
            path = (self.source / row['relative']).resolve()
            if not path.is_relative_to(self.source) or file_sha(path) != row['sha256']:
                raise RuntimeError('Visual source/asset fingerprint mismatch: ' + row['relative'])
        report = json.loads((self.root / 'migration/visual_recovery_candidates/recovery_report.json').read_text(encoding='utf-8'))
        for name in ('common.py', 'models.py', 'vision.py', 'room_env.py', 'depth_net.py'):
            if file_sha(self.legacy_root / 'src' / name) != report['expected'][name]:
                raise RuntimeError('Recovered legacy source changed: ' + name)
        if self.original_protocol != json.loads((self.source / 'assets/historical_protocol/protocol.json').read_text(encoding='utf-8')):
            raise RuntimeError('Initialization and source protocol identities differ')

    def _load_sources(self):
        for name in ('goal_fixed_core', 'derivative_critic_core', 'train_goal_fixed', 'cae_vs_common', 'lac_vs'):
            module = sys.modules.get(name)
            if module is not None and not Path(module.__file__).resolve().is_relative_to(self.source):
                raise RuntimeError('Conflicting visual source imports; use a separate process')
        sys.path.insert(0, str(self.source))
        sys.path.insert(0, str(self.source / 'runtime'))
        self.core = importlib.import_module('goal_fixed_core')
        self.derivative = importlib.import_module('derivative_critic_core')
        self.vision = importlib.import_module('lac_vs.vision')
        self.config_module = importlib.import_module('lac_vs.config')
        common = self.core.legacy.module(self.cfg, 'common')
        # Explicit path rebinding, covered by the new adapter's own provenance.
        old_runtime = str(common.OLD_DEPTH_RUNTIME)
        sys.path[:] = [path for path in sys.path if path != old_runtime]
        common.ROOT = self.legacy_root
        common.WORKSPACE = self.root
        common.OLD_SCENES_DIR = self.source / 'assets/scene_dependencies/bright_natural_dense_servo_20261013'
        common.OLD_DEPTH_MODEL_DIR = self.source / 'assets/old_depth_audit/model'
        common.DEPTH_MODEL_FILE = common.OLD_DEPTH_MODEL_DIR / 'model.safetensors'
        common.OLD_DEPTH_RUNTIME = self.source / 'assets/unused_old_depth_runtime'
        common.OLD_AEVS_DIR = self.source / 'assets/unavailable_old_aevs'
        common.OLD_AEVS_CHECKPOINT = common.OLD_AEVS_DIR / 'best.pt'
        common.OLD_DEPTH_DIR = self.source / 'assets/unused_old_depth'

    def checkpoint(self, name='initial.pt'):
        if name not in ('initial.pt', 'best.pt'):
            raise ValueError('Only audited initial.pt and evaluation-only best.pt are supported')
        manifest = json.loads((self.root / 'migration/visual_initialization_bundle.json').read_text(encoding='utf-8'))
        expected = next(row['sha256'] for row in manifest['entries'] if row['relative'] == name)
        path = self.bundle / name
        if file_sha(path) != expected:
            raise RuntimeError('Imported checkpoint SHA256 changed')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        identity = hashlib.sha256(json.dumps(self.original_protocol, sort_keys=True).encode()).hexdigest()
        if payload['protocol_hash'] != identity:
            raise RuntimeError('Imported checkpoint does not match its original protocol')
        return payload

    def load_models(self, name='best.pt', device='cpu'):
        payload = self.checkpoint(name)
        ae = self.core.make_model(self.cfg, initialize=False).to(device).eval()
        metric = self.core.GoalMetric().to(device).eval()
        ae.load_state_dict(payload['learner']['ae'], strict=True)
        metric.load_state_dict(payload['learner']['metric'], strict=True)
        return ae, metric

    def load_training_initial(self, device='cpu'):
        """Restore every learner/optimizer/RNG field, without writing old runs."""
        payload = self.checkpoint('initial.pt')
        if any(payload['learner']['counts'].values()):
            raise RuntimeError('Expected a zero-update training initial checkpoint')
        learner = self.derivative.DerivativeLearner(self.cfg, payload, device)
        learner.restore(payload['learner'])
        # initial is intentionally absent from learner.payload, and must always
        # remain the original initial AE (not validation-best AE).
        learner.initial.load_state_dict(payload['learner']['ae'])
        self.config_module.restore_rng(payload['rng'])
        return learner, payload

    def reference(self, ae, metric, goal_rgb, rotation, *, depth=None, device='cpu', identity=False):
        if depth is None:
            depth = self.core.FastDepth(self.cfg, device).predict(goal_rgb)
        return self.core.Reference(self.cfg, ae, metric, goal_rgb, depth, rotation,
                                   device=device, identity=identity)

    def encode_error(self, ae, current_rgb, goal_rgb, device='cpu'):
        """Current and desired images share one trainable feature extractor."""
        x = self.vision.image_tensor(self.cfg, [current_rgb, goal_rgb], device)
        z, goal = ae.encode(x).split(1)
        return (z - goal) / 4, goal

    def calculate_action_from_M(self, M, e, reference):
        """Pure differentiable control; no physical step and no pose input."""
        J = reference.j if hasattr(reference, 'j') else reference
        normalized, _ = self.core.policy(e, J, M)
        return normalized * normalized.new_tensor(self.physical_scales)

    def certificate(self, M, e):
        return self.core.energy(e, M)

    def new_environment(self, *args, **kwargs):
        raise FileNotFoundError(
            'Original scene_builders.py lines 131-285 are unavailable. Numerical '
            'migration is supported; original rendering, online collection and '
            'closed-loop training are blocked until that dependency is recovered.')
