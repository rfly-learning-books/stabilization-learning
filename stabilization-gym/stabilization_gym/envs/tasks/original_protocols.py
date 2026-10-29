"""Explicit original research protocols, alongside the portable task baselines.

The physical transition and formal controller stay in the environment package.
The learner imports this contract, rather than replacing state-dependent M with
one Gaussian draw per episode. Vendored research source is kept byte-identical.
"""
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


def workspace_root():
    for p in Path(__file__).resolve().parents:
        if (p / 'migration/original_protocol_assets.json').is_file():
            return p
    raise FileNotFoundError('Run scripts/prepare_original_protocol_migration.py first')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_push_source(source=None, namespace='_stabilization_original_push'):
    root = Path(source or workspace_root() / 'third_party/push_original/dl_push').resolve()
    if namespace in sys.modules:
        if Path(sys.modules[namespace].__file__).parent != root:
            raise RuntimeError('Cannot mix original source roots in one namespace')
    else:
        spec = importlib.util.spec_from_file_location(namespace, root / '__init__.py',
                                                     submodule_search_locations=[str(root)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = package
        spec.loader.exec_module(package)
    return importlib.import_module(namespace + '.certificate.full_m_ppo')


@dataclass(frozen=True)
class CertificateContract:
    conditioning: tuple[str, ...]
    scope: str
    error_dim: int
    full_state_derivative: bool
    differentiable_action: bool
    differentiable_candidate_scores: bool


class OriginalPushTask:
    """Original 32-shape, 12-mode, 100-step Full-M task and control contract.

    ``new_environment`` exposes the exact vector physical/control state machine.
    ``calculate_action_from_certificate`` has no physical transition side effect.
    Advance the returned action through ``environment.advance`` exactly once.
    Native training reward is retained, and is NOT the portable positive cost.
    """
    name = 'push_any_shape'
    contract = CertificateContract(('D', 'C', 'valid_modes'), 'state_dependent', 3,
                                   True, False, True)

    def __init__(self, *, root=None):
        self.root = Path(root or workspace_root()).resolve()
        self.reference = self.root / 'assets/original_push_reference'
        self.library = self.root / 'assets/push_geometric32'
        self.source = self.root / 'third_party/push_original/dl_push'
        self.engine = load_push_source(self.source)
        self.manifest = json.loads((self.reference / 'manifest.json').read_text(encoding='utf-8'))
        if self.engine.source_hashes() != self.manifest['source_sha256']:
            raise RuntimeError('Original Push source hash mismatch')
        self.cfg = self.engine.FullMConfig(**self.manifest['config'])
        self.run_config = self.manifest['run_config']

    def bank(self, device='cpu'):
        bank = self.engine.ModeBank(self.library, device=device)
        if bank.fingerprints != self.manifest['fingerprints']:
            raise RuntimeError('Original shape calibration fingerprints changed')
        return bank

    def new_environment(self, bank, shape_names=None, num_envs=None):
        names = shape_names or self.run_config['train_shapes']
        n = num_envs or self.run_config['num_envs']
        lanes = [names[i % len(names)] for i in range(n)]
        return self.engine.FullMControlEnv(bank, self.cfg, lanes, n)

    def metric_context(self, data, bank):
        ids = data['shape']
        return dict(D=self.engine.coordinates(data['e']), C=bank.C[ids],
                    valid_modes=bank.valid[ids])

    def candidate_scores(self, certificate, data, bank, *, training=False):
        return self.engine.candidates(certificate, data, bank, self.cfg, training=training,
                                     speed_points=self.cfg.train_speed_points if training else self.cfg.eval_speed_points)

    def calculate_action_from_certificate(self, certificate, data, bank, environment):
        candidates = self.candidate_scores(certificate, data, bank)
        goal = torch.as_tensor(environment.at_goal(), device=bank.device)
        recovery = torch.as_tensor(environment.recovery, device=bank.device)
        decision = self.engine.hard_choice(candidates, data, self.cfg, recovery, goal)
        return decision, candidates

    def load_actor(self, checkpoint, device='cpu'):
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved['source_sha256'] != self.engine.source_hashes():
            raise RuntimeError('Checkpoint source mismatch')
        if saved['cfg'] != self.manifest['config'] or saved['fingerprints'] != self.manifest['fingerprints']:
            raise RuntimeError('Checkpoint task/shape mismatch')
        actor = self.engine.FullMActor(self.cfg).to(device)
        actor.load_state_dict(saved['actor'])
        return actor, saved

    def evaluate(self, actor, bank, *, seed=20260913, output=None, video=False):
        return self.engine.evaluate(actor, bank, self.cfg, self.run_config['eval_shapes'],
                                    seed=seed, output=output, video=video)


class OriginalVisualServoTask:
    """Original RGB/FastDepth controller with explicit checkpoint-import support.

    The numerical/controller path works without constructing a room. Live room
    creation remains fail-closed while the original scene builder is missing.
    No substitute renderer is allowed to masquerade as the original experiment.
    """
    name = 'visual_servo'
    contract = CertificateContract(('z_goal',), 'episode_fixed', 16, False, True, True)

    def __init__(self, *, root=None):
        self.root = Path(root or workspace_root()).resolve()
        self.reference = self.root / 'assets/original_visual_reference'
        self.protocol = json.loads((self.reference / 'protocol.json').read_text(encoding='utf-8'))
        self.bundle = self.root / 'assets/original_visual_initialization'
        self.source = self.root / 'third_party/visual_original'

    def numerical_adapter(self):
        from .original_visual import OriginalVisualAdapter
        return OriginalVisualAdapter(self.root)

    def require_assets(self):
        provenance = json.loads((self.root / 'migration/original_protocol_assets.json').read_text(encoding='utf-8'))
        # The original deleted AE is superseded only by an explicit complete
        # initial-checkpoint import. Never report that deleted file as recovered.
        missing = []
        for entry in provenance['visual_missing_dependencies']:
            name = Path(entry.get('path', entry.get('asset', ''))).name
            candidate = self.root / 'assets/recovered_visual_legacy/src' / name
            if name.endswith('.py') and candidate.is_file():
                if digest(candidate) != entry['expected_sha256']:
                    raise RuntimeError('Recovered legacy source hash mismatch: ' + name)
                continue
            if name == 'best.pt' or entry.get('asset') == 'initial_autoencoder':
                continue
            missing.append(entry)
        if missing:
            raise FileNotFoundError('Exact visual protocol is unavailable: ' +
                                    ', '.join(x['asset'] for x in missing) +
                                    '. Original renderer/encoder source must be recovered and hash-verified.')
        raise NotImplementedError('Recovered assets must pass original closed-loop parity before training')


def make_original_task(name, **kwargs):
    if name == 'push_any_shape':
        return OriginalPushTask(**kwargs)
    if name == 'visual_servo':
        return OriginalVisualServoTask(**kwargs)
    raise KeyError(name)
