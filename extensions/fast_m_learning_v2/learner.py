"""Episode-M PPO with a declared sparse validation and unchanged PPO updates."""
from copy import deepcopy
from dataclasses import dataclass
import json
from numbers import Integral
from pathlib import Path
import time

from stabilization_learning.trainer import StabilizationLearner, TrainConfig, atomic_save, module_hash, summarize


@dataclass
class SparseValidationConfig(TrainConfig):
    validation_every_updates: int = 5

    def __post_init__(self):
        super().__post_init__()
        if (not isinstance(self.validation_every_updates, Integral)
                or isinstance(self.validation_every_updates, bool) or self.validation_every_updates <= 0):
            raise ValueError('validation_every_updates must be a positive integer')


class SparseValidationLearner(StabilizationLearner):
    """Initial, every-k and final validation; commit every PPO batch.

    Unevaluated training rows never enter best-policy selection. Timing logs
    are separate from the deterministic, resumable learning history.
    """
    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, config=config or SparseValidationConfig(), **kwargs)

    def _protocol(self):
        protocol = super()._protocol()
        protocol['algorithm'] = 'episode_M_PPO_sparse_validation_v2'
        protocol['validation_schedule'] = {
            'initial': True, 'every_updates': self.config.validation_every_updates,
            'final': True, 'selection': 'validated_checkpoints_only', 'checkpoint': 'every_update'}
        return protocol

    def _selected(self):
        return min((row for row in self.history if isinstance(row.get('validation'), dict)),
                   key=lambda row: row['validation']['objective'])

    def _save_best(self, output):
        selected = self._selected()
        atomic_save({'protocol': self._protocol(), 'actor': self.best_actor, 'score': self.best_score,
                     'selected_update': selected['updates'],
                     'selected_physical_steps': selected['physical_steps']}, output / 'best.pt')

    def train(self, output_dir, *, resume=False, max_updates=None):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        last = output / 'last.pt'
        if resume:
            self.restore(last)
        elif last.exists():
            raise FileExistsError('Existing run: use resume=True or a new output directory')
        if not self.history:
            initial = self.evaluate()['summary']
            self.best_score = initial['objective']
            self.best_actor = deepcopy(self.actor.state_dict())
            self.history.append({'updates': 0, 'episodes': 0, 'physical_steps': 0,
                                 'validation': initial, 'validated': True})
            self.save(last)
            self._save_best(output)
        count = 0
        while self.episodes < self.config.episodes:
            if max_updates is not None and count >= max_updates:
                break
            started = time.perf_counter()
            batch = self._collect(min(self.config.batch_episodes, self.config.episodes-self.episodes))
            collected = time.perf_counter()
            losses = self._update(*batch[:4])
            updated = time.perf_counter()
            should_validate = (self.updates % self.config.validation_every_updates == 0
                               or self.episodes >= self.config.episodes)
            validation = self.evaluate()['summary'] if should_validate else None
            validated = time.perf_counter()
            if validation is not None and validation['objective'] < self.best_score:
                self.best_score = validation['objective']
                self.best_actor = deepcopy(self.actor.state_dict())
            self.history.append({'updates': self.updates, 'episodes': self.episodes,
                'physical_steps': self.physical_steps, 'train': summarize(batch[4]),
                'validation': validation, 'validated': should_validate, **losses})
            if module_hash(self.representation or getattr(self.env, 'representation', None)) != self.representation_hash:
                raise RuntimeError('Representation changed during the policy run')
            self.save(last)
            atomic_save(self.history, output / 'history.pt')
            if should_validate:
                self._save_best(output)
            with (output / 'timing.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'update': self.updates, 'validated': should_validate,
                    'collection_seconds': collected-started, 'PPO_seconds': updated-collected,
                    'validation_seconds': validated-updated,
                    'commit_seconds': time.perf_counter()-validated})+'\n')
            count += 1
        self._save_best(output)
        self._write_report(output)
        return self.history

    def _write_report(self, output):
        selected = self._selected()
        status = {'algorithm': 'episode_M_PPO_sparse_validation_v2',
            'complete': self.episodes >= self.config.episodes, 'episodes': self.episodes,
            'updates': self.updates, 'physical_steps': self.physical_steps,
            'validation_updates': [row['updates'] for row in self.history if row.get('validated')],
            'best_selected_update': selected['updates'], 'best_validation_objective': self.best_score,
            'candidate_certificate_only': True}
        (output/'status.json').write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
        (output/'progress.json').write_text(json.dumps(self.history, indent=2)+'\n', encoding='utf-8')
