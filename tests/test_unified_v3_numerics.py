"""Fault-injection checks for new-run Push numerical boundaries (CPU only)."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from extensions.unified_training_v3.numerical import CheckedReplay, FiniteAdam
from extensions.unified_training_v3.push import FixedPushBackend


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "runs/scratch_apps_so101_full_v1_20261026_184356/so101_environment/time_bank.json"


@pytest.fixture
def backend(tmp_path):
    result = FixedPushBackend(dict(time_bank=str(BANK), device="cpu", seed=74923,
        num_envs=1, initial_steps=3, collection_steps=2, batch_size=4,
        validation_repeats=1, test_repeats=1), tmp_path / "push")
    result.initialize()
    yield result
    result.close()


@pytest.mark.parametrize("field", ["learning_rate", "alpha_learning_rate", "initial_alpha"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_nonfinite_optimizer_settings_rejected(field, value):
    with pytest.raises(ValueError):
        FixedPushBackend({field: value})


def test_extreme_valid_logits_cannot_lose_mass_to_invalid_candidates(tmp_path):
    from extensions.unified_training_v1.backends.push import PushBackend
    cand = dict(logits=torch.tensor([[-2.e9, -1.e9, -2.e9 - 1024.], [-1.e9, -1.e9, -1.e9]]),
                physical=torch.tensor([[True, False, True], [False, False, False]]))
    model = SimpleNamespace(candidates=lambda *args, **kwargs: cand)
    old, new = PushBackend(output=tmp_path / "old"), FixedPushBackend(output=tmp_path / "new")
    for backend in (old, new):
        backend.actor, backend.model = None, model
    old_probability, *_ = old._distribution({})
    assert old_probability[0].sum() == 0  # Concrete failure with the old sentinel.
    probability, logp, valid, _ = new._distribution({})
    torch.testing.assert_close(probability.sum(-1), torch.ones(2))
    assert probability[0, 1] == 0 and torch.isfinite(logp).all()
    assert valid.tolist() == [True, False]
    assert probability[1].tolist() == [1., 0., 0.]
    torch.distributions.Categorical(probs=probability).sample()


def test_corrupt_candidate_latches_failure_before_sampling(tmp_path):
    backend = FixedPushBackend(output=tmp_path)
    backend.actor = None
    backend.model = SimpleNamespace(candidates=lambda *args, **kwargs: dict(
        logits=torch.tensor([[float("nan"), 1.]]), physical=torch.tensor([[True, True]])))
    with pytest.raises(FloatingPointError, match="candidate output"):
        backend._distribution({})
    with pytest.raises(FloatingPointError, match="stopped after numerical failure"):
        backend._guard()


def test_replay_rejects_nonfinite_transition_before_insertion():
    failures = []
    replay = CheckedReplay(failure=failures.append)
    with pytest.raises(FloatingPointError, match="replay transition"):
        replay.append(dict(observation={"q": [0., float("nan"), 0.]}, dt=1., reward=0.))
    assert not replay and len(failures) == 1


def test_corrupt_replay_batch_cannot_update_parameters(backend):
    backend.collect(initial=True)
    for row in backend.replay:
        row["reward"] = float("nan")
    before = [p.detach().clone() for p in backend.q1.parameters()]
    with pytest.raises(FloatingPointError, match="sampled replay batch"):
        backend._critic_update()
    assert all(torch.equal(p, prior) for p, prior in zip(backend.q1.parameters(), before))
    assert backend.counts["evaluator_updates"] == 0
    with pytest.raises(FloatingPointError, match="stopped after numerical failure"):
        backend.save(backend.output / "must_not_exist.pt")
    assert not (backend.output / "must_not_exist.pt").exists()


def test_optimizer_moment_corruption_is_rejected_before_parameter_step():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    failures = []
    optimizer = FiniteAdam([parameter], lr=.01, failure=failures.append, label="test")
    parameter.grad = torch.tensor([.1])
    optimizer.step()
    before = parameter.detach().clone()
    optimizer.state[parameter]["exp_avg"].fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="optimizer state"):
        optimizer.step()
    assert torch.equal(parameter, before) and len(failures) == 1


def test_finite_input_overflow_in_optimizer_state_is_detected():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    failures = []
    optimizer = FiniteAdam([parameter], lr=.01, failure=failures.append, label="test")
    parameter.grad = torch.tensor([1.e30])
    with pytest.raises(FloatingPointError, match="optimizer state"):
        optimizer.step()
    assert len(failures) == 1


@pytest.mark.parametrize("log_alpha", [1000., -1000.])
def test_temperature_overflow_or_underflow_cannot_start_update(backend, log_alpha):
    backend.collect(initial=True)
    with torch.no_grad():
        backend.log_alpha.fill_(log_alpha)
    with pytest.raises(FloatingPointError, match="temperature"):
        backend._actor_update()
    assert backend.counts["certificate_updates"] == 0
    assert backend.counts["alpha_updates"] == 0


def test_checkpoint_rejects_nonfinite_target_network(backend):
    with torch.no_grad():
        next(backend.target1.parameters()).fill_(float("nan"))
    path = backend.output / "bad.pt"
    with pytest.raises(FloatingPointError, match="target1 checkpoint"):
        backend.save(path)
    assert not path.exists()


def test_finite_parameters_with_overflowing_metric_stop_before_probe_record(backend):
    before = len(backend.probe_records)
    with torch.no_grad():
        for parameter in backend.actor.parameters():
            parameter.fill_(1.e20)
    with pytest.raises(FloatingPointError, match="fixed metric probe"):
        backend._probe("overflow_injection")
    assert len(backend.probe_records) == before
    with pytest.raises(FloatingPointError, match="stopped after numerical failure"):
        backend.save(backend.output / "corrupt_metric.pt")


def test_repeated_real_push_updates_remain_finite_with_replay_rollover(tmp_path):
    from extensions.unified_training_v1.backends.push import _module_hash
    from extensions.unified_training_v3.numerical import require_finite
    backend = FixedPushBackend(dict(time_bank=str(BANK), device="cpu", seed=74929,
        num_envs=2, initial_steps=4, collection_steps=2, batch_size=8,
        replay_capacity=32, validation_repeats=1, test_repeats=1), tmp_path / "stress")
    try:
        backend.initialize()
        initial_hash = _module_hash(backend.actor)
        backend.collect(initial=True)
        for epoch in range(1, 25):
            backend.collect()
            result = backend.train_epoch(epoch, evaluator_updates=2, certificate_updates=2)
            require_finite(result, "stress metrics")
        assert backend.counts["transitions"] == 104
        assert backend.counts["evaluator_updates"] == 48
        assert backend.counts["certificate_updates"] > 0
        assert len(backend.replay) == 32
        assert _module_hash(backend.actor) != initial_hash
        checkpoint = tmp_path / "stress_checkpoint.pt"
        backend.save(checkpoint)
        assert checkpoint.is_file()
    finally:
        backend.close()


def test_evaluation_never_selects_invalid_sentinel_over_extreme_valid_logits(backend, monkeypatch):
    from extensions.unified_training_v3.evaluation import evaluate
    original = backend.model.candidates
    expected = []
    def extreme_logits(actor, data, **kwargs):
        candidates = original(actor, data, **kwargs)
        available = torch.where(candidates["physical"][0])[0]
        if len(available):
            chosen = int(available[-1])
            candidates["physical"].fill_(False)
            candidates["physical"][0, chosen] = True
            candidates["logits"] = torch.full_like(candidates["logits"], -1.e9)
            candidates["logits"][0, chosen] = -2.e9
            expected.append(chosen)
        else:
            expected.append(0)
        return candidates
    monkeypatch.setattr(backend.model, "candidates", extreme_logits)
    result = evaluate(backend.actor, backend.model, [backend.config.train_shapes[0]], [74937], backend.costs)
    actions = [row["action"] for row in result["episodes"][0]["trace"]]
    assert actions and actions == expected
    assert any(action != 0 for action in actions)


def test_evaluation_rejects_nonfinite_candidate_metric(backend, monkeypatch):
    from extensions.unified_training_v3.evaluation import evaluate
    original = backend.model.candidates
    def corrupt_metric(actor, data, **kwargs):
        candidates = original(actor, data, **kwargs)
        candidates["M"] = torch.full_like(candidates["M"], float("nan"))
        return candidates
    monkeypatch.setattr(backend.model, "candidates", corrupt_metric)
    with pytest.raises(FloatingPointError, match="evaluation candidate output.M"):
        evaluate(backend.actor, backend.model, [backend.config.train_shapes[0]], [74939], backend.costs)
