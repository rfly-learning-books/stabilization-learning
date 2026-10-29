"""CPU contracts for book reproduction, model selection, and paired reporting."""
from copy import deepcopy
import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stabilization_learning.rgb_book.learner import (
    MODELS, OPTIMIZERS, inference_state, restore_models,
)
from extensions.rgb_book_reproduction_v1.evaluation import (
    paired, summarize, verify_frozen, native_trace, _variant_models, evaluate_cases,
)


def learner_fixture():
    learner = SimpleNamespace(counts={"ae": 7, "metric": 11}, version=13,
                              anchors={"goal": torch.tensor([1., 2.])},
                              cache={"stale": 1}, inputs={"old": 2})
    learner.clear_cache = learner.cache.clear
    for model_name, optimizer_name in zip(MODELS, OPTIMIZERS):
        model = torch.nn.Linear(2, 2)
        model.register_buffer("calibration", torch.tensor([3., 4.]))
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        setattr(learner, model_name, model)
        setattr(learner, optimizer_name, optimizer)
    return learner


def assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested_equal(a, b)
    else:
        assert actual == expected


def test_model_snapshot_is_independent_and_contains_no_optimizer():
    learner = learner_fixture()
    state = inference_state(learner)
    assert set(state) == set(MODELS) | {"counts", "version", "anchors"}
    assert not set(OPTIMIZERS) & state.keys()
    original = learner.ae.weight.detach().clone()
    assert not state["ae"]["weight"].requires_grad
    state["ae"]["weight"].add_(99.)
    state["ae"]["calibration"].zero_()
    state["anchors"]["goal"].zero_()
    state["counts"]["ae"] = 1000
    assert torch.equal(learner.ae.weight, original)
    assert torch.equal(learner.ae.calibration, torch.tensor([3., 4.]))
    assert torch.equal(learner.anchors["goal"], torch.tensor([1., 2.]))
    assert learner.counts["ae"] == 7


def test_book_save_uses_compact_store_and_pins_full_snapshot_replay(tmp_path):
    from extensions.rgb_book_reproduction_v1.backend import BookRGBBackend
    from stabilization_learning.storage import load_checkpoint
    from stabilization_learning.storage.replay import ReplayRetention
    backend = BookRGBBackend.__new__(BookRGBBackend)
    backend.output, backend.deadline, backend.device = tmp_path, None, "cpu"
    backend.config = dict(epochs=2, minimum_free_gib=0,
        storage=dict(checkpoints="lossless_objects_v1", replay="active_and_restart"))
    backend.learner = learner_fixture()
    backend.learner.payload = lambda: inference_state(backend.learner) | {
        name: getattr(backend.learner, name).state_dict() for name in OPTIMIZERS}
    backend.initial_hash, backend.initial_metric_hash, backend.collection_id = "ae", "m", 1
    backend.rng = np.random.default_rng(3)
    replay_root = tmp_path / "replay"
    replay_root.mkdir()
    retention = ReplayRetention(replay_root, "active_and_restart")
    (replay_root / "bootstrap.npz").write_bytes(b"bootstrap")
    retention.register("bootstrap.npz")
    backend.replay = SimpleNamespace(rows=[{"file": "bootstrap.npz"}],
        rng=np.random.default_rng(4), retention=retention)
    ready = tmp_path / "checkpoints/ready.pt"
    result = backend.save(ready)
    assert result["storage"]["format"] == "lossless_objects_v1" and result["optimizer_complete"]
    assert retention.pinned == {"bootstrap.npz"}
    snapshot = load_checkpoint(ready)
    assert set(OPTIMIZERS) <= snapshot["learner"].keys()
    epoch = tmp_path / "checkpoints/epoch_0001.pt"
    result = backend.save(epoch)
    assert not result["optimizer_complete"] and load_checkpoint(epoch)["replay_rows"] is None
    with torch.no_grad():
        backend.learner.metric.weight.add_(1)
    backend.load(epoch)
    assert_nested_equal(inference_state(backend.learner), load_checkpoint(epoch)["learner"])


def test_restore_models_restores_buffers_and_metadata_without_optimizer_or_rng():
    learner = learner_fixture()
    state = inference_state(learner)
    optimizer_states = {name: deepcopy(getattr(learner, name).state_dict())
                        for name in OPTIMIZERS}
    with torch.no_grad():
        for name in MODELS:
            getattr(learner, name).weight.add_(20.)
            getattr(learner, name).calibration.zero_()
    learner.counts, learner.version, learner.anchors = {}, 99, {}
    rng = (torch.random.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate())
    restore_models(learner, state)
    assert_nested_equal(inference_state(learner), state)
    for name in OPTIMIZERS:
        assert_nested_equal(getattr(learner, name).state_dict(), optimizer_states[name])
    assert_nested_equal((torch.random.get_rng_state(), np.random.get_state(), random.getstate()), rng)
    assert learner.cache == {} and learner.inputs == {}
    state["counts"]["ae"] = 999
    state["anchors"]["goal"].zero_()
    assert learner.counts["ae"] == 7
    assert torch.equal(learner.anchors["goal"], torch.tensor([1., 2.]))


@pytest.mark.parametrize("bad_key", ["missing", "unexpected", "wrong_shape"])
def test_model_restore_rejects_incompatible_model(bad_key):
    learner = learner_fixture()
    state = inference_state(learner)
    if bad_key == "missing":
        del state["ae"]["weight"]
    elif bad_key == "unexpected":
        state["ae"]["surprise"] = torch.zeros(1)
    else:
        state["ae"]["weight"] = torch.zeros(9, 9)
    with pytest.raises(RuntimeError):
        restore_models(learner, state)


def row(case_id, variant="selected", plant="P0", success=True, elapsed=2.,
        reason=None, difficulty="near", room="room-a", error=1., effort=2.):
    return dict(case_id=case_id, variant=variant, plant=plant, success=success,
                elapsed_s=elapsed, reason=reason or ("success" if success else "timeout"),
                difficulty=difficulty, scene_id=room, integrated_error=error,
                input_squared_integral=effort)


def test_paired_timing_excludes_gains_losses_and_timeouts():
    rows = [row("common", elapsed=2.), row("common", "identity", elapsed=5.),
            row("gain", elapsed=4.), row("gain", "identity", success=False, elapsed=1000.),
            row("loss", success=False, elapsed=2000.), row("loss", "identity", elapsed=3.),
            row("both_fail", success=False, elapsed=3000.),
            row("both_fail", "identity", success=False, elapsed=6000.),
            row("irrelevant", plant="P1", elapsed=999.)]
    result = paired(rows, "selected", "identity", "P0")
    assert result["episodes"] == 4
    assert result["gained"] == 1 and result["lost"] == 1
    assert result["common_successes"] == 1
    assert result["mean_arrival_time_difference_seconds"] == -3.


def test_paired_returns_no_time_when_no_common_successes():
    result = paired([row("a"), row("a", "identity", success=False)],
                    "selected", "identity", "P0")
    assert result["common_successes"] == 0
    assert result["mean_arrival_time_difference_seconds"] is None


def test_paired_requires_the_same_complete_case_set():
    assert paired([row("a"), row("b", "identity")], "selected", "identity", "P0") is None
    assert paired([], "selected", "identity", "P0") is None


def test_paired_rejects_duplicate_case_instead_of_overwriting_it():
    with pytest.raises(ValueError, match="Duplicate"):
        paired([row("a"), row("a", success=False), row("a", "identity")],
               "selected", "identity", "P0")


def test_summary_keeps_variant_and_plant_denominators_separate():
    rows = [row("a", error=1.), row("b", success=False, reason="collision", error=3.,
                                  difficulty="far", room="room-b"),
            row("c", plant="P1", success=False, reason="outside", error=8.),
            row("a", "identity", error=100.), row("b", "identity", error=100.)]
    result = summarize(rows)
    assert result["episodes"] == 5
    assert result["success_rate"] == pytest.approx(1/3)
    assert result["mean_integrated_error"] == 4.
    p0 = result["groups"]["P0/selected"]
    assert p0["episodes"] == 2 and p0["successes"] == 1
    assert p0["failures"] == {"timeout": 0, "collision": 1, "outside": 0}
    assert p0["difficulty"] == {"near": {"episodes": 1, "successes": 1},
                                "far": {"episodes": 1, "successes": 0}}
    assert p0["scene_id"]["room-b"] == {"episodes": 1, "successes": 0}
    assert result["groups"]["P1/selected"]["episodes"] == 1
    assert result["groups"]["P0/identity"]["success_rate"] == 1.


def test_summary_without_selected_does_not_use_reference_as_primary():
    result = summarize([row("a", "reference_book")])
    assert result["success_rate"] is None
    assert result["mean_integrated_error"] is None


def test_book_macro_scene_selection_does_not_weight_large_rooms_more():
    result = summarize([row(str(i), room="room-a") for i in range(4)] +
                       [row("failure", success=False, room="room-b")])
    group = result["groups"]["P0/selected"]
    assert group["success_rate"] == .8
    assert group["macro_scene_success"] == .5
    assert group["worst_scene_success"] == 0.


def test_book_model_variants_preserve_the_identity_control_ablation():
    learner = learner_fixture()
    learner.initial = torch.nn.Linear(2, 2)
    backend = SimpleNamespace(learner=learner, reference_models={})
    assert _variant_models(backend, "selected") == (learner.ae, learner.metric, False)
    assert _variant_models(backend, "identity") == (learner.ae, learner.metric, True)
    assert _variant_models(backend, "shadow") == (learner.shadow, learner.metric, True)
    assert _variant_models(backend, "initial_identity") == (learner.initial, learner.metric, True)
    with pytest.raises(ValueError, match="Unknown"):
        _variant_models(backend, "misspelled")


def book_arrays():
    offset = np.zeros((4, 4), np.float32)
    offset[:, 0] = [.04, .01, .03, .01]
    return dict(offset=offset, features=np.ones((4, 16), np.float32),
                V=np.full(4, 8., np.float32), M=np.eye(16, dtype=np.float32),
                J_reference=np.zeros((16, 4), np.float32),
                action=np.array([[1., 0, 0, 0], [0, 0, 0, 1.], [0, 0, 0, 0]], np.float32),
                terminal=np.array([False, False, False]),
                truncated=np.array([False, False, True]), cost=np.ones(3, np.float32),
                rgb=np.zeros((4, 4, 4, 3), np.uint8), goal_rgb=np.zeros((4, 4, 3), np.uint8))


def test_native_trace_retains_all_endpoints_and_physical_integrals():
    arrays = book_arrays()
    trace = native_trace(arrays, .1)
    assert trace["error"] is arrays["features"]
    assert trace["J"] is arrays["J_reference"]
    assert trace["V"].shape == (4,)
    np.testing.assert_array_equal(trace["hold"], [1, 0, 1])
    np.testing.assert_allclose(trace["integrated_error"], [.00001, .00009, .00001], rtol=1e-6)
    np.testing.assert_allclose(trace["input_squared"], [.1, .1, 0.])
    assert "rgb" not in trace and "goal_rgb" not in trace
    assert "rgb" in arrays and "goal_rgb" in arrays


def test_native_trace_rejects_missing_successor():
    arrays = book_arrays()
    arrays["features"] = arrays["features"][:-1]
    with pytest.raises(ValueError, match="successor"):
        native_trace(arrays, .1)


def test_book_evaluation_saves_native_trace_and_uses_physical_arrival_time(tmp_path):
    from extensions.unified_training_v1.backends.rgb import Backend
    learner = learner_fixture()
    learner.initial, learner.device = deepcopy(learner.ae), "cpu"
    calls, events, closed = [], [], []
    instance = SimpleNamespace(close=lambda: closed.append(True))

    def episode(cfg, output, scene, goal, offset, view, depth, rng, **kwargs):
        calls.append((kwargs["mode"], kwargs["plant_name"]))
        arrays = book_arrays()
        arrays["action"][:] = 0.
        integral = native_trace(arrays, .1)["integrated_error"].sum()
        return arrays, dict(status="timeout", success=False, steps=3,
            integrated_error=float(integral), elapsed_s=99.,
            actual_V_increase_fraction=0., V_over_E_quantiles=[1., 2., 3.], latency_ms=1.)

    backend = SimpleNamespace(output=tmp_path, cfg={"plant": {"dt": .1},
        "goal_fixed": {"value_lower": .25}}, learner=learner, depth=None, seed=41,
        reference_models={}, evaluation_id=0, guard=lambda: None,
        event=lambda kind, **kw: events.append((kind, kw)),
        lookup={"room-a": {"id": "room-a", "split": "val"}},
        adapter=SimpleNamespace(joint=SimpleNamespace(episode=episode,
            new_instance=lambda cfg, scene: instance)))
    backend._audit = lambda arrays, reason: Backend._audit(backend, arrays, reason)
    cases = [dict(case_id="case-a", scene_id="room-a", goal_id=0,
                  offset=[.04, 0., 0., 0.], difficulty="near")]
    result = evaluate_cases(backend, cases, "val",
        ("selected", "identity", "shadow", "initial_identity"), ("P0", "P1"), 3)
    assert len(result["rows"]) == 8
    assert len(calls) == 8 and calls[:2] == [("learned", "P0"), ("learned", "P1")]
    assert calls[2:] == [("identity_m", p) for _ in range(3) for p in ("P0", "P1")]
    assert len(closed) == 1
    for item in result["rows"]:
        assert item["elapsed_s"] == pytest.approx(.3)
        assert item["wall_s"] == 99.
        assert item["audit"]["passed"]
        with np.load(item["trajectory"]) as trace:
            assert len(trace["V"]) == 4 and len(trace["action"]) == 3
    assert (tmp_path / "RGB_EVAL_000_val" / "RESULTS.json").is_file()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")


def frozen_fixture(tmp_path, compact=False):
    learner = learner_fixture()
    folder = tmp_path / "jobs" / "rgb-book"
    (folder / "checkpoints").mkdir(parents=True)
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {"experiment": "book-reproduction-test"})
    write_json(folder / "protocol.json", {"suite_protocol_sha256": file_hash(protocol)})
    checkpoint = folder / "checkpoints" / "selected.pt"
    if compact:
        from stabilization_learning.storage import save_checkpoint
        save_checkpoint(checkpoint, {"learner": inference_state(learner)},
                        store_root=folder / ".checkpoint_objects")
    else:
        torch.save({"learner": inference_state(learner)}, checkpoint)
    write_json(folder / "FROZEN.json", {"checkpoint": "checkpoints/selected.pt",
               "sha256": file_hash(checkpoint), "protocol_sha256": file_hash(folder / "protocol.json")})
    write_json(tmp_path / "GLOBAL_FROZEN.json", {"protocol_sha256": file_hash(protocol),
               "jobs": {folder.name: file_hash(folder / "FROZEN.json")}})
    return SimpleNamespace(output=folder, learner=learner)


@pytest.mark.parametrize("compact", [False, True])
def test_frozen_evaluation_accepts_only_the_saved_selection(tmp_path, compact):
    backend = frozen_fixture(tmp_path, compact=compact)
    verify_frozen(backend)
    with torch.no_grad():
        backend.learner.metric.weight.add_(.1)
    with pytest.raises(RuntimeError, match="differs from frozen"):
        verify_frozen(backend)


@pytest.mark.parametrize("changed", ["checkpoint", "marker", "suite", "group", "job_protocol"])
def test_frozen_evaluation_rejects_changed_provenance(tmp_path, changed):
    backend = frozen_fixture(tmp_path)
    if changed == "checkpoint":
        with (backend.output / "checkpoints" / "selected.pt").open("ab") as stream:
            stream.write(b"modified")
    elif changed == "marker":
        with (backend.output / "FROZEN.json").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    elif changed == "suite":
        write_json(tmp_path / "protocol.json", {"experiment": "changed"})
    elif changed == "job_protocol":
        write_json(backend.output / "protocol.json", {"suite_protocol_sha256": "other"})
    else:
        record = json.loads((tmp_path / "GLOBAL_FROZEN.json").read_text(encoding="utf-8"))
        record["jobs"]["unfrozen-other-job"] = "0" * 64
        write_json(tmp_path / "GLOBAL_FROZEN.json", record)
    with pytest.raises(RuntimeError):
        verify_frozen(backend)
