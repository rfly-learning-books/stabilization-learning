"""Historical RGB assets fail preflight, before any training or held-out test."""
import ast
from pathlib import Path

import pytest

from extensions.reproducible_training_v4 import assets


@pytest.fixture
def declared(tmp_path, monkeypatch):
    best = tmp_path / "assets/original_visual_initialization/best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"fixture historical checkpoint")
    digest = assets.sha256(best)
    monkeypatch.setattr(assets, "HISTORICAL_BEST_SHA256", digest)
    config = dict(jobs=[dict(id="one_rgb", backend="rgb_legacy",
        config=dict(include_historical_reference=True),
        test_variants=["selected", "historical_trained_reference", "historical_trained_identity"])])
    return tmp_path, best, digest, config


def test_expected_identity_matches_retained_evaluator_without_importing_cuda():
    source = Path(__file__).resolve().parents[1] / "extensions/unified_training_v2/historical_rgb.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    expected = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                    and t.id == "EXPECTED_BEST_SHA256" for t in node.targets))
    assert assets.HISTORICAL_BEST_SHA256 == expected


def test_single_rgb_arm_automatically_freezes_actual_loaded_asset(declared):
    project, best, digest, config = declared
    assert assets.historical_sources(project, config) == {str(best.resolve()): digest}


def test_no_historical_request_does_not_require_unused_asset(tmp_path):
    config = dict(jobs=[dict(backend="rgb_legacy", config={}, test_variants=["selected"])])
    assert assets.historical_sources(tmp_path, config) == {}


@pytest.mark.parametrize("enabled,variants,backend", [
    (False, ["selected", "historical_trained_reference"], "rgb_legacy"),
    (True, ["selected"], "rgb_legacy"),
    (1, ["selected", "historical_trained_reference"], "rgb_legacy"),
    (True, ["selected", "historical_trained_reference"], "episode_ppo"),
])
def test_flag_and_variants_must_declare_the_same_rgb_route(declared, enabled, variants, backend):
    project, _, _, config = declared
    job = config["jobs"][0]
    job.update(backend=backend, test_variants=variants)
    job["config"]["include_historical_reference"] = enabled
    with pytest.raises(ValueError):
        assets.historical_sources(project, config)


@pytest.mark.parametrize("corruption", ["missing", "wrong_bytes"])
def test_missing_or_different_best_rejected_before_initializer(declared, corruption):
    project, best, _, config = declared
    if corruption == "missing":
        best.unlink()
    else:
        best.write_bytes(b"not the historical trained model")
    with pytest.raises((ValueError, FileNotFoundError)):
        assets.historical_sources(project, config)


def test_declared_original_and_all_sources_are_resolved_and_frozen(declared):
    project, best, digest, config = declared
    original = project / "original.pt"
    original.write_bytes(best.read_bytes())
    extra = project / "中文来源.json"
    extra.write_text('{"provenance": true}', encoding="utf-8")
    config["historical_reference"] = dict(checkpoint=str(best), original_checkpoint=str(original),
        checkpoint_sha256=digest, source_paths=[str(best), str(original), str(extra)])
    expected = {str(path.resolve()): assets.sha256(path) for path in (best, original, extra)}
    assert assets.historical_sources(project, config) == expected


@pytest.mark.parametrize("problem", ["wrong_checkpoint", "wrong_original", "wrong_hash", "relative_source",
    "missing_source", "missing_actual", "missing_original", "duplicate_source", "empty_sources", "not_mapping"])
def test_incomplete_or_inconsistent_explicit_provenance_is_rejected(declared, problem):
    project, best, digest, config = declared
    original = project / "original.pt"
    original.write_bytes(best.read_bytes())
    spec = dict(checkpoint=str(best), original_checkpoint=str(original), checkpoint_sha256=digest,
                source_paths=[str(best), str(original)])
    if problem == "wrong_checkpoint":
        spec["checkpoint"] = str(original)
    elif problem == "wrong_original":
        original.write_bytes(b"wrong original")
    elif problem == "wrong_hash":
        spec["checkpoint_sha256"] = "0" * 64
    elif problem == "relative_source":
        spec["source_paths"].append("relative.json")
    elif problem == "missing_source":
        spec["source_paths"].append(str(project / "missing.json"))
    elif problem == "missing_actual":
        spec["source_paths"] = [str(original)]
    elif problem == "missing_original":
        spec["source_paths"] = [str(best)]
    elif problem == "duplicate_source":
        spec["source_paths"].append(str(best))
    elif problem == "empty_sources":
        spec["source_paths"] = []
    else:
        spec = []
    config["historical_reference"] = spec
    with pytest.raises((ValueError, FileNotFoundError)):
        assets.historical_sources(project, config)
