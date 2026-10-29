"""Scientific plotting invariants; synthetic fixtures are not training results."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.book_multiseed_v4.figures import (
    aggregate_curves, build_figures, result_summary, training_series,
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_seed_aggregation_keeps_missing_points_missing_and_uses_sample_sd():
    rows = aggregate_curves({41: [(1, 2.), (2, 9.)], 42: [(1, 4.)]})
    assert rows[0] == dict(update=1, mean=3., std=np.sqrt(2), n=2)
    assert rows[1] == dict(update=2, mean=9., std=None, n=1)
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_curves({41: [(1, 1.), (1, 3.)]})


def results_fixture():
    seeds = {}
    for seed, cost, qualified in ((41, -2., True), (42, 4., False)):
        seeds[str(seed)] = dict(
            selection=dict(updates=2, best_update=1),
            test_gate=dict(qualified=qualified, reasons=[] if qualified else ["physical_cost:2.0"]),
            audit_passed=True, M_state_variation=.2, action_difference_from_P0=.1,
            aliases=dict(initial="fixed_P0", classical_SDRE="fixed_P0"),
            job_path=f"jobs/point_mass_seed{seed}", physical=dict(dt=.1, horizon=3),
            comparisons=[dict(scale=2., cost_percent=cost, baseline_success=2,
                learned_success=2 if qualified else 1, episodes=2, arrival_delay=.01,
                energy_ratio=1.03, max_normalized_decay_rate=-.1,
                cost_vs_classical_SDRE_percent=cost)])
    return dict(schema="book_multiseed_v4_results", tasks={"point_mass": dict(seeds=seeds)})


def test_failed_test_seed_is_kept_and_never_replaced_by_best_seed():
    summary = result_summary(results_fixture())["point_mass"]
    metric = summary["groups"][0]["metrics"]["cost_percent"]
    assert metric["values"] == [-2., 4.]
    assert metric["n"] == 2 and metric["mean"] == 1.
    assert summary["seeds"]["42"]["test_passed"] is False
    assert summary["seeds"]["42"]["reasons"] == ["physical_cost:2.0"]


def test_rejected_validation_candidates_are_plotted_without_selection_filter():
    events = [dict(stage="update_committed", task="point_mass", seed=41, update=1,
                   loss=7., validation=dict(qualified=False, selected=False)),
              dict(stage="uncommitted", task="point_mass", seed=41, update=2, loss=0.)]
    assert training_series(events, "point_mass", lambda e:e["loss"]) == {41:[(1,7.)]}


def trace_fixture(root, role):
    path = root/"jobs"/"point_mass_seed41"/"point_mass"/"test"/f"2.0_{role}"
    path.mkdir(parents=True)
    states = np.asarray([[[1., .5]], [[.8, .4]], [[.64, .32]]])
    following = states*.8
    metric = np.broadcast_to(np.eye(2), (3,1,2,2)).copy()
    if role == "learned":
        metric *= (1+.1*np.square(states).sum(-1))[...,None,None]
        next_metric = np.broadcast_to(np.eye(2), (3,1,2,2)).copy()
        next_metric *= (1+.1*np.square(following).sum(-1))[...,None,None]
    else:
        next_metric = metric.copy()
    before = np.einsum("tbi,tbij,tbj->tb",states,metric,states)
    after = np.einsum("tbi,tbij,tbj->tb",following,next_metric,following)
    np.savez_compressed(path/"trace.npz", states=states, following=following,
        M_current=metric, M_next=next_metric, V_before=before, V_after=after,
        actions=np.ones((3,1,1)),active=np.ones((3,1), dtype=bool))
    digest = hashlib.sha256((path/"trace.npz").read_bytes()).hexdigest()
    dump(path/"GROUP.json", dict(seeds=[220000000], trace_sha256=digest))
    dump(path/"AUDIT.json", dict(passed=True))
    return path


def test_build_figures_requires_selection_commit_and_preserves_trace_evidence(tmp_path):
    result = results_fixture()
    with pytest.raises(ValueError, match="committed"):
        build_figures(tmp_path, result)
    dump(tmp_path/"GLOBAL_SELECTION_COMMITTED.json", dict(test_access_at_selection=False))
    for seed in (41,42):
        for update in (1,2):
            validation = dict(qualified=seed==41, selected=update==1, reasons=[],
                probe_max_normalized_decay_rate=-.1,
                groups=[dict(scale=2.,cost_ratio=.98 if seed==41 else 1.04,
                             successes=2,episodes=2,arrival_delay=.01,energy_ratio=1.03)])
            dump(tmp_path/"events"/f"{seed}_{update}.json", dict(stage="update_committed",
                task="point_mass",seed=seed,update=update,validation=validation,
                loss=3.,physical_training_cost=2.,constraint=.001,gradient_norm=.1,
                model_transitions=update*16*3,wall_seconds=.1,best_update=1))
    for role in ("fixed_P0","learned"):
        trace_fixture(tmp_path,role)
    returned = build_figures(tmp_path,result)
    assert returned["figure_count"] == 4
    manifest = json.loads((tmp_path/"FIGURES.json").read_text(encoding="utf-8"))
    assert len(manifest["illustrations"][0]["groups"]) == 2  # SDRE=P0, not three arms
    assert manifest["illustrations"][0]["training_seed"] == 41
    assert manifest["illustrations"][0]["lane"] == 0
    assert manifest["summary"]["point_mass"]["seeds"]["42"]["test_passed"] is False
    for figure in manifest["figures"]:
        for output in figure["files"].values():
            path = tmp_path/output["path"]
            assert path.stat().st_size > 100
            assert hashlib.sha256(path.read_bytes()).hexdigest() == output["sha256"]
    report = (tmp_path/"REPORT.md").read_text(encoding="utf-8")
    assert "physical_cost:2.0" in report
    assert "未通过" in report and "不是置信区间" in report


def test_bad_trace_digest_is_rejected_before_trajectory_plotting(tmp_path):
    from extensions.book_multiseed_v4.figures import _plot_api, _trajectory_figures
    path = trace_fixture(tmp_path,"fixed_P0")
    dump(path/"GROUP.json", dict(seeds=[220000000],trace_sha256="changed"))
    with pytest.raises(ValueError, match="Unsealed"):
        _trajectory_figures(_plot_api(),tmp_path,tmp_path,results_fixture(),[])
