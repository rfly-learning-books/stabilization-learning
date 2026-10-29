"""Independent CPU arithmetic audit of completed frozen-control diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np

from .common import read_json, sha256, write_json


def describe(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return dict(n=0, mean=None, median=None, q10=None, q90=None)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                q10=float(np.quantile(x, .1)), q90=float(np.quantile(x, .9)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--process-snapshot", required=True, type=Path)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.root.resolve() / "control"
    snapshot = read_json(args.process_snapshot)
    if not snapshot.get("query_success") or not snapshot.get("control_writer_absent"):
        raise RuntimeError("Successful external process snapshot is required")
    if not 0 <= time.time() - snapshot["created_unix"] < 60:
        raise RuntimeError("External process snapshot must be less than 60 seconds old")
    for process in snapshot["processes"]:
        command = process.get("CommandLine")
        if not command:
            raise RuntimeError("A Python process has no command line")
        if "extensions.rgb_diagnostic_v5.control_ablation" in command:
            raise RuntimeError("Control writer is still active")
    frozen = read_json(root / "stage_protocols/control_ablation.json")
    for source, digest in frozen["sources"].items():
        if sha256(source) != digest:
            raise RuntimeError("Control stage source changed: " + source)
    protocol = read_json(root / "PROTOCOL.json")
    old = Path(protocol["old_root"])
    assert sha256(old / protocol["checkpoint"]) == protocol["checkpoint_sha256"]
    assert sha256(root / "PROTOCOL.json") == frozen["protocol_sha256"]
    criteria = protocol["control"]["success"]
    expected_cases = {c["case_id"] for c in read_json(root / "VALIDATION_CASES.json")}
    # Read only immutable events for this already-completed stage, not any
    # other stage's variable checkpoint/results while it may be running.
    events = [read_json(p) for p in (root / "events").glob("*.json")]
    control = [e for e in events if e.get("stage") == "control"]
    first_evaluation = min(e["unix"] for e in control if e.get("kind") == "control_variant_started")
    assert frozen["created_unix"] < first_evaluation
    complete_events = [e for e in control if e.get("kind") == "control_variant_complete"]
    assert len(complete_events) == 10
    maxima = {key: 0. for key in ("action", "V", "physical_error", "integrated_error", "input_integral")}
    values = defaultdict(list)
    initial = {}
    groups = {}
    episodes = transitions = successes = 0
    for path in sorted(output.glob("*/*/RESULTS.json")):
        result = read_json(path)
        rows = result["rows"]
        assert len(rows) == 48 and {r["case_id"] for r in rows} == expected_cases
        group = path.parent.parent.name + "/" + path.parent.name
        groups[group] = dict(episodes=48, successes=sum(r["success"] for r in rows))
        for row in rows:
            trajectory = Path(row["trajectory"])
            assert sha256(trajectory) == row["trajectory_sha256"]
            with np.load(trajectory, allow_pickle=False) as data:
                a = {k: np.asarray(data[k]) for k in data.files}
            n = row["steps"]
            e, J, M = (a[key].astype(float) for key in ("error", "J", "M"))
            assert e.shape == (n+1, 16) and J.shape == (n, 16, 4)
            assert len(a["dt"]) == n and (a["dt"] > 0).all()
            assert not a["terminated"][:-1].any() and not a["truncated"][:-1].any()
            eigenvalues = np.linalg.eigvalsh(M)
            assert eigenvalues[0] >= .1-1e-5 and abs(np.trace(M)-16) < 1e-4
            if row["method"] == "identity_M":
                assert np.allclose(M, np.eye(16), rtol=0, atol=1e-7)
            G = J.transpose(0, 2, 1) @ M @ J
            h = G.copy()
            index = np.arange(4)
            h[:, index, index] += .01 * G[:, index, index]
            damping = (.02*np.trace(G, axis1=1, axis2=2)/4+1e-8
                       if row["method"] == "relative_damping" else np.full(n, .02))
            h[:, index, index] += damping[:, None]
            rhs = (J.transpose(0, 2, 1) @ M @ e[:-1, :, None])[..., 0]
            raw = -.7 * np.linalg.solve(h, rhs[..., None])[..., 0]
            radius = np.maximum(np.linalg.norm(raw[:, :3], axis=1), np.abs(raw[:, 3]))
            action = raw / np.maximum(1, radius)[:, None]
            maxima["action"] = max(maxima["action"], float(np.max(np.abs(action-a["action"]))))
            assert np.allclose(action, a["action"], rtol=5e-5, atol=5e-5)
            V = .5 * np.einsum("ti,ij,tj->t", e, M, e)
            maxima["V"] = max(maxima["V"], float(np.max(np.abs(V-a["V"]))))
            assert np.allclose(V, a["V"], rtol=1e-5, atol=1e-5)
            q = a["offset"].astype(float)
            E = np.sum(q[:, :3]**2, axis=1) + (q[:, 3]/math.radians(25))**2
            maxima["physical_error"] = max(maxima["physical_error"], float(np.max(np.abs(E-a["physical_error"]))))
            integral = float((E[1:]*a["dt"]).sum())
            effort = float((a["action"].astype(float)**2).sum(axis=1) @ a["dt"].astype(float))
            maxima["integrated_error"] = max(maxima["integrated_error"], abs(integral-row["integrated_error"]))
            maxima["input_integral"] = max(maxima["input_integral"], abs(effort-row["input_squared_integral"]))
            assert np.isclose(integral, row["integrated_error"], rtol=1e-6, atol=1e-7)
            assert np.isclose(effort, row["input_squared_integral"], rtol=1e-6, atol=1e-7)
            tolerance = ((np.linalg.norm(q[:, :3], axis=1) < criteria["position_tolerance_m"])
                         & (np.abs(q[:, 3]) < math.radians(criteria["yaw_tolerance_deg"])))
            count, counts = 0, []
            for good in tolerance[1:]:
                count = count+1 if good else 0
                counts.append(count)
            assert np.array_equal(counts, a["hold_observations"])
            if row["success"]:
                assert counts[-1] >= criteria["required_hold_observations"] and a["success"][-1] and a["terminated"][-1]
                assert row["reason"] == "success"
            assert not a["success"][:-1].any()
            initial[(row["plant"], row["method"], row["case_id"])] = dict(error=e[0], M=M, offset=q[0], version=row["version"])
            valid = E > 1e-12
            near = ((np.linalg.norm(q[:, :3], axis=1) < .1)
                    & (np.abs(q[:, 3]) < math.radians(5)) & ~tolerance)
            start = np.arange(len(E)) == 0
            outcome = "success" if row["success"] else "failure"
            feature_energy = np.sum(e**2, axis=1)
            numerical_tolerance = 1e-7 + 1e-4*E
            # The attainable envelope allows a DIFFERENT optimal M per state.
            # It is only a necessary feasibility condition, not a common-goal fit.
            features = np.column_stack((V/np.maximum(E, 1e-30),
                V < .1*E-numerical_tolerance,
                V > 4*E+numerical_tolerance,
                7.25*feature_energy < .1*E,
                7.25*feature_energy < .1*E-numerical_tolerance,
                .05*feature_energy > 4*E,
                .5*feature_energy/np.maximum(E, 1e-30)))
            masks = {"initial_all": start, "initial_"+outcome: start,
                     "all_states": np.ones(len(E), dtype=bool), "states_"+outcome: np.ones(len(E), dtype=bool),
                     "near_outside_success": near, "near_"+outcome: near}
            for name, mask in masks.items():
                values[group+"/"+name].append(features[mask & valid])
            episodes += 1
            transitions += n
            successes += row["success"]
    assert episodes == 480 and len(groups) == 10
    max_initial_feature = max_initial_metric = max_initial_pose = 0.
    for (plant, method, case), record in initial.items():
        baseline = initial[(plant, "baseline", case)]
        max_initial_feature = max(max_initial_feature, float(np.max(np.abs(record["error"]-baseline["error"]))))
        max_initial_pose = max(max_initial_pose, float(np.max(np.abs(record["offset"]-baseline["offset"]))))
        assert record["version"] == baseline["version"]
        assert np.allclose(record["error"], baseline["error"], rtol=1e-6, atol=1e-7)
        assert np.array_equal(record["offset"], baseline["offset"])
        if method != "identity_M":
            max_initial_metric = max(max_initial_metric, float(np.max(np.abs(record["M"]-baseline["M"]))))
            assert np.array_equal(record["M"], baseline["M"])
    summaries = {}
    for key, blocks in sorted(values.items()):
        x = np.concatenate(blocks, axis=0)
        summaries[key] = dict(states=len(x), value_over_E=describe(x[:, 0]),
            lower_bound_violation_fraction=float(x[:, 1].mean()) if len(x) else None,
            upper_bound_violation_fraction=float(x[:, 2].mean()) if len(x) else None,
            lower_bound_impossible_fraction=float(x[:, 3].mean()) if len(x) else None,
            lower_bound_impossible_with_margin_fraction=float(x[:, 4].mean()) if len(x) else None,
            upper_bound_impossible_fraction=float(x[:, 5].mean()) if len(x) else None,
            identity_value_over_E_on_same_states=describe(x[:, 6]))
    result = dict(passed=True, episodes=episodes, transitions=transitions, strict_success_episodes=successes,
        scope="completed control stage only; independent CPU arithmetic; no new policy rollouts or weight changes",
        process_snapshot=str(args.process_snapshot), audit_source=__file__, audit_source_sha256=sha256(__file__),
        input_results_sha256=sha256(output/"RESULTS.json"), groups=groups,
        maximum_absolute_differences=maxima,
        source_and_coordinate_checks=dict(stage_frozen_unix=frozen["created_unix"], first_evaluation_unix=first_evaluation,
            frozen_before_evaluation=True, stage_sources_match=True, selected_checkpoint_matches=True,
            same_case_initial_features_max_difference=max_initial_feature,
            same_case_initial_M_max_difference_except_identity=max_initial_metric,
            same_case_initial_pose_max_difference=max_initial_pose,
            limitation="Archived e and M are independently recomputed into V; images are not rerendered/reencoded in this CPU audit. Fixed parameters/normalization are enforced by frozen inference version checks and frozen selected checkpoint provenance."),
        stratified=summaries,
        definitions=dict(value="0.5 e.T M e, same fixed M at both endpoints",
            physical_error="||position_error_m||^2+(yaw_error_rad/radians(25))^2",
            near="position<0.1m and abs(yaw)<5deg, excluding original success tolerance; descriptive, not a success change",
            all_states="includes terminal; time-correlated, longer episodes have greater weight",
            initial="one state per declared episode; methods and P0/P1 reuse the same starts",
            outcomes="entire trajectory grouped by eventual episode success/failure, not randomized strata",
            feasible_envelope="0.05||e||^2 <= V <= 7.25||e||^2 from trace16/eigenfloor0.1; necessary only",
            numerical_margin="1e-7+1e-4 E", no_confidence_interval=True))
    write_json(output/"CONTROL_SCIENTIFIC_AUDIT.json", result)
    print(json.dumps(dict(passed=True, episodes=episodes, transitions=transitions,
                         maxima=maxima, source_and_coordinate_checks=result["source_and_coordinate_checks"]), indent=2))


if __name__ == "__main__":
    main()
