"""Extend v1 report with full Push updates and reference provenance audit."""
from pathlib import Path
from stabilization_learning.unified_v1 import reporting as v1


def compare_rgb_cases(root, protocol):
    """Compare recorded starts across the two arms only after writer exit."""
    jobs = [j for j in protocol["config"]["jobs"] if j["backend"] == "rgb_legacy"]
    if len(jobs) != 2:
        return dict(status="not_checkable", reason="Expected two declared RGB arms")
    paths = [root / "jobs" / j["id"] / "RGB_CASES.json" for j in jobs]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        return dict(status="not_checkable", reason="A declared arm did not write its cases", missing=missing)
    cases = [v1.read(p) for p in paths]
    comparisons = {}
    for split in ("val", "test"):
        a, b = cases[0].get(split), cases[1].get(split)
        key = "validation_cases" if split == "val" else "test_cases"
        expected = [j.get("config", {}).get(key) for j in jobs]
        valid = all(isinstance(x, list) for x in (a, b))
        comparisons[split] = dict(equal=bool(valid and a == b),
                                  actual_counts=[len(x) if isinstance(x, list) else None for x in (a,b)],
                                  expected_counts=expected,
                                  counts_match=bool(valid and len(a) == expected[0] and len(b) == expected[1]))
    passed = all(x["equal"] and x["counts_match"] for x in comparisons.values())
    return dict(status="matched" if passed else "mismatch", arms=[j["id"] for j in jobs],
                cases_sha256=[v1.sha256(p) for p in paths], splits=comparisons)


def report_suite(root, output, process_snapshot):
    original_curves = v1._curves
    original_evaluate = v1._evaluate
    original_integrity = v1._integrity
    def curves(folder, backend, stages):
        result = original_curves(folder, backend, stages)
        if backend == "push_discrete_sac":
            rows = v1._jsonl(folder / "push_all_updates.jsonl")
            result["all_optimizer_calls"] = rows
            result["all_optimizer_calls_scope"] = "Individual critic and actor calls including pretraining and skipped actor calls"
            result["epoch_averages"] = result["training"]
            result["training"] = [dict(actual_step=r["counts"]["evaluator_updates" if r["kind"] == "critic" else "certificate_updates"],
                                       stage=r["stage"], kind=r["kind"], epoch=r["epoch"],
                                       **{k: v for k, v in r["result"].items() if not (k == "actor_loss" and r["result"].get("actor_skipped"))})
                                  for r in rows]
        return result
    def evaluate(folder, job, test):
        result = original_evaluate(folder, job, test)
        if job["backend"] != "rgb_legacy" or not job.get("config", {}).get("include_historical_reference"):
            return result
        protocol = v1.read(folder.parent.parent / "protocol.json")
        spec = protocol["config"]["historical_reference"]
        directory = v1._safe_path(folder, test["folder"])
        reference = v1.read(directory / "FROZEN.json").get("historical_reference") or {}
        problems = []
        if reference.get("checkpoint_sha256") != spec["checkpoint_sha256"]:
            problems.append("historical checkpoint hash differs from suite protocol")
        if not reference.get("ae_sha256") or not reference.get("metric_sha256"):
            problems.append("historical trained model hashes missing")
        for name in ("historical_trained_reference", "historical_trained_identity"):
            for plant in ("P0", "P1"):
                group = result["groups"].get(f"{plant}/{name}", {})
                if group.get("episodes") != job["config"]["test_cases"]:
                    problems.append(f"incomplete {plant}/{name}")
        result["historical_reference_audit"] = dict(passed=not problems, problems=problems, frozen_provenance=reference)
        if problems:
            result["postprocess_numeric_audit"]["passed"] = False
            result["postprocess_numeric_audit"]["errors"].extend(problems)
        return result
    def integrity(root, protocol, final):
        result, events = original_integrity(root, protocol, final)
        comparison = compare_rgb_cases(root, protocol)
        result["rgb_cross_arm_case_comparison"] = comparison
        if comparison["status"] == "mismatch" or (comparison["status"] == "not_checkable" and final.get("status") == "complete"):
            result["problems"].append("RGB arm cases differ, have unexpected counts, or cannot be verified in a completed run")
            result["passed"] = False
        return result, events
    v1._curves = curves
    v1._evaluate = evaluate
    v1._integrity = integrity
    try:
        return v1.report_suite(root, output, process_snapshot)
    finally:
        v1._curves = original_curves
        v1._evaluate = original_evaluate
        v1._integrity = original_integrity
