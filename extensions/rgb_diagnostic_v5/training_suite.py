"""Four-arm short repair with group freeze before validation.

Uses one GPU lease and one read-only archive/hash cache.  All original models,
archive, tests and STOP files remain untouched.  This is warm-started development
research, not a new from-scratch result or an independent test evaluation.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
import time
import traceback

from .common import (bootstrap, gpu_lease, freeze_stage, read_json, write_json,
                     sha256, guard, event)
from .training_ablation import (ReadOnlyV4Replay, train_arm, parameter_digest,
                               precision_floor_from_task, ARMS)
from .control_ablation import run_ablation
from extensions.joint_certificate_v3.rgb import _new_learner, _restore_for_evaluation
from stabilization_learning.joint_v2.runtime import seed_all
import torch


def run(root):
    root = Path(root).resolve()
    guard(root)
    protocol = read_json(root/"PROTOCOL.json")
    authorization = read_json(root/"TRAINING_AUTHORIZATION.json")
    arms = protocol["conditional_training"]["arms"]
    if not authorization.get("authorized") or set(authorization.get("arms", [])) != set(arms):
        raise RuntimeError("All declared training arms need root's evidence-based authorization")
    if set(arms) != set(ARMS):
        raise ValueError("This wrapper requires the complete predeclared two-by-two design")
    plan_path = root/"TRAINING_KEYS.json"
    plan = read_json(plan_path)
    if (plan["pairs"] != protocol["conditional_training"]["updates_per_arm"]["feature"]
            or plan["pairs"] != protocol["conditional_training"]["updates_per_arm"]["metric"]
            or plan["batch_size"] != protocol["conditional_training"]["batch_size"]):
        raise ValueError("Shared key plan and declared training budgets differ")
    if any((root/"training"/arm).exists() for arm in arms) or (root/"training_evaluation").exists():
        raise FileExistsError("No restart or overwrite of a partial training/evaluation group")
    sources = [Path(__file__), Path(__file__).with_name("training_ablation.py"),
               Path(__file__).with_name("control_ablation.py")]
    frozen = freeze_stage(root, "conditional_training_suite",
        dict(arms=arms, pairs=plan["pairs"], batch_size=plan["batch_size"],
             key_plan_sha256=sha256(plan_path), authorization_sha256=sha256(root/"TRAINING_AUTHORIZATION.json"),
             order="train and freeze all four arms, then evaluate each final model",
             evaluated_methods=["baseline", "identity_M"], plants=["P0", "P1"],
             validation_cases=48, max_steps=200, expected_evaluation_episodes=768,
             scientific_scope="short warm-start repairs can test compensation by loss/schedule, not prove geometry repaired"),
        sources)
    def check_inputs():
        guard(root)
        for path, expected in frozen["sources"].items():
            if sha256(path) != expected:
                raise RuntimeError("Frozen diagnostic source changed: " + path)
        if sha256(plan_path) != frozen["settings"]["key_plan_sha256"]:
            raise RuntimeError("Shared immutable key plan changed")
    replay = ReadOnlyV4Replay(plan["replay_index"])
    if replay.index_sha256 != plan["replay_index_sha256"]:
        raise RuntimeError("Original replay index hash mismatch")
    all_frozen, evaluations = [], []
    begun = time.perf_counter()
    try:
        with gpu_lease(root, "conditional_training_suite"):
            ctx = bootstrap(root)
            floor = precision_floor_from_task(ctx.cfg)
            baseline_hashes = {k: parameter_digest(getattr(ctx.learner, k)) for k in ("features", "metric")}
            for index, arm in enumerate(arms):
                check_inputs()
                seed_all(protocol["seed"])
                if index == 0:
                    base = ctx.learner
                else:
                    base = _new_learner(ctx.cfg, ctx.config)
                    _restore_for_evaluation(base, ctx.checkpoint)
                actual = {k: parameter_digest(getattr(base, k)) for k in ("features", "metric")}
                if actual != baseline_hashes:
                    raise RuntimeError("Arm did not start from identical selected model tensors")
                trained, result = train_arm(base, replay, ctx.geometry, plan, root/"training"/arm,
                    arm=arm, precision_floor=floor, guard=check_inputs, event=ctx.event,
                    source=ctx.provenance)
                if result["initial_hashes"] != baseline_hashes:
                    raise RuntimeError("Learner construction changed selected initialization")
                all_frozen.append(dict(arm=arm, checkpoint=result["checkpoint"], sha256=result["sha256"],
                                       counts=result["counts"], elapsed_s=result["elapsed_s"],
                                       marker=str(root/"training"/arm/"FROZEN.json"),
                                       marker_sha256=sha256(root/"training"/arm/"FROZEN.json")))
                ctx.learner = None
                del trained, base, result
                gc.collect(); torch.cuda.empty_cache()
            # This group boundary must precede the first validation rollout.
            for row in all_frozen:
                if sha256(row["checkpoint"]) != row["sha256"] or sha256(row["marker"]) != row["marker_sha256"]:
                    raise RuntimeError("A frozen repair checkpoint or marker changed")
                if row["counts"]["feature"] != plan["pairs"] or row["counts"]["metric"] != plan["pairs"]:
                    raise RuntimeError("The complete training group is not ready")
            write_json(root/"TRAINING_GROUP_FROZEN.json", dict(arms=all_frozen,
                       key_plan_sha256=sha256(plan_path), initial_hashes=baseline_hashes,
                       final_only=True, validation_not_started=True))
            ctx.event("training_group_frozen", arms=arms, counts=plan["pairs"])
            for row in all_frozen:
                check_inputs()
                if sha256(row["checkpoint"]) != row["sha256"]:
                    raise RuntimeError("Final model changed before validation")
                learner = _new_learner(ctx.cfg, ctx.config)
                ck = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
                _restore_for_evaluation(learner, ck)
                learner.features.eval(); learner.metric.eval()
                out = root/"training_evaluation"/row["arm"]
                ctx.event("repair_validation_started", arm=row["arm"], source_sha256=row["sha256"])
                evaluated = run_ablation(ctx, out, learner=learner,
                                        methods=("baseline", "identity_M"), plants=("P0", "P1"),
                                        max_steps=200, verify_original_baseline=False)
                expected = {p+"/"+m for p in ("P0", "P1") for m in ("baseline", "identity_M")}
                if (set(evaluated["summaries"]) != expected
                        or any(value["episodes"] != 48 for value in evaluated["summaries"].values())):
                    raise RuntimeError("Incomplete final validation population")
                evaluations.append(dict(arm=row["arm"], results=str(out/"RESULTS.json"),
                                        sha256=sha256(out/"RESULTS.json"), episodes=192))
                ctx.event("repair_validation_complete", arm=row["arm"], results=str(out/"RESULTS.json"))
                del learner, ck, evaluated
                gc.collect(); torch.cuda.empty_cache()
            check_inputs()
            result = dict(schema="rgb_v5_training_suite_complete", status="complete",
                          arms=all_frozen, evaluation=evaluations,
                          updates_per_arm=dict(feature=plan["pairs"], metric=plan["pairs"]),
                          expected_evaluation_episodes=768, elapsed_s=time.perf_counter()-begun,
                          completed_evaluation_episodes=sum(r["episodes"] for r in evaluations),
                          group_frozen_before_validation=True, no_intermediate_selection=True,
                          key_plan_sha256=sha256(plan_path), source=replay.source_summary(),
                          limitation="Existing v4 model and training replay reused; validation development only, not independent held-out testing.")
            write_json(root/"TRAINING_SUITE_COMPLETE.json", result)
            ctx.event("training_suite_complete", arms=arms, elapsed_s=result["elapsed_s"])
            return result
    except BaseException as exc:
        event(root, "training_suite_failed", error=repr(exc), completed_training_arms=[r["arm"] for r in all_frozen],
              completed_evaluation_arms=[r["arm"] for r in evaluations])
        write_json(root/"TRAINING_SUITE_FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc(),
                   completed_training_arms=all_frozen, completed_evaluation_arms=evaluations,
                   automatic_resume=False, elapsed_s=time.perf_counter()-begun))
        raise
    finally:
        replay.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args=parser.parse_args()
    summary=run(args.root)
    print(json.dumps({"status":summary["status"],"arms":[r["arm"] for r in summary["arms"]],
                      "elapsed_s":summary["elapsed_s"]},ensure_ascii=False))
