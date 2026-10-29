"""Training-room-only checks for the new collector; no validation/test rollout."""
from pathlib import Path
from copy import deepcopy
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/"stabilization-gym", ROOT/"stabilization-learning-control"):
    sys.path.insert(0, str(path))


def main():
    import numpy as np
    import torch
    from extensions.rgb_longroll_v1.backend import LongRollRGBBackend, phase_ids
    from extensions.unified_training_v1.backends.rgb import _module_hash, _write
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = deepcopy(config["jobs"][0]["config"])
    cfg["train_scene_count"] = 4
    cfg["retention_diagnostic"] = True
    args.output.mkdir(parents=True)
    training_cases = dict(val=[dict(case_id="training_preflight", scene_id="classroom_train_00", goal_id=0,
                                  difficulty="near_training_only", offset=[.01,.005,0.,.001], split="train")])
    training_cases["test"] = deepcopy(training_cases["val"])
    manifest = args.output/"TRAINING_ONLY_CASES.json"
    _write(manifest, training_cases)
    cfg["case_manifest"] = str(manifest.resolve())
    cfg["previous_cases"] = str(manifest.resolve())
    folder = args.output/"jobs"/"probe"
    backend = LongRollRGBBackend(cfg, folder, deadline=time.time()+12*60,
        event=lambda kind, **data: print(json.dumps(dict(event=kind, **{k:v for k,v in data.items() if k in ("collection", "remaining_transitions", "fresh_transitions")})), flush=True))
    begin = time.time()
    try:
        backend.initialize()
        ids = [s["id"] for s in backend.train_scenes]
        cfg = backend.config
        cfg["regular_collection_quotas"] = [{sid:384 for sid in ids} for _ in range(24)]
        initial = backend.collect(initial=True)
        long = backend.collect(initial=False)
        assert long["fresh_transitions"] == 1536
        assert sum(long["phase_counts"].values()) == 1536
        assert max(r["transitions"] for r in long["coverage"]) <= 120
        assert any(r["transitions"] > 24 for r in long["coverage"])
        terminal_rows = 0
        for row in backend.replay.rows:
            with np.load(backend.replay.root/row["file"], allow_pickle=False) as a:
                if "phase" not in a:
                    continue
                length = len(a["action"])
                assert len(a["offset"]) == len(a["rgb"]) == length+1
                assert not a["terminal"][:-1].any() and not a["truncated"][:-1].any()
                assert len(a["smooth_step"]) <= (length-1)//10+1
                terminal_rows += 1
        before = _module_hash(backend.learner.ae)
        backend.warmup_representation(2)
        backend.pretrain_evaluator(2)
        backend.train_epoch(epoch=1, representation_updates=2, evaluator_updates=0, certificate_updates=2)
        after = _module_hash(backend.learner.ae)
        assert before != after
        scene = backend.train_scenes[0]
        case = dict(scene_id=scene["id"], goal_id=0, offset=[.08, .03, .02, .02])
        env = backend._env([case], max_steps=2)
        try:
            public, _ = env.reset(options=dict(offsets=[case["offset"]], goal_ids=[0]))
            first = backend._policy(public)(public)
            checkpoint = folder/"preflight.pt"
            backend.save(checkpoint)
            backend.load(checkpoint)
            restored = backend._policy(public)(public)
            assert torch.allclose(first.action, restored.action, atol=5e-6, rtol=1e-5)
            assert torch.allclose(first.value, restored.value, atol=2e-5, rtol=1e-5)
        finally:
            env.close()
        rejected = False
        try:
            backend.evaluate("test", ("selected",))
        except RuntimeError as error:
            rejected = "freeze" in str(error).lower()
        assert rejected, "Test must reject an unfrozen preflight"
        # Exercise every reference and retention branch on one TRAINING start.
        from stabilization_learning.unified_v1.runner import sha256
        _write(args.output/"protocol.json", dict(scope="training-room-only preflight"))
        suite_hash = sha256(args.output/"protocol.json")
        _write(folder/"protocol.json", dict(suite_protocol_sha256=suite_hash, scope="training-only"))
        _write(folder/"FROZEN.json", dict(checkpoint="preflight.pt", sha256=sha256(checkpoint),
            protocol_sha256=sha256(folder/"protocol.json")))
        _write(args.output/"GLOBAL_FROZEN.json", dict(schema="unified_frozen_group_v1",
            protocol_sha256=suite_hash, jobs={"probe":sha256(folder/"FROZEN.json")}))
        evaluated = backend.evaluate("test", tuple(config["jobs"][0]["test_variants"]))
        assert evaluated["summary"]["episodes"] == 16
        assert len(evaluated["retention"]["rows"]) == 10
        assert all(row["audit"]["passed"] for row in evaluated["rows"]+evaluated["retention"]["rows"])
        assert phase_ids(np.array([[0,0,0,0],[.03,0,0,0],[.15,0,0,0],[.5,0,0,0]]), .02, np.pi/180).tolist() == [0,1,2,3]
        result = dict(passed=True, training_rooms=ids, initial_transitions=initial["fresh_transitions"],
            long_transitions=long["fresh_transitions"], phase_counts=long["phase_counts"],
            episode_lengths=[r["transitions"] for r in long["coverage"]], audited_replay_episodes=terminal_rows,
            strict_success_episodes=sum(r["first_strict_success_step"] is not None for r in long["coverage"]),
            post_success_transitions=sum(r["post_success_transitions"] for r in long["coverage"]),
            parameter_update_observed=before!=after, save_load_action_and_value=True,
            frozen_reference_training_cases=evaluated["summary"]["episodes"],
            retention_training_cases=len(evaluated["retention"]["rows"]),
            unfrozen_test_rejected=rejected, wall_seconds=time.time()-begin,
            scope="training rooms only; no test-driven configuration choice")
        _write(args.output/"PREFLIGHT_RESULT.json", result)
        print(json.dumps(result), flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
