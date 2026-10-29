"""Bounded state-conditioned M training/evaluation for the DC motor.

This is the first seed41 pilot.  It is deliberately one-step differentiable
training; the deployed evaluation uses the hard candidate CLF screen.  It is
not a formal multi-seed study and not a stability proof.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from extensions.state_conditioned_m_v1.core.metric import StateDependentMetricActor
from extensions.state_conditioned_m_v1.core.identity import IdentityMetricActor
from extensions.state_conditioned_m_v1.core.objectives import dc_motor_one_step_loss
from extensions.state_conditioned_m_v1.core.rollout import rollout_dc_motor
from extensions.state_conditioned_m_v1.envs.dc_motor import StateDependentDCMotorEnv
from stabilization_gym.common.types import MetricSpec


def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def other_writer():
    cmd = ("@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and "
           "$_.CommandLine -match 'train_book_m_suite.py' } | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress")
    txt = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True).strip()
    if not txt:
        return []
    value = json.loads(txt)
    return value if isinstance(value, list) else [value]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--allow-active-writer", action="store_true")
    args = parser.parse_args()
    cfg = read(args.config)
    if args.engineering_smoke:
        cfg["updates"] = min(int(cfg["updates"]), 5)
        cfg["batch_states"] = min(int(cfg["batch_states"]), 8)
        cfg["test_seed_count"] = min(int(cfg["test_seed_count"]), 4)
    if other_writer() and not args.allow_active_writer:
        raise RuntimeError("Another formal writer is active; refuse concurrent training")
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.manual_seed(int(cfg["train_seed"]))
    np.random.seed(int(cfg["train_seed"]))
    random.seed(int(cfg["train_seed"]))
    torch.set_num_threads(1)
    spec = MetricSpec(error_dim=2, eigen_floor=.05, trace=2.)
    env = StateDependentDCMotorEnv(target_speed=.5)
    actor = StateDependentMetricActor(context_dim=4, spec=spec, hidden=32)
    initial_state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(cfg["learning_rate"]))
    history, best_score, best_actor = [], float("inf"), {k: v.detach().clone() for k, v in actor.state_dict().items()}
    val_seeds = [int(x) for x in cfg["validation_seeds"]]
    for update in range(1, int(cfg["updates"]) + 1):
        seeds = [int(cfg["train_seed"]) + update * 1000 + i for i in range(int(cfg["batch_states"]))]
        loss, metrics = dc_motor_one_step_loss(actor, env, seeds=seeds,
            decay=float(cfg["decay"]), descent_weight=float(cfg["descent_weight"]))
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0); optimizer.step()
        if update % 10 == 0 or update == 1 or update == int(cfg["updates"]):
            validation = []
            for seed in val_seeds:
                result = rollout_dc_motor(env, actor, seed=seed, control="clf_candidate",
                    alpha=float(cfg["alpha"]), clf_points=int(cfg["clf_points"]))
                validation.append({k: result[k] for k in
                    ("success", "steps", "total_cost", "terminal_physical_error",
                     "V_rise_steps", "descent_residual_sum")})
            success = np.mean([x["success"] for x in validation])
            cost = float(np.mean([x["total_cost"] for x in validation]))
            score = (1.0 - success) + cost
            row = {"update": update, "train": metrics,
                   "validation": {"success_rate": float(success), "mean_cost": cost, "objective": score,
                                  "episodes": validation}}
            history.append(row)
            if score < best_score:
                best_score, best_actor = score, {k: v.detach().clone() for k, v in actor.state_dict().items()}
            write(out / "progress.json", {"config": cfg, "history": history, "best_score": best_score})
    # Deterministic final/test comparison: identity and initial are explicit baselines.
    results = {}
    models = {"identity": IdentityMetricActor(4, spec), "initial": actor, "learned": actor}
    # actor holds the final state; load best into a separate copy for learned.
    best = StateDependentMetricActor(context_dim=4, spec=spec, hidden=32)
    best.load_state_dict(best_actor)
    initial = StateDependentMetricActor(context_dim=4, spec=spec, hidden=32)
    initial.load_state_dict(initial_state)
    models = {"identity": IdentityMetricActor(4, spec), "initial": initial, "learned": best, "final": actor}
    test_seeds = list(range(int(cfg["test_seed_start"]), int(cfg["test_seed_start"]) + int(cfg["test_seed_count"])))
    for name, model in models.items():
        episodes = []
        for seed in test_seeds:
            result = rollout_dc_motor(env, model, seed=seed, control="clf_candidate",
                alpha=float(cfg["alpha"]), clf_points=int(cfg["clf_points"]))
            episodes.append({k: result[k] for k in
                ("success", "steps", "total_cost", "terminal_physical_error",
                 "V_rise_steps", "descent_residual_sum", "clf_infeasible_selected")})
        results[name] = {"episodes": episodes,
            "success_rate": float(np.mean([e["success"] for e in episodes])),
            "mean_cost": float(np.mean([e["total_cost"] for e in episodes])),
            "mean_V_rise_steps": float(np.mean([e["V_rise_steps"] for e in episodes])),
            "clf_infeasible_steps": int(sum(e["clf_infeasible_selected"] for e in episodes))}
    torch.save({"actor": actor.state_dict(), "best_actor": best_actor,
                "protocol": cfg, "train_seed": int(cfg["train_seed"])}, out / "last.pt")
    write(out / "RESULTS.json", {"config": cfg, "history": history, "best_score": best_score,
        "test_seeds": test_seeds, "models": results,
        "candidate_certificate_only": True,
        "scope": "bounded seed41 pilot; identity/initial/learned/final; not a stability proof"})
    write(out / "protocol.json", {"config": cfg, "algorithm": "state_conditioned_M_one_step_pilot_v1",
        "candidate_certificate_only": True, "formal_training_started": True})
    print(json.dumps({"output": str(out), "updates": int(cfg["updates"]),
                      "best_score": best_score, "test_models": results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
