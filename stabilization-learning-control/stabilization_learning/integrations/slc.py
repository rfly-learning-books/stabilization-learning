"""Real upstream SLC LAC on one episode-held Cholesky action.

The imported LAC, actor, critic, replay buffer and update losses are unmodified
upstream implementations. A narrowly scoped single-process logging helper lets
Windows users run this path without installing an MPI runtime. This adapter is
not an MPI implementation and refuses multi-process launch environments.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import types
import warnings

import numpy as np
import torch

from ..wrappers import EpisodeMetricEnv
from ..common.compat import canonical_protocol


UPSTREAM_COMMIT = "27b6fdfed6407a46192d2304027b307750c18ba6"
UPSTREAM_URL = "https://github.com/rickstaa/stable-learning-control"
_SERIAL_MODULE = "stable_learning_control.utils.mpi_utils.mpi_tools"


def _serial_statistics(values, with_min_and_max=False):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        result = (float("nan"), float("nan"))
        return (*result, float("inf"), float("-inf")) if with_min_and_max else result
    result = (float(values.mean()), float(values.std()))
    return (*result, float(values.min()), float(values.max())) if with_min_and_max else result


def _install_serial_logging():
    for variable in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "WORLD_SIZE",
                     "MPI_LOCALNRANKS", "MV2_COMM_WORLD_SIZE"):
        if int(os.environ.get(variable, "1")) > 1:
            raise RuntimeError("The SLC adapter supports one process only; MPI launch detected")
    existing = sys.modules.get(_SERIAL_MODULE)
    if existing is not None:
        if not getattr(existing, "__stabilization_serial_logging__", False):
            raise RuntimeError("An MPI helper was imported before the serial SLC adapter; use a fresh process")
        return
    helper = types.ModuleType(_SERIAL_MODULE)
    helper.__stabilization_serial_logging__ = True
    helper.__doc__ = "Single-process logging statistics only; not an MPI runtime."
    helper.proc_id = lambda: 0
    helper.num_procs = lambda: 1
    helper.mpi_statistics_scalar = _serial_statistics

    def reject_parallel(n, bind_to_core=False):
        if n > 1:
            raise RuntimeError("Distributed MPI execution is unavailable in this serial adapter")

    helper.mpi_fork = reject_parallel
    sys.modules[_SERIAL_MODULE] = helper


def load_upstream():
    """Load the vendored official LAC class, without changing upstream sources."""
    override = os.environ.get("STABILIZATION_SLC_SOURCE")
    candidates = ([Path(override)] if override else []) + [
        parent / "third_party" / "stable-learning-control" for parent in Path(__file__).resolve().parents]
    upstream = next((path for path in candidates if (path / "stable_learning_control").is_dir()), candidates[0])
    source = upstream / "stable_learning_control" / "algos" / "pytorch" / "lac" / "lac.py"
    if not source.is_file():
        raise ImportError(f"Pinned stable-learning-control source is missing: {source}")
    _install_serial_logging()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    try:
        module = importlib.import_module("stable_learning_control.algos.pytorch.lac.lac")
        buffers = importlib.import_module("stable_learning_control.algos.pytorch.common.buffers")
    except ImportError as exc:
        raise ImportError(
            "SLC's PyTorch adapter needs gymnasium, scipy, joblib, tensorboard, "
            "cloudpickle, pandas and the core PyTorch dependencies. No TensorFlow, "
            "Ray or MPI installation is required by this serial adapter."
        ) from exc
    if Path(module.__file__).resolve() != source.resolve():
        raise RuntimeError("A different SLC installation was already imported; use a fresh process")
    provenance = {
        "repository": UPSTREAM_URL,
        "commit": UPSTREAM_COMMIT,
        "implementation": str(source),
        "lac_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "logging_backend": "explicit_single_process_statistics_adapter",
        "algorithm_modified": False,
    }
    return module.LAC, buffers.ReplayBuffer, provenance


def _parameter_hash(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _make_env(env_factory):
    env = env_factory()
    return env if isinstance(env, EpisodeMetricEnv) else EpisodeMetricEnv(env)


def _make_agent(LAC, env, hidden_sizes, device):
    return LAC(env, opt_type="minimize", device=device,
               ac_kwargs={"hidden_sizes": {"actor": list(hidden_sizes), "critic": list(hidden_sizes)}})


def _environment_signature(env):
    spec = env.physical_env.metric_spec
    return {"observation_shape": list(env.observation_space.shape),
            "action_low": env.action_space.low.tolist(), "action_high": env.action_space.high.tolist(),
            "error_dim": spec.error_dim, "eigen_floor": spec.eigen_floor, "trace": spec.trace,
            "physical_environment": type(env.physical_env).__module__ + "." + type(env.physical_env).__name__}


def train_slc(env_factory, output_dir, steps=64, seed=41, *, batch_size=16,
              start_steps=8, update_after=16, updates_per_step=1,
              hidden_sizes=(64, 64), device="cpu"):
    """Train upstream LAC with raw Cholesky parameters as its bounded action.

    ``steps`` counts complete physical episodes, not physical integration steps.
    ``env_factory`` may return a physical environment or EpisodeMetricEnv.
    The final checkpoint includes weights, optimizers, replay and RNG state;
    this small adapter exposes evaluation loading, not automatic resume.
    """
    if any(int(x) != x or x < 1 for x in (steps, batch_size, update_after, updates_per_step)):
        raise ValueError("steps, batch_size, update_after and updates_per_step must be positive integers")
    if start_steps < 0 or int(start_steps) != start_steps:
        raise ValueError("start_steps must be a nonnegative integer")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    LAC, ReplayBuffer, provenance = load_upstream()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = _make_env(env_factory)
    try:
        env.action_space.seed(seed)
        agent = _make_agent(LAC, env, hidden_sizes, device)
        replay = ReplayBuffer(obs_dim=env.observation_space.shape,
                              act_dim=env.action_space.shape, size=max(int(steps), 2), device=agent._device)
        actor_before = _parameter_hash(agent.ac.pi)
        critic_before = _parameter_hash(agent.ac.L)
        records, diagnostics = [], []
        physical_steps = updates = 0
        probe_observation, _ = env.reset(seed=seed)
        for decision in range(int(steps)):
            observation, _ = env.reset(seed=seed + decision)
            action = env.action_space.sample() if decision < start_steps else agent.get_action(observation)
            following, cost, terminated, truncated, info = env.step(action)
            if not terminated or truncated:
                raise RuntimeError("EpisodeMetricEnv must terminate every completed metric decision")
            if not np.isfinite(cost) or cost < 0:
                raise ValueError("SLC LAC requires finite nonnegative costs")
            replay.store(observation, action, cost, following, True)
            physical_steps += int(info["physical_steps"])
            row = {"metric_decisions": decision + 1, "physical_steps": physical_steps,
                   "cost": float(cost), "success": bool(info["success"]),
                   "failure": bool(info["failure"]), "V_rise_fraction": float(info["V_rise_fraction"])}
            records.append(row)
            if replay.size >= max(int(batch_size), int(update_after)):
                for _ in range(int(updates_per_step)):
                    data = replay.sample_batch(batch_size=int(batch_size))
                    # Exactly the upstream update: no replacement loss or fake optimizer.
                    values = agent.update(data)
                    reduced = {key: float(np.asarray(value).mean()) for key, value in values.items()}
                    if not all(np.isfinite(value) for value in reduced.values()):
                        raise FloatingPointError("Nonfinite upstream LAC diagnostics")
                    updates += 1
                    diagnostics.append({"update": updates, **reduced})
        settings = {"steps": int(steps), "seed": int(seed), "batch_size": int(batch_size),
                    "start_steps": int(start_steps), "update_after": int(update_after),
                    "updates_per_step": int(updates_per_step), "hidden_sizes": list(hidden_sizes),
                    "device": device}
        summary = {"backend": "upstream_stable_learning_control.LAC", "provenance": provenance,
                   "metric_decisions": int(steps), "physical_steps": physical_steps, "updates": updates,
                   "actor_initial_sha256": actor_before, "actor_final_sha256": _parameter_hash(agent.ac.pi),
                   "critic_initial_sha256": critic_before, "critic_final_sha256": _parameter_hash(agent.ac.L),
                   "settings": settings, "cost_convention": "positive_fixed_horizon_cost",
                   "certificate_status": "candidate_only",
                   "environment_signature": _environment_signature(env),
                   "critic_note": "Upstream learned L critic is separate from physical V=e^T M e",
                   "probe_observation": probe_observation.tolist(),
                   "probe_action": agent.get_action(probe_observation, deterministic=True).tolist(),
                   "checkpoint": str(output / "lac_final.pt")}
        checkpoint = {"state_dict": agent.state_dict(), "settings": settings, "summary": summary,
                      "optimizers": {"actor": agent._pi_optimizer.state_dict(),
                                     "critic": agent._c_optimizer.state_dict(),
                                     "alpha": agent._log_alpha_optimizer.state_dict(),
                                     "lambda": agent._log_labda_optimizer.state_dict()},
                      "replay": {"obs": replay.obs_buf[:replay.size], "act": replay.act_buf[:replay.size],
                                 "cost": replay.rew_buf[:replay.size],
                                 "obs_next": replay.obs_next_buf[:replay.size],
                                 "done": replay.done_buf[:replay.size]},
                      "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                              "torch": torch.get_rng_state()},
                      "records": records, "diagnostics": diagnostics}
        torch.save(checkpoint, output / "lac_final.pt.tmp")
        (output / "lac_final.pt.tmp").replace(output / "lac_final.pt")
        (output / "RESULTS.json").write_text(json.dumps({**summary, "records": records,
                                                          "diagnostics": diagnostics}, indent=2, allow_nan=False),
                                              encoding="utf-8")
        if updates == 0:
            warnings.warn("No LAC update was performed; increase steps or reduce batch_size/update_after", stacklevel=2)
        return summary
    finally:
        env.close()


def load_slc_policy(env_factory, checkpoint, *, device="cpu"):
    """Load a trusted local adapter checkpoint for deterministic evaluation."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    LAC, _, provenance = load_upstream()
    saved_provenance = payload["summary"]["provenance"]
    for key in ("commit", "lac_source_sha256"):
        if saved_provenance[key] != provenance[key]:
            raise ValueError("The saved policy and current upstream LAC source do not match")
    env = _make_env(env_factory)
    try:
        if canonical_protocol(_environment_signature(env)) != canonical_protocol(payload["summary"]["environment_signature"]):
            raise ValueError("The policy's task coordinates, metric specification or raw action bounds do not match")
        agent = _make_agent(LAC, env, payload["settings"]["hidden_sizes"], device)
        agent.load_state_dict(payload["state_dict"])
        agent.eval()
        return agent
    finally:
        env.close()
