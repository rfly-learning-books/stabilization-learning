"""Measured CUDA RGB sampling and learning workloads; never a policy evaluation.

All benchmark outputs must use a new directory. The frozen v3 engineering model
is a reproducible compute workload, not evidence of a successful visual policy.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import threading
import time
import traceback

import numpy as np
import torch

from stabilization_gym.tasks_v2.reconstructed_rgb import load_runtime, ReconstructedRGBEnv
from stabilization_gym.tasks_v3.reconstructed_rgb import BatchedRGBGeometry
from stabilization_gym.tasks_v2.visual import VisualObservation
from stabilization_gym.tasks_gpu_rgb.environment import GPUParallelRGBEnv, initial_offsets
from stabilization_gym.tasks_gpu_rgb.geometry import GPURGBGeometry
from stabilization_learning.joint_v3.inference import FrozenRGBPolicy
from stabilization_learning.joint_v2.visual import VisualBatch
from extensions.joint_certificate_v3.rgb import RGBRunConfig, _new_learner, _restore_for_evaluation


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_state_hash(learner):
    h = hashlib.sha256()
    for module in (learner.features, learner.metric):
        for key, tensor in module.state_dict().items():
            h.update(key.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class GPUUtilization:
    def __init__(self):
        self.rows = []
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw",
             "--format=csv,noheader,nounits", "-lms", "200"], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, creationflags=flags)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.process.stdout:
            try:
                fields = [float(x.strip()) for x in line.split(",")]
                if len(fields) == 5 and fields[0] == 0:
                    self.rows.append(dict(time=time.perf_counter(), unix=time.time(), gpu=fields[1],
                                          memory_controller=fields[2], memory_mib=fields[3], watts=fields[4]))
            except ValueError:
                pass

    def summarize(self, begin, end):
        # nvidia-smi is a device-wide activity sample with a driver averaging
        # window. Trim phase boundaries; retain raw samples for independent audit.
        rows = [r for r in self.rows if begin+1.5 <= r["time"] <= end-.5]
        return {"samples": len(rows), **{k: statistics.mean(r[k] for r in rows) if rows else None
                                        for k in ("gpu", "memory_controller", "memory_mib", "watts")}}

    def close(self):
        self.process.terminate(); self.process.wait(timeout=10); self.thread.join(timeout=10)


class SerialRGBVectorEnv:
    """Reference v3 CPU environment loop, same next-step reset/count convention."""
    def __init__(self, cfg, scenes, *, plant="P0", max_steps=200, **_):
        self.envs = [ReconstructedRGBEnv(cfg, s, plant=plant, max_steps=max_steps) for s in scenes]
        self.num_envs, self.cfg, self.scenes = len(scenes), cfg, scenes
        self.done = np.zeros(len(scenes), bool)

    def reset(self, *, seed=811, options=None):
        self.options = options
        self.seeds = [seed+i for i in range(self.num_envs)]
        self.done[:] = False
        self.public = []
        for i, env in enumerate(self.envs):
            public, _ = env.reset(seed=self.seeds[i], options={"goal_id": options.get("goal_ids", [0]*self.num_envs)[i],
                                                            "offset": options["offsets"][i]})
            self.public.append(public)
        return self.public, {}

    def step(self, action):
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        active = ~self.done.copy()
        reward, terminal, truncated, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            if self.done[i]:
                public, info = env.reset(seed=self.seeds[i], options={"goal_id": self.options.get("goal_ids", [0]*self.num_envs)[i],
                                                                    "offset": self.options["offsets"][i]})
                r, t, tr = 0., False, False
            else:
                public, r, t, tr, info = env.step(action[i])
            self.public[i] = public
            self.done[i] = t or tr
            reward.append(r); terminal.append(t); truncated.append(tr); infos.append(info)
        return self.public, np.asarray(reward), np.asarray(terminal), np.asarray(truncated), dict(transition_mask=active, rows=infos)

    def close(self):
        for env in self.envs:
            env.close()


def restore_workload(root, cfg):
    run = root / "runs/rgb_pipeline_v3_smoke_20261027_a/rgb_seed81"
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf8"))
    for path, expected in protocol["sources"].items():
        if sha(path) != expected:
            raise RuntimeError("Frozen source changed: " + path)
    frozen = json.loads((run / "FROZEN.json").read_text(encoding="utf8"))
    if sha(run / frozen["checkpoint"]) != frozen["sha256"]:
        raise RuntimeError("Frozen compute workload checkpoint changed")
    config = RGBRunConfig.from_dict({k: v for k, v in protocol["config"].items() if k in RGBRunConfig.__dataclass_fields__})
    learner = _new_learner(cfg, config)
    from stabilization_learning.storage import load_checkpoint
    payload = load_checkpoint(run / frozen["checkpoint"], map_location="cpu")
    _restore_for_evaluation(learner, payload)
    return learner, payload, dict(path=str(run / frozen["checkpoint"]), sha256=frozen["sha256"],
                                 use="frozen three-update engineering model, compute workload only")


def make_env(kind, cfg, scenes, root, plant, max_steps=200):
    cls = GPUParallelRGBEnv if kind == "gpu" else SerialRGBVectorEnv
    return cls(cfg, scenes, plant=plant, max_steps=max_steps,
               cache_dir=root/".cache/rgb_gpu_warp", samples_per_axis=2)


def audit(root, output, cfg, scenes, learner, *, all_assets=False):
    """Independent reference execution, GPU preprocessing and view comparisons."""
    from PIL import Image, ImageDraw
    from lac_vs import legacy
    families = ("classroom", "office", "meeting", "library")
    selected = [next(s for s in scenes if s["split"] == "train" and s["family"] == f) for f in families]
    report = dict(physical=[], geometry={}, visual=[], asset_compatibility=[], passed=False)
    montage = Image.new("RGB", (2*256, 4*(256+28)), "white")
    draw = ImageDraw.Draw(montage)
    cpu_geometry = BatchedRGBGeometry(cfg, "cuda", depth_batch_size=4)
    gpu_geometry = GPURGBGeometry(cfg, "cuda:0")
    original_state = tensor_state_hash(learner)
    for family_index, scene in enumerate(selected):
        offsets = initial_offsets(cfg, [scene]*2, [811, 812])
        for plant in ("P0", "P1"):
            env = make_env("gpu", cfg, [scene]*2, root, plant, max_steps=9)
            old = make_env("serial", cfg, [scene]*2, root, plant, max_steps=9)
            try:
                options = dict(offsets=offsets)
                public, _ = env.reset(options=options)
                reference, _ = old.reset(options=options)
                if plant == "P0":
                    a, b = reference[0]["rgb"], public["rgb"][0].cpu().numpy()
                    delta = a.astype(float)-b.astype(float)
                    mse = float(np.mean(delta**2))
                    report["visual"].append(dict(scene_id=scene["id"], mae=float(np.mean(abs(delta))),
                        psnr_db=float(10*np.log10(255**2/max(mse, 1e-15))), reference_mean=float(a.mean()),
                        gpu_mean=float(b.mean()), pixel_equivalent=False))
                    montage.paste(Image.fromarray(a), (0, family_index*284+28))
                    montage.paste(Image.fromarray(b), (256, family_index*284+28))
                    draw.text((4, family_index*284+6), scene["id"]+" | classic / GPU", fill="black")
                    if b.std() < 5:
                        raise AssertionError("Rendered view is degenerate")
                errors = dict(offset=0., reward=0., actual_velocity=0.)
                rng = np.random.default_rng(938)
                for step in range(24):
                    commands = rng.uniform(-.15, .15, (2, 4)).astype(np.float32)
                    p, r, t, tr, info = env.step(torch.as_tensor(commands, device="cuda"))
                    q, r0, t0, tr0, info0 = old.step(commands)
                    np.testing.assert_array_equal(t.cpu().numpy(), t0)
                    np.testing.assert_array_equal(tr.cpu().numpy(), tr0)
                    np.testing.assert_array_equal(info["transition_mask"].cpu().numpy(), info0["transition_mask"])
                    errors["reward"] = max(errors["reward"], float(np.max(abs(r.cpu().numpy()-r0))))
                    for lane, row in enumerate(info0["rows"]):
                        e = info["supervision"]["next_offset"][lane].cpu().numpy()
                        errors["offset"] = max(errors["offset"], float(np.max(abs(e-row["supervision"]["next_offset"]))))
                        if info0["transition_mask"][lane]:
                            v = info["supervision"]["actual_velocity"][lane].cpu().numpy()
                            errors["actual_velocity"] = max(errors["actual_velocity"], float(np.max(abs(v-row["supervision"]["actual_velocity"]))))
                if max(errors.values()) > 1e-7:
                    raise AssertionError(errors)
                report["physical"].append(dict(scene_id=scene["id"], plant=plant, steps=24, lanes=2, max_abs=errors))
            finally:
                env.close(); old.close(); del env, old; gc.collect()
    # Use identical CPU-generated RGB for a clean geometry-only comparison.
    legacy_env = SerialRGBVectorEnv(cfg, selected, max_steps=8)
    try:
        offsets = initial_offsets(cfg, selected, [913+i for i in range(4)])
        observations, _ = legacy_env.reset(options=dict(offsets=offsets))
        expected = cpu_geometry.observe_batch(observations)
        gpu_public = {k: torch.as_tensor(np.stack([p[k] for p in observations]), device="cuda") for k in observations[0]}
        actual = gpu_geometry.observe(gpu_public)
        fixed_depth = np.stack([cpu_geometry.depth.predict(p["rgb"]) for p in observations])
        gx, gj = gpu_geometry.tangents(gpu_public["rgb"], torch.as_tensor(fixed_depth[:, None], device="cuda"), gpu_public["rotation"])
        for name, got, want in (("preprocessing", gx, expected[0]), ("fixed_depth_tangents", gj, expected[2])):
            diff = got-want
            report["geometry"][name] = dict(max_abs=float(diff.abs().max()), relative_l2=float(diff.norm()/want.norm().clamp_min(1e-12)))
            torch.testing.assert_close(got, want, rtol=5e-5, atol=3e-6)
        z = gpu_geometry.predict_depth(gpu_public["rgb"])
        zref = torch.as_tensor(fixed_depth[:, None], device="cuda")
        report["geometry"]["estimated_depth"] = dict(max_abs=float((z-zref).abs().max()),
                                                     relative_l2=float((z-zref).norm()/zref.norm()))
        if report["geometry"]["estimated_depth"]["relative_l2"] > .01:
            raise AssertionError("GPU depth preprocessing differs materially")
        for name, got, want in zip(("image", "goal", "depth_based_tangents"), actual, expected):
            report["geometry"][name] = dict(max_abs=float((got-want).abs().max()),
                                             relative_l2=float((got-want).norm()/want.norm().clamp_min(1e-12)))
        policy = FrozenRGBPolicy(learner)
        a = policy(VisualObservation(*actual)).action
        b = policy(VisualObservation(*expected)).action
        report["geometry"]["controller_action"] = dict(max_abs=float((a-b).abs().max()),
                                                         relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-12)))
        if float((a-b).abs().max()) > .01:
            raise AssertionError("GPU preprocessing changes actions materially")
        try:
            gpu_geometry.observe(dict(gpu_public, truth=torch.zeros(4, device="cuda")))
            raise AssertionError("Privileged truth accepted")
        except ValueError:
            report["geometry"]["privileged_input_rejected"] = True
    finally:
        legacy_env.close()
    if all_assets:
        for scene in scenes:
            env = make_env("gpu", cfg, [scene], root, "P0")
            try:
                public, _ = env.reset(seed=811)
                rgb = public["rgb"]
                if rgb.float().std(correction=0).item() < 1:
                    raise AssertionError("Empty asset image")
                report["asset_compatibility"].append(dict(split=scene["split"],
                    shape=list(rgb.shape), mean=float(rgb.float().mean()), **env._groups[0][1].stats))
                print(json.dumps(dict(asset=scene["id"], checked=len(report["asset_compatibility"])), ensure_ascii=False), flush=True)
            finally:
                env.close(); del env; gc.collect()
    montage.save(output/"renderer_comparison.png")
    report["frozen_workload_parameters_unchanged"] = tensor_state_hash(learner) == original_state
    report["renderer_equivalence"] = "same assets/cameras/task, different shading/rasterization; not pixel equivalent"
    report["passed"] = True
    write_json(output/"AUDIT.json", report)
    return report


def measure(fn, seconds, telemetry, unit, *, warmups=3):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    count = torch.zeros((), dtype=torch.int64, device="cuda")
    iterations = 0
    begin = time.perf_counter()
    while time.perf_counter()-begin < seconds:
        value = fn()
        count += value
        iterations += 1
    torch.cuda.synchronize()
    end = time.perf_counter()
    n = int(count.item())
    return dict(seconds=end-begin, iterations=iterations, samples=n, samples_per_second=n/(end-begin),
                unit=unit, peak_torch_mib=torch.cuda.max_memory_allocated()/1024**2,
                telemetry=telemetry.summarize(begin, end), begin=begin, end=end)


def rollout_phase(root, cfg, selected, learner, kind, mode, batch_size, plant, seconds, repeats, telemetry):
    # Four fixed training families; mixed rooms grouped, never substituted by a
    # single cheap primitive. At B=1 use the first training classroom.
    scenes = [selected[i % len(selected)] for i in range(batch_size)]
    offsets = initial_offsets(cfg, scenes, [1101+i for i in range(batch_size)])
    env = make_env(kind, cfg, scenes, root, plant)
    geometry = GPURGBGeometry(cfg) if kind == "gpu" else BatchedRGBGeometry(cfg, "cuda", depth_batch_size=batch_size)
    policy = FrozenRGBPolicy(learner, max_goals=max(32, batch_size))
    keys = [(s["id"], 0, kind) for s in scenes]
    # Independent bounded commands prepared once for the environment-only probe.
    rng = np.random.default_rng(724)
    commands = rng.uniform(-.2, .2, (257, batch_size, 4)).astype(np.float32)
    actions = torch.as_tensor(commands, device="cuda") if kind == "gpu" else commands
    rows = []
    try:
        for repeat in range(repeats):
            public, _ = env.reset(options=dict(offsets=offsets))
            state = {"public": public, "step": 0}
            def operation():
                if mode == "closed_loop":
                    observed = (geometry.observe(state["public"]) if kind == "gpu"
                                else geometry.observe_batch(state["public"]))
                    action = policy(VisualObservation(*observed), goal_keys=keys).action
                else:
                    action = actions[state["step"] % len(actions)]
                state["public"], _, _, _, info = env.step(action)
                state["step"] += 1
                mask = info["transition_mask"]
                return mask.sum() if torch.is_tensor(mask) else int(mask.sum())
            row = measure(operation, seconds, telemetry, "valid environment transitions/s")
            row.update(kind=kind, mode=mode, batch_size=batch_size, plant=plant, repeat=repeat,
                       scene_ids=[s["id"] for s in scenes], horizon=200, rgb_resolution=256,
                       policy_input_resolution=128, resets="next_step_reuse_declared_start")
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close(); del env, geometry; gc.collect(); torch.cuda.empty_cache()
    return rows


def learning_phase(root, cfg, selected, learner, payload, batch_size, seconds, repeats, telemetry):
    scenes = [selected[i % len(selected)] for i in range(batch_size)]
    env = make_env("gpu", cfg, scenes, root, "P0")
    geometry = GPURGBGeometry(cfg)
    rows, raw = [], []
    try:
        public, _ = env.reset(seed=1731)
        generator = torch.Generator(device="cuda").manual_seed(894)
        for _ in range(4):
            command = (torch.rand((batch_size, 4), generator=generator, device="cuda")-.5)*.3
            next_public, _, _, _, info = env.step(command)
            if not bool(info["transition_mask"].all()):
                raise AssertionError("Learning workload requires complete transitions")
            raw.append((public, next_public["rgb"], info))
            public = next_public
        for repeat in range(repeats):
            _restore_for_evaluation(learner, payload)
            learner.feature_optimizer.state.clear(); learner.metric_optimizer.state.clear()
            before = tensor_state_hash(learner)
            index = [0]
            def operation():
                obs, following, info = raw[index[0] % len(raw)]; index[0] += 1
                x, goal, tangent = geometry.observe(obs)
                truth = info["supervision"]
                batch = VisualBatch(VisualObservation(x, goal, tangent), geometry.encode_images(following),
                    truth["offset"].float(), truth["next_offset"].float(),
                    (truth["actual_velocity"] / env.scale).float(),
                    torch.full((batch_size,), env.dt, device="cuda"),
                    torch.zeros(batch_size, dtype=torch.bool, device="cuda"),
                    truth["previous_action"].float(), split="train")
                learner.update_features(batch)
                learner.update_metric(batch)
                return batch_size
            row = measure(operation, seconds, telemetry, "transitions per feature+M optimizer pair/s", warmups=2)
            row.update(kind="gpu", mode="learning", batch_size=batch_size, plant="P0", repeat=repeat,
                       pair_updates=row["iterations"], optimizer_updates=2*row["iterations"],
                       parameters_changed=tensor_state_hash(learner) != before,
                       raw_buffer_batches=len(raw), includes="GPU raw batch access, FastDepth, geometry, full phi/M losses/backward/optimizer",
                       excludes="environment stepping/rendering, disk archival")
            if not row["parameters_changed"]:
                raise AssertionError("Optimizer workload did not change parameters")
            rows.append(row); print(json.dumps(row), flush=True)
    finally:
        _restore_for_evaluation(learner, payload)
        env.close(); del env, geometry; gc.collect(); torch.cuda.empty_cache()
    return rows


def run(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(717); np.random.seed(717)
    telemetry = None
    try:
        _, cfg, scenes = load_runtime(root)
        learner, payload, workload = restore_workload(root, cfg)
        sources = {}
        for directory in (root/"stabilization-gym/stabilization_gym/tasks_gpu_rgb", root/"extensions/rgb_gpu_parallel"):
            sources.update({str(p): sha(p) for p in directory.glob("*.py")})
        sources[str(root/"scripts/benchmark_rgb_gpu_parallel.py")] = sha(root/"scripts/benchmark_rgb_gpu_parallel.py")
        import mujoco, mujoco_warp, warp
        protocol = dict(scope="engineering GPU environment audit and throughput; no long training or performance selection",
            created_unix=time.time(), system=platform.platform(), python=platform.python_version(),
            torch=torch.__version__, torch_cuda=torch.version.cuda, mujoco=mujoco.__version__, warp=warp.__version__,
            gpu=torch.cuda.get_device_name(), source_hashes=sources, workload=workload,
            config=vars(args), physical_task="unchanged four-DOF P0/P1; not full quadrotor dynamics",
            renderer="MJWarp, original MJCF assets, 256x256 RGB, textures/shadows, 2x2 subpixel rays",
            geometry="RGB-only fixed public FastDepth; no simulator depth",
            benchmark_scenes="first training scene from each of four families, round-robin by lane",
            notes=["Native Windows measured; Linux code path not executed on this host",
                   "nvidia-smi samples entire GPU, not exclusive process utilization or FLOP occupancy",
                   "Repeated starts and a fixed engineering model are workload fixtures, not independent policy trials",
                   "Renderer shading changes; no pixel-equivalence or learned-performance claim",
                   "Learning loop uses disposable restored weights and fresh small raw buffer; no saved trained policy"])
        write_json(output/"PROTOCOL.json", protocol)
        report = dict(protocol=protocol, phases=[])
        audit(root, output, cfg, scenes, learner, all_assets=args.all_assets)
        if args.audit_only:
            write_json(output/"FINAL_STATE.json", dict(status="complete", stage="audit")); return
        telemetry = GPUUtilization()
        begin=time.perf_counter(); time.sleep(4); end=time.perf_counter()
        report["idle"] = telemetry.summarize(begin, end)
        selected = [next(s for s in scenes if s["split"] == "train" and s["family"] == f)
                    for f in ("classroom", "office", "meeting", "library")]
        frozen_hash = tensor_state_hash(learner)
        for plant in args.plants:
            for mode, batches in (("environment", args.env_batches), ("closed_loop", args.policy_batches)):
                for batch in batches:
                    for kind in ("serial", "gpu"):
                        if (output/"STOP").exists():
                            raise InterruptedError("Benchmark STOP requested")
                        rows = rollout_phase(root, cfg, selected, learner, kind, mode, batch, plant,
                                             args.seconds, args.repeats, telemetry)
                        report["phases"].extend(rows)
                        write_json(output/"PROGRESS.json", report)
        report["rollout_model_unchanged"] = tensor_state_hash(learner) == frozen_hash
        for batch in args.gpu_extra_batches:
            rows = rollout_phase(root, cfg, selected, learner, "gpu", "environment", batch, "P0",
                                 args.seconds, args.repeats, telemetry)
            report["phases"].extend(rows); write_json(output/"PROGRESS.json", report)
        for batch in args.learning_batches:
            rows = learning_phase(root, cfg, selected, learner, payload, batch, args.seconds, args.repeats, telemetry)
            report["phases"].extend(rows); write_json(output/"PROGRESS.json", report)
        for p, expected in sources.items():
            if sha(p) != expected:
                raise RuntimeError("Benchmark source changed during measurement")
        report["restored_model_hash"] = tensor_state_hash(learner)
        report["model_restored"] = report["restored_model_hash"] == frozen_hash
        telemetry.close(); report["telemetry"] = telemetry.rows; telemetry = None
        write_json(output/"PROFILE.json", report)
        write_json(output/"FINAL_STATE.json", dict(status="complete", phases=len(report["phases"])))
    except BaseException as error:
        write_json(output/"FINAL_STATE.json", dict(status="failed", error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        if telemetry is not None:
            telemetry.close(); write_json(output/"PARTIAL_TELEMETRY.json", telemetry.rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=12.)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--plants", nargs="+", choices=("P0", "P1"), default=["P0", "P1"])
    parser.add_argument("--env-batches", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--policy-batches", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--learning-batches", nargs="*", type=int, default=[4, 16])
    parser.add_argument("--gpu-extra-batches", nargs="*", type=int, default=[32, 64])
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--all-assets", action="store_true")
    args = parser.parse_args()
    if args.seconds < 4 or args.repeats < 1 or any(b < 1 for b in args.env_batches+args.policy_batches+args.learning_batches):
        parser.error("Positive batch/repeat counts and at least 4 seconds per window are required")
    run(args)

