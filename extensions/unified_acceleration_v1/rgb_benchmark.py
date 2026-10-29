"""Bounded train-only RGB engineering benchmark; never performance selection."""
from __future__ import annotations
from contextlib import nullcontext
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT/"stabilization-gym", ROOT/"stabilization-learning-control"):
    sys.path.insert(0, str(path))
import numpy as np
import torch
from extensions.unified_training_v1.backends.rgb import Backend
from extensions.unified_acceleration_v1.rgb_execution import PreparedRGBExecutor


class Telemetry:
    def __init__(self, folder):
        self.folder = folder; self.rows=[]; self.stop=threading.Event()
        self.thread=threading.Thread(target=self.loop, daemon=True)
    def loop(self):
        with (self.folder/"GPU_SAMPLES.jsonl").open("x", encoding="utf-8") as stream:
            while not self.stop.is_set():
                row=dict(unix=time.time())
                try:
                    result=subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                        capture_output=True,text=True,check=True,timeout=3,stdin=subprocess.DEVNULL,
                        creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                    values=list(map(float,result.stdout.strip().splitlines()[0].split(",")))
                    row.update(gpu_percent=values[0],memory_mib=values[1],power_watts=values[2])
                except Exception as error: row["error"]=repr(error)
                self.rows.append(row);stream.write(json.dumps(row)+"\n");stream.flush();self.stop.wait(.5)
    def summary(self, start, end):
        samples=[r for r in self.rows if start<=r["unix"]<=end and "gpu_percent" in r]
        return dict(sample_count=len(samples),mean_gpu_percent=float(np.mean([r["gpu_percent"] for r in samples])) if samples else None,
            mean_memory_mib=float(np.mean([r["memory_mib"] for r in samples])) if samples else None,
            peak_memory_mib=max([r["memory_mib"] for r in samples],default=None))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seconds",type=int,default=300);parser.add_argument("--pairs",type=int,default=12)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    begun=time.time(); deadline=begun+args.seconds
    def expired():
        (args.output/"TIMEOUT.json").write_text(json.dumps(dict(timeout=True)),encoding="utf-8");os._exit(124)
    watchdog=threading.Timer(args.seconds,expired);watchdog.daemon=True;watchdog.start()
    telemetry=Telemetry(args.output);telemetry.thread.start()
    config=dict(seed=81,train_scene_count=1,parallel_envs=4,initial_collection_steps=8,collection_steps=8,
        batch_size=4,validation_cases=2,test_cases=2,evaluation_steps=12,evaluation_plants=["P0"])
    backend=Backend(config,args.output/"train_only",deadline=deadline)
    result=dict(scope="train-only engineering; same initial weights; no validation/test/selection",rows=[])
    def publish():
        (args.output/"BENCHMARK.partial.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    try:
        init=time.time();result["initialization"]=backend.initialize();result["initialization_wall_s"]=time.time()-init
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        result["execution_options"]=dict(cudnn_deterministic=True,cudnn_benchmark=False,tf32=False)
        initial=deepcopy(backend.learner.payload())
        for lanes in (4,16):
            backend.config["parallel_envs"]=lanes
            t=time.time(); collection=backend.collect(initial=True);end=time.time()
            row=dict(kind="collection",lanes=lanes,started_unix=t,finished_unix=end,wall_s=end-t,
                fresh_transitions=collection["fresh_transitions"],transitions_per_second=collection["fresh_transitions"]/(end-t),
                includes="environment creation, rendering, FastDepth, old controller, D2H, triplets, compressed replay write")
            result["rows"].append(row);publish();print(json.dumps(row),flush=True)
        learner=backend.learner
        replay_state=deepcopy(backend.replay.rng.bit_generator.state)
        def reset():
            learner.restore(initial);learner.inputs.clear();backend.replay.cache.clear()
            backend.replay.rng.bit_generator.state=deepcopy(replay_state)
            torch.manual_seed(811);torch.cuda.manual_seed_all(811)
        def update(batch):
            learner.representation(batch);learner.metric_update(batch)
        def gradients():
            return {name:{key:(None if parameter.grad is None else parameter.grad.detach().cpu().clone())
                    for key,parameter in getattr(learner,name).named_parameters()}
                    for name in ("ae","shadow","critics","metric")}
        reset();parity_batch=backend.replay.sample(4)
        learner.representation(parity_batch);representation_gradients=gradients()
        learner.metric_update(parity_batch);metric_gradients=gradients()
        reference={name:{key:value.detach().cpu().clone() for key,value in getattr(learner,name).state_dict().items()}
                   for name in ("ae","shadow","critics","metric")}
        reset();update(parity_batch)
        result["original_repeat_max_parameter_difference"]={name:max(float((tensor.detach().cpu().to(torch.float64)-reference[name][key].to(torch.float64)).abs().max())
                for key,tensor in getattr(learner,name).state_dict().items()) for name in reference}
        reset()
        with PreparedRGBExecutor(learner) as prepared:
            prepared.prepare([parity_batch]);learner.representation(parity_batch)
            prepared_representation_gradients=gradients()
            learner.metric_update(parity_batch);prepared_metric_gradients=gradients()
            differences={name:max(float((tensor.detach().cpu().to(torch.float64)-reference[name][key].to(torch.float64)).abs().max())
                for key,tensor in getattr(learner,name).state_dict().items()) for name in reference}
            result["parity_max_parameter_difference"]=differences
            gradient_differences={}
            for phase,first,second in (("representation",representation_gradients,prepared_representation_gradients),
                                       ("metric",metric_gradients,prepared_metric_gradients)):
                maximum=0.
                for name in first:
                    for key,a in first[name].items():
                        b=second[name][key]
                        if (a is None)!=(b is None): raise AssertionError("Gradient isolation changed")
                        if a is not None: maximum=max(maximum,float((a-b).abs().max()))
                gradient_differences[phase]=maximum
            result["parity_max_gradient_difference"]=gradient_differences
            result["prepared_strict_parity_pass"]=(max(differences.values())<=1e-6 and max(gradient_differences.values())<=1e-6)
            result["prepared_execution_status"]="experimental; not enabled in formal training; strict 1e-6 gate retained"
        for batch_size in (4,16,32):
            for mode in ("original_end_to_end","prepared_window"):
                reset();torch.cuda.reset_peak_memory_stats();start=time.time()
                context=PreparedRGBExecutor(learner) if mode=="prepared_window" else nullcontext()
                with context as prepared:
                    if prepared:
                        batches=[backend.replay.sample(batch_size) for _ in range(args.pairs)]
                        prepared.prepare(batches)
                    else: batches=None
                    torch.cuda.synchronize();ready=time.time()
                    for index in range(args.pairs):
                        backend.guard();batch=batches[index] if batches is not None else backend.replay.sample(batch_size)
                        update(batch)
                    torch.cuda.synchronize();finish=time.time()
                    row=dict(kind="updates",batch_size=batch_size,mode=mode,pairs=args.pairs,
                        representation_updates=args.pairs,metric_updates=args.pairs,
                        optimizer_steps=dict(ae=args.pairs,shadow_ae=args.pairs,derivative_critic=args.pairs,metric=args.pairs),
                        transition_presentations=2*args.pairs*batch_size,unique_replay_transitions=backend.replay.transitions,
                        started_unix=start,ready_unix=ready,finished_unix=finish,prepare_wall_s=ready-start,
                        ready_update_wall_s=finish-ready,end_to_end_wall_s=finish-start,
                        pairs_per_second=args.pairs/(finish-start),end_to_end_presentations_per_second=2*args.pairs*batch_size/(finish-start),
                        ready_presentations_per_second=2*args.pairs*batch_size/(finish-ready),
                        peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        prepared=prepared.statistics() if prepared else None,
                        original_mode_ready_includes_replay_and_preprocessing=not bool(prepared),
                        geometry_JVP_recomputed_after_every_representation=True)
                    result["rows"].append(row);publish();print(json.dumps(row),flush=True)
        result["status"]="complete"
    except BaseException as error:
        result.update(status="failed",error=repr(error));raise
    finally:
        backend.close();telemetry.stop.set();telemetry.thread.join(timeout=4);watchdog.cancel()
        finish=time.time();result["wall_s"]=finish-begun;result["full_run_gpu"]=telemetry.summary(begun,finish)
        for row in result["rows"]:row["gpu"]=telemetry.summary(row["started_unix"],row["finished_unix"])
        result["limitations"]=["one training room, seed81; throughput only; no control-quality evidence",
            "batch size changes gradient statistics and samples per update; speed is not training equivalence",
            "GPU rendering with CPU image preprocessing/FastDepth wrapper and replay I/O; not entirely GPU resident",
            "prepared-window update timing excludes measured preparation; end-to-end includes it",
            "whole-run sampling includes initialization/parity/collection; short interval averages have few samples"]
        (args.output/"BENCHMARK.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(dict(completed=True,wall_s=result["wall_s"],full_run_gpu=result["full_run_gpu"])),flush=True)


if __name__=="__main__":main()
