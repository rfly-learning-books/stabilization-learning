"""All-seed selection precedes any test access; old protocols remain untouched."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import os
import shutil
import sys
import time
import traceback

import torch
from stabilization_learning.algos.pytorch.state_metric_full_horizon import FullHorizonConfig
from stabilization_learning.algos.pytorch.metric_ppo.metric_ppo import module_hash
from extensions.metric_full_horizon_v3 import pipeline as old

ROOT = Path(__file__).resolve().parents[2]
TASKS = old.TASKS
SEEDS = (41, 42, 43, 44, 45)
read, write, sha, emit, now = old.read, old.write, old.sha, old.emit, old.now


def make_jobs(settings, sources):
    """Separate training streams; shared heldout states support paired comparisons."""
    jobs = []
    seeds = settings["seeds"]
    offset=settings.get("stream_offset",0)
    if offset and not settings.get("development_only",False):
        raise ValueError("Formal holdout streams are fixed")
    if settings.get("development_only",False) and offset<500000000:
        raise ValueError("Development smoke must use separate streams")
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Unique nonempty training seeds required")
    for ti, task in enumerate(settings["tasks"]):
        if task not in TASKS:
            raise ValueError("Unsupported physical task")
        for si, seed in enumerate(seeds):
            cfg = FullHorizonConfig(**settings["training"])
            cfg = replace(cfg, seed=seed,
                train_seed_start=offset+100000000 + ti*10000000 + si*100000,
                train_probe_seed=offset+150000000 + ti*1000000 + si*1000,
                validation_seeds=tuple(range(offset+210000000+ti*100000,offset+210000000+ti*100000+settings["validation_count"])),
                test_seeds=tuple(range(offset+220000000+ti*100000,offset+220000000+ti*100000+settings["test_count"])),
                validation_probe_seed=offset+230000000+ti*1000,
                **settings["task_overrides"].get(task, {}))
            identifier = f"{task}_seed{seed}"
            jobs.append(dict(id=identifier, task=task, seed=seed, path=f"jobs/{identifier}",
                protocol=dict(schema="book_multiseed_v4_job",tasks=[task],training=asdict(cfg),
                    hidden=settings["hidden"],residual_limit=settings["residual_limit"],sources=sources,
                    test_probe_seed=offset+240000000+ti*1000)))
    # The ranges for training initial-state generation must be disjoint, including
    # initialization's 32 states and the reserved prefix before optimizer updates.
    intervals=[]
    for job in jobs:
        c=job["protocol"]["training"]
        interval=(c["train_seed_start"],c["train_seed_start"]+128+c["updates"]*c["batch_states"])
        if any(max(interval[0],a)<min(interval[1],b) for a,b in intervals):
            raise ValueError("Training streams overlap")
        intervals.append(interval)
    return jobs


def source_manifest():
    result=old.manifest()
    for path in (ROOT/"extensions/book_multiseed_v4").glob("*.py"):
        result[path.relative_to(ROOT).as_posix()]=sha(path)
    for relative in ("scripts/analysis/run_book_multiseed_v4.py",):
        result[relative]=sha(ROOT/relative)
    return result


def snapshot_book(out):
    """Record the reviewed text as it stood at protocol creation, without editing it."""
    book=Path("E:/rfly-books/Rfly-Stable-Learning-Book_Reviewed_20261022")
    paths=["BookEnLatex.tex"]+[f"ch{n}/chapter{n}.tex" for n in (2,5,6,8,9,10,12,13,14,15,16)]
    sources={}
    for relative in paths:
        src=book/relative;dest=out/"book_source_snapshot"/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dest);sources[str(src)]=sha(dest)
    document=ROOT/"docs/BOOK_MULTI_SEED_V4_ALIGNMENT.md"
    shutil.copyfile(document,out/"BOOK_ALIGNMENT.md")
    value=dict(book_root=str(book),sources=sources,alignment_document_sha256=sha(out/"BOOK_ALIGNMENT.md"),
        architecture_html=str(ROOT/"docs/book_toolbox_architecture_v4.html"),
        mechanisms=["coupled_linear","point_mass","pendulum","cartpole"],
        disk_scope="ch8 fixed-V numerical control and ch9 two-parameter D regression; separate protocol",
        specialized_chapters={"13":"multirotor visual servo, existing specialized backends",
            "14":"Push-T / cross-shape manipulation, existing specialized backends",
            "15":"visual relative-pose observer, method plus scalar checks; full 6DoF experiment absent",
            "16":"text-to-image, method plus quadratic checks; full generation experiment absent"},
        claim="Mechanism-level empirical alignment only; exact book physical models and all application chapters are not covered")
    write(out/"BOOK_ALIGNMENT.json",value)
    return value


def train_job(out, job, stop):
    protocol=job["protocol"]; old.check_sources(protocol)
    dest=out/job["path"]; folder=dest/job["task"]; folder.mkdir(parents=True,exist_ok=True)
    if (folder/"TRAINING_COMPLETE.json").exists():
        done=read(folder/"TRAINING_COMPLETE.json")
        if sha(folder/"last.pt")!=done["checkpoint_sha256"]:
            raise ValueError("Completed checkpoint changed")
        selection=dict(tasks={job["task"]:done},protocol_sha256=sha(dest/"protocol.json"),
                       test_access_at_selection=False)
        if (dest/"SELECTION_COMMITTED.json").exists():
            if read(dest/"SELECTION_COMMITTED.json")!=selection:
                raise ValueError("Completed job selection changed")
        else:
            write(dest/"SELECTION_COMMITTED.json",selection)
        return done
    task,policy,learner=old.construct(job["task"],protocol)
    learner.should_stop=stop
    if (folder/"last.pt").exists():
        learner.restore(folder/"last.pt")
        emit(out,"job_resumed",task=job["task"],seed=job["seed"],update=learner.updates)
    else:
        learner.initialize(); learner.save(folder/"last.pt")
        emit(out,"baseline_qualified",task=job["task"],seed=job["seed"],update=0,
             validation=learner.history[-1],cost_scale=learner.cost_scale)
    while learner.updates<learner.config.updates and learner.stale<learner.config.patience:
        if stop():
            raise InterruptedError("STOP at whole-update boundary")
        started=time.perf_counter(); row=learner.update(); validation=None
        if learner.updates%learner.config.validation_every==0 or learner.updates==learner.config.updates:
            validation=learner.validate()
        learner.save(folder/"last.pt")
        emit(out,"update_committed",task=job["task"],seed=job["seed"],**row,
             wall_seconds=time.perf_counter()-started,best_update=learner.best_update,validation=validation)
    policy.load_state_dict(learner.best_actor)
    done=dict(task=job["task"],seed=job["seed"],updates=learner.updates,best_update=learner.best_update,
              best_validation_ratio=learner.best_score,best_actor_sha256=module_hash(policy),
              checkpoint_sha256=sha(folder/"last.pt"),model_transitions=learner.model_transitions,
              physical=dict(dt=task.dt,horizon=task.max_steps,goal=task.goal.tolist()),
              stopped_by="validation_patience" if learner.stale>=learner.config.patience else "update_budget")
    write(folder/"TRAINING_HISTORY.json",learner.history)
    write(folder/"TRAINING_COMPLETE.json",done)
    write(dest/"SELECTION_COMMITTED.json",dict(tasks={job["task"]:done},
        protocol_sha256=sha(dest/"protocol.json"),test_access_at_selection=False))
    emit(out,"task_training_complete",**done)
    return done


def commit_all(out, protocol, selections):
    expected={j["id"] for j in protocol["jobs"]}
    if set(selections)!=expected:
        raise ValueError("Every task/seed must finish before test access")
    value=dict(schema="global_validation_selection_v4",protocol_sha256=sha(out/"protocol.json"),
        selections=selections,test_access_at_selection=False)
    path=out/"GLOBAL_SELECTION_COMMITTED.json"
    if path.exists():
        if read(path)!=value:
            raise ValueError("Global selection changed after commitment")
    else:
        write(path,value)
    return value


def require_commitment(out, protocol):
    value=read(out/"GLOBAL_SELECTION_COMMITTED.json")
    if value["protocol_sha256"]!=sha(out/"protocol.json") or value["test_access_at_selection"]:
        raise ValueError("Invalid global commitment")
    if set(value["selections"])!={j["id"] for j in protocol["jobs"]}:
        raise ValueError("Incomplete global commitment")
    for job in protocol["jobs"]:
        dest=out/job["path"]
        selected=value["selections"][job["id"]]
        if sha(dest/job["task"]/"last.pt")!=selected["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed")
        if read(dest/"SELECTION_COMMITTED.json")["tasks"][job["task"]]!=selected:
            raise ValueError("Job and global selection differ")
    return value


def evaluate_jobs(out, protocol, stop):
    require_commitment(out,protocol)
    result=dict(schema="book_multiseed_v4_results",tasks={t:dict(seeds={}) for t in protocol["tasks"]},
        seeds=protocol["seeds"],audited_episodes=0,formal_stability_proof=False,
        development_only=protocol["settings"].get("development_only",False),
        scope="five independent training seeds; no seed selection; full-horizon CPU float64 model gradients",
        test_after_global_selection=True)
    for job in protocol["jobs"]:
        if stop():
            raise InterruptedError("STOP before test job")
        emit(out,"job_test_started",task=job["task"],seed=job["seed"])
        dest=out/job["path"]
        report=old.evaluate_all(dest,job["protocol"],stop)
        write(dest/"RESULTS.json",report)
        item=report["tasks"][job["task"]]
        item["job_path"]=job["path"]
        item["physical"]=item["selection"]["physical"]
        result["tasks"][job["task"]]["seeds"][str(job["seed"])]=item
        result["audited_episodes"]+=report["audited_episodes"]
        emit(out,"job_test_complete",task=job["task"],seed=job["seed"],
             qualified=item["test_gate"]["qualified"],reasons=item["test_gate"]["reasons"],
             comparisons=item["comparisons"],audit_passed=item["audit_passed"])
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"configs/book_multiseed_v4.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(); out=args.output.resolve()
    if out.parent!=ROOT/"runs":
        raise ValueError("Use a new direct child of runs")
    torch.set_num_threads(1)
    with old.writer_lock():
        if args.resume:
            protocol=read(out/"protocol.json");old.check_sources(protocol)
            if (out/"STOP").exists() or (out/"FINAL_STATE.json").exists():
                raise ValueError("Stopped or finalized run cannot resume")
        else:
            settings=read(args.config);sources=source_manifest();jobs=make_jobs(settings,sources)
            if settings["seeds"]!=list(SEEDS) or settings["tasks"]!=list(TASKS):
                raise ValueError("Formal protocol requires all five seeds and all four bases")
            out.mkdir(exist_ok=False)
            book_alignment=snapshot_book(out)
            protocol=dict(schema="book_multiseed_v4",created_at=now(),sources=sources,jobs=jobs,
                tasks=settings["tasks"],seeds=settings["seeds"],settings=settings,
                book_alignment=book_alignment,
                deadline_unix=time.time()+settings["maximum_seconds"],config_sha256=sha(args.config),
                authorization="对照书稿文本要求，开始多个种子训练，生成训练曲线/结果和工具箱架构",
                existing_runs="sealed; all previous STOP files remain untouched",
                writer_rule="external observers read only protocol.json and committed events/*.json",
                recovery="same frozen source/protocol, whole-update checkpoint only; never bypass STOP; incomplete test groups require diagnosis")
            write(out/"protocol.json",protocol)
            for relative in sources:
                target=out/"source_snapshot"/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/relative,target)
            for job in jobs:
                write(out/job["path"]/"protocol.json",job["protocol"])
        def stop():
            if (out/"STOP").exists():
                return True
            if time.time()>protocol["deadline_unix"]:
                try:
                    with (out/"STOP").open("x",encoding="utf-8") as stream:
                        stream.write("Declared eight-hour budget reached; deletion requires explicit authorization.\n")
                except FileExistsError:
                    pass
                return True
            return False
        try:
            emit(out,"pipeline_started",pid=os.getpid(),tasks=protocol["tasks"],seeds=protocol["seeds"],
                 maximum_seconds=protocol["settings"]["maximum_seconds"])
            selections={}
            for job in protocol["jobs"]:
                emit(out,"job_training_started",task=job["task"],seed=job["seed"])
                selections[job["id"]]=train_job(out,job,stop)
            old.check_sources(protocol)
            commit_all(out,protocol,selections)
            emit(out,"all_models_selected",jobs=len(selections),test_access=False)
            result=evaluate_jobs(out,protocol,stop)
            old.check_sources(protocol)
            write(out/"RESULTS.json",result)
            from .figures import build_figures
            figures=build_figures(out,result=result)
            old.check_sources(protocol)
            final=dict(status="development_smoke_complete" if result["development_only"] else "multiseed_complete",at=now(),jobs=len(selections),
                all_test_gates_passed=all(s["test_gate"]["qualified"] for t in result["tasks"].values() for s in t["seeds"].values()),
                audited_episodes=result["audited_episodes"],formal_stability_proof=False,
                results_sha256=sha(out/"RESULTS.json"),figures=figures)
            write(out/"FINAL_STATE.json",final);emit(out,"pipeline_complete",**final)
        except BaseException as error:
            status="stopped" if isinstance(error,(InterruptedError,KeyboardInterrupt)) else "failed"
            # Timestamped diagnostics preserve earlier failure evidence on recovery.
            write(out/("DIAGNOSTIC_"+str(time.time_ns())+".json"),dict(status=status,at=now(),
                error=repr(error),traceback=traceback.format_exc()))
            emit(out,"pipeline_"+status,error=repr(error))
            raise


if __name__=="__main__":
    main()
