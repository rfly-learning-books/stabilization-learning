"""Protocol-first generation experiment, with held-out prompts and full outputs."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import uuid

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .env import GenerationEnv
from .algo import fit,choose


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024**2),b""):digest.update(block)
    return digest.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with temporary.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    temporary.rename(path)


def csv_write(path,rows):
    if not rows:return
    with Path(path).open("x",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def gpu_sample():
    try:
        raw=subprocess.check_output(["nvidia-smi","--query-gpu=timestamp,utilization.gpu,memory.used,power.draw","--format=csv,noheader,nounits"],timeout=10,text=True)
        values=raw.strip().splitlines()[0].split(",")
        return dict(unix=time.time(),gpu_percent=float(values[1]),memory_mib=float(values[2]),power_watts=float(values[3]))
    except Exception as error:return dict(unix=time.time(),error=repr(error))


def audit_gradient(env,x0):
    value,direction,rms=env.local_gradient(x0)
    point=x0.detach().requires_grad_(True)
    derivative=torch.autograd.grad(env.proxy(point),point)[0]
    predicted=float((derivative*direction).sum())
    records=[]
    with torch.no_grad():
        for delta in (.0005,.001,.002):
            plus=float(env.proxy(x0+delta*direction));minus=float(env.proxy(x0-delta*direction))
            finite=(plus-minus)/(2*delta)
            records.append(dict(step=delta,analytic=predicted,central_difference=finite,
                relative_error=abs(finite-predicted)/max(1e-6,abs(predicted))))
    return dict(proxy=value,gradient_rms=rms,checks=records,
        passed=all(math.isfinite(row["relative_error"]) for row in records) and min(r["relative_error"] for r in records)<.25,
        scope="Derivative of CLIP proxy through VAE with epsilon held fixed; not derivative of the full sampler")


def collect(env,prompts,seed,strengths,output,guard,event):
    features=[];improvements=[];rows=[];gradient_audit=None
    for pindex,prompt in enumerate(prompts):
        guard();latent=env.reset(prompt["text"],seed+pindex)
        snapshots={key:[] for key in ("latent","x0","epsilon","direction")}
        for index in range(env.steps):
            guard();x0,epsilon,previous=env.predict(latent,index)
            value,direction,rms=env.local_gradient(x0)
            label=[]
            with torch.no_grad():
                for strength in strengths:
                    label.append(0. if strength==0 else value-float(env.proxy(x0+strength*direction)))
            features.append(env.features(index,env.steps,value,rms,x0,epsilon));improvements.append(label)
            rows.append(dict(prompt_id=prompt["id"],seed=seed+pindex,index=index,proxy=value,improvements=label))
            for key,tensor in (("latent",latent),("x0",x0),("epsilon",epsilon),("direction",direction)):
                snapshots[key].append(tensor[0].detach().cpu().numpy().astype(np.float32))
            if gradient_audit is None and index==env.steps//2:
                gradient_audit=audit_gradient(env,x0)
                write(output/"GRADIENT_AUDIT.json",gradient_audit)
                if not gradient_audit["passed"]:raise RuntimeError("Local gradient finite-difference audit failed")
            # Training contexts follow the original, uncorrected sampler.
            latent=env.advance(x0,epsilon,previous)
        np.savez_compressed(output/(prompt["id"]+"_contexts.npz"),**{k:np.stack(v) for k,v in snapshots.items()})
        Image.fromarray(env.image(latent)).save(output/(prompt["id"]+"_original.png"))
        event("collection_prompt",prompt=prompt["id"],contexts=len(features))
    data=dict(features=np.stack(features),improvements=np.asarray(improvements,np.float32))
    np.savez_compressed(output/"dataset.npz",**data);write(output/"labels.json",rows)
    return data


def independent_evaluate(assets,records,device,guard):
    from transformers import SiglipModel,AutoProcessor
    from torch.nn import functional as F
    model=SiglipModel.from_pretrained(str(assets/"siglip"),local_files_only=True).to(device).eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(str(assets/"siglip"),local_files_only=True)
    embeddings={};results=[]
    with torch.no_grad():
        for row in records:
            guard()
            image=Image.open(row["image"]).convert("RGB")
            inputs=processor(text=[row["prompt"]],images=image,padding="max_length",return_tensors="pt").to(device)
            outputs=model(**inputs)
            score=float(outputs.logits_per_image[0,0].sigmoid())
            features=F.normalize(outputs.image_embeds.float(),dim=-1)[0].cpu().numpy()
            embeddings[row["id"]]=features
            results.append(dict(id=row["id"],method=row["method"],prompt_id=row["prompt_id"],family=row["family"],
                seed=row["seed"],siglip_match_probability=score,image=row["image"]))
    del model,processor;torch.cuda.empty_cache()
    diversity=[]
    grouped=defaultdict(list)
    for row in results:grouped[(row["method"],row["prompt_id"])].append(row)
    for (method,prompt),group in grouped.items():
        if len(group)!=2:continue
        a,b=group
        pa=np.asarray(Image.open(a["image"]),np.float32)/255
        pb=np.asarray(Image.open(b["image"]),np.float32)/255
        diversity.append(dict(method=method,prompt_id=prompt,embedding_cosine_distance=float(1-np.dot(embeddings[a["id"]],embeddings[b["id"]])),
            pixel_mean_absolute_difference=float(np.abs(pa-pb).mean())))
    return results,diversity


def figures(output,history,records,evaluation,diversity,config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory=output/"figures";directory.mkdir()
    paths=[]
    def save(fig,name):
        for extension in ("png","pdf"):
            path=directory/(name+"."+extension);fig.savefig(path,dpi=300,bbox_inches="tight");paths.append(str(path))
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for key,label in (("train_mse","Train"),("val_mse","Validation")):
        axes[0].semilogy([r["update"] for r in history],[max(1e-12,r[key]) for r in history],label=label)
    axes[0].set(xlabel="GPU update",ylabel="Local improvement MSE");axes[0].legend()
    axes[1].plot([r["update"] for r in history],[r["val_local_regret"] for r in history]);axes[1].set(xlabel="GPU update",ylabel="Validation local action regret")
    save(fig,"training")
    methods=config["methods"];fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for method in methods:
        rows=[r for r in records if r["method"]==method]
        axes[0].plot(np.mean([[s["proxy_before"] for s in r["steps"]] for r in rows],axis=0),label=method)
    axes[0].set(xlabel="DDIM step",ylabel="Mean CLIP proxy");axes[0].legend(fontsize=8)
    means=[np.mean([r["siglip_match_probability"] for r in evaluation if r["method"]==m]) for m in methods]
    times=[np.mean([r["wall_seconds"] for r in records if r["method"]==m]) for m in methods]
    axes[1].bar(methods,means);axes[1].set(ylabel="Independent SigLIP score (descriptive)")
    axes[2].bar(methods,times);axes[2].set(ylabel="Seconds / image, including audit")
    for ax in axes[1:]:ax.tick_params(axis="x",rotation=25)
    save(fig,"semantic_proxy_and_cost")
    # Every test image appears; no aesthetic selection or dropped failures.
    for seed in config["test_seeds"]:
        prompts=config["test_prompts"];fig,axes=plt.subplots(len(prompts),len(methods),figsize=(12,3*len(prompts)),squeeze=False)
        for i,prompt in enumerate(prompts):
            for j,method in enumerate(methods):
                row=next(r for r in records if r["prompt_id"]==prompt["id"] and r["seed"]==seed and r["method"]==method)
                axes[i,j].imshow(Image.open(row["image"]));axes[i,j].axis("off")
                axes[i,j].set_title((method if i==0 else "")+"\n"+(prompt["id"]+": "+prompt["text"] if j==0 else ""),fontsize=7,wrap=True)
        save(fig,"paired_images_seed_"+str(seed))
    return paths


def run(config_path,output):
    config_path=Path(config_path).resolve();config=json.loads(config_path.read_text(encoding="utf-8-sig"))
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);write(output/"WRITER.lock",dict(pid=os.getpid()))
    begun=time.time();hard=min(begun+config["maximum_minutes"]*60,datetime.fromisoformat(config["deadline_iso"]).timestamp())
    assets=Path(config["assets"]);events=0;telemetry=[]
    def event(kind,**data):
        nonlocal events
        events+=1;row=dict(sequence=events,kind=kind,unix=time.time(),**data)
        write(output/"events"/f"{events:06d}_{kind}.json",row);print(json.dumps(row),flush=True)
    def guard():
        if time.time()>hard or (output/"STOP").exists():raise InterruptedError("Generation deadline or STOP")
        if shutil.disk_usage(output).free<config["minimum_free_gib"]*1024**3:raise InterruptedError("Generation disk reserve")
    state=dict(status="running",started_unix=begun,deadline_unix=hard);env=None
    try:
        guard()
        files=list(Path(__file__).parent.glob("*.py"))+[config_path,Path(__file__).resolve().parents[2]/"scripts/run_book_generation_overnight_v1.py"]
        models={str(p.relative_to(assets)):sha(p) for p in assets.rglob("*") if p.is_file() and not p.name.endswith(".download")}
        if not (assets/"MODEL_MANIFEST.json").exists():raise RuntimeError("Pinned model download is incomplete")
        protocol=dict(schema="book_e25_local_diffusion_guidance_v1",config=config,sources={str(p):sha(p) for p in files},models=models,
            runtime={p:importlib.metadata.version(p) for p in ("torch","diffusers","transformers","numpy","safetensors")},
            device=torch.cuda.get_device_name(0),
            proxy="1 - cosine(CLIP_B32(decode(x0)), CLIP_B32(text))",action="x0 correction, epsilon fixed, RMS-bounded normalized negative gradient",
            equality="Paired initial noise; 20 conditional UNet calls; at most 4 local candidate proxy evaluations. Actual costs including audit are recorded.",
            split="Training objects/scenes, validation materials, held-out count/spatial/text/unusual prompt families; no test-based selection",
            limits=["Local x0 proxy decrease is not a pathwise or terminal semantic guarantee","One learned-guidance training seed; two held-out noise seeds",
                    "SigLIP is independent of the guidance CLIP but is not human judgment","No reference distribution or human blind ratings: no FID or broad quality claim"])
        write(output/"protocol.json",protocol)
        torch.manual_seed(config["seed"]);np.random.seed(config["seed"]);torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        env=GenerationEnv(assets,config["image_size"],config["steps"],config["cfg"],config["device"])
        for phase in ("train","validation"):(output/phase).mkdir()
        event("models_loaded");telemetry.append(gpu_sample())
        training=collect(env,config["train_prompts"],config["seed"]+100,config["strengths"],output/"train",guard,event)
        validation=collect(env,config["validation_prompts"],config["seed"]+200,config["strengths"],output/"validation",guard,event)
        critic,history,selection=fit(training,validation,len(config["strengths"]),config["updates"],config["seed"],env.device,guard)
        csv_write(output/"training.csv",history);torch.save(critic.state_dict(),output/"guide_critic.pt")
        write(output/"FREEZE.json",dict(selection=selection,checkpoint_sha256=sha(output/"guide_critic.pt"),
            protocol_sha256=sha(output/"protocol.json"),test_accessed=False))
        event("training_frozen",**selection);records=[];(output/"images").mkdir();(output/"trajectories").mkdir()
        for prompt in config["test_prompts"]:
            for seed in config["test_seeds"]:
                for method in config["methods"]:
                    guard();latent=env.reset(prompt["text"],seed);env.sync();start=time.perf_counter();trace=[];snapshots=[]
                    for index in range(env.steps):
                        guard();x0,epsilon,previous=env.predict(latent,index)
                        corrected,metrics,_=choose(method,env,x0,index,epsilon,config["strengths"],critic)
                        trace.append(dict(index=index,timestep=int(env.scheduler.timesteps[index]),**metrics))
                        snapshots.append(torch.stack([latent[0],x0[0],epsilon[0],corrected[0]-x0[0]]).detach().cpu().numpy())
                        latent=env.advance(corrected,epsilon,previous)
                    pixels=env.image(latent);env.sync();seconds=time.perf_counter()-start
                    identifier=f"{prompt['id']}_{seed}_{method}";image_path=output/"images"/(identifier+".png")
                    Image.fromarray(pixels).save(image_path)
                    np.savez_compressed(output/"trajectories"/(identifier+".npz"),states=np.stack(snapshots),final_latent=latent.detach().cpu().numpy())
                    record=dict(id=identifier,prompt_id=prompt["id"],family=prompt["family"],prompt=prompt["text"],seed=seed,method=method,
                        image=str(image_path),wall_seconds=seconds,counts=dict(env.counts),steps=trace)
                    write(output/"trajectories"/(identifier+".json"),record);records.append(record)
                    event("test_image",id=identifier,wall_seconds=seconds);telemetry.append(gpu_sample())
        env.close();env=None
        evaluation,diversity=independent_evaluate(assets,records,config["device"],guard)
        csv_write(output/"independent_semantics.csv",evaluation);csv_write(output/"diversity.csv",diversity)
        # A fixed blind ordering and answer key are delivered, but no human scores are invented.
        generator=np.random.default_rng(841);order=generator.permutation(len(records))
        blind=[];(output/"blind_review").mkdir()
        for i,index in enumerate(order):
            row=records[int(index)];name=f"image_{i+1:03d}.png"
            shutil.copyfile(row["image"],output/"blind_review"/name)
            blind.append(dict(blind_id=name,id=row["id"],prompt=row["prompt"],method=row["method"]))
        write(output/"BLIND_REVIEW_KEY.json",blind)
        (output/"blind_review"/"INSTRUCTIONS.txt").write_text("Rate text adherence and visible quality separately on 1-5 scales. Judge count, relation and text literally. Randomized identifiers hide method; consult prompts in the review key only through a coordinator. No human ratings have yet been collected.\n",encoding="utf-8")
        summaries=[]
        for method in config["methods"]:
            group=[r for r in records if r["method"]==method];scores=[r["siglip_match_probability"] for r in evaluation if r["method"]==method]
            summaries.append(dict(method=method,images=len(group),siglip_mean=float(np.mean(scores)),wall_seconds_mean=float(np.mean([r["wall_seconds"] for r in group])),
                local_increases=sum(s["local_delta"]>1e-7 for r in group for s in r["steps"]),
                mean_correction_rms=float(np.mean([s["correction_rms"] for r in group for s in r["steps"]]))))
        result=dict(schema="book_e25_result_v1",status="complete",scope="SD1.5 local x0 guidance application; full E25 acceptance awaits human and distributional evaluation",
            summary=summaries,selection=selection,limits=protocol["limits"])
        write(output/"RESULTS.json",result);paths=figures(output,history,records,evaluation,diversity,config)
        write(output/"FIGURES_COMPLETE.json",dict(files=paths));write(output/"GPU_SAMPLES.json",telemetry)
        (output/"REPORT.md").write_text("# 文本条件生成实验\n\n冻结 SD1.5 与 20 步 DDIM。比较原条件采样、固定幅度局部引导、有限回溯和学习幅度选择；每项完整输出均已保存。\n\n学习器拟合固定噪声预测下的局部 CLIP 代理改善，在独立提示词验证集上选模，冻结后才访问测试提示词。SigLIP 独立评价不参与学习。逐步局部下降、终端语义分数和图像质量分别解释。当前未完成人工盲评，也没有 FID 参考分布。\n\n计算费用包含 UNet、VAE、语义梯度、回溯及审计；相同 UNet 步数不等于相同墙钟。原采样的逐步代理解码仅用于审计。\n\n```json\n"+json.dumps(summaries,ensure_ascii=False,indent=2)+"\n```\n",encoding="utf-8")
        state["status"]="complete"
    except BaseException as error:
        state.update(status="stopped" if isinstance(error,(KeyboardInterrupt,InterruptedError)) else "failed",error=repr(error),traceback=traceback.format_exc())
    finally:
        if env is not None:env.close()
        state["finished_unix"]=time.time();write(output/"FINAL_STATE.json",state)
        (output/"WRITER.lock").unlink();event("finished",status=state["status"])
    return state
