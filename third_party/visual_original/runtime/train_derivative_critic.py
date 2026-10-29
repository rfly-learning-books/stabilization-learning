"""Independent protocol: AE-dominant physical-derivative representation learning."""
from __future__ import annotations
import argparse
import copy
import hashlib
import html
import json
import math
import shutil
import time
import traceback
from pathlib import Path
import cv2
import numpy as np
import torch

import train_goal_fixed as joint
from derivative_critic_core import DerivativeLearner, DerivativeCritic, derivative_coefficients, physical_scales
from goal_fixed_core import ROOT, Replay, FastDepth, GoalMetric, policy, pose_error, p0_offset_step, digest, source_fingerprints, resource, emit
from lac_vs.config import load_config, seed_all, json_write, json_append, rng_state, restore_rng
from lac_vs.environment import sample_start
from lac_vs.locking import writer_lock
from lac_vs import legacy


def tensor_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_payload(cfg):
    source=ROOT/cfg['derivative']['source_run']
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    payload=torch.load(source/cfg['derivative']['initialization_checkpoint'],map_location='cpu')
    expected=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if payload['protocol_hash']!=expected:raise RuntimeError('Source protocol mismatch')
    if any(payload['learner']['counts'].values()):raise RuntimeError('Initialization must precede old AE/TD/M optimization')
    if any(r['split']!='train' for r in payload['replay_rows']):raise RuntimeError('Source contains held-out replay')
    for name,value in protocol['sources'].items():
        if digest(ROOT/name)!=value:raise RuntimeError(f'Read-only source changed: {name}')
    return source,payload


def prepare(out,cfg):
    source,_=source_payload(cfg)
    files=source_fingerprints(cfg)
    for name in ('train_derivative_critic.py','derivative_critic_core.py'):
        rel='runtime/'+name; files[rel]=digest(ROOT/rel)
    expected=dict(config=cfg,sources=files,dependencies=joint.dependencies(cfg),
        initialization={name:digest(source/name) for name in
            ('protocol.json','initial.pt','scenes.json','evaluation_starts.json','training_anchors.json')})
    marker=out/'protocol.json'
    if marker.exists():
        saved=json.loads(marker.read_text(encoding='utf-8'))
        for key,value in expected.items():
            if saved[key]!=value:raise RuntimeError(f'Protocol {key} changed; use a new output')
        return saved
    if out.exists() and any(out.iterdir()):raise RuntimeError('Nonempty output without protocol')
    out.mkdir(parents=True,exist_ok=True)
    for name in ('scenes.json','evaluation_starts.json','training_anchors.json'):
        shutil.copy2(source/name,out/name)
    protocol=dict(schema=1,created=joint.now(),**expected,
        algorithm='AE-dominant RGB features with supervised signed physical error-rate Critic and goal-only SPD M',
        critic='D(z,z_goal,v_physical)=a(z,z_goal)^T v; four physical velocity coefficients; goal coefficient exactly zero',
        teacher='a=[2 R(yaw)^T p, 2 yaw/(25deg)^2], frozen static-target P0 kinematics; signed raw physical units',
        representation='AE RGB+DCT plus normalized derivative MSE; derivative encoder gradient cap .25 AE, regular cap .05 AE',
        no_bellman=True,no_critic_bootstrap=True,old_cost_critic_weights_imported=False,
        shadow_control='same initial AE, same sampled RGB/triplets, same warmup/update count and regularization, no derivative loss; data collected by main student',
        metric_update='old SPD controller/other M terms retained; old learned cumulative-cost term replaced by normalized signed derivative prediction',
        input='one RGB per shared encoder call; Critic receives only z,z_goal,physical velocity; pose only in labels and evaluation',
        fixed_networks_within_episode=True,fixed_reference_J=True,depth='same public FastDepth reproduction',
        selection='all 8 validation layouts, macro success/worst scene/negative integrated physical error; no test selection',
        progress_checkpoint='full model checkpoint at optimization/stage boundaries; atomic compact per-episode RNG/replay checkpoint tied to full-model SHA256',
        limitation='derivative predicts P0 command kinematics; P1 hidden actuator state is tested only as closed-loop stress; metric-scale ambiguity of RGB remains',
        guarantee='candidate certificate; true RGB V descent and physical pose outcomes must be checked independently')
    json_write(marker,protocol)
    import yaml
    (out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    for rel in files:
        p=out/'provenance'/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((ROOT/rel).read_bytes())
    return protocol


def import_training_data(out,cfg,replay,payload):
    source=ROOT/cfg['derivative']['source_run']; rows=copy.deepcopy(payload['replay_rows'])
    if cfg.get('engineering_smoke'):rows=rows[:2]
    for row in rows:
        src=source/'replay'/row['file']; dst=replay.root/row['file']
        if not dst.exists():shutil.copy2(src,dst)
        if digest(src)!=digest(dst):raise RuntimeError('Replay evidence copy mismatch')
    replay.rows=rows; json_write(replay.path,rows)
    shutil.copytree(source/'references',out/'references',dirs_exist_ok=True)
    json_write(out/'import_provenance.json',dict(source=str(source),checkpoint='initial.pt',
        source_optimizer_updates=payload['learner']['counts'],transitions=replay.transitions,shards=len(rows),
        critic_reinitialized=True,initial_M_reused=True,train_only=True))


def audit(out,cfg,learner,replay):
    snapshot=copy.deepcopy(learner.payload()); global_rng=rng_state(); rr=copy.deepcopy(replay.rng.bit_generator.state)
    checks={}
    try:
        off=torch.tensor([[.4,-.3,.2,.2],[-.7,.2,-.1,-.35],[.001,-.002,.003,.001]],dtype=torch.double)
        u=torch.tensor([[.2,-.1,.4,-.3],[-.5,.4,.2,.6],[.1,.2,-.3,.4]],dtype=torch.double)
        a=derivative_coefficients(off); true=(a*u*physical_scales(u)).sum(-1)
        eps=1e-5
        fd=(pose_error(p0_offset_step(off,u,eps))-pose_error(p0_offset_step(off,u,-eps)))/(2*eps)
        checks['physical_derivative_matches_pose_finite_difference']=bool(torch.allclose(true,fd,atol=1e-8,rtol=1e-7))
        critic=DerivativeCritic().double(); z=torch.randn(4,16,dtype=torch.double); g=torch.randn_like(z)
        v=torch.randn(4,4,dtype=torch.double,requires_grad=True)
        q=critic(z,g,v)
        grad=torch.autograd.grad(q.sum(),v,retain_graph=True)[0]
        checks['critic_action_gradient_equals_coefficients']=bool(torch.allclose(grad,critic.coefficients(z,g)))
        checks['critic_zero_velocity']=bool(torch.equal(critic(z,g,torch.zeros_like(v)),torch.zeros(4,dtype=torch.double)))
        checks['critic_velocity_odd']=bool(torch.allclose(q,-critic(z,g,-v)))
        checks['goal_rate_zero_for_every_action']=bool(torch.equal(critic(g,g,v),torch.zeros(4,dtype=torch.double)))
        checks['critic_output_can_have_both_signs']=bool((critic(z,g,grad.detach())>0).all() and (critic(z,g,-grad.detach())<0).all())
        numerical=(critic(z,g,v.detach()+eps*torch.eye(4,dtype=torch.double)[0])-critic(z,g,v.detach()-eps*torch.eye(4,dtype=torch.double)[0]))/(2*eps)
        checks['critic_action_gradient_finite_difference']=bool(torch.allclose(numerical,grad[:,0],atol=1e-8,rtol=1e-6))
        b=replay.sample(cfg['goal_fixed']['batch_size'])
        x=learner.encode(b['rgb']); goal=learner.encode(b['goal_rgb'])
        loss,_=learner.supervised_loss(x,goal,learner.tensor(b['offset']),learner.candidate_actions(b))
        eg=torch.autograd.grad(loss,list(learner.ae.base.encoder.parameters()),allow_unused=True)
        enorm=float(torch.stack([p.square().sum() for p in eg if p is not None]).sum().sqrt())
        checks['supervised_derivative_updates_phi']=enorm>0
        before_m=tensor_hash(learner.metric); before_ae=tensor_hash(learner.ae)
        rep=learner.representation(b)
        checks['representation_changes_phi']=tensor_hash(learner.ae)!=before_ae
        checks['representation_does_not_change_M']=tensor_hash(learner.metric)==before_m
        checks['AE_dominant_derivative_gradient']=rep['critic_encoder_scale']*rep['critic_encoder_grad']<=.2501*rep['ae_encoder_grad']+1e-8
        checks['AE_dominant_regular_gradient']=rep['regular_encoder_scale']*rep['regular_encoder_grad']<=.0501*rep['ae_encoder_grad']+1e-8
        checks['paired_AE_control_has_same_update_and_distinct_features']=tensor_hash(learner.shadow)!=tensor_hash(learner.ae)
        before_ae=tensor_hash(learner.ae); before_c=tensor_hash(learner.critics)
        ml=learner.metric_update(b)
        checks['M_nonzero_gradient']=ml['gradient_norm']>0
        checks['M_update_encoder_isolation']=tensor_hash(learner.ae)==before_ae and ml['encoder_leak']==0
        checks['M_update_critic_isolation']=tensor_hash(learner.critics)==before_c and ml['critic_leak']==0
        mm=GoalMetric().double(); gg=torch.randn(8,16,dtype=torch.double)
        e=torch.randn(8,16,dtype=torch.double); j=torch.randn(8,16,4,dtype=torch.double)
        m=mm(gg)[1]; cmd,h=policy(e,j,m); ev=torch.linalg.eigvalsh(m)
        checks['M_SPD_trace_bounds']=bool(ev.min()>=.1-1e-9 and ev.max()<=14.5+1e-9 and
            torch.allclose(m.diagonal(dim1=-2,dim2=-1).sum(-1),torch.full((8,),16.,dtype=torch.double)))
        checks['speed_constraints']=bool(cmd[:,:3].norm(dim=-1).max()<=1.000001 and cmd[:,3].abs().max()<=1.000001)
        checks['nominal_V_nonincrease']=bool(torch.einsum('bi,bij,bj->b',e,m,(j@cmd.unsqueeze(-1)).squeeze(-1)).max()<1e-9)
        checks['replay_training_only']=all(r['split']=='train' for r in replay.rows)
        checks['signed_derivative_network_replaces_cost_network']=len(learner.critics)==1 and isinstance(learner.critics[0],DerivativeCritic)
        result=dict(passed=all(checks.values()),timestamp=joint.now(),checks=checks,
            physical_derivative_max_error=float((true-fd).abs().max()),critic_encoder_gradient=enorm,
            representation_update=rep,metric_update=ml)
        json_write(out/'engineering_audit.json',result)
        if not result['passed']:raise RuntimeError(f'Derivative audit failed: {checks}')
        emit(event='derivative_engineering_audit',**result)
    finally:
        learner.restore(snapshot); learner.zero_all(); restore_rng(global_rng); replay.rng.bit_generator.state=rr


@torch.no_grad()
def critic_diagnostic(learner,arrays,maximum):
    indices=np.linspace(0,len(arrays['features'])-1,min(maximum,len(arrays['features']))).astype(int)
    goal=learner.encode(arrays['goal_rgb'][None])
    z=goal+4*learner.tensor(arrays['features'][indices])
    pred=learner.critics[0].coefficients(z,goal.expand_as(z))
    true=derivative_coefficients(learner.tensor(arrays['offset'][indices]))
    a=pred*physical_scales(pred); b=true*physical_scales(true)
    mask=b.norm(dim=-1)>1e-5
    cos=(a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)+1e-8)
    # Axis commands identify the entire action-linear function, not one recorded action.
    target=torch.cat([b,-b],-1); value=torch.cat([a,-a],-1); active=target.abs()>1e-4
    summary=dict(states=len(indices),rmse=float((value-target).square().mean().sqrt()),
        sign_accuracy=float((value[active]*target[active]>0).float().mean()) if active.any() else 1.,
        action_gradient_cosine=float(cos[mask].mean()) if mask.any() else 1.,
        scope='static-reference P0 instantaneous derivative; physical pose used only for labels')
    return summary,dict(critic_indices=indices,critic_pred_coeff=pred.cpu().numpy(),critic_true_coeff=true.cpu().numpy())


def evaluate(cfg,out,learner,depth,split,tag,methods,plants,stop,cap=None,stress=False):
    manifest=json.loads((out/'scenes.json').read_text(encoding='utf-8')); scenes={s['id']:s for s in manifest['scenes']}
    starts=json.loads((out/'evaluation_starts.json').read_text(encoding='utf-8'))[split]
    if cap:starts=joint.select_validation_subset(starts,cap)
    folder=out/'evaluation'/tag; folder.mkdir(parents=True,exist_ok=True)
    rows=[]; videos=set(); inst=None; sid=None; complete=True
    rng=np.random.default_rng(cfg['seed']+77001)
    try:
        for method in methods:
            main=learner.ae
            if method=='shadow_ae_identity':learner.ae=learner.shadow
            elif method=='initial_ae_identity':learner.ae=learner.initial
            try:
                for plant in plants:
                    for i,start in enumerate(starts):
                        path=folder/f'{method}_{plant}_{i:04d}.json'
                        if path.exists():
                            row=json.loads(path.read_text(encoding='utf-8')); rows.append(row)
                            if row.get('video'):videos.add((method,plant,row['difficulty'],row['success']))
                            continue
                        if stop.stopped():complete=False; break
                        scene=scenes[start['scene_id']]
                        if sid!=scene['id']:
                            if inst:inst.close()
                            inst=joint.new_instance(cfg,scene); sid=scene['id']
                        arrays,row=joint.episode(cfg,out,scene,start['goal_id'],start['offset'],learner,depth,rng,
                            steps=cfg['evaluation']['max_steps'],mode='learned' if method=='learned' else 'identity_m',
                            plant_name=plant,inst=inst,depth_scale=float(tag.rsplit('_',1)[-1]) if stress else 1.)
                        row.update(method=method,difficulty=start['difficulty'],index=i,tag=tag,checkpoint='best.pt',timestamp=joint.now())
                        if plant=='P0' and method in ('learned','identity_m'):
                            diagnostic,extra=critic_diagnostic(learner,arrays,cfg['derivative']['critic_diagnostic_states_per_episode'])
                            row['derivative_critic']=diagnostic; arrays.update(extra)
                        key=(method,plant,row['difficulty'],row['success'])
                        if key not in videos:
                            video=path.with_suffix('.mp4'); joint.write_video(video,arrays,row)
                            row['video']=str(video.relative_to(out)); videos.add(key)
                        telemetry={k:v for k,v in arrays.items() if k not in ('rgb','goal_rgb','smooth_center','smooth_minus','smooth_plus')}
                        with path.with_suffix('.npz').open('wb') as f:np.savez_compressed(f,**telemetry)
                        json_write(path,row); rows.append(row)
                        if (i+1)%12==0:
                            json_write(out/'status.json',dict(stage='evaluation',tag=tag,method=method,plant=plant,index=i+1,
                                total=len(starts),timestamp=joint.now(),resources=resource(out,cfg)))
                            emit(event='derivative_evaluation_progress',tag=tag,method=method,plant=plant,index=i+1,total=len(starts))
                    if not complete:break
            finally:learner.ae=main
            if not complete:break
    finally:
        if inst:inst.close()
    summaries={}
    for method in methods:
        for plant in plants:
            selected=[r for r in rows if r['method']==method and r['plant']==plant]
            if not selected:continue
            agg=joint.aggregate(selected)
            for key in ('certificate_comparable_fraction','latency_ms'):
                agg['mean_'+key]=float(np.mean([r[key] for r in selected]))
            diagnostics=[r['derivative_critic'] for r in selected if 'derivative_critic' in r]
            if diagnostics:
                agg['derivative_critic']={k:float(np.mean([d[k] for d in diagnostics])) for k in ('rmse','sign_accuracy','action_gradient_cosine')}
            summaries[f'{method}/{plant}']=agg
    result=dict(tag=tag,split=split,complete=complete,episodes=len(rows),planned=len(starts)*len(methods)*len(plants),summaries=summaries)
    json_write(folder/'summary.json',result); emit(event='derivative_evaluation_summary',**result)
    return result


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=out/'report'; dest.mkdir(exist_ok=True)
    status=json.loads((out/'status.json').read_text(encoding='utf-8')) if (out/'status.json').exists() else {}
    rows=[]
    for path in sorted((out/'evaluation').glob('*/summary.json')):
        result=json.loads(path.read_text(encoding='utf-8'))
        for key,v in result['summaries'].items():
            diag=v.get('derivative_critic',{})
            rows.append(f'<tr><td>{result["tag"]}</td><td>{key}</td><td>{v["episodes"]}</td><td>{100*v["success_rate"]:.1f}%</td>'
                f'<td>{v["mean_integrated_error"]:.4g}</td><td>{100*v["mean_actual_V_increase_fraction"]:.1f}%</td>'
                f'<td>{diag.get("sign_accuracy","—")}</td><td>{diag.get("action_gradient_cosine","—")}</td></tr>')
    unique={}
    if (out/'training.jsonl').exists():
        for line in (out/'training.jsonl').read_text(encoding='utf-8').splitlines():
            r=json.loads(line); unique[r['phase'],r['round'],r['update']]=r
    plots=''
    if unique:
        fig,axes=plt.subplots(2,2,figsize=(12,7))
        for phase in ('warmup','critic_prefit','representation','metric'):
            records=[r for r in unique.values() if r['phase']==phase]
            for ax,key in zip(axes.ravel(),('reconstruction','derivative_rmse','derivative_gradient_cosine','gradient_norm')):
                pairs=[(i,r[key]) for i,r in enumerate(records) if key in r]
                if pairs:ax.plot([p[0]*25 for p in pairs],[p[1] for p in pairs],label=phase); ax.set_title(key); ax.grid(alpha=.25); ax.legend()
        fig.tight_layout(); fig.savefig(dest/'training.png',dpi=140); plt.close(fig); plots='<img src="training.png">'
    examples=[]
    for tag in ('fixed_test','full_val'):
        for path in sorted((out/'evaluation'/tag).glob('*.json')):
            if path.name=='summary.json':continue
            r=json.loads(path.read_text(encoding='utf-8'))
            if not r.get('video') or r['plant']!='P0':continue
            video=r['video'].replace('\\','/')
            with np.load(path.with_suffix('.npz')) as a:off=a['offset']; v=a['V']; u=a['action']
            fig,axs=plt.subplots(1,3,figsize=(12,3)); t=np.arange(len(v))*.1
            axs[0].plot(t,100*np.linalg.norm(off[:,:3],axis=1)); axs[0].set_title('Position error (cm)')
            axs[1].plot(t,v,label='V'); axs[1].plot(t,pose_error(torch.from_numpy(off)).numpy(),label='E'); axs[1].legend()
            axs[2].plot(np.arange(len(u))*.1,np.linalg.norm(u,axis=-1)); axs[2].set_title('Command norm')
            for ax in axs:ax.grid(alpha=.25); ax.set_xlabel('Time (s)')
            fig.tight_layout(); name=f'{tag}_{path.stem}.png'; fig.savefig(dest/name,dpi=120); plt.close(fig)
            examples.append(f'<p>{r["scene_id"]} {r["method"]} {r["difficulty"]}: {r["status"]}</p><img src="{name}"><video controls preload="none" src="../{video}"></video>')
            if len(examples)>=16:break
        if len(examples)>=16:break
    complete=(out/'complete.json').exists()
    text=f'''<!doctype html><meta charset="utf-8"><title>AE＋物理下降率Critic</title>
<style>body{{max-width:1300px;margin:30px auto;font:16px/1.7 system-ui;padding:20px}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:7px}}img{{max-width:100%}}video{{width:700px;max-width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>AE主导＋物理误差下降率Critic：表征与目标M学习</h1>
<p>状态：{html.escape(str(status.get('stage','preparing')))}。完整测试{'已完成' if complete else '未完成，以下均为已记录数据'}。</p>
<p>Critic输入仅z、z*与物理速度，输出有符号dE/dt；真值由P0位姿运动学计算。无Bellman目标，无旧Critic权重。AE主导，Critic至编码器梯度上限为AE的25%，其余平滑正则上限5%。</p>
<p>同批次AE对照共享初始化、数据和更新次数，缺少下降率损失；其交互数据由主方法采集。M仍为目标条件Cholesky、trace16、0.1I至14.5I；其余原M目标保留，用归一化导数项替换累计代价项。</p>
{plots}<table><tr><th>数据集</th><th>方法/plant</th><th>回合</th><th>成功率</th><th>累计物理误差</th><th>V上升（回合均值）</th><th>导数符号准确率</th><th>动作梯度余弦</th></tr>{''.join(rows)}</table>
<h2>轨迹和成功失败录像</h2>{''.join(examples)}
<p>正定结构与预测下降不等于真实Lyapunov证书。P1是含执行器滞后的闭环压力测试，其指令不能按P0真值导数解释。纯RGB存在尺度与可观测性限制。单训练种子，测试不用于选模。</p>
<p><a href="../protocol.json">协议与哈希</a> · <a href="../engineering_audit.json">数学/梯度验收</a> · <a href="../analysis/RESULTS.json">完整最终统计</a></p>
<pre>{html.escape(json.dumps(status,ensure_ascii=False,indent=2))}</pre>'''
    (dest/'REPORT.html').write_text(text,encoding='utf-8')


def final_analysis(out,cfg):
    folder=out/'evaluation/fixed_test'; summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if not summary['complete']:raise RuntimeError('Incomplete fixed test')
    grouped={}; errors=[]; videos=0; frames=0
    allrows={}
    for path in sorted(folder.glob('*.json')):
        if path.name=='summary.json':continue
        row=json.loads(path.read_text(encoding='utf-8')); key=f'{row["method"]}/{row["plant"]}'
        allrows[row['method'],row['plant'],row['index']]=row
        g=grouped.setdefault(key,dict(transitions=0,V_rises=0,monotonic_successes=0))
        with np.load(path.with_suffix('.npz')) as a:
            e=a['features'].astype(np.float64); m=a['M'].astype(np.float64); v=a['V'].astype(np.float64)
            if np.max(np.abs(.5*np.einsum('ti,ij,tj->t',e,m,e)-v))>1e-3:errors.append(path.name+': energy')
            ev=np.linalg.eigvalsh(m)
            if ev.min()<.1-1e-5 or ev.max()>14.5+1e-4:errors.append(path.name+': SPD')
            diff=np.diff(v); g['transitions']+=len(diff); g['V_rises']+=int((diff>1e-10).sum())
            g['monotonic_successes']+=int(row['success'] and not (diff>1e-10).any())
    for g in grouped.values():g['transition_weighted_V_rise_fraction']=g['V_rises']/max(1,g['transitions'])
    comparisons=[]
    for plant in cfg['goal_fixed']['evaluation_plants']:
        for base in ('identity_m','shadow_ae_identity','initial_ae_identity'):
            pairs=[(allrows['learned',plant,i],allrows[base,plant,i]) for i in range(3 if cfg.get('engineering_smoke') else 360)]
            comparisons.append(dict(plant=plant,baseline=base,episodes=len(pairs),
                learned_only=sum(a['success'] and not b['success'] for a,b in pairs),
                baseline_only=sum(b['success'] and not a['success'] for a,b in pairs)))
    for tag in ('full_val','fixed_test','depth_scale_0.5','depth_scale_2.0'):
        for path in (out/'evaluation'/tag).glob('*.mp4'):
            cap=cv2.VideoCapture(str(path)); n=0
            while True:
                ok,_=cap.read()
                if not ok:break
                n+=1
            cap.release(); expected=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['steps']+1
            if n!=expected:errors.append(path.name+': video frames')
            videos+=1; frames+=n
    dest=out/'analysis'; dest.mkdir(exist_ok=True)
    result=dict(test=summary['summaries'],certificate=grouped,paired=comparisons,
        audit=dict(passed=not errors,errors=errors,decoded_videos=videos,decoded_frames=frames),
        clarification='This run changes representation supervision and replaces the accumulated-cost term; all remaining M terms retained.')
    json_write(dest/'RESULTS.json',result)
    if errors:raise RuntimeError(f'Final derivative-run audit failed: {errors[:5]}')


def run(out,cfg,args):
    seed_all(cfg['seed']); torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(cfg['goal_fixed']['memory_fraction'])
    protocol=prepare(out,cfg)
    if (out/'complete.json').exists():emit(event='already_complete',output=str(out)); return
    _,source=source_payload(cfg); learner=DerivativeLearner(cfg,source); replay=Replay(out,cfg)
    p=cfg['goal_fixed']; d=cfg['derivative']; stop=joint.Stop(out,args.max_minutes*60 if args.max_minutes else 8*3600)
    state=dict(stage='warmup',round=0,index=0,phase_update=0,best=[-1.,-1.,-1e30],created=joint.now())
    full_hash=None
    if (out/'last.pt').exists():
        if not args.resume:raise RuntimeError('Existing checkpoint requires --resume')
        state=joint.restore(out/'last.pt',learner,replay,protocol); full_hash=digest(out/'last.pt')
        progress=out/'episode_progress.pt'
        if progress.exists():
            saved=torch.load(progress,map_location='cpu')
            if saved['base_checkpoint_sha256']==full_hash:
                if saved['protocol_hash']!=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest():raise RuntimeError('Progress protocol mismatch')
                state=saved['state']; replay.rows=saved['replay_rows']; replay.rng.bit_generator.state=saved['replay_rng']; restore_rng(saved['rng'])
                if any(not (replay.root/r['file']).exists() for r in replay.rows):raise RuntimeError('Missing progress shard')
                json_write(replay.path,replay.rows)
        emit(event='derivative_resumed',stage=state['stage'],round=state['round'],counts=learner.counts)
    elif args.resume:raise RuntimeError('No checkpoint to resume')
    else:
        import_training_data(out,cfg,replay,source); audit(out,cfg,learner,replay)
        joint.save(out/'initial.pt',learner,state,replay,protocol)
    manifest=json.loads((out/'scenes.json').read_text(encoding='utf-8')); training=[s for s in manifest['scenes'] if s['split']=='train']
    if cfg.get('engineering_smoke'):training=training[:1]
    rng=np.random.default_rng(cfg['seed']+90017)
    if 'collection_rng' in state:rng.bit_generator.state=state['collection_rng']
    start_time=time.monotonic()

    def checkpoint(full=True):
        nonlocal full_hash
        state['collection_rng']=copy.deepcopy(rng.bit_generator.state); state['updated']=joint.now()
        if full or full_hash is None:
            joint.save(out/'last.pt',learner,state,replay,protocol); full_hash=digest(out/'last.pt')
        else:
            payload=dict(base_checkpoint_sha256=full_hash,state=copy.deepcopy(state),rng=rng_state(),
                replay_rows=copy.deepcopy(replay.rows),replay_rng=replay.rng.bit_generator.state,
                protocol_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest())
            tmp=out/'episode_progress.tmp'; torch.save(payload,tmp); tmp.replace(out/'episode_progress.pt')

    def status():
        json_write(out/'status.json',dict(**state,counts=state.get('training_counts',learner.counts),
            active_model_counts=learner.counts,replay_transitions=replay.transitions,timestamp=joint.now(),resources=resource(out,cfg)))

    try:
        if full_hash is None:checkpoint(); status()
        for phase,budget in (('warmup',p['warmup_updates']),('critic_prefit',d['critic_prefit_updates'])):
            if state['stage']!=phase:continue
            for i in range(state['phase_update'],budget):
                if stop.stopped():checkpoint(); status(); return
                b=replay.sample(p['batch_size'])
                logs=learner.representation(b,warmup=True) if phase=='warmup' else learner.critic_prefit(b)
                state['phase_update']=i+1
                if (i+1)%25==0:
                    row=dict(phase=phase,round=-1,update=i+1,timestamp=joint.now(),counts=learner.counts.copy(),**logs)
                    json_append(out/'training.jsonl',row); emit(event='derivative_update',**row); status()
                if (i+1)%p['checkpoint_every']==0:checkpoint()
            state.update(stage='critic_prefit' if phase=='warmup' else 'collect',phase_update=0); checkpoint(); status()
        depth=FastDepth(cfg)
        while state['round']<p['rounds'] and state['stage'] not in ('trained','complete'):
            if state['stage']=='collect':
                todo=[(s,k) for s in training for k in range(cfg['collection']['episodes_per_scene_per_round'])]
                for i in range(state['index'],len(todo)):
                    if stop.stopped():checkpoint(full=False); status(); return
                    s,k=todo[i]; inst=joint.new_instance(cfg,s)
                    try:
                        gi=(state['round']+k)%len(s['refs']); frame=legacy.module(cfg,'room_env').ReferenceFrame(np.array(s['refs'][gi]['goal_matrix']))
                        difficulty=('small','medium','large')[(state['round']+i)%3]
                        offset=sample_start(cfg,s,frame,rng,difficulty)
                        arrays,row=joint.episode(cfg,out,s,gi,offset,learner,depth,rng,steps=cfg['collection']['steps'],collect=True,inst=inst)
                        replay.add(arrays,row,f'derivative_round{state["round"]:02d}_{i:04d}')
                        json_append(out/'collection.jsonl',dict(phase='student',round=state['round'],index=i,difficulty=difficulty,**row))
                    finally:inst.close()
                    state['index']=i+1; checkpoint(full=False)
                    if (i+1)%6==0:status(); emit(event='derivative_collection',round=state['round'],index=i+1,total=len(todo))
                state.update(stage='representation',index=0,phase_update=0); checkpoint()
            for phase,budget in (('representation',p['representation_updates']),('metric',p['metric_updates'])):
                if state['stage']!=phase:continue
                for i in range(state['phase_update'],budget):
                    if stop.stopped():checkpoint(); status(); return
                    b=replay.sample(p['batch_size']); logs=learner.representation(b) if phase=='representation' else learner.metric_update(b)
                    state['phase_update']=i+1
                    if (i+1)%25==0:
                        row=dict(phase=phase,round=state['round'],update=i+1,timestamp=joint.now(),counts=learner.counts.copy(),**logs)
                        json_append(out/'training.jsonl',row); emit(event='derivative_update',**row); status()
                    if (i+1)%p['checkpoint_every']==0:checkpoint()
                state.update(stage='metric' if phase=='representation' else 'validation',phase_update=0); checkpoint()
            if state['stage']=='validation':
                number=state['round']+1
                if number%d['validation_every_rounds']==0 or number==p['rounds']:
                    val=evaluate(cfg,out,learner,depth,'val',f'round_{number:02d}',['learned'],['P0'],stop,cap=p['evaluation_validation_cap'])
                    if not val['complete']:checkpoint(); status(); return
                    v=val['summaries']['learned/P0']; score=[v['macro_scene_success'],v['worst_scene_success'],-v['mean_integrated_error']]
                    improved=score>state['best']
                    if improved:state['best']=score; state['best_round']=number; joint.save(out/'best.pt',learner,state,replay,protocol)
                    json_append(out/'validation.jsonl',dict(round=number,improved=improved,score=score,**v))
                    emit(event='derivative_validation',round=number,improved=improved,**v)
                    resource(out,cfg,True); report(out)
                state.update(stage='collect',round=number,index=0,phase_update=0); checkpoint(); status()
        if state['stage']!='complete':
            if state['stage']!='trained':
                state.update(stage='trained',training_counts=learner.counts.copy()); checkpoint(); status()
                json_write(out/'training_complete.json',dict(timestamp=joint.now(),counts=learner.counts,rounds=state['round'],
                    best_round=state.get('best_round'),runtime_seconds=time.monotonic()-start_time,resources=resource(out,cfg,True)))
            joint.restore(out/'best.pt',learner,replay,protocol,random_state=False)
            cap=3 if cfg.get('engineering_smoke') else None
            val=evaluate(cfg,out,learner,depth,'val','full_val',p['evaluation_methods'],['P0'],stop,cap=cap)
            if not val['complete']:status(); return
            test=evaluate(cfg,out,learner,depth,'test','fixed_test',p['evaluation_methods'],p['evaluation_plants'],stop,cap=cap)
            if not test['complete']:status(); return
            for scale in p['stress_depth_scales']:
                result=evaluate(cfg,out,learner,depth,'test',f'depth_scale_{scale}',['learned','identity_m'],['P0'],stop,
                    cap=p['stress_test_cap'],stress=True)
                if not result['complete']:status(); return
            final_analysis(out,cfg)
            state['stage']='complete'; status()
            json_write(out/'complete.json',dict(timestamp=joint.now(),training_counts=state['training_counts'],
                validation=val,test=test,protocol_sha256=digest(out/'protocol.json'),best_sha256=digest(out/'best.pt'),resources=resource(out,cfg,True)))
            emit(event='derivative_complete',output=str(out))
    finally:report(out)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--config',default=str(ROOT/'configs/derivative_critic.yaml'))
    p.add_argument('--resume',action='store_true'); p.add_argument('--smoke',action='store_true'); p.add_argument('--max-minutes',type=float)
    args=p.parse_args(); cfg=load_config(args.config)
    cfg['evaluation']['test_starts_per_goal_difficulty']=cfg['goal_fixed']['evaluation_test_repeats']
    if args.smoke:
        cfg['engineering_smoke']=True
        cfg['goal_fixed'].update(rounds=1,warmup_updates=2,representation_updates=2,metric_updates=2,batch_size=2,
            evaluation_validation_cap=3,evaluation_plants=['P0'],stress_depth_scales=[])
        cfg['derivative']['critic_prefit_updates']=2
        cfg['collection'].update(steps=8,episodes_per_scene_per_round=1); cfg['evaluation']['max_steps']=12
    out=Path(args.output).resolve()
    with writer_lock(out):
        try:run(out,cfg,args)
        except Exception as exc:
            out.mkdir(parents=True,exist_ok=True)
            json_write(out/'error.json',dict(timestamp=joint.now(),type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            emit(event='fatal',message=str(exc)); raise


if __name__=='__main__':main()
