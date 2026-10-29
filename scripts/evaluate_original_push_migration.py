"""Frozen test protocol for migrated Full-M: no test-based model selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import html
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'stabilization-gym'), str(ROOT / 'stabilization-learning-control')]
import numpy as np
import torch
from torch import nn
from stabilization_gym.envs.tasks.original_protocols import make_original_task, digest
from stabilization_learning.algos.pytorch.full_m_ppo.trainer import write_json, check_resources, writer_lock


class IdentityCertificate(nn.Module):
    def encoder(self, C, valid, ell):
        return C.new_zeros((len(C), 1))

    def conditioned(self, D, condition):
        eye = torch.eye(3, device=D.device, dtype=D.dtype).expand(*D.shape[:-1], 3, 3)
        return eye, eye

    def value(self, D, condition):
        return D.square().sum(-1)


def episode_statistics(task, actor, bank, report, trace_path):
    with np.load(trace_path, allow_pickle=False) as values:
        data = {k:values[k] for k in values.files}
    # Recover full-precision reference/start poses from the same original reset.
    # The stored Torch context was float32 and must not introduce artificial
    # certificate increases or physical-cost differences near the target.
    ctrl=task.new_environment(bank,shape_names=report['shapes'],num_envs=len(report['shapes']))
    ctrl.reset(seed=report['seed'])
    if 'T' in report['shapes']:
        i=report['shapes'].index('T');q0=ctrl.env.q.copy();target0=ctrl.env.target.copy()
        target0[i]=[.25,.25,0.];q0[i]=[.251959141309,.249951905780,np.deg2rad(10.)]
        ctrl.reset(seed=report['seed'],options={'pose':q0,'target':target0})
    initial_pose=ctrl.env.q.copy()
    exact_target=ctrl.env.target.copy();ctrl.env.close()
    q=data['pose_next'];target=np.broadcast_to(exact_target,data['target'].shape)
    delta = q[..., :2] - target[..., :2]
    ang = target[..., 2]
    e_next = np.stack([(np.cos(ang)*delta[..., 0] + np.sin(ang)*delta[..., 1])/.1,
                       (-np.sin(ang)*delta[..., 0] + np.cos(ang)*delta[..., 1])/.1,
                       q[..., 2] - target[..., 2]], -1)
    with torch.no_grad():
        ids = torch.as_tensor(data['shape'][0], device=bank.device)
        condition = actor.encoder(bank.C, bank.valid, task.cfg.length_scale)[ids]
        # Network batch dimension is lanes, candidate dimension is time.
        D = task.engine.coordinates(torch.as_tensor(e_next.transpose(1, 0, 2), dtype=bank.dtype, device=bank.device))
        V_next = actor.value(D, condition).cpu().numpy().T
    active = data['active'].astype(bool)
    physical = (e_next[..., :2]**2).sum(-1) + (1 - np.cos(e_next[..., 2]))
    cumulative = (np.minimum(physical, task.cfg.error_cap) * active * task.cfg.dt).sum(0)
    if not np.isclose(cumulative.mean(),report['mean_cumulative_error'],rtol=1e-9,atol=1e-10):
        raise RuntimeError('Independent physical cost recomputation disagrees with original report')
    rise = (V_next - data['V']) > 1e-7
    decoded = []
    for i, ep in enumerate(report['episodes']):
        mask = active[:, i]
        a = data['action'][mask, i]
        decoded.append(dict(**ep, cumulative_error=float(cumulative[i]),
                            physical_steps=int(mask.sum()),
                            actual_V_rise_steps=int((rise[:, i] & mask).sum()),
                            actual_V_positive_change=float(np.maximum(V_next[:,i]-data['V'][:,i], 0)[mask].sum()),
                            arrival_capped_s=ep['steps']*task.cfg.dt if ep['success'] else task.cfg.max_steps*task.cfg.dt,
                            speed_total_variation=float(np.abs(np.diff(a[:,1])).sum()) if len(a)>1 else 0.,
                            initial_pose=initial_pose[i].tolist(), target_pose=target[0,i].tolist()))
    return decoded


def summarize(rows):
    total_steps = sum(x['physical_steps'] for x in rows)
    return dict(episodes=len(rows), successes=sum(x['success'] for x in rows),
                success_rate=float(np.mean([x['success'] for x in rows])),
                mean_cumulative_error=float(np.mean([x['cumulative_error'] for x in rows])),
                mean_position_error_m=float(np.mean([x['position_error_m'] for x in rows])),
                mean_abs_yaw_error_deg=float(np.mean([abs(x['yaw_error_deg']) for x in rows])),
                mean_switches=float(np.mean([x['switches'] for x in rows])),
                mean_capped_arrival_s=float(np.mean([x['arrival_capped_s'] for x in rows])),
                actual_V_rise_steps=sum(x['actual_V_rise_steps'] for x in rows),
                physical_steps=total_steps,
                pooled_V_rise_fraction=sum(x['actual_V_rise_steps'] for x in rows)/max(1,total_steps),
                mean_recovery_steps=float(np.mean([x['recovery_steps'] for x in rows])),
                mean_no_action_steps=float(np.mean([x['no_action_steps'] for x in rows])),
                mean_speed_total_variation=float(np.mean([x['speed_total_variation'] for x in rows])))


def paired(a, b):
    assert len(a)==len(b)
    for x,y in zip(a,b):
        assert (x['seed'],x['shape'],x['initial_pose'],x['target_pose']) == (y['seed'],y['shape'],y['initial_pose'],y['target_pose'])
    common = [(x,y) for x,y in zip(a,b) if x['success'] and y['success']]
    delta = np.array([x['cumulative_error']-y['cumulative_error'] for x,y in zip(a,b)])
    return dict(pairs=len(a), new_only_success=sum(x['success'] and not y['success'] for x,y in zip(a,b)),
                baseline_only_success=sum(y['success'] and not x['success'] for x,y in zip(a,b)),
                common_successes=len(common),
                mean_cost_difference=float(delta.mean()),
                common_success_mean_arrival_difference_s=float(np.mean([x['arrival_capped_s']-y['arrival_capped_s'] for x,y in common])) if common else None,
                all_starts_mean_capped_arrival_difference_s=float(np.mean([x['arrival_capped_s']-y['arrival_capped_s'] for x,y in zip(a,b)])))


def make_report(out, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = {}
    for line in (out/'history.jsonl').read_text().splitlines():
        row=json.loads(line);records[row['update']]=row
    rows=[records[k] for k in sorted(records)]
    steps=np.cumsum([x['collection']['transitions'] for x in rows])
    valid=[(steps[i],x) for i,x in enumerate(rows) if 'hard_eval' in x]
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].plot([x[0] for x in valid],[x[1]['hard_eval']['success_rate'] for x in valid]);ax[0].set_ylabel('Development success rate')
    ax[1].plot([x[0] for x in valid],[x[1]['hard_eval']['mean_cumulative_error'] for x in valid]);ax[1].set_ylabel('Development integrated error')
    for a in ax:a.set_xlabel('Collected physical transitions');a.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'analysis/learning.png',dpi=150);plt.close(fig)
    labels={'migrated_best':'迁移后重新训练M','historical_best':'原实验M','identity':'单位M','initial':'原始未训练M','migrated_final':'重新训练最终M'}
    lines=['# Push Any Shape 严格协议迁移与训练测试','',
           '原始核心源码逐字节保留；控制器、奖励、辅助损失、100步时限、32形状与1000轮预算保持一致。',
           '这里的“迁移”是代码/协议迁移。测试形状均来自训练和标定库，不是未见形状零样本测试。','',
           f"训练完成 {results['training']['updates']} 轮，共 {results['training']['transitions']} 条有效物理转移。验证最佳为第 {results['training']['best_update']} 轮。",'',
           '| 方法 | 新起点成功 | 平均累计误差 | 失败封顶平均到达秒 | 实际V回升步比例 |',
           '|---|---:|---:|---:|---:|']
    for method,s in results['new_start_summary'].items():
        lines.append(f"| {labels[method]} | {s['successes']}/{s['episodes']} | {s['mean_cumulative_error']:.6f} | {s['mean_capped_arrival_s']:.3f} | {s['pooled_V_rise_fraction']:.2%} |")
    lines.extend(['','新起点统计排除原程序始终固定的T形回归案例：31种形状 × 10个新种子 = 310回合/方法。T形开发回归另行保留。',
                  '实际V回升从执行后的位姿重新计算，使用后继状态对应的M；分母为有效物理步数，包含目标保持步。',
                  '单种子、固定标定库、降阶数值模型。该实验不包含SO101关节接触仿真，不证明全局Lyapunov稳定性。',
                  '', '视觉伺服严格迁移仍受原室内渲染器与编码器源码缺失阻塞，没有用新场景替代原结果。','',
                  '![训练曲线](learning.png)','', '[完整统计与配对结果](RESULTS.json)',''])
    (out/'analysis/REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    table=''.join('<tr><td>'+labels[k]+'</td><td>'+f"{v['successes']}/{v['episodes']}"+'</td><td>'+f"{v['mean_cumulative_error']:.6f}"+'</td><td>'+f"{v['mean_capped_arrival_s']:.3f}"+'</td><td>'+f"{v['pooled_V_rise_fraction']:.2%}"+'</td></tr>' for k,v in results['new_start_summary'].items())
    page='<!doctype html><meta charset="utf-8"><title>Push迁移训练测试</title><style>body{font:17px/1.7 system-ui;margin:3em auto;max-width:1100px;color:#203040}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd}img,video{max-width:100%}</style><h1>Push Any Shape：原协议迁移</h1>'
    page+='<p>1000轮原配置重新训练；新起点采用310个配对案例。训练形状上的新起点，不是未见形状。</p><table><tr><th>方法</th><th>成功</th><th>累计误差</th><th>失败封顶到达秒</th><th>实际V回升比例</th></tr>'+table+'</table><img src="learning.png"><p>原权重重放、逐步动作与优化器更新等价性已单独审计。实际回升如实保留，仅为候选证书。SO101接触执行层未纳入本实验。</p><video controls src="../final_best/hard_numeric.mp4"></video><p>视觉伺服原始依赖缺失，严格闭环迁移尚未完成。</p>'
    (out/'analysis/RESULTS.html').write_text(page,encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    out=Path(args.run).resolve();protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'));config=protocol['config']
    if not (out/'training_complete.json').is_file():raise RuntimeError('Training must complete before fixed testing')
    if str(torch.__version__)!=config['torch_version']:raise RuntimeError('Runtime mismatch')
    task=make_original_task('push_any_shape');torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if config['device']=='cuda':torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    checkpoints={'migrated_best':out/'best.pt','historical_best':task.reference/'best.pt',
                 'identity':None,'initial':task.reference/'initial.pt','migrated_final':out/'last.pt'}
    initial_hashes={k:digest(p) for k,p in checkpoints.items() if p}
    (out/'analysis').mkdir(exist_ok=True)
    test_protocol=dict(training_protocol_sha256=digest(out/'protocol.json'),checkpoints=initial_hashes,
                       seeds=config['test_seeds'],evaluator_sha256=digest(__file__),T_case='pinned; excluded from new-start summary')
    marker=out/'analysis/test_protocol.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))!=test_protocol:raise RuntimeError('Test protocol changed')
    write_json(marker,test_protocol)
    evidence={};summaries={};dev={}
    with writer_lock(task.root/'runs/.original_push_gpu.lock'),writer_lock(out/'.writer.lock'):
        bank=task.bank(config['device'])
        for method,path in checkpoints.items():
            actor=IdentityCertificate() if path is None else task.load_actor(path,config['device'])[0]
            folder=out/('final_best' if method=='migrated_best' else f'comparison/development/{method}')
            for seed,destination,video in [(20260913,folder,method=='migrated_best')]+[(seed,out/f'comparison/{method}/seed_{seed}',False) for seed in config['test_seeds']]:
                if (out/'STOP').exists():raise RuntimeError('User STOP: evaluation paused')
                reason=check_resources(out,config,scan_size=True)
                if reason:raise RuntimeError('Resource safety stop: '+reason)
                if (destination/'report.json').is_file() and (destination/'trajectory.npz').is_file() and (not video or (destination/'hard_numeric.mp4').is_file()):
                    report=json.loads((destination/'report.json').read_text())
                else:report=task.evaluate(actor,bank,seed=seed,output=destination,video=video)
                rows=episode_statistics(task,actor,bank,report,destination/'trajectory.npz')
                for row in rows:row['seed']=seed
                if seed==20260913:dev[method]=summarize(rows)
                else:evidence.setdefault(method,[]).extend([row for row in rows if row['shape']!='T'])
                write_json(out/'status.json',dict(status='EVALUATING',method=method,seed=seed))
                print(json.dumps(dict(method=method,seed=seed,success_rate=report['success_rate'])),flush=True)
            summaries[method]=summarize(evidence[method])
        for k,p in checkpoints.items():
            if p and digest(p)!=initial_hashes[k]:raise RuntimeError('Evaluation mutated checkpoint')
        history={}
        for s in (out/'history.jsonl').read_text().splitlines():
            r=json.loads(s);history[r['update']]=r
        best=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        result=dict(training=dict(updates=len(history),transitions=sum(r['collection']['transitions'] for r in history.values()),
                                  best_update=best['next_update']),development_summary=dev,new_start_summary=summaries,
                    paired={k:paired(evidence['migrated_best'],v) for k,v in evidence.items() if k!='migrated_best'},
                    episodes=evidence,checkpoints_unchanged=True,visual_strict_migration='blocked_missing_original_sources')
        write_json(out/'analysis/RESULTS.json',result);make_report(out,result)
        import cv2
        video=out/'final_best/hard_numeric.mp4';cap=cv2.VideoCapture(str(video));frames=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if frame is None or not frame.size:raise RuntimeError('Invalid video frame')
            frames+=1
        cap.release()
        if not frames:raise RuntimeError('Final video cannot be decoded')
        write_json(out/'complete.json',dict(training_updates=len(history),test_episodes_per_method=len(evidence['migrated_best']),
                                           checkpoint_hashes=initial_hashes,video_frames_decoded=frames,
                                           results_sha256=digest(out/'analysis/RESULTS.json')))
        write_json(out/'status.json',dict(status='COMPLETE',training_updates=len(history)))
    print(json.dumps(dict(training=result['training'],new_start_summary=summaries,paired=result['paired']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
