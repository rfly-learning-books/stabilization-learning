"""Read-only legacy RGB provenance and stored-trajectory audit (no rollout)."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

from .common import write_json

PROJECT = Path('D:/stabilization-learning')
LEGACY_REPO = Path('D:/书稿论文/lac_ae_certificate_vs')
LEGACY = LEGACY_REPO / 'runs/ae_derivative_critic_seed41_20261019_v1'
SEED71 = PROJECT / 'runs/scratch_apps_so101_full_v1_20261026_184356/rgb_seed71'
V4 = PROJECT / 'runs/rgb_gpu_coverage_v4_seed81_20261027_141542'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def case_key(c):
    return c['scene_id'], c['goal_id'], tuple(c['offset'])


def first_hold(offset):
    # The frozen legacy implementation starts counting after the first action.
    good = (np.linalg.norm(offset[1:, :3], axis=1) < .02) & (np.abs(offset[1:, 3]) < math.radians(1))
    hold = 0
    for step, ok in enumerate(good, 1):
        hold = hold + 1 if ok else 0
        if hold == 11:
            return step
    return None


def main(root):
    root = Path(root)
    if (root/'legacy_audit.json').exists() or (root/'legacy_audit.txt').exists():
        raise FileExistsError('Refuse to replace an existing audit')
    started = time.time()
    proto = read(LEGACY/'protocol.json')
    cfg = proto['config']
    inputs = [LEGACY/'protocol.json', LEGACY/'resolved_config.yaml', LEGACY/'training_complete.json',
              LEGACY/'import_provenance.json', LEGACY/'evaluation_starts.json', LEGACY/'collection.jsonl',
              LEGACY/'evaluation/fixed_test/summary.json', LEGACY/'evaluation/full_val/summary.json',
              LEGACY_REPO/'TRAINING_PLAN.md', LEGACY_REPO/'README.md',
              LEGACY_REPO/'runs/goal_fixed_fastdepth_night_seed41_20261019_v2/repair_provenance.json',
              SEED71/'protocol.json', SEED71/'FROZEN.json', SEED71/'RESULTS.json',
              V4/'protocol.json', V4/'FROZEN.json', V4/'FINAL_STATE.json',
              PROJECT/'migration/visual_scene_reconstruction/README.md',
              PROJECT/'migration/visual_scene_reconstruction/SCENE_AUDIT.json',
              root/'control/RESULTS.json', root/'derivative_geometry_v2/ANALYSIS.json',
              root/'derivative_quantization/SUMMARY.json', Path(__file__)]
    source_checks = []
    for rel, expected in proto['sources'].items():
        p = LEGACY_REPO/rel.replace('\\', '/')
        actual = digest(p) if p.is_file() else None
        source_checks.append(dict(path=str(p), expected=expected, actual=actual, matched=actual==expected))
    if not all(c['matched'] for c in source_checks):
        raise RuntimeError('Legacy source no longer matches frozen protocol')
    starts = read(LEGACY/'evaluation_starts.json')
    v4proto = read(V4/'protocol.json')
    manifest = v4proto['config']['evaluation_manifest']
    overlap = {}
    for split in ('val', 'test'):
        distances = []
        for c in manifest[split]:
            others = [d for d in starts[split] if (c['scene_id'], c['goal_id']) == (d['scene_id'], d['goal_id'])]
            distances.append(min(float(np.max(np.abs(np.asarray(c['offset'])-d['offset']))) for d in others))
        overlap[split] = dict(v4_cases=len(manifest[split]), legacy_cases=len(starts[split]),
                             exact_matches=len(set(map(case_key, manifest[split])) & set(map(case_key, starts[split]))),
                             matches_tolerance_1e_6=sum(d < 1e-6 for d in distances),
                             minimum_coordinate_max_difference=min(distances),
                             median_nearest_coordinate_max_difference=float(np.median(distances)))
    tests = starts['test']
    sample_unit = dict(training_seed=41, distinct_test_room_instances=len({c['scene_id'] for c in tests}),
                       strata=len({(c['scene_id'], c['goal_id'], c['difficulty']) for c in tests}),
                       cases=len(tests), unique_scene_goal_offsets=len(set(map(case_key, tests))),
                       repeat_index_counts=dict(Counter(str(c['repeat']) for c in tests)),
                       interpretation='72 room/goal/difficulty strata with five distinct offsets each; 360 starts, one trained seed. Starts share 12 rooms and are not 360 independent training runs.')
    test_summary = read(LEGACY/'evaluation/fixed_test/summary.json')
    trajectory_checks = []
    horizons = {}
    earliest_test = None
    for method in ('learned', 'identity_m', 'shadow_ae_identity', 'initial_ae_identity'):
        for plant in ('P0', 'P1'):
            files = sorted((LEGACY/'evaluation/fixed_test').glob(f'{method}_{plant}_*.json'))
            passed200 = passed300 = 0
            consistency_errors = []
            for f in files:
                r = read(f)
                stamp = r.get('timestamp')
                earliest_test = stamp if earliest_test is None else min(earliest_test, stamp)
                npz_path = f.with_suffix('.npz')
                with np.load(npz_path, allow_pickle=False) as z:
                    off = np.asarray(z['offset'], np.float64)
                    hold = first_hold(off)
                    count = len(z['action'])
                row_success = bool(r['success'])
                success = hold is not None and r['status'] not in ('collision', 'outside')
                if row_success != success or (success and hold != r['steps']) or count != r['steps']:
                    consistency_errors.append(dict(file=str(f), reported_success=row_success,
                                                   first_hold_step=hold, reported_steps=r['steps'], actions=count))
                passed200 += int(success and hold <= 200)
                passed300 += int(success and hold <= 300)
                trajectory_checks.append(dict(file=str(f), sha256=digest(f), trace_sha256=digest(npz_path),
                                              reported_status=r['status'], first_hold_step=hold, steps=r['steps']))
            if len(files) != 360:
                raise RuntimeError(f'Incomplete legacy group {method}/{plant}')
            horizons[f'{method}/{plant}'] = dict(episodes=len(files), success_200=passed200,
                success_rate_200=passed200/len(files), success_300=passed300, success_rate_300=passed300/len(files),
                extra_successes_201_300=passed300-passed200, extra_percentage_points=100*(passed300-passed200)/len(files),
                hold_reconstruction_mismatches=consistency_errors,
                reported_successes=test_summary['summaries'][f'{method}/{plant}']['status_counts']['success'])
    collection = [json.loads(line) for line in (LEGACY/'collection.jsonl').read_text().splitlines() if line.strip()]
    training = read(LEGACY/'training_complete.json')
    source_import = read(LEGACY/'import_provenance.json')
    source_repair = read(LEGACY_REPO/'runs/goal_fixed_fastdepth_night_seed41_20261019_v2/repair_provenance.json')
    initial_path = Path(cfg['paths']['initial_autoencoder'])
    seed71proto = read(SEED71/'protocol.json')
    seed71results = read(SEED71/'RESULTS.json')['summaries']
    scene_audit = read(PROJECT/'migration/visual_scene_reconstruction/SCENE_AUDIT.json')
    control = read(root/'control/RESULTS.json')
    legacy_val = read(LEGACY/'evaluation/full_val/summary.json')['summaries']['learned/P0']
    report = dict(schema='rgb_v5_legacy_source_audit', created_utc=datetime.now(timezone.utc).isoformat(),
        mode='read-only files and stored trajectories; no old test rollout, no model modification',
        root=str(root), legacy_root=str(LEGACY), elapsed_seconds=time.time()-started,
        source_hash_checks=source_checks, input_hashes={str(p):digest(p) for p in inputs},
        legacy_sample_unit=sample_unit, legacy_v4_start_overlap=overlap,
        legacy_test_summary=test_summary, legacy_horizon_reconstruction=horizons,
        legacy_validation=dict(episodes=48,successes=legacy_val['status_counts']['success'],success_rate=legacy_val['success_rate']),
        legacy_training=dict(counts=training['counts'],rounds=training['rounds'],best_round=training['best_round'],
            online_collection_episodes=len(collection),new_online_transitions=sum(c['steps'] for c in collection),
            imported_bootstrap_transitions=source_import['transitions'],training_completed=training['timestamp'],
            first_fixed_test_timestamp=earliest_test,training_before_test=training['timestamp']<earliest_test),
        legacy_initialization=dict(declared_initial_ae_path=str(initial_path),path_currently_exists=initial_path.exists(),
            declared_initial_ae_sha256=proto['dependencies']['read_only']['initial_autoencoder'],
            source_import=source_import,prior_repair_summary={k:v for k,v in source_repair.items() if k!='reused_shards'},
            established='Existing grayscale AE converted to RGB, mixed-scene RGB reconstruction, then deterministic 32-to-16 weight SVD and smooth activations; imported joint initial snapshot has zero prior joint optimizer updates.',
            unresolved='Original mixed_ae_41/best.pt and fairness/training records absent at declared path and not located by bounded filename search in D:/书稿论文. Cumulative grayscale/RGB pretraining data, actual epochs/updates and split exposure cannot be certified. A design document proposing 50 epochs is not execution evidence.'),
        legacy_algorithm=dict(encoder_lr=cfg['goal_fixed']['encoder_lr'],metric_lr=cfg['goal_fixed']['metric_lr'],
            feature_error='(encode(current)-encode(goal))/4; same scale in J and V',
            interaction='Reference-image J fixed for an entire rollout; recomputed after encoder version changes',
            policy='G=J.T M J; H=G+0.01 diag(diag(G))+max(1e-8,1.01 trace(G)/(10000-1)) I; u=saturate(-0.7 H^-1 J.T M e)',
            representation='Normalized RGB+DCT reconstruction gradients dominate; signed physical derivative Critic gradient capped at 0.25 times reconstruction encoder norm; other regularizers capped at 0.05.',
            metric='Goal-only SPD M, trace16, eigenfloor.1; encoder/critic parameters frozen during M update; differentiate action solve.',
            objective_floor=cfg['goal_fixed']['objective_floor'],value_lower=cfg['goal_fixed']['value_lower'],
            shadow_ae=True,depth='Public FastDepth; reference depth; train augmentation',
            online_observation='RGB images plus known camera calibration and estimated depth; true pose only labels/metrics',
            training_schedule='Warmup and Critic prefit followed by 24 rounds of representation block then M block; this is structured alternating training, not a completely separate one-shot two-stage RGB run.'),
        later_seed71=dict(root=str(SEED71),protocol=seed71proto,freeze=read(SEED71/'FROZEN.json'),
                         summaries={k:{n:v.get(n) for n in ('episodes','success_rate','status_counts','mean_integrated_error')} for k,v in seed71results.items()}),
        v4=dict(root=str(V4),run_config=v4proto['config']['run'],final_state=read(V4/'FINAL_STATE.json')),
        scene_reconstruction=dict(original_rgb_equivalent=scene_audit['original_rgb_equivalent'],
            manifest_byte_identical=digest(LEGACY/'scenes.json')==digest(PROJECT/'migration/visual_scene_reconstruction/scenes.json'),
            structured_geometry_and_dynamics=scene_audit['checks'],train_render_comparisons=scene_audit['train_render_comparisons'],
            limitations=scene_audit['limitations'],visual_changes=scene_audit['visual_changes']),
        current_controlled_evidence=dict(control=control,derivative_geometry=read(root/'derivative_geometry_v2/ANALYSIS.json'),
                                         quantization=read(root/'derivative_quantization/SUMMARY.json')),
        conclusions=[
            'Historical RGB performance is real in stored records: 262/360 P0 and 261/360 P1, with 360 distinct starts. It is not the later seed71 result and not a four-channel optical-flow result.',
            'Extending 200 to 300 steps contributes only 13/360 P0 and 14/360 P1 successes; horizon difference alone cannot explain the v4 gap.',
            'Legacy pretrained initialization matters, but later seed71 random initialization still attains 47/72 and 46/72 in the reconstructed CPU-rendered environment; absence of pretraining alone is insufficient explanation.',
            'Representation objective/gradient routing, learning rate, loss floor, J work point, damping, scene appearance and rendering all changed; the historical-versus-v4 comparison is confounded and cannot isolate a single cause.',
            'The current frozen-v4 controlled tests do not support simply reverting J to the target work point: success gains are small and tail failures rise. FD J reduces average error but does not restore historical success and costs more input.',
            'Encoder JVP checks pass locally. True depth does not improve average J direction. Removing final 8-bit quantization changes direction agreement negligibly; fine motion inconsistency remains.',
            'Preserve env/algo separation. Prioritize physically informative near-goal feature training and closed-loop validation; use geometric/FD diagnostics to bound model limitations before adding residual or horizon complexity.'
        ],
        caveats=['Single training seed per version; distinct initial offsets are not independent training replications.',
                 'Legacy and v4 use different start manifests and non-identical pixels; values are historical context, not a matched causal test.',
                 'Original good success does not establish a valid physical stability certificate; V can rise and original mean comparable fraction is only about .687 under its own bounds.',
                 'Current v5 evidence is development/validation, not untouched held-out test. No hyperparameter was selected from re-reading old test trajectories.',
                 'Stored legacy hold rule is first 11 consecutive post-action observations, not guaranteed indefinite hold.',
                 'True-depth/FD variants are privileged diagnostics, not deployable RGB-only performance.'])
    write_json(root/'legacy_trajectory_audit.json', dict(schema='legacy_stored_hold_audit',trajectories=trajectory_checks))
    report['trajectory_audit_path']=str(root/'legacy_trajectory_audit.json')
    report['trajectory_audit_sha256']=digest(root/'legacy_trajectory_audit.json')
    write_json(root/'legacy_audit.json', report)
    lines = ['原始 RGB 与当前实现：来源及存量轨迹审查', '',
             '此次仅阅读已有模型协议、配置、结果和轨迹。没有重新运行历史测试，也没有更改旧模型、旧配置或测试。',
             f'原始运行：{LEGACY}', f'核验源码：{len(source_checks)} 项全部匹配原冻结 SHA256。', '',
             '1. 原始成功率的真实口径',
             'seed41，12测试房间×2目标×3难度×5个不同offset=360起点。五个repeat索引不是重复同一轨迹，也不是五个训练种子。',
             '成功要求位置<2cm、偏航<1°，执行动作后连续11观测满足。原时域300步、dt=.1秒。',
             '模型/工况 | 200步已完成保持 | 300步成功 | 多100步新增',]
    for k,v in horizons.items():
        lines.append(f"{k} | {v['success_200']}/360 ({100*v['success_rate_200']:.2f}%) | {v['success_300']}/360 ({100*v['success_rate_300']:.2f}%) | {v['extra_successes_201_300']} ({v['extra_percentage_points']:.2f}百分点)")
    mismatches=sum(len(v['hold_reconstruction_mismatches']) for v in horizons.values())
    lines += [f'从2880条保存轨迹独立重算保持：与原成功/步数记录不一致 {mismatches} 条。',
              '原验证37/48=77.08%。原test初始AE+identity已231/360=64.17%；所学M对同特征identity的成功增益为P0 +5.83、P1 +7.22百分点。',
              '200步旧learned仍69.17%/68.61%，所以时域只解释一小部分历史差距。', '',
              '2. 三个版本必须分开',
              '2026-10-19 原seed41：已有灰度AE转RGB再多场景重建，SVD32→16、平滑编码器；目标J；相对对角阻尼和自适应ridge；AE梯度主导、有符号物理导数Critic；24轮表示块/M块交替。',
              '2026-10-26 seed71复现：随机AE/M/Critic、空回放、124222新转移、14000模块更新；重建CPU场景；200步72起点；47/72、46/72。不是原seed41的同一模型。',
              '2026-10-27 v4：随机AE/M，GPU并行渲染、130387新转移、14000模块更新；共同控制目标直接更新表示；当前J、固定.02阻尼；200步72起点；P0/P1均8/72。',
              'v4与seed71共享评测清单；它们与原seed41的val/test起点，即使允许1e-6差异，交集仍均为0。', '',
              '3. 原始训练机制的关键细节',
              f"正式更新计数：{training['counts']}；新在线转移 {sum(c['steps'] for c in collection)}，另导入2822条bootstrap。",
              '导入前序goal-fixed的initial.pt，其joint优化计数全部0；不能说复用了前序已训练M或前序joint更新。但AE本身已有预训练。',
              '原AE lr=1e-5（v4为2e-4）；原M lr=1e-4（v4为3e-4）；导数辅助梯度最多AE梯度的.25，其他正则最多.05；v4没有这种路由限制。',
              '原标定/下降分母E+.0004、下界.25；v4分母E+.01、下界.1。成功阈值附近的损失信号不等价，不能只比较总更新数。',
              '原M仍为目标条件SPD、固定迹16；目标J与M在回合内固定。最终来源不是README最初描述的任意状态相关M控制器。',
              '原实现有同数据、同次数shadow AE；v4未实现shadow AE。', '',
              '4. 缺失的历史证据及环境变化',
              f'原AE声明路径目前不存在：{initial_path}',
              'TRAINING_PLAN可确认其灰度→RGB、多场景重建和SVD来源，但原更早AE实际累计更新/epoch、训练数据和fairness记录尚不可核定。设计文档的50epoch不能当成实际完成数。',
              '原scene_builders.py部分丢失；当前44房间清单逐字节相同，碰撞、相机与P0/P1动力学保留，视觉组装为重建。',
              '训练目标的历史/重建CPU图像MAE：classroom .763、office 2.856、meeting 11.146、library .274（0–255尺度）；v4还新增GPU像素差异。不能声称原环境完全等价。',
              '视觉伺服_代码与经验_20261012 是四通道/光流材料，不能把其中结果当成此RGB AE结果。', '',
              '5. 结合本次固定模型控制诊断',
              '48验证起点P0/P1成功：baseline10/9；target_J11/11；relative_damping9/11；identity_M6/5；FD_current_J11/12。',
              '单换目标J的均值累计误差从8.17/8.17升至28.06/26.51，并增加碰撞与越界；不足以恢复旧效果。',
              'FD J使均值累计误差降低约25.9%/29.3%，但输入代理增加约95.7%/65.2%，且成功仍低；J有贡献，却非唯一瓶颈。',
              'JVP微分数值检验通过；真深度未改善平均方向；去掉最终8bit量化基本不改善细微位移J误差。',
              '因此优先检验近目标表示/物理标定与控制梯度路由，不把全部差距归到深度、量化或单一J工作点。统一env/algo接口可以保留。', '',
              '限制：单种子；历史比较混杂；本次v5属于开发/验证；输入平方只是代理；有限差分与真深度只作诊断；成功率不能替代物理稳定性证明。', '',
              '完整可复算字段与源文件SHA256见同目录 legacy_audit.json；全部2880条保持复算与输入hash见 legacy_trajectory_audit.json。']
    with (root/'legacy_audit.txt').open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(audit=str(root/'legacy_audit.json'),trajectories=len(trajectory_checks),
                          mismatch_count=mismatches,legacy_learned={k:v for k,v in horizons.items() if k.startswith('learned/')},
                          seconds=time.time()-started),ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True,type=Path)
    main(parser.parse_args().root)
