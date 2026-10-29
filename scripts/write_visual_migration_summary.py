"""Summarize completed offline checks without implying a completed retrain."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/visual_original_offline_migration_20261022_v1'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def main():
    c, t, a = (read(name) for name in ('CONTROL_AUDIT.json', 'TRAINING_AUDIT.json', 'ASSET_AUDIT.json'))
    assert c['passed'] and t['passed'] and a['passed']
    compact, complete = c['compact_telemetry'], c['additional_original_telemetry']
    def maximum(group, key, plant=None):
        return max(row[key] for row in group['records'] if plant is None or row['plant'] == plant)
    values = dict(
        compact_action=maximum(compact, 'normalized_action_max_abs'),
        compact_v=maximum(compact, 'V_max_abs'),
        all_action=maximum(complete, 'normalized_action_max_abs'),
        all_v=maximum(complete, 'V_max_abs'),
        p0_step=maximum(complete, 'physics_local_max_abs', 'P0'),
        p1_step=maximum(complete, 'physics_local_max_abs', 'P1'),
    )
    summary = f'''# 多旋翼视觉伺服：迁移验收结果

**原模型、控制器和离线训练链已迁入新项目并通过核验。完整在线重训与闭环复现尚未完成，原因是原房间生成源码缺失一段。**

## 已通过的数值检查

| 检查 | 覆盖 | 结果 |
|---|---|---|
| 文件内容完整性 | {a['files']} 个源码、权重、数据和元数据文件 | 全部 SHA256 一致 |
| 目标图像深度 | {c['goal_depth']['references']} 个目标 RGB | FastDepth 最大差异 {c['goal_depth']['max_abs']:g} |
| 固定参考与目标 M | {c['rebuilt_references']['unique_goals']} 个测试目标，4 类场景 | M、J、H 与历史逐值相同 |
| 独立数据包的动作/V | {compact['episodes']} 回合，{compact['steps']} 步 | 动作最大差异 {values['compact_action']:.3e}；V 为 {values['compact_v']:.3e} |
| 原完整遥测的补充回归 | {complete['episodes']} 回合，{complete['steps']} 步 | 动作最大差异 {values['all_action']:.3e}；V 为 {values['all_v']:.3e} |
| 原 P0/P1 物理步进 | 同上，按已保存命令重算 | 单步最大差异 {values['p0_step']:.3e} / {values['p1_step']:.3e} |
| M/e → 控制量梯度 | {c['gradients']['directions']} 个方向，含饱和段 | 有限差分最大相对误差 {c['gradients']['max_relative_error']:.3e} |
| 当前 RGB 编码接口 | 保留的 bootstrap RGB | e 差异为 0 |
| 离线更新与恢复 | {len(t['checks'])} 项检查 | 全部通过 |

动作差异以归一化命令计；重建目标参考在原 GPU 推理下逐值一致。双精度重算保存的 float32 误差特征/命令出现上述微小差异，未超过预先固定的验收阈值。完整 1440 回合检查读取原目录的历史遥测；独立包中的 24 回合无需原目录。

## 实际执行的训练检查

从原正式 initial.pt 完整导入 AE、shadow AE、M、导数 Critic、优化器、计数与随机状态。以原 batch=4 分别执行一次 AE warmup、Critic 预拟合、表示学习、M 更新，均产生实际参数变化；变化范围符合各阶段隔离要求。

Critic 监督确实到达 Phi，25%/5% 辅助梯度上限通过。M 更新没有改变 AE 或 Critic。四个优化器都有非空状态后保存检查点，再恢复并执行下一次表示更新，采样、日志、全部模型参数、优化器、计数及 CPU 全局/replay 随机状态逐值精确一致。CUDA RNG 保存在原初始化中，本次 CPU 恢复验收不声称验证了 GPU 更新的精确重现。

这证明可迁移的数值与优化链已经跑通，不等于重新完成原 24 轮训练或检验多步长期训练走势。

## 保留的关键设计

- 共用 16 维 RGB 编码器；AE 与物理下降率 Critic 联合塑造 Phi。
- M 只由目标特征生成，Cholesky 正定界与 trace 约束不变；回合内 M 和网络冻结。
- FastDepth 只估计目标深度，固定参考 JVP；原阻尼控制器在归一化命令坐标中计算，最后转物理速度。
- M/e → 控制量保留可微链路；真值位姿只用于训练标签和评价。
- 原源码保持字节不变；路径和完整初始状态导入采用独立迁移协议，不修改旧协议后续写。

## 尚未完成

原 `scene_builders.py` 第 131–285 行未找到，涉及房间外壳与四族场景构造。其余 5 个缺失模块均已按原 SHA 恢复。最早 AE 文件仍缺失，但完整正式 initial.pt 可显式替代“构造后即被覆盖”的那次加载，不影响本次初始训练状态导入。

因此原环境渲染、在线采样、完整重训和新的闭环成功率仍待该源码恢复与渲染回归。测试遥测没有逐帧当前 RGB，不能靠有损视频验证像素等价。原实验的成功率和本次遥测重算不作为新闭环测试结果；V 仍是候选证书，未取得全局稳定性证明。

## 文件

- [迁移接口与使用说明](../../docs/VISUAL_ORIGINAL_MIGRATION.md)
- [控制验收](CONTROL_AUDIT.json)
- [训练验收](TRAINING_AUDIT.json)
- [资产完整性](ASSET_AUDIT.json)
- [迁移协议](protocol.json)
'''
    (OUT / 'MIGRATION_SUMMARY.md').write_text(summary, encoding='utf-8')
    names = ['CONTROL_AUDIT.json', 'TRAINING_AUDIT.json', 'ASSET_AUDIT.json', 'protocol.json']
    final = dict(
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        stage='offline_verified_closed_loop_blocked', offline_audit_passed=True,
        online_training_started=False, new_closed_loop_episodes=0,
        missing_dependency='scene_builders.py lines 131-285',
        source_files_changed=False, summary_values=values,
        evidence_sha256={name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in names},
        audit_script_sha256={name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest()
                            for name in ('audit_visual_control_transfer.py', 'audit_visual_training_transfer.py',
                                         'audit_visual_bundle_integrity.py', 'write_visual_migration_summary.py')},
    )
    (OUT / 'status.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'summary': str(OUT / 'MIGRATION_SUMMARY.md'), 'stage': final['stage']}))


if __name__ == '__main__':
    main()
