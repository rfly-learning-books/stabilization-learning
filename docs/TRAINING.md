# 训练指南

## 入口与流程

通用 CLI `python -m stabilization_learning` 提供回合 M-PPO；`scripts/run_unified_training_v4.py` 提供多后端生命周期；观测任务使用 `scripts/run_observer_training_v5.py`；室内 RGB 书稿协议使用 `scripts/run_rgb_book_reproduction_v1.py`。

统一顺序为：初始化、初始验证与采样、表征预热、Critic 预训练、epoch 循环、全组冻结、独立测试。每个 epoch 包含采样、训练更新、验证和检查点保存。无对应网络的后端跳过预热阶段。

`configs/public/` 提供免训练资产的短流程配置。其他版本化配置保留研究参数；应在新配置中替换历史路径、运行预算及外部资源位置。各后端的 Critic 目标分别定义：PPO 价值估计、SAC Q 与视觉导数监督不共享同一数学目标。

## 基础 M-PPO

```bash
python -m stabilization_learning train --config configs/public/point_mass_smoke.json --output runs/point_mass
python -m stabilization_learning evaluate --run runs/point_mass --seeds 20001 20002
```

`--episodes` 可覆盖训练回合数。恢复训练使用 `--resume`，须保持环境、特征坐标和协议一致。测试比较同起点的学习 M 与单位 M。

## 统一 Runner

```bash
python scripts/run_unified_training_v4.py preflight --config configs/public/unified_cpu_smoke.json --output runs/preflight.json
python scripts/run_unified_training_v4.py train --config configs/public/unified_cpu_smoke.json --output runs/unified_cpu
python scripts/run_observer_training_v5.py train --config configs/public/observer_cpu_smoke.json --output runs/observer_cpu
```

每次使用新的输出目录。`maximum_minutes` 限制运行时间，`minimum_free_gib` 设置磁盘余量。更新预算位于 `jobs[].schedule`。CPU 示例使用状态模型梯度；机器人任务可选 `episode_ppo`；`push_discrete_sac` 需要显式的 `time_bank`。

## GPU 与 RGB

按 PyTorch 官方说明安装适合本机驱动的 CUDA 版本，再检查：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

RGB 后端使用 CUDA 进行网络计算，渲染与采样管线由后端决定。通用 `visual_servo` 是生成纹理和 DCT 特征任务；书稿 RGB 使用 AE、深度预测与目标相关 M。书稿入口需要 [外部资源](ASSETS.md) 中列出的初始化、bootstrap 和参考缓存。

## 运行时锁与复现

公开示例不绑定作者机器的解释器或 GPU UUID。严格复现时，可在自己的环境采集运行时锁：

```bash
python scripts/run_unified_training_v4.py capture-runtime --output runs/runtime_local.json
```

将新配置的 `runtime_lock` 指向该文件。CPU 训练锁应在相同的隐藏 CUDA 环境采集：PowerShell 使用 `$env:CUDA_VISIBLE_DEVICES='-1'`，Linux 使用 `CUDA_VISIBLE_DEVICES=-1 python ...`。锁比较依赖、系统和硬件身份。

```bash
python scripts/run_unified_training_v4.py reproduce --config configs/public/unified_cpu_smoke.json --output runs/reproduction
```

`reproduce` 在独立目录执行两次训练并比较记录。验证集用于选模，测试集仅在模型冻结后使用；改变训练目标或预算须建立新实验。

## 输出与检查

输出包含检查点、历史、协议、阶段状态、冻结标记和评价结果，具体文件名由入口定义。`python -m pytest` 默认执行免外部训练资产的测试。历史研究测试可按文件名显式运行，并提供其资源。`python scripts/check_publication.py` 检查 Git 暂存或跟踪文件中的数据、权重和凭据模式。
