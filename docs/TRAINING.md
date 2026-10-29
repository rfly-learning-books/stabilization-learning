# 训练指南

先完成 [安装](../README.md#安装)。下列命令均在仓库根目录执行，输出使用新的目录名。任务选择、章节对应、初始化与已有结果见 [训练任务与书稿对应关系](BOOK_TASKS.md)。

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

## 从头联合成本 Push 与 SO101

对应 [seed81 联合成本实验](BOOK_TASKS.md#从头-push-与联合成本-so101)。Full-M Actor、Critic 和优化器从头初始化，从第一次更新使用全部成本项。训练使用标定降阶响应模型，冻结后的 SO101 测试使用关节驱动 MuJoCo。

准备仿真依赖：

```bash
python -m pip install -r requirements-simulation.txt
```

若已有协议匹配的时间／响应库，后续直接将 `--time-bank` 指向它。需要重新标定时执行：

```bash
python scripts/run_joint_certificate_v2.py push-prepare --output runs/joint81_calibration --max-hours 2
```

完成后生成 `runs/joint81_calibration/time_bank.json`。标定阶段采集 SO101 接触响应和时间，不读取策略权重。配置默认使用八个形状；训练／验证／测试形状分别为四／二／二个，留出形状保留标定信息。

依次训练、冻结、测试：

```bash
python scripts/run_joint_certificate_v2.py push-train --config configs/joint_certificate_v2/push_overnight.json --costs configs/joint_certificate_v2/push_costs.json --time-bank runs/joint81_calibration/time_bank.json --output runs/joint81/push_seed81 --max-hours 4
python scripts/run_joint_certificate_v2.py push-freeze --runs runs/joint81/push_seed81 --seeds 81 --output runs/joint81/push_group.json --max-hours 0.1
python scripts/run_joint_certificate_v2.py push-test --run runs/joint81/push_seed81 --group-manifest runs/joint81/push_group.json --output runs/joint81/push_test_seed81 --repeats 16 --max-hours 1
python scripts/run_joint_certificate_v2.py push-arm --run runs/joint81/push_seed81 --group-manifest runs/joint81/push_group.json --output runs/joint81/arm_seed81 --repeats 4 --max-hours 2
```

上述时长为各命令的示例上限，不是预计耗时；结束条件同时受任务预算和磁盘余量约束。每条命令成功完成后再执行下一条。`push-train` 的模板为 seed81、1,000 次更新、CPU；可在新的本地配置中设定设备与预算。改变设备或标定库时记录新的运行协议。

`push-freeze` 固定验证所选模型，`push-test` 评价降阶模型的留出形状，`push-arm` 将同一 M 直接用于物理仿真。所选 M 没有再经过 300 次 SO101 微调。多种子实验在全部训练结束后，将全部路径及种子一次传给 `push-freeze`。

全部写入进程结束后可导出训练、M 与验证曲线：

```bash
python scripts/run_joint_certificate_v2.py analyze --root runs/joint81 --output artifacts/joint81_report
```

该报告包含 `ANALYSIS_ZH.md`、曲线与统计来源。单条命令的 `--max-hours` 仅限制该阶段；需要整组共用截止时，先用 `init-night` 创建协议，再对所有阶段传入同一 `--night-root`。正式共用协议的分析另需 `--process-snapshot`，其格式为 `joint_v2_process_snapshot`，见 [分析入口](../extensions/joint_certificate_v2/analysis.py)。

## 从头 Push、SO101 课程与 RGB 套件

此路线对应 seeds71—73 的跨形状 Push 和 seed71 的从头 RGB。顺序为 SO101 标定、Push 训练与测试、SO101 时间训练与测试、RGB 训练与测试。SO101 从本次相同种子的 Push 选择模型开始；时间目标在 SO101 阶段加入。

复制 [套件配置](../configs/scratch_applications_v1.json) 为 `configs/scratch_applications_local.json`。将 `python` 与 `rgb_python` 改为本机具有所需依赖的 Python 路径，设置 `device` 与 `max_hours`，准备 [套件资源](ASSETS.md#从头应用套件资源)。两个解释器可指向同一个已安装完整依赖的环境。

```bash
python scripts/run_scratch_applications_v1.py prepare --config configs/scratch_applications_local.json --output runs/scratch_apps
python scripts/run_scratch_applications_v1.py run --run runs/scratch_apps
python scripts/run_scratch_applications_v1.py report --run runs/scratch_apps
```

`prepare` 冻结配置、源码与资产哈希并开始计算整组预算，随后执行 `run`。首次检查可在新的输出目录执行 `prepare ... --smoke`，使用缩短预算验证管线；正式训练另建目录。套件固定使用已有几何／标定库和 FastDepth，学习网络与采样回放按随机初始化协议建立。

## 对齐书稿的室内 RGB

此路线对应历史初始化的 24 轮实验。复制 [seed41 书稿配置](../configs/rgb_book_reproduction_v1/seed41_book.json) 为 `configs/rgb_book_local.json`，按以下项目配置本机运行：

| 字段 | 设置内容 |
|---|---|
| `jobs[0].config.initial_checkpoint` | 协议匹配的完整 `initial.pt` |
| `bootstrap_root`、`book_cases` | bootstrap 目录与评价起点清单；后者包含 48 个验证、360 个测试起点 |
| `source_files` | 本次实际使用的源码路径 |
| `comparative_sources` | 若引用历史对照，提供对应记录；新实验不引用时使用空对象并记录该变更 |
| `absolute_deadline_unix`、`maximum_minutes` | 本次截止时间和预算；模板中的历史截止时间须更新，正式流程上限为 360 分钟 |
| `runtime_lock`（若设置） | 本机采集的运行时锁 |
| `jobs[0].config.device` | 例如 `cuda:0` |

同时准备参考 RGB／深度缓存、场景与来源清单以及 FastDepth。专用历史后端的一部分资源按仓库相对路径查找，具体清单见 [外部资源](ASSETS.md)。

```bash
python scripts/run_rgb_book_reproduction_v1.py validate --config configs/rgb_book_local.json
python scripts/run_rgb_book_reproduction_v1.py supervise --config configs/rgb_book_local.json --output runs/rgb_book_seed41
```

`validate` 检查配置与起点协议；完整资源加载和渲染仍需首次短流程检查。短流程使用 [engineering 配置](../configs/rgb_book_reproduction_v1/engineering_smoke.json)的新本地副本，独立输出；正式模式固定 1,000 次表示预热、1,000 次 Critic 预训练、24 ×（300 次表示＋200 次 M）更新，每两轮验证，选模后统一测试。

训练、验证及测试的写入进程全部退出后，在外部 PowerShell 枚举完整 Python 进程，立即生成快照并运行报告：

```powershell
$ErrorActionPreference = 'Stop'
$rgbProcesses = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { $_.Name -match '^(pythonw?|py)(\d+(\.\d+)*)?\.exe$' } | Select-Object ProcessId, ParentProcessId, Name, CommandLine)
$rgbSnapshot = @{ schema = 'unified_training_process_snapshot'; query_success = $true; created_unix = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0; processes = @($rgbProcesses) }
$rgbSnapshot | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath runs/rgb_book_process_snapshot.json -Encoding UTF8
python scripts/run_rgb_book_reproduction_v1.py report --root runs/rgb_book_seed41 --output artifacts/rgb_book_seed41_report --process-snapshot runs/rgb_book_process_snapshot.json
```

报告核验 60 秒内的快照、冻结文件、轨迹端点、控制量与保持计数，并导出曲线。快照中仍有写入者或无法识别的 Python 进程时，报告入口停止并说明原因；应核对进程身份，待相关工作结束后重新取快照。Windows 虚拟环境可能显示父子两个 Python 进程，均保留在列表中。

## 书稿数值与 RL 对照

以下是独立子实验入口，具体范围见 [章节对应表](BOOK_TASKS.md#书稿章节与实验入口)：

```bash
python scripts/run_book_foundations_overnight_v1.py --output runs/book_foundations --max-minutes 10
python scripts/run_book_disk_overnight_v1.py --output runs/book_disk --seed 41 --max-minutes 30
python scripts/run_book_grid_overnight_v1.py --output runs/book_grid --seed 41 --episodes 10000 --max-minutes 15
python scripts/run_book_rl_overnight_v1.py train --config extensions/book_rl_overnight_v1/seed41.json --output runs/book_rl
```

基础数值例与有限 MDP 复算、固定 V 的 D 学习、固定证书的 SAC 对照分别记录；这些入口不全部训练 M。生成实验另外需要扩散与评价模型，使用 `scripts/run_book_generation_overnight_v1.py --config 本地配置 --output 新目录`。

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
