# Stabilization Learning

基于可学习度量与 Lyapunov 候选函数的控制学习工具箱。项目将物理环境、学习算法与实验调度分为独立模块，支持状态镇定、轨迹跟踪、观测器、视觉伺服和接触推动研究。

在误差或特征坐标中构造

\[
V_\theta(e)=e^\top M_\theta(e,c)e,\qquad M_\theta\succ0,
\]

结合动力学、局部雅可比或标定动作响应生成控制量；使用采样得到的控制成本、到达保持表现及证书约束更新参数。不同后端分别实现模型梯度、回合度量 PPO、离散 SAC 或视觉特征与度量联合学习。

## 安装

要求 Python 3.10 或更新版本。先克隆仓库并进入根目录：

```bash
git clone git@github.com:rfly-learning-books/stabilization-learning.git
cd stabilization-learning
python -m venv .venv
```

激活环境：Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`；Linux/macOS 使用 `source .venv/bin/activate`。

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m stabilization_learning list
```

若使用 NVIDIA GPU，先按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)安装适合本机驱动的 CUDA 版本，再安装本项目。MuJoCo 接触任务及室内渲染额外使用 `python -m pip install -r requirements-simulation.txt`。

## 快速训练与配对测试

以下质点任务不需要下载训练数据或权重：

```bash
python -m stabilization_learning train --config configs/public/point_mass_smoke.json --output runs/point_mass_smoke
python -m stabilization_learning evaluate --run runs/point_mass_smoke --seeds 20001 20002
```

测试在相同起点比较学习 M 与单位 M。`runs/point_mass_smoke` 保存模型、训练历史和评价结果；生成的文件不进入 Git。示例采用短训练预算，用于检查安装和接口。

Python 接口：

```python
from stabilization_gym import make
from stabilization_learning import StabilizationLearner, TrainConfig

env = make("point_mass", max_steps=50)
learner = StabilizationLearner(
    env,
    config=TrainConfig(episodes=4, batch_episodes=2, epochs=1),
    env_id="point_mass",
    env_kwargs={"max_steps": 50},
)
learner.train("runs/point_mass_api")
env.close()
```

## 统一训练流程

统一 Runner 按以下顺序组织任务：

`初始化 → 初始验证与采样 → 表征预热 → Critic 预训练 → epoch 循环 → 全组模型冻结 → 独立测试`

每轮包含采样、训练、验证和检查点保存。后端没有的预热阶段记为跳过。训练、验证与测试使用独立种子；正式测试不参与选模。

在仓库根目录运行：

```bash
python scripts/run_unified_training_v4.py preflight --config configs/public/unified_cpu_smoke.json --output runs/unified_preflight.json
python scripts/run_unified_training_v4.py train --config configs/public/unified_cpu_smoke.json --output runs/unified_cpu_smoke
```

三个观测任务使用修复了路径注册的 v5 入口：

```bash
python scripts/run_observer_training_v5.py train --config configs/public/observer_cpu_smoke.json --output runs/observer_cpu_smoke
```

详细参数、运行时锁、GPU 配置及输出协议见 [训练指南](docs/TRAINING.md)。按书稿章节选择环境、查找从头训练入口及已有实验结果，见 [训练任务与书稿对应关系](docs/BOOK_TASKS.md)。

RGB 新模板支持检查点无损去重与有界回放，配置和空间测算见 [runs 存储说明](docs/RUN_STORAGE.md)。

GMM-D 与拉格朗日 L 学习的模型、控制器及 Runner 协作接口见 [GMM 与 L 学习接口](docs/GMM_L_LEARNING_INTERFACES.md)。一维圆盘 GMM 已有专用实现；L 学习与统一 v4 后端仍待实现。

## 库与职责

| 层 | 代码 | 职责 |
|---|---|---|
| Env | `stabilization-gym/stabilization_gym` | 状态转移、观测、动作界限、终止与保持判据 |
| Algo | `stabilization-learning-control/stabilization_learning` | 特征、M、策略、Critic、损失与优化器 |
| Runner | `extensions/reproducible_training_v4` 等 | 采样、预热、训练、验证、冻结、测试与保存 |
| 入口 | `scripts/` | 命令行与具体实验配置连接 |
| 配置 | `configs/public/` | 可移植的入门配置 |
| 研究后端 | `extensions/`、`migration/next_envs_v1/` | 保留各任务的算法和物理协议实现 |

两个核心包可分别安装。版本化 Runner 从仓库运行，并调用相应后端；保留的早期模块是明确的实现依赖。接口详见 [架构与扩展](docs/ARCHITECTURE.md)。

## 任务与资源

| 任务 | 入口／后端 | 外部资源 |
|---|---|---|
| 质点、单摆、车摆、双连杆、独轮车、多旋翼 | 通用 CLI / `episode_ppo` | 无训练资产要求 |
| 状态依赖 M | `state_model_gradient` | 基础任务无需训练资产 |
| 观测器 | v5 / `observer_model_gradient` | 三个内置任务无需训练资产 |
| 轻量视觉伺服 | 通用 CLI 的 `visual_servo` | 自生成纹理与 DCT 特征 |
| 书稿室内 RGB | `run_rgb_book_reproduction_v1.py` | AE 初始化、深度模型、bootstrap、参考缓存及来源清单 |
| Push 离散 SAC | v4 / `push_discrete_sac` | 校准响应与时间库 |
| 从头联合成本 Push / SO101 | `run_joint_certificate_v2.py` / Full-M PPO | 校准响应与时间库；MuJoCo |
| 从头 Push、SO101 课程与 RGB | `run_scratch_applications_v1.py` | 几何能力库、场景清单、FastDepth；MuJoCo |

轻量 `visual_servo` 与书稿室内 RGB 是不同任务。书稿分支保留训练程序，但其外部训练数据与预训练模型不在本仓库。资源目录和配置方式见 [外部资源](docs/ASSETS.md)。

已有实验中，随机 Full-M 的 Push 三种子训练将留出形状成功数由各 97/128 提高至各 123/128；从首次更新加入时间、误差、切换和输入成本的 seed81，在 SO101 仿真保持 32/32 成功，平均耗时降低 13.04%，误差积分增加 7.81%。RGB 的收益来自专用室内渲染环境，须按对应初始化和评价协议复现。实验范围、对照定义与结果来源列于 [书稿任务说明](docs/BOOK_TASKS.md)。

## 开发与验证

```bash
python -m pytest
python scripts/check_publication.py
```

默认测试集覆盖免外部资产的核心功能；依赖历史数据、检查点及校准库的测试保留为显式扩展测试。当前研究实现中已定位的问题列于 [实现状态](docs/IMPLEMENTATION_STATUS.md)。正定 M 提供候选函数结构；稳定性结论仍需对应任务的下降条件、适用域和闭环验证。

## 发布内容与许可

仓库发布源代码、配置、测试和带来源声明的静态场景资源；不发布训练数据、权重、回放、运行日志或本机环境。

第三方代码保留各自许可证。项目未统一指定新的整体许可证，详见 [许可与来源](LICENSES.md)及 [第三方说明](THIRD_PARTY_NOTICES.md)。
