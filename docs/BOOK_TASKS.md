# 训练任务与书稿对应关系

本文按 Reviewed 版第 1—17 章组织项目中的环境、学习机制、训练入口和已有结果。安装步骤见 [README](../README.md)，命令见 [训练指南](TRAINING.md)，资源准备见 [外部资源](ASSETS.md)。结果记录截至 2026 年 10 月 29 日。

## 选择训练任务

| 研究目的 | 使用路径 | 初始化与实验含义 |
|---|---|---|
| 安装检查、基础控制与新环境开发 | 通用 CLI；统一 v4 | 免外部训练资产；短配置检验接口，长配置另行声明预算 |
| 从首次更新联合优化时间、误差、切换和输入成本 | [联合 Push / SO101 v2](../scripts/run_joint_certificate_v2.py) | 随机 Full-M Actor、随机 Critic、空优化器；对应 seed81 实验 |
| 从头学习跨形状 Push，再训练 SO101 时间目标 | [scratch 应用套件](../scripts/run_scratch_applications_v1.py) | Push 从随机网络开始；SO101 继承本次 Push 结果，随后加入时间训练 |
| 对齐书稿室内 RGB 的训练流程 | [RGB 书稿入口](../scripts/run_rgb_book_reproduction_v1.py) | 继承声明的历史初始化与 2,822 条 bootstrap；24 轮表示与 M 更新 |
| 从随机 AE、M、Critic 学习 RGB | scratch 应用套件中的 RGB 阶段 | 新回放；公开预训练 FastDepth 保持固定；对应 seed71 实验 |
| 学习观测器度量 | [观测 v5](../scripts/run_observer_training_v5.py) | 三个观测任务，使用估计误差、观测模型与观测器更新 |

书稿 RGB、从头 RGB、从头 Push 和联合成本 Push 是不同实验协议。环境、初始化、奖励、选模和评价时域共同决定结果；后端名称相近不表示检查点或成绩可互换。

## Env、Algo 与 Runner 的关系

Env 负责物理状态、观测、真实转移与任务判据；任务适配层定义误差、模型和控制构造；Algo 拥有表示、M、Critic、策略及优化器；Runner 管理采样、预热、更新、验证、冻结和测试。代码职责见 [架构说明](ARCHITECTURE.md)。

共同研究对象是由度量构造的候选函数，例如

\[
V_\theta(e,c)=e^\top M_\theta(e,c)e,\qquad M_\theta(e,c)\succ0.
\]

RGB 实现采用带系数的形式 `V = 0.5 eᵀMe`。模型或局部交互矩阵将当前 M 转为控制量，闭环样本再用于更新表示、度量或性能估计。对状态相关 M，有限步变化为

\[
D_\theta(s,a)=V_\theta(s')-V_\theta(s),
\]

两端分别计算 M。物理误差、时间和输入成本使用固定定义，避免随着学习 M 改变评价尺度。正定结构、预测下降与真实轨迹下降分别记录。

各算法保留自身目标：模型梯度通过模型传播；PPO 学习价值与策略分布；离散 SAC 学习候选动作 Q；书稿 RGB 的 Critic 使用导数监督。因此，seed81 的 Full-M PPO 结果不属于 `push_discrete_sac` 的实验结果，书稿 RGB 也不是累计奖励 Q 的实验。

## 已注册环境与专用环境

以下九个名称可传给 `stabilization_gym.make(...)`，注册代码见 [envs](../stabilization-gym/stabilization_gym/envs/__init__.py)。

| 环境名 | 物理任务 | 对应书稿内容 |
|---|---|---|
| `coupled_linear` | 双质量耦合线性镇定 | 第 2、6 章的状态反馈与镇定机制 |
| `point_mass` | 二维双积分器 | 第 2、6 章；第 8、11 章机制对照的基础系统 |
| `pendulum` | 力矩驱动倒立摆局部镇定 | 第 2、6 章；需区分书稿中的其他驱动方式 |
| `cartpole` | 全状态车摆局部镇定 | 第 2、6 章；输出反馈另用观测任务 |
| `two_link_arm` | 平面双连杆关节定位 | 第 2、6、8 章控制机制；当前实现使用给定模型 |
| `unicycle_tracking` | 非完整独轮车圆轨迹跟踪 | 第 2、6 章跟踪 |
| `multirotor_tracking` | 平面多旋翼悬停与圆轨迹 | 第 2、5、6 章；模型为平面动力学 |
| `push_any_shape` | 标定响应模型上的通用 M 接口 | 第 14 章推动；通用回合固定 M 与专用状态相关 Full-M 分别实现 |
| `visual_servo` | 自生成纹理、DCT 特征的轻量伺服 | 第 10、13 章机制检查 |

专用书稿任务通过下列接口或 Runner 使用，不全部注册为上述 `make` 名称。

| 专用任务 | 环境／任务代码 | 学习与运行路径 |
|---|---|---|
| 室内重建 RGB 伺服 | [RGB 物理环境](../stabilization-gym/stabilization_gym/envs/tasks/rgb_book_physical.py)、[视觉环境适配](../stabilization-gym/stabilization_gym/envs/tasks/original_visual_online.py) | RGB 书稿 Runner 或 scratch RGB |
| 跨形状标定 Push | [推动任务](../stabilization-gym/stabilization_gym/tasks_v2/push.py) | Full-M PPO、联合成本 v2；离散 SAC 另有 v4 后端 |
| SO101 执行 | [关节驱动仿真评价](../extensions/joint_certificate_v2/arm.py) | 冻结 M 在 MuJoCo 中执行；课程路线可先做时间训练 |
| 电机、车摆输出、已知几何位姿观测 | [观测任务工厂](../stabilization-gym/stabilization_gym/observation/__init__.py) | `make_observation_task`：`motor_speed`、`cartpole_output`、`cooperative_pose`；观测 v5 |
| 电机状态控制 | [DC motor](../extensions/state_conditioned_m_v1/envs/dc_motor.py) | v4 的 `dc_motor_gradient` |

室内 RGB 提供场景渲染、深度与参考交互矩阵；P0 为名义运动学，P1 增加隐藏执行器动态。它与 DCT `visual_servo` 使用不同观测和控制链路。SO101 使用关节驱动接触仿真，不能用平面双连杆 `two_link_arm` 的成绩替代。

## 书稿章节与实验入口

| 章节 | 本项目对应内容 | 入口／实现 | 实验范围 |
|---|---|---|---|
| 第 1 章 | 问题定义与共同机制 | [架构](ARCHITECTURE.md) | 方法说明 |
| 第 2—6 章 | 状态镇定、跟踪、稳定性与内模数值例 | [统一 v4](../scripts/run_unified_training_v4.py)、[基础数值例](../scripts/run_book_foundations_overnight_v1.py) | 基础环境训练与数值复算；不同输入和状态定义逐任务对齐 |
| 第 7、15 章 | 已知目标几何的位姿观测，以及电机、车摆观测 | [观测 v5](../scripts/run_observer_training_v5.py) | `cooperative_pose` 使用合成投影关键点；未知场景自身运动与完整估计控制耦合尚待实现 |
| 第 8 章 | 圆盘的已知模型、直接 D 学习、模型辨识和 GMM 对照 | [圆盘实验](../scripts/run_book_disk_overnight_v1.py) | 数值圆盘、固定 V 的下降量与控制学习 |
| 第 9 章 | 随机镇定与目标／陷阱统计 | [95 状态网格](../scripts/run_book_grid_overnight_v1.py) | 有限 MDP 的策略与概率复算；与神经 M 训练分别统计 |
| 第 10 章 | 表示、动作信息与可学习度量 | RGB 表示／M 训练和轻量视觉环境 | 完整视觉接触圆盘及时序消融尚待补齐 |
| 第 11 章 | RL、下降软惩罚与约束过滤 | [RL 对照](../scripts/run_book_rl_overnight_v1.py) | 二维积分器上的等采样预算 SAC 比较，使用固定证书 |
| 第 12 章 | 工具箱接口、安装、冻结与复现 | [训练指南](TRAINING.md)、[发布验证](RELEASE_VALIDATION.md) | 两个可安装库和版本化 Runner |
| 第 13 章 | 室内 RGB 视觉伺服 | [RGB 书稿流程](../scripts/run_rgb_book_reproduction_v1.py)、[早期迁移训练](../scripts/train_original_visual_migration.py) | 仿真布局、初态与深度尺度测试；完整飞行执行需单独实验 |
| 第 14 章 | Push-T／跨形状推动、模式选择与时间成本 | [联合成本 v2](../scripts/run_joint_certificate_v2.py)、[scratch 套件](../scripts/run_scratch_applications_v1.py) | 标定降阶模型训练与 SO101 MuJoCo 执行；实物实验另列 |
| 第 16 章 | 原采样、固定引导、回溯与学习引导 | [生成对照](../scripts/run_book_generation_overnight_v1.py) | 冻结 SD1.5 上的 64 图对照；人类盲评尚无评分 |
| 第 17 章及跨章 | 预测下降量、真实 V 变化、误差尺度及失败边界 | 各 Runner 的冻结评价和独立复算 | 报告轨迹与采样域的经验性质；全域认证另需误差界与可行域证明 |

数值算例、神经 M 训练和完整应用验证分别归类。多机协同、虚拟管道、流匹配规划等教学或建模材料不计入已经完成的应用训练。

## 已有训练结果与初始化

### 从头 Push 与联合成本 SO101

| 实验 | 初始化、训练与测试 | 已记录结果 |
|---|---|---|
| 跨形状 Push，seeds 71／72／73 [R1] | 随机 Full-M Actor／Critic、空优化器；各 1,000 次 PPO 更新；20／4／8 个训练／验证／测试形状；每测试形状 16 起点 | 各种子由随机初始 97/128 提高至所选模型 123/128；种子均值误差积分 0.605063 → 0.377666，切换 28.69 → 9.61；输入平方代理 0.055359 → 0.107478 |
| 从首次更新联合全部成本，seed81 [R2] | 随机 Full-M Actor／Critic、空优化器；1,000 次 PPO 更新；4／2／2 个形状；验证选第 420 次更新；同一冻结 M 直接部署 SO101 | 8 形状 × 4 起点：随机初始和所选 M 均 32/32 成功；耗时 33.451 → 29.088 秒（−13.04%）；切换 3.656 → 2.563；误差积分 0.903208 → 0.973718（+7.81%） |
| seed81 的 SO101 留出形状子集 [R2] | 上述 32 个组合中的两个留出形状，共 8 个组合 | 双方均 8/8；耗时 32.793 → 31.343 秒（−4.42%），误差积分增加 14.51% |

这两条路线保留了形状几何、标定响应和控制构造先验。随机 M 位于单位阵附近，经已有模型构造的控制器可以在训练前取得较高成功率。“从头训练”指学习网络及优化器的来源；留出形状排除在策略优化和选模之外，其响应／时间标定仍可用。

seed81 的任务成本为

\[
J=w_t T/T_{\rm ref}+w_e\int E_{\rm phys}\,dt+w_sN_{\rm switch}
  +w_u\int u_{\rm push}^{2}\,dt+w_f\mathbf1_{\rm failure}-w_g\mathbf1_{\rm success}.
\]

[已用配置](../configs/joint_certificate_v2/push_costs.json)设 `w_t=1`、`w_e=1`、`w_s=0.25`、`w_u=0.02`、`w_f=600`、`w_g=200`。时间覆盖换点、接近、推动、稳定与保持阶段；输入平方项是推杆速度代理。M 用于候选 V、后继变化与动作分布，PPO 根据联合回报更新。SO101 全体联合成本从 −164.732 降至 −169.298，包含成功奖励；时间与切换收益抵消了误差成本增加。

上述结果支持该协议下学习证书与模式选择的经验收益；独立分离两者贡献还需要固定 M／固定选择规则的消融。所选 SO101 轨迹仍有约 2.60% 的实际 V 回升。seed81 同批次 RGB 分支的 P0／P1 均为 0/16，推动的成功不延伸到该 RGB 分支。

### RGB 伺服

下表中的单位 M 与学习 M 使用同一所选编码器，以比较度量的作用。

| 实验 | 初始化与评价协议 | 单位 M → 学习 M |
|---|---|---|
| 重建环境迁移，seed41 [R3] | 历史初始化与 bootstrap；24 轮，验证选第 22 轮；P0／P1 各 360 测试起点，最多 300 步 | P0：243 → 271；P1：242 → 270，均净增 28 个成功起点 |
| Env／Algo／Runner 书稿流程，seed41 [R4] | 历史初始化与 2,822 条 bootstrap；24 轮，实际冻结第 8 轮；最高验证 37/48；P0／P1 各 360 测试起点，最多 300 步 | P0：239 → 250；P1：237 → 248，均净增 11 个成功起点 |
| 从头 RGB，seed71 [R1] | 随机 AE／M／导数 Critic、新回放，固定预训练 FastDepth；24 轮；P0／P1 各 72 测试起点，最多 200 步 | P0：41 → 47；P1：42 → 46；物理误差积分分别增加约 6.8% 和 10.7% |

室内重建环境的测试布局来自与训练相同的场景族，测试布局和起点分别保留。P1 测试执行器动态变化；深度倍率测试检查指定的深度尺度扰动。这些结果描述仿真泛化范围，原场景源码缺失部分由重建实现补齐，像素外观与历史书稿资源有差异。

书稿 Runner 的第 8 轮验证误差积分为 20.432，第 24 轮为 41.302；后期成功数有回落。实际选模还受到浮点同分比较影响：按有理数解释原规则会选择第 14 轮，表中仍保留实际冻结第 8 轮的测试结果。问题说明见 [实现状态](IMPLEMENTATION_STATUS.md)。历史 `φ20＋M13` 的 38/48 属于交叉检查点诊断，不能替代上述独立训练协议与 360 起点测试。

### 基础控制与观测

| 任务组 | 已有完整训练 | 结果与对照 |
|---|---|---|
| 耦合线性、质点、摆、车摆 [R5] | 各五种子 41—45，各 256 次模型梯度更新；解析 P0 初始化与锚定 | 学习策略和固定 P0 在 3,840 次评价均成功；误差成本分别降低 3.64%—5.24%、1.64%—2.16%、2.96%—4.43%、2.69%—3.21%；部分任务输入能量增加，摆到达稍慢 |
| 电机、车摆输出、已知几何位姿观测 [R6] | 各五种子 41—45，各 1,024 次更新 | 名义电机和车摆成功率 100%；位姿观测固定 M 为 22/32，学习 M 各种子为 32/32；KF／EKF 的成本仍更低，车摆证书下降违例仍高 |

这些历史长训结果与发布入口的短流程验证分别记录。v5 修复了观测入口重复路径导致的运行时锁核对问题；新入口已通过短流程检查，其长训成绩仍应由新的实验生成。`two_link_arm`、`unicycle_tracking`、`multirotor_tracking` 已接入训练，但已有对照存在改善和退化混合结果。

## 评价与复现口径

新实验先固定训练／验证／测试划分、种子、预算和选模规则，再生成数据。对照保持相同环境、起点、约束和评价时域。RGB 隔离 M 作用时固定编码器；Push 隔离 M 作用时固定响应模型、候选动作和选择规则。

每个实验同时报告成功保持率、固定物理误差积分、完整执行时间、模式切换、输入成本与真实下降违例。时间比较采用共同成功起点配对，并列出新增与丢失成功数。训练中原始候选、最后模型和验证所选模型分别保存，保留后期退化。

当前仓库包含上述训练实现与配置。获取相应外部资产后可开展同类实验；精确复算历史成绩还需要该次冻结源码、配置、运行时、资产和权重。源代码发布本身不携带这些训练产物。

## 结果来源索引

以下为原实验项目中的相对记录路径，用作结果标识。运行记录、权重和原始数据不随仓库上传。

| 编号 | 来源记录 |
|---|---|
| R1 | `runs/scratch_apps_so101_full_v1_20261026_184356/POST_ANALYSIS/ANALYSIS_ZH.md` |
| R2 | `runs/joint_certificate_v2_night_20261027_015308/launch_retry_20261027_015623/DELIVERY_REPORT_FINAL/REPORT.md` |
| R3 | `runs/visual_reconstructed_seed41_20261023_v1/analysis/MIGRATION_SUMMARY.md` |
| R4 | `artifacts/book_overnight_seed41_20261029_report/scientific_analysis_v1/SCIENTIFIC_ANALYSIS.md` |
| R5 | `artifacts/book_multiseed_v4_delivery_20261025/POST_REVIEW.md` |
| R6 | `runs/observer_applications_night_v1_20261026_010507/FINAL_REPORT/index.html` |
| R7 | `artifacts/push_initial_m_provenance_20261026/CORRECTION.md` |

R7 更正了旧 Push 对照的来源：304/310 的“初始 M”继承此前已训练 1,000 次更新的 Full-M 权重；本阶段计数和优化器曾被重置。继续训练后 307/310、单位 M 270/310 是该历史协议下的成绩。该起点称为“预训练 M”，与 R1、R2 的随机初始化分别引用。
