# Env / Algo / Runner

`stabilization-gym` 定义物理环境和公共规格，不依赖学习包。`stabilization-learning-control` 提供特征、M、Critic、策略与优化器。仓库级 Runner 调用后端连接两者。

```text
配置与命令行
    ↓
Runner：初始化 → 预热 → 采样/更新/验证 → 冻结 → 测试
    ├── Algo：特征 φ、M、Critic、优化器
    └── Env：状态、观测、动作、物理步进、任务判据
```

环境提供任务模型或控制接口；算法不承担真实状态步进。可微模型、PPO 和 SAC 共享阶段组织，各自保留采样与梯度契约。

| 路径 | 内容 |
|---|---|
| `stabilization-gym/stabilization_gym/common` | 公共类型、动作空间、资产路径 |
| `stabilization-gym/stabilization_gym/envs` | 经典控制、机器人、视觉和推动任务 |
| `stabilization-learning-control/stabilization_learning/algos` | 学习算法 |
| `stabilization-learning-control/stabilization_learning/policies` | 控制策略 |
| `extensions/reproducible_training_v4` | 生命周期、确定性运行和后端 |
| `extensions/observer_training_v5` | 观测入口规范化与诊断 |
| `extensions/rgb_book_reproduction_v1` | 书稿视觉采样与评测协议 |
| `scripts/m_learning_suite` | 回合任务公共适配 |
| `migration/next_envs_v1` | 补充环境运行代码 |
| `third_party` | 保留来源和许可的实现依赖 |

新增任务时，先定义物理步进、观测、输入界限、成功与保持判据，再声明误差坐标及 M 维度，实现控制生成与学习目标，最后接入后端各阶段并编写配对测试。

状态依赖 M 的有限步变化必须计算为 `V_theta(e_next)-V_theta(e)`，两端分别计算函数值。跨 M 的评价应同时报告物理误差、保持率、完整时间和输入成本。
