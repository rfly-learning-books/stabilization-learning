# GMM-D 与拉格朗日 L 学习：接口和协作任务

本文为第 8 章的两条模型学习路线预留接口。物理环境、学习算法、实验调度分别归属于 Env、Algo 和 Runner；接口不改变现有实验协议，也不预设某种算法一定优于基线。

## 当前状态

| 线路 | 已有 | 本次接入 | 待协作实现 |
|---|---|---|---|
| GMM-D | 固定接触侧、标量圆盘的专用 GMM 局部增量模型；[专用实验入口](../scripts/run_book_disk_overnight_v1.py) | 模型和数据契约；[圆盘 GMM 适配器](../extensions/book_disk_overnight_v1/contract_adapter.py)；预测与真实离散 D 的共用计算 | 多维模型、一般联合分布 GMR、迭代式 V／策略更新，以及 v4 后端 |
| L 学习 | 双连杆等物理环境及书稿模型定义；文献中的机械臂和四旋翼结果 | 机械数据、结构动力学、模型拟合、控制器、误差能量和 Runner 契约 | 结构模型、训练器、跟踪控制器、积分补偿、误差能量实现与独立实验 |

**上述 L 学习是拉格朗日动力学学习，不是 Lyapunov critic 的 LAC，也不是现有神经 M 训练。** 当前接口和圆盘桥接可用于开发；L 算法及两条路线的 v4 后端尚未实现或注册，不能作为统一训练命令的可运行后端。书稿文献图和历史圆盘实验均不是本次新增训练结果。

GMM 的现有实现先对归一化的状态—动作进行混合分区，再拟合非负局部增量增益，预测形式为 <code>delta = a * sum(h_k * b_k)</code>。它是该一维任务的保零结构，不等同于一般联合高斯分布条件回归。

## 代码位置和职责

~~~text
Env: stabilization-gym/stabilization_gym/...
  状态与动作定义、实际物理步进、实际作用量、任务/成功/约束

Algo: stabilization-learning-control/stabilization_learning/algos/model_based/
  数据批次、模型/证书/控制器契约，离散 D 的 NumPy 诊断
  拟接入的 GMM 拟合器、拉格朗日模型与控制器属于这一层

Runner: extensions/reproducible_training_v4/...
  初始化、采样与预热、逐轮更新与验证、冻结、测试、记录

Existing adapter: extensions/book_disk_overnight_v1/contract_adapter.py
  将原圆盘 GMM 拟合与一步预测适配为共用模型契约
~~~

接口代码见 [contracts.py](../stabilization-learning-control/stabilization_learning/algos/model_based/contracts.py)。Env 不通过真实后继替代预测；模型不修改真实环境状态。Runner 接口 <code>RunnerBackend</code> 只约定方法形状，未自动注册实现。

## GMM：模型、评价和 D

<code>DiscreteTransitionBatch(state, applied_action, next_state, dt_seconds)</code> 使用二维批次，首维是样本。<code>GMMFitter.fit(batch, seed=...)</code> 返回拟合后的 <code>GMMTransitionModel</code> 和诊断；模型的 <code>predict_next(state, action)</code> 必须返回**完整后继状态**，<code>memberships</code> 返回分量权重。模型训练使用真实施加的动作；固定采样周期及坐标单位随协议保存。

当前候选评价 <code>DiscreteCertificate.value(state)</code> 每个状态返回一个标量。离散形式分别计算

~~~text
预测 D = V(model.predict_next(s, a)) - V(s)
实际 D = V(s_next) - V(s)
~~~

<code>predicted_discrete_change</code> 与 <code>observed_discrete_change</code> 在两个端点分别调用 V，因此状态相关 M 在后继处重新计算。它们是 NumPy 离线诊断工具，不保留 PyTorch 梯度；需要梯度或 GPU 的训练实现应以张量实现相同数学关系。模型不直接给出 V，替换 V 后可重算 D，但相关误差裕度和闭环评价需要重新核验。连续时间导数 ∇V·f 与一步差分的单位和采样条件不同，不能混写。

以下示例使用已安装的两个库，在仓库根目录运行；只构造小规模内存数据，不生成历史实验结果：

~~~python
import numpy as np

from extensions.book_disk_overnight_v1.env import DiskConfig, DiskEnv, transitions
from extensions.book_disk_overnight_v1.contract_adapter import DiskGMMFitter
from stabilization_learning.algos.model_based import (
    DiscreteTransitionBatch,
    observed_discrete_change,
    predicted_discrete_change,
)

config = DiskConfig(response="varying")
env = DiskEnv(config)
train = transitions(env, 256, np.random.default_rng(41))
validation = transitions(env, 64, np.random.default_rng(42))
batch = DiscreteTransitionBatch(
    train["observed_x"][:, None], train["action"][:, None],
    train["observed_next_x"][:, None], config.dt,
)
model, diagnostics = DiskGMMFitter(config, components=4).fit(batch, seed=41)

class DiskValue:
    def value(self, state):
        return 0.5 * state[:, 0] ** 2

state, action = validation["x"][:, None], validation["action"][:, None]
predicted = predicted_discrete_change(model, DiskValue(), state, action)
actual = observed_discrete_change(DiskValue(), state, validation["next_x"][:, None])
print("validation D RMSE:", np.sqrt(np.mean((predicted - actual) ** 2)))
~~~

较完整的原圆盘实验可运行：

~~~bash
python scripts/run_book_disk_overnight_v1.py --smoke --output runs/book_disk_smoke
~~~

输出路径必须是新的运行目录。该实验比较已知模型、直接 D、单一增量模型和 GMM 局部模型；短流程只验证工程链，不替代多种子长训。直接 D 模型没有后继状态预测器，不能接成 <code>GMMTransitionModel</code>。

## L 学习：模型、控制器和误差能量

<code>MechanicalTrainingBatch</code> 约定 <code>q</code>、<code>velocity</code>、<code>acceleration</code>、<code>applied_generalized_force</code> 的形状均为 (样本数, 广义坐标数)，并可另存指令广义力与时间戳。加速度的来源、滤波、时延、力矩映射、饱和和单位须由任务协议说明。契约保留传入的数组或张量；它不会自动完成设备转移或验证有限性。

<code>LagrangianFitter</code> 根据训练批次返回 <code>LagrangianDynamics</code>。模型暴露同一结构下的 <code>mass_matrix</code>、<code>potential</code>、<code>coriolis_matrix</code>、<code>gravity</code> 和 <code>inverse_dynamics</code>：

~~~text
L(q, qdot) = 1/2 qdotᵀ M_mass(q) qdot - P(q)
tau_hat = M_mass(q) qddot + C(q, qdot) qdot + g(q)
~~~

其中 <code>M_mass</code> 是机械系统惯性矩阵，不能与前面 <code>V=eᵀ M_metric e</code> 的可学习证书度量混称。模型实现需检查惯性矩阵正定性、导数一致性和力学结构；摩擦、外力和未建模部分应明确处理，不能默认为不存在。

<code>LLearningController</code> 使用 <code>MechanicalReference(q, velocity, acceleration)</code> 接收位置、速度和加速度参考，并显式接收／返回每回合的内部状态，便于实现积分补偿与重置。它输出期望**广义力**；Env／任务适配层负责执行器映射、限幅和真实作用量记录。按书稿约定误差 <code>e = q_d - q</code>、复合误差 <code>r = qdot_d - qdot + Λe</code>；公式、离散积分方式与符号需逐项对照书稿。

<code>LLearningEnergy</code> 为基于同一模型的跟踪误差能量预留单独接口。离线审核所需真值或残差可通过 <code>audit_context</code> 提供；不可把仅在仿真中可获取的真实扰动当作在线控制输入。仅在规定的固定模型、残差、输入和工作域条件满足时，才可使用书稿对应的解析结论。

## Runner 对接清单

1. 编写具体 backend，实现 <code>RunnerBackend</code> 的初始化、采样、预热、逐 epoch 更新、验证、保存／加载和测试；不需要的阶段返回明确的跳过记录。
2. v4 当前先做初始验证，再采样和预训练；新后端须定义可评价的初始模型／策略，且不能把测试集用于选模。
3. <code>validate()</code> 返回有限、按优先级排列且越大越好的 <code>selection_score</code>；将成功计数、误差和约束解释写入冻结协议。<code>save()</code> 必须保存恢复训练所需的模型、优化器、随机状态与配置，不能覆盖既有运行。
4. 在 [v4 后端工厂](../extensions/reproducible_training_v4/runner.py)和[配置校验](../extensions/reproducible_training_v4/validation.py)共同注册具体实现，补齐源码／资产清单和 checkpoint 恢复检查。当前 v4 要求正数 <code>certificate_updates</code> 且默认比较 selected／initial／identity；纯动力学拟合与 L 跟踪未必适合这一约定，需先为新后端定义合理阶段和基线，不能把任意模型更新伪称为证书更新。
5. 同起点、同动作界限、同训练预算下比较模型与控制器；输出训练和验证曲线、逐回合物理指标、预测／实际 D 或误差能量、失败及模型适用域。全部模型冻结后访问独立测试集。

## 建议分工与验收

- **GMM 维护者**：完成一般 GMR 与多维环境接口、样本／分量数对照、预测误差和闭环差异；保留现有一维结构作为单独基线。
- **L 学习维护者**：先完成双连杆模型辨识、结构恒等式检查和固定模型跟踪控制；再评估积分、残差与力矩饱和；四旋翼姿态单列。
- **框架维护者**：补 v4 backend 与状态恢复、来源清单、训练／验证／测试拆分和配对报告；在算法具备实际实现及测试后再注册名称。
- **共同交付**：协议、实现、训练与验证曲线、冻结模型索引、独立测试与失败边界、可复算图表。训练数据、权重、回放和日志放在本地运行目录，不提交仓库。

完整研究任务见[书稿仓库的 GMM 与 L 学习实验列表](https://github.com/rfly-learning-books/Rfly-Stable-Learning-Book/blob/main/docs/GMM%E4%B8%8EL%E5%AD%A6%E4%B9%A0%E5%AE%9E%E9%AA%8C%E4%BB%BB%E5%8A%A1%E5%88%97%E8%A1%A8_20261029.md)。该列表是待办与验收要求，不表示相应新实验已完成。
