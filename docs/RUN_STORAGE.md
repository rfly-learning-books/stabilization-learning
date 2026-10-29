# 训练 runs 存储

书稿 RGB 和冻结特征 Critic/M 的新配置已启用存储优化。学习目标、采样索引、随机状态、训练轮数和评价协议保持原样；旧运行目录不由此配置自动清理。

## 原来的空间主要花在哪里

2026 年 10 月 29 日对已完成目录的统计：

| 运行 | 主要占用 | 原因 |
|---|---|---|
| `rgb_book_seed41_20261029_0036`，共 13.73 GiB | 回放 6.86 GiB；检查点目录 4.50 GiB；另有预热快照、测试轨迹和 M 探针 | 回放有 1,200 个片段，活动采样索引只含 227 个；移出索引的图像文件仍保存。各轮重复写入完整模型张量 |
| `rgb_cost_m_seed41_20261028_213608` | 检查点目录 11.36 GiB；计入初始快照和原始候选后为 11.88 GiB | 特征网络和部分优化器已冻结，每轮仍重复保存相同内容 |

RGB 片段中的图像和局部扰动探针占据主要回放空间。NPZ 已经压缩，单纯再压一次图像不能解决旧片段累积的问题。

## 新规则

### 检查点无损去重

张量存储按 4 MiB 分块，使用 SHA-256 标识内容，并进行无损压缩。同一任务中相同的数据块只保存一次；每个检查点保存自己的完整内容清单。全部模型候选、优化器状态、随机状态和张量精度都保留。

```text
jobs/rgb_book_seed41/
  .checkpoint_objects/       # 各检查点共享的无损数据块
  checkpoints/initial.pt     # 内容清单
  checkpoints/ready.pt
  checkpoints/epoch_0001.pt
  ...
  pretraining/ae_warmup.pt
```

复制或归档任务时，包含整个任务目录及隐藏的 `.checkpoint_objects`。读取时逐块校验内容；旧版独立 `.pt` 继续兼容：

```python
from stabilization_learning.storage import load_checkpoint, export_checkpoint

state = load_checkpoint("runs/example/jobs/rgb_book_seed41/checkpoints/ready.pt", map_location="cpu")

# 需要单文件交付时，导出到一个新文件，随后可直接使用 torch.load。
export_checkpoint(
    "runs/example/jobs/rgb_book_seed41/checkpoints/ready.pt",
    "exports/ready_standalone.pt",
)
```

冻结特征任务的原始候选仍通过 `extensions.rgb_cost_m_v2.checkpoint.load_checkpoint` 读取，该入口会同时展开其对 `ready.pt` 的差分引用。

### 有界回放

书稿 RGB 的 `active_and_restart` 模式保留活动采样池，以及完整检查点引用的数据：初始模型、AE 预热完成、Critic 预训练完成和末轮快照。其他轮次继续保存完整的推理模型。

一个片段同时满足“由本次写入器创建、已经移出活动索引、未被完整检查点引用”时，才在新的运行中释放其磁盘文件。删除前记录文件哈希和字节数。导入的 bootstrap、未知文件、模型和测试轨迹保留。该模式要求全新的回放目录，不能用来接管并清理历史 runs。

因此，新规则保留活动采样和完整快照所需的数据，但不保存所有过期的训练图像。需要全量原始图像归档时，设 `replay` 为 `all`。

## 使用配置

[书稿正式配置](../configs/rgb_book_reproduction_v1/seed41_book.json)和[短流程配置](../configs/rgb_book_reproduction_v1/engineering_smoke.json)已包含：

```json
"storage": {
  "checkpoints": "lossless_objects_v1",
  "replay": "active_and_restart"
}
```

字段位于 `jobs[0].config` 内。[冻结特征配置](../configs/rgb_cost_m_v2/seed41.json)启用检查点去重，保留其原回放策略。旧本地配置未声明 `storage` 时，仍使用原保存方式；新实验可从更新后的模板建立配置。

训练、冻结、测试与报告入口保持原样，见 [训练指南](TRAINING.md)。临时序列化还需要最多一个标准检查点大小的额外空间，该临时文件在操作结束时关闭并释放。

## 实测文件测算与验证

以下读取已有文件，逐块计算真实压缩长度和去重后大小，没有改写或删除源运行：

| 对象 | 原大小 | 按新规则测算 | 减少 |
|---|---:|---:|---:|
| 书稿 RGB 的 27 个检查点，含 AE 预热 | 4.95 GiB | 3.74 GiB | 24.5% |
| 同一运行的回放 | 6.86 GiB | 1.47 GiB | 78.5% |
| 冻结特征任务的 37 个检查点，含初始和原始候选 | 11.88 GiB | 0.65 GiB | 94.6% |

保留其余已有文件时，书稿 RGB 整轮由 13.73 GiB 测算为约 7.13 GiB；新产生的回放保留记录还有少量开销。不同模型变化量、轨迹长度和评价输出会改变实际占用，这不是新一轮长训的实测总量。

2026 年 10 月 29 日在 Windows、Python 3.10 环境运行相关回归，共 191 项通过。验证覆盖：模型及优化器读回、恢复后的下一次优化更新、随机状态、张量视图与共享存储、CPU／CUDA 读取、旧检查点兼容、完整快照回放保护、真实回放类的采样一致性、冻结模型核对以及缺失／损坏数据检测。检查点对象目录也已加入 Git 忽略和发布检查。
