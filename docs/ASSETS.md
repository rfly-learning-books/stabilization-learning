# 外部资源

仓库不包含训练样本、校准响应数据、权重、检查点或回放。基本状态控制及内置观测任务直接运行；专用任务需要另备资源。

| 后端 | 资源 |
|---|---|
| 通用 PushAnyShape | 模式库、几何与标定响应 |
| 时间感知 Push / SAC / SO101 | 配置中的 `time_bank`、几何与响应标定 |
| 书稿 RGB | 完整 `initial.pt`、bootstrap、参考RGB/深度、FastDepth、场景与来源清单 |
| 生成实验 | 扩散与评价模型及其许可证 |

复制 `assets.example.json` 为 `assets.json`，填写本机路径；相对路径相对于该 JSON 所在目录解析。也可令 `STABILIZATION_ASSET_CONFIG` 指向其他配置文件。`assets.json` 和 `external_assets/` 被 Git 忽略。

未指定配置时，通用环境使用 `STABILIZATION_ASSET_ROOT`，默认位于用户目录 `.stabilization/assets`。只在实际调用相关任务时加载资产。

上述资产配置用于支持该解析接口的通用环境。版本化研究入口还会读取自身配置及固定的仓库相对路径；`assets.json` 不统一覆盖所有历史后端。

版本化书稿入口还会核对源码、清单及初始化数据哈希。资源必须来自同一协议，不能用任意同名文件替代。其配置中的历史路径应在新的本地配置中替换；原机器硬件锁没有发布。

## 从头应用套件资源

`run_scratch_applications_v1.py prepare` 会核对以下内容，须在准备套件前就位：

| 仓库相对位置 | 内容及用途 |
|---|---|
| `assets/push_geometric32/` | 32 形状几何与标定能力库；提供环境响应 |
| `migration/visual_scene_reconstruction/scenes.json` | 室内场景划分与来源清单 |
| `third_party/so101_timeaware_v1/MIGRATION.json` | SO101 资源来源清单 |
| `third_party/visual_original/runtime/dependencies/fastdepth_mobilenet_l1/extracted_state_dict.pt` | 固定 FastDepth 权重 |
| 同目录 `models.py`、SO101 目录中的 PNG 纹理 | 已保留的模型源码及静态渲染资源 |

前四项中的实验数据、权重及原运行资源清单没有随源码发布，需按选定协议另备。套件随机初始化 AE、M 和 Critic；固定几何、响应库与 FastDepth 属于保留先验。

联合成本 v2 的 `push-prepare` 可从 MuJoCo 接触试验新建八形状响应／时间库，再将生成的 `time_bank.json` 传入 `push-train`；也支持显式提供已有匹配库。命令见 [训练指南](TRAINING.md#从头联合成本-push-与-so101)。

`assets/recovered_visual_legacy/src` 中的 Python 文件是运行源码。Poly Haven 网格与纹理为带 CC0 来源说明的静态渲染资源，不是采样数据。
