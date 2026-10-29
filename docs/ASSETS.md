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

版本化书稿入口还会核对源码、清单及初始化数据哈希。资源必须来自同一协议，不能用任意同名文件替代。其配置中的历史路径应在新的本地配置中替换；原机器硬件锁没有发布。

`assets/recovered_visual_legacy/src` 中的 Python 文件是运行源码。Poly Haven 网格与纹理为带 CC0 来源说明的静态渲染资源，不是采样数据。
