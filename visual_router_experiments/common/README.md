# common

本目录用于保存 Visual Router 各阶段共享代码。

后续建议放置：

- prediction cache schema 和读写工具；
- item-channel-window sample key 工具；
- MAE/MSE/regret/oracle label 计算函数；
- 伪图像 2D 堆叠张量构造函数；
- ViT/视觉 encoder 运行内 embedding 工具；
- 通用评估和汇总表生成工具。

当前已有文件：

| 文件 | 功能 |
| --- | --- |
| `__init__.py` | 将 `common/` 标记为可导入 Python package |
| `prediction_cache_schema.py` | 定义 item-channel-window 级 prediction cache key、manifest record、窗口级 MAE/MSE 计算和基础一致性校验工具 |
| `pseudo_imageization.py` | 定义在线 tensor-first 伪图像化工具，包括 per-window normalization、FFT top3 周期选择、3view/top3fold 语义通道构造，以及 encoder 前 `hf_vit_0_5` / `torchvision_imagenet` normalization |
| `round2_layout_registry.py` | 定义 Visual Router V2 Round2 layout registry 和 GPU tensor imageization adapter，覆盖 `current_rgb_3view`、`spatial_panel_3view`、`line_only`、`line_difference_band`、`fft_absolute_energy`、`top3fold_period_layout`，并保留 deferred layout stub |
| `vit_embedding_utils.py` | 定义 online Visual Router 与历史离线 embedding pilot 共用的 ViT `pixel_values` 构造、CLS/patch pooling、dtype 解析和窗口 batch 索引工具；不保存 `.npy` 或长期 embedding cache |
