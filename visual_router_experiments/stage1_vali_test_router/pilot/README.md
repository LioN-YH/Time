# Stage 1 Pilot Scripts

本目录保存 Stage 1 正式实验前用于验证数据流、cache schema 和 baseline 口径的 pilot 脚本。

这些脚本的共同约束：

- 输入和输出规模应保持可控，用于快速验证而不是生成正式结论；
- 输出仍写入 `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_*_pilot/`；
- 运行后如果产生新的长期结果或调整口径，需要同步写中文实验日志；
- 当某段逻辑被正式实验复用时，应上收到 `stage1_vali_test_router/` 或 `visual_router_experiments/common/`，避免把正式流程长期依赖在 pilot 目录中。

| 文件 | 功能 |
| --- | --- |
| `build_prediction_cache_pilot.py` | 小规模生成 window-level prediction cache，验证专家预测、窗口 key、数组落盘和 MAE/MSE 对齐 |
| `build_online_pseudo_image_pilot.py` | 基于已有 oracle label 样本清单重新加载 Quito 历史窗口 x，在线生成 3view/top3fold 伪图像并记录 index、metadata、latency 和少量 debug PNG；不保存全量 tensor cache |
| `build_structure_feature_cache_pilot.py` | 基于 TimeFuse 单变量元特征生成 window-level 数值结构 feature cache；删除多变量/跨变量特征，仅作为非视觉 router baseline 输入 |
| `compute_window_oracle_from_cache.py` | 基于 pilot manifest 计算 window-level oracle label、expert regret 和 best-single-vs-oracle 汇总 |
| `enrich_cache_with_tsf_cell.py` | 为 pilot manifest/oracle labels 合并 TSF cell 元信息，并生成分层 oracle summary |
| `train_structure_router_pilot.py` | 使用 TimeFuse 单变量元特征训练最小 LogisticRegression router；vali fit scaler/router，test 评估专家选择 MAE |
