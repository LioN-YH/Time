# Visual Router V2 Round2 全局总结与主线推荐

日志日期：2026-06-22 09:31:55 CST

## 目的

在不启动新实验、不训练、不运行 ViT、不生成 feature cache、不新增样本、不修改 imageization 或 router head 主逻辑的前提下，汇总 Visual Router V2 Round2a-Round2f 的阶段结果，形成 Round2 全局总结、主线推荐、关键结果表和 metadata。

## 背景

Round1 已确认 `film_mean_patch_aux` 是当前最强后端主线，Visual Router V2 的主要收益来自 raw-soft expert fusion，而不是 oracle-label hard classifier。Round2 在固定 `film_mean_patch_aux` 后端的前提下比较 pseudo image / view layout，希望判断 `spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout` 等 layout 是否能改善 ViT visual embedding。

Round2 已完成 35k screening、65k expanded validation 和 P0-scale validation。本轮需要收束全局结论，重点回答 `spatial_panel_3view + film_mean_patch_aux` 是否应升级为 Round2 mainline candidate，以及下一步是否进入 full-scale validation planning。

## 操作

1. 读取用户提供的目标文件，确认本轮边界为只做 global summary 和 recommendation。
2. 检查当前分支和提交历史，确认位于 `exp/visual-router-v2-pilot`，HEAD 为 `db0cd06 Add round2 P0 spatial panel validation`。
3. 查阅现有轻量 summary 和 CSV，包括：
   - `experiment_summaries/visual_router_v2_round2/layout_screening/round2_layout_screening_summary.md`
   - `experiment_summaries/visual_router_v2_round2/expanded_layout_validation/round2_expanded_layout_validation_summary.md`
   - `experiment_summaries/visual_router_v2_round2/p0_spatial_panel_mainline/round2_p0_spatial_summary.md`
   - `experiment_summaries/visual_router_v2_round2/period_continuity/round2_period_continuity_summary.md`
   - `experiment_summaries/visual_router_v2_round2/period_continuity_addendum/round2_period_continuity_addendum_summary.md`
4. 新增全局汇总产物：
   - `experiment_summaries/visual_router_v2_round2/round2_global_summary.md`
   - `experiment_summaries/visual_router_v2_round2/round2_mainline_recommendation.md`
   - `experiment_summaries/visual_router_v2_round2/round2_global_key_results.csv`
   - `experiment_summaries/visual_router_v2_round2/round2_global_metadata.json`
5. 更新 `WORKSPACE_STRUCTURE.md`，登记 Round2 global summary 四个长期轻量产物。
6. 更新 `experiment_logs/README.md`，登记本日志。

## 结果

新增的全局 summary 明确记录了 Round2a-Round2f 的 timeline、每一步为什么做、输入、输出、结论，以及是否影响 mainline recommendation。

关键结果表覆盖 35k、65k 和 P0 三层核心 raw-soft MAE/MSE/regret：

- 35k `spatial_panel_3view + film_mean_patch_aux` selection raw-soft MAE=0.310385，test_small raw-soft MAE=0.398598。
- 65k `spatial_panel_3view + film_mean_patch_aux` selection raw-soft MAE=0.307233，test_expanded raw-soft MAE=0.394336。
- P0 `spatial_panel_3view + film_mean_patch_aux` pilot_selection raw-soft MAE=0.300227，frozen pilot_test raw-soft MAE=0.413558。
- Round1 `film_mean_patch_aux` frozen pilot_test raw-soft MAE=0.417824。
- P0 frozen pilot_test 上 spatial panel 相对 Round1 fallback 的 delta 为 MAE=-0.004266、MSE=-0.582819、regret=-0.004266。

文档中已明确：

- `spatial_panel_3view + film_mean_patch_aux` 是当前 Round2 mainline candidate。
- Round1 `film_mean_patch_aux` 是历史最强 fallback baseline。
- Round2 `current_rgb_3view + film_mean_patch_aux` 是 Round2 registry 下的 layout baseline。
- `top3fold_period_layout` 在 continuity/diagnostic 上有价值，但没有转化为 65k/P0 主指标优势，暂作为 side branch。
- `line_only`、`line_difference_band`、`fft_absolute_energy` 不建议作为当前主线。
- 下一步推荐进入 Visual Router V2 Round2 full-scale validation planning，而不是继续扩展 small layout 支线。

## 结论

Round2 当前应收束到 `spatial_panel_3view + film_mean_patch_aux` 主线候选。它的收益是稳定小幅收益，不应夸大为决定性突破；但 35k、65k、P0 三层验证方向一致，且 P0 frozen pilot_test 相对 Round1 fallback 在 MAE/MSE/regret 上同时改善，足以支持进入 full-scale validation planning。

period continuity 方向保留为后续支线。`top3fold_period_layout` 的连续性优势是真实诊断价值，但当前没有转化为 expanded/P0 主指标优势，因此不阻塞主线推进。

## 下一步方案

优先进入 `Visual Router V2 Round2 full-scale validation planning`。规划阶段只做方案设计，不直接启动 full-scale feature extraction、training 或 final eval。规划需明确 staged validation scale、shard-aware feature cache、batch/SQLite prediction lookup、不全量加载 116M prediction manifest、多 GPU 并行、frozen eval 协议，以及与 Stage 1 canonical migration 的边界。
