# spatial_panel_3view 与 TimeFuse 抽样子集对比核查

日志日期：2026-06-24 03:44:18 CST

## 目的

核查本地历史结果中是否存在 `spatial_panel_3view` 与 TimeFuse 在 65k 或 275k 抽样子集上的对比，并明确哪些是同一子集直接对比，哪些只是参考行。

## 背景

Visual Router V2 Round2 中 `spatial_panel_3view + film_mean_patch_aux` 经过 35k、65k expanded 和 P0/275k 三层验证。用户关注它是否曾与 TimeFuse 在 65k/275k 抽样子集上对比过，因此需要从 summary、comparison CSV 和 Round0 TimeFuse 输出中复核口径。

## 操作

1. 搜索 `spatial_panel_3view`、`TimeFuse`、`65k`、`275000`、`P0` 等关键词，定位 Round2 expanded validation、P0 spatial panel mainline 和 Round0 TimeFuse 结果。
2. 读取 `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation/round2_expanded_layout_selection_comparison_with_references.csv`，确认 65k expanded 中 TimeFuse 行的来源和 sample_count。
3. 读取 `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/round2_p0_spatial_selection_comparison_with_references.csv`、`round2_p0_spatial_test_summary.csv` 和 Round0 `round0_main_comparison.csv` / `round0_selection_comparison.csv`，复核 P0 pilot_selection 与 pilot_test 对比。
4. 读取 `experiment_summaries/visual_router_v2_round2/round2_mainline_recommendation.md` 和相关 summary，确认已有结论表述。

## 结果

1. 65k expanded validation 已完成，样本为 `round2_train_expanded=30000`、`round2_selection_expanded=10000`、`round2_diagnostic_balanced_expanded=10000`、`round2_test_expanded=15000`，总计 65000。`spatial_panel_3view` 在 selection/test_expanded 上均为 best：raw-soft MAE 分别为 `0.307233` 和 `0.394336`。
2. 65k expanded 的 comparison_with_references 中存在 `Round0 TimeFuse` 参考行，但来源是 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/round0_selection_comparison.csv`，sample_count 为 `30000` 的 `pilot_selection`，不是在 65k expanded 的 `round2_selection_expanded=10000` 或 `round2_test_expanded=15000` 上直接重算。
3. 因此 65k 中可引用的 TimeFuse 对比只能写作参考口径：`spatial_panel_3view` selection raw-soft MAE `0.307233` vs Round0 TimeFuse pilot_selection raw-soft MAE `0.317530`，delta `-0.010297`；但 MSE 为 `2.043914` vs `1.370167`，且 sample set/sample_count 不一致。
4. P0/275k spatial panel mainline 使用 pilot 协议样本：`pilot_train=150000`、`pilot_selection=30000`、`diagnostic_balanced=20000`、`pilot_test=75000`，total feature rows `275000`。这里与 Round0 TimeFuse 使用同一 pilot sample protocol，属于更接近直接横向的对比。
5. P0 pilot_selection：`spatial_panel_3view` raw-soft MAE/MSE/regret 为 `0.300227/1.269413/0.033496`，Round0 TimeFuse raw-soft 为 `0.317530/1.370167/0.050799`，MAE delta `-0.017303`。hard-top1 MAE 为 `0.318488` vs `0.334912`，delta `-0.016424`。
6. P0 frozen pilot_test：`spatial_panel_3view` raw-soft MAE/MSE/regret 为 `0.413558/182.771166/0.073273`，Round0 TimeFuse raw-soft 为 `0.535220/568.502401/0.194935`，MAE delta `-0.121662`。hard-top1 MAE 为 `0.428479` vs `0.547432`，delta `-0.118954`。

## 结论

有 275k/P0 pilot 协议上的 `spatial_panel_3view` 与 Round0 TimeFuse 对比，且 selection 与 frozen pilot_test 上 `spatial_panel_3view` 的 MAE 都优于 TimeFuse。65k expanded 上没有严格同一 65k 子集重算的 TimeFuse 对比，只有从 Round0 pilot_selection 引入的参考行；引用时必须说明 sample_set/sample_count 不一致。

## 下一步方案

如果后续需要正式写论文式对比，应避免把 65k expanded 的 TimeFuse 参考行写成同一子集直接对比；如要补齐 65k 直接比较，应在 `round2_selection_expanded` 和 `round2_test_expanded` 上复用相同 prediction/oracle 和 TimeFuse 权重口径重新生成 TimeFuse evaluation。
