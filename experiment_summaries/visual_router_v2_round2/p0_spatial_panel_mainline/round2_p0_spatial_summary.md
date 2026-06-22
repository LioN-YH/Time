# Visual Router V2 Round2f P0 Spatial Panel Mainline Summary

生成时间：2026-06-22 08:17:25 CST

## 结论

- 本轮只验证 `spatial_panel_3view + film_mean_patch_aux`，选择口径只使用 `pilot_selection` raw-soft MAE mean。
- pilot_selection raw-soft：MAE=0.300227，MSE=1.269413，regret=0.033496。
- frozen pilot_test raw-soft：MAE=0.413558，MSE=182.771166，regret=0.073273。
- 是否超过 Round1 当前 best：P0 spatial pilot_test raw-soft MAE=0.413558 vs Round1 film_mean_patch_aux=0.417824，delta=-0.004266（改善）；MSE delta=-0.582819（改善）；regret delta=-0.004266（改善）。
- 下一步建议：full-scale validation。

## 必答问题

1. P0 pilot_selection 表现：MAE=0.300227，MSE=1.269413，regret=0.033496。
2. frozen pilot_test 是否优于 Round1 `film_mean_patch_aux`：P0 spatial pilot_test raw-soft MAE=0.413558 vs Round1 film_mean_patch_aux=0.417824，delta=-0.004266（改善）；MSE delta=-0.582819（改善）；regret delta=-0.004266（改善）。
3. 是否优于 Round0 TimeFuse / original Visual：pilot_selection raw-soft MAE delta vs Round0 TimeFuse=-0.017303（改善）。 pilot_test raw-soft MAE delta vs Round1 visual_cls_mean_concat=-0.029504（改善）。 pilot_selection raw-soft MAE delta vs Round0 original Visual=-0.033842（改善）。
4. MAE / MSE / regret 是否同时改善：见第 2 条和 `round2_p0_spatial_delta_summary.csv`。
5. seed stability 是否保持：seed stability：pilot_selection raw-soft MAE_std=0.000546，MSE_std=0.204500。
6. CrossFormer / PatchTST / ES / DLinear strata 是否改善：frozen pilot_test 按 oracle_model 聚合后相对 Round1 `film_mean_patch_aux` 均改善；MAE delta 分别为 CrossFormer=-0.009317、PatchTST=-0.008063、ES=-0.000946、DLinear=-0.003066，且四者 MSE delta 也均为负。
7. selected_model ratio 是否健康：pilot_test raw-soft selected_model ratio(seed mean) 为 CrossFormer=0.059、DLinear=0.111、ES=0.292、NaiveForecaster=0.213、PatchTST=0.325；未出现单专家极端塌缩。
8. 是否建议把 spatial panel 作为 Visual Router V2 当前主线：建议。P0 pilot_selection 略优于 Round1 `film_mean_patch_aux`，frozen pilot_test 的 MAE/MSE/regret 同时优于 Round1 当前 best，seed MAE_std=0.002209/0.000546(test/selection) 可接受。
9. 下一步：full-scale validation。

## 产物

- 输出目录：`/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline`
- 轻量 summary 目录：`/home/shiyuhong/Time-visual-router-v2/experiment_summaries/visual_router_v2_round2/p0_spatial_panel_mainline`
- metadata：`round2_p0_spatial_metadata.json`
- final test summary：`round2_p0_spatial_final_test_summary.csv`

## Metadata 摘要

- devices_requested：`cuda:0,cuda:1,cuda:2,cuda:3`
- devices_used：`['cuda:0', 'cuda:1', 'cuda:2']`
- backend_style：`film_mean_patch_aux`
- used_frozen_test_for_selection：`False`
