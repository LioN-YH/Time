# Visual Router V2 Round2f P0 Spatial Panel Mainline Summary

生成时间：2026-06-22 08:17:24 CST

## 结论

- best_layout：`spatial_panel_3view`
- 后端固定：`film_mean_patch_aux`，即 `mean_patch_embedding -> visual hidden`，`revin_aux -> FiLM gamma/beta`。
- 选择口径只使用 `pilot_selection` raw-soft MAE mean；`diagnostic_balanced` 只诊断，`pilot_test` 只做 frozen validation。
- 下一步建议：建议升级 `spatial_panel_3view` 为 Round2 主线，并扩大到 P0/P2a 规模。

## 必答问题

1. 65k selection best：`spatial_panel_3view`，raw-soft MAE=0.300227，MSE=1.269413。
2. 65k test_expanded best：`spatial_panel_3view`，raw-soft MAE=0.413558，MSE=182.771166。
3. selection best 与 test best 是否一致：一致。
4. `spatial_panel_3view` 是否仍优于 `current_rgb_3view`：`spatial_panel_3view` 或 `current_rgb_3view` 缺少结果。
5. `top3fold_period_layout` 的 continuity / diagnostic 优势是否转化为 expanded 性能：`top3fold_period_layout` 或 `current_rgb_3view` 缺少结果。 diagnostic CrossFormer/PatchTST=diagnostic_balanced/oracle_model 缺少可比较 strata 结果。；seasonality=diagnostic_balanced/season_strength_cat 缺少可比较 strata 结果。。
6. 35k 结论是否在 65k 上稳定：稳定。35k screening 结论为 `spatial_panel_3view` 是 selection/test_small best，本轮以 65k selection/test 是否继续支持该 layout 判定。
7. seed stability / MSE tail / CrossFormer / PatchTST strata：seed stability=`spatial_panel_3view` 最稳定，selection MAE_std=0.000546，MSE_std=0.204500。 MSE tail=selection MSE 最低：`spatial_panel_3view`=1.269413；pilot_test MSE 最低：`spatial_panel_3view`=182.771166。 CrossFormer/PatchTST=diagnostic_balanced/oracle_model 缺少可比较 strata 结果。
8. 是否建议把 `spatial_panel_3view` 升级为 Round2 主线：建议。
9. 下一步：建议升级 `spatial_panel_3view` 为 Round2 主线，并扩大到 P0/P2a 规模。
10. latency 检查：未发现超过 median 1.5x 的明显过慢 layout。latency 排序：spatial_panel_3view=0.941 ms/sample。

## Selection

| sample_set | variant | method | seed_count | sample_count_per_seed | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_selection | spatial_panel_3view | spatial_panel_3view_raw_soft_fusion | 3 | 30000 | 0.300227 | 0.000546 | 1.269413 | 0.204500 | 0.033496 | 0.000546 | 0.549656 | 0.150012 | 1.142732 | 0.008502 | 0.710019 | 0.005282 | 0.498033 | 0.005323 |
| pilot_selection | spatial_panel_3view | spatial_panel_3view_hard_top1 | 3 | 30000 | 0.318488 | 0.000583 | 1.329941 | 0.208590 | 0.051756 | 0.000583 | 0.549656 | 0.150012 | 1.142732 | 0.008502 | 0.710019 | 0.005282 | 0.498033 | 0.005323 |

## Diagnostic Balanced

| sample_set | variant | method | seed_count | sample_count_per_seed | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diagnostic_balanced | spatial_panel_3view | spatial_panel_3view_raw_soft_fusion | 3 | 20000 | 0.345245 | 0.002725 | 1.410329 | 0.019554 | 0.039755 | 0.002725 | 0.457367 | 0.103027 | 1.067131 | 0.010050 | 0.663046 | 0.006245 | 0.538637 | 0.005577 |
| diagnostic_balanced | spatial_panel_3view | spatial_panel_3view_hard_top1 | 3 | 20000 | 0.370313 | 0.003907 | 1.469294 | 0.013604 | 0.064823 | 0.003907 | 0.457367 | 0.103027 | 1.067131 | 0.010050 | 0.663046 | 0.006245 | 0.538637 | 0.005577 |

## Frozen Test

| sample_set | variant | method | seed_count | sample_count_per_seed | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_test | spatial_panel_3view | spatial_panel_3view_raw_soft_fusion | 3 | 75000 | 0.413558 | 0.002209 | 182.771166 | 0.092882 | 0.073273 | 0.002209 | 0.521733 | 0.158075 | 1.105016 | 0.005121 | 0.686585 | 0.003182 | 0.518161 | 0.010963 |
| pilot_test | spatial_panel_3view | spatial_panel_3view_hard_top1 | 3 | 75000 | 0.428479 | 0.002479 | 182.957533 | 0.172605 | 0.088194 | 0.002479 | 0.521733 | 0.158075 | 1.105016 | 0.005121 | 0.686585 | 0.003182 | 0.518161 | 0.010963 |

## Delta Summary

| delta_name | status | sample_set | method_kind | left_variant | right_variant | left_MAE_mean | right_MAE_mean | delta_MAE_mean | left_MSE_mean | right_MSE_mean | delta_MSE_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spatial_panel_3view - current_rgb_3view | missing_reference |  |  |  |  |  |  |  |  |  |  |
| line_only - current_rgb_3view | missing_reference |  |  |  |  |  |  |  |  |  |  |
| line_difference_band - line_only | missing_reference |  |  |  |  |  |  |  |  |  |  |
| line_difference_band - current_rgb_3view | missing_reference |  |  |  |  |  |  |  |  |  |  |
| fft_absolute_energy - current_rgb_3view | missing_reference |  |  |  |  |  |  |  |  |  |  |
| top3fold_period_layout - current_rgb_3view | missing_reference |  |  |  |  |  |  |  |  |  |  |
| top3fold_period_layout - fft_absolute_energy | missing_reference |  |  |  |  |  |  |  |  |  |  |
| spatial_panel_3view - film_mean_patch_aux | ok | pilot_selection | raw_soft_fusion | spatial_panel_3view | film_mean_patch_aux | 0.300227 | 0.300393 | -0.000166 | 1.269413 | 1.289872 | -0.020458 |
| spatial_panel_3view - Round0 TimeFuse | ok | pilot_selection | raw_soft_fusion | spatial_panel_3view | Round0 TimeFuse | 0.300227 | 0.317530 | -0.017303 | 1.269413 | 1.370167 | -0.100753 |

## Reference-Inclusive Comparison

| stage | sample_set | variant | method | method_kind | seed_count | sample_count | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std | source_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round2f P0 spatial panel mainline | diagnostic_balanced | spatial_panel_3view | spatial_panel_3view_hard_top1 | hard_top1 | 3 | 20000 | 0.370313 | 0.003907 | 1.469294 | 0.013604 | 0.064823 | 0.003907 | 0.457367 | 0.103027 | 1.067131 | 0.010050 | 0.663046 | 0.006245 | 0.538637 | 0.005577 | /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/round2_p0_spatial_diagnostic_summary.csv |
| Round2f P0 spatial panel mainline | diagnostic_balanced | spatial_panel_3view | spatial_panel_3view_raw_soft_fusion | raw_soft_fusion | 3 | 20000 | 0.345245 | 0.002725 | 1.410329 | 0.019554 | 0.039755 | 0.002725 | 0.457367 | 0.103027 | 1.067131 | 0.010050 | 0.663046 | 0.006245 | 0.538637 | 0.005577 | /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/round2_p0_spatial_diagnostic_summary.csv |
| Round1 reference | pilot_selection | film_mean_patch_aux | film_mean_patch_aux_hard_top1 | hard_top1 | 3 | 30000 | 0.317828 | 0.000429 | 1.378600 | 0.155791 | 0.051097 | 0.000429 | 0.552167 | 0.148492 | 1.135975 | 0.020092 | 0.705821 | 0.012484 | 0.502353 | 0.006659 | /data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round1_film/round1_film_selection_comparison.csv |
| Round1 reference | pilot_selection | film_cls_mean_concat_aux | film_cls_mean_concat_aux_hard_top1 | hard_top1 | 3 | 30000 | 0.318243 | 0.000603 | 1.364084 | 0.166895 | 0.051512 | 0.000603 | 0.465378 | 0.147915 | 1.129449 | 0.028893 | 0.701766 | 0.017952 | 0.505618 | 0.013839 | /data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round1_film/round1_film_selection_comparison.csv |
| Round2f P0 spatial panel mainline | pilot_selection | spatial_panel_3view | spatial_panel_3view_hard_top1 | hard_top1 | 3 | 30000 | 0.318488 | 0.000583 | 1.329941 | 0.208590 | 0.051756 | 0.000583 | 0.549656 | 0.150012 | 1.142732 | 0.008502 | 0.710019 | 0.005282 | 0.498033 | 0.005323 | /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/round2_p0_spatial_selection_layout_only.csv |
| Round1 reference | pilot_selection | visual_cls_mean_concat | visual_cls_mean_concat_hard_top1 | hard_top1 | 3 | 30000 | 0.320836 | 0.003389 | 1.391806 | 0.176839 | 0.054105 | 0.003389 | 0.549878 | 0.159907 | 1.112957 | 0.023586 | 0.691519 | 0.014655 | 0.523813 | 0.011781 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling/visual_pooling_selection_comparison.csv |
| Round0 | pilot_selection | Round0 TimeFuse | timefuse_hard_top1 | hard_top1 | 1 | 30000 | 0.334912 | 0.000000 | 1.429099 | 0.000000 | 0.068181 | 0.000000 | 0.585767 | 0.000000 | 0.855693 | 0.000000 | 0.531672 | 0.000000 | 0.618703 | 0.000000 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/round0_selection_comparison.csv |
| Round0 | pilot_selection | Round0 original Visual | visual_router_hard_top1 | hard_top1 | 1 | 30000 | 0.356267 | 0.000000 | 1.367826 | 0.000000 | 0.089536 | 0.000000 | 0.579200 | 0.000000 | 1.292436 | 0.000000 | 0.803035 | 0.000000 | 0.439655 | 0.000000 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/round0_selection_comparison.csv |
| Round0 | pilot_selection | oracle_top1 | oracle_top1 | oracle | 1 | 30000 | 0.266731 | 0.000000 |  | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |  | 0.000000 |  | 0.000000 |  | 0.000000 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/round0_selection_comparison.csv |
| Round2f P0 spatial panel mainline | pilot_selection | spatial_panel_3view | spatial_panel_3view_raw_soft_fusion | raw_soft_fusion | 3 | 30000 | 0.300227 | 0.000546 | 1.269413 | 0.204500 | 0.033496 | 0.000546 | 0.549656 | 0.150012 | 1.142732 | 0.008502 | 0.710019 | 0.005282 | 0.498033 | 0.005323 | /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/round2_p0_spatial_selection_layout_only.csv |
| Round1 reference | pilot_selection | film_mean_patch_aux | film_mean_patch_aux_raw_soft_fusion | raw_soft_fusion | 3 | 30000 | 0.300393 | 0.000542 | 1.289872 | 0.164650 | 0.033662 | 0.000542 | 0.552167 | 0.148492 | 1.135975 | 0.020092 | 0.705821 | 0.012484 | 0.502353 | 0.006659 | /data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round1_film/round1_film_selection_comparison.csv |
| Round1 reference | pilot_selection | film_cls_mean_concat_aux | film_cls_mean_concat_aux_raw_soft_fusion | raw_soft_fusion | 3 | 30000 | 0.300486 | 0.000859 | 1.313162 | 0.168122 | 0.033754 | 0.000859 | 0.465378 | 0.147915 | 1.129449 | 0.028893 | 0.701766 | 0.017952 | 0.505618 | 0.013839 | /data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round1_film/round1_film_selection_comparison.csv |
| Round1 reference | pilot_selection | visual_cls_mean_concat | visual_cls_mean_concat_raw_soft_fusion | raw_soft_fusion | 3 | 30000 | 0.302220 | 0.003929 | 1.217317 | 0.118708 | 0.035489 | 0.003929 | 0.549878 | 0.159907 | 1.112957 | 0.023586 | 0.691519 | 0.014655 | 0.523813 | 0.011781 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling/visual_pooling_selection_comparison.csv |
| Round0 | pilot_selection | Round0 TimeFuse | timefuse_raw_soft_fusion | raw_soft_fusion | 1 | 30000 | 0.317530 | 0.000000 | 1.370167 | 0.000000 | 0.050799 | 0.000000 | 0.585767 | 0.000000 | 0.855693 | 0.000000 | 0.531672 | 0.000000 | 0.618703 | 0.000000 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/round0_selection_comparison.csv |
| Round0 | pilot_selection | Round0 original Visual | visual_router_raw_soft_fusion | raw_soft_fusion | 1 | 30000 | 0.334069 | 0.000000 | 1.181831 | 0.000000 | 0.067337 | 0.000000 | 0.579200 | 0.000000 | 1.292436 | 0.000000 | 0.803035 | 0.000000 | 0.439655 | 0.000000 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/round0_selection_comparison.csv |
| Round0 | pilot_selection | global_best_single | global_best_single | single | 1 | 30000 | 0.357820 | 0.000000 |  | 0.000000 | 0.091089 | 0.000000 | 0.309667 | 0.000000 |  | 0.000000 |  | 0.000000 |  | 0.000000 | /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/round0_selection_comparison.csv |
| Round2f P0 spatial panel mainline | pilot_test | spatial_panel_3view | spatial_panel_3view_hard_top1 | hard_top1 | 3 | 75000 | 0.428479 | 0.002479 | 182.957533 | 0.172605 | 0.088194 | 0.002479 | 0.521733 | 0.158075 | 1.105016 | 0.005121 | 0.686585 | 0.003182 | 0.518161 | 0.010963 | /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/round2_p0_spatial_test_summary.csv |
| Round2f P0 spatial panel mainline | pilot_test | spatial_panel_3view | spatial_panel_3view_raw_soft_fusion | raw_soft_fusion | 3 | 75000 | 0.413558 | 0.002209 | 182.771166 | 0.092882 | 0.073273 | 0.002209 | 0.521733 | 0.158075 | 1.105016 | 0.005121 | 0.686585 | 0.003182 | 0.518161 | 0.010963 | /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/round2_p0_spatial_test_summary.csv |
