# Visual Router V2 Round2 panel gating 35k screening

生成时间：2026-06-22 20:38:59 CST

## 结论

- selection best：`film_panel_lowrank_aux`。
- 升级判断：Drop / defer panel-aware path。
- 本结果只覆盖 Round2 35k small screening，不影响并行 full-scale 主线；full-scale 仍可继续使用 `spatial_panel_3view + film_mean_patch_aux`。
- 本轮不重新跑 ViT，不保存 pseudo image tensor，不做 65k/P0/full-scale；`round2_test_small` 未用于训练、调参、选择 variant、选择 seed 或选择 epoch。

## Selection Raw-Soft / Hard

| sample_set | variant | method | seed_count | sample_count_per_seed | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| round2_selection_small | film_panel_lowrank_aux | film_panel_lowrank_aux_raw_soft_fusion | 3 | 5000 | 0.308605 | 0.007982 | 3.153192 | 0.688477 | 0.045155 | 0.007982 | 0.541867 | 0.147110 | 1.130508 | 0.028563 | 0.702424 | 0.017747 | 0.503382 | 0.019453 |
| round2_selection_small | film_mean_patch_aux | film_mean_patch_aux_raw_soft_fusion | 3 | 5000 | 0.310385 | 0.008199 | 3.329199 | 0.573350 | 0.046935 | 0.008199 | 0.538000 | 0.142235 | 1.135332 | 0.010910 | 0.705421 | 0.006779 | 0.515005 | 0.014563 |
| round2_selection_small | film_panel_gated_mean_aux | film_panel_gated_mean_aux_raw_soft_fusion | 3 | 5000 | 0.314789 | 0.002958 | 3.651150 | 0.289802 | 0.051340 | 0.002958 | 0.541400 | 0.149996 | 1.155383 | 0.010834 | 0.717880 | 0.006732 | 0.494802 | 0.008160 |
| round2_selection_small | film_mean_patch_aux | film_mean_patch_aux_hard_top1 | 3 | 5000 | 0.328295 | 0.008512 | 3.450927 | 0.576780 | 0.064845 | 0.008512 | 0.538000 | 0.142235 | 1.135332 | 0.010910 | 0.705421 | 0.006779 | 0.515005 | 0.014563 |
| round2_selection_small | film_panel_lowrank_aux | film_panel_lowrank_aux_hard_top1 | 3 | 5000 | 0.328300 | 0.009876 | 3.553516 | 0.900581 | 0.064851 | 0.009876 | 0.541867 | 0.147110 | 1.130508 | 0.028563 | 0.702424 | 0.017747 | 0.503382 | 0.019453 |
| round2_selection_small | film_panel_gated_mean_aux | film_panel_gated_mean_aux_hard_top1 | 3 | 5000 | 0.333933 | 0.001295 | 4.051177 | 0.052093 | 0.070483 | 0.001295 | 0.541400 | 0.149996 | 1.155383 | 0.010834 | 0.717880 | 0.006732 | 0.494802 | 0.008160 |

## Diagnostic Balanced

| sample_set | variant | method | seed_count | sample_count_per_seed | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | film_panel_lowrank_aux_raw_soft_fusion | 3 | 5000 | 0.361316 | 0.009896 | 1.488833 | 0.720438 | 0.055223 | 0.009896 | 0.453333 | 0.071520 | 1.048663 | 0.036166 | 0.651571 | 0.022471 | 0.548995 | 0.017492 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | film_panel_gated_mean_aux_raw_soft_fusion | 3 | 5000 | 0.369126 | 0.003514 | 2.482110 | 0.483008 | 0.063033 | 0.003514 | 0.458400 | 0.087022 | 1.076061 | 0.015898 | 0.668594 | 0.009878 | 0.538608 | 0.009257 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | film_mean_patch_aux_raw_soft_fusion | 3 | 5000 | 0.370516 | 0.005574 | 3.388182 | 0.944727 | 0.064422 | 0.005574 | 0.449800 | 0.066598 | 1.054758 | 0.012455 | 0.655358 | 0.007739 | 0.555304 | 0.008914 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | film_panel_lowrank_aux_hard_top1 | 3 | 5000 | 0.386403 | 0.012429 | 1.807692 | 0.806439 | 0.080310 | 0.012429 | 0.453333 | 0.071520 | 1.048663 | 0.036166 | 0.651571 | 0.022471 | 0.548995 | 0.017492 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | film_panel_gated_mean_aux_hard_top1 | 3 | 5000 | 0.394692 | 0.002352 | 3.039984 | 1.064280 | 0.088598 | 0.002352 | 0.458400 | 0.087022 | 1.076061 | 0.015898 | 0.668594 | 0.009878 | 0.538608 | 0.009257 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | film_mean_patch_aux_hard_top1 | 3 | 5000 | 0.395243 | 0.006540 | 3.580370 | 0.969172 | 0.089149 | 0.006540 | 0.449800 | 0.066598 | 1.054758 | 0.012455 | 0.655358 | 0.007739 | 0.555304 | 0.008914 |

## Frozen Test Small

| sample_set | variant | method | seed_count | sample_count_per_seed | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| round2_test_small | film_panel_lowrank_aux | film_panel_lowrank_aux_raw_soft_fusion | 3 | 5000 | 0.397761 | 0.001878 | 3.481281 | 0.015328 | 0.064412 | 0.001878 | 0.518800 | 0.162191 | 1.111045 | 0.043629 | 0.690331 | 0.027108 | 0.521709 | 0.017853 |
| round2_test_small | film_panel_gated_mean_aux | film_panel_gated_mean_aux_raw_soft_fusion | 3 | 5000 | 0.398413 | 0.000501 | 3.479709 | 0.010087 | 0.065064 | 0.000501 | 0.519333 | 0.158257 | 1.121942 | 0.050122 | 0.697101 | 0.031142 | 0.521551 | 0.023808 |
| round2_test_small | film_mean_patch_aux | film_mean_patch_aux_raw_soft_fusion | 3 | 5000 | 0.398598 | 0.001161 | 3.484102 | 0.009068 | 0.065249 | 0.001161 | 0.518667 | 0.161140 | 1.086936 | 0.037663 | 0.675351 | 0.023401 | 0.549078 | 0.012821 |
| round2_test_small | film_panel_lowrank_aux | film_panel_lowrank_aux_hard_top1 | 3 | 5000 | 0.411200 | 0.000420 | 3.572681 | 0.007930 | 0.077851 | 0.000420 | 0.518800 | 0.162191 | 1.111045 | 0.043629 | 0.690331 | 0.027108 | 0.521709 | 0.017853 |
| round2_test_small | film_panel_gated_mean_aux | film_panel_gated_mean_aux_hard_top1 | 3 | 5000 | 0.411244 | 0.001341 | 3.555384 | 0.035575 | 0.077895 | 0.001341 | 0.519333 | 0.158257 | 1.121942 | 0.050122 | 0.697101 | 0.031142 | 0.521551 | 0.023808 |
| round2_test_small | film_mean_patch_aux | film_mean_patch_aux_hard_top1 | 3 | 5000 | 0.411577 | 0.000517 | 3.567902 | 0.022592 | 0.078228 | 0.000517 | 0.518667 | 0.161140 | 1.086936 | 0.037663 | 0.675351 | 0.023401 | 0.549078 | 0.012821 |

## Selection Key Strata Delta

负数表示好于 baseline `film_mean_patch_aux`。

| sample_set | stratum_column | stratum_value | variant | sample_count | MAE_mean | MAE_delta_vs_baseline | MSE_delta_vs_baseline | regret_delta_vs_baseline | oracle_label_accuracy_mean | mean_max_weight_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| round2_selection_small | error_gap_quantile | q5 | film_panel_gated_mean_aux | 811.000000 | 0.641852 | 0.030381 | 1.986404 | 0.030381 | 0.688861 | 0.675145 |
| round2_selection_small | error_gap_quantile | q5 | film_panel_lowrank_aux | 811.000000 | 0.607568 | -0.003904 | -1.083095 | -0.003904 | 0.688450 | 0.695189 |
| round2_selection_small | oracle_model | PatchTST | film_panel_gated_mean_aux | 814.000000 | 0.481476 | 0.035485 | 2.102680 | 0.035485 | 0.393120 | 0.561415 |
| round2_selection_small | oracle_model | PatchTST | film_panel_lowrank_aux | 814.000000 | 0.448854 | 0.002864 | -0.926598 | 0.002864 | 0.424242 | 0.575249 |
| round2_selection_small | oracle_model | CrossFormer | film_panel_gated_mean_aux | 479.000000 | 0.481063 | -0.000005 | 0.000325 | -0.000005 | 0.179541 | 0.545993 |
| round2_selection_small | oracle_model | CrossFormer | film_panel_lowrank_aux | 479.000000 | 0.482785 | 0.001717 | 0.003635 | 0.001717 | 0.158664 | 0.553486 |
| round2_selection_small | oracle_model | DLinear | film_panel_gated_mean_aux | 1547.000000 | 0.374473 | -0.000650 | 0.001200 | -0.000650 | 0.758242 | 0.612226 |
| round2_selection_small | oracle_model | DLinear | film_panel_lowrank_aux | 1547.000000 | 0.371924 | -0.003198 | -0.004855 | -0.003198 | 0.748546 | 0.629670 |
| round2_selection_small | oracle_model | ES | film_panel_gated_mean_aux | 1774.000000 | 0.139435 | -0.001393 | -0.057719 | -0.001393 | 0.551297 | 0.313247 |
| round2_selection_small | oracle_model | ES | film_panel_lowrank_aux | 1774.000000 | 0.139181 | -0.001647 | -0.067363 | -0.001647 | 0.558437 | 0.312172 |
| round2_selection_small | group_name | LOW_LOW_HIGH | film_panel_gated_mean_aux | 210.000000 | 0.153471 | 0.001938 | 0.004040 | 0.001938 | 0.601587 | 0.377263 |
| round2_selection_small | group_name | LOW_LOW_HIGH | film_panel_lowrank_aux | 210.000000 | 0.150777 | -0.000756 | -0.000142 | -0.000756 | 0.596825 | 0.378311 |

## Selected Model Ratio

| sample_set | variant | seed | method | selected_model | count | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | CrossFormer | 550 | 0.110000 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | DLinear | 1804 | 0.360800 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | ES | 1249 | 0.249800 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | NaiveForecaster | 451 | 0.090200 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | PatchTST | 946 | 0.189200 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | CrossFormer | 550 | 0.110000 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | DLinear | 1804 | 0.360800 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | ES | 1249 | 0.249800 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | NaiveForecaster | 451 | 0.090200 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | PatchTST | 946 | 0.189200 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | CrossFormer | 492 | 0.098400 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | DLinear | 1984 | 0.396800 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | ES | 205 | 0.041000 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | NaiveForecaster | 1328 | 0.265600 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | PatchTST | 991 | 0.198200 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | CrossFormer | 492 | 0.098400 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | DLinear | 1984 | 0.396800 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | ES | 205 | 0.041000 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | NaiveForecaster | 1328 | 0.265600 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | PatchTST | 991 | 0.198200 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | CrossFormer | 319 | 0.063800 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | DLinear | 2253 | 0.450600 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | ES | 1311 | 0.262200 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | NaiveForecaster | 295 | 0.059000 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | PatchTST | 822 | 0.164400 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | CrossFormer | 319 | 0.063800 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | DLinear | 2253 | 0.450600 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | ES | 1311 | 0.262200 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | NaiveForecaster | 295 | 0.059000 |
| round2_diagnostic_balanced_small | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | PatchTST | 822 | 0.164400 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_hard_top1 | CrossFormer | 479 | 0.095800 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_hard_top1 | DLinear | 2087 | 0.417400 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_hard_top1 | ES | 405 | 0.081000 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_hard_top1 | NaiveForecaster | 506 | 0.101200 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_hard_top1 | PatchTST | 1523 | 0.304600 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_raw_soft_fusion | CrossFormer | 479 | 0.095800 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_raw_soft_fusion | DLinear | 2087 | 0.417400 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_raw_soft_fusion | ES | 405 | 0.081000 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_raw_soft_fusion | NaiveForecaster | 506 | 0.101200 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_raw_soft_fusion | PatchTST | 1523 | 0.304600 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_hard_top1 | CrossFormer | 452 | 0.090400 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_hard_top1 | DLinear | 1863 | 0.372600 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_hard_top1 | ES | 1086 | 0.217200 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_hard_top1 | NaiveForecaster | 560 | 0.112000 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_hard_top1 | PatchTST | 1039 | 0.207800 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_raw_soft_fusion | CrossFormer | 452 | 0.090400 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_raw_soft_fusion | DLinear | 1863 | 0.372600 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_raw_soft_fusion | ES | 1086 | 0.217200 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_raw_soft_fusion | NaiveForecaster | 560 | 0.112000 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_raw_soft_fusion | PatchTST | 1039 | 0.207800 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_hard_top1 | CrossFormer | 311 | 0.062200 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_hard_top1 | DLinear | 2228 | 0.445600 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_hard_top1 | ES | 1186 | 0.237200 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_hard_top1 | NaiveForecaster | 403 | 0.080600 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_hard_top1 | PatchTST | 872 | 0.174400 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_raw_soft_fusion | CrossFormer | 311 | 0.062200 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_raw_soft_fusion | DLinear | 2228 | 0.445600 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_raw_soft_fusion | ES | 1186 | 0.237200 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_raw_soft_fusion | NaiveForecaster | 403 | 0.080600 |
| round2_diagnostic_balanced_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_raw_soft_fusion | PatchTST | 872 | 0.174400 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_hard_top1 | CrossFormer | 579 | 0.115800 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_hard_top1 | DLinear | 2607 | 0.521400 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_hard_top1 | ES | 357 | 0.071400 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_hard_top1 | NaiveForecaster | 555 | 0.111000 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_hard_top1 | PatchTST | 902 | 0.180400 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_raw_soft_fusion | CrossFormer | 579 | 0.115800 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_raw_soft_fusion | DLinear | 2607 | 0.521400 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_raw_soft_fusion | ES | 357 | 0.071400 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_raw_soft_fusion | NaiveForecaster | 555 | 0.111000 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_raw_soft_fusion | PatchTST | 902 | 0.180400 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_hard_top1 | CrossFormer | 414 | 0.082800 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_hard_top1 | DLinear | 1884 | 0.376800 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_hard_top1 | ES | 1128 | 0.225600 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_hard_top1 | NaiveForecaster | 437 | 0.087400 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_hard_top1 | PatchTST | 1137 | 0.227400 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_raw_soft_fusion | CrossFormer | 414 | 0.082800 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_raw_soft_fusion | DLinear | 1884 | 0.376800 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_raw_soft_fusion | ES | 1128 | 0.225600 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_raw_soft_fusion | NaiveForecaster | 437 | 0.087400 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_raw_soft_fusion | PatchTST | 1137 | 0.227400 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_hard_top1 | CrossFormer | 238 | 0.047600 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_hard_top1 | DLinear | 2265 | 0.453000 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_hard_top1 | ES | 1295 | 0.259000 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_hard_top1 | NaiveForecaster | 402 | 0.080400 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_hard_top1 | PatchTST | 800 | 0.160000 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_raw_soft_fusion | CrossFormer | 238 | 0.047600 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_raw_soft_fusion | DLinear | 2265 | 0.453000 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_raw_soft_fusion | ES | 1295 | 0.259000 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_raw_soft_fusion | NaiveForecaster | 402 | 0.080400 |
| round2_diagnostic_balanced_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_raw_soft_fusion | PatchTST | 800 | 0.160000 |
| round2_selection_small | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | CrossFormer | 439 | 0.087800 |
| round2_selection_small | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | DLinear | 1865 | 0.373000 |
| round2_selection_small | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | ES | 1680 | 0.336000 |
| round2_selection_small | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | NaiveForecaster | 231 | 0.046200 |
| round2_selection_small | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | PatchTST | 785 | 0.157000 |
| round2_selection_small | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | CrossFormer | 439 | 0.087800 |
| round2_selection_small | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | DLinear | 1865 | 0.373000 |
| round2_selection_small | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | ES | 1680 | 0.336000 |
| round2_selection_small | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | NaiveForecaster | 231 | 0.046200 |
| round2_selection_small | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | PatchTST | 785 | 0.157000 |
| round2_selection_small | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | CrossFormer | 357 | 0.071400 |
| round2_selection_small | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | DLinear | 1991 | 0.398200 |
| round2_selection_small | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | ES | 160 | 0.032000 |
| round2_selection_small | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | NaiveForecaster | 1636 | 0.327200 |
| round2_selection_small | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | PatchTST | 856 | 0.171200 |
| round2_selection_small | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | CrossFormer | 357 | 0.071400 |
| round2_selection_small | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | DLinear | 1991 | 0.398200 |
| round2_selection_small | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | ES | 160 | 0.032000 |
| round2_selection_small | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | NaiveForecaster | 1636 | 0.327200 |
| round2_selection_small | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | PatchTST | 856 | 0.171200 |
| round2_selection_small | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | CrossFormer | 251 | 0.050200 |
| round2_selection_small | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | DLinear | 2207 | 0.441400 |
| round2_selection_small | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | ES | 1733 | 0.346600 |
| round2_selection_small | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | NaiveForecaster | 135 | 0.027000 |
| round2_selection_small | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | PatchTST | 674 | 0.134800 |
| round2_selection_small | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | CrossFormer | 251 | 0.050200 |
| round2_selection_small | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | DLinear | 2207 | 0.441400 |
| round2_selection_small | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | ES | 1733 | 0.346600 |
| round2_selection_small | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | NaiveForecaster | 135 | 0.027000 |
| round2_selection_small | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | PatchTST | 674 | 0.134800 |
| round2_selection_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_hard_top1 | CrossFormer | 362 | 0.072400 |
| round2_selection_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_hard_top1 | DLinear | 2132 | 0.426400 |
| round2_selection_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_hard_top1 | ES | 296 | 0.059200 |
| round2_selection_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_hard_top1 | NaiveForecaster | 277 | 0.055400 |
| round2_selection_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_hard_top1 | PatchTST | 1933 | 0.386600 |
| round2_selection_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_raw_soft_fusion | CrossFormer | 362 | 0.072400 |
| round2_selection_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_raw_soft_fusion | DLinear | 2132 | 0.426400 |
| round2_selection_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_raw_soft_fusion | ES | 296 | 0.059200 |
| round2_selection_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_raw_soft_fusion | NaiveForecaster | 277 | 0.055400 |
| round2_selection_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_raw_soft_fusion | PatchTST | 1933 | 0.386600 |
| round2_selection_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_hard_top1 | CrossFormer | 371 | 0.074200 |
| round2_selection_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_hard_top1 | DLinear | 1921 | 0.384200 |
| round2_selection_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_hard_top1 | ES | 1556 | 0.311200 |
| round2_selection_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_hard_top1 | NaiveForecaster | 289 | 0.057800 |
| round2_selection_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_hard_top1 | PatchTST | 863 | 0.172600 |
| round2_selection_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_raw_soft_fusion | CrossFormer | 371 | 0.074200 |
| round2_selection_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_raw_soft_fusion | DLinear | 1921 | 0.384200 |
| round2_selection_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_raw_soft_fusion | ES | 1556 | 0.311200 |
| round2_selection_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_raw_soft_fusion | NaiveForecaster | 289 | 0.057800 |
| round2_selection_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_raw_soft_fusion | PatchTST | 863 | 0.172600 |
| round2_selection_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_hard_top1 | CrossFormer | 240 | 0.048000 |
| round2_selection_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_hard_top1 | DLinear | 2233 | 0.446600 |
| round2_selection_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_hard_top1 | ES | 1623 | 0.324600 |
| round2_selection_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_hard_top1 | NaiveForecaster | 190 | 0.038000 |
| round2_selection_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_hard_top1 | PatchTST | 714 | 0.142800 |
| round2_selection_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_raw_soft_fusion | CrossFormer | 240 | 0.048000 |
| round2_selection_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_raw_soft_fusion | DLinear | 2233 | 0.446600 |
| round2_selection_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_raw_soft_fusion | ES | 1623 | 0.324600 |
| round2_selection_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_raw_soft_fusion | NaiveForecaster | 190 | 0.038000 |
| round2_selection_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_raw_soft_fusion | PatchTST | 714 | 0.142800 |
| round2_selection_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_hard_top1 | CrossFormer | 428 | 0.085600 |
| round2_selection_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_hard_top1 | DLinear | 3232 | 0.646400 |
| round2_selection_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_hard_top1 | ES | 280 | 0.056000 |
| round2_selection_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_hard_top1 | NaiveForecaster | 296 | 0.059200 |
| round2_selection_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_hard_top1 | PatchTST | 764 | 0.152800 |
| round2_selection_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_raw_soft_fusion | CrossFormer | 428 | 0.085600 |
| round2_selection_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_raw_soft_fusion | DLinear | 3232 | 0.646400 |
| round2_selection_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_raw_soft_fusion | ES | 280 | 0.056000 |
| round2_selection_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_raw_soft_fusion | NaiveForecaster | 296 | 0.059200 |
| round2_selection_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_raw_soft_fusion | PatchTST | 764 | 0.152800 |
| round2_selection_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_hard_top1 | CrossFormer | 309 | 0.061800 |
| round2_selection_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_hard_top1 | DLinear | 1891 | 0.378200 |
| round2_selection_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_hard_top1 | ES | 1618 | 0.323600 |
| round2_selection_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_hard_top1 | NaiveForecaster | 218 | 0.043600 |
| round2_selection_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_hard_top1 | PatchTST | 964 | 0.192800 |
| round2_selection_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_raw_soft_fusion | CrossFormer | 309 | 0.061800 |
| round2_selection_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_raw_soft_fusion | DLinear | 1891 | 0.378200 |
| round2_selection_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_raw_soft_fusion | ES | 1618 | 0.323600 |
| round2_selection_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_raw_soft_fusion | NaiveForecaster | 218 | 0.043600 |
| round2_selection_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_raw_soft_fusion | PatchTST | 964 | 0.192800 |
| round2_selection_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_hard_top1 | CrossFormer | 192 | 0.038400 |
| round2_selection_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_hard_top1 | DLinear | 2226 | 0.445200 |
| round2_selection_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_hard_top1 | ES | 1713 | 0.342600 |
| round2_selection_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_hard_top1 | NaiveForecaster | 205 | 0.041000 |
| round2_selection_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_hard_top1 | PatchTST | 664 | 0.132800 |
| round2_selection_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_raw_soft_fusion | CrossFormer | 192 | 0.038400 |
| round2_selection_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_raw_soft_fusion | DLinear | 2226 | 0.445200 |
| round2_selection_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_raw_soft_fusion | ES | 1713 | 0.342600 |
| round2_selection_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_raw_soft_fusion | NaiveForecaster | 205 | 0.041000 |
| round2_selection_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_raw_soft_fusion | PatchTST | 664 | 0.132800 |
| round2_test_small | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | CrossFormer | 250 | 0.050000 |
| round2_test_small | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | DLinear | 354 | 0.070800 |
| round2_test_small | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | ES | 1999 | 0.399800 |
| round2_test_small | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | NaiveForecaster | 564 | 0.112800 |
| round2_test_small | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | PatchTST | 1833 | 0.366600 |
| round2_test_small | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | CrossFormer | 250 | 0.050000 |
| round2_test_small | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | DLinear | 354 | 0.070800 |
| round2_test_small | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | ES | 1999 | 0.399800 |
| round2_test_small | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | NaiveForecaster | 564 | 0.112800 |
| round2_test_small | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | PatchTST | 1833 | 0.366600 |
| round2_test_small | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | CrossFormer | 217 | 0.043400 |
| round2_test_small | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | DLinear | 435 | 0.087000 |
| round2_test_small | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | ES | 345 | 0.069000 |
| round2_test_small | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | NaiveForecaster | 2155 | 0.431000 |
| round2_test_small | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | PatchTST | 1848 | 0.369600 |
| round2_test_small | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | CrossFormer | 217 | 0.043400 |
| round2_test_small | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | DLinear | 435 | 0.087000 |
| round2_test_small | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | ES | 345 | 0.069000 |
| round2_test_small | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | NaiveForecaster | 2155 | 0.431000 |
| round2_test_small | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | PatchTST | 1848 | 0.369600 |
| round2_test_small | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | CrossFormer | 290 | 0.058000 |
| round2_test_small | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | DLinear | 570 | 0.114000 |
| round2_test_small | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | ES | 2145 | 0.429000 |
| round2_test_small | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | NaiveForecaster | 443 | 0.088600 |
| round2_test_small | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | PatchTST | 1552 | 0.310400 |
| round2_test_small | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | CrossFormer | 290 | 0.058000 |
| round2_test_small | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | DLinear | 570 | 0.114000 |
| round2_test_small | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | ES | 2145 | 0.429000 |
| round2_test_small | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | NaiveForecaster | 443 | 0.088600 |
| round2_test_small | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | PatchTST | 1552 | 0.310400 |
| round2_test_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_hard_top1 | CrossFormer | 238 | 0.047600 |
| round2_test_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_hard_top1 | DLinear | 512 | 0.102400 |
| round2_test_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_hard_top1 | ES | 531 | 0.106200 |
| round2_test_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_hard_top1 | NaiveForecaster | 568 | 0.113600 |
| round2_test_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_hard_top1 | PatchTST | 3151 | 0.630200 |
| round2_test_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_raw_soft_fusion | CrossFormer | 238 | 0.047600 |
| round2_test_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_raw_soft_fusion | DLinear | 512 | 0.102400 |
| round2_test_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_raw_soft_fusion | ES | 531 | 0.106200 |
| round2_test_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_raw_soft_fusion | NaiveForecaster | 568 | 0.113600 |
| round2_test_small | film_panel_gated_mean_aux | 16 | film_panel_gated_mean_aux_raw_soft_fusion | PatchTST | 3151 | 0.630200 |
| round2_test_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_hard_top1 | CrossFormer | 225 | 0.045000 |
| round2_test_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_hard_top1 | DLinear | 466 | 0.093200 |
| round2_test_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_hard_top1 | ES | 1947 | 0.389400 |
| round2_test_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_hard_top1 | NaiveForecaster | 618 | 0.123600 |
| round2_test_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_hard_top1 | PatchTST | 1744 | 0.348800 |
| round2_test_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_raw_soft_fusion | CrossFormer | 225 | 0.045000 |
| round2_test_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_raw_soft_fusion | DLinear | 466 | 0.093200 |
| round2_test_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_raw_soft_fusion | ES | 1947 | 0.389400 |
| round2_test_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_raw_soft_fusion | NaiveForecaster | 618 | 0.123600 |
| round2_test_small | film_panel_gated_mean_aux | 17 | film_panel_gated_mean_aux_raw_soft_fusion | PatchTST | 1744 | 0.348800 |
| round2_test_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_hard_top1 | CrossFormer | 250 | 0.050000 |
| round2_test_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_hard_top1 | DLinear | 609 | 0.121800 |
| round2_test_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_hard_top1 | ES | 2007 | 0.401400 |
| round2_test_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_hard_top1 | NaiveForecaster | 519 | 0.103800 |
| round2_test_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_hard_top1 | PatchTST | 1615 | 0.323000 |
| round2_test_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_raw_soft_fusion | CrossFormer | 250 | 0.050000 |
| round2_test_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_raw_soft_fusion | DLinear | 609 | 0.121800 |
| round2_test_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_raw_soft_fusion | ES | 2007 | 0.401400 |
| round2_test_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_raw_soft_fusion | NaiveForecaster | 519 | 0.103800 |
| round2_test_small | film_panel_gated_mean_aux | 18 | film_panel_gated_mean_aux_raw_soft_fusion | PatchTST | 1615 | 0.323000 |
| round2_test_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_hard_top1 | CrossFormer | 395 | 0.079000 |
| round2_test_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_hard_top1 | DLinear | 1836 | 0.367200 |
| round2_test_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_hard_top1 | ES | 483 | 0.096600 |
| round2_test_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_hard_top1 | NaiveForecaster | 628 | 0.125600 |
| round2_test_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_hard_top1 | PatchTST | 1658 | 0.331600 |
| round2_test_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_raw_soft_fusion | CrossFormer | 395 | 0.079000 |
| round2_test_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_raw_soft_fusion | DLinear | 1836 | 0.367200 |
| round2_test_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_raw_soft_fusion | ES | 483 | 0.096600 |
| round2_test_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_raw_soft_fusion | NaiveForecaster | 628 | 0.125600 |
| round2_test_small | film_panel_lowrank_aux | 16 | film_panel_lowrank_aux_raw_soft_fusion | PatchTST | 1658 | 0.331600 |
| round2_test_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_hard_top1 | CrossFormer | 235 | 0.047000 |
| round2_test_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_hard_top1 | DLinear | 418 | 0.083600 |
| round2_test_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_hard_top1 | ES | 1968 | 0.393600 |
| round2_test_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_hard_top1 | NaiveForecaster | 523 | 0.104600 |
| round2_test_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_hard_top1 | PatchTST | 1856 | 0.371200 |
| round2_test_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_raw_soft_fusion | CrossFormer | 235 | 0.047000 |
| round2_test_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_raw_soft_fusion | DLinear | 418 | 0.083600 |
| round2_test_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_raw_soft_fusion | ES | 1968 | 0.393600 |
| round2_test_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_raw_soft_fusion | NaiveForecaster | 523 | 0.104600 |
| round2_test_small | film_panel_lowrank_aux | 17 | film_panel_lowrank_aux_raw_soft_fusion | PatchTST | 1856 | 0.371200 |
| round2_test_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_hard_top1 | CrossFormer | 263 | 0.052600 |
| round2_test_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_hard_top1 | DLinear | 530 | 0.106000 |
| round2_test_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_hard_top1 | ES | 2172 | 0.434400 |
| round2_test_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_hard_top1 | NaiveForecaster | 497 | 0.099400 |
| round2_test_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_hard_top1 | PatchTST | 1538 | 0.307600 |
| round2_test_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_raw_soft_fusion | CrossFormer | 263 | 0.052600 |
| round2_test_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_raw_soft_fusion | DLinear | 530 | 0.106000 |
| round2_test_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_raw_soft_fusion | ES | 2172 | 0.434400 |
| round2_test_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_raw_soft_fusion | NaiveForecaster | 497 | 0.099400 |
| round2_test_small | film_panel_lowrank_aux | 18 | film_panel_lowrank_aux_raw_soft_fusion | PatchTST | 1538 | 0.307600 |

## 判断依据

- film_panel_lowrank_aux: selection MAE delta=-0.001780, MSE delta=-0.176006, regret delta=-0.001780, q5_ok=True, PatchTST/CrossFormer_ok=False, test_ok=True
- film_panel_gated_mean_aux: selection MAE delta=+0.004405, MSE delta=+0.321951, regret delta=+0.004405, q5_ok=False, PatchTST/CrossFormer_ok=False, test_ok=True

## 产物

- `panel_gating_35k_variant_seed_results.csv`
- `panel_gating_35k_selection_summary.csv`
- `panel_gating_35k_diagnostic_summary.csv`
- `panel_gating_35k_test_small_summary.csv`
- `panel_gating_35k_selected_model_counts.csv`
- `panel_gating_35k_stratified_summary.csv`
- `panel_gating_35k_key_strata_delta.csv`
- `panel_gating_35k_metadata.json`
- `panel_gating_35k_screening_summary.md`
