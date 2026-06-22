# Visual Router V2 Round2 65k Pooling Ablation Summary

生成时间：2026-06-23 03:31:17 CST

## 结论

- 固定 layout：`spatial_panel_3view`。
- 固定后端：FiLM aux 注入，condition input 为 `revin_aux`，不直接 concat aux。
- best pooling variant：`film_mean_patch_aux`，选择口径为 `round2_selection_expanded` raw-soft MAE mean。
- frozen test best：`film_mean_patch_aux`；与 selection best 一致。
- 是否支持继续使用 `film_mean_patch_aux` 作为 Round2 主线后端：支持。

## 必答问题

1. 65k selection best pooling variant：`film_mean_patch_aux`，raw-soft MAE=0.307233，MSE=2.043914。
2. 65k test_expanded frozen best：`film_mean_patch_aux`，raw-soft MAE=0.394336，MSE=2.008546；与 selection best 一致。
3. `cls` 是否弱于 `mean_patch`：是。selection raw-soft MAE delta(cls-mean_patch)=+0.001362，MSE delta=+0.002500。
4. `cls_mean_concat` 是否优于单独 `cls` 或 `mean_patch`：相对 cls MAE delta=-0.000567；相对 mean_patch MAE delta=+0.000795。因此 concat 优于 cls 但未优于 mean_patch。
5. `film_cls_mean_concat_aux` 是否带来更好 MAE/MSE：未优于 anchor mean_patch；selection MAE 高 +0.000795，MSE 高 +0.225216，说明增加维度没有带来收益。
6. seed stability：selection raw-soft MAE_std 最低为 `film_cls_aux`=0.000715；anchor mean_patch MAE_std=0.001518。
7. MSE tail：selection raw-soft MSE 最低为 `film_mean_patch_aux`=2.043914；test raw-soft MSE 最低为 `film_mean_patch_aux`=2.008546。
8. CrossFormer / PatchTST / ES / DLinear / NaiveForecaster 分层 best：CrossFormer: film_cls_mean_concat_aux MAE=0.497906, MSE=0.825527; DLinear: film_cls_mean_concat_aux MAE=0.375560, MSE=5.014533; ES: film_cls_mean_concat_aux MAE=0.129188, MSE=0.642071; NaiveForecaster: film_cls_mean_concat_aux MAE=0.364827, MSE=1.999646; PatchTST: film_mean_patch_aux MAE=0.464945, MSE=6.035855
9. selected_model ratio 是否出现单专家塌缩：未见单专家塌缩；selection raw-soft 各 variant 最高平均 selected_model ratio 为 film_cls_aux: max DLinear=0.433; film_cls_mean_concat_aux: max DLinear=0.507; film_mean_patch_aux: max DLinear=0.414。
10. 是否支持继续使用 `film_mean_patch_aux` 作为 Round2 主线后端：支持。

## Selection Summary

| variant | visual_input_mode | visual_dim | seed_count | sample_set | method_kind | method | sample_count_per_seed | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| film_mean_patch_aux | mean_patch | 768 | 3 | round2_selection_expanded | raw_soft_fusion | film_mean_patch_aux_raw_soft_fusion | 10000 | 0.307233 | 0.001518 | 2.043914 | 0.158331 | 0.043993 | 0.001518 | 0.454233 | 0.154265 | 1.124301 | 0.030309 | 0.698568 | 0.018832 | 0.524017 | 0.034404 |
| film_cls_mean_concat_aux | cls_mean_concat | 1536 | 3 | round2_selection_expanded | raw_soft_fusion | film_cls_mean_concat_aux_raw_soft_fusion | 10000 | 0.308029 | 0.000988 | 2.269129 | 0.056146 | 0.044788 | 0.000988 | 0.460100 | 0.147917 | 1.129805 | 0.014668 | 0.701987 | 0.009114 | 0.510928 | 0.012426 |
| film_cls_aux | cls | 768 | 3 | round2_selection_expanded | raw_soft_fusion | film_cls_aux_raw_soft_fusion | 10000 | 0.308595 | 0.000715 | 2.046413 | 0.223704 | 0.045355 | 0.000715 | 0.453200 | 0.150558 | 1.136652 | 0.026585 | 0.706242 | 0.016518 | 0.503964 | 0.022954 |
| film_mean_patch_aux | mean_patch | 768 | 3 | round2_selection_expanded | hard_top1 | film_mean_patch_aux_hard_top1 | 10000 | 0.324800 | 0.001343 | 2.219885 | 0.236141 | 0.061559 | 0.001343 | 0.454233 | 0.154265 | 1.124301 | 0.030309 | 0.698568 | 0.018832 | 0.524017 | 0.034404 |
| film_cls_mean_concat_aux | cls_mean_concat | 1536 | 3 | round2_selection_expanded | hard_top1 | film_cls_mean_concat_aux_hard_top1 | 10000 | 0.325364 | 0.001936 | 2.360953 | 0.009054 | 0.062123 | 0.001936 | 0.460100 | 0.147917 | 1.129805 | 0.014668 | 0.701987 | 0.009114 | 0.510928 | 0.012426 |
| film_cls_aux | cls | 768 | 3 | round2_selection_expanded | hard_top1 | film_cls_aux_hard_top1 | 10000 | 0.327544 | 0.001340 | 2.361622 | 0.019894 | 0.064303 | 0.001340 | 0.453200 | 0.150558 | 1.136652 | 0.026585 | 0.706242 | 0.016518 | 0.503964 | 0.022954 |

## Diagnostic Summary

| variant | visual_input_mode | visual_dim | seed_count | sample_set | method_kind | method | sample_count_per_seed | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| film_mean_patch_aux | mean_patch | 768 | 3 | round2_diagnostic_balanced_expanded | raw_soft_fusion | film_mean_patch_aux_raw_soft_fusion | 10000 | 0.367737 | 0.001179 | 2.904819 | 0.539291 | 0.057210 | 0.001179 | 0.416467 | 0.069560 | 1.049741 | 0.018294 | 0.652241 | 0.011367 | 0.560353 | 0.020153 |
| film_cls_mean_concat_aux | cls_mean_concat | 1536 | 3 | round2_diagnostic_balanced_expanded | raw_soft_fusion | film_cls_mean_concat_aux_raw_soft_fusion | 10000 | 0.367778 | 0.003235 | 2.657263 | 0.106038 | 0.057252 | 0.003235 | 0.411500 | 0.075194 | 1.049236 | 0.012792 | 0.651927 | 0.007948 | 0.554959 | 0.011549 |
| film_cls_aux | cls | 768 | 3 | round2_diagnostic_balanced_expanded | raw_soft_fusion | film_cls_aux_raw_soft_fusion | 10000 | 0.370664 | 0.004035 | 2.616089 | 0.099487 | 0.060138 | 0.004035 | 0.397833 | 0.067421 | 1.056387 | 0.023562 | 0.656370 | 0.014640 | 0.547197 | 0.019686 |
| film_mean_patch_aux | mean_patch | 768 | 3 | round2_diagnostic_balanced_expanded | hard_top1 | film_mean_patch_aux_hard_top1 | 10000 | 0.391445 | 0.003912 | 3.062045 | 0.468718 | 0.080919 | 0.003912 | 0.416467 | 0.069560 | 1.049741 | 0.018294 | 0.652241 | 0.011367 | 0.560353 | 0.020153 |
| film_cls_mean_concat_aux | cls_mean_concat | 1536 | 3 | round2_diagnostic_balanced_expanded | hard_top1 | film_cls_mean_concat_aux_hard_top1 | 10000 | 0.392606 | 0.003287 | 2.832046 | 0.077523 | 0.082080 | 0.003287 | 0.411500 | 0.075194 | 1.049236 | 0.012792 | 0.651927 | 0.007948 | 0.554959 | 0.011549 |
| film_cls_aux | cls | 768 | 3 | round2_diagnostic_balanced_expanded | hard_top1 | film_cls_aux_hard_top1 | 10000 | 0.393264 | 0.005529 | 2.669731 | 0.100527 | 0.082738 | 0.005529 | 0.397833 | 0.067421 | 1.056387 | 0.023562 | 0.656370 | 0.014640 | 0.547197 | 0.019686 |

## Frozen Test Summary

| variant | visual_input_mode | visual_dim | seed_count | sample_set | method_kind | method | sample_count_per_seed | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| film_mean_patch_aux | mean_patch | 768 | 3 | round2_test_expanded | raw_soft_fusion | film_mean_patch_aux_raw_soft_fusion | 15000 | 0.394336 | 0.002355 | 2.008546 | 0.031969 | 0.069543 | 0.002355 | 0.417822 | 0.156928 | 1.123550 | 0.046254 | 0.698101 | 0.028739 | 0.534448 | 0.046124 |
| film_cls_aux | cls | 768 | 3 | round2_test_expanded | raw_soft_fusion | film_cls_aux_raw_soft_fusion | 15000 | 0.396900 | 0.001465 | 2.039682 | 0.089669 | 0.072107 | 0.001465 | 0.416556 | 0.159508 | 1.132392 | 0.025605 | 0.703595 | 0.015909 | 0.517094 | 0.014060 |
| film_cls_mean_concat_aux | cls_mean_concat | 1536 | 3 | round2_test_expanded | raw_soft_fusion | film_cls_mean_concat_aux_raw_soft_fusion | 15000 | 0.397700 | 0.002912 | 2.064184 | 0.082819 | 0.072907 | 0.002912 | 0.420022 | 0.155808 | 1.119044 | 0.034673 | 0.695301 | 0.021543 | 0.525952 | 0.020435 |
| film_mean_patch_aux | mean_patch | 768 | 3 | round2_test_expanded | hard_top1 | film_mean_patch_aux_hard_top1 | 15000 | 0.409695 | 0.002857 | 2.108045 | 0.032389 | 0.084902 | 0.002857 | 0.417822 | 0.156928 | 1.123550 | 0.046254 | 0.698101 | 0.028739 | 0.534448 | 0.046124 |
| film_cls_aux | cls | 768 | 3 | round2_test_expanded | hard_top1 | film_cls_aux_hard_top1 | 15000 | 0.411267 | 0.002942 | 2.122390 | 0.094995 | 0.086474 | 0.002942 | 0.416556 | 0.159508 | 1.132392 | 0.025605 | 0.703595 | 0.015909 | 0.517094 | 0.014060 |
| film_cls_mean_concat_aux | cls_mean_concat | 1536 | 3 | round2_test_expanded | hard_top1 | film_cls_mean_concat_aux_hard_top1 | 15000 | 0.412720 | 0.003088 | 2.155146 | 0.077687 | 0.087927 | 0.003088 | 0.420022 | 0.155808 | 1.119044 | 0.034673 | 0.695301 | 0.021543 | 0.525952 | 0.020435 |

## Delta Summary

| delta_name | sample_set | method_kind | left_variant | right_variant | left_MAE_mean | right_MAE_mean | delta_MAE_mean | left_MSE_mean | right_MSE_mean | delta_MSE_mean | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| film_cls_aux - film_mean_patch_aux | round2_selection_expanded | raw_soft_fusion | film_cls_aux | film_mean_patch_aux | 0.308595 | 0.307233 | 0.001362 | 2.046413 | 2.043914 | 0.002500 | ok |
| film_cls_mean_concat_aux - film_cls_aux | round2_selection_expanded | raw_soft_fusion | film_cls_mean_concat_aux | film_cls_aux | 0.308029 | 0.308595 | -0.000567 | 2.269129 | 2.046413 | 0.222716 | ok |
| film_cls_mean_concat_aux - film_mean_patch_aux | round2_selection_expanded | raw_soft_fusion | film_cls_mean_concat_aux | film_mean_patch_aux | 0.308029 | 0.307233 | 0.000795 | 2.269129 | 2.043914 | 0.225216 | ok |
