# Visual Router V2 Round 1 P2e FiLM Frozen Pilot Test Extension Summary

生成时间：2026-06-21 18:13:09 CST

## 结论回答

1. `film_mean_patch_aux` 是否在 pilot_test 上优于 `visual_mean_patch_only`：是。raw-soft MAE=0.417824 vs 0.452976，delta=-0.035152。
2. `film_mean_patch_aux` 是否避免 `mean_patch_plus_aux` 明显退化：是。film MAE=0.417824，mean_patch_plus_aux MAE=0.516108，delta=-0.098284。
3. `film_cls_mean_concat_aux` 是否优于 `visual_cls_mean_concat`：是。raw-soft MAE=0.419568 vs 0.443062，delta=-0.023493。
4. `film_cls_mean_concat_aux` 是否优于 `cls_mean_concat_plus_aux`：是。raw-soft MAE=0.419568 vs 0.452942，delta=-0.033373。
5. 两个 FiLM 变体中 raw-soft MAE 更好的是 `film_mean_patch_aux`，MSE 更好的是 `film_mean_patch_aux`，regret 更好的是 `film_mean_patch_aux`；film_mean_patch_aux MAE/MSE/regret=0.417824/183.353985/0.077539，film_cls_mean_concat_aux=0.419568/183.463846/0.079283，二者 MAE delta(cls-mean)=+0.001744。
6. FiLM 是否改善 seed stability：mean_patch 路线 是，cls+mean 路线 是；FiLM MAE_std=0.000657/0.001850。
7. FiLM 是否改善 MSE tail：mean_patch 路线 是，cls+mean 路线 是；MSE delta 分别为 -302.748534 和 -60.774642。
8. FiLM 是否改善 CrossFormer / PatchTST strata：见下方 oracle_model 分层摘录；完整表在 `round1_film_final_test_extension_stratified_summary.csv`。
9. FiLM 是否仍主要依赖 soft fusion 而不是 hard oracle-label accuracy：是。normalized entropy=0.697282/0.691494，mean max weight=0.514508/0.517568，oracle-label accuracy=0.515560/0.426618。
10. P2e 后续建议：可进入下一轮 Round2/P2f，但必须以 frozen test 风险为约束。该判断只用于后续路线，不改变 P2e selection best 历史结论，也不使用 pilot_test 做模型选择。

## Comparison

| sample_set | method | variant | seed_count | sample_count | hard_top1_MAE | hard_top1_MSE | hard_top1_regret_to_oracle | hard_top1_oracle_label_accuracy | raw_soft_fusion_MAE | raw_soft_fusion_MSE | raw_soft_fusion_regret_to_oracle | raw_soft_fusion_oracle_label_accuracy | weight_entropy | normalized_weight_entropy | mean_max_weight | MAE_std | MSE_std | regret_to_oracle_std | oracle_label_accuracy_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_test | film_mean_patch_aux_hard_top1 | film_mean_patch_aux | 3 | 75000 | 0.431851 | 183.477411 | 0.091566 | 0.515560 |  |  |  |  | 1.122232 | 0.697282 | 0.514508 | 0.000909 | 0.129478 | 0.000909 | 0.153230 |
| pilot_test | film_mean_patch_aux_raw_soft_fusion | film_mean_patch_aux | 3 | 75000 |  |  |  |  | 0.417824 | 183.353985 | 0.077539 | 0.515560 | 1.122232 | 0.697282 | 0.514508 | 0.000657 | 0.157147 | 0.000657 | 0.153230 |
| pilot_test | film_cls_mean_concat_aux_hard_top1 | film_cls_mean_concat_aux | 3 | 75000 | 0.434611 | 183.614013 | 0.094326 | 0.426618 |  |  |  |  | 1.112916 | 0.691494 | 0.517568 | 0.001566 | 0.145622 | 0.001566 | 0.158805 |
| pilot_test | film_cls_mean_concat_aux_raw_soft_fusion | film_cls_mean_concat_aux | 3 | 75000 |  |  |  |  | 0.419568 | 183.463846 | 0.079283 | 0.426618 | 1.112916 | 0.691494 | 0.517568 | 0.001850 | 0.132809 | 0.001850 | 0.158805 |
| pilot_test | visual_cls_mean_concat_hard_top1 | visual_cls_mean_concat | 3 | 75000 | 0.450969 | 257.265910 | 0.110684 | 0.517329 |  |  |  |  | 1.122043 | 0.697165 | 0.524337 | 0.025854 | 109.709509 | 0.025854 | 0.171314 |
| pilot_test | visual_cls_mean_concat_raw_soft_fusion | visual_cls_mean_concat | 3 | 75000 |  |  |  |  | 0.443062 | 244.238487 | 0.102777 | 0.517329 | 1.122043 | 0.697165 | 0.524337 | 0.021419 | 90.916281 | 0.021419 | 0.171314 |
| pilot_test | visual_mean_patch_only_hard_top1 | visual_mean_patch_only | 3 | 75000 | 0.463490 | 334.751485 | 0.123205 | 0.337782 |  |  |  |  | 1.143872 | 0.710728 | 0.505401 | 0.057565 | 263.565738 | 0.057565 | 0.005231 |
| pilot_test | visual_mean_patch_only_raw_soft_fusion | visual_mean_patch_only | 3 | 75000 |  |  |  |  | 0.452976 | 303.486492 | 0.112691 | 0.337782 | 1.143872 | 0.710728 | 0.505401 | 0.044625 | 189.502593 | 0.044625 | 0.005231 |
| pilot_test | cls_mean_concat_plus_aux_hard_top1 | cls_mean_concat_plus_aux | 3 | 75000 | 0.467320 | 245.625781 | 0.127035 | 0.432360 |  |  |  |  | 1.112345 | 0.691139 | 0.521259 | 0.038801 | 67.800287 | 0.038801 | 0.154035 |
| pilot_test | cls_mean_concat_plus_aux_raw_soft_fusion | cls_mean_concat_plus_aux | 3 | 75000 |  |  |  |  | 0.452942 | 245.459475 | 0.112657 | 0.432360 | 1.112345 | 0.691139 | 0.521259 | 0.039445 | 67.971945 | 0.039445 | 0.154035 |
| pilot_test | mean_patch_plus_aux_hard_top1 | mean_patch_plus_aux | 3 | 75000 | 0.531247 | 486.346829 | 0.190962 | 0.429196 |  |  |  |  | 1.121209 | 0.696646 | 0.514989 | 0.047731 | 293.369741 | 0.047731 | 0.161040 |
| pilot_test | mean_patch_plus_aux_raw_soft_fusion | mean_patch_plus_aux | 3 | 75000 |  |  |  |  | 0.516108 | 486.102519 | 0.175823 | 0.429196 | 1.121209 | 0.696646 | 0.514989 | 0.048081 | 293.536057 | 0.048081 | 0.161040 |
| pilot_test | round0_timefuse_hard_top1 |  | 1 | 75000 | 0.547432 | 568.559825 | 0.207147 | 0.587240 |  |  |  |  | 0.730438 | 0.453847 | 0.701544 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pilot_test | round0_timefuse_raw_soft_fusion |  | 1 | 75000 |  |  |  |  | 0.535220 | 568.502401 | 0.194935 | 0.587240 | 0.730438 | 0.453847 | 0.701544 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pilot_test | round0_original_visual_hard_top1 |  | 1 | 75000 | 0.664912 | 596.442288 | 0.324627 | 0.457960 |  |  |  |  | 1.263102 | 0.784809 | 0.450189 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pilot_test | round0_original_visual_raw_soft_fusion |  | 1 | 75000 |  |  |  |  | 0.603009 | 510.975952 | 0.262724 | 0.457960 | 1.263102 | 0.784809 | 0.450189 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pilot_test | global_best_single |  | 1 | 75000 | 0.599744 |  | 0.259460 | 0.125760 |  |  |  |  |  |  |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pilot_test | oracle_top1 |  | 1 | 75000 | 0.340285 |  | 0.000000 | 1.000000 |  |  |  |  |  |  |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Delta Summary

| delta_name | sample_set | method_kind | left_variant | right_variant | metric | left_value | right_value | delta_left_minus_right | lower_is_better |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| film_mean_patch_aux - visual_mean_patch_only | pilot_test | hard_top1 | film_mean_patch_aux | visual_mean_patch_only | hard_top1_MAE | 0.431851 | 0.463490 | -0.031639 | True |
| film_mean_patch_aux - visual_mean_patch_only | pilot_test | hard_top1 | film_mean_patch_aux | visual_mean_patch_only | hard_top1_MSE | 183.477411 | 334.751485 | -151.274073 | True |
| film_mean_patch_aux - visual_mean_patch_only | pilot_test | hard_top1 | film_mean_patch_aux | visual_mean_patch_only | hard_top1_regret_to_oracle | 0.091566 | 0.123205 | -0.031639 | True |
| film_mean_patch_aux - visual_mean_patch_only | pilot_test | hard_top1 | film_mean_patch_aux | visual_mean_patch_only | hard_top1_oracle_label_accuracy | 0.515560 | 0.337782 | 0.177778 | False |
| film_mean_patch_aux - visual_mean_patch_only | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_mean_patch_only | raw_soft_fusion_MAE | 0.417824 | 0.452976 | -0.035152 | True |
| film_mean_patch_aux - visual_mean_patch_only | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_mean_patch_only | raw_soft_fusion_MSE | 183.353985 | 303.486492 | -120.132506 | True |
| film_mean_patch_aux - visual_mean_patch_only | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_mean_patch_only | raw_soft_fusion_regret_to_oracle | 0.077539 | 0.112691 | -0.035152 | True |
| film_mean_patch_aux - visual_mean_patch_only | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_mean_patch_only | raw_soft_fusion_oracle_label_accuracy | 0.515560 | 0.337782 | 0.177778 | False |
| film_mean_patch_aux - visual_mean_patch_only | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_mean_patch_only | weight_entropy | 1.122232 | 1.143872 | -0.021640 | False |
| film_mean_patch_aux - visual_mean_patch_only | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_mean_patch_only | normalized_weight_entropy | 0.697282 | 0.710728 | -0.013446 | False |
| film_mean_patch_aux - visual_mean_patch_only | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_mean_patch_only | mean_max_weight | 0.514508 | 0.505401 | 0.009107 | False |
| film_mean_patch_aux - visual_mean_patch_only | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_mean_patch_only | MAE_std | 0.000657 | 0.044625 | -0.043968 | True |
| film_mean_patch_aux - visual_mean_patch_only | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_mean_patch_only | MSE_std | 0.157147 | 189.502593 | -189.345446 | True |
| film_mean_patch_aux - visual_mean_patch_only | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_mean_patch_only | regret_to_oracle_std | 0.000657 | 0.044625 | -0.043968 | True |
| film_mean_patch_aux - visual_mean_patch_only | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_mean_patch_only | oracle_label_accuracy_std | 0.153230 | 0.005231 | 0.147999 | True |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_test | hard_top1 | film_mean_patch_aux | mean_patch_plus_aux | hard_top1_MAE | 0.431851 | 0.531247 | -0.099396 | True |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_test | hard_top1 | film_mean_patch_aux | mean_patch_plus_aux | hard_top1_MSE | 183.477411 | 486.346829 | -302.869418 | True |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_test | hard_top1 | film_mean_patch_aux | mean_patch_plus_aux | hard_top1_regret_to_oracle | 0.091566 | 0.190962 | -0.099396 | True |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_test | hard_top1 | film_mean_patch_aux | mean_patch_plus_aux | hard_top1_oracle_label_accuracy | 0.515560 | 0.429196 | 0.086364 | False |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | mean_patch_plus_aux | raw_soft_fusion_MAE | 0.417824 | 0.516108 | -0.098284 | True |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | mean_patch_plus_aux | raw_soft_fusion_MSE | 183.353985 | 486.102519 | -302.748534 | True |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | mean_patch_plus_aux | raw_soft_fusion_regret_to_oracle | 0.077539 | 0.175823 | -0.098284 | True |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | mean_patch_plus_aux | raw_soft_fusion_oracle_label_accuracy | 0.515560 | 0.429196 | 0.086364 | False |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | mean_patch_plus_aux | weight_entropy | 1.122232 | 1.121209 | 0.001023 | False |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | mean_patch_plus_aux | normalized_weight_entropy | 0.697282 | 0.696646 | 0.000636 | False |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | mean_patch_plus_aux | mean_max_weight | 0.514508 | 0.514989 | -0.000481 | False |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | mean_patch_plus_aux | MAE_std | 0.000657 | 0.048081 | -0.047424 | True |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | mean_patch_plus_aux | MSE_std | 0.157147 | 293.536057 | -293.378911 | True |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | mean_patch_plus_aux | regret_to_oracle_std | 0.000657 | 0.048081 | -0.047424 | True |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | mean_patch_plus_aux | oracle_label_accuracy_std | 0.153230 | 0.161040 | -0.007811 | True |
| film_mean_patch_aux - visual_cls_mean_concat | pilot_test | hard_top1 | film_mean_patch_aux | visual_cls_mean_concat | hard_top1_MAE | 0.431851 | 0.450969 | -0.019118 | True |
| film_mean_patch_aux - visual_cls_mean_concat | pilot_test | hard_top1 | film_mean_patch_aux | visual_cls_mean_concat | hard_top1_MSE | 183.477411 | 257.265910 | -73.788499 | True |
| film_mean_patch_aux - visual_cls_mean_concat | pilot_test | hard_top1 | film_mean_patch_aux | visual_cls_mean_concat | hard_top1_regret_to_oracle | 0.091566 | 0.110684 | -0.019118 | True |
| film_mean_patch_aux - visual_cls_mean_concat | pilot_test | hard_top1 | film_mean_patch_aux | visual_cls_mean_concat | hard_top1_oracle_label_accuracy | 0.515560 | 0.517329 | -0.001769 | False |
| film_mean_patch_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_cls_mean_concat | raw_soft_fusion_MAE | 0.417824 | 0.443062 | -0.025238 | True |
| film_mean_patch_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_cls_mean_concat | raw_soft_fusion_MSE | 183.353985 | 244.238487 | -60.884502 | True |
| film_mean_patch_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_cls_mean_concat | raw_soft_fusion_regret_to_oracle | 0.077539 | 0.102777 | -0.025238 | True |
| film_mean_patch_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_cls_mean_concat | raw_soft_fusion_oracle_label_accuracy | 0.515560 | 0.517329 | -0.001769 | False |
| film_mean_patch_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_cls_mean_concat | weight_entropy | 1.122232 | 1.122043 | 0.000189 | False |
| film_mean_patch_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_cls_mean_concat | normalized_weight_entropy | 0.697282 | 0.697165 | 0.000117 | False |
| film_mean_patch_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_cls_mean_concat | mean_max_weight | 0.514508 | 0.524337 | -0.009829 | False |
| film_mean_patch_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_cls_mean_concat | MAE_std | 0.000657 | 0.021419 | -0.020762 | True |
| film_mean_patch_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_cls_mean_concat | MSE_std | 0.157147 | 90.916281 | -90.759134 | True |
| film_mean_patch_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_cls_mean_concat | regret_to_oracle_std | 0.000657 | 0.021419 | -0.020762 | True |
| film_mean_patch_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_mean_patch_aux | visual_cls_mean_concat | oracle_label_accuracy_std | 0.153230 | 0.171314 | -0.018084 | True |
| film_mean_patch_aux - cls_mean_concat_plus_aux | pilot_test | hard_top1 | film_mean_patch_aux | cls_mean_concat_plus_aux | hard_top1_MAE | 0.431851 | 0.467320 | -0.035469 | True |
| film_mean_patch_aux - cls_mean_concat_plus_aux | pilot_test | hard_top1 | film_mean_patch_aux | cls_mean_concat_plus_aux | hard_top1_MSE | 183.477411 | 245.625781 | -62.148370 | True |
| film_mean_patch_aux - cls_mean_concat_plus_aux | pilot_test | hard_top1 | film_mean_patch_aux | cls_mean_concat_plus_aux | hard_top1_regret_to_oracle | 0.091566 | 0.127035 | -0.035469 | True |
| film_mean_patch_aux - cls_mean_concat_plus_aux | pilot_test | hard_top1 | film_mean_patch_aux | cls_mean_concat_plus_aux | hard_top1_oracle_label_accuracy | 0.515560 | 0.432360 | 0.083200 | False |
| film_mean_patch_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | cls_mean_concat_plus_aux | raw_soft_fusion_MAE | 0.417824 | 0.452942 | -0.035118 | True |
| film_mean_patch_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | cls_mean_concat_plus_aux | raw_soft_fusion_MSE | 183.353985 | 245.459475 | -62.105489 | True |
| film_mean_patch_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | cls_mean_concat_plus_aux | raw_soft_fusion_regret_to_oracle | 0.077539 | 0.112657 | -0.035118 | True |
| film_mean_patch_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | cls_mean_concat_plus_aux | raw_soft_fusion_oracle_label_accuracy | 0.515560 | 0.432360 | 0.083200 | False |
| film_mean_patch_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | cls_mean_concat_plus_aux | weight_entropy | 1.122232 | 1.112345 | 0.009887 | False |
| film_mean_patch_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | cls_mean_concat_plus_aux | normalized_weight_entropy | 0.697282 | 0.691139 | 0.006143 | False |
| film_mean_patch_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | cls_mean_concat_plus_aux | mean_max_weight | 0.514508 | 0.521259 | -0.006751 | False |
| film_mean_patch_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | cls_mean_concat_plus_aux | MAE_std | 0.000657 | 0.039445 | -0.038787 | True |
| film_mean_patch_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | cls_mean_concat_plus_aux | MSE_std | 0.157147 | 67.971945 | -67.814798 | True |
| film_mean_patch_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | cls_mean_concat_plus_aux | regret_to_oracle_std | 0.000657 | 0.039445 | -0.038787 | True |
| film_mean_patch_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_mean_patch_aux | cls_mean_concat_plus_aux | oracle_label_accuracy_std | 0.153230 | 0.154035 | -0.000805 | True |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_test | hard_top1 | film_cls_mean_concat_aux | visual_cls_mean_concat | hard_top1_MAE | 0.434611 | 0.450969 | -0.016358 | True |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_test | hard_top1 | film_cls_mean_concat_aux | visual_cls_mean_concat | hard_top1_MSE | 183.614013 | 257.265910 | -73.651897 | True |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_test | hard_top1 | film_cls_mean_concat_aux | visual_cls_mean_concat | hard_top1_regret_to_oracle | 0.094326 | 0.110684 | -0.016358 | True |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_test | hard_top1 | film_cls_mean_concat_aux | visual_cls_mean_concat | hard_top1_oracle_label_accuracy | 0.426618 | 0.517329 | -0.090711 | False |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | visual_cls_mean_concat | raw_soft_fusion_MAE | 0.419568 | 0.443062 | -0.023493 | True |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | visual_cls_mean_concat | raw_soft_fusion_MSE | 183.463846 | 244.238487 | -60.774642 | True |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | visual_cls_mean_concat | raw_soft_fusion_regret_to_oracle | 0.079283 | 0.102777 | -0.023493 | True |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | visual_cls_mean_concat | raw_soft_fusion_oracle_label_accuracy | 0.426618 | 0.517329 | -0.090711 | False |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | visual_cls_mean_concat | weight_entropy | 1.112916 | 1.122043 | -0.009127 | False |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | visual_cls_mean_concat | normalized_weight_entropy | 0.691494 | 0.697165 | -0.005671 | False |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | visual_cls_mean_concat | mean_max_weight | 0.517568 | 0.524337 | -0.006769 | False |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | visual_cls_mean_concat | MAE_std | 0.001850 | 0.021419 | -0.019569 | True |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | visual_cls_mean_concat | MSE_std | 0.132809 | 90.916281 | -90.783472 | True |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | visual_cls_mean_concat | regret_to_oracle_std | 0.001850 | 0.021419 | -0.019569 | True |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | visual_cls_mean_concat | oracle_label_accuracy_std | 0.158805 | 0.171314 | -0.012508 | True |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_test | hard_top1 | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | hard_top1_MAE | 0.434611 | 0.467320 | -0.032709 | True |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_test | hard_top1 | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | hard_top1_MSE | 183.614013 | 245.625781 | -62.011768 | True |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_test | hard_top1 | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | hard_top1_regret_to_oracle | 0.094326 | 0.127035 | -0.032709 | True |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_test | hard_top1 | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | hard_top1_oracle_label_accuracy | 0.426618 | 0.432360 | -0.005742 | False |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | raw_soft_fusion_MAE | 0.419568 | 0.452942 | -0.033373 | True |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | raw_soft_fusion_MSE | 183.463846 | 245.459475 | -61.995629 | True |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | raw_soft_fusion_regret_to_oracle | 0.079283 | 0.112657 | -0.033373 | True |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | raw_soft_fusion_oracle_label_accuracy | 0.426618 | 0.432360 | -0.005742 | False |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | weight_entropy | 1.112916 | 1.112345 | 0.000571 | False |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | normalized_weight_entropy | 0.691494 | 0.691139 | 0.000355 | False |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | mean_max_weight | 0.517568 | 0.521259 | -0.003691 | False |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | MAE_std | 0.001850 | 0.039445 | -0.037594 | True |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | MSE_std | 0.132809 | 67.971945 | -67.839136 | True |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | regret_to_oracle_std | 0.001850 | 0.039445 | -0.037594 | True |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | oracle_label_accuracy_std | 0.158805 | 0.154035 | 0.004770 | True |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_test | hard_top1 | film_cls_mean_concat_aux | film_mean_patch_aux | hard_top1_MAE | 0.434611 | 0.431851 | 0.002760 | True |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_test | hard_top1 | film_cls_mean_concat_aux | film_mean_patch_aux | hard_top1_MSE | 183.614013 | 183.477411 | 0.136602 | True |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_test | hard_top1 | film_cls_mean_concat_aux | film_mean_patch_aux | hard_top1_regret_to_oracle | 0.094326 | 0.091566 | 0.002760 | True |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_test | hard_top1 | film_cls_mean_concat_aux | film_mean_patch_aux | hard_top1_oracle_label_accuracy | 0.426618 | 0.515560 | -0.088942 | False |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | film_mean_patch_aux | raw_soft_fusion_MAE | 0.419568 | 0.417824 | 0.001744 | True |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | film_mean_patch_aux | raw_soft_fusion_MSE | 183.463846 | 183.353985 | 0.109860 | True |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | film_mean_patch_aux | raw_soft_fusion_regret_to_oracle | 0.079283 | 0.077539 | 0.001744 | True |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | film_mean_patch_aux | raw_soft_fusion_oracle_label_accuracy | 0.426618 | 0.515560 | -0.088942 | False |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | film_mean_patch_aux | weight_entropy | 1.112916 | 1.122232 | -0.009315 | False |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | film_mean_patch_aux | normalized_weight_entropy | 0.691494 | 0.697282 | -0.005788 | False |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | film_mean_patch_aux | mean_max_weight | 0.517568 | 0.514508 | 0.003060 | False |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | film_mean_patch_aux | MAE_std | 0.001850 | 0.000657 | 0.001193 | True |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | film_mean_patch_aux | MSE_std | 0.132809 | 0.157147 | -0.024338 | True |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | film_mean_patch_aux | regret_to_oracle_std | 0.001850 | 0.000657 | 0.001193 | True |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | film_mean_patch_aux | oracle_label_accuracy_std | 0.158805 | 0.153230 | 0.005576 | True |
| film_mean_patch_aux - Round0 TimeFuse | pilot_test | hard_top1 | film_mean_patch_aux | Round0 TimeFuse | hard_top1_MAE | 0.431851 | 0.547432 | -0.115582 | True |
| film_mean_patch_aux - Round0 TimeFuse | pilot_test | hard_top1 | film_mean_patch_aux | Round0 TimeFuse | hard_top1_MSE | 183.477411 | 568.559825 | -385.082413 | True |
| film_mean_patch_aux - Round0 TimeFuse | pilot_test | hard_top1 | film_mean_patch_aux | Round0 TimeFuse | hard_top1_regret_to_oracle | 0.091566 | 0.207147 | -0.115582 | True |
| film_mean_patch_aux - Round0 TimeFuse | pilot_test | hard_top1 | film_mean_patch_aux | Round0 TimeFuse | hard_top1_oracle_label_accuracy | 0.515560 | 0.587240 | -0.071680 | False |
| film_mean_patch_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_mean_patch_aux | Round0 TimeFuse | raw_soft_fusion_MAE | 0.417824 | 0.535220 | -0.117396 | True |
| film_mean_patch_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_mean_patch_aux | Round0 TimeFuse | raw_soft_fusion_MSE | 183.353985 | 568.502401 | -385.148416 | True |
| film_mean_patch_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_mean_patch_aux | Round0 TimeFuse | raw_soft_fusion_regret_to_oracle | 0.077539 | 0.194935 | -0.117396 | True |
| film_mean_patch_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_mean_patch_aux | Round0 TimeFuse | raw_soft_fusion_oracle_label_accuracy | 0.515560 | 0.587240 | -0.071680 | False |
| film_mean_patch_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_mean_patch_aux | Round0 TimeFuse | weight_entropy | 1.122232 | 0.730438 | 0.391794 | False |
| film_mean_patch_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_mean_patch_aux | Round0 TimeFuse | normalized_weight_entropy | 0.697282 | 0.453847 | 0.243435 | False |
| film_mean_patch_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_mean_patch_aux | Round0 TimeFuse | mean_max_weight | 0.514508 | 0.701544 | -0.187036 | False |
| film_mean_patch_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_mean_patch_aux | Round0 TimeFuse | MAE_std | 0.000657 | 0.000000 | 0.000657 | True |
| film_mean_patch_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_mean_patch_aux | Round0 TimeFuse | MSE_std | 0.157147 | 0.000000 | 0.157147 | True |
| film_mean_patch_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_mean_patch_aux | Round0 TimeFuse | regret_to_oracle_std | 0.000657 | 0.000000 | 0.000657 | True |
| film_mean_patch_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_mean_patch_aux | Round0 TimeFuse | oracle_label_accuracy_std | 0.153230 | 0.000000 | 0.153230 | True |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_test | hard_top1 | film_cls_mean_concat_aux | Round0 TimeFuse | hard_top1_MAE | 0.434611 | 0.547432 | -0.112822 | True |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_test | hard_top1 | film_cls_mean_concat_aux | Round0 TimeFuse | hard_top1_MSE | 183.614013 | 568.559825 | -384.945812 | True |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_test | hard_top1 | film_cls_mean_concat_aux | Round0 TimeFuse | hard_top1_regret_to_oracle | 0.094326 | 0.207147 | -0.112822 | True |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_test | hard_top1 | film_cls_mean_concat_aux | Round0 TimeFuse | hard_top1_oracle_label_accuracy | 0.426618 | 0.587240 | -0.160622 | False |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | Round0 TimeFuse | raw_soft_fusion_MAE | 0.419568 | 0.535220 | -0.115652 | True |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | Round0 TimeFuse | raw_soft_fusion_MSE | 183.463846 | 568.502401 | -385.038556 | True |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | Round0 TimeFuse | raw_soft_fusion_regret_to_oracle | 0.079283 | 0.194935 | -0.115652 | True |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | Round0 TimeFuse | raw_soft_fusion_oracle_label_accuracy | 0.426618 | 0.587240 | -0.160622 | False |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | Round0 TimeFuse | weight_entropy | 1.112916 | 0.730438 | 0.382478 | False |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | Round0 TimeFuse | normalized_weight_entropy | 0.691494 | 0.453847 | 0.237647 | False |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | Round0 TimeFuse | mean_max_weight | 0.517568 | 0.701544 | -0.183976 | False |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | Round0 TimeFuse | MAE_std | 0.001850 | 0.000000 | 0.001850 | True |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | Round0 TimeFuse | MSE_std | 0.132809 | 0.000000 | 0.132809 | True |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | Round0 TimeFuse | regret_to_oracle_std | 0.001850 | 0.000000 | 0.001850 | True |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_test | raw_soft_fusion | film_cls_mean_concat_aux | Round0 TimeFuse | oracle_label_accuracy_std | 0.158805 | 0.000000 | 0.158805 | True |

## Per-Seed Result

| sample_set | variant | seed | method | sample_count | MAE | MSE | regret_to_oracle | oracle_label_accuracy | weight_entropy | normalized_weight_entropy | mean_max_weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_test | film_cls_mean_concat_aux | 16 | film_cls_mean_concat_aux_hard_top1 | 75000 | 0.434395 | 183.782065 | 0.094110 | 0.335933 | 1.150370 | 0.714765 | 0.507366 |
| pilot_test | film_cls_mean_concat_aux | 16 | film_cls_mean_concat_aux_raw_soft_fusion | 75000 | 0.419974 | 183.612512 | 0.079689 | 0.335933 | 1.150370 | 0.714765 | 0.507366 |
| pilot_test | film_cls_mean_concat_aux | 17 | film_cls_mean_concat_aux_hard_top1 | 75000 | 0.436273 | 183.534959 | 0.095988 | 0.333933 | 1.073434 | 0.666962 | 0.530731 |
| pilot_test | film_cls_mean_concat_aux | 17 | film_cls_mean_concat_aux_raw_soft_fusion | 75000 | 0.421182 | 183.422099 | 0.080897 | 0.333933 | 1.073434 | 0.666962 | 0.530731 |
| pilot_test | film_cls_mean_concat_aux | 18 | film_cls_mean_concat_aux_hard_top1 | 75000 | 0.433164 | 183.525016 | 0.092879 | 0.609987 | 1.114946 | 0.692755 | 0.514607 |
| pilot_test | film_cls_mean_concat_aux | 18 | film_cls_mean_concat_aux_raw_soft_fusion | 75000 | 0.417549 | 183.356926 | 0.077264 | 0.609987 | 1.114946 | 0.692755 | 0.514607 |
| pilot_test | film_mean_patch_aux | 16 | film_mean_patch_aux_hard_top1 | 75000 | 0.432898 | 183.504503 | 0.092613 | 0.338667 | 1.119645 | 0.695674 | 0.527603 |
| pilot_test | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | 75000 | 0.418138 | 183.265267 | 0.077853 | 0.338667 | 1.119645 | 0.695674 | 0.527603 |
| pilot_test | film_mean_patch_aux | 17 | film_mean_patch_aux_hard_top1 | 75000 | 0.431391 | 183.336531 | 0.091106 | 0.607307 | 1.119688 | 0.695701 | 0.504908 |
| pilot_test | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | 75000 | 0.417069 | 183.261261 | 0.076784 | 0.607307 | 1.119688 | 0.695701 | 0.504908 |
| pilot_test | film_mean_patch_aux | 18 | film_mean_patch_aux_hard_top1 | 75000 | 0.431264 | 183.591201 | 0.090979 | 0.600707 | 1.127363 | 0.700470 | 0.511013 |
| pilot_test | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | 75000 | 0.418265 | 183.535428 | 0.077980 | 0.600707 | 1.127363 | 0.700470 | 0.511013 |

## CrossFormer / PatchTST Strata 摘录

| sample_set | variant | seed | method | stratum_column | stratum_kind | stratum_value | sample_count | MAE | MSE | regret_to_oracle | oracle_label_accuracy | weight_entropy | normalized_weight_entropy | mean_max_weight | dataset_name | oracle_model | error_gap_quantile | cluster | group_name | forecastability_cat | season_strength_cat | trend_strength_cat | cv_cat | missing_ratio_cat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_test | cls_mean_concat_plus_aux | 16 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.635800 | 1.416192 | 0.105149 | 0.111905 | 1.085296 | 0.674332 | 0.544954 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | cls_mean_concat_plus_aux | 17 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.624165 | 1.423400 | 0.093514 | 0.134286 | 1.035823 | 0.643593 | 0.562990 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | cls_mean_concat_plus_aux | 18 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.625234 | 1.437301 | 0.094583 | 0.128893 | 1.020350 | 0.633979 | 0.568769 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | film_cls_mean_concat_aux | 16 | film_cls_mean_concat_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.639834 | 1.470466 | 0.109183 | 0.156667 | 1.097338 | 0.681814 | 0.547010 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | film_cls_mean_concat_aux | 17 | film_cls_mean_concat_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.631889 | 1.455788 | 0.101237 | 0.114467 | 1.001599 | 0.622329 | 0.572213 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | film_cls_mean_concat_aux | 18 | film_cls_mean_concat_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.621518 | 1.468610 | 0.090866 | 0.170689 | 1.031359 | 0.640819 | 0.556851 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.642837 | 1.518439 | 0.112186 | 0.082243 | 1.055106 | 0.655574 | 0.571056 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.620849 | 1.423138 | 0.090198 | 0.170015 | 1.059557 | 0.658340 | 0.545045 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.642261 | 1.512334 | 0.111609 | 0.080221 | 1.058514 | 0.657692 | 0.565129 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | mean_patch_plus_aux | 16 | mean_patch_plus_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.639142 | 1.438215 | 0.108490 | 0.105568 | 1.062983 | 0.660469 | 0.558117 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | mean_patch_plus_aux | 17 | mean_patch_plus_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.616115 | 1.401524 | 0.085464 | 0.194958 | 1.070990 | 0.665444 | 0.532647 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | mean_patch_plus_aux | 18 | mean_patch_plus_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.634559 | 1.466574 | 0.103908 | 0.071053 | 1.069664 | 0.664620 | 0.553279 |  |  |  |  |  |  |  |  |  |  |
| pilot_test |  | -1 | round0_timefuse_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.658768 | 1.466646 | 0.128117 | 0.060941 | 0.858139 | 0.533192 | 0.635861 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_cls_mean_concat | 16 | visual_cls_mean_concat_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.629668 | 1.438379 | 0.099017 | 0.122017 | 1.081653 | 0.672069 | 0.550924 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_cls_mean_concat | 17 | visual_cls_mean_concat_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.610474 | 1.377995 | 0.079822 | 0.254011 | 1.052087 | 0.653699 | 0.542495 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_cls_mean_concat | 18 | visual_cls_mean_concat_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.626052 | 1.462753 | 0.095401 | 0.136848 | 1.049996 | 0.652399 | 0.551801 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_mean_patch_only | 16 | visual_mean_patch_only_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.637202 | 1.453442 | 0.106551 | 0.105703 | 1.119787 | 0.695763 | 0.538325 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_mean_patch_only | 17 | visual_mean_patch_only_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.618395 | 1.385014 | 0.087743 | 0.163004 | 1.070048 | 0.664858 | 0.550830 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_mean_patch_only | 18 | visual_mean_patch_only_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.638592 | 1.431614 | 0.107941 | 0.066334 | 1.047794 | 0.651031 | 0.566056 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | cls_mean_concat_plus_aux | 16 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.574480 | 4.451115 | 0.075975 | 0.715472 | 0.928359 | 0.576822 | 0.629596 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | cls_mean_concat_plus_aux | 17 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.581622 | 5.901474 | 0.083117 | 0.758492 | 0.899433 | 0.558849 | 0.637656 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | cls_mean_concat_plus_aux | 18 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.567098 | 6.669592 | 0.068593 | 0.793082 | 0.868870 | 0.539859 | 0.651046 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | film_cls_mean_concat_aux | 16 | film_cls_mean_concat_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.604167 | 9.817582 | 0.105662 | 0.692620 | 0.933833 | 0.580223 | 0.630481 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | film_cls_mean_concat_aux | 17 | film_cls_mean_concat_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.605344 | 8.779589 | 0.106839 | 0.743382 | 0.868958 | 0.539914 | 0.643802 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | film_cls_mean_concat_aux | 18 | film_cls_mean_concat_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.583249 | 7.788517 | 0.084744 | 0.738636 | 0.894458 | 0.555758 | 0.629057 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.582299 | 8.457405 | 0.083794 | 0.760989 | 0.883022 | 0.548652 | 0.657278 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.588150 | 8.072144 | 0.089645 | 0.718157 | 0.922312 | 0.573065 | 0.615990 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.586620 | 9.252342 | 0.088115 | 0.762737 | 0.906117 | 0.563002 | 0.643464 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | mean_patch_plus_aux | 16 | mean_patch_plus_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.592184 | 7.042653 | 0.093679 | 0.702672 | 0.896145 | 0.556806 | 0.644944 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | mean_patch_plus_aux | 17 | mean_patch_plus_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.587288 | 5.886750 | 0.088783 | 0.718531 | 0.955746 | 0.593839 | 0.595528 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | mean_patch_plus_aux | 18 | mean_patch_plus_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.580862 | 5.802504 | 0.082357 | 0.752498 | 0.924688 | 0.574541 | 0.630311 |  |  |  |  |  |  |  |  |  |  |
| pilot_test |  | -1 | round0_timefuse_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.542706 | 3.035039 | 0.044201 | 0.840347 | 0.744881 | 0.462820 | 0.704559 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_cls_mean_concat | 16 | visual_cls_mean_concat_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.566604 | 4.247729 | 0.068099 | 0.720030 | 0.929444 | 0.577496 | 0.629562 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_cls_mean_concat | 17 | visual_cls_mean_concat_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.592471 | 6.329496 | 0.093966 | 0.668769 | 0.934914 | 0.580895 | 0.605558 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_cls_mean_concat | 18 | visual_cls_mean_concat_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.567292 | 4.595499 | 0.068787 | 0.743319 | 0.903251 | 0.561222 | 0.630963 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_mean_patch_only | 16 | visual_mean_patch_only_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.586393 | 5.850754 | 0.087888 | 0.723214 | 0.971932 | 0.603895 | 0.618340 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_mean_patch_only | 17 | visual_mean_patch_only_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.579154 | 5.628166 | 0.080649 | 0.733641 | 0.939264 | 0.583598 | 0.619947 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_mean_patch_only | 18 | visual_mean_patch_only_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.575612 | 5.628816 | 0.077107 | 0.770355 | 0.913530 | 0.567608 | 0.637138 |  |  |  |  |  |  |  |  |  |  |

## 边界记录

- 本扩展只评估 frozen P2e checkpoint；未训练新模型，未改变 variant/seed/epoch/hyperparams。
- pilot_test 只用于最终评估，不用于模型选择。
- 使用 final_test_only feature cache；未重建 P2a feature cache，未保存 pseudo image tensor。
- commit hash：`500bef09238257fa214b39652430c2de7bf6b9ee`
