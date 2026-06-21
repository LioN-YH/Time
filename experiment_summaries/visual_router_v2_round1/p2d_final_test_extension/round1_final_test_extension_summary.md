# Visual Router V2 Round 1 Frozen Final Test Extension Summary

生成时间：2026-06-21 16:48:27 CST

## 核心结论

1. mean_patch_plus_aux 是否足够强：仍与 P2d best 有可见差距。mean_patch_plus_aux raw-soft MAE=0.516108，P2d best=0.452942，delta=+0.063166。
2. aux 对 mean_patch 的 test 边际贡献：delta(mean_patch_plus_aux - visual_mean_patch_only)=+0.063132，负值表示 aux 改善；visual_mean_patch_only raw-soft MAE=0.452976。
3. aux 对 cls+mean 的 test 边际贡献：delta(cls_mean_concat_plus_aux - visual_cls_mean_concat)=+0.009880，负值表示 aux 改善；visual_cls_mean_concat raw-soft MAE=0.443062。
4. visual-only 是否已经超过 Round0 TimeFuse：visual_mean_patch_only 超过，visual_cls_mean_concat 超过；TimeFuse raw-soft MAE=0.535220。
5. cls_mean_concat_plus_aux 优势是否由单一 seed 驱动：P2d best raw-soft MAE_std=0.039445，best seed=17；mean_patch_plus_aux MAE_std=0.048081，是否更稳定=否。
6. 后续 P2e FiLM 主线建议：仍建议以 cls_mean_concat 为主线，mean_patch 作为简洁强基线保留；该建议仅基于冻结 pilot_test 解释，不改变 Round 1 best。
7. CrossFormer / PatchTST strata 见下方分层摘录；完整分层在 `round1_final_test_extension_stratified_summary.csv`。

## Extension Comparison

| sample_set | method | variant | seed_count | sample_count | hard_top1_MAE | hard_top1_MSE | hard_top1_regret_to_oracle | hard_top1_oracle_label_accuracy | raw_soft_fusion_MAE | raw_soft_fusion_MSE | raw_soft_fusion_regret_to_oracle | raw_soft_fusion_oracle_label_accuracy | weight_entropy | normalized_weight_entropy | mean_max_weight | MAE_std | MSE_std | regret_to_oracle_std | oracle_label_accuracy_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_test | cls_mean_concat_plus_aux_hard_top1 | cls_mean_concat_plus_aux | 3 | 75000 | 0.467320 | 245.625781 | 0.127035 | 0.432360 |  |  |  |  | 1.112345 | 0.691139 | 0.521259 | 0.038801 | 67.800287 | 0.038801 | 0.154035 |
| pilot_test | cls_mean_concat_plus_aux_raw_soft_fusion | cls_mean_concat_plus_aux | 3 | 75000 |  |  |  |  | 0.452942 | 245.459475 | 0.112657 | 0.432360 | 1.112345 | 0.691139 | 0.521259 | 0.039445 | 67.971945 | 0.039445 | 0.154035 |
| pilot_test | mean_patch_plus_aux_hard_top1 | mean_patch_plus_aux | 3 | 75000 | 0.531247 | 486.346829 | 0.190962 | 0.429196 |  |  |  |  | 1.121209 | 0.696646 | 0.514989 | 0.047731 | 293.369741 | 0.047731 | 0.161040 |
| pilot_test | mean_patch_plus_aux_raw_soft_fusion | mean_patch_plus_aux | 3 | 75000 |  |  |  |  | 0.516108 | 486.102519 | 0.175823 | 0.429196 | 1.121209 | 0.696646 | 0.514989 | 0.048081 | 293.536057 | 0.048081 | 0.161040 |
| pilot_test | visual_mean_patch_only_hard_top1 | visual_mean_patch_only | 3 | 75000 | 0.463490 | 334.751485 | 0.123205 | 0.337782 |  |  |  |  | 1.143872 | 0.710728 | 0.505401 | 0.057565 | 263.565738 | 0.057565 | 0.005231 |
| pilot_test | visual_mean_patch_only_raw_soft_fusion | visual_mean_patch_only | 3 | 75000 |  |  |  |  | 0.452976 | 303.486492 | 0.112691 | 0.337782 | 1.143872 | 0.710728 | 0.505401 | 0.044625 | 189.502593 | 0.044625 | 0.005231 |
| pilot_test | visual_cls_mean_concat_hard_top1 | visual_cls_mean_concat | 3 | 75000 | 0.450969 | 257.265910 | 0.110684 | 0.517329 |  |  |  |  | 1.122043 | 0.697165 | 0.524337 | 0.025854 | 109.709509 | 0.025854 | 0.171314 |
| pilot_test | visual_cls_mean_concat_raw_soft_fusion | visual_cls_mean_concat | 3 | 75000 |  |  |  |  | 0.443062 | 244.238487 | 0.102777 | 0.517329 | 1.122043 | 0.697165 | 0.524337 | 0.021419 | 90.916281 | 0.021419 | 0.171314 |
| pilot_test | round0_timefuse_hard_top1 |  | 1 | 75000 | 0.547432 | 568.559825 | 0.207147 | 0.587240 |  |  |  |  | 0.730438 | 0.453847 | 0.701544 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pilot_test | round0_timefuse_raw_soft_fusion |  | 1 | 75000 |  |  |  |  | 0.535220 | 568.502401 | 0.194935 | 0.587240 | 0.730438 | 0.453847 | 0.701544 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pilot_test | round0_original_visual_hard_top1 |  | 1 | 75000 | 0.664912 | 596.442288 | 0.324627 | 0.457960 |  |  |  |  | 1.263102 | 0.784809 | 0.450189 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pilot_test | round0_original_visual_raw_soft_fusion |  | 1 | 75000 |  |  |  |  | 0.603009 | 510.975952 | 0.262724 | 0.457960 | 1.263102 | 0.784809 | 0.450189 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pilot_test | global_best_single |  | 1 | 75000 | 0.599744 |  | 0.259460 | 0.125760 |  |  |  |  |  |  |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pilot_test | oracle_top1 |  | 1 | 75000 | 0.340285 |  | 0.000000 | 1.000000 |  |  |  |  |  |  |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Delta Summary

| comparison | lhs | rhs | method_kind | metric | lhs_value | rhs_value | delta_lhs_minus_rhs | lower_is_better |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean_patch_plus_aux vs cls_mean_concat_plus_aux | mean_patch_plus_aux | cls_mean_concat_plus_aux | hard_top1 | hard_top1_MAE | 0.531247 | 0.467320 | 0.063927 | True |
| mean_patch_plus_aux vs cls_mean_concat_plus_aux | mean_patch_plus_aux | cls_mean_concat_plus_aux | hard_top1 | hard_top1_MSE | 486.346829 | 245.625781 | 240.721048 | True |
| mean_patch_plus_aux vs cls_mean_concat_plus_aux | mean_patch_plus_aux | cls_mean_concat_plus_aux | hard_top1 | hard_top1_regret_to_oracle | 0.190962 | 0.127035 | 0.063927 | True |
| mean_patch_plus_aux vs cls_mean_concat_plus_aux | mean_patch_plus_aux | cls_mean_concat_plus_aux | hard_top1 | hard_top1_oracle_label_accuracy | 0.429196 | 0.432360 | -0.003164 | False |
| mean_patch_plus_aux vs cls_mean_concat_plus_aux | mean_patch_plus_aux | cls_mean_concat_plus_aux | raw_soft_fusion | raw_soft_fusion_MAE | 0.516108 | 0.452942 | 0.063166 | True |
| mean_patch_plus_aux vs cls_mean_concat_plus_aux | mean_patch_plus_aux | cls_mean_concat_plus_aux | raw_soft_fusion | raw_soft_fusion_MSE | 486.102519 | 245.459475 | 240.643045 | True |
| mean_patch_plus_aux vs cls_mean_concat_plus_aux | mean_patch_plus_aux | cls_mean_concat_plus_aux | raw_soft_fusion | raw_soft_fusion_regret_to_oracle | 0.175823 | 0.112657 | 0.063166 | True |
| mean_patch_plus_aux vs cls_mean_concat_plus_aux | mean_patch_plus_aux | cls_mean_concat_plus_aux | raw_soft_fusion | raw_soft_fusion_oracle_label_accuracy | 0.429196 | 0.432360 | -0.003164 | False |
| mean_patch_plus_aux vs cls_mean_concat_plus_aux | mean_patch_plus_aux | cls_mean_concat_plus_aux | raw_soft_fusion | weight_entropy | 1.121209 | 1.112345 | 0.008863 | False |
| mean_patch_plus_aux vs cls_mean_concat_plus_aux | mean_patch_plus_aux | cls_mean_concat_plus_aux | raw_soft_fusion | normalized_weight_entropy | 0.696646 | 0.691139 | 0.005507 | False |
| mean_patch_plus_aux vs cls_mean_concat_plus_aux | mean_patch_plus_aux | cls_mean_concat_plus_aux | raw_soft_fusion | mean_max_weight | 0.514989 | 0.521259 | -0.006270 | False |
| mean_patch_plus_aux vs cls_mean_concat_plus_aux | mean_patch_plus_aux | cls_mean_concat_plus_aux | raw_soft_fusion | MAE_std | 0.048081 | 0.039445 | 0.008637 | True |
| mean_patch_plus_aux vs cls_mean_concat_plus_aux | mean_patch_plus_aux | cls_mean_concat_plus_aux | raw_soft_fusion | MSE_std | 293.536057 | 67.971945 | 225.564113 | True |
| mean_patch_plus_aux vs cls_mean_concat_plus_aux | mean_patch_plus_aux | cls_mean_concat_plus_aux | raw_soft_fusion | regret_to_oracle_std | 0.048081 | 0.039445 | 0.008637 | True |
| mean_patch_plus_aux vs cls_mean_concat_plus_aux | mean_patch_plus_aux | cls_mean_concat_plus_aux | raw_soft_fusion | oracle_label_accuracy_std | 0.161040 | 0.154035 | 0.007006 | True |
| mean_patch_plus_aux vs visual_mean_patch_only | mean_patch_plus_aux | visual_mean_patch_only | hard_top1 | hard_top1_MAE | 0.531247 | 0.463490 | 0.067757 | True |
| mean_patch_plus_aux vs visual_mean_patch_only | mean_patch_plus_aux | visual_mean_patch_only | hard_top1 | hard_top1_MSE | 486.346829 | 334.751485 | 151.595344 | True |
| mean_patch_plus_aux vs visual_mean_patch_only | mean_patch_plus_aux | visual_mean_patch_only | hard_top1 | hard_top1_regret_to_oracle | 0.190962 | 0.123205 | 0.067757 | True |
| mean_patch_plus_aux vs visual_mean_patch_only | mean_patch_plus_aux | visual_mean_patch_only | hard_top1 | hard_top1_oracle_label_accuracy | 0.429196 | 0.337782 | 0.091413 | False |
| mean_patch_plus_aux vs visual_mean_patch_only | mean_patch_plus_aux | visual_mean_patch_only | raw_soft_fusion | raw_soft_fusion_MAE | 0.516108 | 0.452976 | 0.063132 | True |
| mean_patch_plus_aux vs visual_mean_patch_only | mean_patch_plus_aux | visual_mean_patch_only | raw_soft_fusion | raw_soft_fusion_MSE | 486.102519 | 303.486492 | 182.616028 | True |
| mean_patch_plus_aux vs visual_mean_patch_only | mean_patch_plus_aux | visual_mean_patch_only | raw_soft_fusion | raw_soft_fusion_regret_to_oracle | 0.175823 | 0.112691 | 0.063132 | True |
| mean_patch_plus_aux vs visual_mean_patch_only | mean_patch_plus_aux | visual_mean_patch_only | raw_soft_fusion | raw_soft_fusion_oracle_label_accuracy | 0.429196 | 0.337782 | 0.091413 | False |
| mean_patch_plus_aux vs visual_mean_patch_only | mean_patch_plus_aux | visual_mean_patch_only | raw_soft_fusion | weight_entropy | 1.121209 | 1.143872 | -0.022664 | False |
| mean_patch_plus_aux vs visual_mean_patch_only | mean_patch_plus_aux | visual_mean_patch_only | raw_soft_fusion | normalized_weight_entropy | 0.696646 | 0.710728 | -0.014082 | False |
| mean_patch_plus_aux vs visual_mean_patch_only | mean_patch_plus_aux | visual_mean_patch_only | raw_soft_fusion | mean_max_weight | 0.514989 | 0.505401 | 0.009587 | False |
| mean_patch_plus_aux vs visual_mean_patch_only | mean_patch_plus_aux | visual_mean_patch_only | raw_soft_fusion | MAE_std | 0.048081 | 0.044625 | 0.003456 | True |
| mean_patch_plus_aux vs visual_mean_patch_only | mean_patch_plus_aux | visual_mean_patch_only | raw_soft_fusion | MSE_std | 293.536057 | 189.502593 | 104.033465 | True |
| mean_patch_plus_aux vs visual_mean_patch_only | mean_patch_plus_aux | visual_mean_patch_only | raw_soft_fusion | regret_to_oracle_std | 0.048081 | 0.044625 | 0.003456 | True |
| mean_patch_plus_aux vs visual_mean_patch_only | mean_patch_plus_aux | visual_mean_patch_only | raw_soft_fusion | oracle_label_accuracy_std | 0.161040 | 0.005231 | 0.155809 | True |
| cls_mean_concat_plus_aux vs visual_cls_mean_concat | cls_mean_concat_plus_aux | visual_cls_mean_concat | hard_top1 | hard_top1_MAE | 0.467320 | 0.450969 | 0.016351 | True |
| cls_mean_concat_plus_aux vs visual_cls_mean_concat | cls_mean_concat_plus_aux | visual_cls_mean_concat | hard_top1 | hard_top1_MSE | 245.625781 | 257.265910 | -11.640129 | True |
| cls_mean_concat_plus_aux vs visual_cls_mean_concat | cls_mean_concat_plus_aux | visual_cls_mean_concat | hard_top1 | hard_top1_regret_to_oracle | 0.127035 | 0.110684 | 0.016351 | True |
| cls_mean_concat_plus_aux vs visual_cls_mean_concat | cls_mean_concat_plus_aux | visual_cls_mean_concat | hard_top1 | hard_top1_oracle_label_accuracy | 0.432360 | 0.517329 | -0.084969 | False |
| cls_mean_concat_plus_aux vs visual_cls_mean_concat | cls_mean_concat_plus_aux | visual_cls_mean_concat | raw_soft_fusion | raw_soft_fusion_MAE | 0.452942 | 0.443062 | 0.009880 | True |
| cls_mean_concat_plus_aux vs visual_cls_mean_concat | cls_mean_concat_plus_aux | visual_cls_mean_concat | raw_soft_fusion | raw_soft_fusion_MSE | 245.459475 | 244.238487 | 1.220987 | True |
| cls_mean_concat_plus_aux vs visual_cls_mean_concat | cls_mean_concat_plus_aux | visual_cls_mean_concat | raw_soft_fusion | raw_soft_fusion_regret_to_oracle | 0.112657 | 0.102777 | 0.009880 | True |
| cls_mean_concat_plus_aux vs visual_cls_mean_concat | cls_mean_concat_plus_aux | visual_cls_mean_concat | raw_soft_fusion | raw_soft_fusion_oracle_label_accuracy | 0.432360 | 0.517329 | -0.084969 | False |
| cls_mean_concat_plus_aux vs visual_cls_mean_concat | cls_mean_concat_plus_aux | visual_cls_mean_concat | raw_soft_fusion | weight_entropy | 1.112345 | 1.122043 | -0.009698 | False |
| cls_mean_concat_plus_aux vs visual_cls_mean_concat | cls_mean_concat_plus_aux | visual_cls_mean_concat | raw_soft_fusion | normalized_weight_entropy | 0.691139 | 0.697165 | -0.006026 | False |
| cls_mean_concat_plus_aux vs visual_cls_mean_concat | cls_mean_concat_plus_aux | visual_cls_mean_concat | raw_soft_fusion | mean_max_weight | 0.521259 | 0.524337 | -0.003078 | False |
| cls_mean_concat_plus_aux vs visual_cls_mean_concat | cls_mean_concat_plus_aux | visual_cls_mean_concat | raw_soft_fusion | MAE_std | 0.039445 | 0.021419 | 0.018025 | True |
| cls_mean_concat_plus_aux vs visual_cls_mean_concat | cls_mean_concat_plus_aux | visual_cls_mean_concat | raw_soft_fusion | MSE_std | 67.971945 | 90.916281 | -22.944336 | True |
| cls_mean_concat_plus_aux vs visual_cls_mean_concat | cls_mean_concat_plus_aux | visual_cls_mean_concat | raw_soft_fusion | regret_to_oracle_std | 0.039445 | 0.021419 | 0.018025 | True |
| cls_mean_concat_plus_aux vs visual_cls_mean_concat | cls_mean_concat_plus_aux | visual_cls_mean_concat | raw_soft_fusion | oracle_label_accuracy_std | 0.154035 | 0.171314 | -0.017279 | True |
| visual_mean_patch_only vs Round0 TimeFuse | visual_mean_patch_only | round0_timefuse | hard_top1 | hard_top1_MAE | 0.463490 | 0.547432 | -0.083943 | True |
| visual_mean_patch_only vs Round0 TimeFuse | visual_mean_patch_only | round0_timefuse | hard_top1 | hard_top1_MSE | 334.751485 | 568.559825 | -233.808340 | True |
| visual_mean_patch_only vs Round0 TimeFuse | visual_mean_patch_only | round0_timefuse | hard_top1 | hard_top1_regret_to_oracle | 0.123205 | 0.207147 | -0.083943 | True |
| visual_mean_patch_only vs Round0 TimeFuse | visual_mean_patch_only | round0_timefuse | hard_top1 | hard_top1_oracle_label_accuracy | 0.337782 | 0.587240 | -0.249458 | False |
| visual_mean_patch_only vs Round0 TimeFuse | visual_mean_patch_only | round0_timefuse | raw_soft_fusion | raw_soft_fusion_MAE | 0.452976 | 0.535220 | -0.082245 | True |
| visual_mean_patch_only vs Round0 TimeFuse | visual_mean_patch_only | round0_timefuse | raw_soft_fusion | raw_soft_fusion_MSE | 303.486492 | 568.502401 | -265.015910 | True |
| visual_mean_patch_only vs Round0 TimeFuse | visual_mean_patch_only | round0_timefuse | raw_soft_fusion | raw_soft_fusion_regret_to_oracle | 0.112691 | 0.194935 | -0.082245 | True |
| visual_mean_patch_only vs Round0 TimeFuse | visual_mean_patch_only | round0_timefuse | raw_soft_fusion | raw_soft_fusion_oracle_label_accuracy | 0.337782 | 0.587240 | -0.249458 | False |
| visual_mean_patch_only vs Round0 TimeFuse | visual_mean_patch_only | round0_timefuse | raw_soft_fusion | weight_entropy | 1.143872 | 0.730438 | 0.413434 | False |
| visual_mean_patch_only vs Round0 TimeFuse | visual_mean_patch_only | round0_timefuse | raw_soft_fusion | normalized_weight_entropy | 0.710728 | 0.453847 | 0.256881 | False |
| visual_mean_patch_only vs Round0 TimeFuse | visual_mean_patch_only | round0_timefuse | raw_soft_fusion | mean_max_weight | 0.505401 | 0.701544 | -0.196143 | False |
| visual_mean_patch_only vs Round0 TimeFuse | visual_mean_patch_only | round0_timefuse | raw_soft_fusion | MAE_std | 0.044625 | 0.000000 | 0.044625 | True |
| visual_mean_patch_only vs Round0 TimeFuse | visual_mean_patch_only | round0_timefuse | raw_soft_fusion | MSE_std | 189.502593 | 0.000000 | 189.502593 | True |
| visual_mean_patch_only vs Round0 TimeFuse | visual_mean_patch_only | round0_timefuse | raw_soft_fusion | regret_to_oracle_std | 0.044625 | 0.000000 | 0.044625 | True |
| visual_mean_patch_only vs Round0 TimeFuse | visual_mean_patch_only | round0_timefuse | raw_soft_fusion | oracle_label_accuracy_std | 0.005231 | 0.000000 | 0.005231 | True |
| visual_cls_mean_concat vs Round0 TimeFuse | visual_cls_mean_concat | round0_timefuse | hard_top1 | hard_top1_MAE | 0.450969 | 0.547432 | -0.096464 | True |
| visual_cls_mean_concat vs Round0 TimeFuse | visual_cls_mean_concat | round0_timefuse | hard_top1 | hard_top1_MSE | 257.265910 | 568.559825 | -311.293914 | True |
| visual_cls_mean_concat vs Round0 TimeFuse | visual_cls_mean_concat | round0_timefuse | hard_top1 | hard_top1_regret_to_oracle | 0.110684 | 0.207147 | -0.096464 | True |
| visual_cls_mean_concat vs Round0 TimeFuse | visual_cls_mean_concat | round0_timefuse | hard_top1 | hard_top1_oracle_label_accuracy | 0.517329 | 0.587240 | -0.069911 | False |
| visual_cls_mean_concat vs Round0 TimeFuse | visual_cls_mean_concat | round0_timefuse | raw_soft_fusion | raw_soft_fusion_MAE | 0.443062 | 0.535220 | -0.092158 | True |
| visual_cls_mean_concat vs Round0 TimeFuse | visual_cls_mean_concat | round0_timefuse | raw_soft_fusion | raw_soft_fusion_MSE | 244.238487 | 568.502401 | -324.263914 | True |
| visual_cls_mean_concat vs Round0 TimeFuse | visual_cls_mean_concat | round0_timefuse | raw_soft_fusion | raw_soft_fusion_regret_to_oracle | 0.102777 | 0.194935 | -0.092158 | True |
| visual_cls_mean_concat vs Round0 TimeFuse | visual_cls_mean_concat | round0_timefuse | raw_soft_fusion | raw_soft_fusion_oracle_label_accuracy | 0.517329 | 0.587240 | -0.069911 | False |
| visual_cls_mean_concat vs Round0 TimeFuse | visual_cls_mean_concat | round0_timefuse | raw_soft_fusion | weight_entropy | 1.122043 | 0.730438 | 0.391605 | False |
| visual_cls_mean_concat vs Round0 TimeFuse | visual_cls_mean_concat | round0_timefuse | raw_soft_fusion | normalized_weight_entropy | 0.697165 | 0.453847 | 0.243318 | False |
| visual_cls_mean_concat vs Round0 TimeFuse | visual_cls_mean_concat | round0_timefuse | raw_soft_fusion | mean_max_weight | 0.524337 | 0.701544 | -0.177207 | False |
| visual_cls_mean_concat vs Round0 TimeFuse | visual_cls_mean_concat | round0_timefuse | raw_soft_fusion | MAE_std | 0.021419 | 0.000000 | 0.021419 | True |
| visual_cls_mean_concat vs Round0 TimeFuse | visual_cls_mean_concat | round0_timefuse | raw_soft_fusion | MSE_std | 90.916281 | 0.000000 | 90.916281 | True |
| visual_cls_mean_concat vs Round0 TimeFuse | visual_cls_mean_concat | round0_timefuse | raw_soft_fusion | regret_to_oracle_std | 0.021419 | 0.000000 | 0.021419 | True |
| visual_cls_mean_concat vs Round0 TimeFuse | visual_cls_mean_concat | round0_timefuse | raw_soft_fusion | oracle_label_accuracy_std | 0.171314 | 0.000000 | 0.171314 | True |
| cls_mean_concat_plus_aux vs Round0 TimeFuse | cls_mean_concat_plus_aux | round0_timefuse | hard_top1 | hard_top1_MAE | 0.467320 | 0.547432 | -0.080113 | True |
| cls_mean_concat_plus_aux vs Round0 TimeFuse | cls_mean_concat_plus_aux | round0_timefuse | hard_top1 | hard_top1_MSE | 245.625781 | 568.559825 | -322.934044 | True |
| cls_mean_concat_plus_aux vs Round0 TimeFuse | cls_mean_concat_plus_aux | round0_timefuse | hard_top1 | hard_top1_regret_to_oracle | 0.127035 | 0.207147 | -0.080113 | True |
| cls_mean_concat_plus_aux vs Round0 TimeFuse | cls_mean_concat_plus_aux | round0_timefuse | hard_top1 | hard_top1_oracle_label_accuracy | 0.432360 | 0.587240 | -0.154880 | False |
| cls_mean_concat_plus_aux vs Round0 TimeFuse | cls_mean_concat_plus_aux | round0_timefuse | raw_soft_fusion | raw_soft_fusion_MAE | 0.452942 | 0.535220 | -0.082279 | True |
| cls_mean_concat_plus_aux vs Round0 TimeFuse | cls_mean_concat_plus_aux | round0_timefuse | raw_soft_fusion | raw_soft_fusion_MSE | 245.459475 | 568.502401 | -323.042927 | True |
| cls_mean_concat_plus_aux vs Round0 TimeFuse | cls_mean_concat_plus_aux | round0_timefuse | raw_soft_fusion | raw_soft_fusion_regret_to_oracle | 0.112657 | 0.194935 | -0.082279 | True |
| cls_mean_concat_plus_aux vs Round0 TimeFuse | cls_mean_concat_plus_aux | round0_timefuse | raw_soft_fusion | raw_soft_fusion_oracle_label_accuracy | 0.432360 | 0.587240 | -0.154880 | False |
| cls_mean_concat_plus_aux vs Round0 TimeFuse | cls_mean_concat_plus_aux | round0_timefuse | raw_soft_fusion | weight_entropy | 1.112345 | 0.730438 | 0.381907 | False |
| cls_mean_concat_plus_aux vs Round0 TimeFuse | cls_mean_concat_plus_aux | round0_timefuse | raw_soft_fusion | normalized_weight_entropy | 0.691139 | 0.453847 | 0.237292 | False |
| cls_mean_concat_plus_aux vs Round0 TimeFuse | cls_mean_concat_plus_aux | round0_timefuse | raw_soft_fusion | mean_max_weight | 0.521259 | 0.701544 | -0.180285 | False |
| cls_mean_concat_plus_aux vs Round0 TimeFuse | cls_mean_concat_plus_aux | round0_timefuse | raw_soft_fusion | MAE_std | 0.039445 | 0.000000 | 0.039445 | True |
| cls_mean_concat_plus_aux vs Round0 TimeFuse | cls_mean_concat_plus_aux | round0_timefuse | raw_soft_fusion | MSE_std | 67.971945 | 0.000000 | 67.971945 | True |
| cls_mean_concat_plus_aux vs Round0 TimeFuse | cls_mean_concat_plus_aux | round0_timefuse | raw_soft_fusion | regret_to_oracle_std | 0.039445 | 0.000000 | 0.039445 | True |
| cls_mean_concat_plus_aux vs Round0 TimeFuse | cls_mean_concat_plus_aux | round0_timefuse | raw_soft_fusion | oracle_label_accuracy_std | 0.154035 | 0.000000 | 0.154035 | True |

## Per-Seed Result

| sample_set | variant | seed | method | sample_count | MAE | MSE | regret_to_oracle | oracle_label_accuracy | weight_entropy | normalized_weight_entropy | mean_max_weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_test | cls_mean_concat_plus_aux | 16 | cls_mean_concat_plus_aux_hard_top1 | 75000 | 0.475915 | 238.280722 | 0.135630 | 0.610200 | 1.118346 | 0.694867 | 0.528712 |
| pilot_test | cls_mean_concat_plus_aux | 16 | cls_mean_concat_plus_aux_raw_soft_fusion | 75000 | 0.461857 | 238.172610 | 0.121572 | 0.610200 | 1.118346 | 0.694867 | 0.528712 |
| pilot_test | cls_mean_concat_plus_aux | 17 | cls_mean_concat_plus_aux_hard_top1 | 75000 | 0.424941 | 181.797077 | 0.084656 | 0.345973 | 1.105651 | 0.686980 | 0.520417 |
| pilot_test | cls_mean_concat_plus_aux | 17 | cls_mean_concat_plus_aux_raw_soft_fusion | 75000 | 0.409803 | 181.424539 | 0.069518 | 0.345973 | 1.105651 | 0.686980 | 0.520417 |
| pilot_test | cls_mean_concat_plus_aux | 18 | cls_mean_concat_plus_aux_hard_top1 | 75000 | 0.501102 | 316.799544 | 0.160817 | 0.340907 | 1.113039 | 0.691570 | 0.514648 |
| pilot_test | cls_mean_concat_plus_aux | 18 | cls_mean_concat_plus_aux_raw_soft_fusion | 75000 | 0.487166 | 316.781275 | 0.146881 | 0.340907 | 1.113039 | 0.691570 | 0.514648 |
| pilot_test | mean_patch_plus_aux | 16 | mean_patch_plus_aux_hard_top1 | 75000 | 0.586362 | 825.100897 | 0.246077 | 0.335413 | 1.108697 | 0.688872 | 0.522932 |
| pilot_test | mean_patch_plus_aux | 16 | mean_patch_plus_aux_raw_soft_fusion | 75000 | 0.571626 | 825.048750 | 0.231341 | 0.335413 | 1.108697 | 0.688872 | 0.522932 |
| pilot_test | visual_mean_patch_only | 16 | visual_mean_patch_only_hard_top1 | 75000 | 0.429297 | 182.412190 | 0.089012 | 0.338507 | 1.162898 | 0.722549 | 0.497990 |
| pilot_test | visual_mean_patch_only | 16 | visual_mean_patch_only_raw_soft_fusion | 75000 | 0.420698 | 183.956838 | 0.080413 | 0.338507 | 1.162898 | 0.722549 | 0.497990 |
| pilot_test | visual_cls_mean_concat | 16 | visual_cls_mean_concat_hard_top1 | 75000 | 0.475890 | 383.047106 | 0.135605 | 0.613760 | 1.130390 | 0.702351 | 0.530807 |
| pilot_test | visual_cls_mean_concat | 16 | visual_cls_mean_concat_raw_soft_fusion | 75000 | 0.467214 | 349.214436 | 0.126929 | 0.613760 | 1.130390 | 0.702351 | 0.530807 |
| pilot_test | mean_patch_plus_aux | 17 | mean_patch_plus_aux_hard_top1 | 75000 | 0.503710 | 317.226720 | 0.163425 | 0.337027 | 1.124447 | 0.698658 | 0.509142 |
| pilot_test | mean_patch_plus_aux | 17 | mean_patch_plus_aux_raw_soft_fusion | 75000 | 0.488110 | 316.548285 | 0.147825 | 0.337027 | 1.124447 | 0.698658 | 0.509142 |
| pilot_test | visual_mean_patch_only | 17 | visual_mean_patch_only_hard_top1 | 75000 | 0.431221 | 182.751343 | 0.090936 | 0.332227 | 1.147755 | 0.713140 | 0.503404 |
| pilot_test | visual_mean_patch_only | 17 | visual_mean_patch_only_raw_soft_fusion | 75000 | 0.434330 | 204.519692 | 0.094045 | 0.332227 | 1.147755 | 0.713140 | 0.503404 |
| pilot_test | visual_cls_mean_concat | 17 | visual_cls_mean_concat_hard_top1 | 75000 | 0.452744 | 207.432664 | 0.112459 | 0.319533 | 1.137317 | 0.706655 | 0.503335 |
| pilot_test | visual_cls_mean_concat | 17 | visual_cls_mean_concat_raw_soft_fusion | 75000 | 0.435602 | 192.649303 | 0.095317 | 0.319533 | 1.137317 | 0.706655 | 0.503335 |
| pilot_test | mean_patch_plus_aux | 18 | mean_patch_plus_aux_hard_top1 | 75000 | 0.503669 | 316.712870 | 0.163384 | 0.615147 | 1.130482 | 0.702408 | 0.512892 |
| pilot_test | mean_patch_plus_aux | 18 | mean_patch_plus_aux_raw_soft_fusion | 75000 | 0.488586 | 316.710523 | 0.148301 | 0.615147 | 1.130482 | 0.702408 | 0.512892 |
| pilot_test | visual_mean_patch_only | 18 | visual_mean_patch_only_hard_top1 | 75000 | 0.529951 | 639.090921 | 0.189667 | 0.342613 | 1.120965 | 0.696495 | 0.514811 |
| pilot_test | visual_mean_patch_only | 18 | visual_mean_patch_only_raw_soft_fusion | 75000 | 0.503899 | 521.982944 | 0.163614 | 0.342613 | 1.120965 | 0.696495 | 0.514811 |
| pilot_test | visual_cls_mean_concat | 18 | visual_cls_mean_concat_hard_top1 | 75000 | 0.424273 | 181.317960 | 0.083988 | 0.618693 | 1.098422 | 0.682488 | 0.538871 |
| pilot_test | visual_cls_mean_concat | 18 | visual_cls_mean_concat_raw_soft_fusion | 75000 | 0.426370 | 190.851723 | 0.086085 | 0.618693 | 1.098422 | 0.682488 | 0.538871 |

## CrossFormer Stratum 摘录

| sample_set | variant | seed | method | stratum_column | stratum_kind | stratum_value | sample_count | MAE | MSE | regret_to_oracle | oracle_label_accuracy | weight_entropy | normalized_weight_entropy | mean_max_weight | dataset_name | oracle_model | error_gap_quantile | cluster | group_name | forecastability_cat | season_strength_cat | trend_strength_cat | cv_cat | missing_ratio_cat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_test | cls_mean_concat_plus_aux | 16 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.635800 | 1.416192 | 0.105149 | 0.111905 | 1.085296 | 0.674332 | 0.544954 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | cls_mean_concat_plus_aux | 17 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.624165 | 1.423400 | 0.093514 | 0.134286 | 1.035823 | 0.643593 | 0.562990 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | cls_mean_concat_plus_aux | 18 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.625234 | 1.437301 | 0.094583 | 0.128893 | 1.020350 | 0.633979 | 0.568769 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | mean_patch_plus_aux | 16 | mean_patch_plus_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.639142 | 1.438215 | 0.108490 | 0.105568 | 1.062983 | 0.660469 | 0.558117 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_mean_patch_only | 16 | visual_mean_patch_only_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.637202 | 1.453442 | 0.106551 | 0.105703 | 1.119787 | 0.695763 | 0.538325 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_cls_mean_concat | 16 | visual_cls_mean_concat_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.629668 | 1.438379 | 0.099017 | 0.122017 | 1.081653 | 0.672069 | 0.550924 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | mean_patch_plus_aux | 17 | mean_patch_plus_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.616115 | 1.401524 | 0.085464 | 0.194958 | 1.070990 | 0.665444 | 0.532647 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_mean_patch_only | 17 | visual_mean_patch_only_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.618395 | 1.385014 | 0.087743 | 0.163004 | 1.070048 | 0.664858 | 0.550830 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_cls_mean_concat | 17 | visual_cls_mean_concat_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.610474 | 1.377995 | 0.079822 | 0.254011 | 1.052087 | 0.653699 | 0.542495 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | mean_patch_plus_aux | 18 | mean_patch_plus_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.634559 | 1.466574 | 0.103908 | 0.071053 | 1.069664 | 0.664620 | 0.553279 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_mean_patch_only | 18 | visual_mean_patch_only_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.638592 | 1.431614 | 0.107941 | 0.066334 | 1.047794 | 0.651031 | 0.566056 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_cls_mean_concat | 18 | visual_cls_mean_concat_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.626052 | 1.462753 | 0.095401 | 0.136848 | 1.049996 | 0.652399 | 0.551801 |  |  |  |  |  |  |  |  |  |  |
| pilot_test |  | -1 | round0_timefuse_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.658768 | 1.466646 | 0.128117 | 0.060941 | 0.858139 | 0.533192 | 0.635861 |  |  |  |  |  |  |  |  |  |  |

## PatchTST Stratum 摘录

| sample_set | variant | seed | method | stratum_column | stratum_kind | stratum_value | sample_count | MAE | MSE | regret_to_oracle | oracle_label_accuracy | weight_entropy | normalized_weight_entropy | mean_max_weight | dataset_name | oracle_model | error_gap_quantile | cluster | group_name | forecastability_cat | season_strength_cat | trend_strength_cat | cv_cat | missing_ratio_cat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_test | cls_mean_concat_plus_aux | 16 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.574480 | 4.451115 | 0.075975 | 0.715472 | 0.928359 | 0.576822 | 0.629596 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | cls_mean_concat_plus_aux | 17 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.581622 | 5.901474 | 0.083117 | 0.758492 | 0.899433 | 0.558849 | 0.637656 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | cls_mean_concat_plus_aux | 18 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.567098 | 6.669592 | 0.068593 | 0.793082 | 0.868870 | 0.539859 | 0.651046 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | mean_patch_plus_aux | 16 | mean_patch_plus_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.592184 | 7.042653 | 0.093679 | 0.702672 | 0.896145 | 0.556806 | 0.644944 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_mean_patch_only | 16 | visual_mean_patch_only_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.586393 | 5.850754 | 0.087888 | 0.723214 | 0.971932 | 0.603895 | 0.618340 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_cls_mean_concat | 16 | visual_cls_mean_concat_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.566604 | 4.247729 | 0.068099 | 0.720030 | 0.929444 | 0.577496 | 0.629562 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | mean_patch_plus_aux | 17 | mean_patch_plus_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.587288 | 5.886750 | 0.088783 | 0.718531 | 0.955746 | 0.593839 | 0.595528 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_mean_patch_only | 17 | visual_mean_patch_only_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.579154 | 5.628166 | 0.080649 | 0.733641 | 0.939264 | 0.583598 | 0.619947 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_cls_mean_concat | 17 | visual_cls_mean_concat_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.592471 | 6.329496 | 0.093966 | 0.668769 | 0.934914 | 0.580895 | 0.605558 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | mean_patch_plus_aux | 18 | mean_patch_plus_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.580862 | 5.802504 | 0.082357 | 0.752498 | 0.924688 | 0.574541 | 0.630311 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_mean_patch_only | 18 | visual_mean_patch_only_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.575612 | 5.628816 | 0.077107 | 0.770355 | 0.913530 | 0.567608 | 0.637138 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | visual_cls_mean_concat | 18 | visual_cls_mean_concat_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.567292 | 4.595499 | 0.068787 | 0.743319 | 0.903251 | 0.561222 | 0.630963 |  |  |  |  |  |  |  |  |  |  |
| pilot_test |  | -1 | round0_timefuse_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.542706 | 3.035039 | 0.044201 | 0.840347 | 0.744881 | 0.462820 | 0.704559 |  |  |  |  |  |  |  |  |  |  |

## 边界记录

- p2d_best_variant_path：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_concat/round1_concat_best_variant.json`
- variant 固定为 `cls_mean_concat_plus_aux`；未训练新模型；未按 pilot_test 改 seed/epoch/hyperparams。
- pilot_test feature cache 独立写入 final_test_only 目录，不覆盖 P2a 原始 feature cache。
- commit hash：`456a3f41712f5f7af7ec41a1dbac462da53a6f99`
