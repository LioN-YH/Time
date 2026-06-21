# Visual Router V2 Round 1 Frozen Final Test Summary

生成时间：2026-06-21 15:27:10 CST

## 核心结论

1. P2d best 是否在 pilot_test raw-soft MAE 上超过 Round0 TimeFuse：是。P2d=0.452942，TimeFuse=0.535220，delta=-0.082279。
2. regret_to_oracle 是否超过 Round0 TimeFuse：是。P2d=0.112657，TimeFuse=0.194935，delta=-0.082279。
3. MSE 是否保留或改善原始 Visual Router 优势：是。P2d raw-soft MSE=245.459475，Round0 Visual raw-soft MSE=510.975952。
4. oracle-label accuracy 是否仍低于 TimeFuse：是。P2d=0.432360，TimeFuse=0.587240；该差异需要结合 MAE/regret 判断，不能单独作为选择依据。
5. raw-soft 是否明显优于 hard top-1：是。P2d hard MAE=0.467320、raw-soft MAE=0.452942；hard regret=0.127035、raw-soft regret=0.112657。
6. CrossFormer / PatchTST strata 见下方分层摘录；完整分层在 `round1_final_test_stratified_summary.csv`。
7. 是否存在 selection 提升、test 退化风险：未观察到相对 TimeFuse 的 test 退化。
8. 是否建议进入 P2e FiLM/conditional modulation：可以作为后续方向，但本次 final eval 不支持用 pilot_test 重新选型。
9. 是否建议进入 Round 2 pseudo image / view layout 消融：建议进入，且继续冻结 pilot_test 只作最终验证。
10. 是否足够支持后续 full-scale-safe pilot rerun：支持作为下一步候选。

## Final Comparison

| sample_set | method | variant | seed_count | sample_count | hard_top1_MAE | hard_top1_MSE | hard_top1_regret_to_oracle | hard_top1_oracle_label_accuracy | raw_soft_fusion_MAE | raw_soft_fusion_MSE | raw_soft_fusion_regret_to_oracle | raw_soft_fusion_oracle_label_accuracy | weight_entropy | normalized_weight_entropy | mean_max_weight | MAE_std | MSE_std | regret_to_oracle_std | oracle_label_accuracy_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_test | cls_mean_concat_plus_aux_hard_top1 | cls_mean_concat_plus_aux | 3 | 75000 | 0.467320 | 245.625781 | 0.127035 | 0.432360 |  |  |  |  | 1.112345 | 0.691139 | 0.521259 | 0.038801 | 67.800287 | 0.038801 | 0.154035 |
| pilot_test | cls_mean_concat_plus_aux_raw_soft_fusion | cls_mean_concat_plus_aux | 3 | 75000 |  |  |  |  | 0.452942 | 245.459475 | 0.112657 | 0.432360 | 1.112345 | 0.691139 | 0.521259 | 0.039445 | 67.971945 | 0.039445 | 0.154035 |
| pilot_test | round0_timefuse_hard_top1 |  | 1 | 75000 | 0.547432 | 568.559825 | 0.207147 | 0.587240 |  |  |  |  | 0.730438 | 0.453847 | 0.701544 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pilot_test | round0_timefuse_raw_soft_fusion |  | 1 | 75000 |  |  |  |  | 0.535220 | 568.502401 | 0.194935 | 0.587240 | 0.730438 | 0.453847 | 0.701544 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pilot_test | round0_original_visual_hard_top1 |  | 1 | 75000 | 0.664912 | 596.442288 | 0.324627 | 0.457960 |  |  |  |  | 1.263102 | 0.784809 | 0.450189 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pilot_test | round0_original_visual_raw_soft_fusion |  | 1 | 75000 |  |  |  |  | 0.603009 | 510.975952 | 0.262724 | 0.457960 | 1.263102 | 0.784809 | 0.450189 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pilot_test | global_best_single |  | 1 | 75000 | 0.599744 |  | 0.259460 | 0.125760 |  |  |  |  |  |  |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pilot_test | oracle_top1 |  | 1 | 75000 | 0.340285 |  | 0.000000 | 1.000000 |  |  |  |  |  |  |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Per-Seed Result

| sample_set | variant | seed | method | sample_count | MAE | MSE | regret_to_oracle | oracle_label_accuracy | weight_entropy | normalized_weight_entropy | mean_max_weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_test | cls_mean_concat_plus_aux | 16 | cls_mean_concat_plus_aux_hard_top1 | 75000 | 0.475915 | 238.280722 | 0.135630 | 0.610200 | 1.118346 | 0.694867 | 0.528712 |
| pilot_test | cls_mean_concat_plus_aux | 16 | cls_mean_concat_plus_aux_raw_soft_fusion | 75000 | 0.461857 | 238.172610 | 0.121572 | 0.610200 | 1.118346 | 0.694867 | 0.528712 |
| pilot_test | cls_mean_concat_plus_aux | 17 | cls_mean_concat_plus_aux_hard_top1 | 75000 | 0.424941 | 181.797077 | 0.084656 | 0.345973 | 1.105651 | 0.686980 | 0.520417 |
| pilot_test | cls_mean_concat_plus_aux | 17 | cls_mean_concat_plus_aux_raw_soft_fusion | 75000 | 0.409803 | 181.424539 | 0.069518 | 0.345973 | 1.105651 | 0.686980 | 0.520417 |
| pilot_test | cls_mean_concat_plus_aux | 18 | cls_mean_concat_plus_aux_hard_top1 | 75000 | 0.501102 | 316.799544 | 0.160817 | 0.340907 | 1.113039 | 0.691570 | 0.514648 |
| pilot_test | cls_mean_concat_plus_aux | 18 | cls_mean_concat_plus_aux_raw_soft_fusion | 75000 | 0.487166 | 316.781275 | 0.146881 | 0.340907 | 1.113039 | 0.691570 | 0.514648 |

## CrossFormer Stratum 摘录

| sample_set | variant | seed | method | stratum_column | stratum_kind | stratum_value | sample_count | MAE | MSE | regret_to_oracle | oracle_label_accuracy | weight_entropy | normalized_weight_entropy | mean_max_weight | dataset_name | oracle_model | error_gap_quantile | cluster | group_name | forecastability_cat | season_strength_cat | trend_strength_cat | cv_cat | missing_ratio_cat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_test | cls_mean_concat_plus_aux | 16 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.635800 | 1.416192 | 0.105149 | 0.111905 | 1.085296 | 0.674332 | 0.544954 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | cls_mean_concat_plus_aux | 17 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.624165 | 1.423400 | 0.093514 | 0.134286 | 1.035823 | 0.643593 | 0.562990 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | cls_mean_concat_plus_aux | 18 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.625234 | 1.437301 | 0.094583 | 0.128893 | 1.020350 | 0.633979 | 0.568769 |  |  |  |  |  |  |  |  |  |  |
| pilot_test |  | -1 | round0_timefuse_raw_soft_fusion | oracle_model | single_column | CrossFormer | 7417 | 0.658768 | 1.466646 | 0.128117 | 0.060941 | 0.858139 | 0.533192 | 0.635861 |  |  |  |  |  |  |  |  |  |  |

## PatchTST Stratum 摘录

| sample_set | variant | seed | method | stratum_column | stratum_kind | stratum_value | sample_count | MAE | MSE | regret_to_oracle | oracle_label_accuracy | weight_entropy | normalized_weight_entropy | mean_max_weight | dataset_name | oracle_model | error_gap_quantile | cluster | group_name | forecastability_cat | season_strength_cat | trend_strength_cat | cv_cat | missing_ratio_cat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_test | cls_mean_concat_plus_aux | 16 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.574480 | 4.451115 | 0.075975 | 0.715472 | 0.928359 | 0.576822 | 0.629596 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | cls_mean_concat_plus_aux | 17 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.581622 | 5.901474 | 0.083117 | 0.758492 | 0.899433 | 0.558849 | 0.637656 |  |  |  |  |  |  |  |  |  |  |
| pilot_test | cls_mean_concat_plus_aux | 18 | cls_mean_concat_plus_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.567098 | 6.669592 | 0.068593 | 0.793082 | 0.868870 | 0.539859 | 0.651046 |  |  |  |  |  |  |  |  |  |  |
| pilot_test |  | -1 | round0_timefuse_raw_soft_fusion | oracle_model | single_column | PatchTST | 16016 | 0.542706 | 3.035039 | 0.044201 | 0.840347 | 0.744881 | 0.462820 | 0.704559 |  |  |  |  |  |  |  |  |  |  |

## 边界记录

- p2d_best_variant_path：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_concat/round1_concat_best_variant.json`
- variant 固定为 `cls_mean_concat_plus_aux`；未训练新模型；未按 pilot_test 改 seed/epoch/hyperparams。
- pilot_test feature cache 独立写入 final_test_only 目录，不覆盖 P2a 原始 feature cache。
- commit hash：`5b21c8246be44da9a7018c128fa6f9465a66a5c2`
