# Visual Router V2 Round 1 P2b Visual Pooling Summary

生成时间：2026-06-21 09:29:34 CST

## 结论回答

1. mean_patch 是否优于 CLS？是。pilot_selection raw-soft MAE：mean_patch=0.300996，CLS=0.302048。
2. CLS+mean concat 是否优于单一 pooling？否。concat raw-soft MAE=0.302220，最佳单一 pooling=0.300996。
3. visual-only pooling 变体相对 P1 Round 0 Visual baseline 是否有改善？是。best=visual_mean_patch_only，raw-soft MAE=0.300996 vs Round0 Visual raw-soft=0.334069；hard MAE=0.319706 vs Round0 Visual hard=0.356267。
4. 是否建议后续 visual+aux concat 使用哪个 visual pooling？建议使用 `visual_mean_patch_only`，依据为 pilot_selection raw-soft MAE mean 最低；diagnostic_balanced 未参与选择。

## Pilot Selection Mean/Std

| sample_set | variant | method | seed_count | sample_count_per_seed | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_selection | visual_mean_patch_only | visual_mean_patch_only_raw_soft_fusion | 3 | 30000 | 0.300996 | 0.001000 | 1.234168 | 0.163449 | 0.034265 | 0.001000 | 0.378856 | 0.006665 | 1.132446 | 0.019601 | 0.703628 | 0.012179 | 0.504598 | 0.008922 |
| pilot_selection | visual_cls_only | visual_cls_only_raw_soft_fusion | 3 | 30000 | 0.302048 | 0.001219 | 1.220514 | 0.109631 | 0.035317 | 0.001219 | 0.376122 | 0.006574 | 1.127909 | 0.020945 | 0.700809 | 0.013014 | 0.508793 | 0.008400 |
| pilot_selection | visual_cls_mean_concat | visual_cls_mean_concat_raw_soft_fusion | 3 | 30000 | 0.302220 | 0.003929 | 1.217317 | 0.118708 | 0.035489 | 0.003929 | 0.549878 | 0.159907 | 1.112957 | 0.023586 | 0.691519 | 0.014655 | 0.523813 | 0.011781 |
| pilot_selection | visual_mean_patch_only | visual_mean_patch_only_hard_top1 | 3 | 30000 | 0.319706 | 0.001077 | 1.351444 | 0.193305 | 0.052975 | 0.001077 | 0.378856 | 0.006665 | 1.132446 | 0.019601 | 0.703628 | 0.012179 | 0.504598 | 0.008922 |
| pilot_selection | visual_cls_only | visual_cls_only_hard_top1 | 3 | 30000 | 0.320451 | 0.000508 | 1.364566 | 0.191218 | 0.053719 | 0.000508 | 0.376122 | 0.006574 | 1.127909 | 0.020945 | 0.700809 | 0.013014 | 0.508793 | 0.008400 |
| pilot_selection | visual_cls_mean_concat | visual_cls_mean_concat_hard_top1 | 3 | 30000 | 0.320836 | 0.003389 | 1.391806 | 0.176839 | 0.054105 | 0.003389 | 0.549878 | 0.159907 | 1.112957 | 0.023586 | 0.691519 | 0.014655 | 0.523813 | 0.011781 |

## Diagnostic Balanced Mean/Std

| sample_set | variant | method | seed_count | sample_count_per_seed | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diagnostic_balanced | visual_mean_patch_only | visual_mean_patch_only_raw_soft_fusion | 3 | 20000 | 0.346525 | 0.003262 | 1.388554 | 0.016225 | 0.041035 | 0.003262 | 0.356317 | 0.023181 | 1.051701 | 0.020309 | 0.653459 | 0.012619 | 0.548390 | 0.008799 |
| diagnostic_balanced | visual_cls_mean_concat | visual_cls_mean_concat_raw_soft_fusion | 3 | 20000 | 0.348457 | 0.008445 | 1.359039 | 0.023454 | 0.042966 | 0.008445 | 0.461067 | 0.105146 | 1.036698 | 0.024174 | 0.644137 | 0.015020 | 0.559072 | 0.009164 |
| diagnostic_balanced | visual_cls_only | visual_cls_only_raw_soft_fusion | 3 | 20000 | 0.349310 | 0.004724 | 1.369623 | 0.055323 | 0.043820 | 0.004724 | 0.350133 | 0.029228 | 1.043780 | 0.020659 | 0.648537 | 0.012836 | 0.553225 | 0.007214 |
| diagnostic_balanced | visual_mean_patch_only | visual_mean_patch_only_hard_top1 | 3 | 20000 | 0.372762 | 0.004210 | 1.477363 | 0.015128 | 0.067272 | 0.004210 | 0.356317 | 0.023181 | 1.051701 | 0.020309 | 0.653459 | 0.012619 | 0.548390 | 0.008799 |
| diagnostic_balanced | visual_cls_mean_concat | visual_cls_mean_concat_hard_top1 | 3 | 20000 | 0.373441 | 0.007979 | 1.456282 | 0.034102 | 0.067951 | 0.007979 | 0.461067 | 0.105146 | 1.036698 | 0.024174 | 0.644137 | 0.015020 | 0.559072 | 0.009164 |
| diagnostic_balanced | visual_cls_only | visual_cls_only_hard_top1 | 3 | 20000 | 0.374926 | 0.005695 | 1.449180 | 0.038335 | 0.069435 | 0.005695 | 0.350133 | 0.029228 | 1.043780 | 0.020659 | 0.648537 | 0.012836 | 0.553225 | 0.007214 |

## Best Variant

- best_variant：`visual_mean_patch_only`
- selection_basis：pilot_selection raw_soft_fusion MAE_mean; tie-breakers hard_top1 MAE_mean, regret_to_oracle_mean, oracle_label_accuracy_mean
- 本轮未使用 pilot_test；未训练 ViT；未使用 RevIN aux 或 visual+aux concat。
