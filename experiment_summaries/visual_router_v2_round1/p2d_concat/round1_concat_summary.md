# Visual Router V2 Round 1 P2d Visual+Aux Concat Summary

生成时间：2026-06-21 11:44:38 CST

## 结论回答

1. mean_patch_plus_aux 是否优于 P2b visual_mean_patch_only？是。selection raw-soft MAE=0.300831 vs 0.300996，delta=-0.000165；hard MAE=0.318148。
2. cls_mean_concat_plus_aux 是否优于 P2b visual_cls_mean_concat？是。selection raw-soft MAE=0.300605 vs 0.302220，delta=-0.001615；hard MAE=0.318751。
3. RevIN aux 与 visual embedding 是否存在互补？存在可测互补。P2c aux-only raw-soft MAE=0.332987，P1 visual baseline=0.334069，但最终判断以 concat 是否超过对应 visual-only 为准。
4. aux 的主要作用：当前从 selection 看主要体现在 改善 MAE/regret；entropy、selected_model 稳定性需结合下表和 counts 文件查看。
5. cls_mean_concat_plus_aux 是否缓解 cls_mean_concat seed 不稳定？是。raw-soft MAE_std=0.001287 vs P2b cls_mean_concat=0.003929。
6. CrossFormer / PatchTST 相关 strata 是否改善？CrossFormer 未明显改善（seed16 cls+mean raw-soft MAE 0.479498 vs P2b 0.477138），PatchTST 有改善（0.397557 vs P2b 0.398446）。完整三 seed 见 `round1_concat_stratified_summary.csv`。
7. Round 1 最终 best variant：`cls_mean_concat_plus_aux`（stage=P2d），只按 pilot_selection raw-soft MAE_mean 选择。
8. 是否建议做 pilot_test final eval：建议仅在确认 Round 1 best 后做一次冻结 final eval；pilot_test 不能参与 variant/seed/epoch 选择。
9. 是否建议进入 Round 2 pseudo image / view layout 消融：建议进入，因为当前 best 仍依赖 P2a visual embedding 质量。
10. 是否值得后续单独开 P2e 探索 FiLM/gating/conditional modulation：值得，但应单独开 P2e；本 P2d 未做这些结构。

## Pilot Selection Mean/Std

| sample_set | variant | method | seed_count | sample_count_per_seed | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_selection | cls_mean_concat_plus_aux | cls_mean_concat_plus_aux_raw_soft_fusion | 3 | 30000 | 0.300605 | 0.001287 | 1.205401 | 0.058046 | 0.033874 | 0.001287 | 0.467911 | 0.146558 | 1.118137 | 0.016793 | 0.694737 | 0.010434 | 0.513148 | 0.006964 |
| pilot_selection | mean_patch_plus_aux | mean_patch_plus_aux_raw_soft_fusion | 3 | 30000 | 0.300831 | 0.000548 | 1.239938 | 0.182619 | 0.034100 | 0.000548 | 0.468133 | 0.148860 | 1.123557 | 0.017389 | 0.698105 | 0.010804 | 0.510815 | 0.010831 |
| pilot_selection | mean_patch_plus_aux | mean_patch_plus_aux_hard_top1 | 3 | 30000 | 0.318148 | 0.000756 | 1.306914 | 0.196183 | 0.051417 | 0.000756 | 0.468133 | 0.148860 | 1.123557 | 0.017389 | 0.698105 | 0.010804 | 0.510815 | 0.010831 |
| pilot_selection | cls_mean_concat_plus_aux | cls_mean_concat_plus_aux_hard_top1 | 3 | 30000 | 0.318751 | 0.001585 | 1.339381 | 0.161119 | 0.052020 | 0.001585 | 0.467911 | 0.146558 | 1.118137 | 0.016793 | 0.694737 | 0.010434 | 0.513148 | 0.006964 |

## Diagnostic Balanced Mean/Std

| sample_set | variant | method | seed_count | sample_count_per_seed | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diagnostic_balanced | cls_mean_concat_plus_aux | cls_mean_concat_plus_aux_raw_soft_fusion | 3 | 20000 | 0.346514 | 0.001458 | 1.370437 | 0.033517 | 0.041024 | 0.001458 | 0.411800 | 0.073895 | 1.035812 | 0.017886 | 0.643586 | 0.011113 | 0.554780 | 0.006442 |
| diagnostic_balanced | mean_patch_plus_aux | mean_patch_plus_aux_raw_soft_fusion | 3 | 20000 | 0.346638 | 0.001640 | 1.378566 | 0.020956 | 0.041148 | 0.001640 | 0.404933 | 0.094110 | 1.042023 | 0.013289 | 0.647445 | 0.008257 | 0.552775 | 0.007238 |
| diagnostic_balanced | cls_mean_concat_plus_aux | cls_mean_concat_plus_aux_hard_top1 | 3 | 20000 | 0.371844 | 0.001548 | 1.466458 | 0.013000 | 0.066353 | 0.001548 | 0.411800 | 0.073895 | 1.035812 | 0.017886 | 0.643586 | 0.011113 | 0.554780 | 0.006442 |
| diagnostic_balanced | mean_patch_plus_aux | mean_patch_plus_aux_hard_top1 | 3 | 20000 | 0.372101 | 0.002317 | 1.468723 | 0.011989 | 0.066610 | 0.002317 | 0.404933 | 0.094110 | 1.042023 | 0.013289 | 0.647445 | 0.008257 | 0.552775 | 0.007238 |

## Best Variant

- best_variant：`cls_mean_concat_plus_aux`
- selection_basis：pilot_selection raw_soft_fusion MAE_mean; tie-breakers MAE_std, regret_to_oracle_mean, weight_entropy_std, mean_max_weight_std
- 本轮未使用 pilot_test；未训练 ViT；未修改 P2a builder/schema；未做 FiLM/gating/attention。
