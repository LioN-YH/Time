# Visual Router V2 Round2 panel pooling 35k small screening

生成时间：2026-06-22 14:55:46 CST

## 实验边界

- 本结果是 Round2 exploration 的 `spatial_panel_3view` panel-wise / view-region pooling 35k small screening。
- 训练集为 `round2_train_small` 20,000 vali；selection/diagnostic/test_small 各 5,000。
- selection 只按 `round2_selection_small` raw-soft MAE 选择 variant；`round2_test_small` 只作为 frozen screening。
- 未做 full-scale validation，未启动 1M/116M 长跑，未修改 full-scale pipeline，未保存 pseudo image tensor。

## 结论

- selection raw-soft MAE 最优是 `film_mean_patch_aux`：0.310385；baseline `film_mean_patch_aux` 为 0.310385。
- `film_global_panel_mean_aux` 相对 baseline 的 selection raw-soft MAE delta 为 0.000578，MSE delta 为 0.215574，regret delta 为 0.000578。
- `film_panel_mean_aux` 相对 baseline 的 selection raw-soft MAE delta 为 0.001768，MSE delta 为 0.251441，regret delta 为 0.001768。
- 升级判断：Keep as side branch; do not enter 65k from this 35k screening。test_small 虽然 panel variants raw-soft 略好，但不得用于选择 variant；selection 上 panel variants 未稳定优于 baseline。
- 建议：35k screening 后暂不进入 65k expanded validation；panel-wise pooling 保留为 architecture side branch，full-scale 并行主线不受本结果影响。

## Overall Raw-Soft Summary

### Selection
| variant | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | weight_entropy_mean | mean_max_weight_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| film_mean_patch_aux | 0.310385 | 0.008199 | 3.329199 | 0.573350 | 0.046935 | 0.008199 | 0.538000 | 1.135332 | 0.515005 |
| film_global_panel_mean_aux | 0.310962 | 0.005712 | 3.544773 | 0.473582 | 0.047512 | 0.005712 | 0.457467 | 1.084603 | 0.542975 |
| film_panel_mean_aux | 0.312153 | 0.005447 | 3.580640 | 0.523736 | 0.048703 | 0.005447 | 0.538533 | 1.116768 | 0.515510 |


### Diagnostic Balanced
| variant | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | weight_entropy_mean | mean_max_weight_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| film_panel_mean_aux | 0.368445 | 0.002679 | 2.685900 | 0.977037 | 0.062352 | 0.002679 | 0.451933 | 1.035628 | 0.559990 |
| film_global_panel_mean_aux | 0.369304 | 0.002499 | 2.884644 | 0.827879 | 0.063211 | 0.002499 | 0.393867 | 1.007850 | 0.582771 |
| film_mean_patch_aux | 0.370516 | 0.005574 | 3.388182 | 0.944727 | 0.064422 | 0.005574 | 0.449800 | 1.054758 | 0.555304 |


### Frozen Test Small
| variant | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | weight_entropy_mean | mean_max_weight_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| film_global_panel_mean_aux | 0.396863 | 0.000574 | 3.467099 | 0.027124 | 0.063514 | 0.000574 | 0.431267 | 1.073060 | 0.550976 |
| film_panel_mean_aux | 0.397086 | 0.000618 | 3.471120 | 0.014760 | 0.063737 | 0.000618 | 0.522867 | 1.081380 | 0.541671 |
| film_mean_patch_aux | 0.398598 | 0.001161 | 3.484102 | 0.009068 | 0.065249 | 0.001161 | 0.518667 | 1.086936 | 0.549078 |


## Delta vs Baseline

| split | variant | MAE_delta_vs_baseline | MSE_delta_vs_baseline | regret_delta_vs_baseline | MAE_std | oracle_label_accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- |
| selection | film_global_panel_mean_aux | 0.000578 | 0.215574 | 0.000578 | 0.005712 | 0.457467 |
| selection | film_panel_mean_aux | 0.001768 | 0.251441 | 0.001768 | 0.005447 | 0.538533 |
| diagnostic | film_panel_mean_aux | -0.002071 | -0.702282 | -0.002071 | 0.002679 | 0.451933 |
| diagnostic | film_global_panel_mean_aux | -0.001211 | -0.503538 | -0.001211 | 0.002499 | 0.393867 |
| test_small | film_global_panel_mean_aux | -0.001735 | -0.017002 | -0.001735 | 0.000574 | 0.431267 |
| test_small | film_panel_mean_aux | -0.001513 | -0.012982 | -0.001513 | 0.000618 | 0.522867 |


## Hard Top1 Selection Snapshot

| variant | MAE_mean | MAE_std | MSE_mean | regret_to_oracle_mean | oracle_label_accuracy_mean |
| --- | --- | --- | --- | --- | --- |
| film_mean_patch_aux | 0.328295 | 0.008512 | 3.450927 | 0.064845 | 0.538000 |
| film_panel_mean_aux | 0.329898 | 0.005509 | 3.706792 | 0.066448 | 0.538533 |
| film_global_panel_mean_aux | 0.330073 | 0.004882 | 3.741217 | 0.066623 | 0.457467 |


## Selected Model Ratio

selection 上三类 variant 的平均 selected_model ratio 如下；panel/global+panel 的 seed 间分配波动较明显，尤其 CrossFormer/ES/NaiveForecaster 比例随 seed 摆动。
| variant | CrossFormer | DLinear | ES | NaiveForecaster | PatchTST |
| --- | --- | --- | --- | --- | --- |
| film_global_panel_mean_aux | 0.133867 | 0.443200 | 0.157067 | 0.135067 | 0.130800 |
| film_mean_patch_aux | 0.069800 | 0.404200 | 0.238200 | 0.133467 | 0.154333 |
| film_panel_mean_aux | 0.157400 | 0.419000 | 0.235600 | 0.044933 | 0.143067 |


## Key Strata

下表只列 selection 中 panel variants 相对 baseline 的关键 strata delta，包括 oracle_model=CrossFormer/PatchTST/ES/DLinear、error_gap_quantile=q5 和 group_name=LOW_LOW_HIGH。负数表示好于 baseline。
| stratum_column | stratum_value | variant | sample_count | MAE_mean | MAE_delta_vs_baseline | MSE_delta_vs_baseline | regret_delta_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| error_gap_quantile | q5 | film_global_panel_mean_aux | 811.000000 | 0.620596 | 0.009124 | 1.327428 | 0.009124 |
| error_gap_quantile | q5 | film_panel_mean_aux | 811.000000 | 0.627609 | 0.016137 | 1.548404 | 0.016137 |
| group_name | LOW_LOW_HIGH | film_global_panel_mean_aux | 210.000000 | 0.149697 | -0.001836 | -0.002328 | -0.001836 |
| group_name | LOW_LOW_HIGH | film_panel_mean_aux | 210.000000 | 0.151306 | -0.000227 | 0.002646 | -0.000227 |
| oracle_model | CrossFormer | film_global_panel_mean_aux | 479.000000 | 0.488467 | 0.007399 | 0.018126 | 0.007399 |
| oracle_model | CrossFormer | film_panel_mean_aux | 479.000000 | 0.481220 | 0.000152 | 0.002049 | 0.000152 |
| oracle_model | DLinear | film_global_panel_mean_aux | 1547.000000 | 0.368280 | -0.006843 | -0.008357 | -0.006843 |
| oracle_model | DLinear | film_panel_mean_aux | 1547.000000 | 0.371176 | -0.003947 | -0.007096 | -0.003947 |
| oracle_model | ES | film_global_panel_mean_aux | 1774.000000 | 0.140306 | -0.000522 | 0.039610 | -0.000522 |
| oracle_model | ES | film_panel_mean_aux | 1774.000000 | 0.139068 | -0.001761 | -0.047896 | -0.001761 |
| oracle_model | PatchTST | film_global_panel_mean_aux | 814.000000 | 0.465256 | 0.019266 | 1.244899 | 0.019266 |
| oracle_model | PatchTST | film_panel_mean_aux | 814.000000 | 0.470082 | 0.024092 | 1.659027 | 0.024092 |


## 文件归档

- `panel_pooling_35k_selection_summary.csv`
- `panel_pooling_35k_diagnostic_summary.csv`
- `panel_pooling_35k_test_small_summary.csv`
- `panel_pooling_35k_selected_model_counts.csv`
- `panel_pooling_35k_selected_model_ratio_summary.csv`
- `panel_pooling_35k_stratified_summary.csv`
- `panel_pooling_35k_key_strata_delta.csv`
- `panel_pooling_35k_metadata.json`
