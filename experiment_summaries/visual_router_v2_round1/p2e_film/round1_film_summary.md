# Visual Router V2 Round 1 P2e FiLM / Aux Modulation Summary

生成时间：2026-06-21 17:31:36 CST

## 结论回答

1. FiLM 是否优于 `visual_cls_mean_concat`？是。film_cls_mean_concat_aux selection raw-soft MAE=0.300486 vs visual_cls_mean_concat=0.302220，delta=-0.001734。
2. FiLM 是否优于 `cls_mean_concat_plus_aux`？是。film_cls_mean_concat_aux MAE=0.300486 vs cls_mean_concat_plus_aux=0.300605，delta=-0.000120。
3. FiLM 是否优于 `visual_mean_patch_only`？是。film_mean_patch_aux MAE=0.300393 vs visual_mean_patch_only=0.300996，delta=-0.000603。
4. mean_patch 路线用 FiLM 是否比直接 concat aux 更好？是。film_mean_patch_aux MAE=0.300393 vs mean_patch_plus_aux=0.300831，delta=-0.000438。
5. FiLM 是否改善 seed stability？是。film_cls_mean_concat_aux MAE_std=0.000859 vs visual_cls_mean_concat=0.003929。
6. FiLM 是否改善 MSE tail？否。film_cls_mean_concat_aux MSE_mean=1.313162 vs visual_cls_mean_concat=1.217317。
7. FiLM 是否改善 CrossFormer / PatchTST strata？见下方 oracle_model 分层表；判断以 `round1_film_stratified_summary.csv` 的三 seed 均值为准。
8. 是否值得进入下一步 frozen pilot_test eval extension？值得；本轮未使用也未评估 pilot_test。

## Oracle Model Strata

| sample_set | variant | seed | method | stratum_column | stratum_kind | stratum_value | sample_count | MAE | MSE | regret_to_oracle | oracle_label_accuracy | weight_entropy | normalized_weight_entropy | mean_max_weight | dataset_name | oracle_model | error_gap_quantile | cluster | group_name | forecastability_cat | season_strength_cat | trend_strength_cat | cv_cat | missing_ratio_cat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_selection | film_cls_mean_concat_aux | 18 | film_cls_mean_concat_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 2893 | 0.476672 | 0.758771 | 0.036685 | 0.206015 | 1.036832 | 0.644220 | 0.551244 |  |  |  |  |  |  |  |  |  |  |
| pilot_selection | film_cls_mean_concat_aux | 16 | film_cls_mean_concat_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 2893 | 0.479963 | 0.712562 | 0.039975 | 0.191842 | 1.100307 | 0.683659 | 0.524456 |  |  |  |  |  |  |  |  |  |  |
| pilot_selection | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 2893 | 0.480648 | 0.728294 | 0.040661 | 0.160387 | 1.080503 | 0.671354 | 0.535702 |  |  |  |  |  |  |  |  |  |  |
| pilot_selection | film_cls_mean_concat_aux | 17 | film_cls_mean_concat_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 2893 | 0.481465 | 0.706049 | 0.041478 | 0.143104 | 1.035622 | 0.643468 | 0.562913 |  |  |  |  |  |  |  |  |  |  |
| pilot_selection | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 2893 | 0.484540 | 0.761824 | 0.044553 | 0.146561 | 1.048650 | 0.651563 | 0.553682 |  |  |  |  |  |  |  |  |  |  |
| pilot_selection | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | oracle_model | single_column | CrossFormer | 2893 | 0.487275 | 0.737402 | 0.047288 | 0.144141 | 1.046938 | 0.650499 | 0.556814 |  |  |  |  |  |  |  |  |  |  |
| pilot_selection | film_cls_mean_concat_aux | 18 | film_cls_mean_concat_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 4822 | 0.395170 | 0.700289 | 0.023359 | 0.440689 | 1.005078 | 0.624490 | 0.553170 |  |  |  |  |  |  |  |  |  |  |
| pilot_selection | film_mean_patch_aux | 16 | film_mean_patch_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 4822 | 0.396447 | 0.711506 | 0.024636 | 0.402945 | 1.030600 | 0.640348 | 0.545169 |  |  |  |  |  |  |  |  |  |  |
| pilot_selection | film_mean_patch_aux | 18 | film_mean_patch_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 4822 | 0.399069 | 0.706587 | 0.027258 | 0.381377 | 1.015963 | 0.631253 | 0.556117 |  |  |  |  |  |  |  |  |  |  |
| pilot_selection | film_cls_mean_concat_aux | 16 | film_cls_mean_concat_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 4822 | 0.400241 | 0.708613 | 0.028430 | 0.394235 | 1.045432 | 0.649564 | 0.537211 |  |  |  |  |  |  |  |  |  |  |
| pilot_selection | film_mean_patch_aux | 17 | film_mean_patch_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 4822 | 0.404093 | 0.708332 | 0.032282 | 0.346951 | 0.995083 | 0.618280 | 0.568081 |  |  |  |  |  |  |  |  |  |  |
| pilot_selection | film_cls_mean_concat_aux | 17 | film_cls_mean_concat_aux_raw_soft_fusion | oracle_model | single_column | PatchTST | 4822 | 0.404282 | 0.721696 | 0.032471 | 0.364372 | 0.999332 | 0.620920 | 0.565296 |  |  |  |  |  |  |  |  |  |  |

## Pilot Selection Mean/Std

| sample_set | variant | method | seed_count | sample_count_per_seed | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_selection | film_mean_patch_aux | film_mean_patch_aux_raw_soft_fusion | 3 | 30000 | 0.300393 | 0.000542 | 1.289872 | 0.164650 | 0.033662 | 0.000542 | 0.552167 | 0.148492 | 1.135975 | 0.020092 | 0.705821 | 0.012484 | 0.502353 | 0.006659 |
| pilot_selection | film_cls_mean_concat_aux | film_cls_mean_concat_aux_raw_soft_fusion | 3 | 30000 | 0.300486 | 0.000859 | 1.313162 | 0.168122 | 0.033754 | 0.000859 | 0.465378 | 0.147915 | 1.129449 | 0.028893 | 0.701766 | 0.017952 | 0.505618 | 0.013839 |
| pilot_selection | film_mean_patch_aux | film_mean_patch_aux_hard_top1 | 3 | 30000 | 0.317828 | 0.000429 | 1.378600 | 0.155791 | 0.051097 | 0.000429 | 0.552167 | 0.148492 | 1.135975 | 0.020092 | 0.705821 | 0.012484 | 0.502353 | 0.006659 |
| pilot_selection | film_cls_mean_concat_aux | film_cls_mean_concat_aux_hard_top1 | 3 | 30000 | 0.318243 | 0.000603 | 1.364084 | 0.166895 | 0.051512 | 0.000603 | 0.465378 | 0.147915 | 1.129449 | 0.028893 | 0.701766 | 0.017952 | 0.505618 | 0.013839 |

## Diagnostic Balanced Mean/Std

| sample_set | variant | method | seed_count | sample_count_per_seed | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | regret_to_oracle_std | oracle_label_accuracy_mean | oracle_label_accuracy_std | weight_entropy_mean | weight_entropy_std | normalized_weight_entropy_mean | normalized_weight_entropy_std | mean_max_weight_mean | mean_max_weight_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diagnostic_balanced | film_mean_patch_aux | film_mean_patch_aux_raw_soft_fusion | 3 | 20000 | 0.345809 | 0.001276 | 1.394350 | 0.012115 | 0.040319 | 0.001276 | 0.453950 | 0.073645 | 1.060335 | 0.021147 | 0.658823 | 0.013140 | 0.542973 | 0.009147 |
| diagnostic_balanced | film_cls_mean_concat_aux | film_cls_mean_concat_aux_raw_soft_fusion | 3 | 20000 | 0.346838 | 0.004585 | 1.407221 | 0.003519 | 0.041348 | 0.004585 | 0.400700 | 0.100132 | 1.049559 | 0.036832 | 0.652128 | 0.022885 | 0.547955 | 0.018009 |
| diagnostic_balanced | film_mean_patch_aux | film_mean_patch_aux_hard_top1 | 3 | 20000 | 0.370468 | 0.001834 | 1.465920 | 0.003500 | 0.064977 | 0.001834 | 0.453950 | 0.073645 | 1.060335 | 0.021147 | 0.658823 | 0.013140 | 0.542973 | 0.009147 |
| diagnostic_balanced | film_cls_mean_concat_aux | film_cls_mean_concat_aux_hard_top1 | 3 | 20000 | 0.371248 | 0.004565 | 1.469448 | 0.004283 | 0.065758 | 0.004565 | 0.400700 | 0.100132 | 1.049559 | 0.036832 | 0.652128 | 0.022885 | 0.547955 | 0.018009 |

## Delta Summary

| delta_name | sample_set | method_kind | left_variant | right_variant | left_MAE_mean | right_MAE_mean | delta_MAE_mean | left_MAE_std | right_MAE_std | delta_MAE_std | left_MSE_mean | right_MSE_mean | delta_MSE_mean | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| film_cls_mean_concat_aux - visual_cls_mean_concat | pilot_selection | raw_soft_fusion | film_cls_mean_concat_aux | visual_cls_mean_concat | 0.300486 | 0.302220 | -0.001734 | 0.000859 | 0.003929 | -0.003071 | 1.313162 | 1.217317 | 0.095845 | ok |
| film_cls_mean_concat_aux - cls_mean_concat_plus_aux | pilot_selection | raw_soft_fusion | film_cls_mean_concat_aux | cls_mean_concat_plus_aux | 0.300486 | 0.300605 | -0.000120 | 0.000859 | 0.001287 | -0.000429 | 1.313162 | 1.205401 | 0.107761 | ok |
| film_mean_patch_aux - visual_mean_patch_only | pilot_selection | raw_soft_fusion | film_mean_patch_aux | visual_mean_patch_only | 0.300393 | 0.300996 | -0.000603 | 0.000542 | 0.001000 | -0.000459 | 1.289872 | 1.234168 | 0.055704 | ok |
| film_mean_patch_aux - mean_patch_plus_aux | pilot_selection | raw_soft_fusion | film_mean_patch_aux | mean_patch_plus_aux | 0.300393 | 0.300831 | -0.000438 | 0.000542 | 0.000548 | -0.000006 | 1.289872 | 1.239938 | 0.049934 | ok |
| film_cls_mean_concat_aux - film_mean_patch_aux | pilot_selection | raw_soft_fusion | film_cls_mean_concat_aux | film_mean_patch_aux | 0.300486 | 0.300393 | 0.000092 | 0.000859 | 0.000542 | 0.000317 | 1.313162 | 1.289872 | 0.023290 | ok |
| film_cls_mean_concat_aux - Round0 TimeFuse | pilot_selection | raw_soft_fusion | film_cls_mean_concat_aux | Round0 TimeFuse | 0.300486 | 0.317530 | -0.017044 | 0.000859 | 0.000000 | 0.000859 | 1.313162 | 1.370167 | -0.057005 | ok |
| film_mean_patch_aux - Round0 TimeFuse | pilot_selection | raw_soft_fusion | film_mean_patch_aux | Round0 TimeFuse | 0.300393 | 0.317530 | -0.017137 | 0.000542 | 0.000000 | 0.000542 | 1.289872 | 1.370167 | -0.080295 | ok |

## Best FiLM Variant

- best_variant：`film_mean_patch_aux`
- selection_basis：pilot_selection raw-soft MAE_mean; tie-breakers MAE_std, regret_to_oracle_mean, MSE_mean, weight_entropy_std, mean_max_weight_std
- 本轮只训练 pilot_train，只用 pilot_selection 选择，diagnostic_balanced 只诊断；pilot_test_evaluated=false。
