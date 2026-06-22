# Visual Router V2 Round2 staged full-scale report schema

生成时间：2026-06-22 11:06:48 CST

## Overall

输出：`round2_staged_fullscale_overall_report.csv`

必需字段：

| 字段 | 含义 |
| --- | --- |
| `sample_set` | `staged_selection` / `staged_diagnostic` / `staged_test` |
| `layout_name` | layout 名称 |
| `seed_count` | 聚合 seed 数 |
| `sample_count_per_seed` | 每 seed 样本数 |
| `raw_soft_MAE_mean/std` | raw-soft MAE seed mean/std |
| `raw_soft_MSE_mean/std` | raw-soft MSE seed mean/std |
| `raw_soft_regret_mean/std` | raw-soft regret seed mean/std |
| `hard_top1_MAE_mean/std` | hard top1 MAE seed mean/std |
| `hard_top1_MSE_mean/std` | hard top1 MSE seed mean/std |
| `hard_top1_regret_mean/std` | hard top1 regret seed mean/std |
| `oracle_label_accuracy_mean/std` | oracle-label accuracy 解释指标 |
| `entropy_mean/std` | router weight entropy |
| `mean_max_weight_mean/std` | max weight 均值 |
| `raw_soft_vs_hard_top1_MAE_gap_mean/std` | raw-soft 与 hard top1 MAE 差距 |
| `per_seed_metrics` | JSON 字符串，保留每 seed 指标 |

## Strata

输出：`round2_staged_fullscale_strata_report.csv`

固定 strata：

- `oracle_model`
- `dataset_name`
- `group_name`
- `error_gap_quantile`
- `forecastability_cat`
- `season_strength_cat`
- `trend_strength_cat`
- `cv_cat`

每个 stratum 输出 `sample_count`、raw-soft MAE/MSE/regret、hard top1 MAE、entropy、mean max weight 和 oracle-label accuracy。

## Tail

输出：`round2_staged_fullscale_tail_report.csv`

必需字段：

- `top1pct_soft_MAE`
- `top5pct_soft_MAE`
- `top1pct_regret`
- `top5pct_regret`
- `tail_overlap_top1pct_mae_regret_jaccard`
- `tail_oracle_model_distribution`
- `tail_selected_model_distribution`

tail 字段用于监控 high-error/high-regret 样本，不能替代 overall 主指标。

## Router Behavior

输出：`round2_staged_fullscale_router_behavior_report.csv`

必需字段：

- `selected_ratio_DLinear`
- `selected_ratio_PatchTST`
- `selected_ratio_CrossFormer`
- `selected_ratio_ES`
- `selected_ratio_NaiveForecaster`
- `entropy`
- `mean_max_weight`
- `raw_soft_vs_hard_top1_MAE_gap`

## Metadata

输出：`round2_staged_fullscale_metadata.json`

必须记录：

- `sample_scale`
- `layouts`
- `backend`
- `seeds`
- feature manifest 检查结果
- prediction SQLite lookup 检查结果
- `not_1m_run=true`
- `not_116m_full_scale_run=true`
- `loaded_116m_prediction_manifest_to_memory=false`
- `saved_pseudo_image_tensor=false`
- `test_used_for_training_or_selection=false`
