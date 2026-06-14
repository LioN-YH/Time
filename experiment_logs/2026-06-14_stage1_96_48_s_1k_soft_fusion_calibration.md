# Stage 1 `96_48_S` 1k Soft Fusion Calibration

日志日期：2026-06-14 17:54:38 CST

## 目的

对 `96_48_S` 1k online visual router 输出的五专家权重做 soft fusion calibration，比较 raw soft、top-1 hard、top-2/top-3 fusion 和 temperature sweep，确定当前 1k 设置下最优的可部署融合策略。

## 背景

最终 online visual router 输出目录为：

```text
experiment_logs/run_outputs/2026-06-14_175036_visual_router_stage1_online_visual_router_96_48_s_1k_local_only/
```

该 run 已验证 `local_files_only=True`、online embedding 仅运行内存暂存、不保存 `.npy` 或伪图像 tensor。未校准 raw soft fusion MAE 为 0.437221，hard top-1 MAE 为 0.459729。

## 操作

运行 calibration：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  /home/shiyuhong/Time/visual_router_experiments/stage1_vali_test_router/evaluate_soft_fusion_calibration.py \
  --router-predictions-path experiment_logs/run_outputs/2026-06-14_175036_visual_router_stage1_online_visual_router_96_48_s_1k_local_only/visual_router_predictions.csv \
  --prediction-manifest-path experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/merged_cache/manifest.csv \
  --labels-path experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/merged_cache/window_oracle_labels_with_tsf_cell.csv \
  --metric mae \
  --output-dir experiment_logs/run_outputs/2026-06-14_175338_visual_router_stage1_soft_fusion_calibration_96_48_s_1k
```

完成后检查：

- `soft_fusion_calibration_summary.csv`
- `soft_fusion_calibration_predictions.csv`
- `soft_fusion_calibration_selected_model_counts.csv`
- `soft_fusion_calibration_metadata.json`
- `soft_fusion_calibration_comparison.csv`
- `soft_fusion_calibration_summary.md`

并校验每个 calibration strategy 覆盖 500 个 test sample_key。

## 结果

输出目录：

```text
experiment_logs/run_outputs/2026-06-14_175338_visual_router_stage1_soft_fusion_calibration_96_48_s_1k/
```

校验结果：

- summary 行数为 19；
- predictions 行数为 9500；
- 每个 strategy 的 sample_key 唯一数均为 500；
- strategy family 覆盖：`raw_soft`、`temperature_soft`、`top1_hard`、`top2_fusion`、`top2_temperature_fusion`、`top3_fusion`、`top3_temperature_fusion`；
- 输出文件均存在。

Top comparison：

| method | method_family | sample_count | mae_like_value | oracle_value | regret_to_oracle | relative_improvement_vs_global_best_single |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| oracle_top1 | upper_bound | 500 | 0.356273 | 0.356273 | 0.000000 | 0.238175 |
| calibration_top3_fusion | top3_fusion | 500 | 0.436033 | 0.356273 | 0.079760 | 0.067623 |
| calibration_soft_T0p75 | temperature_soft | 500 | 0.436598 | 0.356273 | 0.080325 | 0.066414 |
| calibration_top3_fusion_T0p75 | top3_temperature_fusion | 500 | 0.436798 | 0.356273 | 0.080525 | 0.065986 |
| calibration_raw_soft | raw_soft | 500 | 0.437221 | 0.356273 | 0.080948 | 0.065082 |
| dataset_tsf_cell | mean_metric_best | 500 | 0.439672 | 0.356273 | 0.083399 | 0.059841 |

## 结论

当前 `96_48_S` 1k 中等规模实验下，最佳非 oracle calibration 策略为 `calibration_top3_fusion`，MAE 为 0.436033，优于 raw soft fusion 0.437221、online hard top-1 0.459729、非视觉 `dataset_tsf_cell` baseline 0.439672 和 `global_best_single` 0.467657。

## 下一步方案

1. 写最终统一中文汇总日志，串联 preflight、prediction cache、merge、oracle/TSF/baseline、online router 和 calibration。
2. 更新 `experiment_logs/README.md` 总览表。
3. 检查 `WORKSPACE_STRUCTURE.md` 是否需要补充本次新增长期输出目录口径。
