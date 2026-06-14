# Stage 1 `96_48_S` 1k Oracle、TSF Enrichment 与非视觉 Baseline

日志日期：2026-06-14 17:46:53 CST

## 目的

在已合并的 `96_48_S` 1k 五专家 prediction cache 上生成 window-level oracle labels，补充 TSF cell 元信息，并运行非视觉 router baseline，为 online visual router 训练和后续 soft fusion calibration 提供监督标签与对照指标。

## 背景

`merged_cache/manifest.csv` 已通过完整性校验：5000 行、1000 个 sample_key、每个 sample_key 五专家完整且共享 `y_true` 一致。后续 router 训练需要 `window_oracle_labels_with_tsf_cell.csv`，非视觉 baseline 则用于衡量 visual router 是否超过 dataset/TSF-cell shortcut。

## 操作

1. 生成 window-level oracle labels：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     /home/shiyuhong/Time/visual_router_experiments/stage1_vali_test_router/pilot/compute_window_oracle_from_cache.py \
     --cache-dir experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/merged_cache
   ```

2. 补充 TSF cell 元信息：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     /home/shiyuhong/Time/visual_router_experiments/stage1_vali_test_router/pilot/enrich_cache_with_tsf_cell.py \
     --cache-dir experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/merged_cache
   ```

3. 运行非视觉 baseline：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     /home/shiyuhong/Time/visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py \
     --labels-path experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/merged_cache/window_oracle_labels_with_tsf_cell.csv \
     --metric mae
   ```

4. 完成后检查以下输出是否存在并读取校验：

   - `window_oracle_labels.csv`
   - `window_oracle_summary.csv`
   - `manifest_with_tsf_cell.csv`
   - `window_oracle_labels_with_tsf_cell.csv`
   - `window_oracle_summary_by_tsf_cell.csv`
   - `window_oracle_summary_by_dataset_tsf_cell.csv`
   - `baseline_predictions.csv`
   - `baseline_summary.csv`
   - `baseline_summary_by_config.csv`
   - `baseline_summary_macro.csv`
   - `baseline_summary_by_dataset.csv`
   - `baseline_summary_by_tsf_cell.csv`
   - `baseline_summary_by_dataset_tsf_cell.csv`
   - `summary.md`

## 结果

### Oracle Labels

- `window_oracle_labels.csv` 行数为 2000；
- `metric=mae` 覆盖 1000 个 sample_key；
- `metric=mse` 覆盖 1000 个 sample_key。

MAE test oracle summary：

| metric | split | dataset_name | sample_count | best_single_model | best_single_value | oracle_value | oracle_gap_abs | oracle_gap_pct |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| mae | test | TEST_DATA_HOUR | 250 | DLinear | 0.409528 | 0.323967 | 0.085562 | 0.208928 |
| mae | test | TEST_DATA_MIN | 250 | PatchTST | 0.492851 | 0.388579 | 0.104271 | 0.211568 |

### TSF Enrichment

- `window_oracle_labels_with_tsf_cell.csv` 行数为 2000；
- oracle labels 中 `group_name` 缺失数为 0；
- oracle labels 中 `cluster` 缺失数为 0；
- `manifest_with_tsf_cell.csv` 行数为 5000；
- manifest 中 `group_name` 缺失数为 0。

### 非视觉 Baseline

`baseline_summary.csv` 关键结果如下：

| baseline | rule_kind | sample_count | selected_value | oracle_value | regret_to_oracle | oracle_label_accuracy | relative_improvement_vs_global |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| global_best_single | mean_metric_best | 500 | 0.467657 | 0.356273 | 0.111384 | 0.262 | 0.000000 |
| dataset_only | mean_metric_best | 500 | 0.451190 | 0.356273 | 0.094917 | 0.388 | 0.035213 |
| dataset_tsf_cell | mean_metric_best | 500 | 0.439672 | 0.356273 | 0.083399 | 0.424 | 0.059841 |
| tsf_cell_only | mean_metric_best | 500 | 0.469368 | 0.356273 | 0.113095 | 0.360 | -0.003659 |
| oracle_top1 | upper_bound | 500 | 0.356273 | 0.356273 | 0.000000 | 1.000 | 0.238175 |

## 结论

Oracle labels、TSF cell enrichment 和非视觉 baseline 均已完成并通过文件级校验。当前 1k test split 上，`dataset_tsf_cell` 是可部署非视觉 baseline 中最强者，MAE 为 0.439672；oracle top-1 上限 MAE 为 0.356273。

## 下一步方案

1. 使用 `train_visual_router_online.py` 在该 1k cache 上训练 online visual router。
2. 训练时显式传入 `--local-files-only`，并设置本地 HF 离线环境变量。
3. 验证 online router 输出不包含长期 ViT embedding `.npy` 或伪图像 tensor。
