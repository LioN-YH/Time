# Stage 1 `96_48_S` 1k Prediction Cache Merge 校验

日志日期：2026-06-14 17:45:12 CST

## 目的

合并 `96_48_S` 1k 五专家 prediction cache shard，并验证合并后的 cache 是否满足 Stage 1 cache contract：`sample_key + model_name` 唯一、每个 sample_key 覆盖五专家、共享 `y_true` 一致、预测数组存在且指标可重算。

## 背景

上一阶段已完成 DLinear、PatchTST、CrossFormer、ES、NaiveForecaster 五个 shard。每个 shard 各包含 1000 个 sample_key 的单专家预测记录。后续 oracle labels、TSF enrichment、非视觉 baseline、online visual router 和 soft fusion calibration 都依赖合并后的五专家 manifest。

## 操作

1. 确认目标合并目录此前没有有效文件：

   ```text
   experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/merged_cache/
   ```

2. 执行合并：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     /home/shiyuhong/Time/visual_router_experiments/stage1_vali_test_router/merge_prediction_cache_shards.py \
     --shard-dirs \
       experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/shards/DLinear \
       experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/shards/PatchTST \
       experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/shards/CrossFormer \
       experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/shards/ES \
       experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/shards/NaiveForecaster \
     --output-dir experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/merged_cache
   ```

3. 合并后使用 `quito` 环境做独立校验：

   - 读取 `status.json` 和 `manifest.csv`；
   - 检查行数、sample 数、重复键；
   - 检查每个 sample_key 的专家数；
   - 检查每个 sample_key 的共享 `y_true_path`；
   - 检查所有 `y_true_path` 和 `y_pred_path` 文件存在；
   - 对前 25 条记录读取数组，校验 shape 为 `(48, 1)`，并重算 MAE/MSE。

## 结果

合并输出目录：

```text
experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/merged_cache/
```

关键校验结果：

- `status.json.status = completed`；
- manifest 行数为 5000；
- `sample_key` 唯一数为 1000；
- `sample_key + model_name` 重复数为 0；
- 每个 sample_key 的专家数为 5，共 1000 个 sample_key；
- 每个 sample_key 的 `y_true_path` 唯一数为 1，共 1000 个 sample_key；
- 缺失 `y_true` 文件数为 0；
- 缺失 `y_pred` 文件数为 0；
- 前 25 条记录重算 `max_mae_delta = 2.220446049250313e-16`；
- 前 25 条记录重算 `max_mse_delta = 4.440892098500626e-16`。

覆盖统计：

| config_name | split | dataset_name | model_name | rows |
| --- | --- | --- | --- | ---: |
| 96_48_S | test | TEST_DATA_HOUR | CrossFormer | 250 |
| 96_48_S | test | TEST_DATA_HOUR | DLinear | 250 |
| 96_48_S | test | TEST_DATA_HOUR | ES | 250 |
| 96_48_S | test | TEST_DATA_HOUR | NaiveForecaster | 250 |
| 96_48_S | test | TEST_DATA_HOUR | PatchTST | 250 |
| 96_48_S | test | TEST_DATA_MIN | CrossFormer | 250 |
| 96_48_S | test | TEST_DATA_MIN | DLinear | 250 |
| 96_48_S | test | TEST_DATA_MIN | ES | 250 |
| 96_48_S | test | TEST_DATA_MIN | NaiveForecaster | 250 |
| 96_48_S | test | TEST_DATA_MIN | PatchTST | 250 |
| 96_48_S | vali | TEST_DATA_HOUR | CrossFormer | 250 |
| 96_48_S | vali | TEST_DATA_HOUR | DLinear | 250 |
| 96_48_S | vali | TEST_DATA_HOUR | ES | 250 |
| 96_48_S | vali | TEST_DATA_HOUR | NaiveForecaster | 250 |
| 96_48_S | vali | TEST_DATA_HOUR | PatchTST | 250 |
| 96_48_S | vali | TEST_DATA_MIN | CrossFormer | 250 |
| 96_48_S | vali | TEST_DATA_MIN | DLinear | 250 |
| 96_48_S | vali | TEST_DATA_MIN | ES | 250 |
| 96_48_S | vali | TEST_DATA_MIN | NaiveForecaster | 250 |
| 96_48_S | vali | TEST_DATA_MIN | PatchTST | 250 |

## 结论

`96_48_S` 1k 五专家 prediction cache merge 已完成，合并产物满足 Stage 1 cache contract。合并过程只复制数组和重写路径，没有删除原 shard。

## 下一步方案

1. 在 `merged_cache/` 上生成 window-level oracle labels。
2. 对 manifest 和 oracle labels 补充 TSF cell 字段，并生成 TSF 分层 oracle summary。
3. 基于 `window_oracle_labels_with_tsf_cell.csv` 运行非视觉 router baseline。
