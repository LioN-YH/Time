# Stage 1 扩大版五专家 Window Oracle Pilot

日志日期：2026-06-12 13:01:01 CST

## 目的

在最小五专家 cache 闭环通过后，扩大 item/window 数量，初步检查 window-level oracle gap、专家胜率和 cache 规模，为后续是否接入伪图像 tensor/embedding 提供依据。

## 背景

上一轮五专家 pilot 只有 8 个 `sample_key`，用于验证同一窗口下五专家预测能否完整对齐。该规模太小，不足以判断 window-level 专家互补性。因此本轮将范围扩大到每个 dataset 3 个 item、每个 item-channel 10 个窗口，同时覆盖 vali/test。

## 操作

1. 修改 `build_prediction_cache_pilot.py`，新增 `--print-rows`，避免扩大 pilot 后打印完整 manifest。
2. 运行扩大版五专家 cache：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/build_prediction_cache_pilot.py \
     --five-expert-96-48-s \
     --splits vali test \
     --max-items 3 \
     --max-windows 10 \
     --max-channels 1 \
     --print-rows 12
   ```

3. 运行 window oracle 计算：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/compute_window_oracle_from_cache.py \
     --cache-dir experiment_logs/run_outputs/2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot
   ```

4. 校验 manifest 行数、sample_key 数量、专家集合完整性、oracle label/summary 行数和磁盘规模。

## 输出

输出目录：

```text
experiment_logs/run_outputs/2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot/
```

输出规模：

| 项目 | 数量 |
| --- | ---: |
| `manifest.csv` 行数 | 600 |
| `sample_key` 数量 | 120 |
| 每个 `sample_key` 的专家数 | 5 |
| `window_oracle_labels.csv` 行数 | 240 |
| `window_oracle_summary.csv` 行数 | 8 |
| `.npy` 数组文件数 | 1200 |
| 输出目录大小 | 5.4 MB |

## 结果

MAE 口径按 split/dataset 的 oracle gap：

| split | dataset | sample_count | best_single_model | best_single_MAE | oracle_MAE | oracle_gap_pct | 主要 oracle 胜者 |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| `test` | `TEST_DATA_HOUR` | 30 | DLinear | 0.111696 | 0.109931 | 1.58% | DLinear 83.33% |
| `test` | `TEST_DATA_MIN` | 30 | PatchTST | 1.957175 | 1.500853 | 23.32% | ES 50.00%，NaiveForecaster 33.33% |
| `vali` | `TEST_DATA_HOUR` | 30 | DLinear | 0.121876 | 0.098023 | 19.57% | PatchTST 40.00%，DLinear 33.33%，CrossFormer 26.67% |
| `vali` | `TEST_DATA_MIN` | 30 | ES | 0.273968 | 0.258632 | 5.60% | ES 60.00% |

整体按 split 汇总：

| metric | split | sample_count | best_single_model | best_single_value | oracle_value | oracle_gap_pct |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| MAE | `test` | 60 | PatchTST | 1.050780 | 0.805392 | 23.35% |
| MAE | `vali` | 60 | CrossFormer | 0.238761 | 0.178328 | 25.31% |
| MSE | `test` | 60 | PatchTST | 4.374439 | 3.698563 | 15.45% |
| MSE | `vali` | 60 | DLinear | 0.137431 | 0.096525 | 29.76% |

## 验证

已验证：

- `manifest.csv` 为 600 行，等于 120 个 `sample_key` × 5 个专家。
- 每个 `sample_key` 下专家集合完整，均包含 DLinear、PatchTST、CrossFormer、ES、NaiveForecaster。
- `window_oracle_labels.csv` 为 240 行，等于 120 个 `sample_key` × MAE/MSE。
- `window_oracle_summary.csv` 为 8 行，等于 2 个指标 × 2 个 split × 2 个 dataset。
- 输出目录大小为 5.4 MB，当前小规模缓存成本可接受。

## 结论

扩大版 pilot 已显示明显 window-level 专家互补性。整体 MAE oracle gap 在 test split 为 23.35%，vali split 为 25.31%；不同 dataset 的专家胜率差异明显，尤其分钟级 test window 中 ES/NaiveForecaster 的窗口级胜率高于配置级直觉。

这说明 Stage 1 不应直接进入 ViT embedding 训练前的大规模视觉缓存，而应先扩大五专家 prediction cache 并系统统计 window-level oracle gap 和胜率；当前路线仍成立。

## 下一步方案

1. 将五专家 cache 扩大到更稳定的样本范围，例如每个 dataset 10 个 item、每个 item-channel 50 个 window。
2. 同时记录运行时间和输出目录大小，估算全量 cache 成本。
3. 若更大范围仍保持明显 oracle gap，再实现伪图像 tensor 生成与冻结视觉 encoder embedding 缓存。
