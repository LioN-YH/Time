# Stage 1 Prediction Cache 补充 TSF Cell 元信息

日志日期：2026-06-12 20:01:29 CST

## 目的

为 Stage 1 prediction cache 与 window oracle 结果补充 TSF cell 元信息，区分 `dataset_name` 与 `cluster/group_name`，支持后续按 dataset、TSF cell、dataset+TSF cell 分层分析。

## 背景

此前扩大版五专家 pilot 的 manifest 中包含 `dataset_name`，但该字段只表示 Quito 的数据来源/频率层级，例如 `TEST_DATA_MIN` 或 `TEST_DATA_HOUR`，不等于 TSF cell。TSF cell 需要通过 `item_id` 从 `item_clusters.csv` 映射得到。为了判断 window-level 互补性究竟来自 dataset 差异、TSF cell 差异还是窗口结构差异，需要显式合并 `cluster/group_name`。

## 操作

1. 新增脚本：

   ```text
   visual_router_experiments/stage1_vali_test_router/enrich_cache_with_tsf_cell.py
   ```

2. 脚本输入：

   - `manifest.csv`
   - `window_oracle_labels.csv`
   - `quito/examples/datasets/cluster_data/item_clusters.csv`

3. 对脚本执行语法检查：

   ```bash
   python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/enrich_cache_with_tsf_cell.py

   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     -m py_compile \
     visual_router_experiments/stage1_vali_test_router/enrich_cache_with_tsf_cell.py
   ```

4. 在扩大版五专家 pilot 目录上运行：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/enrich_cache_with_tsf_cell.py \
     --cache-dir experiment_logs/run_outputs/2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot
   ```

## 输出

输出目录：

```text
experiment_logs/run_outputs/2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot/
```

新增文件：

| 文件 | 功能 | 行数 |
| --- | --- | ---: |
| `manifest_with_tsf_cell.csv` | 带 TSF cell 元信息的五专家 cache manifest | 600 |
| `window_oracle_labels_with_tsf_cell.csv` | 带 TSF cell 元信息的 window oracle label/regret | 240 |
| `window_oracle_summary_by_tsf_cell.csv` | 按 split + TSF cell 汇总 oracle gap 和胜率 | 20 |
| `window_oracle_summary_by_dataset_tsf_cell.csv` | 按 split + dataset + TSF cell 汇总 oracle gap 和胜率 | 20 |

## 结果

当前扩大版 pilot 覆盖的 item 与 TSF cell：

| dataset_name | item_id | cluster | group_name |
| --- | ---: | ---: | --- |
| `TEST_DATA_HOUR` | 100011 | 0 | HIGH_HIGH_HIGH |
| `TEST_DATA_HOUR` | 100012 | 0 | HIGH_HIGH_HIGH |
| `TEST_DATA_HOUR` | 100019 | 2 | HIGH_HIGH_LOW |
| `TEST_DATA_MIN` | 153 | 8 | HIGH_LOW_LOW |
| `TEST_DATA_MIN` | 191 | 18 | LOW_HIGH_HIGH |
| `TEST_DATA_MIN` | 286 | 26 | LOW_LOW_LOW |

MAE 口径下，按 dataset + TSF cell 的代表性结果：

| split | dataset | group_name | sample_count | best_single_model | oracle_gap_pct | 主要胜者 |
| --- | --- | --- | ---: | --- | ---: | --- |
| `test` | `TEST_DATA_HOUR` | HIGH_HIGH_HIGH | 20 | DLinear | 3.18% | DLinear 75.00% |
| `test` | `TEST_DATA_HOUR` | HIGH_HIGH_LOW | 10 | DLinear | 0.00% | DLinear 100.00% |
| `test` | `TEST_DATA_MIN` | HIGH_LOW_LOW | 10 | ES | 1.73% | ES 80.00% |
| `test` | `TEST_DATA_MIN` | LOW_HIGH_HIGH | 10 | PatchTST | 17.30% | PatchTST 40.00%，NaiveForecaster 30.00% |
| `test` | `TEST_DATA_MIN` | LOW_LOW_LOW | 10 | ES | 3.23% | ES 50.00%，NaiveForecaster 50.00% |
| `vali` | `TEST_DATA_HOUR` | HIGH_HIGH_HIGH | 20 | PatchTST | 4.26% | PatchTST 60.00%，CrossFormer 40.00% |
| `vali` | `TEST_DATA_MIN` | LOW_LOW_LOW | 10 | ES | 7.50% | ES 60.00%，CrossFormer 30.00% |

## 验证

已验证：

- `manifest_with_tsf_cell.csv` 行数为 600，与原 manifest 一致。
- `window_oracle_labels_with_tsf_cell.csv` 行数为 240，与原 oracle labels 一致。
- `group_name` 缺失数量为 0。
- 当前 pilot 覆盖 5 个 TSF cell：`HIGH_HIGH_HIGH`、`HIGH_HIGH_LOW`、`HIGH_LOW_LOW`、`LOW_HIGH_HIGH`、`LOW_LOW_LOW`。
- `WORKSPACE_STRUCTURE.md` 和 Stage 1 README 已更新新增脚本说明。

## 结论

Stage 1 cache 已具备 dataset 与 TSF cell 双重分层分析能力。后续扩大样本或训练 router 时，必须同时报告 dataset-only、TSF-cell-only 和 dataset+TSF-cell 的 baseline/summary，避免把 dataset shortcut 或 cell shortcut 误判为视觉结构泛化能力。

## 下一步方案

1. 下一轮扩大 cache 时直接在 manifest 生成或后处理阶段保留 TSF cell 字段。
2. 设计 router baseline：
   - dataset-only；
   - TSF-cell-only；
   - dataset+TSF-cell；
   - window visual/numeric structure。
3. 继续扩大样本范围前，先把 summary 脚本固定下来，确保每次输出都包含 dataset 和 TSF cell 分层。
