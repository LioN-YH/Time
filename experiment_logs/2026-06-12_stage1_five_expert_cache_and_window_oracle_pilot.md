# Stage 1 五专家 Prediction Cache 与 Window Oracle Pilot

日志日期：2026-06-12 12:51:17 CST

## 目的

将 Stage 1 prediction cache pilot 从单专家扩展到五专家，并基于同一批 item-channel-window 计算 window-level oracle label 和 expert regret。

## 背景

前一步已经跑通 DLinear 单专家的 vali+test cache。Visual Router 训练需要同一个 `sample_key` 下存在完整专家集合，才能计算 hard oracle label、专家 regret 和后续 softmax fusion。因此本次使用当前已有五专家 `96_48_S` evaluate config 做最小范围对齐验证。

## 操作

1. 修改 `visual_router_experiments/stage1_vali_test_router/build_prediction_cache_pilot.py`：

   - 新增 `--config-paths`，支持多个专家 evaluate config；
   - 新增 `--five-expert-96-48-s`，直接使用当前已有 DLinear、PatchTST、CrossFormer、ES、SNaive/NaiveForecaster 的 `96_48_S` config；
   - 保留 `sample_key` 不含模型名，使同一窗口下不同专家可以对齐；
   - 顺序加载专家，避免并发占用内存。

2. 新增脚本 `visual_router_experiments/stage1_vali_test_router/compute_window_oracle_from_cache.py`，基于 manifest 计算：

   - `window_oracle_labels.csv`；
   - `window_oracle_summary.csv`。

3. 执行语法检查：

   ```bash
   python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/build_prediction_cache_pilot.py \
     visual_router_experiments/common/prediction_cache_schema.py

   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     -m py_compile \
     visual_router_experiments/stage1_vali_test_router/build_prediction_cache_pilot.py \
     visual_router_experiments/common/prediction_cache_schema.py
   ```

4. 运行五专家 cache pilot：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/build_prediction_cache_pilot.py \
     --five-expert-96-48-s \
     --splits vali test
   ```

5. 运行 window oracle 计算：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/compute_window_oracle_from_cache.py \
     --cache-dir experiment_logs/run_outputs/2026-06-12_124825_765773_visual_router_stage1_prediction_cache_pilot
   ```

## 输出

输出目录：

```text
experiment_logs/run_outputs/2026-06-12_124825_765773_visual_router_stage1_prediction_cache_pilot/
```

主要输出文件：

| 文件 | 功能 |
| --- | --- |
| `manifest.csv` | 五专家 window-level prediction cache manifest，共 40 条记录 |
| `metadata.json` | 五专家 pilot 参数和 config 路径 |
| `arrays/` | 五专家 `y_true/y_pred` 小规模数组 |
| `window_oracle_labels.csv` | 每个 sample_key 的 MAE/MSE oracle label 和专家 regret，共 16 条记录 |
| `window_oracle_summary.csv` | 按 split/dataset/metric 汇总 best single 与 oracle gap，共 8 条记录 |

## 结果

五专家 manifest 校验：

- `manifest.csv` 行数为 40。
- `sample_key` 数量为 8。
- 每个 `sample_key` 下都有 5 个专家：DLinear、PatchTST、CrossFormer、ES、NaiveForecaster。
- 每条 `y_true/y_pred` 数组形状均为 `(48, 1)`。
- 逐条重算 MAE 与 manifest 记录一致。

MAE oracle label 结果：

| split | dataset | window | oracle_model | oracle_MAE |
| --- | --- | ---: | --- | ---: |
| `vali` | `TEST_DATA_MIN` | 0 | ES | 0.476450 |
| `vali` | `TEST_DATA_MIN` | 1 | ES | 0.517364 |
| `vali` | `TEST_DATA_HOUR` | 0 | CrossFormer | 0.078136 |
| `vali` | `TEST_DATA_HOUR` | 1 | CrossFormer | 0.080290 |
| `test` | `TEST_DATA_MIN` | 0 | ES | 0.711675 |
| `test` | `TEST_DATA_MIN` | 1 | ES | 0.649370 |
| `test` | `TEST_DATA_HOUR` | 0 | PatchTST | 0.078579 |
| `test` | `TEST_DATA_HOUR` | 1 | PatchTST | 0.087196 |

由于本次每个 split/dataset 只有 2 个窗口，MAE 口径下 best single 与 oracle 刚好一致，oracle gap 为 0。这只是小样本 pilot 的结果，不代表全量 window-level 没有 oracle gap；本次核心目标是验证五专家对齐和 label/regret 生成链路。

MSE 口径下 `TEST_DATA_HOUR` 的两个 split 已出现不同窗口最优专家变化：

- `test / TEST_DATA_HOUR`：PatchTST 与 CrossFormer 各赢 1 个窗口；
- `vali / TEST_DATA_HOUR`：PatchTST 与 CrossFormer 各赢 1 个窗口。

## 验证

已验证：

- 五专家 manifest 行数、sample_key 数和专家集合完整性均符合预期。
- `window_oracle_labels.csv` 行数为 16，即 8 个 sample_key × MAE/MSE。
- `window_oracle_summary.csv` 行数为 8，即 2 个指标 × 2 个 split × 2 个 dataset。
- `compute_window_oracle_from_cache.py` 在系统 Python 和 Quito conda 环境下均通过语法检查。

## 结论

Stage 1 已跑通五专家 window-level prediction cache 与 oracle label/regret 最小闭环。下一步可以扩大 pilot 范围，优先增加 item/window 数量，而不是马上接入 ViT embedding；先确认更大样本下 window-level oracle gap 和专家胜率是否稳定。

## 下一步方案

1. 扩大五专家 pilot 到更多 item 和窗口，例如每个 dataset 5 到 10 个 item、每个 item-channel 20 到 50 个 window。
2. 汇总更大样本下的 window-level oracle gap、专家胜率和 regret 分布。
3. 若 window-level 互补性仍明显，再接入伪图像 tensor 和冻结视觉 encoder embedding。
