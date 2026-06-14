# Stage 1 `96_48_S` 1k 五专家 Prediction Cache Shard

日志日期：2026-06-14 17:43:17 CST

## 目的

启动并完成 Stage 1 `96_48_S` 1k 中等规模五专家 prediction cache shard，为后续 merge、oracle labels、非视觉 baseline、online visual router 和 soft fusion calibration 提供统一的专家预测缓存。

## 背景

执行前检查已确认 1k sample manifest 完整，五专家 shard 尚不存在。实验要求所有阶段可恢复：若已有 shard 完成则跳过；若缺失则只补跑缺失部分。本轮开始时五个 shard 均缺失，因此需要运行 DLinear、PatchTST、CrossFormer、ES 和 NaiveForecaster 五个专家。

## 操作

1. 首先执行已有 launcher：

   ```text
   bash experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/launcher.sh
   ```

2. 检查 PID、GPU、`status.json` 和 `main.log` 后发现：

   - launcher 写出了五个 pid 文件；
   - 随后这些 PID 均已退出；
   - 五个 `main.log` 均为 0 字节；
   - 五个 shard 均未写出 `status.json`；
   - 未生成有效 `manifest.csv`。

   该次启动没有产生有效 prediction cache，且没有出现 429、503、timeout、connection 或远端服务临时失败信息。

3. 保留上述空日志事实后，改用同一批命令的直接会话方式补跑五个 shard，输出仍写入原 launcher 目录下对应 shard：

   - DLinear：`CUDA_VISIBLE_DEVICES=0`，`--batch-size 512`，`--local-rank 0`；
   - PatchTST：`CUDA_VISIBLE_DEVICES=1`，`--batch-size 512`，`--local-rank 0`；
   - CrossFormer：`CUDA_VISIBLE_DEVICES=2`，`--batch-size 512`，`--local-rank 0`；
   - ES：CPU，`--batch-size 64`，`--local-rank -1`；
   - NaiveForecaster：CPU，`--batch-size 64`，`--local-rank -1`。

4. 完成后使用 `quito` 环境读取每个 shard 的 `status.json` 和 `manifest.csv`，并对每个专家校验：

   - `status == completed`；
   - manifest 行数为 1000；
   - `sample_key` 唯一数为 1000；
   - 仅包含对应单个 `model_name`；
   - `sample_key + model_name` 无重复；
   - `y_true_path` 和 `y_pred_path` 各 1000 个；
   - 前 5 条记录的数组存在，shape 为 `(48, 1)`，重算 MAE/MSE 与 manifest 一致。

## 结果

五个 shard 均已完成：

| model_name | status | sample_count | record_count | manifest rows | sample_unique | duplicate_pairs | main.log bytes | head5 max MAE delta | head5 max MSE delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DLinear | completed | 1000 | 1000 | 1000 | 1000 | 0 | 4929 | 5.551115e-17 | 5.551115e-17 |
| PatchTST | completed | 1000 | 1000 | 1000 | 1000 | 0 | 4944 | 8.326673e-17 | 6.938894e-17 |
| CrossFormer | completed | 1000 | 1000 | 1000 | 1000 | 0 | 5000 | 5.551115e-17 | 5.551115e-17 |
| ES | completed | 1000 | 1000 | 1000 | 1000 | 0 | 3846 | 8.326673e-17 | 6.245005e-17 |
| NaiveForecaster | completed | 1000 | 1000 | 1000 | 1000 | 0 | 4096 | 5.551115e-17 | 8.326673e-17 |

输出目录：

```text
experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/shards/
```

本步骤没有发生 429、503、网络超时或远端服务临时失败；没有删除任何已有有效输出。

## 结论

`96_48_S` 1k 五专家 prediction cache shard 已全部完成并通过独立校验。原 launcher 的后台启动方式在当前工具执行环境下没有保住子进程，但直接会话补跑使用了相同专家命令和同一输出目录，最终产物有效且可恢复。

## 下一步方案

1. 执行 `merge_prediction_cache_shards.py`，将五个 shard 合并到 launcher 目录下的 `merged_cache/`。
2. 合并后校验 5000 行、1000 个 sample_key、五专家完整、`sample_key + model_name` 唯一、共享 `y_true` 一致，并重算首批 MAE/MSE。
3. Merge 通过后继续生成 oracle labels、TSF cell enrichment 和非视觉 baseline。
