# Stage 1 TimeFuse-style Fusor GPU2/3 状态检查

日志日期：2026-06-19 11:26:10 CST

## 目的

检查 Stage 1 `96_48_S` full-scale TimeFuse-style fusor GPU2/3 正式后台任务当前完成情况，确认是否已经进入训练、评估或结果汇总阶段。

## 背景

正式 GPU2/3 任务目录为：

```text
/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/
```

该任务使用 `CUDA_VISIBLE_DEVICES=2,3` 和 `--device cuda` 启动，目标是在公平口径下完成 TimeFuse-style fusor baseline。此前已经确认 64/64 oracle SQLite index 与 64/64 prediction SQLite index 完成，但当时仍处于 `scaler_partial_fit`。

## 操作

1. 使用 `ps` 检查 launcher 与 Python 子进程是否仍存活。
2. 读取 `status.json`。
3. 查看 `main.log` 尾部。
4. 检查 checkpoint、summary、predictions 等关键输出是否出现。
5. 统计 SQLite index 数量。
6. 查看 GPU 与磁盘状态。

## 结果

进程仍在运行：

```text
launcher PID/PGID: 1271090 / 1271090
Python PID: 1271092
elapsed: 约 1-10:06:43
Python CPU: 约 137%
RSS: 约 3.9GB
```

当前 `status.json`：

```text
status: running
phase: scaler_partial_fit
current_shard: sample_shard_0051_of_0064
vali_batches: 29400
vali_samples: 7509342
updated_at: 2026-06-19 11:21:52 CST
CUDA_VISIBLE_DEVICES: 2,3
torch_cuda logical_index 0/1: NVIDIA GeForce RTX 3090
```

SQLite index 状态：

```text
oracle_labels_index.sqlite: 64
prediction_manifest_index.sqlite: 64
```

关键输出检查：

```text
metadata.json 已存在
checkpoints/latest_timefuse_fusor.pt 尚未出现
checkpoints/timefuse_fusor_epoch_*.pt 尚未出现
summary.md 尚未出现
timefuse_fusor_summary.csv 尚未出现
timefuse_fusor_raw_soft_fusion_summary.csv 尚未出现
timefuse_fusor_selected_model_counts.csv 尚未出现
timefuse_fusor_predictions.csv 尚未出现
sample_predictions.csv 尚未出现
```

GPU 快照：

```text
GPU2: 12 MiB, 0% util
GPU3: 12 MiB, 0% util
```

这说明当前仍未进入 fusor 模型训练阶段；GPU2/GPU3 尚未实际承担训练前向/反向。

磁盘状态：

```text
/data2: 约 2.2T 可用
/home: 约 23G 可用，接近满载
```

## 结论

截至本次检查，TimeFuse fusor GPU2/3 正式任务尚未完成，也尚未进入真正的 GPU 模型训练阶段。当前仍在 `StandardScaler.partial_fit`，已处理 `7,509,342` 个 vali sample，进度到 `sample_shard_0051_of_0064`。64 个 shard-local SQLite index 已全部完成，但 checkpoint、summary 和 predictions 尚未生成。

因此当前不是“信息汇总收尾阶段”，而是训练前 scaler 阶段的后半段。

## 下一步方案

1. 继续监控 `status.json`，等待 `phase` 从 `scaler_partial_fit` 进入 `train`。
2. 进入 `train` 后确认 `main.log` 出现 `启用 DataParallel 双卡训练`。
3. 训练完成后检查 `checkpoints/latest_timefuse_fusor.pt`。
4. 进入并完成 `test_eval` 后再检查 summary、prediction CSV 和最终 `status=completed`。
5. 后续若要提速，应优先实现 feature-only scaler path，避免 scaler 阶段读取五专家 prediction arrays。

## 追加复核：2026-06-19 11:32:48 CST

### 目的

响应用户“检查任务完成情况”的请求，在前一次 11:26 检查后再次确认 TimeFuse fusor GPU2/3 正式任务、visual router eval-only 和关键输出文件状态。

### 操作

1. 使用 `ps` 检查 `train_timefuse_fusor_streaming.py` 进程。
2. 读取 GPU2/3 TimeFuse fusor 的 `status.json` 和 `main.log` 尾部。
3. 检查 GPU2/3 TimeFuse fusor 的 checkpoint、summary 和 prediction 文件是否出现。
4. 读取 visual router eval-only 的 `status.json`、summary CSV 和 soft fusion summary。
5. 查看 GPU、`/data2` 和 `/home` 状态。

### 结果

TimeFuse fusor GPU2/3 仍在运行，尚未完成：

```text
launcher PID/PGID: 1271090 / 1271090
Python PID: 1271092
elapsed: 约 1-10:13:17
status: running
phase: scaler_partial_fit
current_shard: sample_shard_0051_of_0064
vali_batches: 29600
vali_samples: 7560542
updated_at: 2026-06-19 11:27:09 CST
RSS: 约 3.8GB
```

关键产物仍未出现：

```text
checkpoints/latest_timefuse_fusor.pt: 不存在
summary.md: 不存在
timefuse_fusor_summary.csv: 不存在
timefuse_fusor_raw_soft_fusion_summary.csv: 不存在
timefuse_fusor_selected_model_counts.csv: 不存在
timefuse_fusor_predictions.csv: 不存在
sample_predictions.csv: 不存在
```

GPU 快照：

```text
GPU2: 12 MiB, 0% util
GPU3: 12 MiB, 0% util
```

说明当前仍在 scaler 阶段，尚未进入模型训练；因此 GPU2/GPU3 尚未实际承担 fusor 训练。

visual router eval-only 已完成，并已产出 full-scale router predictions：

```text
status: completed
phase: done
router_predictions: 13924650
updated_at: 2026-06-18 17:48:18 CST
```

visual router full-scale 指标：

```text
hard top-1 MAE: 0.5615367653135453
raw soft fusion MAE: 0.5174675759559787
oracle MAE: 0.33862214116809347
sample_count: 13924650
oracle_label_accuracy: 0.4621166779775434
```

磁盘状态：

```text
/data2: 约 2.2T 可用
/home: 约 22G 可用，仍接近满载
```

### 结论

本次复核确认：visual router eval-only 已经完成；TimeFuse fusor GPU2/3 正式任务仍未完成，当前仍在 `scaler_partial_fit` 后半段，尚未进入 train、尚未生成 checkpoint、summary 或 prediction CSV。

### 下一步方案

继续轻量监控 TimeFuse fusor GPU2/3 的 `status.json`。按 2026-06-18 20:51 到 2026-06-19 11:27 的进度粗估，scaler 阶段仍可能需要数小时才能进入 train；进入 train 后再检查 `main.log` 中的 `启用 DataParallel 双卡训练` 和 GPU2/GPU3 占用。

## 追加复核：2026-06-19 11:35:05 CST

### 背景

11:32 复核后，任务状态发生变化：旧 GPU2/3 进程被停止，原因是旧 `scaler_partial_fit` 路径复用了完整 reader，错误地读取 oracle/prediction arrays，导致 scaler 阶段过慢。随后同一输出目录通过 `command_resume.sh` 重启，计划复用已有 64 个 shard-local SQLite index，并走 feature-only scaler 路径。

### 结果

旧进程状态曾被写为：

```text
status=stopped_for_scaler_feature_only_optimization
phase=stopped
stopped_at=2026-06-19 11:33:00 CST
stop_reason=旧 scaler_partial_fit 复用完整 reader，错误地读取 oracle/prediction arrays；已停机以启用 feature-only scaler 和 SQLite index 复用。
```

新的 resume 进程已经启动：

```text
launcher PID/PGID: 1840046 / 1840046
Python PID: 1840048
status: running
phase: reuse_shard_indexes
shard_name: sample_shard_0024_of_0064
updated_at: 2026-06-19 11:35:05 CST
```

`main.log` 显示复用 index 正常推进：

```text
[2026-06-19 11:34:10 CST] 启动 Stage 1 TimeFuse-style fusor streaming 训练/eval
[2026-06-19 11:34:12 CST] 复用已有 shard-local index：sample_shard_0000_of_0064 sample_key=363675
...
[2026-06-19 11:35:05 CST] 复用已有 shard-local index：sample_shard_0024_of_0064 sample_key=363675
```

GPU2/GPU3 仍基本空闲，符合 index 复用和 feature-only scaler 前置阶段预期。

### 结论

TimeFuse fusor GPU2/3 仍未完成，但已从旧的低效 scaler 路径切换到优化后的 resume 路径。当前不是失败终止，而是在同一正式目录中继续运行，正在复用已有 SQLite index。正式 baseline 仍需等待后续 scaler、train、checkpoint 和 test_eval 完成。

### 下一步方案

继续监控新 PID `1840048`。重点看 `phase` 是否从 `reuse_shard_indexes` 进入 feature-only scaler / train，并确认后续 `main.log` 中出现 DataParallel 双卡训练信息。

### 最新状态补充：2026-06-19 11:36:04 CST

复用 index 阶段已完成并进入 feature-only scaler：

```text
status=running
phase=scaler_partial_fit
scaler_mode=feature_only
current_shard=sample_shard_0001_of_0064
vali_samples=292204
pid=1840048
```

`main.log` 明确写出：

```text
[2026-06-19 11:36:01 CST] scaler 使用 feature-only streaming，不读取 oracle/prediction arrays
[2026-06-19 11:36:03 CST] feature-only scaler 完成 shard=sample_shard_0000_of_0064 vali_samples=146102
[2026-06-19 11:36:04 CST] feature-only scaler 完成 shard=sample_shard_0001_of_0064 vali_samples=146102
```
