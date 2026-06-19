# Stage 1 TimeFuse-style Fusor GPU2/3 完成情况检查

日志日期：2026-06-18 20:51:35 CST

## 目的

检查 Stage 1 `96_48_S` full-scale TimeFuse-style fusor GPU2/3 正式后台任务是否已经完成，并同步确认同时间运行的 visual router eval-only 状态。

## 背景

TimeFuse fusor 正式公平口径要求训练时使用 GPU2/GPU3 双卡。当前正式目录为：

```text
/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/
```

CPU 半程目录已因公平性要求停止并标记为非正式结果，不应引用为正式 baseline。

## 操作

1. 使用 `ps` 检查 TimeFuse fusor GPU2/3 进程、visual router eval-only 进程。
2. 读取 TimeFuse fusor `status.json` 和 `main.log` 尾部。
3. 检查 TimeFuse fusor 输出目录下关键产物是否出现。
4. 统计 shard-local SQLite index 数量。
5. 读取 visual router eval-only `status.json`。
6. 使用 `nvidia-smi` 查看 GPU 占用。

## 结果

### TimeFuse fusor GPU2/3

当前仍在运行，尚未完成：

```text
launcher PID/PGID: 1271090 / 1271090
Python PID: 1271092
elapsed: 约 19:32:26
status: running
phase: scaler_partial_fit
current_shard: sample_shard_0027_of_0064
vali_batches: 16000
vali_samples: 4086861
CUDA_VISIBLE_DEVICES: 2,3
```

资源快照：

```text
RSS: 约 3818 MB
process read: 约 96875 MB
process write: 约 65801 MB
torch 可见 logical CUDA 0/1 均为 NVIDIA GeForce RTX 3090
```

SQLite index 进度：

```text
oracle_labels_index.sqlite: 64
prediction_manifest_index.sqlite: 64
```

说明：64 个 feature shard 对应的 oracle/prediction shard-local SQLite index 已全部构建完成。`main.log` 显示 `sample_shard_0063_of_0064` 的 prediction index 已完成，随后进入 scaler partial fit 阶段。

输出产物检查：

```text
metadata.json 已存在
summary.md 尚未出现
timefuse_fusor_summary.csv 尚未出现
timefuse_fusor_raw_soft_fusion_summary.csv 尚未出现
timefuse_fusor_selected_model_counts.csv 尚未出现
timefuse_fusor_predictions.csv 尚未出现
latest_timefuse_fusor.pt 尚未出现
```

因此 TimeFuse fusor 目前还没有完成训练/eval，也还没有 checkpoint 和正式 summary。

### Visual router eval-only

visual router eval-only 已完成：

```text
status: completed
phase: done
router_predictions: 13924650
completed_epochs: 1
updated_at: 2026-06-18 17:48:18 CST
output_dir: /data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt
```

## 结论

截至本次检查，visual router eval-only 已完成；TimeFuse fusor GPU2/3 正式任务尚未完成，但已经通过最重的 64 shard SQLite index 构建阶段，当前正在对 vali feature stream 做 `StandardScaler.partial_fit`。正式 TimeFuse baseline 结果仍需等待后续 `train`、checkpoint 保存和 `test_eval` 完成后才能引用。

## 下一步方案

1. 继续轻量监控 TimeFuse fusor GPU2/3 的 `status.json` 和 `main.log`。
2. 进入 `train` 阶段后确认 `main.log` 中出现 `启用 DataParallel 双卡训练`，并观察 GPU2/GPU3 占用。
3. 完成后检查 `status=completed`、`phase=done`、`checkpoints/latest_timefuse_fusor.pt`、`summary.md` 和三份 summary CSV。
4. 后续引用正式 TimeFuse fusor baseline 时只使用 GPU2/3 目录，不使用 CPU 停止目录。
