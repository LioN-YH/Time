# Stage 1 `96_48_S` Full-Scale Merge 进度复核

日志日期：2026-06-16 12:43:12 CST

## 目的

复核昨晚 goal“Stage 1 `96_48_S` full-scale prediction cache 的 merge 与完整性校验”的实际进展，确认 `HANDOFF.md`、正式输出目录、完整性校验结果和总览文档是否一致。

## 背景

根目录 `HANDOFF.md` 顶部记录该 goal 已完成，正式 merged cache 位于：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache/
```

完整性校验位于：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache_validation/2026-06-16_011835_full_integrity_validation_compact_retry/
```

本次只做只读复核，不启动新的 oracle labels、router、calibration 或训练任务。

## 操作

1. 读取根目录 `HANDOFF.md`，确认顶部最新状态记录为 merge 与完整性校验已完成。
2. 读取 `experiment_logs/2026-06-16_stage1_96_48_s_full_scale_merge_and_integrity_validation.md`，确认昨晚已写入正式中文实验日志。
3. 使用 Quito 环境解析 `merged_cache/status.json`、`merged_cache/metadata.json` 和完整性校验 `integrity_summary.json` 的关键字段。
4. 查看最终 merge 日志 `merge_command_retry_compact_2026-06-16_011812.log` 尾部，确认 64 个 sample shard 已全部写完，并输出 `sample_count=23,275,170`、`record_count=116,375,850`。
5. 使用 `ps` 检查是否仍有 `merge_prediction_cache_shards`、完整性校验或 `prediction_cache_full_scale_launcher` 相关遗留进程。
6. 使用 `du -sh` 查看正式 `merged_cache/` 和校验目录占用。
7. 检查 `experiment_logs/README.md` 与 `WORKSPACE_STRUCTURE.md`，确认两者已记录最终 completed / passed 结果。

## 结果

`merged_cache/status.json` 和 `metadata.json` 均显示：

```text
status: completed
generated_at: 2026-06-16 02:09:58 CST
sample_count: 23,275,170
record_count: 116,375,850
array_storage: packed_npy_v1
merge_strategy: packed_npy_v1_streaming_by_sample_shard
shared_y_true_path: true
```

完整性校验 `integrity_summary.json` 显示：

```text
status: completed
generated_at: 2026-06-16 02:27:14 CST
duration_sec: 4118.754
passed: true
actual_record_count: 116,375,850
actual_sample_key_unique_count: 23,275,170
sample_group_count: 23,275,170
sample_key_model_uniqueness_violations: 0
expert_completeness_violations: 0
shared_y_true_violations: 0
stable_metadata_violations: 0
array_storage_violations: 0
array_path_violations: 0
```

五专家覆盖均为 `23,275,170`：

```text
DLinear: 23,275,170
PatchTST: 23,275,170
CrossFormer: 23,275,170
ES: 23,275,170
NaiveForecaster: 23,275,170
```

未发现 merge 或完整性校验相关遗留进程。目录占用为：

```text
merged_cache: 77G
merged_cache_validation/2026-06-16_011835_full_integrity_validation_compact_retry: 280K
```

`experiment_logs/README.md` 已包含 `2026-06-16_stage1_96_48_s_full_scale_merge_and_integrity_validation.md` 的完成记录；`WORKSPACE_STRUCTURE.md` 已记录正式 full-scale prediction cache launcher、`merged_cache/` 和完整性校验目录口径。

## 结论

昨晚 goal 已完成：Stage 1 `96_48_S` full-scale 五专家 prediction cache 已合并为正式 `merged_cache/`，并通过完整性校验。最终可引用结果应以 `merge_command_retry_compact_2026-06-16_011812.log`、`merged_cache/status.json`、`metadata.json` 和 `integrity_summary.json` 为准，不应引用早期 failed retry。

## 下一步方案

1. 后续 Stage 1 主线可以基于该 `merged_cache/` 生成 oracle labels 和 TSF enrichment。
2. 生成 labels 后再进入 baseline / TimeFuse-style fusor / streaming visual router / calibration。
3. `HANDOFF.md` 中的 TimeFuse feature cache 是另一条并行任务状态，继续推进前应单独复核其 launcher 和 64 个 feature shard 状态。
