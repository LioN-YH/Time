# Stage 1 `96_48_S` Full-Scale Prediction Cache Merge 与完整性校验

日志日期：2026-06-16 02:33:22 CST

## 目的

完成 Stage 1 `96_48_S` 正式 full-scale 五专家 prediction cache 的 merge，并在 `merged_cache/` 上执行可复核的完整性校验。本轮任务边界只包含 merge 和校验，不启动 oracle labels、streaming visual router 或 calibration。

## 背景

正式 full-scale prediction cache 已完成全部五专家 shard：`status_files=320`、`completed=320`、`running=0`、`failed=0`。`prediction_cache_full_scale_launcher/status.json` 中的 `merge_command` 是正式 merge 入口，包含 320 个 shard 输入和输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache/
```

上一轮 merge 在 `sample_shard=0014` 失败，报错为 `ES/sample_shard_0014_of_0064` 的 packed `y_true` 文件 shape 是 DLinear 的两倍。接手后继续按 shard 级别定位，不删除任何已完成 prediction cache shard。

## 操作

1. 复核 `AGENTS.md`、外部 `HANDOFF.md`、`experiment_logs/README.md`、Stage 1 cache contract、protocol、正式 merge 脚本和 launcher `status.json`。
2. 检查原 merge 日志 `merge_command_2026-06-15_234723.log`，确认失败点为 `ES/sample_shard_0014_of_0064` 的 `test/TEST_DATA_HOUR` packed `y_true` shape 翻倍。
3. 对 `sample_shard_0014_of_0064` 五专家 manifest 和 packed 文件做只读诊断：
   - 五专家 manifest 行数均为 `363,675`；
   - ES 四个 `y_true` 和四个 `y_pred` packed 文件均为 manifest 引用行数的 2 倍；
   - manifest row index 只引用前半段；
   - ES 被引用前半段 `y_true` 与 DLinear 完全一致，尾部是重复追加。
4. 修改正式脚本 `visual_router_experiments/stage1_vali_test_router/merge_prediction_cache_shards.py`：
   - packed y_true 校验改为按 manifest row index 比较被引用行，而不是要求整文件 shape 完全相同；
   - y_pred 复制到 merged cache 时只保留 manifest 引用行，并重写为 compact `0..N-1` row index；
   - merged y_true row index 对所有专家统一改写为 reference expert 的共享行号；
   - 保持执行入口仍为 launcher `status.json` 中的同一条 `merge_command`。
5. 使用 Quito 环境做语法检查：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile visual_router_experiments/stage1_vali_test_router/merge_prediction_cache_shards.py
```

6. 对异常 shard 做隔离回归：
   - `shard0014_merge_regression_fast_2026-06-16_002553`：通过，ES y_pred 被裁剪到 manifest 引用行数；
   - `shard0054_merge_regression_shared_true_2026-06-16_011652`：通过，输出 `1,818,370` 行、`363,674` 个 sample_key、五专家完整、共享 y_true 违规为 `0`。
7. 用 launcher `status.json` 中的 `merge_command` 重新后台启动正式 compact retry，主日志为：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merge_command_retry_compact_2026-06-16_011812.log
```

8. 启动等待型完整性校验任务，等待 merge completed 后自动扫描 merged manifest：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache_validation/2026-06-16_011835_full_integrity_validation_compact_retry/
```

## 结果

正式 merged cache 已生成：

```text
merged_cache/status.json: completed
generated_at: 2026-06-16 02:09:58 CST
sample_count: 23,275,170
record_count: 116,375,850
array_storage: packed_npy_v1
merge_strategy: packed_npy_v1_streaming_by_sample_shard
shared_y_true_path: true
```

merged y_true 行数为：

```text
test/TEST_DATA_HOUR: 1,305,425
test/TEST_DATA_MIN: 12,619,225
vali/TEST_DATA_HOUR: 7,530,105
vali/TEST_DATA_MIN: 1,820,415
```

完整性校验结果：

```text
validation_status: completed
passed: true
duration_sec: 4118.754
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

五专家覆盖计数均为 `23,275,170`：

```text
DLinear: 23,275,170
PatchTST: 23,275,170
CrossFormer: 23,275,170
ES: 23,275,170
NaiveForecaster: 23,275,170
```

校验产物：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache_validation/2026-06-16_011835_full_integrity_validation_compact_retry/integrity_summary.json
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache_validation/2026-06-16_011835_full_integrity_validation_compact_retry/model_counts.csv
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache_validation/2026-06-16_011835_full_integrity_validation_compact_retry/array_path_checks.csv
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache_validation/2026-06-16_011835_full_integrity_validation_compact_retry/bad_examples.json
```

本轮未删除任何已完成 prediction cache shard，未启动 oracle labels、streaming visual router 或 calibration。

## 结论

Stage 1 `96_48_S` full-scale 五专家 prediction cache 的正式 `merged_cache/` 已完成并通过完整性校验。`ES` 个别 shard 的恢复残留表现为 packed 数组重复尾部或 manifest 引用后半段 row index；这些问题已经在 merge 阶段按 manifest row index 读取、按 reference y_true 共享行号重写、按 compact y_pred row index 输出的方式处理，最终 merged cache 满足 Stage 1 cache contract。

## 下一步方案

1. 后续若继续 Stage 1 主线，可在该 `merged_cache/` 上生成 oracle labels 和 TSF cell enrichment。
2. 生成 labels 后再进入 baseline / TimeFuse-style fusor / streaming visual router / calibration；本轮已明确没有启动这些步骤。
3. 若复查本轮 merge，优先查看 `merge_command_retry_compact_2026-06-16_011812.log` 和 `integrity_summary.json`，不要读取早期 failed retry 作为最终结果。
