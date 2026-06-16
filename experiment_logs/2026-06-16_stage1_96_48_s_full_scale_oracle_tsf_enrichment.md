# Stage 1 `96_48_S` Full-Scale Oracle Labels 与 TSF Enrichment

日志日期：2026-06-16 14:13:11 CST

## 目的

基于已经完成并通过完整性校验的 Stage 1 `96_48_S` 正式 full-scale merged prediction cache，生成后续 baseline、router 和 fusor 可 join 的 window-level oracle labels 与 sample_key 级 TSF enrichment。

## 背景

本轮输入固定为正式 merged cache，不重新 merge，不启动 visual router、streaming router、soft fusion calibration 或 TimeFuse fusor 训练：

```text
merged cache:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache/

integrity validation:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache_validation/2026-06-16_011835_full_integrity_validation_compact_retry/integrity_summary.json
```

预检确认：

```text
merged_cache/status.json: completed
merged_cache/metadata.json: completed
integrity_summary.json: status=completed, passed=true
actual_record_count: 116,375,850
actual_sample_key_unique_count: 23,275,170
sample_key_model_uniqueness_violations: 0
expert_completeness_violations: 0
shared_y_true_violations: 0
stable_metadata_violations: 0
array_storage_violations: 0
array_path_violations: 0
```

## 操作

1. 阅读并复用 Stage 1 cache contract、pilot oracle/TSF enrichment 口径和正式 `packed_npy_v1` merged manifest 字段。
2. 新增正式 full-scale 脚本：

```text
visual_router_experiments/stage1_vali_test_router/build_full_scale_window_oracle_labels.py
visual_router_experiments/stage1_vali_test_router/build_full_scale_tsf_enrichment.py
visual_router_experiments/stage1_vali_test_router/validate_full_scale_oracle_tsf_outputs.py
```

3. 使用 Quito 环境做语法检查与临时小 manifest smoke，覆盖 chunk 边界 carry-over、oracle labels、TSF enrichment 和 validation 串联。
4. 启动第一版 oracle 后发现逐 `sample_key` Python groupby 吞吐不足，已停止该半成品进程并将状态标记为 `stopped_for_optimization`：

```text
stopped dir:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/

stop_reason:
首版逐 sample_key Python groupby 在 full-scale oracle 生成上吞吐不足，改用向量化 pivot 后在新输出目录重启。
```

5. 将 oracle 生成改为 chunk 内向量化 pivot：每个完整 chunk 按 `sample_key` 形成五专家宽表，用 numpy `argmin` 生成 `mae`/`mse` 两套 oracle label 和 regret 字段，避免逐窗口 Python 循环。
6. 使用后台 `setsid` 并行生成正式产物；后续已将 completed 的 `_v2` 目录重命名为无后缀 canonical 路径，见 `2026-06-16_stage1_oracle_partial_cleanup_and_canonical_rename.md`：

```text
oracle:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/

TSF enrichment:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/tsf_enrichment_full_scale_2026-06-16/
```

7. 使用 `deepseek-sidecar` 做只读巡检，检查 PID、`status.json`、`main.log` 和文件描述符；sidecar 未修改文件，未启动训练。
8. 两个生成任务完成后运行正式 join/覆盖验证：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_tsf_validation_2026-06-16/
```

## 结果

Oracle labels 正式输出：

```text
output dir:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/

main log:
.../oracle_labels_full_scale_2026-06-16/main.log

status:
.../oracle_labels_full_scale_2026-06-16/status.json

labels:
.../oracle_labels_full_scale_2026-06-16/window_oracle_labels.parquet

summary:
.../oracle_labels_full_scale_2026-06-16/window_oracle_summary.csv
```

Oracle 状态：

```text
status: completed
input_record_count: 116,375,850
sample_key_unique_count: 23,275,170
output_row_count: 46,550,340
metric_counts: mae=23,275,170, mse=23,275,170
duplicate_or_order_violations: 0
missing sample_key/oracle_model/oracle_value/metric: 0
elapsed_sec: 646.805
gpu_used: false
```

TSF enrichment 正式输出：

```text
output dir:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/tsf_enrichment_full_scale_2026-06-16/

main log:
.../tsf_enrichment_full_scale_2026-06-16/main.log

status:
.../tsf_enrichment_full_scale_2026-06-16/status.json

enrichment:
.../tsf_enrichment_full_scale_2026-06-16/sample_tsf_enrichment.parquet

missing summary:
.../tsf_enrichment_full_scale_2026-06-16/tsf_missing_summary.csv
```

TSF 状态：

```text
status: completed
input_record_count: 116,375,850
sample_key_unique_count: 23,275,170
sample_key_unique: true
duplicate_sample_key_count: 0
cluster/group_name/forecastability_cat/season_strength_cat/trend_strength_cat/cv_cat/missing_ratio_cat missing_count: 0
elapsed_sec: 586.541
gpu_used: false
```

正式 validation：

```text
validation dir:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_tsf_validation_2026-06-16/

validation_summary:
.../oracle_tsf_validation_2026-06-16/validation_summary.json

status: passed
expected_sample_count: 23,275,170
expected_oracle_rows: 46,550,340
oracle_rows: 46,550,340
oracle_sample_key_unique_count: 23,275,170
tsf_rows: 23,275,170
tsf_sample_key_unique_count: 23,275,170
oracle_minus_tsf_count: 0
tsf_minus_oracle_count: 0
oracle_missing_counts: all 0
tsf_missing_counts: all 0
```

Oracle summary 中，test split 的 MAE oracle 上限示例：

```text
TEST_DATA_HOUR: best_single=DLinear, best_single_value=0.339654, oracle_value=0.261869, oracle_gap_pct=22.90%
TEST_DATA_MIN: best_single=PatchTST, best_single_value=0.532548, oracle_value=0.346562, oracle_gap_pct=34.92%
```

本轮没有重新 merge，没有删除正式 merged cache 或 validation 结果，没有启动 visual router、calibration 或 TimeFuse fusor 训练，也没有生成伪图像 tensor 或 ViT embedding 产物。

## 结论

Stage 1 `96_48_S` full-scale oracle labels 与 TSF enrichment 已完成并通过 join/覆盖验证。正式可引用输入为：

```text
oracle labels:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/window_oracle_labels.parquet

TSF enrichment:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/tsf_enrichment_full_scale_2026-06-16/sample_tsf_enrichment.parquet

validation:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_tsf_validation_2026-06-16/validation_summary.json
```

这些产物均以 `sample_key` 为 join 键，覆盖范围与正式 merged cache 的 `23,275,170` 个 sample_key 完全一致，可作为后续 metadata baseline、TimeFuse-style fusor 压力测试、streaming visual router 或 calibration 的正式监督/元信息输入。

## 下一步方案

1. 若继续非视觉 baseline，应先让 evaluator 支持 Parquet 输入或生成受控切片 CSV，避免 full-scale CSV 长表造成不必要 I/O。
2. 若继续 TimeFuse-style fusor，应先做小切片内存压力测试，再基于已完成的 TimeFuse feature cache、oracle labels 和 merged prediction cache 设计 streaming/memmap 路径。
3. 若继续 visual router，应使用本轮 oracle labels 与 TSF enrichment 作为监督和分层评估输入，不重新生成 prediction cache。
