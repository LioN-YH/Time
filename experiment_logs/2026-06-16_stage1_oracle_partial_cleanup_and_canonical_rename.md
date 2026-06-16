# Stage 1 Oracle 半成品清理与正式目录重命名

日志日期：2026-06-16 14:27:28 CST

## 目的

清理 Stage 1 `96_48_S` full-scale oracle labels 首版慢实现留下的半成品目录，并将正式 completed 产物从 `_v2` 目录重命名为无后缀 canonical 目录，避免后续读取时误以为存在多个正式版本。

## 背景

此前 oracle labels 首版输出目录为：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/
```

该目录的 `status.json` 为 `stopped_for_optimization`，只处理了 `99,999` 个 sample_key、`199,998` 条 oracle label，是逐 `sample_key` Python groupby 吞吐不足后停止的半成品，不作为正式结果引用。

正式 completed 产物原目录为：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16_v2/
```

该目录覆盖 `23,275,170` 个 sample_key，输出 `46,550,340` 条 `mae/mse` oracle label，并已通过 oracle/TSF join 覆盖验证。

## 操作

1. 执行删除前安全检查：
   - 半成品目录 `status=stopped_for_optimization`；
   - `_v2` 正式目录 `status=completed`；
   - `_v2` 正式目录 `sample_key_unique_count=23,275,170`；
   - `_v2` 正式目录 `output_row_count=46,550,340`。
2. 精确删除半成品目录：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/
```

3. 将正式目录重命名为无后缀 canonical 路径：

```text
from:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16_v2/

to:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/
```

4. 更新正式 oracle `status.json` 中的 `output_dir`、`output_path`、`summary_path`，并记录 `renamed_from`、`renamed_to` 和 `rename_note`。
5. 使用 canonical oracle 目录重跑 oracle/TSF validation，更新：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_tsf_validation_2026-06-16/validation_summary.json
```

6. 同步更新 `experiment_logs/README.md`、`WORKSPACE_STRUCTURE.md` 和上一条 oracle/TSF 生成日志中的路径引用。

## 结果

当前正式 oracle labels 目录为：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/
```

当前 `_v2` 目录已不存在。

正式 oracle 状态：

```text
status: completed
sample_key_unique_count: 23,275,170
output_row_count: 46,550,340
output_path: .../oracle_labels_full_scale_2026-06-16/window_oracle_labels.parquet
summary_path: .../oracle_labels_full_scale_2026-06-16/window_oracle_summary.csv
```

重跑 validation 结果：

```text
status: passed
oracle_rows: 46,550,340
oracle_sample_key_unique_count: 23,275,170
tsf_rows: 23,275,170
tsf_sample_key_unique_count: 23,275,170
oracle_minus_tsf_count: 0
tsf_minus_oracle_count: 0
```

本次没有删除正式 `merged_cache/`、`merged_cache_validation/`、TSF enrichment、oracle/TSF validation 或其他正式结果。

## 结论

首版 stopped oracle 半成品已清理，正式 oracle labels 现在使用无后缀 canonical 目录。后续引用 oracle labels 时应使用：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/window_oracle_labels.parquet
```

## 下一步方案

1. 后续 baseline、TimeFuse-style fusor 压力测试或 visual router 应引用 canonical oracle 目录，不再引用 `_v2`。
2. 若脚本或文档中出现 `_v2` 路径，应视为陈旧引用并改为无后缀路径。
