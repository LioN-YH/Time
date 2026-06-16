# Stage 1 `96_48_S` full-scale TimeFuse feature cache 完成情况检查

日志日期：2026-06-16 13:59:55 CST

## 目的

检查昨日启动的 Stage 1 `96_48_S` full-scale TimeFuse-derived 单变量元特征预计算任务是否已经完成，并确认产物是否满足后续 TimeFuse-style fusor baseline 使用的基本条件。

## 背景

上一份日志 `2026-06-16_stage1_timefuse_full_scale_feature_cache_launcher.md` 记录了正式 8 lane CPU 后台任务已经启动，输出目录为：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/
```

该任务目标是从 full-scale sample manifest 独立预计算 17 维 `timefuse_single_variable_meta_v1` 单变量结构特征。该特征只允许使用历史窗口 `x`，不读取未来 `y`、专家预测、prediction cache manifest、oracle label 或 TSF label 文件。

## 操作

1. 使用 `rg` 在 `HANDOFF.md`、`experiment_logs/`、`visual_router_experiments/` 和 `WORKSPACE_STRUCTURE.md` 中定位 TimeFuse feature cache 任务的正式输出目录、预期总行数和检查口径。
2. 使用 `ps -eo pid,ppid,stat,etime,cmd | rg -i "timefuse|metadata|feature_cache|96_48_S|train_visual|launcher|python"` 检查是否还有相关 worker 或 launcher 进程。
3. 查看正式输出目录、根级 `main.log` 和 8 个 lane 日志尾部，确认每个 lane 是否完成。
4. 使用 Quito 环境解释器 `/home/shiyuhong/application/miniconda3/envs/quito/bin/python` 汇总 64 个 shard 的 `status.json` 和 `metadata.json`，检查 shard 状态、声明样本数、特征维度和特征版本。
5. 顺序扫描 64 个 `feature_cache.csv`，统计实际 CSV 数据行数、表头一致性，并与 shard 元数据声明行数交叉验证。
6. 抽看 `sample_shard_0000_of_0064` 与 `sample_shard_0063_of_0064` 的 `status.json` 和 CSV 表头/首行，确认字段口径包含稳定 sample 元信息和 17 个 TimeFuse-derived 特征列。

## 结果

- 当前没有发现仍在运行的 TimeFuse feature cache worker、launcher 或相关 Python 进程。
- 8 个 lane 均正常完成：
  - lane 00 完成于 `2026-06-16 06:03:18 CST`；
  - lane 01 完成于 `2026-06-16 06:03:39 CST`；
  - lane 02 完成于 `2026-06-16 06:04:56 CST`；
  - lane 03 完成于 `2026-06-16 06:11:09 CST`；
  - lane 04 完成于 `2026-06-16 06:04:24 CST`；
  - lane 05 完成于 `2026-06-16 06:01:04 CST`；
  - lane 06 完成于 `2026-06-16 06:01:58 CST`；
  - lane 07 完成于 `2026-06-16 06:15:14 CST`。
- 正式输出目录大小约 `9.3G`，包含 64 个 shard 目录、64 个 `status.json` 和 64 个 `feature_cache.csv`。
- shard 状态汇总：

```text
status_counts {'completed': 64}
metadata_status_counts {'completed': 64}
status_rows_total 23275170
metadata_rows_total 23275170
expected_total 23275170
feature_dims {17: 64}
feature_versions {'timefuse_single_variable_meta_v1': 64}
bad_status [] count= 0
```

- 全量 CSV 行数扫描结果：

```text
shard_count 64
status_counts {'completed': 64}
csv_rows_total 23275170
expected_total 23275170
feature_dims {17: 64}
feature_versions {'timefuse_single_variable_meta_v1': 64}
header_variants 1
header_count 64 columns 29
bad_count 0
```

- CSV 表头为 29 列，其中前 12 列为版本和 sample 稳定元信息，后 17 列为 TimeFuse-derived 单变量特征：

```text
mean, std, min, max, skewness, kurtosis,
autocorrelation_mean, stationarity,
rate_of_change_mean, rate_of_change_std,
autoreg_coef_mean, residual_std_mean,
frequency_mean, frequency_peak,
spectral_entropy, spectral_skewness, spectral_kurtosis
```

- 根级 `status.json` 与 `metadata.json` 仍显示 `status: running`。结合文件内容和时间戳判断，它们是 launcher 生成时写入的静态状态快照，并未在 8 个 lane 完成后自动回写；本次完成判断以 lane 日志、64 个 shard 的 `status.json`/`metadata.json` 和全量 CSV 行数扫描为准。

## 结论

Stage 1 `96_48_S` full-scale TimeFuse-derived 单变量 feature cache 预计算主体已经完成。64 个 shard 全部 `completed`，实际 CSV 行数总和为 `23,275,170`，与 full-scale sample manifest 预期样本数一致；所有 shard 均为 17 维 `timefuse_single_variable_meta_v1`，表头一致，未发现缺失 shard、失败 shard、行数不匹配或临时残留文件。

需要注意的是，根级 launcher 状态文件未自动从 `running` 更新为 `completed`，后续引用完成状态时应优先引用本日志、lane 日志和 shard 级状态，而不是单独引用根级 `status.json` 的 `status` 字段。

## 下一步方案

1. 若要进入 TimeFuse-style fusor full-scale baseline，先确认正式 `merged_cache/` 和后续 oracle labels / TSF enrichment 均已生成并通过完整性检查。
2. 在正式 full-scale fusor 前，建议先做小切片压力测试，确认 `evaluate_router_baselines.py --timefuse-fusor on` 在大规模 prediction cache 与 feature cache 对齐时的内存峰值和运行时间。
3. 如需长期消除歧义，可后续增加一个只读汇总/收尾脚本，生成独立的 feature cache completion summary，而不是覆盖原 launcher 生成时的根级 `status.json`。
