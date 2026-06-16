# Stage 1 `96_48_S` Full-Scale Completed Shard 严格 Audit

日志日期：2026-06-15 03:43:43 CST

## 目的

在 Stage 1 `96_48_S` 正式 full-scale 全候选窗口 prediction cache 长跑过程中，补充此前尚未严格抽检的新增 completed shard，确认已完成 shard 的 manifest、packed 数组索引、`sample_key + model_name` 唯一性和共享 `y_true` 对齐口径是否仍然满足正式 merge 前置约束。

## 背景

上一个 handoff 的最新状态停留在 `2026-06-15 03:21:21 CST`，当时轻量状态为 `completed=20`、`running=5`、`failed=0`，严格 CSV 抽检只覆盖到 12 个 completed shard，即四个非 ES 专家的 shard 0000 到 0002。新增 shard 0003/0004 尚未严格抽检，不满足 merge 条件。

本次仍处于 prediction cache 生产和监控阶段。正式 merge 条件没有变化：必须等五个专家各 64 个 sample shard 全部完成，即 `completed=320` 且 `failed=0` 后，才能执行 launcher `status.json` 中的 `merge_command`。

## 操作

1. 复核当前仓库、资源和运行状态：
   - `git status --short`
   - `git rev-parse --short HEAD`
   - `nvidia-smi --query-gpu=...`
   - `df -h /data2 /home`
   - `tmux ls`
   - `ps -eo ... | rg build_prediction_cache_from_manifest.py`
2. 读取正式输出根目录和 launcher 目录：
   - `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/HANDOFF.md`
   - `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/status.json`
   - `prediction_cache_full_scale_launcher/shards/*/sample_shard_*/status.json`
3. 使用 `quito` Python 执行只读 audit：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python
   ```

   audit 读取快照时所有 `status=completed` 的 shard，不触碰正在运行的 shard，也不改写 prediction cache。
4. audit 对每个 completed shard 检查：
   - `manifest.csv` 是否存在；
   - `validate_manifest_frame(..., require_shared_y_true_path=True)` 是否通过；
   - 行数是否与 `status.json` 的 `record_count` / `sample_manifest_total_count` 一致；
   - `sample_key + model_name` 是否重复；
   - `model_name` 是否与目录专家一致；
   - `array_storage` 是否为 `packed_npy_v1`；
   - `y_true_row_index` / `y_pred_row_index` 是否落在对应 `.npy` 第一维范围内；
   - 同一 packed 文件内 row index 是否唯一。
5. audit 按 sample shard 合并已完成专家，检查：
   - 跨已完成专家的 `sample_key + model_name` 是否重复；
   - 同一 `sample_key` 的稳定元信息是否一致；
   - 同一 `sample_key` 的 `y_true_path + y_true_row_index` 是否一致；
   - 当前已完成专家数量分布是否符合预期。
6. 将 audit 输出写入：

   ```text
   /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/audits/2026-06-15_033558_completed_manifest_audit/
   ```

7. audit 后再次做轻量状态复核，并更新根目录 `HANDOFF.md` 与正式输出根目录 `HANDOFF.md`。

## 结果

audit 快照时间为 `2026-06-15 03:35:58 CST`，状态如下：

- `status_files=33`；
- `completed=28`；
- `running=5`；
- `failed=0`；
- 四个非 ES 专家均已完成 shard 0000 到 0006；
- 四个非 ES 专家正在运行 shard 0007；
- `ES` 仍在运行 shard 0000。

audit 输出文件：

- `audit_summary.json`
- `audit_summary.md`
- `status_snapshot.csv`
- `completed_shards.csv`
- `running_shards.csv`
- `failed_shards.csv`
- `per_shard_manifest_audit.csv`
- `sample_shard_coverage_audit.csv`

严格检查结果：

- 严格覆盖 `28` 个 completed shard；
- 共读取并校验 `10,182,900` 行 completed shard manifest；
- 按 sample shard 合并覆盖检查 `2,545,725` 个 sample_key-shard 组合；
- `array_storage_values=['packed_npy_v1']`；
- `global_duplicate_sample_model=0`；
- `shared_y_true_violations=0`；
- `stable_metadata_violations=0`。

按专家 completed 覆盖：

```text
CrossFormer: 0000,0001,0002,0003,0004,0005,0006
DLinear: 0000,0001,0002,0003,0004,0005,0006
NaiveForecaster: 0000,0001,0002,0003,0004,0005,0006
PatchTST: 0000,0001,0002,0003,0004,0005,0006
```

每个已 audit 的 sample shard 当前均只有四个非 ES 专家结果，专家数量分布为 `{4: 363675}`。这是因为 `ES` 尚未完成首个 shard，不是共享 `y_true` 或 sample key 对齐错误。

audit 后在 `2026-06-15 03:41:31 CST` 再次轻量复核，状态已前进为：

- `status_files=37`；
- `completed=32`；
- `running=5`；
- `failed=0`；
- 四个非 ES 专家已完成 shard 0000 到 0007，并运行 shard 0008；
- `ES` 仍在 shard 0000。

最新运行 Python 子进程：

```text
DLinear PID 461965 -> sample_shard_0008_of_0064
PatchTST PID 463302 -> sample_shard_0008_of_0064
CrossFormer PID 464446 -> sample_shard_0008_of_0064
ES PID 427585 -> sample_shard_0000_of_0064
NaiveForecaster PID 462203 -> sample_shard_0008_of_0064
```

写入日志前在 `2026-06-15 03:45:47 CST` 再次做轻量 sanity check，现场状态继续前进为 `status_files=39`、`completed=34`、`running=5`、`failed=0`。其中 `DLinear`、`NaiveForecaster` 已完成到 shard 0008 并运行 shard 0009，`PatchTST`、`CrossFormer` 已完成到 shard 0007 并运行 shard 0008，`ES` 仍在 shard 0000。本次 03:45 检查是轻量状态修正，不是新的严格 audit。

资源状态：

- 正式输出根目录约 `14G`；
- `/data2` 可用空间约 `2.6T`；
- `/home` 仍接近满盘，仅约 `18G` 可用。

## 结论

截至 audit 快照，已完成的 28 个 prediction cache shard 满足正式 packed cache 口径：manifest schema、行数、`sample_key + model_name` 唯一性、`packed_npy_v1` 数组索引、稳定元信息和共享 `y_true_path + y_true_row_index` 均通过检查。

当前阶段仍不能 merge。原因是最新轻量状态只有 `completed=34`，远低于五专家 64 shard 全部完成所需的 `completed=320`，且 `ES` 仍未完成首个 shard。

## 下一步方案

1. 继续监控五专家 worker，重点关注 `ES/sample_shard_0000_of_0064` 是否正常完成。
2. 如果四个非 ES 专家继续快速完成新 shard，可定期做只读 audit，但不要过于频繁读取全部 completed CSV，以免浪费 I/O。
3. 出现 failed shard 时只精确定位和重跑失败 shard，不删除已完成 shard。
4. 只有当 `completed=320` 且 `failed=0` 时，才执行 launcher `status.json` 中的 `merge_command`。
5. merge 后必须校验 `sample_key + model_name` 唯一、五专家完整、共享 `y_true_path + y_true_row_index` 一致和 `array_storage=packed_npy_v1`。
6. merge 完成并通过校验后，再生成 oracle labels、TSF enrichment/baseline，随后使用 `train_visual_router_online_streaming.py` 跑 1 epoch 正式 streaming visual router，并执行 soft fusion calibration。
