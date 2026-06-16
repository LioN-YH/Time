# Stage 1 `96_48_S` Full-Scale Merge 启动与 Handoff

日志日期：2026-06-15 23:49:51 CST

## 目的

在五专家 full-scale prediction cache 已全部完成后，启动正式 `merged_cache` 合并任务，并在当前上下文达到项目 handoff 阈值前记录可接手状态。

## 背景

上一阶段已经确认 `prediction_cache_full_scale_launcher/shards/` 下共有 `320` 个 completed shard，覆盖 `DLinear`、`PatchTST`、`CrossFormer`、`ES` 和 `NaiveForecaster` 五个专家，每个专家 `64/64` 个 sample shard。本轮任务边界只允许做 merge 与完整性校验，不进入 oracle labels、streaming visual router 或 calibration。

## 操作

1. 阅读并复核 `AGENTS.md`、正式输出目录 `HANDOFF.md`、`experiment_logs/README.md`、ES 完成检查日志、Stage 1 README、cache contract、protocol/plan、`merge_prediction_cache_shards.py` 和 launcher `status.json`。
2. 启动前检查资源：`/data2` 约 `2.5T` 可用、inode 充足；CPU 和 I/O 空闲；4 张 RTX 3090 基本空闲。merge 属于 CPU/I/O-bound CSV 与 `.npy` 处理，因此没有改成 GPU 路径。
3. 只读预检 launcher `status.json` 中的 `merge_command`：该字段为 list，`argc=325`，输入 `320` 个 shard，输出目录为 `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache`。
4. 并行/只读预检所有 shard：`status_counts={'completed': 320}`，五专家各 `64` 个 shard；逐 manifest 行数统计得到五专家各 `23,275,170` 行，总计 `116,375,850` 行，预期 sample_key 数为 `23,275,170`；首行 `array_storage` 检查全部为 `packed_npy_v1`。
5. 发现原 `merge_prediction_cache_shards.py` 的 packed 分支会逐行反复 `np.load` 读取 `y_true`，在 `116,375,850` 行 full-scale manifest 上不可接受；因此在同一个正式 merge 脚本内增加 `packed_npy_v1_streaming_by_sample_shard` 分支。该分支仍由 launcher `status.json` 的同一条 `merge_command` 调用，不新增另一套命令入口。
6. 使用 Quito 环境执行语法检查，并用历史 full-scale dry-run 的 2 个 sample shard 做回归：输出 `20` 行、`4` 个 sample_key，`sample_key + model_name` 无重复，五专家完整，共享 `y_true_path + y_true_row_index` 一致，`array_storage=['packed_npy_v1']`。
7. 后台启动 `status.json` 字段 `merge_command`，包装层只负责保活、日志和 PID 记录。启动信息写入 `prediction_cache_full_scale_launcher/merge_command_run_status.json`。

## 结果

正式 merge 已启动但尚未完成。当前 handoff 快照为：

```text
启动时间：2026-06-15 23:47:23 CST
PID/PGID：675597 / 675597
命令来源：/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/status.json 字段 merge_command
主日志：/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merge_command_2026-06-15_234723.log
输出目录：/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache
停止命令：kill -TERM -675597
```

截至 `2026-06-15 23:49:51 CST`：

```text
merged_cache/status.json: running
最新进度日志：[2026-06-15 23:49:41 CST] merged sample_shard=0000 progress=1/64 records_written=1818375
merged_cache 当前占用：约 1.2G
```

本轮没有启动 oracle labels、streaming router 或 calibration。

## 结论

full-scale merge 已按 launcher `status.json` 的 `merge_command` 正式进入后台执行，并已完成第 1 个 sample shard 的合并写出。由于当前上下文已经达到项目 handoff 阈值，暂停继续监控和验证，改由新窗口接手。

## 下一步方案

1. 新窗口先读取根目录 `HANDOFF.md` 和正式输出目录 `HANDOFF.md`，确认 PID `675597` 是否仍在运行。
2. 使用 `tail -n 80` 监控主日志；若 `merged_cache/status.json` 变为 `completed`，进入完整性校验。
3. 完整性校验必须记录并保存：manifest 行数、sample_key 数、专家覆盖计数、`sample_key + model_name` 唯一、每个 sample_key 五专家完整、共享 `y_true_path + y_true_row_index` 一致、`array_storage=packed_npy_v1`。
4. 若 merge 失败，只根据 `merged_cache/status.json` 和主日志定位失败 sample shard 或阶段；不要删除任何已完成 prediction cache shard。

## 暂停前补充监控

补充时间：2026-06-15 23:56:28 CST

用户建议在确认 merge 正常后台运行后暂停窗口会话，后续再唤醒继续 merge 后步骤。按该建议做最后一次轻量检查：

```text
PID/PGID: 675597 / 675597
进程状态：running
merged_cache/status.json: running
最新进度日志：[2026-06-15 23:56:04 CST] merged sample_shard=0008 progress=9/64 records_written=16365375
merged_cache 当前占用：约 12G
```

从 `sample_shard=0000` 到 `sample_shard=0008` 的日志间隔估算，最近平均约 `48` 秒完成一个 sample shard；剩余 `55` 个 sample shard 约需 `44` 分钟，考虑收尾写 `metadata.json`、`status.json` 和 `merge_summary.md`，预计在 `2026-06-16 00:40-00:45 CST` 左右完成。此处仅暂停窗口会话，不停止后台进程。
