# Stage 1 `96_48_S` Full-Scale ES 完成情况检查

日志日期：2026-06-15 22:25:26 CST

## 目的

确认 Stage 1 `96_48_S` 正式 full-scale prediction cache 中 `ES` 的最终完成情况，判断当前是否已经解除对 merge 的阻塞。

## 背景

此前 `ES` 是整轮 full-scale prediction cache 的最后慢路径，曾通过原 worker、backfill 和 accelerator 三个并行会话推进。检查前需要确认是否已经全部写完、是否仍有 running/failed shard，以及相关 tmux 会话是否还有活动进程。

## 操作

1. 只读检查 `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/shards/` 下所有 `status.json`。
2. 汇总五个专家的 shard 状态，重点统计 `ES` 的 completed/running/failed/missing。
3. 查看 `stage1_fullscale_launcher_shell`、`stage1_es_backfill_0016_0063`、`stage1_es_accelerator_0010_0015_0048_0063` 相关日志末尾，确认各 lane 是否已经结束。
4. 检查 tmux 会话与 `pgrep`，确认没有残留的 `build_prediction_cache_from_manifest.py --models ES` 进程。

## 结果

检查时间为 `2026-06-15 22:25:26 CST` 时，状态为：

```text
global_status_files=320
completed=320
running=0
failed=0
```

五个专家均已完成全部 `64/64` 个 shard：

- `DLinear`：64/64
- `PatchTST`：64/64
- `CrossFormer`：64/64
- `NaiveForecaster`：64/64
- `ES`：64/64

`ES` 具体结果为：

```text
completed: 0000-0063
running: none
failed: none
missing: none
```

日志确认：

- 原 `ES` worker 在 `2026-06-15 18:50:16 CST` 完成；
- `stage1_es_backfill_0016_0063` 的四个 lane 已在 `2026-06-15 17:23:06 CST` 到 `17:42:13 CST` 之间结束；
- `stage1_es_accelerator_0010_0015_0048_0063` 的四个 lane 已在 `2026-06-15 17:23:06 CST` 到 `17:29:39 CST` 之间结束。

当前 `tmux` 里只剩与旧 launcher 相关的 shell 会话，不再存在活动的 ES 预测进程。

## 结论

`ES` 已经完全完成，`completed=64/64`，且没有 `running` 或 `failed` shard。整轮五专家 prediction cache 现在满足 `completed=320 && failed=0` 的 merge 前置条件。

## 下一步方案

下一步可以进入：

1. merge prediction cache；
2. 校验 `sample_key + model_name` 唯一、五专家完整、共享 `y_true_path + y_true_row_index` 一致；
3. 在 merged cache 上生成 oracle labels；
4. 继续正式 streaming visual router 和 soft fusion calibration。
