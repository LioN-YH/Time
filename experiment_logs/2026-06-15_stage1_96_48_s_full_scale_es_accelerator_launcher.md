# Stage 1 `96_48_S` Full-Scale ES Accelerator 启动

日志日期：2026-06-15 13:37:01 CST

## 目的

进一步加速 Stage 1 `96_48_S` 正式 full-scale prediction cache 的 `ES` 剩余 shard。当前四个非 ES 专家已经完成，后续 merge、oracle、streaming router 和 calibration 都被 `ES` 未完成阻塞，因此在服务器 CPU 和内存仍有明显余量时，新增安全的 `ES` accelerator tmux launcher。

## 背景

加速前只读检查显示机器有 64 个 CPU 线程、约 217GiB available memory。已有 `ES` 原 worker 和 `stage1_es_backfill_0016_0063` 共 5 个 ES Python 进程，合计约使用 16 个 CPU 核，仍具备增加并行度的条件。

加速前状态为：

```text
completed=292
running=5
failed=0
```

`ES` 已完成 `36/64`，正在运行 `0008,0044,0045,0046,0047`，缺失 `0009-0015,0048-0063`。

## 操作

1. 只读检查资源、已有 tmux 会话、`ES` status 和 backfill lane 日志。
2. 确认主 launcher 只跳过 `completed` shard，不跳过 `running` shard。因此不能抢跑 `0009`，否则原 worker 完成 `0008` 后可能进入 `0009` 并造成双写风险。
3. 新增 accelerator 目录：

   ```text
   /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/es_accelerator_0010_0015_0048_0063/
   ```

4. 新增文件：
   - `run_lane.sh`
   - `tmux_launcher.sh`
   - `launch_plan.md`
   - `status.json`
5. accelerator 只覆盖尚未出现 `status.json` 的 `ES` shard：`0010-0015,0048-0063`。
6. 明确排除：
   - `0008`：原 worker 正在运行；
   - `0009`：保留给原 worker，避免双写；
   - `0044-0047`：原 backfill 正在运行。
7. 启动 tmux：

   ```bash
   bash /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/es_accelerator_0010_0015_0048_0063/tmux_launcher.sh
   ```

## 结果

已创建 tmux 会话：

```text
stage1_es_accelerator_0010_0015_0048_0063
```

窗口包括：

```text
lane0
lane1
lane2
lane3
```

加速后健康检查状态：

```text
status_files=301
completed=292
running=9
failed=0
```

当前 `ES` running shard：

```text
0008,0010,0011,0012,0013,0044,0045,0046,0047
```

新增 accelerator 进程已稳定运行，`0010-0013` 均进入 CPU 计算，未出现 failed shard。

## 结论

新增 accelerator 成功启动，并把 `ES` 并发度从 5 个进程提高到 9 个进程。当前仍不能 merge；必须继续等待 `completed=320` 且 `failed=0`。本次加速没有启动 oracle、router 或 calibration，也没有删除任何已完成 shard。

## 下一步方案

1. 继续监控三个会话：

   ```bash
   tmux list-sessions
   tmux list-windows -t stage1_fullscale_launcher_shell
   tmux list-windows -t stage1_es_backfill_0016_0063
   tmux list-windows -t stage1_es_accelerator_0010_0015_0048_0063
   ```

2. 监控 accelerator：

   ```bash
   ACCEL=/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/es_accelerator_0010_0015_0048_0063
   tail -n 80 "$ACCEL/logs/lane_0.log"
   tail -n 80 "$ACCEL/logs/lane_1.log"
   tail -n 80 "$ACCEL/logs/lane_2.log"
   tail -n 80 "$ACCEL/logs/lane_3.log"
   ```

3. 若需要停止 accelerator：

   ```bash
   tmux kill-session -t stage1_es_accelerator_0010_0015_0048_0063
   ```

4. 等全部五专家 prediction cache 达到 `completed=320` 且 `failed=0` 后，再单独进入 merge 与完整性校验目标。
