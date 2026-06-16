# Stage 1 `96_48_S` Full-Scale ES Backfill 并行加速

日志日期：2026-06-15 03:59:44 CST

## 目的

在 Stage 1 `96_48_S` 正式 full-scale prediction cache 长跑中，`ES` 单 worker 明显慢于其他专家。为避免 `ES` 顺序运行 64 个 sample shard 成为不可接受的总时长瓶颈，本次新增一个仅覆盖 `ES` 高编号 shard 的补充 tmux launcher，在不删除、不覆盖已完成 shard 的前提下并行推进 `ES` prediction cache。

## 背景

截至本次操作前，四个非 ES 专家已经快速推进到 shard 0008 之后，而 `ES` 原 worker 仍在低编号 shard。机器资源检查显示 64 个 CPU 线程、约 206GiB 可用内存，具备增加少量 CPU 并行的条件。现有主 launcher 对每个 shard 都有 `status=completed` skip 逻辑，因此补充完成高编号 `ES` shard 后，原 `ES` worker 后续到达这些 shard 时会自动跳过。

## 操作

1. 检查 `ES` 原 worker 状态：
   - `ES/sample_shard_0000_of_0064` 已于 `2026-06-15 03:50:43 CST` 完成；
   - 原 worker 已进入 `ES/sample_shard_0001_of_0064`；
   - `ES` 进程仍有高 CPU 使用，说明不是卡死。
2. 新增 backfill 目录：

   ```text
   /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/es_parallel_backfill_0016_0063/
   ```

3. 写入 backfill 文件：
   - `launcher.sh`：早期直接后台尝试使用；
   - `run_lane.sh`：单 lane 前台执行脚本；
   - `tmux_launcher.sh`：权威启动入口；
   - `launch_plan.md`；
   - `status.json`。
4. 第一次直接从 exec 会话运行 `launcher.sh`，发现 lane 进程未保活，只留下 0 字节 `main.log`，未生成有效 shard `status.json`。该尝试没有产出有效 shard，不作为成功运行记录。
5. 改用持久 tmux 会话启动：

   ```bash
   bash /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/es_parallel_backfill_0016_0063/tmux_launcher.sh
   ```

6. backfill 分 4 个 lane：
   - lane 0：`0016,0020,...,0060`
   - lane 1：`0017,0021,...,0061`
   - lane 2：`0018,0022,...,0062`
   - lane 3：`0019,0023,...,0063`

## 结果

tmux 会话已成功创建：

```text
stage1_es_backfill_0016_0063
```

当前窗口：

```text
lane0
lane1
lane2
lane3
```

最新状态快照：

- `status_files=52`；
- `completed=43`；
- `running=9`；
- `failed=0`。

本轮结束前在 `2026-06-15 04:02:18 CST` 做固定轻量快照，状态为 `status_files=53`、`completed=44`、`running=9`、`failed=0`。各专家运行 shard 未发生失败：`DLinear` 运行 0011、`NaiveForecaster` 运行 0011、`PatchTST` 运行 0010、`CrossFormer` 运行 0010、`ES` 运行 0001/0016/0017/0018/0019。

各专家状态：

- `DLinear`：完成到 shard 0010，运行 0011；
- `NaiveForecaster`：完成到 shard 0010，运行 0011；
- `PatchTST`：完成到 shard 0009，运行 0010；
- `CrossFormer`：完成到 shard 0009，运行 0010；
- `ES`：完成 shard 0000，运行 shard 0001、0016、0017、0018、0019。

ES backfill 子进程：

```text
lane0 PID 471036 -> ES sample_shard_0016_of_0064
lane1 PID 471044 -> ES sample_shard_0017_of_0064
lane2 PID 471057 -> ES sample_shard_0018_of_0064
lane3 PID 471053 -> ES sample_shard_0019_of_0064
```

资源状态仍可接受：

- `/data2` 约 `2.6T` 可用；
- `/home` 仍约 `18G` 可用且接近满盘；
- 内存约 `206GiB` available；
- backfill 启动后未出现 failed shard。

## 结论

ES backfill 的 tmux 版启动成功，当前 4 个高编号 `ES` shard 已进入 `status=running`。第一次直接后台尝试未保活，不产生有效 shard，应忽略。当前仍远未达到 merge 条件，不能执行 merge、oracle、router 或 calibration。

## 下一步方案

1. 继续监控原 `stage1_fullscale_launcher_shell` 和新 `stage1_es_backfill_0016_0063`。
2. backfill 监控命令：

   ```bash
   tmux list-windows -t stage1_es_backfill_0016_0063
   BACKFILL=/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/es_parallel_backfill_0016_0063
   tail -n 80 "$BACKFILL/logs/lane_0.log"
   tail -n 80 "$BACKFILL/logs/lane_1.log"
   tail -n 80 "$BACKFILL/logs/lane_2.log"
   tail -n 80 "$BACKFILL/logs/lane_3.log"
   ```

3. 如需停止 backfill，仅停止该 tmux 会话：

   ```bash
   tmux kill-session -t stage1_es_backfill_0016_0063
   ```

4. 若 backfill 某个 shard 失败，只精确重跑对应 `ES/sample_shard_XXXX_of_0064`，不要删除其他 completed shard。
5. 等所有五专家 shard 达到 `completed=320` 且 `failed=0` 后，再执行正式 merge 和后续 oracle/router/calibration。
