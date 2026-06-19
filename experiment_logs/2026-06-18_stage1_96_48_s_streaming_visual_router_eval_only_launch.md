# Stage 1 `96_48_S` Streaming Visual Router Full-Scale Eval-Only 启动

日志日期：2026-06-18 01:04:15 CST

## 目的

按照 Stage 1 `96_48_S` full-scale 后续任务拆分，先执行 visual router checkpoint 的 eval-only 步骤，在 test split 上产出 full-scale `visual_router_predictions.csv`、`visual_router_summary.csv` 和 raw soft fusion 相关输出，为后续 soft fusion calibration 与最终汇总提供输入。

## 背景

上一阶段已经完成 `96_48_S` full-scale streaming visual router v2 的 1 epoch train-only 训练，正式 checkpoint 为：

```text
/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/latest_96_48_S.pt
```

该训练输出 `status=completed`、`phase=train_only_done`、`completed_epochs=1`，但因为使用了 `--train-only`，没有生成 test predictions，也没有完成 calibration。本轮只做 eval-only，不追加训练 epoch。

## 操作

1. 按 DeepSeek sidecar skill 规范读取本地 skill 说明，并启动只读预检任务：

   ```text
   task_id=stage1-vr-evalonly-precheck
   ```

   任务约束为只读检查 checkpoint、GPU、进程和轻量监控命令，不扫描 52GB merged manifest。

2. 检查当前资源：
   - 未发现正在运行的 `train_visual_router_online_streaming.py` 同类进程；
   - GPU2/GPU3 初始显存均约 `12 MiB`；
   - `/data2` 可用约 `2.3T`；
   - 系统可用内存约 `240GiB`。

3. 新建 eval-only 输出目录：

   ```text
   /data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/
   ```

4. 写入并检查启动文件：
   - `launcher.sh`
   - `command.sh`
   - `launch_metadata.json`

5. 使用后台 `setsid` 启动 eval-only：

   ```text
   /data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/command.sh
   ```

   核心参数：

   ```text
   CUDA_VISIBLE_DEVICES=2,3
   --resume-checkpoint /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/latest_96_48_S.pt
   --epochs 0
   --embedding-batch-size 128
   --batch-size 64
   --device cuda
   --vit-data-parallel
   --local-files-only
   --period-selection fixed_candidates
   --dtype auto
   --chunk-read-rows 1000000
   --status-update-interval 100
   ```

6. 启动 DeepSeek sidecar 轻量监控任务：

   ```text
   task_id=stage1-vr-evalonly-monitor
   ```

   任务约束为只采样 3 轮，只允许 `ps`、`cat status.json`、`tail -n <=80 main.log`、`stat`、`nvidia-smi`、`free`、`sleep`，禁止扫描 52GB merged manifest。

7. 主线程完成短时健康检查：
   - 启动后约 2 秒确认 PID/PGID；
   - 启动后约 26 秒检查 `status.json`、GPU、文件 mtime；
   - 启动后约 5 分钟再次检查进程、日志、SQLite 临时索引和 GPU。

## 结果

eval-only 已成功后台启动：

```text
PID=1264073
PGID=1264073
PPID=1
```

当前输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/
```

启动后约 5 分钟的健康检查结果：

```text
PID 1264073 alive
STAT=Rsl
ELAPSED=00:05:08
RSS=17,545,916 KiB
```

`status.json` 当前仍处于 init 阶段：

```text
status=running
phase=init
resume_checkpoint=/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/latest_96_48_S.pt
completed_epochs=1
```

`main.log` 已出现 manifest index 进度：

```text
[manifest_index] chunks=1 rows_seen=1000000 matched_rows=1000000 target_sample_keys=13924650
```

SQLite 临时索引已经开始增长：

```text
prediction_manifest_index.sqlite.tmp = 2.1G
```

GPU 状态：

```text
GPU2 memory=693 MiB, util=0
GPU3 memory=12 MiB, util=0
```

这说明当前仍处于 CPU/I/O 为主的 test prediction manifest SQLite index 构建阶段；进入 `test_predict` 后 GPU2/GPU3 才会明显参与 ViT 前向。

DeepSeek sidecar 状态：

```text
stage1-vr-evalonly-precheck: idle
stage1-vr-evalonly-monitor: idle
```

尝试 resume `stage1-vr-evalonly-monitor` 取简报时，wrapper 命中旧 profile 兼容错误：

```text
Error: legacy `profile = "ds-sidecar"` config is no longer supported; use `--profile ds-sidecar` with `ds-sidecar.config.toml` instead
```

该错误发生在后续取简报阶段，不影响已启动的 eval-only 主进程；主线程已独立完成健康检查。

## 结论

Stage 1 `96_48_S` full-scale visual router eval-only 已成功转入后台运行。当前已确认：

1. 后台进程脱离当前会话，`PPID=1`；
2. 使用 checkpoint `latest_96_48_S.pt`，`--epochs 0`，没有追加训练；
3. GPU 限制为 `CUDA_VISIBLE_DEVICES=2,3`；
4. 已进入 test split SQLite index 构建，未见异常退出；
5. 输出目录已具备 launcher、PID、主日志、状态文件和临时索引。

当前尚未完成 eval-only，因此还没有最终 `visual_router_predictions.csv`、`visual_router_summary.csv` 或 calibration 输入。

## 下一步方案

1. 继续轻量监控：

   ```bash
   ps -p 1264073 -o pid,ppid,pgid,stat,etime,%cpu,%mem,rss,cmd
   cat /data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/status.json
   tail -n 120 /data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/main.log
   stat -c '%y %s %n' /data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/prediction_manifest_index.sqlite.tmp /data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/prediction_manifest_index.sqlite 2>/dev/null
   nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
   ```

2. 不要对 52GB merged manifest 执行 `wc -l`、`head`、`tail` 或全表扫描。
3. 等 `status.json` 变为 `completed`、`phase=done` 后，检查：
   - `visual_router_predictions.csv`
   - `visual_router_summary.csv`
   - `visual_router_soft_fusion_predictions.csv`
   - `visual_router_soft_fusion_summary.csv`
   - `visual_router_metadata.json`
4. eval-only 完成后，再进入 soft fusion calibration。注意 calibration 脚本当前可能仍需 full-scale streaming/SQLite 适配，不能默认直接全量加载 116M 行 manifest。
