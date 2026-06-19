# Stage 1 TimeFuse-style Fusor Full-Scale Launcher 与正式启动

日志日期：2026-06-18 01:13:30 CST

## 目的

为 Stage 1 `96_48_S` full-scale TimeFuse-style fusor baseline 实现正式后台 launcher，完成启动前 preflight，覆盖 64 个 feature/prediction shard 启动正式长跑，并记录监控、停止和恢复方式。

## 背景

前序步骤已经完成 TimeFuse fusor streaming reader、train/eval 入口、1-shard smoke、checkpoint eval-only 复验和 2-shard 小切片压力测试。正式 full-scale 输入包括：

- 64 个 TimeFuse-derived feature shard；
- full-scale oracle labels parquet；
- 五专家 `packed_npy_v1` prediction shard manifests，共 320 个 manifest；
- 已完成并校验的 merged prediction cache，`record_count=116,375,850`、`sample_count=23,275,170`。

同时，`train_timefuse_fusor_streaming.py` 已经复用 shard-local SQLite + batch reader，不依赖全量 manifest lookup 或全量 DataFrame join。本轮目标是新增正式后台 launcher，而不是改动训练/reader 数据口径。

## 操作

1. 阅读并复核：
   - `AGENTS.md`
   - `HANDOFF.md`
   - `experiment_logs/2026-06-18_stage1_timefuse_fusor_streaming_reader.md`
   - `experiment_logs/2026-06-18_stage1_timefuse_fusor_streaming_train_eval_pressure.md`
   - `experiment_logs/2026-06-17_stage1_timefuse_fusor_full_scale_gpu_feasibility_review.md`
   - `visual_router_experiments/stage1_vali_test_router/stage1_timefuse_fusor_streaming_reader.py`
   - `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`
2. 新增正式 launcher：
   - `visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py`
   - 该 launcher 只负责 preflight、脚本生成、后台启动和接手信息写入；训练/eval 仍调用 `train_timefuse_fusor_streaming.py`。
3. launcher 支持：
   - preflight-only；
   - `setsid` 后台启动；
   - 写出 `pid.txt`、`pgid.txt`、`main.log`、`launcher.log`、`status.json`、`metadata.json`、`preflight_report.json`；
   - 写出 `command.sh`、`command_resume.sh`、`launcher.sh`、`stop.sh`、`resume.sh`；
   - `resume.sh` 优先检测 `checkpoints/latest_timefuse_fusor.pt`，若存在则用 `--resume-checkpoint` 跳过已完成 epoch 并继续 eval；若不存在则重新开始。
4. preflight 检查并通过：
   - feature shard 数量 `64`，64/64 为 `completed`；
   - feature status 行数合计 `23,275,170`；
   - prediction manifest 数量 `320`；
   - oracle labels parquet 存在，大小约 `2.93GB`，oracle status 为 `completed`；
   - merged cache status 为 `completed`，`record_count=116,375,850`，`sample_count=23,275,170`；
   - `/data2` 剩余约 `2.3T`；
   - 未发现已有 `train_timefuse_fusor_streaming.py` 正式进程；
   - 未发现同名输出目录中的训练产物或存活 PID。
5. 资源策略：
   - 本轮正式 fusor 使用 CPU，`CUDA_VISIBLE_DEVICES` 显式置空；
   - 原因：TimeFuse fusor 是 17 维线性权重模型，主要瓶颈在 shard-local SQLite/packed array I/O；同时 GPU2/GPU3 仍有 full-scale visual router eval-only 进程 `1264073`，为避免资源争抢不使用 GPU。
6. 启动正式后台任务：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py --auto-start --allow-existing-output-dir
   ```

## 结果

正式输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/
```

启动信息：

```text
launcher PID/PGID: 1268584 / 1268584
training Python PID: 1268586
device: cpu
CUDA_VISIBLE_DEVICES: ""
batch_size: 256
epochs: 1
prediction_num_workers: 4
prefetch_batches: 1
status_update_interval: 200
```

短时健康检查结果：

```text
2026-06-18 01:13 CST:
PID 1268584 alive, PPID=1, PGID=1268584
Python child PID 1268586 alive, PGID=1268584
status=running
phase=build_prediction_index
shard_name=sample_shard_0002_of_0064
sample_key_count=363675
RSS about 1.2GB for Python child
GPU2/GPU3 memory unchanged at about 693MiB / 12MiB
```

`main.log` 显示：

- `sample_shard_0000_of_0064` oracle SQLite 已完成，记录数 `363,675`；
- `sample_shard_0000_of_0064` prediction SQLite 已完成，记录数 `1,818,375`，即 `363,675 × 5`；
- `sample_shard_0001_of_0064` oracle/prediction SQLite 已完成；
- `sample_shard_0002_of_0064` oracle SQLite 已完成，正在构建 prediction index。

关键接手文件：

```text
/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/preflight_report.json
/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/metadata.json
/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/status.json
/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/main.log
/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/launcher.log
```

监控命令：

```bash
ps -p 1268584,1268586 -o pid,ppid,pgid,stat,etime,%cpu,%mem,rss,cmd
cat /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/status.json
tail -n 120 /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/main.log
find /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/indexes -maxdepth 2 \( -name '*.sqlite' -o -name '*.sqlite.tmp' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 20
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
df -h /data2 /home
```

停止命令：

```bash
bash /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/stop.sh
```

恢复命令：

```bash
bash /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/resume.sh
```

## 结论

Stage 1 `96_48_S` full-scale TimeFuse-style fusor baseline 的正式后台 launcher 已实现并通过 preflight。正式任务已在 `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/` 后台运行，当前处于 shard-local SQLite index 构建阶段，进程、日志、status、资源占用和阶段进展均可监控。训练/eval 继续复用现有 streaming reader 和 `train_timefuse_fusor_streaming.py`，没有引入全量 manifest lookup 或全量 join。

本轮没有删除任何已有 smoke/pressure 结果，也没有启动重复正式 TimeFuse fusor 任务。

## 下一步方案

1. 持续轻量监控 `status.json`、`main.log`、PID/PGID 和 index 文件增长，避免扫描 full merged `manifest.csv`。
2. 等 64 shard index 构建完成后，观察 `scaler_partial_fit`、`train` 和 `test_eval` 阶段的 batch/sample 进展。
3. 若任务中断，先确认 PID/PGID 是否已停止，再使用 `resume.sh`；若 checkpoint 已存在，恢复会走 `--resume-checkpoint`，避免重复已完成 epoch。
4. 完成后检查 `metadata.json`、`summary.md`、`timefuse_fusor_summary.csv`、`timefuse_fusor_raw_soft_fusion_summary.csv`、`timefuse_fusor_selected_model_counts.csv`、`sample_predictions.csv` 和 `checkpoints/latest_timefuse_fusor.pt`。
