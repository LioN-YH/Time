# Stage 1 `96_48_S` Full-Scale Handoff

## 最新状态（2026-06-19 11:55:23 CST）

### 当前结论：visual router eval-only 已完成；TimeFuse-style fusor GPU2/3 已完成 reader/scaler 优化并进入 train

用户指出 TimeFuse fusor index/scaler 阶段持续超过 24 小时不合理。复核确认旧路径确实有问题：旧 scaler 复用完整 reader，误读 oracle/prediction arrays；旧 train reader 的 split 过滤发生在 prediction arrays 读取之后；packed `.npy` 在 batch 内按 sample 重复 `np.load`。当前已实现 shard-local index 复用、feature-only scaler、reader split 下推、batch-level grouped packed npy 读取，以及大块 CSV 读取后再切 `batch_size`。正式 GPU2/3 后台任务已恢复并进入训练。

| 项目 | 值 |
| --- | --- |
| fusor 正式输出目录 | `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/` |
| launcher 脚本 | `/home/shiyuhong/Time/visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py` |
| 训练入口 | `/home/shiyuhong/Time/visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py` |
| reader | `/home/shiyuhong/Time/visual_router_experiments/stage1_vali_test_router/stage1_timefuse_fusor_streaming_reader.py` |
| 当前 PID/PGID | `1845436 / 1845436` |
| 当前 Python 子进程 | `1845438` |
| 设备策略 | `--device cuda`，`CUDA_VISIBLE_DEVICES=2,3` |
| 双卡口径 | `main.log` 已写 `启用 DataParallel 双卡训练 logical_devices=[0, 1]` |
| SQLite index | 64/64 oracle index 已复用，64/64 prediction index 已复用 |
| scaler | `feature-only`，约 `11:50:32` 到 `11:51:49 CST` 完成 64 shard，`vali_samples=9,350,520` |
| 当前状态 | `status=running`，`phase=train`，`epoch=1`，`current_shard=sample_shard_0002_of_0064`，`train_batches=1200`，`train_samples=307,052`，`latest_loss=0.11619359254837036` |
| GPU 占用 | GPU2/GPU3 均有 PyTorch 显存占用，约 `441 MiB / 441 MiB` |
| checkpoint/summary | 截至本记录尚未写出，等待训练 epoch 完成后检查 |

推荐轻量监控命令：

```bash
ps -p 1845436,1845438 -o pid,ppid,pgid,stat,etime,%cpu,%mem,rss,cmd
cat /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/status.json
tail -n 120 /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/main.log
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
find /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23 -maxdepth 3 \( -name 'latest_timefuse_fusor.pt' -o -name 'summary.md' -o -name 'timefuse_fusor_summary.csv' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort
```

停止命令：

```bash
bash /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/stop.sh
```

恢复命令：

```bash
bash /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/resume.sh
```

禁止监控方式：

```text
不要对 /data2/.../prediction_cache_full_scale_launcher/merged_cache/manifest.csv 执行 wc -l、head、tail 或全表扫描。
不要删除已有 smoke/pressure 结果，也不要另启重复 full-scale fusor 任务。
不要引用 CPU 停止目录作为正式 TimeFuse fusor baseline。
```

本轮同步更新：

```text
visual_router_experiments/stage1_vali_test_router/stage1_timefuse_fusor_streaming_reader.py
visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py
experiment_logs/2026-06-19_stage1_timefuse_fusor_reader_scaler_optimization_restart.md
experiment_logs/README.md
HANDOFF.md
WORKSPACE_STRUCTURE.md
visual_router_experiments/stage1_vali_test_router/README.md
```

---

## 最新状态（2026-06-19 11:36:04 CST）

### 当前结论：visual router eval-only 已完成；TimeFuse-style fusor GPU2/3 已优化重启，正在 feature-only scaler

用户要求为了公平比较，TimeFuse fusor 至少训练时也要使用 GPU2/GPU3 双卡。因此 CPU 版半程任务已停止并标记为非正式结果。GPU2/3 版旧进程在 `scaler_partial_fit` 阶段被优化性停止，因为旧 scaler 路径复用了完整 reader 并读取 oracle/prediction arrays；同一正式目录已通过 `command_resume.sh` 重启，已复用已有 shard-local SQLite index，并进入 feature-only scaler。visual router eval-only 任务已在 `2026-06-18 17:48:18 CST` 完成。

| 项目 | 值 |
| --- | --- |
| fusor 正式输出目录 | `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/` |
| CPU 停止留痕目录 | `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/`，`status=stopped_for_gpu_fairness_requirement`，不作为正式结果 |
| launcher 脚本 | `/home/shiyuhong/Time/visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py` |
| 训练入口 | `/home/shiyuhong/Time/visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`，已支持 CUDA 多卡 `nn.DataParallel` |
| 当前 PID/PGID | `1840046 / 1840046` |
| 当前 Python 子进程 | `1840048` |
| 旧 GPU2/3 进程 | `1271090 / 1271092`，已写入 `stopped_for_scaler_feature_only_optimization` |
| 启动方式 | `setsid bash command.sh > main.log 2>&1 < /dev/null &` |
| 设备策略 | `--device cuda`，`CUDA_VISIBLE_DEVICES=2,3` |
| 双卡口径 | `CUDA_VISIBLE_DEVICES=2,3` 下 PyTorch 可见 `device_count=2`；进入模型训练后 `DataParallel` 使用两个 logical CUDA device |
| preflight | 通过；64 个 feature shard completed，feature 行数 `23,275,170`，320 个 prediction manifest，oracle/merged cache completed，`/data2` 约 2.3T 可用 |
| 当前状态 | `status=running`，`phase=scaler_partial_fit`，`scaler_mode=feature_only`，`current_shard=sample_shard_0001_of_0064`，`vali_samples=292,204` |
| SQLite index | 64/64 oracle index 完成，64/64 prediction index 完成 |
| 已有产物 | `metadata.json` 和 64 shard indexes 已存在；checkpoint、summary、predictions 尚未出现 |
| 主日志 | `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/main.log` |
| 状态文件 | `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/status.json` |
| metadata | `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/metadata.json` |
| preflight report | `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/preflight_report.json` |

短时健康检查：

```text
2026-06-19 11:36 CST:
PID 1840046 alive, PPID=1, PGID=1840046
Python child PID 1840048 alive, PGID=1840046
status=running, phase=scaler_partial_fit, scaler_mode=feature_only
current shard=sample_shard_0001_of_0064
vali_samples=292204
64/64 oracle SQLite and 64/64 prediction SQLite reused
visual router eval-only status=completed, router_predictions=13924650
```

说明：当前尚未进入 fusor 模型训练，所以 checkpoint、summary 和 predictions 尚未生成。当前阶段是 feature-only scaler，只读取 feature CSV，不读取 oracle/prediction arrays；后续应进入 train。进入训练阶段后应确认 `main.log` 出现 `启用 DataParallel 双卡训练`。visual router eval-only 已产出 full-scale 结果：hard top-1 MAE=`0.5615367653135453`，raw soft fusion MAE=`0.5174675759559787`，oracle MAE=`0.33862214116809347`。

推荐轻量监控命令：

```bash
ps -p 1840046,1840048 -o pid,ppid,pgid,stat,etime,%cpu,%mem,rss,cmd
cat /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/status.json
tail -n 120 /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/main.log
find /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/indexes -maxdepth 2 \( -name '*.sqlite' -o -name '*.sqlite.tmp' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 20
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
df -h /data2 /home
```

停止命令：

```bash
bash /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/stop.sh
```

恢复命令：

```bash
bash /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/resume.sh
```

恢复语义：若 `checkpoints/latest_timefuse_fusor.pt` 已存在，`resume.sh` 会使用 `--resume-checkpoint`，跳过已完成 epoch 并继续 eval；若 checkpoint 尚不存在，则重新构建 shard-local index 并从头训练。

预期完成产物：

```text
metadata.json
status.json  # 预期 status=completed, phase=done
summary.md
timefuse_fusor_predictions.csv
timefuse_fusor_summary.csv
timefuse_fusor_raw_soft_fusion_summary.csv
timefuse_fusor_selected_model_counts.csv
sample_predictions.csv
checkpoints/latest_timefuse_fusor.pt
checkpoints/latest_checkpoint_index.json
indexes/*/oracle_labels_index.sqlite
indexes/*/prediction_manifest_index.sqlite
```

禁止监控方式：

```text
不要对 /data2/.../prediction_cache_full_scale_launcher/merged_cache/manifest.csv 执行 wc -l、head、tail 或全表扫描。
不要删除已有 smoke/pressure 结果，也不要另启重复 full-scale fusor 任务。
不要引用 CPU 停止目录作为正式 TimeFuse fusor baseline。
```

本轮同步更新：

```text
experiment_logs/2026-06-18_stage1_timefuse_fusor_gpu23_fairness_relaunch.md
experiment_logs/2026-06-18_stage1_timefuse_fusor_gpu23_completion_check.md
experiment_logs/2026-06-18_stage1_timefuse_fusor_full_scale_launcher_launch.md
experiment_logs/README.md
visual_router_experiments/stage1_vali_test_router/README.md
WORKSPACE_STRUCTURE.md
```

---

## 最新状态（2026-06-18 01:04:15 CST）

### 当前结论：1 epoch checkpoint eval-only 已后台启动，正在构建 test SQLite index

本轮按任务拆分执行 Calibration 前置步骤：用已完成的 `96_48_S` full-scale 1 epoch checkpoint 跑 eval-only，目标是在 test split 上生成 full-scale router predictions，供后续 soft fusion calibration 使用。

| 项目 | 值 |
| --- | --- |
| eval-only 输出目录 | `/data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/` |
| checkpoint | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/latest_96_48_S.pt` |
| PID/PGID | `1264073 / 1264073` |
| 启动方式 | `setsid bash ... > main.log 2>&1 < /dev/null &` |
| GPU 限制 | `CUDA_VISIBLE_DEVICES=2,3` |
| 命令语义 | `--resume-checkpoint ... --epochs 0`，不追加训练，不使用 `--train-only` |
| 主日志 | `/data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/main.log` |
| 状态文件 | `/data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/status.json` |
| launcher | `/data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/launcher.sh` |

短时健康检查：

```text
2026-06-18 01:03 CST:
PID 1264073 alive, PPID=1, PGID=1264073
RSS=17,545,916 KiB
status=running, phase=init, completed_epochs=1
main.log:
[manifest_index] chunks=1 rows_seen=1000000 matched_rows=1000000 target_sample_keys=13924650
prediction_manifest_index.sqlite.tmp=2.1G
GPU2=693 MiB, GPU3=12 MiB
```

说明：当前仍在 CPU/I/O 为主的 test prediction manifest SQLite index 构建阶段。进入 `test_predict` 后，GPU2/GPU3 才会明显执行 ViT 前向。`status.json` 在 index 构建期间仍可能停留在 `phase=init`，以 `main.log` 的 `[manifest_index]` 进度和 SQLite tmp 增长辅助判断。

DeepSeek sidecar 已按用户要求启用：

```text
stage1-vr-evalonly-precheck: idle
stage1-vr-evalonly-monitor: idle
```

两个任务均为只读辅助检查/监督。后续尝试 resume 取简报时 wrapper 报旧 profile 兼容错误，但这发生在取报告阶段，不影响 eval-only 主进程。

推荐轻量监控命令：

```bash
ps -p 1264073 -o pid,ppid,pgid,stat,etime,%cpu,%mem,rss,cmd
cat /data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/status.json
tail -n 120 /data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/main.log
stat -c '%y %s %n' /data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/prediction_manifest_index.sqlite.tmp /data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/prediction_manifest_index.sqlite 2>/dev/null
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

禁止监控方式：

```text
不要对 /data2/.../prediction_cache_full_scale_launcher/merged_cache/manifest.csv 执行 wc -l、head、tail 或全表扫描。
```

预期完成产物：

```text
visual_router_predictions.csv
visual_router_summary.csv
visual_router_soft_fusion_predictions.csv
visual_router_soft_fusion_summary.csv
visual_router_selected_model_counts.csv
visual_router_comparison.csv
visual_router_metadata.json
visual_router_online_metadata.json
visual_router_streaming_summary.md
status.json  # 预期 status=completed, phase=done
```

完成后下一步：再启动 soft fusion calibration。注意 `evaluate_soft_fusion_calibration.py` 可能仍需 full-scale streaming/SQLite 适配，不能默认直接全量加载 116M 行 manifest。

停止命令：

```bash
kill -TERM 1264073
```

---

## 最新状态（2026-06-18 00:38:33 CST）

### 当前结论：v2 训练已完成，1 epoch train-only checkpoint 已写出

`96_48_S` full-scale streaming visual router v2 已正常结束，当前没有遗留 `PID 919803` 或 `train_visual_router_online_streaming.py` 进程。

| 项目 | 值 |
| --- | --- |
| 正式完成目录 | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/` |
| 状态 | `status=completed`，`phase=train_only_done` |
| 完成 epoch | `completed_epochs=1` |
| 完成时间 | `2026-06-17 20:09:12 CST` |
| latest checkpoint | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/latest_96_48_S.pt` |
| epoch checkpoint | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/router_96_48_S_epoch_0001.pt` |
| checkpoint index | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/latest_checkpoint_index.json` |

训练摘要：

```text
vali_sample_count=9350520
test_sample_count=13924650
test_predictions=0
epoch=1
loss=0.2646199787870476
huber_loss=0.2595411924736033
kl_loss=0.5078786429804912
```

注意：本轮使用 `--train-only`，因此没有生成 `visual_router_predictions.csv` 或 `visual_router_summary.csv`，也没有完成 test 评估或 calibration。若要评估当前 checkpoint，应使用 `--resume-checkpoint .../checkpoints/latest_96_48_S.pt --epochs 0` 并去掉 `--train-only`，建议写入新的独立输出目录。

旧目录 `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/` 是 OOM 失败产物，`status.json` 仍停在 `running/training` 且没有 checkpoint，不应引用为完成结果。

---

## 最新状态（2026-06-17 09:46:07 CST）

### 当前结论：OOM 修复已穿过旧崩溃点，训练 v2 正在运行

本轮先审查 qoder 的 OOM 修复，再重启 `96_48_S` full-scale streaming visual router `--epochs 1 --train-only`。请后续接手优先阅读本节。

#### 关键审查结论

- qoder 原方案“轻量级路径索引 + 按需读取”方向是对的，但实现不够可靠：
  - full-scale `packed_npy_v1` 必须保留 `array_storage`、`y_true_row_index`、`y_pred_row_index`，只保存路径会误读 packed 大数组或触发 shape/内存错误；
  - 即使只保存路径，约 4675 万条 `(sample_key, model_name)` 仍是千万级 Python dict/string 对象，不足以保证不 OOM。
- 已将 `train_visual_router_online_streaming.py` 改为 SQLite 磁盘索引：
  - `SQLitePredictionIndex` 负责 batch 级查询；
  - `build_lightweight_prediction_index()` 分块扫描 manifest，只把匹配行写入 SQLite，不常驻 Python dict；
  - `load_prediction_tensors_from_lightweight_index()` 每个 embedding batch 查询当前 sample_key 的五专家 record，并用 packed row index 读取单行；
  - soft fusion 分支也改为 batch 查询，避免 eval-only 退回全量 lookup。
- 验证：
  - `py_compile` 通过；
  - 正式 manifest 前 5 行 mini packed 验证通过：`y_pred_shape=(1,5,48,1)`、`y_true_shape=(1,48,1)`，复算 MAE 与 manifest 对齐。

#### 当前运行状态

| 项目 | 值 |
| --- | --- |
| 输出目录 | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/` |
| Python PID/PGID | `919803 / 919803` |
| 启动时间 | 约 2026-06-17 08:39 CST |
| 主日志 | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/main.log` |
| 状态文件 | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/status.json` |
| SQLite 索引 | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/prediction_manifest_index.sqlite.tmp`（构建中） |
| 物理 GPU 限制 | `CUDA_VISIBLE_DEVICES=2,3`，不应占用 GPU0/GPU1 |
| 当前阶段 | manifest -> SQLite index，尚未进入 `scaler_fit` |

#### 已穿过旧 OOM 压力点

上一轮 OOM killer 发生在约 `rows_seen=100M / matched_rows≈40M`。本轮截至 2026-06-17 09:46 CST：

```text
[manifest_index] chunks=25 rows_seen=25000000 matched_rows=9769890 target_sample_keys=9350520
[manifest_index] chunks=50 rows_seen=50000000 matched_rows=19723770 target_sample_keys=9350520
[manifest_index] chunks=75 rows_seen=75000000 matched_rows=29950910 target_sample_keys=9350520
[manifest_index] chunks=100 rows_seen=100000000 matched_rows=40167530 target_sample_keys=9350520
```

同等压力下进程 RSS 仍约 `16-17GB`，SQLite 临时索引约 `14.3GB`，没有复现旧方案 `117GB anon-rss` 的线性内存膨胀。

#### GPU 状态说明

当前仍在 CPU/I/O 为主的 manifest index 阶段，GPU3 未明显使用是正常现象。`--vit-data-parallel` 只有进入 ViT embedding forward（`scaler_fit` / `train_epoch_1`）后才会把 batch 分发到物理 GPU2/GPU3。

最新观测：

```text
GPU0: 17342 MiB, 0% util（既有 python3.10，不是本训练）
GPU1:    12 MiB, 0% util
GPU2:   693 MiB, 0% util（本训练 CUDA 上下文）
GPU3:    12 MiB, 0% util
```

#### 启动命令与修正

旧 v2 launcher 失败过两次，均已保留日志：

- `main_failed_missing_numpy_2026-06-17_0003.log`：错误 conda 路径导致 `ModuleNotFoundError: No module named 'numpy'`。
- `main_failed_bad_config_path_2026-06-17_0837.log`：handoff 中旧 config path 不存在。

当前有效 launcher：

```text
/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/launcher.sh
```

关键点：

```text
PY=/home/shiyuhong/application/miniconda3/envs/quito/bin/python
CUDA_VISIBLE_DEVICES=2,3
--config-path /home/shiyuhong/Time/quito/outputs/default_baseline/dlinear/96_48_S/seed_16/EVALUATE/ver_0/config.yaml
--output-dir /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2
```

#### 轻量监控命令

不要对 52GB `merged_cache/manifest.csv` 执行 `wc -l`、`head`、`tail` 或全表扫描；这些会和主训练抢 I/O。推荐只用：

```bash
ps -p 919803 -o pid,pgid,stat,etime,%cpu,%mem,rss,cmd
tail -n 220 /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/main.log
stat -c '%y %s %n' /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/prediction_manifest_index.sqlite.tmp /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/prediction_manifest_index.sqlite 2>/dev/null
cat /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/status.json
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -h
```

DeepSeek sidecar：

- 首轮 `monitor-stage1-96-48-s` 启动成功，但误扫大 manifest，已中断。
- 当前轻量版 `monitor-stage1-lite` 已启动，要求只做 3 轮轻量采样后停止。

#### ETA

截至 2026-06-17 09:46 CST：

- SQLite index 预计还需约 `10-25` 分钟完成；
- 进入 `scaler_fit + train_epoch_1` 后预计还需 `4-7` 小时；
- train-only checkpoint 保守完成窗口：`2026-06-17 14:00-17:00 CST`。

完成后检查：

```bash
cat /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/status.json
ls -lah /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/
```

预期需要看到 `status=completed` / `phase=train_only_done`，以及 `checkpoints/latest_96_48_S.pt`。

---

## 最新状态（2026-06-16 23:55:07 CST）

### ⚠️ 重要更新：OOM 问题修复与重启计划

**上一轮任务状态**：FAILED (OOM Killed)
- **被杀时间**：2026-06-16 22:24:57 CST
- **原因**：Linux OOM Killer 强制终止进程 PID 82124
- **内存占用**：anon-rss 117 GB（远超系统可用内存）
- **失败阶段**：manifest lookup 阶段（扫描到第 100M 行，匹配约 4000 万条记录时崩溃）
- **根本原因**：预加载全量 `prediction_lookup` dict 导致内存爆炸（~4675 万条记录 × Python 对象开销）

**本轮修复**（2026-06-16 23:50 CST）：
1. ✅ 新增 `build_lightweight_prediction_index()` 函数，只存储文件路径而非完整 record
2. ✅ 新增 `load_prediction_tensors_from_lightweight_index()` 函数，实现按需即时加载
3. ✅ 修改 `train_on_stream_batch()` 支持两种模式（轻量级索引优先，向后兼容旧接口）
4. ✅ 内存占用从 ~117 GB 降至 < 1 GB（预计）
5. ✅ 代码已通过语法检查和编译验证

**预期影响**：
- **训练结果**：完全一致（数据源和计算逻辑不变）
- **训练速度**：略慢 1-5%（I/O 开销），但可接受
- **稳定性**：大幅提升，不再依赖"有足够内存"的假设

---

## 当前任务：Stage 1 `96_48_S` full-scale streaming visual router 单轮 epoch 训练（重启）

目标：先完成 `--epochs 1 --train-only` 并保存可续训 checkpoint。

### 本轮代码与启动前验证

- 已在 `train_visual_router_online_streaming.py` 增加 `--vit-data-parallel`，CUDA 多卡可用时用 `torch.nn.DataParallel` 并行冻结 ViT 前向。
- **已重构 prediction manifest 读取机制**：
  - ❌ 旧方案：预加载全量 `prediction_lookup` dict → OOM (~117 GB)
  - ✅ 新方案：轻量级路径索引 + 按需即时加载 → < 1 GB
- 启动前 smoke 已通过：
  - `py_compile` 通过；
  - 小样本 `--vit-data-parallel --train-only` 通过；
  - full-scale oracle parquet `metric=mae` filter 读取通过。

### 准备重启

| 项目 | 值 |
| --- | --- |
| 输出目录 | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/` |
| 启动方式 | `setsid bash -c ... > main.log 2>&1`，断开终端不会中断 |
| 主日志 | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/main.log` |
| 状态文件 | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/status.json` |
| 代码版本 | `train_visual_router_online_streaming.py` (2026-06-16 23:50 修订) |

核心参数（保持不变）：

```
--epochs 1
--train-only
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

输入路径（保持不变）：

```text
labels:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/window_oracle_labels.parquet

prediction manifest:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache/manifest.csv
```

### 监控命令

```
# 查看进程状态
ps aux | grep train_visual_router_online_streaming | grep -v grep

# 查看实时日志
tail -f /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/main.log

# 查看状态文件
cat /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/status.json | python3 -m json.tool

# 查看内存使用
watch -n 5 'ps aux | grep train_visual_router | grep -v grep | awk "{print $6/1024, $11}"'

# 查看 GPU 使用
nvidia-smi dmon -s u -d 5
```

### 停止命令

```
# 温和停止（保存 checkpoint）
kill -SIGTERM <PID>

# 强制停止（不推荐，可能丢失进度）
kill -SIGKILL <PID>
```

### 接手方式

如果任务中断，可以：
1. 检查 `status.json` 了解最后完成的 epoch
2. 使用 `--resume-checkpoint <path_to_checkpoint>` 从 checkpoint 恢复
3. 重新运行相同的启动命令（会自动检测已有输出）

---

## 历史状态（2026-06-16 16:55:07 CST）- 已失效

当前任务：Stage 1 `96_48_S` full-scale streaming visual router 单轮 epoch 训练，目标是先完成 `--epochs 1 --train-only` 并保存可续训 checkpoint。

### 本轮代码与启动前验证

- 已在 `train_visual_router_online_streaming.py` 增加 `--vit-data-parallel`，CUDA 多卡可用时用 `torch.nn.DataParallel` 并行冻结 ViT 前向。
- 已把 full-scale prediction manifest 读取改为按本次需要的 sample_key 子集分块扫描，避免一次性把 52GB manifest 的 vali/test 全量记录都建成 Python lookup。
- 启动前 smoke 已通过：
  - `py_compile` 通过；
  - 小样本 `--vit-data-parallel --train-only` 通过；
  - full-scale oracle parquet `metric=mae` filter 读取通过。

### 正在运行

| 项目 | 值 |
| --- | --- |
| 输出目录 | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/` |
| 父进程 PID/PGID | `82121 / 82121` |
| Python 子进程 PID/PGID | `82124 / 82121` |
| 启动方式 | `setsid bash -c ... > main.log 2>&1`，断开终端不会中断 |
| 主日志 | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/main.log` |
| 状态文件 | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/status.json` |
| 启动脚本 | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/launcher.sh` |
| PID 文件 | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/pid.txt` |

实际启动命令保存在：

```
/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/command.sh
```

核心参数：

```
--epochs 1
--train-only
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

输入路径：

```text
labels:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/window_oracle_labels.parquet

prediction manifest:
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache/manifest.csv
```

### 当前健康检查

截至 `2026-06-16 16:55 CST`：

- 进程已运行约 `56` 分钟，Python 子进程仍为 running。
- full-scale manifest lookup 已完成并进入 `scaler_fit` 阶段；`status.json` 仍显示 `phase=init` 是脚本当前只在 scaler 完成后更新状态的表现，不代表卡死。
- 4 张 GPU 均已参与 ViT DataParallel 前向：

```text
GPU0: 1287 MiB / 24576 MiB, util 67%
GPU1:  787 MiB / 24576 MiB, util 67%
GPU2:  787 MiB / 24576 MiB, util 63%
GPU3:  787 MiB / 24576 MiB, util 61%
```

- 内存状态：

```text
Mem used: 116Gi / 251Gi
Mem available: 133Gi
Swap used: 175Mi / 8Gi
```

- `scaler_fit` 进度：

```text
online_embedding_latency_summary.csv: 9688 行
online_embedding_manifest.csv: 1,201,000 行左右
vali total sample_key: 9,350,520
当前 scaler_fit 约完成 12.8%
```

- latency 最新批次口径：

```text
embedding_batch_size=128
encoder_forward_per_window_ms≈0.42-0.46
imageization_per_window_ms≈0.02-0.03
phase=scaler_fit
```

### ETA

粗略估计：

- manifest lookup：已完成，耗时约 45 分钟；
- scaler_fit：按当前约 2k windows/s 估计总计约 1.2-1.5 小时；
- train_epoch_1：还需要再次遍历 9,350,520 个 vali windows，并读取五专家 y_pred/y_true 做 fusion loss，预计慢于 scaler_fit；
- 完整 `--epochs 1 --train-only` 从启动到 checkpoint 写出，保守估计总耗时约 `4-7` 小时。

后续应以进入 `train_epoch_1` 后的真实吞吐重新修正 ETA。

### 监控命令

```bash
cd /home/shiyuhong/Time
OUT=/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch

ps -o pid,pgid,stat,etime,%cpu,%mem,rss,cmd -p 82124
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
free -h
du -sh "$OUT"
tail -n 120 "$OUT/main.log"
cat "$OUT/status.json"
wc -l "$OUT/online_embedding_latency_summary.csv" "$OUT/online_embedding_manifest.csv"
tail -n 5 "$OUT/online_embedding_latency_summary.csv"
```

### 成功完成后的检查

训练完成后应看到：

```text
$OUT/status.json:
status=completed
phase=train_only_done
completed_epochs=1
latest_checkpoint_path=$OUT/checkpoints/latest_96_48_S.pt
```

同时检查：

```bash
ls -lh "$OUT/checkpoints/"
cat "$OUT/checkpoints/latest_checkpoint_index.json"
```

### 停止命令

如必须停止：

```bash
OUT=/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch
kill -TERM -- -$(cat "$OUT/pgid.txt")
```

### 后续恢复/追加训练

如果本轮完成 epoch 1 后追加 epoch 2：

```bash
OUT=/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -u \
  /home/shiyuhong/Time/visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py \
  --labels-path /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/window_oracle_labels.parquet \
  --prediction-manifest-path /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache/manifest.csv \
  --output-dir "$OUT" \
  --resume-checkpoint "$OUT/checkpoints/latest_96_48_S.pt" \
  --epochs 1 \
  --train-only \
  --embedding-batch-size 128 \
  --batch-size 64 \
  --device cuda \
  --vit-data-parallel \
  --local-files-only \
  --period-selection fixed_candidates \
  --dtype auto \
  --chunk-read-rows 1000000 \
  --status-update-interval 100 \
  --print-rows 5
```

如只评估 checkpoint：使用同样输入和 `--resume-checkpoint "$OUT/checkpoints/latest_96_48_S.pt" --epochs 0`，并去掉 `--train-only`。

## 最新状态（2026-06-16 02:33:22 CST）

本节记录当前窗口完成的 prediction cache merge 与完整性校验。下方 `2026-06-16 00:52:36 CST` 的 TimeFuse feature cache 状态是另一条并行任务的 handoff 信息，保留供接手时参考。

- 当前任务边界：本轮只做 Stage 1 `96_48_S` full-scale prediction cache merge 与完整性校验；没有启动 oracle labels、streaming visual router 或 calibration。
- 正式 merged cache 已完成：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache/
```

- 最终 merge 日志：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merge_command_retry_compact_2026-06-16_011812.log
```

- `merged_cache/status.json` / `metadata.json` 核心结果：

```text
status: completed
generated_at: 2026-06-16 02:09:58 CST
sample_count: 23,275,170
record_count: 116,375,850
array_storage: packed_npy_v1
merge_strategy: packed_npy_v1_streaming_by_sample_shard
shared_y_true_path: true
```

- 完整性校验已通过：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache_validation/2026-06-16_011835_full_integrity_validation_compact_retry/integrity_summary.json
```

校验摘要：

```text
passed: true
actual_record_count: 116,375,850
actual_sample_key_unique_count: 23,275,170
sample_group_count: 23,275,170
model_counts: each expert = 23,275,170
sample_key_model_uniqueness_violations: 0
expert_completeness_violations: 0
shared_y_true_violations: 0
stable_metadata_violations: 0
array_storage_violations: 0
array_path_violations: 0
```

- 处理过的 merge 失败点：
  - `sample_shard=0014`：ES packed `y_true/y_pred` 文件重复尾部，manifest 只引用前半段；已按 manifest row index 处理。
  - `sample_shard=0054`：ES manifest 对部分 split/dataset 引用重复追加后的后半段 row index；已在 merge 中按各专家 row index 读取内容，并在 merged cache 中重写共享 y_true 与 compact y_pred row index。
- 已修改 `visual_router_experiments/stage1_vali_test_router/merge_prediction_cache_shards.py`，并用 Quito 环境 `py_compile`、0014/0054 隔离回归和 full-scale validation 验证通过。
- 没有删除任何已完成 prediction cache shard；没有进入 oracle、router 或 calibration。

后续接力：如继续 Stage 1 主线，下一步可以在 completed `merged_cache/` 上生成 oracle labels 和 TSF enrichment，再进入 baseline / TimeFuse-style fusor / streaming visual router / calibration。最终 merge/validation 结果以本节路径为准，不要引用早期 failed retry 作为最终结果。

## 最新状态（2026-06-16 00:52:36 CST）

本节是当前接手时的最新状态；下方 `2026-06-15 23:49:51 CST` 及更早内容是 prediction cache merge 历史记录。当前新目标是 **TimeFuse-derived 单变量 feature cache full-scale 预计算**。

- 当前任务边界：只监控和收尾 TimeFuse feature cache 预计算；不要在 feature cache、merged prediction cache 和 oracle labels 齐全前启动 TimeFuse fusor 训练。
- 已新增正式脚本：
  - `visual_router_experiments/stage1_vali_test_router/build_timefuse_feature_cache_from_manifest.py`
  - `visual_router_experiments/stage1_vali_test_router/launch_timefuse_feature_cache_full_scale.py`
- 语法检查已通过；64 行 builder smoke、1024 行 builder smoke 和 2-shard launcher smoke 均通过。
- 特征口径：17 维 TimeFuse-derived 单变量元特征，只使用历史窗口 `x`；不读取未来 `y`、专家预测、prediction cache manifest、oracle label 或 TSF label 文件。
- GPU 策略：未使用 GPU。ADF、ACF、AutoReg、periodogram 等小窗口统计是 `numpy/scipy/statsmodels` CPU 计算，GPU 没有实际收益。

### 正在运行

| 项目 | 值 |
| --- | --- |
| 任务 | full-scale TimeFuse feature cache 预计算 |
| 输出目录 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/` |
| launcher | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/launcher.sh` |
| 根状态 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/status.json` |
| lane 日志 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/logs/lane_*.log` |
| shard 输出 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/shards/sample_shard_XXXX_of_0064/` |
| 启动命令 | 见本节下方“恢复/重启命令” |

当前 lane PID/PGID：

```text
lane_00: 720659 / 720659
lane_01: 720661 / 720661
lane_02: 720662 / 720662
lane_03: 720664 / 720664
lane_04: 720665 / 720665
lane_05: 720666 / 720666
lane_06: 720667 / 720667
lane_07: 720668 / 720668
```

当前首批 builder PID：

```text
sample_shard_0000_of_0064: PID 720690
sample_shard_0001_of_0064: PID 720686
sample_shard_0002_of_0064: PID 720692
sample_shard_0003_of_0064: PID 720687
sample_shard_0004_of_0064: PID 720693
sample_shard_0005_of_0064: PID 720691
sample_shard_0006_of_0064: PID 720688
sample_shard_0007_of_0064: PID 720694
```

截至 `2026-06-16 00:52:36 CST`：

```text
status_files=8
running=8
rows_written_or_completed=363,489
sum_builder_rows_per_second≈1,179.24
eta_hours_at_current_sum_rps≈5.40
当前只看到首批 8 个 shard status；后续 lane 会顺序启动 shard 0008、0016、... 等分配到各 lane 的任务。
```

### 监控命令

```bash
cd /home/shiyuhong/Time
ROOT=/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher

/home/shiyuhong/application/miniconda3/envs/quito/bin/python - <<'PY'
import collections, json, pathlib, statistics
root = pathlib.Path('/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/shards')
files = sorted(root.glob('sample_shard_*/status.json'))
c = collections.Counter()
rows = 0
rps = []
for p in files:
    s = json.loads(p.read_text())
    status = s.get('status', 'unknown')
    c[status] += 1
    done = int(s.get('rows_written', s.get('sample_count', 0)) or 0)
    if status == 'completed':
        done = int(s.get('sample_count', done) or done)
    rows += done
    if float(s.get('rows_per_second', 0) or 0) > 0:
        rps.append(float(s['rows_per_second']))
print({'status_files': len(files), **dict(c), 'rows_done_or_written': rows, 'sum_rows_per_second': sum(rps)})
PY

tail -n 80 "$ROOT/logs/lane_00.log"
tail -n 80 "$ROOT/shards/sample_shard_0000_of_0064/main.log"
du -sh "$ROOT"
```

### 停止命令

```bash
ROOT=/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher
for p in "$ROOT"/pids/*.pid; do
  kill -TERM -- -$(cat "$p") 2>/dev/null || kill -TERM $(cat "$p") 2>/dev/null || true
done
```

### 恢复/重启命令

```bash
cd /home/shiyuhong/Time
bash /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/launcher.sh
```

恢复时会跳过 `status=completed` 的 shard；单 shard 中断时 builder 的 `--resume` 会保留完整 item 组并续跑。

### 后续验证口径

feature cache 64 个 shard 全部完成后，至少检查：

- 64 个 `feature_cache.csv` 行数总和为 `23,275,170`；
- 全局 `sample_key` 唯一；
- 每行 `feature_dim=17`，`feature_version=timefuse_single_variable_meta_v1`；
- 17 个特征列全部有限；
- `sample_key` 与 `config_name/split/dataset_name/item_id/channel_id/window_index` 一致；
- 每个 shard 的 `metadata.json` / `status.json` 均为 `completed`。

正式 TimeFuse fusor 训练仍需等待 feature cache、merged prediction cache 和 oracle labels 全部齐全。

## 最新状态（2026-06-15 23:49:51 CST）

本节是当前接手时的最新状态；下方 `2026-06-15 22:25:26 CST` 及更早内容是历史监控记录。

- 当前任务边界：只做 full-scale prediction cache merge 与完整性校验；不要启动 oracle labels、streaming visual router 或 calibration。
- 五专家 prediction cache 已完成：`status_files=320`、`completed=320`、`running=0`、`failed=0`，五个专家各 `64/64` 个 sample shard。
- 资源启动前检查：`/data2` 约 `2.5T` 可用、inode 充足；CPU/I/O 空闲；4 张 RTX 3090 基本空闲。本轮 merge 是 CPU/I/O-bound，没有改成 GPU 计算。
- `prediction_cache_full_scale_launcher/status.json` 中的 `merge_command` 已读取并用于正式启动；该字段为 list，`argc=325`，输入 `320` 个 shard，输出目录为 `prediction_cache_full_scale_launcher/merged_cache`。
- merge 前只读预检：五专家各 `23,275,170` 行，总 manifest 行数 `116,375,850`，预期 sample_key 数 `23,275,170`；首行 `array_storage` 检查全部为 `packed_npy_v1`；缺失/坏 shard 数均为 `0`。
- 已修改正式脚本 `visual_router_experiments/stage1_vali_test_router/merge_prediction_cache_shards.py`：新增 `packed_npy_v1_streaming_by_sample_shard` 分支，避免旧 packed 分支对 full-scale 逐行反复 `np.load`。执行入口仍是 status.json 的同一条 `merge_command`。
- 语法检查已通过；历史 dry-run 的 2 个 sample shard 回归通过，结果为 `20` 行、`4` 个 sample_key、五专家完整、共享 y_true 一致、`array_storage=['packed_npy_v1']`。

### 正在运行

| 项目 | 值 |
| --- | --- |
| 进程 | `merge_prediction_cache_shards.py` |
| PID / PGID | `675597` / `675597` |
| 启动时间 | `2026-06-15 23:47:23 CST` |
| 命令来源 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/status.json` 的 `merge_command` 字段 |
| 主日志 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merge_command_2026-06-15_234723.log` |
| 运行状态文件 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merge_command_run_status.json` |
| 输出目录 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache/` |
| 停止命令 | `kill -TERM -675597` |

截至 `2026-06-15 23:56:28 CST`：

```text
merged_cache/status.json: running
最新进度日志：[2026-06-15 23:56:04 CST] merged sample_shard=0008 progress=9/64 records_written=16365375
merged_cache 当前占用：约 12G
按最近 shard 平均约 48 秒/个估算，预计 2026-06-16 00:40-00:45 CST 完成 merge 主体和收尾写 metadata/summary。
```

### 接手监控命令

```bash
cd /home/shiyuhong/Time
ROOT=/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale
LAUNCHER=$ROOT/prediction_cache_full_scale_launcher

ps -o pid,pgid,etime,%cpu,%mem,rss,vsz,stat,comm -p 675597
tail -n 80 "$LAUNCHER/merge_command_2026-06-15_234723.log"
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m json.tool "$LAUNCHER/merged_cache/status.json"
du -sh "$LAUNCHER/merged_cache"
```

### 后续验证口径

merge 完成后，必须只在 `merged_cache/` 上做完整性校验，并保存可复核输出。至少检查：

- `manifest.csv` 行数应为 `116,375,850`；
- `sample_key` 唯一数应为 `23,275,170`；
- `sample_key + model_name` 唯一；
- 每个 `sample_key` 的专家数分布应为 `{5: 23,275,170}`；
- 同一 `sample_key` 的 `y_true_path + y_true_row_index` 必须唯一；
- `array_storage` 只能是 `packed_npy_v1`；
- 五专家覆盖计数各为 `23,275,170`；
- `metadata.json` / `status.json` 应为 `completed`，且 `merge_strategy` 应为 `packed_npy_v1_streaming_by_sample_shard`。

如果 merge 失败，先读 `merged_cache/status.json` 和主日志定位失败 sample shard 或阶段；不要删除任何已完成 prediction cache shard。

## 最新状态（2026-06-15 22:25:26 CST）

本节为当前接手时应优先采用的最新状态；下方 03:21 前后的内容仅作为历史追溯。

- 正式输出根目录：`/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/`
- 有效 launcher 会话：`tmux` 会话 `stage1_fullscale_launcher_shell`
- 历史 ES backfill 会话：`stage1_es_backfill_0016_0063`，已结束
- 历史 ES accelerator 会话：`stage1_es_accelerator_0010_0015_0048_0063`，已结束
- 最新完成状态：`status_files=320`、`completed=320`、`running=0`、`failed=0`
- `DLinear`、`PatchTST`、`CrossFormer`、`NaiveForecaster` 均已完成全部 `64/64` 个 sample shard
- `ES` 已完成全部 `64/64` 个 sample shard，没有 running 或 failed shard
- 五专家 prediction cache 已满足 `completed=320 && failed=0`，现在可以进入 merge

最新运行进程：

| 专家 | worker PID | 当前 Python PID | 当前 shard | 设备 |
| --- | ---: | ---: | --- | --- |
| ES 原 worker | `stage1_fullscale_launcher_shell` | 573353 | completed | CPU |
| ES backfill lane0 | tmux `stage1_es_backfill_0016_0063` | 573855 | completed | CPU |
| ES backfill lane1 | tmux `stage1_es_backfill_0016_0063` | 574010 | completed | CPU |
| ES backfill lane2 | tmux `stage1_es_backfill_0016_0063` | 573543 | completed | CPU |
| ES backfill lane3 | tmux `stage1_es_backfill_0016_0063` | 573698 | completed | CPU |
| ES accelerator lane0 | tmux `stage1_es_accelerator_0010_0015_0048_0063` | 589498 | completed | CPU |
| ES accelerator lane1 | tmux `stage1_es_accelerator_0010_0015_0048_0063` | 589513 | completed | CPU |
| ES accelerator lane2 | tmux `stage1_es_accelerator_0010_0015_0048_0063` | 589520 | completed | CPU |
| ES accelerator lane3 | tmux `stage1_es_accelerator_0010_0015_0048_0063` | 589523 | completed | CPU |

最新严格只读 audit：

- audit 快照时间：`2026-06-15 03:35:58 CST`
- audit 输出目录：`/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/audits/2026-06-15_033558_completed_manifest_audit/`
- audit 快照状态：`status_files=33`、`completed=28`、`running=5`、`failed=0`
- 严格覆盖：28 个 completed shard，即四个非 ES 专家的 shard `0000` 到 `0006`
- 读取并校验 manifest 行数：`10,182,900`
- 按 sample shard 合并覆盖检查：`2,545,725` 个 sample_key-shard 组合
- 结论：`global_duplicate_sample_model=0`、`shared_y_true_violations=0`、`stable_metadata_violations=0`、`array_storage_values=['packed_npy_v1']`
- 每个已 audit 的 sample shard 当前均只有 4 个专家结果，原因是 `ES` 尚未完成首个 shard；这不是错误，但仍不能 merge

资源状态：

- 正式输出根目录当前约 `88G`
- `/data2` 约 `2.6T` 可用
- `/home` 仍约 `18G` 可用且接近满盘，不要把正式产物写回 `/home`

下一步建议：

1. 继续只做 `ES` 收尾监控，不要 merge、oracle、router 或 calibration。
2. 等 `completed=320` 且 `failed=0` 后，再单独开启 merge 与完整性校验小目标。
3. 后续拆分为 prediction cache 收尾、merge/校验、oracle/TSF、streaming visual router、calibration/最终汇总五个小目标，避免再次把全流程塞进同一上下文。

ES backfill 说明：

- 目录：`/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/es_parallel_backfill_0016_0063/`
- 第一次直接从 exec 会话启动 `launcher.sh` 没有保活，仅留下 0 字节 `main.log`，未生成有效 `status.json`，不要视为成功运行。
- 有效启动方式：`bash .../es_parallel_backfill_0016_0063/tmux_launcher.sh`
- 监控：

```bash
tmux list-windows -t stage1_es_backfill_0016_0063
BACKFILL=/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/es_parallel_backfill_0016_0063
tail -n 80 "$BACKFILL/logs/lane_0.log"
tail -n 80 "$BACKFILL/logs/lane_1.log"
tail -n 80 "$BACKFILL/logs/lane_2.log"
tail -n 80 "$BACKFILL/logs/lane_3.log"
```

- 停止：

```bash
tmux kill-session -t stage1_es_backfill_0016_0063
```

## 触发信息

- 触发时间：2026-06-15 03:21:21 CST
- 触发原因：当前目标上下文已远超项目 65% 阈值，本轮只做最小轻量状态复核后立即 handoff；当前不再继续推进主要任务。
- 当前窗口/线程：继续推进 Stage 1 `96_48_S` 正式 full-scale 全候选窗口实验的长跑监控窗口。
- 建议继承方式：新窗口或 `/fork` 继续，先读取本文件和正式输出根目录 `HANDOFF.md`。

## 当前目标

- 本轮用户目标：推进 Stage 1 的 `96_48_S` 正式 full-scale 全候选窗口实验，得到第一版 full-scale streaming visual router 结果。
- 当前任务边界：只做视觉主线，路径固定为 `x -> pseudo image -> frozen ViT -> router`；当前阶段仍是五专家 prediction cache 生产和监控。
- 不应继续做的事项：不要重复 1k；不要把 dry-run 或 `launcher_compat_check/` 当正式结果；不要启动 ViT embedding cache；不要在 completed 达到 `320` 前执行 merge；不要删除已完成 shard。

## 已完成步骤

1. 正式 full-scale sample manifest 已完成：
   - 目录：`/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/sample_manifest_full_scale/`
   - `sample_count=23,275,170`
   - sample shards：`64`
   - selection_strategy：`all_candidate_windows`
2. 正式 prediction cache launcher 已生成并在持久 tmux shell 中启动：
   - 目录：`/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/`
   - tmux 会话：`stage1_fullscale_launcher_shell`
   - `DLinear`/`PatchTST`/`CrossFormer` 走 GPU 0/1/2，`ES`/`NaiveForecaster` 走 CPU。
3. 截至 `2026-06-15 03:06:09 CST`，完成 12 个 completed shard 的严格 CSV 抽检：
   - 覆盖 `DLinear`、`PatchTST`、`CrossFormer`、`NaiveForecaster` 的 shard 0000、0001 和 0002；
   - 每个 shard `363,675` 行；
   - `sample_key + model_name` 无重复；
   - `array_storage=packed_npy_v1`；
   - 合并抽检覆盖 `1,091,025` 个 sample_key，共享 `y_true_path + y_true_row_index` 无违规；
   - 每个已抽检 sample_key 当前只有四个专家结果，原因是 ES 尚未完成首 shard。
4. 截至 `2026-06-15 03:10:09 CST`，完成一次轻量状态复核：
   - `completed=16`、`running=5`、`failed=0`；
   - `DLinear`、`PatchTST`、`CrossFormer`、`NaiveForecaster` 已完成到 shard 0003 并运行 shard 0004；
   - `ES` 仍在 shard 0000；
   - 本次未做新增 shard 0003 的严格 CSV 抽检，严格抽检仍截至 12 个 completed shard。
5. 截至 `2026-06-15 03:12:36 CST`，再次轻量确认：
   - `completed=16`、`running=5`、`failed=0` 未变；
   - 四个非 ES 专家仍在 shard 0004，ES 仍在 shard 0000；
   - 本次未新增严格 CSV 抽检，仍不满足 merge 条件。
6. 截至 `2026-06-15 03:19:02 CST`，再次轻量状态更新：
   - `completed=20`、`running=5`、`failed=0`；
   - 四个非 ES 专家已完成到 shard 0004 并运行 shard 0005；
   - ES 仍在 shard 0000；
   - 本次未新增严格 CSV 抽检，严格抽检仍截至 12 个 completed shard。
7. 截至 `2026-06-15 03:21:21 CST`，再次轻量确认：
   - `completed=20`、`running=5`、`failed=0` 未变；
   - 四个非 ES 专家仍在 shard 0005，ES 仍在 shard 0000；
   - 本次未新增严格 CSV 抽检，仍不满足 merge 条件。
8. 已新增并更新日志：
   - `experiment_logs/2026-06-15_stage1_96_48_s_full_scale_prediction_cache_monitoring.md`
   - `experiment_logs/README.md`
   - `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/HANDOFF.md`

## handoff 前最后状态

最后一次轻量状态检查时间：`2026-06-15 03:21:21 CST`

```
status_files=25
completed=20
running=5
failed=0
```

各专家状态：

| 专家 | worker PID | 当前 Python PID | 当前 shard | 状态 |
| --- | ---: | ---: | --- | --- |
| DLinear | 427566 | 451835 | sample_shard_0005_of_0064 | running |
| PatchTST | 427571 | 452615 | sample_shard_0005_of_0064 | running |
| CrossFormer | 427576 | 453105 | sample_shard_0005_of_0064 | running |
| ES | 427581 | 427585 | sample_shard_0000_of_0064 | running |
| NaiveForecaster | 427586 | 451980 | sample_shard_0005_of_0064 | running |

注意：严格 CSV 抽检截至 12 个 completed shard。`completed=20` 是轻量状态复核；新窗口接手后建议先复核最新实时状态，并对新增 completed shard 0003/0004 做严格 CSV 抽检。

## 正在运行的命令

| 命令/进程 | 状态 | PID/会话 | 输出路径 | 处理建议 |
| --- | --- | --- | --- | --- |
| `bash launcher.sh` workers | running | tmux `stage1_fullscale_launcher_shell`；worker PID 见 launcher `pids/` | `prediction_cache_full_scale_launcher/shards/` | 继续监控，不要中止 |
| `build_prediction_cache_from_manifest.py` DLinear | running | Python `451835` | `shards/DLinear/sample_shard_0005_of_0064/` | 等待完成 |
| `build_prediction_cache_from_manifest.py` PatchTST | running | Python `452615` | `shards/PatchTST/sample_shard_0005_of_0064/` | 等待完成 |
| `build_prediction_cache_from_manifest.py` CrossFormer | running | Python `453105` | `shards/CrossFormer/sample_shard_0005_of_0064/` | 等待完成 |
| `build_prediction_cache_from_manifest.py` ES | running | Python `427585` | `shards/ES/sample_shard_0000_of_0064/` | 等待完成；这是当前慢路径 |
| `build_prediction_cache_from_manifest.py` NaiveForecaster | running | Python `451980` | `shards/NaiveForecaster/sample_shard_0005_of_0064/` | 等待完成 |

## 关键路径

| 路径 | 用途 | 备注 |
| --- | --- | --- |
| `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/` | 正式 full-scale 输出根目录 | 后续所有正式结果继续写这里 |
| `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/HANDOFF.md` | 实验运行 handoff | 已更新到 03:21:21 CST |
| `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/sample_manifest_full_scale/sample_manifest_shard_index.csv` | 正式 sample shard index | 64 shards，23,275,170 sample_key |
| `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/status.json` | launcher 元信息和 merge command | completed=320 后执行其中 `merge_command` |
| `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/shards/` | 五专家 shard 输出 | 不要删除 completed shard |
| `experiment_logs/2026-06-15_stage1_96_48_s_full_scale_prediction_cache_monitoring.md` | 本轮监控中文日志 | 已同步 README |

## 失败点和风险

- 尚未出现 failed shard。
- 第一次一次性 tmux 启动方式曾经快速退出，不要把那次短暂尝试视为有效运行；当前有效运行在持久 `stage1_fullscale_launcher_shell`。
- `/home` 几乎满盘，正式输出必须留在 `/data2`。
- ES 明显慢于其他专家，后续完成进度会被 ES 限制。
- ES 明显落后，目前四个非 ES 专家已完成到 shard 0003 并运行 shard 0004，但 ES 仍在 shard 0000；后续 completed 到 320 前不应执行 merge。

## 下一步命令

```
cd /home/shiyuhong/Time
ROOT=/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale
LAUNCHER=$ROOT/prediction_cache_full_scale_launcher

tmux ls
pgrep -af 'build_prediction_cache_from_manifest.py|prediction_cache_full_scale_launcher'
nvidia-smi

python - <<'PY'
import json, pathlib, collections, re
root = pathlib.Path('/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/shards')
status_files = list(root.glob('*/sample_shard_*/status.json'))
by_status = collections.Counter()
latest = {}
for p in status_files:
    expert = p.parts[-3]
    try:
        obj = json.loads(p.read_text())
    except Exception:
        obj = {'status': 'bad_json'}
    st = obj.get('status', 'missing')
    by_status[st] += 1
    m = re.search(r'sample_shard_(\d{4})_of_0064', str(p))
    if m:
        latest.setdefault(expert, []).append((int(m.group(1)), st))
print('status_files', len(status_files))
print('by_status', dict(sorted(by_status.items())))
for expert in sorted(latest):
    comp = sorted(idx for idx, st in latest[expert] if st == 'completed')
    run = sorted(idx for idx, st in latest[expert] if st == 'running')
    fail = sorted(idx for idx, st in latest[expert] if st == 'failed')
    print(expert, 'completed_count', len(comp), 'completed_max', comp[-1] if comp else None, 'running', run, 'failed', fail)
PY
```

当 completed 达到 `320` 后，再执行：

```
/home/shiyuhong/application/miniconda3/envs/quito/bin/python - <<'PY'
import json, subprocess
from pathlib import Path
status = json.loads(Path('/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/status.json').read_text())
cmd = status['merge_command']
print(' '.join(cmd))
subprocess.run(cmd, check=True)
PY
```

## 验证口径

- prediction cache 完整条件：五专家各 64 个 sample shard，`completed=320`，`failed=0`。
- merge 后必须校验：
  - `sample_key + model_name` 唯一；
  - 每个 sample_key 五专家完整；
  - 共享 `y_true_path + y_true_row_index` 一致；
  - `array_storage=packed_npy_v1`。
- 后续正式 streaming router 必须使用 `train_visual_router_online_streaming.py`，1 epoch。
- streaming 输出必须包含 `visual_router_predictions.csv`、`visual_router_summary.csv`、`visual_router_metadata.json`。
- calibration 输出必须包含 summary/comparison。
- streaming 目录最终检查不能有 `.npy`、`embeddings/` 或伪图像 tensor cache。

## 给下个窗口的继续 prompt

继续推进 `/home/shiyuhong/Time` 中 Stage 1 `96_48_S` 正式 full-scale 全候选窗口实验。先阅读 `AGENTS.md`、根目录 `HANDOFF.md` 和 `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/HANDOFF.md`。当前 prediction cache launcher 在 tmux 会话 `stage1_fullscale_launcher_shell` 中运行；截至 2026-06-15 03:10:09 CST，轻量状态为 completed=16、running=5、failed=0，DLinear/PatchTST/CrossFormer/NaiveForecaster 已完成到 shard 0003 并运行 shard 0004，ES 仍在 shard 0000。严格 CSV 抽检已通过 12 个 completed shard：四个非 ES 专家的 shard 0000、0001、0002，每个 363,675 行，`sample_key + model_name` 无重复，`array_storage=packed_npy_v1`，共享 y_true 无违规；新增 shard 0003 尚未严格抽检。请先复核最新 shard 状态，并在合适时对新增 completed shard 做严格 CSV 抽检；继续监控直到 completed=320，再执行 launcher `status.json` 中的 merge_command，并按 sample_key/model_name 唯一、五专家完整、共享 y_true 一致、packed_npy_v1 校验。之后生成 oracle labels、TSF enrichment/baseline，使用 `train_visual_router_online_streaming.py` 跑 1 epoch 正式 streaming visual router，再运行 soft fusion calibration。全程使用 quito Python `/home/shiyuhong/application/miniconda3/envs/quito/bin/python`，输出保留在 `/data2`，不要重复 1k，不要启动 ViT embedding cache，不要删除已完成 shard。每完成独立步骤写中文实验日志并更新 `experiment_logs/README.md` 和必要 handoff。

补充：`2026-06-15 03:21:21 CST` 轻量确认状态未变，仍为 `completed=20`、`running=5`、`failed=0`；四个非 ES 专家仍在 shard 0005，ES 仍在 shard 0000；新增 shard 0003/0004 仍未严格抽检。
