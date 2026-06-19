# Stage 1 TimeFuse-style Fusor GPU2/3 公平性修正与正式重启

日志日期：2026-06-18 01:19:45 CST

## 目的

根据公平比较要求，将 Stage 1 `96_48_S` full-scale TimeFuse-style fusor baseline 从 CPU 正式长跑修正为训练时使用 GPU2/GPU3 双卡，并重新启动正式后台任务。

## 背景

上一轮 launcher 已经启动 CPU 版 full-scale fusor，输出目录为：

```text
/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/
```

但用户明确要求：为了公平比较，TimeFuse fusor 至少训练时也要用双卡 GPU 2 和 3。因此 CPU 版不能作为正式 baseline 结果继续推进。由于原 `train_timefuse_fusor_streaming.py --device cuda` 只会使用可见 `cuda:0`，本轮需要先补齐双卡训练封装，再重启 GPU2/3 版本。

## 操作

1. 检查 CPU 版 fusor 进程：
   - launcher PID/PGID：`1268584 / 1268584`
   - Python 子进程：`1268586`
   - 当时阶段：`build_prediction_index`
   - 当时 shard：`sample_shard_0004_of_0064`
2. 执行停止命令：

   ```bash
   bash /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/stop.sh
   ```

3. 停止后确认 `1268584` 和 `1268586` 均已不存在。
4. 更新 CPU 输出目录 `status.json`：
   - `status=stopped_for_gpu_fairness_requirement`
   - `phase=stopped`
   - 说明该目录只保留为中止记录，不作为正式结果。
5. 修改 `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`：
   - 新增 `unwrap_fusor()`，确保 DataParallel 训练后 checkpoint 保存未包裹模型的 state_dict；
   - 新增 `maybe_wrap_data_parallel()`，当 `--device cuda` 且 `torch.cuda.device_count() > 1` 时使用 `nn.DataParallel`；
   - 在 fresh train 和 resume checkpoint 两条路径中都启用同一封装；
   - 在最终 metadata 中记录 `data_parallel`、`data_parallel_device_count` 和 `cuda_visible_devices`。
6. 使用 Quito conda 环境验证语法：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py \
     visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py
   ```

7. 使用 `CUDA_VISIBLE_DEVICES=2,3` 验证 PyTorch 可见双卡：

   ```text
   cuda_available True
   device_count 2
   names ['NVIDIA GeForce RTX 3090', 'NVIDIA GeForce RTX 3090']
   ```

8. 启动 GPU2/3 正式后台任务：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py \
     --auto-start \
     --device cuda \
     --cuda-visible-devices 2,3 \
     --output-dir /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23
   ```

## 结果

CPU 版结果状态：

```text
/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/
status=stopped_for_gpu_fairness_requirement
phase=stopped
不作为正式 baseline 结果引用
```

GPU2/3 正式输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/
```

启动信息：

```text
launcher PID/PGID: 1271090 / 1271090
training Python PID: 1271092
device: cuda
CUDA_VISIBLE_DEVICES: 2,3
torch visible CUDA devices: 2
epochs: 1
batch_size: 256
prediction_num_workers: 4
prefetch_batches: 1
```

短时健康检查：

```text
2026-06-18 01:19 CST:
PID 1271090 alive, PPID=1, PGID=1271090
Python child PID 1271092 alive, PGID=1271090
status=running
phase=build_prediction_index
shard_name=sample_shard_0001_of_0064
cuda_visible_devices=2,3
torch_cuda logical_index 0/1 均为 NVIDIA GeForce RTX 3090
```

说明：当前仍在 shard-local SQLite index 构建阶段，GPU memory 未明显新增是正常现象。`DataParallel` 会在 scaler fit 之后创建 fusor 模型时写入 `main.log`，训练/eval 前向阶段才会实际使用可见双卡。

监控命令：

```bash
ps -p 1271090,1271092 -o pid,ppid,pgid,stat,etime,%cpu,%mem,rss,cmd
cat /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/status.json
tail -n 120 /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/main.log
find /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/indexes -maxdepth 2 \( -name '*.sqlite' -o -name '*.sqlite.tmp' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 20
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

停止命令：

```bash
bash /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/stop.sh
```

恢复命令：

```bash
bash /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/resume.sh
```

## 结论

CPU 版正式 fusor 已因公平性要求停止并保留为中止记录。`train_timefuse_fusor_streaming.py` 已支持在 `CUDA_VISIBLE_DEVICES=2,3` 下使用 PyTorch `DataParallel` 双卡训练，同时保持 checkpoint 可被未包裹模型恢复。新的 GPU2/3 正式后台任务已启动，当前处于 shard-local SQLite index 构建阶段。

## 下一步方案

1. 继续轻量监控 GPU2/3 版输出目录的 PID、`status.json`、`main.log` 和 index 文件增长。
2. 等进入 scaler/model 阶段后，确认 `main.log` 出现 `启用 DataParallel 双卡训练`，并观察 `status.json`/`nvidia-smi` 中 GPU2/GPU3 的实际占用。
3. 完成后只引用 `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/` 的 summary、prediction 和 checkpoint 作为正式 TimeFuse fusor baseline。
