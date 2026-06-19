# Stage 1 GPU2/GPU3 Streaming Visual Router 运行状态检查

日志日期：2026-06-17 19:31:41 CST

## 目的

检查当前物理 GPU2/GPU3 上运行的 Stage 1 `96_48_S` full-scale streaming visual router 训练是否正常，确认进程、GPU 利用率、输出文件更新时间、训练进度和预计剩余时间。

## 背景

`2026-06-17_stage1_96_48_s_streaming_visual_router_oom_fix_review_restart.md` 中记录了 v2 训练重启：使用 SQLite 磁盘索引降低内存压力，并将正式 `--epochs 1 --train-only` 训练限制到物理 GPU2/GPU3。用户要求检查当前训练是否正常运行、完成进度和剩余时间。

## 操作

1. 运行 `nvidia-smi --query-gpu` 和 `nvidia-smi --query-compute-apps` 检查 GPU2/GPU3 的利用率、显存、功耗和计算进程。
2. 运行 `ps -eo pid,ppid,pgid,stat,etime,lstart,cmd | rg -i 'visual|router|train|python|torchrun|accelerate'` 确认视觉路由器训练进程和启动命令。
3. 检查输出目录 `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/` 下的 `status.json`、`main.log`、`online_embedding_latency_summary.csv`、`online_embedding_manifest.csv` 和 SQLite index 文件更新时间。
4. 用 `awk` 统计 `online_embedding_latency_summary.csv` 中 `scaler_fit` 与 `train_epoch_1` 已处理 batch 数和窗口数。
5. 间隔约 60 秒再次采样 `status.json` 与训练窗口数，用实际增长量估算近期吞吐和 ETA。

## 结果

1. 物理 GPU2/GPU3 均由同一个训练进程 PID `919803` 使用：
   - GPU2：显存约 `1303 MiB / 24576 MiB`，GPU util 约 `32%`，功耗约 `140 W`。
   - GPU3：显存约 `981 MiB / 24576 MiB`，GPU util 约 `31%`，功耗约 `143 W`。
2. PID `919803` 启动于 `2026-06-17 08:39:47 CST`，检查时已运行约 `10:49:30`。命令为 `/home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py ... --epochs 1 --train-only --device cuda --vit-data-parallel ...`，输出目录为 `_1epoch_v2`。
3. `status.json` 在检查期间持续更新，状态为：
   - `status=running`
   - `phase=training`
   - `epoch=1`
   - `completed_epochs=0`
   - `embedding_batches=141700`
   - `updated_at=2026-06-17 19:30:49 CST`
4. `online_embedding_manifest.csv` 行数为 `9,350,521`，扣除表头后为 `9,350,520` 个窗口，与当前训练集规模一致。
5. `scaler_fit` 阶段已完成 `9,350,520` 个窗口。
6. `train_epoch_1` 在第一次采样时为 `8,629,054 / 9,350,520` 个窗口；第二次采样为 `8,651,043 / 9,350,520` 个窗口，完成约 `92.52%`，剩余约 `699,477` 个窗口。
7. 两次采样之间训练窗口增加约 `21,989` 个，折合近期实际吞吐约 `300 windows/s`。按该速度估算，剩余训练时间约 `38-40` 分钟，即预计在 `2026-06-17 20:10 CST` 左右完成 1 epoch train-only 主体。该估计不包含训练结束后的 checkpoint 写盘、状态收尾和可能的 I/O 抖动。
8. `main.log` 只记录 SQLite manifest index 构建阶段，已在 `2026-06-17 09:58:47 CST` 停止增长；当前实时训练进度主要由 `status.json` 和 `online_embedding_latency_summary.csv` 反映。

## 结论

当前 GPU2/GPU3 上的视觉路由器训练正常运行，没有看到进程退出、GPU 空转、状态文件停止更新或重新触发 OOM 的迹象。训练已进入 epoch 1 后段，按最近一分钟吞吐估算已完成约 `92.5%`，大约还需要 `40` 分钟完成主体训练。

## 下一步方案

1. 约 `2026-06-17 20:10 CST` 后再次检查 `status.json`、`checkpoints/latest_96_48_S.pt` 和 `latest_checkpoint_index.json`。
2. 若 `status.json` 变为 `completed` 且 checkpoint 存在，再决定是否追加 epoch 或启动 eval-only/calibration。
3. 若 20:20 CST 后仍未完成，优先复查 `online_embedding_latency_summary.csv` 是否继续增长、PID `919803` 是否仍存在，以及 GPU2/GPU3 是否仍有计算功耗。
