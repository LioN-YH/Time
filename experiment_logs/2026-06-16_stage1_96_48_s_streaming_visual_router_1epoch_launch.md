# Stage 1 `96_48_S` Streaming Visual Router Full-Scale 1 Epoch 启动

日志日期：2026-06-16 16:55:07 CST

## 目的

启动 Stage 1 `96_48_S` full-scale streaming visual router 单轮 epoch 完整训练，使用 `--train-only` 先保存可续训 checkpoint，不在本轮扫描 test 评估。

## 背景

前序日志 `2026-06-16_stage1_streaming_visual_router_resume_implementation_smoke.md` 已完成 checkpoint/resume smoke。用户要求下一步启动 full-scale 1 epoch 长周期训练，并最大化利用可用 GPU。由于 streaming router 的主要耗时来自冻结 ViT embedding 前向，本轮在正式启动前补充并验证了 ViT DataParallel 路径。

## 操作

1. 启动前检查 4 张 NVIDIA GeForce RTX 3090 均空闲。
2. 确认正式输入存在：
   - oracle labels parquet：`/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/window_oracle_labels.parquet`
   - merged prediction manifest：`/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache/manifest.csv`
3. 在 `train_visual_router_online_streaming.py` 增加 `--vit-data-parallel`，CUDA 多卡可用时用 `torch.nn.DataParallel` 并行冻结 ViT 前向；router/scaler/checkpoint 仍保持单进程语义。
4. 为 full-scale train-only 路径增加 prediction manifest 子集读取：只根据本次需要的 vali sample_key 分块扫描 52GB manifest，避免同时索引 vali/test 全量记录。
5. 运行小样本 `--vit-data-parallel --train-only` smoke，确认多卡 ViT 路径和子集 lookup 可运行。
6. 使用 `setsid` 后台启动正式任务，输出目录：

   ```text
   /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/
   ```

7. 启动后持续监控约 56 分钟，覆盖 manifest lookup 完成、进入 scaler_fit、4 卡 GPU 稳定参与 ViT 前向。
8. 将完整 handoff 写入根目录 `HANDOFF.md` 顶部。

## 结果

正式进程：

```text
父进程 PID/PGID: 82121 / 82121
Python 子进程 PID/PGID: 82124 / 82121
启动方式: setsid 后台运行，断开终端不会中断
```

核心参数：

```text
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
```

监控结果：

- manifest lookup 已完成并进入 `scaler_fit`。
- `online_embedding_manifest.csv` 已写约 `1,201,000` 行，约为 vali 总样本 `9,350,520` 的 `12.8%`。
- 4 张 GPU 均参与前向，最近一次采样利用率为 `67% / 67% / 63% / 61%`。
- 内存使用约 `116Gi / 251Gi`，可用约 `133Gi`，swap 使用约 `175Mi`。
- 最新 latency 显示 `embedding_batch_size=128` 时，ViT forward 约 `0.42-0.46 ms/window`，伪图像构造约 `0.02-0.03 ms/window`。

当前 `status.json` 仍显示 `phase=init`，这是因为脚本只在 scaler_fit 完成后写下一次状态；从 latency/manifest 增长和 GPU 利用率看，任务实际已进入 `scaler_fit` 且运行正常。

## 结论

Stage 1 `96_48_S` full-scale streaming visual router 1 epoch 训练已经成功后台启动，并确认进入多卡 ViT streaming 阶段。当前无需保持交互会话；后续断开终端不会中断训练。

## 下一步方案

1. 后续按 `HANDOFF.md` 中的监控命令检查进度。
2. 完成后重点确认：
   - `status.json` 为 `completed`；
   - `phase=train_only_done`；
   - `completed_epochs=1`；
   - `checkpoints/latest_96_48_S.pt` 存在；
   - `checkpoints/latest_checkpoint_index.json` 指向 epoch 1。
3. 若需要追加训练，使用 `--resume-checkpoint checkpoints/latest_96_48_S.pt --epochs 1 --train-only`。
4. 若需要评估当前 checkpoint，使用 `--resume-checkpoint checkpoints/latest_96_48_S.pt --epochs 0` 并去掉 `--train-only`。
