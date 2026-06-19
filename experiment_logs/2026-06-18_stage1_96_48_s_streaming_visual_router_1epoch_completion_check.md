# Stage 1 `96_48_S` Streaming Visual Router Full-Scale 1 Epoch 完成检查

日志日期：2026-06-18 00:38:33 CST

## 目的

确认 Stage 1 `96_48_S` full-scale streaming visual router v2 的 `--epochs 1 --train-only` 后台训练是否已经正常结束，并核对 checkpoint、状态文件和关键训练摘要。

## 背景

旧输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/
```

已在 2026-06-16 晚间因旧 manifest lookup 方案内存膨胀而失败，未生成 checkpoint。随后按 `HANDOFF.md` 记录重启 v2：

```text
/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/
```

v2 使用 SQLite 磁盘索引和 batch 查询方式读取 full-scale prediction manifest，限制在物理 GPU2/GPU3 上运行，目标是先完成 1 个训练 epoch 并保存可续训 checkpoint，不在本轮扫描 test 评估。

## 操作

1. 使用 `ps` 检查 `PID 919803` 和 `train_visual_router_online_streaming.py` 相关进程。
2. 检查 v2 输出目录文件、mtime 和大小：
   - `status.json`
   - `main.log`
   - `visual_router_metadata.json`
   - `visual_router_online_metadata.json`
   - `checkpoints/latest_checkpoint_index.json`
   - `checkpoints/latest_96_48_S.pt`
   - `checkpoints/router_96_48_S_epoch_0001.pt`
3. 读取 `status.json` 和 `latest_checkpoint_index.json`，确认完成状态与 checkpoint 指向。
4. 读取 `main.log` 末尾，确认 SQLite index 构建完成和 train-only checkpoint 写出信息。
5. 使用 Quito conda 环境读取 `visual_router_metadata.json` 中的 `config_metadata`，核对训练样本规模、epoch 摘要、label 分布和是否未执行 test 预测。

## 结果

v2 训练进程已经结束，当前没有遗留 `train_visual_router_online_streaming.py` / `PID 919803` 进程。

`status.json` 显示：

```text
status=completed
phase=train_only_done
completed_epochs=1
current_epoch=1
latest_checkpoint_path=/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/latest_96_48_S.pt
updated_at=2026-06-17 20:09:12 CST
```

checkpoint 文件存在且更新时间为 2026-06-17 20:09:08 CST：

```text
checkpoints/latest_checkpoint_index.json
checkpoints/latest_96_48_S.pt
checkpoints/router_96_48_S_epoch_0001.pt
```

`latest_checkpoint_index.json` 指向：

```text
checkpoint_path=/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/router_96_48_S_epoch_0001.pt
latest_checkpoint_path=/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/latest_96_48_S.pt
completed_epochs=1
```

`main.log` 末尾确认：

```text
[manifest_index] sqlite_index_ready ... records=46752600 target_sample_keys=9350520
wrote train-only streaming checkpoint outputs to /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2
latest_checkpoint_path=/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/latest_96_48_S.pt
```

`visual_router_metadata.json` 中 `config_metadata` 显示：

```text
config_name=96_48_S
router_mode=fusion_huber_kl
vali_sample_count=9350520
test_sample_count=13924650
scaler_batches=73625
scaler_samples=9350520
test_predictions=0
embedding_dim=768
epochs_requested_this_run=1
previous_completed_epochs=0
completed_epochs=1
latest_checkpoint_path=/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/latest_96_48_S.pt
```

单 epoch 训练摘要：

```text
epoch=1
loss=0.2646199787870476
huber_loss=0.2595411924736033
kl_loss=0.5078786429804912
```

vali oracle label 分布：

```text
DLinear=2895456
PatchTST=1495198
CrossFormer=912483
ES=3279429
NaiveForecaster=767954
```

输出目录中没有 `visual_router_predictions.csv` 或 `visual_router_summary.csv`，这与本轮 `--train-only` 口径一致；本轮只保存可续训 checkpoint，没有执行 test 评估。

旧目录 `_1epoch/` 的 `status.json` 仍停在 `running/training` 且没有 checkpoint；该目录应视为旧 OOM 失败产物，不应作为完成结果引用。

## 结论

Stage 1 `96_48_S` full-scale streaming visual router v2 的 1 epoch train-only 训练已经正常完成。正式可引用的 checkpoint 是：

```text
/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/latest_96_48_S.pt
```

当前完成状态只代表训练 checkpoint 已写出，不代表 test 评估或 calibration 已完成。

## 下一步方案

1. 若需要评估当前 checkpoint，使用 v2 checkpoint 运行 `--resume-checkpoint ... --epochs 0` 并去掉 `--train-only`，生成 `visual_router_predictions.csv`、`visual_router_summary.csv` 和 metadata。
2. 若需要继续训练，使用 `--resume-checkpoint checkpoints/latest_96_48_S.pt --epochs 1 --train-only` 追加 epoch。
3. 启动下一步长任务前优先使用新的独立输出目录，并保留 v2 目录作为 1 epoch checkpoint 基线。
4. 后续日志和 README 中统一把旧 `_1epoch/` 标记为 OOM 失败产物，把 `_1epoch_v2/` 标记为 1 epoch train-only 完成产物。
