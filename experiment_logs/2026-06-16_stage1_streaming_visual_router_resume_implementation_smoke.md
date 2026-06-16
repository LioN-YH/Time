# Stage 1 Streaming Visual Router 续训实现与 Smoke 验证

日志日期：2026-06-16 15:26:25 CST

## 目的

为 `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py` 补齐 checkpoint/resume 能力，支持 `96_48_S` full-scale streaming visual router 先安全训练 1 个 epoch，后续从 checkpoint 继续追加训练，并用小规模 smoke 验证 fresh、resume 和最终 eval 闭环。

## 背景

前序审计日志 `2026-06-16_stage1_streaming_visual_router_resume_audit.md` 确认 streaming visual router 入口原本没有 `--resume-checkpoint`、checkpoint 保存、scaler/optimizer 恢复或 train-only 模式。正式 full-scale 样本量为 `23,275,170` 个 `sample_key`，每个 epoch 都需要重新生成 vali split 的伪图像和 ViT embedding，因此必须先保证 1 epoch 产物可续训，避免长任务中断后只能从头训练。

## 操作

1. 修改 `train_visual_router_online_streaming.py`：
   - 新增 `--resume-checkpoint PATH`。
   - 新增 `--train-only`，允许只训练并保存 checkpoint，不扫 test。
   - 新增 `output_dir/checkpoints/`，每个 epoch 保存 `router_{config_name}_epoch_{epoch:04d}.pt`，并维护 `latest_{config_name}.pt` 和 `latest_checkpoint_index.json`。
   - checkpoint 保存 `router_state_dict`、`optimizer_state_dict`、`scaler_state`、`completed_epochs`、`config_name`、`model_columns`、训练超参、embedding 口径、stream shard 参数和关键输入路径。
   - resume 时严格校验 `config_name`、`MODEL_COLUMNS`、`router_mode`、`metric`、模型结构超参、训练 loss 超参、ViT/伪图像字段、`stream_shard_index/count`、`labels_path`、`prediction_manifest_path` 和 `config_path`。
   - 将 `--epochs` 明确定义为本次追加训练 epoch 数；resume 时从 `completed_epochs + 1` 开始写全局 epoch。
   - `status.json` 在 init、checkpoint_loaded、training、checkpoint_saved 和 completed 阶段记录 `current_epoch`、`completed_epochs` 和 `latest_checkpoint_path`。
   - fresh run 清理会重写的预测/summary CSV；resume 不删除 `checkpoints/`。

2. 修改 `train_visual_router.py` 的 `load_labels()`：
   - 保持 CSV pilot 输入兼容。
   - 增加 parquet 读取支持，并对 full-scale oracle parquet 使用 `metric` filter 和列选择，避免把 mae/mse 双份全量行同时读入内存。

3. 使用 `quito` 环境执行语法检查：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/train_visual_router.py \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py
   ```

4. 使用 full-scale dry-run 的小 CSV 输入执行三段 smoke，输出目录为：

   ```text
   experiment_logs/run_outputs/2026-06-16_stage1_streaming_resume_smoke/
   ```

   fresh 1 epoch train-only：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py \
     --labels-path experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/window_oracle_labels_with_tsf_cell.csv \
     --prediction-manifest-path experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/manifest.csv \
     --output-dir experiment_logs/run_outputs/2026-06-16_stage1_streaming_resume_smoke \
     --epochs 1 \
     --train-only \
     --max-samples-per-split 2 \
     --embedding-batch-size 2 \
     --batch-size 2 \
     --device auto \
     --local-files-only \
     --period-selection fixed_candidates \
     --dtype auto \
     --status-update-interval 1 \
     --print-rows 2
   ```

   resume 追加 1 epoch train-only：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py \
     --labels-path experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/window_oracle_labels_with_tsf_cell.csv \
     --prediction-manifest-path experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/manifest.csv \
     --output-dir experiment_logs/run_outputs/2026-06-16_stage1_streaming_resume_smoke \
     --resume-checkpoint experiment_logs/run_outputs/2026-06-16_stage1_streaming_resume_smoke/checkpoints/latest_96_48_S.pt \
     --epochs 1 \
     --train-only \
     --max-samples-per-split 2 \
     --embedding-batch-size 2 \
     --batch-size 2 \
     --device auto \
     --local-files-only \
     --period-selection fixed_candidates \
     --dtype auto \
     --status-update-interval 1 \
     --print-rows 2
   ```

   checkpoint eval-only：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py \
     --labels-path experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/window_oracle_labels_with_tsf_cell.csv \
     --prediction-manifest-path experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/manifest.csv \
     --output-dir experiment_logs/run_outputs/2026-06-16_stage1_streaming_resume_smoke \
     --resume-checkpoint experiment_logs/run_outputs/2026-06-16_stage1_streaming_resume_smoke/checkpoints/latest_96_48_S.pt \
     --epochs 0 \
     --max-samples-per-split 2 \
     --embedding-batch-size 2 \
     --batch-size 2 \
     --device auto \
     --local-files-only \
     --period-selection fixed_candidates \
     --dtype auto \
     --status-update-interval 1 \
     --print-rows 2
   ```

5. 读取 smoke checkpoint 和 `status.json` 做字段核验。
6. 对 full-scale oracle parquet 执行 metric filter 读取 smoke，确认 `metric=mae` 后为 `23,275,170` 行。
7. 检查 GPU 空闲情况：4 张 NVIDIA GeForce RTX 3090 均约 `10 MiB / 24576 MiB`、利用率 `0%`。

## 结果

语法检查通过。

三段 smoke 均完成：

- fresh 1 epoch 写出 `checkpoints/router_96_48_S_epoch_0001.pt` 和 `checkpoints/latest_96_48_S.pt`。
- resume 追加 1 epoch 写出 `checkpoints/router_96_48_S_epoch_0002.pt`，`latest_96_48_S.pt` 内部 `completed_epochs=2`。
- `--epochs 0 --resume-checkpoint ...` 成功加载 checkpoint 并生成最终 eval summary。

checkpoint 核验结果：

- `checkpoint_version=stage1_streaming_router_checkpoint_v1`
- `completed_epochs=2`
- `config_name=96_48_S`
- `model_columns=[DLinear, PatchTST, CrossFormer, ES, NaiveForecaster]`
- `router_mode=fusion_huber_kl`
- `metric=mae`
- `hidden_dim=64`
- `dropout=0.0`
- `lr=0.001`
- `weight_decay=0.0001`
- `huber_beta=0.1`
- `kl_tau=0.1`
- `lambda_kl=0.01`
- `stream_shard_index=0`
- `stream_shard_count=1`
- `embedding_metadata` 记录了 `encoder_name=google/vit-base-patch16-224`、`variant=variant_a_3view`、`pooling=cls`、`normalization_preset=hf_vit_0_5`、`image_size=224`、`norm_mode=revin_aux`、`pixel_mode=vision`、`clip=5.0`、`period_selection=fixed_candidates` 和默认候选周期。
- `scaler_state` 包含 `mean_`、`scale_`、`var_`、`n_features_in_`、`n_samples_seen_`。
- `epoch_summaries` 为 2 条，epoch 1 loss 为 `0.297589510679245`，epoch 2 loss 为 `0.19881759583950043`。

最终 `status.json`：

```text
status=completed
phase=done
router_predictions=2
completed_epochs=2
current_epoch=2
latest_checkpoint_path=experiment_logs/run_outputs/2026-06-16_stage1_streaming_resume_smoke/checkpoints/latest_96_48_S.pt
```

最终 eval summary 只用于 smoke，不作为正式性能结论：

- hard top-1：`sample_count=2`，`selected_value=1.502748`，`oracle_value=0.586628`。
- raw soft fusion：`sample_count=2`，`soft_fusion_mae=1.462209`。

full-scale oracle parquet 读取验证：

```text
{'rows': 23275170, 'cols': 15, 'metrics': ['mae'], 'head_config': ['96_48_S']}
```

## 结论

Stage 1 streaming visual router 的续训机制已经实现并通过小规模 smoke 验证。当前入口已经支持：

- fresh 训练每个 epoch 保存 checkpoint；
- 从 checkpoint 恢复 scaler/router/optimizer；
- resume 后按全局 epoch 继续追加训练；
- train-only 跳过 test；
- `--epochs 0 --resume-checkpoint` 做 eval-only；
- `status.json` 记录当前 epoch、累计完成 epoch 和最新 checkpoint。

本轮没有启动正式 full-scale 训练，只给出安全启动建议。

## 下一步方案

建议 full-scale 1 epoch 使用后台方式启动，避免占用当前交互会话。建议输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/
```

建议命令：

```text
OUT=/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch
mkdir -p "$OUT"
nohup /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  /home/shiyuhong/Time/visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py \
  --labels-path /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/window_oracle_labels.parquet \
  --prediction-manifest-path /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache/manifest.csv \
  --output-dir "$OUT" \
  --epochs 1 \
  --train-only \
  --embedding-batch-size 16 \
  --batch-size 32 \
  --device cuda \
  --local-files-only \
  --period-selection fixed_candidates \
  --dtype auto \
  --status-update-interval 100 \
  --print-rows 5 \
  > "$OUT/main.log" 2>&1 &
echo $! > "$OUT/pid.txt"
```

监控命令：

```text
tail -n 80 /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/main.log
cat /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/status.json
cat /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/checkpoints/latest_checkpoint_index.json
```

停止命令：

```text
kill "$(cat /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/pid.txt)"
```

完成 1 epoch 后，优先检查：

1. `status.json` 是否为 `completed`、`phase=train_only_done`、`completed_epochs=1`。
2. `checkpoints/latest_96_48_S.pt` 是否存在。
3. `checkpoints/latest_checkpoint_index.json` 是否指向 epoch 1。
4. 如需追加第 2 个 epoch，使用同一命令加：

   ```text
   --resume-checkpoint /data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/checkpoints/latest_96_48_S.pt
   --epochs 1
   --train-only
   ```

5. 如需只评估当前 checkpoint，使用 `--resume-checkpoint ... --epochs 0` 且去掉 `--train-only`。
