# Stage 1 96_48_S 1k Manifest 与 Launcher 准备

日志日期：2026-06-14 10:18:05 CST

## 目的

继续推进 Stage 1 Visual Router，在不扩展到三 config 的前提下，为 `96_48_S` 中等规模 1k sample_key 闸门实验准备可复核样本清单、prediction cache shard builder、ViT embedding smoke 路径和后台 launcher。

## 背景

此前 120 sample_key 的 `fusion_huber_kl` Visual Router 已完成：

- 旧代表 hard top-1 MAE=`0.982425`；
- 旧代表 raw soft fusion MAE=`1.085451`；
- 旧代表最佳 calibrated soft 为 `top2_fusion_T0p25`，MAE=`0.999014`；
- fixed_candidates embedding 更快，但 120 sample_key 下 hard top-1 MAE=`1.011773`，最佳 calibrated soft MAE=`1.021081`。

当前主要问题是确认小样本结论放大后是否稳定，以及 prediction cache、embedding cache 和 online embedding 口径如何进入中等规模。

## 操作

1. 读取并遵守 `AGENTS.md`，确认实验 Python 使用：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python
   ```

2. 检查 GPU：

   ```text
   nvidia-smi
   ```

   2026-06-14 09:46:52 CST 和 10:14:01 CST 两次检查均显示 4 张 RTX 3090 基本空闲，仅 Xorg 占用少量显存。

3. 审计现有 120 sample_key cache：

   - 实际 cache 目录：`experiment_logs/run_outputs/2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot/`
   - 构成：`2 split * 2 dataset * 3 item * 1 channel * 10 window = 120 sample_key`
   - manifest 行数：`600`
   - oracle labels 行数：`240`

4. 新增正式样本清单脚本：

   ```text
   visual_router_experiments/stage1_vali_test_router/build_stage1_sample_manifest.py
   ```

   运行并生成：

   ```text
   experiment_logs/run_outputs/2026-06-14_095911_486696_visual_router_stage1_sample_manifest_96_48_s_1k/
   ```

5. 新增 prediction cache shard 与合并脚本：

   ```text
   visual_router_experiments/stage1_vali_test_router/build_prediction_cache_from_manifest.py
   visual_router_experiments/stage1_vali_test_router/merge_prediction_cache_shards.py
   visual_router_experiments/stage1_vali_test_router/launch_96_48_s_1k_prediction_cache.py
   ```

6. 修改 `build_vit_embeddings.py`，新增 `--sample-manifest-path`，允许在 prediction cache/oracle labels 完成前直接基于历史窗口 x 生成 embedding smoke。

7. 新增 ViT embedding launcher：

   ```text
   visual_router_experiments/stage1_vali_test_router/launch_96_48_s_1k_vit_embedding.py
   ```

8. 进行最小验证：

   - `py_compile` 覆盖新增和修改脚本；
   - 8 sample_key DLinear CPU prediction cache smoke；
   - 8 sample_key DLinear GPU prediction cache smoke；
   - 8 sample_key 单专家 merge smoke；
   - 8 sample_key ViT embedding GPU smoke，embedding cache 写入 `/data2/syh/Time/cache_shards/`。

9. 更新文档：

   - `visual_router_experiments/stage1_vali_test_router/README.md`
   - `visual_router_experiments/stage1_vali_test_router/stage1_protocol_and_plan.md`
   - `WORKSPACE_STRUCTURE.md`

## 结果

### 1k sample manifest

输出目录：

```text
experiment_logs/run_outputs/2026-06-14_095911_486696_visual_router_stage1_sample_manifest_96_48_s_1k/
```

关键验证：

- `sample_manifest.csv` shape 为 `1000 x 17`；
- `sample_key` 唯一数为 `1000`；
- `vali=500`、`test=500`；
- 每个 split 下 `TEST_DATA_MIN=250`、`TEST_DATA_HOUR=250`；
- 每个 dataset 选 50 个 item，每个 item 选 ch0 的 5 个中心等距 window；
- item 使用 TSF cell 均衡轮转后在 cell 内等距抽样。

候选窗口规模：

| split | dataset | candidate windows |
| --- | --- | ---: |
| vali | TEST_DATA_MIN | 1,820,415 |
| vali | TEST_DATA_HOUR | 7,530,105 |
| test | TEST_DATA_MIN | 12,619,225 |
| test | TEST_DATA_HOUR | 1,305,425 |

### 成本估算

- prediction manifest：`5000` 行；
- 旧 pilot 重复 y_true 小文件口径线性外推目录占用约 `49.17 MiB`；
- 新 shard builder 已实现同一 sample_key 共享 y_true，小数组逻辑体积约 `1.83 MiB`；
- 1k ViT embedding float32 约 `2.93 MiB`，fp16 约 `1.46 MiB`；
- 5k 仅估算：prediction manifest 约 `25,000` 行，旧小文件口径目录占用约 `245.8 MiB`，float32 ViT embedding 约 `14.65 MiB`；本轮不直接跑 5k。

### Prediction cache smoke

8 sample_key CPU DLinear smoke 输出：

```text
experiment_logs/run_outputs/2026-06-14_100000_manual_prediction_cache_builder_smoke/dlinear_cache_subset/
```

验证结果：

- manifest shape 为 `8 x 17`；
- `sample_key + model_name` 无重复；
- 同一 sample_key 的 `y_true_path` 唯一数为 1；
- 第一条样本 `y_true/y_pred` shape 均为 `(48, 1)`；
- 重算 MAE 与 manifest MAE 一致。

8 sample_key GPU DLinear smoke 首次失败：

```text
experiment_logs/run_outputs/2026-06-14_100000_manual_prediction_cache_builder_smoke/dlinear_cache_gpu_smoke/status.json
```

失败原因为模型权重仍在 CPU、batch tensor 在 CUDA。已在 `build_prediction_cache_from_manifest.py` 的 `prepare_model()` 中显式执行：

```text
model = model.to(model.device)
```

修复后 GPU smoke 通过：

```text
experiment_logs/run_outputs/2026-06-14_100000_manual_prediction_cache_builder_smoke/dlinear_cache_gpu_smoke_fixed/
```

### Merge smoke

单专家 merge smoke 通过：

```text
experiment_logs/run_outputs/2026-06-14_100000_manual_prediction_cache_builder_smoke/dlinear_cache_merge_smoke/
```

验证结果：

- manifest shape 为 `8 x 17`；
- `sample_key + model_name` 无重复；
- 每个 sample_key 有 1 个专家；
- 重算第一条 MAE 与 manifest 一致。

### ViT embedding smoke

8 sample_key GPU embedding smoke 输出：

```text
experiment_logs/run_outputs/2026-06-14_100000_manual_prediction_cache_builder_smoke/vit_embedding_sample_manifest_gpu_smoke/
```

embedding cache 写入：

```text
/data2/syh/Time/cache_shards/2026-06-14_100000_manual_vit_embedding_sample_manifest_gpu_smoke/
```

验证结果：

- sample_count=`8`；
- embedding_dim=`768`；
- device=`cuda`；
- 未保存伪图像 tensor。

### 已生成 launcher

Prediction cache launcher：

```text
experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/launcher.sh
```

ViT embedding launcher：

```text
experiment_logs/run_outputs/2026-06-14_101500_visual_router_stage1_vit_embedding_96_48_s_1k_launcher/launcher.sh
```

两者均未自动启动。

## 结论

`96_48_S` 1k 中等规模实验已完成 manifest-only dry-run 和最小链路验证。prediction cache 与 embedding 路线目前可进入后台 smoke，但本轮没有直接启动 1k 长任务，以避免在统计模型耗时和深度模型显存实际峰值尚未完整实测时占用资源。

GPU 使用策略固定为：

- DLinear/PatchTST/CrossFormer 独立进程，分别绑定 GPU 0/1/2；
- ES/SNaive 走 CPU；
- ViT embedding smoke 绑定 GPU 3；
- 不使用 DDP；
- 每个任务独立 `main.log`、`status.json` 和 shard 目录；
- 合并前必须检查五专家完整性、`sample_key + model_name` 唯一性和 y_true 一致性。

embedding 策略：

- 本轮不长期缓存伪图像 tensor；
- 后续路线已调整为 online/运行内 embedding，不再先启动 1k ViT embedding cache；
- 1k ViT embedding launcher 已生成但当前暂不启动，避免长期依赖 embedding npy；
- prediction cache launcher 仍可用于后续 1k，因为 oracle、router 训练、soft fusion 和 calibration 需要五专家 `y_pred/y_true`。

## 下一步方案

1. 如要启动 1k 五专家 prediction cache：

   ```text
   bash experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/launcher.sh
   ```

   查看进度：

   ```text
   tail -f experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/shards/DLinear/main.log
   cat experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/shards/DLinear/status.json
   nvidia-smi
   ```

2. 五个 shard 全部完成后，执行 launcher plan 中的 merge 命令，再运行：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/pilot/compute_window_oracle_from_cache.py \
     --cache-dir <merged_cache_dir>

   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/pilot/enrich_cache_with_tsf_cell.py \
     --cache-dir <merged_cache_dir>

   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py \
     --labels-path <merged_cache_dir>/window_oracle_labels_with_tsf_cell.csv
   ```

3. 训练 1k visual router 前，必须确认 oracle labels 和 prediction manifest 的 `sample_key` 集合完全一致；router 阶段改用 `train_visual_router_online.py` 在线生成运行内 ViT embedding，不读取长期 embedding manifest。

4. 暂不启动以下 1k ViT embedding launcher：

   ```text
   experiment_logs/run_outputs/2026-06-14_101500_visual_router_stage1_vit_embedding_96_48_s_1k_launcher/launcher.sh
   ```

## Handoff

当前没有正在运行的后台 cache/embedding 进程。2026-06-14 10:14:01 CST 的 GPU 检查显示 4 张卡基本空闲。后续若用户确认推进 1k，应优先启动 prediction cache launcher；不要启动 1k ViT embedding launcher，router 阶段改用 online/in-memory embedding。
