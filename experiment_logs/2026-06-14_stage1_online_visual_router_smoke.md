# Stage 1 Online Visual Router Smoke

日志日期：2026-06-14 14:23:47 CST

## 目的

调整 Stage 1 `96_48_S` Visual Router 后续路线：不再先缓存 1k ViT embedding，不启动 1k ViT embedding launcher，不长期保存伪图像 tensor 或 ViT embedding `.npy`；实现 online Visual Router 训练入口，并用已有 120 sample_key prediction cache 完成 smoke。

## 背景

上一轮已生成：

- 1k sample manifest：`experiment_logs/run_outputs/2026-06-14_095911_486696_visual_router_stage1_sample_manifest_96_48_s_1k/sample_manifest.csv`
- 1k prediction cache launcher：`experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/launcher.sh`
- 1k ViT embedding launcher：`experiment_logs/run_outputs/2026-06-14_101500_visual_router_stage1_vit_embedding_96_48_s_1k_launcher/launcher.sh`

本轮路线调整后，1k ViT embedding launcher 暂不启动；prediction cache launcher 仍可保留用于后续 1k，因为 router 训练、oracle、soft fusion 和 calibration 仍需要五专家 `y_pred/y_true`。

已有离线 embedding 代表结果目录：

```text
experiment_logs/run_outputs/2026-06-14_025727_562553_visual_router_stage1_visual_router_smoke/
```

关键指标为：

- hard top-1 MAE=`0.982425`
- raw soft fusion MAE=`1.085451`
- oracle MAE=`0.805392`
- `global_best_single=1.055190`

## 操作

1. 读取并遵守 `AGENTS.md`，并读取上一轮日志、Stage 1 协议和 README。

2. 检查 GPU：

   ```text
   nvidia-smi
   ```

   2026-06-14 14:08:38 CST 显示 4 张 RTX 3090 基本空闲，仅 Xorg 少量显存占用；本轮 online smoke 使用 GPU 3 单卡，不使用 DDP。

3. 修改 `train_visual_router.py`：

   - `load_embedding_matrix()` 新增 `feature_lookup` 参数；
   - `train_router_for_config()` 和 `predict_router_for_config()` 支持运行内 `sample_key -> embedding` 字典；
   - 原离线 `.npy` embedding manifest 路径保持兼容。

4. 新增正式 online 入口：

   ```text
   visual_router_experiments/stage1_vali_test_router/train_visual_router_online.py
   ```

   该脚本在线执行：

   ```text
   Quito 历史窗口 x -> pseudo image -> frozen HF ViT -> CLS embedding -> MLP router
   ```

   并复用：

   - `build_vit_embeddings.py` 的 `make_pseudo_images()`、`pool_vit_outputs()`、dtype/device 解析等逻辑；
   - `train_visual_router.py` 的 `VisualMLPRouter`、`fusion_huber_kl`/`classification` 训练、hard top-1、soft fusion、baseline comparison 和 summary 逻辑。

5. 使用 quito conda 环境做语法检查：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_online.py \
     visual_router_experiments/stage1_vali_test_router/train_visual_router.py
   ```

6. 运行 120 sample_key online smoke：

   ```text
   CUDA_VISIBLE_DEVICES=3 \
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_online.py \
     --device cuda \
     --local-files-only \
     --embedding-batch-size 16 \
     --router-mode fusion_huber_kl \
     --epochs 300 \
     --batch-size 32 \
     --hidden-dim 64 \
     --dropout 0.0 \
     --huber-beta 0.1 \
     --kl-tau 0.1 \
     --lambda-kl 0.01
   ```

   第一次调试 run 输出到：

   ```text
   experiment_logs/run_outputs/2026-06-14_141850_549276_visual_router_stage1_online_visual_router_smoke/
   ```

   该 run 成功输出，但 hard top-1 MAE=`1.135703`，明显弱于离线代表。排查发现 online 入口先构造 ViT，会消耗 PyTorch RNG，导致随后 MLP 初始化和 DataLoader shuffle 与离线训练入口不一致。

7. 在 `train_visual_router_online.py` 中完成 online embedding 后重新 `set_seed(seed)`，再复跑代表 smoke，输出到：

   ```text
   experiment_logs/run_outputs/2026-06-14_142004_461629_visual_router_stage1_online_visual_router_smoke/
   ```

8. 验证代表输出：

   - CSV shape；
   - `sample_key` 覆盖；
   - hard top-1 summary；
   - soft fusion summary；
   - online/offline 对比；
   - latency 和 GPU/CPU 设备；
   - 输出目录是否存在 `.npy`、`embeddings/` 或伪图像 tensor cache。

9. 更新文档：

   - `visual_router_experiments/stage1_vali_test_router/README.md`
   - `visual_router_experiments/stage1_vali_test_router/stage1_protocol_and_plan.md`
   - `WORKSPACE_STRUCTURE.md`

## 结果

### 代码与入口

新增：

```text
visual_router_experiments/stage1_vali_test_router/train_visual_router_online.py
```

修改：

```text
visual_router_experiments/stage1_vali_test_router/train_visual_router.py
```

online 入口不保存伪图像 tensor，不保存 ViT embedding `.npy`。第一版在单次运行内把 120 个 vali/test embedding 暂存在内存字典中，避免 router 每个 epoch 重复 ViT 前向。

### 代表 online smoke 输出

输出目录：

```text
experiment_logs/run_outputs/2026-06-14_142004_461629_visual_router_stage1_online_visual_router_smoke/
```

关键文件：

- `online_embedding_manifest.csv`
- `online_embedding_latency_summary.csv`
- `visual_router_predictions.csv`
- `visual_router_summary.csv`
- `visual_router_soft_fusion_predictions.csv`
- `visual_router_soft_fusion_summary.csv`
- `visual_router_selected_model_counts.csv`
- `visual_router_comparison.csv`
- `online_vs_offline_reference_comparison.csv`
- `visual_router_online_metadata.json`
- `visual_router_online_summary.md`

### Shape 与覆盖

| 文件 | shape | 说明 |
| --- | ---: | --- |
| `online_embedding_manifest.csv` | `120 x 19` | 覆盖 labels 中 120 个 `metric=mae` sample_key |
| `visual_router_predictions.csv` | `60 x 22` | 覆盖 test split 的 60 个 sample_key |
| `visual_router_soft_fusion_predictions.csv` | `60 x 36` | 包含 soft fusion 与五专家数组级指标 |
| `visual_router_summary.csv` | `1 x 10` | hard top-1 汇总 |
| `visual_router_soft_fusion_summary.csv` | `1 x 11` | raw soft fusion 汇总 |
| `visual_router_selected_model_counts.csv` | `5 x 5` | hard top-1 选中专家分布 |

`online_embedding_manifest.csv` 与 labels 的 sample_key 集合完全一致；`visual_router_predictions.csv` 与 test labels 的 sample_key 集合完全一致。

### 指标

Hard top-1：

| method | test MAE | oracle MAE | regret | label accuracy | normalized weight entropy | mean max weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `visual_router_mlp_v2_fusion_huber_kl_online_vit` | 0.982425 | 0.805392 | 0.177033 | 0.466667 | 0.757180 | 0.483784 |

Raw soft fusion：

| method | soft fusion MAE | soft fusion MSE | hard top-1 MAE from array | oracle MAE |
| --- | ---: | ---: | ---: | ---: |
| `visual_router_mlp_v2_fusion_huber_kl_online_vit_soft_fusion` | 1.085451 | 5.311244 | 0.982425 | 0.805392 |

选中专家分布：

| model | count | ratio |
| --- | ---: | ---: |
| DLinear | 26 | 0.433333 |
| PatchTST | 3 | 0.050000 |
| CrossFormer | 12 | 0.200000 |
| ES | 18 | 0.300000 |
| NaiveForecaster | 1 | 0.016667 |

与离线代表对比：

| run | method | MAE | oracle MAE | delta vs online |
| --- | --- | ---: | ---: | ---: |
| online in-memory ViT | hard top-1 | 0.982425 | 0.805392 | 0.000000 |
| offline embedding reference | hard top-1 | 0.982425 | 0.805392 | 0.000000 |
| online in-memory ViT | raw soft fusion | 1.085451 | 0.805392 | 0.000000 |
| offline embedding reference | raw soft fusion | 1.085451 | 0.805392 | 0.000000 |

`visual_router_comparison.csv` 中 `global_best_single=1.055190`，online hard top-1 相对 `global_best_single` 提升约 `6.895970%`；online raw soft fusion MAE=`1.085451`，仍弱于 `global_best_single`。

### Latency 与设备

代表 run metadata：

- device=`cuda`
- embedding device=`cuda`
- forward_dtype=`float16`
- embedding_storage=`in_memory_only`
- pseudo_image_tensor_storage=`not_saved`

平均 latency：

| 阶段 | ms/window |
| --- | ---: |
| imageization | 2.473705 |
| encoder forward | 2.591103 |
| in-memory store | 0.023265 |

2026-06-14 14:20:53 CST 再次检查 `nvidia-smi`，GPU 3 回到仅 Xorg 少量显存占用，无 online router 进程残留。

### 缓存检查

对代表输出目录执行：

```text
find experiment_logs/run_outputs/2026-06-14_142004_461629_visual_router_stage1_online_visual_router_smoke \
  -maxdepth 4 \( -name '*.npy' -o -name '*pseudo*' -o -name '*tensor*' -o -type d -name 'embeddings' \) -print
```

输出为空，确认没有写出 embedding `.npy`、`embeddings/` 目录或伪图像 tensor cache。代表输出目录大小约 `140K`，只包含 CSV/JSON/Markdown。

## 结论

1. `train_visual_router_online.py` 已跑通 120 sample_key online Visual Router smoke，且复现离线 embedding 代表指标。
2. online 入口满足当前约束：输入只来自 Quito 历史窗口 x；ViT embedding 只在运行内暂存；不保存伪图像 tensor；不保存 ViT embedding `.npy`。
3. online 入口复用现有 embedding 与 router 训练评估逻辑，降低后续维护分叉。
4. 1k 后续路线应改为：先完成五专家 prediction cache，再用 online router 训练和评估；不应先跑 1k ViT embedding cache。
5. 由于 120 online smoke 已完成且输出指标与离线代表完全对齐，下一步可以在用户确认后启动 1k prediction cache launcher；当前不启动 1k ViT embedding launcher。

## 下一步方案

1. 若用户确认推进 1k，启动：

   ```text
   bash experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/launcher.sh
   ```

2. 五专家 shard 完成后，先 merge，再生成 oracle labels、TSF cell enrichment 和非视觉 baseline。

3. 使用 `train_visual_router_online.py` 指向 1k 合并后的 labels 与 prediction manifest，在线生成运行内 ViT embedding 并训练 router。

4. 暂不启动：

   ```text
   experiment_logs/run_outputs/2026-06-14_101500_visual_router_stage1_vit_embedding_96_48_s_1k_launcher/launcher.sh
   ```

5. 不扩 5k，不扩到 `576_288_S` 或 `1024_512_S`，直到 `96_48_S` 1k prediction cache 与 online router 路线完成复核。
