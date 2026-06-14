# Stage 1 Fusion Huber KL Visual Router 改造

日志日期：2026-06-14 02:58:34 CST

## 目的

将 Stage 1 Visual Router 从旧版 oracle hard-label 分类器改为更接近 TimeFuse 的权重式 fusion router：

1. router 输出五专家 softmax 权重；
2. 训练时用权重直接融合五专家 `y_pred`，主损失使用 `SmoothL1Loss(fused_pred, y_true)`；
3. 用五专家误差构造 soft oracle distribution `q_i = softmax(-error_i / tau)`，增加 KL 辅助损失；
4. 保留旧版分类 router 作为 baseline；
5. 在已有 120 个 sample_key 上完成 smoke test，并输出同表比较和权重塌缩诊断。

## 背景

上一轮 Stage 1 smoke 已完成 120 个 `96_48_S metric=mae` sample_key 的 ViT embedding 与分类式 Visual Router：

- embedding manifest：`experiment_logs/run_outputs/2026-06-14_010821_165988_visual_router_stage1_vit_embedding_smoke/embedding_manifest.csv`
- 分类 router 输出：`experiment_logs/run_outputs/2026-06-14_010907_224073_visual_router_stage1_visual_router_smoke/`
- 旧版分类 hard top-1 MAE 为 `1.013099`；
- 旧版分类 soft fusion MAE 为 `1.022590`；
- `global_best_single` MAE 为 `1.055190`；
- `oracle_top1` MAE 为 `0.805392`。

旧版脚本虽然输出 softmax 权重，但训练目标是 `CrossEntropyLoss(oracle_model)`，更像 hard label 分类器；本次改造希望主训练目标直接对齐五专家加权融合预测误差。

## 操作

1. 阅读并遵守 `AGENTS.md`，确认实验 Python 使用 conda `quito` 环境：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python
   ```

2. 阅读和修改伪图像工具：
   - 文件：`visual_router_experiments/common/pseudo_imageization.py`
   - 新增 `make_default_period_candidates()`，为 GPU-first 路径构造固定候选周期桶；
   - 新增 `parse_period_candidates()`，支持外部候选周期解析；
   - 扩展 `select_fft_periods(..., period_candidates=...)`，在候选周期存在时用向量化 gather/top-k 选择周期，减少动态路径中的逐样本 `.tolist()`/`.item()`；
   - 将 period fold 改为按周期值分桶批量 fold，同一周期下批量 `reshape` 和 `F.interpolate`；
   - 保留动态 FFT top-k 兼容路径，便于后续 ablation。

3. 修改 ViT embedding 脚本：
   - 文件：`visual_router_experiments/stage1_vali_test_router/build_vit_embeddings.py`
   - 新增 `--period-selection fixed_candidates|dynamic_fft_topk`，默认 `fixed_candidates`；
   - 新增 `--period-candidates`，支持逗号分隔固定候选周期；
   - metadata 和 summary 中记录 `period_selection` 与实际候选周期列表。

4. 修改 Visual Router 训练脚本：
   - 文件：`visual_router_experiments/stage1_vali_test_router/train_visual_router.py`
   - 新增 `--router-mode classification|fusion_huber_kl`，默认 `fusion_huber_kl`；
   - `classification` 保留旧版 oracle hard-label `CrossEntropyLoss` 训练；
   - `fusion_huber_kl` 按 sample_key 读取五专家 `y_pred` 与共享 `y_true`，构造 `[N, 5, pred_len, channel]` 训练张量；
   - 主损失为 `SmoothL1Loss(beta=--huber-beta)`；
   - KL 辅助目标为 `q_i = softmax(-error_i / --kl-tau)`，损失为 `KL(q || p_router)`，权重为 `--lambda-kl`；
   - 输出 `weight_entropy`、`normalized_weight_entropy`、`max_weight`；
   - 新增 `visual_router_selected_model_counts.csv`，记录 hard top-1 选中专家分布；
   - `visual_router_comparison.csv` 同表包含 `oracle_top1`、`global_best_single`、TimeFuse 结构特征 router、visual hard top-1 和 visual soft fusion。

5. 使用 quito 环境完成语法和函数级验证：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/common/pseudo_imageization.py \
     visual_router_experiments/stage1_vali_test_router/build_vit_embeddings.py \
     visual_router_experiments/stage1_vali_test_router/train_visual_router.py
   ```

   另做伪图像张量 smoke，使用 4 个随机 `[96, 1]` 历史窗口和固定候选周期，确认输出形状为 `(4, 3, 32, 32)`，finite 为 `True`，值域约为 `[0.0, 0.999963]`。

6. 运行默认 fusion smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/train_visual_router.py \
     --embedding-manifest-path experiment_logs/run_outputs/2026-06-14_010821_165988_visual_router_stage1_vit_embedding_smoke/embedding_manifest.csv \
     --router-mode fusion_huber_kl \
     --epochs 300 \
     --batch-size 32 \
     --hidden-dim 64 \
     --huber-beta 0.1 \
     --kl-tau 0.5 \
     --lambda-kl 0.1 \
     --print-rows 5
   ```

   输出目录：

   ```text
   experiment_logs/run_outputs/2026-06-14_025441_431557_visual_router_stage1_visual_router_smoke/
   ```

   该组默认超参 hard top-1 MAE 为 `1.055653`，soft fusion MAE 为 `1.192512`，权重熵接近均匀，说明 `tau=0.5`、`lambda_kl=0.1` 对当前小样本过平滑。

7. 运行小规模超参诊断：
   - `lambda_kl=0.0, tau=0.5, dropout=0.1`
   - `lambda_kl=0.01, tau=0.2, dropout=0.1`
   - `lambda_kl=0.01, tau=0.1, dropout=0.0`
   - `lambda_kl=0.1, tau=0.1, dropout=0.0`

   其中 `lambda_kl=0.01, tau=0.1, dropout=0.0` 表现最好。

8. 运行代表性 fusion smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/train_visual_router.py \
     --embedding-manifest-path experiment_logs/run_outputs/2026-06-14_010821_165988_visual_router_stage1_vit_embedding_smoke/embedding_manifest.csv \
     --router-mode fusion_huber_kl \
     --epochs 300 \
     --batch-size 32 \
     --hidden-dim 64 \
     --dropout 0.0 \
     --huber-beta 0.1 \
     --kl-tau 0.1 \
     --lambda-kl 0.01 \
     --print-rows 5
   ```

   输出目录：

   ```text
   experiment_logs/run_outputs/2026-06-14_025727_562553_visual_router_stage1_visual_router_smoke/
   ```

9. 复验旧版分类 baseline 模式：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/train_visual_router.py \
     --embedding-manifest-path experiment_logs/run_outputs/2026-06-14_010821_165988_visual_router_stage1_vit_embedding_smoke/embedding_manifest.csv \
     --router-mode classification \
     --epochs 300 \
     --batch-size 32 \
     --hidden-dim 64 \
     --output-dir experiment_logs/run_outputs/2026-06-14_manual_classification_verify_visual_router_stage1_visual_router_smoke \
     --print-rows 0
   ```

   复验得到 hard top-1 MAE `1.013099`、soft fusion MAE `1.022590`，与上一轮 smoke 一致，说明 classification baseline 路径未被破坏。

10. 更新文档：
    - `visual_router_experiments/stage1_vali_test_router/README.md`
    - `visual_router_experiments/stage1_vali_test_router/stage1_protocol_and_plan.md`
    - `WORKSPACE_STRUCTURE.md`
    - `experiment_logs/README.md`

## 结果

### 代表性 Fusion Router 输出

输出目录：

```text
experiment_logs/run_outputs/2026-06-14_025727_562553_visual_router_stage1_visual_router_smoke/
```

关键输出文件：

- `visual_router_predictions.csv`
- `visual_router_summary.csv`
- `visual_router_soft_fusion_predictions.csv`
- `visual_router_soft_fusion_summary.csv`
- `visual_router_selected_model_counts.csv`
- `visual_router_comparison.csv`
- `visual_router_metadata.json`
- `visual_router_summary.md`

CSV 形状校验：

| 文件 | shape |
| --- | --- |
| `visual_router_predictions.csv` | `(60, 22)` |
| `visual_router_summary.csv` | `(1, 10)` |
| `visual_router_soft_fusion_predictions.csv` | `(60, 36)` |
| `visual_router_soft_fusion_summary.csv` | `(1, 11)` |
| `visual_router_selected_model_counts.csv` | `(5, 5)` |
| `visual_router_comparison.csv` | `(12, 12)` |

### 主要指标

| 方法 | test MAE | oracle MAE | regret | label accuracy | normalized weight entropy | mean max weight |
| --- | --- | --- | --- | --- | --- | --- |
| `oracle_top1` | 0.805392 | 0.805392 | 0.000000 | 1.000000 | NA | NA |
| `visual_router_mlp_v2_fusion_huber_kl` hard top-1 | 0.982425 | 0.805392 | 0.177033 | 0.466667 | 0.757180 | 0.483784 |
| `global_best_single` | 1.055190 | 0.805392 | 0.249798 | 0.050000 | NA | NA |
| `timefuse_single_variable_logistic_regression` | 1.079743 | 0.805392 | 0.274351 | 0.466667 | NA | NA |
| `visual_router_mlp_v2_fusion_huber_kl_soft_fusion` | 1.085451 | 0.805392 | 0.280059 | NA | 0.757180 | 0.483784 |

相对 `global_best_single`：

- fusion hard top-1 相对提升 `6.895970%`；
- fusion soft fusion 相对下降 `2.867833%`；
- TimeFuse 结构特征 router 相对下降 `2.326899%`；
- `oracle_top1` 相对提升 `23.673284%`。

hard top-1 选中专家分布：

| 专家 | count | ratio |
| --- | --- | --- |
| DLinear | 26 | 0.433333 |
| PatchTST | 3 | 0.050000 |
| CrossFormer | 12 | 0.200000 |
| ES | 18 | 0.300000 |
| NaiveForecaster | 1 | 0.016667 |

### 验证结果

以下验证通过：

- quito 环境 `py_compile`；
- 伪图像固定候选周期张量 smoke；
- 代表性 `fusion_huber_kl` 120 sample_key router smoke；
- 旧版 `classification` baseline 模式复验；
- 输出 CSV shape 和同表比较读取校验。

## 结论

1. Stage 1 Visual Router 已支持权重式 fusion 训练目标；`fusion_huber_kl` 不再用 oracle hard label 作为唯一训练目标，而是直接用 router 权重融合五专家预测并反传 SmoothL1 主损失。
2. KL 辅助损失已接入，支持 `--kl-tau` 和 `--lambda-kl`；在当前小样本上，较小的 `tau=0.1` 和 `lambda_kl=0.01` 比默认平滑设置更合理。
3. 代表性 fusion hard top-1 MAE 为 `0.982425`，优于旧版分类 hard top-1 `1.013099`、`global_best_single=1.055190` 和 TimeFuse 结构特征 router `1.079743`。
4. soft fusion MAE 为 `1.085451`，仍弱于 hard top-1 和 `global_best_single`；当前权重归一化熵 `0.757180`、平均最大权重 `0.483784`，说明权重还偏平滑，直接全专家加权融合会混入较差专家。
5. 伪图像化路径已向 GPU-first 方向推进，固定候选周期和分桶 fold 减少了动态 top-k 路径中的逐样本同步点；后续重新生成 embedding 时可用新默认口径。
6. 当前结果仍只是 `96_48_S`、120 sample_key 的 smoke，不应作为三 config 正式结论。

## 下一步方案

1. 针对 soft fusion 做校准实验：温度缩放、top-k 权重重归一化、熵正则、稀疏化或 confidence-aware fusion。
2. 用新 `fixed_candidates` 周期口径重新生成一轮小规模 ViT embedding，对比旧动态周期 embedding 的 latency 与 router 指标。
3. 扩大 `96_48_S` 样本规模，检查 fusion hard top-1 的收益是否稳定，而不是小样本偶然结果。
4. 将 `fusion_huber_kl` 的超参搜索纳入正式 smoke 配置记录，避免单点超参被误读为最终方案。
5. 在 `96_48_S` 链路稳定后，再扩展到 `576_288_S` 和 `1024_512_S`，继续保持 per-config router 口径。
