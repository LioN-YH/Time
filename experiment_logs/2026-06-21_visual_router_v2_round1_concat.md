# Visual Router V2 Round 1 P2d visual+aux concat 收束实验

日志日期：2026-06-21 11:52:15 CST

## 目的

完成 Visual Router V2 Round 1 P2d visual embedding 与 RevIN aux 的简单 concat 收束实验，验证六维 `revin_aux` 是否能在 P2b 最优 visual-only 表示基础上提供互补增益，并生成 Round 1 总表与中文结论。

## 背景

P2a 已生成 P0 fixed sample set 对齐的 sharded feature cache，包含 `cls_embedding`、`mean_patch_embedding` 和六维 `revin_aux`。P2b 显示 `visual_mean_patch_only` 是 visual-only 中 selection raw-soft MAE 最优变体；P2probe 显示 `cls_mean_concat` 对 oracle expert suitability 的线性 probe 最强；P2c 显示 `revin_aux` 单独有路由信号但明显弱于 P2b best visual-only。因此本步只验证简单 concat，不做 FiLM、gating、attention、adapter、ViT finetune、pseudo image/view layout 改造或 P2a feature 重建。

## 操作

1. 新增 `visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round1_concat.py`。
   - 只读取 P2a `round1_feature_manifest.csv` 指向的 `.npz` shard。
   - 严格按 P0 `order_index` 和 `sample_key` 校验 `pilot_train`、`pilot_selection`、`diagnostic_balanced` 对齐。
   - 实现 `mean_patch_plus_aux = concat(mean_patch_embedding, revin_aux)`，维度 774。
   - 实现 `cls_mean_concat_plus_aux = concat(cls_embedding, mean_patch_embedding, revin_aux)`，维度 1542。
   - 复用 P2b 的 batch prediction 读取、`fusion_huber_kl` 训练目标、hard top-1/raw-soft 指标与 selected model 统计逻辑。
   - 复用 P2b 已有 P0 subset SQLite index，避免重新扫描 116M prediction manifest。
2. 用 Quito 环境完成语法检查和 128 样本 smoke：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round1_concat.py

   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round1_concat.py \
     --output-dir /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_concat_smoke \
     --seeds 16 --epochs 1 --batch-size 64 --eval-batch-size 64 \
     --device cpu --max-samples-per-set 128 --overwrite
   ```

3. 启动正式后台任务：

   ```bash
   CUDA_VISIBLE_DEVICES=1 /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     /home/shiyuhong/Time-visual-router-v2/visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round1_concat.py \
     --output-dir /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_concat \
     --seeds 16,17,18 --epochs 3 --batch-size 256 --eval-batch-size 512 --device cuda
   ```

   后台记录：
   - PID/PGID：`2807053/2807053`
   - 主日志：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_concat/main.log`
   - 停止脚本：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_concat/stop.sh`

4. 正式任务完成后，基于已生成 prediction CSV 重建一次轻量 summary，使 `round1_concat_stratified_summary.csv` 同时包含 `single_column` 分层和 `tsf_cell` 联合分层，便于直接查看 `oracle_model=CrossFormer/PatchTST`。
5. 将轻量结果复制到仓库内 `experiment_summaries/visual_router_v2_round1/p2d_concat/`，并同步复制 Round 1 总表到 `experiment_summaries/visual_router_v2_round1/` 根目录。

## 结果

正式输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_concat/
```

必须产物均已生成：

- `round1_concat_variant_seed_results.csv`
- `round1_concat_selection_comparison.csv`
- `round1_concat_diagnostic_summary.csv`
- `round1_concat_selected_model_counts.csv`
- `round1_concat_stratified_summary.csv`
- `round1_concat_best_variant.json`
- `round1_concat_metadata.json`
- `round1_concat_summary.md`
- `round1_all_variant_comparison.csv`
- `round1_all_variant_summary.md`

两变体均完成 seeds 16/17/18、3 epochs。`pilot_test_used_for_selection=false`，`diagnostic_balanced_used_for_selection=false`，scaler 只在 `pilot_train` fit。

P2d selection raw-soft 结果：

| variant | MAE_mean | MAE_std | regret_to_oracle_mean | oracle_label_accuracy_mean |
| --- | ---: | ---: | ---: | ---: |
| `cls_mean_concat_plus_aux` | 0.300605 | 0.001287 | 0.033874 | 0.467911 |
| `mean_patch_plus_aux` | 0.300831 | 0.000548 | 0.034100 | 0.468133 |

Round 1 selection raw-soft 排名：

| stage | variant | MAE_mean |
| --- | --- | ---: |
| P2d | `cls_mean_concat_plus_aux` | 0.300605 |
| P2d | `mean_patch_plus_aux` | 0.300831 |
| P2b | `visual_mean_patch_only` | 0.300996 |
| P2b | `visual_cls_only` | 0.302048 |
| P2b | `visual_cls_mean_concat` | 0.302220 |
| P2c | `revin_aux_only_fusion_huber_kl` | 0.332987 |
| P1 | `p1_round0_visual_baseline` | 0.334069 |

分层检查：

- `cls_mean_concat_plus_aux` 相比 P2b `visual_cls_mean_concat` 在 PatchTST oracle stratum 有改善，三 seed 平均 raw-soft MAE 约 0.398208 vs 0.400511。
- CrossFormer oracle stratum 未改善，三 seed 平均 raw-soft MAE 约 0.481549，略差于 P2b `visual_cls_mean_concat` 约 0.478257，仍是后续重点短板。

## 结论

1. `mean_patch_plus_aux` 略优于 P2b `visual_mean_patch_only`：selection raw-soft MAE 0.300831 vs 0.300996。
2. `cls_mean_concat_plus_aux` 优于 P2b `visual_cls_mean_concat`：selection raw-soft MAE 0.300605 vs 0.302220。
3. RevIN aux 与 visual embedding 存在可测互补，但增益幅度较小；主要改善体现在 MAE/regret。
4. `cls_mean_concat_plus_aux` 缓解了 P2b `visual_cls_mean_concat` 的 seed 不稳定：raw-soft MAE std 0.001287 vs 0.003929。
5. Round 1 最终 best variant 为 `cls_mean_concat_plus_aux`，只按 `pilot_selection` raw-soft MAE_mean 选择。
6. 建议做一次冻结 `pilot_test` final eval，但只能在当前 best variant/seed/epoch 选择完成后执行，不能参与选择。
7. 建议进入 Round 2 pseudo image / view layout 消融；P2e FiLM/gating/conditional modulation 值得单独开，但不能混入本次 P2d 结论。

## 下一步方案

1. 如需最终报告，可对 `cls_mean_concat_plus_aux` 做一次冻结 `pilot_test` final eval，并在 metadata 中明确 `pilot_test_not_used_for_selection=true`。
2. Round 2 优先围绕 pseudo image / view layout 消融，重点观察 CrossFormer oracle stratum recall/MAE 是否改善。
3. 若开 P2e，只探索 FiLM/gating/conditional modulation，并保持与 P2d 相同的 P0/P2a 输入、selection 选择和 diagnostic 诊断边界。
