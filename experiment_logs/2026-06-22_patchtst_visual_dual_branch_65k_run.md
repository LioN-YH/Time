# PatchTST + Visual dual-branch 65k 正式运行

日志日期：2026-06-22 22:41:16 CST

## 目的

在 Visual Router V2 探索分支上，使用已有 fixed visual embedding cache 和 PatchTST frozen prediction cache，完成 65k expanded 数据边界上的 PatchTST baseline 与 PatchTST+Visual dual-branch 轻量融合对比。

## 背景

前一步已经新增 `visual_router_experiments/dual_branch_fusion/` 代码、文档和 synthetic smoke。本步继续补齐真实 65k 输入：现有 Round2 expanded layout validation 已保存 `spatial_panel_3view` fixed visual embedding shards，并已有 prediction subset SQLite 覆盖 65,000 个 expanded sample_key × 5 experts。由于没有单独 PatchTST hidden cache，本步从 SQLite 中抽取 PatchTST `y_pred/y_true`，并将 `y_patchtst` flatten 作为 `h_ts` fallback，符合目标中 “h_ts 和/或 y_ts” 的 y_ts 口径。

## 操作

1. 修复 `cache_dataset.py`，允许读取既有 Round2 `.npz` feature shard 中 object-array 形式的 `sample_key`，读取后立即转为字符串。
2. 新增 `build_patchtst_cache_from_prediction_index.py`，从 sample manifest 和 prediction subset SQLite 派生 PatchTST frozen cache。
3. 构建 PatchTST cache：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m visual_router_experiments.dual_branch_fusion.build_patchtst_cache_from_prediction_index \
  --sample_manifest /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/round2_expanded_sample_manifest.csv \
  --prediction_index /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation/prediction_index_round2_layout_subset.sqlite \
  --output_npz /data2/syh/Time/run_outputs/2026-06-22_patchtst_visual_dual_branch_65k/inputs/patchtst_frozen_cache_from_round2_expanded.npz \
  --output_metadata /data2/syh/Time/run_outputs/2026-06-22_patchtst_visual_dual_branch_65k/inputs/patchtst_frozen_cache_metadata.json \
  --model_name PatchTST \
  --split_field sample_set \
  --sample_sets round2_train_expanded,round2_selection_expanded,round2_diagnostic_balanced_expanded,round2_test_expanded \
  --batch_size 2048
```

4. 使用 `spatial_panel_3view` fixed visual embedding cache，分别运行四个 fusion mode：
   - `feature_concat`
   - `film`
   - `residual_feature`
   - `visual_residual`
5. 每个 mode 使用 seed 1、20 epochs、batch size 256、lr 1e-3、hidden dim 256、dropout 0.1；训练 split 为 `round2_train_expanded`，validation split 为 `round2_selection_expanded`，test split 为 `round2_test_expanded`。
6. 运行 `summarize_results.py` 生成统一汇总：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m visual_router_experiments.dual_branch_fusion.summarize_results \
  --results_root /data2/syh/Time/run_outputs/2026-06-22_patchtst_visual_dual_branch_65k/patchtst_visual/spatial_panel_3view \
  --output_dir /data2/syh/Time/run_outputs/2026-06-22_patchtst_visual_dual_branch_65k/patchtst_visual/spatial_panel_3view/summary
```

## 结果

- PatchTST frozen cache 构建完成，`sample_count=65000`，target shape 为 `[48, 1]`，`h_ts_source=flattened_y_patchtst`。
- 每个 run 的 `aligned_sample_count=65000`，`train_samples=30000`，`val_samples=10000`，`test_samples=15000`。
- 每个 run 均写出 `config.json`、`metrics.json`、`predictions.npz`、`training_log.txt`、`summary.md`。
- `predictions.npz` 中 `y_patchtst`、`y_fusion`、`y_true` shape 均为 `[15000, 48, 1]`。
- 统一汇总写出到 `/data2/syh/Time/run_outputs/2026-06-22_patchtst_visual_dual_branch_65k/patchtst_visual/spatial_panel_3view/summary/`。

核心指标如下：

| fusion_mode | PatchTST MAE | PatchTST MSE | Dual-branch MAE | Dual-branch MSE | delta_mae | delta_mse | MAE beats | MSE beats |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| feature_concat | 0.47232839 | 2.13463616 | 0.50705421 | 3.02721500 | -0.03472582 | -0.89257884 | false | false |
| film | 0.47232839 | 2.13463616 | 0.47495529 | 2.15019655 | -0.00262690 | -0.01556039 | false | false |
| residual_feature | 0.47232839 | 2.13463616 | 0.47987416 | 2.09826279 | -0.00754577 | 0.03637338 | false | true |
| visual_residual | 0.47232839 | 2.13463616 | 0.47753343 | 2.15689278 | -0.00520504 | -0.02225661 | false | false |

## 结论

在 `spatial_panel_3view` fixed visual embedding、PatchTST frozen prediction flatten 作为 `h_ts` fallback、seed 1 的 65k expanded 设置下，四个双分支变体都没有在 MAE 上超过 PatchTST baseline。`residual_feature` 在 MSE 上超过 PatchTST，MSE delta 为 `0.03637338`，但 MAE 仍退化 `0.00754577`。本轮不能声称视觉双分支稳定优于 PatchTST；更合理的结论是固定视觉 embedding 对 PatchTST 预测的轻量融合在当前设置下没有带来 MAE 增益。

## 下一步方案

1. 若继续探索，应优先补真实 PatchTST hidden representation，而不是只使用 `y_patchtst` flatten 作为 `h_ts`。
2. 可对 `visual_residual` 增加更强约束，例如残差零初始化、较小学习率或 residual scale，避免破坏强 baseline。
3. 如需正式结论，应增加多 seed，并固定报告 mean/std；当前结果是 seed 1 单次 65k expanded 实验。
4. 如需比较其他视觉编码，可复用同一 PatchTST frozen cache，替换 `--visual_embedding_cache` 为其他 layout 的 fixed embedding cache。
