# PatchTST + Visual Dual-Branch 65k 实验说明

## 目的

本实验属于 Visual Router V2 探索分支，用于验证固定视觉编码对 PatchTST 时序预测的增益。它不是 Stage 1 canonical 重构主线，也不重新设计 full-scale router 框架。

## 实验边界

- PatchTST baseline 使用已有 frozen prediction/cache，不在本入口重新训练 PatchTST。
- 视觉分支使用已有 fixed visual embedding cache，不重新生成图像，不重新跑 ViT。
- PatchTST baseline 与 dual-branch 必须使用同一批对齐后的 `sample_key` 和同一 train/val/test split。
- 第一批实现 `feature_concat`、`film`、`residual_feature`、`visual_residual` 四个轻量变体。
- 新增稳健性变体 `patchtst_residual_visual`：以 PatchTST frozen prediction 为基线，只学习 `y_fusion = y_patchtst + residual_scale * delta_y`，delta head 最后一层 zero-init，默认 `residual_scale=0.1`。
- `pred_gate` 已作为可选 mode 支持，但不作为第一批硬门槛。
- 暂不实现 cross-attention 或 token-level 融合。
- 训练入口默认启用 train-only feature standardization：`h_ts` 和 `h_vis` 的 scaler 都只在 train split 上 fit，val/test 只 transform；`config.json` 记录 `feature_standardization.enabled`。
- test 阶段默认使用 validation loss 最优 checkpoint，而不是最后一轮模型；`metrics.json`、`config.json` 和单 run `summary.md` 会记录 `best_val_epoch`、`best_val_loss` 与 `test_checkpoint=best_validation_checkpoint`。
- 当前已完成的 af96c53 结果中 `h_ts_source=flattened_y_patchtst`，不是 PatchTST encoder hidden representation。因此该结果只能解释 prediction-level/fallback-level 融合效果，不能证明真实 PatchTST hidden 与视觉表示融合无效。

## Cache Contract

`--patchtst_cache` 支持单个 `.npz` 或包含多个 `.npz` shard 的目录。必须包含：

- `sample_key` 或 `sample_keys`
- `split`、`splits` 或 `sample_set`
- `h_ts`、`patchtst_hidden`、`ts_embedding` 或 `ts_feature`
- `y_patchtst`、`patchtst_pred`、`y_pred` 或 `prediction`
- `y_true`、`target` 或 `label`

`--visual_embedding_cache` 支持单个 `.npz` 或包含多个 `.npz` shard 的目录。必须包含：

- `sample_key` 或 `sample_keys`
- `h_vis`、`visual_embedding`、`mean_patch_embedding`、`cls_embedding` 或 `visual_feature`

入口会按 `sample_key` 对齐两类 cache，并要求 `train_split`、`val_split`、`test_split` 均非空。

## 从 Prediction Index 派生 PatchTST Cache

如果没有单独保存 PatchTST hidden cache，但已有 prediction subset SQLite，可以先从 PatchTST frozen prediction 派生本实验需要的 cache。此时 `h_ts` 使用 `y_patchtst` flatten 得到，属于 “h_ts 和/或 y_ts” 中的 y_ts fallback 表示。

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

## 单次运行

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m visual_router_experiments.dual_branch_fusion.train_patchtst_visual_65k \
  --data_subset 65k \
  --ts_model patchtst \
  --visual_embedding_cache <VISUAL_CACHE_NPZ_OR_DIR> \
  --patchtst_cache <PATCHTST_CACHE_NPZ_OR_DIR> \
  --fusion_mode film \
  --train_split round2_train_expanded \
  --val_split round2_selection_expanded \
  --test_split round2_test_expanded \
  --epochs 20 \
  --batch_size 256 \
  --lr 1e-3 \
  --residual_scale 0.1 \
  --seed 1 \
  --output_dir outputs/dual_branch_65k/patchtst_visual/film_seed1
```

如果现有 split 名称使用 `vali`，将 `--val_split vali` 传入即可。

## 输出文件

每个 fusion mode / seed 的输出目录包含：

- `config.json`
- `metrics.json`
- `predictions.npz`
- `training_log.txt`
- `summary.md`

`metrics.json` 固定包含：

- `patchtst_mae`
- `patchtst_mse`
- `dual_branch_mae`
- `dual_branch_mse`
- `delta_mae_vs_patchtst`
- `delta_mse_vs_patchtst`
- `beats_patchtst_mae`
- `beats_patchtst_mse`

## 多 seed 汇总

推荐至少使用 seeds `1,2,3` 或 `16,17,18`。示例：

```bash
for seed in 1 2 3; do
  for mode in feature_concat film residual_feature visual_residual patchtst_residual_visual; do
    /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m visual_router_experiments.dual_branch_fusion.train_patchtst_visual_65k \
      --data_subset 65k \
      --ts_model patchtst \
      --visual_embedding_cache <VISUAL_CACHE_NPZ_OR_DIR> \
      --patchtst_cache <PATCHTST_CACHE_NPZ_OR_DIR> \
      --fusion_mode "${mode}" \
      --train_split round2_train_expanded \
      --val_split round2_selection_expanded \
      --test_split round2_test_expanded \
      --epochs 20 \
      --batch_size 256 \
      --lr 1e-3 \
      --residual_scale 0.1 \
      --seed "${seed}" \
      --output_dir "outputs/dual_branch_65k/patchtst_visual/${mode}_seed${seed}" \
      --overwrite
  done
done
```

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m visual_router_experiments.dual_branch_fusion.summarize_results \
  --results_root outputs/dual_branch_65k/patchtst_visual \
  --output_dir outputs/dual_branch_65k/patchtst_visual/summary
```

汇总脚本写出：

- `dual_branch_run_metrics.csv`
- `dual_branch_summary.csv`
- `dual_branch_summary.json`
- `dual_branch_summary.md`

`dual_branch_summary.csv/json/md` 会按 fusion mode 报告 mean/std，并显式包含 `beats_patchtst_mae_rate` 和 `beats_patchtst_mse_rate`。

## Smoke

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/dual_branch_fusion_patchtst_65k_smoke.py
```

smoke 构造 synthetic cache，验证 `feature_concat`、`film`、`residual_feature`、`visual_residual` 和 `patchtst_residual_visual` 可以 forward、跑 2 个 mini train epoch，输出 shape 与 `y_true` 一致，并正常写出 `metrics.json` 与 `summary.md`。

## af96c53 结果解释边界

af96c53 完成的 65k 结果使用从 PatchTST frozen prediction 派生的 `h_ts=flattened_y_patchtst`，不是 PatchTST encoder hidden representation。该设置适合回答“在已有 PatchTST 预测和 fixed visual embedding 上做 prediction-level/fallback-level 轻量融合是否稳定改进 baseline”，但不能回答“视觉信息与真实 PatchTST hidden representation 融合是否有效”。如果后续能够导出真实 PatchTST encoder hidden，应复用同一 cache contract、同一 train-only scaler、同一 best-val checkpoint 规则和同一多 seed 汇总口径重新运行。
