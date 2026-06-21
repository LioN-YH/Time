# Visual Router V2 Round2 Small Screening Protocol

生成日期：2026-06-21 21:25:00 CST

## 目的

Round2 的核心变量是 pseudo image / view layout。后端 router/head 固定沿用 Round1 已验证最强的 `film_mean_patch_aux` 路线：base visual input 使用 mean patch embedding，condition input 使用 RevIN aux，aux 通过 FiLM gamma/beta 调制 visual hidden representation。主指标固定为 raw-soft MAE、raw-soft MSE 和 raw-soft regret；oracle-label accuracy 只作为解释指标。

本步只冻结 small screening 的样本集、候选 layout 设计和 top3fold 复用边界，不生成 full feature cache，不训练 router，不跑 ViT，不启动 200k P2a-style feature cache。

## 冻结样本集

外部输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_small_samples/
```

| sample_set | split | count | 用途 |
| --- | --- | ---: | --- |
| `round2_train_small` | vali | 20,000 | 后续 small layout router 训练 |
| `round2_selection_small` | vali | 5,000 | 后续 layout、seed、epoch、hyperparameter 选择 |
| `round2_diagnostic_balanced_small` | vali | 5,000 | oracle expert balanced 诊断，不参与选择 |
| `round2_test_small` | test | 5,000 | frozen screening only，不参与训练、调参、选择 layout、选择 seed、选择 epoch 或 hyperparams |

采样规则：

- 复用 P0 的稳定 hash / deterministic sampling 思路，seed 为 `20260621`。
- `round2_train_small` 和 `round2_selection_small` 均来自 vali，且 sample_key 不重叠。
- `round2_diagnostic_balanced_small` 来自 vali，按五个 oracle expert 近似均衡抽样，每个 expert 1000 条；仅用于诊断。
- `round2_test_small` 来自 test，只用于 frozen screening。
- 不全量加载 116M prediction manifest 到内存；只扫描 full-scale oracle labels parquet 和 TSF enrichment parquet。
- 保留 `sample_key`、`dataset_name`、`group_name`、`oracle_model`、`error_gap_quantile`、`forecastability_cat`、`season_strength_cat`、`trend_strength_cat`、`cv_cat`、`missing_ratio_cat` 等诊断字段。

可选 expanded plan 已写入 metadata，但本轮未构建：

| sample_set | optional count |
| --- | ---: |
| train | 30,000 |
| selection | 10,000 |
| diagnostic | 10,000 |
| test | 15,000 |

## 输出文件

外部目录保存：

- `round2_train_small_sample_keys.csv`
- `round2_selection_small_sample_keys.csv`
- `round2_diagnostic_balanced_small_sample_keys.csv`
- `round2_test_small_sample_keys.csv`
- `round2_small_sample_manifest.csv`
- `round2_layout_candidates.json`
- `round2_top3fold_reuse_audit.md`
- `round2_small_sample_metadata.json`
- `round2_small_screening_summary.md`
- `round2_coverage_summary.csv`
- `round2_validation_summary.json`

轻量 summary 已复制到：

```text
experiment_summaries/visual_router_v2_round2/small_samples/
```

## Layout Candidates

第一轮默认 layout set：

1. `current_rgb_3view`
2. `spatial_panel_3view`
3. `line_only`
4. `line_difference_band`
5. `fft_absolute_energy`
6. `top3fold_period_layout`

第一轮暂缓：

1. `period_soft_mixture`
2. `independent_view_encoder`

每个 candidate 的 `layout_name`、`layout_family`、`input_source`、`pseudo_image_size`、`channel_design`、`panel_design`、`uses_revin_normalized_shape`、`uses_difference_or_volatility`、`uses_frequency_information`、`uses_period_folding`、`uses_scale_statistics`、`shortcut_risk`、`expected_helped_strata`、`expected_failure_modes`、`implementation_status`、`estimated_single_layout_feature_cache_time`、`default_in_round2a` 和 `notes` 均记录在 `round2_layout_candidates.json`。

## top3fold 复用边界

已有 top3fold 相关实现可复用：

- `visual_router_experiments/common/pseudo_imageization.py::select_fft_periods`
- `visual_router_experiments/common/pseudo_imageization.py::imageize_top3fold`
- `visual_router_experiments/common/vit_embedding_utils.py::make_pseudo_images`

当前实现已经支持 FFT top-k 周期选择、固定候选周期桶、按周期分桶批量 fold，以及 `[B, 3, H, W]` ViT-compatible tensor 输出。Round2 下一步需要把它接入 layout registry/adapter，避免把 `top3fold_period_layout` 与新 spatial panel、line-only、difference-band layout 混在旧 `variant_b_top3fold` 参数语义里。

## 后续耗时估计

- P2a 约 200k feature cache ≈ 5h。
- P2d/P2e final_test_only 约 75k feature cache/eval ≈ 2.5h。
- 35k samples 的单 layout small feature cache 预计约 1-1.5h。
- 5 个 layout 顺序跑约 5-8h。
- 若使用 3 张 GPU 进程级并行，wall time 预计约 2-3h。
- `independent_view_encoder` 若每个 view 单独过 ViT，可能接近 2-3 倍成本，不建议第一轮默认执行。

## 下一步

下一步应实现 Round2 small feature cache screening 的 layout registry 和 feature builder 参数化入口。默认只处理本协议冻结的 35k small samples，后端固定为 `film_mean_patch_aux`，不得在同一轮同时改变 router/head、loss、selection split 或 test usage 规则。
