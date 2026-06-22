# Visual Router V2 Round2 panel gating 35k screening

日志日期：2026-06-22 20:39:31 CST

## 目的

在 Round2 35k small 样本上比较 `spatial_panel_3view` 的 panel-aware gating / residual 候选是否相对当前 `film_mean_patch_aux` baseline 有稳定收益，并判断是否值得进入 65k expanded validation。

## 背景

Round2 当前主线为 `spatial_panel_3view + film_mean_patch_aux`。此前 35k panel-wise pooling 直接 concat 分支未能优于 baseline，且 selection 的 q5 high-error、PatchTST、CrossFormer strata 出现退化，因此 direct panel concat 保留为 side branch。本次只验证更轻量的 768 维 residual/gating 表示，严格限制为 35k small screening，不做 full-scale、65k、P0 或新的 layout search。

## 操作

1. 新增 `visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_panel_gating_probe.py`。
   - baseline `film_mean_patch_aux` 继续使用 `global_mean_patch` 和 FiLM aux；
   - `film_panel_gated_mean_aux` 与 `film_panel_lowrank_aux` 读取 `panel_mean_concat`，复原为 `[B,3,768]` 后与 FiLM router 端到端训练；
   - 所有候选输出保持 768 维 visual representation；
   - 明确记录不重新跑 ViT、不保存 pseudo image tensor、不做 full-scale/65k/P0，`round2_test_small` 不参与训练、调参、选择 variant、选择 seed 或选择 epoch。
2. 使用既有 feature cache：
   - `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_panel_pooling_35k_features/`
   - 确认 shard 中包含 `global_mean_patch`、`panel_mean_concat`、`revin_aux`，不包含 pseudo image tensor。
3. 预构建本次 35k screening 共用 prediction SQLite index：
   - 输出：`/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_panel_gating_35k_screening/prediction_index_round2_panel_gating_subset.sqlite`
   - 结果：`records=175000`，`target_sample_keys=35000`。
4. 先用 seed 999、每集合 64 条样本做 smoke：
   - `film_mean_patch_aux`
   - `film_panel_gated_mean_aux`
   - `film_panel_lowrank_aux`
   - 三条路径均完成训练、预测和 method rows 写出；seed 999 只用于路径验证，不进入正式汇总。
5. 正式训练/评估：
   - variants：`film_mean_patch_aux`、`film_panel_gated_mean_aux`、`film_panel_lowrank_aux`
   - seeds：16、17、18
   - epochs：3
   - train：`round2_train_small` 20000 vali
   - selection：`round2_selection_small` 5000 vali
   - diagnostic：`round2_diagnostic_balanced_small` 5000 vali
   - frozen screening：`round2_test_small` 5000 test
6. 聚合时首次在 key strata delta 处失败，原因是 stratified 表按 seed 保留多行，同一 MultiIndex 命中多个 row；已修正为先按 `sample_set/variant/stratum_column/stratum_value` 做 seed mean，再计算相对 baseline delta。修正后重新运行 aggregate 成功。

## 结果

正式 9 个 task 均完成，输出目录为：

`/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_panel_gating_35k_screening/`

轻量归档已复制到：

`experiment_summaries/visual_router_v2_round2/`

主要文件包括：

- `panel_gating_35k_screening_summary.md`
- `panel_gating_35k_selection_summary.csv`
- `panel_gating_35k_diagnostic_summary.csv`
- `panel_gating_35k_test_small_summary.csv`
- `panel_gating_35k_selected_model_counts.csv`
- `panel_gating_35k_stratified_summary.csv`
- `panel_gating_35k_key_strata_delta.csv`
- `panel_gating_35k_metadata.json`
- `panel_gating_35k_variant_seed_results.csv`

selection raw-soft 主指标：

| variant | MAE_mean | MSE_mean | regret_mean | MAE delta vs baseline |
| --- | ---: | ---: | ---: | ---: |
| `film_mean_patch_aux` | 0.310385 | 3.329199 | 0.046935 | 0.000000 |
| `film_panel_lowrank_aux` | 0.308605 | 3.153192 | 0.045155 | -0.001780 |
| `film_panel_gated_mean_aux` | 0.314789 | 3.651150 | 0.051340 | +0.004405 |

frozen test_small raw-soft：

| variant | MAE_mean | MSE_mean | regret_mean |
| --- | ---: | ---: | ---: |
| `film_mean_patch_aux` | 0.398598 | 3.484102 | 0.065249 |
| `film_panel_lowrank_aux` | 0.397761 | 3.481281 | 0.064412 |
| `film_panel_gated_mean_aux` | 0.398413 | 3.479709 | 0.065064 |

selection 关键 strata：

- `film_panel_lowrank_aux` 在 q5 high-error 上改善：MAE delta=-0.003904，MSE delta=-1.083095。
- `film_panel_lowrank_aux` 在 PatchTST/CrossFormer strata 仍退化：PatchTST MAE delta=+0.002864，CrossFormer MAE delta=+0.001717。
- `film_panel_gated_mean_aux` 明显退化：q5 MAE delta=+0.030381，PatchTST MAE delta=+0.035485，selection overall MAE/MSE/regret 也退化。
- `LOW_LOW_HIGH` group：lowrank 轻微改善，gated 轻微退化。

## 结论

本次 35k screening 不建议进入 65k expanded validation。

`film_panel_lowrank_aux` 虽然在 selection overall raw-soft MAE/MSE/regret 和 frozen test_small 上略优于 baseline，但 PatchTST 与 CrossFormer 关键 strata 仍有退化，不满足“selection 主指标、MSE/regret、seed 稳定性、q5、PatchTST/CrossFormer、selected_model ratio、test_small 同向”同时通过的升级条件。

`film_panel_gated_mean_aux` 在 selection overall、q5 和 PatchTST 上明显退化，应丢弃或推迟。

因此 panel-aware gating/residual 当前不影响并行 full-scale 主线；full-scale 仍继续使用 `spatial_panel_3view + film_mean_patch_aux`。

## 下一步方案

1. 将 panel-aware gating/residual 保留为 side/drop 记录，不进入 65k。
2. Round2 exploration 后续如继续支线，应优先转向 period-aware、padding-mask、resize-interpolation 等方向，而不是继续扩大 panel-aware pooling/gating。
3. 若未来重新考虑 low-rank residual，需要先解决 PatchTST/CrossFormer strata 退化，再谈更大规模验证。
