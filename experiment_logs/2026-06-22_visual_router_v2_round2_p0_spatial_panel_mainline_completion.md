# Visual Router V2 Round2f P0 spatial panel mainline 完成

日志日期：2026-06-22 08:18:30 CST

## 目的

记录 `spatial_panel_3view + film_mean_patch_aux` 在 P0 pilot 协议规模上的完整验证结果，并完成验收检查、轻量 summary 入仓和结构文档更新。

## 背景

本轮承接 Round2e-b 65k expanded validation 结论：`spatial_panel_3view` 是 selection/test_expanded 双 best。目标是在 P0 原协议规模上只验证 spatial panel 主线，固定后端为 `film_mean_patch_aux`，训练 seeds 16/17/18，只用 `pilot_selection` raw-soft MAE mean 做选择，`pilot_test` 只做 frozen final eval。

## 操作

1. 继续监控后台 launcher：

   ```text
   /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/launcher_stdout.log
   ```

2. 确认 feature 阶段完成：
   - `pilot_train`: 150000
   - `pilot_selection`: 30000
   - `diagnostic_balanced`: 20000
   - `pilot_test`: 75000
3. 抽检正式 `round2_p0_spatial_feature_manifest.csv` 和首尾 shard：
   - manifest rows = 138；
   - total feature rows = 275000；
   - `mean_patch_embedding` 维度为 768；
   - `revin_aux` 维度为 6；
   - finite 检查通过；
   - 四个 sample_set 的 `order_index` 首尾边界均正确。
4. 确认 prediction subset SQLite 构建完成：
   - `target_sample_keys=275000`
   - records = 1375000
   - 未全量加载 116M prediction manifest 到内存。
5. 确认 seeds 16/17/18 的 fixed FiLM task 均完成，写出 checkpoint、selection/diagnostic/pilot_test prediction CSV、`method_rows.csv`、`seed_results.csv` 和 `task_metadata.json`。
6. 聚合后发现 P0 launcher 后处理第一次因 `final_test_summary.csv` 无 `method_kind` 列而失败；修复 `_raw_soft_row`，允许从 `method` 后缀恢复 raw-soft/hard-top1 口径。
7. 用 `--aggregate-only` 重跑单进程 feature aggregation、training aggregation 和 P0 summary 后处理；未重跑 feature cache 或训练。
8. 更新 `/data2` 输出 summary，并同步到：

   ```text
   experiment_summaries/visual_router_v2_round2/p0_spatial_panel_mainline/
   ```

9. 更新 `WORKSPACE_STRUCTURE.md`，登记 P0 输出目录和轻量 summary 目录的完成状态。

## 结果

### 关键指标

`pilot_selection` raw-soft：

- MAE = 0.3002268343315664
- MSE = 1.269413468185334
- regret = 0.03349563069950783
- MAE_std = 0.0005459253748224863

`frozen pilot_test` raw-soft：

- MAE = 0.41355815497163473
- MSE = 182.77116559596718
- regret = 0.07327323095887356
- MAE_std = 0.0022087742458668323

相对 Round1 `film_mean_patch_aux` frozen pilot_test：

- MAE delta = -0.004266，改善；
- MSE delta = -0.582819，改善；
- regret delta = -0.004266，改善。

### Strata

frozen pilot_test 按 `oracle_model` 聚合后，相对 Round1 `film_mean_patch_aux` 的 MAE delta：

- CrossFormer = -0.009317
- PatchTST = -0.008063
- ES = -0.000946
- DLinear = -0.003066

以上四类的 MSE delta 也均为负。

### Selected Model Ratio

pilot_test raw-soft selected_model ratio 的 seed mean：

- CrossFormer = 0.059
- DLinear = 0.111
- ES = 0.292
- NaiveForecaster = 0.213
- PatchTST = 0.325

未出现单专家极端塌缩。

### 必需输出

以下验收文件均已生成：

- `round2_p0_spatial_feature_manifest.csv`
- `round2_p0_spatial_feature_latency.csv`
- `round2_p0_spatial_variant_seed_results.csv`
- `round2_p0_spatial_selection_comparison.csv`
- `round2_p0_spatial_diagnostic_summary.csv`
- `round2_p0_spatial_final_test_summary.csv`
- `round2_p0_spatial_selected_model_counts.csv`
- `round2_p0_spatial_stratified_summary.csv`
- `round2_p0_spatial_delta_summary.csv`
- `round2_p0_spatial_metadata.json`
- `round2_p0_spatial_summary.md`
- `status.json`

## 结论

`spatial_panel_3view + film_mean_patch_aux` 在 P0 pilot_selection 上略优于 Round1 `film_mean_patch_aux`，并且在 frozen pilot_test 上 MAE/MSE/regret 同时优于 Round1 当前 best。seed stability 可接受，oracle_model strata 中 CrossFormer、PatchTST、ES、DLinear 均改善，selected_model ratio 未出现极端塌缩。

因此建议把 `spatial_panel_3view + film_mean_patch_aux` 作为 Visual Router V2 当前主线。

## 下一步方案

优先进入 full-scale validation。`period_soft_mixture` 可作为独立支线继续验证，但不应阻塞 spatial panel 主线推进；canonical migration 可在 full-scale validation 口径稳定后进行。
