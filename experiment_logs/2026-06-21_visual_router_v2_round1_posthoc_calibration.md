# Visual Router V2 Round 1 P2f post-hoc calibration diagnostic

日志日期：2026-06-21 19:45:06 CST

## 目的

在不重新训练 router、不修改 checkpoint、不重建 P2a feature、不使用 `pilot_test` 调参的前提下，对 Round 1 主要变体做 post-hoc soft weight calibration diagnostic，判断 temperature / entropy-style calibration 是否能进一步改善 soft fusion 的 MAE、MSE、regret、seed stability、CrossFormer/PatchTST strata 和 selected_model ratio。

## 背景

Round 1 global summary 显示 frozen `pilot_test` 当前最强变体为 `film_mean_patch_aux`，raw-soft MAE/MSE/regret=0.417824/183.353985/0.077539；`film_cls_mean_concat_aux` 次之，`visual_cls_mean_concat` 是 visual-only strong baseline。由于 raw-soft 明显优于 hard top-1，且 CrossFormer hard selected ratio 偏低，本轮只对既有 soft weights 做 post-hoc calibration 诊断。

## 操作

1. 新增脚本 `visual_router_experiments/stage1_vali_test_router/calibrate_visual_router_v2_round1_posthoc.py`。
2. 脚本只读取四个候选变体既有 prediction CSV：
   - `film_mean_patch_aux`
   - `film_cls_mean_concat_aux`
   - `visual_cls_mean_concat`
   - `cls_mean_concat_plus_aux`
3. 因既有 prediction CSV 不含 logits，按任务允许口径使用 `normalize(w ** (1/T))` 做 temperature-like power transform；额外测试 uniform entropy interpolation `alpha`。
4. temperature grid 为 `[0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]`，alpha grid 为 `[0.0, 0.02, 0.05, 0.1]`。
5. 复用已有 SQLite prediction index 批量读取专家预测数组并重算 soft fusion MAE/MSE：
   - selection/diagnostic：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling/prediction_index_p2b_train_selection_diagnostic.sqlite`
   - pilot_test：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_final_test_extension/prediction_index_round1_final_test_pilot_test.sqlite`
6. 正式运行命令：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/calibrate_visual_router_v2_round1_posthoc.py --overwrite
   ```

7. 首次运行已完成全部 seed/sample_set 数组重算并写出 seed raw CSV，但在最终 delta 汇总阶段因 Round0 baseline 在全局表中 `variant="Round0 TimeFuse"`、`method="round0_timefuse"` 命名不一致触发 `KeyError: 'round0_timefuse'`。
8. 修正脚本：baseline 查询兼容 `variant` 与 `method`；新增 `--reuse-raw` 以复用已生成的 seed raw CSV；修正 selection delta 口径，并在 stratified summary 中追加相对 original raw-soft 的 delta。
9. 复用 raw 重建汇总：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/calibrate_visual_router_v2_round1_posthoc.py --overwrite --reuse-raw
   ```

10. 执行语法检查：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile visual_router_experiments/stage1_vali_test_router/calibrate_visual_router_v2_round1_posthoc.py
   ```

11. 生成 `/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round1_calibration/`，并复制轻量交付物到 `experiment_summaries/visual_router_v2_round1/p2f_calibration/`。

## 结果

1. 必需交付物均已生成：
   - `round1_calibration_grid_results.csv`
   - `round1_calibration_best_params.csv`
   - `round1_calibration_selection_comparison.csv`
   - `round1_calibration_diagnostic_summary.csv`
   - `round1_calibration_final_test_summary.csv`
   - `round1_calibration_delta_summary.csv`
   - `round1_calibration_selected_model_counts.csv`
   - `round1_calibration_stratified_summary.csv`
   - `round1_calibration_metadata.json`
   - `round1_calibration_summary.md`
2. `/data2` 输出目录额外保留 seed 级复核文件：
   - `round1_calibration_seed_grid_raw.csv`
   - `round1_calibration_seed_selected_counts_raw.csv`
   - `round1_calibration_seed_strata_raw.csv`
3. 四个候选变体在 `pilot_selection` 上均选择 `power_temperature_T0p85`、`entropy_alpha=0.0`。
4. frozen `pilot_test` calibrated 结果：
   - `film_mean_patch_aux`：MAE=0.417536、MSE=183.369086、regret=0.077251；相对 original delta_MAE=-0.000288、delta_MSE=+0.015101。
   - `film_cls_mean_concat_aux`：MAE=0.419259、MSE=183.478104、regret=0.078974；相对 original delta_MAE=-0.000309、delta_MSE=+0.014259。
   - `visual_cls_mean_concat`：MAE=0.441577、MSE=247.844653、regret=0.101292；相对 original delta_MAE=-0.001485、delta_MSE=+3.606165。
   - `cls_mean_concat_plus_aux`：MAE=0.452601、MSE=245.478623、regret=0.112316；相对 original delta_MAE=-0.000341、delta_MSE=+0.019148。
5. `film_mean_patch_aux` 的 CrossFormer stratum 在 `pilot_test` 上相对 original 变差：delta_MAE=+0.002571、delta_MSE=+0.009384。
6. `film_mean_patch_aux` 的 PatchTST stratum 在 `pilot_test` 上 MAE 改善但 MSE 变差：delta_MAE=-0.004724、delta_MSE=+0.009593。
7. `film_mean_patch_aux` CrossFormer hard selected ratio 未改善：original=0.042587、calibrated=0.042587。
8. metadata 明确记录 `calibration_only=true`、`trained_new_model=false`、`changed_checkpoint=false`、`rebuilt_p2a_feature=false`、`used_pilot_test_for_selection=false`、`pilot_test_evaluated=true`、`loaded_116m_prediction_manifest_to_memory=false`、`saved_pseudo_image_tensor=false`、`read_feature_shard=false`、`read_checkpoint=false`。

## 结论

P2f calibration 在四个候选变体上都带来极小 MAE/regret 改善，但 MSE tail 没有改善，且 `film_mean_patch_aux` 的 CrossFormer ratio 不变、CrossFormer stratum 变差。综合来看，不建议把 calibrated `film_mean_patch_aux` 升级为 Round 1 综合 enhanced recommendation；应保持 raw `film_mean_patch_aux` 为当前主推荐，并把本次 calibration 作为 soft weight shape diagnostic。

## 下一步方案

1. 优先推进 view layout Round2，而不是基于本次 calibration 立即扩大 FiLM hyperparameter search。
2. 后续若继续做 calibration，只应在 `pilot_selection` 选择参数，`pilot_test` 保持 frozen eval。
3. CrossFormer hard selected ratio 和 PatchTST/CrossFormer strata 仍需在 Round2 view layout 中重点跟踪。
