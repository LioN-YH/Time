# Visual Router V2 Round2 spatial panel 分层与错误尾部分析

日志日期：2026-06-22 10:04:52 CST

## 目的

基于已有 Round1 `film_mean_patch_aux` 和 Round2f `spatial_panel_3view + film_mean_patch_aux` 的 frozen eval 结果，分析 spatial panel 的收益来自哪些 strata，以及哪些 strata 和 high-regret tail 仍然较弱。

## 背景

Round2f P0-scale validation 已确认 `spatial_panel_3view + film_mean_patch_aux` 在 frozen `pilot_test` 上相对 Round1 `film_mean_patch_aux` 有稳定小幅收益：raw-soft MAE 从 0.417824 降到 0.413558，MSE/regret 也同时改善。但该收益不是决定性突破，需要进一步解释 oracle_model、dataset/group、结构属性、selected_model ratio 和 tail 层面的来源。

## 操作

1. 读取用户粘贴目标文件，确认本窗口边界：不做 full-scale planning，不检查 current layout equivalence，不训练，不跑 ViT，不生成 feature cache，不新建样本。
2. 检查已有 summary 与 run_outputs，确认可复用以下逐样本 prediction CSV：
   - `/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round1_film_final_test_extension/tasks/film_mean_patch_aux_seed{16,17,18}/predictions_film_mean_patch_aux_seed{16,17,18}_pilot_test.csv`
   - `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/tasks/spatial_panel_3view_seed{16,17,18}/predictions_spatial_panel_3view_seed{16,17,18}_pilot_test.csv`
3. 新增只读分析脚本 `visual_router_experiments/stage1_vali_test_router/analyze_visual_router_v2_round2_spatial_panel_strata.py`。脚本按 `sample_key` 聚合三 seed，生成 strata metrics、tail metrics 和 metadata。
4. 使用 Quito 环境执行语法检查和分析：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile visual_router_experiments/stage1_vali_test_router/analyze_visual_router_v2_round2_spatial_panel_strata.py
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/analyze_visual_router_v2_round2_spatial_panel_strata.py
   ```

5. 新增分析产物：
   - `experiment_summaries/visual_router_v2_round2/spatial_panel_strata_error_analysis.md`
   - `experiment_summaries/visual_router_v2_round2/spatial_panel_strata_metrics.csv`
   - `experiment_summaries/visual_router_v2_round2/spatial_panel_error_tail_metrics.csv`
   - `experiment_summaries/visual_router_v2_round2/spatial_panel_strata_metadata.json`
6. 更新 `WORKSPACE_STRUCTURE.md`，登记新增分析脚本和轻量 summary 产物。

## 结果

脚本运行完成并写出三份结构化产物。`spatial_panel_strata_metrics.csv` 共 40 行、65 列，覆盖 overall、oracle_model、dataset_name、group_name、error_gap_quantile、forecastability、season、trend、CV 和 missing ratio 分层。`spatial_panel_error_tail_metrics.csv` 共 96 行、21 列，覆盖 top 1%/5% raw-soft MAE 与 raw-soft regret tail，并记录 tail overlap、oracle_model 分布和 selected_model 分布。

关键结果：

- frozen `pilot_test` raw-soft MAE/MSE/regret delta 分别为 -0.004266、-0.582819、-0.004266；hard top1 MAE/MSE/regret 也改善，但 MAE delta 较小，为 -0.003372。
- oracle_model 分层中 CrossFormer、PatchTST、NaiveForecaster、DLinear、ES 均改善；CrossFormer 和 PatchTST 改善最大，MAE delta 分别为 -0.009317 和 -0.008063，ES 改善最小，为 -0.000946。
- dataset 层面 `TEST_DATA_MIN` 和 `TEST_DATA_HOUR` 均改善；`TEST_DATA_MIN` 样本占 67,808/75,000，对 overall 贡献更大。
- group 层面 `HIGH_HIGH_LOW` 改善最大，MAE delta=-0.015945；`LOW_LOW_HIGH` 明确退化，MAE delta=+0.001635，MSE delta=+0.030233。
- error tail 层面，Round1 top 5% regret tail 上 Round2 mean regret 从 0.887476 降到 0.726327，mean MSE 从 77.979016 降到 65.967985；Round2 自身 top 5% regret tail 仍然存在，selected_model mode 中 PatchTST 占 49.4%。
- selected_model ratio 未出现单专家极端塌缩；Round2 相比 Round1 提高 CrossFormer、ES、NaiveForecaster 选择比例，降低 DLinear 和 PatchTST 选择比例。

## 结论

`spatial_panel_3view + film_mean_patch_aux` 的收益主要来自困难 strata 和 high-regret tail 缓解，不只是整体均匀微小平移。它能作为 Round2 mainline candidate，但当前证据仍应表述为稳定小幅收益。CrossFormer/PatchTST 绝对 MAE 仍高，`LOW_LOW_HIGH` group 退化，`q4` 基本持平，Round2 high-regret tail 中 PatchTST selected mode 偏重仍是风险点。

## 下一步方案

本窗口不做 full-scale planning。给后续 full-scale reporting 的建议是：冻结报告中保留 oracle_model、dataset/group、error_gap_quantile、forecastability/season/trend/CV 分层；固定输出 raw-soft、hard top1、oracle-label accuracy、entropy、mean max weight、selected_model ratio；加入 top 1%/5% high-error/high-regret tail、tail overlap 和 tail 内 oracle/selected 分布。
