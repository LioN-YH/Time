# Visual Router V2 Round2 staged full-scale validation thin slice

日志日期：2026-06-22 11:06:48 CST

## 目的

落地 Visual Router V2 Round2 staged full-scale validation 的 thin slice，把后续 full-scale 验证拆成 sample、shard-aware feature cache、SQLite prediction lookup、fixed FiLM train/eval 和 report schema 五个可恢复阶段。

## 背景

Round2 global summary、current layout equivalence check 和 spatial panel strata/error analysis 已完成。当前主线候选为 `spatial_panel_3view + film_mean_patch_aux`，required baseline 为 `current_rgb_3view + film_mean_patch_aux`。本步不是启动 1M 或 116M 正式长跑，而是验证可扩展链路。

## 操作

1. 新增 `build_visual_router_v2_round2_staged_samples.py`：
   - 从 full-scale `sample_manifest_full_scale/sample_shards/sample_shard_0000_of_0064.csv` 构建 `staged_train`、`staged_selection`、`staged_diagnostic`、`staged_test`。
   - `train/selection/diagnostic` 使用 `vali`，`test` 使用 `test`。
   - 通过 oracle parquet batch scan 补 `oracle_model`、`error_gap`、`error_gap_quantile`。

2. 新增 `launch_visual_router_v2_round2_staged_validation_parallel.py`：
   - 复用现有 Round2 layout feature builder 和 fixed FiLM trainer。
   - 支持 `--layouts spatial_panel_3view,current_rgb_3view`、`--backend film_mean_patch_aux`、`--sample-scale smoke/one_shard`、`--devices`、`--feature-only`、`--train-only`、`--eval-only`、`--aggregate-only`、`--dry-run`、`--local-files-only`、`--overwrite`。
   - 先构建 subset SQLite prediction index，再启动 layout × seed worker。

3. 新增 `summarize_visual_router_v2_round2_staged_validation.py`：
   - 检查 feature manifest、SQLite prediction lookup 覆盖。
   - 输出 overall、strata、tail、router behavior 和 metadata。
   - 覆盖 top 1%/5% soft MAE、top 1%/5% regret、tail overlap、tail 内 oracle/selected model distribution。

4. 新增轻量文档：
   - `experiment_summaries/visual_router_v2_round2/staged_fullscale_validation_plan.md`
   - `experiment_summaries/visual_router_v2_round2/staged_fullscale_report_schema.md`
   - `experiment_summaries/visual_router_v2_round2/staged_fullscale_metadata.json`

5. 更新 `WORKSPACE_STRUCTURE.md`，登记新增脚本、输出目录和轻量 summary 文件。

## 结果

1. `py_compile` 通过：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round2_staged_samples.py \
     visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_staged_validation_parallel.py \
     visual_router_experiments/stage1_vali_test_router/summarize_visual_router_v2_round2_staged_validation.py
   ```

2. launcher dry-run 通过：
   - 输出 `staged_launcher_metadata.json` 和完整 command preview。
   - command preview 覆盖 sample builder、feature worker、feature aggregation、prediction index、layout×seed train task、trainer aggregation 和 staged summary。
   - metadata 明确 `not_1m_run=true`、`not_116m_full_scale_run=true`、`loaded_116m_prediction_manifest_to_memory=false`、`saved_pseudo_image_tensor=false`。

3. very small smoke 完成：
   - 输出目录：`/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_staged_fullscale_validation_thin_slice/`
   - 样本：`staged_train/staged_selection/staged_diagnostic/staged_test` 各 32 条，共 128 个唯一 sample_key。
   - layouts：`spatial_panel_3view,current_rgb_3view`。
   - seed：16。
   - epochs：1。
   - feature manifest：`round2_staged_fullscale_feature_manifest.csv`，8 行，覆盖 2 layouts × 4 sample sets，每格 32 条，missing shard count=0。
   - prediction lookup：`prediction_index_round2_layout_subset.sqlite`，128 sample_key × 5 experts = 640 records，五专家各 128 条，检查通过。
   - 训练/评估：两个 layout×seed task 均完成，写出 staged selection/diagnostic/test prediction CSV。
   - 聚合：写出 `round2_staged_fullscale_overall_report.csv`、`round2_staged_fullscale_strata_report.csv`、`round2_staged_fullscale_tail_report.csv`、`round2_staged_fullscale_router_behavior_report.csv`、`round2_staged_fullscale_metadata.json` 和 `round2_staged_fullscale_validation_summary.md`。
   - 轻量 summary 已复制到 `experiment_summaries/visual_router_v2_round2/staged_fullscale_validation/`。

4. 修复项：
   - 首次 smoke 在最后 summary 阶段失败，原因是 `pandas.DataFrame.to_markdown()` 需要当前 `quito` 环境未安装的可选依赖 `tabulate`。
   - 已把 summary markdown 输出改为脚本内置简单 Markdown 表生成。
   - 同时兼容现有 prediction CSV 的 `soft_fusion_mae/soft_fusion_mse` 与 `hard_top1_mae_from_array/hard_top1_mse_from_array` 字段，并由 `soft_fusion_mae - oracle_value` 计算 raw-soft regret。
   - 修复后重新运行 summary 和 launcher `--aggregate-only`，均通过，launcher metadata 更新为 completed。

## 结论

staged full-scale validation thin slice 已完成并通过 very small smoke。当前产物证明 P0/65k 后续可以沿同一 pipeline 扩大到 one-shard、1M staged planning 和 near-full scale：样本构建不读取 116M prediction manifest，feature cache 按 shard/layout/sample_set 组织，prediction lookup 使用 subset SQLite，report schema 覆盖 overall、strata、tail、selected_model ratio、entropy 和 per-seed metrics。

## 下一步方案

1. 后续单独 goal 可运行 `--sample-scale one_shard --dry-run` 或 one-shard 小规模执行，验证单 shard 更大切片的 I/O 和 report 稳定性。
2. 扩到 1M 前应先增加 shard list 参数，并保持 `selection` 只用于选择、`test` 只做 frozen eval。
3. full-scale report 继续监控 CrossFormer/PatchTST、`LOW_LOW_HIGH`、q4/q5 error_gap、low forecastability/strong trend/highly_variable CV 和 high-regret tail 中 PatchTST selected mode 偏重。
