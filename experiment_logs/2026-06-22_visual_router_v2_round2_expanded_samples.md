# Visual Router V2 Round2e-a 65k expanded sample builder

日志日期：2026-06-22 03:16:44 CST

## 目的

冻结 Visual Router V2 Round2e-a 的 65k expanded validation sample boundary，为后续 Round2e-b 65k layout validation 准备固定样本集与 manifest。

## 背景

Round2a 已冻结 35k small screening samples，Round2c 完成六个 layout 的 35k feature cache 与 fixed FiLM screening，Round2d/addendum 完成 period continuity 诊断。后续建议把 `spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout` 推进到 65k expanded validation，但此前 65k expanded samples 只存在于 metadata/protocol 的 optional plan 中，尚未实际构建。

本步只构建样本边界，不训练 router、不运行 ViT、不生成 feature cache、不保存 pseudo image tensor、不读取 116M prediction manifest 到内存。

## 操作

1. 新增脚本 `visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round2_expanded_samples.py`。
2. 脚本读取 `/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_small_samples/round2_small_sample_manifest.csv`，把四个 small sample set 作为对应 expanded sample set 的保留种子。
3. 脚本流式扫描 full-scale oracle labels parquet：
   - 使用 stable hash reservoir 估计 `error_gap_quantile` 边界；
   - 从 `vali` 中补齐 `round2_train_expanded` 和 `round2_selection_expanded`；
   - 从 `test` 中补齐 `round2_test_expanded`；
   - 从 `vali` 中按 `oracle_model` 分桶补齐 `round2_diagnostic_balanced_expanded`。
4. 脚本流式读取 TSF enrichment parquet，只为补齐样本 join `group_name`、forecastability/season/trend/cv/missing 等诊断字段。
5. 使用 `quito` 环境执行语法和正式构建：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round2_expanded_samples.py
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round2_expanded_samples.py
   ```

6. 输出写入 `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/`，轻量 summary/metadata/coverage/validation/status 复制到 `experiment_summaries/visual_router_v2_round2/expanded_samples/`。
7. 独立读取最终 manifest 复核 counts、split、集合内重复、跨集合重复、order_index、small subset 和 diagnostic expert counts。
8. 更新 `visual_router_experiments/stage1_vali_test_router/README.md`、`WORKSPACE_STRUCTURE.md` 和本实验日志总览。

## 结果

`py_compile` 通过。正式构建完成，用时约 386.862 秒。

输出样本数：

| sample_set | split | count |
| --- | --- | ---: |
| `round2_train_expanded` | `vali` | 30,000 |
| `round2_selection_expanded` | `vali` | 10,000 |
| `round2_diagnostic_balanced_expanded` | `vali` | 10,000 |
| `round2_test_expanded` | `test` | 15,000 |

独立验收结果：

- 四个 sample_set 内 `sample_key` 重复数均为 0；
- 四个 sample_set 跨集合重复数为 0；
- `round2_train_expanded ∩ round2_selection_expanded = 0`；
- train/selection/diagnostic 全部来自 `vali`；
- test 全部来自 `test`；
- 每个 sample_set 的 `order_index` 均从 0 开始连续；
- 35k small sets 均为对应 expanded set 的严格子集：
  - `round2_train_small`: 20,000/20,000；
  - `round2_selection_small`: 5,000/5,000；
  - `round2_diagnostic_balanced_small`: 5,000/5,000；
  - `round2_test_small`: 5,000/5,000；
- diagnostic expanded 五专家均衡：`CrossFormer=2000`、`DLinear=2000`、`ES=2000`、`NaiveForecaster=2000`、`PatchTST=2000`。

主要产物：

- `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/round2_train_expanded_sample_keys.csv`
- `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/round2_selection_expanded_sample_keys.csv`
- `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/round2_diagnostic_balanced_expanded_sample_keys.csv`
- `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/round2_test_expanded_sample_keys.csv`
- `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/round2_expanded_sample_manifest.csv`
- `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/round2_expanded_overlap_with_small.csv`
- `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/round2_expanded_coverage_summary.csv`
- `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/round2_expanded_validation_summary.csv`
- `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/round2_expanded_sample_metadata.json`
- `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/round2_expanded_sample_summary.md`
- `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/status.json`

metadata 明确记录：

- `round2_stage=expanded_sample_builder`
- `trained_model=false`
- `built_feature_cache=false`
- `ran_vit=false`
- `saved_pseudo_image_tensor=false`
- `used_test_expanded_for_selection=false`
- `loaded_116m_prediction_manifest_to_memory=false`
- `recommended_layouts_for_round2e_b=["spatial_panel_3view","current_rgb_3view","top3fold_period_layout"]`
- `next_step_recommendation=round2e_b_65k_layout_validation`

## 结论

Round2e-a 65k expanded sample boundary 已冻结并通过验收。35k small sample sets 均严格包含于对应 expanded sample sets，可保证 Round2a/Round2c/Round2d 与后续 Round2e-b 的样本边界连续。`round2_test_expanded` 完全来自 test split，且只用于 frozen expanded validation，不用于训练、调参、选择 layout、选择 seed、选择 epoch 或 hyperparams。

## 下一步方案

1. Round2e-b 使用 `round2_expanded_sample_manifest.csv` 构建 65k feature cache。
2. Round2e-b 只验证 `spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout`。
3. Round2e-b 继续固定 `film_mean_patch_aux` 风格后端，避免 layout 变量与 head/hparam 改动混杂。
4. Round2e-b 继续使用多 GPU 进程级并行：feature cache 按 layout 并行，training/eval 按 layout×seed 并行。
