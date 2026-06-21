# Visual Router V2 Round2d addendum spatial panel period continuity diagnostic

日志日期：2026-06-22 02:07:04 CST

## 目的

补齐 Round2c best layout `spatial_panel_3view` 的 period continuity diagnostic，按 Round2d 完全相同口径检查 hard top1 period fold 对 pseudo image、ViT embedding、router weight、selected model 和 soft fused prediction 的扰动传播，并与已有 `current_rgb_3view`、`top3fold_period_layout` 结果合并比较。

## 背景

Round2d 已完成 `current_rgb_3view` 与 `top3fold_period_layout` 诊断，结论是 `top3fold_period_layout` 的 image / embedding / router weight 变化显著低于 current hard top1，且 `period_soft_mixture` 不是进入 65k expanded validation 的前置硬门槛。但 Round2c best layout 是 `spatial_panel_3view`，其 selection/test_small raw-soft MAE 分别为 0.310385/0.398598，并且 layout 内也包含 top1 period fold panel，因此需要补充同口径连续性诊断。

## 操作

1. 扩展 `visual_router_experiments/stage1_vali_test_router/diagnose_visual_router_v2_round2_period_continuity.py`：
   - 新增 `--result-prefix`，用于输出 `round2_period_continuity_addendum_*` 命名产物；
   - 新增 `--compare-with-existing` / `--append-to-existing`，用于读取已有 Round2d raw results 做合并对照；
   - 新增 `layout_comparison` 输出；
   - 为 `period_continuity_addendum` metadata 生成七项 spatial panel 结论摘要；
   - 保持 worker 侧 period selection、pseudo image、ViT embedding、router weight、selected model flip 和 fused prediction 指标计算不变。
2. 使用 `quito` 环境验证脚本编译：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile visual_router_experiments/stage1_vali_test_router/diagnose_visual_router_v2_round2_period_continuity.py
   ```

3. 启动 addendum 诊断：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/diagnose_visual_router_v2_round2_period_continuity.py \
     --layouts spatial_panel_3view \
     --output-dir /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_period_continuity_addendum \
     --summary-copy-dir /home/shiyuhong/Time-visual-router-v2/experiment_summaries/visual_router_v2_round2/period_continuity_addendum \
     --result-prefix round2_period_continuity_addendum \
     --compare-with-existing /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_period_continuity/round2_period_continuity_raw_results.csv \
     --devices cuda:0,cuda:1,cuda:2,cuda:3 \
     --overwrite
   ```

4. 任务按 `process_per_layout_seed_sample_set` 并行，单任务隔离到 `tasks/spatial_panel_3view_seed<seed>_<sample_set>/`。共运行 seeds 16/17/18、`round2_selection_small` 与 `round2_diagnostic_balanced_small`，每集合 512 样本，sigma=0.001/0.005/0.01，每 sigma 3 次扰动。
5. 聚合后再次运行 `py_compile` 和 `--aggregate-only`，刷新 addendum summary 与轻量副本。

## 结果

1. 新增输出目录：
   - `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_period_continuity_addendum/`
2. 轻量副本目录：
   - `experiment_summaries/visual_router_v2_round2/period_continuity_addendum/`
3. 关键输出文件已生成：
   - `round2_period_continuity_addendum_raw_results.csv`
   - `round2_period_continuity_addendum_layout_comparison.csv`
   - `round2_period_continuity_addendum_stratified_summary.csv`
   - `round2_period_continuity_addendum_high_change_examples.csv`
   - `round2_period_continuity_addendum_metadata.json`
   - `round2_period_continuity_addendum_summary.md`
4. raw rows 为 27,648，符合 `1 layout × 3 seeds × 2 sample_sets × 512 samples × 3 sigma × 3 perturbations`。
5. metadata 记录：
   - `round2_stage=period_continuity_addendum`
   - `trained_new_model=false`
   - `built_feature_cache=false`
   - `ran_vit_for_embedding_diagnostic=true`
   - `saved_pseudo_image_tensor=false`
   - `used_test_small_for_selection=false`
   - `layouts_diagnosed=["spatial_panel_3view"]`
   - `compared_against_existing=["current_rgb_3view","top3fold_period_layout"]`
   - `parallel_backend=process_per_layout_seed_sample_set`
6. layout-level comparison：

| layout | top1 changed | image cos | mean patch cos | weight JS | weight L1 | selected flip | fused abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_rgb_3view` | 0.095775 | 0.012659 | 0.060437 | 0.020471 | 0.121888 | 0.042969 | 0.003293 |
| `spatial_panel_3view` | 0.095775 | 0.011196 | 0.024991 | 0.021074 | 0.126698 | 0.015553 | 0.003997 |
| `top3fold_period_layout` | 0.095775 | 0.000262 | 0.005865 | 0.001715 | 0.018025 | 0.037507 | 0.003129 |

## 结论

1. `spatial_panel_3view` 也受 hard top1 period fold 扰动影响，top1 period changed ratio 为 0.095775。
2. 相比 `current_rgb_3view`，spatial panel 对 image / embedding 和 selected model flip 有改善：image cosine 0.011196 < 0.012659，mean-patch embedding cosine 0.024991 < 0.060437，selected flip 0.015553 < 0.042969。
3. spatial panel 没有完整降低 router weight 不连续传播：weight JS 0.021074 略高于 current 的 0.020471，weight L1 0.126698 也高于 current 的 0.121888。
4. 相比 `top3fold_period_layout`，spatial panel 的 image / embedding / router weight 连续性整体更弱，但 selected flip 更低。
5. fold panel 是 spatial panel 高变化来源：top1 period changed 行的 image cosine、router JS 和 selected flip 均明显高于 unchanged 行。
6. `spatial_panel_3view` 仍应进入 65k expanded validation，因为它是 Round2c best layout，且 addendum 没有发现需要阻塞 65k 的失败模式。
7. `period_soft_mixture` 不作为 65k 前置硬门槛，可作为后续表达改进单独 smoke。

## 下一步方案

1. 65k expanded validation 推荐 layout 保持：
   - `spatial_panel_3view`
   - `current_rgb_3view`
   - `top3fold_period_layout`
2. 65k 阶段继续禁止使用 test_small 做选择，test_small 只保留 frozen screening 口径。
3. 后续若要优化 period 不连续传播，优先单独做 `period_soft_mixture` smoke，不阻塞当前 65k expanded validation。
