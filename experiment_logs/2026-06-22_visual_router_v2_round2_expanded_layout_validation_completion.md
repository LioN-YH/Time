# Visual Router V2 Round2e-b 65k expanded layout validation 完成日志

日志日期：2026-06-22 04:26:29 CST

## 目的

完成 Round2e-b 65k expanded samples 上的三 layout validation，固定后端为 `film_mean_patch_aux`，并按 selection raw-soft MAE mean 选择 layout，同时验证 frozen `round2_test_expanded` 上的稳定性。

## 背景

上一阶段已经启动正式后台 launcher，但最初 feature worker 使用 `deepcopy(dataset) + select_user_data(item_id)` 读取历史窗口，在 expanded shard 上遇到严重 CPU 瓶颈。为了在不改变样本、layout、seed、selection 规则和 metric 口径的前提下完成目标，本轮先优化 feature loader，再重启正式任务。

## 操作

1. 停止旧正式 launcher 和三个 feature worker，确认无残留相关进程。
2. 修改 `build_visual_router_v2_round2_layout_features.py` 的 `Round2HistoryWindowLoader`：
   - 新增 `_item_rows_cache`。
   - 通过 Quito dataset 的 `id_mask[:,0,0]` 直接定位 `item_id` 对应的 `(item, channel)` 行。
   - 用 `channel_id` 在该 item 的行数组中选择实际 data 行。
   - 去掉 shard 内每个 item 的 `deepcopy + select_user_data`。
3. 使用 `quito` 环境重新执行 py_compile，结果通过。
4. 运行 fast-loader feature smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_expanded_validation_parallel.py \
     --output-dir /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation_smoke_fast_loader \
     --layouts spatial_panel_3view \
     --devices cuda:0 \
     --feature-only \
     --max-samples-per-set 2 \
     --overwrite \
     --poll-seconds 2
   ```

5. 验证 fast-loader smoke 的 feature manifest 和 shard shape/finite 通过。
6. 用优化后的代码重启正式 launcher：

   ```text
   setsid bash -c 'exec /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_expanded_validation_parallel.py --devices cuda:0,cuda:1,cuda:2,cuda:3 --layouts spatial_panel_3view,current_rgb_3view,top3fold_period_layout --overwrite > /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation/launcher.nohup.log 2>&1' &
   ```

7. 监控 feature、prediction index、training/eval 和 aggregation：
   - feature 阶段三个 layout 进程并行完成。
   - prediction subset SQLite 构建完成。
   - 9 个 layout×seed task 分批在 4 张 GPU 上完成。
   - aggregation 写出统一 CSV/JSON/Markdown，并复制轻量 summary 到仓库。
8. 执行完整验收脚本，覆盖必需文件、feature counts、shard shape/finite/order、9 task、metadata 约束、selection/test best 和 summary 必答问题。

## 结果

正式输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation/
```

轻量 summary 目录：

```text
experiment_summaries/visual_router_v2_round2/expanded_layout_validation/
```

关键运行结果：

1. feature 阶段完成：
   - 3 layouts × 65,000 rows = 195,000 feature rows。
   - 每个 layout 33 shards。
   - 每个 layout 覆盖：
     - `round2_train_expanded=30,000`
     - `round2_selection_expanded=10,000`
     - `round2_diagnostic_balanced_expanded=10,000`
     - `round2_test_expanded=15,000`
   - shard 抽检和完整遍历检查均通过，`cls_embedding=(N,768)`、`mean_patch_embedding=(N,768)`、`revin_aux=(N,6)`，数值 finite，`order_index` 连续且与 manifest 对齐。
2. prediction SQLite 完成：
   - `prediction_index_round2_layout_subset.sqlite`
   - `records=325000`
   - `target_sample_keys=65000`
   - 对应 65,000 sample_key × 5 experts。
3. training/eval 完成：
   - 3 layouts × 3 seeds = 9 task。
   - seeds = 16, 17, 18。
   - epochs = 3。
   - 所有 task 的 `status.json` 均为 `completed`。
   - 每个 task 均生成 selection、diagnostic、test_expanded prediction CSV、checkpoint、method rows、seed results 和 task metadata。
4. aggregation 完成，`status.json` 为 `completed`。
5. 必需输出文件全部存在：
   - `round2_expanded_layout_feature_manifest.csv`
   - `round2_expanded_layout_feature_latency.csv`
   - `round2_expanded_layout_variant_seed_results.csv`
   - `round2_expanded_layout_selection_comparison.csv`
   - `round2_expanded_layout_diagnostic_summary.csv`
   - `round2_expanded_layout_test_summary.csv`
   - `round2_expanded_layout_selected_model_counts.csv`
   - `round2_expanded_layout_stratified_summary.csv`
   - `round2_expanded_layout_delta_summary.csv`
   - `round2_expanded_layout_best_layout.json`
   - `round2_expanded_layout_validation_metadata.json`
   - `round2_expanded_layout_validation_summary.md`
   - `status.json`
6. 完整验收脚本输出：

   ```text
   expanded_layout_validation_verification=passed
   best_layout spatial_panel_3view
   selection_raw_soft_best_MAE 0.3072330755222998
   test_raw_soft_best_MAE 0.3943355328734411
   ```

## 主要结论

1. 65k selection best 为 `spatial_panel_3view`：
   - raw-soft MAE = 0.307233
   - raw-soft MSE = 2.043914
2. 65k `round2_test_expanded` best 也是 `spatial_panel_3view`：
   - raw-soft MAE = 0.394336
   - raw-soft MSE = 2.008546
3. selection best 与 test best 一致。
4. `spatial_panel_3view` 仍优于 `current_rgb_3view`：
   - selection raw-soft MAE delta = -0.002420
   - selection raw-soft MSE delta = -0.077461
5. `top3fold_period_layout` 的 continuity / diagnostic 价值没有转化为 expanded selection/test 主指标优势：
   - selection raw-soft MAE = 0.312709，弱于 `current_rgb_3view` 和 `spatial_panel_3view`
   - test raw-soft MAE = 0.399203，弱于 `spatial_panel_3view`，略优于 `current_rgb_3view`
6. 35k 结论在 65k 上稳定：`spatial_panel_3view` 仍是 selection/test best。
7. seed stability、selection/test MSE tail 和 CrossFormer/PatchTST strata 综合上推荐 `spatial_panel_3view`。
8. 建议把 `spatial_panel_3view` 升级为 Round2 主线。
9. 下一步建议扩大到 P0/P2a 规模，而不是继续把 `period_soft_mixture` / period tokens 作为前置阻塞项。

## 结论

Round2e-b 65k expanded layout validation 已完成并通过验收。`spatial_panel_3view` 在 65k selection 与 frozen test_expanded 上均为 best layout，支持将其升级为 Round2 主线。

## 下一步方案

1. 以 `spatial_panel_3view + film_mean_patch_aux` 作为 Round2 主线候选，设计 P0/P2a 规模扩展方案。
2. 保留 `current_rgb_3view` 作为 RGB channel mixing baseline。
3. 保留 `top3fold_period_layout` 作为 diagnostic/period continuity 对照，但不作为下一步主线。
4. `period_soft_mixture` / period tokens 可作为后续独立改进实验，不作为扩大规模前置门槛。
