# Visual Router V2 Round2e-b Expanded Layout Validation Handoff

## 最新状态（2026-06-22 04:26:29 CST）

当前目标 `Visual Router V2 Round2e-b 65k expanded layout validation` 已完成并通过验收。

## 关键路径

| 项目 | 路径 |
| --- | --- |
| worktree | `/home/shiyuhong/Time-visual-router-v2` |
| branch | `exp/visual-router-v2-pilot` |
| conda python | `/home/shiyuhong/application/miniconda3/envs/quito/bin/python` |
| 用户目标文件 | `/home/shiyuhong/.codex-tianyu/attachments/b841a4a9-c37e-4f2a-a4ff-3cb3921f7e77/pasted-text-1.txt` |
| expanded sample manifest | `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/round2_expanded_sample_manifest.csv` |
| 正式输出目录 | `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation/` |
| 轻量 summary 目录 | `experiment_summaries/visual_router_v2_round2/expanded_layout_validation/` |
| 启动日志 | `experiment_logs/2026-06-22_visual_router_v2_round2_expanded_layout_validation_launch.md` |
| 完成日志 | `experiment_logs/2026-06-22_visual_router_v2_round2_expanded_layout_validation_completion.md` |

## 已完成内容

1. 参数化并修复 Round2 layout feature builder：
   - `--artifact-prefix`
   - `--layout` worker 不再写 unified manifest，避免并行写冲突
   - `Round2HistoryWindowLoader` 使用 Quito `id_mask` 直接定位 item/channel 行，去掉 expanded shard 内 `deepcopy + select_user_data` 瓶颈
2. 参数化 fixed FiLM training/aggregation：
   - 支持 expanded sample set 参数
   - 支持 `round2_expanded_layout_*` 输出前缀
   - summary 回答用户目标的 9 个问题
3. 新增 expanded 专用 launcher：
   - `visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_expanded_validation_parallel.py`
   - 支持 `--devices`、`--layouts`、`--feature-only`、`--train-only`、`--aggregate-only`、`--overwrite`
4. py_compile 通过。
5. fast-loader feature smoke 通过。
6. 正式任务完成：
   - 3 layouts × 65,000 feature rows
   - prediction subset SQLite `records=325000`
   - 9 个 layout×seed fixed FiLM task 全部 completed
   - aggregation completed
   - 轻量 summary 已复制到仓库

## 验收结果

完整验收脚本输出：

```text
expanded_layout_validation_verification=passed
best_layout spatial_panel_3view
selection_raw_soft_best_MAE 0.3072330755222998
test_raw_soft_best_MAE 0.3943355328734411
```

验收覆盖：

- 必需输出文件全部存在；
- feature manifest 覆盖 `spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout` 和四个 expanded sample sets；
- 每个 layout 样本数为 train 30,000、selection 10,000、diagnostic 10,000、test 15,000；
- 99 个 feature shards 的 required fields、shape、finite 和 order_index 检查通过；
- 9 个 task 均 `status=completed`；
- 每个 task 均生成 selection/diagnostic/test_expanded prediction CSV；
- metadata 记录 only variable 是 layout、base visual input 是 `mean_patch_embedding`、condition input 是 `revin_aux`、使用 FiLM、不 concat aux、不用 test 选择、不保存 pseudo image tensor、不使用 DataParallel/DDP；
- summary 回答目标中的 9 个问题；
- 轻量 summary 已复制到 `experiment_summaries/visual_router_v2_round2/expanded_layout_validation/`。

## 正式结论

- 65k selection best：`spatial_panel_3view`
  - raw-soft MAE = 0.307233
  - raw-soft MSE = 2.043914
- 65k `round2_test_expanded` best：`spatial_panel_3view`
  - raw-soft MAE = 0.394336
  - raw-soft MSE = 2.008546
- selection best 与 test best 一致。
- `spatial_panel_3view` 仍优于 `current_rgb_3view`。
- `top3fold_period_layout` 的 continuity / diagnostic 价值未转化为 expanded 主指标优势。
- 35k 结论在 65k 上稳定。
- 建议把 `spatial_panel_3view + film_mean_patch_aux` 升级为 Round2 主线。
- 下一步建议扩大到 P0/P2a 规模；`period_soft_mixture` / period tokens 作为后续独立实验，不作为前置门槛。

## 后续建议

1. 基于 `spatial_panel_3view + film_mean_patch_aux` 设计 P0/P2a 规模扩展。
2. 保留 `current_rgb_3view` 为 baseline。
3. 保留 `top3fold_period_layout` 为 period diagnostic 对照。
4. 不提交 `/data2` 中的 feature cache、checkpoint、SQLite 或逐样本 prediction CSV。
