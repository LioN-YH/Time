# Visual Router V2 Round2c Layout Screening Handoff

## 最新状态（2026-06-22 00:12:37 CST）

当前目标 `Visual Router V2 Round2c 35k layout feature cache and fixed FiLM screening` 已完成并通过验收。

## 关键路径

| 项目 | 路径 |
| --- | --- |
| worktree | `/home/shiyuhong/Time-visual-router-v2` |
| branch | `exp/visual-router-v2-pilot` |
| conda python | `/home/shiyuhong/application/miniconda3/envs/quito/bin/python` |
| 正式输出目录 | `/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_layout_screening/` |
| 轻量 summary | `experiment_summaries/visual_router_v2_round2/layout_screening/` |
| 实验日志 | `experiment_logs/2026-06-21_visual_router_v2_round2_layout_screening.md` |

## 已完成内容

1. 新增并验证三个 Round2c 入口：
   - `visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round2_layout_features.py`
   - `visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round2_layout_film.py`
   - `visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_layout_screening_parallel.py`
2. 六个默认 layout 均完成 35k frozen ViT feature cache：
   - `current_rgb_3view`
   - `spatial_panel_3view`
   - `line_only`
   - `line_difference_band`
   - `fft_absolute_energy`
   - `top3fold_period_layout`
3. 每个 layout 覆盖四个 frozen sample set：
   - `round2_train_small=20000`
   - `round2_selection_small=5000`
   - `round2_diagnostic_balanced_small=5000`
   - `round2_test_small=5000`
4. prediction subset SQLite 已完成：
   - `prediction_index_round2c_35k.sqlite`
   - 35,000 个 sample_key
   - 175,000 条 `sample_key + model_name` 记录
5. 18 个 `layout × seed` 固定 FiLM 训练任务全部完成：
   - seeds: 16, 17, 18
   - epochs: 3
   - backend 固定为 `film_mean_patch_aux` 风格
6. 正式 aggregation 已完成并写出目标要求的 16 个核心输出文件。

## 验收结果

已用 quito 环境运行完整验收脚本，结果为：

```text
Round2c verification passed
best_layout spatial_panel_3view
```

验收覆盖：

- 所有必需 CSV/JSON/Markdown 输出存在；
- feature manifest 中六个 layout、四个 sample set 的样本计数正确；
- seed results 覆盖六个 layout、三 seeds、selection/diagnostic/test_small，且包含 hard top-1 与 raw-soft fusion；
- metadata 明确 only variable 是 layout，base visual input 为 `mean_patch_embedding`，condition input 为 `revin_aux`，使用 FiLM、不 concat aux、不用 test_small 做选择；
- comparison 覆盖六个 Round2 layout、Round1 `film_mean_patch_aux`、Round1 `visual_cls_mean_concat`、Round0 TimeFuse、oracle_top1 和 global_best_single；
- delta summary 覆盖目标要求的核心 pairs；
- 轻量 summary 已复制到 `experiment_summaries/visual_router_v2_round2/layout_screening/`。

## 正式结论

- best layout：`spatial_panel_3view`
- selection raw-soft MAE/MSE：0.310385 / 3.329199
- frozen test_small raw-soft MAE/MSE：0.398598 / 3.484102
- selection best 与 test_small best 一致
- 推荐进入 65k expanded validation：`spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout`

## 后续建议

1. 先做 period continuity diagnostic，重点检查 `top3fold_period_layout` 和 current period fold 的 hard FFT period selection 连续性。
2. 再基于 `spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout` 跑 65k expanded validation。
3. 65k 阶段继续固定后端为 `film_mean_patch_aux` 风格，不同时搜索 head、loss、dropout、calibration 或 seed selection 规则。
