# Visual Router V2 Round2 65k visual pooling ablation

日志日期：2026-06-23 03:31:28 CST

## 目的

在已冻结的 Round2 65k expanded samples 上，固定 layout 为 `spatial_panel_3view`，固定训练/评估协议和 FiLM aux 注入方式，只比较视觉特征池化输入方式：`cls`、`mean_patch`、`cls_mean_concat`。本轮目标是补齐 65k 规模下的 pooling ablation，并验证是否继续支持 `film_mean_patch_aux` 作为 Round2 主线后端。

## 背景

Round2e-b 65k expanded layout validation 已证明 `spatial_panel_3view` 是 best layout，Round2f P0 spatial panel mainline 也支持 `spatial_panel_3view + film_mean_patch_aux`。但 65k expanded samples 上尚未独立比较 `cls_embedding`、`mean_patch_embedding` 和 `[cls_embedding; mean_patch_embedding]` 三种视觉输入。已有 `spatial_panel_3view` feature cache 中包含 `cls_embedding`、`mean_patch_embedding` 和 `revin_aux`，因此本轮不重新生成 samples、不重跑 ViT、不保存 pseudo image tensor。

## 操作

1. 扩展 `visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round2_layout_film.py`：
   - 新增 `--visual-input-mode` / `--visual-input-modes`，支持 `cls`、`mean_patch`、`cls_mean_concat`。
   - 新增 `--feature-artifact-prefix`，使 pooling 输出前缀 `round2_65k_pooling` 可以复用既有 `round2_expanded_layout_feature_manifest.csv`。
   - feature 读取按 mode 选择 `cls_embedding`、`mean_patch_embedding` 或二者按 `[cls; mean_patch]` 拼接，并记录 `visual_dim`。
   - 聚合阶段输出 `best_variant`、pooling 专用 metadata、delta summary 和中文 summary。
2. 新增 `visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_pooling_ablation_parallel.py`：
   - 只调度 prediction subset index 复用/构建、variant×seed 训练评估和单进程聚合。
   - 不调用 feature builder，不重新跑 ViT。
3. 使用 `quito` 环境执行 `py_compile`，通过新增/修改脚本编译检查。
4. 执行 smoke：
   - 输出目录：`/data2/syh/Time/run_outputs/2026-06-23_visual_router_v2_round2_65k_pooling_ablation_smoke/`
   - 参数：3 variants × seed16 × 1 epoch，每个 sample_set 截断 64 条。
   - 首次 smoke 构建 256 sample_key 的 prediction subset SQLite，覆盖 1280 条专家记录；修复了 method_rows 中 variant 仍写 layout 名称的问题。
   - 修复后重跑 smoke，确认三个 variant 均进入 summary，`cls_mean_concat` 的 `visual_dim=1536`。
5. 正式运行：
   - 输出目录：`/data2/syh/Time/run_outputs/2026-06-23_visual_router_v2_round2_65k_pooling_ablation/`
   - 复用 Round2e-b 的 65k prediction subset SQLite，核验 `prediction_index` 有 325000 条记录、65000 个 sample_key。
   - 运行 `cls,mean_patch,cls_mean_concat` × seeds 16/17/18 × 3 epochs，4 张 RTX 3090 进程级并行，9/9 task completed。
   - 运行 aggregate-only 重新生成 pooling 专用 summary，并同步轻量 summary 到 `experiment_summaries/visual_router_v2_round2/65k_pooling_ablation/`。

## 结果

正式输出 `status.json` 为 `completed`，metadata 记录：

- `round2_stage = 65k_pooling_ablation`
- `layout_fixed_to = spatial_panel_3view`
- `compared_visual_input_modes = ["cls", "mean_patch", "cls_mean_concat"]`
- `condition_input = revin_aux`
- `used_film = true`
- `used_concat_aux = false`
- `used_test_for_selection = false`
- `loaded_116m_prediction_manifest_to_memory = false`
- `saved_pseudo_image_tensor = false`
- `reused_existing_feature_cache = true`

关键 raw-soft 结果：

| sample_set | best variant | MAE_mean | MSE_mean | 备注 |
| --- | --- | ---: | ---: | --- |
| `round2_selection_expanded` | `film_mean_patch_aux` | 0.307233 | 2.043914 | selection 只按 raw-soft MAE mean 选择 |
| `round2_test_expanded` | `film_mean_patch_aux` | 0.394336 | 2.008546 | frozen final eval，与 selection best 一致 |

selection raw-soft 对比：

- `film_mean_patch_aux`：MAE=0.307233，MSE=2.043914，MAE_std=0.001518。
- `film_cls_mean_concat_aux`：MAE=0.308029，MSE=2.269129，MAE_std=0.000988。
- `film_cls_aux`：MAE=0.308595，MSE=2.046413，MAE_std=0.000715。

主要结论：

- `cls` 弱于 `mean_patch`：selection raw-soft MAE delta(cls-mean_patch)=+0.001362，MSE delta=+0.002500。
- `cls_mean_concat` 优于单独 `cls` 的 MAE，但未优于 `mean_patch`；相对 `mean_patch` MAE 高 +0.000795，MSE 高 +0.225216。
- 增加到 1536 维的 concat 输入没有带来更好的 MAE/MSE，且 selection MSE 明显高于 mean patch anchor。
- seed stability 上，selection raw-soft MAE_std 最低为 `film_cls_aux`=0.000715，但 best overall 仍是 `film_mean_patch_aux`。
- MSE tail 上，selection 与 frozen test 的 raw-soft MSE 最低均为 `film_mean_patch_aux`。
- selection raw-soft selected_model ratio 未出现单专家塌缩：各 variant 最高平均 selected_model ratio 约为 `film_cls_aux` DLinear=0.433、`film_cls_mean_concat_aux` DLinear=0.507、`film_mean_patch_aux` DLinear=0.414。
- diagnostic oracle_model 分层中，CrossFormer/DLinear/ES/NaiveForecaster strata 的 best 多为 concat，但 PatchTST strata best 为 `film_mean_patch_aux`；overall selection/test 仍以 `film_mean_patch_aux` 为准。

## 结论

本轮 65k pooling ablation 支持继续使用 `film_mean_patch_aux` 作为 Round2 主线后端。`cls_embedding` 单独输入整体弱于 `mean_patch_embedding`；`cls_mean_concat` 虽略优于 cls 的 MAE，但没有超过 mean patch anchor，并带来更高 selection MSE，因此不建议为了 concat 维度扩张替换主线后端。

## 下一步方案

1. Round2 主线继续采用 `spatial_panel_3view + film_mean_patch_aux`。
2. `film_cls_mean_concat_aux` 可保留为分层诊断参考，但不作为主线后端。
3. 后续 full-scale/staged validation 应继续报告 pooling 后端、selected_model ratio、MSE tail 和 oracle_model strata，避免把局部分层收益误读为 overall 主线收益。
