# PatchTST + Visual dual-branch 65k 多 seed 稳健复跑

日志日期：2026-06-23 00:03:43 CST

## 目的

基于 `7f9f840` 之后的稳健训练入口，复跑 65k expanded sample set 上的 PatchTST frozen prediction + `spatial_panel_3view` fixed visual embedding 双分支融合实验，获得 PatchTST baseline 与 PatchTST+Visual 双分支的多 seed MAE/MSE 对比。

## 背景

本任务属于 Visual Router V2 Round2 探索分支，不属于 Stage 1 canonical 重构主线。前序提交 `af96c53` 完成 seed1 首版 65k 双分支实验，`7f9f840` 修补了 train-only feature standardization、best validation checkpoint 和 `patchtst_residual_visual` residual-safe mode。本轮不修改模型结构，只复用既有 cache 执行 5 个 fusion mode × 3 个 seeds 的稳健复跑。

当前解释边界必须保持清楚：`h_ts` 仍是 `flattened_y_patchtst` fallback，不是真实 PatchTST encoder hidden。本轮只能说明 fixed visual embedding 与 PatchTST frozen prediction fallback 的轻量融合效果，不能据此否定真实 PatchTST encoder hidden 与视觉 embedding 融合的可能性。

## 操作

1. 确认当前分支和提交：branch 为 `exp/visual-router-v2-round2-exploration`，commit 为 `7f9f840`。
2. PatchTST frozen cache 使用 `/data2/syh/Time/run_outputs/2026-06-22_patchtst_visual_dual_branch_65k/inputs/patchtst_frozen_cache_from_round2_expanded.npz`。
3. fixed visual embedding cache 使用 `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation/features/spatial_panel_3view`。
4. output root 使用 `/data2/syh/Time/run_outputs/2026-06-22_patchtst_visual_dual_branch_65k_robust_multiseed/`，避免覆盖 `af96c53` 历史结果。
5. 新增调度脚本 `visual_router_experiments/dual_branch_fusion/run_patchtst_visual_65k_robust_multiseed.sh`，仅调用既有训练入口，不生成图像、不运行 ViT、不训练 PatchTST、不修改 Stage 1 canonical 文件。
6. 初次普通后台启动时，训练子进程被当前执行会话回收，单 run stdout 为空且未进入 GPU；随后改用 `setsid` 脱离会话成功运行。
7. 本轮 seeds 使用任务推荐的 `1,2,3`；fusion modes 为 `feature_concat`、`film`、`residual_feature`、`visual_residual`、`patchtst_residual_visual`。
8. 每个 run 参数为 `epochs=20`、`batch_size=256`、`lr=1e-3`、`hidden_dim=256`、`dropout=0.1`、`residual_scale=0.1`，GPU 并行度为 4。
9. split 为 train=`round2_train_expanded`、val=`round2_selection_expanded`、test=`round2_test_expanded`。
10. 所有 run 默认启用 train-only feature standardization，test 使用 best validation checkpoint。
11. 训练完成后运行 `summarize_results.py`，输出到 `patchtst_visual/spatial_panel_3view/summary/`。
12. 使用 `quito` conda 环境执行验收脚本，检查 15 个 run 的必需文件、config/metrics 字段、split 数量、checkpoint 口径和 `predictions.npz` shape。

## 结果

launcher 从 `2026-06-23 00:01:20 CST` 运行到 `2026-06-23 00:02:33 CST`，15 个 run 全部 `rc=0`，汇总 `summary_rc=0`。每个 run 均包含 `config.json`、`metrics.json`、`predictions.npz`、`training_log.txt` 和 `summary.md`。

验收脚本结果：

- `validated_runs=15`
- `errors=0`
- 每个 run 的 `aligned_sample_count=65000`
- split 样本数为 train/val/test = `30000/10000/15000`
- `feature_standardization.enabled=true`
- `test_checkpoint=best_validation_checkpoint`
- `predictions.npz` 中 `y_patchtst`、`y_fusion`、`y_true` shape 均为 `[15000, 48, 1]`

summary 目录包含 `dual_branch_run_metrics.csv`、`dual_branch_summary.csv`、`dual_branch_summary.json` 和 `dual_branch_summary.md`。

PatchTST baseline 在所有 run 中一致：

- PatchTST MAE：`0.47232839`
- PatchTST MSE：`2.13463616`

多 seed 汇总如下，delta 定义为 PatchTST 指标减去 dual-branch 指标，正数表示双分支更好：

| fusion_mode | runs | mean PatchTST MAE | std PatchTST MAE | mean PatchTST MSE | std PatchTST MSE | mean Dual MAE | std Dual MAE | mean Dual MSE | std Dual MSE | mean delta MAE | std delta MAE | mean delta MSE | std delta MSE | MAE beat rate | MSE beat rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| feature_concat | 3 | 0.47232839 | 0.00000000 | 2.13463616 | 0.00000000 | 0.51214854 | 0.00157591 | 2.18839304 | 0.03876851 | -0.03982014 | 0.00157591 | -0.05375687 | 0.03876851 | 0.00000000 | 0.33333333 |
| film | 3 | 0.47232839 | 0.00000000 | 2.13463616 | 0.00000000 | 0.48310875 | 0.00586938 | 2.15683460 | 0.05592405 | -0.01078035 | 0.00586938 | -0.02219844 | 0.05592405 | 0.00000000 | 0.66666667 |
| patchtst_residual_visual | 3 | 0.47232839 | 0.00000000 | 2.13463616 | 0.00000000 | 0.47826815 | 0.00356270 | 2.04493093 | 0.00157841 | -0.00593975 | 0.00356270 | 0.08970523 | 0.00157841 | 0.00000000 | 1.00000000 |
| residual_feature | 3 | 0.47232839 | 0.00000000 | 2.13463616 | 0.00000000 | 0.49294524 | 0.00452657 | 2.16017278 | 0.05661966 | -0.02061685 | 0.00452657 | -0.02553662 | 0.05661966 | 0.00000000 | 0.33333333 |
| visual_residual | 3 | 0.47232839 | 0.00000000 | 2.13463616 | 0.00000000 | 0.47860580 | 0.00261630 | 2.12113794 | 0.00315994 | -0.00627740 | 0.00261630 | 0.01349823 | 0.00315994 | 0.00000000 | 1.00000000 |

按 MAE 超过 PatchTST 的 mode：无。

按 MSE 超过 PatchTST 的 mode：

- `patchtst_residual_visual`：mean delta MSE = `0.08970523`，3/3 seeds 超过 PatchTST。
- `visual_residual`：mean delta MSE = `0.01349823`，3/3 seeds 超过 PatchTST。

## 结论

在 65k expanded sample set、`spatial_panel_3view` fixed visual embedding、PatchTST frozen prediction cache、`h_ts=flattened_y_patchtst` fallback、train-only feature standardization 和 best validation checkpoint 口径下，没有任何双分支变体在 MAE 上稳定超过 PatchTST baseline。

`patchtst_residual_visual` 和 `visual_residual` 在 MSE 上稳定超过 PatchTST，但 MAE 仍退化，说明它们可能改善了部分平方误差尾部或极端误差，而没有改善整体绝对误差。本轮结果不支持继续直接堆叠更复杂融合结构来追求 MAE。

## 下一步方案

1. 不建议立即继续跑剩余复杂方法或 cross-attention/feature_gate 类方法。
2. 由于只有 MSE 稳定提升，应先做 error-tail、strata、residual-scale 和 loss 口径分析，判断 MSE 改善来自哪些样本、TSF cell、oracle_model 或高误差尾部。
3. 若目标是验证视觉信息是否能帮助 PatchTST，优先导出真实 PatchTST encoder hidden，或更换/改进视觉编码，而不是继续基于 `flattened_y_patchtst` fallback 堆复杂融合方法。
4. 后续所有报告都应继续明确：本轮不能否定真实 PatchTST encoder hidden 与视觉 embedding 融合的可能性。
