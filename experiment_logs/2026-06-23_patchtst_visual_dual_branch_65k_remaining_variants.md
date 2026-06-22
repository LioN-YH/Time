# PatchTST + Visual dual-branch 65k 剩余变体与完整大表

日志日期：2026-06-23 00:35:01 CST

## 目的

补齐 Visual Router V2 Round2 探索分支中 PatchTST frozen prediction + fixed visual embedding 双分支路线的剩余轻量变体，并合并 `1b20e72` 已完成的 robust multiseed 结果，生成完整比较大表。该步骤的目标是完整评估当前 fallback-level 融合路线，而不是证明视觉分支一定稳定优于 PatchTST。

## 背景

- 当前分支：`exp/visual-router-v2-round2-exploration`。
- 当前基准提交：`1b20e72 exp: run robust 65k PatchTST visual dual-branch multi-seed`；本次在该提交之后有未提交代码改动。
- 复用历史结果路径：`/data2/syh/Time/run_outputs/2026-06-22_patchtst_visual_dual_branch_65k_robust_multiseed/patchtst_visual/spatial_panel_3view/`。
- 本轮输出根目录：`/data2/syh/Time/run_outputs/2026-06-23_patchtst_visual_dual_branch_65k_remaining_variants/`。
- PatchTST cache：`/data2/syh/Time/run_outputs/2026-06-22_patchtst_visual_dual_branch_65k/inputs/patchtst_frozen_cache_from_round2_expanded.npz`。
- visual embedding cache：`/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation/features/spatial_panel_3view`。
- 数据和 split 继承 robust multiseed：`round2_train_expanded`、`round2_selection_expanded`、`round2_test_expanded`，总对齐样本为 65,000。
- 当前 `h_ts` 仍为 `flattened_y_patchtst` fallback，不是真实 PatchTST encoder hidden。

## 操作

1. 扩展 `visual_router_experiments/dual_branch_fusion/fusion_heads.py`：
   - 保留已有 `pred_gate`；
   - 新增 `feature_gate`：`z_ts/z_vis` 投影到同一隐空间后逐维门控融合；
   - 新增 `gated_residual_feature`：视觉 residual 通过 gate 和 `residual_scale` 保守修正 `h_ts`，再接 feature-level prediction head；
   - 在注释中明确 `gated_residual_feature` 不是 prediction-level residual，不保证初始等于 PatchTST。
2. 扩展 `train_patchtst_visual_65k.py` 的 `--fusion_mode` choices，允许训练新增两个 mode。
3. 新增 `run_patchtst_visual_65k_remaining_variants.sh`，用于调度：
   - `pred_gate`、`feature_gate`、`gated_residual_feature` × seeds `1/2/3`；
   - `patchtst_residual_visual` residual_scale sweep：`0.01/0.03/0.05/0.1` × seeds `1/2/3`。
4. 新增 `summarize_full_comparison.py`：
   - 输出本轮 `remaining_run_metrics.csv` 和 `remaining_summary.csv`；
   - 合并历史 5 个 mode 与本轮新增/scale sweep；
   - 生成 `full_dual_branch_comparison.csv/json/md` 和 `full_dual_branch_ranking.md`；
   - scale sweep 以 `patchtst_residual_visual_scale0p01` 等独立 method 进入大表。
5. 使用 quito 环境做语法检查：
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile ...`
   - `bash -n visual_router_experiments/dual_branch_fusion/run_patchtst_visual_65k_remaining_variants.sh`
6. 执行本轮 21 个 run。调试性前台命令为：

   ```bash
   OUTPUT_ROOT=/data2/syh/Time/run_outputs/2026-06-23_patchtst_visual_dual_branch_65k_remaining_variants MAX_PARALLEL=1 bash -x visual_router_experiments/dual_branch_fusion/run_patchtst_visual_65k_remaining_variants.sh
   ```

   实际完成时间约为 `2026-06-23 00:34:28 CST`，`training_failures=0`，`summary_rc=0`。

## 结果

本轮新增 run 全部完成：

- `pred_gate` seeds `1/2/3`；
- `feature_gate` seeds `1/2/3`；
- `gated_residual_feature` seeds `1/2/3`；
- `patchtst_residual_visual` residual_scale `0.01/0.03/0.05/0.1` × seeds `1/2/3`。

所有新增 run 均使用：

- train-only feature standardization：是；
- test checkpoint：best validation checkpoint；
- epochs：20；
- batch_size：256；
- lr：`1e-3`；
- hidden_dim：256；
- dropout：0.1；
- 默认 residual_scale：0.1，scale sweep 除外。

验收检查：

- 新增 21 个 run 目录均有 `config.json`、`metrics.json`、`predictions.npz`、`training_log.txt`、`summary.md`，共 105 个必需单 run 产物。
- 汇总目录 `/data2/syh/Time/run_outputs/2026-06-23_patchtst_visual_dual_branch_65k_remaining_variants/patchtst_visual/spatial_panel_3view/summary/` 已生成：
  - `remaining_run_metrics.csv`
  - `remaining_summary.csv`
  - `full_dual_branch_comparison.csv`
  - `full_dual_branch_comparison.json`
  - `full_dual_branch_comparison.md`
  - `full_dual_branch_ranking.md`

完整大表核心结果如下，delta 定义为 PatchTST 指标减去 dual-branch 指标，正数表示双分支更好：

| method | delta_mae_vs_patchtst_mean | delta_mse_vs_patchtst_mean | beats_patchtst_mae_rate | beats_patchtst_mse_rate |
| --- | ---: | ---: | ---: | ---: |
| `pred_gate` | 0.01929374 | 0.07061227 | 1.00000000 | 1.00000000 |
| `patchtst_residual_visual_scale0p01` | 0.01238557 | 0.09501719 | 1.00000000 | 1.00000000 |
| `patchtst_residual_visual_scale0p03` | 0.00888681 | 0.08889906 | 1.00000000 | 1.00000000 |
| `feature_gate` | 0.00352377 | -0.00204198 | 1.00000000 | 0.66666667 |
| `patchtst_residual_visual_scale0p05` | 0.00271306 | 0.09043733 | 0.66666667 | 1.00000000 |
| `gated_residual_feature` | 0.00080566 | 0.05652316 | 0.66666667 | 1.00000000 |

历史 robust multiseed 5 个 mode 合并后仍保留在 full comparison 中；其中历史默认 `patchtst_residual_visual` 与本轮 `scale0p1` 指标一致，但在大表中分别保留为历史默认方法和 scale sweep 方法，便于追踪来源。

## 结论

1. 本轮有方法在 MAE 上超过 PatchTST：`pred_gate`、`patchtst_residual_visual_scale0p01`、`patchtst_residual_visual_scale0p03`、`feature_gate`、`patchtst_residual_visual_scale0p05`、`gated_residual_feature` 的 mean delta 均为正，其中 `pred_gate` 最好，`delta_mae_vs_patchtst_mean=0.01929374`。
2. 本轮有方法在 MSE 上超过 PatchTST：`pred_gate`、`patchtst_residual_visual_scale0p01/0p03/0p05/0p1`、`gated_residual_feature` 等为正，其中 `patchtst_residual_visual_scale0p01` 最好，`delta_mse_vs_patchtst_mean=0.09501719`。
3. 小 residual scale 比默认 `0.1` 更适合 `patchtst_residual_visual`：`0.01` 和 `0.03` 同时取得 3/3 seeds 的 MAE 与 MSE 改善，而默认 `0.1` 仅稳定改善 MSE、MAE 退化。
4. `pred_gate` 在当前 fallback-level 设置下取得最强 MAE 改善，说明 fixed visual embedding + PatchTST frozen prediction fallback 并非完全无效。
5. 但当前 `h_ts` 仍是 `flattened_y_patchtst` fallback，本轮结果只能说明 prediction-level/fallback-level 轻量融合有效，不能替代真实 PatchTST encoder hidden + visual embedding 的融合实验。

## 下一步方案

1. 不建议继续把主要资源投入更复杂的 cross-attention 或伪 token attention，除非已有真实 token-level `H_ts/H_vis` cache。
2. 可保留 `pred_gate` 与 `patchtst_residual_visual_scale0p01/0p03` 作为 fallback-level 强对照，用于论文或中期报告中的消融表。
3. 后续优先级应转向：
   - 导出真实 PatchTST encoder hidden，复用同一 split 和 fixed visual embedding 重新验证；
   - 对 `pred_gate` 和小 residual scale 做 error-tail / strata 分析，确认增益来自哪些数据段；
   - 比较视觉编码替换或更强 visual representation，而不是继续堆叠复杂融合头。
4. 不能把本轮正结果写成视觉分支已被充分证明；应表述为“在 frozen PatchTST prediction fallback 设置下，轻量 gating/residual 变体已观察到 3 seed 平均 MAE/MSE 改善”。
