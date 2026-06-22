# PatchTST + Visual dual-branch 65k 稳健性修补

日志日期：2026-06-22 23:24:02 CST

## 目的

在不重新跑 ViT、不修改 Stage 1 canonical、不扩大 65k 实验规模的前提下，修补 PatchTST + fixed visual embedding 双分支实验的稳健性和解释性问题。

## 背景

af96c53 已完成 65k PatchTST frozen prediction + fixed visual embedding 双分支实验，但四个轻量变体均未在 MAE 上超过 PatchTST，并且当前 `h_ts_source=flattened_y_patchtst`，不是 PatchTST encoder hidden representation。因此需要先让该实验入口具备更稳健的训练/评估口径，再把文档中的解释边界写清楚。

## 操作

1. 修改 `visual_router_experiments/dual_branch_fusion/train_patchtst_visual_65k.py`：
   - 新增 train-only feature standardization，`h_ts` 和 `h_vis` 的 scaler 只在 train split 上拟合，val/test 只执行 transform；
   - `config.json` 记录 `feature_standardization.enabled`、fit split 和标准化后 train mean 检查摘要；
   - 每个 epoch 后按 `val_loss` 保存 best state dict，test 前回载 best validation checkpoint；
   - `metrics.json`、`config.json`、`training_log.txt` 和单 run `summary.md` 记录 `test_checkpoint=best_validation_checkpoint`、`best_val_epoch` 和 `best_val_loss`。
2. 修改 `visual_router_experiments/dual_branch_fusion/fusion_heads.py`：
   - 新增 `patchtst_residual_visual` mode；
   - 使用 `y_fusion = y_patchtst + residual_scale * delta_y`；
   - delta head 最后一层 zero-init，默认 `residual_scale=0.1`，使初始预测严格退化为 PatchTST baseline。
3. 修改 `visual_router_experiments/dual_branch_fusion/summarize_results.py`，在汇总 Markdown 中说明新版单 run test 指标使用 best validation checkpoint，并保留 mean/std、`beats_patchtst_mae_rate`、`beats_patchtst_mse_rate` 汇总口径。
4. 修改 `docs/experiments/patchtst_visual_dual_branch_65k.md`：
   - 补充 train-only scaler、best-val checkpoint、新 residual-safe mode 和多 seed 命令；
   - 明确 af96c53 结果的解释边界：当前 `h_ts=flattened_y_patchtst` 只能说明 prediction-level/fallback-level 融合效果，不能代表真实 PatchTST hidden 融合结论；
   - 说明后续若能导出真实 PatchTST encoder hidden，应按同一框架重新运行。
5. 修改 `tests/smoke/dual_branch_fusion_patchtst_65k_smoke.py`，把 smoke 从四个原始 mode 扩展到五个 mode，并检查新增 config/metrics/summary 字段。
6. 同步更新 `WORKSPACE_STRUCTURE.md` 中 dual branch 代码目录和 65k 输出目录说明。

## 结果

语法检查通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
  visual_router_experiments/dual_branch_fusion/fusion_heads.py \
  visual_router_experiments/dual_branch_fusion/train_patchtst_visual_65k.py \
  visual_router_experiments/dual_branch_fusion/summarize_results.py \
  tests/smoke/dual_branch_fusion_patchtst_65k_smoke.py
```

synthetic smoke 通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/dual_branch_fusion_patchtst_65k_smoke.py
```

smoke 覆盖 `feature_concat`、`film`、`residual_feature`、`visual_residual` 和 `patchtst_residual_visual`，均完成 forward、2 epoch mini train、metrics 写出、prediction shape 校验和 summary 字段校验。新增 `patchtst_residual_visual` 在 synthetic test 上 PatchTST MAE 为 `0.05268485`，dual MAE 为 `0.05368329`，相比其他非 residual-safe mode 更接近 PatchTST baseline，符合保守残差设计预期。

## 结论

本次修补完成了 65k dual-branch 实验入口的三个稳健性要求：test 使用 best validation checkpoint、特征标准化不泄漏 val/test、新增 residual-safe PatchTST 残差模式。同时文档已明确 af96c53 的结果不能解释真实 PatchTST hidden 融合，只能解释基于 frozen prediction fallback 的轻量融合。

## 下一步方案

1. 若继续跑 65k，应使用 seeds `1,2,3` 或 `16,17,18`，并把 `patchtst_residual_visual` 纳入同表汇总。
2. 汇总时重点看 `dual_branch_summary.csv/json/md` 中的 mean/std、`beats_patchtst_mae_rate` 和 `beats_patchtst_mse_rate`。
3. 如果后续能导出真实 PatchTST encoder hidden，应复用当前 train-only scaler、best-val checkpoint 和 residual-safe mode 重新运行同一框架。
