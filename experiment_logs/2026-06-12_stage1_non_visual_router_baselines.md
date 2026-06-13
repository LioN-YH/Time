# Stage 1 非视觉 Router Baseline 评估

日志日期：2026-06-12 20:45:24 CST

## 目的

在训练任何 visual router 之前，先用 `vali` split 上可学习的非视觉统计规则建立参照，回答“不使用视觉输入，只靠全局、dataset、TSF cell 等元信息规则，在 `test` split 上能达到什么误差水平”。

## 背景

前序步骤已经在 `experiment_logs/run_outputs/2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot/` 中生成扩大版五专家 window-level prediction cache，并通过 `enrich_cache_with_tsf_cell.py` 合并了 TSF cell 元信息。可用输入文件为：

- `experiment_logs/run_outputs/2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot/window_oracle_labels_with_tsf_cell.csv`

该文件包含 `vali/test` split、`dataset_name`、`item_id`、`channel_id`、`window_index`、`cluster`、`group_name`、五专家 MAE/MSE、`oracle_model` 和 `oracle_value`，足以评估不依赖视觉输入的 routing baseline。

## 操作

1. 检查 `visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py`，确认脚本已覆盖以下规则：
   - `global_best_single`：在全部 `vali` window 上按平均 MAE 选单一专家；
   - `dataset_only`：每个 `dataset_name` 在 `vali` 上选最佳专家；
   - `tsf_cell_only`：每个 `group_name` 在 `vali` 上选最佳专家；
   - `dataset_tsf_cell`：每个 `(dataset_name, group_name)` 在 `vali` 上选最佳专家；
   - `oracle_top1`：在 `test` 上逐窗口事后选择最佳专家，只作为上限；
   - `global_majority_label`：在 `vali` 上按 oracle label 胜出次数最多选单一专家。
2. 确认脚本额外输出了 `dataset_majority_label`、`tsf_cell_majority_label` 和 `dataset_tsf_cell_majority_label`，用于对比同样分组粒度下“胜出次数最多”和“平均 MAE 最低”两种口径。
3. 重新运行 baseline 评估命令：

   ```bash
   python visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py \
     --labels-path experiment_logs/run_outputs/2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot/window_oracle_labels_with_tsf_cell.csv \
     --metric mae
   ```

4. 校验必需输出文件存在：
   - `baseline_summary.csv`
   - `baseline_predictions.csv`
   - `baseline_summary_by_dataset.csv`
   - `baseline_summary_by_tsf_cell.csv`
   - `summary.md`
5. 额外确认 `baseline_summary_by_dataset_tsf_cell.csv` 也已生成，作为 dataset + TSF cell 细分层级的补充汇总。

## 结果

脚本成功写入输出目录：

- `experiment_logs/run_outputs/2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot/`

复现后的关键校验结果：

- 必需输出文件无缺失；
- `baseline_summary.csv` 包含 9 条 baseline 汇总记录；
- `baseline_predictions.csv` 包含 540 条逐窗口 baseline 选择记录；
- `global_best_single` 对应 60 个唯一 test window；
- 用户要求的 baseline 集合均已包含在整体汇总中。

整体 test MAE 口径结果如下：

| baseline | test window 数 | test MAE | oracle MAE | 相对 global best single 改善 |
| --- | ---: | ---: | ---: | ---: |
| `global_best_single` | 60 | 1.055190 | 0.805392 | 0.000000 |
| `dataset_only` | 60 | 1.367750 | 0.805392 | -0.296212 |
| `tsf_cell_only` | 60 | 1.401091 | 0.805392 | -0.327809 |
| `dataset_tsf_cell` | 60 | 1.401091 | 0.805392 | -0.327809 |
| `global_majority_label` | 60 | 1.488905 | 0.805392 | -0.411030 |
| `oracle_top1` | 60 | 0.805392 | 0.805392 | 0.236733 |

当前 pilot 上，可部署的非视觉规则中 `global_best_single` 最好；dataset/TSF-cell shortcut 没有超过全局单专家，反而带来更高 MAE。`oracle_top1` 相对 `global_best_single` 仍有约 23.67% 的 MAE 改善空间，说明窗口级专家互补性仍存在，但不能被这些简单元信息规则捕获。

## 结论

Stage 1 mixed visual router 的后续结果至少需要与 `global_best_single` 的 test MAE 1.055190 对齐比较；如果视觉 router 只能学到 dataset 或 TSF-cell shortcut，则在当前 pilot 上可能比全局单专家更差。`oracle_top1` 的 0.805392 MAE 是当前窗口级 hard top-1 routing 的事后上限。

## 下一步方案

1. 基于同一 prediction cache 构造伪图像张量或结构特征，先做轻量 embedding cache。
2. 训练 Stage 1 visual router 时，把 `global_best_single`、`dataset_only`、`tsf_cell_only`、`dataset_tsf_cell` 和 `oracle_top1` 同表报告。
3. 扩大 pilot 覆盖范围后，重新运行 `evaluate_router_baselines.py`，确认 dataset/TSF-cell shortcut 在更大样本上是否仍然弱于全局单专家。
