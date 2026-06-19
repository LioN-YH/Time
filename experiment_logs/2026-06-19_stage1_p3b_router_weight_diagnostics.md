# Stage 1 P3b router weight diagnostics helper 抽取

日志日期：2026-06-19 18:15:56 CST

## 目的

在已完成 P3a 最小 fusion/metrics helper 的基础上，继续小步抽取 router/fusor weights 的纯 numpy 诊断 helper，只覆盖 selected counts、weight entropy 和 max weight。

## 背景

Stage 1 后续需要逐步收束 Visual Router 与 TimeFuse-style fusor 的共享评估逻辑。P3a 已经完成 MAE/MSE、hard top-1 fusion 和 raw soft fusion 的最小共享 helper。本次 P3b 只补充权重诊断，不迁移正式训练入口，不改变输出 schema，也不接入 oracle/TSF 或 prediction reader。

## 操作

1. 在 `time_router/evaluation/metrics.py` 中新增纯 numpy helper：
   - `compute_selected_counts(selected_indices, model_columns)`
   - `compute_weight_entropy(weights, eps=1e-12)`
   - `compute_max_weight(weights)`
2. 为 P3b helper 增加必要输入校验：
   - `weights` 必须是二维 `[sample, expert]`；
   - `selected_indices` 必须是一维 `[sample]`；
   - `model_columns` 不能为空且不允许重复；
   - `selected_indices` 不能小于 0 或大于等于专家数；
   - 权重必须有限且非负，避免 entropy/max-weight 诊断产生无效结果。
3. 更新 `time_router/evaluation/__init__.py`，导出 P3b helper。
4. 更新 `tests/smoke/stage1_golden_smoke.py`，基于 `GOLDEN_WEIGHTS` 的 hard top-1 `selected_indices` 复算 selected counts，并验证 entropy/max_weight shape 和 max_weight 数值。
5. 更新 `docs/refactor/stage1_refactor_roadmap.md`，追加 P3b 当前状态、完成范围和明确不做事项。
6. 更新 `WORKSPACE_STRUCTURE.md`，记录 P3b diagnostics helper 和 golden smoke 覆盖范围。

## 结果

验收命令均已在 `quito` conda 环境下通过：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

`stage1_golden_smoke.py` 保持原 hard top-1 和 raw soft fusion golden 数值不变：

- hard top-1 MAE=`0.416048437`，MSE=`0.456369758`
- raw soft fusion MAE=`0.410296679`，MSE=`0.488154024`
- selected counts=`{'DLinear': 2, 'PatchTST': 1, 'CrossFormer': 1, 'ES': 0, 'NaiveForecaster': 0}`

## 结论

P3b router weight diagnostics extraction 已完成。新增 helper 只接受显式数组和专家列名输入，不读取 manifest、prediction cache、oracle/TSF 或正式训练输出目录；本次没有迁移 Visual Router / TimeFuse fusor 正式训练入口，没有实现 calibration/top-k/temperature，没有改变 summary/comparison/prediction output schema，也没有修改模型结构、loss 或正式输出目录。

## 下一步方案

后续如果继续推进 P3，应另起小步处理 summary/comparison/per-sample schema 或 calibration 相关逻辑；正式 Visual Router / TimeFuse fusor 入口迁移仍保留到 P6，并继续以 golden smoke、oracle/TSF smoke 和 compileall 作为门禁。
