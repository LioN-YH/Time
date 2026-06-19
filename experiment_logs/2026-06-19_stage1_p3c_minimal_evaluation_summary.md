# Stage 1 P3c minimal evaluation summary builder

日志日期：2026-06-19 18:24:40 CST

## 目的

在已完成 P3a/P3b 的 `time_router/evaluation/metrics.py` 基础上，小步抽取最小 evaluation summary helper，用于把 hard top-1、raw soft fusion、selected counts、weight entropy 和 max weight 汇总成稳定 summary dict。

## 背景

Stage 1 重构路线要求先在 golden smoke 中锁定行为，再逐步下沉公共 helper。本次 P3c 只处理 summary builder，不迁移 Visual Router 或 TimeFuse-style fusor 正式训练入口，不改变正式 summary / comparison / prediction output schema，也不实现 calibration、oracle regret 或 oracle/TSF 读取。

## 操作

1. 新增 `time_router/evaluation/summary.py`，实现 `build_fusion_summary(...)`。
2. 在 `time_router/evaluation/__init__.py` 导出 `build_fusion_summary`。
3. 在 `tests/smoke/stage1_golden_smoke.py` 中基于 `GOLDEN_WEIGHTS`、`hard_result` 和 `raw_soft_result` 增加 summary 断言。
4. 更新 `docs/refactor/stage1_refactor_roadmap.md`，补充 P3c 当前状态、完成范围和明确不做事项。
5. 更新 `WORKSPACE_STRUCTURE.md`，记录新增 `time_router/evaluation/summary.py` 以及 golden smoke 的 P3c summary 覆盖范围。
6. 使用 Quito 环境运行验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

## 结果

- `build_fusion_summary(...)` 只消费显式传入的 `FusionMetricsResult`、`weights` 和 `model_columns`。
- summary dict 包含 `hard_mae`、`hard_mse`、`raw_soft_mae`、`raw_soft_mse`、`selected_counts`、`mean_entropy`、`mean_max_weight`、`num_samples`、`num_experts` 和 `model_columns`。
- golden smoke 复算结果保持不变：
  - hard top-1 MAE = `0.41604843735694885`
  - hard top-1 MSE = `0.4563697576522827`
  - raw soft MAE = `0.4102966785430908`
  - raw soft MSE = `0.48815402388572693`
  - selected counts = `{'DLinear': 2, 'PatchTST': 1, 'CrossFormer': 1, 'ES': 0, 'NaiveForecaster': 0}`
  - mean entropy = `1.217490315`
  - mean max weight = `0.550000012`
- 三条验收命令均通过。

## 结论

P3c minimal evaluation summary builder 已完成。该 helper 保持纯 numpy / Python 标准库边界，不读取 manifest、prediction cache、oracle/TSF 或正式输出目录；本次未迁移正式训练入口，未实现 temperature/top-k/calibration，未改变正式 output schema，未改 reader、模型结构、loss 或正式输出目录。

## 下一步方案

后续如继续 P3，应另起小步处理 comparison、per-sample prediction schema 或 calibration；正式 Visual Router / TimeFuse-style fusor 入口迁移仍保留到 P6，并继续使用 golden smoke、oracle/TSF smoke 和 compileall 作为门禁。
