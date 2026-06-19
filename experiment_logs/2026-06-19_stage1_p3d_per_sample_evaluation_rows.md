# Stage 1 P3d minimal per-sample evaluation rows builder

日志日期：2026-06-19 18:47:30 CST

## 目的

在已完成 P3a/P3b/P3c 的 `time_router/evaluation` helper 基础上，小步抽取最小 per-sample evaluation rows builder，用于把当前 batch 的 sample_key、hard top-1 选择、hard/raw-soft 逐样本指标和 router weight diagnostics 组合成稳定逐样本 rows。

## 背景

Stage 1 重构路线要求继续先在 golden smoke 中锁定行为，再逐步抽取公共 helper。本次 P3d 只处理逐样本 rows builder，不迁移 Visual Router 或 TimeFuse-style fusor 正式训练入口，不改变正式 summary / comparison / prediction output schema，不写正式输出目录，不实现 calibration、oracle regret 或 oracle/TSF 读取。

## 操作

1. 新增 `time_router/evaluation/prediction_rows.py`，实现 `build_per_sample_fusion_rows(...)`。
2. 在 `time_router/evaluation/__init__.py` 导出 `build_per_sample_fusion_rows`。
3. 在 `tests/smoke/stage1_golden_smoke.py` 中基于 `EXPECTED_SAMPLE_KEYS`、`GOLDEN_WEIGHTS`、`hard_result`、`raw_soft_result` 和 `y_true` 增加 per-sample rows 断言。
4. 更新 `docs/refactor/stage1_refactor_roadmap.md`，补充 P3d 当前状态、完成范围和明确不做事项。
5. 更新 `WORKSPACE_STRUCTURE.md`，记录新增 `time_router/evaluation/prediction_rows.py` 以及 golden smoke 的 P3d rows 覆盖范围。
6. 使用 Quito 环境运行验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

## 结果

- `build_per_sample_fusion_rows(...)` 只消费显式传入的 `sample_keys`、`FusionMetricsResult`、`y_true`、`weights` 和 `model_columns`。
- 每个 row 包含 `sample_key`、`selected_model`、`selected_index`、`hard_mae`、`hard_mse`、`raw_soft_mae`、`raw_soft_mse`、`max_weight` 和 `weight_entropy`。
- 输入校验覆盖 sample_key 数量、hard/raw-soft fused_pred 与 y_true shape、hard selected_indices、raw_soft 不含 selected 信息、weights `[sample, expert]` shape、model_columns 长度和重复专家名。
- golden smoke 复算结果保持不变，并新增 rows 检查：
  - rows 数量 = `4`
  - selected_model 顺序 = `['CrossFormer', 'DLinear', 'PatchTST', 'DLinear']`
  - max_weight 与 `np.max(GOLDEN_WEIGHTS, axis=1)` 一致
  - weight_entropy 与 `compute_weight_entropy(GOLDEN_WEIGHTS)` 一致
- 三条验收命令均通过。

## 结论

P3d minimal per-sample evaluation rows builder 已完成。该 helper 保持纯 numpy / Python 标准库边界，不读取 manifest、prediction cache、oracle/TSF 或正式输出目录，不写 CSV/JSON/Parquet；本次未迁移正式训练入口，未实现 temperature/top-k/calibration，未改变正式 output schema，未改 reader、模型结构、loss 或正式输出目录。

## 下一步方案

后续如继续 P3，应另起小步处理 comparison、正式 per-sample prediction schema 或 calibration；正式 Visual Router / TimeFuse-style fusor 入口迁移仍保留到 P6，并继续使用 golden smoke、oracle/TSF smoke 和 compileall 作为门禁。
