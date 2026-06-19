# Stage 1 P3e evaluation package boundary review

日志日期：2026-06-19 19:07:54 CST

## 目的

在已完成 P3a/P3b/P3c/P3d 的基础上，对 `time_router/evaluation` 当前 package 边界做一次架构复核和整理规划，明确 `metrics.py`、`summary.py`、`prediction_rows.py` 和 `__init__.py` 的职责、public API、private helper 与后续 consolidation 原则。

## 背景

当前 P3 已抽取 hard top-1/raw soft fusion、基础 MAE/MSE、router weight diagnostics、minimal summary dict builder 和 per-sample evaluation rows builder。随着 `time_router/evaluation` 下模块变多，需要先明确边界，避免后续继续加入 comparison、calibration、report schema 时出现 import churn 或职责混乱。

本次 P3e 只做文档化 review 和 consolidation plan，不迁移 Visual Router / TimeFuse-style fusor 正式训练入口，不改 helper 行为，不改 golden 数值，不实现 comparison/calibration，不改变正式 output schema，不移动或重命名 evaluation 文件。

## 操作

1. 阅读 `time_router/evaluation/metrics.py`、`summary.py`、`prediction_rows.py` 和 `__init__.py` 的当前职责与导出。
2. 新增 `docs/refactor/evaluation_package_boundary.md`，记录 evaluation package 边界、public/private API、consolidation 判断和后续整理门禁。
3. 更新 `docs/refactor/stage1_refactor_roadmap.md`，追加 P3e 当前状态、完成范围和明确不做事项。
4. 更新 `WORKSPACE_STRUCTURE.md`，登记新增边界复核文档，并补充 `docs/refactor/` 和 roadmap 的描述。
5. 使用 Quito 环境运行验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

## 结果

- `docs/refactor/evaluation_package_boundary.md` 明确：
  - `metrics.py` 当前包含基础 MAE/MSE、fusion helper 和 router weight diagnostics；暂不拆成 `fusion.py` / `diagnostics.py`。
  - `summary.py` 只负责 `build_fusion_summary(...)` 的内存 summary dict，不代表正式 summary output schema。
  - `prediction_rows.py` 只负责内存中的 per-sample fusion rows，不写正式 CSV/JSON/Parquet，不代表正式 prediction output schema。
  - `__init__.py` 是稳定 public API 聚合入口；后续正式入口迁移应优先从 `time_router.evaluation` 导入。
- 文档列出当前 public API：`FusionMetricsResult`、`compute_mae`、`compute_mse`、`hard_top1_fusion`、`raw_soft_fusion`、`compute_selected_counts`、`compute_weight_entropy`、`compute_max_weight`、`build_fusion_summary`、`build_per_sample_fusion_rows`。
- 文档列出 private helper：`_validate_model_columns`、`_validate_weight_matrix`、`_validate_summary_inputs`、`_validate_rows_inputs`、`_per_sample_mae`、`_per_sample_mse`。
- consolidation 判断为：当前不把 `summary.py` / `prediction_rows.py` 合回 `metrics.py`，也不把 `metrics.py` 立即拆成 `fusion.py` / `diagnostics.py`；后续若整理，应优先保持 `__init__.py` public API 不变。
- 三条验收命令均通过。

## 结论

P3e evaluation package boundary review / consolidation plan 已完成。本次只新增和更新文档，没有修改 helper 函数签名或行为，没有移动/重命名 evaluation 文件，没有实现 comparison、temperature/top-k/calibration、正式 output schema 或正式训练 CLI。

## 下一步方案

后续如继续 P3，可在 comparison/calibration/report schema 边界明确后再考虑 P3 cleanup；正式 Visual Router / TimeFuse-style fusor 入口迁移仍保留到 P6。任何 evaluation 文件移动或内部拆分都应保持 `time_router.evaluation` public API 不变，并运行 golden smoke、oracle/TSF smoke 和 compileall。
