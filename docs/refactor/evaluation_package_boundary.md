# Stage 1 Evaluation Package Boundary Review

设计日期：2026-06-19

## 1. 目的

本文记录 Stage 1 P3e 对 `time_router/evaluation` 的 package 边界复核结果。当前 P3 已完成：

- P3a：hard top-1 / raw soft fusion / MAE / MSE；
- P3b：selected counts / weight entropy / max weight；
- P3c：minimal summary dict builder；
- P3d：minimal per-sample evaluation rows builder。

本次只做文档化 review、API 边界说明和 consolidation 规划，不迁移 Visual Router / TimeFuse-style fusor 正式训练入口，不改变 helper 行为，不改变 golden 数值，不实现 comparison/calibration，不改变正式 output schema，也不移动或重命名 evaluation 文件。

## 2. 当前模块职责

### `metrics.py`

当前职责：

- 定义 `FusionMetricsResult`，作为 hard top-1 和 raw soft fusion 的数组级结果对象。
- 提供基础全局指标：
  - `compute_mae(...)`
  - `compute_mse(...)`
- 提供 fusion helper：
  - `validate_fusion_inputs(...)`
  - `hard_top1_fusion(...)`
  - `raw_soft_fusion(...)`
- 提供 router/fusor weight diagnostics：
  - `compute_selected_counts(...)`
  - `compute_weight_entropy(...)`
  - `compute_max_weight(...)`

当前判断：

- `metrics.py` 已经包含 metrics、fusion 和 diagnostics 三类概念，长期看存在拆成 `metrics.py` + `fusion.py` + `diagnostics.py` 的可能。
- 但当前 P3 只在 golden smoke 中锁定了最小行为，正式入口尚未迁移；此时拆文件会带来 import churn，收益主要是“看起来更整齐”，不值得现在做。
- P6 入口迁移前，保留 `metrics.py` 现状更稳妥。后续入口应通过 `time_router.evaluation` public API 导入，而不是依赖 `metrics.py` 的文件位置。

### `summary.py`

当前职责：

- 提供 `build_fusion_summary(...)`，将 hard top-1 result、raw soft result、weights 和 model_columns 汇总为最小 summary dict。

边界说明：

- 只消费调用方显式传入的 `FusionMetricsResult`、`weights` 和 `model_columns`。
- 不读取 manifest、prediction cache、oracle/TSF 或正式训练输出目录。
- 不代表正式 summary output schema；当前字段只是 P3c golden smoke 使用的稳定最小 dict。
- 不实现 comparison、calibration、oracle regret 或 output writer。

当前判断：

- 不建议把 `summary.py` 合回 `metrics.py`。summary 是组合层，和底层数组指标/diagnostics 的抽象层级不同。
- 保持单独文件可以为后续 comparison/report schema 留出边界，同时避免 `metrics.py` 继续膨胀。

### `prediction_rows.py`

当前职责：

- 提供 `build_per_sample_fusion_rows(...)`，将当前 batch 的 sample_keys、hard/raw-soft result、y_true、weights 和 model_columns 组合为内存中的逐样本 rows。

边界说明：

- 只消费调用方显式传入的对象和数组。
- 不读取 manifest、prediction cache、oracle/TSF 或正式训练输出目录。
- 不写 CSV/JSON/Parquet。
- 不代表正式 prediction output schema；当前 row 字段只是 P3d golden smoke 使用的稳定最小逐样本评估信息。

当前判断：

- 不建议把 `prediction_rows.py` 合回 `metrics.py`。per-sample rows 是输出准备层，虽然会调用 MAE/MSE 和 diagnostics 口径，但职责不是基础数值计算。
- 也不建议现在把它扩展成正式 prediction schema 层。正式 schema 需要等 comparison/calibration/report 边界明确后再设计。

### `__init__.py`

当前职责：

- 作为 `time_router.evaluation` 的稳定 public API 聚合入口。
- 对外导出 P3a-P3d 的最小 helper，隐藏文件拆分细节。

边界说明：

- 后续 Visual Router / TimeFuse-style fusor 正式入口迁移时，应优先使用：

```python
from time_router.evaluation import hard_top1_fusion, build_fusion_summary
```

- 下游代码不应直接依赖 `_validate_*`、`_per_sample_*` 等下划线 helper。
- 如果未来移动或拆分内部文件，必须保持 `time_router.evaluation` 的 public API 尽量不变，避免下游入口迁移时发生 import churn。

## 3. Public API 与 Private Helper

### Public API

当前稳定 public API 至少包括：

- `FusionMetricsResult`
- `compute_mae`
- `compute_mse`
- `hard_top1_fusion`
- `raw_soft_fusion`
- `compute_selected_counts`
- `compute_weight_entropy`
- `compute_max_weight`
- `build_fusion_summary`
- `build_per_sample_fusion_rows`

`validate_fusion_inputs(...)` 当前也从 `__init__.py` 导出，用于低层 fusion 输入一致性检查。它比下划线 helper 更接近公共校验函数，但不应被正式入口当作必须调用的业务接口；正式入口优先调用更高层的 fusion/summary/rows helper。

### Private Helper

以下 helper 仅用于模块内部，不应作为下游正式入口依赖：

- `_validate_model_columns`
- `_validate_weight_matrix`
- `_validate_summary_inputs`
- `_validate_rows_inputs`
- `_per_sample_mae`
- `_per_sample_mse`

后续如果需要复用私有校验逻辑，应先评估是否值得提升为明确 public API，而不是从下游代码直接 import 下划线函数。

## 4. Consolidation 判断

### 现在是否应该把 `summary.py` / `prediction_rows.py` 合回 `metrics.py`？

不应该。

理由：

- `metrics.py` 已经承担基础指标、fusion 和 diagnostics；继续合并 summary/rows 会让一个文件同时承担数值计算、组合汇总和输出准备三类职责。
- `summary.py` 和 `prediction_rows.py` 目前虽然很小，但边界清晰：它们都只消费显式传入对象，不做 IO，不代表正式 schema。
- 后续 comparison、calibration 和 report schema 的边界尚未确定；现在合并会降低后续拆分的可读性。

### 现在是否应该把 `metrics.py` 拆成 `fusion.py` / `diagnostics.py`？

暂时不应该。

理由：

- 当前 public API 刚通过 golden smoke 固定，正式入口尚未迁移；此时拆文件主要制造 import churn。
- P3a/P3b 的 helper 体量仍可审查，且都属于同一个纯 numpy 数组层。
- 等 P3 cleanup 或 P4 前，如果 comparison/calibration 需要更清晰的基础层边界，再考虑拆分。

### 哪种结构最适合后续 P6 入口迁移？

当前最适合 P6 的结构是：

- 保持内部文件现状：
  - `metrics.py`：基础数组指标、fusion、diagnostics；
  - `summary.py`：最小 batch summary 组合；
  - `prediction_rows.py`：最小 per-sample rows 组合；
  - `__init__.py`：稳定 public API 聚合入口。
- P6 入口迁移时从 `time_router.evaluation` 导入 public API，而不是深层依赖具体文件。
- 如果 P6 前必须整理内部文件，也应保证 `__init__.py` 兼容导出不变。

### 如果未来要整理，应该在哪个阶段做？

建议顺序：

1. P3 cleanup：只在 comparison/calibration/report schema 边界明确后，评估是否拆出 `fusion.py`、`diagnostics.py`、`comparison.py` 或 `calibration.py`。
2. P4 前：如果 logging/path/config 抽取需要统一输出命名，再补充文档和最小 schema 边界，但仍避免重排 P3 helper。
3. P6 前：如果正式入口迁移需要更清晰的 import 层，可做一次内部文件整理；前提是保持 `time_router.evaluation` public API 不变，并运行全部 smoke 门禁。

## 5. 后续判断原则

- 不为“看起来整齐”而移动文件。
- 只有当 public API 稳定、golden smoke 充分、comparison/calibration/report schema 边界明确后，才做 consolidation。
- 若要整理内部文件，应优先保持 `__init__.py` public API 不变。
- 任何文件移动、函数拆分或导入路径整理，都必须运行：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

## 6. P3e 明确不做

- 不迁移 Visual Router / TimeFuse-style fusor 正式训练入口。
- 不实现 comparison。
- 不实现 temperature/top-k/calibration。
- 不改变正式 summary / comparison / prediction output schema。
- 不新增正式训练 CLI。
- 不读取 oracle/TSF，不实现 oracle regret。
- 不接入 `OracleTsfReader` 或 full-scale 输出目录。
- 不写 CSV / JSON / Parquet 到正式输出目录。
- 不改 `PredictionBatchReader` / `OracleTsfReader`。
- 不改模型结构、loss 或正式输出目录。
- 不为整理而移动或重命名 evaluation 文件。
- 不改变现有 public API 和 helper 行为。
