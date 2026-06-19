# Stage 1 P6b FusionEvaluator Compat Adapter

创建日期：2026-06-19

P6c consolidation 更新（2026-06-20）：`FusionEvaluator` 不再是独立 evaluation adapter 实现，而是较早 P6b 命名下的 legacy/compat wrapper。canonical adapter 是 `EvaluationInputAdapter`，详见 `docs/refactor/evaluation_input_adapter.md`。

## 1. 目标

本文件记录 `FusionEvaluator` 兼容包装的实现边界。它位于 `time_router/evaluation/`，只负责保留旧 public API，并委托 `EvaluationInputAdapter` 将 P5c protocol objects 适配成 `time_router.evaluation` public API 可消费的内存对象，复算当前 golden smoke 锁定的 summary 与 per-sample rows。

当前实现只供 smoke 使用，不接 Visual Router / TimeFuse-style fusor 正式训练入口。

## 2. API

新增文件：

- `time_router/evaluation/fusion_evaluator.py`

新增 public API：

- `FusionEvaluator`
- `FusionEvaluationResult`

兼容调用形态：

```python
from time_router.evaluation import FusionEvaluator

result = FusionEvaluator().evaluate(
    expert_batch=expert_batch,
    router_output=router_output,
)
```

也可以在调用方已经构造好 `EvaluationInput` 时直接传入：

```python
result = FusionEvaluator().evaluate(evaluation_input=evaluation_input)
```

## 3. 输入输出

输入优先使用 P5c protocol types：

- `ExpertBatch`
- `RouterOutput`
- `EvaluationInput`

`FusionEvaluator.build_evaluation_input(...)` 会检查 `ExpertBatch.sample_keys` 与 `RouterOutput.sample_keys` 完全一致，并检查 `model_columns` 完全一致。检查通过后，adapter 原样复用：

- `ExpertBatch.y_pred`
- `ExpertBatch.y_true`
- `RouterOutput.weights`
- `RouterOutput.logits`

输出为纯内存 `FusionEvaluationResult`：

- `evaluation_input: EvaluationInput`
- `hard_result: FusionMetricsResult`
- `raw_soft_result: FusionMetricsResult`
- `summary: dict[str, Any]`
- `per_sample_rows: list[dict[str, Any]]`
- `diagnostics: dict[str, Any]`

`diagnostics` 只包含 adapter 名称、sample/model 顺序和输入对象已经携带的轻量 lineage。若需要 row index lineage，只从 `ExpertBatch.row_index_metadata` 和 `ExpertBatch.extra` 转递，不重新读取外部文件。

## 4. 复用的 evaluation public API

P6c 后，`FusionEvaluator` 自身不直接调用以下 public API：

- `hard_top1_fusion(...)`
- `raw_soft_fusion(...)`
- `build_fusion_summary(...)`
- `build_per_sample_fusion_rows(...)`

这些调用集中在 canonical `EvaluationInputAdapter.evaluate_input(...)` 中。`FusionEvaluator` 只委托 canonical adapter，并把返回结果包装为 `FusionEvaluationResult`，因此 hard top-1、raw soft fusion、MAE/MSE、selected counts、max weight、weight entropy 和 per-sample rows 口径继续由 P3a-P3d helper 决定，不新增 calibration、temperature、top-k、oracle regret 或正式 output schema。

## 5. 明确不做

- 不修改 `PredictionBatchReader` 行为。
- 不修改 `PredictionCacheExpertProvider` 行为。
- 不重新读取 manifest。
- 不重新读取 packed npy。
- 不访问 `/data2`。
- 不创建 `run_dir`。
- 不写 `status.json` / `metadata.json`。
- 不写 CSV / JSON / Parquet。
- 不实现 runtime / launcher。
- 不新增 Bash 或 `scripts/` entrypoint。
- 不实现 config system。
- 不迁移 Visual Router / TimeFuse fusor 正式入口。
- 不新增 calibration / temperature / top-k。
- 不新增 oracle regret。
- 不读取 oracle/TSF。
- 不改模型结构、loss 或正式输出目录。

## 6. Smoke 覆盖

兼容 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_fusion_evaluator_adapter_smoke.py
```

该 smoke 使用 2026-06-14 的 4 sample packed golden fixture，只检查旧 `FusionEvaluator` 路径不漂移；主验收路径是 `stage1_evaluation_input_adapter_smoke.py`：

1. 构造 `PredictionCacheExpertProvider`。
2. 显式传入 golden sample_keys 得到 `ExpertBatch`。
3. 构造包含 golden weights 的 `RouterOutput`。
4. 通过 `FusionEvaluator` 构造 `EvaluationInput`，并由 canonical `EvaluationInputAdapter` 复算 hard top-1 / raw soft result。
5. 复算 summary 和 per-sample rows。
6. 检查 sample_keys 保序、model_columns 为固定五专家顺序。
7. 检查 summary 数值与 golden smoke 当前锁定指标一致。
8. 检查 per-sample rows 数量、字段集合、selected_model、selected_index、hard/raw-soft MAE/MSE、max_weight 和 weight_entropy。
9. 在 adapter 调用阶段阻断 `open`、`Path.open` 与 `np.load`，证明 adapter 不重新读取 prediction cache、oracle/TSF 或其他文件。
10. 检查 `experiment_logs/run_outputs/` 一层目录集合不变，证明 adapter 不创建正式输出目录。
11. 检查 diagnostics 中 `canonical_adapter_name=EvaluationInputAdapter`，证明兼容路径委托 canonical adapter。

## 7. 后续接入

P6b 完成后，下一步仍不应直接迁移正式入口。更稳妥的顺序是：

1. 新代码继续优先使用 `EvaluationInputAdapter`。
2. 在正式入口迁移前，先明确 Evaluator interface 与 runtime/report writer 的分工。
3. Visual Router / TimeFuse-style fusor 正式入口后续只在小规模等价门禁通过后，再逐步替换内部 evaluation 复算逻辑。
