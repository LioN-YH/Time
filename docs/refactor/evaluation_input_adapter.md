# Stage 1 P6b EvaluationInput Adapter

创建日期：2026-06-20

P6c consolidation 更新（2026-06-20）：`EvaluationInputAdapter` 是 evaluation adapter 的 canonical 实现。`FusionEvaluator` 若继续保留，只作为 legacy/compat wrapper，内部必须委托 `EvaluationInputAdapter`，不得复制 `hard_top1_fusion`、`raw_soft_fusion`、`build_fusion_summary` 或 `build_per_sample_fusion_rows` 调用逻辑。

## 1. 目标

本文记录 Stage 1 P6b 最小 `EvaluationInputAdapter`。该 adapter 基于 P5c protocol types、P5d adapter boundary、P6a `PredictionCacheExpertProvider` 和 P6a.5 专家系统边界审计，只把当前 Stage 1 canonical experiment 的 `ExpertBatch + RouterOutput.weights` 或显式 fusion weights 包装为 `EvaluationInput`，再复用 `time_router.evaluation` public API 生成内存 summary 和 per-sample rows。

本步只供 smoke 使用，不接正式 Visual Router / TimeFuse fusor 训练入口。

新增代码：

- `time_router/evaluation/evaluation_input_adapter.py`
- `tests/smoke/stage1_evaluation_input_adapter_smoke.py`

同步保留：

- `time_router/evaluation/fusion_evaluator.py`
- `tests/smoke/stage1_fusion_evaluator_adapter_smoke.py`

`FusionEvaluator` 是较早 P6b 命名下的兼容 adapter；P6c 后验收主入口以 `EvaluationInputAdapter` 为准，兼容 smoke 只检查旧路径不漂移。

## 2. Public API

最小调用形态：

```python
from time_router.evaluation import EvaluationInputAdapter

result = EvaluationInputAdapter().evaluate(
    expert_batch=expert_batch,
    router_output=router_output,
)
```

显式传入 fusion weights：

```python
result = EvaluationInputAdapter().evaluate(
    expert_batch=expert_batch,
    fusion_weights=weights,
)
```

输出 `EvaluationInputAdapterResult` 包含：

- `evaluation_input`
- `hard_result`
- `raw_soft_result`
- `summary`
- `per_sample_rows`
- `extra`

## 3. 适配流程

```text
ExpertBatch + RouterOutput.weights
  或 ExpertBatch + explicit fusion_weights
  -> EvaluationInput
  -> hard_top1_fusion
  -> raw_soft_fusion
  -> build_fusion_summary
  -> build_per_sample_fusion_rows
```

adapter 原样复用：

- `ExpertBatch.sample_keys`
- `ExpertBatch.model_columns`
- `ExpertBatch.y_pred`
- `ExpertBatch.y_true`
- `RouterOutput.weights` 或显式 `fusion_weights`

如需 lineage，只从 `ExpertBatch.row_index_metadata`、`ExpertBatch.extra` 和 `RouterOutput.extra` 读取轻量信息，并放入 `EvaluationInput.extra` / result `extra`。

P6c 后，`EvaluationInputAdapter.evaluate_input(...)` 是 adapter 层唯一调用 evaluation public API 的实现点。兼容包装应传入已经构造好的 `EvaluationInput` 并复用该方法，避免 adapter 层出现两份 summary/rows 复算逻辑。

## 4. 边界

明确不做：

- 不实现长期 `ExpertSystem`。
- 不新增 expert registry。
- 不新增 expert training。
- 不新增 online expert serving。
- 不修改 `PredictionBatchReader` 行为。
- 不修改 `PredictionCacheExpertProvider` 行为。
- 不重新读取 manifest。
- 不重新读取 packed npy。
- 不读取 oracle/TSF。
- 不访问 `/data2`。
- 不创建 run_dir。
- 不写 `status.json` / `metadata.json`。
- 不写 CSV / JSON / Parquet。
- 不实现 runtime / launcher。
- 不新增 Bash 或 scripts entrypoint。
- 不实现 config system。
- 不迁移 Visual Router / TimeFuse fusor 正式入口。
- 不新增 calibration / temperature / top-k。
- 不新增 oracle regret。
- 不改模型结构、loss 或正式输出目录。

当前实现可以继续服务固定五专家顺序，但这是 Stage 1 canonical experiment 契约，不是 Time framework 长期全局 Expert System 契约。

## 5. Smoke 覆盖

验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py
```

覆盖内容：

- 使用 golden fixture。
- 构造 `PredictionCacheExpertProvider`。
- 显式传入 golden sample_keys 得到 `ExpertBatch`。
- 构造 `RouterOutput`，并单独覆盖显式 fusion weights 输入。
- 通过 `EvaluationInputAdapter` 得到 `EvaluationInput`。
- 复算 hard top-1 summary。
- 复算 raw soft fusion summary。
- 生成 per-sample rows。
- 检查 sample_keys 保序。
- 检查 model_columns 与固定五专家顺序一致。
- 检查 summary 数值与 golden smoke 当前锁定指标一致。
- 检查 per-sample rows 数量、字段集合、selected_model、selected_index、hard/raw-soft MAE/MSE、max_weight、weight_entropy。
- adapter 调用阶段阻断 `open`、`Path.open` 和 `np.load`，验证不重新读取 prediction cache、oracle/TSF。
- 检查 `experiment_logs/run_outputs/` 一层目录集合不变，验证不创建正式输出目录。
- 兼容 `FusionEvaluator` smoke 额外检查 diagnostics 中的 `canonical_adapter_name=EvaluationInputAdapter`。

## 6. 后续接入顺序

下一步仍不直接迁移正式入口。更稳妥的顺序是：

1. 继续保持 `PredictionCacheExpertProvider` 与 `EvaluationInputAdapter` 只由 smoke 使用。
2. 后续新增 feature provider / router head adapter 时，先用同一 golden fixture 做最小集成 smoke。
3. 正式 Visual Router / TimeFuse fusor 入口迁移前后都运行 golden smoke、provider smoke、adapter smoke 和 compileall。
