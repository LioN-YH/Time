# Stage 1 P6a PredictionCacheExpertProvider

创建日期：2026-06-19

## 1. 目标

本文记录 Stage 1 P6a 的最小 `PredictionCacheExpertProvider`。该 adapter 基于 P1 `PredictionBatchReader`、P5c protocol types、P5d adapter boundary 和 P5e/P5f migration/launcher 设计，只把 prediction cache reader 输出包装为 `time_router.protocols.ExpertBatch`，先供 smoke 使用，不接正式 Visual Router 或 TimeFuse-style fusor 训练入口。

P6a.5 专家系统边界审计见 `docs/refactor/expert_system_boundary_review.md`。审计结论是：`ExpertProvider / ExpertBatch` 是 Time framework 长期专家系统边界；`PredictionCacheExpertProvider` 只是当前 Stage 1 canonical experiment 的 prediction-cache adapter implementation。固定五专家顺序是当前 Stage 1 canonical experiment 的契约，不是 Time framework 长期必须绑定的全局专家系统契约。

本阶段新增代码：

- `time_router/experts/__init__.py`
- `time_router/experts/prediction_cache.py`
- `tests/smoke/stage1_prediction_cache_expert_provider_smoke.py`

## 2. Public API

最小调用形态：

```python
from time_router.experts import PredictionCacheExpertProvider

provider = PredictionCacheExpertProvider(fixture_root=fixture_root)
expert_batch = provider.load_batch(sample_keys, verify_metrics=True)
```

`PredictionCacheExpertProvider.load_batch(...)` 的输入和输出：

- 输入 `sample_keys: Sequence[str]`，必须显式传入，不能为空且不能重复。
- 输入 `verify_metrics: bool = True`，沿用 `PredictionBatchReader` 对 manifest MAE/MSE 的复算校验。
- 输出 `ExpertBatch.sample_keys: tuple[str, ...]`，保持调用方传入顺序。
- 输出 `ExpertBatch.model_columns: tuple[str, ...]`，保持固定五专家顺序。
- 输出 `ExpertBatch.y_pred`，来自 reader，形状为 `[num_samples, num_experts, pred_len, channels]`。
- 输出 `ExpertBatch.y_true`，来自 reader，形状为 `[num_samples, pred_len, channels]`。
- 输出 `ExpertBatch.row_index_metadata`，保留 reader 的 `row_indices_by_sample_model`。
- 输出 `ExpertBatch.extra`，记录 `provider_name`、`array_storage` 和轻量 `reader_metadata`。

## 3. 与 PredictionBatchReader 的关系

`PredictionCacheExpertProvider` 不重新实现 prediction cache 读取逻辑。它内部只复用：

```python
time_router.io.PredictionBatchReader
```

由 reader 继续负责：

- 读取 `packed_npy_v1` 或 legacy `per_sample_npy`。
- 保持 sample_key 顺序。
- 固定五专家顺序。
- 校验同一 sample_key 下五专家共享 `y_true`。
- 保留 packed row index lineage。
- 可选复算 manifest MAE/MSE，确认 row index 读取和 manifest 指标一致。

provider 只负责协议包装：

```text
PredictionBatchReader.load(...)
  -> PredictionBatch
  -> ExpertBatch
```

这里的 cache 是 implementation，不是 interface。未来 `ExpertProvider` 可以来自 prediction cache、statistical baselines、online expert models、external expert systems、dynamic expert pools 或 TimeFuse-style fusor branch 所需专家输出；这些实现都应输出 `ExpertBatch` 或等价 protocol object，而不是要求下游直接绑定 packed prediction cache。

## 4. ExpertBatch Metadata

P6a 输出的 `ExpertBatch.extra` 只保留轻量信息：

```text
extra = {
  "provider_name": "PredictionCacheExpertProvider",
  "array_storage": "packed_npy_v1" 或 tuple[str, ...],
  "reader_metadata": {
    "manifest_path": ".../manifest.csv",
    "manifest_row_count": 当前 batch 命中 manifest 行数,
    "manifest_model_order_by_sample": 当前 batch 的原始 manifest 专家顺序,
    "verify_metrics": True/False,
    "chunk_rows": reader chunk 大小,
  },
}
```

不把 `manifest_rows` DataFrame 或全量 manifest lookup 放入 `ExpertBatch.extra`。row index lineage 放在 `ExpertBatch.row_index_metadata`，便于 evaluator smoke 或后续 runtime metadata 选择性引用。

## 5. 明确边界

P6a 明确不做：

- 不修改 `PredictionBatchReader` 行为。
- 不移动 `prediction_array_io`。
- 不读取 oracle/TSF。
- 不生成 Visual Router 或 TimeFuse feature。
- 不计算 loss。
- 不做 evaluation；smoke 中的 evaluation 只用于验收 provider 输出未漂移。
- 不访问 `/data2`。
- 不创建 `run_dir`。
- 不写 `status.json` 或 `metadata.json`。
- 不实现 config system。
- 不实现 runtime / launcher。
- 不新增 Bash 或 `scripts/` entrypoint。
- 不修改 Visual Router / TimeFuse fusor 正式入口。
- 不改模型结构、loss 或正式输出目录。

进一步边界：

- `PredictionCacheExpertProvider` 可以继续校验固定五专家顺序，因为它服务的是当前 Stage 1 canonical experiment。
- 固定五专家顺序不应上升为所有 `ExpertProvider` 的全局专家系统契约。
- Visual Router 主线和 TimeFuse-style fusor 支线后续应依赖 `ExpertBatch` / protocol types，而不是直接读取 packed prediction cache。
- P6b EvaluationInput adapter 后续应消费 `ExpertBatch + RouterOutput.weights` 或显式 fusion weights，不重新读取 prediction cache。
- `ExpertProvider` 不承担 feature generation、oracle/TSF supervision、loss、evaluation、runtime artifact、run_dir、Bash launcher 或 config system 职责。

## 6. 为什么放在 `time_router/experts/`

P5d/P5e 的 `ExpertProvider` 语义只覆盖专家预测与共享 `y_true`，不同于更泛化的 provider 概念。P6a 选择 `time_router/experts/` 而不是 `time_router/providers/`，是为了避免过早创建“大一统 provider”层。

当前目录含义：

- `time_router/experts/`：专家预测来源 adapter，例如 prediction cache expert provider、future online expert provider。
- `time_router/features/`：后续再承载 Visual/TimeFuse feature provider。
- `time_router/runtime/`：后续再承载 protocol 执行、run_dir、checkpoint/status/metadata 调度。

## 7. Smoke 覆盖

新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_cache_expert_provider_smoke.py
```

覆盖内容：

- 使用 2026-06-14 4 sample packed golden fixture。
- 构造 `PredictionCacheExpertProvider`。
- 显式传入 golden sample_keys。
- 验证返回对象是 `ExpertBatch`。
- 验证 sample_keys tuple 保序。
- 验证 model_columns tuple 与固定五专家顺序一致。
- 验证 `y_pred=(4, 5, 48, 1)`、`y_true=(4, 48, 1)`。
- 验证 `row_index_metadata` 包含 packed_npy_v1 的 `y_true_row_index` / `y_pred_row_index`。
- 验证 `extra.provider_name`、`extra.array_storage` 和轻量 reader metadata。
- 使用 `time_router.evaluation` public API 复算 hard top-1 与 raw soft fusion golden 指标。
- 不创建正式输出目录，不访问 `/data2`。

## 8. 后续接入顺序

P6a 之后建议继续小步推进：

1. 保持 `PredictionCacheExpertProvider` 先只由 smoke 使用。
2. 新增 evaluator adapter smoke，从 `ExpertBatch + RouterOutput.weights` 或显式 fusion weights 构造 `EvaluationInput` 并复算 summary/rows。
3. 再考虑最小 config skeleton，让 smoke 可以从 config 构造 provider。
4. 最后再新增 `scripts/` thin entrypoint 和 `exp_scripts/` Bash launcher。

正式 Visual Router / TimeFuse fusor 入口只有在 provider/evaluator/config smoke 稳定后再小步接入，且每次接入前后必须运行 golden smoke 和相关 P6a smoke。
