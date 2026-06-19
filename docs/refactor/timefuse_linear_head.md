# Stage 1 P7b TimeFuseLinearSoftmaxHead

本文记录 Stage 1 P7b 的最小 `TimeFuseLinearSoftmaxHead`。该 adapter 基于 P5c protocol types、P7a `TimeFuseFeatureCacheProvider` 和 P6b/P6c Evaluation adapter 边界，只把 `FeatureBatch.features` 线性映射为 `RouterOutput(logits, weights)`，先供 smoke 使用。

## 1. 目标

- 输入 `FeatureBatch` 和调用方显式传入的 `model_columns`。
- 输出 `RouterOutput`。
- 保持 `FeatureBatch.sample_keys` 顺序。
- `logits` / `weights` 的专家维度严格与 `model_columns` 对齐。
- `weights` 沿专家维度做 softmax，逐样本和为 1。
- smoke 使用固定小矩阵、固定权重和固定 bias，验证 deterministic 输出。

## 2. 非目标

- 不训练。
- 不计算 loss。
- 不创建 optimizer。
- 不保存 checkpoint。
- 不读取 prediction cache。
- 不读取 oracle/TSF。
- 不读取 feature CSV。
- 不访问 `/data2`。
- 不创建 run_dir。
- 不写 status、metadata、CSV、JSON 或 Parquet。
- 不迁移正式 TimeFuse fusor、Visual Router 或 canonical runtime 入口。

## 3. Public API

```python
from time_router.models import TimeFuseLinearSoftmaxHead

head = TimeFuseLinearSoftmaxHead(weight=weight, bias=bias)
router_output = head.predict(feature_batch, model_columns)
```

`weight` 形状为 `[feature_dim, num_experts]`，`bias` 可选，长度为 `num_experts`。`model_columns` 是唯一专家顺序来源，其长度必须等于 `num_experts`。

## 4. 输入输出 Contract

输入：

- `feature_batch.sample_keys: tuple[str, ...]`
- `feature_batch.features: numpy-compatible 2D array`
- `model_columns: Sequence[str]`

输出：

- `RouterOutput.sample_keys`：与 `FeatureBatch.sample_keys` 完全一致。
- `RouterOutput.model_columns`：与调用方传入的 `model_columns` 完全一致。
- `RouterOutput.logits`：`features @ weight + bias`，形状 `[num_samples, num_experts]`。
- `RouterOutput.weights`：对 logits 的专家维度做 stable softmax，形状 `[num_samples, num_experts]`。
- `RouterOutput.extra`：只记录 `head_name`、`feature_schema`、`feature_dim` 和 `num_experts` 等轻量 lineage。

## 5. Smoke 验收

P7b 新增：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_linear_head_smoke.py
```

完整本步验收：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_feature_cache_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_linear_head_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

smoke 阶段会阻断 `open`、`Path.open`、`np.load`、`np.save` 和 `np.savez`，并检查 `experiment_logs/run_outputs` 一层目录集合不变，用于证明 head 只做纯内存前向映射。
