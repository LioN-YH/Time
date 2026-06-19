# Stage 1 P7a TimeFuseFeatureCacheProvider

创建日期：2026-06-20

## 1. 目标

本文记录 Stage 1 P7a 的最小 `TimeFuseFeatureCacheProvider`。该 adapter 基于 P5c protocol types、P5d adapter boundary、P5e/P5f migration/launcher 设计，以及 P6a/P6b 已稳定的 Expert/Evaluation adapter 边界，只把显式传入的小规模 TimeFuse 17 维 feature CSV 包装为 `time_router.protocols.FeatureBatch`，先供 smoke 使用。

本阶段不接正式 TimeFuse fusor、Visual Router、runtime、launcher 或 config system。

本阶段新增代码：

- `time_router/features/__init__.py`
- `time_router/features/timefuse_cache.py`
- `tests/smoke/stage1_timefuse_feature_cache_provider_smoke.py`

## 2. Public API

最小调用形态：

```python
from time_router.features import TimeFuseFeatureCacheProvider

provider = TimeFuseFeatureCacheProvider(
    feature_csv_path=feature_csv_path,
    feature_columns=feature_columns,
)
feature_batch = provider.load_batch(sample_keys)
```

`TimeFuseFeatureCacheProvider.load_batch(...)` 的输入和输出：

- 输入 `sample_keys: Sequence[str]`，必须显式传入，不能为空且不能重复。
- 输出 `FeatureBatch.sample_keys: tuple[str, ...]`，保持调用方传入顺序。
- 输出 `FeatureBatch.features`，当前实现为 `numpy.float32` array，形状为 `[num_samples, feature_dim]`。
- 输出 `FeatureBatch.feature_schema`，记录 `feature_schema_name`、`feature_columns`、`feature_dim` 和 `source`。
- 输出 `FeatureBatch.extra`，记录 `provider_name`、`sample_key_column`、`feature_csv_path`、`num_available_rows` 和 `dtype`。

## 3. 与 TimeFuse Feature Cache 的关系

`TimeFuseFeatureCacheProvider` 不重新定义 TimeFuse 特征语义。它只读取调用方显式传入的 feature CSV：

```text
feature CSV
  -> sample_key + feature columns
  -> FeatureBatch
```

P7a 当前只覆盖小规模 batch 包装；正式 full-scale shard streaming、split 下推、scaler fit、checkpoint、status/metadata 和评估产物写出仍留在现有 TimeFuse fusor 入口与后续 runtime 迁移中。

## 4. FeatureBatch Metadata

P7a 输出的 `FeatureBatch.feature_schema` 示例：

```text
feature_schema = {
  "feature_schema_name": "timefuse_single_variable_meta_v1",
  "feature_columns": ("timefuse_feature_00", ..., "timefuse_feature_16"),
  "feature_dim": 17,
  "source": ".../timefuse_features.csv",
}
```

P7a 输出的 `FeatureBatch.extra` 示例：

```text
extra = {
  "provider_name": "TimeFuseFeatureCacheProvider",
  "sample_key_column": "sample_key",
  "feature_csv_path": ".../timefuse_features.csv",
  "num_available_rows": 3,
  "dtype": "float32",
}
```

这些 metadata 只用于轻量 lineage，不包含 prediction manifest、oracle label、TSF enrichment、expert error、scaler state 或运行输出目录。

## 5. 明确边界

P7a 明确不做：

- 不读取 prediction cache。
- 不读取 oracle/TSF。
- 不读取 `y_true`。
- 不读取 expert error 或 oracle top-1。
- 不做 scaler fit；scaler 属于 training/runtime。
- 不创建 `run_dir`。
- 不写 `status.json`、`metadata.json`、CSV、JSON 或 Parquet。
- 不访问 `/data2`。
- 不新增 Bash 或 `scripts/` entrypoint。
- 不迁移正式 TimeFuse fusor / Visual Router 入口。
- 不改模型结构、loss、checkpoint 或正式输出目录。

## 6. 为什么放在 `time_router/features/`

P5b/P5d 的 `FeatureProvider` 语义覆盖 router/fusor 的输入特征，和 P6a 的 `ExpertProvider` 分离：

- `time_router/experts/`：专家预测来源 adapter，例如 `PredictionCacheExpertProvider`。
- `time_router/features/`：特征来源 adapter，例如 `TimeFuseFeatureCacheProvider` 和后续 Visual online ViT provider。
- `time_router/evaluation/`：显式 evaluation input 到内存 summary/rows 的评估 adapter。
- `time_router/runtime/`：后续再承载 protocol 执行、run_dir、checkpoint/status/metadata 调度。

P7a 只新增 feature-only adapter，不把 prediction cache 或 evaluation 职责混入 `time_router/features/`。

## 7. Smoke 覆盖

新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_feature_cache_provider_smoke.py
```

覆盖内容：

- 使用测试内临时 feature CSV，不依赖 `/data2`。
- 构造 `TimeFuseFeatureCacheProvider`。
- 显式传入 sample_keys。
- 验证返回对象是 `FeatureBatch`。
- 验证 sample_keys tuple 保序。
- 验证 `features=(2, 17)` 且 dtype 为 `float32`。
- 验证 17 维 feature 数值按 CSV 与请求顺序对齐。
- 验证 `feature_schema` 中的 `feature_schema_name`、`feature_columns`、`feature_dim` 和 `source`。
- 验证 `extra` 中的 provider metadata。
- 拒绝空 sample_keys 和重复 sample_key。
- provider 阶段只允许读取临时 feature CSV，并阻断 `np.load`。
- 检查 `experiment_logs/run_outputs/` 一层目录集合不变，证明不创建输出目录。

## 8. 后续接入顺序

P7a 之后建议继续小步推进：

1. 保持 `TimeFuseFeatureCacheProvider` 先只由 smoke 使用。
2. 另起小步实现 TimeFuse linear-softmax head 的纯 protocol smoke。
3. 再考虑最小 config skeleton，让 smoke 可以从 config 构造 expert/feature/evaluator adapter。
4. 最后再按正式入口迁移计划，把 TimeFuse fusor 的 feature 读取从 runtime orchestration 中小步下沉。

正式 TimeFuse fusor / Visual Router 入口只有在 provider/evaluator/head smoke 稳定后再接入。
