# Stage 1 P7c TimeFuse Protocol Chain Smoke

本文记录 Stage 1 P7c 的 smoke-only TimeFuse protocol chain。该 smoke 把 P6/P7 已完成 adapter 串起来：

```text
PredictionCacheExpertProvider -> ExpertBatch
TimeFuseFeatureCacheProvider -> FeatureBatch
TimeFuseLinearSoftmaxHead -> RouterOutput
EvaluationInputAdapter -> summary / per-sample rows
```

本步只验证协议对象可组合，不迁移正式 TimeFuse fusor / Visual Router 入口。

## 1. 目标

- 使用 golden prediction fixture 构造 `ExpertBatch`。
- 使用测试内临时 TimeFuse feature CSV 构造 `FeatureBatch`。
- `FeatureBatch.sample_keys` 必须与 `ExpertBatch.sample_keys` 对齐。
- 使用固定 `TimeFuseLinearSoftmaxHead` 权重生成 `RouterOutput`。
- 使用 `EvaluationInputAdapter` 复算内存 summary 和 per-sample rows。
- 验证 `sample_keys`、`model_columns`、features、weights、summary、rows 均保序且 deterministic。

## 2. 非目标

- 不训练。
- 不计算 loss。
- 不创建 optimizer。
- 不保存 checkpoint。
- 不访问 `/data2`。
- 不新增 Bash 或 `scripts/`。
- 不创建 run_dir。
- 不写 status、metadata、CSV、JSON 或 Parquet。
- 不迁移正式 TimeFuse fusor 或 Visual Router 入口。

## 3. IO 边界

链路 smoke 分阶段控制 IO：

- `PredictionCacheExpertProvider` 阶段允许读取 golden prediction fixture。
- `TimeFuseFeatureCacheProvider` 阶段只允许读取测试内临时 feature CSV，并阻断 `np.load`。
- `TimeFuseLinearSoftmaxHead` 和 `EvaluationInputAdapter` 阶段阻断 `open`、`Path.open`、`np.load`、`np.save` 和 `np.savez`。
- smoke 前后检查 `experiment_logs/run_outputs` 一层目录集合不变。

这证明 head/evaluator 阶段不重新读取 prediction cache、oracle/TSF 或 feature CSV，也不写运行产物。

## 4. Deterministic 口径

`tests/smoke/stage1_timefuse_protocol_chain_smoke.py` 使用固定 4 sample golden fixture、固定 17 维 feature matrix、固定 `17 x 5` linear weight 和 bias。smoke 锁定：

- `FeatureBatch.features`
- `RouterOutput.logits`
- `RouterOutput.weights`
- weights 逐样本和为 1
- summary hard/raw-soft MAE/MSE
- selected counts
- mean entropy / mean max weight
- per-sample rows 的 sample_key、selected model/index、hard/raw-soft MAE/MSE、max weight 和 entropy

## 5. 验收

P7c 新增：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
```

完整本步验收：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_feature_cache_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_linear_head_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```
