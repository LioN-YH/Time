# Stage 1 P7c TimeFuse Protocol Chain Smoke

日志日期：2026-06-20 01:13:04 CST

## 目的

新增 smoke-only TimeFuse protocol chain，把已完成的 `PredictionCacheExpertProvider`、`TimeFuseFeatureCacheProvider`、`TimeFuseLinearSoftmaxHead` 和 `EvaluationInputAdapter` 串起来，验证 Stage 1 协议对象可以组合。

## 背景

P6a 已有 `PredictionCacheExpertProvider -> ExpertBatch`，P7a 已有 `TimeFuseFeatureCacheProvider -> FeatureBatch`，P7b 已有 `TimeFuseLinearSoftmaxHead -> RouterOutput`，P6b/P6c 已有 `EvaluationInputAdapter -> summary / per-sample rows`。本步只做链路 smoke，不迁移正式 TimeFuse fusor / Visual Router 入口，不新增 Bash/scripts。

## 操作

1. 新增 `tests/smoke/stage1_timefuse_protocol_chain_smoke.py`：
   - 使用 golden prediction fixture 构造 `ExpertBatch`。
   - 使用测试内临时 TimeFuse feature CSV 构造 `FeatureBatch`。
   - CSV 行顺序刻意不同于请求顺序，验证 `FeatureBatch.sample_keys` 必须与 `ExpertBatch.sample_keys` 对齐。
   - 使用固定 17 维 feature matrix、固定 `17 x 5` linear weight 和 bias 生成 `RouterOutput`。
   - 使用 `EvaluationInputAdapter` 复算内存 summary 和 per-sample rows。
   - 锁定 features、logits、weights、summary、selected counts、rows hard/raw-soft MAE/MSE、max weight 和 entropy 的 deterministic 输出。
   - head/evaluator 阶段阻断 `open`、`Path.open`、`np.load`、`np.save` 和 `np.savez`，并检查 `experiment_logs/run_outputs` 一层目录集合不变。
2. 新增 `docs/refactor/timefuse_protocol_chain_smoke.md`，记录链路目标、非目标、IO 边界、deterministic 口径和验收命令。
3. 更新 `docs/refactor/stage1_refactor_roadmap.md`，补充 P7b/P7c 当前状态、完成范围和明确不做范围。
4. 更新 `docs/refactor/stage1_target_architecture.md`，把 P7b `TimeFuseLinearSoftmaxHead` 和 P7c protocol chain smoke 纳入当前 adapter 链路说明。
5. 更新 `WORKSPACE_STRUCTURE.md`，登记新增文档和 smoke 文件。

## 结果

以下命令均已在 `quito` conda 环境下通过：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_feature_cache_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_linear_head_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

关键验证结果：

- P7a provider smoke 通过，已有 `FeatureBatch` provider contract 未漂移。
- P7b head smoke 通过，已有 `TimeFuseLinearSoftmaxHead` deterministic 输出未漂移。
- P6b evaluation adapter smoke 通过，下游 evaluation 适配路径未漂移。
- P7c protocol chain smoke 通过，链路输出 `hard_mae=1.093573928`、`raw_soft_mae=0.556751269`，features/weights/summary/rows 均保序且 deterministic。
- `compileall` 通过，新增 smoke 和现有 `time_router` 包无语法错误。
- head/evaluator 阶段未重新读取 prediction cache、oracle/TSF 或 feature CSV，未创建 run_dir，未写 status/metadata/CSV/JSON/Parquet。

## 结论

Stage 1 P7c smoke-only TimeFuse protocol chain 已完成。当前 adapter 链路可以从 golden prediction fixture 和临时 TimeFuse feature CSV 组合出 `ExpertBatch -> FeatureBatch -> RouterOutput -> EvaluationInputAdapterResult`，但正式 TimeFuse fusor / Visual Router 入口仍未迁移。

## 下一步方案

1. 小步提交并推送 `refactor/stage1-route-audit` 分支。
2. 后续如继续推进正式入口迁移，应另起小步，先设计 runtime/protocol boundary 或正式 RouterHead interface 接入门禁。
