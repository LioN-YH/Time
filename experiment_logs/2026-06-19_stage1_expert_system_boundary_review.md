# Stage 1 Expert System Boundary Review

日志日期：2026-06-19 23:45:11 CST

## 目的

在 P6a `PredictionCacheExpertProvider` 之后、P6b FusionEvaluator adapter 之前，补充 Stage 1 专家系统边界审计和文档冻结，明确 `ExpertProvider / ExpertBatch` 是 Time framework 长期专家系统边界，而 `PredictionCacheExpertProvider` 只是当前 Stage 1 canonical experiment 的 prediction-cache adapter implementation。

## 背景

P6a 已新增 smoke-only `PredictionCacheExpertProvider`，它复用 `PredictionBatchReader` 并输出 `ExpertBatch`。该实现服务于当前 Stage 1 canonical experiment，因此保留固定五专家顺序、packed prediction cache、sample_key 保序和 verify_metrics 等约束。

本次需要避免把 P6a 的实现约束误提升为 Time framework 的全局专家系统契约，尤其需要明确固定五专家顺序只属于当前 Stage 1 canonical experiment。

## 操作

1. 读取并复核现有文档与代码边界：
   - `docs/refactor/stage1_refactor_roadmap.md`
   - `docs/refactor/stage1_target_architecture.md`
   - `docs/refactor/prediction_cache_expert_provider.md`
   - `docs/refactor/protocol_types.md`
   - `docs/refactor/stage1_provider_interface.md`
   - `time_router/protocols/types.py`
   - `time_router/experts/prediction_cache.py`
2. 新增 `docs/refactor/expert_system_boundary_review.md`，文档化三层契约：
   - Time framework long-term expert-system contract；
   - Stage 1 canonical experiment contract；
   - PredictionCacheExpertProvider implementation constraint。
3. 更新 `docs/refactor/stage1_refactor_roadmap.md`，新增 P6a.5 expert system boundary review only。
4. 更新 `docs/refactor/stage1_target_architecture.md`，在目标架构和共享主干表中补充 `ExpertProvider / ExpertBatch` 长期边界与 `PredictionCacheExpertProvider` adapter 边界。
5. 更新 `docs/refactor/prediction_cache_expert_provider.md`，补充 P6a.5 审计结论、cache 是 implementation 不是 interface、P6b FusionEvaluator adapter 后续消费 `ExpertBatch + RouterOutput/EvaluationInput` 的边界。
6. 更新 `WORKSPACE_STRUCTURE.md`，登记新增长期文档和结构说明。
7. 运行用户指定验收命令。

## 结果

新增文档明确以下结论：

- `ExpertProvider` 是专家系统边界，不是 prediction cache 边界。
- `ExpertBatch` 是下游 Router / Fusor / Evaluator 的统一专家输出载体。
- `PredictionCacheExpertProvider` 只是 Stage 1 canonical experiment 的 prediction-cache adapter implementation。
- cache 是 implementation，不是 interface。
- 固定五专家顺序属于 Stage 1 canonical experiment 契约，不上升为 Time framework 全局专家系统契约。
- 当前 P6a provider 可以保留固定五专家顺序校验，因为它服务的是 Stage 1 canonical experiment。
- 未来 `ExpertProvider` 可以来自 prediction cache、statistical baselines、online expert models、external expert systems、dynamic expert pools 和 TimeFuse-style fusor branch 所需专家输出。
- Visual Router 主线和 TimeFuse-style fusor 支线后续都应依赖 `ExpertBatch` / protocol types，而不是直接绑定 packed prediction cache。
- P6b FusionEvaluator adapter 后续应该消费 `ExpertBatch + RouterOutput/EvaluationInput`，不重新读取 prediction cache。
- `ExpertProvider` 不承担 feature generation、oracle/TSF supervision、loss、evaluation、runtime artifact、run_dir、Bash launcher 或 config system 职责。

本次明确未执行：

- 未改 `PredictionBatchReader` 行为。
- 未改 `PredictionCacheExpertProvider` smoke 语义。
- 未移动 `prediction_array_io`。
- 未访问 `/data2`。
- 未创建 `run_dir`。
- 未写 `status.json` / `metadata.json`。
- 未实现 config system。
- 未实现 runtime / launcher。
- 未新增 Bash 或 `scripts/` entrypoint。
- 未修改 Visual Router / TimeFuse fusor 正式入口。
- 未改模型结构、loss 或正式输出目录。
- 未新增正式 provider abstraction 代码。

验收命令均通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_protocol_types_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_cache_expert_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

## 结论

Stage 1 P6a 到 P6b 之间的专家系统边界已完成文档冻结。后续 P6b FusionEvaluator adapter 应从 `ExpertBatch + RouterOutput` 构造 `EvaluationInput` 并调用 evaluator / `time_router.evaluation` public API，不应重新读取 prediction cache 或把 packed cache 细节扩散到 evaluator。

## 下一步方案

1. 提交并推送 `refactor/stage1-route-audit` 分支。
2. 后续进入 P6b FusionEvaluator adapter smoke-only 时，优先验证 `ExpertBatch + RouterOutput -> EvaluationInput -> summary/rows`，并继续不迁移正式 Visual Router / TimeFuse fusor 入口。
