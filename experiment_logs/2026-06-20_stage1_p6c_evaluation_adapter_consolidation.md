# Stage 1 P6c Evaluation Adapter Consolidation

日志日期：2026-06-20 00:31:07 CST

## 目的

收束 P6b 后 evaluation adapter 的命名和职责，避免 `EvaluationInputAdapter` 与 `FusionEvaluator` 两套相似逻辑并行生长。保持 `EvaluationInputAdapter` 作为 canonical adapter：`ExpertBatch + RouterOutput.weights / explicit fusion weights -> EvaluationInput -> evaluation public API`。`FusionEvaluator` 若保留，只作为 legacy/compat wrapper。

## 背景

P6b 已新增 `EvaluationInputAdapter`，但仓库中仍保留较早命名的 `FusionEvaluator`。在 P6c 前，`fusion_evaluator.py` 内部仍独立调用 `hard_top1_fusion`、`raw_soft_fusion`、`build_fusion_summary` 和 `build_per_sample_fusion_rows`，与 `EvaluationInputAdapter` 形成重复实现风险。

## 操作

1. 在 `EvaluationInputAdapter` 中新增 `evaluate_input(evaluation_input=...)`，集中承载 adapter 层唯一的 evaluation public API 调用逻辑。
2. 修改 `FusionEvaluator`，使其从 `ExpertBatch + RouterOutput` 或显式 `EvaluationInput` 进入后委托 `EvaluationInputAdapter`，只负责兼容旧 result 类型和 diagnostics。
3. 更新 `tests/smoke/stage1_fusion_evaluator_adapter_smoke.py`，将其标注为 compat smoke，并断言 diagnostics 中 `canonical_adapter_name=EvaluationInputAdapter`。
4. 更新 `time_router/evaluation/__init__.py` public API 注释，明确 `EvaluationInputAdapter` 是 canonical adapter，`FusionEvaluator` 是旧命名兼容包装。
5. 更新 `docs/refactor/evaluation_input_adapter.md`、`docs/refactor/fusion_evaluator_adapter.md`、`docs/refactor/evaluation_package_boundary.md` 和 `docs/refactor/stage1_refactor_roadmap.md`。
6. 同步更新 `WORKSPACE_STRUCTURE.md`。

## 结果

当前职责收束为：

- `EvaluationInputAdapter`：canonical adapter，负责构造 `EvaluationInput` 和通过 `evaluate_input(...)` 复算内存 summary / per-sample rows。
- `FusionEvaluator`：legacy/compat wrapper，保留旧 public API，不再直接调用 metrics/summary/rows helper。
- 主验收路径仍是 `tests/smoke/stage1_evaluation_input_adapter_smoke.py`。
- `tests/smoke/stage1_fusion_evaluator_adapter_smoke.py` 只验证兼容路径不漂移。

本步未修改 `metrics.py`、`summary.py`、`prediction_rows.py`、`PredictionBatchReader` 或 `PredictionCacheExpertProvider`，未接正式 Visual Router / TimeFuse fusor 入口，未访问 `/data2`，未创建 run_dir，也未写 CSV/JSON/Parquet 结果。

已运行验证：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_cache_expert_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_fusion_evaluator_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

以上命令均通过。

## 结论

P6c consolidation 已完成代码与文档层面的职责收束。后续新增 evaluation adapter 逻辑应进入 `EvaluationInputAdapter`，不要再扩展 `FusionEvaluator` 的独立实现。

## 下一步方案

小步提交并推送到远程 `refactor/stage1-route-audit` 分支。后续新增 evaluation adapter 逻辑只进入 canonical `EvaluationInputAdapter`，兼容 `FusionEvaluator` 不再扩展独立实现。
