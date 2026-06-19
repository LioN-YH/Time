# Stage 1 P6b FusionEvaluator Adapter

日志日期：2026-06-19 23:59:34 CST

## 目的

在 P6a `PredictionCacheExpertProvider` 和 Expert System Boundary Review 之后，新增最小 `FusionEvaluator` adapter，使 evaluation 层可以从 `ExpertBatch + RouterOutput` 或显式 `EvaluationInput` 复算 hard top-1、raw soft fusion、summary 和 per-sample rows。

## 背景

P5c 已定义 `ExpertBatch`、`RouterOutput` 和 `EvaluationInput` 等 protocol types。P6a 已实现 `PredictionCacheExpertProvider`，可从 golden fixture 的 prediction cache 显式加载 sample_keys 并输出 `ExpertBatch`。P6a.5 已明确专家系统边界：P6b evaluator adapter 不应重新读取 prediction cache、manifest、packed npy、oracle/TSF 或正式输出目录，而应只消费显式内存对象。

## 操作

1. 新增 `time_router/evaluation/fusion_evaluator.py`。
   - 定义 `FusionEvaluationResult`。
   - 定义 `FusionEvaluator`。
   - `build_evaluation_input(...)` 检查 `ExpertBatch` 与 `RouterOutput` 的 `sample_keys` 和 `model_columns` 完全一致。
   - `evaluate(...)` 支持 `expert_batch + router_output` 或直接 `evaluation_input` 两种输入。
   - 内部只调用 `hard_top1_fusion`、`raw_soft_fusion`、`build_fusion_summary` 和 `build_per_sample_fusion_rows`。
2. 更新 `time_router/evaluation/__init__.py`，从 public API 导出 `FusionEvaluator` 和 `FusionEvaluationResult`。
3. 新增 `tests/smoke/stage1_fusion_evaluator_adapter_smoke.py`。
   - 使用 2026-06-14 的 4 sample packed golden fixture。
   - 先构造 `PredictionCacheExpertProvider`，显式传入 golden sample_keys 得到 `ExpertBatch`。
   - 用 golden weights 构造 `RouterOutput`。
   - 调用 `FusionEvaluator` 复算 summary 和 per-sample rows。
   - 在 adapter 调用阶段阻断 `open`、`Path.open` 和 `np.load`，验证 adapter 不重新读取 prediction cache、oracle/TSF 或其他文件。
   - 比对 `experiment_logs/run_outputs/` 一层目录集合，验证 adapter 不创建正式输出目录。
4. 新增 `docs/refactor/fusion_evaluator_adapter.md`，记录 P6b API、输入输出、复用 public API、明确不做范围、smoke 覆盖和后续接入顺序。
5. 更新 `docs/refactor/stage1_refactor_roadmap.md`，新增 P6b 完成状态和验收范围。
6. 更新 `docs/refactor/stage1_target_architecture.md`，补充 `FusionEvaluator` adapter 在共享主干中的位置。
7. 更新 `docs/refactor/evaluation_package_boundary.md`，补充 `fusion_evaluator.py` 的职责边界和 public API。
8. 更新 `docs/refactor/expert_system_boundary_review.md`，将 P6b 从后续边界更新为已按边界实现 smoke-only adapter。
9. 更新 `WORKSPACE_STRUCTURE.md`，登记新增 adapter、smoke 和文档。
10. 运行用户指定的全部验收命令。

## 结果

新增 adapter 输出纯内存对象：

- `evaluation_input`
- `hard_result`
- `raw_soft_result`
- `summary`
- `per_sample_rows`
- `diagnostics`

新增 smoke 已验证：

- `sample_keys` 保持 golden 顺序。
- `model_columns` 与固定五专家顺序一致。
- `EvaluationInput.y_pred` / `y_true` 复用 `ExpertBatch` 对象。
- `EvaluationInput.weights` 复用 `RouterOutput.weights`。
- summary hard top-1 MAE/MSE 和 raw soft MAE/MSE 与 golden smoke 当前锁定值一致。
- per-sample rows 的字段集合、数量、`selected_model`、`selected_index`、hard/raw-soft MAE/MSE、`max_weight` 和 `weight_entropy` 与现有 golden 口径一致。
- adapter 调用阶段未调用 `open`、`Path.open` 或 `np.load`。
- adapter 未创建新的 `experiment_logs/run_outputs/` 一层输出目录。

验收命令均通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_protocol_types_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_cache_expert_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_fusion_evaluator_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

## 结论

P6b 最小 FusionEvaluator adapter 已完成。当前实现只作为 smoke-only adapter 存在，严格保持在 evaluation package 内，不修改 reader/provider 行为，不访问 `/data2`，不创建 run_dir，不写 status/metadata/CSV/JSON/Parquet，也不接入 Visual Router / TimeFuse-style fusor 正式训练入口。

## 下一步方案

后续可以在新的小步中设计正式 Evaluator interface 与 runtime/report writer 分工。正式 Visual Router / TimeFuse-style fusor 入口迁移仍应保持小步推进：先在小规模等价门禁通过后，再逐步替换入口内部 evaluation 复算逻辑。
