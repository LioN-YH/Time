# Stage 1 P6b EvaluationInput Adapter

日志日期：2026-06-20 00:15:25 CST

## 目的

基于 P5c protocol types、P5d adapter boundary、P6a `PredictionCacheExpertProvider` 和 P6a.5 专家系统边界审计，新增最小 `EvaluationInputAdapter`。本步只把当前 Stage 1 canonical experiment 的 `ExpertBatch + RouterOutput.weights` 或显式 fusion weights 包装为 `EvaluationInput`，再复用 `time_router.evaluation` public API 生成内存 summary 和 per-sample rows。

## 背景

前一小步已实现较早命名的 `FusionEvaluator` adapter，但本轮目标明确要求新增 `time_router/evaluation/evaluation_input_adapter.py`、`tests/smoke/stage1_evaluation_input_adapter_smoke.py` 和 `docs/refactor/evaluation_input_adapter.md`。当前 Stage 1 仍以 QuitoBench、固定五专家顺序、prediction cache 和 Visual Router 主实验为核心；固定五专家顺序只属于 Stage 1 canonical experiment 契约，不是 Time framework 长期全局 Expert System 契约。

## 操作

1. 新增 `time_router/evaluation/evaluation_input_adapter.py`，实现 `EvaluationInputAdapter` 和 `EvaluationInputAdapterResult`。
2. 在 `time_router/evaluation/__init__.py` 导出 `EvaluationInputAdapter` 和 `EvaluationInputAdapterResult`，保留既有 `FusionEvaluator` 兼容导出。
3. 新增 `tests/smoke/stage1_evaluation_input_adapter_smoke.py`，使用 golden fixture 和 `PredictionCacheExpertProvider` 显式构造 `ExpertBatch`，覆盖 `RouterOutput.weights` 和显式 `fusion_weights` 两种输入路径。
4. 新增 `docs/refactor/evaluation_input_adapter.md`，记录 API、适配流程、边界和 smoke 覆盖。
5. 更新 `docs/refactor/stage1_refactor_roadmap.md`、`docs/refactor/stage1_target_architecture.md`、`docs/refactor/evaluation_package_boundary.md`、`docs/refactor/expert_system_boundary_review.md`、`docs/refactor/prediction_cache_expert_provider.md` 和 `WORKSPACE_STRUCTURE.md`。
6. 运行目标文件列出的完整验收命令，并额外运行旧 `FusionEvaluator` adapter smoke 做兼容回归。

## 结果

新增 adapter 的关键行为：

- 使用 `ExpertBatch.sample_keys` 保持样本顺序。
- 使用 `ExpertBatch.model_columns` 保持专家列顺序。
- 原样复用 `ExpertBatch.y_pred` 和 `ExpertBatch.y_true`。
- 使用 `RouterOutput.weights` 或显式 `fusion_weights` 作为融合权重。
- 只从 `ExpertBatch.row_index_metadata`、`ExpertBatch.extra` 和 `RouterOutput.extra` 转递轻量 lineage。
- 内部只调用 `hard_top1_fusion`、`raw_soft_fusion`、`build_fusion_summary` 和 `build_per_sample_fusion_rows`。
- 输出纯内存 `EvaluationInput`、`summary`、`per_sample_rows`、`hard_result`、`raw_soft_result` 和 `extra`。

已运行验证：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_protocol_types_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_cache_expert_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

以上目标验收命令均通过。另额外运行：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_fusion_evaluator_adapter_smoke.py
```

兼容 smoke 也通过。新增 smoke 输出确认 adapter 阶段未调用 `open`、`Path.open` 或 `np.load`，不重新读取 prediction cache 或 oracle/TSF；同时确认 `run_outputs` 一层目录集合不变，没有创建正式输出目录。

## 结论

P6b 目标命名下的最小 `EvaluationInputAdapter` 已实现并通过完整验收。它只服务 Stage 1 canonical experiment smoke，不实现长期 `ExpertSystem`、expert registry、expert training、online expert serving、runtime、launcher、config、正式入口迁移、calibration、temperature、top-k 或 oracle regret。

## 下一步方案

进行小步提交，并推送到远程 `refactor/stage1-route-audit` 分支。后续正式入口迁移仍应保持 golden/provider/adapter smoke 先行。
