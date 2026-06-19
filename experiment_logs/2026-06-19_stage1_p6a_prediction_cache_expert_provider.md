# Stage 1 P6a PredictionCacheExpertProvider 最小适配器

日志日期：2026-06-19 23:27:09 CST

## 目的

新增最小 `PredictionCacheExpertProvider`，基于已有 `PredictionBatchReader` 输出 `time_router.protocols.ExpertBatch`，先只供 smoke 使用，为后续 canonical ExpertProvider / Evaluator / runtime 小步迁移提供低风险落点。

## 背景

P1 已完成共享 `PredictionBatchReader`，P5c 已定义 `ExpertBatch` 等 protocol types，P5d-P5f 已文档化 adapter boundary、entrypoint migration plan 和 launcher architecture。P5f 后推荐的第一步是先实现 `PredictionCacheExpertProvider` smoke-only adapter，而不是先做完整 config、runtime、launcher 或正式入口迁移。

## 操作

1. 新增 `time_router/experts/__init__.py`，导出 `PredictionCacheExpertProvider`。
2. 新增 `time_router/experts/prediction_cache.py`，实现 `PredictionCacheExpertProvider`。
3. `PredictionCacheExpertProvider` 内部复用 `time_router.io.PredictionBatchReader`，不重新实现 prediction cache 读取逻辑。
4. 新增显式 batch 方法 `load_batch(sample_keys, verify_metrics=True) -> ExpertBatch`。
5. `load_batch(...)` 要求调用方显式传入非空且不重复的 sample_keys，不默认扫描全量 manifest。
6. `ExpertBatch` 输出保留 sample_keys tuple、model_columns tuple、`y_pred`、`y_true`、`row_index_metadata` 和 `extra`。
7. `extra` 中记录 `provider_name`、`array_storage` 和轻量 `reader_metadata`，不塞入 manifest DataFrame。
8. 新增 `tests/smoke/stage1_prediction_cache_expert_provider_smoke.py`，使用 golden fixture 显式传入 4 个 sample_key，验证 ExpertBatch contract、packed row index metadata、provider extra，并用 `time_router.evaluation` public API 复算 hard top-1 与 raw soft fusion golden 指标。
9. 新增 `docs/refactor/prediction_cache_expert_provider.md`，记录 API、reader 关系、metadata、明确不做范围和后续接入顺序。
10. 更新 `docs/refactor/stage1_refactor_roadmap.md`、`docs/refactor/stage1_target_architecture.md` 和 `WORKSPACE_STRUCTURE.md`，登记 P6a 当前状态和新增文件职责。
11. 单独运行新增 provider smoke，确认通过。

## 结果

完整验收命令均已在 `quito` 环境通过：

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

新增 provider smoke 结果确认：

- provider 拒绝空 sample_keys 和重复 sample_key，不默认扫描全量 manifest。
- `ExpertBatch.sample_keys` 按 golden sample_keys 保序。
- `ExpertBatch.model_columns` 与固定五专家顺序一致。
- `y_pred=(4, 5, 48, 1)`，`y_true=(4, 48, 1)`。
- `row_index_metadata` 保留 packed_npy_v1 的 `y_true_row_index` / `y_pred_row_index` 信息。
- `extra` 包含 provider name、`array_storage=packed_npy_v1` 和轻量 reader metadata。
- hard top-1 MAE 为 `0.416048437`，raw soft fusion MAE 为 `0.410296679`，与 golden smoke 口径一致。

本次没有修改 `PredictionBatchReader` 行为，没有移动 `prediction_array_io`，没有访问 `/data2`，没有创建 run_dir，没有写 status/metadata，没有实现 config/runtime/launcher，没有新增 Bash 或 `scripts/` entrypoint，没有修改 Visual Router / TimeFuse fusor 正式入口，也没有改变模型结构、loss 或正式输出目录。

## 结论

`PredictionCacheExpertProvider` 已作为 P6a smoke-only ExpertProvider adapter 落地。它只负责把 prediction cache reader 输出包装为 `ExpertBatch`，并通过显式 sample_keys 避免默认全量 manifest 扫描。后续正式入口迁移仍需继续小步推进，不应直接把该 provider 接入 full-scale 训练入口。

## 下一步方案

提交并推送到远程 `refactor/stage1-route-audit` 分支。后续建议新增 evaluator adapter smoke，再考虑最小 config skeleton。
