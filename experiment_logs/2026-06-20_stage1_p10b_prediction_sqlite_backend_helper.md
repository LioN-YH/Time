# Stage 1 P10b Prediction SQLite Backend Helper

日志日期：2026-06-20 03:39:35 CST

## 目的

新增一个最小 shared prediction SQLite backend smoke helper，验证 Stage 1 prediction manifest 子集索引、batch fetch records、metadata、missing report 和 packed row index lineage。该步骤只实现小规模 fixture/smoke 能力，不接 Visual Router 或 TimeFuse-style fusor 正式入口。

## 背景

P10a 已完成 shared prediction SQLite backend 审计，结论是 SQLite backend 属于 full-scale prediction backend implementation，不是 framework provider interface。可共享范围应限制在 manifest chunk scan、调用方传入的 target sample_keys、SQLite 子集索引、batch fetch records、packed row index lineage、grouped mmap loading、index metadata 和 atomic replace / cleanup。

本轮必须保持以下边界：不修改 `train_visual_router_online_streaming.py`、`train_timefuse_fusor_streaming.py`、`launch_timefuse_fusor_full_scale.py`；不接正式入口；不替换现有正式 SQLite index；不改 `PredictionBatchReader`、`PredictionCacheExpertProvider` 或 `EvaluationInputAdapter`；不访问 `/data2`；不启动 pressure/full-scale。

## 操作

1. 新增 `time_router/io/prediction_sqlite_backend.py`。
   - 实现 `build_prediction_sqlite_backend(...)`，接收 `manifest_path`、`target_sample_keys`、`index_db_path`、`model_columns` 和 `chunk_read_rows`。
   - SQLite `prediction_index` 使用 `(sample_key, model_name)` 主键，记录 `y_true_path`、`y_pred_path`、`mae`、`mse`、`array_storage`、`y_true_row_index` 和 `y_pred_row_index`。
   - 分块读取 manifest，只写目标 sample/model records，不加载全量 manifest 到 Python 内存。
   - 使用同目录临时 SQLite 文件构建，成功后 `os.replace(...)` 原子替换；异常时关闭连接并删除临时文件。
   - 实现 `PreparedPredictionSQLiteBackend.fetch_records(...)`、`load_prediction_sqlite_backend(...)` 和 `records_to_ordered_rows(...)`。
   - metadata 记录 target keys、expected/actual records、chunk rows、model columns、manifest/index path、created_at 和 missing sample/model report。

2. 更新 `time_router/io/__init__.py`，导出 P10b helper public API。

3. 新增 `tests/smoke/stage1_prediction_sqlite_backend_smoke.py`。
   - 在 tempfile 下构造 4 个 sample、5 个 model 的 `packed_npy_v1` manifest 和 `.npy` 数组。
   - 构建 SQLite index，并用显式乱序 sample_keys 查询 records。
   - 用 `records_to_ordered_rows(...)` 校验 `sample_keys + model_columns` 恢复顺序。
   - 用已有 `load_prediction_arrays_grouped(...)` 读回 `y_true/y_pred`，校验 shape、数值和 row index lineage。
   - 覆盖缺失 sample/model 默认报错，以及 `allow_missing=True` 时写入 missing report。
   - 明确检查不访问 `/data2`。

4. 新增 `docs/refactor/prediction_sqlite_backend.md`，记录 API、SQLite schema、metadata、构建失败语义、fetch 边界、smoke 覆盖和明确不做范围。

5. 更新以下文档：
   - `docs/refactor/stage1_refactor_roadmap.md`
   - `docs/refactor/stage1_entrypoint_migration_plan.md`
   - `docs/refactor/shared_prediction_sqlite_backend_audit.md`
   - `WORKSPACE_STRUCTURE.md`

6. 已先执行新增 smoke 与语法检查：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile time_router/io/prediction_sqlite_backend.py tests/smoke/stage1_prediction_sqlite_backend_smoke.py
   ```

7. 完整执行用户指定验收命令：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_cache_expert_provider_smoke.py
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_training_expert_batch_bypass_smoke.py
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router
   ```

## 结果

新增 smoke 已通过：

- 完整 fixture 可构建 SQLite 子集索引、fetch records，并按 row index 读回 packed arrays。
- 缺失 sample/model 默认报错。
- `allow_missing=True` 时 metadata 写入 missing report。

完整验收命令均已通过：

- P10b SQLite backend smoke 通过。
- P6a PredictionCacheExpertProvider smoke 通过。
- P9f Visual Router training ExpertBatch bypass smoke 通过。
- P7c TimeFuse protocol chain smoke 通过。
- `compileall` 覆盖 `time_router`、`tests/smoke` 和 `visual_router_experiments/stage1_vali_test_router` 通过。

本轮未修改 Visual Router / TimeFuse 正式入口，未新增 Bash/scripts，未访问 `/data2`，未启动 pressure/full-scale，未修改正式 CSV、summary、metadata、status、checkpoint、loss、optimizer、scaler 或 resume 口径。

## 结论

P10b 最小 shared prediction SQLite backend helper 已按 P10a 边界落地为 smoke-only 实现。它可作为后续 runtime/index prepare 的底层候选，但不是 provider，也不代表 `PredictionCacheExpertProvider` 已接入正式入口。

## 下一步方案

1. 提交并 push 到 `refactor/stage1-route-audit`。
2. 后续 P10c 可整理 launcher / run scripts 边界；provider prepared backend 接入仍推迟到 Stage 1.5 / Stage 2。
