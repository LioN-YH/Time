# Stage 1 P10b Prediction SQLite Backend

日志日期：2026-06-20 03:39:35 CST

## 1. 目标

本文记录 Stage 1 P10b 最小 shared prediction SQLite backend helper。该 helper 只覆盖小规模 fixture/smoke 能力，用于验证 shared prediction index prepare / fetch / metadata 边界，不接 Visual Router 或 TimeFuse-style fusor 正式入口。

新增实现：

- `time_router/io/prediction_sqlite_backend.py`
- `tests/smoke/stage1_prediction_sqlite_backend_smoke.py`

## 2. API

核心入口：

```python
build_prediction_sqlite_backend(
    manifest_path=...,
    target_sample_keys=...,
    index_db_path=...,
    model_columns=...,
    chunk_read_rows=...,
)
```

输入由调用方显式提供：

- `manifest_path`
- `target_sample_keys`
- `index_db_path`
- `model_columns`
- `chunk_read_rows`

返回 `PreparedPredictionSQLiteBackend`，提供：

- `fetch_records(sample_keys)`
- `metadata`
- `close()`

辅助函数：

- `load_prediction_sqlite_backend(index_db_path)`：从 P10b helper 写出的 SQLite metadata 恢复 prepared backend。
- `records_to_ordered_rows(records, sample_keys, model_columns)`：按调用方输入顺序恢复稳定 record 列表。

## 3. SQLite Schema

`prediction_index` 使用 `(sample_key, model_name)` 作为主键，字段包括：

- `sample_key`
- `model_name`
- `y_true_path`
- `y_pred_path`
- `mae`
- `mse`
- `array_storage`
- `y_true_row_index`
- `y_pred_row_index`

`index_metadata` 记录：

- `target_sample_keys`
- `expected_records`
- `actual_records`
- `chunk_read_rows`
- `model_columns`
- `manifest_path`
- `manifest_dir`
- `index_db_path`
- `created_at`
- `missing_report`

## 4. 构建与失败语义

helper 使用 pandas `chunksize=chunk_read_rows` 分块扫描 manifest，只把 `target_sample_keys` 且 `model_name in model_columns` 的记录写入 SQLite，不构建全量 Python lookup。

SQLite 文件先写到同目录临时路径，成功后通过 `os.replace(...)` 原子替换 `index_db_path`。构建失败时会关闭连接并删除临时 SQLite 文件；默认 `allow_missing=False` 时，缺失任一 sample/model record 会报错，不留下目标 SQLite 半成品。

可选 `allow_missing=True` 只用于 smoke/审计场景：缺失记录会写入 `missing_report`，但调用方仍需自行决定是否允许继续。

## 5. Fetch 与数组读取边界

`fetch_records(sample_keys)` 只查询当前 batch 的 records，并把 manifest 中相对数组路径解析到 `manifest_dir` 下的可读路径。返回值是 `(sample_key, model_name) -> record` 字典；调用方需要使用 `records_to_ordered_rows(...)` 或等价逻辑按输入 `sample_keys + model_columns` 恢复顺序。

helper 不直接读取数组。smoke 使用已有 `visual_router_experiments.common.prediction_array_io.load_prediction_arrays_grouped(...)` 复原 `packed_npy_v1` 的 `y_pred/y_true`，以验证 row index lineage 和 grouped mmap 读取边界。

## 6. Smoke 覆盖

`tests/smoke/stage1_prediction_sqlite_backend_smoke.py` 在临时目录构造 4 个 sample、5 个 model 的 packed fixture，覆盖：

- 构建 SQLite 子集索引；
- `target_sample_keys` metadata 保序；
- `expected_records` / `actual_records`；
- `index_metadata` 字段完整性；
- `fetch_records(...)` 后按显式 `sample_keys + model_columns` 恢复顺序；
- 用 grouped packed loading 读回 `y_pred/y_true`；
- 校验 `y_true_row_index` / `y_pred_row_index` lineage 和数组 shape；
- 默认缺失 sample/model record 报错；
- `allow_missing=True` 写入 missing report；
- 不访问 `/data2`，不运行正式入口。

已执行：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile time_router/io/prediction_sqlite_backend.py tests/smoke/stage1_prediction_sqlite_backend_smoke.py
```

结果：通过。

## 7. 明确不做

P10b 不做以下事项：

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不接正式入口。
- 不替换 Visual Router / TimeFuse 现有 SQLite index。
- 不改 `PredictionBatchReader`。
- 不改 `PredictionCacheExpertProvider`。
- 不改 `EvaluationInputAdapter`。
- 不新增 provider/head/runtime 代码。
- 不新增 Bash/scripts。
- 不访问 `/data2`。
- 不启动 pressure/full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler 或 checkpoint/resume。

## 8. 后续

P10b 之后可以继续整理 launcher / run scripts 边界，或在 Stage 1.5 / Stage 2 评估让 `PredictionCacheExpertProvider` 消费 prepared backend。真正接入正式入口前，仍需分别做 Visual Router 和 TimeFuse-style fusor 的小步旁路验证，不能用本 smoke 代替 full-scale runtime 行为验证。
