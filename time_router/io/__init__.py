"""
文件功能：
    `time_router` 共享 IO 工具包。

边界说明：
    本入口只聚合稳定 public API，供后续正式入口优先从 `time_router.io`
    导入。这里不执行配置读取、路径探测、输出目录创建或任何训练相关副作用。
"""

from time_router.io.json_utils import atomic_write_json, build_status_payload, write_status_json
from time_router.io.path_resolver import (
    find_repo_root,
    resolve_metadata_path,
    resolve_status_path,
    resolve_under_root,
)
from time_router.io.prediction_array_io import (
    PACKED_NPY_STORAGE,
    PER_SAMPLE_NPY_STORAGE,
    load_prediction_array,
    load_prediction_arrays_grouped,
    resolve_cache_array_path,
)
from time_router.io.prediction_cache_reader import DEFAULT_MODEL_COLUMNS, PredictionBatch, PredictionBatchReader
from time_router.io.prediction_sqlite_backend import (
    PreparedPredictionSQLiteBackend,
    PredictionSQLiteBackendMetadata,
    build_prediction_sqlite_backend,
    load_prediction_sqlite_backend,
    records_to_ordered_rows,
)
from time_router.io.run_metadata import build_run_metadata, write_run_metadata

__all__ = [
    "DEFAULT_MODEL_COLUMNS",
    "PACKED_NPY_STORAGE",
    "PER_SAMPLE_NPY_STORAGE",
    "PreparedPredictionSQLiteBackend",
    "PredictionBatch",
    "PredictionBatchReader",
    "PredictionSQLiteBackendMetadata",
    "atomic_write_json",
    "build_prediction_sqlite_backend",
    "build_run_metadata",
    "build_status_payload",
    "find_repo_root",
    "load_prediction_array",
    "load_prediction_arrays_grouped",
    "load_prediction_sqlite_backend",
    "records_to_ordered_rows",
    "resolve_cache_array_path",
    "resolve_metadata_path",
    "resolve_status_path",
    "resolve_under_root",
    "write_run_metadata",
    "write_status_json",
]
