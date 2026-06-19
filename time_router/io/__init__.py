"""
文件功能：
    `time_router` 共享 IO 工具包。
"""

from time_router.io.json_utils import atomic_write_json, build_status_payload, write_status_json
from time_router.io.path_resolver import (
    find_repo_root,
    resolve_metadata_path,
    resolve_status_path,
    resolve_under_root,
)
from time_router.io.prediction_cache_reader import DEFAULT_MODEL_COLUMNS, PredictionBatch, PredictionBatchReader
from time_router.io.run_metadata import build_run_metadata, write_run_metadata

__all__ = [
    "DEFAULT_MODEL_COLUMNS",
    "PredictionBatch",
    "PredictionBatchReader",
    "atomic_write_json",
    "build_run_metadata",
    "build_status_payload",
    "find_repo_root",
    "resolve_metadata_path",
    "resolve_status_path",
    "resolve_under_root",
    "write_run_metadata",
    "write_status_json",
]
