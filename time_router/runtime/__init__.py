"""
文件功能：
    Stage 1 canonical Runtime artifact 写出 helper 的 public API。

边界说明：
    本包只负责 Runtime 层显式传入 run_dir 后的最小磁盘 artifact 写出；
    Provider、Head、Evaluator 不应从这里读取或推导 run_dir。
"""

from time_router.runtime.artifact_writer import (
    CANONICAL_RUN_SUBDIRS,
    create_run_dir,
    write_evaluation_summary,
    write_json_atomic,
    write_prediction_rows_csv,
    write_run_metadata,
    write_run_status,
    write_sample_manifest_ref,
    write_split_summary,
)

__all__ = [
    "CANONICAL_RUN_SUBDIRS",
    "create_run_dir",
    "write_evaluation_summary",
    "write_json_atomic",
    "write_prediction_rows_csv",
    "write_run_metadata",
    "write_run_status",
    "write_sample_manifest_ref",
    "write_split_summary",
]
