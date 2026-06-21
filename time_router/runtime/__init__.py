"""
文件功能：
    Stage 1 canonical Runtime helper 的 public API。

边界说明：
    本包只负责 Runtime 层显式传入 run_dir 后的最小磁盘 artifact 写出；
    以及 P16i 显式 checkpoint path/payload 到 legacy VisualMLPRouter state_dict
    strict load 的最小边界。Provider、Head、Evaluator 不应从这里读取或推导
    run_dir，也不应接收 checkpoint path。
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
from time_router.runtime.visual_mlp_checkpoint import (
    extract_router_state_dict,
    load_checkpoint_payload,
    load_router_state_dict,
    strip_dataparallel_prefix,
)

__all__ = [
    "CANONICAL_RUN_SUBDIRS",
    "create_run_dir",
    "extract_router_state_dict",
    "load_checkpoint_payload",
    "load_router_state_dict",
    "strip_dataparallel_prefix",
    "write_evaluation_summary",
    "write_json_atomic",
    "write_prediction_rows_csv",
    "write_run_metadata",
    "write_run_status",
    "write_sample_manifest_ref",
    "write_split_summary",
]
