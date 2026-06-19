"""
文件功能：
    `time_router` 共享评估工具包入口。

说明：
    当前只暴露 Stage 1 P3a/P3b/P3c/P3d 的最小 numpy fusion/metrics/diagnostics/summary/rows helper；
    正式训练入口、calibration 和报告 schema 尚未迁移。
"""

from time_router.evaluation.metrics import (
    FusionMetricsResult,
    compute_mae,
    compute_max_weight,
    compute_mse,
    compute_selected_counts,
    compute_weight_entropy,
    hard_top1_fusion,
    raw_soft_fusion,
    validate_fusion_inputs,
)
from time_router.evaluation.prediction_rows import build_per_sample_fusion_rows
from time_router.evaluation.summary import build_fusion_summary

__all__ = [
    "FusionMetricsResult",
    "build_per_sample_fusion_rows",
    "build_fusion_summary",
    "compute_mae",
    "compute_max_weight",
    "compute_mse",
    "compute_selected_counts",
    "compute_weight_entropy",
    "hard_top1_fusion",
    "raw_soft_fusion",
    "validate_fusion_inputs",
]
