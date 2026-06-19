#!/usr/bin/env python3
"""
文件功能：
    提供 Stage 1 P3c 最小 evaluation summary builder。

设计边界：
    - 只依赖 numpy 和 Python 标准库；
    - 只汇总调用方显式传入的 fusion result、weights 和 model_columns；
    - 不读取 manifest、prediction cache、oracle/TSF 或正式训练输出目录；
    - 不实现 calibration、oracle regret、comparison 或正式 output schema 迁移。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from time_router.evaluation.metrics import (
    FusionMetricsResult,
    compute_max_weight,
    compute_selected_counts,
    compute_weight_entropy,
)


def _validate_summary_inputs(
    *,
    model_columns: Sequence[str],
    hard_result: FusionMetricsResult,
    raw_soft_result: FusionMetricsResult,
    weights: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """
    函数功能：
        校验 P3c summary builder 的最小输入契约，并返回稳定列名与权重数组。

    关键约束：
        summary 的专家顺序和样本数只来自显式传入对象；这里不回读任何外部
        manifest 或输出目录，避免把 summary builder 变成新的 IO 入口。
    """
    columns = [str(model_name) for model_name in model_columns]
    if not columns:
        raise ValueError("model_columns 不能为空")
    if len(set(columns)) != len(columns):
        raise ValueError(f"model_columns 存在重复专家名：{columns}")

    weight_arr = np.asarray(weights)
    if weight_arr.ndim != 2:
        raise ValueError(f"weights 必须是二维 [sample, expert]：actual={weight_arr.shape}")
    if weight_arr.shape[1] != len(columns):
        raise ValueError(f"weights expert 维度与 model_columns 不一致：weights={weight_arr.shape} columns={len(columns)}")

    if hard_result.selected_indices is None:
        raise ValueError("hard_result 必须包含 selected_indices")
    selected_indices = np.asarray(hard_result.selected_indices)
    if selected_indices.ndim != 1:
        raise ValueError(f"hard_result.selected_indices 必须是一维 [sample]：actual={selected_indices.shape}")
    if selected_indices.shape[0] != weight_arr.shape[0]:
        raise ValueError(
            "hard_result.selected_indices 样本数与 weights 不一致："
            f"selected={selected_indices.shape[0]} weights={weight_arr.shape[0]}"
        )
    if raw_soft_result.selected_indices is not None or raw_soft_result.selected_models is not None:
        raise ValueError("raw_soft_result 应来自 raw soft fusion，不能包含 hard top-1 选择信息")

    return columns, weight_arr


def build_fusion_summary(
    *,
    model_columns: Sequence[str],
    hard_result: FusionMetricsResult,
    raw_soft_result: FusionMetricsResult,
    weights: np.ndarray,
) -> dict[str, object]:
    """
    函数功能：
        汇总 hard top-1、raw soft fusion 和 router/fusor 权重诊断指标。

    输入：
        model_columns: 专家名称顺序。
        hard_result: `hard_top1_fusion(...)` 返回的结果对象。
        raw_soft_result: `raw_soft_fusion(...)` 返回的结果对象。
        weights: 二维 `[sample, expert]` 权重矩阵。

    输出：
        稳定 summary dict，字段覆盖 hard/raw-soft MAE/MSE、selected counts、
        平均 entropy、平均 max weight、样本数、专家数和专家顺序。
    """
    columns, weight_arr = _validate_summary_inputs(
        model_columns=model_columns,
        hard_result=hard_result,
        raw_soft_result=raw_soft_result,
        weights=weights,
    )
    entropy = compute_weight_entropy(weight_arr)
    max_weight = compute_max_weight(weight_arr)
    selected_counts = compute_selected_counts(hard_result.selected_indices, columns)

    return {
        "hard_mae": float(hard_result.mae),
        "hard_mse": float(hard_result.mse),
        "raw_soft_mae": float(raw_soft_result.mae),
        "raw_soft_mse": float(raw_soft_result.mse),
        "selected_counts": selected_counts,
        "mean_entropy": float(np.mean(entropy)),
        "mean_max_weight": float(np.mean(max_weight)),
        "num_samples": int(weight_arr.shape[0]),
        "num_experts": int(weight_arr.shape[1]),
        "model_columns": columns,
    }
