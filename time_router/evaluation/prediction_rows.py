#!/usr/bin/env python3
"""
文件功能：
    提供 Stage 1 P3d 最小 per-sample evaluation rows builder。

设计边界：
    - 只依赖 numpy 和 Python 标准库；
    - 只消费调用方显式传入的 sample_keys、fusion result、y_true、weights 和 model_columns；
    - 不读取 manifest、prediction cache、oracle/TSF 或正式训练输出目录；
    - 不写 CSV/JSON/Parquet，不实现 calibration、oracle regret 或正式 output schema 迁移。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from time_router.evaluation.metrics import (
    FusionMetricsResult,
    compute_max_weight,
    compute_weight_entropy,
)


def _validate_rows_inputs(
    *,
    sample_keys: Sequence[str],
    model_columns: Sequence[str],
    hard_result: FusionMetricsResult,
    raw_soft_result: FusionMetricsResult,
    y_true: np.ndarray,
    weights: np.ndarray,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    函数功能：
        校验 P3d rows builder 的最小输入契约，并返回稳定化后的对象。

    关键约束：
        rows builder 只组合当前 batch 已经显式传入的数据；这里不回读外部
        manifest、prediction cache 或正式输出目录，避免产生新的 IO/schema 入口。
    """
    keys = [str(sample_key) for sample_key in sample_keys]
    columns = [str(model_name) for model_name in model_columns]
    if not columns:
        raise ValueError("model_columns 不能为空")
    if len(set(columns)) != len(columns):
        raise ValueError(f"model_columns 存在重复专家名：{columns}")

    true_arr = np.asarray(y_true)
    hard_pred = np.asarray(hard_result.fused_pred)
    raw_soft_pred = np.asarray(raw_soft_result.fused_pred)
    weight_arr = np.asarray(weights)

    if true_arr.ndim < 2:
        raise ValueError(f"y_true 必须至少包含 sample/target 两维：actual={true_arr.shape}")
    sample_count = true_arr.shape[0]
    if len(keys) != sample_count:
        raise ValueError(f"sample_keys 长度必须等于样本数：keys={len(keys)} samples={sample_count}")
    if hard_pred.shape != true_arr.shape:
        raise ValueError(f"hard_result.fused_pred shape 必须等于 y_true：hard={hard_pred.shape} y_true={true_arr.shape}")
    if raw_soft_pred.shape != true_arr.shape:
        raise ValueError(f"raw_soft_result.fused_pred shape 必须等于 y_true：raw_soft={raw_soft_pred.shape} y_true={true_arr.shape}")
    if weight_arr.shape != (sample_count, len(columns)):
        raise ValueError(f"weights shape 必须为 [sample, expert]：actual={weight_arr.shape} expected={(sample_count, len(columns))}")

    if hard_result.selected_indices is None:
        raise ValueError("hard_result 必须包含 selected_indices")
    selected_indices = np.asarray(hard_result.selected_indices)
    if selected_indices.ndim != 1:
        raise ValueError(f"hard_result.selected_indices 必须是一维 [sample]：actual={selected_indices.shape}")
    if selected_indices.shape[0] != sample_count:
        raise ValueError(
            "hard_result.selected_indices 样本数必须等于 y_true："
            f"selected={selected_indices.shape[0]} samples={sample_count}"
        )
    if not np.issubdtype(selected_indices.dtype, np.integer):
        if not np.all(np.equal(selected_indices, np.floor(selected_indices))):
            raise ValueError("hard_result.selected_indices 必须是整数专家下标")
        selected_indices = selected_indices.astype(np.int64)
    if selected_indices.size > 0:
        if int(selected_indices.min()) < 0 or int(selected_indices.max()) >= len(columns):
            raise ValueError(
                "hard_result.selected_indices 越界："
                f"min={int(selected_indices.min())} max={int(selected_indices.max())} expert_count={len(columns)}"
            )

    if raw_soft_result.selected_indices is not None or raw_soft_result.selected_models is not None:
        raise ValueError("raw_soft_result 应来自 raw soft fusion，不能包含 hard top-1 选择信息")
    if hard_result.selected_models is not None:
        selected_models = [columns[int(index)] for index in selected_indices]
        if list(hard_result.selected_models) != selected_models:
            raise ValueError(
                "hard_result.selected_models 与 selected_indices/model_columns 不一致："
                f"actual={hard_result.selected_models} expected={selected_models}"
            )

    return keys, columns, true_arr, hard_pred, raw_soft_pred, weight_arr


def _per_sample_mae(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """函数功能：按 sample 聚合除第 0 维外所有 target/channel 维度的 MAE。"""
    reduce_axes = tuple(range(1, y_true.ndim))
    return np.mean(np.abs(y_pred - y_true), axis=reduce_axes)


def _per_sample_mse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """函数功能：按 sample 聚合除第 0 维外所有 target/channel 维度的 MSE。"""
    reduce_axes = tuple(range(1, y_true.ndim))
    error = y_pred - y_true
    return np.mean(error**2, axis=reduce_axes)


def build_per_sample_fusion_rows(
    *,
    sample_keys: Sequence[str],
    model_columns: Sequence[str],
    hard_result: FusionMetricsResult,
    raw_soft_result: FusionMetricsResult,
    y_true: np.ndarray,
    weights: np.ndarray,
) -> list[dict[str, object]]:
    """
    函数功能：
        构造当前 batch 的逐样本 fusion evaluation rows。

    输入：
        sample_keys: 当前 batch 的样本 key，顺序即 rows 输出顺序。
        model_columns: 专家名称顺序。
        hard_result: `hard_top1_fusion(...)` 返回的结果对象。
        raw_soft_result: `raw_soft_fusion(...)` 返回的结果对象。
        y_true: 真实值数组，shape 必须等于两个 fused_pred。
        weights: 二维 `[sample, expert]` 权重矩阵。

    输出：
        `list[dict[str, object]]`，每行包含 sample_key、hard top-1 选择、
        逐样本 hard/raw-soft MAE/MSE、max weight 和 weight entropy。
    """
    keys, columns, true_arr, hard_pred, raw_soft_pred, weight_arr = _validate_rows_inputs(
        sample_keys=sample_keys,
        model_columns=model_columns,
        hard_result=hard_result,
        raw_soft_result=raw_soft_result,
        y_true=y_true,
        weights=weights,
    )
    selected_indices = np.asarray(hard_result.selected_indices, dtype=np.int64)
    hard_mae = _per_sample_mae(hard_pred, true_arr)
    hard_mse = _per_sample_mse(hard_pred, true_arr)
    raw_soft_mae = _per_sample_mae(raw_soft_pred, true_arr)
    raw_soft_mse = _per_sample_mse(raw_soft_pred, true_arr)
    max_weight = compute_max_weight(weight_arr)
    weight_entropy = compute_weight_entropy(weight_arr)

    rows: list[dict[str, object]] = []
    for row_index, sample_key in enumerate(keys):
        selected_index = int(selected_indices[row_index])
        rows.append(
            {
                "sample_key": sample_key,
                "selected_model": columns[selected_index],
                "selected_index": selected_index,
                "hard_mae": float(hard_mae[row_index]),
                "hard_mse": float(hard_mse[row_index]),
                "raw_soft_mae": float(raw_soft_mae[row_index]),
                "raw_soft_mse": float(raw_soft_mse[row_index]),
                "max_weight": float(max_weight[row_index]),
                "weight_entropy": float(weight_entropy[row_index]),
            }
        )
    return rows
