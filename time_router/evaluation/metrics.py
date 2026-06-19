#!/usr/bin/env python3
"""
文件功能：
    提供 Stage 1 P3a 最小共享 fusion/metrics helper。

设计边界：
    - 只依赖 numpy，不引入 torch/sklearn 训练依赖；
    - 只抽取 hard top-1 fusion、raw soft fusion、MAE、MSE 和必要输入校验；
    - 不实现 calibration，不改变报告 schema，不迁移正式 Visual Router / TimeFuse 入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class FusionMetricsResult:
    """
    类功能：
        承载 hard top-1 或 raw soft fusion 的数组级复算结果。

    字段说明：
        selected_indices: hard top-1 选中的专家下标；raw soft fusion 为 None。
        selected_models: hard top-1 选中的专家名；raw soft fusion 为 None。
        fused_pred: 融合后的预测数组，shape 与 y_true 一致。
        mae: fused_pred 与 y_true 的全元素 MAE。
        mse: fused_pred 与 y_true 的全元素 MSE。
    """

    selected_indices: Optional[np.ndarray]
    selected_models: Optional[List[str]]
    fused_pred: np.ndarray
    mae: float
    mse: float


def compute_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    函数功能：
        计算两个同形数组之间的全元素 MAE。

    输入：
        y_pred: 预测数组。
        y_true: 真实值数组，shape 必须与 y_pred 一致。

    输出：
        Python float 类型 MAE。
    """
    pred_arr = np.asarray(y_pred)
    true_arr = np.asarray(y_true)
    if pred_arr.shape != true_arr.shape:
        raise ValueError(f"MAE 输入 shape 不一致：y_pred={pred_arr.shape} y_true={true_arr.shape}")
    return float(np.mean(np.abs(pred_arr - true_arr)))


def compute_mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    函数功能：
        计算两个同形数组之间的全元素 MSE。

    输入：
        y_pred: 预测数组。
        y_true: 真实值数组，shape 必须与 y_pred 一致。

    输出：
        Python float 类型 MSE。
    """
    pred_arr = np.asarray(y_pred)
    true_arr = np.asarray(y_true)
    if pred_arr.shape != true_arr.shape:
        raise ValueError(f"MSE 输入 shape 不一致：y_pred={pred_arr.shape} y_true={true_arr.shape}")
    error = pred_arr - true_arr
    return float(np.mean(error**2))


def validate_fusion_inputs(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    weights: np.ndarray,
    model_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    函数功能：
        校验 Stage 1 fusion helper 的公共输入，并返回 numpy 化后的数组。

    关键约束：
        y_pred 的 shape 必须为 `(sample, expert, ...)`；y_true 必须为
        `(sample, ...)`；weights 必须为 `(sample, expert)`；model_columns 的长度
        必须等于 expert 维度。这里不强制归一化 weights，保留上游既有 raw soft
        fusion 口径。
    """
    pred_arr = np.asarray(y_pred)
    true_arr = np.asarray(y_true)
    weight_arr = np.asarray(weights)
    columns = [str(model_name) for model_name in model_columns]

    if pred_arr.ndim < 3:
        raise ValueError(f"y_pred 必须至少包含 sample/expert/target 三维：actual={pred_arr.shape}")
    sample_count = pred_arr.shape[0]
    expert_count = pred_arr.shape[1]
    if true_arr.shape != (sample_count, *pred_arr.shape[2:]):
        raise ValueError(f"y_true shape 与 y_pred 不匹配：y_pred={pred_arr.shape} y_true={true_arr.shape}")
    if weight_arr.shape != (sample_count, expert_count):
        raise ValueError(f"weights shape 必须为 (sample, expert)：actual={weight_arr.shape} expected={(sample_count, expert_count)}")
    if len(columns) != expert_count:
        raise ValueError(f"model_columns 长度必须等于专家数：actual={len(columns)} expected={expert_count}")
    if len(set(columns)) != len(columns):
        raise ValueError(f"model_columns 存在重复专家名：{columns}")
    return pred_arr, true_arr, weight_arr, columns


def hard_top1_fusion(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    weights: np.ndarray,
    model_columns: Sequence[str],
) -> FusionMetricsResult:
    """
    函数功能：
        根据 weights 的 argmax 执行 hard top-1 fusion，并复算 MAE/MSE。

    输出：
        FusionMetricsResult，其中包含 selected_indices、selected_models、
        fused_pred、mae 和 mse。
    """
    pred_arr, true_arr, weight_arr, columns = validate_fusion_inputs(y_pred, y_true, weights, model_columns)
    selected_indices = weight_arr.argmax(axis=1)
    selected_models = [columns[int(index)] for index in selected_indices]
    fused_pred = pred_arr[np.arange(pred_arr.shape[0]), selected_indices]
    return FusionMetricsResult(
        selected_indices=selected_indices,
        selected_models=selected_models,
        fused_pred=fused_pred,
        mae=compute_mae(fused_pred, true_arr),
        mse=compute_mse(fused_pred, true_arr),
    )


def raw_soft_fusion(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    weights: np.ndarray,
    model_columns: Sequence[str],
) -> FusionMetricsResult:
    """
    函数功能：
        执行当前 golden smoke 使用的 raw soft fusion 线性加权口径。

    说明：
        本函数只做 `sum_expert(weights * y_pred)`，不做 temperature scaling、
        top-k 截断或额外归一化，以保持 P3a 的最小抽取边界。
    """
    pred_arr, true_arr, weight_arr, _columns = validate_fusion_inputs(y_pred, y_true, weights, model_columns)
    reshape_dims = (weight_arr.shape[0], weight_arr.shape[1], *([1] * (pred_arr.ndim - 2)))
    fused_pred = (weight_arr.reshape(reshape_dims) * pred_arr).sum(axis=1)
    return FusionMetricsResult(
        selected_indices=None,
        selected_models=None,
        fused_pred=fused_pred,
        mae=compute_mae(fused_pred, true_arr),
        mse=compute_mse(fused_pred, true_arr),
    )
