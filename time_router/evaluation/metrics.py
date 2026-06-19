#!/usr/bin/env python3
"""
文件功能：
    提供 Stage 1 P3a/P3b 最小共享 fusion/metrics/diagnostics helper。

设计边界：
    - 只依赖 numpy，不引入 torch/sklearn 训练依赖；
    - 只抽取 hard top-1 fusion、raw soft fusion、MAE、MSE、router 权重诊断和必要输入校验；
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


def _validate_model_columns(model_columns: Sequence[str]) -> List[str]:
    """
    函数功能：
        校验专家列名并返回稳定的字符串列表。

    关键约束：
        model_columns 是 router/fusor 权重诊断的唯一专家顺序来源，因此必须非空且
        不允许重复，避免 selected counts 汇总时把两个专家合并到同一名称。
    """
    columns = [str(model_name) for model_name in model_columns]
    if not columns:
        raise ValueError("model_columns 不能为空")
    if len(set(columns)) != len(columns):
        raise ValueError(f"model_columns 存在重复专家名：{columns}")
    return columns


def _validate_weight_matrix(weights: np.ndarray) -> np.ndarray:
    """
    函数功能：
        校验 router/fusor 权重诊断输入，并返回 numpy 二维数组。

    关键约束：
        P3b diagnostics 只消费调用方显式传入的权重矩阵，不读取 manifest、
        prediction cache、oracle/TSF 或正式输出目录。
    """
    weight_arr = np.asarray(weights)
    if weight_arr.ndim != 2:
        raise ValueError(f"weights 必须是二维 [sample, expert]：actual={weight_arr.shape}")
    if weight_arr.shape[1] <= 0:
        raise ValueError(f"weights expert 维度必须大于 0：actual={weight_arr.shape}")
    if not np.all(np.isfinite(weight_arr)):
        raise ValueError("weights 包含 NaN 或 Inf")
    if np.any(weight_arr < 0):
        raise ValueError("weights 必须非负，才能计算 entropy/max-weight 诊断")
    return weight_arr


def compute_selected_counts(selected_indices: np.ndarray, model_columns: Sequence[str]) -> dict[str, int]:
    """
    函数功能：
        根据 hard top-1 选中的专家下标统计每个专家被选择的次数。

    输入：
        selected_indices: 一维 `[sample]` 专家下标数组。
        model_columns: 专家名称顺序，长度定义合法专家数。

    输出：
        按 model_columns 顺序构造的 `{model_name: count}` 字典，未被选中的专家计为 0。
    """
    columns = _validate_model_columns(model_columns)
    index_arr = np.asarray(selected_indices)
    if index_arr.ndim != 1:
        raise ValueError(f"selected_indices 必须是一维 [sample]：actual={index_arr.shape}")
    if not np.issubdtype(index_arr.dtype, np.integer):
        if not np.all(np.equal(index_arr, np.floor(index_arr))):
            raise ValueError("selected_indices 必须是整数专家下标")
        index_arr = index_arr.astype(np.int64)
    if index_arr.size > 0:
        if int(index_arr.min()) < 0 or int(index_arr.max()) >= len(columns):
            raise ValueError(
                f"selected_indices 越界：min={int(index_arr.min())} max={int(index_arr.max())} expert_count={len(columns)}"
            )
    bincount = np.bincount(index_arr.astype(np.int64), minlength=len(columns))
    return {model_name: int(bincount[index]) for index, model_name in enumerate(columns)}


def compute_weight_entropy(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    函数功能：
        计算每个 sample 的权重熵诊断值。

    输入：
        weights: 二维 `[sample, expert]` 权重矩阵。
        eps: log 计算的数值稳定项。

    输出：
        shape 为 `[sample]` 的 numpy 数组。这里不做 temperature/top-k/calibration，
        只按调用方传入的权重执行 `-sum(weights * log(weights + eps))`。
    """
    if eps <= 0:
        raise ValueError(f"eps 必须为正数：actual={eps}")
    weight_arr = _validate_weight_matrix(weights)
    return -np.sum(weight_arr * np.log(weight_arr + eps), axis=1)


def compute_max_weight(weights: np.ndarray) -> np.ndarray:
    """
    函数功能：
        计算每个 sample 的最大 router/fusor 权重。

    输入：
        weights: 二维 `[sample, expert]` 权重矩阵。

    输出：
        shape 为 `[sample]` 的 numpy 数组，等于每行 weights 的最大值。
    """
    weight_arr = _validate_weight_matrix(weights)
    return np.max(weight_arr, axis=1)


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
