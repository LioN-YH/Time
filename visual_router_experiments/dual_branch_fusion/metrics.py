#!/usr/bin/env python3
"""
文件功能：
    PatchTST baseline 与 PatchTST+Visual dual-branch 预测实验的最小指标工具。

输入：
    numpy 数组形式的预测值和真实值，shape 必须完全一致。

输出：
    MAE/MSE 与 baseline-vs-dual 对比字典。
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """函数功能：计算全元素 MAE，并显式检查预测和真实值 shape。"""
    pred = np.asarray(y_pred)
    true = np.asarray(y_true)
    if pred.shape != true.shape:
        raise ValueError(f"MAE 输入 shape 不一致：y_pred={pred.shape} y_true={true.shape}")
    return float(np.mean(np.abs(pred - true)))


def compute_mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """函数功能：计算全元素 MSE，并显式检查预测和真实值 shape。"""
    pred = np.asarray(y_pred)
    true = np.asarray(y_true)
    if pred.shape != true.shape:
        raise ValueError(f"MSE 输入 shape 不一致：y_pred={pred.shape} y_true={true.shape}")
    error = pred - true
    return float(np.mean(error * error))


def build_comparison_metrics(
    *,
    y_patchtst: np.ndarray,
    y_dual_branch: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, object]:
    """
    函数功能：
        生成验收要求的 PatchTST baseline 与 dual-branch 对比指标。

    关键约束：
        `delta_*` 定义为 PatchTST 指标减去 dual-branch 指标；正数表示双分支更好。
    """
    patchtst_mae = compute_mae(y_patchtst, y_true)
    patchtst_mse = compute_mse(y_patchtst, y_true)
    dual_mae = compute_mae(y_dual_branch, y_true)
    dual_mse = compute_mse(y_dual_branch, y_true)
    delta_mae = patchtst_mae - dual_mae
    delta_mse = patchtst_mse - dual_mse
    return {
        "patchtst_mae": patchtst_mae,
        "patchtst_mse": patchtst_mse,
        "dual_branch_mae": dual_mae,
        "dual_branch_mse": dual_mse,
        "delta_mae_vs_patchtst": delta_mae,
        "delta_mse_vs_patchtst": delta_mse,
        "beats_patchtst_mae": bool(delta_mae > 0.0),
        "beats_patchtst_mse": bool(delta_mse > 0.0),
    }
