#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P9f Visual Router training loss ExpertBatch 旁路校验 smoke。

输入：
    只使用本文件内构造的小型 numpy arrays，不启动 ViT，不访问 `/data2`，
    不运行正式 Visual Router entrypoint。

输出：
    标准输出打印中文检查日志；若 ExpertBatch.y_pred/y_true 复算 expert_errors
    与 legacy expert_errors 不一致，或失败信息缺少定位上下文，则抛出
    AssertionError。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.stage1_vali_test_router.fusion_utils import MODEL_COLUMNS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import (  # noqa: E402
    verify_training_expert_errors_from_expert_batch,
)


def _build_arrays() -> tuple[np.ndarray, np.ndarray]:
    """
    函数功能：
        构造两个样本、五个专家、三步单通道的最小 training prediction 张量。

    关键约束：
        数值刻意覆盖正负误差与完美专家，便于同时验证 MAE/MSE 复算。
    """
    y_true = np.asarray(
        [
            [[1.0], [2.0], [3.0]],
            [[2.0], [0.0], [-1.0]],
        ],
        dtype=np.float32,
    )
    y_pred = np.asarray(
        [
            [
                [[1.1], [2.2], [2.9]],
                [[0.8], [2.1], [3.3]],
                [[1.0], [2.0], [3.0]],
                [[1.5], [1.5], [2.5]],
                [[0.5], [2.5], [3.5]],
            ],
            [
                [[1.5], [0.2], [-1.2]],
                [[2.1], [-0.1], [-0.9]],
                [[2.4], [0.4], [-0.5]],
                [[1.9], [0.1], [-1.1]],
                [[2.0], [0.0], [-1.0]],
            ],
        ],
        dtype=np.float32,
    )
    return y_pred, y_true


def _compute_errors(y_pred: np.ndarray, y_true: np.ndarray, metric: str) -> np.ndarray:
    """
    函数功能：
        用测试内独立公式生成 legacy expert_errors，作为 helper 的对照输入。
    """
    diff = y_pred - y_true[:, None, ...]
    reduce_axes = tuple(range(2, diff.ndim))
    if metric == "mae":
        return np.mean(np.abs(diff), axis=reduce_axes, dtype=np.float32).astype(np.float32)
    if metric == "mse":
        return np.mean(diff ** 2, axis=reduce_axes, dtype=np.float32).astype(np.float32)
    raise ValueError(f"未知 metric={metric}")


def _assert_message_contains(message: str, fragments: list[str]) -> None:
    """函数功能：确认失败信息包含后续定位 training batch 所需的关键字段。"""
    missing = [fragment for fragment in fragments if fragment not in message]
    if missing:
        raise AssertionError(f"失败信息缺少定位字段：missing={missing}\nmessage={message}")


def run_smoke() -> None:
    """函数功能：执行 P9f Visual Router training ExpertBatch 旁路 smoke。"""
    output_dir = REPO_ROOT / "experiment_logs" / "run_outputs" / "p9f_training_expert_batch_bypass_smoke_not_created"
    if str(output_dir).startswith("/data2/"):
        raise AssertionError("P9f smoke 不应访问 /data2")

    sample_keys = ["p9f_sample_0", "p9f_sample_1"]
    y_pred, y_true = _build_arrays()

    for metric in ["mae", "mse"]:
        legacy_errors = _compute_errors(y_pred, y_true, metric)
        verify_training_expert_errors_from_expert_batch(
            sample_keys=sample_keys,
            y_pred=y_pred,
            y_true=y_true,
            legacy_expert_errors=legacy_errors,
            model_columns=MODEL_COLUMNS,
            error_metric=metric,
            training_batch_index=11,
            epoch=3,
            output_dir=output_dir,
            atol=1e-6,
            rtol=1e-6,
        )
        print(f"通过：{metric} expert_errors 可由 ExpertBatch.y_pred/y_true 显式复算且与 legacy 一致")

    bad_errors = _compute_errors(y_pred, y_true, "mae")
    bad_errors[1, 4] = float(bad_errors[1, 4]) + 0.25
    try:
        verify_training_expert_errors_from_expert_batch(
            sample_keys=sample_keys,
            y_pred=y_pred,
            y_true=y_true,
            legacy_expert_errors=bad_errors,
            model_columns=MODEL_COLUMNS,
            error_metric="mae",
            training_batch_index=12,
            epoch=4,
            output_dir=output_dir,
            atol=1e-6,
            rtol=1e-6,
        )
    except AssertionError as exc:
        message = str(exc)
        _assert_message_contains(
            message,
            [
                "phase=training",
                "router_mode=fusion_huber_kl",
                "metric=mae",
                "batch_index=12",
                "training_batch_index=12",
                "sample_key=p9f_sample_1",
                f"model_name={MODEL_COLUMNS[4]}",
                "expert_index=4",
                "old_value=",
                "legacy_value=",
                "expert_batch_value=",
                "recomputed_value=",
                f"output_dir={output_dir}",
            ],
        )
        print("通过：故意 mismatch 时失败信息包含 phase/router_mode/metric/sample/model/value/output_dir 定位上下文")
    else:
        raise AssertionError("故意制造的 expert_errors mismatch 未触发 AssertionError")


if __name__ == "__main__":
    run_smoke()
