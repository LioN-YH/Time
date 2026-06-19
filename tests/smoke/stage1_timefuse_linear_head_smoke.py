#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P7b TimeFuseLinearSoftmaxHead smoke。

输入：
    使用测试内固定 FeatureBatch、小矩阵权重和 bias。

输出：
    标准输出打印中文检查日志；任一 head contract 漂移时抛出 AssertionError。

关键约束：
    该 smoke 不训练、不算 loss、不建 optimizer、不保存 checkpoint；不读取
    prediction cache、oracle/TSF、feature CSV，不访问 /data2，不创建 run_dir，
    不写 status/metadata/CSV/JSON/Parquet，不接正式 TimeFuse fusor 或
    Visual Router 入口。
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.models import TimeFuseLinearSoftmaxHead  # noqa: E402
from time_router.protocols import FeatureBatch, RouterOutput  # noqa: E402


SAMPLE_KEYS = ("sample_b", "sample_a", "sample_c")
MODEL_COLUMNS = ("DLinear", "PatchTST", "CrossFormer")
RUN_OUTPUTS_ROOT = REPO_ROOT / "experiment_logs" / "run_outputs"


def snapshot_run_outputs() -> set[str]:
    """函数功能：记录 run_outputs 一层目录名，用于检查 head 不创建输出目录。"""
    if not RUN_OUTPUTS_ROOT.exists():
        return set()
    return {path.name for path in RUN_OUTPUTS_ROOT.iterdir()}


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """函数功能：在 smoke 中独立复算逐样本 softmax，作为 deterministic oracle。"""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def fail_file_access(*args: object, **kwargs: object) -> object:
    """函数功能：阻断 head 阶段任何文件读取/写入。"""
    raise AssertionError(f"TimeFuseLinearSoftmaxHead 不应访问文件系统：args={args} kwargs={kwargs}")


def fail_training_side_effect(*args: object, **kwargs: object) -> object:
    """函数功能：阻断 checkpoint/run artifact 等非 smoke 行为。"""
    raise AssertionError(f"TimeFuseLinearSoftmaxHead smoke 不应产生训练或运行产物：args={args} kwargs={kwargs}")


def build_feature_batch() -> FeatureBatch:
    """函数功能：构造固定小矩阵 FeatureBatch，sample_keys 顺序刻意不排序。"""
    features = np.asarray(
        [
            [1.0, 0.5, -1.0, 2.0],
            [0.0, -2.0, 1.5, 0.5],
            [3.0, 1.0, 0.0, -0.5],
        ],
        dtype=np.float64,
    )
    return FeatureBatch(
        sample_keys=SAMPLE_KEYS,
        features=features,
        feature_schema={
            "feature_schema_name": "timefuse_linear_head_smoke_v1",
            "feature_columns": ("f0", "f1", "f2", "f3"),
            "feature_dim": 4,
        },
        extra={"fixture": "in_memory"},
    )


def run_smoke() -> None:
    """函数功能：执行 TimeFuseLinearSoftmaxHead 最小前向 contract smoke。"""
    print("开始 Stage 1 TimeFuseLinearSoftmaxHead smoke")
    before_outputs = snapshot_run_outputs()

    feature_batch = build_feature_batch()
    weight = np.asarray(
        [
            [0.20, -0.30, 0.10],
            [0.00, 0.40, -0.20],
            [-0.50, 0.25, 0.30],
            [0.10, -0.10, 0.20],
        ],
        dtype=np.float64,
    )
    bias = np.asarray([0.05, -0.10, 0.20], dtype=np.float64)

    head = TimeFuseLinearSoftmaxHead(weight=weight, bias=bias)
    with patch.object(builtins, "open", side_effect=fail_file_access), patch.object(
        Path, "open", fail_file_access
    ), patch.object(np, "load", side_effect=fail_file_access), patch.object(
        np, "save", side_effect=fail_training_side_effect
    ), patch.object(
        np, "savez", side_effect=fail_training_side_effect
    ):
        router_output = head.predict(feature_batch, MODEL_COLUMNS)

    if not isinstance(router_output, RouterOutput):
        raise AssertionError(f"head 未返回 RouterOutput：actual={type(router_output)!r}")
    if router_output.sample_keys != SAMPLE_KEYS:
        raise AssertionError(f"RouterOutput sample_keys 未保序：actual={router_output.sample_keys}")
    if router_output.model_columns != MODEL_COLUMNS:
        raise AssertionError(f"RouterOutput model_columns 未与输入对齐：actual={router_output.model_columns}")
    print("通过：RouterOutput 保持 sample_keys 顺序，并按 model_columns 对齐专家维度")

    expected_logits = feature_batch.features @ weight + bias
    expected_weights = stable_softmax(expected_logits)
    np.testing.assert_allclose(router_output.logits, expected_logits, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(router_output.weights, expected_weights, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(router_output.weights, axis=1), np.ones(len(SAMPLE_KEYS)), rtol=0.0, atol=1e-12)
    if tuple(router_output.logits.shape) != (len(SAMPLE_KEYS), len(MODEL_COLUMNS)):
        raise AssertionError(f"logits shape 漂移：actual={router_output.logits.shape}")
    if tuple(router_output.weights.shape) != (len(SAMPLE_KEYS), len(MODEL_COLUMNS)):
        raise AssertionError(f"weights shape 漂移：actual={router_output.weights.shape}")
    print("通过：固定权重 deterministic logits/weights 与独立 softmax 复算一致")

    extra = router_output.extra
    if extra.get("head_name") != "TimeFuseLinearSoftmaxHead":
        raise AssertionError(f"head_name metadata 漂移：{extra}")
    if extra.get("feature_dim") != 4 or extra.get("num_experts") != len(MODEL_COLUMNS):
        raise AssertionError(f"head extra 维度 metadata 漂移：{extra}")
    if extra.get("feature_schema", {}).get("feature_schema_name") != "timefuse_linear_head_smoke_v1":
        raise AssertionError(f"head extra 未保留轻量 feature_schema lineage：{extra}")
    print("通过：head extra 只记录轻量 lineage，不写运行产物")

    callable_output = head(feature_batch, MODEL_COLUMNS)
    np.testing.assert_allclose(callable_output.weights, expected_weights, rtol=0.0, atol=1e-12)
    print("通过：__call__ 与 predict 输出一致")

    for invalid_columns, expected_message in (
        ((), "非空 model_columns"),
        (("DLinear", "DLinear", "PatchTST"), "重复 model_columns"),
        (("DLinear", "PatchTST"), "专家输出维度"),
    ):
        try:
            head.predict(feature_batch, invalid_columns)
        except ValueError as exc:
            if expected_message not in str(exc):
                raise AssertionError(f"head 拒绝非法 model_columns 的错误信息不清晰：{exc}") from exc
        else:
            raise AssertionError(f"head 应拒绝非法 model_columns：{invalid_columns}")
    print("通过：head 拒绝空、重复和专家数不匹配的 model_columns")

    after_outputs = snapshot_run_outputs()
    if after_outputs != before_outputs:
        raise AssertionError(f"head smoke 不应创建输出目录：新增={sorted(after_outputs - before_outputs)}")
    print("通过：head 不读取外部输入，不训练，不创建输出目录，不写运行产物")
    print("完成：Stage 1 TimeFuseLinearSoftmaxHead smoke 全部通过")


def main() -> None:
    """函数功能：脚本入口。"""
    run_smoke()


if __name__ == "__main__":
    main()
