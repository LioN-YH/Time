#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P10e canonical SampleManifest / SupervisionBatch 协议骨架 smoke。

输入：
    无命令行输入；只在内存中构造小型 manifest 与 supervision batch。

输出：
    标准输出打印中文检查日志；若 sample_key、split、专家列或监督矩阵维度
    不符合最小协议约束则抛出异常。

关键约束：
    不修改正式入口，不读取 /data2，不创建运行目录，不接 Visual Router 或
    TimeFuse-style fusor 训练链路。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.protocols import SampleManifest, SampleManifestRow, SupervisionBatch  # noqa: E402


MODEL_COLUMNS = ("DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster")


def expect_value_error(func, expected_text: str) -> None:
    """
    函数功能：
        smoke 内部断言 helper，确认故意构造的坏输入会给出清晰 ValueError。
    """
    try:
        func()
    except ValueError as exc:
        message = str(exc)
        if expected_text not in message:
            raise AssertionError(f"错误信息未包含 {expected_text!r}，实际为：{message}") from exc
        return
    raise AssertionError(f"预期触发 ValueError：{expected_text}")


def build_manifest() -> SampleManifest:
    """
    函数功能：
        构造包含 vali/test 两个 split 的 4 行 canonical SampleManifest。
    """
    rows = (
        SampleManifestRow(
            sample_key="96_48_S::ETTh1::item0::ch0::win0",
            split="vali",
            config_name="96_48_S",
            dataset_name="ETTh1",
            item_id=0,
            channel_id=0,
            window_index=0,
            seq_len=96,
            pred_len=48,
        ),
        SampleManifestRow(
            sample_key="96_48_S::ETTh1::item0::ch0::win1",
            split="test",
            config_name="96_48_S",
            dataset_name="ETTh1",
            item_id=0,
            channel_id=0,
            window_index=1,
            seq_len=96,
            pred_len=48,
        ),
        SampleManifestRow(
            sample_key="96_48_S::ETTm2::item3::ch0::win0",
            split="vali",
            config_name="96_48_S",
            dataset_name="ETTm2",
            item_id=3,
            channel_id=0,
            window_index=0,
            seq_len=96,
            pred_len=48,
            extra={"manifest_shard": "sample_shard_0000"},
        ),
        SampleManifestRow(
            sample_key="96_48_S::weather::item8::ch0::win2",
            split="test",
            config_name="96_48_S",
            dataset_name="weather",
            item_id=8,
            channel_id=0,
            window_index=2,
            seq_len=96,
            pred_len=48,
        ),
    )
    return SampleManifest(rows=rows, extra={"schema": "sample_manifest_protocol_smoke_v1"})


def build_supervision_batch(sample_keys: tuple[str, ...], metric: str) -> SupervisionBatch:
    """
    函数功能：
        按显式 sample_keys 和固定五专家顺序构造小型 oracle/error supervision batch。
    """
    per_model_errors = np.array(
        [
            [0.9, 0.4, 0.7, 1.1, 0.8],
            [0.3, 0.6, 0.2, 0.5, 0.7],
        ],
        dtype=np.float32,
    )
    oracle_indices = per_model_errors.argmin(axis=1)
    oracle_model = np.array([MODEL_COLUMNS[index] for index in oracle_indices], dtype=object)
    oracle_value = per_model_errors.min(axis=1)
    return SupervisionBatch(
        sample_keys=sample_keys,
        model_columns=MODEL_COLUMNS,
        metric=metric,
        oracle_model=oracle_model,
        oracle_value=oracle_value,
        per_model_errors=per_model_errors,
        extra={"source": "in_memory_smoke"},
    )


def run_smoke() -> None:
    """函数功能：执行 SampleManifest / SupervisionBatch 最小协议 smoke。"""
    print("开始 Stage 1 P10e SampleManifest / SupervisionBatch protocol smoke")

    manifest = build_manifest()
    manifest.validate_unique_sample_keys()
    if manifest.sample_keys() != tuple(row.sample_key for row in manifest.rows):
        raise AssertionError("SampleManifest.sample_keys() 未按 rows 原始顺序返回")
    if manifest.sample_keys(split="vali") != (
        "96_48_S::ETTh1::item0::ch0::win0",
        "96_48_S::ETTm2::item3::ch0::win0",
    ):
        raise AssertionError("vali split sample_keys 未保持原始顺序")
    if manifest.sample_keys(split="test") != (
        "96_48_S::ETTh1::item0::ch0::win1",
        "96_48_S::weather::item8::ch0::win2",
    ):
        raise AssertionError("test split sample_keys 未保持原始顺序")
    if manifest.split_counts() != {"vali": 2, "test": 2}:
        raise AssertionError(f"split_counts 不符合预期：{manifest.split_counts()}")
    print("通过：SampleManifest 唯一性、split 过滤、split_counts 和 ordered sample_keys")

    vali_supervision = build_supervision_batch(manifest.sample_keys(split="vali"), metric="mae")
    test_supervision = build_supervision_batch(manifest.sample_keys(split="test"), metric="mse")
    for supervision in (vali_supervision, test_supervision):
        supervision.validate_shapes()
        if supervision.model_columns != MODEL_COLUMNS:
            raise AssertionError("SupervisionBatch 未保持 model_columns 顺序")
        if supervision.metric not in {"mae", "mse"}:
            raise AssertionError(f"SupervisionBatch metric 异常：{supervision.metric}")
        if supervision.per_model_errors.shape != (2, 5):
            raise AssertionError(f"per_model_errors shape 异常：{supervision.per_model_errors.shape}")
    if tuple(vali_supervision.oracle_model.tolist()) != ("PatchTST", "CrossFormer"):
        raise AssertionError(f"vali oracle_model 异常：{vali_supervision.oracle_model}")
    if not np.allclose(vali_supervision.oracle_value, np.array([0.4, 0.2], dtype=np.float32)):
        raise AssertionError(f"vali oracle_value 异常：{vali_supervision.oracle_value}")
    print("通过：SupervisionBatch 保持 sample/model 顺序、metric 和监督矩阵 shape")

    duplicate_manifest = SampleManifest(rows=manifest.rows + (manifest.rows[0],))
    expect_value_error(duplicate_manifest.validate_unique_sample_keys, "重复值")
    bad_error_shape = SupervisionBatch(
        sample_keys=vali_supervision.sample_keys,
        model_columns=MODEL_COLUMNS,
        metric="mae",
        oracle_model=vali_supervision.oracle_model,
        oracle_value=vali_supervision.oracle_value,
        per_model_errors=np.zeros((2, 4), dtype=np.float32),
    )
    expect_value_error(bad_error_shape.validate_shapes, "model_columns")
    bad_oracle_shape = SupervisionBatch(
        sample_keys=vali_supervision.sample_keys,
        model_columns=MODEL_COLUMNS,
        metric="mae",
        oracle_model=np.array(["DLinear"], dtype=object),
        oracle_value=vali_supervision.oracle_value,
        per_model_errors=vali_supervision.per_model_errors,
    )
    expect_value_error(bad_oracle_shape.validate_shapes, "oracle_model")
    print("通过：重复 sample_key 和 supervision shape mismatch 均给出清晰报错")

    print("Stage 1 P10e SampleManifest / SupervisionBatch protocol smoke 通过")


if __name__ == "__main__":
    run_smoke()
