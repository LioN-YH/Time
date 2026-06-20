#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P10f Visual labels -> SampleManifest / SupervisionBatch adapter smoke。

输入：
    无命令行输入；在内存中构造 4 行 vali/test labels fixture，并额外写入
    tempfile CSV 验证 CSV 入口。

输出：
    标准输出打印中文检查日志；若 manifest、split、oracle、per-model error
    或清晰报错不符合 P10f 最小契约则抛出异常。

关键约束：
    不修改正式 Visual Router 入口，不读取 `/data2`，不新增 Bash/scripts，
    不改变正式 CSV / summary / metadata / status / checkpoint schema。
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.data import (  # noqa: E402
    visual_labels_to_sample_manifest,
    visual_labels_to_supervision_batch,
)


MODEL_COLUMNS = ("DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster")


def expect_value_error(func, expected_text: str) -> None:
    """函数功能：确认坏输入触发 ValueError，且错误信息包含预期中文片段。"""
    try:
        func()
    except ValueError as exc:
        message = str(exc)
        if expected_text not in message:
            raise AssertionError(f"错误信息未包含 {expected_text!r}，实际为：{message}") from exc
        return
    raise AssertionError(f"预期触发 ValueError：{expected_text}")


def build_visual_labels_fixture() -> pd.DataFrame:
    """
    函数功能：
        构造 P10f 最小 Visual labels fixture。

    说明：
        当前 fixture 使用 `模型名_metric_error` 作为 smoke 误差列命名约定。
        正式入口接入前仍需与真实历史 labels CSV schema 做字段映射审计。
    """
    rows = [
        {
            "sample_key": "96_48_S::ETTh1::item0::ch0::win0",
            "split": "vali",
            "config_name": "96_48_S",
            "dataset_name": "ETTh1",
            "item_id": 0,
            "channel_id": 0,
            "window_index": 0,
            "seq_len": 96,
            "pred_len": 48,
            "manifest_shard": "labels_shard_0000",
            "DLinear_mae_error": 0.90,
            "PatchTST_mae_error": 0.40,
            "CrossFormer_mae_error": 0.70,
            "ES_mae_error": 1.10,
            "NaiveForecaster_mae_error": 0.80,
        },
        {
            "sample_key": "96_48_S::ETTh1::item0::ch0::win1",
            "split": "test",
            "config_name": "96_48_S",
            "dataset_name": "ETTh1",
            "item_id": 0,
            "channel_id": 0,
            "window_index": 1,
            "seq_len": 96,
            "pred_len": 48,
            "manifest_shard": "labels_shard_0000",
            "DLinear_mae_error": 0.30,
            "PatchTST_mae_error": 0.60,
            "CrossFormer_mae_error": 0.20,
            "ES_mae_error": 0.50,
            "NaiveForecaster_mae_error": 0.70,
        },
        {
            "sample_key": "96_48_S::ETTm2::item3::ch0::win0",
            "split": "vali",
            "config_name": "96_48_S",
            "dataset_name": "ETTm2",
            "item_id": 3,
            "channel_id": 0,
            "window_index": 0,
            "seq_len": 96,
            "pred_len": 48,
            "manifest_shard": "labels_shard_0001",
            "DLinear_mae_error": 0.50,
            "PatchTST_mae_error": 0.60,
            "CrossFormer_mae_error": 0.70,
            "ES_mae_error": 0.20,
            "NaiveForecaster_mae_error": 0.90,
        },
        {
            "sample_key": "96_48_S::weather::item8::ch0::win2",
            "split": "test",
            "config_name": "96_48_S",
            "dataset_name": "weather",
            "item_id": 8,
            "channel_id": 0,
            "window_index": 2,
            "seq_len": 96,
            "pred_len": 48,
            "manifest_shard": "labels_shard_0001",
            "DLinear_mae_error": 0.80,
            "PatchTST_mae_error": 0.70,
            "CrossFormer_mae_error": 0.60,
            "ES_mae_error": 0.50,
            "NaiveForecaster_mae_error": 0.10,
        },
    ]
    return pd.DataFrame(rows)


def assert_supervision(
    frame: pd.DataFrame,
    sample_keys: tuple[str, ...],
    expected_models: tuple[str, ...],
    expected_values: np.ndarray,
) -> None:
    """函数功能：按 split 校验 adapter 产出的 SupervisionBatch 内容和 shape。"""
    supervision = visual_labels_to_supervision_batch(
        frame,
        sample_keys=sample_keys,
        model_columns=MODEL_COLUMNS,
        metric="mae",
    )
    if supervision.sample_keys != sample_keys:
        raise AssertionError("SupervisionBatch 未保持输入 sample_keys 顺序")
    if supervision.model_columns != MODEL_COLUMNS:
        raise AssertionError("SupervisionBatch 未保持输入 model_columns 顺序")
    if supervision.per_model_errors.shape != (len(sample_keys), len(MODEL_COLUMNS)):
        raise AssertionError(f"per_model_errors shape 异常：{supervision.per_model_errors.shape}")
    if tuple(supervision.oracle_model.tolist()) != expected_models:
        raise AssertionError(f"oracle_model 异常：{supervision.oracle_model}")
    if not np.allclose(supervision.oracle_value, expected_values.astype(np.float32)):
        raise AssertionError(f"oracle_value 异常：{supervision.oracle_value}")


def run_smoke() -> None:
    """函数功能：执行 P10f Visual labels adapter 最小 smoke 验收。"""
    print("开始 Stage 1 P10f Visual labels -> SampleManifest/SupervisionBatch adapter smoke")
    frame = build_visual_labels_fixture()

    manifest = visual_labels_to_sample_manifest(frame)
    if manifest.sample_keys() != tuple(frame["sample_key"].astype(str).tolist()):
        raise AssertionError("SampleManifest 未保持 labels 原始 sample_key 顺序")
    if manifest.sample_keys(split="vali") != (
        "96_48_S::ETTh1::item0::ch0::win0",
        "96_48_S::ETTm2::item3::ch0::win0",
    ):
        raise AssertionError("vali split sample_keys 未保序")
    if manifest.sample_keys(split="test") != (
        "96_48_S::ETTh1::item0::ch0::win1",
        "96_48_S::weather::item8::ch0::win2",
    ):
        raise AssertionError("test split sample_keys 未保序")
    if manifest.split_counts() != {"vali": 2, "test": 2}:
        raise AssertionError(f"split_counts 异常：{manifest.split_counts()}")
    if "PatchTST_mae_error" in manifest.rows[0].extra:
        raise AssertionError("oracle/error 字段不应进入 SampleManifestRow.extra")
    if manifest.rows[0].extra.get("manifest_shard") != "labels_shard_0000":
        raise AssertionError(f"manifest lineage extra 异常：{manifest.rows[0].extra}")
    print("通过：SampleManifest 唯一性、split 保序、split_counts 和 lineage extra")

    assert_supervision(
        frame,
        manifest.sample_keys(split="vali"),
        expected_models=("PatchTST", "ES"),
        expected_values=np.array([0.40, 0.20], dtype=np.float32),
    )
    assert_supervision(
        frame,
        manifest.sample_keys(split="test"),
        expected_models=("CrossFormer", "NaiveForecaster"),
        expected_values=np.array([0.20, 0.10], dtype=np.float32),
    )
    print("通过：vali/test SupervisionBatch oracle、oracle_value 和 per-model error shape")

    with TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "visual_labels_fixture.csv"
        frame.to_csv(csv_path, index=False)
        csv_manifest = visual_labels_to_sample_manifest(csv_path)
        if csv_manifest.sample_keys() != manifest.sample_keys():
            raise AssertionError("CSV 入口未保持 sample_key 顺序")
    print("通过：CSV 与 DataFrame 入口一致")

    missing_expert = frame.drop(columns=["ES_mae_error"])
    expect_value_error(
        lambda: visual_labels_to_supervision_batch(
            missing_expert,
            sample_keys=manifest.sample_keys(split="vali"),
            model_columns=MODEL_COLUMNS,
            metric="mae",
        ),
        "缺少必需列",
    )

    duplicate_key = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    expect_value_error(lambda: visual_labels_to_sample_manifest(duplicate_key), "重复值")

    unknown_split = frame.copy()
    unknown_split.loc[0, "split"] = "validation"
    expect_value_error(lambda: visual_labels_to_sample_manifest(unknown_split), "未知 split")
    print("通过：缺失专家列、重复 sample_key 和未知 split 均给出清晰报错")

    print("Stage 1 P10f Visual labels adapter smoke 通过")


if __name__ == "__main__":
    run_smoke()
