#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P10b shared prediction SQLite backend 最小 smoke。

输入：
    测试内临时构造 4 个 sample、5 个 model 的 packed_npy_v1 manifest 与数组。

输出：
    标准输出打印中文检查日志；任一 SQLite index、fetch、metadata、row index 或
    missing report 契约漂移时抛出 AssertionError。

关键约束：
    该 smoke 不访问 /data2，不运行正式入口，不修改 Visual Router / TimeFuse 训练代码，
    不创建正式输出目录。
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.io import (  # noqa: E402
    DEFAULT_MODEL_COLUMNS,
    build_prediction_sqlite_backend,
    load_prediction_sqlite_backend,
    records_to_ordered_rows,
)
from visual_router_experiments.common.prediction_array_io import load_prediction_arrays_grouped  # noqa: E402


MODEL_COLUMNS = tuple(DEFAULT_MODEL_COLUMNS)
SAMPLE_KEYS = tuple(f"p10b_sample_{idx}" for idx in range(4))
PRED_LEN = 6
CHANNELS = 1


def build_fixture(root: Path, *, drop_pair: Tuple[str, str] | None = None) -> Path:
    """
    函数功能：
        构造 packed_npy_v1 prediction manifest fixture。

    输入：
        root: tempfile 下的 fixture 根目录。
        drop_pair: 可选删除一个 `(sample_key, model_name)` record，用于缺失路径测试。

    输出：
        manifest.csv 路径。
    """
    arrays_dir = root / "arrays" / "packed"
    y_true_dir = arrays_dir / "y_true" / "test" / "P10B_FIXTURE"
    y_true_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.arange(len(SAMPLE_KEYS) * PRED_LEN * CHANNELS, dtype=np.float32).reshape(
        len(SAMPLE_KEYS), PRED_LEN, CHANNELS
    )
    y_true_path = y_true_dir / "y_true.npy"
    np.save(y_true_path, y_true)

    rows: List[Dict[str, object]] = []
    for model_idx, model_name in enumerate(MODEL_COLUMNS):
        y_pred_dir = arrays_dir / "y_pred" / model_name / "test" / "P10B_FIXTURE"
        y_pred_dir.mkdir(parents=True, exist_ok=True)
        y_pred = y_true + np.float32((model_idx + 1) / 10.0)
        y_pred_path = y_pred_dir / "y_pred.npy"
        np.save(y_pred_path, y_pred)
        for sample_idx, sample_key in enumerate(SAMPLE_KEYS):
            if drop_pair == (sample_key, model_name):
                continue
            sample_true = y_true[sample_idx]
            sample_pred = y_pred[sample_idx]
            rows.append(
                {
                    "sample_key": sample_key,
                    "model_name": model_name,
                    "y_true_path": str(y_true_path.relative_to(root)),
                    "y_pred_path": str(y_pred_path.relative_to(root)),
                    "mae": float(np.mean(np.abs(sample_pred - sample_true))),
                    "mse": float(np.mean((sample_pred - sample_true) ** 2)),
                    "array_storage": "packed_npy_v1",
                    "y_true_row_index": int(sample_idx),
                    "y_pred_row_index": int(sample_idx),
                }
            )

    manifest_path = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def assert_sqlite_metadata(index_db_path: Path, *, expected_records: int) -> None:
    """函数功能：直接检查 SQLite metadata 表，确认 index 文件可独立审计。"""
    connection = sqlite3.connect(str(index_db_path))
    try:
        metadata_keys = {row[0] for row in connection.execute("SELECT key FROM index_metadata").fetchall()}
        required = {
            "target_sample_keys",
            "expected_records",
            "actual_records",
            "chunk_read_rows",
            "model_columns",
            "manifest_path",
            "manifest_dir",
            "index_db_path",
            "created_at",
            "missing_report",
        }
        if not required.issubset(metadata_keys):
            raise AssertionError(f"index_metadata 字段不完整：actual={sorted(metadata_keys)}")
        actual_records = int(connection.execute("SELECT COUNT(*) FROM prediction_index").fetchone()[0])
        if actual_records != expected_records:
            raise AssertionError(f"prediction_index 行数漂移：actual={actual_records} expected={expected_records}")
    finally:
        connection.close()


def run_complete_backend_smoke(tmp_root: Path) -> None:
    """函数功能：验证完整 manifest 的 build/fetch/grouped loading 契约。"""
    manifest_path = build_fixture(tmp_root / "complete")
    index_db_path = tmp_root / "complete.sqlite"
    requested_keys = [SAMPLE_KEYS[2], SAMPLE_KEYS[0], SAMPLE_KEYS[3]]
    backend = build_prediction_sqlite_backend(
        manifest_path=manifest_path,
        target_sample_keys=requested_keys,
        index_db_path=index_db_path,
        model_columns=MODEL_COLUMNS,
        chunk_read_rows=3,
    )
    try:
        expected_records = len(requested_keys) * len(MODEL_COLUMNS)
        if not index_db_path.exists():
            raise AssertionError("SQLite index 未写入目标路径")
        if backend.metadata.target_sample_keys != tuple(requested_keys):
            raise AssertionError(f"metadata target_sample_keys 未保序：{backend.metadata.target_sample_keys}")
        if backend.metadata.expected_records != expected_records or backend.metadata.actual_records != expected_records:
            raise AssertionError(f"metadata record count 漂移：{backend.metadata}")
        if backend.metadata.missing_report["missing_records"] != 0:
            raise AssertionError(f"完整 fixture 不应有 missing report：{backend.metadata.missing_report}")
        assert_sqlite_metadata(index_db_path, expected_records=expected_records)

        fetch_keys = [SAMPLE_KEYS[3], SAMPLE_KEYS[0]]
        records = backend.fetch_records(fetch_keys)
        ordered_rows = records_to_ordered_rows(records, sample_keys=fetch_keys, model_columns=MODEL_COLUMNS)
        if len(ordered_rows) != len(fetch_keys) * len(MODEL_COLUMNS):
            raise AssertionError("ordered rows 数量不正确")
        if [row["sample_key"] for row in ordered_rows[: len(MODEL_COLUMNS)]] != [fetch_keys[0]] * len(MODEL_COLUMNS):
            raise AssertionError("ordered rows 未按输入 sample_keys 保序")
        if [row["model_name"] for row in ordered_rows[: len(MODEL_COLUMNS)]] != list(MODEL_COLUMNS):
            raise AssertionError("ordered rows 未按 model_columns 保序")

        first_model_rows = [records[(sample_key, MODEL_COLUMNS[0])] for sample_key in fetch_keys]
        y_true = load_prediction_arrays_grouped(first_model_rows, "y_true")
        if tuple(y_true.shape) != (len(fetch_keys), PRED_LEN, CHANNELS):
            raise AssertionError(f"y_true shape 漂移：{y_true.shape}")
        for expected_position, sample_key in enumerate(fetch_keys):
            expected_row_index = SAMPLE_KEYS.index(sample_key)
            if int(first_model_rows[expected_position]["y_true_row_index"]) != expected_row_index:
                raise AssertionError("y_true row index lineage 漂移")

        model_pred_rows = [records[(sample_key, MODEL_COLUMNS[1])] for sample_key in fetch_keys]
        y_pred = load_prediction_arrays_grouped(model_pred_rows, "y_pred")
        if tuple(y_pred.shape) != (len(fetch_keys), PRED_LEN, CHANNELS):
            raise AssertionError(f"y_pred shape 漂移：{y_pred.shape}")
        if not np.allclose(y_pred - y_true, np.float32(0.2)):
            raise AssertionError("grouped packed loading 读回的 y_pred/y_true 数值不符合 fixture")
    finally:
        backend.close()

    reloaded = load_prediction_sqlite_backend(index_db_path)
    try:
        if reloaded.metadata.target_sample_keys != tuple(requested_keys):
            raise AssertionError("只读恢复 backend 后 metadata target_sample_keys 漂移")
        if reloaded.model_columns != MODEL_COLUMNS:
            raise AssertionError(f"只读恢复 backend 后 model_columns 漂移：{reloaded.model_columns}")
    finally:
        reloaded.close()
    print("通过：完整 fixture 可构建 SQLite 子集索引、fetch records，并按 row index 读回 packed arrays")


def run_missing_backend_smoke(tmp_root: Path) -> None:
    """函数功能：验证缺失 sample/model 的报错和 missing report。"""
    drop_pair = (SAMPLE_KEYS[1], MODEL_COLUMNS[3])
    manifest_path = build_fixture(tmp_root / "missing", drop_pair=drop_pair)
    strict_index = tmp_root / "missing_strict.sqlite"
    try:
        build_prediction_sqlite_backend(
            manifest_path=manifest_path,
            target_sample_keys=[SAMPLE_KEYS[0], SAMPLE_KEYS[1]],
            index_db_path=strict_index,
            model_columns=MODEL_COLUMNS,
            chunk_read_rows=4,
        )
    except ValueError as exc:
        message = str(exc)
        if "子集不完整" not in message or drop_pair[0] not in message:
            raise AssertionError(f"缺失 record 报错信息不清晰：{message}") from exc
    else:
        raise AssertionError("默认 allow_missing=False 时应拒绝缺失 sample/model")
    if strict_index.exists():
        raise AssertionError("失败构建不应留下目标 SQLite 半成品")

    report_index = tmp_root / "missing_report.sqlite"
    backend = build_prediction_sqlite_backend(
        manifest_path=manifest_path,
        target_sample_keys=[SAMPLE_KEYS[0], SAMPLE_KEYS[1]],
        index_db_path=report_index,
        model_columns=MODEL_COLUMNS,
        chunk_read_rows=4,
        allow_missing=True,
    )
    try:
        report = backend.metadata.missing_report
        if report["missing_records"] != 1:
            raise AssertionError(f"missing_records 漂移：{report}")
        missing_models = report["missing_models_by_sample"].get(drop_pair[0])
        if missing_models != [drop_pair[1]]:
            raise AssertionError(f"missing_models_by_sample 漂移：{report}")
    finally:
        backend.close()
    print("通过：缺失 sample/model 默认报错，allow_missing=True 时写入 missing report")


def main() -> None:
    """函数功能：脚本入口。"""
    with tempfile.TemporaryDirectory(prefix="stage1_p10b_sqlite_backend_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        if str(tmp_root).startswith("/data2/"):
            raise AssertionError("P10b smoke 不应访问 /data2")
        run_complete_backend_smoke(tmp_root)
        run_missing_backend_smoke(tmp_root)
    print("完成：Stage 1 P10b shared prediction SQLite backend smoke 全部通过")


if __name__ == "__main__":
    main()
