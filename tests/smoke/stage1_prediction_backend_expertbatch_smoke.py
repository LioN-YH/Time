#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P13d prediction backend -> ExpertBatch small smoke。

输入：
    读取 P13b real-derived small fixture 的 `sample_manifest.csv` 与
    `expert_predictions.json`；在 tempfile 内把 JSON 参考值转换成小型
    `packed_npy_v1` prediction cache manifest、数组和 SQLite backend。

输出：
    标准输出打印中文检查日志；若 backend records、PredictionBatchReader、
    PredictionCacheExpertProvider 或 ExpertBatch 契约相对 P13b 参考值漂移则抛错。

关键约束：
    本 smoke 只验证 prediction backend -> ExpertBatch 小链路，不迁移正式入口，
    不访问 `/data2`，不启动训练、pressure 或 full-scale，也不把 P13b JSON 升级为
    正式 backend schema。
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.experts import PredictionCacheExpertProvider  # noqa: E402
from time_router.io import (  # noqa: E402
    PACKED_NPY_STORAGE,
    PredictionBatchReader,
    build_prediction_sqlite_backend,
    load_prediction_arrays_grouped,
    records_to_ordered_rows,
)


FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small"
SAMPLE_MANIFEST_PATH = FIXTURE_ROOT / "sample_manifest.csv"
EXPERT_REFERENCE_PATH = FIXTURE_ROOT / "expert_predictions.json"


def load_manifest_sample_keys(path: Path) -> tuple[str, ...]:
    """
    函数功能：
        从 P13b sample manifest 中读取 ordered sample_keys。

    输入：
        path: P13b `sample_manifest.csv` 路径。

    输出：
        按 CSV 行顺序返回的 sample_key tuple；若为空、重复或指向 `/data2` 则报错。
    """
    assert_repo_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError("sample_manifest.csv 不应为空")
    sample_keys = tuple(str(row["sample_key"]) for row in rows)
    if len(sample_keys) != len(set(sample_keys)):
        raise AssertionError(f"sample_manifest.csv 存在重复 sample_key：{sample_keys}")
    return sample_keys


def load_reference(path: Path, ordered_sample_keys: Sequence[str]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """
    函数功能：
        读取 P13b expert JSON，并按 manifest 行顺序组装参考 y_pred/y_true。

    输入：
        path: P13b `expert_predictions.json` 路径。
        ordered_sample_keys: `sample_manifest.csv` 行顺序。

    输出：
        `(model_columns, y_pred, y_true)`，其中 y_pred shape 为
        `[sample, expert, pred_len, channel]`，y_true shape 为
        `[sample, pred_len, channel]`。

    关键约束：
        JSON 只作为 P13d 数值参考；本函数不会把它当作正式 prediction backend schema。
    """
    assert_repo_file(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError("expert_predictions.json 不是 JSON object")

    model_columns = tuple(str(model_name) for model_name in payload["model_columns"])
    if not model_columns or len(model_columns) != len(set(model_columns)):
        raise AssertionError(f"model_columns 异常：{model_columns}")

    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise AssertionError("expert_predictions.json 缺少 samples list")
    reference_by_key: Dict[str, Mapping[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise AssertionError(f"sample 不是 object：{sample}")
        sample_key = str(sample["sample_key"])
        if sample_key in reference_by_key:
            raise AssertionError(f"expert_predictions.json 存在重复 sample_key：{sample_key}")
        reference_by_key[sample_key] = sample
    if set(reference_by_key) != set(ordered_sample_keys):
        raise AssertionError(
            "expert_predictions.json sample_key 集合未与 manifest 对齐："
            f"json={sorted(reference_by_key)} manifest={sorted(ordered_sample_keys)}"
        )

    y_true_rows: List[np.ndarray] = []
    y_pred_rows: List[np.ndarray] = []
    for sample_key in ordered_sample_keys:
        sample = reference_by_key[sample_key]
        y_true_rows.append(np.asarray(sample["y_true"], dtype=np.float32))
        y_pred = np.asarray(sample["y_pred"], dtype=np.float32)
        if y_pred.shape[0] != len(model_columns):
            raise AssertionError(f"{sample_key} y_pred 专家维与 model_columns 不一致：shape={y_pred.shape}")
        y_pred_rows.append(y_pred)

    y_true_array = np.stack(y_true_rows, axis=0).astype(np.float32)
    y_pred_array = np.stack(y_pred_rows, axis=0).astype(np.float32)
    if y_pred_array.shape[0] != len(ordered_sample_keys) or y_pred_array.shape[1] != len(model_columns):
        raise AssertionError(f"参考 y_pred shape 异常：{y_pred_array.shape}")
    if y_pred_array.shape[2:] != y_true_array.shape[1:]:
        raise AssertionError(f"参考 y_pred/y_true shape 不一致：{y_pred_array.shape} vs {y_true_array.shape}")
    return model_columns, y_pred_array, y_true_array


def build_prediction_cache_fixture(
    root: Path,
    *,
    ordered_sample_keys: Sequence[str],
    model_columns: Sequence[str],
    y_pred_reference: np.ndarray,
    y_true_reference: np.ndarray,
) -> Path:
    """
    函数功能：
        在 tempfile 内把 P13b 参考数组转换成 packed_npy_v1 prediction cache fixture。

    输入：
        root: 临时 fixture 根目录。
        ordered_sample_keys/model_columns: 期望顺序。
        y_pred_reference/y_true_reference: 按上述顺序排列的参考数组。

    输出：
        prediction cache `manifest.csv` 路径。

    关键约束：
        数组和 manifest 只存在于 tempfile；P13b JSON 仍只是参考来源，不作为正式
        backend schema 长期保留。
    """
    arrays_dir = root / "arrays" / "packed"
    y_true_dir = arrays_dir / "y_true" / "test" / "P13D_FIXTURE"
    y_true_dir.mkdir(parents=True, exist_ok=True)
    y_true_path = y_true_dir / "y_true.npy"
    np.save(y_true_path, y_true_reference.astype(np.float32))

    rows: List[Dict[str, object]] = []
    for model_idx, model_name in enumerate(model_columns):
        y_pred_dir = arrays_dir / "y_pred" / str(model_name) / "test" / "P13D_FIXTURE"
        y_pred_dir.mkdir(parents=True, exist_ok=True)
        y_pred_model = y_pred_reference[:, model_idx, :, :].astype(np.float32)
        y_pred_path = y_pred_dir / "y_pred.npy"
        np.save(y_pred_path, y_pred_model)
        for sample_idx, sample_key in enumerate(ordered_sample_keys):
            sample_true = y_true_reference[sample_idx]
            sample_pred = y_pred_model[sample_idx]
            rows.append(
                {
                    "sample_key": str(sample_key),
                    "model_name": str(model_name),
                    "y_true_path": str(y_true_path.relative_to(root)),
                    "y_pred_path": str(y_pred_path.relative_to(root)),
                    "mae": float(np.mean(np.abs(sample_pred - sample_true))),
                    "mse": float(np.mean((sample_pred - sample_true) ** 2)),
                    "array_storage": PACKED_NPY_STORAGE,
                    "y_true_row_index": int(sample_idx),
                    "y_pred_row_index": int(sample_idx),
                }
            )

    manifest_path = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def assert_backend_fetch_parity(
    *,
    manifest_path: Path,
    tmp_root: Path,
    ordered_sample_keys: Sequence[str],
    model_columns: Sequence[str],
    y_pred_reference: np.ndarray,
    y_true_reference: np.ndarray,
) -> None:
    """
    函数功能：
        先经 shared SQLite backend fetch records，再用 grouped array IO 验证数值 parity。

    关键约束：
        该检查覆盖 backend prepare / fetch / row index lineage，不直接构造 ExpertBatch；
        ExpertBatch 包装由后续 provider 检查覆盖。
    """
    index_db_path = tmp_root / "p13d_prediction_backend.sqlite"
    backend = build_prediction_sqlite_backend(
        manifest_path=manifest_path,
        target_sample_keys=ordered_sample_keys,
        index_db_path=index_db_path,
        model_columns=model_columns,
        chunk_read_rows=3,
    )
    try:
        expected_records = len(ordered_sample_keys) * len(model_columns)
        if backend.metadata.expected_records != expected_records or backend.metadata.actual_records != expected_records:
            raise AssertionError(f"SQLite metadata record count 异常：{backend.metadata}")
        if backend.metadata.target_sample_keys != tuple(ordered_sample_keys):
            raise AssertionError(f"SQLite metadata 未保留 manifest sample_key 顺序：{backend.metadata.target_sample_keys}")
        if backend.metadata.model_columns != tuple(model_columns):
            raise AssertionError(f"SQLite metadata model_columns 异常：{backend.metadata.model_columns}")
        if backend.metadata.missing_report["missing_records"] != 0:
            raise AssertionError(f"完整 P13d fixture 不应有 missing report：{backend.metadata.missing_report}")

        records = backend.fetch_records(ordered_sample_keys)
        ordered_rows = records_to_ordered_rows(records, sample_keys=ordered_sample_keys, model_columns=model_columns)
        if len(ordered_rows) != expected_records:
            raise AssertionError(f"fetch ordered rows 数量异常：{len(ordered_rows)} vs {expected_records}")

        y_true_rows = [records[(sample_key, model_columns[0])] for sample_key in ordered_sample_keys]
        y_true = load_prediction_arrays_grouped(y_true_rows, "y_true")
        assert_array_equal("SQLite backend grouped y_true", y_true, y_true_reference)

        for model_idx, model_name in enumerate(model_columns):
            pred_rows = [records[(sample_key, model_name)] for sample_key in ordered_sample_keys]
            y_pred = load_prediction_arrays_grouped(pred_rows, "y_pred")
            assert_array_equal(f"SQLite backend grouped y_pred[{model_name}]", y_pred, y_pred_reference[:, model_idx])
            for row_position, row in enumerate(pred_rows):
                if int(row["y_pred_row_index"]) != row_position or int(row["y_true_row_index"]) != row_position:
                    raise AssertionError(f"row index lineage 漂移：model={model_name} row={row}")
    finally:
        backend.close()
    print("通过：shared SQLite backend 可按 manifest 顺序 fetch records，并按 row index 读回参考数组")


def assert_reader_and_provider_parity(
    *,
    manifest_path: Path,
    ordered_sample_keys: Sequence[str],
    model_columns: Sequence[str],
    y_pred_reference: np.ndarray,
    y_true_reference: np.ndarray,
) -> None:
    """
    函数功能：
        经 PredictionBatchReader 和 PredictionCacheExpertProvider 输出 ExpertBatch 并验证 parity。
    """
    reader = PredictionBatchReader(
        manifest_path=manifest_path,
        model_columns=model_columns,
        chunk_rows=4,
        validate_manifest_schema=False,
    )
    prediction_batch = reader.load(ordered_sample_keys, verify_metrics=True)
    if tuple(prediction_batch.sample_keys) != tuple(ordered_sample_keys):
        raise AssertionError(f"PredictionBatchReader sample_keys 未保序：{prediction_batch.sample_keys}")
    assert_array_equal("PredictionBatchReader y_pred", prediction_batch.y_pred, y_pred_reference)
    assert_array_equal("PredictionBatchReader y_true", prediction_batch.y_true, y_true_reference)
    if tuple(prediction_batch.metadata["model_columns"]) != tuple(model_columns):
        raise AssertionError(f"PredictionBatchReader metadata model_columns 异常：{prediction_batch.metadata}")

    provider = PredictionCacheExpertProvider(
        manifest_path=manifest_path,
        model_columns=model_columns,
        chunk_rows=4,
        validate_manifest_schema=False,
    )
    expert_batch = provider.load_batch(ordered_sample_keys, verify_metrics=True)
    if expert_batch.sample_keys != tuple(ordered_sample_keys):
        raise AssertionError(f"ExpertBatch sample_keys 未保留 manifest 行顺序：{expert_batch.sample_keys}")
    if expert_batch.model_columns != tuple(model_columns):
        raise AssertionError(f"ExpertBatch model_columns 未与参考 JSON 一致：{expert_batch.model_columns}")
    assert_array_equal("ExpertBatch y_pred", expert_batch.y_pred, y_pred_reference)
    assert_array_equal("ExpertBatch y_true", expert_batch.y_true, y_true_reference)

    if tuple(expert_batch.y_pred.shape) != tuple(y_pred_reference.shape):
        raise AssertionError(f"ExpertBatch y_pred shape 漂移：{expert_batch.y_pred.shape}")
    if tuple(expert_batch.y_true.shape) != tuple(y_true_reference.shape):
        raise AssertionError(f"ExpertBatch y_true shape 漂移：{expert_batch.y_true.shape}")
    if not expert_batch.row_index_metadata:
        raise AssertionError("ExpertBatch 缺少 row_index_metadata")
    if expert_batch.extra.get("provider_name") != "PredictionCacheExpertProvider":
        raise AssertionError(f"ExpertBatch.extra 缺少轻量 provider 来源信息：{expert_batch.extra}")
    if expert_batch.extra.get("array_storage") != PACKED_NPY_STORAGE:
        raise AssertionError(f"ExpertBatch.extra array_storage 异常：{expert_batch.extra}")
    reader_metadata = expert_batch.extra.get("reader_metadata")
    if not isinstance(reader_metadata, dict) or reader_metadata.get("manifest_path") != str(manifest_path):
        raise AssertionError(f"ExpertBatch.extra.reader_metadata 来源信息异常：{expert_batch.extra}")
    if reader_metadata.get("validate_manifest_schema") is not False:
        raise AssertionError(f"P13d smoke 应显式记录关闭 canonical schema 校验：{expert_batch.extra}")
    print("通过：PredictionBatchReader / PredictionCacheExpertProvider 输出 ExpertBatch 且数值对齐 P13b 参考")


def assert_repo_file(path: Path) -> None:
    """函数功能：确认输入 fixture 是仓库内普通文件且不位于 `/data2`。"""
    if not path.is_file():
        raise AssertionError(f"fixture 文件缺失：{path}")
    resolved = str(path.resolve())
    if resolved.startswith("/data2/") or resolved == "/data2":
        raise AssertionError(f"P13d smoke 不应访问 /data2 fixture：{path}")


def assert_array_equal(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    """函数功能：用严格 shape 和 allclose 校验数组，失败时给出明确对象名。"""
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{name} shape 漂移：actual={actual.shape} expected={expected.shape}")
    if not np.allclose(actual, expected, rtol=1e-6, atol=1e-6):
        max_abs = float(np.max(np.abs(actual - expected)))
        raise AssertionError(f"{name} 数值与参考不一致：max_abs_diff={max_abs}")


def assert_no_data2(paths: Iterable[Path]) -> None:
    """函数功能：确认本 smoke 使用路径均未落到 `/data2`。"""
    for path in paths:
        resolved = str(path.resolve())
        if resolved.startswith("/data2/") or resolved == "/data2":
            raise AssertionError(f"P13d smoke 不应访问 /data2：{path}")


def main() -> None:
    """函数功能：脚本入口，执行 P13d backend -> ExpertBatch smoke。"""
    print("开始 Stage 1 P13d prediction backend -> ExpertBatch small smoke")
    ordered_sample_keys = load_manifest_sample_keys(SAMPLE_MANIFEST_PATH)
    model_columns, y_pred_reference, y_true_reference = load_reference(EXPERT_REFERENCE_PATH, ordered_sample_keys)
    print("通过：已按 P13b sample_manifest 行顺序读取 sample_keys，并按该顺序组装参考数组")

    with tempfile.TemporaryDirectory(prefix="stage1_p13d_prediction_backend_expertbatch_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        assert_no_data2([tmp_root])
        manifest_path = build_prediction_cache_fixture(
            tmp_root / "prediction_cache",
            ordered_sample_keys=ordered_sample_keys,
            model_columns=model_columns,
            y_pred_reference=y_pred_reference,
            y_true_reference=y_true_reference,
        )
        assert_no_data2([manifest_path])
        print("通过：已在 tempfile 构造 packed_npy_v1 prediction cache/backend fixture")

        assert_backend_fetch_parity(
            manifest_path=manifest_path,
            tmp_root=tmp_root,
            ordered_sample_keys=ordered_sample_keys,
            model_columns=model_columns,
            y_pred_reference=y_pred_reference,
            y_true_reference=y_true_reference,
        )
        assert_reader_and_provider_parity(
            manifest_path=manifest_path,
            ordered_sample_keys=ordered_sample_keys,
            model_columns=model_columns,
            y_pred_reference=y_pred_reference,
            y_true_reference=y_true_reference,
        )

    print("完成：Stage 1 P13d prediction backend -> ExpertBatch small smoke 全部通过")


if __name__ == "__main__":
    main()
