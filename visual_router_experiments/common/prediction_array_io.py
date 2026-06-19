#!/usr/bin/env python3
"""
文件功能：
    为 Stage 1 prediction cache 提供统一数组读取接口。

设计约束：
    - 兼容早期每个 sample 一个 `.npy` 文件的 legacy/per-sample cache；
    - 支持 full-scale shard 推荐的 packed `.npy`，即一个 shard 文件保存多行窗口数组；
    - 读取层只返回单个 sample 的 y_true/y_pred 数组，不改变 manifest 契约中的
      `sample_key + model_name` 主键；
    - 该文件不负责写 cache，写入策略由 prediction cache builder/merge 决定。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


PACKED_NPY_STORAGE = "packed_npy_v1"
PER_SAMPLE_NPY_STORAGE = "per_sample_npy"


def resolve_cache_array_path(path_text: Union[str, Path], manifest_dir: Path) -> Path:
    """函数功能：解析 manifest 中相对或绝对数组路径。"""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return manifest_dir / path


def _optional_int(record: Mapping[str, object], key: str) -> Optional[int]:
    """函数功能：从 manifest record 中安全读取可选整数索引。"""
    if key not in record:
        return None
    value = record[key]
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return int(value)


def load_prediction_array(record: Mapping[str, object], array_kind: str) -> np.ndarray:
    """
    函数功能：
        根据 manifest record 读取一个 y_true 或 y_pred 数组。

    输入：
        record: `load_prediction_lookup()` 生成的字典，至少包含 `y_true_path`
            或 `y_pred_path`，并可选包含 `array_storage`、`y_true_row_index`、
            `y_pred_row_index`。
        array_kind: 只能是 `y_true` 或 `y_pred`。

    输出：
        单个 sample 的 float32 数组。

    关键约束：
        packed `.npy` 的第一维必须是 sample 维；row index 只在该维上取一行。
        这样可以把全量 shard 从海量小文件收敛为少量大数组，同时保持下游逐样本
        fusion / calibration 逻辑不变。
    """
    if array_kind not in {"y_true", "y_pred"}:
        raise ValueError(f"array_kind 只能是 y_true 或 y_pred，实际为 {array_kind}")

    path_key = f"{array_kind}_path"
    if path_key not in record:
        raise KeyError(f"prediction record 缺少字段：{path_key}")
    path = Path(str(record[path_key]))
    if not path.exists():
        raise FileNotFoundError(f"找不到 prediction cache 数组：{path}")

    storage = str(record.get("array_storage", PER_SAMPLE_NPY_STORAGE) or PER_SAMPLE_NPY_STORAGE)
    row_index = _optional_int(record, f"{array_kind}_row_index")
    array = np.load(path, mmap_mode="r")
    if storage == PACKED_NPY_STORAGE or row_index is not None:
        if row_index is None:
            raise ValueError(f"packed 数组缺少 {array_kind}_row_index：{path}")
        if row_index < 0 or row_index >= int(array.shape[0]):
            raise IndexError(f"{array_kind}_row_index 越界：index={row_index} shape={array.shape} path={path}")
        return np.asarray(array[row_index], dtype=np.float32)
    return np.asarray(array, dtype=np.float32)


def load_prediction_arrays_grouped(records: Sequence[Mapping[str, object]], array_kind: str) -> np.ndarray:
    """
    函数功能：
        按 record 顺序批量读取 y_true 或 y_pred，并对 packed npy 按路径分组复用
        mmap 文件句柄。

    输入：
        records: manifest record 序列，路径应已解析为可访问路径。
        array_kind: 只能是 `y_true` 或 `y_pred`。

    输出：
        第一维与 records 对齐的 float32 数组。

    关键约束：
        full-scale packed cache 中，同一个 batch 往往有多行指向同一个 `.npy`
        shard；这里每个路径只打开一次，避免把“每 sample 重复 np.load packed
        文件”变成正式 reader 的默认策略。legacy per-sample 小文件仍复用
        `load_prediction_array()`，保持旧 cache 语义一致。
    """
    if array_kind not in {"y_true", "y_pred"}:
        raise ValueError(f"array_kind 只能是 y_true 或 y_pred，实际为 {array_kind}")
    if not records:
        raise ValueError("records 不能为空")

    path_key = f"{array_kind}_path"
    row_key = f"{array_kind}_row_index"
    output: List[Optional[np.ndarray]] = [None] * len(records)
    grouped: dict[Path, List[Tuple[int, int]]] = {}
    for position, record in enumerate(records):
        storage = str(record.get("array_storage", PER_SAMPLE_NPY_STORAGE) or PER_SAMPLE_NPY_STORAGE)
        row_index = _optional_int(record, row_key)
        if storage == PACKED_NPY_STORAGE or row_index is not None:
            if row_index is None:
                raise ValueError(f"packed 数组缺少 {row_key}：{record.get(path_key)}")
            grouped.setdefault(Path(str(record[path_key])), []).append((position, int(row_index)))
            continue
        output[position] = load_prediction_array(record, array_kind)

    for path, positions in grouped.items():
        if not path.exists():
            raise FileNotFoundError(f"找不到 prediction cache 数组：{path}")
        array = np.load(path, mmap_mode="r")
        row_indices = np.asarray([row_index for _, row_index in positions], dtype=np.int64)
        if int(row_indices.min()) < 0 or int(row_indices.max()) >= int(array.shape[0]):
            raise IndexError(f"{array_kind}_row_index 越界：index={row_indices.tolist()} shape={array.shape} path={path}")
        values = np.asarray(array[row_indices], dtype=np.float32)
        for local_idx, (position, _) in enumerate(positions):
            output[position] = values[local_idx]

    if any(value is None for value in output):
        raise RuntimeError("batch 数组读取后仍存在空位")
    return np.stack([np.asarray(value, dtype=np.float32) for value in output], axis=0)
