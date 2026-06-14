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
from typing import Mapping, Optional, Union

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

