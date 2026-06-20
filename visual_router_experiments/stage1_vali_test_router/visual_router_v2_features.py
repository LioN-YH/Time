#!/usr/bin/env python3
"""
文件功能：
    定义 Visual Router V2 Round 1 pilot feature cache 的共享 schema、校验和
    RevIN auxiliary feature 构造逻辑。

设计约束：
    - 只从历史窗口 x 计算可部署特征，不读取 future y、oracle error 或专家预测；
    - visual feature 与 aux feature 均以 float32 保存；
    - 所有写盘 shard 必须可独立校验 sample_key、order_index、shape 和 finite。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


FEATURE_SCHEMA_VERSION = "visual_router_v2_round1_feature_cache_v1"
AUX_FEATURE_COLUMNS = ["mean", "log_std", "min", "max", "range", "clip_ratio"]
DEFAULT_ROUND1_SAMPLE_SETS = ("pilot_train", "pilot_selection", "diagnostic_balanced")
FINAL_TEST_ONLY_SAMPLE_SET = "pilot_test"


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：用同目录临时文件原子写出 JSON，避免中断留下半文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def load_and_validate_sample_csv(path: Path, *, sample_set: str, max_samples: int | None = None) -> pd.DataFrame:
    """
    函数功能：
        读取 P0 sample CSV，并校验 sample_set、order_index、sample_key 和稳定元信息。

    输入：
        path: P0 生成的 `*_sample_keys.csv`。
        sample_set: 期望的集合名。
        max_samples: smoke 截断样本数；None 表示读取完整集合。

    输出：
        按 `order_index` 升序排列的 DataFrame。
    """
    required_cols = {
        "sample_set",
        "order_index",
        "sample_key",
        "config_name",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
    }
    if not path.exists():
        raise FileNotFoundError(f"找不到 P0 sample CSV：{path}")
    frame = pd.read_csv(path)
    missing = sorted(required_cols - set(frame.columns))
    if missing:
        raise ValueError(f"{path} 缺少必要字段：{missing}")
    frame = frame.sort_values("order_index", kind="mergesort").reset_index(drop=True)
    if max_samples is not None:
        frame = frame.head(int(max_samples)).copy()
    if frame.empty:
        raise ValueError(f"{path} 读取后为空")
    bad_set = frame["sample_set"].astype(str) != str(sample_set)
    if bool(bad_set.any()):
        bad_row = frame.loc[bad_set].iloc[0]
        raise ValueError(f"{path} 中 sample_set 不一致：expected={sample_set} actual={bad_row['sample_set']}")
    order_index = frame["order_index"].to_numpy(dtype=np.int64, copy=False)
    expected = np.arange(int(order_index[0]), int(order_index[0]) + len(frame), dtype=np.int64)
    if not np.array_equal(order_index, expected):
        raise ValueError(f"{path} 的 order_index 不连续或未按顺序排列")
    if int(order_index[0]) != 0:
        raise ValueError(f"{path} 的 order_index 必须从 0 开始，实际为 {int(order_index[0])}")
    if frame["sample_key"].astype(str).duplicated().any():
        dup = frame.loc[frame["sample_key"].astype(str).duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"{path} 中 sample_key 重复，示例：{dup}")
    return frame


def compute_revin_aux_from_x(x_batch: np.ndarray, *, clip: float) -> np.ndarray:
    """
    函数功能：
        从历史窗口 x 计算 Round 1 限定的 6 维 RevIN auxiliary feature。

    输入：
        x_batch: `[N, L, C]`、`[N, L]` 或等价历史窗口数组；多通道会按最后一维均值折成
            单变量序列，与当前视觉伪图像 `_as_series_batch` 口径一致。
        clip: 与伪图像 pixel 映射一致的截断阈值；`clip_ratio` 表示 RevIN 标准化后
            绝对值超过该阈值的比例。

    输出：
        `[N, 6]` float32，列顺序为 `mean, log_std, min, max, range, clip_ratio`。
    """
    array = np.asarray(x_batch, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[-1] == 1:
            series = array[..., 0]
        else:
            series = array.mean(axis=-1, dtype=np.float32)
    elif array.ndim == 2:
        series = array
    else:
        raise ValueError(f"x_batch 维度必须为 2/3，实际为 {array.shape}")
    mean = series.mean(axis=1, dtype=np.float32)
    std = series.std(axis=1, dtype=np.float32)
    std = np.maximum(std, np.float32(1e-6))
    min_value = series.min(axis=1)
    max_value = series.max(axis=1)
    value_range = max_value - min_value
    normalized = (series - mean[:, None]) / std[:, None]
    clip_ratio = (np.abs(normalized) > float(clip)).mean(axis=1, dtype=np.float32)
    aux = np.stack([mean, np.log(std), min_value, max_value, value_range, clip_ratio], axis=1).astype(np.float32)
    if aux.shape[1] != len(AUX_FEATURE_COLUMNS):
        raise ValueError(f"RevIN aux 维度异常：{aux.shape}")
    if not np.isfinite(aux).all():
        raise ValueError("RevIN aux 中存在 NaN/Inf")
    return aux


def validate_feature_arrays(
    *,
    sample_keys: Sequence[str],
    order_index: np.ndarray,
    cls_embedding: np.ndarray,
    mean_patch_embedding: np.ndarray,
    revin_aux: np.ndarray,
) -> None:
    """函数功能：集中校验一个 shard 即将写盘的字段完整性和 feature shape。"""
    sample_count = len(sample_keys)
    if sample_count <= 0:
        raise ValueError("shard 样本数必须大于 0")
    if np.asarray(order_index).shape != (sample_count,):
        raise ValueError(f"order_index shape 异常：{np.asarray(order_index).shape} sample_count={sample_count}")
    expected_order = np.arange(int(order_index[0]), int(order_index[0]) + sample_count, dtype=np.int64)
    if not np.array_equal(np.asarray(order_index, dtype=np.int64), expected_order):
        raise ValueError("shard 内 order_index 不连续")
    cls = np.asarray(cls_embedding)
    mean_patch = np.asarray(mean_patch_embedding)
    aux = np.asarray(revin_aux)
    if cls.ndim != 2 or cls.shape[0] != sample_count:
        raise ValueError(f"cls_embedding shape 异常：{cls.shape}")
    if mean_patch.shape != cls.shape:
        raise ValueError(f"mean_patch_embedding shape 与 cls 不一致：mean_patch={mean_patch.shape} cls={cls.shape}")
    if aux.shape != (sample_count, len(AUX_FEATURE_COLUMNS)):
        raise ValueError(f"revin_aux shape 异常：{aux.shape}")
    if cls.dtype != np.float32 or mean_patch.dtype != np.float32 or aux.dtype != np.float32:
        raise ValueError(f"feature dtype 必须为 float32：cls={cls.dtype} mean_patch={mean_patch.dtype} aux={aux.dtype}")
    if not (np.isfinite(cls).all() and np.isfinite(mean_patch).all() and np.isfinite(aux).all()):
        raise ValueError("feature shard 中存在 NaN/Inf")


def write_feature_shard_atomic(
    *,
    shard_path: Path,
    sample_keys: Sequence[str],
    order_index: np.ndarray,
    cls_embedding: np.ndarray,
    mean_patch_embedding: np.ndarray,
    revin_aux: np.ndarray,
) -> None:
    """函数功能：用 tmp 文件 + atomic rename 写出一个 `.npz` feature shard。"""
    validate_feature_arrays(
        sample_keys=sample_keys,
        order_index=order_index,
        cls_embedding=cls_embedding,
        mean_patch_embedding=mean_patch_embedding,
        revin_aux=revin_aux,
    )
    shard_path = Path(shard_path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_key=np.asarray([str(key) for key in sample_keys], dtype=object),
            order_index=np.asarray(order_index, dtype=np.int64),
            cls_embedding=np.asarray(cls_embedding, dtype=np.float32),
            mean_patch_embedding=np.asarray(mean_patch_embedding, dtype=np.float32),
            revin_aux=np.asarray(revin_aux, dtype=np.float32),
        )
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(shard_path)


def validate_existing_shard(
    *,
    shard_path: Path,
    expected_sample_keys: Sequence[str],
    expected_order_index: np.ndarray,
) -> tuple[int, int, int]:
    """
    函数功能：
        校验已有 shard 是否可安全 skip。

    输出：
        `(sample_count, visual_feature_dim, aux_feature_dim)`。
    """
    with np.load(shard_path, allow_pickle=True) as data:
        sample_keys = [str(value) for value in data["sample_key"].tolist()]
        order_index = np.asarray(data["order_index"], dtype=np.int64)
        cls_embedding = np.asarray(data["cls_embedding"], dtype=np.float32)
        mean_patch_embedding = np.asarray(data["mean_patch_embedding"], dtype=np.float32)
        revin_aux = np.asarray(data["revin_aux"], dtype=np.float32)
    if sample_keys != [str(key) for key in expected_sample_keys]:
        raise ValueError(f"已有 shard sample_key 与 P0 CSV 不一致：{shard_path}")
    if not np.array_equal(order_index, np.asarray(expected_order_index, dtype=np.int64)):
        raise ValueError(f"已有 shard order_index 与 P0 CSV 不一致：{shard_path}")
    validate_feature_arrays(
        sample_keys=sample_keys,
        order_index=order_index,
        cls_embedding=cls_embedding,
        mean_patch_embedding=mean_patch_embedding,
        revin_aux=revin_aux,
    )
    return int(len(sample_keys)), int(cls_embedding.shape[1]), int(revin_aux.shape[1])
