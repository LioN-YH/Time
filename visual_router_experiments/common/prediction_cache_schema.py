#!/usr/bin/env python3
"""
文件功能：
    定义 Visual Router Stage 1 使用的 window-level prediction cache schema。

设计背景：
    Stage 1 需要冻结多个专家模型，在 vali/test 的每个 item-channel-window 上保存
    y_true、y_pred 和窗口级 MAE/MSE。router 训练、hard top-1 routing、softmax
    fusion 都应复用同一套 cache key，避免后续不同专家或不同 split 难以对齐。

关键约束：
    - 路由粒度是 item-channel-window。
    - config_name 表示历史长度、预测长度和特征模式的组合；正式 Stage 1 只能在
      同一 config_name 内比较专家和训练 router。
    - 当前 Quito 的 S 配置会把 item-channel 展平成 dataset 的样本序列。
    - 大数组可以外置为 npy/npz/zarr/memmap，本 schema 的记录只保存路径和元信息。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


CACHE_SCHEMA_VERSION = "visual_router_prediction_cache_v1"
SPLITS = ("vali", "test")


@dataclass(frozen=True)
class PredictionCacheKey:
    """
    类功能：
        唯一标识一个 item-channel-window 样本。

    字段说明：
        config_name: Quito 配置名，例如 96_48_S。
        split: vali 或 test。
        dataset_name: Quito dataset 配置名，例如 TEST_DATA_MIN / TEST_DATA_HOUR。
        item_id: 原始 item_id。
        channel_id: 原始通道编号；单变量或 S 展开后仍保留该字段。
        window_index: 当前 item-channel 在 split 内的滑动窗口序号。
    """

    config_name: str
    split: str
    dataset_name: str
    item_id: int
    channel_id: int
    window_index: int

    def as_string(self) -> str:
        """函数功能：生成稳定字符串 key，便于文件命名和多表 join。"""
        return (
            f"{self.config_name}__{self.split}__{self.dataset_name}"
            f"__item{self.item_id}__ch{self.channel_id}__win{self.window_index}"
        )


@dataclass(frozen=True)
class PredictionCacheRecord:
    """
    类功能：
        描述某个专家在一个 item-channel-window 上的预测缓存记录。

    设计说明：
        y_true/y_pred 可能很大，因此记录中默认保存外置数组路径；小规模 pilot 可选择
        直接把数组保存为 .npy，并在 manifest 中记录相对路径。
    """

    cache_version: str
    sample_key: str
    config_name: str
    split: str
    dataset_name: str
    item_id: int
    channel_id: int
    window_index: int
    history_length: int
    pred_length: int
    model_name: str
    expert_version: str
    checkpoint_selection: str
    y_true_path: str
    y_pred_path: str
    mae: float
    mse: float
    array_storage: str = "per_sample_npy"
    y_true_row_index: Optional[int] = None
    y_pred_row_index: Optional[int] = None


def compute_window_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    函数功能：
        计算单个窗口、单个专家的 MAE/MSE。

    输入：
        y_true: 真实未来序列，形状通常为 [pred_len, channels] 或 [pred_len]。
        y_pred: 专家预测序列，形状必须与 y_true 一致。

    输出：
        包含 mae 和 mse 的字典。
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true 与 y_pred 形状不一致：{y_true.shape} vs {y_pred.shape}")

    error = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error ** 2)),
    }


def make_prediction_record(
    *,
    key: PredictionCacheKey,
    history_length: int,
    pred_length: int,
    model_name: str,
    expert_version: str,
    checkpoint_selection: str,
    y_true_path: Path,
    y_pred_path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    array_storage: str = "per_sample_npy",
    y_true_row_index: Optional[int] = None,
    y_pred_row_index: Optional[int] = None,
) -> PredictionCacheRecord:
    """
    函数功能：
        基于 sample key、专家信息和数组内容创建一条标准 cache manifest 记录。

    约束：
        该函数只创建元信息记录，不负责写数组文件；数组写入由具体 cache builder 决定。
    """
    if key.split not in SPLITS:
        raise ValueError(f"split 必须是 {SPLITS} 之一，实际为 {key.split}")

    metrics = compute_window_metrics(y_true=y_true, y_pred=y_pred)
    return PredictionCacheRecord(
        cache_version=CACHE_SCHEMA_VERSION,
        sample_key=key.as_string(),
        config_name=key.config_name,
        split=key.split,
        dataset_name=key.dataset_name,
        item_id=int(key.item_id),
        channel_id=int(key.channel_id),
        window_index=int(key.window_index),
        history_length=int(history_length),
        pred_length=int(pred_length),
        model_name=model_name,
        expert_version=expert_version,
        checkpoint_selection=checkpoint_selection,
        y_true_path=str(y_true_path),
        y_pred_path=str(y_pred_path),
        mae=metrics["mae"],
        mse=metrics["mse"],
        array_storage=str(array_storage),
        y_true_row_index=y_true_row_index,
        y_pred_row_index=y_pred_row_index,
    )


def records_to_frame(records: Iterable[PredictionCacheRecord]) -> pd.DataFrame:
    """函数功能：将 cache manifest 记录转换为 DataFrame，便于写 parquet/csv。"""
    rows = []
    for record in records:
        if isinstance(record, PredictionCacheRecord):
            rows.append(asdict(record))
        elif isinstance(record, Mapping):
            rows.append(dict(record))
        else:
            raise TypeError(f"不支持的 prediction cache record 类型：{type(record)!r}")
    return pd.DataFrame(rows)


def validate_manifest_frame(
    manifest_df: pd.DataFrame,
    *,
    expected_models: Optional[List[str]] = None,
    require_unique_model_per_sample: bool = True,
    require_shared_y_true_path: bool = False,
) -> None:
    """
    函数功能：
        对 prediction cache manifest 做基础一致性检查。

    检查项：
        - 必要字段是否存在；
        - cache_version 是否一致；
        - sample_key 是否与 config/split/dataset/item/channel/window 字段一致；
        - 同一个 sample_key 下的 config_name、history_length、pred_length 等元信息是否一致；
        - 可选检查同一个 sample_key 下 y_true_path 是否一致；
        - sample_key + model_name 是否重复；
        - 可选检查每个 sample_key 是否覆盖 expected_models。
    """
    required_cols = {
        "cache_version",
        "sample_key",
        "config_name",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "history_length",
        "pred_length",
        "model_name",
        "expert_version",
        "checkpoint_selection",
        "y_true_path",
        "y_pred_path",
        "mae",
        "mse",
    }
    missing_cols = sorted(required_cols.difference(manifest_df.columns))
    if missing_cols:
        raise ValueError(f"prediction cache manifest 缺少字段：{missing_cols}")

    versions = set(manifest_df["cache_version"].unique())
    if versions != {CACHE_SCHEMA_VERSION}:
        raise ValueError(f"cache_version 不一致：{versions}")

    expected_keys = manifest_df.apply(
        lambda row: PredictionCacheKey(
            config_name=str(row["config_name"]),
            split=str(row["split"]),
            dataset_name=str(row["dataset_name"]),
            item_id=int(row["item_id"]),
            channel_id=int(row["channel_id"]),
            window_index=int(row["window_index"]),
        ).as_string(),
        axis=1,
    )
    bad_key_mask = manifest_df["sample_key"].astype(str) != expected_keys
    if bool(bad_key_mask.any()):
        bad_row = manifest_df.loc[bad_key_mask].iloc[0]
        raise ValueError(
            "sample_key 与 config/split/dataset/item/channel/window 字段不一致；"
            f"示例 sample_key={bad_row['sample_key']}"
        )

    # 同一个 sample_key 表示同一个待路由窗口，因此这些元信息必须在五专家记录间完全一致。
    stable_cols = [
        "config_name",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "history_length",
        "pred_length",
    ]
    if require_shared_y_true_path:
        stable_cols.append("y_true_path")
        if "y_true_row_index" in manifest_df.columns:
            stable_cols.append("y_true_row_index")
    unstable = manifest_df.groupby("sample_key")[stable_cols].nunique(dropna=False)
    bad_stable = unstable[(unstable > 1).any(axis=1)]
    if not bad_stable.empty:
        example_key = bad_stable.index[0]
        bad_cols = bad_stable.columns[bad_stable.loc[example_key] > 1].tolist()
        raise ValueError(f"同一 sample_key 的元信息不一致；示例 {example_key}: {bad_cols}")

    if require_unique_model_per_sample:
        duplicate_count = int(manifest_df.duplicated(["sample_key", "model_name"]).sum())
        if duplicate_count:
            raise ValueError(f"sample_key + model_name 存在 {duplicate_count} 条重复记录。")

    if "array_storage" in manifest_df.columns:
        packed_mask = manifest_df["array_storage"].astype(str) == "packed_npy_v1"
        if packed_mask.any():
            packed_frame = manifest_df.loc[packed_mask, :]
            if "y_true_row_index" not in packed_frame.columns or "y_pred_row_index" not in packed_frame.columns:
                raise ValueError("packed_npy_v1 记录必须包含 y_true_row_index 和 y_pred_row_index")
            if packed_frame["y_true_row_index"].isna().any() or packed_frame["y_pred_row_index"].isna().any():
                raise ValueError("packed_npy_v1 记录的 row_index 不能为空")

    if expected_models:
        expected_set = set(expected_models)
        model_sets = manifest_df.groupby("sample_key")["model_name"].apply(set)
        bad_keys = model_sets[model_sets.map(lambda models: models != expected_set)]
        if not bad_keys.empty:
            example_key = bad_keys.index[0]
            raise ValueError(
                "部分 sample_key 的专家集合不完整或不一致；"
                f"示例 {example_key}: {sorted(bad_keys.iloc[0])}"
            )


def infer_window_index_from_batch(
    *,
    batch_idx: int,
    batch_size: int,
    row_in_batch: int,
) -> int:
    """
    函数功能：
        在单 item-channel dataset 且 DataLoader shuffle=False 时，从 batch 位置恢复窗口序号。

    设计说明：
        Stage 1 pilot 计划先对每个 user/item 独立 select_user_data，再构造不打乱的
        DataLoader。此时 dataset 的样本序列就是该 item-channel 在当前 split 内的
        滑动窗口序列，因此全局样本序号可作为 window_index。
    """
    if batch_idx < 0 or batch_size <= 0 or row_in_batch < 0:
        raise ValueError("batch_idx、batch_size、row_in_batch 参数非法。")
    if row_in_batch >= batch_size:
        raise ValueError("row_in_batch 不能大于或等于 batch_size。")
    return int(batch_idx * batch_size + row_in_batch)
