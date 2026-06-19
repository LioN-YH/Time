#!/usr/bin/env python3
"""
文件功能：
    提供 Stage 1 共享 PredictionBatchReader，把 prediction cache manifest 组装为
    五专家 y_pred 与共享 y_true batch。

设计约束：
    - 支持 `packed_npy_v1` 与 legacy `per_sample_npy`；
    - manifest 原始行顺序只用于推断 sample_key 首次出现顺序，不能作为专家动作空间；
    - 专家动作空间顺序固定由 `model_columns` 控制；
    - 对同一 sample_key 的五专家 y_true 做内容一致性校验；
    - packed npy 读取必须使用 manifest 中的 row index；
    - 当传入 sample_key 子集时，只保留命中行，不构建全量 Python lookup。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from time_router.io.prediction_array_io import (
    load_prediction_arrays_grouped,
    resolve_cache_array_path,
)
from visual_router_experiments.common.prediction_cache_schema import (
    compute_window_metrics,
    validate_manifest_frame,
)


DEFAULT_MODEL_COLUMNS = ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]
REQUIRED_MANIFEST_COLUMNS = [
    "sample_key",
    "model_name",
    "y_true_path",
    "y_pred_path",
    "mae",
    "mse",
    "array_storage",
    "y_true_row_index",
    "y_pred_row_index",
]


@dataclass(frozen=True)
class PredictionBatch:
    """
    类功能：
        描述从 prediction cache 读取出的一个 batch。

    字段说明：
        sample_keys: 与 y_pred/y_true 第一维严格对齐的 sample_key 顺序。
        y_pred: `[num_samples, num_experts, pred_len, channels]`。
        y_true: `[num_samples, pred_len, channels]`。
        metadata: 记录 row index、原始 manifest 行、manifest 专家行顺序等诊断信息。
    """

    sample_keys: List[str]
    y_pred: np.ndarray
    y_true: np.ndarray
    metadata: Dict[str, object]


class PredictionBatchReader:
    """
    类功能：
        从 `merged_cache/manifest.csv` 或 fixture root 读取 prediction batch。

    关键约束：
        该 reader 不迁移任何正式训练入口，只收束 prediction cache 的读取和校验
        契约。正式 full-scale 场景应优先显式传入当前 shard/batch 的 sample_key；
        若不传 sample_key，reader 会按 manifest 首次出现顺序读取全部样本，适合
        golden fixture 或小规模 smoke，不适合作为 23M sample 的整批入口。
    """

    def __init__(
        self,
        *,
        manifest_path: Optional[Path] = None,
        fixture_root: Optional[Path] = None,
        model_columns: Optional[Sequence[str]] = None,
        chunk_rows: int = 200_000,
    ) -> None:
        if manifest_path is None and fixture_root is None:
            raise ValueError("必须提供 manifest_path 或 fixture_root")
        if manifest_path is not None and fixture_root is not None:
            raise ValueError("manifest_path 与 fixture_root 只能提供一个")
        if fixture_root is not None:
            manifest_path = Path(fixture_root) / "manifest.csv"
        assert manifest_path is not None
        self.manifest_path = Path(manifest_path)
        self.manifest_dir = self.manifest_path.parent
        self.model_columns = list(model_columns or DEFAULT_MODEL_COLUMNS)
        self.chunk_rows = int(chunk_rows)
        if self.chunk_rows <= 0:
            raise ValueError("chunk_rows 必须为正整数")
        if len(self.model_columns) != len(set(self.model_columns)):
            raise ValueError(f"model_columns 存在重复：{self.model_columns}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"找不到 prediction cache manifest：{self.manifest_path}")

    def load(
        self,
        sample_keys: Optional[Sequence[str]] = None,
        *,
        verify_metrics: bool = True,
    ) -> PredictionBatch:
        """
        函数功能：
            读取一个 prediction batch，并按 `model_columns` 重排专家维度。

        输入：
            sample_keys: 可选 sample_key 顺序；为空时按 manifest 中首次出现顺序推断。
            verify_metrics: 是否用 y_true/y_pred 复算每个专家 manifest 的 MAE/MSE。

        输出：
            `PredictionBatch`，其数组维度与 sample_key/model_columns 顺序严格对齐。
        """
        explicit_keys = None if sample_keys is None else [str(key) for key in sample_keys]
        if explicit_keys is not None and len(explicit_keys) != len(set(explicit_keys)):
            raise ValueError("sample_keys 中存在重复 key")

        manifest_df, ordered_keys = self._read_manifest_subset(explicit_keys)
        self._validate_manifest_subset(manifest_df)
        record_lookup, manifest_model_order = self._build_record_lookup(manifest_df)

        y_true_records = [record_lookup[(sample_key, self.model_columns[0])] for sample_key in ordered_keys]
        y_true = load_prediction_arrays_grouped(y_true_records, "y_true").astype(np.float32)

        model_pred_arrays: List[np.ndarray] = []
        row_indices: Dict[str, Dict[str, Tuple[Optional[int], Optional[int]]]] = {}
        for model_name in self.model_columns:
            model_records = [record_lookup[(sample_key, model_name)] for sample_key in ordered_keys]
            model_y_pred = load_prediction_arrays_grouped(model_records, "y_pred").astype(np.float32)
            model_pred_arrays.append(model_y_pred)
            for sample_idx, (sample_key, first_record, current_record) in enumerate(
                zip(ordered_keys, y_true_records, model_records)
            ):
                row_indices.setdefault(sample_key, {})[model_name] = (
                    _optional_int(current_record, "y_true_row_index"),
                    _optional_int(current_record, "y_pred_row_index"),
                )
                self._validate_shared_y_true(
                    sample_key=sample_key,
                    sample_idx=sample_idx,
                    first_record=first_record,
                    current_record=current_record,
                    expected_y_true=y_true[sample_idx],
                )
                if verify_metrics:
                    self._validate_manifest_metrics(
                        sample_key=sample_key,
                        model_name=model_name,
                        record=current_record,
                        y_true=y_true[sample_idx],
                        y_pred=model_y_pred[sample_idx],
                    )

        y_pred = np.stack(model_pred_arrays, axis=1).astype(np.float32)
        if y_pred.shape[0] != y_true.shape[0] or y_pred.shape[2:] != y_true.shape[1:]:
            raise ValueError(f"y_pred/y_true shape 不一致：y_pred={y_pred.shape} y_true={y_true.shape}")
        if not np.isfinite(y_pred).all() or not np.isfinite(y_true).all():
            raise ValueError("prediction batch 中存在非有限值")

        return PredictionBatch(
            sample_keys=list(ordered_keys),
            y_pred=y_pred,
            y_true=y_true,
            metadata={
                "manifest_path": str(self.manifest_path),
                "model_columns": list(self.model_columns),
                "manifest_rows": manifest_df.copy(),
                "manifest_model_order_by_sample": manifest_model_order,
                "row_indices_by_sample_model": row_indices,
            },
        )

    def _read_manifest_subset(self, explicit_keys: Optional[Sequence[str]]) -> Tuple[pd.DataFrame, List[str]]:
        """函数功能：按 chunk 扫描 manifest，只保留目标 sample_key 的记录。"""
        target_set = None if explicit_keys is None else set(explicit_keys)
        chunks: List[pd.DataFrame] = []
        inferred_keys: List[str] = []
        seen_keys: set[str] = set()
        matched_pairs: set[Tuple[str, str]] = set()
        for chunk_df in pd.read_csv(self.manifest_path, chunksize=self.chunk_rows):
            missing = sorted(set(REQUIRED_MANIFEST_COLUMNS).difference(chunk_df.columns))
            if missing:
                raise ValueError(f"prediction cache manifest 缺少字段：{missing}")
            chunk_df = chunk_df.copy()
            chunk_df["sample_key"] = chunk_df["sample_key"].astype(str)
            chunk_df["model_name"] = chunk_df["model_name"].astype(str)
            if target_set is None:
                selected_df = chunk_df
            else:
                selected_df = chunk_df[chunk_df["sample_key"].isin(target_set)].copy()
            if selected_df.empty:
                continue
            for sample_key in selected_df["sample_key"].tolist():
                if sample_key not in seen_keys:
                    inferred_keys.append(sample_key)
                    seen_keys.add(sample_key)
            chunks.append(selected_df)
            if target_set is not None:
                matched_pairs.update(zip(selected_df["sample_key"], selected_df["model_name"]))
                # 显式 sample_key 场景下，命中当前 batch 的五专家记录后即可停止扫描，
                # 避免 full-scale merged manifest 被整表读入或构建全量 Python lookup。
                if len(matched_pairs) == len(target_set) * len(self.model_columns):
                    break

        if not chunks:
            raise ValueError("manifest 中没有命中的 sample_key")
        manifest_df = pd.concat(chunks, ignore_index=True)
        ordered_keys = list(explicit_keys) if explicit_keys is not None else inferred_keys
        missing_keys = [sample_key for sample_key in ordered_keys if sample_key not in set(manifest_df["sample_key"])]
        if missing_keys:
            raise ValueError(f"manifest 缺少 sample_key，示例：{missing_keys[:5]}")
        return manifest_df, ordered_keys

    def _validate_manifest_subset(self, manifest_df: pd.DataFrame) -> None:
        """函数功能：复用 schema 校验，并确认每个 sample 覆盖固定专家集合。"""
        validate_manifest_frame(
            manifest_df,
            expected_models=list(self.model_columns),
            require_unique_model_per_sample=True,
            require_shared_y_true_path=False,
        )

    def _build_record_lookup(
        self, manifest_df: pd.DataFrame
    ) -> Tuple[Dict[Tuple[str, str], Dict[str, object]], Dict[str, List[str]]]:
        """函数功能：构建当前命中子集的 sample/model record lookup 和原始专家顺序元数据。"""
        record_lookup: Dict[Tuple[str, str], Dict[str, object]] = {}
        manifest_model_order: Dict[str, List[str]] = {}
        for sample_key, group in manifest_df.groupby("sample_key", sort=False):
            manifest_model_order[str(sample_key)] = group["model_name"].astype(str).tolist()
            for row in group.to_dict(orient="records"):
                record = dict(row)
                record["y_true_path"] = str(resolve_cache_array_path(str(record["y_true_path"]), self.manifest_dir))
                record["y_pred_path"] = str(resolve_cache_array_path(str(record["y_pred_path"]), self.manifest_dir))
                record_lookup[(str(record["sample_key"]), str(record["model_name"]))] = record
        return record_lookup, manifest_model_order

    def _validate_shared_y_true(
        self,
        *,
        sample_key: str,
        sample_idx: int,
        first_record: Mapping[str, object],
        current_record: Mapping[str, object],
        expected_y_true: np.ndarray,
    ) -> None:
        """函数功能：确认同一 sample_key 下五专家共享 y_true 内容一致。"""
        first_identity = (str(first_record["y_true_path"]), _optional_int(first_record, "y_true_row_index"))
        current_identity = (str(current_record["y_true_path"]), _optional_int(current_record, "y_true_row_index"))
        if current_identity == first_identity:
            return
        current_y_true = load_prediction_arrays_grouped([current_record], "y_true")[0]
        if not np.array_equal(expected_y_true, current_y_true):
            raise ValueError(f"同一 sample_key 的五专家 y_true 不一致：{sample_key} sample_idx={sample_idx}")

    def _validate_manifest_metrics(
        self,
        *,
        sample_key: str,
        model_name: str,
        record: Mapping[str, object],
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """函数功能：复算单专家 MAE/MSE，确认 row index 读取与 manifest 指标一致。"""
        metrics = compute_window_metrics(y_true=y_true, y_pred=y_pred)
        for metric_name in ("mae", "mse"):
            actual = float(metrics[metric_name])
            expected = float(record[metric_name])
            if not np.isclose(actual, expected, rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"{sample_key}/{model_name} {metric_name} 与 manifest 不一致："
                    f"actual={actual:.12f} expected={expected:.12f}"
                )


def _optional_int(record: Mapping[str, object], key: str) -> Optional[int]:
    """函数功能：安全读取 manifest 中可能为空的 row index。"""
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return int(value)
