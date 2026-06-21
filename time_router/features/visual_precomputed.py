"""
文件功能：
    提供 Stage 1 P16c 最小 precomputed/head-ready Visual FeatureProvider。

设计边界：
    该 provider 只读取调用方显式传入的、已经预计算好的 head-ready visual
    embedding/feature CSV，并通过 `load_batch(sample_keys)` 输出 canonical
    FeatureBatch。它不是 ViT provider，不构造 pseudo image，不处理 scaler，
    不读取 checkpoint，不拥有 run_dir，也不访问 prediction/oracle/expert error。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from time_router.protocols import FeatureBatch


class VisualPrecomputedFeatureProvider:
    """
    类功能：
        将已准备好的 head-ready visual feature CSV 适配为 canonical FeatureBatch。

    输入：
        feature_source_path 指向只读 CSV fixture；feature_columns 可显式指定特征列，
        未指定时自动选择表头中以 `feature_` 开头的列。

    输出：
        `load_batch(sample_keys)` 返回 FeatureBatch，sample_keys 按调用方请求顺序
        输出，features 固定为 `numpy.float32` 二维矩阵。

    关键约束：
        provider 只表达 precomputed/head-ready feature 边界；真实 history window、
        pseudo image、ViT encoder、scaler transform、checkpoint loading 和训练入口迁移
        都必须留给后续独立步骤。
    """

    provider_name = "VisualPrecomputedFeatureProvider"

    def __init__(
        self,
        *,
        feature_source_path: Path,
        feature_columns: Optional[Sequence[str]] = None,
        sample_key_column: str = "sample_key",
        feature_schema_name: str = "visual_precomputed_head_ready_v1",
        source_name: str | None = None,
        provider_name: str | None = None,
        dtype: Any = np.float32,
    ) -> None:
        self.feature_source_path = Path(feature_source_path)
        self.sample_key_column = str(sample_key_column)
        self.feature_schema_name = str(feature_schema_name)
        self.source_name = str(source_name) if source_name is not None else str(self.feature_source_path)
        self.provider_name = str(provider_name) if provider_name is not None else self.__class__.provider_name
        self.dtype = dtype

        self._rows_by_sample_key, inferred_columns = self._read_feature_csv()
        selected_columns = tuple(str(column) for column in (feature_columns or inferred_columns))
        if not selected_columns:
            raise ValueError("VisualPrecomputedFeatureProvider 需要至少一个 feature column")
        missing_columns = [column for column in selected_columns if column not in inferred_columns]
        if missing_columns:
            raise ValueError(f"visual precomputed feature CSV 缺少指定 feature column：{missing_columns}")
        self.feature_columns = selected_columns
        self.feature_dim = len(self.feature_columns)

        # 在初始化阶段完成数值校验，保证 load_batch 不会按样本路径延迟暴露坏 fixture。
        self._features_by_sample_key = self._build_feature_index()

    def load_batch(self, sample_keys: Sequence[str]) -> FeatureBatch:
        """
        函数功能：
            按调用方传入的 ordered sample_keys 输出 head-ready FeatureBatch。

        输入：
            sample_keys: manifest 或 split strategy 已确定的样本顺序；不能为空且
                不能重复。

        输出：
            FeatureBatch；`features` shape 为 `[len(sample_keys), feature_dim]`，
            dtype 固定为 `np.float32`。
        """
        ordered_keys = tuple(str(sample_key) for sample_key in sample_keys)
        if not ordered_keys:
            raise ValueError("VisualPrecomputedFeatureProvider.load_batch 必须显式传入非空 sample_keys")
        if len(ordered_keys) != len(set(ordered_keys)):
            raise ValueError("VisualPrecomputedFeatureProvider.load_batch 收到重复 sample_key")

        missing_keys = [sample_key for sample_key in ordered_keys if sample_key not in self._features_by_sample_key]
        if missing_keys:
            raise KeyError(f"visual precomputed feature CSV 缺少 sample_key：{missing_keys}")

        features = np.stack([self._features_by_sample_key[sample_key] for sample_key in ordered_keys], axis=0).astype(
            np.float32,
            copy=False,
        )
        if features.ndim != 2 or features.shape != (len(ordered_keys), self.feature_dim):
            raise ValueError(f"visual precomputed features shape 漂移：actual={features.shape}")
        if features.dtype != np.float32:
            raise ValueError(f"visual precomputed features dtype 必须为 float32：actual={features.dtype}")
        if not np.all(np.isfinite(features)):
            raise ValueError("visual precomputed features 包含 NaN 或 Inf")

        return FeatureBatch(
            sample_keys=ordered_keys,
            features=features,
            feature_schema={
                "provider_name": self.provider_name,
                "feature_schema_name": self.feature_schema_name,
                "feature_dim": self.feature_dim,
                "feature_columns": self.feature_columns,
                "head_ready": True,
                "loads_real_vit": False,
                "handles_scaler": False,
                "precomputed": True,
                "dtype": str(features.dtype),
            },
            extra={
                "provider_name": self.provider_name,
                "source_name": self.source_name,
                "feature_source_path": str(self.feature_source_path),
                "sample_key_column": self.sample_key_column,
                "num_available_rows": len(self._features_by_sample_key),
            },
        )

    def _read_feature_csv(self) -> tuple[dict[str, dict[str, str]], tuple[str, ...]]:
        """
        函数功能：
            读取 precomputed feature CSV 并建立 sample_key 行索引。

        关键约束：
            这里只解析显式传入的 feature CSV；重复或空 sample_key 立即失败，
            避免后续 FeatureBatch 静默错位。
        """
        rows_by_sample_key: dict[str, dict[str, str]] = {}
        with self.feature_source_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("visual precomputed feature CSV 缺少表头")
            fieldnames = tuple(str(field_name) for field_name in reader.fieldnames)
            if self.sample_key_column not in fieldnames:
                raise ValueError(f"visual precomputed feature CSV 缺少 sample_key 列：{self.sample_key_column}")
            feature_columns = tuple(column for column in fieldnames if column.startswith("feature_"))
            if not feature_columns:
                raise ValueError("visual precomputed feature CSV 未发现 feature_ 前缀列")

            for row in reader:
                sample_key = str(row[self.sample_key_column])
                if not sample_key:
                    raise ValueError("visual precomputed feature CSV 存在空 sample_key")
                if sample_key in rows_by_sample_key:
                    raise ValueError(f"visual precomputed feature CSV 存在重复 sample_key：{sample_key}")
                rows_by_sample_key[sample_key] = row
        if not rows_by_sample_key:
            raise ValueError("visual precomputed feature CSV 没有任何 feature row")
        return rows_by_sample_key, feature_columns

    def _build_feature_index(self) -> dict[str, np.ndarray]:
        """
        函数功能：
            将 CSV 字符串特征转换为有限的 float32 向量索引。
        """
        features_by_sample_key: dict[str, np.ndarray] = {}
        for sample_key, csv_row in self._rows_by_sample_key.items():
            try:
                feature_values = [float(csv_row[column]) for column in self.feature_columns]
            except ValueError as exc:
                raise ValueError(f"visual precomputed feature CSV 存在非数值特征：sample_key={sample_key}") from exc
            feature_vector = np.asarray(feature_values, dtype=self.dtype).astype(np.float32, copy=False)
            if feature_vector.shape != (self.feature_dim,):
                raise ValueError(f"visual precomputed feature vector shape 漂移：sample_key={sample_key}")
            if not np.all(np.isfinite(feature_vector)):
                raise ValueError(f"visual precomputed feature CSV 存在 NaN 或 Inf：sample_key={sample_key}")
            features_by_sample_key[sample_key] = feature_vector
        return features_by_sample_key
