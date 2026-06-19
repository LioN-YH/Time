"""
文件功能：
    提供 Stage 1 P7a 最小 TimeFuseFeatureCacheProvider。

设计边界：
    该 provider 只读取调用方显式传入的小规模 feature CSV，并把指定
    sample_key batch 包装为 FeatureBatch。它不读取 prediction cache、
    oracle/TSF、y_true 或 expert error，不做 scaler fit，不创建 run_dir，
    也不写 status/metadata/CSV/JSON/Parquet。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from time_router.protocols import FeatureBatch


class TimeFuseFeatureCacheProvider:
    """
    类功能：
        将显式 TimeFuse feature CSV 适配为 canonical FeatureBatch。

    输入：
        feature_csv_path 指向只读 feature CSV；sample_key_column 指定样本键列；
        feature_columns 可显式指定特征列，未指定时从表头中排除 sample_key 列
        后按 CSV 原始顺序推断。

    输出：
        `load_batch(...)` 返回 FeatureBatch，其中 sample_keys 为 tuple，
        features 为 numpy float32 array，feature_schema 记录 schema 名称、
        feature_columns、feature_dim 和 source。

    关键约束：
        调用方必须显式传入 sample_keys。provider 只读 feature CSV，不读取
        prediction/oracle，不拟合 scaler，不决定输出目录。
    """

    provider_name = "TimeFuseFeatureCacheProvider"

    def __init__(
        self,
        *,
        feature_csv_path: Path,
        feature_columns: Optional[Sequence[str]] = None,
        sample_key_column: str = "sample_key",
        feature_schema_name: str = "timefuse_single_variable_meta_v1",
        dtype: Any = np.float32,
    ) -> None:
        self.feature_csv_path = Path(feature_csv_path)
        self.sample_key_column = str(sample_key_column)
        self.feature_schema_name = str(feature_schema_name)
        self.dtype = dtype
        self._rows_by_sample_key, inferred_columns = self._read_feature_csv()
        selected_columns = tuple(str(column) for column in (feature_columns or inferred_columns))
        if not selected_columns:
            raise ValueError("TimeFuseFeatureCacheProvider 需要至少一个 feature column")
        missing_columns = [column for column in selected_columns if column not in inferred_columns]
        if missing_columns:
            raise ValueError(f"feature CSV 缺少指定 feature column：{missing_columns}")
        self.feature_columns = selected_columns

    def load_batch(self, sample_keys: Sequence[str]) -> FeatureBatch:
        """
        函数功能：
            显式读取一个 sample_key batch，并包装为 FeatureBatch。

        输入：
            sample_keys: 调用方指定的 sample_key 顺序；不能为空且不能重复。

        输出：
            FeatureBatch，sample_keys 保持调用方顺序，features 第一维与之严格对齐。
        """
        ordered_keys = tuple(str(sample_key) for sample_key in sample_keys)
        if not ordered_keys:
            raise ValueError("TimeFuseFeatureCacheProvider.load_batch 必须显式传入非空 sample_keys")
        if len(ordered_keys) != len(set(ordered_keys)):
            raise ValueError("TimeFuseFeatureCacheProvider.load_batch 收到重复 sample_key")

        missing_keys = [sample_key for sample_key in ordered_keys if sample_key not in self._rows_by_sample_key]
        if missing_keys:
            raise KeyError(f"feature CSV 缺少 sample_key：{missing_keys}")

        feature_rows = []
        for sample_key in ordered_keys:
            csv_row = self._rows_by_sample_key[sample_key]
            try:
                feature_rows.append([float(csv_row[column]) for column in self.feature_columns])
            except ValueError as exc:
                raise ValueError(f"feature CSV 存在非数值特征：sample_key={sample_key}") from exc

        features = np.asarray(feature_rows, dtype=self.dtype)
        source = str(self.feature_csv_path)
        return FeatureBatch(
            sample_keys=ordered_keys,
            features=features,
            feature_schema={
                "feature_schema_name": self.feature_schema_name,
                "feature_columns": self.feature_columns,
                "feature_dim": len(self.feature_columns),
                "source": source,
            },
            extra={
                "provider_name": self.provider_name,
                "sample_key_column": self.sample_key_column,
                "feature_csv_path": source,
                "num_available_rows": len(self._rows_by_sample_key),
                "dtype": str(features.dtype),
            },
        )

    def _read_feature_csv(self) -> tuple[dict[str, dict[str, str]], tuple[str, ...]]:
        """
        函数功能：
            读取显式 feature CSV，并按 sample_key 建立小规模内存索引。

        关键约束：
            这里只解析 feature CSV 本身，不推断 split、不读取 prediction/oracle，
            也不写任何运行产物；重复 sample_key 会被拒绝，避免静默覆盖。
        """
        rows_by_sample_key: dict[str, dict[str, str]] = {}
        with self.feature_csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("feature CSV 缺少表头")
            fieldnames = tuple(str(field_name) for field_name in reader.fieldnames)
            if self.sample_key_column not in fieldnames:
                raise ValueError(f"feature CSV 缺少 sample_key 列：{self.sample_key_column}")
            feature_columns = tuple(column for column in fieldnames if column != self.sample_key_column)
            for row in reader:
                sample_key = str(row[self.sample_key_column])
                if not sample_key:
                    raise ValueError("feature CSV 存在空 sample_key")
                if sample_key in rows_by_sample_key:
                    raise ValueError(f"feature CSV 存在重复 sample_key：{sample_key}")
                rows_by_sample_key[sample_key] = row
        if not rows_by_sample_key:
            raise ValueError("feature CSV 没有任何 feature row")
        return rows_by_sample_key, feature_columns
