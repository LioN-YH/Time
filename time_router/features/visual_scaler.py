"""
文件功能：
    提供 Stage 1 P16d 最小 loaded Visual FeatureScaler 边界。

设计边界：
    LoadedFeatureScaler 只消费调用方显式传入或显式路径读取的 scaler state，
    将 raw/pre-head FeatureBatch.features 执行 `(raw - mean) / scale`，输出
    head-ready float32 FeatureBatch。它不执行训练期参数估计，不读取 checkpoint，
    不接 ViT，不拥有 run_dir，也不访问 prediction/oracle/expert error。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from time_router.protocols import FeatureBatch


class LoadedFeatureScaler:
    """
    类功能：
        使用已加载 scaler state 对 raw/pre-head FeatureBatch 做确定性 transform。

    输入：
        mean 和 scale 是 Runtime 已经加载好的 scaler 参数；feature_columns 可选，
        用于和 FeatureBatch.feature_schema 中的 feature_columns 显式对齐。

    输出：
        `transform(feature_batch)` 返回新的 FeatureBatch，sample_keys 保持输入顺序，
        features shape 不变，dtype 固定为 np.float32。

    关键约束：
        本类没有训练期参数估计入口，也不根据 batch 数据自动估计 mean/std；
        scaler state discovery、checkpoint loading、ViT embedding 和正式训练入口
        都必须留在 Runtime/entrypoint 侧。
    """

    scaler_name = "LoadedFeatureScaler"
    default_schema_version = "stage1_visual_feature_scaler_v1"

    def __init__(
        self,
        *,
        mean: Sequence[float],
        scale: Sequence[float],
        feature_columns: Sequence[str] | None = None,
        scaler_schema_version: str = default_schema_version,
        source_name: str | None = None,
    ) -> None:
        self.scaler_schema_version = str(scaler_schema_version)
        self.source_name = str(source_name) if source_name is not None else "runtime_supplied_loaded_state"
        self.mean = self._validate_vector(mean, vector_name="mean")
        self.scale = self._validate_vector(scale, vector_name="scale")
        if self.mean.shape != self.scale.shape:
            raise ValueError(f"mean/scale 长度必须一致：mean={self.mean.shape} scale={self.scale.shape}")
        if np.any(self.scale == 0.0):
            raise ValueError("scale 不能包含 0")

        self.feature_dim = int(self.mean.shape[0])
        if feature_columns is None:
            self.feature_columns = tuple(f"feature_{idx}" for idx in range(self.feature_dim))
        else:
            self.feature_columns = tuple(str(column) for column in feature_columns)
            if len(self.feature_columns) != self.feature_dim:
                raise ValueError(
                    "feature_columns 长度必须等于 scaler feature_dim："
                    f"columns={len(self.feature_columns)} feature_dim={self.feature_dim}"
                )
            if len(self.feature_columns) != len(set(self.feature_columns)):
                raise ValueError(f"feature_columns 不能重复：{self.feature_columns}")

    @classmethod
    def from_json(cls, path: Path) -> "LoadedFeatureScaler":
        """
        函数功能：
            从显式 JSON 路径读取 loaded scaler state。

        输入：
            path: Runtime 或 smoke 显式传入的 JSON state 路径。

        输出：
            LoadedFeatureScaler 实例。

        关键约束：
            这里只读取 scaler state JSON，不发现 run_dir，不读取 checkpoint。
        """
        state_path = Path(path)
        with state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("scaler state JSON 必须是 object")
        return cls(
            mean=payload.get("mean", ()),
            scale=payload.get("scale", ()),
            feature_columns=payload.get("feature_columns"),
            scaler_schema_version=str(payload.get("scaler_schema_version", cls.default_schema_version)),
            source_name=str(state_path),
        )

    def transform(self, feature_batch: FeatureBatch) -> FeatureBatch:
        """
        函数功能：
            将 raw/pre-head FeatureBatch transform 为 head-ready float32 FeatureBatch。

        输入：
            feature_batch: raw/pre-head visual features；features 必须是二维有限数值，
                feature_dim 必须与 loaded scaler state 对齐。

        输出：
            新 FeatureBatch；不修改输入 FeatureBatch 或其 metadata。
        """
        raw_features = self._validate_input_features(feature_batch)
        input_schema = dict(feature_batch.feature_schema)
        input_feature_columns = self._feature_columns_from_schema(input_schema, raw_features.shape[1])
        if input_feature_columns != self.feature_columns:
            raise ValueError(
                "FeatureBatch.feature_schema feature_columns 必须与 scaler state 对齐："
                f"input={input_feature_columns} scaler={self.feature_columns}"
            )

        # 中文注释：astype(copy=True) 保证 transform 不会复用或修改输入数组内存。
        raw_float64 = raw_features.astype(np.float64, copy=True)
        scaled = ((raw_float64 - self.mean) / self.scale).astype(np.float32, copy=False)
        if scaled.shape != raw_features.shape:
            raise ValueError(f"scaled features shape 漂移：actual={scaled.shape} expected={raw_features.shape}")
        if scaled.dtype != np.float32:
            raise ValueError(f"scaled features dtype 必须为 float32：actual={scaled.dtype}")
        if not np.all(np.isfinite(scaled)):
            raise ValueError("scaled features 包含 NaN 或 Inf")

        return FeatureBatch(
            sample_keys=tuple(str(sample_key) for sample_key in feature_batch.sample_keys),
            features=scaled,
            feature_schema={
                "transformed_by": self.scaler_name,
                "scaler_schema_version": self.scaler_schema_version,
                "feature_dim": self.feature_dim,
                "feature_columns": self.feature_columns,
                "head_ready": True,
                "handles_scaler": True,
                "input_schema": input_schema,
                "dtype": str(scaled.dtype),
            },
            extra={
                "scaler_name": self.scaler_name,
                "scaler_state_source": self.source_name,
                "fit_performed": False,
            },
        )

    def __call__(self, feature_batch: FeatureBatch) -> FeatureBatch:
        """函数功能：提供与 transformer 常见调用习惯一致的转发入口。"""
        return self.transform(feature_batch)

    def _validate_vector(self, values: Sequence[float], *, vector_name: str) -> np.ndarray:
        """函数功能：校验 scaler state 中的一维有限向量。"""
        vector = np.asarray(values, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError(f"{vector_name} 必须是一维数组：actual_shape={vector.shape}")
        if vector.size == 0:
            raise ValueError(f"{vector_name} 不能为空")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{vector_name} 必须全为有限数值")
        return vector

    def _validate_input_features(self, feature_batch: FeatureBatch) -> np.ndarray:
        """
        函数功能：
            校验输入 FeatureBatch 的 raw/pre-head features contract。
        """
        sample_keys = tuple(str(sample_key) for sample_key in feature_batch.sample_keys)
        if not sample_keys:
            raise ValueError("LoadedFeatureScaler.transform 需要非空 sample_keys")
        if len(sample_keys) != len(set(sample_keys)):
            raise ValueError(f"LoadedFeatureScaler.transform 收到重复 sample_key：{sample_keys}")

        features = np.asarray(feature_batch.features)
        if features.ndim != 2:
            raise ValueError(f"FeatureBatch.features 必须是二维矩阵：actual_shape={features.shape}")
        if features.shape[0] != len(sample_keys):
            raise ValueError(
                "FeatureBatch.features 样本维度必须等于 sample_keys 数量："
                f"features={features.shape[0]} sample_keys={len(sample_keys)}"
            )
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                "FeatureBatch.features 特征维度必须等于 scaler feature_dim："
                f"features={features.shape[1]} scaler={self.feature_dim}"
            )
        if not np.issubdtype(features.dtype, np.number):
            raise ValueError(f"FeatureBatch.features 必须是数值 dtype：actual={features.dtype}")
        if not np.all(np.isfinite(features)):
            raise ValueError("FeatureBatch.features 包含 NaN 或 Inf")
        return features

    def _feature_columns_from_schema(self, feature_schema: Mapping[str, Any], feature_dim: int) -> tuple[str, ...]:
        """
        函数功能：
            从输入 schema 读取 feature_columns；缺省时按 feature_dim 生成默认列名。
        """
        raw_columns = feature_schema.get("feature_columns")
        if raw_columns is None:
            return tuple(f"feature_{idx}" for idx in range(feature_dim))
        columns = tuple(str(column) for column in raw_columns)
        if len(columns) != feature_dim:
            raise ValueError(
                "FeatureBatch.feature_schema feature_columns 长度必须等于 feature_dim："
                f"columns={len(columns)} feature_dim={feature_dim}"
            )
        if len(columns) != len(set(columns)):
            raise ValueError(f"FeatureBatch.feature_schema feature_columns 不能重复：{columns}")
        return columns
