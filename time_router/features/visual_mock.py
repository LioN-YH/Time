"""
文件功能：
    提供 Stage 1 P14b smoke-only VisualMockFeatureProvider。

设计边界：
    该 provider 只消费调用方显式传入的 ordered sample_keys 和内存中的
    history window source，并通过 deterministic encoder stub 输出 FeatureBatch。
    它不读取 prediction cache、oracle/error、y_true、run_dir、status、checkpoint
    或 `/data2`，也不加载真实 Hugging Face ViT、不使用 GPU/DataParallel。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from time_router.protocols import FeatureBatch


class DeterministicVisualEncoderStub:
    """
    类功能：
        把小型 history window array 转成固定 8 维 float32 embedding。

    输入：
        history_window: 一维历史窗口，只代表过去 x，不包含 future y。

    输出：
        8 维 numpy.float32 向量，维度依次为 mean、std、min、max、first、
        last、range 和 mean_square。

    关键约束：
        这是 P14b smoke 的 deterministic stub，不加载真实 ViT，不访问 GPU，
        不依赖 Hugging Face cache；未来正式 encoder 应另由 Runtime/encoder
        factory 管理。
    """

    encoder_name = "deterministic_visual_history_stats_stub_v1"
    feature_dim = 8

    def encode_one(self, history_window: Sequence[float] | np.ndarray) -> np.ndarray:
        """函数功能：将单个 history window 编码为固定 8 维 float32 特征。"""
        window = np.asarray(history_window, dtype=np.float32)
        if window.ndim != 1:
            raise ValueError(f"history window 必须是一维数组：actual_shape={window.shape}")
        if window.size == 0:
            raise ValueError("history window 不能为空")
        if not np.all(np.isfinite(window)):
            raise ValueError("history window 包含 NaN 或 Inf")

        feature_values = np.asarray(
            [
                float(np.mean(window)),
                float(np.std(window)),
                float(np.min(window)),
                float(np.max(window)),
                float(window[0]),
                float(window[-1]),
                float(np.max(window) - np.min(window)),
                float(np.mean(np.square(window))),
            ],
            dtype=np.float32,
        )
        return feature_values

    def encode_batch(self, history_windows: Sequence[Sequence[float] | np.ndarray]) -> np.ndarray:
        """
        函数功能：
            将一组 history windows 编码为 `[sample, 8]` float32 特征矩阵。
        """
        if not history_windows:
            raise ValueError("encoder stub 需要至少一个 history window")
        return np.stack([self.encode_one(window) for window in history_windows], axis=0).astype(np.float32)


class VisualMockFeatureProvider:
    """
    类功能：
        P14b 最小 Visual-style FeatureProvider mock。

    输入：
        history_windows: `sample_key -> history window` 的内存映射。
        encoder: deterministic encoder stub；默认使用
            DeterministicVisualEncoderStub。
        history_source_name: schema 中记录的轻量 history source 口径。

    输出：
        `load_batch(sample_keys)` 返回 FeatureBatch，sample_keys 与调用方传入
        顺序完全一致，features 为 numpy.float32 `[sample, 8]`。

    关键约束：
        provider 不拥有 manifest、oracle、prediction backend 或 run_dir；它只按
        调用方显式 sample_keys 从内存 history source 取过去窗口并编码。
    """

    provider_name = "VisualMockFeatureProvider"

    def __init__(
        self,
        *,
        history_windows: Mapping[str, Sequence[float] | np.ndarray],
        encoder: DeterministicVisualEncoderStub | None = None,
        history_source_name: str = "in_memory_history_window_fixture",
        feature_schema_name: str = "visual_mock_history_encoder_v1",
        source: str = "tests_fixture_in_memory",
    ) -> None:
        if not history_windows:
            raise ValueError("VisualMockFeatureProvider 需要非空 history_windows")
        self.history_windows = {str(sample_key): value for sample_key, value in history_windows.items()}
        if len(self.history_windows) != len(history_windows):
            raise ValueError("history_windows 存在重复或不可区分的 sample_key")
        self.encoder = encoder or DeterministicVisualEncoderStub()
        self.history_source_name = str(history_source_name)
        self.feature_schema_name = str(feature_schema_name)
        self.source = str(source)

    def load_batch(self, sample_keys: Sequence[str]) -> FeatureBatch:
        """
        函数功能：
            按调用方传入顺序读取内存 history windows，编码后包装为 FeatureBatch。

        输入：
            sample_keys: manifest 或 SplitStrategy 已确定的 ordered sample_keys；
                不能为空且不能重复。

        输出：
            FeatureBatch；`sample_keys` 保序，`features` dtype 固定为 float32。
        """
        ordered_keys = tuple(str(sample_key) for sample_key in sample_keys)
        if not ordered_keys:
            raise ValueError("VisualMockFeatureProvider.load_batch 必须显式传入非空 sample_keys")
        if len(ordered_keys) != len(set(ordered_keys)):
            raise ValueError("VisualMockFeatureProvider.load_batch 收到重复 sample_key")

        missing_keys = [sample_key for sample_key in ordered_keys if sample_key not in self.history_windows]
        if missing_keys:
            raise KeyError(f"history_windows 缺少 sample_key：{missing_keys}")

        ordered_windows = [self.history_windows[sample_key] for sample_key in ordered_keys]
        features = self.encoder.encode_batch(ordered_windows).astype(np.float32, copy=False)
        if features.ndim != 2 or features.shape[1] != self.encoder.feature_dim:
            raise ValueError(f"encoder stub 输出 shape 漂移：actual={features.shape}")

        return FeatureBatch(
            sample_keys=ordered_keys,
            features=features,
            feature_schema={
                "feature_schema_name": self.feature_schema_name,
                "feature_dim": int(features.shape[1]),
                "history_source": self.history_source_name,
                "pseudo_image": {
                    "variant": "mock_not_materialized",
                    "storage": "not_saved",
                    "input": "history_window_x_only",
                },
                "encoder_stub": {
                    "name": self.encoder.encoder_name,
                    "feature_dim": self.encoder.feature_dim,
                    "loads_real_vit": False,
                    "uses_gpu": False,
                    "uses_huggingface_cache": False,
                },
                "storage": "batch_runtime_only_not_saved",
                "dtype": str(features.dtype),
            },
            extra={
                "provider_name": self.provider_name,
                "source": self.source,
                "num_available_rows": len(self.history_windows),
            },
        )

