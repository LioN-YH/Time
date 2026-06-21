"""
文件功能：
    `time_router` 特征适配器入口。

说明：
    当前长期 public boundary 只暴露 canonical 迁移链路需要的 FeatureProvider /
    FeatureTransform contract：TimeFuseFeatureCacheProvider、VisualPrecomputedFeatureProvider、
    LoadedFeatureScaler，以及 P16f Visual feature chain protocol skeleton。
    VisualMockFeatureProvider 和 DeterministicVisualEncoderStub 仍作为 P14/P15
    兼容属性导入，避免破坏既有 smoke 和 Visual small 默认路径；但它们是
    smoke-only scaffold，不进入 __all__，后续 P18b/P18c 应迁到 tests/helpers
    或专门 legacy smoke helper。
"""

from time_router.features.timefuse_cache import TimeFuseFeatureCacheProvider
from time_router.features.visual_chain import (
    FeatureTransform,
    LineageMetadata,
    PoolingStrategy,
    PreImageBatch,
    PreImageTransform,
    PseudoImageTransformer,
    RawWindowBatch,
    RawWindowProvider,
    ResizePolicy,
    SampleKeys,
    VisualEmbeddingBatch,
    VisualEncoderProvider,
    VisualFeatureChainSpec,
    VisualInputBatch,
)
from time_router.features.visual_scaler import LoadedFeatureScaler
from time_router.features.visual_mock import DeterministicVisualEncoderStub, VisualMockFeatureProvider
from time_router.features.visual_precomputed import VisualPrecomputedFeatureProvider

__all__ = [
    "FeatureTransform",
    "LineageMetadata",
    "LoadedFeatureScaler",
    "PoolingStrategy",
    "PreImageBatch",
    "PreImageTransform",
    "PseudoImageTransformer",
    "RawWindowBatch",
    "RawWindowProvider",
    "ResizePolicy",
    "SampleKeys",
    "TimeFuseFeatureCacheProvider",
    "VisualEmbeddingBatch",
    "VisualEncoderProvider",
    "VisualFeatureChainSpec",
    "VisualInputBatch",
    "VisualPrecomputedFeatureProvider",
]
