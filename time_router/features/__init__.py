"""
文件功能：
    `time_router` 特征适配器入口。

说明：
    当前长期 public boundary 只暴露 canonical 迁移链路需要的 FeatureProvider /
    FeatureTransform contract：TimeFuseFeatureCacheProvider、VisualPrecomputedFeatureProvider、
    LoadedFeatureScaler，以及 P16f Visual feature chain protocol skeleton。
    VisualMockFeatureProvider 和 DeterministicVisualEncoderStub 已在 P18b 迁到
    tests.helpers.visual_smoke_providers，不再从 feature package 入口导入。
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
