"""
文件功能：
    `time_router` 特征适配器入口。

说明：
    当前长期 public boundary 只暴露 canonical 迁移链路需要的 FeatureProvider /
    FeatureTransform contract：TimeFuseFeatureCacheProvider、VisualPrecomputedFeatureProvider、
    LoadedFeatureScaler、P16f Visual feature chain protocol skeleton、P19a
    VisualFeatureChainRunner dry-run 编排骨架，以及 P19b guarded VisualVitEncoderProvider。
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
from time_router.features.visual_chain_runner import (
    CHAIN_LINEAGE,
    VisualFeatureChainResult,
    VisualFeatureChainRunner,
)
from time_router.features.visual_scaler import LoadedFeatureScaler
from time_router.features.visual_precomputed import VisualPrecomputedFeatureProvider
from time_router.features.visual_vit_encoder import VisualVitEncoderProvider, build_visual_vit_encoder_provider

__all__ = [
    "CHAIN_LINEAGE",
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
    "VisualFeatureChainResult",
    "VisualFeatureChainRunner",
    "VisualFeatureChainSpec",
    "VisualInputBatch",
    "VisualPrecomputedFeatureProvider",
    "VisualVitEncoderProvider",
    "build_visual_vit_encoder_provider",
]
