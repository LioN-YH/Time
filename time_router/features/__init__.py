"""
文件功能：
    `time_router` 特征适配器入口。

说明：
    当前暴露 smoke-only / 边界验证用 FeatureProvider adapter：TimeFuseFeatureCacheProvider
    只把显式 feature CSV 中的 TimeFuse 结构特征包装为 FeatureBatch；
    VisualMockFeatureProvider 只用内存 history window 和 deterministic encoder
    stub 验证 Visual-style provider contract；VisualPrecomputedFeatureProvider
    读取已预计算的 head-ready visual embedding fixture；LoadedFeatureScaler
    只使用已加载 scaler state 做 raw/pre-head -> head-ready transform。
    visual_chain 只暴露 P16f protocol skeleton，不实现真实 ViT 链路。它们都
    不接正式训练入口。
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
    "DeterministicVisualEncoderStub",
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
    "VisualMockFeatureProvider",
    "VisualPrecomputedFeatureProvider",
]
