"""
文件功能：
    `time_router` 特征适配器入口。

说明：
    当前暴露 smoke-only FeatureProvider adapter：TimeFuseFeatureCacheProvider
    只把显式 feature CSV 中的 TimeFuse 结构特征包装为 FeatureBatch；
    VisualMockFeatureProvider 只用内存 history window 和 deterministic encoder
    stub 验证 Visual-style provider contract。二者都不接正式训练入口。
"""

from time_router.features.timefuse_cache import TimeFuseFeatureCacheProvider
from time_router.features.visual_mock import DeterministicVisualEncoderStub, VisualMockFeatureProvider

__all__ = ["DeterministicVisualEncoderStub", "TimeFuseFeatureCacheProvider", "VisualMockFeatureProvider"]
