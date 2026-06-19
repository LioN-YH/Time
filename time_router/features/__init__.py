"""
文件功能：
    `time_router` 特征适配器入口。

说明：
    当前只暴露 Stage 1 P7a 的最小 TimeFuseFeatureCacheProvider。该 provider
    只把显式 feature CSV 中的 TimeFuse 结构特征包装为 FeatureBatch，供
    smoke 验证 canonical FeatureProvider contract，不接正式训练入口。
"""

from time_router.features.timefuse_cache import TimeFuseFeatureCacheProvider

__all__ = ["TimeFuseFeatureCacheProvider"]
