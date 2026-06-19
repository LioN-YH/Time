"""
文件功能：
    `time_router` 专家预测适配器入口。

说明：
    当前只暴露 Stage 1 P6a 的最小 PredictionCacheExpertProvider。该 provider
    只把 PredictionBatchReader 的输出包装为 ExpertBatch，供 smoke 验证
    canonical ExpertProvider contract，不接正式训练入口。
"""

from time_router.experts.prediction_cache import PredictionCacheExpertProvider

__all__ = ["PredictionCacheExpertProvider"]
