"""
文件功能：
    `time_router` 最小模型/RouterHead 适配器入口。

说明：
    当前只暴露 Stage 1 P7b smoke-only 的 TimeFuseLinearSoftmaxHead。该 head
    只把 FeatureBatch.features 线性映射为 RouterOutput(logits, weights)，
    不接正式 TimeFuse fusor、Visual Router 入口或训练流程。
"""

from time_router.models.timefuse_linear import TimeFuseLinearSoftmaxHead

__all__ = ["TimeFuseLinearSoftmaxHead"]
