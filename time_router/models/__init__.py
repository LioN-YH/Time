"""
文件功能：
    `time_router` 最小模型/RouterHead 适配器入口。

说明：
    当前暴露 Stage 1 canonical RouterHead 边界所需的轻量模型适配器。
    TimeFuseLinearSoftmaxHead 服务 TimeFuse smoke；LoadedTorchMLPRouterHeadAdapter
    服务 P16a Visual 已加载 torch module + head-ready FeatureBatch 边界。
"""

from time_router.models.timefuse_linear import TimeFuseLinearSoftmaxHead
from time_router.models.visual_mlp_adapter import LoadedTorchMLPRouterHeadAdapter

__all__ = ["LoadedTorchMLPRouterHeadAdapter", "TimeFuseLinearSoftmaxHead"]
