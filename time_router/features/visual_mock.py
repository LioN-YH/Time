"""
文件功能：
    Stage 1 P18b 兼容包装。

说明：
    VisualMockFeatureProvider / DeterministicVisualEncoderStub 已迁到
    tests.helpers.visual_smoke_providers。该模块只保留旧子模块 import 的短期
    compatibility alias；不再由 `time_router.features` package 入口导入，也不进入
    public `__all__`。P18c 可删除该兼容文件。
"""

from tests.helpers.visual_smoke_providers import DeterministicVisualEncoderStub, VisualMockFeatureProvider

