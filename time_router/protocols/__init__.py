"""
文件功能：
    Stage 1 protocol lightweight contract container 的 public API 入口。

边界说明：
    本包只导出 P5c dataclass 类型骨架，用于描述 provider/protocol 之间
    传递的轻量 contract object。这里不实例化 provider，不读取配置或路径，
    不创建 run_dir，也不执行任何训练、评估或文件 IO。
"""

from time_router.protocols.types import (
    EvaluationInput,
    ExpertBatch,
    ExperimentProtocolSpec,
    FeatureBatch,
    RouterOutput,
    SampleManifest,
    SampleManifestRow,
    SplitSpec,
    SupervisionBatch,
)

__all__ = [
    "EvaluationInput",
    "ExpertBatch",
    "ExperimentProtocolSpec",
    "FeatureBatch",
    "RouterOutput",
    "SampleManifest",
    "SampleManifestRow",
    "SplitSpec",
    "SupervisionBatch",
]
