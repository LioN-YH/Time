"""
文件功能：
    `time_router` 共享数据读取与小型 schema adapter 工具包。

说明：
    当前导出 Stage 1 oracle/TSF 只读 reader，以及 P10f Visual labels / P10g
    TimeFuse feature-oracle 到 SampleManifest / SupervisionBatch 的 smoke adapter。
    这些接口都只负责读取、校验和轻量协议对象转换，不承载训练策略或可部署
    特征逻辑。
"""

from time_router.data.oracle_tsf_reader import OracleTsfBatch, OracleTsfReader
from time_router.data.timefuse_supervision_adapter import (
    timefuse_features_to_sample_manifest,
    timefuse_oracle_to_supervision_batch,
)
from time_router.data.visual_labels_adapter import (
    visual_labels_to_sample_manifest,
    visual_labels_to_supervision_batch,
)

__all__ = [
    "OracleTsfBatch",
    "OracleTsfReader",
    "timefuse_features_to_sample_manifest",
    "timefuse_oracle_to_supervision_batch",
    "visual_labels_to_sample_manifest",
    "visual_labels_to_supervision_batch",
]
