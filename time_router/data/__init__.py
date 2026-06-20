"""
文件功能：
    `time_router` 共享数据读取与小型 schema adapter 工具包。

说明：
    当前导出 Stage 1 oracle/TSF 只读 reader，以及 P10f Visual labels 到
    SampleManifest / SupervisionBatch 的 smoke adapter。二者都只负责读取、
    校验和轻量协议对象转换，不承载训练策略或可部署特征逻辑。
"""

from time_router.data.oracle_tsf_reader import OracleTsfBatch, OracleTsfReader
from time_router.data.visual_labels_adapter import (
    visual_labels_to_sample_manifest,
    visual_labels_to_supervision_batch,
)

__all__ = [
    "OracleTsfBatch",
    "OracleTsfReader",
    "visual_labels_to_sample_manifest",
    "visual_labels_to_supervision_batch",
]
