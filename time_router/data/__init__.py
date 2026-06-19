"""
文件功能：
    `time_router` 共享数据读取工具包。

说明：
    当前只导出 Stage 1 oracle/TSF 只读 reader；该 reader 只负责读取、
    校验和按 sample_key join，不承载训练策略或可部署特征逻辑。
"""

from time_router.data.oracle_tsf_reader import OracleTsfBatch, OracleTsfReader

__all__ = ["OracleTsfBatch", "OracleTsfReader"]
