"""
文件功能：
    提供 Stage 1 P6a 最小 PredictionCacheExpertProvider。

设计边界：
    该 provider 只复用 PredictionBatchReader，把显式 sample_key batch 包装为
    ExpertBatch。它不读取 oracle/TSF，不生成 router feature，不计算 loss，
    不做 evaluation，不创建 run_dir，也不写 status/metadata。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from time_router.io import DEFAULT_MODEL_COLUMNS, PredictionBatchReader
from time_router.protocols import ExpertBatch


class PredictionCacheExpertProvider:
    """
    类功能：
        将 prediction cache reader 输出适配为 canonical ExpertBatch。

    输入：
        manifest_path 或 fixture_root 用于初始化底层 PredictionBatchReader；
        model_columns 控制专家动作空间顺序，默认沿用 Stage 1 固定五专家顺序。

    输出：
        `load_batch(...)` 返回 `ExpertBatch`，其中 sample_keys/model_columns 为 tuple，
        y_pred/y_true 原样来自 reader，row_index_metadata 保留 packed row index lineage。

    关键约束：
        调用方必须显式传入 sample_keys。provider 不默认扫描全量 manifest，
        不决定 run_dir，不接触 /data2，不承担训练或评估职责。
    """

    provider_name = "PredictionCacheExpertProvider"

    def __init__(
        self,
        *,
        manifest_path: Optional[Path] = None,
        fixture_root: Optional[Path] = None,
        model_columns: Optional[Sequence[str]] = None,
        chunk_rows: int = 200_000,
        validate_manifest_schema: bool = True,
    ) -> None:
        self.reader = PredictionBatchReader(
            manifest_path=manifest_path,
            fixture_root=fixture_root,
            model_columns=model_columns,
            chunk_rows=chunk_rows,
            validate_manifest_schema=validate_manifest_schema,
        )
        self.model_columns = tuple(str(model_name) for model_name in (model_columns or DEFAULT_MODEL_COLUMNS))
        self.chunk_rows = int(chunk_rows)
        self.validate_manifest_schema = bool(validate_manifest_schema)

    def load_batch(self, sample_keys: Sequence[str], *, verify_metrics: bool = True) -> ExpertBatch:
        """
        函数功能：
            显式读取一个 sample_key batch，并包装为 ExpertBatch。

        输入：
            sample_keys: 调用方指定的 sample_key 顺序；不能为空且不能重复。
            verify_metrics: 是否沿用 PredictionBatchReader 的 manifest MAE/MSE 复算校验。

        输出：
            ExpertBatch，sample_keys 和 model_columns 与数组维度严格对齐。
        """
        ordered_keys = tuple(str(sample_key) for sample_key in sample_keys)
        if not ordered_keys:
            raise ValueError("PredictionCacheExpertProvider.load_batch 必须显式传入非空 sample_keys")
        if len(ordered_keys) != len(set(ordered_keys)):
            raise ValueError("PredictionCacheExpertProvider.load_batch 收到重复 sample_key")

        batch = self.reader.load(ordered_keys, verify_metrics=verify_metrics)
        row_index_metadata = batch.metadata.get("row_indices_by_sample_model")
        manifest_rows = batch.metadata.get("manifest_rows")
        array_storage = _extract_array_storage(manifest_rows)

        return ExpertBatch(
            sample_keys=tuple(batch.sample_keys),
            model_columns=tuple(self.reader.model_columns),
            y_pred=batch.y_pred,
            y_true=batch.y_true,
            row_index_metadata=row_index_metadata,
            extra={
                "provider_name": self.provider_name,
                "array_storage": array_storage,
                "reader_metadata": {
                    "manifest_path": batch.metadata.get("manifest_path"),
                    "manifest_row_count": _safe_len(manifest_rows),
                    "manifest_model_order_by_sample": batch.metadata.get("manifest_model_order_by_sample"),
                    "verify_metrics": bool(verify_metrics),
                    "chunk_rows": self.chunk_rows,
                    "validate_manifest_schema": self.validate_manifest_schema,
                },
            },
        )


def _safe_len(value: Any) -> int | None:
    """函数功能：读取轻量行数元数据，避免 provider 依赖 pandas 类型。"""
    if value is None:
        return None
    try:
        return int(len(value))
    except TypeError:
        return None


def _extract_array_storage(manifest_rows: Any) -> str | tuple[str, ...] | None:
    """
    函数功能：
        从 reader 的 manifest_rows 轻量提取 array_storage。

    关键约束：
        这里不解析 manifest schema，也不读取数组文件；只把 reader 已经返回的
        当前 batch metadata 缩减为适合 ExpertBatch.extra 的 lineage 信息。
    """
    if manifest_rows is None or not hasattr(manifest_rows, "__getitem__"):
        return None
    try:
        values = manifest_rows["array_storage"]
    except Exception:
        return None
    unique_values = tuple(str(value) for value in sorted(set(values.astype(str).tolist())))
    if len(unique_values) == 1:
        return unique_values[0]
    return unique_values
