#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P10f Visual labels CSV 到 canonical SampleManifest / SupervisionBatch
    的最小 smoke adapter。

设计约束：
    该模块只验证历史 labels 表能被拆解为 canonical sample/split/supervision
    协议对象；不接正式 Visual Router 入口，不读取 `/data2`，不写运行产物，
    不改变正式 labels CSV、summary、metadata、status 或 checkpoint schema。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from time_router.protocols import SampleManifest, SampleManifestRow, SupervisionBatch


ALLOWED_SPLITS = frozenset({"train", "vali", "test", "heldout"})
MANIFEST_REQUIRED_COLUMNS = (
    "sample_key",
    "split",
    "config_name",
    "dataset_name",
    "item_id",
    "channel_id",
    "window_index",
)
OPTIONAL_LENGTH_COLUMNS = ("seq_len", "pred_len")
DEFAULT_LINEAGE_COLUMNS = ("source_label_path", "label_schema_version", "manifest_shard")


def load_visual_labels_frame(labels: pd.DataFrame | str | Path) -> pd.DataFrame:
    """
    函数功能：
        读取小型 Visual labels fixture，统一返回 DataFrame 副本。

    输入：
        labels 可以是已在内存中的 `pd.DataFrame`，也可以是 CSV 路径。

    输出：
        DataFrame 副本；调用方后续校验字段、split 和 sample_key。

    关键约束：
        只支持 smoke / 小型 fixture。正式 full-scale CSV schema 对齐前，不在这里
        推断历史入口私有字段或接管 streaming 训练入口。
    """
    if isinstance(labels, pd.DataFrame):
        return labels.copy()
    path = Path(labels)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"P10f adapter 目前只支持 CSV 或 DataFrame 输入：{path}")
    return pd.read_csv(path)


def visual_labels_to_sample_manifest(
    labels: pd.DataFrame | str | Path,
    *,
    allowed_splits: Sequence[str] = tuple(sorted(ALLOWED_SPLITS)),
    lineage_columns: Sequence[str] = DEFAULT_LINEAGE_COLUMNS,
) -> SampleManifest:
    """
    函数功能：
        从 Visual labels DataFrame/CSV 构造 canonical `SampleManifest`。

    输入：
        labels: 小型 labels fixture 或 CSV 路径。
        allowed_splits: 允许出现的 split 名称。
        lineage_columns: 允许进入 `SampleManifestRow.extra` 的轻量 lineage 列。

    输出：
        `SampleManifest`，rows 顺序与 labels 原始行顺序一致。

    关键约束：
        `extra` 只保存 lineage，不保存 oracle、专家 error 或未来信息，避免把监督
        信息误传给 deployable FeatureProvider。
    """
    frame = load_visual_labels_frame(labels)
    _require_columns(frame, MANIFEST_REQUIRED_COLUMNS, source="visual labels manifest")
    frame = frame.copy()
    frame["sample_key"] = frame["sample_key"].astype(str)
    frame["split"] = frame["split"].astype(str)
    _validate_unique_sample_key_column(frame["sample_key"].tolist())
    _validate_allowed_splits(frame["split"].tolist(), allowed_splits)

    rows: list[SampleManifestRow] = []
    for _, row in frame.iterrows():
        rows.append(
            SampleManifestRow(
                sample_key=str(row["sample_key"]),
                split=str(row["split"]),
                config_name=str(row["config_name"]),
                dataset_name=str(row["dataset_name"]),
                item_id=_to_int(row["item_id"], "item_id"),
                channel_id=_to_int(row["channel_id"], "channel_id"),
                window_index=_to_int(row["window_index"], "window_index"),
                seq_len=_optional_int(row, "seq_len"),
                pred_len=_optional_int(row, "pred_len"),
                extra=_lineage_extra(row, lineage_columns),
            )
        )

    manifest = SampleManifest(
        rows=tuple(rows),
        extra={
            "source": "visual_labels_adapter",
            "adapter_scope": "p10f_smoke_only_not_formal_entrypoint",
        },
    )
    manifest.validate_unique_sample_keys()
    return manifest


def visual_labels_to_supervision_batch(
    labels: pd.DataFrame | str | Path,
    *,
    sample_keys: Sequence[str],
    model_columns: Sequence[str],
    metric: str,
) -> SupervisionBatch:
    """
    函数功能：
        从同一 Visual labels DataFrame/CSV 构造 canonical `SupervisionBatch`。

    输入：
        sample_keys: 显式目标样本顺序；输出必须保持该顺序。
        model_columns: 显式专家顺序；每个专家要求存在对应 metric error 列。
        metric: 监督指标，例如 `mae` 或 `mse`。

    输出：
        `SupervisionBatch`，包含 oracle top-1 专家名、oracle metric 值和
        `[sample, expert]` per-model error 矩阵。

    关键约束：
        oracle/error 只进入 supervision batch，不进入 SampleManifest 或 FeatureProvider。
    """
    ordered_keys = _normalize_unique_strings(sample_keys, field_name="sample_keys")
    ordered_models = _normalize_unique_strings(model_columns, field_name="model_columns")
    if not str(metric):
        raise ValueError("metric 不能为空")

    frame = load_visual_labels_frame(labels)
    _require_columns(frame, ("sample_key",), source="visual labels supervision")
    error_columns = tuple(_error_column_name(model_name, metric) for model_name in ordered_models)
    _require_columns(frame, error_columns, source="visual labels supervision")
    frame = frame.copy()
    frame["sample_key"] = frame["sample_key"].astype(str)
    _validate_unique_sample_key_column(frame["sample_key"].tolist())

    indexed = frame.set_index("sample_key", drop=False)
    missing_keys = [sample_key for sample_key in ordered_keys if sample_key not in indexed.index]
    if missing_keys:
        raise ValueError(f"labels 缺少请求的 sample_key：{missing_keys}")

    ordered_frame = indexed.loc[list(ordered_keys)]
    per_model_errors = ordered_frame.loc[:, list(error_columns)].to_numpy(dtype=np.float32)
    oracle_indices = per_model_errors.argmin(axis=1)
    oracle_model = np.array([ordered_models[index] for index in oracle_indices], dtype=object)
    oracle_value = per_model_errors.min(axis=1)

    batch = SupervisionBatch(
        sample_keys=tuple(ordered_keys),
        model_columns=tuple(ordered_models),
        metric=str(metric),
        oracle_model=oracle_model,
        oracle_value=oracle_value,
        per_model_errors=per_model_errors,
        extra={
            "source": "visual_labels_adapter",
            "error_column_pattern": "{model_name}_{metric}_error",
            "adapter_scope": "p10f_smoke_only_not_formal_entrypoint",
        },
    )
    batch.validate_shapes()
    return batch


def _error_column_name(model_name: str, metric: str) -> str:
    """函数功能：集中定义 smoke fixture 的专家误差列命名约定。"""
    return f"{model_name}_{metric}_error"


def _require_columns(frame: pd.DataFrame, required_columns: Sequence[str], *, source: str) -> None:
    """函数功能：检查 DataFrame 必需列，缺失时给出来源和列名。"""
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} 缺少必需列：{missing}")


def _validate_unique_sample_key_column(sample_keys: Sequence[str]) -> None:
    """函数功能：检查 labels 源表 sample_key 唯一，避免 manifest/supervision 错位。"""
    seen: set[str] = set()
    duplicates: list[str] = []
    for sample_key in sample_keys:
        if sample_key in seen and sample_key not in duplicates:
            duplicates.append(sample_key)
        seen.add(sample_key)
    if duplicates:
        raise ValueError(f"labels sample_key 必须唯一，重复值：{duplicates}")


def _validate_allowed_splits(splits: Sequence[str], allowed_splits: Sequence[str]) -> None:
    """函数功能：限制 split 取值，防止历史 labels 中拼写错误静默进入 manifest。"""
    allowed = {str(split) for split in allowed_splits}
    unknown = sorted({split for split in splits if split not in allowed})
    if unknown:
        raise ValueError(f"labels 出现未知 split：{unknown}；允许值：{sorted(allowed)}")


def _normalize_unique_strings(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    """函数功能：规范化显式顺序字段，并拒绝空列表和重复值。"""
    normalized = tuple(str(value) for value in values)
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    duplicates = sorted({value for value in normalized if normalized.count(value) > 1})
    if duplicates:
        raise ValueError(f"{field_name} 存在重复值：{duplicates}")
    return normalized


def _to_int(value: object, field_name: str) -> int:
    """函数功能：将 manifest 身份字段转为 int，并在失败时保留字段名。"""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须可转换为整数，实际值：{value!r}") from exc


def _optional_int(row: pd.Series, field_name: str) -> int | None:
    """函数功能：读取可选 seq_len/pred_len，缺失或 NaN 时返回 None。"""
    if field_name not in row.index or pd.isna(row[field_name]):
        return None
    return _to_int(row[field_name], field_name)


def _lineage_extra(row: pd.Series, lineage_columns: Sequence[str]) -> dict[str, object]:
    """
    函数功能：
        只提取白名单 lineage 字段，避免 oracle/error 字段进入 manifest extra。
    """
    extra: dict[str, object] = {}
    for column in lineage_columns:
        if column in row.index and not pd.isna(row[column]):
            value = row[column]
            if hasattr(value, "item"):
                value = value.item()
            extra[str(column)] = value
    return extra
