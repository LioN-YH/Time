#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P10g TimeFuse feature/oracle 表到 canonical SampleManifest /
    SupervisionBatch 的最小 smoke adapter。

设计约束：
    本模块只验证 TimeFuse-style fusor 历史 feature source 与 oracle/supervision
    source 可拆解为 canonical sample/split/supervision 协议对象；不接正式
    TimeFuse 入口，不读取 `/data2`，不改变正式 feature/oracle/prediction/runtime
    artifact schema。
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
DEFAULT_LINEAGE_COLUMNS = ("feature_shard", "feature_schema_version")


def load_timefuse_frame(source: pd.DataFrame | str | Path, *, source_name: str) -> pd.DataFrame:
    """
    函数功能：
        读取小型 TimeFuse feature/oracle fixture，统一返回 DataFrame 副本。

    输入：
        source 可以是内存中的 `pd.DataFrame`，也可以是 CSV 路径。
        source_name 用于错误信息中区分 feature source 与 oracle source。

    输出：
        DataFrame 副本；调用方随后按 manifest 或 supervision 语义做字段校验。

    关键约束：
        只支持 smoke / 小型 fixture，不扫描正式 shard、不读取 oracle SQLite/parquet，
        也不推断 full-scale 私有 schema。
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"{source_name} 目前只支持 CSV 或 DataFrame 输入：{path}")
    return pd.read_csv(path)


def timefuse_features_to_sample_manifest(
    features: pd.DataFrame | str | Path,
    *,
    allowed_splits: Sequence[str] = tuple(sorted(ALLOWED_SPLITS)),
    lineage_columns: Sequence[str] = DEFAULT_LINEAGE_COLUMNS,
) -> SampleManifest:
    """
    函数功能：
        从小型 TimeFuse feature DataFrame/CSV 构造 canonical `SampleManifest`。

    输入：
        features: 小型 feature fixture 或 CSV 路径。
        allowed_splits: 允许出现的 split 名称。
        lineage_columns: 允许进入 `SampleManifestRow.extra` 的轻量 lineage 列。

    输出：
        `SampleManifest`，rows 顺序与 feature source 原始行顺序一致。

    关键约束：
        TimeFuse 17 维 feature 值属于未来 `FeatureProvider`，不得进入 manifest
        extra；这里仅保存 feature shard / schema version 等轻量 lineage。
    """
    frame = load_timefuse_frame(features, source_name="TimeFuse feature source")
    _require_columns(frame, MANIFEST_REQUIRED_COLUMNS, source="TimeFuse feature manifest")
    frame = frame.copy()
    frame["sample_key"] = frame["sample_key"].astype(str)
    frame["split"] = frame["split"].astype(str)
    _validate_unique_sample_key_column(frame["sample_key"].tolist(), source="TimeFuse feature")
    _validate_allowed_splits(frame["split"].tolist(), allowed_splits, source="TimeFuse feature")

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
            "source": "timefuse_supervision_adapter",
            "adapter_scope": "p10g_smoke_only_not_formal_entrypoint",
            "feature_values_in_manifest": False,
        },
    )
    manifest.validate_unique_sample_keys()
    return manifest


def timefuse_oracle_to_supervision_batch(
    oracle: pd.DataFrame | str | Path,
    *,
    sample_keys: Sequence[str],
    model_columns: Sequence[str],
    metric: str,
) -> SupervisionBatch:
    """
    函数功能：
        从小型 TimeFuse oracle/supervision DataFrame/CSV 构造 `SupervisionBatch`。

    输入：
        oracle: 小型 oracle fixture 或 CSV 路径。
        sample_keys: 显式目标样本顺序；输出必须保持该顺序。
        model_columns: 显式专家顺序；每个专家要求存在对应 metric error 列。
        metric: 监督指标，例如 `mae` 或 `mse`。

    输出：
        `SupervisionBatch`，包含 oracle top-1 专家名、oracle metric 值和
        `[sample, expert]` per-model error 矩阵。

    关键约束：
        supervision 只来自 oracle/error source，不读取 feature source，不把 17 维
        feature 当作 supervision，也不读取 prediction cache。
    """
    ordered_keys = _normalize_unique_strings(sample_keys, field_name="sample_keys")
    ordered_models = _normalize_unique_strings(model_columns, field_name="model_columns")
    if not str(metric):
        raise ValueError("metric 不能为空")

    frame = load_timefuse_frame(oracle, source_name="TimeFuse oracle source")
    _require_columns(frame, ("sample_key",), source="TimeFuse oracle supervision")
    error_columns = tuple(_error_column_name(model_name, metric) for model_name in ordered_models)
    _require_columns(frame, error_columns, source="TimeFuse oracle supervision")
    frame = frame.copy()
    frame["sample_key"] = frame["sample_key"].astype(str)
    _validate_unique_sample_key_column(frame["sample_key"].tolist(), source="TimeFuse oracle")

    indexed = frame.set_index("sample_key", drop=False)
    missing_keys = [sample_key for sample_key in ordered_keys if sample_key not in indexed.index]
    if missing_keys:
        raise ValueError(f"TimeFuse oracle 缺少请求的 sample_key：{missing_keys}")

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
            "source": "timefuse_supervision_adapter",
            "error_column_pattern": "{model_name}_{metric}_error",
            "adapter_scope": "p10g_smoke_only_not_formal_entrypoint",
        },
    )
    batch.validate_shapes()
    return batch


def _error_column_name(model_name: str, metric: str) -> str:
    """函数功能：集中定义 P10g smoke fixture 的专家误差列命名约定。"""
    return f"{model_name}_{metric}_error"


def _require_columns(frame: pd.DataFrame, required_columns: Sequence[str], *, source: str) -> None:
    """函数功能：检查 DataFrame 必需列，缺失时给出来源和列名。"""
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} 缺少必需列：{missing}")


def _validate_unique_sample_key_column(sample_keys: Sequence[str], *, source: str) -> None:
    """函数功能：检查源表 sample_key 唯一，避免 manifest/supervision 错位。"""
    seen: set[str] = set()
    duplicates: list[str] = []
    for sample_key in sample_keys:
        if sample_key in seen and sample_key not in duplicates:
            duplicates.append(sample_key)
        seen.add(sample_key)
    if duplicates:
        raise ValueError(f"{source} sample_key 必须唯一，重复值：{duplicates}")


def _validate_allowed_splits(splits: Sequence[str], allowed_splits: Sequence[str], *, source: str) -> None:
    """函数功能：限制 split 取值，防止历史 feature 表拼写错误静默进入 manifest。"""
    allowed = {str(split) for split in allowed_splits}
    unknown = sorted({split for split in splits if split not in allowed})
    if unknown:
        raise ValueError(f"{source} 出现未知 split：{unknown}；允许值：{sorted(allowed)}")


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
        只提取白名单 lineage 字段，避免 TimeFuse 17 维 feature 值进入 manifest extra。
    """
    extra: dict[str, object] = {}
    for column in lineage_columns:
        if column in row.index and not pd.isna(row[column]):
            value = row[column]
            if hasattr(value, "item"):
                value = value.item()
            extra[str(column)] = value
    return extra
