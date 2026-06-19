#!/usr/bin/env python3
"""
文件功能：
    提供 Stage 1 共享 OracleTsfReader，按 sample_key 批量读取 window-level
    oracle label 与 TSF enrichment / TSF-cell 元信息。

设计约束：
    - 只做读取、字段校验、缺失统计和 join，不实现任何训练策略；
    - oracle label 可用于监督、上限或诊断，但不得进入可部署 FeatureProvider；
    - TSF enrichment 可用于统计 baseline、分层汇总或诊断，不和未来信息泄漏混用；
    - 显式传入 sample_key 时必须保持输入顺序；
    - full-scale 场景应显式传入当前 batch/shard 的 sample_key。未显式传入 key
      的全扫描模式只适合小规模 fixture 或 smoke。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_ORACLE_COLUMNS = ["sample_key", "metric", "oracle_model", "oracle_value"]
DEFAULT_TSF_COLUMNS = [
    "sample_key",
    "cluster",
    "group_name",
    "forecastability_cat",
    "season_strength_cat",
    "trend_strength_cat",
    "cv_cat",
    "missing_ratio_cat",
]
STABLE_JOIN_COLUMNS = ["config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]
VALID_MISSING_POLICIES = {"error", "report"}


@dataclass(frozen=True)
class OracleTsfBatch:
    """
    类功能：
        描述 oracle、TSF 或二者 join 后的只读 batch。

    字段说明：
        sample_keys: 与输入 sample_key 顺序一致；若未显式输入，则为源文件首次出现顺序。
        frame: 输出 DataFrame，第一列为 sample_key，行顺序与 sample_keys 对齐。
        missing_report: 记录 oracle、TSF 和 joined 阶段缺失或重复的 sample_key。
        metadata: 记录源路径、metric、列来源和读取策略，便于实验复核。
    """

    sample_keys: List[str]
    frame: pd.DataFrame
    missing_report: Dict[str, object]
    metadata: Dict[str, object]


class OracleTsfReader:
    """
    类功能：
        从既有 oracle labels 与 TSF enrichment 文件批量读取 sample_key 级元信息。

    关键约束：
        该 reader 不改变 oracle/TSF 的生成逻辑，也不迁移正式训练入口。CSV 输入
        在显式 sample_keys 场景下按 chunk 过滤；Parquet 输入依赖 pyarrow dataset
        过滤。未显式 sample_keys 的读取只用于小规模 smoke，full-scale 后续应优先
        使用 SQLite / shard-local / batch query。
    """

    def __init__(
        self,
        *,
        oracle_path: Optional[Path] = None,
        tsf_path: Optional[Path] = None,
        fixture_root: Optional[Path] = None,
        missing_policy: str = "error",
        chunk_rows: int = 200_000,
        allow_full_scan: bool = False,
    ) -> None:
        if missing_policy not in VALID_MISSING_POLICIES:
            raise ValueError(f"missing_policy 必须为 {sorted(VALID_MISSING_POLICIES)}")
        if fixture_root is not None:
            root = Path(fixture_root)
            oracle_path = oracle_path or _first_existing(
                root / "window_oracle_labels_with_tsf_cell.csv",
                root / "window_oracle_labels.csv",
                root / "window_oracle_labels.parquet",
            )
            tsf_path = tsf_path or _first_existing(
                root / "sample_tsf_enrichment.parquet",
                root / "manifest_with_tsf_cell.csv",
                root / "window_oracle_labels_with_tsf_cell.csv",
            )
        self.oracle_path = Path(oracle_path) if oracle_path is not None else None
        self.tsf_path = Path(tsf_path) if tsf_path is not None else None
        self.missing_policy = missing_policy
        self.chunk_rows = int(chunk_rows)
        self.allow_full_scan = bool(allow_full_scan)
        if self.chunk_rows <= 0:
            raise ValueError("chunk_rows 必须为正整数")
        if self.oracle_path is not None and not self.oracle_path.exists():
            raise FileNotFoundError(f"找不到 oracle labels 文件：{self.oracle_path}")
        if self.tsf_path is not None and not self.tsf_path.exists():
            raise FileNotFoundError(f"找不到 TSF enrichment 文件：{self.tsf_path}")

    def load_oracle(self, sample_keys: Optional[Sequence[str]], metric: str = "mae") -> OracleTsfBatch:
        """
        函数功能：
            按 sample_key + metric 读取 oracle label。

        输入：
            sample_keys: 目标 sample_key 列表；显式传入时输出保持该顺序。
            metric: oracle 指标，目前沿用既有 `mae` / `mse` 字段值。

        输出：
            OracleTsfBatch，frame 至少包含 sample_key、metric、oracle_model、oracle_value。
        """
        if self.oracle_path is None:
            raise ValueError("未配置 oracle_path，无法读取 oracle labels")
        ordered_keys, explicit = _normalize_sample_keys(sample_keys)
        df = self._read_rows(self.oracle_path, ordered_keys if explicit else None, metric=metric)
        _require_columns(df, DEFAULT_ORACLE_COLUMNS, source=str(self.oracle_path))
        df["sample_key"] = df["sample_key"].astype(str)
        df["metric"] = df["metric"].astype(str)
        df = df[df["metric"] == str(metric)].copy()
        ordered_keys = ordered_keys if explicit else _first_seen_keys(df)
        df, missing_report = _dedupe_and_order(
            df,
            ordered_keys,
            stage="oracle",
            missing_policy=self.missing_policy,
        )
        metadata = {
            "oracle_path": str(self.oracle_path),
            "metric": str(metric),
            "explicit_sample_keys": explicit,
            "usage": "supervision_or_upper_bound_or_diagnostic_only_not_feature_provider",
            "reader_scope": "read_validate_join_only",
        }
        return OracleTsfBatch(ordered_keys, df.reset_index(drop=True), missing_report, metadata)

    def load_tsf(self, sample_keys: Optional[Sequence[str]]) -> OracleTsfBatch:
        """
        函数功能：
            按 sample_key 读取 TSF enrichment / TSF-cell metadata。

        输入：
            sample_keys: 目标 sample_key 列表；显式传入时输出保持该顺序。

        输出：
            OracleTsfBatch，frame 包含可用 TSF 元信息列；若源文件来自
            `manifest_with_tsf_cell.csv`，会先检查 sample_key 是否被重复记录。
        """
        if self.tsf_path is None:
            raise ValueError("未配置 tsf_path，无法读取 TSF enrichment")
        ordered_keys, explicit = _normalize_sample_keys(sample_keys)
        df = self._read_rows(self.tsf_path, ordered_keys if explicit else None)
        _require_columns(df, ["sample_key"], source=str(self.tsf_path))
        df["sample_key"] = df["sample_key"].astype(str)
        ordered_keys = ordered_keys if explicit else _first_seen_keys(df)
        tsf_columns = _select_tsf_columns(df)
        if not tsf_columns:
            raise ValueError(f"TSF 源文件缺少可识别的 enrichment 字段：{self.tsf_path}")
        stable_columns = [col for col in STABLE_JOIN_COLUMNS if col in df.columns]
        selected_columns = ["sample_key", *stable_columns, *tsf_columns]
        df = _collapse_identical_rows(
            df[selected_columns].copy(),
            stage="tsf",
        )
        df, missing_report = _dedupe_and_order(
            df,
            ordered_keys,
            stage="tsf",
            missing_policy=self.missing_policy,
        )
        metadata = {
            "tsf_path": str(self.tsf_path),
            "explicit_sample_keys": explicit,
            "tsf_columns": tsf_columns,
            "stable_columns": stable_columns,
            "usage": "stratified_summary_or_baseline_or_diagnostic_only",
            "reader_scope": "read_validate_join_only",
        }
        return OracleTsfBatch(ordered_keys, df.reset_index(drop=True), missing_report, metadata)

    def load_joined(self, sample_keys: Optional[Sequence[str]], metric: str = "mae") -> OracleTsfBatch:
        """
        函数功能：
            读取 oracle 与 TSF，并按 sample_key 做一对一 join。

        输入：
            sample_keys: 目标 sample_key 列表；显式传入时输出保持该顺序。
            metric: oracle 指标。

        输出：
            OracleTsfBatch，frame 包含 oracle label、oracle error/regret 列和 TSF 元信息。
        """
        oracle = self.load_oracle(sample_keys, metric=metric)
        tsf = self.load_tsf(oracle.sample_keys)
        joined = oracle.frame.merge(tsf.frame, on="sample_key", how="left", suffixes=("", "_tsf"), validate="one_to_one")
        joined, joined_missing = _dedupe_and_order(
            joined,
            oracle.sample_keys,
            stage="joined",
            missing_policy=self.missing_policy,
        )
        missing_report = {
            "oracle": oracle.missing_report,
            "tsf": tsf.missing_report,
            "joined": joined_missing,
        }
        metadata = {
            "oracle": oracle.metadata,
            "tsf": tsf.metadata,
            "lineage": {
                "metric": str(metric),
                "join_key": "sample_key",
                "join_mode": "left_oracle_to_tsf_validate_one_to_one",
                "leakage_guard": "oracle_and_tsf_not_exported_as_deployable_test_time_dynamic_features",
            },
        }
        return OracleTsfBatch(oracle.sample_keys, joined.reset_index(drop=True), missing_report, metadata)

    def _read_rows(self, path: Path, sample_keys: Optional[Sequence[str]], metric: Optional[str] = None) -> pd.DataFrame:
        """函数功能：根据文件后缀和 sample_key 显式性选择小批量读取策略。"""
        if sample_keys is None and not self.allow_full_scan:
            raise ValueError(
                f"未显式传入 sample_keys 时默认禁止全扫描：{path}；"
                "小规模 smoke 可设置 allow_full_scan=True。"
            )
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return _read_csv_subset(path, sample_keys, metric=metric, chunk_rows=self.chunk_rows)
        if suffix in {".parquet", ".pq"}:
            return _read_parquet_subset(path, sample_keys, metric=metric)
        raise ValueError(f"暂不支持的 oracle/TSF 文件格式：{path}")


def _first_existing(*paths: Path) -> Path:
    """函数功能：按优先级返回第一个存在的 fixture 文件。"""
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"fixture 中找不到 oracle/TSF 候选文件：{[str(path) for path in paths]}")


def _normalize_sample_keys(sample_keys: Optional[Sequence[str]]) -> Tuple[List[str], bool]:
    """函数功能：规范化 sample_key 输入，并检查显式输入是否重复。"""
    if sample_keys is None:
        return [], False
    ordered_keys = [str(key) for key in sample_keys]
    if len(ordered_keys) != len(set(ordered_keys)):
        raise ValueError("sample_keys 中存在重复 key")
    return ordered_keys, True


def _read_csv_subset(
    path: Path,
    sample_keys: Optional[Sequence[str]],
    *,
    metric: Optional[str],
    chunk_rows: int,
) -> pd.DataFrame:
    """函数功能：chunk 读取 CSV，并只保留目标 sample_key / metric。"""
    target_set = None if sample_keys is None else set(sample_keys)
    chunks: List[pd.DataFrame] = []
    seen_columns: Optional[List[str]] = None
    for chunk_df in pd.read_csv(path, chunksize=chunk_rows):
        if seen_columns is None:
            seen_columns = list(chunk_df.columns)
        if "sample_key" not in chunk_df.columns:
            raise ValueError(f"{path} 缺少 sample_key 字段")
        chunk_df = chunk_df.copy()
        chunk_df["sample_key"] = chunk_df["sample_key"].astype(str)
        if target_set is not None:
            chunk_df = chunk_df[chunk_df["sample_key"].isin(target_set)]
        if metric is not None and "metric" in chunk_df.columns:
            chunk_df = chunk_df[chunk_df["metric"].astype(str) == str(metric)]
        if not chunk_df.empty:
            chunks.append(chunk_df)
    if not chunks:
        # 即使目标 key 全部缺失，也保留源文件表头，后续才能生成明确 missing report。
        return pd.DataFrame(columns=seen_columns or [])
    return pd.concat(chunks, ignore_index=True)


def _read_parquet_subset(path: Path, sample_keys: Optional[Sequence[str]], *, metric: Optional[str]) -> pd.DataFrame:
    """
    函数功能：
        用 pyarrow dataset 对 Parquet 做列过滤，避免显式 batch 读取退化为全表加载。
    """
    try:
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise ImportError("读取 Parquet oracle/TSF 需要 pyarrow，以便按 sample_key 过滤。") from exc

    dataset = ds.dataset(path, format="parquet")
    filters = []
    if sample_keys is not None:
        filters.append(pc.field("sample_key").isin([str(key) for key in sample_keys]))
    if metric is not None:
        filters.append(pc.field("metric") == str(metric))
    expression = None
    for current in filters:
        expression = current if expression is None else expression & current
    table = dataset.to_table(filter=expression)
    return table.to_pandas()


def _require_columns(df: pd.DataFrame, required_columns: Sequence[str], *, source: str) -> None:
    """函数功能：检查输入 DataFrame 是否包含必要字段。"""
    missing = sorted(set(required_columns).difference(df.columns))
    if missing:
        raise ValueError(f"{source} 缺少字段：{missing}")


def _first_seen_keys(df: pd.DataFrame) -> List[str]:
    """函数功能：按源文件首次出现顺序推断 sample_key。"""
    if df.empty:
        return []
    return df["sample_key"].drop_duplicates().astype(str).tolist()


def _select_tsf_columns(df: pd.DataFrame) -> List[str]:
    """函数功能：选择源文件中存在的 TSF enrichment 字段。"""
    return [col for col in DEFAULT_TSF_COLUMNS if col in df.columns and col != "sample_key"]


def _collapse_identical_rows(df: pd.DataFrame, *, stage: str) -> pd.DataFrame:
    """
    函数功能：
        折叠完全一致的重复元信息行，并拒绝同一 sample_key 下字段冲突的重复行。
    """
    collapsed = df.drop_duplicates().copy()
    duplicated_keys = sorted(collapsed.loc[collapsed["sample_key"].duplicated(), "sample_key"].unique().tolist())
    if duplicated_keys:
        raise ValueError(f"{stage} 存在重复 sample_key 或冲突元信息，示例：{duplicated_keys[:5]}")
    return collapsed


def _dedupe_and_order(
    df: pd.DataFrame,
    ordered_keys: Sequence[str],
    *,
    stage: str,
    missing_policy: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    函数功能：
        检查 sample_key 覆盖和重复，并按 ordered_keys 恢复输出顺序。
    """
    if "sample_key" not in df.columns:
        raise ValueError(f"{stage} frame 缺少 sample_key 字段")
    df = df.copy()
    df["sample_key"] = df["sample_key"].astype(str)
    duplicated_keys = sorted(df.loc[df["sample_key"].duplicated(), "sample_key"].unique().tolist())
    if duplicated_keys:
        raise ValueError(f"{stage} 存在重复 sample_key，示例：{duplicated_keys[:5]}")
    present_keys = set(df["sample_key"].tolist())
    missing_keys = [key for key in ordered_keys if key not in present_keys]
    extra_keys = [key for key in df["sample_key"].tolist() if key not in set(ordered_keys)]
    if missing_keys and missing_policy == "error":
        raise KeyError(f"{stage} 缺少 sample_key，示例：{missing_keys[:5]}")
    order_frame = pd.DataFrame({"sample_key": list(ordered_keys), "_reader_order": range(len(ordered_keys))})
    ordered = order_frame.merge(df, on="sample_key", how="left", validate="one_to_one")
    ordered = ordered.sort_values("_reader_order").drop(columns=["_reader_order"])
    missing_report: Dict[str, object] = {
        "stage": stage,
        "missing_count": len(missing_keys),
        "missing_sample_keys": missing_keys,
        "duplicate_count": len(duplicated_keys),
        "duplicate_sample_keys": duplicated_keys,
        "extra_count": len(extra_keys),
    }
    return ordered, missing_report
