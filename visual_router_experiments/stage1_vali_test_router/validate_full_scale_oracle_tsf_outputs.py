#!/usr/bin/env python3
"""
文件功能：
    校验 full-scale oracle labels 与 TSF enrichment 的覆盖、唯一性和 join 一致性。

输入：
    - oracle labels 输出目录；
    - TSF enrichment 输出目录；
    - merged cache 完整性校验 summary。

输出：
    - validation_summary.json：记录总行数、sample_key 唯一数、缺失率和覆盖一致性；
    - joined_oracle_tsf_sample.parquet：少量抽样 join 结果，便于人工检查字段契约。

设计说明：
    full-scale 文件很大，验证优先使用 parquet metadata 和分列扫描，避免把所有列一次性
    读入内存。join 一致性通过 sample_key 集合差异和 oracle metric 分布检查完成。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def now_cst() -> str:
    """函数功能：生成状态文件使用的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Validate full-scale oracle and TSF enrichment outputs.")
    parser.add_argument("--oracle-dir", type=Path, required=True, help="oracle labels 输出目录。")
    parser.add_argument("--tsf-dir", type=Path, required=True, help="TSF enrichment 输出目录。")
    parser.add_argument("--integrity-summary", type=Path, required=True, help="merged cache 完整性校验 summary。")
    parser.add_argument("--output-dir", type=Path, required=True, help="验证输出目录。")
    return parser.parse_args()


def load_status(path: Path) -> Dict[str, object]:
    """函数功能：读取并确认生成任务状态为 completed。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError(f"{path} 不是 completed 状态：{payload.get('status')}")
    return payload


def parquet_row_count(path: Path) -> int:
    """函数功能：读取 parquet metadata 中的总行数。"""
    return int(pq.ParquetFile(path).metadata.num_rows)


def distinct_count(path: Path, column: str) -> int:
    """函数功能：计算单列 distinct 数，用于 sample_key 唯一性校验。"""
    table = ds.dataset(path, format="parquet").to_table(columns=[column])
    return int(len(pc.unique(table[column])))


def value_counts(path: Path, column: str) -> Dict[str, int]:
    """函数功能：计算单列 value_counts。"""
    table = ds.dataset(path, format="parquet").to_table(columns=[column])
    counts = pc.value_counts(table[column]).to_pylist()
    return {str(item["values"]): int(item["counts"]) for item in counts}


def unique_key_table(path: Path):
    """函数功能：读取 sample_key 列并返回去重后的 Arrow Table。"""
    table = ds.dataset(path, format="parquet").to_table(columns=["sample_key"])
    # pyarrow 24 的 Table 没有 drop_duplicates，因此使用 compute.unique 明确构造 key 表。
    return pa_table_from_unique(pc.unique(table["sample_key"]))


def pa_table_from_unique(unique_values):
    """函数功能：把 unique() 结果转换为带 sample_key 列名的 Arrow Table。"""
    import pyarrow as pa

    return pa.Table.from_arrays([unique_values], names=["sample_key"])


def null_counts(path: Path, columns) -> Dict[str, int]:
    """函数功能：统计指定列缺失数量。"""
    table = ds.dataset(path, format="parquet").to_table(columns=list(columns))
    return {col: int(pc.sum(pc.is_null(table[col])).as_py() or 0) for col in columns}


def main() -> None:
    """函数功能：执行 oracle/TSF 输出验证。"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oracle_status = load_status(args.oracle_dir / "status.json")
    tsf_status = load_status(args.tsf_dir / "status.json")
    integrity = json.loads(args.integrity_summary.read_text(encoding="utf-8"))

    oracle_path = args.oracle_dir / "window_oracle_labels.parquet"
    tsf_path = args.tsf_dir / "sample_tsf_enrichment.parquet"
    expected_samples = int(integrity["actual_sample_key_unique_count"])
    expected_oracle_rows = expected_samples * 2

    oracle_rows = parquet_row_count(oracle_path)
    tsf_rows = parquet_row_count(tsf_path)
    oracle_unique_keys = distinct_count(oracle_path, "sample_key")
    tsf_unique_keys = distinct_count(tsf_path, "sample_key")
    metric_counts = value_counts(oracle_path, "metric")
    tsf_nulls = null_counts(
        tsf_path,
        ["sample_key", "cluster", "group_name", "forecastability_cat", "season_strength_cat", "trend_strength_cat", "cv_cat", "missing_ratio_cat"],
    )
    oracle_nulls = null_counts(oracle_path, ["sample_key", "metric", "oracle_model", "oracle_value"])

    # 用 anti-join 检查 sample_key 集合是否一致；只读 sample_key 列，避免加载全部宽列。
    oracle_keys = unique_key_table(oracle_path)
    tsf_keys = unique_key_table(tsf_path)
    oracle_minus_tsf = oracle_keys.join(tsf_keys, keys="sample_key", join_type="left anti")
    tsf_minus_oracle = tsf_keys.join(oracle_keys, keys="sample_key", join_type="left anti")

    passed = (
        oracle_rows == expected_oracle_rows
        and tsf_rows == expected_samples
        and oracle_unique_keys == expected_samples
        and tsf_unique_keys == expected_samples
        and metric_counts.get("mae", 0) == expected_samples
        and metric_counts.get("mse", 0) == expected_samples
        and oracle_minus_tsf.num_rows == 0
        and tsf_minus_oracle.num_rows == 0
        and all(count == 0 for count in oracle_nulls.values())
        and all(count == 0 for count in tsf_nulls.values())
    )

    sample = (
        ds.dataset(oracle_path, format="parquet")
        .to_table(columns=["sample_key", "config_name", "split", "dataset_name", "metric", "oracle_model", "oracle_value"])
        .slice(0, 1000)
        .join(ds.dataset(tsf_path, format="parquet").to_table().slice(0, 1000), keys="sample_key", join_type="inner")
    )
    pq.write_table(sample, args.output_dir / "joined_oracle_tsf_sample.parquet", compression="zstd")

    summary = {
        "status": "passed" if passed else "failed",
        "generated_at": now_cst(),
        "oracle_dir": str(args.oracle_dir),
        "tsf_dir": str(args.tsf_dir),
        "expected_sample_count": expected_samples,
        "expected_oracle_rows": expected_oracle_rows,
        "oracle_rows": oracle_rows,
        "oracle_sample_key_unique_count": oracle_unique_keys,
        "tsf_rows": tsf_rows,
        "tsf_sample_key_unique_count": tsf_unique_keys,
        "metric_counts": metric_counts,
        "oracle_missing_counts": oracle_nulls,
        "tsf_missing_counts": tsf_nulls,
        "oracle_minus_tsf_count": int(oracle_minus_tsf.num_rows),
        "tsf_minus_oracle_count": int(tsf_minus_oracle.num_rows),
        "oracle_status_path": str(args.oracle_dir / "status.json"),
        "tsf_status_path": str(args.tsf_dir / "status.json"),
        "oracle_status": oracle_status,
        "tsf_status": tsf_status,
    }
    output_path = args.output_dir / "validation_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
