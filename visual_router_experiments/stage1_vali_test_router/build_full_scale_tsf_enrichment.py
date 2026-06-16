#!/usr/bin/env python3
"""
文件功能：
    基于 Stage 1 full-scale merged manifest 生成 sample_key 级 TSF enrichment。

输入：
    - 已完成且通过完整性校验的 merged_cache/manifest.csv；
    - Quito item_clusters.csv。

输出：
    - sample_tsf_enrichment.parquet：每个 sample_key 一行，可直接与 oracle/router join；
    - tsf_missing_summary.csv：关键字段缺失率；
    - status.json：记录唯一性、覆盖范围、缺失率和运行状态。

设计说明：
    TSF cell 元信息由 item_id 映射而来，与 dataset_name 不是同一个概念。full-scale
    manifest 每个 sample_key 有五专家记录，本脚本只保留每个 sample_key 的第一条
    稳定元信息，避免 enrichment 重复 5 倍。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


WORKSPACE = Path("/home/shiyuhong/Time")
DEFAULT_CLUSTER_PATH = WORKSPACE / "quito" / "examples" / "datasets" / "cluster_data" / "item_clusters.csv"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from visual_router_experiments.common.prediction_cache_schema import CACHE_SCHEMA_VERSION  # noqa: E402


STABLE_COLS = ["sample_key", "config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]
USECOLS = [*STABLE_COLS, "cache_version", "model_name"]
TSF_COLS = [
    "cluster",
    "group_name",
    "forecastability_cat",
    "season_strength_cat",
    "trend_strength_cat",
    "cv_cat",
    "missing_ratio_cat",
]


def now_cst() -> str:
    """函数功能：生成实验状态文件使用的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Build full-scale Stage 1 sample TSF enrichment.")
    parser.add_argument("--merged-cache-dir", type=Path, required=True, help="已完成的 merged_cache 目录。")
    parser.add_argument("--integrity-summary", type=Path, required=True, help="完整性校验 integrity_summary.json。")
    parser.add_argument("--output-dir", type=Path, required=True, help="TSF enrichment 输出目录。")
    parser.add_argument("--cluster-path", type=Path, default=DEFAULT_CLUSTER_PATH, help="item_id 到 TSF cell 的映射 CSV。")
    parser.add_argument("--chunk-size", type=int, default=500_000, help="CSV 流式读取行数。")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    """函数功能：读取 JSON 文件并返回字典。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 JSON 文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_inputs(merged_cache_dir: Path, integrity_summary_path: Path) -> Dict[str, object]:
    """函数功能：只读确认 merged cache completed 且完整性校验通过。"""
    status_path = merged_cache_dir / "status.json"
    metadata_path = merged_cache_dir / "metadata.json"
    manifest_path = merged_cache_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 merged manifest：{manifest_path}")
    merge_status = load_json(status_path)
    merge_metadata = load_json(metadata_path)
    integrity = load_json(integrity_summary_path)
    if merge_status.get("status") != "completed" or merge_metadata.get("status") != "completed":
        raise RuntimeError("merged cache 不是 completed 状态")
    violation_keys = [key for key in integrity if key.endswith("_violations")]
    violation_total = sum(int(integrity.get(key, 0) or 0) for key in violation_keys)
    passed = bool(integrity.get("passed", violation_total == 0))
    if integrity.get("status") != "completed" or not passed or violation_total != 0:
        raise RuntimeError(f"integrity validation 未通过：status={integrity.get('status')}, passed={passed}, violations={violation_total}")
    return {
        "manifest_path": str(manifest_path),
        "merge_status": merge_status,
        "merge_metadata": merge_metadata,
        "integrity_summary": integrity,
        "integrity_passed": passed,
        "integrity_violation_total": violation_total,
    }


def load_cluster_mapping(cluster_path: Path) -> pd.DataFrame:
    """函数功能：读取 TSF cell 映射，并检查关键字段。"""
    if not cluster_path.exists():
        raise FileNotFoundError(f"找不到 cluster 映射：{cluster_path}")
    cluster_df = pd.read_csv(cluster_path)
    required = ["item_id", *TSF_COLS]
    missing = sorted(set(required).difference(cluster_df.columns))
    if missing:
        raise ValueError(f"cluster 映射缺少字段：{missing}")
    duplicate_items = int(cluster_df.duplicated("item_id").sum())
    if duplicate_items:
        raise ValueError(f"cluster 映射 item_id 重复：{duplicate_items}")
    return cluster_df[required]


def write_status(path: Path, payload: Dict[str, object]) -> None:
    """函数功能：原子化写出状态文件，便于后台任务监控。"""
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def append_parquet(writer: Optional[pq.ParquetWriter], output_path: Path, frame: pd.DataFrame) -> pq.ParquetWriter:
    """函数功能：追加写出 enrichment parquet。"""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def main() -> None:
    """函数功能：执行 full-scale sample_key 级 TSF enrichment 生成。"""
    args = parse_args()
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "sample_tsf_enrichment.parquet"
    missing_summary_path = args.output_dir / "tsf_missing_summary.csv"
    status_path = args.output_dir / "status.json"

    input_info = check_inputs(args.merged_cache_dir, args.integrity_summary)
    manifest_path = Path(str(input_info["manifest_path"]))
    cluster_df = load_cluster_mapping(args.cluster_path)

    status: Dict[str, object] = {
        "status": "running",
        "started_at": now_cst(),
        "updated_at": now_cst(),
        "merged_cache_dir": str(args.merged_cache_dir),
        "integrity_summary_path": str(args.integrity_summary),
        "cluster_path": str(args.cluster_path),
        "output_dir": str(args.output_dir),
        "output_path": str(output_path),
        "missing_summary_path": str(missing_summary_path),
        "chunk_size": int(args.chunk_size),
        "gpu_used": False,
        "gpu_note": "TSF enrichment 是 manifest/metadata join，未使用 GPU。",
        "input_record_count": 0,
        "sample_key_unique_count": 0,
        "duplicate_sample_key_count": 0,
        "missing_counts": {col: 0 for col in TSF_COLS},
    }
    write_status(status_path, {**status, **input_info})

    writer: Optional[pq.ParquetWriter] = None
    pending = pd.DataFrame(columns=USECOLS)
    last_key: Optional[str] = None

    try:
        for chunk_idx, chunk in enumerate(pd.read_csv(manifest_path, usecols=USECOLS, chunksize=args.chunk_size)):
            if set(chunk["cache_version"].unique()) != {CACHE_SCHEMA_VERSION}:
                raise ValueError(f"chunk={chunk_idx} cache_version 异常")
            frame = chunk if pending.empty else pd.concat([pending, chunk], ignore_index=True)
            tail_key = frame["sample_key"].iloc[-1]
            complete = frame[frame["sample_key"] != tail_key]
            pending = frame[frame["sample_key"] == tail_key].copy()

            sample_frame = complete.drop_duplicates("sample_key", keep="first")[STABLE_COLS]
            if last_key is not None and not sample_frame.empty and sample_frame["sample_key"].iloc[0] == last_key:
                status["duplicate_sample_key_count"] = int(status["duplicate_sample_key_count"]) + 1
                raise ValueError(f"sample_key={last_key} 在非连续位置重复出现")
            if not sample_frame.empty:
                last_key = str(sample_frame["sample_key"].iloc[-1])
                enriched = sample_frame.merge(cluster_df, on="item_id", how="left")
                for col in TSF_COLS:
                    status["missing_counts"][col] = int(status["missing_counts"][col]) + int(enriched[col].isna().sum())
                writer = append_parquet(writer, output_path, enriched)
                status["sample_key_unique_count"] = int(status["sample_key_unique_count"]) + int(len(enriched))

            status["input_record_count"] = int(status["input_record_count"]) + len(chunk)
            if chunk_idx % 20 == 0:
                status["updated_at"] = now_cst()
                status["pending_record_count"] = int(len(pending))
                status["elapsed_sec"] = round(time.time() - start_time, 3)
                write_status(status_path, {**status, **input_info})
                print(f"[{now_cst()}] processed chunk={chunk_idx}, input_rows={status['input_record_count']}, samples={status['sample_key_unique_count']}", flush=True)

        if not pending.empty:
            sample_frame = pending.drop_duplicates("sample_key", keep="first")[STABLE_COLS]
            if last_key is not None and not sample_frame.empty and sample_frame["sample_key"].iloc[0] == last_key:
                status["duplicate_sample_key_count"] = int(status["duplicate_sample_key_count"]) + 1
                raise ValueError(f"sample_key={last_key} 在尾部非连续重复出现")
            enriched = sample_frame.merge(cluster_df, on="item_id", how="left")
            for col in TSF_COLS:
                status["missing_counts"][col] = int(status["missing_counts"][col]) + int(enriched[col].isna().sum())
            writer = append_parquet(writer, output_path, enriched)
            status["sample_key_unique_count"] = int(status["sample_key_unique_count"]) + int(len(enriched))

        if writer is not None:
            writer.close()

        missing_rows: List[Dict[str, object]] = []
        total = int(status["sample_key_unique_count"])
        for col, count in status["missing_counts"].items():
            missing_rows.append({"field": col, "missing_count": int(count), "missing_rate": float(count) / total if total else 0.0})
        pd.DataFrame(missing_rows).to_csv(missing_summary_path, index=False)

        status.update(
            {
                "status": "completed",
                "completed_at": now_cst(),
                "updated_at": now_cst(),
                "elapsed_sec": round(time.time() - start_time, 3),
                "missing_summary_row_count": len(missing_rows),
                "sample_key_unique": True,
            }
        )
        write_status(status_path, {**status, **input_info})
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        if writer is not None:
            writer.close()
        status.update({"status": "failed", "failed_at": now_cst(), "updated_at": now_cst(), "error": repr(exc)})
        write_status(status_path, {**status, **input_info})
        raise


if __name__ == "__main__":
    main()
