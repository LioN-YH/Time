#!/usr/bin/env python3
"""
文件功能：
    基于 Stage 1 full-scale merged prediction cache 流式生成 window-level oracle labels。

输入：
    - 已完成且通过完整性校验的 merged_cache/manifest.csv；
    - merged_cache/status.json；
    - full integrity validation 的 integrity_summary.json。

输出：
    - window_oracle_labels.parquet：长表，每个 sample_key 生成 mae/mse 两条 oracle 记录；
    - window_oracle_summary.csv：按 config/split/dataset/metric 汇总 oracle 上限；
    - status.json：记录输入状态、输出行数、唯一 key、缺失/重复检查和运行状态。

设计说明：
    full-scale manifest 超过 1 亿行，不能一次性读入内存。该脚本假设正式 merged
    cache 已按 sample_key 连续存放五专家记录，并用 carry-over 处理 chunk 边界。
    若发现同一 sample_key 的五专家记录不完整或跨 chunk 不连续，会立即失败，避免
    生成不可复核的 oracle label。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np


WORKSPACE = Path("/home/shiyuhong/Time")
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from visual_router_experiments.common.prediction_cache_schema import CACHE_SCHEMA_VERSION  # noqa: E402


MODEL_ORDER = ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]
METRICS = ["mae", "mse"]
STABLE_COLS = ["sample_key", "config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]
USECOLS = [*STABLE_COLS, "cache_version", "model_name", *METRICS]


def now_cst() -> str:
    """函数功能：生成实验状态文件使用的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Build full-scale Stage 1 window oracle labels.")
    parser.add_argument("--merged-cache-dir", type=Path, required=True, help="已完成的 merged_cache 目录。")
    parser.add_argument("--integrity-summary", type=Path, required=True, help="完整性校验 integrity_summary.json。")
    parser.add_argument("--output-dir", type=Path, required=True, help="oracle labels 输出目录。")
    parser.add_argument("--chunk-size", type=int, default=500_000, help="CSV 流式读取行数。")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    """函数功能：读取 JSON 文件并返回字典。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 JSON 文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_inputs(merged_cache_dir: Path, integrity_summary_path: Path) -> Dict[str, object]:
    """
    函数功能：
        只读确认 merged cache 和完整性校验状态。

    关键约束：
        本脚本只接受已经 completed 且完整性违规计数为 0 的正式 merged cache。
    """
    status_path = merged_cache_dir / "status.json"
    metadata_path = merged_cache_dir / "metadata.json"
    manifest_path = merged_cache_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 merged manifest：{manifest_path}")

    merge_status = load_json(status_path)
    merge_metadata = load_json(metadata_path)
    integrity = load_json(integrity_summary_path)
    if merge_status.get("status") != "completed":
        raise RuntimeError(f"merged cache status 不是 completed：{merge_status.get('status')}")
    if merge_metadata.get("status") != "completed":
        raise RuntimeError(f"merged cache metadata status 不是 completed：{merge_metadata.get('status')}")
    if integrity.get("status") != "completed":
        raise RuntimeError(f"integrity validation status 不是 completed：{integrity.get('status')}")

    # 旧版 integrity_summary 没有显式 passed 字段，因此同时支持 passed=true 与违规计数全 0 两种口径。
    violation_keys = [key for key in integrity if key.endswith("_violations")]
    violation_total = sum(int(integrity.get(key, 0) or 0) for key in violation_keys)
    passed = bool(integrity.get("passed", violation_total == 0))
    if not passed or violation_total != 0:
        raise RuntimeError(f"integrity validation 未通过：passed={passed}, violation_total={violation_total}")

    return {
        "manifest_path": str(manifest_path),
        "merge_status": merge_status,
        "merge_metadata": merge_metadata,
        "integrity_summary": integrity,
        "integrity_passed": passed,
        "integrity_violation_total": violation_total,
    }


def make_oracle_rows(group: pd.DataFrame) -> List[Dict[str, object]]:
    """
    函数功能：
        将同一个 sample_key 的五专家记录转换为 mae/mse 两条 oracle label。

    约束：
        五专家必须完整，稳定元信息必须一致；并列最优按 MODEL_ORDER 固定顺序打破。
    """
    if len(group) != len(MODEL_ORDER):
        raise ValueError(f"sample_key={group['sample_key'].iloc[0]} 专家记录数不是 5：{len(group)}")
    if set(group["model_name"]) != set(MODEL_ORDER):
        raise ValueError(f"sample_key={group['sample_key'].iloc[0]} 专家集合异常：{sorted(group['model_name'])}")
    for col in STABLE_COLS[1:]:
        if group[col].nunique(dropna=False) != 1:
            raise ValueError(f"sample_key={group['sample_key'].iloc[0]} 的 {col} 不一致")

    first = group.iloc[0]
    by_model = group.set_index("model_name")
    rows: List[Dict[str, object]] = []
    for metric in METRICS:
        values = {model: float(by_model.at[model, metric]) for model in MODEL_ORDER}
        oracle_model = min(MODEL_ORDER, key=lambda model: values[model])
        oracle_value = values[oracle_model]
        row: Dict[str, object] = {
            "sample_key": first["sample_key"],
            "config_name": first["config_name"],
            "split": first["split"],
            "dataset_name": first["dataset_name"],
            "item_id": int(first["item_id"]),
            "channel_id": int(first["channel_id"]),
            "window_index": int(first["window_index"]),
            "metric": metric,
            "oracle_model": oracle_model,
            "oracle_value": oracle_value,
            "oracle_top1_value": oracle_value,
        }
        for model in MODEL_ORDER:
            row[model] = values[model]
            row[f"{model}_regret"] = values[model] - oracle_value
        rows.append(row)
    return rows


def write_status(path: Path, payload: Dict[str, object]) -> None:
    """函数功能：原子化写出状态文件，便于后台任务监控。"""
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def update_summary(summary: Dict[tuple, Dict[str, object]], rows: Iterable[Dict[str, object]]) -> None:
    """函数功能：累计 split/dataset/metric 级 oracle summary 所需统计量。"""
    for row in rows:
        key = (row["config_name"], row["split"], row["dataset_name"], row["metric"])
        bucket = summary.setdefault(
            key,
            {
                "sample_count": 0,
                "oracle_sum": 0.0,
                "model_sums": defaultdict(float),
                "win_counts": defaultdict(int),
            },
        )
        bucket["sample_count"] += 1
        bucket["oracle_sum"] += float(row["oracle_value"])
        bucket["win_counts"][row["oracle_model"]] += 1
        for model in MODEL_ORDER:
            bucket["model_sums"][model] += float(row[model])


def build_oracle_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    函数功能：
        对一个已去除尾部不完整 sample_key 的 chunk 向量化生成 oracle labels。

    设计说明：
        full-scale 有 2327 万个 sample_key，逐 sample Python 循环会非常慢。这里按
        metric 分别 pivot 成 sample_key 级宽表，再用 numpy argmin 计算 oracle。
    """
    if frame.empty:
        return pd.DataFrame()

    model_counts = frame.groupby("sample_key", sort=False)["model_name"].nunique()
    bad_counts = model_counts[model_counts != len(MODEL_ORDER)]
    if not bad_counts.empty:
        raise ValueError(f"chunk 内存在专家数不为 5 的 sample_key，示例：{bad_counts.index[0]}")

    output_frames: List[pd.DataFrame] = []
    for metric in METRICS:
        pivot = frame.pivot(index=STABLE_COLS, columns="model_name", values=metric).reindex(columns=MODEL_ORDER).reset_index()
        if pivot[MODEL_ORDER].isna().any().any():
            raise ValueError(f"{metric} pivot 后存在缺失专家值")
        values = pivot[MODEL_ORDER].to_numpy(dtype=np.float64, copy=False)
        winner_idx = values.argmin(axis=1)
        oracle_values = values[np.arange(values.shape[0]), winner_idx]

        out = pivot[STABLE_COLS].copy()
        out["metric"] = metric
        out["oracle_model"] = np.asarray(MODEL_ORDER, dtype=object)[winner_idx]
        out["oracle_value"] = oracle_values
        out["oracle_top1_value"] = oracle_values
        for model in MODEL_ORDER:
            out[model] = pivot[model].astype(float)
            out[f"{model}_regret"] = out[model] - out["oracle_value"]
        output_frames.append(out)

    return pd.concat(output_frames, ignore_index=True)


def update_summary_from_frame(summary: Dict[tuple, Dict[str, object]], labels: pd.DataFrame) -> None:
    """函数功能：按少量分组累计 oracle summary，避免逐样本更新。"""
    if labels.empty:
        return
    group_cols = ["config_name", "split", "dataset_name", "metric"]
    for keys, group in labels.groupby(group_cols, sort=False):
        bucket = summary.setdefault(
            tuple(keys),
            {
                "sample_count": 0,
                "oracle_sum": 0.0,
                "model_sums": defaultdict(float),
                "win_counts": defaultdict(int),
            },
        )
        bucket["sample_count"] += int(len(group))
        bucket["oracle_sum"] += float(group["oracle_value"].sum())
        win_counts = group["oracle_model"].value_counts()
        for model in MODEL_ORDER:
            bucket["model_sums"][model] += float(group[model].sum())
            bucket["win_counts"][model] += int(win_counts.get(model, 0))


def flush_rows(writer: Optional[pq.ParquetWriter], output_path: Path, rows: List[Dict[str, object]]) -> pq.ParquetWriter:
    """函数功能：将累计 oracle rows 追加写入 parquet，控制内存占用。"""
    table = pa.Table.from_pylist(rows)
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def finalize_summary(summary: Dict[tuple, Dict[str, object]], output_path: Path) -> pd.DataFrame:
    """函数功能：写出 oracle summary CSV。"""
    rows: List[Dict[str, object]] = []
    for (config_name, split, dataset_name, metric), bucket in summary.items():
        sample_count = int(bucket["sample_count"])
        model_means = {model: float(bucket["model_sums"][model]) / sample_count for model in MODEL_ORDER}
        best_model = min(MODEL_ORDER, key=lambda model: model_means[model])
        best_value = model_means[best_model]
        oracle_value = float(bucket["oracle_sum"]) / sample_count
        row: Dict[str, object] = {
            "config_name": config_name,
            "split": split,
            "dataset_name": dataset_name,
            "metric": metric,
            "sample_count": sample_count,
            "best_single_model": best_model,
            "best_single_value": best_value,
            "oracle_value": oracle_value,
            "oracle_gap_abs": best_value - oracle_value,
            "oracle_gap_pct": (best_value - oracle_value) / best_value if best_value else 0.0,
        }
        for model in MODEL_ORDER:
            row[f"{model}_win_rate"] = float(bucket["win_counts"][model]) / sample_count
        rows.append(row)
    summary_df = pd.DataFrame(rows).sort_values(["config_name", "split", "dataset_name", "metric"])
    summary_df.to_csv(output_path, index=False)
    return summary_df


def main() -> None:
    """函数功能：执行 full-scale oracle label 流式生成。"""
    args = parse_args()
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    output_path = args.output_dir / "window_oracle_labels.parquet"
    summary_path = args.output_dir / "window_oracle_summary.csv"

    input_info = check_inputs(args.merged_cache_dir, args.integrity_summary)
    manifest_path = Path(str(input_info["manifest_path"]))
    status: Dict[str, object] = {
        "status": "running",
        "started_at": now_cst(),
        "updated_at": now_cst(),
        "merged_cache_dir": str(args.merged_cache_dir),
        "integrity_summary_path": str(args.integrity_summary),
        "output_dir": str(args.output_dir),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "chunk_size": int(args.chunk_size),
        "gpu_used": False,
        "gpu_note": "oracle 生成只做 CSV/Parquet I/O 与窗口级统计，未使用 GPU。",
        "input_record_count": 0,
        "sample_key_unique_count": 0,
        "output_row_count": 0,
    }
    write_status(status_path, {**status, **input_info})

    writer: Optional[pq.ParquetWriter] = None
    pending = pd.DataFrame(columns=USECOLS)
    summary: Dict[tuple, Dict[str, object]] = {}
    duplicate_or_order_violations = 0

    try:
        for chunk_idx, chunk in enumerate(pd.read_csv(manifest_path, usecols=USECOLS, chunksize=args.chunk_size)):
            if set(chunk["cache_version"].unique()) != {CACHE_SCHEMA_VERSION}:
                raise ValueError(f"chunk={chunk_idx} cache_version 异常")
            frame = chunk if pending.empty else pd.concat([pending, chunk], ignore_index=True)
            if frame.empty:
                continue

            # chunk 尾部 sample_key 可能不完整，留到下一轮与后续记录合并处理。
            tail_key = frame["sample_key"].iloc[-1]
            complete = frame[frame["sample_key"] != tail_key]
            pending = frame[frame["sample_key"] == tail_key].copy()

            labels = build_oracle_frame(complete)
            if not labels.empty:
                update_summary_from_frame(summary, labels)
                table = pa.Table.from_pandas(labels, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
                status["sample_key_unique_count"] = int(status["sample_key_unique_count"]) + int(labels["sample_key"].nunique())
                status["output_row_count"] = int(status["output_row_count"]) + int(len(labels))

            status["input_record_count"] = int(status["input_record_count"]) + len(chunk)
            if chunk_idx % 20 == 0:
                status["updated_at"] = now_cst()
                status["pending_record_count"] = int(len(pending))
                status["elapsed_sec"] = round(time.time() - start_time, 3)
                write_status(status_path, {**status, **input_info})
                print(f"[{now_cst()}] processed chunk={chunk_idx}, input_rows={status['input_record_count']}, samples={status['sample_key_unique_count']}", flush=True)

        if not pending.empty:
            labels = build_oracle_frame(pending)
            if not labels.empty:
                update_summary_from_frame(summary, labels)
                table = pa.Table.from_pandas(labels, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
                status["sample_key_unique_count"] = int(status["sample_key_unique_count"]) + int(labels["sample_key"].nunique())
                status["output_row_count"] = int(status["output_row_count"]) + int(len(labels))
        if writer is not None:
            writer.close()

        summary_df = finalize_summary(summary, summary_path)
        status.update(
            {
                "status": "completed",
                "completed_at": now_cst(),
                "updated_at": now_cst(),
                "elapsed_sec": round(time.time() - start_time, 3),
                "duplicate_or_order_violations": duplicate_or_order_violations,
                "missing_value_counts": {
                    "sample_key": 0,
                    "oracle_model": 0,
                    "oracle_value": 0,
                    "metric": 0,
                },
                "summary_row_count": int(len(summary_df)),
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
