#!/usr/bin/env python3
"""
文件功能：
    从已有 prediction subset SQLite 派生 PatchTST frozen cache，供双分支预测实验使用。

输入：
    - 65k sample manifest，提供 `sample_key` 与 split/sample_set 字段；
    - 已构建的 prediction SQLite index，提供 PatchTST 的 y_pred/y_true 数组路径；
    - 既有 packed/per-sample prediction array cache。

输出：
    单个 `.npz`，包含 `sample_key`、`split`、`h_ts`、`y_patchtst`、`y_true`。

关键约束：
    本脚本不训练 PatchTST，不重新评估原始时序模型；`h_ts` 默认由 `y_patchtst`
    flatten 得到，是目标中 “h_ts 和/或 y_ts” 的 y_ts fallback 表示。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from visual_router_experiments.common.prediction_array_io import (
    load_prediction_arrays_grouped,
    resolve_cache_array_path,
)


def display_time() -> str:
    """函数功能：生成 metadata 中使用的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def _decode_metadata_value(value: object) -> object:
    """函数功能：解析 SQLite metadata 中 JSON 编码的 value。"""
    text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_index_metadata(connection: sqlite3.Connection) -> Dict[str, object]:
    """函数功能：读取 prediction SQLite index 的 metadata 表。"""
    try:
        rows = connection.execute("SELECT key, value FROM index_metadata").fetchall()
    except sqlite3.Error as exc:
        raise ValueError("prediction SQLite 缺少 index_metadata 表，无法解析相对数组路径") from exc
    return {str(key): _decode_metadata_value(value) for key, value in rows}


def load_sample_manifest(
    *,
    sample_manifest: Path,
    split_field: str,
    sample_sets: Sequence[str] | None,
    max_samples: int | None,
) -> pd.DataFrame:
    """函数功能：读取并筛选 65k sample manifest，保留稳定顺序。"""
    df = pd.read_csv(sample_manifest)
    required = {"sample_key", split_field}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"sample manifest 缺少字段：{missing}")
    if sample_sets:
        wanted = {str(item) for item in sample_sets}
        df = df[df[split_field].astype(str).isin(wanted)].copy()
    if max_samples is not None:
        df = df.head(int(max_samples)).copy()
    if df.empty:
        raise ValueError("sample manifest 筛选后为空")
    if df["sample_key"].duplicated().any():
        dup = df.loc[df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"sample manifest 中 sample_key 重复，示例：{dup}")
    return df.reset_index(drop=True)


def fetch_patchtst_records(
    *,
    connection: sqlite3.Connection,
    sample_keys: Sequence[str],
    model_name: str,
    manifest_dir: Path,
) -> List[Mapping[str, object]]:
    """
    函数功能：
        从 SQLite 中按 sample_key 顺序读取 PatchTST records，并解析数组路径。
    """
    if not sample_keys:
        return []
    placeholders = ",".join(["?"] * len(sample_keys))
    rows = connection.execute(
        f"""
        SELECT sample_key, model_name, y_true_path, y_pred_path, mae, mse,
               array_storage, y_true_row_index, y_pred_row_index
        FROM prediction_index
        WHERE model_name = ? AND sample_key IN ({placeholders})
        """,
        [str(model_name), *[str(key) for key in sample_keys]],
    ).fetchall()
    by_key: Dict[str, Mapping[str, object]] = {}
    for row in rows:
        record = dict(row)
        sample_key = str(record["sample_key"])
        record["y_true_path"] = str(resolve_cache_array_path(str(record["y_true_path"]), manifest_dir))
        record["y_pred_path"] = str(resolve_cache_array_path(str(record["y_pred_path"]), manifest_dir))
        by_key[sample_key] = record

    missing = [str(key) for key in sample_keys if str(key) not in by_key]
    if missing:
        raise KeyError(f"prediction index 缺少 {model_name} records，示例：{missing[:10]}")
    return [by_key[str(key)] for key in sample_keys]


def build_cache(args: argparse.Namespace) -> None:
    """函数功能：执行 PatchTST prediction records 读取、数组加载和 npz 写出。"""
    sample_sets = [item for item in str(args.sample_sets).split(",") if item] if args.sample_sets else None
    manifest_df = load_sample_manifest(
        sample_manifest=args.sample_manifest,
        split_field=args.split_field,
        sample_sets=sample_sets,
        max_samples=args.max_samples,
    )
    sample_keys = manifest_df["sample_key"].astype(str).tolist()
    split = manifest_df[args.split_field].astype(str).to_numpy()

    connection = sqlite3.connect(str(args.prediction_index))
    connection.row_factory = sqlite3.Row
    try:
        metadata = load_index_metadata(connection)
        manifest_dir_value = metadata.get("manifest_dir")
        if manifest_dir_value is None:
            raise ValueError("prediction SQLite metadata 缺少 manifest_dir")
        manifest_dir = Path(str(manifest_dir_value))

        all_pred: List[np.ndarray] = []
        all_true: List[np.ndarray] = []
        for start in range(0, len(sample_keys), int(args.batch_size)):
            batch_keys = sample_keys[start : start + int(args.batch_size)]
            records = fetch_patchtst_records(
                connection=connection,
                sample_keys=batch_keys,
                model_name=args.model_name,
                manifest_dir=manifest_dir,
            )
            all_pred.append(load_prediction_arrays_grouped(records, "y_pred"))
            all_true.append(load_prediction_arrays_grouped(records, "y_true"))
    finally:
        connection.close()

    y_patchtst = np.concatenate(all_pred, axis=0).astype(np.float32, copy=False)
    y_true = np.concatenate(all_true, axis=0).astype(np.float32, copy=False)
    if y_patchtst.shape != y_true.shape:
        raise ValueError(f"PatchTST y_pred/y_true shape 不一致：{y_patchtst.shape} vs {y_true.shape}")
    # 以 PatchTST 预测 y_ts 作为冻结时序分支表示，保证没有 hidden cache 时仍能训练四类轻量融合头。
    h_ts = y_patchtst.reshape(y_patchtst.shape[0], -1).astype(np.float32, copy=False)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        sample_key=np.asarray(sample_keys, dtype=object),
        split=split.astype(object),
        h_ts=h_ts,
        y_patchtst=y_patchtst,
        y_true=y_true,
    )
    payload = {
        "created_at": display_time(),
        "sample_manifest": str(args.sample_manifest),
        "prediction_index": str(args.prediction_index),
        "output_npz": str(args.output_npz),
        "model_name": str(args.model_name),
        "split_field": str(args.split_field),
        "sample_sets": sample_sets,
        "sample_count": int(len(sample_keys)),
        "target_shape": list(y_true.shape[1:]),
        "h_ts_source": "flattened_y_patchtst",
    }
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"完成：PatchTST frozen cache 写入 {args.output_npz}，samples={len(sample_keys)}")


def parse_args() -> argparse.Namespace:
    """函数功能：解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Build PatchTST frozen npz cache from prediction subset SQLite.")
    parser.add_argument("--sample_manifest", type=Path, required=True)
    parser.add_argument("--prediction_index", type=Path, required=True)
    parser.add_argument("--output_npz", type=Path, required=True)
    parser.add_argument("--output_metadata", type=Path, required=True)
    parser.add_argument("--model_name", default="PatchTST")
    parser.add_argument("--split_field", default="sample_set")
    parser.add_argument("--sample_sets", default="")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """函数功能：脚本入口。"""
    build_cache(parse_args())


if __name__ == "__main__":
    main()
