#!/usr/bin/env python3
"""
文件功能：
    从 Stage 1 full-scale sample shards 中构建 Visual Router V2 Round2 staged
    full-scale validation 的小规模样本 manifest。

核心约束：
    - 只读取指定 sample shard 的 CSV，不读取 116M prediction manifest；
    - train / selection / diagnostic / test 通过 sample_set 严格分离；
    - 只用 oracle parquet 补齐审计标签，不把 oracle label 当作可部署特征；
    - 输出 manifest 继续复用现有 Round2 feature builder 和 fixed FiLM trainer。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.stage1_vali_test_router.evaluate_visual_router_v2_round0 import (  # noqa: E402
    DEFAULT_ORACLE_LABELS,
    load_oracle_subset,
)
from visual_router_experiments.stage1_vali_test_router.fusion_utils import MODEL_COLUMNS  # noqa: E402


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_FULL_SCALE_SAMPLE_SHARDS = (
    DATA2_RUN_OUTPUT_ROOT
    / "2026-06-15_stage1_96_48_s_full_scale"
    / "sample_manifest_full_scale"
    / "sample_shards"
)
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-22_visual_router_v2_round2_staged_fullscale_validation_thin_slice"
SCRIPT_VERSION = "visual_router_v2_round2_staged_samples_v1"
SAMPLE_SETS = ("staged_train", "staged_selection", "staged_diagnostic", "staged_test")


def display_time() -> str:
    """函数功能：生成写入 metadata/status 的本地时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 staged sample builder 参数。"""
    parser = argparse.ArgumentParser(description="Build Round2 staged full-scale validation sample manifest.")
    parser.add_argument("--full-scale-sample-shard-dir", type=Path, default=DEFAULT_FULL_SCALE_SAMPLE_SHARDS)
    parser.add_argument("--oracle-labels-path", type=Path, default=DEFAULT_ORACLE_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-scale", choices=["smoke", "one_shard"], default="smoke")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=64)
    parser.add_argument("--smoke-count-per-set", type=int, default=32)
    parser.add_argument("--one-shard-count-per-set", type=int, default=512)
    parser.add_argument("--parquet-batch-rows", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def git_commit_hash() -> str:
    """函数功能：记录当前 commit，失败时返回 unknown。"""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def shard_path(args: argparse.Namespace) -> Path:
    """函数功能：按 full-scale shard 编号返回 sample shard CSV 路径。"""
    return Path(args.full_scale_sample_shard_dir) / f"sample_shard_{int(args.shard_index):04d}_of_{int(args.shard_count):04d}.csv"


def _take_split_rows(frame: pd.DataFrame, split: str, start: int, count: int) -> pd.DataFrame:
    """
    函数功能：
        从单个 shard 的 split 内按稳定排序截取一段样本。

    说明：
        staged thin slice 只验证链路，不做代表性抽样结论；因此使用确定性 head/offset，
        便于重跑和审计。
    """
    rows = frame[frame["split"].astype(str) == str(split)].sort_values(
        ["dataset_name", "item_id", "channel_id", "window_index"], kind="mergesort"
    )
    part = rows.iloc[int(start) : int(start) + int(count)].copy()
    if len(part) != int(count):
        raise ValueError(f"split={split} 可用样本不足：need={count} got={len(part)} start={start}")
    return part.reset_index(drop=True)


def build_base_manifest(args: argparse.Namespace) -> pd.DataFrame:
    """函数功能：构建 train/selection/diagnostic/test 四个 staged sample_set。"""
    path = shard_path(args)
    if not path.exists():
        raise FileNotFoundError(f"找不到 full-scale sample shard：{path}")
    frame = pd.read_csv(path)
    required = {"sample_key", "config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} 缺少必要字段：{missing}")
    count = int(args.smoke_count_per_set if args.sample_scale == "smoke" else args.one_shard_count_per_set)
    slices = {
        "staged_train": _take_split_rows(frame, "vali", 0, count),
        "staged_selection": _take_split_rows(frame, "vali", count, count),
        "staged_diagnostic": _take_split_rows(frame, "vali", count * 2, count),
        "staged_test": _take_split_rows(frame, "test", 0, count),
    }
    rows: List[pd.DataFrame] = []
    for sample_set, part in slices.items():
        part = part.copy()
        part.insert(0, "order_index", np.arange(len(part), dtype=np.int64))
        part.insert(0, "sample_set", sample_set)
        rows.append(part)
    manifest = pd.concat(rows, ignore_index=True)
    if manifest["sample_key"].astype(str).duplicated().any():
        dup = manifest.loc[manifest["sample_key"].astype(str).duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"staged sample sets 之间存在重复 sample_key：{dup}")
    return manifest


def add_oracle_fields(manifest: pd.DataFrame, labels_path: Path, batch_rows: int) -> pd.DataFrame:
    """
    函数功能：
        从 oracle parquet 补齐 oracle_model、error_gap 和 error_gap_quantile。

    说明：
        error_gap 定义为五专家 MAE 第二优与第一优的差值，仅用于 strata/report。
    """
    keys = manifest["sample_key"].astype(str).tolist()
    labels = load_oracle_subset(labels_path, keys, batch_rows=int(batch_rows))
    label_cols = ["sample_key", "oracle_model", *MODEL_COLUMNS]
    merged = manifest.merge(labels[label_cols], on="sample_key", how="left", validate="one_to_one")
    if merged["oracle_model"].isna().any():
        raise ValueError("oracle labels merge 后存在缺失 oracle_model")
    expert_values = merged[MODEL_COLUMNS].to_numpy(dtype=np.float64)
    sorted_values = np.sort(expert_values, axis=1)
    merged["error_gap"] = (sorted_values[:, 1] - sorted_values[:, 0]).astype(np.float64)
    # 小样本可能有重复值，duplicates=drop 后再兜底为 q3，保证 schema 稳定。
    try:
        merged["error_gap_quantile"] = pd.qcut(merged["error_gap"], q=5, labels=["q1", "q2", "q3", "q4", "q5"], duplicates="drop")
        merged["error_gap_quantile"] = merged["error_gap_quantile"].astype(str).replace({"nan": "q3"})
    except ValueError:
        merged["error_gap_quantile"] = "q3"
    return merged.drop(columns=MODEL_COLUMNS)


def validate_manifest(frame: pd.DataFrame) -> Dict[str, object]:
    """函数功能：校验 staged manifest 的分离性、顺序和基础 schema。"""
    counts = {name: int((frame["sample_set"].astype(str) == name).sum()) for name in SAMPLE_SETS}
    errors: List[str] = []
    for sample_set in SAMPLE_SETS:
        part = frame[frame["sample_set"].astype(str) == sample_set]
        if part.empty:
            errors.append(f"{sample_set} empty")
            continue
        expected = np.arange(len(part), dtype=np.int64)
        actual = part["order_index"].to_numpy(dtype=np.int64, copy=False)
        if not np.array_equal(actual, expected):
            errors.append(f"{sample_set} order_index not contiguous")
        if part["sample_key"].astype(str).duplicated().any():
            errors.append(f"{sample_set} duplicated sample_key")
    split_by_set = {
        sample_set: sorted(frame.loc[frame["sample_set"].astype(str) == sample_set, "split"].astype(str).unique().tolist())
        for sample_set in SAMPLE_SETS
    }
    if set(frame.loc[frame["sample_set"].astype(str).isin(["staged_train", "staged_selection", "staged_diagnostic"]), "split"].astype(str)) != {"vali"}:
        errors.append("train/selection/diagnostic must use vali split")
    if set(frame.loc[frame["sample_set"].astype(str) == "staged_test", "split"].astype(str)) != {"test"}:
        errors.append("staged_test must use test split")
    return {
        "passed": not errors,
        "errors": errors,
        "counts": counts,
        "split_by_set": split_by_set,
        "unique_sample_key_count": int(frame["sample_key"].astype(str).nunique()),
    }


def main() -> None:
    """函数功能：写出 staged sample manifest、metadata 和 validation summary。"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "inputs" / f"round2_staged_{args.sample_scale}_sample_manifest.csv"
    metadata_path = output_dir / "inputs" / f"round2_staged_{args.sample_scale}_sample_metadata.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在；如需覆盖请传 --overwrite：{manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_base_manifest(args)
    manifest = add_oracle_fields(manifest, Path(args.oracle_labels_path), int(args.parquet_batch_rows))
    validation = validate_manifest(manifest)
    if not validation["passed"]:
        raise ValueError(f"staged manifest validation failed: {validation['errors']}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    metadata = {
        "status": "completed",
        "generated_at": display_time(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "commit_hash": git_commit_hash(),
        "sample_scale": str(args.sample_scale),
        "source_sample_shard": str(shard_path(args)),
        "oracle_labels_path": str(args.oracle_labels_path),
        "manifest_path": str(manifest_path),
        "validation": validation,
        "constraints": {
            "loaded_116m_prediction_manifest_to_memory": False,
            "saved_pseudo_image_tensor": False,
            "train_selection_diagnostic_split": "vali",
            "test_split": "test",
            "test_used_for_training_or_selection": False,
            "oracle_labels_used_as_report_labels_not_deployable_features": True,
        },
    }
    write_json(metadata_path, metadata)
    write_json(output_dir / "inputs" / "status.json", {"status": "completed", "updated_at": display_time(), "manifest_path": str(manifest_path)})
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
