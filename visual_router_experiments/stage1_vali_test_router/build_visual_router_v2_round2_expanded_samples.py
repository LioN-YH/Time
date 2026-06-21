#!/usr/bin/env python3
"""
文件功能：
    构建 Visual Router V2 Round2e-a 65k expanded validation 固定样本集。

输入：
    - Round2a small sample manifest，用作 strict subset 边界；
    - full-scale window oracle labels parquet；
    - full-scale sample TSF enrichment parquet。

输出：
    - round2_train_expanded / selection / diagnostic / test 四个 sample_key CSV；
    - round2_expanded_sample_manifest.csv；
    - overlap / coverage / validation / metadata / summary / status 轻量产物。

关键约束：
    本脚本只冻结 expanded sample boundary，不运行 ViT，不生成 feature cache，
    不训练 router，不保存 pseudo image tensor，也不读取 116M prediction manifest。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from build_visual_router_v2_pilot_samples import (
    MODEL_ORDER,
    OUTPUT_COLS,
    TSF_COLS,
    CandidateRow,
    SmallestHeap,
    attach_tsf,
    gap_quantile_label,
    load_tsf_subset,
    make_candidate_frame,
    rows_from_frame,
    scan_oracle_batches,
    validate_inputs,
)


DEFAULT_FULL_SCALE_ROOT = Path(
    "/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher"
)
DEFAULT_SMALL_SAMPLE_DIR = Path("/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_small_samples")
DEFAULT_OUTPUT_DIR = Path("/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples")
DEFAULT_LIGHT_SUMMARY_DIR = Path("experiment_summaries/visual_router_v2_round2/expanded_samples")
SCRIPT_VERSION = "visual_router_v2_round2e_a_expanded_sample_builder_v1"
RECOMMENDED_LAYOUTS = ["spatial_panel_3view", "current_rgb_3view", "top3fold_period_layout"]


def now_cst() -> str:
    """函数功能：返回 metadata、summary 和 status 使用的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 Round2e-a expanded sample builder 参数。"""
    parser = argparse.ArgumentParser(description="Build Visual Router V2 Round2e-a expanded sample sets.")
    parser.add_argument(
        "--oracle-labels-path",
        type=Path,
        default=DEFAULT_FULL_SCALE_ROOT / "oracle_labels_full_scale_2026-06-16" / "window_oracle_labels.parquet",
        help="full-scale window_oracle_labels.parquet 路径。",
    )
    parser.add_argument(
        "--tsf-enrichment-path",
        type=Path,
        default=DEFAULT_FULL_SCALE_ROOT / "tsf_enrichment_full_scale_2026-06-16" / "sample_tsf_enrichment.parquet",
        help="full-scale sample_tsf_enrichment.parquet 路径。",
    )
    parser.add_argument(
        "--small-sample-manifest",
        type=Path,
        default=DEFAULT_SMALL_SAMPLE_DIR / "round2_small_sample_manifest.csv",
        help="Round2a small sample manifest，用于 strict subset 边界。",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="expanded sample 输出目录。")
    parser.add_argument("--light-summary-dir", type=Path, default=DEFAULT_LIGHT_SUMMARY_DIR, help="仓库轻量 summary 目录。")
    parser.add_argument("--seed", type=int, default=20260622, help="expanded 补齐样本的固定抽样 seed。")
    parser.add_argument("--train-size", type=int, default=30_000, help="round2_train_expanded vali window 数。")
    parser.add_argument("--selection-size", type=int, default=10_000, help="round2_selection_expanded vali window 数。")
    parser.add_argument("--diagnostic-balanced-size", type=int, default=10_000, help="round2_diagnostic_balanced_expanded vali window 数。")
    parser.add_argument("--test-size", type=int, default=15_000, help="round2_test_expanded test window 数。")
    parser.add_argument("--gap-quantile-reservoir-size", type=int, default=500_000, help="估计 error_gap 分位边界的稳定哈希 reservoir 大小。")
    parser.add_argument("--batch-size", type=int, default=250_000, help="Parquet 扫描 batch 行数。")
    parser.add_argument("--no-copy-light-summary", action="store_true", help="只写外部输出目录，不复制轻量 summary 到仓库。")
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：写出 UTF-8 JSON，保留中文字段可读性。"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_small_frames(small_manifest_path: Path) -> Dict[str, pd.DataFrame]:
    """
    函数功能：
        读取 Round2a small manifest，并按 expanded 命名重映射 sample_set。

    设计说明：
        small manifest 只有 35k 行，读取到内存成本可控；它是本步 strict subset 的
        权威边界，而不是用于抽样的 feature 来源。
    """
    if not small_manifest_path.exists():
        raise FileNotFoundError(f"找不到 Round2 small sample manifest：{small_manifest_path}")
    small = pd.read_csv(small_manifest_path)
    required = set(OUTPUT_COLS)
    missing = sorted(required.difference(small.columns))
    if missing:
        raise RuntimeError(f"small manifest 缺少必要字段：{missing}")
    mapping = {
        "round2_train_small": "round2_train_expanded",
        "round2_selection_small": "round2_selection_expanded",
        "round2_diagnostic_balanced_small": "round2_diagnostic_balanced_expanded",
        "round2_test_small": "round2_test_expanded",
    }
    frames: Dict[str, pd.DataFrame] = {}
    for small_name, expanded_name in mapping.items():
        frame = small[small["sample_set"] == small_name].copy()
        if frame.empty:
            raise RuntimeError(f"small manifest 中没有 {small_name}")
        frame["sample_set"] = expanded_name
        frames[expanded_name] = frame[OUTPUT_COLS].reset_index(drop=True)
    return frames


def collect_gap_boundaries(oracle_path: Path, seed: int, batch_size: int, reservoir_size: int) -> Tuple[List[float], Dict[str, int]]:
    """
    函数功能：
        单独扫描 oracle labels，估计 error_gap 五分位边界。

    说明：
        使用稳定哈希 reservoir，而不是全量 error_gap 常驻内存；这延续 P0/Round2a
        的 deterministic hash sampling 口径。
    """
    gap_heap = SmallestHeap(reservoir_size)
    counters = {"mae_rows_seen": 0, "vali_rows_seen": 0, "test_rows_seen": 0}
    for batch_idx, batch in enumerate(scan_oracle_batches(oracle_path, batch_size)):
        frame = make_candidate_frame(batch, seed)
        if frame.empty:
            continue
        counters["mae_rows_seen"] += int(len(frame))
        counters["vali_rows_seen"] += int((frame["split"] == "vali").sum())
        counters["test_rows_seen"] += int((frame["split"] == "test").sum())
        for row in rows_from_frame(frame):
            gap_heap.push(row)
        if batch_idx % 20 == 0:
            print(
                f"[{now_cst()}] gap pass batch={batch_idx} mae_rows={counters['mae_rows_seen']} "
                f"vali_seen={counters['vali_rows_seen']} test_seen={counters['test_rows_seen']}",
                flush=True,
            )
    gap_values = np.asarray([row.error_gap for row in gap_heap.rows_sorted()], dtype=np.float64)
    if len(gap_values) == 0:
        raise RuntimeError("无法估计 error_gap quantile：reservoir 为空")
    return np.quantile(gap_values, [0.2, 0.4, 0.6, 0.8]).astype(float).tolist(), counters


def collect_natural_fillers(
    oracle_path: Path,
    seed: int,
    batch_size: int,
    *,
    vali_needed: int,
    test_needed: int,
    excluded_keys: Set[str],
) -> Tuple[List[CandidateRow], List[CandidateRow], Dict[str, int]]:
    """
    函数功能：
        从 oracle labels 中按稳定哈希抽取 expanded 主集合补齐样本。

    约束：
        `excluded_keys` 包含所有 small set 的 sample_key，避免补齐样本抢占 small
        selection/diagnostic/test 的 strict subset 位置。
    """
    vali_heap = SmallestHeap(vali_needed)
    test_heap = SmallestHeap(test_needed)
    counters = {"mae_rows_seen": 0, "vali_candidate_rows": 0, "test_candidate_rows": 0}
    for batch_idx, batch in enumerate(scan_oracle_batches(oracle_path, batch_size)):
        frame = make_candidate_frame(batch, seed)
        if frame.empty:
            continue
        counters["mae_rows_seen"] += int(len(frame))
        frame = frame[~frame["sample_key"].isin(excluded_keys)]
        if frame.empty:
            continue
        counters["vali_candidate_rows"] += int((frame["split"] == "vali").sum())
        counters["test_candidate_rows"] += int((frame["split"] == "test").sum())
        for row in rows_from_frame(frame):
            if row.split == "vali":
                vali_heap.push(row)
            elif row.split == "test":
                test_heap.push(row)
        if batch_idx % 20 == 0:
            print(
                f"[{now_cst()}] filler pass batch={batch_idx} "
                f"vali_heap={len(vali_heap.rows_sorted())} test_heap={len(test_heap.rows_sorted())}",
                flush=True,
            )
    vali_rows = vali_heap.rows_sorted()
    test_rows = test_heap.rows_sorted()
    if len(vali_rows) < vali_needed:
        raise RuntimeError(f"vali expanded 补齐候选不足：got={len(vali_rows)}, required={vali_needed}")
    if len(test_rows) < test_needed:
        raise RuntimeError(f"test expanded 补齐候选不足：got={len(test_rows)}, required={test_needed}")
    return vali_rows, test_rows, counters


def collect_diagnostic_fillers(
    oracle_path: Path,
    seed: int,
    batch_size: int,
    *,
    needed_by_model: Mapping[str, int],
    excluded_keys: Set[str],
) -> Tuple[Dict[str, List[CandidateRow]], Dict[str, int]]:
    """
    函数功能：
        从 vali 中为 diagnostic expanded 按 oracle_model 分桶补齐。

    设计说明：
        small diagnostic 已经作为每个 expert 的保留种子；这里只为低于目标的 expert
        补齐缺口，并排除 expanded train/selection/test，保证四个 sample_set 互斥。
    """
    diag_seed = seed + 10_003
    heaps = {model: SmallestHeap(max(0, int(needed_by_model.get(model, 0)))) for model in MODEL_ORDER}
    counters = {"mae_rows_seen": 0, "vali_candidate_rows": 0}
    for batch_idx, batch in enumerate(scan_oracle_batches(oracle_path, batch_size)):
        if all(heap.capacity == 0 for heap in heaps.values()):
            break
        frame = make_candidate_frame(batch, diag_seed)
        if frame.empty:
            continue
        counters["mae_rows_seen"] += int(len(frame))
        frame = frame[(frame["split"] == "vali") & (~frame["sample_key"].isin(excluded_keys))]
        if frame.empty:
            continue
        counters["vali_candidate_rows"] += int(len(frame))
        for row in rows_from_frame(frame):
            heaps[row.oracle_model].push(row)
        if batch_idx % 20 == 0:
            counts = {model: len(heaps[model].rows_sorted()) for model in MODEL_ORDER}
            print(f"[{now_cst()}] diagnostic filler pass batch={batch_idx} heap_counts={counts}", flush=True)
    fillers: Dict[str, List[CandidateRow]] = {}
    for model, needed in needed_by_model.items():
        rows = heaps[model].rows_sorted()
        if len(rows) < needed:
            raise RuntimeError(f"diagnostic expanded 缺少 {model} 补齐候选：got={len(rows)}, required={needed}")
        fillers[model] = rows[:needed]
    return fillers, counters


def rows_to_frame_with_start(
    sample_set: str,
    rows: Sequence[CandidateRow],
    boundaries: Sequence[float],
    *,
    start_index: int,
) -> pd.DataFrame:
    """函数功能：把补齐候选行转换为带连续 order_index 的输出表。"""
    records: List[Dict[str, object]] = []
    for offset, row in enumerate(rows):
        payload = row.to_dict()
        payload["sample_set"] = sample_set
        payload["order_index"] = start_index + offset
        payload["error_gap_quantile"] = gap_quantile_label(row.error_gap, boundaries)
        records.append(payload)
    return pd.DataFrame(records)


def append_fillers(
    small_frame: pd.DataFrame,
    filler_rows: Sequence[CandidateRow],
    sample_set: str,
    boundaries: Sequence[float],
    tsf_subset: pd.DataFrame,
) -> pd.DataFrame:
    """函数功能：把 small 保留行与 oracle 补齐行合并，并重建连续 order_index。"""
    base = small_frame.copy().reset_index(drop=True)
    base["order_index"] = np.arange(len(base), dtype=np.int64)
    filler_raw = rows_to_frame_with_start(sample_set, filler_rows, boundaries, start_index=len(base))
    if filler_raw.empty:
        combined = base
    else:
        filler = attach_tsf(filler_raw, tsf_subset)
        combined = pd.concat([base, filler], ignore_index=True)
    combined["order_index"] = np.arange(len(combined), dtype=np.int64)
    return combined[OUTPUT_COLS]


def build_coverage_summary(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """函数功能：生成 expanded samples 的 split/sample_set/coverage 字段分布长表。"""
    rows: List[Dict[str, object]] = []
    fields = [
        "split",
        "sample_set",
        "dataset_name",
        "group_name",
        "oracle_model",
        "error_gap_quantile",
        "forecastability_cat",
        "season_strength_cat",
        "trend_strength_cat",
        "cv_cat",
        "missing_ratio_cat",
    ]
    for sample_set, frame in frames.items():
        total = int(len(frame))
        for field in fields:
            counts = frame[field].astype("string").value_counts(dropna=False).sort_index()
            for value, count in counts.items():
                rows.append(
                    {
                        "split": str(frame["split"].iloc[0]) if len(frame) else "",
                        "sample_set": sample_set,
                        "field": field,
                        "value": str(value),
                        "count": int(count),
                        "fraction": float(count) / total if total else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def compute_overlap_with_small(frames: Mapping[str, pd.DataFrame], small_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """函数功能：统计每个 small set 是否严格包含于对应 expanded set。"""
    rows: List[Dict[str, object]] = []
    for expanded_name, expanded_frame in frames.items():
        small_keys = set(small_frames[expanded_name]["sample_key"].astype(str))
        expanded_keys = set(expanded_frame["sample_key"].astype(str))
        overlap_count = len(small_keys.intersection(expanded_keys))
        rows.append(
            {
                "expanded_sample_set": expanded_name,
                "small_sample_set": expanded_name.replace("_expanded", "_small"),
                "small_count": int(len(small_keys)),
                "expanded_count": int(len(expanded_keys)),
                "small_in_expanded_count": int(overlap_count),
                "small_in_expanded_ratio": float(overlap_count / len(small_keys)) if small_keys else 0.0,
                "strict_subset": bool(overlap_count == len(small_keys)),
                "reason_if_not_strict_subset": "" if overlap_count == len(small_keys) else "small sample key 未全部保留到对应 expanded set",
            }
        )
    return pd.DataFrame(rows)


def validate_expanded_sample_sets(frames: Mapping[str, pd.DataFrame], target_counts: Mapping[str, int]) -> Dict[str, object]:
    """函数功能：执行 Round2e-a 验收所需的 counts、唯一性、split 和 order_index 检查。"""
    per_set_counts = {name: int(len(frame)) for name, frame in frames.items()}
    per_set_duplicate_counts = {name: int(frame["sample_key"].duplicated().sum()) for name, frame in frames.items()}
    split_values = {name: sorted(frame["split"].unique().tolist()) for name, frame in frames.items()}
    order_index_contiguous = {
        name: bool((frame["order_index"].to_numpy() == np.arange(len(frame))).all()) for name, frame in frames.items()
    }
    all_keys: List[str] = []
    for frame in frames.values():
        all_keys.extend(frame["sample_key"].astype(str).tolist())
    cross_set_duplicate_count = len(all_keys) - len(set(all_keys))
    train_keys = set(frames["round2_train_expanded"]["sample_key"].astype(str))
    selection_keys = set(frames["round2_selection_expanded"]["sample_key"].astype(str))
    diagnostic_expert_counts = (
        frames["round2_diagnostic_balanced_expanded"]["oracle_model"].astype(str).value_counts().sort_index().to_dict()
    )
    validation = {
        "status": "passed",
        "generated_at": now_cst(),
        "per_set_counts": per_set_counts,
        "target_counts": dict(target_counts),
        "per_set_duplicate_counts": per_set_duplicate_counts,
        "cross_set_duplicate_count": int(cross_set_duplicate_count),
        "split_values": split_values,
        "round2_train_selection_intersection_count": int(len(train_keys.intersection(selection_keys))),
        "order_index_contiguous_from_zero": order_index_contiguous,
        "diagnostic_expert_counts": {str(k): int(v) for k, v in diagnostic_expert_counts.items()},
    }
    checks = [
        per_set_counts == dict(target_counts),
        all(count == 0 for count in per_set_duplicate_counts.values()),
        cross_set_duplicate_count == 0,
        split_values["round2_train_expanded"] == ["vali"],
        split_values["round2_selection_expanded"] == ["vali"],
        split_values["round2_diagnostic_balanced_expanded"] == ["vali"],
        split_values["round2_test_expanded"] == ["test"],
        validation["round2_train_selection_intersection_count"] == 0,
        all(order_index_contiguous.values()),
    ]
    if not all(checks):
        validation["status"] = "failed"
    return validation


def write_validation_csv(path: Path, validation: Mapping[str, object], overlap: pd.DataFrame) -> None:
    """函数功能：把验收结果写成长表 CSV，方便无 JSON parser 时快速审阅。"""
    rows: List[Dict[str, object]] = []
    for sample_set, count in validation["per_set_counts"].items():
        rows.append({"check": "sample_count", "sample_set": sample_set, "value": int(count), "passed": count == validation["target_counts"][sample_set]})
    for sample_set, count in validation["per_set_duplicate_counts"].items():
        rows.append({"check": "per_set_duplicate_count", "sample_set": sample_set, "value": int(count), "passed": count == 0})
    for sample_set, splits in validation["split_values"].items():
        expected = ["test"] if sample_set == "round2_test_expanded" else ["vali"]
        rows.append({"check": "split_values", "sample_set": sample_set, "value": ",".join(splits), "passed": splits == expected})
    for sample_set, passed in validation["order_index_contiguous_from_zero"].items():
        rows.append({"check": "order_index_contiguous_from_zero", "sample_set": sample_set, "value": bool(passed), "passed": bool(passed)})
    rows.append({"check": "cross_set_duplicate_count", "sample_set": "all", "value": int(validation["cross_set_duplicate_count"]), "passed": validation["cross_set_duplicate_count"] == 0})
    rows.append({"check": "train_selection_intersection_count", "sample_set": "train_selection", "value": int(validation["round2_train_selection_intersection_count"]), "passed": validation["round2_train_selection_intersection_count"] == 0})
    for model, count in validation["diagnostic_expert_counts"].items():
        rows.append({"check": "diagnostic_expert_count", "sample_set": model, "value": int(count), "passed": True})
    for row in overlap.itertuples(index=False):
        rows.append(
            {
                "check": "small_strict_subset",
                "sample_set": row.expanded_sample_set,
                "value": row.small_in_expanded_ratio,
                "passed": bool(row.strict_subset),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_summary(
    path: Path,
    *,
    metadata: Mapping[str, object],
    validation: Mapping[str, object],
    overlap: pd.DataFrame,
) -> None:
    """函数功能：写出 Round2e-a expanded sample builder 的人工审阅摘要。"""
    counts = metadata["sample_counts"]
    strict_subset = bool(metadata["strict_small_subset"])
    overlap_lines = "\n".join(
        f"| {row.small_sample_set} | {row.expanded_sample_set} | {row.small_in_expanded_count}/{row.small_count} | {row.small_in_expanded_ratio:.6f} | {row.strict_subset} |"
        for row in overlap.itertuples(index=False)
    )
    diag_counts = ", ".join(f"{model}={count}" for model, count in validation["diagnostic_expert_counts"].items())
    text = f"""# Visual Router V2 Round2e-a Expanded Samples Summary

生成时间：{metadata["generated_at"]}

## 为什么先构建 65k expanded samples

Round2a/Round2c 只冻结并验证了 35k small screening 边界。后续 Round2e-b 要比较 `spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout` 的 65k layout validation，必须先固定 train/selection/diagnostic/test expanded 边界，避免 feature cache、router 训练和 layout 选择过程中重新抽样造成口径漂移。

## Expanded sample sets

| sample_set | split | count | 用途 |
| --- | --- | ---: | --- |
| round2_train_expanded | vali | {counts["round2_train_expanded"]} | 后续 65k fixed FiLM 风格 router 训练 |
| round2_selection_expanded | vali | {counts["round2_selection_expanded"]} | 后续 layout/seed/epoch/hparam 选择；不含 train |
| round2_diagnostic_balanced_expanded | vali | {counts["round2_diagnostic_balanced_expanded"]} | oracle expert balanced 诊断，不用于选择 |
| round2_test_expanded | test | {counts["round2_test_expanded"]} | frozen expanded validation only，不用于训练、调参或选择 |

验证状态：`{validation["status"]}`；跨集合 sample_key 重复数：`{validation["cross_set_duplicate_count"]}`；train/selection 交集：`{validation["round2_train_selection_intersection_count"]}`。

## Small subset

35k small samples 是否是 65k expanded samples 的子集：`{strict_subset}`。

| small set | expanded set | small_in_expanded | ratio | strict_subset |
| --- | --- | ---: | ---: | --- |
{overlap_lines}

{"若 strict subset 为 false，原因见 `round2_expanded_overlap_with_small.csv`。" if not strict_subset else "本次通过先保留 small 边界、再稳定哈希补齐的方式保证四个 small set 均为对应 expanded set 的严格子集。"}

## Diagnostic balance

`round2_diagnostic_balanced_expanded` 保持 oracle expert balanced：{diag_counts}。该集合只用于诊断，不参与 layout/seed/epoch/hparam 选择。

## Test boundary

`round2_test_expanded` 全部来自 `test` split，且只用于 frozen expanded validation；metadata 中 `used_test_expanded_for_selection=false`。

## Round2e-b recommendation

后续 Round2e-b 应验证以下 layout：`spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout`。

Round2e-b 应继续固定 `film_mean_patch_aux` 风格后端，即 mean_patch visual embedding + RevIN aux FiLM modulation，避免把 layout 效果与 head/hparam 改动混在一起。

Round2e-b 应继续使用多 GPU 进程级并行：feature cache 可按 layout 分配 GPU，training/eval 可按 layout×seed 分配 GPU；本步没有启动 GPU、ViT、feature cache 或 router 训练。
"""
    path.write_text(text, encoding="utf-8")


def copy_light_summary(output_dir: Path, light_summary_dir: Path, files: Sequence[str]) -> Dict[str, str]:
    """函数功能：复制适合随仓库审阅的轻量文件，不复制大规模 cache 或逐样本 feature。"""
    light_summary_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    for filename in files:
        source = output_dir / filename
        target = light_summary_dir / filename
        shutil.copy2(source, target)
        copied[filename] = str(target)
    return copied


def main() -> None:
    """函数功能：执行 Round2e-a 65k expanded sample boundary 构建。"""
    args = parse_args()
    start = time.time()
    validate_inputs(args.oracle_labels_path, args.tsf_enrichment_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    small_frames = load_small_frames(args.small_sample_manifest)
    target_counts = {
        "round2_train_expanded": int(args.train_size),
        "round2_selection_expanded": int(args.selection_size),
        "round2_diagnostic_balanced_expanded": int(args.diagnostic_balanced_size),
        "round2_test_expanded": int(args.test_size),
    }
    small_counts = {name: int(len(frame)) for name, frame in small_frames.items()}
    if any(small_counts[name] > target_counts[name] for name in target_counts):
        raise RuntimeError(f"small count 大于 expanded target：small={small_counts}, target={target_counts}")

    gap_boundaries, gap_scan_counters = collect_gap_boundaries(
        args.oracle_labels_path,
        seed=args.seed,
        batch_size=args.batch_size,
        reservoir_size=args.gap_quantile_reservoir_size,
    )

    all_small_keys = set().union(*(set(frame["sample_key"].astype(str)) for frame in small_frames.values()))
    vali_needed = (args.train_size - small_counts["round2_train_expanded"]) + (
        args.selection_size - small_counts["round2_selection_expanded"]
    )
    test_needed = args.test_size - small_counts["round2_test_expanded"]
    vali_fillers, test_fillers, filler_scan_counters = collect_natural_fillers(
        args.oracle_labels_path,
        seed=args.seed,
        batch_size=args.batch_size,
        vali_needed=vali_needed,
        test_needed=test_needed,
        excluded_keys=all_small_keys,
    )
    train_extra_needed = args.train_size - small_counts["round2_train_expanded"]
    selection_extra_needed = args.selection_size - small_counts["round2_selection_expanded"]
    train_fillers = vali_fillers[:train_extra_needed]
    selection_fillers = vali_fillers[train_extra_needed : train_extra_needed + selection_extra_needed]

    main_frames_raw = {
        "round2_train_expanded": small_frames["round2_train_expanded"],
        "round2_selection_expanded": small_frames["round2_selection_expanded"],
        "round2_test_expanded": small_frames["round2_test_expanded"],
    }
    main_filler_rows = {
        "round2_train_expanded": train_fillers,
        "round2_selection_expanded": selection_fillers,
        "round2_test_expanded": test_fillers,
    }
    main_filler_keys = set(row.sample_key for rows in main_filler_rows.values() for row in rows)

    main_selected_keys = set().union(*(set(frame["sample_key"].astype(str)) for frame in main_frames_raw.values()))
    main_selected_keys.update(main_filler_keys)
    diag_small = small_frames["round2_diagnostic_balanced_expanded"].copy()
    diag_small_counts = diag_small["oracle_model"].astype(str).value_counts().to_dict()
    per_model_target = int(math.ceil(args.diagnostic_balanced_size / len(MODEL_ORDER)))
    needed_by_model = {
        model: max(0, per_model_target - int(diag_small_counts.get(model, 0))) for model in MODEL_ORDER
    }
    # 若 small diagnostic 已经不是完全均衡，先按每类 ceil 目标补齐，再按全局稳定顺序裁到 10k。
    diag_excluded = set(main_selected_keys).union(set(diag_small["sample_key"].astype(str)))
    diag_fillers_by_model, diag_scan_counters = collect_diagnostic_fillers(
        args.oracle_labels_path,
        seed=args.seed,
        batch_size=args.batch_size,
        needed_by_model=needed_by_model,
        excluded_keys=diag_excluded,
    )
    diag_fillers = sorted(
        [row for rows in diag_fillers_by_model.values() for row in rows],
        key=lambda row: (row.oracle_model, row.score, row.sample_key),
    )

    filler_keys_for_tsf = set(row.sample_key for rows in main_filler_rows.values() for row in rows)
    filler_keys_for_tsf.update(row.sample_key for row in diag_fillers)
    tsf_subset = load_tsf_subset(args.tsf_enrichment_path, filler_keys_for_tsf, args.batch_size)

    frames = {
        name: append_fillers(main_frames_raw[name], rows, name, gap_boundaries, tsf_subset)
        for name, rows in main_filler_rows.items()
    }
    diag_frame = append_fillers(diag_small, diag_fillers, "round2_diagnostic_balanced_expanded", gap_boundaries, tsf_subset)
    if len(diag_frame) > args.diagnostic_balanced_size:
        # 保持 small 行优先，然后按 oracle_model/score 排序补齐裁剪，保证 order_index 连续。
        diag_frame = diag_frame.iloc[: args.diagnostic_balanced_size].copy()
        diag_frame["order_index"] = np.arange(len(diag_frame), dtype=np.int64)
    frames["round2_diagnostic_balanced_expanded"] = diag_frame[OUTPUT_COLS]

    ordered_names = [
        "round2_train_expanded",
        "round2_selection_expanded",
        "round2_diagnostic_balanced_expanded",
        "round2_test_expanded",
    ]
    frames = {name: frames[name] for name in ordered_names}

    output_files: Dict[str, str] = {}
    for name, frame in frames.items():
        output_path = args.output_dir / f"{name}_sample_keys.csv"
        frame.to_csv(output_path, index=False)
        output_files[f"{name}_sample_keys"] = str(output_path)

    manifest = pd.concat([frames[name] for name in ordered_names], ignore_index=True)
    manifest_path = args.output_dir / "round2_expanded_sample_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    overlap = compute_overlap_with_small(frames, small_frames)
    overlap_path = args.output_dir / "round2_expanded_overlap_with_small.csv"
    overlap.to_csv(overlap_path, index=False)

    coverage = build_coverage_summary(frames)
    coverage_path = args.output_dir / "round2_expanded_coverage_summary.csv"
    coverage.to_csv(coverage_path, index=False)

    validation = validate_expanded_sample_sets(frames, target_counts)
    validation_path = args.output_dir / "round2_expanded_validation_summary.csv"
    write_validation_csv(validation_path, validation, overlap)
    if validation["status"] != "passed":
        raise RuntimeError(f"Round2 expanded 样本集验证失败，详见 {validation_path}")

    strict_small_subset = bool(overlap["strict_subset"].all())
    small_to_expanded_overlap = {
        str(row.expanded_sample_set): {
            "small_sample_set": str(row.small_sample_set),
            "small_count": int(row.small_count),
            "expanded_count": int(row.expanded_count),
            "small_in_expanded_count": int(row.small_in_expanded_count),
            "small_in_expanded_ratio": float(row.small_in_expanded_ratio),
            "strict_subset": bool(row.strict_subset),
            "reason_if_not_strict_subset": str(row.reason_if_not_strict_subset),
        }
        for row in overlap.itertuples(index=False)
    }
    metadata = {
        "status": "completed",
        "script_version": SCRIPT_VERSION,
        "generated_at": now_cst(),
        "elapsed_sec": round(time.time() - start, 3),
        "script": str(Path(__file__).resolve()),
        "round2_stage": "expanded_sample_builder",
        "trained_model": False,
        "built_feature_cache": False,
        "ran_vit": False,
        "saved_pseudo_image_tensor": False,
        "used_test_expanded_for_selection": False,
        "loaded_116m_prediction_manifest_to_memory": False,
        "oracle_labels_path": str(args.oracle_labels_path),
        "tsf_enrichment_path": str(args.tsf_enrichment_path),
        "output_dir": str(args.output_dir),
        "light_summary_dir": None if args.no_copy_light_summary else str(args.light_summary_dir),
        "sample_source": {
            "round2_train_expanded": "Round2 train small strict subset + vali metric=mae oracle labels stable-hash fillers",
            "round2_selection_expanded": "Round2 selection small strict subset + disjoint vali metric=mae oracle labels stable-hash fillers",
            "round2_diagnostic_balanced_expanded": "Round2 diagnostic small strict subset + disjoint vali oracle_model balanced fillers",
            "round2_test_expanded": "Round2 test small strict subset + test metric=mae oracle labels stable-hash fillers; frozen validation only",
        },
        "sample_counts": {name: int(len(frame)) for name, frame in frames.items()},
        "sample_set_boundaries": {
            "round2_train_expanded_and_round2_selection_expanded_disjoint": True,
            "round2_diagnostic_balanced_expanded_used_for_selection": False,
            "round2_test_expanded_split": "test",
            "round2_test_expanded_used_for_training_tuning_or_selection": False,
            "all_sets_cross_disjoint": True,
            "small_sets_preserved_before_hash_fill": True,
        },
        "hash_seed_or_sampling_rule": {
            "seed": int(args.seed),
            "main_fillers": "先保留各自 small set，再从排除全部 small sample_key 的 oracle labels 中按 seed+sample_key 稳定哈希补齐；train 取 vali filler 前段，selection 取后段，test 从 test filler 补齐。",
            "diagnostic_seed": int(args.seed + 10_003),
            "diagnostic_fillers": "先保留 diagnostic small，再排除 expanded train/selection/test 和 diagnostic small，按 oracle_model 分桶稳定哈希补齐。",
            "error_gap_quantile": "基于 metric=mae oracle 行的稳定哈希 reservoir 估计五分位边界。",
        },
        "oracle_balance_rule": {
            "sample_set": "round2_diagnostic_balanced_expanded",
            "models": MODEL_ORDER,
            "per_model_target": int(per_model_target),
            "small_diagnostic_expert_counts": {str(k): int(v) for k, v in diag_small_counts.items()},
            "filler_needed_by_model": {str(k): int(v) for k, v in needed_by_model.items()},
            "final_expert_counts": validation["diagnostic_expert_counts"],
            "selection_usage": "diagnostic_only_not_for_layout_selection",
        },
        "small_sample_source": str(args.small_sample_manifest),
        "small_to_expanded_overlap": small_to_expanded_overlap,
        "strict_small_subset": strict_small_subset,
        "recommended_layouts_for_round2e_b": RECOMMENDED_LAYOUTS,
        "next_step_recommendation": "round2e_b_65k_layout_validation",
        "round2e_b_backend_recommendation": "continue fixed film_mean_patch_aux style backend",
        "round2e_b_parallel_recommendation": "continue process-level multi-GPU parallelism for layout feature cache and layout×seed training/eval",
        "gap_quantile_boundaries": gap_boundaries,
        "oracle_scan_counters": {
            "gap_pass": gap_scan_counters,
            "main_filler_pass": filler_scan_counters,
            "diagnostic_filler_pass": diag_scan_counters,
        },
        "validation": validation,
        "output_files": {
            **output_files,
            "round2_expanded_sample_manifest": str(manifest_path),
            "round2_expanded_overlap_with_small": str(overlap_path),
            "round2_expanded_coverage_summary": str(coverage_path),
            "round2_expanded_validation_summary": str(validation_path),
        },
    }

    metadata_path = args.output_dir / "round2_expanded_sample_metadata.json"
    write_json(metadata_path, metadata)

    summary_path = args.output_dir / "round2_expanded_sample_summary.md"
    write_summary(summary_path, metadata=metadata, validation=validation, overlap=overlap)
    metadata["output_files"]["round2_expanded_sample_metadata"] = str(metadata_path)
    metadata["output_files"]["round2_expanded_sample_summary"] = str(summary_path)

    status = {
        "status": "completed",
        "generated_at": now_cst(),
        "elapsed_sec": round(time.time() - start, 3),
        "script_version": SCRIPT_VERSION,
        "output_dir": str(args.output_dir),
        "validation_status": validation["status"],
        "sample_counts": metadata["sample_counts"],
        "strict_small_subset": strict_small_subset,
        "trained_model": False,
        "built_feature_cache": False,
        "ran_vit": False,
        "saved_pseudo_image_tensor": False,
        "loaded_116m_prediction_manifest_to_memory": False,
    }
    status_path = args.output_dir / "status.json"
    write_json(status_path, status)
    metadata["output_files"]["status"] = str(status_path)

    if not args.no_copy_light_summary:
        copied = copy_light_summary(
            args.output_dir,
            args.light_summary_dir,
            files=[
                "round2_expanded_overlap_with_small.csv",
                "round2_expanded_coverage_summary.csv",
                "round2_expanded_validation_summary.csv",
                "round2_expanded_sample_metadata.json",
                "round2_expanded_sample_summary.md",
                "status.json",
            ],
        )
        metadata["light_summary_files"] = copied

    write_json(metadata_path, metadata)
    if not args.no_copy_light_summary:
        shutil.copy2(metadata_path, args.light_summary_dir / "round2_expanded_sample_metadata.json")

    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
