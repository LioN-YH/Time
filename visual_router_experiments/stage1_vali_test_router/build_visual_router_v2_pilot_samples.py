#!/usr/bin/env python3
"""
文件功能：
    为 Visual Router V2 小规模架构诊断实验构建固定 sample_key 集合。

输入：
    - full-scale window oracle labels parquet；
    - full-scale sample TSF enrichment parquet。

输出：
    - pilot_train_sample_keys.csv；
    - pilot_selection_sample_keys.csv；
    - pilot_test_sample_keys.csv；
    - diagnostic_balanced_sample_keys.csv；
    - sample_set_metadata.json；
    - coverage_summary.csv；
    - validation_summary.json。

关键约束：
    本脚本只读取已经生成好的 oracle/TSF parquet，不读取 116M 行 merged
    prediction manifest，不读取 future y 作为 feature，也不启动任何训练。
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds


MODEL_ORDER = ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]
TSF_COLS = [
    "cluster",
    "group_name",
    "forecastability_cat",
    "season_strength_cat",
    "trend_strength_cat",
    "cv_cat",
    "missing_ratio_cat",
]
META_COLS = ["config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]
ORACLE_COLS = ["sample_key", *META_COLS, "metric", "oracle_model", *MODEL_ORDER]
OUTPUT_COLS = [
    "sample_set",
    "order_index",
    "sample_key",
    *META_COLS,
    "oracle_model",
    "error_gap",
    "error_gap_quantile",
    *TSF_COLS,
]


def now_cst() -> str:
    """函数功能：返回日志和 metadata 使用的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析命令行参数，并给出当前 full-scale 成品的默认路径。"""
    default_root = Path("/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher")
    parser = argparse.ArgumentParser(description="Build fixed Visual Router V2 pilot sample sets.")
    parser.add_argument(
        "--oracle-labels-path",
        type=Path,
        default=default_root / "oracle_labels_full_scale_2026-06-16" / "window_oracle_labels.parquet",
        help="full-scale window_oracle_labels.parquet 路径。",
    )
    parser.add_argument(
        "--tsf-enrichment-path",
        type=Path,
        default=default_root / "tsf_enrichment_full_scale_2026-06-16" / "sample_tsf_enrichment.parquet",
        help="full-scale sample_tsf_enrichment.parquet 路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples"),
        help="样本集输出目录。",
    )
    parser.add_argument("--seed", type=int, default=20260620, help="固定抽样 seed。")
    parser.add_argument("--pilot-train-size", type=int, default=150_000, help="pilot_train vali window 数。")
    parser.add_argument("--pilot-selection-size", type=int, default=30_000, help="pilot_selection vali window 数。")
    parser.add_argument("--pilot-test-size", type=int, default=75_000, help="pilot_test test window 数。")
    parser.add_argument("--diagnostic-balanced-size", type=int, default=20_000, help="diagnostic_balanced vali window 数。")
    parser.add_argument(
        "--gap-quantile-reservoir-size",
        type=int,
        default=1_000_000,
        help="用于估计全量 error_gap 分位边界的稳定哈希 reservoir 大小。",
    )
    parser.add_argument("--batch-size", type=int, default=250_000, help="Parquet 扫描 batch 行数。")
    return parser.parse_args()


@dataclass(frozen=True)
class CandidateRow:
    """函数功能：保存一个候选 sample_key 的轻量监督与覆盖字段。"""

    score: int
    sample_key: str
    config_name: str
    split: str
    dataset_name: str
    item_id: int
    channel_id: int
    window_index: int
    oracle_model: str
    error_gap: float

    def to_dict(self) -> Dict[str, object]:
        """函数功能：转换为后续 DataFrame/CSV 使用的字典。"""
        return {
            "sample_key": self.sample_key,
            "config_name": self.config_name,
            "split": self.split,
            "dataset_name": self.dataset_name,
            "item_id": self.item_id,
            "channel_id": self.channel_id,
            "window_index": self.window_index,
            "oracle_model": self.oracle_model,
            "error_gap": self.error_gap,
        }


class SmallestHeap:
    """
    函数功能：
        保留 score 最小的固定数量候选。

    设计说明：
        score 由 seed 和 sample_key 的稳定哈希决定。使用 heap 后只需保存目标规模附近的
        样本，不会把 2327 万个 oracle sample_key 全部常驻内存。
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._heap: List[Tuple[int, int, CandidateRow]] = []
        self._counter = 0

    def push(self, row: CandidateRow) -> None:
        """函数功能：尝试把候选加入 top-k 最小 score 集合。"""
        if self.capacity <= 0:
            return
        entry = (-int(row.score), self._counter, row)
        self._counter += 1
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, entry)
        elif entry > self._heap[0]:
            # entry 的第一项是负 score；更大的 entry 代表真实 score 更小。
            heapq.heapreplace(self._heap, entry)

    def rows_sorted(self) -> List[CandidateRow]:
        """函数功能：按稳定 score 从小到大返回候选。"""
        return [item[2] for item in sorted(self._heap, key=lambda entry: (-entry[0], entry[1]))]


def validate_inputs(oracle_path: Path, tsf_path: Path) -> None:
    """函数功能：检查输入文件存在，避免误启动后才失败。"""
    if not oracle_path.exists():
        raise FileNotFoundError(f"找不到 oracle labels parquet：{oracle_path}")
    if not tsf_path.exists():
        raise FileNotFoundError(f"找不到 TSF enrichment parquet：{tsf_path}")


def stable_hash_scores(keys: pd.Series, seed: int) -> np.ndarray:
    """
    函数功能：
        基于 sample_key 和 seed 生成可复现 uint64 score。

    设计说明：
        pandas 的 SipHash 向量化实现比逐行 hashlib 更适合千万级 sample_key。这里把
        seed 作为字符串前缀参与哈希，保证同一 parquet 读取顺序变化时样本仍稳定。
    """
    seeded = str(seed) + "::" + keys.astype("string")
    return pd.util.hash_pandas_object(seeded, index=False).to_numpy(dtype=np.uint64, copy=False)


def make_candidate_frame(batch: object, seed: int) -> pd.DataFrame:
    """
    函数功能：
        从一个 oracle batch 中取 `metric=mae` 行，并计算 oracle 第一/第二专家误差差距。
    """
    frame = batch.to_pandas()
    if frame.empty:
        return frame
    frame = frame[frame["metric"] == "mae"].copy()
    if frame.empty:
        return frame
    values = frame[MODEL_ORDER].to_numpy(dtype=np.float64, copy=False)
    sorted_values = np.sort(values, axis=1)
    frame["error_gap"] = sorted_values[:, 1] - sorted_values[:, 0]
    frame["score"] = stable_hash_scores(frame["sample_key"], seed)
    return frame


def rows_from_frame(frame: pd.DataFrame) -> Iterable[CandidateRow]:
    """函数功能：把 DataFrame 行转换为 CandidateRow，集中处理类型转换。"""
    for row in frame.itertuples(index=False):
        yield CandidateRow(
            score=int(getattr(row, "score")),
            sample_key=str(getattr(row, "sample_key")),
            config_name=str(getattr(row, "config_name")),
            split=str(getattr(row, "split")),
            dataset_name=str(getattr(row, "dataset_name")),
            item_id=int(getattr(row, "item_id")),
            channel_id=int(getattr(row, "channel_id")),
            window_index=int(getattr(row, "window_index")),
            oracle_model=str(getattr(row, "oracle_model")),
            error_gap=float(getattr(row, "error_gap")),
        )


def scan_oracle_batches(oracle_path: Path, batch_size: int):
    """函数功能：按 batch 只读扫描 oracle parquet 所需列。"""
    scanner = ds.dataset(oracle_path, format="parquet").scanner(
        columns=ORACLE_COLS,
        filter=pc.field("metric") == "mae",
        batch_size=int(batch_size),
    )
    yield from scanner.to_batches()


def collect_main_sets_and_gap_boundaries(
    oracle_path: Path,
    seed: int,
    batch_size: int,
    train_size: int,
    selection_size: int,
    test_size: int,
    gap_reservoir_size: int,
) -> Tuple[List[CandidateRow], List[CandidateRow], List[CandidateRow], List[float], Dict[str, int]]:
    """
    函数功能：
        第一遍扫描 oracle：抽取自然分布 vali/test 主样本，并估计 error_gap 分位边界。
    """
    vali_heap = SmallestHeap(train_size + selection_size)
    test_heap = SmallestHeap(test_size)
    gap_heap = SmallestHeap(gap_reservoir_size)
    counters = {"mae_rows_seen": 0, "vali_rows_seen": 0, "test_rows_seen": 0}

    for batch_idx, batch in enumerate(scan_oracle_batches(oracle_path, batch_size)):
        frame = make_candidate_frame(batch, seed)
        if frame.empty:
            continue
        counters["mae_rows_seen"] += int(len(frame))
        counters["vali_rows_seen"] += int((frame["split"] == "vali").sum())
        counters["test_rows_seen"] += int((frame["split"] == "test").sum())

        for row in rows_from_frame(frame):
            if row.split == "vali":
                vali_heap.push(row)
            elif row.split == "test":
                test_heap.push(row)
            gap_heap.push(row)

        if batch_idx % 20 == 0:
            print(
                f"[{now_cst()}] pass1 batch={batch_idx} mae_rows={counters['mae_rows_seen']} "
                f"vali_seen={counters['vali_rows_seen']} test_seen={counters['test_rows_seen']}",
                flush=True,
            )

    vali_rows = vali_heap.rows_sorted()
    test_rows = test_heap.rows_sorted()
    if len(vali_rows) < train_size + selection_size:
        raise RuntimeError(f"vali 候选不足：got={len(vali_rows)}, required={train_size + selection_size}")
    if len(test_rows) < test_size:
        raise RuntimeError(f"test 候选不足：got={len(test_rows)}, required={test_size}")

    gap_values = np.asarray([row.error_gap for row in gap_heap.rows_sorted()], dtype=np.float64)
    if len(gap_values) == 0:
        raise RuntimeError("无法估计 error_gap quantile：reservoir 为空")
    boundaries = np.quantile(gap_values, [0.2, 0.4, 0.6, 0.8]).astype(float).tolist()
    return vali_rows[:train_size], vali_rows[train_size : train_size + selection_size], test_rows[:test_size], boundaries, counters


def collect_diagnostic_rows(
    oracle_path: Path,
    seed: int,
    batch_size: int,
    total_size: int,
    excluded_keys: Set[str],
) -> List[CandidateRow]:
    """
    函数功能：
        第二遍扫描 oracle：从 vali 中抽取 oracle expert 近似均衡诊断样本。

    约束：
        diagnostic_balanced 只用于诊断，不替代自然分布主指标；这里默认从 vali 中抽取，
        并排除 pilot_train/pilot_selection，避免架构选择诊断提前混入 test。
    """
    per_model = int(math.ceil(total_size / len(MODEL_ORDER)))
    heaps = {model: SmallestHeap(per_model) for model in MODEL_ORDER}
    diag_seed = seed + 10_003

    for batch_idx, batch in enumerate(scan_oracle_batches(oracle_path, batch_size)):
        frame = make_candidate_frame(batch, diag_seed)
        if frame.empty:
            continue
        frame = frame[(frame["split"] == "vali") & (~frame["sample_key"].isin(excluded_keys))]
        if frame.empty:
            continue
        for row in rows_from_frame(frame):
            heaps[row.oracle_model].push(row)
        if batch_idx % 20 == 0:
            counts = {model: len(heaps[model].rows_sorted()) for model in MODEL_ORDER}
            print(f"[{now_cst()}] pass2 batch={batch_idx} diagnostic_heap_counts={counts}", flush=True)

    rows: List[CandidateRow] = []
    for model in MODEL_ORDER:
        model_rows = heaps[model].rows_sorted()
        if not model_rows:
            raise RuntimeError(f"diagnostic_balanced 缺少 oracle_model={model} 候选")
        rows.extend(model_rows[:per_model])
    rows = sorted(rows, key=lambda row: (row.score, row.sample_key))[:total_size]
    if len(rows) < total_size:
        raise RuntimeError(f"diagnostic_balanced 候选不足：got={len(rows)}, required={total_size}")
    return rows


def gap_quantile_label(value: float, boundaries: Sequence[float]) -> str:
    """函数功能：把 error_gap 映射到固定的五分位标签。"""
    # 当第一条边界为 0 时，许多并列最优样本的 gap 也为 0；使用 side="left"
    # 可把这些最难区分样本保留在最低 gap 桶，避免 q1 人为空桶。
    idx = int(np.searchsorted(np.asarray(boundaries, dtype=np.float64), float(value), side="left"))
    return f"q{idx + 1}"


def rows_to_frame(sample_set: str, rows: Sequence[CandidateRow], boundaries: Sequence[float]) -> pd.DataFrame:
    """函数功能：把候选行转换为带 sample_set/order_index 的输出表。"""
    records: List[Dict[str, object]] = []
    for order_idx, row in enumerate(rows):
        payload = row.to_dict()
        payload["sample_set"] = sample_set
        payload["order_index"] = order_idx
        payload["error_gap_quantile"] = gap_quantile_label(row.error_gap, boundaries)
        records.append(payload)
    return pd.DataFrame(records)


def load_tsf_subset(tsf_path: Path, sample_keys: Set[str], batch_size: int) -> pd.DataFrame:
    """
    函数功能：
        从 TSF enrichment parquet 中只保留选中 sample_key 的元信息。

    设计说明：
        选中样本约几十万，直接把 sample_key set 放在内存中做 batch 过滤即可；这比用
        巨大 isin 表达式交给 Arrow dataset 更容易定位缺失并保持兼容。
    """
    columns = ["sample_key", *TSF_COLS]
    scanner = ds.dataset(tsf_path, format="parquet").scanner(columns=columns, batch_size=int(batch_size))
    frames: List[pd.DataFrame] = []
    for batch_idx, batch in enumerate(scanner.to_batches()):
        frame = batch.to_pandas()
        subset = frame[frame["sample_key"].isin(sample_keys)]
        if not subset.empty:
            frames.append(subset)
        if batch_idx % 20 == 0:
            print(f"[{now_cst()}] tsf batch={batch_idx} matched={sum(len(item) for item in frames)}", flush=True)
    if not frames:
        raise RuntimeError("TSF enrichment 中没有匹配到任何选中 sample_key")
    tsf = pd.concat(frames, ignore_index=True)
    duplicates = int(tsf.duplicated("sample_key").sum())
    if duplicates:
        raise RuntimeError(f"TSF enrichment 子集 sample_key 重复：{duplicates}")
    return tsf


def attach_tsf(frame: pd.DataFrame, tsf_subset: pd.DataFrame) -> pd.DataFrame:
    """函数功能：为样本集补充 TSF cell 元信息并保持原有顺序。"""
    merged = frame.merge(tsf_subset, on="sample_key", how="left", validate="one_to_one")
    missing = {col: int(merged[col].isna().sum()) for col in TSF_COLS}
    if any(missing.values()):
        raise RuntimeError(f"TSF 字段存在缺失：{missing}")
    return merged[OUTPUT_COLS]


def build_coverage_summary(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """函数功能：生成 split/dataset/oracle/error-gap/TSF cell 覆盖分布长表。"""
    rows: List[Dict[str, object]] = []
    fields = ["split", "dataset_name", "oracle_model", "error_gap_quantile", *TSF_COLS]
    for sample_set, frame in frames.items():
        total = int(len(frame))
        for field in fields:
            counts = frame[field].astype("string").value_counts(dropna=False).sort_index()
            for value, count in counts.items():
                rows.append(
                    {
                        "sample_set": sample_set,
                        "field": field,
                        "value": str(value),
                        "count": int(count),
                        "fraction": float(count) / total if total else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：写出 UTF-8 JSON。"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_sample_sets(frames: Mapping[str, pd.DataFrame]) -> Dict[str, object]:
    """
    函数功能：
        执行验收所需的边界、唯一性和交集检查。
    """
    per_set_counts = {name: int(len(frame)) for name, frame in frames.items()}
    per_set_duplicate_counts = {name: int(frame["sample_key"].duplicated().sum()) for name, frame in frames.items()}
    all_keys: List[str] = []
    for frame in frames.values():
        all_keys.extend(frame["sample_key"].astype(str).tolist())
    cross_set_duplicate_count = len(all_keys) - len(set(all_keys))

    train_keys = set(frames["pilot_train"]["sample_key"])
    selection_keys = set(frames["pilot_selection"]["sample_key"])
    split_values = {name: sorted(frame["split"].unique().tolist()) for name, frame in frames.items()}
    validation = {
        "status": "passed",
        "generated_at": now_cst(),
        "per_set_counts": per_set_counts,
        "per_set_duplicate_counts": per_set_duplicate_counts,
        "cross_set_duplicate_count": int(cross_set_duplicate_count),
        "split_values": split_values,
        "pilot_train_selection_intersection_count": int(len(train_keys.intersection(selection_keys))),
        "all_outputs_have_order_index": all(bool((frame["order_index"].to_numpy() == np.arange(len(frame))).all()) for frame in frames.values()),
    }
    checks = [
        all(count == 0 for count in per_set_duplicate_counts.values()),
        cross_set_duplicate_count == 0,
        split_values["pilot_train"] == ["vali"],
        split_values["pilot_selection"] == ["vali"],
        split_values["pilot_test"] == ["test"],
        split_values["diagnostic_balanced"] == ["vali"],
        validation["pilot_train_selection_intersection_count"] == 0,
        bool(validation["all_outputs_have_order_index"]),
    ]
    if not all(checks):
        validation["status"] = "failed"
    return validation


def main() -> None:
    """函数功能：执行固定 pilot sample set 构建。"""
    args = parse_args()
    start = time.time()
    validate_inputs(args.oracle_labels_path, args.tsf_enrichment_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows, selection_rows, test_rows, gap_boundaries, scan_counters = collect_main_sets_and_gap_boundaries(
        oracle_path=args.oracle_labels_path,
        seed=args.seed,
        batch_size=args.batch_size,
        train_size=args.pilot_train_size,
        selection_size=args.pilot_selection_size,
        test_size=args.pilot_test_size,
        gap_reservoir_size=args.gap_quantile_reservoir_size,
    )
    main_keys = {row.sample_key for row in train_rows + selection_rows + test_rows}
    diagnostic_rows = collect_diagnostic_rows(
        oracle_path=args.oracle_labels_path,
        seed=args.seed,
        batch_size=args.batch_size,
        total_size=args.diagnostic_balanced_size,
        excluded_keys=main_keys,
    )

    raw_frames = {
        "pilot_train": rows_to_frame("pilot_train", train_rows, gap_boundaries),
        "pilot_selection": rows_to_frame("pilot_selection", selection_rows, gap_boundaries),
        "pilot_test": rows_to_frame("pilot_test", test_rows, gap_boundaries),
        "diagnostic_balanced": rows_to_frame("diagnostic_balanced", diagnostic_rows, gap_boundaries),
    }
    all_selected_keys = set().union(*(set(frame["sample_key"]) for frame in raw_frames.values()))
    tsf_subset = load_tsf_subset(args.tsf_enrichment_path, all_selected_keys, args.batch_size)
    frames = {name: attach_tsf(frame, tsf_subset) for name, frame in raw_frames.items()}

    output_files: Dict[str, str] = {}
    for name, frame in frames.items():
        output_path = args.output_dir / f"{name}_sample_keys.csv"
        frame.to_csv(output_path, index=False)
        output_files[name] = str(output_path)

    coverage = build_coverage_summary(frames)
    coverage_path = args.output_dir / "coverage_summary.csv"
    coverage.to_csv(coverage_path, index=False)

    validation = validate_sample_sets(frames)
    validation_path = args.output_dir / "validation_summary.json"
    write_json(validation_path, validation)
    if validation["status"] != "passed":
        raise RuntimeError(f"样本集验证失败，详见 {validation_path}")

    metadata = {
        "status": "completed",
        "generated_at": now_cst(),
        "elapsed_sec": round(time.time() - start, 3),
        "script": str(Path(__file__).resolve()),
        "oracle_labels_path": str(args.oracle_labels_path),
        "tsf_enrichment_path": str(args.tsf_enrichment_path),
        "output_dir": str(args.output_dir),
        "seed": int(args.seed),
        "sampling_rules": {
            "pilot_train": "从 vali 中按 seed+sample_key 稳定哈希取自然分布前 N 个。",
            "pilot_selection": "从同一 vali 稳定哈希序列中取 pilot_train 之后的独立样本。",
            "pilot_test": "从 test 中按 seed+sample_key 稳定哈希取自然分布前 N 个。",
            "diagnostic_balanced": "从 vali 中排除主样本后，按 oracle_model 分桶近似均衡抽样；仅用于诊断，不替代自然分布主指标。",
            "error_gap_quantile": "基于全量 metric=mae oracle 行的稳定哈希 reservoir 估计五分位边界。",
        },
        "target_counts": {
            "pilot_train": int(args.pilot_train_size),
            "pilot_selection": int(args.pilot_selection_size),
            "pilot_test": int(args.pilot_test_size),
            "diagnostic_balanced": int(args.diagnostic_balanced_size),
        },
        "actual_counts": {name: int(len(frame)) for name, frame in frames.items()},
        "gap_quantile_boundaries": gap_boundaries,
        "oracle_scan_counters": scan_counters,
        "output_files": {**output_files, "coverage_summary": str(coverage_path), "validation_summary": str(validation_path)},
        "constraints": {
            "read_full_prediction_manifest": False,
            "read_future_y_as_feature": False,
            "started_training": False,
            "natural_distribution_sets": ["pilot_train", "pilot_selection", "pilot_test"],
            "diagnostic_set_replaces_main_metric": False,
        },
    }
    metadata_path = args.output_dir / "sample_set_metadata.json"
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
