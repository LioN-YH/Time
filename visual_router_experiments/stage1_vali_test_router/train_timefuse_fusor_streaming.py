#!/usr/bin/env python3
"""
文件功能：
    基于 `stage1_timefuse_fusor_streaming_reader.py` 训练和评估 Stage 1
    `96_48_S` full-scale TimeFuse-style fusor。

核心约束：
    - 复用 shard-local SQLite + batch reader，不做全量 manifest lookup、
      全量 DataFrame join 或全量 prediction 常驻内存；
    - fusor 固定为原生 TimeFuse-style `nn.Linear -> softmax -> weighted fusion`；
    - `StandardScaler` 只在 vali feature streaming 上 `partial_fit`；
    - train/test 阶段均按 batch 读取 packed 五专家 `y_pred/y_true`；
    - test streaming 同时输出 hard top-1 与 raw soft fusion 指标；
    - 该入口支持 1-2 shard 压力测试，不提供正式 64-shard launcher。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


WORKSPACE = Path("/home/shiyuhong/Time")
DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
FULL_SCALE_ROOT = DATA2_RUN_OUTPUT_ROOT / "2026-06-15_stage1_96_48_s_full_scale"
DEFAULT_FEATURE_SHARD_ROOT = FULL_SCALE_ROOT / "timefuse_feature_cache_full_scale_launcher" / "shards"
DEFAULT_LABELS_PATH = (
    FULL_SCALE_ROOT
    / "prediction_cache_full_scale_launcher"
    / "oracle_labels_full_scale_2026-06-16"
    / "window_oracle_labels.parquet"
)
DEFAULT_PREDICTION_SHARD_ROOT = FULL_SCALE_ROOT / "prediction_cache_full_scale_launcher" / "shards"

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from time_router.evaluation import EvaluationInputAdapter  # noqa: E402
from time_router.protocols import EvaluationInput  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.fusion_utils import (  # noqa: E402
    EPS,
    MODEL_COLUMNS,
    TimeFuseFusor,
    frame_to_markdown,
    summarize_hard_predictions,
    summarize_selected_model_counts,
    summarize_soft_fusion,
)
from visual_router_experiments.stage1_vali_test_router.stage1_timefuse_fusor_streaming_reader import (  # noqa: E402
    FEATURE_CACHE_COLUMNS,
    build_oracle_sqlite_index,
    build_prediction_sqlite_index,
    collect_feature_sample_keys,
    discover_prediction_shard_manifests,
    infer_feature_columns,
    Stage1TimeFuseFusorStreamingReader,
    TimeFuseFusorBatch,
)


FUSOR_VERSION = "stage1_timefuse_fusor_streaming_v1"


def display_time() -> str:
    """函数功能：生成写入中文日志、metadata、status 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def now_token() -> str:
    """函数功能：生成默认输出目录使用的时间戳。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def append_log(output_dir: Path, message: str) -> None:
    """函数功能：追加写主日志，便于压力测试中途接手。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "main.log").open("a", encoding="utf-8") as log_f:
        log_f.write(f"[{display_time()}] {message}\n")


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：原子写 JSON，避免监控时读到半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def to_jsonable(value: object) -> object:
    """
    函数功能：
        将 Path、numpy 标量/数组等运行时对象转为 JSON 可写格式。

    说明：
        checkpoint 可以保留 Python 对象给 `torch.save`，但 `metadata.json` 和
        `status.json` 面向人工与脚本监控，必须使用稳定的文本表示。
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_status(output_dir: Path, payload: Mapping[str, object]) -> None:
    """函数功能：写出 status.json，记录当前阶段、进度和资源快照。"""
    status = dict(payload)
    status["updated_at"] = display_time()
    status["output_dir"] = str(output_dir)
    status["pid"] = int(os.getpid())
    status["resources"] = collect_resource_snapshot()
    write_json_atomic(output_dir / "status.json", status)


def collect_resource_snapshot() -> Dict[str, object]:
    """
    函数功能：
        采集轻量资源快照。

    说明：
        psutil/nvidia-smi 不一定在所有环境可用，因此这里采用 best-effort；
        失败不会影响训练，只在 status/metadata 中记录错误。
    """
    snapshot: Dict[str, object] = {}
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory = process.memory_info()
        io_counters = process.io_counters()
        snapshot["process_memory"] = {
            "rss_mb": float(memory.rss / 1024 / 1024),
            "vms_mb": float(memory.vms / 1024 / 1024),
        }
        snapshot["process_io"] = {
            "read_mb": float(io_counters.read_bytes / 1024 / 1024),
            "write_mb": float(io_counters.write_bytes / 1024 / 1024),
        }
    except Exception as exc:  # pragma: no cover - 只影响运行环境观测
        snapshot["psutil_error"] = str(exc)

    snapshot["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        if torch.cuda.is_available():
            snapshot["torch_cuda"] = [
                {
                    "logical_index": int(idx),
                    "name": torch.cuda.get_device_name(idx),
                    "memory_allocated_mb": float(torch.cuda.memory_allocated(idx) / 1024 / 1024),
                    "memory_reserved_mb": float(torch.cuda.memory_reserved(idx) / 1024 / 1024),
                }
                for idx in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # pragma: no cover
        snapshot["torch_cuda_error"] = str(exc)

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            snapshot["nvidia_smi"] = result.stdout.strip().splitlines()
        else:
            snapshot["nvidia_smi_error"] = result.stderr.strip()
    except Exception as exc:  # pragma: no cover
        snapshot["nvidia_smi_error"] = str(exc)
    return snapshot


def resolve_device(device_arg: str) -> torch.device:
    """函数功能：解析训练设备，并强制 GPU 只允许暴露物理 2/3。"""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg in {"cuda", "auto"} and torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible and visible.replace(" ", "") not in {"2", "3", "2,3"}:
            raise ValueError("使用 GPU 时只允许 CUDA_VISIBLE_DEVICES=2,3 或其单卡子集")
        return torch.device("cuda:0")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 streaming TimeFuse fusor 训练/eval 参数。"""
    parser = argparse.ArgumentParser(description="Train/evaluate Stage 1 TimeFuse-style fusor with streaming shards.")
    parser.add_argument("--feature-shard-path", type=Path, action="append", default=None, help="feature_cache.csv shard；可重复传入 1-2 个。")
    parser.add_argument("--feature-shard-root", type=Path, default=DEFAULT_FEATURE_SHARD_ROOT, help="未显式传 shard 时自动取 shard 0000。")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH, help="full-scale oracle labels parquet。")
    parser.add_argument("--prediction-shard-root", type=Path, default=DEFAULT_PREDICTION_SHARD_ROOT, help="五专家 prediction shard 根目录。")
    parser.add_argument("--output-root", type=Path, default=DATA2_RUN_OUTPUT_ROOT, help="输出根目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式输出目录。")
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae", help="oracle label 和专家误差口径。")
    parser.add_argument("--epochs", type=int, default=1, help="训练 epoch。")
    parser.add_argument("--batch-size", type=int, default=128, help="streaming batch size。")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate。")
    parser.add_argument("--huber-beta", type=float, default=0.01, help="SmoothL1Loss beta。")
    parser.add_argument("--seed", type=int, default=16, help="随机种子。")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="运行设备。")
    parser.add_argument("--max-feature-shards", type=int, default=2, help="安全阈值：最多允许输入多少 feature shard。")
    parser.add_argument("--max-rows-per-split-per-shard", type=int, default=None, help="pressure/smoke 子集：每个 shard 每个 split 最多取前 N 行。None 表示使用完整 shard。")
    parser.add_argument("--feature-read-chunk-rows", type=int, default=200000, help="feature key/subset 扫描 chunk 行数。")
    parser.add_argument("--prediction-chunk-rows", type=int, default=200000, help="prediction manifest 扫描 chunk 行数。")
    parser.add_argument("--oracle-parquet-batch-rows", type=int, default=200000, help="oracle parquet 扫描 batch 行数。")
    parser.add_argument("--prediction-num-workers", type=int, default=2, help="当前 batch 内 prediction row 读取线程数。")
    parser.add_argument("--prefetch-batches", type=int, default=1, help="reader 预取 batch 数；底层限制 0/1。")
    parser.add_argument("--status-update-interval", type=int, default=20, help="每多少个 batch 更新 status.json。")
    parser.add_argument("--sample-prediction-limit", type=int, default=200, help="保存多少条 sample predictions。")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="从 checkpoint 加载 fusor/scaler。")
    parser.add_argument("--eval-only", action="store_true", help="只加载 checkpoint 并执行 test streaming eval。")
    parser.add_argument("--train-only", action="store_true", help="只训练和保存 checkpoint，不执行 test eval。")
    parser.add_argument(
        "--verify-evaluation-adapter",
        action="store_true",
        help="仅用于 pressure/smoke：在 evaluation batch 内用 EvaluationInputAdapter 旁路复算并校验指标一致性。",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """函数功能：固定主要随机源，便于 smoke/压力测试复验。"""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass(frozen=True)
class PreparedShard:
    """类功能：记录一个 feature/prediction shard 的 reader 依赖。"""

    shard_name: str
    feature_shard_path: Path
    original_feature_shard_path: Path
    prediction_manifest_paths: List[Path]
    sample_keys: List[str]
    oracle_db_path: Path
    prediction_db_path: Path


def default_feature_shards(root: Path) -> List[Path]:
    """函数功能：默认只取 shard 0000，避免误启动 64-shard 全量训练。"""
    path = Path(root) / "sample_shard_0000_of_0064" / "feature_cache.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到默认 feature shard：{path}")
    return [path]


def make_split_limited_feature_subset(
    feature_shard_path: Path,
    *,
    output_dir: Path,
    max_rows_per_split: Optional[int],
    chunk_rows: int,
) -> Path:
    """
    函数功能：
        为 smoke/压力测试创建每个 split 最多 N 行的小 feature subset。

    说明：
        full-scale shard 文件按 sample manifest 顺序排列，开头可能全是 test。
        直接取 head(N) 可能没有 vali 样本，无法验证训练闭环；该函数只在用户显式
        传 `--max-rows-per-split-per-shard` 时启用，正式完整 shard 训练不走此路径。
    """
    if max_rows_per_split is None:
        return feature_shard_path
    if int(max_rows_per_split) <= 0:
        raise ValueError("--max-rows-per-split-per-shard 必须为正整数")

    shard_name = feature_shard_path.parent.name
    subset_path = output_dir / "feature_subsets" / shard_name / "feature_cache.csv"
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    if subset_path.exists():
        subset_path.unlink()
    counts = {"vali": 0, "test": 0}
    wrote_header = False
    for chunk_df in pd.read_csv(feature_shard_path, chunksize=int(chunk_rows)):
        keep_parts: List[pd.DataFrame] = []
        for split_name in ["vali", "test"]:
            remaining = int(max_rows_per_split) - counts[split_name]
            if remaining <= 0:
                continue
            part = chunk_df[chunk_df["split"].astype(str) == split_name].head(remaining)
            if not part.empty:
                counts[split_name] += int(len(part))
                keep_parts.append(part)
        if keep_parts:
            out_df = pd.concat(keep_parts, ignore_index=True)
            out_df.to_csv(subset_path, mode="a", index=False, header=not wrote_header)
            wrote_header = True
        if all(value >= int(max_rows_per_split) for value in counts.values()):
            break
    if not wrote_header:
        raise ValueError(f"feature shard 未写出任何子集行：{feature_shard_path}")
    if counts["vali"] == 0 or counts["test"] == 0:
        raise ValueError(f"feature subset 缺少 vali 或 test：counts={counts} shard={feature_shard_path}")
    return subset_path


def sqlite_table_count(db_path: Path, table_name: str) -> Optional[int]:
    """
    函数功能：
        读取 SQLite 表行数，用于判断已有 shard-local index 是否可复用。

    说明：
        full-scale 任务重启时，64 个 oracle/prediction index 构建代价很高。
        只要最终 sqlite 文件存在且行数符合当前 shard 的 sample_key 数，就直接
        复用；若缺失、损坏或行数不匹配，则回退到重建该 shard。
    """
    if not db_path.exists():
        return None
    try:
        connection = sqlite3.connect(str(db_path))
        try:
            row = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            return int(row[0]) if row is not None else None
        finally:
            connection.close()
    except sqlite3.Error:
        return None


def reusable_shard_indexes(
    *,
    oracle_db_path: Path,
    prediction_db_path: Path,
    sample_key_count: int,
) -> bool:
    """
    函数功能：
        判断当前 shard 的 oracle/prediction SQLite 是否完整可复用。

    关键约束：
        prediction index 必须有 `sample_key_count * 5` 条记录，确保五专家完整；
        oracle index 必须有当前 shard 每个 sample_key 的一条指定 metric label。
    """
    oracle_count = sqlite_table_count(oracle_db_path, "oracle_index")
    prediction_count = sqlite_table_count(prediction_db_path, "prediction_index")
    return oracle_count == int(sample_key_count) and prediction_count == int(sample_key_count) * len(MODEL_COLUMNS)


def prepare_shards(args: argparse.Namespace, output_dir: Path, feature_cols: Sequence[str]) -> List[PreparedShard]:
    """
    函数功能：
        为每个输入 feature shard 构建 shard-local oracle/prediction SQLite。

    关键约束：
        每次只为 1-2 个 feature shard 的 sample_key 建索引；不会读取或保留
        116M 行 merged manifest 的全量 lookup。
    """
    feature_shards = args.feature_shard_path or default_feature_shards(args.feature_shard_root)
    if len(feature_shards) > int(args.max_feature_shards):
        raise ValueError(
            f"当前入口用于 smoke/压力测试，feature_shard_count={len(feature_shards)} "
            f"超过 --max-feature-shards={args.max_feature_shards}"
        )

    prepared: List[PreparedShard] = []
    for raw_feature_path in feature_shards:
        raw_feature_path = Path(raw_feature_path)
        subset_path = make_split_limited_feature_subset(
            raw_feature_path,
            output_dir=output_dir,
            max_rows_per_split=args.max_rows_per_split_per_shard,
            chunk_rows=int(args.feature_read_chunk_rows),
        )
        shard_name = raw_feature_path.parent.name
        sample_keys = collect_feature_sample_keys(
            subset_path,
            max_rows=None,
            chunk_rows=int(args.feature_read_chunk_rows),
        )
        prediction_manifest_paths = discover_prediction_shard_manifests(args.prediction_shard_root, raw_feature_path)
        shard_index_dir = output_dir / "indexes" / shard_name
        oracle_db_path = shard_index_dir / "oracle_labels_index.sqlite"
        prediction_db_path = shard_index_dir / "prediction_manifest_index.sqlite"
        if reusable_shard_indexes(
            oracle_db_path=oracle_db_path,
            prediction_db_path=prediction_db_path,
            sample_key_count=len(sample_keys),
        ):
            write_status(
                output_dir,
                {
                    "status": "running",
                    "phase": "reuse_shard_indexes",
                    "shard_name": shard_name,
                    "sample_key_count": len(sample_keys),
                },
            )
            append_log(output_dir, f"复用已有 shard-local index：{shard_name} sample_key={len(sample_keys)}")
            prepared.append(
                PreparedShard(
                    shard_name=shard_name,
                    feature_shard_path=subset_path,
                    original_feature_shard_path=raw_feature_path,
                    prediction_manifest_paths=prediction_manifest_paths,
                    sample_keys=sample_keys,
                    oracle_db_path=oracle_db_path,
                    prediction_db_path=prediction_db_path,
                )
            )
            continue
        write_status(
            output_dir,
            {
                "status": "running",
                "phase": "build_oracle_index",
                "shard_name": shard_name,
                "sample_key_count": len(sample_keys),
            },
        )
        oracle_index = build_oracle_sqlite_index(
            args.labels_path,
            sample_keys=sample_keys,
            metric=str(args.metric),
            index_db_path=oracle_db_path,
            parquet_batch_rows=int(args.oracle_parquet_batch_rows),
        )
        oracle_index.close()
        write_status(
            output_dir,
            {
                "status": "running",
                "phase": "build_prediction_index",
                "shard_name": shard_name,
                "sample_key_count": len(sample_keys),
            },
        )
        prediction_index = build_prediction_sqlite_index(
            prediction_manifest_paths,
            sample_keys=sample_keys,
            index_db_path=prediction_db_path,
            chunk_read_rows=int(args.prediction_chunk_rows),
        )
        prediction_index.close()
        prepared.append(
            PreparedShard(
                shard_name=shard_name,
                feature_shard_path=subset_path,
                original_feature_shard_path=raw_feature_path,
                prediction_manifest_paths=prediction_manifest_paths,
                sample_keys=sample_keys,
                oracle_db_path=oracle_db_path,
                prediction_db_path=prediction_db_path,
            )
        )
        append_log(output_dir, f"完成 shard-local index：{shard_name} sample_key={len(sample_keys)}")
    return prepared


def iter_reader_batches(
    prepared_shards: Sequence[PreparedShard],
    *,
    feature_cols: Sequence[str],
    args: argparse.Namespace,
    split: Optional[str],
) -> Iterator[Tuple[str, TimeFuseFusorBatch]]:
    """函数功能：跨 shard 顺序产出 batch，可按 split 过滤。"""
    from visual_router_experiments.stage1_vali_test_router.stage1_timefuse_fusor_streaming_reader import (
        OracleSQLiteIndex,
        PredictionSQLiteIndex,
    )

    for shard in prepared_shards:
        oracle_index = OracleSQLiteIndex(shard.oracle_db_path)
        prediction_index = PredictionSQLiteIndex(shard.prediction_db_path)
        try:
            reader = Stage1TimeFuseFusorStreamingReader(
                feature_shard_path=shard.feature_shard_path,
                oracle_index=oracle_index,
                prediction_index=prediction_index,
                feature_columns=feature_cols,
                batch_size=int(args.batch_size),
                metric=str(args.metric),
                max_rows=None,
                prediction_num_workers=int(args.prediction_num_workers),
                prefetch_batches=int(args.prefetch_batches),
                split_filter=split,
            )
            for batch in reader:
                if split is None:
                    yield shard.shard_name, batch
                    continue
                mask = batch.metadata_df["split"].astype(str).to_numpy() == str(split)
                if not mask.any():
                    continue
                yield shard.shard_name, filter_batch(batch, mask)
        finally:
            oracle_index.close()
            prediction_index.close()


def filter_batch(batch: TimeFuseFusorBatch, mask: np.ndarray) -> TimeFuseFusorBatch:
    """函数功能：按 split mask 过滤 reader batch，同时保持各字段顺序一致。"""
    indices = np.where(mask)[0]
    labels = [batch.labels[int(idx)] for idx in indices]
    sample_keys = [batch.sample_keys[int(idx)] for idx in indices]
    return TimeFuseFusorBatch(
        sample_keys=sample_keys,
        metadata_df=batch.metadata_df.iloc[indices].reset_index(drop=True),
        features=batch.features[indices],
        labels=labels,
        y_pred=batch.y_pred[indices],
        y_true=batch.y_true[indices],
        expert_errors=batch.expert_errors[indices],
    )


def scaler_to_state(scaler: StandardScaler) -> Dict[str, object]:
    """函数功能：将 StandardScaler 转为可 checkpoint 的透明状态。"""
    return {
        "mean_": np.asarray(scaler.mean_, dtype=np.float64),
        "scale_": np.asarray(scaler.scale_, dtype=np.float64),
        "var_": np.asarray(scaler.var_, dtype=np.float64),
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": getattr(scaler, "n_samples_seen_", None),
    }


def scaler_from_state(state: Mapping[str, object]) -> StandardScaler:
    """函数功能：从 checkpoint 状态恢复 StandardScaler。"""
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(state["mean_"], dtype=np.float64)
    scaler.scale_ = np.asarray(state["scale_"], dtype=np.float64)
    scaler.var_ = np.asarray(state["var_"], dtype=np.float64)
    scaler.n_features_in_ = int(state["n_features_in_"])
    n_samples_seen = state.get("n_samples_seen_")
    if n_samples_seen is not None:
        scaler.n_samples_seen_ = np.asarray(n_samples_seen) if isinstance(n_samples_seen, (list, tuple, np.ndarray)) else n_samples_seen
    return scaler


def broadcast_weights(weights: torch.Tensor, prediction_tensor: torch.Tensor) -> torch.Tensor:
    """函数功能：把 `[B, M]` 权重广播到 `[B, M, ...]` 专家预测张量。"""
    return weights.view((weights.shape[0], weights.shape[1], *([1] * (prediction_tensor.ndim - 2))))


def compute_weight_stats(weights_np: np.ndarray) -> Dict[str, float]:
    """函数功能：计算权重熵、归一化熵和最大权重均值。"""
    weights_np = np.asarray(weights_np, dtype=np.float64)
    entropy = -(weights_np * np.log(np.clip(weights_np, EPS, 1.0))).sum(axis=1)
    return {
        "mean_weight_entropy": float(entropy.mean()),
        "mean_normalized_weight_entropy": float((entropy / math.log(len(MODEL_COLUMNS))).mean()),
        "mean_max_weight": float(weights_np.max(axis=1).mean()),
    }


def fit_scaler_streaming(
    *,
    prepared_shards: Sequence[PreparedShard],
    feature_cols: Sequence[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[StandardScaler, Dict[str, object]]:
    """
    函数功能：
        只在 vali feature streaming 上 partial_fit StandardScaler。

    优化说明：
        scaler 只需要 17 维 TimeFuse feature，不需要 oracle label、五专家
        `y_pred/y_true` 或 expert errors。旧实现复用完整 reader，导致 scaler
        阶段反复读取 packed prediction arrays，GPU 训练前就消耗十几小时 I/O。
        这里改为 feature-only CSV streaming，语义仍是只在 vali split 上 fit，
        但避免所有 prediction array 读取。
    """
    scaler = StandardScaler()
    chunk_count = 0
    sample_count = 0
    start = time.perf_counter()
    usecols = ["split", *feature_cols]
    append_log(output_dir, "scaler 使用 feature-only streaming，不读取 oracle/prediction arrays")
    for shard in prepared_shards:
        shard_sample_count = 0
        for chunk_df in pd.read_csv(
            shard.feature_shard_path,
            usecols=usecols,
            chunksize=int(args.feature_read_chunk_rows),
        ):
            vali_df = chunk_df[chunk_df["split"].astype(str) == "vali"]
            if vali_df.empty:
                continue
            features = vali_df[list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
            scaler.partial_fit(features)
            chunk_count += 1
            current_count = int(features.shape[0])
            sample_count += current_count
            shard_sample_count += current_count
            write_status(
                output_dir,
                {
                    "status": "running",
                    "phase": "scaler_partial_fit",
                    "scaler_mode": "feature_only",
                    "current_shard": shard.shard_name,
                    "scaler_chunks": chunk_count,
                    "vali_samples": sample_count,
                    "current_shard_vali_samples": shard_sample_count,
                },
            )
        append_log(output_dir, f"feature-only scaler 完成 shard={shard.shard_name} vali_samples={shard_sample_count}")
    if sample_count == 0:
        raise ValueError("没有读到 vali 样本，无法 fit StandardScaler")
    return scaler, {
        "scaler_mode": "feature_only",
        "scaler_batches": int(chunk_count),
        "scaler_chunks": int(chunk_count),
        "scaler_samples": int(sample_count),
        "elapsed_seconds": float(time.perf_counter() - start),
    }


def unwrap_fusor(fusor: nn.Module) -> TimeFuseFusor:
    """
    函数功能：
        取出可能被 DataParallel 包裹的 TimeFuse fusor 本体。

    说明：
        正式公平比较要求 GPU2/GPU3 双卡训练时，训练过程会用
        `nn.DataParallel` 包裹模型；checkpoint 仍保存未包裹模型的
        state_dict，避免后续 CPU/eval-only 加载时出现 `module.` 前缀不兼容。
    """
    if isinstance(fusor, nn.DataParallel):
        return fusor.module  # type: ignore[return-value]
    return fusor  # type: ignore[return-value]


def maybe_wrap_data_parallel(fusor: TimeFuseFusor, device: torch.device, output_dir: Path) -> nn.Module:
    """
    函数功能：
        在 CUDA 双卡可见时启用 DataParallel。

    关键约束：
        `resolve_device()` 已经限制物理 GPU 只能是 2/3 或其单卡子集。这里仅在
        当前进程实际可见 GPU 数量大于 1 时包裹模型，从而满足正式 fusor 训练
        使用 GPU2/GPU3 双卡的公平性要求；CPU 和单卡 smoke 不受影响。
    """
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        append_log(output_dir, f"启用 DataParallel 双卡训练 logical_devices={list(range(torch.cuda.device_count()))}")
        return nn.DataParallel(fusor)
    return fusor


def train_streaming(
    *,
    fusor: nn.Module,
    scaler: StandardScaler,
    prepared_shards: Sequence[PreparedShard],
    feature_cols: Sequence[str],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    start_epoch: int,
) -> List[Dict[str, object]]:
    """函数功能：按 vali batch 读取 packed predictions 并训练 fusor。"""
    optimizer = torch.optim.Adam(fusor.parameters(), lr=float(args.lr))
    criterion = nn.SmoothL1Loss(beta=float(args.huber_beta))
    epoch_summaries: List[Dict[str, object]] = []
    for epoch in range(int(start_epoch) + 1, int(args.epochs) + 1):
        fusor.train()
        losses: List[float] = []
        sample_count = 0
        batch_count = 0
        epoch_start = time.perf_counter()
        for shard_name, batch in iter_reader_batches(prepared_shards, feature_cols=feature_cols, args=args, split="vali"):
            x_np = scaler.transform(batch.features).astype(np.float32)
            batch_x = torch.from_numpy(x_np).to(device=device)
            batch_pred = torch.from_numpy(batch.y_pred).to(device=device)
            batch_true = torch.from_numpy(batch.y_true).to(device=device)
            optimizer.zero_grad(set_to_none=True)
            weights = fusor(batch_x)
            fused = (broadcast_weights(weights, batch_pred) * batch_pred).sum(dim=1)
            loss = criterion(fused, batch_true)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            sample_count += int(batch_x.shape[0])
            batch_count += 1
            if batch_count == 1 or batch_count % int(args.status_update_interval) == 0:
                write_status(
                    output_dir,
                    {
                        "status": "running",
                        "phase": "train",
                        "epoch": int(epoch),
                        "current_shard": shard_name,
                        "train_batches": int(batch_count),
                        "train_samples": int(sample_count),
                        "latest_loss": float(losses[-1]),
                    },
                )
        if not losses:
            raise ValueError("没有读到 vali batch，训练无法进行")
        summary = {
            "epoch": int(epoch),
            "train_batches": int(batch_count),
            "train_samples": int(sample_count),
            "mean_loss": float(np.mean(losses)),
            "last_loss": float(losses[-1]),
            "elapsed_seconds": float(time.perf_counter() - epoch_start),
        }
        epoch_summaries.append(summary)
        append_log(output_dir, f"epoch={epoch} 完成 train_samples={sample_count} mean_loss={summary['mean_loss']:.6f}")
        save_checkpoint(
            output_dir=output_dir,
            fusor=fusor,
            scaler=scaler,
            args=args,
            feature_cols=feature_cols,
            prepared_shards=prepared_shards,
            completed_epoch=epoch,
            epoch_summaries=epoch_summaries,
        )
    return epoch_summaries


def array_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """函数功能：基于数组计算单样本 MAE/MSE。"""
    diff = np.asarray(y_pred, dtype=np.float32) - np.asarray(y_true, dtype=np.float32)
    return {"mae": float(np.mean(np.abs(diff))), "mse": float(np.mean(diff ** 2))}


def verify_evaluation_adapter_batch(
    *,
    batch: TimeFuseFusorBatch,
    weights_np: np.ndarray,
    selected_indices: np.ndarray,
    entropy: np.ndarray,
    max_weight: np.ndarray,
    shard_name: str,
    batch_index: int,
    atol: float = 1e-5,
) -> None:
    """
    函数功能：
        在正式 evaluation batch 内构造 EvaluationInput，并用 EvaluationInputAdapter
        复算 hard/raw-soft 指标，确认与当前手写 CSV 逻辑一致。

    输入：
        batch: 当前 reader 已经读出的 test batch，不额外回读 prediction cache。
        weights_np: torch fusor 当前 batch 输出的权重矩阵。
        selected_indices/entropy/max_weight: 现有手写评估逻辑已计算的诊断数组。
        shard_name/batch_index: 失败时用于定位的上下文。

    输出：
        无返回值；发现不一致时抛出携带 shard、batch 和 sample_key 的错误。

    关键约束：
        该函数只做内存旁路校验，不写任何 CSV/summary/checkpoint/status/metadata，
        不改变正式 evaluation 输出 schema。
    """
    evaluation_input = EvaluationInput(
        sample_keys=tuple(str(sample_key) for sample_key in batch.sample_keys),
        model_columns=tuple(MODEL_COLUMNS),
        y_pred=batch.y_pred,
        y_true=batch.y_true,
        weights=weights_np,
        extra={
            "source": "train_timefuse_fusor_streaming.evaluate_streaming",
            "shard_name": str(shard_name),
            "batch_index": int(batch_index),
        },
    )
    try:
        result = EvaluationInputAdapter().evaluate_input(evaluation_input=evaluation_input)
    except Exception as exc:
        preview_keys = list(evaluation_input.sample_keys[:5])
        raise RuntimeError(
            "EvaluationInputAdapter 复算失败："
            f"shard={shard_name} batch={batch_index} sample_key_preview={preview_keys}"
        ) from exc

    if len(result.per_sample_rows) != len(batch.sample_keys):
        raise AssertionError(
            "EvaluationInputAdapter rows 数量不一致："
            f"shard={shard_name} batch={batch_index} "
            f"adapter_rows={len(result.per_sample_rows)} expected={len(batch.sample_keys)}"
        )

    for row_idx, adapter_row in enumerate(result.per_sample_rows):
        sample_key = str(batch.sample_keys[row_idx])
        if adapter_row["sample_key"] != sample_key:
            raise AssertionError(
                "EvaluationInputAdapter sample_key 顺序不一致："
                f"shard={shard_name} batch={batch_index} row={row_idx} "
                f"adapter={adapter_row['sample_key']} expected={sample_key}"
            )

        label = batch.labels[row_idx]
        hard_pred = batch.y_pred[row_idx, int(selected_indices[row_idx])]
        soft_pred = np.sum(
            batch.y_pred[row_idx] * weights_np[row_idx].reshape((-1, *([1] * (batch.y_pred.ndim - 2)))),
            axis=0,
        )
        hard_metrics = array_metrics(batch.y_true[row_idx], hard_pred)
        soft_metrics = array_metrics(batch.y_true[row_idx], soft_pred)
        comparisons = {
            "selected_index": (int(adapter_row["selected_index"]), int(selected_indices[row_idx])),
            "hard_mae": (float(adapter_row["hard_mae"]), hard_metrics["mae"]),
            "hard_mse": (float(adapter_row["hard_mse"]), hard_metrics["mse"]),
            "raw_soft_mae": (float(adapter_row["raw_soft_mae"]), soft_metrics["mae"]),
            "raw_soft_mse": (float(adapter_row["raw_soft_mse"]), soft_metrics["mse"]),
            "max_weight": (float(adapter_row["max_weight"]), float(max_weight[row_idx])),
            "weight_entropy": (float(adapter_row["weight_entropy"]), float(entropy[row_idx])),
        }
        for metric_name, (adapter_value, manual_value) in comparisons.items():
            if metric_name == "selected_index":
                if adapter_value != manual_value:
                    raise AssertionError(
                        "EvaluationInputAdapter selected_index 不一致："
                        f"shard={shard_name} batch={batch_index} row={row_idx} sample_key={sample_key} "
                        f"config_name={label['config_name']} adapter={adapter_value} manual={manual_value}"
                    )
                continue
            if not np.isclose(adapter_value, manual_value, rtol=1e-5, atol=atol):
                raise AssertionError(
                    "EvaluationInputAdapter 指标不一致："
                    f"shard={shard_name} batch={batch_index} row={row_idx} sample_key={sample_key} "
                    f"config_name={label['config_name']} metric={metric_name} "
                    f"adapter={adapter_value} manual={manual_value}"
                )


def evaluate_streaming(
    *,
    fusor: nn.Module,
    scaler: StandardScaler,
    prepared_shards: Sequence[PreparedShard],
    feature_cols: Sequence[str],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """函数功能：对 test split 流式输出 hard top-1 与 raw soft fusion 指标。"""
    fusor.eval()
    prediction_path = output_dir / "timefuse_fusor_predictions.csv"
    sample_path = output_dir / "sample_predictions.csv"
    fieldnames = [
        "router_name",
        "config_name",
        "sample_key",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "selected_model",
        "selected_value",
        "oracle_model",
        "oracle_value",
        "regret_to_oracle",
        "oracle_label_correct",
        "weight_entropy",
        "normalized_weight_entropy",
        "max_weight",
        *[f"weight_{model_name}" for model_name in MODEL_COLUMNS],
        "soft_fusion_mae",
        "soft_fusion_mse",
        "hard_top1_mae_from_array",
        "hard_top1_mse_from_array",
    ]
    rows_for_summary: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    batch_count = 0
    sample_count = 0
    start = time.perf_counter()
    with prediction_path.open("w", encoding="utf-8", newline="") as pred_f:
        writer = csv.DictWriter(pred_f, fieldnames=fieldnames)
        writer.writeheader()
        with torch.inference_mode():
            for shard_name, batch in iter_reader_batches(prepared_shards, feature_cols=feature_cols, args=args, split="test"):
                x_np = scaler.transform(batch.features).astype(np.float32)
                weights_np = fusor(torch.from_numpy(x_np).to(device=device)).detach().cpu().numpy()
                selected_indices = weights_np.argmax(axis=1)
                entropy = -(weights_np * np.log(np.clip(weights_np, EPS, 1.0))).sum(axis=1)
                normalized_entropy = entropy / math.log(len(MODEL_COLUMNS))
                max_weight = weights_np.max(axis=1)
                current_batch_index = batch_count + 1
                if bool(getattr(args, "verify_evaluation_adapter", False)):
                    verify_evaluation_adapter_batch(
                        batch=batch,
                        weights_np=weights_np,
                        selected_indices=selected_indices,
                        entropy=entropy,
                        max_weight=max_weight,
                        shard_name=shard_name,
                        batch_index=current_batch_index,
                    )
                for row_idx, sample_key in enumerate(batch.sample_keys):
                    label = batch.labels[row_idx]
                    selected_model = MODEL_COLUMNS[int(selected_indices[row_idx])]
                    selected_value = float(label[selected_model])
                    soft_pred = np.sum(batch.y_pred[row_idx] * weights_np[row_idx].reshape((-1, *([1] * (batch.y_pred.ndim - 2)))), axis=0)
                    hard_pred = batch.y_pred[row_idx, int(selected_indices[row_idx])]
                    soft_metrics = array_metrics(batch.y_true[row_idx], soft_pred)
                    hard_metrics = array_metrics(batch.y_true[row_idx], hard_pred)
                    output_row: Dict[str, object] = {
                        "router_name": "timefuse_style_fusor_streaming",
                        "config_name": label["config_name"],
                        "sample_key": sample_key,
                        "split": label["split"],
                        "dataset_name": label["dataset_name"],
                        "item_id": int(label["item_id"]),
                        "channel_id": int(label["channel_id"]),
                        "window_index": int(label["window_index"]),
                        "selected_model": selected_model,
                        "selected_value": selected_value,
                        "oracle_model": label["oracle_model"],
                        "oracle_value": float(label["oracle_value"]),
                        "regret_to_oracle": float(selected_value - float(label["oracle_value"])),
                        "oracle_label_correct": bool(selected_model == label["oracle_model"]),
                        "weight_entropy": float(entropy[row_idx]),
                        "normalized_weight_entropy": float(normalized_entropy[row_idx]),
                        "max_weight": float(max_weight[row_idx]),
                        "soft_fusion_mae": soft_metrics["mae"],
                        "soft_fusion_mse": soft_metrics["mse"],
                        "hard_top1_mae_from_array": hard_metrics["mae"],
                        "hard_top1_mse_from_array": hard_metrics["mse"],
                    }
                    for model_idx, model_name in enumerate(MODEL_COLUMNS):
                        output_row[f"weight_{model_name}"] = float(weights_np[row_idx, model_idx])
                    writer.writerow(output_row)
                    rows_for_summary.append(output_row)
                    if len(sample_rows) < int(args.sample_prediction_limit):
                        sample_rows.append(output_row)
                sample_count += len(batch.sample_keys)
                batch_count += 1
                if batch_count == 1 or batch_count % int(args.status_update_interval) == 0:
                    write_status(
                        output_dir,
                        {
                            "status": "running",
                            "phase": "test_eval",
                            "current_shard": shard_name,
                            "test_batches": int(batch_count),
                            "test_samples": int(sample_count),
                        },
                    )
    if sample_count == 0:
        raise ValueError("没有读到 test 样本，无法评估")
    pred_df = pd.DataFrame(rows_for_summary)
    hard_summary_df = summarize_hard_predictions(pred_df)
    soft_summary_df = summarize_soft_fusion(pred_df)
    selected_counts_df = summarize_selected_model_counts(pred_df)
    hard_summary_df.to_csv(output_dir / "timefuse_fusor_summary.csv", index=False)
    soft_summary_df.to_csv(output_dir / "timefuse_fusor_raw_soft_fusion_summary.csv", index=False)
    selected_counts_df.to_csv(output_dir / "timefuse_fusor_selected_model_counts.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(sample_path, index=False)
    append_log(output_dir, f"test eval 完成 test_samples={sample_count} elapsed={time.perf_counter() - start:.2f}s")
    return hard_summary_df, soft_summary_df, selected_counts_df


def checkpoint_payload(
    *,
    fusor: nn.Module,
    scaler: StandardScaler,
    args: argparse.Namespace,
    feature_cols: Sequence[str],
    prepared_shards: Sequence[PreparedShard],
    completed_epoch: int,
    epoch_summaries: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """函数功能：组装 checkpoint payload，记录训练参数和已完成 epoch/shard 信息。"""
    return {
        "checkpoint_version": FUSOR_VERSION,
        "saved_at": display_time(),
        "fusor_state_dict": unwrap_fusor(fusor).state_dict(),
        "scaler_state": scaler_to_state(scaler),
        "completed_epoch": int(completed_epoch),
        "completed_shards": [shard.shard_name for shard in prepared_shards],
        "model_columns": list(MODEL_COLUMNS),
        "feature_columns": list(feature_cols),
        "train_args": vars(args),
        "epoch_summaries": [dict(row) for row in epoch_summaries],
    }


def save_checkpoint(
    *,
    output_dir: Path,
    fusor: nn.Module,
    scaler: StandardScaler,
    args: argparse.Namespace,
    feature_cols: Sequence[str],
    prepared_shards: Sequence[PreparedShard],
    completed_epoch: int,
    epoch_summaries: Sequence[Mapping[str, object]],
) -> Path:
    """函数功能：保存 epoch checkpoint 和 latest checkpoint。"""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        fusor=fusor,
        scaler=scaler,
        args=args,
        feature_cols=feature_cols,
        prepared_shards=prepared_shards,
        completed_epoch=completed_epoch,
        epoch_summaries=epoch_summaries,
    )
    epoch_path = checkpoint_dir / f"timefuse_fusor_epoch_{int(completed_epoch):04d}.pt"
    latest_path = checkpoint_dir / "latest_timefuse_fusor.pt"
    for path in [epoch_path, latest_path]:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    write_json_atomic(
        checkpoint_dir / "latest_checkpoint_index.json",
        {
            "checkpoint_path": str(epoch_path),
            "latest_checkpoint_path": str(latest_path),
            "completed_epoch": int(completed_epoch),
            "updated_at": display_time(),
        },
    )
    return latest_path


def load_checkpoint(path: Path, *, device: torch.device) -> Tuple[TimeFuseFusor, StandardScaler, Dict[str, object]]:
    """函数功能：加载 fusor/scaler checkpoint，用于 eval-only 或复验。"""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    feature_cols = list(payload["feature_columns"])
    fusor = TimeFuseFusor(input_dim=len(feature_cols), output_dim=len(MODEL_COLUMNS)).to(device)
    fusor.load_state_dict(payload["fusor_state_dict"])
    scaler = scaler_from_state(payload["scaler_state"])
    return fusor, scaler, payload


def write_markdown_summary(
    output_dir: Path,
    *,
    hard_summary_df: Optional[pd.DataFrame],
    soft_summary_df: Optional[pd.DataFrame],
    selected_counts_df: Optional[pd.DataFrame],
    metadata: Mapping[str, object],
) -> None:
    """函数功能：写中文 Markdown 汇总，便于人工快速复核。"""
    lines = [
        "# Stage 1 TimeFuse-style Fusor Streaming Summary",
        "",
        f"- 生成时间：{display_time()}",
        f"- 输出目录：`{output_dir}`",
        f"- 版本：`{FUSOR_VERSION}`",
        f"- 设备：`{metadata.get('device')}`",
        f"- feature shards：{metadata.get('feature_shard_count')}",
        f"- 训练样本：{metadata.get('train_samples')}",
        f"- 测试样本：{metadata.get('test_samples')}",
        "",
        "## Hard Top-1 Summary",
        "",
        frame_to_markdown(hard_summary_df if hard_summary_df is not None else pd.DataFrame()),
        "",
        "## Raw Soft Fusion Summary",
        "",
        frame_to_markdown(soft_summary_df if soft_summary_df is not None else pd.DataFrame()),
        "",
        "## Selected Model Counts",
        "",
        frame_to_markdown(selected_counts_df if selected_counts_df is not None else pd.DataFrame()),
        "",
        "## 资源与读取口径",
        "",
        "- feature CSV 按 batch streaming；",
        "- oracle parquet 和 prediction manifest 只为当前 1-2 个 feature shard 建 shard-local SQLite；",
        "- 训练和 eval 均按 batch 读取 packed `y_pred/y_true`，不常驻全量 prediction；",
        "- `StandardScaler` 只在 vali feature streaming 上 `partial_fit`，test 只 transform/eval。",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """函数功能：执行 1-2 shard streaming fusor 训练与 test eval 闭环。"""
    args = parse_args()
    if args.prefetch_batches not in {0, 1}:
        raise ValueError("--prefetch-batches 当前只支持 0 或 1")
    if args.eval_only and args.resume_checkpoint is None:
        raise ValueError("--eval-only 必须配合 --resume-checkpoint")
    if args.train_only and args.eval_only:
        raise ValueError("--train-only 与 --eval-only 不能同时使用")
    set_seed(int(args.seed))
    output_dir = args.output_dir or (args.output_root / f"{now_token()}_stage1_timefuse_fusor_streaming")
    output_dir.mkdir(parents=True, exist_ok=True)
    append_log(output_dir, "启动 Stage 1 TimeFuse-style fusor streaming 训练/eval")
    device = resolve_device(str(args.device))
    feature_shards = args.feature_shard_path or default_feature_shards(args.feature_shard_root)
    feature_cols = infer_feature_columns(Path(feature_shards[0]))
    prepared_shards = prepare_shards(args, output_dir, feature_cols)

    start_epoch = 0
    epoch_summaries: List[Dict[str, object]] = []
    scaler_info: Dict[str, object] = {}
    if args.resume_checkpoint is not None:
        base_fusor, scaler, checkpoint = load_checkpoint(args.resume_checkpoint, device=device)
        fusor = maybe_wrap_data_parallel(base_fusor, device, output_dir)
        start_epoch = int(checkpoint.get("completed_epoch", 0))
        epoch_summaries = [dict(row) for row in checkpoint.get("epoch_summaries", [])]
        append_log(output_dir, f"已加载 checkpoint={args.resume_checkpoint} completed_epoch={start_epoch}")
    else:
        scaler, scaler_info = fit_scaler_streaming(
            prepared_shards=prepared_shards,
            feature_cols=feature_cols,
            args=args,
            output_dir=output_dir,
        )
        base_fusor = TimeFuseFusor(input_dim=len(feature_cols), output_dim=len(MODEL_COLUMNS)).to(device)
        fusor = maybe_wrap_data_parallel(base_fusor, device, output_dir)

    if not args.eval_only and int(args.epochs) > start_epoch:
        new_epoch_summaries = train_streaming(
            fusor=fusor,
            scaler=scaler,
            prepared_shards=prepared_shards,
            feature_cols=feature_cols,
            args=args,
            output_dir=output_dir,
            device=device,
            start_epoch=start_epoch,
        )
        epoch_summaries.extend(new_epoch_summaries)
    elif not args.eval_only:
        save_checkpoint(
            output_dir=output_dir,
            fusor=fusor,
            scaler=scaler,
            args=args,
            feature_cols=feature_cols,
            prepared_shards=prepared_shards,
            completed_epoch=start_epoch,
            epoch_summaries=epoch_summaries,
        )

    hard_summary_df: Optional[pd.DataFrame] = None
    soft_summary_df: Optional[pd.DataFrame] = None
    selected_counts_df: Optional[pd.DataFrame] = None
    if not args.train_only:
        hard_summary_df, soft_summary_df, selected_counts_df = evaluate_streaming(
            fusor=fusor,
            scaler=scaler,
            prepared_shards=prepared_shards,
            feature_cols=feature_cols,
            args=args,
            output_dir=output_dir,
            device=device,
        )

    train_samples = int(sum(row.get("train_samples", 0) for row in epoch_summaries[-1:]))
    test_samples = 0 if hard_summary_df is None else int(hard_summary_df["sample_count"].sum())
    active_checkpoint_path = (
        str(args.resume_checkpoint)
        if args.eval_only and args.resume_checkpoint is not None
        else str(output_dir / "checkpoints" / "latest_timefuse_fusor.pt")
    )
    metadata = {
        "status": "completed",
        "generated_at": display_time(),
        "version": FUSOR_VERSION,
        "device": str(device),
        "data_parallel": bool(isinstance(fusor, nn.DataParallel)),
        "data_parallel_device_count": int(torch.cuda.device_count()) if device.type == "cuda" else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "metric": str(args.metric),
        "model_columns": list(MODEL_COLUMNS),
        "feature_columns": list(feature_cols),
        "feature_dim": int(len(feature_cols)),
        "feature_shard_count": int(len(prepared_shards)),
        "feature_shards": [
            {
                "shard_name": shard.shard_name,
                "feature_shard_path": str(shard.feature_shard_path),
                "original_feature_shard_path": str(shard.original_feature_shard_path),
                "sample_key_count": int(len(shard.sample_keys)),
                "prediction_manifest_paths": [str(path) for path in shard.prediction_manifest_paths],
                "oracle_db_path": str(shard.oracle_db_path),
                "prediction_db_path": str(shard.prediction_db_path),
            }
            for shard in prepared_shards
        ],
        "args": vars(args),
        "scaler_info": scaler_info,
        "epoch_summaries": epoch_summaries,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "checkpoint_path": active_checkpoint_path,
        "resources_final": collect_resource_snapshot(),
    }
    write_json_atomic(output_dir / "metadata.json", metadata)
    write_markdown_summary(
        output_dir,
        hard_summary_df=hard_summary_df,
        soft_summary_df=soft_summary_df,
        selected_counts_df=selected_counts_df,
        metadata=metadata,
    )
    write_status(
        output_dir,
        {
            "status": "completed",
            "phase": "done",
            "metadata_path": str(output_dir / "metadata.json"),
            "summary_path": str(output_dir / "summary.md"),
            "checkpoint_path": active_checkpoint_path,
            "train_samples": train_samples,
            "test_samples": test_samples,
        },
    )
    append_log(output_dir, "Stage 1 TimeFuse-style fusor streaming 训练/eval 完成")
    print(json.dumps(to_jsonable(metadata), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
