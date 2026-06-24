#!/usr/bin/env python3
"""
文件功能：
    在 Visual Router V2 P0/275k pilot 样本上，对比当前优化的
    `spatial_panel_3view`、TimeFuse-style fusor baseline 和非视觉/统计基线。

输出内容：
    - full 275k overall 对比；
    - 按 sample_set、TSF cell、sample_set+TSF cell、dataset+TSF cell 分层；
    - 统计规则 baseline 在 pilot_train 上学习到的 expert policy；
    - 中文 Markdown 摘要和 metadata/status。

关键约束：
    - 只读复用既有 P0 feature cache、TimeFuse full-scale feature cache/checkpoint
      和 P0 subset SQLite prediction index；
    - 不训练新模型，不读取 test 作为 policy 选择依据；
    - soft fusion MAE/MSE 从五专家预测数组逐 batch 复算，不用专家 MAE 简单加权；
    - `pilot_train` 上的统计规则 baseline 为 in-sample 诊断，不能当 frozen test 结论。
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.stage1_vali_test_router.fusion_utils import MODEL_COLUMNS, TimeFuseFusor, frame_to_markdown  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.stage1_timefuse_fusor_streaming_reader import FEATURE_COLUMNS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import SQLitePredictionIndex, scaler_from_state  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_v2_round1_film import FiLMRouter  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_v2_round2_layout_film import load_layout_features  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_round1_training import load_prediction_batch_from_index  # noqa: E402


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
P0_RUN_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline"
FULL_SCALE_ROOT = DATA2_RUN_OUTPUT_ROOT / "2026-06-15_stage1_96_48_s_full_scale"
TIMEFUSE_RUN_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-18_stage1_timefuse_fusor_full_scale_gpu23"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-24_p0_275k_spatial_timefuse_statistical_comparison"
MERGED_CACHE_DIR = FULL_SCALE_ROOT / "prediction_cache_full_scale_launcher" / "merged_cache"
TIMEFUSE_FEATURE_ROOT = FULL_SCALE_ROOT / "timefuse_feature_cache_full_scale_launcher" / "shards"

SAMPLE_SETS = ("pilot_train", "pilot_selection", "diagnostic_balanced", "pilot_test")
TSF_COLUMNS = [
    "dataset_name",
    "cluster",
    "group_name",
    "forecastability_cat",
    "season_strength_cat",
    "trend_strength_cat",
    "cv_cat",
    "missing_ratio_cat",
]


def display_time() -> str:
    """函数功能：生成写入 status/metadata/summary 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_status(output_dir: Path, status: str, **extra: object) -> None:
    """函数功能：写出可监控状态。"""
    payload = {"status": status, "updated_at": display_time(), **extra}
    write_json(output_dir / "status.json", payload)


def parse_args() -> argparse.Namespace:
    """函数功能：解析 P0/275k 对比参数。"""
    parser = argparse.ArgumentParser(description="Compare P0 275k spatial_panel_3view, TimeFuse and statistical baselines.")
    parser.add_argument("--p0-run-dir", type=Path, default=P0_RUN_DIR)
    parser.add_argument("--timefuse-run-dir", type=Path, default=TIMEFUSE_RUN_DIR)
    parser.add_argument("--timefuse-feature-root", type=Path, default=TIMEFUSE_FEATURE_ROOT)
    parser.add_argument("--merged-cache-dir", type=Path, default=MERGED_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--feature-chunk-rows", type=int, default=200_000)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@dataclass
class MetricAccumulator:
    """类功能：以流式方式累加某个 method/group 的指标均值。"""

    count: int = 0
    mae_sum: float = 0.0
    mse_sum: float = 0.0
    regret_sum: float = 0.0
    oracle_correct_sum: float = 0.0
    entropy_sum: float = 0.0
    normalized_entropy_sum: float = 0.0
    max_weight_sum: float = 0.0
    entropy_count: int = 0
    selected_counts: MutableMapping[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(
        self,
        *,
        mae: np.ndarray,
        mse: np.ndarray,
        oracle_mae: np.ndarray,
        selected_model: Sequence[str],
        oracle_model: Sequence[str],
        entropy: np.ndarray | None = None,
        normalized_entropy: np.ndarray | None = None,
        max_weight: np.ndarray | None = None,
    ) -> None:
        """函数功能：追加一批逐样本指标。"""
        mae_arr = np.asarray(mae, dtype=np.float64)
        mse_arr = np.asarray(mse, dtype=np.float64)
        oracle_arr = np.asarray(oracle_mae, dtype=np.float64)
        self.count += int(mae_arr.size)
        self.mae_sum += float(mae_arr.sum())
        self.mse_sum += float(mse_arr.sum())
        self.regret_sum += float((mae_arr - oracle_arr).sum())
        selected = [str(x) for x in selected_model]
        oracle = [str(x) for x in oracle_model]
        self.oracle_correct_sum += float(sum(1 for a, b in zip(selected, oracle) if a == b))
        for model_name in selected:
            self.selected_counts[model_name] += 1
        if entropy is not None and normalized_entropy is not None and max_weight is not None:
            ent = np.asarray(entropy, dtype=np.float64)
            norm_ent = np.asarray(normalized_entropy, dtype=np.float64)
            max_w = np.asarray(max_weight, dtype=np.float64)
            self.entropy_sum += float(ent.sum())
            self.normalized_entropy_sum += float(norm_ent.sum())
            self.max_weight_sum += float(max_w.sum())
            self.entropy_count += int(ent.size)

    def to_row(self, key: Mapping[str, object]) -> Dict[str, object]:
        """函数功能：导出一行 summary。"""
        if self.count <= 0:
            raise ValueError("空 accumulator 不能导出")
        row = dict(key)
        row.update(
            {
                "sample_count": int(self.count),
                "MAE": self.mae_sum / self.count,
                "MSE": self.mse_sum / self.count,
                "regret_to_oracle": self.regret_sum / self.count,
                "oracle_label_accuracy": self.oracle_correct_sum / self.count,
            }
        )
        if self.entropy_count:
            row.update(
                {
                    "weight_entropy": self.entropy_sum / self.entropy_count,
                    "normalized_weight_entropy": self.normalized_entropy_sum / self.entropy_count,
                    "mean_max_weight": self.max_weight_sum / self.entropy_count,
                }
            )
        else:
            row.update({"weight_entropy": np.nan, "normalized_weight_entropy": np.nan, "mean_max_weight": np.nan})
        total_selected = sum(self.selected_counts.values())
        for model_name in MODEL_COLUMNS:
            row[f"selected_ratio_{model_name}"] = (
                self.selected_counts.get(model_name, 0) / total_selected if total_selected else np.nan
            )
        return row


class SummaryBook:
    """类功能：维护 overall、sample_set、TSF cell 等多张聚合表。"""

    def __init__(self) -> None:
        self.tables: Dict[str, Dict[Tuple[Tuple[str, object], ...], MetricAccumulator]] = {
            "overall": {},
            "sample_set": {},
            "tsf_cell": {},
            "sample_set_tsf_cell": {},
            "dataset_tsf_cell": {},
        }

    @staticmethod
    def _acc(table: Dict[Tuple[Tuple[str, object], ...], MetricAccumulator], key: Mapping[str, object]) -> MetricAccumulator:
        frozen = tuple(sorted(key.items()))
        if frozen not in table:
            table[frozen] = MetricAccumulator()
        return table[frozen]

    def add_method_batch(
        self,
        *,
        method: str,
        family: str,
        method_kind: str,
        seed: int | str,
        sample_meta: pd.DataFrame,
        mae: np.ndarray,
        mse: np.ndarray,
        oracle_mae: np.ndarray,
        selected_model: Sequence[str],
        oracle_model: Sequence[str],
        entropy: np.ndarray | None = None,
        normalized_entropy: np.ndarray | None = None,
        max_weight: np.ndarray | None = None,
    ) -> None:
        """函数功能：把同一批样本写入所有聚合维度。"""
        base = {"method": method, "family": family, "method_kind": method_kind, "seed": seed}
        self._acc(self.tables["overall"], {**base, "scope": "all_275k"}).add(
            mae=mae,
            mse=mse,
            oracle_mae=oracle_mae,
            selected_model=selected_model,
            oracle_model=oracle_model,
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            max_weight=max_weight,
        )
        for sample_set, idx in sample_meta.groupby("sample_set", sort=False).groups.items():
            mask_idx = np.asarray(list(idx), dtype=np.int64)
            self._acc(self.tables["sample_set"], {**base, "sample_set": str(sample_set)}).add(
                mae=mae[mask_idx],
                mse=mse[mask_idx],
                oracle_mae=oracle_mae[mask_idx],
                selected_model=[selected_model[i] for i in mask_idx],
                oracle_model=[oracle_model[i] for i in mask_idx],
                entropy=entropy[mask_idx] if entropy is not None else None,
                normalized_entropy=normalized_entropy[mask_idx] if normalized_entropy is not None else None,
                max_weight=max_weight[mask_idx] if max_weight is not None else None,
            )
        for group_name, idx in sample_meta.groupby("group_name", sort=False).groups.items():
            mask_idx = np.asarray(list(idx), dtype=np.int64)
            self._acc(self.tables["tsf_cell"], {**base, "group_name": str(group_name)}).add(
                mae=mae[mask_idx],
                mse=mse[mask_idx],
                oracle_mae=oracle_mae[mask_idx],
                selected_model=[selected_model[i] for i in mask_idx],
                oracle_model=[oracle_model[i] for i in mask_idx],
                entropy=entropy[mask_idx] if entropy is not None else None,
                normalized_entropy=normalized_entropy[mask_idx] if normalized_entropy is not None else None,
                max_weight=max_weight[mask_idx] if max_weight is not None else None,
            )
        for (sample_set, group_name), idx in sample_meta.groupby(["sample_set", "group_name"], sort=False).groups.items():
            mask_idx = np.asarray(list(idx), dtype=np.int64)
            self._acc(
                self.tables["sample_set_tsf_cell"],
                {**base, "sample_set": str(sample_set), "group_name": str(group_name)},
            ).add(
                mae=mae[mask_idx],
                mse=mse[mask_idx],
                oracle_mae=oracle_mae[mask_idx],
                selected_model=[selected_model[i] for i in mask_idx],
                oracle_model=[oracle_model[i] for i in mask_idx],
                entropy=entropy[mask_idx] if entropy is not None else None,
                normalized_entropy=normalized_entropy[mask_idx] if normalized_entropy is not None else None,
                max_weight=max_weight[mask_idx] if max_weight is not None else None,
            )
        for (dataset_name, group_name), idx in sample_meta.groupby(["dataset_name", "group_name"], sort=False).groups.items():
            mask_idx = np.asarray(list(idx), dtype=np.int64)
            self._acc(
                self.tables["dataset_tsf_cell"],
                {**base, "dataset_name": str(dataset_name), "group_name": str(group_name)},
            ).add(
                mae=mae[mask_idx],
                mse=mse[mask_idx],
                oracle_mae=oracle_mae[mask_idx],
                selected_model=[selected_model[i] for i in mask_idx],
                oracle_model=[oracle_model[i] for i in mask_idx],
                entropy=entropy[mask_idx] if entropy is not None else None,
                normalized_entropy=normalized_entropy[mask_idx] if normalized_entropy is not None else None,
                max_weight=max_weight[mask_idx] if max_weight is not None else None,
            )

    def to_frame(self, table_name: str) -> pd.DataFrame:
        """函数功能：导出指定聚合表。"""
        rows: List[Dict[str, object]] = []
        for frozen_key, acc in self.tables[table_name].items():
            rows.append(acc.to_row(dict(frozen_key)))
        if not rows:
            return pd.DataFrame()
        sort_cols = [col for col in ["sample_set", "group_name", "dataset_name", "family", "method", "seed"] if col in rows[0]]
        return pd.DataFrame(rows).sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def resolve_device(device_arg: str) -> torch.device:
    """函数功能：解析推理设备。"""
    if device_arg == "cuda" or (device_arg == "auto" and torch.cuda.is_available()):
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_p0_manifest(path: Path) -> pd.DataFrame:
    """函数功能：读取并校验 P0/275k sample manifest。"""
    frame = pd.read_csv(path)
    missing = sorted({"sample_set", "order_index", "sample_key", "split", *TSF_COLUMNS}.difference(frame.columns))
    if missing:
        raise ValueError(f"P0 manifest 缺少字段：{missing}")
    if frame["sample_key"].duplicated().any():
        raise ValueError("P0 manifest sample_key 不应重复")
    ordered_parts: List[pd.DataFrame] = []
    for sample_set in SAMPLE_SETS:
        part = frame[frame["sample_set"].astype(str) == sample_set].sort_values("order_index", kind="mergesort").copy()
        expected = np.arange(len(part), dtype=np.int64)
        actual = part["order_index"].to_numpy(dtype=np.int64, copy=False)
        if not np.array_equal(actual, expected):
            raise ValueError(f"{sample_set} order_index 不连续")
        ordered_parts.append(part)
    return pd.concat(ordered_parts, ignore_index=True)


def checkpoint_scaler_state_to_scaler(state: Mapping[str, object]):
    """函数功能：从 checkpoint 状态恢复 StandardScaler。"""
    return scaler_from_state(state)


def load_spatial_checkpoints(p0_run_dir: Path, device: torch.device) -> Dict[int, Tuple[FiLMRouter, object, object]]:
    """函数功能：加载 spatial_panel_3view 三个 seed 的 FiLM router 和 scaler。"""
    routers: Dict[int, Tuple[FiLMRouter, object, object]] = {}
    for seed in (16, 17, 18):
        path = p0_run_dir / "tasks" / f"spatial_panel_3view_seed{seed}" / f"checkpoint_spatial_panel_3view_seed{seed}.pt"
        checkpoint = torch.load(path, map_location="cpu")
        hp = checkpoint["hyperparameters"]
        router = FiLMRouter(
            visual_dim=768,
            aux_dim=6,
            hidden_dim=int(hp["hidden_dim"]),
            film_hidden_dim=int(hp["film_hidden_dim"]),
            output_dim=len(MODEL_COLUMNS),
            dropout=float(hp["dropout"]),
        )
        router.load_state_dict(checkpoint["router_state_dict"])
        router.to(device=device)
        router.eval()
        routers[seed] = (
            router,
            checkpoint_scaler_state_to_scaler(checkpoint["visual_scaler_state"]),
            checkpoint_scaler_state_to_scaler(checkpoint["aux_scaler_state"]),
        )
    return routers


def load_timefuse_checkpoint(timefuse_run_dir: Path, device: torch.device) -> Tuple[TimeFuseFusor, object, List[str]]:
    """函数功能：加载 full-scale TimeFuse-style fusor checkpoint。"""
    checkpoint_path = timefuse_run_dir / "checkpoints" / "latest_timefuse_fusor.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    feature_columns = list(checkpoint["feature_columns"])
    fusor = TimeFuseFusor(input_dim=len(feature_columns), output_dim=len(MODEL_COLUMNS))
    fusor.load_state_dict(checkpoint["fusor_state_dict"])
    fusor.to(device=device)
    fusor.eval()
    return fusor, checkpoint_scaler_state_to_scaler(checkpoint["scaler_state"]), feature_columns


def find_timefuse_feature_files(root: Path) -> List[Path]:
    """函数功能：发现 full-scale TimeFuse feature shard CSV。"""
    files = sorted(root.glob("sample_shard_*_of_0064/feature_cache.csv"))
    if len(files) != 64:
        raise ValueError(f"期望 64 个 TimeFuse feature shard，实际 {len(files)}：{root}")
    return files


def load_timefuse_feature_subset(
    *,
    feature_files: Sequence[Path],
    manifest: pd.DataFrame,
    feature_columns: Sequence[str],
    chunk_rows: int,
) -> np.ndarray:
    """
    函数功能：
        从 64 个 full-scale TimeFuse feature shard 中抽取 P0/275k sample_key。

    说明：
        只保留 275k 命中行，扫描过程不把 23M 全量 feature 常驻内存。
    """
    key_to_pos = {str(key): idx for idx, key in enumerate(manifest["sample_key"].astype(str).tolist())}
    feature_array = np.full((len(key_to_pos), len(feature_columns)), np.nan, dtype=np.float32)
    hit = np.zeros(len(key_to_pos), dtype=bool)
    usecols = ["sample_key", *feature_columns]
    matched = 0
    for file_idx, path in enumerate(feature_files, start=1):
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunk_rows)):
            mask = chunk["sample_key"].astype(str).isin(key_to_pos)
            if not mask.any():
                continue
            part = chunk.loc[mask].copy()
            positions = part["sample_key"].astype(str).map(key_to_pos).to_numpy(dtype=np.int64)
            values = part[list(feature_columns)].to_numpy(dtype=np.float32)
            feature_array[positions] = values
            hit[positions] = True
            matched += int(len(part))
        if file_idx == 1 or file_idx % 8 == 0:
            print(f"[{display_time()}] TimeFuse feature scan {file_idx}/64 matched={matched}/{len(key_to_pos)}", flush=True)
    if not hit.all():
        missing = manifest.loc[~hit, "sample_key"].head(10).tolist()
        raise ValueError(f"TimeFuse feature subset 缺失 {int((~hit).sum())} 条，示例：{missing}")
    if not np.isfinite(feature_array).all():
        raise ValueError("TimeFuse feature subset 存在 NaN/Inf")
    return feature_array


def load_spatial_features_for_all_sets(p0_run_dir: Path, manifest: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """函数功能：读取 P0 四个 sample_set 的 spatial_panel_3view mean_patch/revin_aux feature。"""
    feature_manifest = p0_run_dir / "round2_p0_spatial_feature_manifest.csv"
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sample_set in SAMPLE_SETS:
        part = manifest[manifest["sample_set"].astype(str) == sample_set].copy()
        visual, aux = load_layout_features(
            feature_manifest_path=feature_manifest,
            sample_df=part,
            sample_set=sample_set,
            layout_name="spatial_panel_3view",
            visual_input_mode="mean_patch",
        )
        out[sample_set] = (visual, aux)
    return out


def compute_router_metrics(
    *,
    weights: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """函数功能：按 router weights 计算 hard/soft 指标和权重诊断。"""
    selected_idx = weights.argmax(axis=1)
    hard_pred = y_pred[np.arange(y_pred.shape[0]), selected_idx]
    # prediction cache 的 y_pred 可能是 [B, M, 48] 或 [B, M, 1, 48]；
    # 权重需要广播到专家维度之后的所有预测维度。
    weight_shape = (weights.shape[0], weights.shape[1], *([1] * (y_pred.ndim - 2)))
    fused = (weights.reshape(weight_shape) * y_pred).sum(axis=1)
    reduce_axes = tuple(range(1, hard_pred.ndim))
    hard_mae = np.mean(np.abs(hard_pred - y_true), axis=reduce_axes)
    hard_mse = np.mean((hard_pred - y_true) ** 2, axis=reduce_axes)
    soft_mae = np.mean(np.abs(fused - y_true), axis=reduce_axes)
    soft_mse = np.mean((fused - y_true) ** 2, axis=reduce_axes)
    selected_model = [MODEL_COLUMNS[int(i)] for i in selected_idx]
    return hard_mae, hard_mse, soft_mae, soft_mse, selected_model


def weight_stats(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """函数功能：计算逐样本 entropy / normalized entropy / max weight。"""
    clipped = np.clip(weights, 1e-12, 1.0)
    entropy = -(weights * np.log(clipped)).sum(axis=1)
    return entropy, entropy / math.log(len(MODEL_COLUMNS)), weights.max(axis=1)


def fit_statistical_policies(manifest: pd.DataFrame, prediction_index: SQLitePredictionIndex, batch_size: int) -> pd.DataFrame:
    """
    函数功能：
        在 pilot_train 上拟合非视觉统计规则 baseline。

    baseline:
        - global_best_single：全 pilot_train 平均 MAE 最低专家；
        - dataset_best_single：每个 dataset_name 平均 MAE 最低专家；
        - tsf_cell_best_single：每个 group_name 平均 MAE 最低专家；
        - dataset_tsf_cell_best_single：每个 (dataset_name, group_name) 平均 MAE 最低专家。
    """
    train = manifest[manifest["sample_set"].astype(str) == "pilot_train"].reset_index(drop=True)
    rows: List[Dict[str, object]] = []
    for start in range(0, len(train), int(batch_size)):
        part = train.iloc[start : start + int(batch_size)].reset_index(drop=True)
        keys = part["sample_key"].astype(str).tolist()
        key_to_meta = {
            str(row.sample_key): (str(row.dataset_name), str(row.group_name))
            for row in part[["sample_key", "dataset_name", "group_name"]].itertuples(index=False)
        }
        placeholders = ",".join(["?"] * len(keys))
        # 统计规则只需要专家 MAE，不需要读取 y_pred/y_true 数组；直接查 SQLite
        # 可以避免在 150k pilot_train 上做一次昂贵的数组 I/O。
        sql_rows = prediction_index.connection.execute(
            f"""
            SELECT sample_key, model_name, mae
            FROM prediction_index
            WHERE sample_key IN ({placeholders})
            """,
            keys,
        ).fetchall()
        if len(sql_rows) != len(keys) * len(MODEL_COLUMNS):
            raise ValueError(f"pilot_train policy SQL 查询不完整：expected={len(keys) * len(MODEL_COLUMNS)} actual={len(sql_rows)}")
        for sql_row in sql_rows:
            sample_key = str(sql_row["sample_key"])
            dataset_name, group_name = key_to_meta[sample_key]
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "group_name": group_name,
                    "model_name": str(sql_row["model_name"]),
                    "mae": float(sql_row["mae"]),
                }
            )
    train_long = pd.DataFrame(rows)
    policies: List[Dict[str, object]] = []

    def best_by(group_cols: Sequence[str], policy_name: str) -> None:
        grouped = train_long.groupby([*group_cols, "model_name"], dropna=False)["mae"].mean().reset_index()
        idx = grouped.groupby(list(group_cols), dropna=False)["mae"].idxmin()
        best = grouped.loc[idx].reset_index(drop=True)
        for row in best.to_dict("records"):
            policies.append({"policy": policy_name, **{col: row[col] for col in group_cols}, "selected_model": row["model_name"], "train_mae": row["mae"]})

    global_means = train_long.groupby("model_name")["mae"].mean().reset_index()
    global_best = global_means.sort_values("mae", kind="mergesort").iloc[0]
    policies.append({"policy": "global_best_single", "selected_model": global_best["model_name"], "train_mae": float(global_best["mae"])})
    best_by(["dataset_name"], "dataset_best_single")
    best_by(["group_name"], "tsf_cell_best_single")
    best_by(["dataset_name", "group_name"], "dataset_tsf_cell_best_single")
    return pd.DataFrame(policies)


def policy_selection(policy_df: pd.DataFrame, policy_name: str, sample_meta: pd.DataFrame) -> List[str]:
    """函数功能：根据统计 policy 为一批样本选择专家。"""
    policy = policy_df[policy_df["policy"].astype(str) == policy_name].copy()
    if policy_name == "global_best_single":
        return [str(policy.iloc[0]["selected_model"])] * len(sample_meta)
    if policy_name == "dataset_best_single":
        mapping = dict(zip(policy["dataset_name"].astype(str), policy["selected_model"].astype(str)))
        return [mapping[str(value)] for value in sample_meta["dataset_name"]]
    if policy_name == "tsf_cell_best_single":
        mapping = dict(zip(policy["group_name"].astype(str), policy["selected_model"].astype(str)))
        return [mapping[str(value)] for value in sample_meta["group_name"]]
    if policy_name == "dataset_tsf_cell_best_single":
        mapping = {
            (str(row.dataset_name), str(row.group_name)): str(row.selected_model)
            for row in policy.itertuples(index=False)
        }
        return [mapping[(str(row.dataset_name), str(row.group_name))] for row in sample_meta.itertuples(index=False)]
    raise ValueError(f"未知统计 policy：{policy_name}")


def add_single_expert_and_policy_methods(
    *,
    book: SummaryBook,
    sample_meta: pd.DataFrame,
    expert_mae: np.ndarray,
    expert_mse: np.ndarray,
    oracle_mae: np.ndarray,
    oracle_model: Sequence[str],
    policy_df: pd.DataFrame,
) -> None:
    """函数功能：写入五专家单模型、oracle 和非视觉统计规则 baseline。"""
    for model_idx, model_name in enumerate(MODEL_COLUMNS):
        family = "statistical_expert" if model_name in {"ES", "NaiveForecaster"} else "deep_expert"
        book.add_method_batch(
            method=f"single_{model_name}",
            family=family,
            method_kind="single_expert",
            seed="na",
            sample_meta=sample_meta,
            mae=expert_mae[:, model_idx],
            mse=expert_mse[:, model_idx],
            oracle_mae=oracle_mae,
            selected_model=[model_name] * len(sample_meta),
            oracle_model=oracle_model,
        )
    oracle_idx = expert_mae.argmin(axis=1)
    book.add_method_batch(
        method="oracle_top1",
        family="oracle",
        method_kind="oracle",
        seed="na",
        sample_meta=sample_meta,
        mae=oracle_mae,
        mse=expert_mse[np.arange(len(sample_meta)), oracle_idx],
        oracle_mae=oracle_mae,
        selected_model=[MODEL_COLUMNS[int(i)] for i in oracle_idx],
        oracle_model=oracle_model,
    )
    for policy_name in ["global_best_single", "dataset_best_single", "tsf_cell_best_single", "dataset_tsf_cell_best_single"]:
        selected = policy_selection(policy_df, policy_name, sample_meta)
        indices = np.asarray([MODEL_COLUMNS.index(model) for model in selected], dtype=np.int64)
        book.add_method_batch(
            method=policy_name,
            family="non_visual_statistical_policy",
            method_kind="policy",
            seed="fit_pilot_train",
            sample_meta=sample_meta,
            mae=expert_mae[np.arange(len(sample_meta)), indices],
            mse=expert_mse[np.arange(len(sample_meta)), indices],
            oracle_mae=oracle_mae,
            selected_model=selected,
            oracle_model=oracle_model,
        )


def write_outputs(output_dir: Path, book: SummaryBook, policy_df: pd.DataFrame, metadata: Mapping[str, object]) -> None:
    """函数功能：写出 CSV/Markdown/metadata。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    table_paths: Dict[str, str] = {}
    for table_name in ["overall", "sample_set", "tsf_cell", "sample_set_tsf_cell", "dataset_tsf_cell"]:
        frame = book.to_frame(table_name)
        path = output_dir / f"p0_275k_{table_name}_comparison.csv"
        frame.to_csv(path, index=False)
        table_paths[table_name] = str(path)
    policy_path = output_dir / "p0_275k_statistical_policy_mapping.csv"
    policy_df.to_csv(policy_path, index=False)
    table_paths["statistical_policy_mapping"] = str(policy_path)

    overall = pd.read_csv(table_paths["overall"])
    selected_methods = [
        "spatial_panel_3view_raw_soft_fusion",
        "timefuse_style_raw_soft_fusion",
        "dataset_tsf_cell_best_single",
        "tsf_cell_best_single",
        "global_best_single",
        "single_ES",
        "single_NaiveForecaster",
        "oracle_top1",
    ]
    summary_view = overall[overall["method"].isin(selected_methods)].copy()
    summary_view = summary_view.sort_values(["MAE", "family", "method"], kind="mergesort")

    lines = [
        "# P0/275k spatial_panel_3view vs TimeFuse vs statistical baseline",
        "",
        f"生成时间：{display_time()}",
        "",
        "## 口径",
        "",
        "- 样本：P0 pilot 协议四个 sample_set，总计 275000 条。",
        "- spatial：`spatial_panel_3view + film_mean_patch_aux`，seeds 16/17/18，报告 seed 级指标；可再对 CSV 做 mean/std。",
        "- TimeFuse：复用 full-scale `latest_timefuse_fusor.pt`，只做 P0 275k subset eval。",
        "- 统计基线：包括 ES / NaiveForecaster 单专家，以及在 `pilot_train` 上拟合的 global/dataset/TSF cell/dataset+TSF cell best-single policy。",
        "- TSF cell：使用 `group_name`；另输出 dataset+TSF cell 表。",
        "",
        "## Full 275k 关键方法",
        "",
        frame_to_markdown(
            summary_view[
                [
                    "family",
                    "method",
                    "method_kind",
                    "seed",
                    "sample_count",
                    "MAE",
                    "MSE",
                    "regret_to_oracle",
                    "oracle_label_accuracy",
                    "weight_entropy",
                    "mean_max_weight",
                ]
            ],
            float_digits=6,
        ),
        "",
        "## 输出文件",
        "",
    ]
    for name, path in table_paths.items():
        lines.append(f"- `{name}`：`{path}`")
    (output_dir / "p0_275k_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_json(output_dir / "p0_275k_comparison_metadata.json", {**metadata, "outputs": table_paths})


def main() -> None:
    """脚本入口：执行 P0/275k 全面对比。"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"输出目录已存在；如需覆盖请传 --overwrite：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_status(output_dir, "started", stage="init")

    device = resolve_device(str(args.device))
    manifest_path = args.p0_run_dir / "inputs" / "p0_sample_manifest.csv"
    manifest = load_p0_manifest(manifest_path)
    if len(manifest) != 275_000:
        raise ValueError(f"P0 manifest 行数异常：{len(manifest)}")

    prediction_index = SQLitePredictionIndex(
        args.p0_run_dir / "prediction_index_round2_layout_subset.sqlite",
        Path(args.merged_cache_dir),
    )
    book = SummaryBook()

    try:
        write_status(output_dir, "running", stage="load_models")
        spatial_models = load_spatial_checkpoints(args.p0_run_dir, device)
        timefuse_model, timefuse_scaler, timefuse_feature_columns = load_timefuse_checkpoint(args.timefuse_run_dir, device)

        write_status(output_dir, "running", stage="fit_statistical_policies")
        policy_df = fit_statistical_policies(manifest, prediction_index, int(args.batch_size))
        policy_df.to_csv(output_dir / "p0_275k_statistical_policy_mapping.partial.csv", index=False)

        write_status(output_dir, "running", stage="load_spatial_features")
        spatial_features = load_spatial_features_for_all_sets(args.p0_run_dir, manifest)

        write_status(output_dir, "running", stage="load_timefuse_features")
        timefuse_features = load_timefuse_feature_subset(
            feature_files=find_timefuse_feature_files(args.timefuse_feature_root),
            manifest=manifest,
            feature_columns=timefuse_feature_columns,
            chunk_rows=int(args.feature_chunk_rows),
        )
        timefuse_features_scaled = timefuse_scaler.transform(timefuse_features).astype(np.float32)

        # spatial 特征按 sample_set 分块加载；这里拼成全 275k 顺序，便于统一 batch eval。
        spatial_visual_all: Dict[int, np.ndarray] = {}
        spatial_aux_all: Dict[int, np.ndarray] = {}
        visual_parts: List[np.ndarray] = []
        aux_parts: List[np.ndarray] = []
        for sample_set in SAMPLE_SETS:
            visual, aux = spatial_features[sample_set]
            visual_parts.append(visual)
            aux_parts.append(aux)
        visual_all = np.concatenate(visual_parts, axis=0).astype(np.float32, copy=False)
        aux_all = np.concatenate(aux_parts, axis=0).astype(np.float32, copy=False)

        write_status(output_dir, "running", stage="evaluate_batches", total_samples=len(manifest))
        sample_keys = manifest["sample_key"].astype(str).tolist()
        for start in range(0, len(manifest), int(args.batch_size)):
            end = min(start + int(args.batch_size), len(manifest))
            sample_meta = manifest.iloc[start:end].reset_index(drop=True)
            batch_keys = sample_keys[start:end]
            y_pred, y_true, _ = load_prediction_batch_from_index(prediction_index, batch_keys, error_metric="mae")
            y_true_for_experts = np.expand_dims(y_true, axis=1)
            reduce_axes = tuple(range(2, y_pred.ndim))
            expert_mae = np.mean(np.abs(y_pred - y_true_for_experts), axis=reduce_axes)
            expert_mse = np.mean((y_pred - y_true_for_experts) ** 2, axis=reduce_axes)
            oracle_idx = expert_mae.argmin(axis=1)
            oracle_mae = expert_mae[np.arange(len(sample_meta)), oracle_idx]
            oracle_model = [MODEL_COLUMNS[int(i)] for i in oracle_idx]

            add_single_expert_and_policy_methods(
                book=book,
                sample_meta=sample_meta,
                expert_mae=expert_mae,
                expert_mse=expert_mse,
                oracle_mae=oracle_mae,
                oracle_model=oracle_model,
                policy_df=policy_df,
            )

            with torch.inference_mode():
                tf_x = torch.from_numpy(timefuse_features_scaled[start:end]).to(device=device)
                tf_weights = timefuse_model(tf_x).detach().cpu().numpy()
            tf_entropy, tf_norm_entropy, tf_max_weight = weight_stats(tf_weights)
            tf_hard_mae, tf_hard_mse, tf_soft_mae, tf_soft_mse, tf_selected = compute_router_metrics(
                weights=tf_weights,
                y_pred=y_pred,
                y_true=y_true,
            )
            book.add_method_batch(
                method="timefuse_style_hard_top1",
                family="timefuse",
                method_kind="hard_top1",
                seed="fullscale_epoch1_seed16",
                sample_meta=sample_meta,
                mae=tf_hard_mae,
                mse=tf_hard_mse,
                oracle_mae=oracle_mae,
                selected_model=tf_selected,
                oracle_model=oracle_model,
                entropy=tf_entropy,
                normalized_entropy=tf_norm_entropy,
                max_weight=tf_max_weight,
            )
            book.add_method_batch(
                method="timefuse_style_raw_soft_fusion",
                family="timefuse",
                method_kind="raw_soft_fusion",
                seed="fullscale_epoch1_seed16",
                sample_meta=sample_meta,
                mae=tf_soft_mae,
                mse=tf_soft_mse,
                oracle_mae=oracle_mae,
                selected_model=tf_selected,
                oracle_model=oracle_model,
                entropy=tf_entropy,
                normalized_entropy=tf_norm_entropy,
                max_weight=tf_max_weight,
            )

            for seed, (router, visual_scaler, aux_scaler) in spatial_models.items():
                visual_scaled = visual_scaler.transform(visual_all[start:end]).astype(np.float32)
                aux_scaled = aux_scaler.transform(aux_all[start:end]).astype(np.float32)
                with torch.inference_mode():
                    logits = router(
                        torch.from_numpy(visual_scaled).to(device=device),
                        torch.from_numpy(aux_scaled).to(device=device),
                    )
                    weights = torch.softmax(logits, dim=1).detach().cpu().numpy()
                entropy, norm_entropy, max_weight = weight_stats(weights)
                hard_mae, hard_mse, soft_mae, soft_mse, selected = compute_router_metrics(
                    weights=weights,
                    y_pred=y_pred,
                    y_true=y_true,
                )
                book.add_method_batch(
                    method="spatial_panel_3view_hard_top1",
                    family="spatial_panel_3view",
                    method_kind="hard_top1",
                    seed=seed,
                    sample_meta=sample_meta,
                    mae=hard_mae,
                    mse=hard_mse,
                    oracle_mae=oracle_mae,
                    selected_model=selected,
                    oracle_model=oracle_model,
                    entropy=entropy,
                    normalized_entropy=norm_entropy,
                    max_weight=max_weight,
                )
                book.add_method_batch(
                    method="spatial_panel_3view_raw_soft_fusion",
                    family="spatial_panel_3view",
                    method_kind="raw_soft_fusion",
                    seed=seed,
                    sample_meta=sample_meta,
                    mae=soft_mae,
                    mse=soft_mse,
                    oracle_mae=oracle_mae,
                    selected_model=selected,
                    oracle_model=oracle_model,
                    entropy=entropy,
                    normalized_entropy=norm_entropy,
                    max_weight=max_weight,
                )
            if start == 0 or end % (int(args.batch_size) * 25) == 0 or end == len(manifest):
                write_status(output_dir, "running", stage="evaluate_batches", processed=end, total_samples=len(manifest))
                print(f"[{display_time()}] evaluated {end}/{len(manifest)}", flush=True)

        write_status(output_dir, "running", stage="write_outputs")
        write_outputs(
            output_dir,
            book,
            policy_df,
            metadata={
                "generated_at": display_time(),
                "script": str(Path(__file__).resolve()),
                "p0_manifest": str(manifest_path),
                "p0_run_dir": str(args.p0_run_dir),
                "timefuse_run_dir": str(args.timefuse_run_dir),
                "timefuse_checkpoint": str(args.timefuse_run_dir / "checkpoints" / "latest_timefuse_fusor.pt"),
                "prediction_index": str(args.p0_run_dir / "prediction_index_round2_layout_subset.sqlite"),
                "merged_cache_dir": str(args.merged_cache_dir),
                "sample_count": int(len(manifest)),
                "sample_sets": {name: int((manifest["sample_set"].astype(str) == name).sum()) for name in SAMPLE_SETS},
                "tsf_cell_column": "group_name",
                "statistical_policy_fit_split": "pilot_train",
                "device": str(device),
            },
        )
        write_status(output_dir, "completed", run_output_dir=str(output_dir))
    finally:
        prediction_index.close()


if __name__ == "__main__":
    main()
