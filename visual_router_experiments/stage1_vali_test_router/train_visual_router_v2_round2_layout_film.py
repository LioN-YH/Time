#!/usr/bin/env python3
"""
文件功能：
    使用固定的 Round1 best `film_mean_patch_aux` 风格后端，对 Round2c 六个
    layout feature cache 做小样本 screening。

实验边界：
    - 本轮变量只能是 layout；
    - visual base 固定为 `mean_patch_embedding`，condition 固定为 `revin_aux`；
    - aux 只通过 FiLM gamma/beta 调制 visual hidden representation，不 concat 到输入；
    - `round2_test_small` 只做 frozen screening，不参与 layout/seed/epoch/hyperparameter 选择。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.common.round2_layout_registry import DEFAULT_ROUND2_LAYOUTS, DEFERRED_ROUND2_LAYOUTS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.evaluate_visual_router_v2_round0 import DEFAULT_ORACLE_LABELS, DEFAULT_PREDICTION_MANIFEST, load_oracle_subset  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.fusion_utils import MODEL_COLUMNS, frame_to_markdown  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import SQLitePredictionIndex, build_lightweight_prediction_index, scaler_to_state  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_v2_round1_concat import normalize_comparison_frame  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_v2_round1_film import (  # noqa: E402
    FiLMRouter,
    build_film_stratified_summary,
    git_commit_hash,
    predict_film_router,
    train_film_router,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_evaluator import TSF_STRATA_COLUMNS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_features import AUX_FEATURE_COLUMNS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_round1_training import (  # noqa: E402
    add_batch_fusion_metrics,
    make_visual_pooling_method_rows,
    resolve_device,
    selected_model_counts_with_variant,
    summarize_mean_std,
    summarize_rows_with_seed,
)


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_SAMPLE_MANIFEST = DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round2_small_samples" / "round2_small_sample_manifest.csv"
DEFAULT_ROUND0_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round0"
DEFAULT_FEATURE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round2_layout_screening"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round2_layout_screening"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "experiment_summaries" / "visual_router_v2_round2" / "layout_screening"
ROUND2_SAMPLE_SETS = (
    "round2_train_small",
    "round2_selection_small",
    "round2_diagnostic_balanced_small",
    "round2_test_small",
)
EVAL_SAMPLE_SETS = (
    "round2_selection_small",
    "round2_diagnostic_balanced_small",
    "round2_test_small",
)
SCRIPT_VERSION = "visual_router_v2_round2c_layout_film_screening_v1"


def display_time() -> str:
    """函数功能：生成日志和 metadata 时间戳。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def log_stage(message: str) -> None:
    """函数功能：输出阶段进度，便于后台监控。"""
    print(f"[{display_time()}] {message}", flush=True)


def parse_csv(text: str) -> List[str]:
    """函数功能：解析逗号分隔参数，并去重保序。"""
    values: List[str] = []
    for part in str(text).split(","):
        value = part.strip()
        if value and value not in values:
            values.append(value)
    if not values:
        raise ValueError("逗号分隔参数不能为空")
    return values


def parse_seed_list(seed_text: str) -> List[int]:
    """函数功能：解析 seed 列表。"""
    return [int(value) for value in parse_csv(seed_text)]


def parse_args() -> argparse.Namespace:
    """函数功能：解析 Round2c layout FiLM screening 参数。"""
    parser = argparse.ArgumentParser(description="Train/aggregate fixed FiLM backend for Round2 layout screening.")
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--round0-dir", type=Path, default=DEFAULT_ROUND0_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-copy-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--oracle-labels-path", type=Path, default=DEFAULT_ORACLE_LABELS)
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST)
    parser.add_argument("--layouts", default=",".join(DEFAULT_ROUND2_LAYOUTS))
    parser.add_argument("--layout", default=None)
    parser.add_argument("--artifact-prefix", default="round2_layout", help="聚合产物文件名前缀；expanded validation 使用 round2_expanded_layout。")
    parser.add_argument("--train-sample-set", default="round2_train_small")
    parser.add_argument("--selection-sample-set", default="round2_selection_small")
    parser.add_argument("--diagnostic-sample-set", default="round2_diagnostic_balanced_small")
    parser.add_argument("--test-sample-set", default="round2_test_small")
    parser.add_argument("--experiment-label", default="Round2c")
    parser.add_argument("--summary-title", default="Visual Router V2 Round2c Layout Screening Summary")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", default="16,17,18")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--film-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--kl-tau", type=float, default=0.1)
    parser.add_argument("--lambda-kl", type=float, default=0.01)
    parser.add_argument("--metric", choices=["mae"], default="mae")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--devices-requested", default="")
    parser.add_argument("--parallel-launcher-used", action="store_true")
    parser.add_argument("--csv-chunksize", type=int, default=200_000)
    parser.add_argument("--parquet-batch-rows", type=int, default=250_000)
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="仅用于 smoke。")
    parser.add_argument("--run-single", action="store_true")
    parser.add_argument("--build-index-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def task_dir(output_dir: Path, layout: str, seed: int) -> Path:
    """函数功能：返回 layout/seed 隔离输出目录。"""
    return Path(output_dir) / "tasks" / f"{layout}_seed{int(seed)}"


def sample_sets_from_args(args: argparse.Namespace) -> Tuple[str, str, str, str]:
    """函数功能：返回本轮训练、选择、诊断和 frozen test 的 sample_set 名称。"""
    return (
        str(args.train_sample_set),
        str(args.selection_sample_set),
        str(args.diagnostic_sample_set),
        str(args.test_sample_set),
    )


def eval_sample_sets_from_args(args: argparse.Namespace) -> Tuple[str, str, str]:
    """函数功能：返回本轮需要评估但不用于训练的 sample_set 名称。"""
    return (str(args.selection_sample_set), str(args.diagnostic_sample_set), str(args.test_sample_set))


def artifact_path(args: argparse.Namespace, suffix: str) -> Path:
    """函数功能：按 artifact prefix 生成统一汇总产物路径。"""
    return Path(args.output_dir) / f"{args.artifact_prefix}_{suffix}"


def read_round2_sample_set(sample_manifest: Path, sample_set: str, *, max_samples: int | None = None) -> pd.DataFrame:
    """函数功能：按 order_index 读取 Round2 small sample_set。"""
    frame = pd.read_csv(sample_manifest)
    part = frame[frame["sample_set"].astype(str) == str(sample_set)].sort_values("order_index", kind="mergesort").reset_index(drop=True)
    if max_samples is not None:
        part = part.head(int(max_samples)).copy()
    if part.empty:
        raise ValueError(f"{sample_set} 样本为空")
    order_index = part["order_index"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(order_index, np.arange(0, len(part), dtype=np.int64)):
        raise ValueError(f"{sample_set} order_index 必须从 0 连续递增")
    if part["sample_key"].astype(str).duplicated().any():
        raise ValueError(f"{sample_set} 存在重复 sample_key")
    return part


def read_all_sample_frames(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    """函数功能：读取训练、选择、诊断、frozen test 四个 Round2 sample set。"""
    return {name: read_round2_sample_set(args.sample_manifest, name, max_samples=args.max_samples_per_set) for name in sample_sets_from_args(args)}


def prediction_index_path(output_dir: Path) -> Path:
    """函数功能：返回本轮共用 prediction subset SQLite 路径。"""
    return Path(output_dir) / "prediction_index_round2_layout_subset.sqlite"


def ensure_round2_prediction_index(args: argparse.Namespace, sample_keys: Sequence[str]) -> SQLitePredictionIndex:
    """
    函数功能：
        获取 Round2c 训练/评估所需 prediction SQLite index。

    说明：
        launcher 会先单进程预构建该 index；worker 只复用已存在文件，避免并行扫描
        116M prediction manifest 或竞争写同一个 SQLite。
    """
    index_path = prediction_index_path(args.output_dir)
    if index_path.exists():
        return SQLitePredictionIndex(index_path, args.prediction_manifest_path.parent)
    if not args.build_index_only and args.run_single:
        raise FileNotFoundError(f"Round2c prediction index 尚未构建：{index_path}")
    return build_lightweight_prediction_index(
        args.prediction_manifest_path,
        sample_keys=[str(key) for key in sample_keys],
        chunk_read_rows=int(args.csv_chunksize),
        index_db_path=index_path,
    )


def load_layout_features(
    *,
    feature_manifest_path: Path,
    sample_df: pd.DataFrame,
    sample_set: str,
    layout_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """函数功能：从 Round2 layout feature cache 读取 mean_patch_embedding 和 revin_aux。"""
    manifest = pd.read_csv(feature_manifest_path)
    rows = manifest[
        (manifest["layout_name"].astype(str) == str(layout_name))
        & (manifest["sample_set"].astype(str) == str(sample_set))
    ].sort_values("start_order_index", kind="mergesort")
    if rows.empty:
        raise ValueError(f"feature manifest 缺少 layout={layout_name} sample_set={sample_set}")
    wanted_count = int(len(sample_df))
    expected_keys = sample_df["sample_key"].astype(str).tolist()
    visual_parts: List[np.ndarray] = []
    aux_parts: List[np.ndarray] = []
    key_parts: List[str] = []
    order_parts: List[np.ndarray] = []
    loaded = 0
    for row in rows.itertuples(index=False):
        if loaded >= wanted_count:
            break
        shard_path = Path(str(row.shard_path))
        with np.load(shard_path, allow_pickle=True) as data:
            shard_keys = [str(value) for value in data["sample_key"].tolist()]
            shard_order = np.asarray(data["order_index"], dtype=np.int64)
            shard_layouts = {str(value) for value in data["layout_name"].tolist()}
            shard_sets = {str(value) for value in data["sample_set"].tolist()}
            visual = np.asarray(data["mean_patch_embedding"], dtype=np.float32)
            aux = np.asarray(data["revin_aux"], dtype=np.float32)
        if shard_layouts != {str(layout_name)} or shard_sets != {str(sample_set)}:
            raise ValueError(f"feature shard layout/sample_set 字段不一致：{shard_path}")
        take = min(int(visual.shape[0]), wanted_count - loaded)
        visual_parts.append(visual[:take])
        aux_parts.append(aux[:take])
        key_parts.extend(shard_keys[:take])
        order_parts.append(shard_order[:take])
        loaded += take
    if loaded != wanted_count:
        raise ValueError(f"feature shard 样本数不足：layout={layout_name} sample_set={sample_set}")
    if key_parts != expected_keys:
        raise ValueError(f"{layout_name}/{sample_set} feature sample_key 与 manifest 顺序不一致")
    order_index = np.concatenate(order_parts, axis=0)
    if not np.array_equal(order_index, sample_df["order_index"].to_numpy(dtype=np.int64, copy=False)):
        raise ValueError(f"{layout_name}/{sample_set} feature order_index 与 manifest 不一致")
    visual_features = np.concatenate(visual_parts, axis=0).astype(np.float32, copy=False)
    aux_features = np.concatenate(aux_parts, axis=0).astype(np.float32, copy=False)
    if visual_features.ndim != 2 or visual_features.shape[0] != wanted_count:
        raise ValueError(f"{layout_name}/{sample_set} visual feature shape 异常：{visual_features.shape}")
    if aux_features.shape != (wanted_count, len(AUX_FEATURE_COLUMNS)):
        raise ValueError(f"{layout_name}/{sample_set} aux feature shape 异常：{aux_features.shape}")
    if not np.isfinite(visual_features).all() or not np.isfinite(aux_features).all():
        raise ValueError(f"{layout_name}/{sample_set} feature 中存在 NaN/Inf")
    return visual_features, aux_features


def run_build_index_only(args: argparse.Namespace) -> None:
    """函数功能：单进程预构建本轮 sample subset prediction SQLite index。"""
    frames = read_all_sample_frames(args)
    keys: List[str] = []
    for name in sample_sets_from_args(args):
        keys.extend(frames[name]["sample_key"].astype(str).tolist())
    index = ensure_round2_prediction_index(args, keys)
    index.close()
    write_json(args.output_dir / "prediction_index_status.json", {"status": "completed", "index_path": str(prediction_index_path(args.output_dir)), "sample_key_count": len(set(keys)), "updated_at": display_time()})


def run_single(args: argparse.Namespace) -> None:
    """函数功能：训练并评估一个 layout/seed 的固定 FiLM backend。"""
    if args.layout is None or args.seed is None:
        raise ValueError("--run-single 必须同时提供 --layout 和 --seed")
    layout_name = str(args.layout)
    seed = int(args.seed)
    out_dir = task_dir(args.output_dir, layout_name, seed)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "task_metadata.json").exists() and not args.overwrite:
        raise FileExistsError(f"单任务输出已存在；如需覆盖请传 --overwrite：{out_dir}")
    write_json(out_dir / "status.json", {"status": "started", "layout_name": layout_name, "seed": seed, "updated_at": display_time()})
    frames = read_all_sample_frames(args)
    all_keys: List[str] = []
    train_set, selection_set, diagnostic_set, test_set = sample_sets_from_args(args)
    eval_sets = eval_sample_sets_from_args(args)
    for name in sample_sets_from_args(args):
        all_keys.extend(frames[name]["sample_key"].astype(str).tolist())
    log_stage(f"读取 oracle labels 子集：layout={layout_name} seed={seed}")
    labels_all = load_oracle_subset(args.oracle_labels_path, all_keys, batch_rows=args.parquet_batch_rows)
    labels_by_set = {name: labels_all[labels_all["sample_key"].isin(frames[name]["sample_key"].astype(str))].copy() for name in eval_sets}
    prediction_index = ensure_round2_prediction_index(args, all_keys)
    device = resolve_device(args.device)
    feature_manifest_path = Path(args.feature_dir) / f"{args.artifact_prefix}_feature_manifest.csv"
    try:
        log_stage(f"读取 Round2 layout features：layout={layout_name}")
        train_visual, train_aux = load_layout_features(
            feature_manifest_path=feature_manifest_path,
            sample_df=frames[train_set],
            sample_set=train_set,
            layout_name=layout_name,
        )
        visual_scaler = StandardScaler()
        aux_scaler = StandardScaler()
        train_visual_scaled = visual_scaler.fit_transform(train_visual).astype(np.float32)
        train_aux_scaled = aux_scaler.fit_transform(train_aux).astype(np.float32)
        del train_visual, train_aux
        router, train_meta = train_film_router(
            train_visual_scaled=train_visual_scaled,
            train_aux_scaled=train_aux_scaled,
            train_sample_keys=frames[train_set]["sample_key"].astype(str).tolist(),
            prediction_index=prediction_index,
            seed=seed,
            device=device,
            hidden_dim=int(args.hidden_dim),
            film_hidden_dim=int(args.film_hidden_dim),
            dropout=float(args.dropout),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            huber_beta=float(args.huber_beta),
            kl_tau=float(args.kl_tau),
            lambda_kl=float(args.lambda_kl),
            metric=str(args.metric),
        )
        checkpoint_path = out_dir / f"checkpoint_{layout_name}_seed{seed}.pt"
        torch.save(
            {
                "script_version": SCRIPT_VERSION,
                "layout_name": layout_name,
                "backend_style": "film_mean_patch_aux",
                "seed": seed,
                "router_state_dict": router.state_dict(),
                "visual_scaler_state": scaler_to_state(visual_scaler),
                "aux_scaler_state": scaler_to_state(aux_scaler),
                "model_columns": list(MODEL_COLUMNS),
                "aux_feature_columns": list(AUX_FEATURE_COLUMNS),
                "hyperparameters": {
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "hidden_dim": int(args.hidden_dim),
                    "film_hidden_dim": int(args.film_hidden_dim),
                    "dropout": float(args.dropout),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "huber_beta": float(args.huber_beta),
                    "kl_tau": float(args.kl_tau),
                    "lambda_kl": float(args.lambda_kl),
                },
            },
            checkpoint_path,
        )
        method_frames: List[pd.DataFrame] = []
        for sample_set in eval_sets:
            visual_features, aux_features = load_layout_features(
                feature_manifest_path=feature_manifest_path,
                sample_df=frames[sample_set],
                sample_set=sample_set,
                layout_name=layout_name,
            )
            pred = predict_film_router(
                router=router,
                visual_scaler=visual_scaler,
                aux_scaler=aux_scaler,
                visual_features=visual_features,
                aux_features=aux_features,
                sample_df=frames[sample_set],
                labels_df=labels_by_set[sample_set],
                variant=layout_name,
                seed=seed,
                sample_set=sample_set,
                device=device,
            )
            pred = add_batch_fusion_metrics(pred, prediction_index=prediction_index, metric=str(args.metric), batch_size=int(args.eval_batch_size))
            pred.to_csv(out_dir / f"predictions_{layout_name}_seed{seed}_{sample_set}.csv", index=False)
            method_frames.append(make_visual_pooling_method_rows(pred, sample_set=sample_set, variant=layout_name, seed=seed))
        method_rows = pd.concat(method_frames, ignore_index=True)
        seed_results = summarize_rows_with_seed(method_rows)
        method_rows.to_csv(out_dir / "method_rows.csv", index=False)
        seed_results.to_csv(out_dir / "seed_results.csv", index=False)
        write_json(
            out_dir / "task_metadata.json",
            {
                "status": "completed",
                "generated_at": display_time(),
                "script_version": SCRIPT_VERSION,
                "layout_name": layout_name,
                "backend_style": "film_mean_patch_aux",
                "seed": seed,
                "device": str(device),
                "checkpoint_path": str(checkpoint_path),
                "train_metadata": train_meta,
                "constraints": {
                    "only_variable_is_layout": True,
                    "base_visual_input": "mean_patch_embedding",
                    "condition_input": "revin_aux",
                    "used_film": True,
                    "used_concat_aux": False,
                    "frozen_test_used_for_selection": False,
                    "train_sample_set": train_set,
                    "selection_sample_set": selection_set,
                    "diagnostic_sample_set": diagnostic_set,
                    "test_sample_set": test_set,
                    "trained_vit": False,
                    "saved_pseudo_image_tensor": False,
                    "single_task_output_isolated": True,
                },
            },
        )
        write_json(out_dir / "status.json", {"status": "completed", "layout_name": layout_name, "seed": seed, "updated_at": display_time()})
        log_stage(f"单任务完成：{out_dir}")
    finally:
        prediction_index.close()


def comparison_from_reference(path: Path, *, stage: str, sample_set: str) -> pd.DataFrame:
    """函数功能：读取 Round1/Round0 reference 表并规范成 Round2 comparison 行。"""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "sample_set" in df.columns:
        df = df.copy()
        df["sample_set"] = sample_set
    return normalize_comparison_frame(df, stage=stage, source_path=path)


def normalize_round0_reference(path: Path, sample_set: str) -> pd.DataFrame:
    """函数功能：抽取 Round0 TimeFuse、oracle 和 global single reference。"""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows: List[Dict[str, object]] = []
    name_map = {
        "timefuse_raw_soft_fusion": ("Round0 TimeFuse", "raw_soft_fusion"),
        "timefuse_hard_top1": ("Round0 TimeFuse", "hard_top1"),
        "global_best_single": ("global_best_single", "single"),
        "oracle_top1": ("oracle_top1", "oracle"),
    }
    for row in df.itertuples(index=False):
        data = row._asdict()
        method = str(data["method"])
        if method not in name_map:
            continue
        variant, kind = name_map[method]
        rows.append(
            {
                "stage": "Round0",
                "sample_set": sample_set,
                "variant": variant,
                "method": method,
                "method_kind": kind,
                "seed_count": 1,
                "sample_count": int(data["sample_count"]),
                "MAE_mean": float(data["MAE"]),
                "MAE_std": 0.0,
                "MSE_mean": float(data["MSE"]) if "MSE" in data and not pd.isna(data["MSE"]) else np.nan,
                "MSE_std": 0.0,
                "regret_to_oracle_mean": float(data["regret_to_oracle"]),
                "regret_to_oracle_std": 0.0,
                "oracle_label_accuracy_mean": float(data["oracle_label_accuracy"]),
                "oracle_label_accuracy_std": 0.0,
                "weight_entropy_mean": float(data["weight_entropy"]) if "weight_entropy" in data and not pd.isna(data["weight_entropy"]) else np.nan,
                "weight_entropy_std": 0.0,
                "normalized_weight_entropy_mean": float(data["normalized_weight_entropy"]) if "normalized_weight_entropy" in data and not pd.isna(data["normalized_weight_entropy"]) else np.nan,
                "normalized_weight_entropy_std": 0.0,
                "mean_max_weight_mean": float(data["mean_max_weight"]) if "mean_max_weight" in data and not pd.isna(data["mean_max_weight"]) else np.nan,
                "mean_max_weight_std": 0.0,
                "source_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def choose_best_layout(selection_summary: pd.DataFrame, *, selection_sample_set: str, diagnostic_sample_set: str, test_sample_set: str) -> Dict[str, object]:
    """函数功能：按用户指定 selection/tie-breaker 选择 best layout。"""
    soft = selection_summary[selection_summary["method"].astype(str).str.endswith("_raw_soft_fusion")].copy()
    if soft.empty:
        raise ValueError("selection summary 缺少 raw_soft_fusion 行")
    soft = soft.sort_values(
        ["MAE_mean", "MAE_std", "MSE_mean", "regret_to_oracle_mean", "weight_entropy_std", "mean_max_weight_std"],
        ascending=[True, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best = soft.iloc[0].to_dict()
    return {
        "best_layout": str(best["variant"]),
        "backend_style": "film_mean_patch_aux",
        "selection_basis": f"{selection_sample_set} raw-soft MAE_mean; tie-breakers MAE_std, MSE_mean, regret_to_oracle_mean, weight_entropy_std, mean_max_weight_std",
        "selected_from_sample_set": str(selection_sample_set),
        f"{diagnostic_sample_set}_used_for_selection": False,
        f"{test_sample_set}_used_for_selection": False,
        "best_row": {key: (float(value) if isinstance(value, (float, np.floating)) else int(value) if isinstance(value, (int, np.integer)) else value) for key, value in best.items()},
    }


def build_delta_summary(comparison: pd.DataFrame, *, selection_sample_set: str, layout_stage: str) -> pd.DataFrame:
    """函数功能：生成用户指定的 layout delta summary。"""
    selection_soft = comparison[
        (comparison["sample_set"].astype(str) == str(selection_sample_set))
        & (comparison["method_kind"].astype(str) == "raw_soft_fusion")
    ].set_index("variant")
    layout_only = selection_soft[selection_soft["stage"].astype(str) == str(layout_stage)] if "stage" in selection_soft.columns else selection_soft
    best_layout_name = ""
    if not layout_only.empty:
        best_layout_name = str(layout_only.sort_values(["MAE_mean", "MAE_std"], kind="mergesort").index[0])
    pairs = [
        ("spatial_panel_3view", "current_rgb_3view"),
        ("line_only", "current_rgb_3view"),
        ("line_difference_band", "line_only"),
        ("line_difference_band", "current_rgb_3view"),
        ("fft_absolute_energy", "current_rgb_3view"),
        ("top3fold_period_layout", "current_rgb_3view"),
        ("top3fold_period_layout", "fft_absolute_energy"),
    ]
    if best_layout_name:
        pairs.extend([(best_layout_name, "film_mean_patch_aux"), (best_layout_name, "Round0 TimeFuse")])
    rows: List[Dict[str, object]] = []
    for left, right in pairs:
        if left not in selection_soft.index or right not in selection_soft.index:
            rows.append({"delta_name": f"{left} - {right}", "status": "missing_reference"})
            continue
        left_row = selection_soft.loc[left]
        right_row = selection_soft.loc[right]
        rows.append(
            {
                "delta_name": f"{left} - {right}",
                "sample_set": str(selection_sample_set),
                "method_kind": "raw_soft_fusion",
                "left_variant": left,
                "right_variant": right,
                "left_MAE_mean": float(left_row["MAE_mean"]),
                "right_MAE_mean": float(right_row["MAE_mean"]),
                "delta_MAE_mean": float(left_row["MAE_mean"] - right_row["MAE_mean"]),
                "left_MSE_mean": float(left_row["MSE_mean"]) if not pd.isna(left_row["MSE_mean"]) else np.nan,
                "right_MSE_mean": float(right_row["MSE_mean"]) if not pd.isna(right_row["MSE_mean"]) else np.nan,
                "delta_MSE_mean": float(left_row["MSE_mean"] - right_row["MSE_mean"]) if not pd.isna(left_row["MSE_mean"]) and not pd.isna(right_row["MSE_mean"]) else np.nan,
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def _raw_soft_rows(frame: pd.DataFrame, sample_set: str | None = None) -> pd.DataFrame:
    """函数功能：筛选 raw-soft fusion 汇总行，可选限制 sample_set。"""
    rows = frame[frame["method"].astype(str).str.endswith("_raw_soft_fusion")].copy()
    if sample_set is not None:
        rows = rows[rows["sample_set"].astype(str) == str(sample_set)].copy()
    return rows


def _best_layout_from_summary(summary: pd.DataFrame) -> Dict[str, object]:
    """函数功能：从 layout-only summary 中按 raw-soft MAE/MSE 选最优 layout。"""
    rows = _raw_soft_rows(summary)
    if rows.empty:
        return {"layout": "", "MAE_mean": np.nan, "MSE_mean": np.nan}
    rows = rows.sort_values(["MAE_mean", "MAE_std", "MSE_mean"], kind="mergesort").reset_index(drop=True)
    row = rows.iloc[0]
    return {"layout": str(row["variant"]), "MAE_mean": float(row["MAE_mean"]), "MSE_mean": float(row["MSE_mean"])}


def _metric_delta(summary: pd.DataFrame, left: str, right: str, metric: str = "MAE_mean") -> str:
    """函数功能：生成两个 layout 在 raw-soft 指标上的差值描述。"""
    rows = _raw_soft_rows(summary).set_index("variant")
    if left not in rows.index or right not in rows.index:
        return f"`{left}` 或 `{right}` 缺少结果。"
    left_value = float(rows.loc[left][metric])
    right_value = float(rows.loc[right][metric])
    delta = left_value - right_value
    verdict = "优于" if delta < 0 else "未优于"
    return f"`{left}` {metric}={left_value:.6f}，`{right}`={right_value:.6f}，delta={delta:+.6f}，结论：{verdict}。"


def recommend_expanded_layouts(selection_summary: pd.DataFrame, test_summary: pd.DataFrame, limit: int = 3) -> List[str]:
    """
    函数功能：
        根据 selection raw-soft MAE 推荐进入 65k 的 layout，并在 test best 不同
        时把 test best 纳入 2-3 个候选。
    """
    selection_rows = _raw_soft_rows(selection_summary).sort_values(["MAE_mean", "MAE_std", "MSE_mean"], kind="mergesort")
    picks: List[str] = []
    for value in selection_rows["variant"].astype(str).tolist():
        if value not in picks:
            picks.append(value)
        if len(picks) >= int(limit):
            break
    test_best = _best_layout_from_summary(test_summary)["layout"]
    if test_best and test_best not in picks:
        if len(picks) >= int(limit):
            picks[-1] = test_best
        else:
            picks.append(test_best)
    return picks[: int(limit)]


def _latency_assessment(output_dir: Path) -> str:
    """函数功能：基于 feature latency 粗略判断 layout 是否过慢。"""
    candidates = sorted(Path(output_dir).glob("*_feature_latency.csv"))
    path = candidates[0] if candidates else Path(output_dir) / "round2_layout_feature_latency.csv"
    if not path.exists():
        return "feature latency 文件不存在，暂无法判断。"
    latency = pd.read_csv(path)
    if latency.empty:
        return "feature latency 文件为空，暂无法判断。"
    latency = latency.copy()
    latency["total_ms"] = latency["imageization_latency_ms"].astype(float) + latency["encoder_forward_ms"].astype(float)
    grouped = latency.groupby("layout_name", as_index=False).agg(total_ms=("total_ms", "sum"), samples=("batch_size", "sum"))
    grouped["ms_per_sample"] = grouped["total_ms"] / grouped["samples"].clip(lower=1)
    grouped = grouped.sort_values("ms_per_sample", ascending=False, kind="mergesort")
    median = float(grouped["ms_per_sample"].median())
    slow = grouped[grouped["ms_per_sample"] > median * 1.5]["layout_name"].astype(str).tolist()
    head = "; ".join(f"{row.layout_name}={row.ms_per_sample:.3f} ms/sample" for row in grouped.itertuples(index=False))
    if slow:
        return f"相对过慢 layout：{', '.join(slow)}。latency 排序：{head}。"
    return f"未发现超过 median 1.5x 的明显过慢 layout。latency 排序：{head}。"


def _strata_help_text(stratified: pd.DataFrame, sample_set: str, stratum_column: str, values: Sequence[str], baseline: str = "current_rgb_3view") -> str:
    """函数功能：描述特定 strata 中哪些 layout 相比 baseline 改善 MAE/MSE。"""
    rows = stratified[
        (stratified["sample_set"].astype(str) == str(sample_set))
        & (stratified["method"].astype(str).str.endswith("_raw_soft_fusion"))
        & (stratified["stratum_column"].astype(str) == stratum_column)
        & (stratified["stratum_value"].astype(str).isin([str(v) for v in values]))
    ].copy()
    if rows.empty or baseline not in rows["variant"].astype(str).unique():
        return f"{sample_set}/{stratum_column} 缺少可比较 strata 结果。"
    grouped = rows.groupby(["variant", "stratum_value"], as_index=False).agg(MAE=("MAE", "mean"), MSE=("MSE", "mean"))
    improved: List[str] = []
    for value in values:
        part = grouped[grouped["stratum_value"].astype(str) == str(value)].copy()
        base = part[part["variant"].astype(str) == baseline]
        if base.empty:
            continue
        base_mae = float(base.iloc[0]["MAE"])
        better = part[part["MAE"].astype(float) < base_mae]["variant"].astype(str).tolist()
        if better:
            improved.append(f"{value}: {', '.join(better)}")
    return "；".join(improved) if improved else f"未发现 layout 在 {', '.join(values)} 上相对 `{baseline}` 改善 MAE。"


def _mse_tail_text(selection_summary: pd.DataFrame, test_summary: pd.DataFrame, *, test_label: str) -> str:
    """函数功能：描述哪些 layout 在 selection/frozen test 上降低 MSE。"""
    selection = _raw_soft_rows(selection_summary).sort_values("MSE_mean", kind="mergesort")
    test = _raw_soft_rows(test_summary).sort_values("MSE_mean", kind="mergesort")
    if selection.empty or test.empty:
        return "缺少 raw-soft MSE summary。"
    return (
        f"selection MSE 最低：`{selection.iloc[0]['variant']}`={float(selection.iloc[0]['MSE_mean']):.6f}；"
        f"{test_label} MSE 最低：`{test.iloc[0]['variant']}`={float(test.iloc[0]['MSE_mean']):.6f}。"
    )


def _seed_stability_text(selection_summary: pd.DataFrame) -> str:
    """函数功能：按 selection raw-soft MAE std 描述 seed stability。"""
    rows = _raw_soft_rows(selection_summary).sort_values(["MAE_std", "MSE_std", "MAE_mean"], kind="mergesort")
    if rows.empty:
        return "缺少 raw-soft seed stability summary。"
    row = rows.iloc[0]
    return f"`{row['variant']}` 最稳定，selection MAE_std={float(row['MAE_std']):.6f}，MSE_std={float(row['MSE_std']):.6f}。"


def write_summary_md(
    *,
    output_dir: Path,
    selection_summary: pd.DataFrame,
    diagnostic_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    stratified: pd.DataFrame,
    delta_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    best_layout: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    """函数功能：写中文 screening/expanded validation summary。"""
    selection_best = _best_layout_from_summary(selection_summary)
    test_best = _best_layout_from_summary(test_summary)
    consistency = "一致" if selection_best["layout"] == test_best["layout"] else "不一致"
    selection_set = str(metadata["selection_sample_set"])
    diagnostic_set = str(metadata["diagnostic_sample_set"])
    test_set = str(metadata["test_sample_set"])
    summary_title = str(metadata.get("summary_title", "Visual Router V2 Round2 Layout Summary"))
    seasonality_text = _strata_help_text(stratified, diagnostic_set, "season_strength_cat", ["strong"])
    patch_text = _strata_help_text(stratified, diagnostic_set, "oracle_model", ["CrossFormer", "PatchTST"])
    mse_text = _mse_tail_text(selection_summary, test_summary, test_label=test_set)
    stability_text = _seed_stability_text(selection_summary)
    latency_text = _latency_assessment(output_dir)
    spatial_delta = _metric_delta(selection_summary, "spatial_panel_3view", "current_rgb_3view")
    top3_delta = _metric_delta(selection_summary, "top3fold_period_layout", "current_rgb_3view")
    stable_35k = "稳定" if selection_best["layout"] == "spatial_panel_3view" or test_best["layout"] == "spatial_panel_3view" else "不稳定"
    next_step = "建议升级 `spatial_panel_3view` 为 Round2 主线，并扩大到 P0/P2a 规模。" if best_layout["best_layout"] == "spatial_panel_3view" else "不建议直接升级 `spatial_panel_3view`，应优先复查 period_soft_mixture / period tokens 或扩大候选诊断。"
    lines = [
        f"# {summary_title}",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 结论",
        "",
        f"- best_layout：`{best_layout['best_layout']}`",
        f"- 后端固定：`film_mean_patch_aux`，即 `mean_patch_embedding -> visual hidden`，`revin_aux -> FiLM gamma/beta`。",
        f"- 选择口径只使用 `{selection_set}` raw-soft MAE mean；`{diagnostic_set}` 只诊断，`{test_set}` 只做 frozen validation。",
        f"- 下一步建议：{next_step}",
        "",
        "## 必答问题",
        "",
        f"1. 65k selection best：`{selection_best['layout']}`，raw-soft MAE={selection_best['MAE_mean']:.6f}，MSE={selection_best['MSE_mean']:.6f}。",
        f"2. 65k test_expanded best：`{test_best['layout']}`，raw-soft MAE={test_best['MAE_mean']:.6f}，MSE={test_best['MSE_mean']:.6f}。",
        f"3. selection best 与 test best 是否一致：{consistency}。",
        f"4. `spatial_panel_3view` 是否仍优于 `current_rgb_3view`：{spatial_delta}",
        f"5. `top3fold_period_layout` 的 continuity / diagnostic 优势是否转化为 expanded 性能：{top3_delta} diagnostic CrossFormer/PatchTST={patch_text}；seasonality={seasonality_text}。",
        f"6. 35k 结论是否在 65k 上稳定：{stable_35k}。35k screening 结论为 `spatial_panel_3view` 是 selection/test_small best，本轮以 65k selection/test 是否继续支持该 layout 判定。",
        f"7. seed stability / MSE tail / CrossFormer / PatchTST strata：seed stability={stability_text} MSE tail={mse_text} CrossFormer/PatchTST={patch_text}",
        f"8. 是否建议把 `spatial_panel_3view` 升级为 Round2 主线：{'建议' if best_layout['best_layout'] == 'spatial_panel_3view' else '暂不建议'}。",
        f"9. 下一步：{next_step}",
        f"10. latency 检查：{latency_text}",
        "",
        "## Selection",
        "",
        frame_to_markdown(selection_summary, float_digits=6),
        "",
        "## Diagnostic Balanced",
        "",
        frame_to_markdown(diagnostic_summary, float_digits=6),
        "",
        "## Frozen Test",
        "",
        frame_to_markdown(test_summary, float_digits=6),
        "",
        "## Delta Summary",
        "",
        frame_to_markdown(delta_summary, float_digits=6),
        "",
        "## Reference-Inclusive Comparison",
        "",
        frame_to_markdown(comparison, float_digits=6),
    ]
    (output_dir / f"{metadata['artifact_prefix']}_validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_light_summaries(output_dir: Path, summary_dir: Path, names: Sequence[str]) -> None:
    """函数功能：复制轻量 summary 到仓库 experiment_summaries。"""
    summary_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(output_dir / name, summary_dir / name)


def aggregate(args: argparse.Namespace) -> None:
    """函数功能：汇总 layout × seed 单任务输出，生成 layout validation 统一产物。"""
    layouts = [str(args.layout)] if args.layout else parse_csv(args.layouts)
    seeds = parse_seed_list(args.seeds)
    train_set, selection_set, diagnostic_set, test_set = sample_sets_from_args(args)
    method_frames: List[pd.DataFrame] = []
    seed_frames: List[pd.DataFrame] = []
    task_meta: List[Mapping[str, object]] = []
    missing: List[str] = []
    for layout in layouts:
        for seed in seeds:
            out_dir = task_dir(args.output_dir, layout, seed)
            for name in ["method_rows.csv", "seed_results.csv", "task_metadata.json"]:
                if not (out_dir / name).exists():
                    missing.append(str(out_dir / name))
            if not missing:
                method_frames.append(pd.read_csv(out_dir / "method_rows.csv"))
                seed_frames.append(pd.read_csv(out_dir / "seed_results.csv"))
                task_meta.append(json.loads((out_dir / "task_metadata.json").read_text(encoding="utf-8")))
    if missing:
        raise FileNotFoundError("layout validation task 输出不完整：" + "; ".join(missing[:20]))
    method_rows = pd.concat(method_frames, ignore_index=True)
    seed_results = pd.concat(seed_frames, ignore_index=True)
    selection_summary = summarize_mean_std(seed_results, sample_set=selection_set)
    diagnostic_summary = summarize_mean_std(seed_results, sample_set=diagnostic_set)
    test_summary = summarize_mean_std(seed_results, sample_set=test_set)
    selected_counts = selected_model_counts_with_variant(method_rows)
    stratified = build_film_stratified_summary(method_rows)
    best_layout = choose_best_layout(selection_summary, selection_sample_set=selection_set, diagnostic_sample_set=diagnostic_set, test_sample_set=test_set)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.artifact_prefix)
    seed_results_path = artifact_path(args, "variant_seed_results.csv")
    selection_layout_only_path = artifact_path(args, "selection_layout_only.csv")
    selection_comparison_path = artifact_path(args, "selection_comparison.csv")
    diagnostic_path = artifact_path(args, "diagnostic_summary.csv")
    test_path = artifact_path(args, "test_summary.csv")
    selected_counts_path = artifact_path(args, "selected_model_counts.csv")
    stratified_path = artifact_path(args, "stratified_summary.csv")
    delta_path = artifact_path(args, "delta_summary.csv")
    best_path = artifact_path(args, "best_layout.json")
    metadata_path = artifact_path(args, "validation_metadata.json")
    summary_path = artifact_path(args, "validation_summary.md")
    seed_results.to_csv(seed_results_path, index=False)
    selection_summary.to_csv(selection_layout_only_path, index=False)
    diagnostic_summary.to_csv(diagnostic_path, index=False)
    test_summary.to_csv(test_path, index=False)
    selected_counts.to_csv(selected_counts_path, index=False)
    stratified.to_csv(stratified_path, index=False)
    write_json(best_path, best_layout)

    comparison_frames = [
        normalize_comparison_frame(selection_summary, stage=str(args.experiment_label), source_path=selection_layout_only_path),
        normalize_comparison_frame(diagnostic_summary, stage=str(args.experiment_label), source_path=diagnostic_path),
        normalize_comparison_frame(test_summary, stage=str(args.experiment_label), source_path=test_path),
    ]
    for path, stage, sample_set in [
        (DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round1_film" / "round1_film_selection_comparison.csv", "Round1 reference", selection_set),
        (DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_visual_pooling" / "visual_pooling_selection_comparison.csv", "Round1 reference", selection_set),
    ]:
        ref = comparison_from_reference(path, stage=stage, sample_set=sample_set)
        if not ref.empty:
            comparison_frames.append(ref[ref["variant"].astype(str).isin(["film_mean_patch_aux", "visual_cls_mean_concat"])].copy())
    round0_ref = normalize_round0_reference(Path(args.round0_dir) / "round0_selection_comparison.csv", selection_set)
    if not round0_ref.empty:
        comparison_frames.append(round0_ref)
    comparison = pd.concat(comparison_frames, ignore_index=True)
    comparison = comparison.sort_values(["sample_set", "method_kind", "MAE_mean", "stage", "variant"], kind="mergesort").reset_index(drop=True)
    # 目标文件要求 comparison 本身包含 Round2 layouts 与 Round1/Round0/oracle/global references；
    # 因此 required 文件名写 reference-inclusive 版本，同时保留 layout-only 副本。
    comparison.to_csv(selection_comparison_path, index=False)
    comparison.to_csv(artifact_path(args, "selection_comparison_with_references.csv"), index=False)
    delta_summary = build_delta_summary(comparison, selection_sample_set=selection_set, layout_stage=str(args.experiment_label))
    delta_summary.to_csv(delta_path, index=False)

    devices_used = sorted({str(meta.get("device", "")) for meta in task_meta if str(meta.get("device", "")).strip()})
    next_step = (
        "Upgrade spatial_panel_3view to Round2 mainline and expand to P0/P2a scale."
        if str(best_layout["best_layout"]) == "spatial_panel_3view"
        else "Continue period_soft_mixture / period tokens diagnostics before promoting the mainline."
    )
    metadata = {
        "status": "completed",
        "round2_stage": "expanded_layout_validation" if "expanded" in prefix else "layout_feature_cache_and_fixed_film_screening",
        "generated_at": display_time(),
        "artifact_prefix": prefix,
        "summary_title": str(args.summary_title),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "commit_hash": git_commit_hash(),
        "output_dir": str(output_dir),
        "summary_copy_dir": str(args.summary_copy_dir),
        "inputs": {
            "sample_manifest": str(args.sample_manifest),
            "feature_dir": str(args.feature_dir),
            "oracle_labels_path": str(args.oracle_labels_path),
            "prediction_manifest_path": str(args.prediction_manifest_path),
            "prediction_index": str(prediction_index_path(output_dir)),
        },
        "layouts": layouts,
        "layouts_screened": layouts,
        "deferred_layouts": list(DEFERRED_ROUND2_LAYOUTS),
        "seeds": seeds,
        "train_sample_set": train_set,
        "selection_sample_set": selection_set,
        "diagnostic_sample_set": diagnostic_set,
        "test_sample_set": test_set,
        "backend_style": "film_mean_patch_aux",
        "backend_fixed_to": "film_mean_patch_aux_style",
        "trained_model": True,
        "built_feature_cache": True,
        "ran_vit": True,
        "saved_pseudo_image_tensor": False,
        "used_frozen_test_for_selection": False,
        "loaded_116m_prediction_manifest_to_memory": False,
        "layout_registry_used": True,
        "changed_router_head": False,
        "changed_loss": False,
        "changed_selection_rule": False,
        "changed_sample_split": False,
        "devices_requested": str(args.devices_requested),
        "devices_used": devices_used,
        "parallel_backend": "process_per_layout_and_seed",
        "single_task_output_isolated": True,
        "feature_cache_parallel_used": bool(args.parallel_launcher_used),
        "training_parallel_used": bool(args.parallel_launcher_used),
        "next_step_recommendation": next_step,
        "constraints": {
            "only_variable_is_layout": True,
            "base_visual_input": "mean_patch_embedding",
            "condition_input": "revin_aux",
            "used_film": True,
            "used_concat_aux": False,
            "searched_head_loss_film_dim_dropout_calibration": False,
            f"{selection_set}_used_for_selection": True,
            f"{diagnostic_set}_used_for_selection": False,
            f"{test_set}_used_for_selection": False,
            "parallel_launcher_used": bool(args.parallel_launcher_used),
            "parallel_backend": "process_per_layout_and_seed",
            "devices_requested": str(args.devices_requested),
            "devices_used": devices_used,
            "single_task_output_isolated": True,
            "feature_cache_parallel_used": bool(args.parallel_launcher_used),
            "training_parallel_used": bool(args.parallel_launcher_used),
        },
        "best_layout": best_layout,
        "task_metadata": task_meta,
    }
    write_json(metadata_path, metadata)
    write_summary_md(
        output_dir=output_dir,
        selection_summary=selection_summary,
        diagnostic_summary=diagnostic_summary,
        test_summary=test_summary,
        stratified=stratified,
        delta_summary=delta_summary,
        comparison=comparison,
        best_layout=best_layout,
        metadata=metadata,
    )
    copy_light_summaries(
        output_dir,
        args.summary_copy_dir,
        [
            seed_results_path.name,
            selection_comparison_path.name,
            diagnostic_path.name,
            test_path.name,
            selected_counts_path.name,
            stratified_path.name,
            delta_path.name,
            best_path.name,
            metadata_path.name,
            summary_path.name,
        ],
    )
    write_json(output_dir / "status.json", {"status": "completed", "best_layout": best_layout, "updated_at": display_time()})
    log_stage(f"layout validation aggregation outputs written to {output_dir}")


def run_serial(args: argparse.Namespace) -> None:
    """函数功能：无 launcher 时提供串行 fallback。"""
    if args.build_index_only:
        run_build_index_only(args)
        return
    seeds = parse_seed_list(args.seeds)
    layouts = [str(args.layout)] if args.layout else parse_csv(args.layouts)
    run_build_index_only(args)
    for layout in layouts:
        for seed in seeds:
            child = argparse.Namespace(**vars(args))
            child.layout = layout
            child.seed = seed
            child.run_single = True
            child.aggregate_only = False
            child.build_index_only = False
            run_single(child)
    aggregate(args)


def main() -> None:
    """函数功能：根据 CLI 模式执行 build-index、单任务、汇总或串行 fallback。"""
    args = parse_args()
    if args.build_index_only:
        run_build_index_only(args)
    elif args.aggregate_only:
        aggregate(args)
    elif args.run_single:
        run_single(args)
    else:
        run_serial(args)


if __name__ == "__main__":
    main()
