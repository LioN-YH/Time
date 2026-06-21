#!/usr/bin/env python3
"""
文件功能：
    对 Visual Router V2 Round 1 P2d 已冻结 best variant 做 pilot_test final
    evaluation，并与 Round0 TimeFuse / Round0 Visual / global / oracle 同表比较。

核心边界：
    - 只加载 P2d 已保存 checkpoint，不训练新模型、不调参、不按 pilot_test 选择；
    - variant 固定为 `cls_mean_concat_plus_aux`；
    - pilot_test feature cache 必须来自独立 final_test_only 输出目录，且 metadata
      标明 final_test_only；
    - 只为 P0 pilot_test sample_key 构建/复用 SQLite prediction 子集索引，避免把
      116M 行 manifest 读入内存。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.stage1_vali_test_router.evaluate_visual_router_v2_round0 import (  # noqa: E402
    DEFAULT_ORACLE_LABELS,
    DEFAULT_PREDICTION_MANIFEST,
    DEFAULT_TIMEFUSE_DIR,
    DEFAULT_VISUAL_DIR,
    extract_csv_rows_by_sample_keys,
    load_oracle_subset,
)
from visual_router_experiments.stage1_vali_test_router.fusion_utils import MODEL_COLUMNS, frame_to_markdown  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router import VisualMLPRouter  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import (  # noqa: E402
    SQLitePredictionIndex,
    build_lightweight_prediction_index,
    scaler_from_state,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_evaluator import (  # noqa: E402
    TSF_STRATA_COLUMNS,
    add_oracle_and_global_rows,
    align_with_sample_frame,
    choose_global_best_model,
    make_method_rows,
    read_sample_csv,
    selected_model_counts,
    summarize_method_rows,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_round1_training import (  # noqa: E402
    add_batch_fusion_metrics,
    predict_visual_pooling_router,
    summarize_mean_std,
    summarize_rows_with_seed,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_v2_round1_concat import (  # noqa: E402
    load_concat_features,
)


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_SAMPLE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_pilot_samples"
DEFAULT_ROUND0_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round0"
DEFAULT_P2A_FEATURE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_features"
DEFAULT_P2D_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_concat"
DEFAULT_FINAL_FEATURE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_features_final_test_only"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_final_test"
SCRIPT_VERSION = "visual_router_v2_round1_final_test_eval_v1"
FROZEN_BEST_VARIANT = "cls_mean_concat_plus_aux"


def display_time() -> str:
    """函数功能：生成本地日志和 metadata 时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 frozen final eval 参数；默认路径均来自 Round 1 协议。"""
    parser = argparse.ArgumentParser(description="Frozen final eval for Visual Router V2 Round 1 P2d best on P0 pilot_test.")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--round0-dir", type=Path, default=DEFAULT_ROUND0_DIR)
    parser.add_argument("--p2a-feature-dir", type=Path, default=DEFAULT_P2A_FEATURE_DIR)
    parser.add_argument("--p2d-dir", type=Path, default=DEFAULT_P2D_DIR)
    parser.add_argument("--final-feature-dir", type=Path, default=DEFAULT_FINAL_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--oracle-labels-path", type=Path, default=DEFAULT_ORACLE_LABELS)
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST)
    parser.add_argument("--visual-dir", type=Path, default=DEFAULT_VISUAL_DIR)
    parser.add_argument("--timefuse-dir", type=Path, default=DEFAULT_TIMEFUSE_DIR)
    parser.add_argument("--variant", default=FROZEN_BEST_VARIANT)
    parser.add_argument("--seeds", default="16,17,18", help="只评估 P2d 已保存 seeds；主结论使用 mean/std，不按 test 选 seed。")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--csv-chunksize", type=int, default=200_000)
    parser.add_argument("--parquet-batch-rows", type=int, default=250_000)
    parser.add_argument("--feature-shard-size", type=int, default=2000)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--vit-data-parallel", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-feature-build", action="store_true", help="只校验并复用已有 final_test_only feature cache。")
    return parser.parse_args()


def parse_seed_list(seed_text: str) -> List[int]:
    """函数功能：解析逗号分隔 seed 列表，去重并保序。"""
    seeds: List[int] = []
    for part in str(seed_text).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in seeds:
            seeds.append(value)
    if not seeds:
        raise ValueError("--seeds 不能为空")
    return seeds


def git_commit_hash() -> str:
    """函数功能：记录当前 commit；失败时写 unknown，不影响评估。"""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def log_stage(message: str) -> None:
    """函数功能：输出阶段进度，便于长任务后台日志监控。"""
    print(f"[{display_time()}] {message}", flush=True)


def resolve_device(device_arg: str) -> torch.device:
    """函数功能：解析 frozen eval 设备；auto 优先使用 CUDA。"""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 --device cuda，但当前 CUDA 不可用")
    return torch.device(device_arg)


def required_outputs() -> List[str]:
    """函数功能：列出 final eval 顶层必需产物，避免无意覆盖。"""
    return [
        "round1_final_test_comparison.csv",
        "round1_final_test_variant_seed_results.csv",
        "round1_final_test_selected_model_counts.csv",
        "round1_final_test_stratified_summary.csv",
        "round1_final_test_metadata.json",
        "round1_final_test_summary.md",
        "status.json",
    ]


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """函数功能：创建输出目录，并在未 overwrite 时保护既有 final eval 产物。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in required_outputs() if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"输出目录已有 final eval 产物；如需覆盖请传 --overwrite：{existing}")
    if overwrite:
        for name in required_outputs():
            path = output_dir / name
            if path.exists():
                path.unlink()
        for path in output_dir.glob("predictions_*.csv"):
            path.unlink()


def write_status(output_dir: Path, payload: Mapping[str, object]) -> None:
    """函数功能：写 status.json，记录 final eval 当前阶段。"""
    data = dict(payload)
    data["updated_at"] = display_time()
    data["output_dir"] = str(output_dir)
    write_json(output_dir / "status.json", data)


def ensure_frozen_best(args: argparse.Namespace, seeds: Sequence[int]) -> Mapping[str, object]:
    """
    函数功能：
        校验 P2d best metadata，确保当前 final eval 没有改变 variant/seed 来源。
    """
    if str(args.variant) != FROZEN_BEST_VARIANT:
        raise ValueError(f"final eval variant 必须固定为 {FROZEN_BEST_VARIANT}，实际为 {args.variant}")
    best_path = Path(args.p2d_dir) / "round1_concat_best_variant.json"
    if not best_path.exists():
        raise FileNotFoundError(f"找不到 P2d best metadata：{best_path}")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    if str(best.get("best_variant")) != FROZEN_BEST_VARIANT:
        raise ValueError(f"P2d best variant 与协议不一致：{best.get('best_variant')}")
    if bool(best.get("pilot_test_used_for_selection")):
        raise ValueError("P2d metadata 显示 pilot_test_used_for_selection=true，不能做 frozen final eval")
    for seed in seeds:
        ckpt = Path(args.p2d_dir) / f"checkpoint_{FROZEN_BEST_VARIANT}_seed{int(seed)}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"找不到 P2d frozen checkpoint：{ckpt}")
    return best


def run_feature_builder_if_needed(args: argparse.Namespace) -> None:
    """
    函数功能：
        确保 pilot_test final_test_only feature cache 存在。

    说明：
        正式 P2a 默认不含 pilot_test；这里显式调用 feature builder 的
        final_test_only 开关，并写入独立目录，避免污染训练/选择 cache。
    """
    manifest_path = Path(args.final_feature_dir) / "round1_feature_manifest.csv"
    metadata_path = Path(args.final_feature_dir) / "round1_feature_metadata.json"
    if manifest_path.exists() and metadata_path.exists() and bool(args.skip_feature_build):
        log_stage(f"按 --skip-feature-build 复用 final_test_only feature cache：{args.final_feature_dir}")
        return
    if manifest_path.exists() and metadata_path.exists() and not bool(args.overwrite):
        log_stage(f"复用已存在 final_test_only feature cache：{args.final_feature_dir}")
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round1_features.py"),
        "--p0-sample-dir",
        str(args.sample_dir),
        "--round0-dir",
        str(args.round0_dir),
        "--output-dir",
        str(args.final_feature_dir),
        "--sample-sets",
        "pilot_test",
        "--include-pilot-test-final-test-only",
        "--shard-size",
        str(int(args.feature_shard_size)),
        "--embedding-batch-size",
        str(int(args.embedding_batch_size)),
        "--local-files-only",
    ]
    if str(args.device) != "auto":
        cmd.extend(["--device", str(args.device)])
    if bool(args.vit_data_parallel):
        cmd.append("--vit-data-parallel")
    if bool(args.overwrite):
        cmd.append("--overwrite")
    log_stage("生成 pilot_test final_test_only feature cache")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def validate_final_feature_cache(feature_dir: Path, expected_count: int) -> Path:
    """
    函数功能：
        校验 final_test_only feature cache 的 sample_set、数量和 metadata 标记。
    """
    manifest_path = Path(feature_dir) / "round1_feature_manifest.csv"
    metadata_path = Path(feature_dir) / "round1_feature_metadata.json"
    if not manifest_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"final_test_only feature cache 不完整：{feature_dir}")
    manifest = pd.read_csv(manifest_path)
    if set(manifest["sample_set"].astype(str)) != {"pilot_test"}:
        raise ValueError("final_test_only feature manifest 必须只包含 pilot_test")
    if "final_test_only" not in manifest.columns or not manifest["final_test_only"].astype(bool).all():
        raise ValueError("final_test_only feature manifest 未全部标记 final_test_only=true")
    count = int(manifest["sample_count"].sum())
    if count != int(expected_count):
        raise ValueError(f"pilot_test feature 数量不一致：expected={expected_count} actual={count}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "pilot_test" not in [str(value) for value in metadata.get("final_test_only_sets", [])]:
        raise ValueError("feature metadata 缺少 final_test_only_sets=['pilot_test'] 标记")
    if bool(metadata.get("feature_constraints", {}).get("train_router_or_encoder", True)):
        raise ValueError("feature metadata 显示训练了 router/encoder，不符合 final_test_only feature 约束")
    return manifest_path


def ensure_prediction_index(
    *,
    output_dir: Path,
    prediction_manifest_path: Path,
    sample_keys: Sequence[str],
    chunk_read_rows: int,
) -> SQLitePredictionIndex:
    """函数功能：为 pilot_test sample_key 构建或复用 SQLite prediction 子集索引。"""
    index_path = Path(output_dir) / "prediction_index_round1_final_test_pilot_test.sqlite"
    expected_records = len(set(str(key) for key in sample_keys)) * len(MODEL_COLUMNS)
    if index_path.exists():
        import sqlite3

        connection = sqlite3.connect(str(index_path))
        try:
            count = int(connection.execute("SELECT COUNT(*) FROM prediction_index").fetchone()[0])
        finally:
            connection.close()
        if count == expected_records:
            log_stage(f"复用 pilot_test prediction SQLite index：{index_path}")
            return SQLitePredictionIndex(index_path, Path(prediction_manifest_path).parent)
        index_path.unlink()
    log_stage("构建 pilot_test prediction SQLite index")
    return build_lightweight_prediction_index(
        prediction_manifest_path,
        sample_keys=[str(key) for key in sample_keys],
        chunk_read_rows=int(chunk_read_rows),
        index_db_path=index_path,
    )


def load_round1_router(checkpoint_path: Path, *, device: torch.device) -> tuple[VisualMLPRouter, object, Mapping[str, object]]:
    """函数功能：加载 P2d frozen checkpoint，恢复 router 和 scaler。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("variant")) != FROZEN_BEST_VARIANT:
        raise ValueError(f"checkpoint variant 不一致：{checkpoint_path} variant={checkpoint.get('variant')}")
    if [str(value) for value in checkpoint.get("model_columns", [])] != list(MODEL_COLUMNS):
        raise ValueError(f"checkpoint model_columns 与当前动作空间不一致：{checkpoint_path}")
    scaler = scaler_from_state(checkpoint["scaler_state"])
    hyper = checkpoint.get("hyperparameters", {})
    router = VisualMLPRouter(
        input_dim=int(scaler.n_features_in_),
        hidden_dim=int(hyper.get("hidden_dim", 64)),
        output_dim=len(MODEL_COLUMNS),
        dropout=float(hyper.get("dropout", 0.0)),
    ).to(device)
    router.load_state_dict(checkpoint["router_state_dict"])
    router.eval()
    return router, scaler, checkpoint


def make_round0_test_rows(sample_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """函数功能：从 Round0 full-scale test CSV 抽取同一 P0 pilot_test sample set 的对照行。"""
    keys = sample_df["sample_key"].astype(str).tolist()
    visual = extract_csv_rows_by_sample_keys(
        Path(args.visual_dir) / "visual_router_soft_fusion_predictions.csv",
        sample_keys=keys,
        chunksize=int(args.csv_chunksize),
    )
    visual = align_with_sample_frame(sample_df, visual)
    timefuse = extract_csv_rows_by_sample_keys(
        Path(args.timefuse_dir) / "timefuse_fusor_predictions.csv",
        sample_keys=keys,
        chunksize=int(args.csv_chunksize),
    )
    timefuse = align_with_sample_frame(sample_df, timefuse)
    frames = [
        make_method_rows(
            sample_set="pilot_test",
            method="round0_original_visual_hard_top1",
            pred_df=visual,
            mae_col="hard_top1_mae_from_array",
            mse_col="hard_top1_mse_from_array",
        ),
        make_method_rows(
            sample_set="pilot_test",
            method="round0_original_visual_raw_soft_fusion",
            pred_df=visual,
            mae_col="soft_fusion_mae",
            mse_col="soft_fusion_mse",
        ),
        make_method_rows(
            sample_set="pilot_test",
            method="round0_timefuse_hard_top1",
            pred_df=timefuse,
            mae_col="hard_top1_mae_from_array",
            mse_col="hard_top1_mse_from_array",
        ),
        make_method_rows(
            sample_set="pilot_test",
            method="round0_timefuse_raw_soft_fusion",
            pred_df=timefuse,
            mae_col="soft_fusion_mae",
            mse_col="soft_fusion_mse",
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def make_round1_method_rows(pred_df: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """函数功能：把单 seed P2d prediction rows 转为 hard/raw-soft 统一评估行。"""
    hard = make_method_rows(
        sample_set="pilot_test",
        method=f"{FROZEN_BEST_VARIANT}_hard_top1",
        pred_df=pred_df,
        mae_col="hard_top1_mae_from_array",
        mse_col="hard_top1_mse_from_array",
    )
    soft = make_method_rows(
        sample_set="pilot_test",
        method=f"{FROZEN_BEST_VARIANT}_raw_soft_fusion",
        pred_df=pred_df,
        mae_col="soft_fusion_mae",
        mse_col="soft_fusion_mse",
    )
    rows = pd.concat([hard, soft], ignore_index=True)
    rows.insert(1, "variant", FROZEN_BEST_VARIANT)
    rows.insert(2, "seed", int(seed))
    return rows


def make_final_comparison(round1_seed_summary: pd.DataFrame, baseline_rows: pd.DataFrame) -> pd.DataFrame:
    """函数功能：生成最终 comparison 表，同时保留 P2d seed mean/std 与 baseline 单值。"""
    round1_mean_std = summarize_mean_std(round1_seed_summary, sample_set="pilot_test")
    round1_rows: List[Dict[str, object]] = []
    for row in round1_mean_std.itertuples(index=False):
        data = row._asdict()
        method = str(data["method"])
        round1_rows.append(
            {
                "sample_set": "pilot_test",
                "method": method,
                "variant": FROZEN_BEST_VARIANT,
                "seed_count": int(data["seed_count"]),
                "sample_count": int(data["sample_count_per_seed"]),
                "hard_top1_MAE": float(data["MAE_mean"]) if method.endswith("_hard_top1") else np.nan,
                "hard_top1_MSE": float(data["MSE_mean"]) if method.endswith("_hard_top1") else np.nan,
                "hard_top1_regret_to_oracle": float(data["regret_to_oracle_mean"]) if method.endswith("_hard_top1") else np.nan,
                "hard_top1_oracle_label_accuracy": float(data["oracle_label_accuracy_mean"]) if method.endswith("_hard_top1") else np.nan,
                "raw_soft_fusion_MAE": float(data["MAE_mean"]) if method.endswith("_raw_soft_fusion") else np.nan,
                "raw_soft_fusion_MSE": float(data["MSE_mean"]) if method.endswith("_raw_soft_fusion") else np.nan,
                "raw_soft_fusion_regret_to_oracle": float(data["regret_to_oracle_mean"]) if method.endswith("_raw_soft_fusion") else np.nan,
                "raw_soft_fusion_oracle_label_accuracy": float(data["oracle_label_accuracy_mean"]) if method.endswith("_raw_soft_fusion") else np.nan,
                "weight_entropy": float(data["weight_entropy_mean"]),
                "normalized_weight_entropy": float(data["normalized_weight_entropy_mean"]),
                "mean_max_weight": float(data["mean_max_weight_mean"]),
                "MAE_std": float(data["MAE_std"]),
                "MSE_std": float(data["MSE_std"]),
                "regret_to_oracle_std": float(data["regret_to_oracle_std"]),
                "oracle_label_accuracy_std": float(data["oracle_label_accuracy_std"]),
            }
        )
    base_summary = summarize_method_rows(baseline_rows)
    baseline_out: List[Dict[str, object]] = []
    for row in base_summary.itertuples(index=False):
        method = str(row.method)
        is_soft = method.endswith("_raw_soft_fusion")
        is_hard = method.endswith("_hard_top1") or method in {"global_best_single", "oracle_top1"}
        baseline_out.append(
            {
                "sample_set": str(row.sample_set),
                "method": method,
                "variant": "",
                "seed_count": 1,
                "sample_count": int(row.sample_count),
                "hard_top1_MAE": float(row.MAE) if is_hard else np.nan,
                "hard_top1_MSE": float(row.MSE) if is_hard else np.nan,
                "hard_top1_regret_to_oracle": float(row.regret_to_oracle) if is_hard else np.nan,
                "hard_top1_oracle_label_accuracy": float(row.oracle_label_accuracy) if is_hard else np.nan,
                "raw_soft_fusion_MAE": float(row.MAE) if is_soft else np.nan,
                "raw_soft_fusion_MSE": float(row.MSE) if is_soft else np.nan,
                "raw_soft_fusion_regret_to_oracle": float(row.regret_to_oracle) if is_soft else np.nan,
                "raw_soft_fusion_oracle_label_accuracy": float(row.oracle_label_accuracy) if is_soft else np.nan,
                "weight_entropy": float(row.weight_entropy) if not pd.isna(row.weight_entropy) else np.nan,
                "normalized_weight_entropy": float(row.normalized_weight_entropy) if not pd.isna(row.normalized_weight_entropy) else np.nan,
                "mean_max_weight": float(row.mean_max_weight) if not pd.isna(row.mean_max_weight) else np.nan,
                "MAE_std": 0.0,
                "MSE_std": 0.0,
                "regret_to_oracle_std": 0.0,
                "oracle_label_accuracy_std": 0.0,
            }
        )
    out = pd.DataFrame(round1_rows + baseline_out)
    order = {
        f"{FROZEN_BEST_VARIANT}_hard_top1": 0,
        f"{FROZEN_BEST_VARIANT}_raw_soft_fusion": 1,
        "round0_timefuse_hard_top1": 2,
        "round0_timefuse_raw_soft_fusion": 3,
        "round0_original_visual_hard_top1": 4,
        "round0_original_visual_raw_soft_fusion": 5,
        "global_best_single": 6,
        "oracle_top1": 7,
    }
    out["_order"] = out["method"].map(order).fillna(999)
    return out.sort_values(["_order", "method"], kind="mergesort").drop(columns=["_order"]).reset_index(drop=True)


def build_stratified_summary(round1_rows: pd.DataFrame, baseline_rows: pd.DataFrame) -> pd.DataFrame:
    """
    函数功能：
        输出 final eval 分层 summary，包含单字段分层和完整 TSF cell 联合分层。
    """
    round1_summary_frames: List[pd.DataFrame] = []
    for col in TSF_STRATA_COLUMNS:
        grouped = summarize_rows_with_seed(round1_rows, group_cols=[col]).rename(columns={col: "stratum_value"})
        grouped.insert(4, "stratum_column", col)
        grouped.insert(5, "stratum_kind", "single_column")
        round1_summary_frames.append(grouped)
    tsf_cell = summarize_rows_with_seed(round1_rows, group_cols=TSF_STRATA_COLUMNS)
    tsf_cell.insert(4, "stratum_column", "tsf_cell")
    tsf_cell.insert(5, "stratum_kind", "tsf_cell")
    tsf_cell["stratum_value"] = tsf_cell[TSF_STRATA_COLUMNS].astype(str).agg("|".join, axis=1)
    round1_summary_frames.append(tsf_cell)
    round1_summary = pd.concat(round1_summary_frames, ignore_index=True)
    baseline_frames: List[pd.DataFrame] = []
    for col in TSF_STRATA_COLUMNS:
        grouped = summarize_method_rows(baseline_rows, group_cols=[col]).rename(columns={col: "stratum_value"})
        grouped.insert(2, "variant", "")
        grouped.insert(3, "seed", -1)
        grouped.insert(4, "stratum_column", col)
        grouped.insert(5, "stratum_kind", "single_column")
        baseline_frames.append(grouped)
    baseline_tsf = summarize_method_rows(baseline_rows, group_cols=TSF_STRATA_COLUMNS)
    baseline_tsf.insert(2, "variant", "")
    baseline_tsf.insert(3, "seed", -1)
    baseline_tsf.insert(4, "stratum_column", "tsf_cell")
    baseline_tsf.insert(5, "stratum_kind", "tsf_cell")
    baseline_tsf["stratum_value"] = baseline_tsf[TSF_STRATA_COLUMNS].astype(str).agg("|".join, axis=1)
    baseline_frames.append(baseline_tsf)
    baseline_summary = pd.concat(baseline_frames, ignore_index=True)
    return pd.concat([round1_summary, baseline_summary], ignore_index=True)


def selected_model_counts_final(round1_rows: pd.DataFrame, baseline_rows: pd.DataFrame) -> pd.DataFrame:
    """
    函数功能：
        输出 final eval selected_model count/ratio。

    说明：
        P2d 三个 seed 是独立 frozen checkpoint，不能把三个 seed 的逐样本行混在
        一个 `mean_over_seeds` 分母里统计；这里保留 per-seed counts，baseline
        仍按单个确定性方法统计。
    """
    round1_counts = (
        round1_rows.groupby(["sample_set", "variant", "seed", "method", "selected_model"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = round1_counts.groupby(["sample_set", "variant", "seed", "method"])["count"].transform("sum")
    round1_counts["ratio"] = round1_counts["count"] / totals
    baseline_counts = selected_model_counts(baseline_rows)
    baseline_counts.insert(1, "variant", "")
    baseline_counts.insert(2, "seed", "baseline")
    columns = ["sample_set", "variant", "seed", "method", "selected_model", "count", "ratio"]
    return (
        pd.concat([round1_counts[columns], baseline_counts[columns]], ignore_index=True)
        .sort_values(["sample_set", "variant", "seed", "method", "selected_model"], kind="mergesort")
        .reset_index(drop=True)
    )


def metric_lookup(comparison: pd.DataFrame, method: str, col: str) -> float:
    """函数功能：从 comparison 表取单个指标。"""
    values = comparison.loc[comparison["method"].astype(str) == method, col]
    if values.empty:
        return float("nan")
    return float(values.iloc[0])


def write_summary_md(
    *,
    output_dir: Path,
    comparison: pd.DataFrame,
    seed_results: pd.DataFrame,
    stratified: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    """函数功能：写中文 final eval 摘要，逐项回答验收问题。"""
    p2d_soft = f"{FROZEN_BEST_VARIANT}_raw_soft_fusion"
    p2d_hard = f"{FROZEN_BEST_VARIANT}_hard_top1"
    tf_soft = "round0_timefuse_raw_soft_fusion"
    visual_soft = "round0_original_visual_raw_soft_fusion"
    p2d_soft_mae = metric_lookup(comparison, p2d_soft, "raw_soft_fusion_MAE")
    tf_soft_mae = metric_lookup(comparison, tf_soft, "raw_soft_fusion_MAE")
    p2d_soft_mse = metric_lookup(comparison, p2d_soft, "raw_soft_fusion_MSE")
    visual_soft_mse = metric_lookup(comparison, visual_soft, "raw_soft_fusion_MSE")
    p2d_soft_regret = metric_lookup(comparison, p2d_soft, "raw_soft_fusion_regret_to_oracle")
    tf_soft_regret = metric_lookup(comparison, tf_soft, "raw_soft_fusion_regret_to_oracle")
    p2d_acc = metric_lookup(comparison, p2d_soft, "raw_soft_fusion_oracle_label_accuracy")
    tf_acc = metric_lookup(comparison, tf_soft, "raw_soft_fusion_oracle_label_accuracy")
    p2d_hard_mae = metric_lookup(comparison, p2d_hard, "hard_top1_MAE")
    p2d_hard_regret = metric_lookup(comparison, p2d_hard, "hard_top1_regret_to_oracle")
    cross = stratified[
        (stratified["stratum_column"].astype(str) == "oracle_model")
        & (stratified["stratum_value"].astype(str) == "CrossFormer")
        & (stratified["method"].astype(str).isin([p2d_soft, tf_soft]))
    ].copy()
    patch = stratified[
        (stratified["stratum_column"].astype(str) == "oracle_model")
        & (stratified["stratum_value"].astype(str) == "PatchTST")
        & (stratified["method"].astype(str).isin([p2d_soft, tf_soft]))
    ].copy()
    lines = [
        "# Visual Router V2 Round 1 Frozen Final Test Summary",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 核心结论",
        "",
        f"1. P2d best 是否在 pilot_test raw-soft MAE 上超过 Round0 TimeFuse：{'是' if p2d_soft_mae < tf_soft_mae else '否'}。P2d={p2d_soft_mae:.6f}，TimeFuse={tf_soft_mae:.6f}，delta={p2d_soft_mae - tf_soft_mae:+.6f}。",
        f"2. regret_to_oracle 是否超过 Round0 TimeFuse：{'是' if p2d_soft_regret < tf_soft_regret else '否'}。P2d={p2d_soft_regret:.6f}，TimeFuse={tf_soft_regret:.6f}，delta={p2d_soft_regret - tf_soft_regret:+.6f}。",
        f"3. MSE 是否保留或改善原始 Visual Router 优势：{'是' if p2d_soft_mse <= visual_soft_mse else '否'}。P2d raw-soft MSE={p2d_soft_mse:.6f}，Round0 Visual raw-soft MSE={visual_soft_mse:.6f}。",
        f"4. oracle-label accuracy 是否仍低于 TimeFuse：{'是' if p2d_acc < tf_acc else '否'}。P2d={p2d_acc:.6f}，TimeFuse={tf_acc:.6f}；该差异需要结合 MAE/regret 判断，不能单独作为选择依据。",
        f"5. raw-soft 是否明显优于 hard top-1：{'是' if p2d_soft_mae < p2d_hard_mae else '否'}。P2d hard MAE={p2d_hard_mae:.6f}、raw-soft MAE={p2d_soft_mae:.6f}；hard regret={p2d_hard_regret:.6f}、raw-soft regret={p2d_soft_regret:.6f}。",
        "6. CrossFormer / PatchTST strata 见下方分层摘录；完整分层在 `round1_final_test_stratified_summary.csv`。",
        f"7. 是否存在 selection 提升、test 退化风险：{'未观察到相对 TimeFuse 的 test 退化' if p2d_soft_mae < tf_soft_mae else '观察到 test 泛化风险或优势不足'}。",
        f"8. 是否建议进入 P2e FiLM/conditional modulation：{'可以作为后续方向，但本次 final eval 不支持用 pilot_test 重新选型' if p2d_soft_mae < tf_soft_mae else '不建议直接进入，需先解决 final test 差距'}。",
        f"9. 是否建议进入 Round 2 pseudo image / view layout 消融：{'建议进入，且继续冻结 pilot_test 只作最终验证' if p2d_soft_mae < tf_soft_mae else '建议先复盘 Round 1 泛化短板，再决定 Round 2'}。",
        f"10. 是否足够支持后续 full-scale-safe pilot rerun：{'支持作为下一步候选' if p2d_soft_mae < tf_soft_mae else '证据不足'}。",
        "",
        "## Final Comparison",
        "",
        frame_to_markdown(comparison, float_digits=6),
        "",
        "## Per-Seed Result",
        "",
        frame_to_markdown(seed_results, float_digits=6),
        "",
        "## CrossFormer Stratum 摘录",
        "",
        frame_to_markdown(cross, float_digits=6),
        "",
        "## PatchTST Stratum 摘录",
        "",
        frame_to_markdown(patch, float_digits=6),
        "",
        "## 边界记录",
        "",
        f"- p2d_best_variant_path：`{metadata['p2d_best_variant_path']}`",
        "- variant 固定为 `cls_mean_concat_plus_aux`；未训练新模型；未按 pilot_test 改 seed/epoch/hyperparams。",
        "- pilot_test feature cache 独立写入 final_test_only 目录，不覆盖 P2a 原始 feature cache。",
        f"- commit hash：`{metadata['commit_hash']}`",
        "",
    ]
    (output_dir / "round1_final_test_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行 Round 1 frozen final eval，并写出全部验收产物。"""
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    prepare_output_dir(Path(args.output_dir), overwrite=bool(args.overwrite))
    write_status(args.output_dir, {"status": "started", "script_version": SCRIPT_VERSION})
    best_metadata = ensure_frozen_best(args, seeds)

    pilot_test = read_sample_csv(Path(args.sample_dir) / "pilot_test_sample_keys.csv")
    run_feature_builder_if_needed(args)
    feature_manifest_path = validate_final_feature_cache(args.final_feature_dir, expected_count=len(pilot_test))

    log_stage("读取 pilot_test oracle labels")
    test_labels = load_oracle_subset(
        Path(args.oracle_labels_path),
        pilot_test["sample_key"].astype(str).tolist(),
        batch_rows=int(args.parquet_batch_rows),
    )
    log_stage("读取 pilot_selection oracle labels 用于 global_best_single 来源")
    pilot_selection = read_sample_csv(Path(args.sample_dir) / "pilot_selection_sample_keys.csv")
    selection_labels = load_oracle_subset(
        Path(args.oracle_labels_path),
        pilot_selection["sample_key"].astype(str).tolist(),
        batch_rows=int(args.parquet_batch_rows),
    )
    global_best_model = choose_global_best_model(selection_labels)

    prediction_index = ensure_prediction_index(
        output_dir=args.output_dir,
        prediction_manifest_path=args.prediction_manifest_path,
        sample_keys=pilot_test["sample_key"].astype(str).tolist(),
        chunk_read_rows=int(args.csv_chunksize),
    )
    device = resolve_device(str(args.device))
    log_stage("读取 final_test_only concat features")
    test_features = load_concat_features(
        feature_manifest_path=feature_manifest_path,
        sample_df=pilot_test,
        sample_set="pilot_test",
        variant=FROZEN_BEST_VARIANT,
    )
    all_round1_rows: List[pd.DataFrame] = []
    try:
        for seed in seeds:
            checkpoint_path = Path(args.p2d_dir) / f"checkpoint_{FROZEN_BEST_VARIANT}_seed{int(seed)}.pt"
            log_stage(f"评估 frozen checkpoint：seed={seed}")
            router, scaler, checkpoint = load_round1_router(checkpoint_path, device=device)
            pred = predict_visual_pooling_router(
                router=router,
                scaler=scaler,
                features=test_features,
                sample_df=pilot_test,
                labels_df=test_labels,
                variant=FROZEN_BEST_VARIANT,
                seed=int(seed),
                sample_set="pilot_test",
                device=device,
            )
            pred["router_name"] = f"p2d_{FROZEN_BEST_VARIANT}_seed{int(seed)}_frozen_final_test"
            pred = add_batch_fusion_metrics(
                pred,
                prediction_index=prediction_index,
                metric="mae",
                batch_size=int(args.eval_batch_size),
            )
            pred.to_csv(args.output_dir / f"predictions_{FROZEN_BEST_VARIANT}_seed{int(seed)}_pilot_test.csv", index=False)
            all_round1_rows.append(make_round1_method_rows(pred, seed=int(seed)))
            write_status(args.output_dir, {"status": "running", "completed_seed": int(seed)})
    finally:
        prediction_index.close()

    round1_rows = pd.concat(all_round1_rows, ignore_index=True)
    seed_results = summarize_rows_with_seed(round1_rows)
    log_stage("抽取 Round0 Visual/TimeFuse pilot_test 对照")
    round0_rows = make_round0_test_rows(pilot_test, args)
    baseline_rows = pd.concat(
        [
            round0_rows,
            add_oracle_and_global_rows(
                sample_set="pilot_test",
                sample_df=pilot_test,
                label_df=test_labels,
                global_best_model=global_best_model,
            ),
        ],
        ignore_index=True,
    )
    comparison = make_final_comparison(seed_results, baseline_rows)
    selected_counts = selected_model_counts_final(round1_rows, baseline_rows)
    stratified = build_stratified_summary(round1_rows, baseline_rows)

    seed_results.to_csv(args.output_dir / "round1_final_test_variant_seed_results.csv", index=False)
    comparison.to_csv(args.output_dir / "round1_final_test_comparison.csv", index=False)
    selected_counts.to_csv(args.output_dir / "round1_final_test_selected_model_counts.csv", index=False)
    stratified.to_csv(args.output_dir / "round1_final_test_stratified_summary.csv", index=False)

    feature_metadata = json.loads((Path(args.final_feature_dir) / "round1_feature_metadata.json").read_text(encoding="utf-8"))
    metadata = {
        "status": "completed",
        "generated_at": display_time(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "commit_hash": git_commit_hash(),
        "p2d_best_variant_path": str(Path(args.p2d_dir) / "round1_concat_best_variant.json"),
        "p2d_best_variant": FROZEN_BEST_VARIANT,
        "p2d_best_metadata": best_metadata,
        "pilot_test_used_for_selection": False,
        "pilot_test_not_used_for_selection": True,
        "pilot_test_feature_final_test_only": True,
        "trained_new_model": False,
        "changed_variant": False,
        "changed_seed_by_test": False,
        "changed_epoch_by_test": False,
        "changed_hyperparams_by_test": False,
        "input_sample_paths": {
            "pilot_test_sample_keys": str(Path(args.sample_dir) / "pilot_test_sample_keys.csv"),
            "pilot_selection_sample_keys_for_global_best_only": str(Path(args.sample_dir) / "pilot_selection_sample_keys.csv"),
        },
        "reference_paths": {
            "round0_dir": str(args.round0_dir),
            "round0_visual_dir": str(args.visual_dir),
            "round0_timefuse_dir": str(args.timefuse_dir),
            "oracle_labels_path": str(args.oracle_labels_path),
            "prediction_manifest_path": str(args.prediction_manifest_path),
            "p2a_feature_dir": str(args.p2a_feature_dir),
            "final_test_only_feature_dir": str(args.final_feature_dir),
            "final_test_only_feature_manifest": str(feature_manifest_path),
            "prediction_index_path": str(Path(args.output_dir) / "prediction_index_round1_final_test_pilot_test.sqlite"),
        },
        "p2d_checkpoint_model_paths": [
            str(Path(args.p2d_dir) / f"checkpoint_{FROZEN_BEST_VARIANT}_seed{int(seed)}.pt") for seed in seeds
        ],
        "sample_counts": {
            "pilot_test": int(len(pilot_test)),
            "pilot_selection_for_global_best": int(len(pilot_selection)),
        },
        "seeds_evaluated": [int(seed) for seed in seeds],
        "main_conclusion_uses": "3 seeds mean/std; no seed selected by pilot_test",
        "global_best_single_source": "pilot_selection_vali",
        "global_best_single_model": global_best_model,
        "feature_metadata_final_test_only": {
            "sample_sets": feature_metadata.get("sample_sets"),
            "final_test_only_sets": feature_metadata.get("final_test_only_sets"),
            "feature_constraints": feature_metadata.get("feature_constraints"),
        },
        "constraints": {
            "used_film": False,
            "used_gating": False,
            "used_attention": False,
            "changed_view_layout": False,
            "rebuilt_p2a_feature": False,
            "overwrote_p0_p1_p2a_p2d": False,
            "loaded_116m_prediction_manifest_to_memory": False,
            "saved_pseudo_image_tensor": False,
            "full_scale_eval": False,
            "used_full_17_dim_timefuse_feature": False,
        },
        "outputs": {name: str(Path(args.output_dir) / name) for name in required_outputs()},
    }
    write_json(args.output_dir / "round1_final_test_metadata.json", metadata)
    write_summary_md(
        output_dir=args.output_dir,
        comparison=comparison,
        seed_results=seed_results,
        stratified=stratified,
        metadata=metadata,
    )
    write_status(args.output_dir, {"status": "completed", "sample_count": int(len(pilot_test))})
    log_stage(f"Round 1 final test outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
