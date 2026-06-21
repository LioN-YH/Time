#!/usr/bin/env python3
"""
文件功能：
    对 Visual Router V2 Round 1 三个关键候选变体做 frozen pilot_test final
    eval extension，并把结果与已完成 P2d best final eval、Round0 baseline、
    global_best_single 和 oracle_top1 放入同一张对比表。

核心边界：
    - 只加载 Round 1 已保存 checkpoint 和 scaler，不训练新模型、不调参；
    - `cls_mean_concat_plus_aux` 作为已完成 P2d best 引用，不改变 best 结论；
    - 新增评测只覆盖 mean_patch_plus_aux、visual_mean_patch_only、
      visual_cls_mean_concat 的 seed 16/17/18；
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
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_round1_training import (  # noqa: E402
    load_pooling_features,
)


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_SAMPLE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_pilot_samples"
DEFAULT_ROUND0_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round0"
DEFAULT_P2A_FEATURE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_features"
DEFAULT_P2D_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_concat"
DEFAULT_P2B_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_visual_pooling"
DEFAULT_FINAL_FEATURE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_features_final_test_only"
DEFAULT_P2D_FINAL_TEST_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_final_test"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_final_test_extension"
SCRIPT_VERSION = "visual_router_v2_round1_final_test_extension_v1"
FROZEN_BEST_VARIANT = "cls_mean_concat_plus_aux"
EXTENSION_VARIANTS = ("mean_patch_plus_aux", "visual_mean_patch_only", "visual_cls_mean_concat")
ALL_REPORT_VARIANTS = (FROZEN_BEST_VARIANT, *EXTENSION_VARIANTS)
VARIANT_CHECKPOINT_DIR_ATTR = {
    "mean_patch_plus_aux": "p2d_dir",
    "visual_mean_patch_only": "p2b_dir",
    "visual_cls_mean_concat": "p2b_dir",
}


def display_time() -> str:
    """函数功能：生成本地日志和 metadata 时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 frozen final eval extension 参数；默认路径均来自 Round 1 协议。"""
    parser = argparse.ArgumentParser(description="Frozen final eval extension for Visual Router V2 Round 1 key variants on P0 pilot_test.")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--round0-dir", type=Path, default=DEFAULT_ROUND0_DIR)
    parser.add_argument("--p2a-feature-dir", type=Path, default=DEFAULT_P2A_FEATURE_DIR)
    parser.add_argument("--p2d-dir", type=Path, default=DEFAULT_P2D_DIR)
    parser.add_argument("--p2b-dir", type=Path, default=DEFAULT_P2B_DIR)
    parser.add_argument("--final-feature-dir", type=Path, default=DEFAULT_FINAL_FEATURE_DIR)
    parser.add_argument("--p2d-final-test-dir", type=Path, default=DEFAULT_P2D_FINAL_TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--oracle-labels-path", type=Path, default=DEFAULT_ORACLE_LABELS)
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST)
    parser.add_argument("--visual-dir", type=Path, default=DEFAULT_VISUAL_DIR)
    parser.add_argument("--timefuse-dir", type=Path, default=DEFAULT_TIMEFUSE_DIR)
    parser.add_argument("--extension-variants", default=",".join(EXTENSION_VARIANTS), help="只补测这些 frozen variants；P2d best 从既有 final eval 引入。")
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
    """函数功能：列出 extension 顶层必需产物，避免无意覆盖。"""
    return [
        "round1_final_test_extension_comparison.csv",
        "round1_final_test_extension_variant_seed_results.csv",
        "round1_final_test_extension_selected_model_counts.csv",
        "round1_final_test_extension_stratified_summary.csv",
        "round1_final_test_extension_delta_summary.csv",
        "round1_final_test_extension_metadata.json",
        "round1_final_test_extension_summary.md",
        "status.json",
    ]


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """函数功能：创建输出目录，并在未 overwrite 时保护既有 final eval 产物。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in required_outputs() if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"输出目录已有 extension 产物；如需覆盖请传 --overwrite：{existing}")
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


def parse_variant_list(variant_text: str) -> List[str]:
    """函数功能：解析并校验本次 extension 允许补测的 frozen variants。"""
    variants: List[str] = []
    for part in str(variant_text).split(","):
        name = part.strip()
        if not name:
            continue
        if name not in EXTENSION_VARIANTS:
            raise ValueError(f"未知 extension variant={name}，允许值={EXTENSION_VARIANTS}")
        if name not in variants:
            variants.append(name)
    if not variants:
        raise ValueError("--extension-variants 不能为空")
    return variants


def ensure_frozen_sources(args: argparse.Namespace, seeds: Sequence[int], extension_variants: Sequence[str]) -> Mapping[str, object]:
    """
    函数功能：
        校验 P2d/P2b 已冻结来源，确保本次 extension 没有训练或改变 Round 1 best。
    """
    best_path = Path(args.p2d_dir) / "round1_concat_best_variant.json"
    if not best_path.exists():
        raise FileNotFoundError(f"找不到 P2d best metadata：{best_path}")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    if str(best.get("best_variant")) != FROZEN_BEST_VARIANT:
        raise ValueError(f"P2d best variant 与协议不一致：{best.get('best_variant')}")
    if bool(best.get("pilot_test_used_for_selection")):
        raise ValueError("P2d metadata 显示 pilot_test_used_for_selection=true，不能做 frozen final eval")
    for seed in seeds:
        p2d_pred = Path(args.p2d_final_test_dir) / f"predictions_{FROZEN_BEST_VARIANT}_seed{int(seed)}_pilot_test.csv"
        if not p2d_pred.exists():
            raise FileNotFoundError(f"找不到已完成 P2d final eval prediction：{p2d_pred}")
        for variant in extension_variants:
            ckpt = checkpoint_path_for_variant(args, variant, int(seed))
            if not ckpt.exists():
                raise FileNotFoundError(f"找不到 frozen checkpoint：variant={variant} seed={seed} path={ckpt}")
    return best


def checkpoint_path_for_variant(args: argparse.Namespace, variant: str, seed: int) -> Path:
    """函数功能：按 variant 定位对应 P2b/P2d frozen checkpoint。"""
    source_attr = VARIANT_CHECKPOINT_DIR_ATTR.get(str(variant))
    if source_attr is None:
        raise ValueError(f"不支持的 extension variant={variant}")
    return Path(getattr(args, source_attr)) / f"checkpoint_{variant}_seed{int(seed)}.pt"


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


def load_round1_router(checkpoint_path: Path, *, variant: str, device: torch.device) -> tuple[VisualMLPRouter, object, Mapping[str, object]]:
    """函数功能：加载指定 variant 的 frozen checkpoint，恢复 router 和 scaler。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("variant")) != str(variant):
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


def load_features_for_variant(feature_manifest_path: Path, sample_df: pd.DataFrame, variant: str) -> np.ndarray:
    """函数功能：按 variant 从同一 final_test_only feature cache 构造现场输入特征。"""
    if variant == "mean_patch_plus_aux":
        return load_concat_features(
            feature_manifest_path=feature_manifest_path,
            sample_df=sample_df,
            sample_set="pilot_test",
            variant=variant,
        )
    if variant in {"visual_mean_patch_only", "visual_cls_mean_concat"}:
        return load_pooling_features(
            feature_manifest_path=feature_manifest_path,
            sample_df=sample_df,
            sample_set="pilot_test",
            variant=variant,
        )
    raise ValueError(f"不支持的 extension variant={variant}")


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


def make_round1_method_rows(pred_df: pd.DataFrame, *, variant: str, seed: int) -> pd.DataFrame:
    """函数功能：把单 seed prediction rows 转为 hard/raw-soft 统一评估行。"""
    hard = make_method_rows(
        sample_set="pilot_test",
        method=f"{variant}_hard_top1",
        pred_df=pred_df,
        mae_col="hard_top1_mae_from_array",
        mse_col="hard_top1_mse_from_array",
    )
    soft = make_method_rows(
        sample_set="pilot_test",
        method=f"{variant}_raw_soft_fusion",
        pred_df=pred_df,
        mae_col="soft_fusion_mae",
        mse_col="soft_fusion_mse",
    )
    rows = pd.concat([hard, soft], ignore_index=True)
    rows.insert(1, "variant", variant)
    rows.insert(2, "seed", int(seed))
    return rows


def make_final_comparison(round1_seed_summary: pd.DataFrame, baseline_rows: pd.DataFrame) -> pd.DataFrame:
    """函数功能：生成 extension comparison 表，保留多 variant seed mean/std 与 baseline 单值。"""
    round1_mean_std = summarize_mean_std(round1_seed_summary, sample_set="pilot_test")
    round1_rows: List[Dict[str, object]] = []
    for row in round1_mean_std.itertuples(index=False):
        data = row._asdict()
        method = str(data["method"])
        variant = str(data["variant"])
        round1_rows.append(
            {
                "sample_set": "pilot_test",
                "method": method,
                "variant": variant,
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
    order: Dict[str, int] = {}
    idx = 0
    for variant in ALL_REPORT_VARIANTS:
        order[f"{variant}_hard_top1"] = idx
        order[f"{variant}_raw_soft_fusion"] = idx + 1
        idx += 2
    order.update(
        {
            "round0_timefuse_hard_top1": idx,
            "round0_timefuse_raw_soft_fusion": idx + 1,
            "round0_original_visual_hard_top1": idx + 2,
            "round0_original_visual_raw_soft_fusion": idx + 3,
            "global_best_single": idx + 4,
            "oracle_top1": idx + 5,
        }
    )
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


def build_delta_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """
    函数功能：
        输出用户关心的 pairwise delta；delta 定义为 lhs - rhs，误差类指标越小越好。
    """
    pairs = [
        ("mean_patch_plus_aux", "cls_mean_concat_plus_aux", "mean_patch_plus_aux vs cls_mean_concat_plus_aux"),
        ("mean_patch_plus_aux", "visual_mean_patch_only", "mean_patch_plus_aux vs visual_mean_patch_only"),
        ("cls_mean_concat_plus_aux", "visual_cls_mean_concat", "cls_mean_concat_plus_aux vs visual_cls_mean_concat"),
        ("visual_mean_patch_only", "round0_timefuse", "visual_mean_patch_only vs Round0 TimeFuse"),
        ("visual_cls_mean_concat", "round0_timefuse", "visual_cls_mean_concat vs Round0 TimeFuse"),
        ("cls_mean_concat_plus_aux", "round0_timefuse", "cls_mean_concat_plus_aux vs Round0 TimeFuse"),
    ]
    metrics = [
        ("hard_top1_MAE", "hard_top1"),
        ("hard_top1_MSE", "hard_top1"),
        ("hard_top1_regret_to_oracle", "hard_top1"),
        ("hard_top1_oracle_label_accuracy", "hard_top1"),
        ("raw_soft_fusion_MAE", "raw_soft_fusion"),
        ("raw_soft_fusion_MSE", "raw_soft_fusion"),
        ("raw_soft_fusion_regret_to_oracle", "raw_soft_fusion"),
        ("raw_soft_fusion_oracle_label_accuracy", "raw_soft_fusion"),
        ("weight_entropy", "raw_soft_fusion"),
        ("normalized_weight_entropy", "raw_soft_fusion"),
        ("mean_max_weight", "raw_soft_fusion"),
        ("MAE_std", "raw_soft_fusion"),
        ("MSE_std", "raw_soft_fusion"),
        ("regret_to_oracle_std", "raw_soft_fusion"),
        ("oracle_label_accuracy_std", "raw_soft_fusion"),
    ]
    rows: List[Dict[str, object]] = []
    for lhs, rhs, label in pairs:
        for metric_col, method_kind in metrics:
            lhs_method = f"{lhs}_{method_kind}" if lhs != "round0_timefuse" else f"round0_timefuse_{method_kind}"
            rhs_method = f"{rhs}_{method_kind}" if rhs != "round0_timefuse" else f"round0_timefuse_{method_kind}"
            lhs_value = metric_lookup(comparison, lhs_method, metric_col)
            rhs_value = metric_lookup(comparison, rhs_method, metric_col)
            rows.append(
                {
                    "comparison": label,
                    "lhs": lhs,
                    "rhs": rhs,
                    "method_kind": method_kind,
                    "metric": metric_col,
                    "lhs_value": lhs_value,
                    "rhs_value": rhs_value,
                    "delta_lhs_minus_rhs": lhs_value - rhs_value if np.isfinite(lhs_value) and np.isfinite(rhs_value) else np.nan,
                    "lower_is_better": not metric_col.endswith("accuracy") and metric_col not in {"weight_entropy", "normalized_weight_entropy", "mean_max_weight"},
                }
            )
    return pd.DataFrame(rows)


def _delta_value(delta_summary: pd.DataFrame, comparison_name: str, metric: str, method_kind: str = "raw_soft_fusion") -> float:
    """函数功能：从 delta summary 提取单个 delta，缺失时返回 NaN。"""
    subset = delta_summary[
        (delta_summary["comparison"].astype(str) == comparison_name)
        & (delta_summary["metric"].astype(str) == metric)
        & (delta_summary["method_kind"].astype(str) == method_kind)
    ]
    if subset.empty:
        return float("nan")
    return float(subset["delta_lhs_minus_rhs"].iloc[0])


def write_summary_md(
    *,
    output_dir: Path,
    comparison: pd.DataFrame,
    seed_results: pd.DataFrame,
    stratified: pd.DataFrame,
    delta_summary: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    """函数功能：写中文 extension 摘要，逐项回答验收问题。"""
    p2d_soft = f"{FROZEN_BEST_VARIANT}_raw_soft_fusion"
    tf_soft = "round0_timefuse_raw_soft_fusion"
    p2d_soft_mae = metric_lookup(comparison, p2d_soft, "raw_soft_fusion_MAE")
    mean_aux_mae = metric_lookup(comparison, "mean_patch_plus_aux_raw_soft_fusion", "raw_soft_fusion_MAE")
    visual_mean_mae = metric_lookup(comparison, "visual_mean_patch_only_raw_soft_fusion", "raw_soft_fusion_MAE")
    visual_concat_mae = metric_lookup(comparison, "visual_cls_mean_concat_raw_soft_fusion", "raw_soft_fusion_MAE")
    tf_soft_mae = metric_lookup(comparison, tf_soft, "raw_soft_fusion_MAE")
    mean_vs_p2d = _delta_value(delta_summary, "mean_patch_plus_aux vs cls_mean_concat_plus_aux", "raw_soft_fusion_MAE")
    aux_mean_delta = _delta_value(delta_summary, "mean_patch_plus_aux vs visual_mean_patch_only", "raw_soft_fusion_MAE")
    aux_concat_delta = _delta_value(delta_summary, "cls_mean_concat_plus_aux vs visual_cls_mean_concat", "raw_soft_fusion_MAE")
    mean_std = metric_lookup(comparison, "mean_patch_plus_aux_raw_soft_fusion", "MAE_std")
    p2d_std = metric_lookup(comparison, p2d_soft, "MAE_std")
    seed_soft = seed_results[
        (seed_results["variant"].astype(str) == FROZEN_BEST_VARIANT)
        & (seed_results["method"].astype(str) == p2d_soft)
    ][["seed", "MAE", "MSE", "regret_to_oracle", "oracle_label_accuracy"]].copy()
    best_seed = int(seed_soft.sort_values("MAE", kind="mergesort").iloc[0]["seed"]) if not seed_soft.empty else -1
    cross = stratified[
        (stratified["stratum_column"].astype(str) == "oracle_model")
        & (stratified["stratum_value"].astype(str) == "CrossFormer")
        & (stratified["method"].astype(str).isin([p2d_soft, "mean_patch_plus_aux_raw_soft_fusion", "visual_mean_patch_only_raw_soft_fusion", "visual_cls_mean_concat_raw_soft_fusion", tf_soft]))
    ].copy()
    patch = stratified[
        (stratified["stratum_column"].astype(str) == "oracle_model")
        & (stratified["stratum_value"].astype(str) == "PatchTST")
        & (stratified["method"].astype(str).isin([p2d_soft, "mean_patch_plus_aux_raw_soft_fusion", "visual_mean_patch_only_raw_soft_fusion", "visual_cls_mean_concat_raw_soft_fusion", tf_soft]))
    ].copy()
    lines = [
        "# Visual Router V2 Round 1 Frozen Final Test Extension Summary",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 核心结论",
        "",
        f"1. mean_patch_plus_aux 是否足够强：{'接近或超过 P2d best' if mean_vs_p2d <= 0.001 else '仍与 P2d best 有可见差距'}。mean_patch_plus_aux raw-soft MAE={mean_aux_mae:.6f}，P2d best={p2d_soft_mae:.6f}，delta={mean_vs_p2d:+.6f}。",
        f"2. aux 对 mean_patch 的 test 边际贡献：delta(mean_patch_plus_aux - visual_mean_patch_only)={aux_mean_delta:+.6f}，负值表示 aux 改善；visual_mean_patch_only raw-soft MAE={visual_mean_mae:.6f}。",
        f"3. aux 对 cls+mean 的 test 边际贡献：delta(cls_mean_concat_plus_aux - visual_cls_mean_concat)={aux_concat_delta:+.6f}，负值表示 aux 改善；visual_cls_mean_concat raw-soft MAE={visual_concat_mae:.6f}。",
        f"4. visual-only 是否已经超过 Round0 TimeFuse：visual_mean_patch_only {'超过' if visual_mean_mae < tf_soft_mae else '未超过'}，visual_cls_mean_concat {'超过' if visual_concat_mae < tf_soft_mae else '未超过'}；TimeFuse raw-soft MAE={tf_soft_mae:.6f}。",
        f"5. cls_mean_concat_plus_aux 优势是否由单一 seed 驱动：P2d best raw-soft MAE_std={p2d_std:.6f}，best seed={best_seed}；mean_patch_plus_aux MAE_std={mean_std:.6f}，是否更稳定={'是' if mean_std < p2d_std else '否'}。",
        f"6. 后续 P2e FiLM 主线建议：{'可优先 mean_patch 主线，因其更简洁且差距很小' if mean_vs_p2d <= 0.001 else '仍建议以 cls_mean_concat 为主线，mean_patch 作为简洁强基线保留'}；该建议仅基于冻结 pilot_test 解释，不改变 Round 1 best。",
        "7. CrossFormer / PatchTST strata 见下方分层摘录；完整分层在 `round1_final_test_extension_stratified_summary.csv`。",
        "",
        "## Extension Comparison",
        "",
        frame_to_markdown(comparison, float_digits=6),
        "",
        "## Delta Summary",
        "",
        frame_to_markdown(delta_summary, float_digits=6),
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
    (output_dir / "round1_final_test_extension_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行 Round 1 frozen final eval extension，并写出全部验收产物。"""
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    extension_variants = parse_variant_list(args.extension_variants)
    prepare_output_dir(Path(args.output_dir), overwrite=bool(args.overwrite))
    write_status(args.output_dir, {"status": "started", "script_version": SCRIPT_VERSION})
    best_metadata = ensure_frozen_sources(args, seeds, extension_variants)

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
    all_round1_rows: List[pd.DataFrame] = []
    try:
        for seed in seeds:
            p2d_pred_path = Path(args.p2d_final_test_dir) / f"predictions_{FROZEN_BEST_VARIANT}_seed{int(seed)}_pilot_test.csv"
            log_stage(f"读取已完成 P2d best final eval prediction：seed={seed}")
            p2d_pred = pd.read_csv(p2d_pred_path)
            p2d_pred = align_with_sample_frame(pilot_test, p2d_pred)
            all_round1_rows.append(make_round1_method_rows(p2d_pred, variant=FROZEN_BEST_VARIANT, seed=int(seed)))
        for seed in seeds:
            for variant in extension_variants:
                checkpoint_path = checkpoint_path_for_variant(args, variant, int(seed))
                log_stage(f"评估 extension frozen checkpoint：variant={variant} seed={seed}")
                test_features = load_features_for_variant(feature_manifest_path, pilot_test, variant)
                router, scaler, checkpoint = load_round1_router(checkpoint_path, variant=variant, device=device)
                pred = predict_visual_pooling_router(
                    router=router,
                    scaler=scaler,
                    features=test_features,
                    sample_df=pilot_test,
                    labels_df=test_labels,
                    variant=variant,
                    seed=int(seed),
                    sample_set="pilot_test",
                    device=device,
                )
                pred["router_name"] = f"round1_{variant}_seed{int(seed)}_frozen_final_test_extension"
                pred = add_batch_fusion_metrics(
                    pred,
                    prediction_index=prediction_index,
                    metric="mae",
                    batch_size=int(args.eval_batch_size),
                )
                pred.to_csv(args.output_dir / f"predictions_{variant}_seed{int(seed)}_pilot_test.csv", index=False)
                all_round1_rows.append(make_round1_method_rows(pred, variant=variant, seed=int(seed)))
                write_status(args.output_dir, {"status": "running", "completed_variant": variant, "completed_seed": int(seed)})
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
    delta_summary = build_delta_summary(comparison)
    selected_counts = selected_model_counts_final(round1_rows, baseline_rows)
    stratified = build_stratified_summary(round1_rows, baseline_rows)

    seed_results.to_csv(args.output_dir / "round1_final_test_extension_variant_seed_results.csv", index=False)
    comparison.to_csv(args.output_dir / "round1_final_test_extension_comparison.csv", index=False)
    selected_counts.to_csv(args.output_dir / "round1_final_test_extension_selected_model_counts.csv", index=False)
    stratified.to_csv(args.output_dir / "round1_final_test_extension_stratified_summary.csv", index=False)
    delta_summary.to_csv(args.output_dir / "round1_final_test_extension_delta_summary.csv", index=False)

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
        "extension_variants": list(extension_variants),
        "all_report_variants": list(ALL_REPORT_VARIANTS),
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
            "p2b_dir": str(args.p2b_dir),
            "p2d_dir": str(args.p2d_dir),
            "p2d_final_test_dir": str(args.p2d_final_test_dir),
            "final_test_only_feature_dir": str(args.final_feature_dir),
            "final_test_only_feature_manifest": str(feature_manifest_path),
            "prediction_index_path": str(Path(args.output_dir) / "prediction_index_round1_final_test_pilot_test.sqlite"),
        },
        "p2d_checkpoint_model_paths": [
            str(Path(args.p2d_dir) / f"checkpoint_{FROZEN_BEST_VARIANT}_seed{int(seed)}.pt") for seed in seeds
        ],
        "extension_checkpoint_model_paths": {
            variant: [str(checkpoint_path_for_variant(args, variant, int(seed))) for seed in seeds] for variant in extension_variants
        },
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
    write_json(args.output_dir / "round1_final_test_extension_metadata.json", metadata)
    write_summary_md(
        output_dir=args.output_dir,
        comparison=comparison,
        seed_results=seed_results,
        stratified=stratified,
        delta_summary=delta_summary,
        metadata=metadata,
    )
    write_status(args.output_dir, {"status": "completed", "sample_count": int(len(pilot_test))})
    log_stage(f"Round 1 final test extension outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
