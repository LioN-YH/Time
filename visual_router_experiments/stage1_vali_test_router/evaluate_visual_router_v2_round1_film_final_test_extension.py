#!/usr/bin/env python3
"""
文件功能：
    对 Visual Router V2 Round 1 P2e FiLM / aux modulation 已训练 checkpoint
    做 frozen pilot_test final eval extension。

核心边界：
    - 只加载 P2e 已保存的两个 FiLM 变体、seeds 16/17/18 checkpoint；
    - 不训练新模型、不调参、不按 pilot_test 改 variant/seed/epoch；
    - 只读取 final_test_only pilot_test feature cache，不重建 P2a feature cache；
    - 不保存 pseudo image tensor，不全量加载 116M prediction manifest；
    - 单任务只写独立 task 子目录，统一 comparison/delta/strata/summary 由 aggregation 单独生成。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.stage1_vali_test_router.evaluate_visual_router_v2_round0 import (  # noqa: E402
    DEFAULT_ORACLE_LABELS,
    DEFAULT_PREDICTION_MANIFEST,
    load_oracle_subset,
)
from visual_router_experiments.stage1_vali_test_router.fusion_utils import MODEL_COLUMNS, frame_to_markdown  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import (  # noqa: E402
    SQLitePredictionIndex,
    build_lightweight_prediction_index,
    scaler_from_state,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_v2_round1_film import (  # noqa: E402
    FEATURE_ARRAY_BY_FILM_VARIANT,
    FILM_VARIANTS,
    FiLMRouter,
    load_film_features,
    predict_film_router,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_evaluator import (  # noqa: E402
    TSF_STRATA_COLUMNS,
    align_with_sample_frame,
    read_sample_csv,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_round1_training import (  # noqa: E402
    add_batch_fusion_metrics,
    make_visual_pooling_method_rows,
    summarize_mean_std,
    summarize_rows_with_seed,
)


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_SAMPLE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_pilot_samples"
DEFAULT_FINAL_FEATURE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_features_final_test_only"
DEFAULT_FILM_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round1_film"
DEFAULT_BASELINE_FINAL_EXTENSION_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_final_test_extension"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round1_film_final_test_extension"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "experiment_summaries" / "visual_router_v2_round1" / "p2e_film_final_test_extension"
SCRIPT_VERSION = "visual_router_v2_round1_film_final_test_extension_v1"
BASELINE_VARIANTS = (
    "visual_cls_mean_concat",
    "visual_mean_patch_only",
    "cls_mean_concat_plus_aux",
    "mean_patch_plus_aux",
)
ROUND0_METHODS = {
    "round0_timefuse_hard_top1",
    "round0_timefuse_raw_soft_fusion",
    "round0_original_visual_hard_top1",
    "round0_original_visual_raw_soft_fusion",
    "global_best_single",
    "oracle_top1",
}
DELTA_PAIRS = [
    ("film_mean_patch_aux", "visual_mean_patch_only"),
    ("film_mean_patch_aux", "mean_patch_plus_aux"),
    ("film_mean_patch_aux", "visual_cls_mean_concat"),
    ("film_mean_patch_aux", "cls_mean_concat_plus_aux"),
    ("film_cls_mean_concat_aux", "visual_cls_mean_concat"),
    ("film_cls_mean_concat_aux", "cls_mean_concat_plus_aux"),
    ("film_cls_mean_concat_aux", "film_mean_patch_aux"),
    ("film_mean_patch_aux", "Round0 TimeFuse"),
    ("film_cls_mean_concat_aux", "Round0 TimeFuse"),
]


def display_time() -> str:
    """函数功能：生成日志、metadata 与 summary 使用的本地时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def log_stage(message: str) -> None:
    """函数功能：输出阶段进度，便于后台日志监控。"""
    print(f"[{display_time()}] {message}", flush=True)


def git_commit_hash() -> str:
    """函数功能：记录当前 commit；失败时写 unknown，不影响评估。"""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 P2e FiLM frozen pilot_test eval extension 参数。"""
    parser = argparse.ArgumentParser(description="Frozen pilot_test eval extension for P2e FiLM variants.")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--final-feature-dir", type=Path, default=DEFAULT_FINAL_FEATURE_DIR)
    parser.add_argument("--film-dir", type=Path, default=DEFAULT_FILM_DIR)
    parser.add_argument("--baseline-final-extension-dir", type=Path, default=DEFAULT_BASELINE_FINAL_EXTENSION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-copy-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--oracle-labels-path", type=Path, default=DEFAULT_ORACLE_LABELS)
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST)
    parser.add_argument("--variant", choices=list(FILM_VARIANTS), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--variants", default=",".join(FILM_VARIANTS))
    parser.add_argument("--seeds", default="16,17,18")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--devices-requested", default="")
    parser.add_argument("--parallel-eval-used", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--csv-chunksize", type=int, default=200_000)
    parser.add_argument("--parquet-batch-rows", type=int, default=250_000)
    parser.add_argument("--run-single", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_csv(text: str) -> List[str]:
    """函数功能：解析逗号分隔字符串，去重并保序。"""
    values: List[str] = []
    for part in str(text).split(","):
        value = part.strip()
        if value and value not in values:
            values.append(value)
    if not values:
        raise ValueError("逗号分隔参数不能为空")
    return values


def parse_seed_list(text: str) -> List[int]:
    """函数功能：解析 seed 列表。"""
    return [int(value) for value in parse_csv(text)]


def parse_variant_list(text: str) -> List[str]:
    """函数功能：解析并校验本次 frozen eval 的 FiLM 变体。"""
    variants = parse_csv(text)
    bad = [variant for variant in variants if variant not in FILM_VARIANTS]
    if bad:
        raise ValueError(f"未知 FiLM variant={bad}，允许值={FILM_VARIANTS}")
    return variants


def task_dir(output_dir: Path, variant: str, seed: int) -> Path:
    """函数功能：返回单个 variant/seed 的隔离输出目录。"""
    return Path(output_dir) / "tasks" / f"{variant}_seed{int(seed)}"


def prediction_index_path(output_dir: Path) -> Path:
    """函数功能：返回本 eval extension 专用 SQLite prediction 子集索引路径。"""
    return Path(output_dir) / "prediction_index_round1_film_final_test_pilot_test.sqlite"


def resolve_device(device_arg: str) -> torch.device:
    """函数功能：解析 frozen eval 设备；auto 优先使用 CUDA。"""
    if str(device_arg) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device_arg).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"请求 {device_arg}，但当前 CUDA 不可用")
    return torch.device(str(device_arg))


def required_outputs() -> List[str]:
    """函数功能：列出顶层必需产物。"""
    return [
        "round1_film_final_test_extension_variant_seed_results.csv",
        "round1_film_final_test_extension_comparison.csv",
        "round1_film_final_test_extension_selected_model_counts.csv",
        "round1_film_final_test_extension_stratified_summary.csv",
        "round1_film_final_test_extension_delta_summary.csv",
        "round1_film_final_test_extension_metadata.json",
        "round1_film_final_test_extension_summary.md",
        "status.json",
    ]


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """函数功能：创建输出目录，并在未 overwrite 时保护既有聚合产物。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in required_outputs() if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"输出目录已有 P2e final test extension 产物；如需覆盖请传 --overwrite：{existing}")
    if overwrite:
        for name in required_outputs():
            path = output_dir / name
            if path.exists():
                path.unlink()


def write_status(output_dir: Path, payload: Mapping[str, object]) -> None:
    """函数功能：写 status.json，记录当前阶段和接手信息。"""
    data = dict(payload)
    data["updated_at"] = display_time()
    data["output_dir"] = str(output_dir)
    write_json(Path(output_dir) / "status.json", data)


def checkpoint_path(film_dir: Path, variant: str, seed: int) -> Path:
    """函数功能：定位 P2e 已训练 FiLM checkpoint。"""
    return task_dir(Path(film_dir), variant, seed) / f"checkpoint_{variant}_seed{int(seed)}.pt"


def validate_film_sources(film_dir: Path, variants: Sequence[str], seeds: Sequence[int]) -> Mapping[str, object]:
    """函数功能：校验 P2e selection best metadata 和全部 checkpoint 存在。"""
    best_path = Path(film_dir) / "round1_film_best_variant.json"
    meta_path = Path(film_dir) / "round1_film_metadata.json"
    if not best_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"P2e FiLM 汇总 metadata 不完整：{film_dir}")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(best.get("best_variant")) != "film_mean_patch_aux":
        raise ValueError(f"P2e best variant 与目标背景不一致：{best.get('best_variant')}")
    constraints = meta.get("constraints", {})
    if bool(constraints.get("pilot_test_evaluated")) or bool(constraints.get("pilot_test_used_for_selection")):
        raise ValueError("P2e 历史 metadata 显示已使用 pilot_test，不符合 frozen extension 前提")
    missing = [str(checkpoint_path(film_dir, variant, seed)) for variant in variants for seed in seeds if not checkpoint_path(film_dir, variant, seed).exists()]
    if missing:
        raise FileNotFoundError("缺少 P2e FiLM checkpoint：" + "; ".join(missing))
    return {"best_variant_path": str(best_path), "metadata_path": str(meta_path), "best_variant": best, "metadata_constraints": constraints}


def validate_final_feature_cache(feature_dir: Path, expected_count: int) -> Path:
    """函数功能：校验 final_test_only pilot_test feature cache，不触发重建。"""
    manifest_path = Path(feature_dir) / "round1_feature_manifest.csv"
    metadata_path = Path(feature_dir) / "round1_feature_metadata.json"
    if not manifest_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"final_test_only feature cache 不完整：{feature_dir}")
    manifest = pd.read_csv(manifest_path)
    if set(manifest["sample_set"].astype(str)) != {"pilot_test"}:
        raise ValueError("final_test_only feature manifest 必须只包含 pilot_test")
    if "final_test_only" not in manifest.columns or not manifest["final_test_only"].astype(bool).all():
        raise ValueError("feature manifest 未全部标记 final_test_only=true")
    count = int(manifest["sample_count"].sum())
    if count != int(expected_count):
        raise ValueError(f"pilot_test feature 数量不一致：expected={expected_count} actual={count}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "pilot_test" not in [str(value) for value in metadata.get("final_test_only_sets", [])]:
        raise ValueError("feature metadata 缺少 final_test_only_sets=['pilot_test'] 标记")
    if bool(metadata.get("feature_constraints", {}).get("train_router_or_encoder", True)):
        raise ValueError("feature metadata 显示训练了 router/encoder，不符合 frozen eval feature 约束")
    return manifest_path


def ensure_prediction_index(
    *,
    output_dir: Path,
    baseline_final_extension_dir: Path,
    prediction_manifest_path: Path,
    sample_keys: Sequence[str],
    chunk_read_rows: int,
) -> SQLitePredictionIndex:
    """函数功能：复用或构建 pilot_test SQLite prediction 子集索引，避免加载全量 manifest。"""
    index_path = prediction_index_path(output_dir)
    baseline_index = Path(baseline_final_extension_dir) / "prediction_index_round1_final_test_pilot_test.sqlite"
    expected_records = len(set(str(key) for key in sample_keys)) * len(MODEL_COLUMNS)
    if baseline_index.exists():
        import sqlite3

        connection = sqlite3.connect(str(baseline_index))
        try:
            count = int(connection.execute("SELECT COUNT(*) FROM prediction_index").fetchone()[0])
        finally:
            connection.close()
        if count == expected_records:
            # 多个 eval 子进程会并行启动；直接只读使用已完成 P2d index，避免首次复制同一个 SQLite 文件时产生写竞争。
            log_stage(f"直接复用 P2d final extension SQLite index：{baseline_index}")
            return SQLitePredictionIndex(baseline_index, Path(prediction_manifest_path).parent)
    if index_path.exists():
        import sqlite3

        try:
            connection = sqlite3.connect(str(index_path))
            try:
                count = int(connection.execute("SELECT COUNT(*) FROM prediction_index").fetchone()[0])
            finally:
                connection.close()
        except sqlite3.DatabaseError:
            # 上一次异常中断可能留下半成品 SQLite；只删除本 extension 自己的 index。
            index_path.unlink()
            count = -1
        if count == expected_records:
            log_stage(f"复用 pilot_test prediction SQLite index：{index_path}")
            return SQLitePredictionIndex(index_path, Path(prediction_manifest_path).parent)
        if index_path.exists():
            index_path.unlink()
    log_stage("构建 pilot_test prediction SQLite index")
    return build_lightweight_prediction_index(
        prediction_manifest_path,
        sample_keys=[str(key) for key in sample_keys],
        chunk_read_rows=int(chunk_read_rows),
        index_db_path=index_path,
    )


def load_film_router(checkpoint: Path, *, variant: str, device: torch.device) -> Tuple[FiLMRouter, object, object, Mapping[str, object]]:
    """函数功能：加载 P2e FiLM checkpoint，恢复 router 和两个 StandardScaler。"""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if str(ckpt.get("variant")) != str(variant):
        raise ValueError(f"checkpoint variant 不一致：{checkpoint} variant={ckpt.get('variant')}")
    if [str(value) for value in ckpt.get("model_columns", [])] != list(MODEL_COLUMNS):
        raise ValueError(f"checkpoint model_columns 与当前动作空间不一致：{checkpoint}")
    visual_scaler = scaler_from_state(ckpt["visual_scaler_state"])
    aux_scaler = scaler_from_state(ckpt["aux_scaler_state"])
    hyper = ckpt.get("hyperparameters", {})
    router = FiLMRouter(
        visual_dim=int(visual_scaler.n_features_in_),
        aux_dim=int(aux_scaler.n_features_in_),
        hidden_dim=int(hyper.get("hidden_dim", 64)),
        film_hidden_dim=int(hyper.get("film_hidden_dim", 32)),
        output_dim=len(MODEL_COLUMNS),
        dropout=float(hyper.get("dropout", 0.0)),
    ).to(device)
    router.load_state_dict(ckpt["router_state_dict"])
    router.eval()
    return router, visual_scaler, aux_scaler, ckpt


def load_baseline_comparison(path: Path) -> pd.DataFrame:
    """函数功能：读取并筛选 P2d final extension comparison，作为统一 baseline 来源。"""
    comparison_path = Path(path) / "round1_final_test_extension_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"找不到 baseline final extension comparison：{comparison_path}")
    df = pd.read_csv(comparison_path)
    keep_methods = set()
    for variant in BASELINE_VARIANTS:
        keep_methods.add(f"{variant}_hard_top1")
        keep_methods.add(f"{variant}_raw_soft_fusion")
    keep_methods.update(ROUND0_METHODS)
    out = df[df["method"].astype(str).isin(keep_methods)].copy()
    if len(out) != len(keep_methods):
        missing = sorted(keep_methods.difference(set(out["method"].astype(str))))
        raise ValueError(f"baseline comparison 缺少必要 method：{missing}")
    return out


def normalize_comparison_from_seed_summary(seed_results: pd.DataFrame) -> pd.DataFrame:
    """函数功能：把 FiLM per-seed summary 转成与 P2d final extension 一致的 comparison schema。"""
    mean_std = summarize_mean_std(seed_results, sample_set="pilot_test")
    rows: List[Dict[str, object]] = []
    for row in mean_std.itertuples(index=False):
        data = row._asdict()
        method = str(data["method"])
        is_hard = method.endswith("_hard_top1")
        is_soft = method.endswith("_raw_soft_fusion")
        rows.append(
            {
                "sample_set": "pilot_test",
                "method": method,
                "variant": str(data["variant"]),
                "seed_count": int(data["seed_count"]),
                "sample_count": int(data["sample_count_per_seed"]),
                "hard_top1_MAE": float(data["MAE_mean"]) if is_hard else np.nan,
                "hard_top1_MSE": float(data["MSE_mean"]) if is_hard else np.nan,
                "hard_top1_regret_to_oracle": float(data["regret_to_oracle_mean"]) if is_hard else np.nan,
                "hard_top1_oracle_label_accuracy": float(data["oracle_label_accuracy_mean"]) if is_hard else np.nan,
                "raw_soft_fusion_MAE": float(data["MAE_mean"]) if is_soft else np.nan,
                "raw_soft_fusion_MSE": float(data["MSE_mean"]) if is_soft else np.nan,
                "raw_soft_fusion_regret_to_oracle": float(data["regret_to_oracle_mean"]) if is_soft else np.nan,
                "raw_soft_fusion_oracle_label_accuracy": float(data["oracle_label_accuracy_mean"]) if is_soft else np.nan,
                "weight_entropy": float(data["weight_entropy_mean"]),
                "normalized_weight_entropy": float(data["normalized_weight_entropy_mean"]),
                "mean_max_weight": float(data["mean_max_weight_mean"]),
                "MAE_std": float(data["MAE_std"]),
                "MSE_std": float(data["MSE_std"]),
                "regret_to_oracle_std": float(data["regret_to_oracle_std"]),
                "oracle_label_accuracy_std": float(data["oracle_label_accuracy_std"]),
            }
        )
    return pd.DataFrame(rows)


def order_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """函数功能：按用户关心顺序排列 comparison。"""
    order: Dict[str, int] = {}
    idx = 0
    for variant in ["film_mean_patch_aux", "film_cls_mean_concat_aux", *BASELINE_VARIANTS]:
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
    out = comparison.copy()
    out["_order"] = out["method"].astype(str).map(order).fillna(999)
    return out.sort_values(["_order", "method"], kind="mergesort").drop(columns=["_order"]).reset_index(drop=True)


def build_stratified_summary(film_rows: pd.DataFrame, baseline_path: Path) -> pd.DataFrame:
    """函数功能：合并 FiLM 新分层和 P2d final extension 既有 baseline 分层。"""
    frames: List[pd.DataFrame] = []
    for col in TSF_STRATA_COLUMNS:
        grouped = summarize_rows_with_seed(film_rows, group_cols=[col]).rename(columns={col: "stratum_value"})
        grouped.insert(4, "stratum_column", col)
        grouped.insert(5, "stratum_kind", "single_column")
        frames.append(grouped)
    tsf_cell = summarize_rows_with_seed(film_rows, group_cols=TSF_STRATA_COLUMNS)
    tsf_cell.insert(4, "stratum_column", "tsf_cell")
    tsf_cell.insert(5, "stratum_kind", "tsf_cell")
    tsf_cell["stratum_value"] = tsf_cell[TSF_STRATA_COLUMNS].astype(str).agg("|".join, axis=1)
    frames.append(tsf_cell)
    film_strata = pd.concat(frames, ignore_index=True)

    baseline_file = Path(baseline_path) / "round1_final_test_extension_stratified_summary.csv"
    if not baseline_file.exists():
        raise FileNotFoundError(f"找不到 baseline stratified summary：{baseline_file}")
    baseline = pd.read_csv(baseline_file)
    keep_variants = set(BASELINE_VARIANTS)
    baseline_keep = baseline[
        baseline["variant"].astype(str).isin(keep_variants)
        | baseline["method"].astype(str).isin(ROUND0_METHODS)
    ].copy()
    return pd.concat([film_strata, baseline_keep], ignore_index=True)


def selected_model_counts_final(film_rows: pd.DataFrame, baseline_path: Path) -> pd.DataFrame:
    """函数功能：合并 FiLM per-seed selected_model counts 与既有 baseline counts。"""
    film_counts = (
        film_rows.groupby(["sample_set", "variant", "seed", "method", "selected_model"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = film_counts.groupby(["sample_set", "variant", "seed", "method"])["count"].transform("sum")
    film_counts["ratio"] = film_counts["count"] / totals
    baseline_file = Path(baseline_path) / "round1_final_test_extension_selected_model_counts.csv"
    if not baseline_file.exists():
        raise FileNotFoundError(f"找不到 baseline selected_model_counts：{baseline_file}")
    baseline = pd.read_csv(baseline_file)
    keep_variants = set(BASELINE_VARIANTS)
    baseline_keep = baseline[
        baseline["variant"].astype(str).isin(keep_variants)
        | baseline["method"].astype(str).isin(ROUND0_METHODS)
    ].copy()
    columns = ["sample_set", "variant", "seed", "method", "selected_model", "count", "ratio"]
    return pd.concat([film_counts[columns], baseline_keep[columns]], ignore_index=True).reset_index(drop=True)


def metric_lookup(comparison: pd.DataFrame, variant: str, method_kind: str, metric: str) -> float:
    """函数功能：按 variant/method_kind 从 comparison 读取指标。"""
    if variant == "Round0 TimeFuse":
        method = f"round0_timefuse_{method_kind}"
    elif variant == "Round0 original Visual":
        method = f"round0_original_visual_{method_kind}"
    elif variant in {"global_best_single", "oracle_top1"}:
        method = variant
    else:
        method = f"{variant}_{method_kind}"
    values = comparison.loc[comparison["method"].astype(str) == method, metric]
    if values.empty:
        return float("nan")
    return float(values.iloc[0])


def build_delta_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """函数功能：输出用户指定 pairwise delta；delta=left-right，误差类指标越小越好。"""
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
    for left, right in DELTA_PAIRS:
        for metric, method_kind in metrics:
            left_value = metric_lookup(comparison, left, method_kind, metric)
            right_value = metric_lookup(comparison, right, method_kind, metric)
            rows.append(
                {
                    "delta_name": f"{left} - {right}",
                    "sample_set": "pilot_test",
                    "method_kind": method_kind,
                    "left_variant": left,
                    "right_variant": right,
                    "metric": metric,
                    "left_value": left_value,
                    "right_value": right_value,
                    "delta_left_minus_right": left_value - right_value if np.isfinite(left_value) and np.isfinite(right_value) else np.nan,
                    "lower_is_better": not metric.endswith("accuracy") and metric not in {"weight_entropy", "normalized_weight_entropy", "mean_max_weight"},
                }
            )
    return pd.DataFrame(rows)


def raw_metric(comparison: pd.DataFrame, variant: str, metric: str) -> float:
    """函数功能：读取 raw-soft 指标。"""
    return metric_lookup(comparison, variant, "raw_soft_fusion", metric)


def delta_value(delta: pd.DataFrame, name: str, metric: str, method_kind: str = "raw_soft_fusion") -> float:
    """函数功能：从 delta summary 读取一个 delta 值。"""
    subset = delta[
        (delta["delta_name"].astype(str) == name)
        & (delta["metric"].astype(str) == metric)
        & (delta["method_kind"].astype(str) == method_kind)
    ]
    if subset.empty:
        return float("nan")
    return float(subset.iloc[0]["delta_left_minus_right"])


def write_summary_md(
    *,
    output_dir: Path,
    comparison: pd.DataFrame,
    seed_results: pd.DataFrame,
    stratified: pd.DataFrame,
    delta_summary: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    """函数功能：写中文 P2e frozen pilot_test summary，逐项回答目标问题。"""
    fm_mae = raw_metric(comparison, "film_mean_patch_aux", "raw_soft_fusion_MAE")
    fm_mse = raw_metric(comparison, "film_mean_patch_aux", "raw_soft_fusion_MSE")
    fm_regret = raw_metric(comparison, "film_mean_patch_aux", "raw_soft_fusion_regret_to_oracle")
    fc_mae = raw_metric(comparison, "film_cls_mean_concat_aux", "raw_soft_fusion_MAE")
    fc_mse = raw_metric(comparison, "film_cls_mean_concat_aux", "raw_soft_fusion_MSE")
    fc_regret = raw_metric(comparison, "film_cls_mean_concat_aux", "raw_soft_fusion_regret_to_oracle")
    visual_mean = raw_metric(comparison, "visual_mean_patch_only", "raw_soft_fusion_MAE")
    mean_aux = raw_metric(comparison, "mean_patch_plus_aux", "raw_soft_fusion_MAE")
    visual_cls = raw_metric(comparison, "visual_cls_mean_concat", "raw_soft_fusion_MAE")
    cls_aux = raw_metric(comparison, "cls_mean_concat_plus_aux", "raw_soft_fusion_MAE")
    tf_mae = raw_metric(comparison, "Round0 TimeFuse", "raw_soft_fusion_MAE")
    fm_std = raw_metric(comparison, "film_mean_patch_aux", "MAE_std")
    fc_std = raw_metric(comparison, "film_cls_mean_concat_aux", "MAE_std")
    visual_cls_std = raw_metric(comparison, "visual_cls_mean_concat", "MAE_std")
    mean_aux_mse = raw_metric(comparison, "mean_patch_plus_aux", "raw_soft_fusion_MSE")
    visual_cls_mse = raw_metric(comparison, "visual_cls_mean_concat", "raw_soft_fusion_MSE")
    fm_acc = raw_metric(comparison, "film_mean_patch_aux", "raw_soft_fusion_oracle_label_accuracy")
    fc_acc = raw_metric(comparison, "film_cls_mean_concat_aux", "raw_soft_fusion_oracle_label_accuracy")
    fm_entropy = raw_metric(comparison, "film_mean_patch_aux", "normalized_weight_entropy")
    fc_entropy = raw_metric(comparison, "film_cls_mean_concat_aux", "normalized_weight_entropy")
    fm_max = raw_metric(comparison, "film_mean_patch_aux", "mean_max_weight")
    fc_max = raw_metric(comparison, "film_cls_mean_concat_aux", "mean_max_weight")

    oracle_focus = stratified[
        (stratified["stratum_column"].astype(str) == "oracle_model")
        & (stratified["stratum_value"].astype(str).isin(["CrossFormer", "PatchTST"]))
        & (stratified["method"].astype(str).str.endswith("_raw_soft_fusion"))
    ].copy()
    keep_methods = [
        "film_mean_patch_aux_raw_soft_fusion",
        "film_cls_mean_concat_aux_raw_soft_fusion",
        "visual_mean_patch_only_raw_soft_fusion",
        "visual_cls_mean_concat_raw_soft_fusion",
        "mean_patch_plus_aux_raw_soft_fusion",
        "cls_mean_concat_plus_aux_raw_soft_fusion",
        "round0_timefuse_raw_soft_fusion",
    ]
    oracle_focus = oracle_focus[oracle_focus["method"].astype(str).isin(keep_methods)]

    fm_vs_visual_mean = delta_value(delta_summary, "film_mean_patch_aux - visual_mean_patch_only", "raw_soft_fusion_MAE")
    fm_vs_mean_aux = delta_value(delta_summary, "film_mean_patch_aux - mean_patch_plus_aux", "raw_soft_fusion_MAE")
    fc_vs_visual_cls = delta_value(delta_summary, "film_cls_mean_concat_aux - visual_cls_mean_concat", "raw_soft_fusion_MAE")
    fc_vs_cls_aux = delta_value(delta_summary, "film_cls_mean_concat_aux - cls_mean_concat_plus_aux", "raw_soft_fusion_MAE")
    fc_vs_fm = delta_value(delta_summary, "film_cls_mean_concat_aux - film_mean_patch_aux", "raw_soft_fusion_MAE")
    fc_vs_visual_cls_mse = delta_value(delta_summary, "film_cls_mean_concat_aux - visual_cls_mean_concat", "raw_soft_fusion_MSE")
    fm_vs_mean_aux_mse = delta_value(delta_summary, "film_mean_patch_aux - mean_patch_plus_aux", "raw_soft_fusion_MSE")

    best_mae_variant = "film_cls_mean_concat_aux" if fc_mae < fm_mae else "film_mean_patch_aux"
    best_mse_variant = "film_cls_mean_concat_aux" if fc_mse < fm_mse else "film_mean_patch_aux"
    best_regret_variant = "film_cls_mean_concat_aux" if fc_regret < fm_regret else "film_mean_patch_aux"
    soft_reliance = (fm_entropy > 0.60 and fc_entropy > 0.60 and fm_max < 0.65 and fc_max < 0.65)
    p2f_suggestion = "回退到 visual-only cls+mean 主线" if fc_mae >= visual_cls and fm_mae >= visual_mean else "可进入下一轮 Round2/P2f，但必须以 frozen test 风险为约束"

    lines = [
        "# Visual Router V2 Round 1 P2e FiLM Frozen Pilot Test Extension Summary",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 结论回答",
        "",
        f"1. `film_mean_patch_aux` 是否在 pilot_test 上优于 `visual_mean_patch_only`：{'是' if fm_mae < visual_mean else '否'}。raw-soft MAE={fm_mae:.6f} vs {visual_mean:.6f}，delta={fm_vs_visual_mean:+.6f}。",
        f"2. `film_mean_patch_aux` 是否避免 `mean_patch_plus_aux` 明显退化：{'是' if fm_mae < mean_aux else '否'}。film MAE={fm_mae:.6f}，mean_patch_plus_aux MAE={mean_aux:.6f}，delta={fm_vs_mean_aux:+.6f}。",
        f"3. `film_cls_mean_concat_aux` 是否优于 `visual_cls_mean_concat`：{'是' if fc_mae < visual_cls else '否'}。raw-soft MAE={fc_mae:.6f} vs {visual_cls:.6f}，delta={fc_vs_visual_cls:+.6f}。",
        f"4. `film_cls_mean_concat_aux` 是否优于 `cls_mean_concat_plus_aux`：{'是' if fc_mae < cls_aux else '否'}。raw-soft MAE={fc_mae:.6f} vs {cls_aux:.6f}，delta={fc_vs_cls_aux:+.6f}。",
        f"5. 两个 FiLM 变体中 raw-soft MAE 更好的是 `{best_mae_variant}`，MSE 更好的是 `{best_mse_variant}`，regret 更好的是 `{best_regret_variant}`；film_mean_patch_aux MAE/MSE/regret={fm_mae:.6f}/{fm_mse:.6f}/{fm_regret:.6f}，film_cls_mean_concat_aux={fc_mae:.6f}/{fc_mse:.6f}/{fc_regret:.6f}，二者 MAE delta(cls-mean)={fc_vs_fm:+.6f}。",
        f"6. FiLM 是否改善 seed stability：mean_patch 路线 {'是' if fm_std < raw_metric(comparison, 'visual_mean_patch_only', 'MAE_std') else '否'}，cls+mean 路线 {'是' if fc_std < visual_cls_std else '否'}；FiLM MAE_std={fm_std:.6f}/{fc_std:.6f}。",
        f"7. FiLM 是否改善 MSE tail：mean_patch 路线 {'是' if fm_mse < mean_aux_mse else '否'}，cls+mean 路线 {'是' if fc_mse < visual_cls_mse else '否'}；MSE delta 分别为 {fm_vs_mean_aux_mse:+.6f} 和 {fc_vs_visual_cls_mse:+.6f}。",
        "8. FiLM 是否改善 CrossFormer / PatchTST strata：见下方 oracle_model 分层摘录；完整表在 `round1_film_final_test_extension_stratified_summary.csv`。",
        f"9. FiLM 是否仍主要依赖 soft fusion 而不是 hard oracle-label accuracy：{'是' if soft_reliance else '不充分'}。normalized entropy={fm_entropy:.6f}/{fc_entropy:.6f}，mean max weight={fm_max:.6f}/{fc_max:.6f}，oracle-label accuracy={fm_acc:.6f}/{fc_acc:.6f}。",
        f"10. P2e 后续建议：{p2f_suggestion}。该判断只用于后续路线，不改变 P2e selection best 历史结论，也不使用 pilot_test 做模型选择。",
        "",
        "## Comparison",
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
        "## CrossFormer / PatchTST Strata 摘录",
        "",
        frame_to_markdown(oracle_focus.sort_values(["stratum_value", "method", "seed"], kind="mergesort"), float_digits=6),
        "",
        "## 边界记录",
        "",
        "- 本扩展只评估 frozen P2e checkpoint；未训练新模型，未改变 variant/seed/epoch/hyperparams。",
        "- pilot_test 只用于最终评估，不用于模型选择。",
        "- 使用 final_test_only feature cache；未重建 P2a feature cache，未保存 pseudo image tensor。",
        f"- commit hash：`{metadata['commit_hash']}`",
        "",
    ]
    (output_dir / "round1_film_final_test_extension_summary.md").write_text("\n".join(lines), encoding="utf-8")


def copy_light_summaries(output_dir: Path, summary_dir: Path) -> None:
    """函数功能：只复制轻量 summary/metadata/CSV 到 experiment_summaries。"""
    summary_dir.mkdir(parents=True, exist_ok=True)
    for name in required_outputs():
        if name == "status.json":
            continue
        shutil.copy2(output_dir / name, summary_dir / name)


def run_single(args: argparse.Namespace) -> None:
    """函数功能：运行一个 FiLM variant/seed 的 frozen pilot_test eval，并写入隔离子目录。"""
    if args.variant is None or args.seed is None:
        raise ValueError("--run-single 必须同时提供 --variant 和 --seed")
    out_dir = task_dir(args.output_dir, args.variant, int(args.seed))
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "task_metadata.json").exists() and not args.overwrite:
        raise FileExistsError(f"单任务输出已存在；如需覆盖请传 --overwrite：{out_dir}")
    write_json(out_dir / "status.json", {"status": "started", "variant": args.variant, "seed": int(args.seed), "updated_at": display_time()})

    pilot_test = read_sample_csv(Path(args.sample_dir) / "pilot_test_sample_keys.csv")
    feature_manifest = validate_final_feature_cache(args.final_feature_dir, expected_count=len(pilot_test))
    checkpoint = checkpoint_path(args.film_dir, args.variant, int(args.seed))
    if not checkpoint.exists():
        raise FileNotFoundError(f"找不到 P2e FiLM checkpoint：{checkpoint}")
    log_stage(f"读取 pilot_test oracle labels：variant={args.variant} seed={args.seed}")
    test_labels = load_oracle_subset(args.oracle_labels_path, pilot_test["sample_key"].astype(str).tolist(), batch_rows=int(args.parquet_batch_rows))
    prediction_index = ensure_prediction_index(
        output_dir=args.output_dir,
        baseline_final_extension_dir=args.baseline_final_extension_dir,
        prediction_manifest_path=args.prediction_manifest_path,
        sample_keys=pilot_test["sample_key"].astype(str).tolist(),
        chunk_read_rows=int(args.csv_chunksize),
    )
    device = resolve_device(str(args.device))
    try:
        log_stage(f"读取 final_test_only FiLM features：variant={args.variant} seed={args.seed}")
        visual_features, aux_features = load_film_features(
            feature_manifest_path=feature_manifest,
            sample_df=pilot_test,
            sample_set="pilot_test",
            variant=args.variant,
        )
        router, visual_scaler, aux_scaler, ckpt = load_film_router(checkpoint, variant=args.variant, device=device)
        pred = predict_film_router(
            router=router,
            visual_scaler=visual_scaler,
            aux_scaler=aux_scaler,
            visual_features=visual_features,
            aux_features=aux_features,
            sample_df=pilot_test,
            labels_df=test_labels,
            variant=args.variant,
            seed=int(args.seed),
            sample_set="pilot_test",
            device=device,
        )
        pred = align_with_sample_frame(pilot_test, pred)
        pred["router_name"] = f"p2e_{args.variant}_seed{int(args.seed)}_frozen_final_test_extension"
        pred = add_batch_fusion_metrics(pred, prediction_index=prediction_index, metric="mae", batch_size=int(args.eval_batch_size))
        pred_path = out_dir / f"predictions_{args.variant}_seed{int(args.seed)}_pilot_test.csv"
        pred.to_csv(pred_path, index=False)
        method_rows = make_visual_pooling_method_rows(pred, sample_set="pilot_test", variant=args.variant, seed=int(args.seed))
        seed_results = summarize_rows_with_seed(method_rows)
        method_rows.to_csv(out_dir / "method_rows.csv", index=False)
        seed_results.to_csv(out_dir / "seed_results.csv", index=False)
        write_json(
            out_dir / "task_metadata.json",
            {
                "status": "completed",
                "generated_at": display_time(),
                "script_version": SCRIPT_VERSION,
                "variant": args.variant,
                "seed": int(args.seed),
                "device": str(device),
                "checkpoint_path": str(checkpoint),
                "prediction_path": str(pred_path),
                "feature_manifest": str(feature_manifest),
                "checkpoint_script_version": ckpt.get("script_version"),
                "hyperparameters": ckpt.get("hyperparameters", {}),
                "constraints": {
                    "pilot_test_used_for_selection": False,
                    "pilot_test_evaluated": True,
                    "trained_new_model": False,
                    "changed_variant_by_test": False,
                    "changed_seed_by_test": False,
                    "changed_epoch_by_test": False,
                    "changed_hyperparams_by_test": False,
                    "rebuilt_p2a_feature": False,
                    "used_final_test_only_feature_cache": True,
                    "loaded_116m_prediction_manifest_to_memory": False,
                    "saved_pseudo_image_tensor": False,
                    "used_film": True,
                    "used_gating": False,
                    "used_attention": False,
                    "used_concat_aux": False,
                    "single_task_output_isolated": True,
                },
            },
        )
        write_json(out_dir / "status.json", {"status": "completed", "variant": args.variant, "seed": int(args.seed), "updated_at": display_time()})
        log_stage(f"单任务完成：{out_dir}")
    finally:
        prediction_index.close()


def aggregate(args: argparse.Namespace) -> None:
    """函数功能：聚合 6 个 frozen eval task，生成统一 comparison/delta/strata/summary。"""
    seeds = parse_seed_list(args.seeds)
    variants = parse_variant_list(args.variants)
    prepare_output_dir(args.output_dir, overwrite=bool(args.overwrite))
    write_status(args.output_dir, {"status": "aggregating", "script_version": SCRIPT_VERSION})
    source_meta = validate_film_sources(args.film_dir, variants, seeds)
    pilot_test = read_sample_csv(Path(args.sample_dir) / "pilot_test_sample_keys.csv")
    feature_manifest = validate_final_feature_cache(args.final_feature_dir, expected_count=len(pilot_test))

    method_frames: List[pd.DataFrame] = []
    seed_frames: List[pd.DataFrame] = []
    task_meta: List[Mapping[str, object]] = []
    missing: List[str] = []
    for variant in variants:
        for seed in seeds:
            out_dir = task_dir(args.output_dir, variant, seed)
            for name in ["method_rows.csv", "seed_results.csv", "task_metadata.json"]:
                if not (out_dir / name).exists():
                    missing.append(str(out_dir / name))
            if not missing:
                method_frames.append(pd.read_csv(out_dir / "method_rows.csv"))
                seed_frames.append(pd.read_csv(out_dir / "seed_results.csv"))
                task_meta.append(json.loads((out_dir / "task_metadata.json").read_text(encoding="utf-8")))
    if missing:
        raise FileNotFoundError("FiLM final eval task 输出不完整：" + "; ".join(missing[:20]))

    film_rows = pd.concat(method_frames, ignore_index=True)
    seed_results = pd.concat(seed_frames, ignore_index=True)
    film_comparison = normalize_comparison_from_seed_summary(seed_results)
    baseline_comparison = load_baseline_comparison(args.baseline_final_extension_dir)
    comparison = order_comparison(pd.concat([film_comparison, baseline_comparison], ignore_index=True))
    selected_counts = selected_model_counts_final(film_rows, args.baseline_final_extension_dir)
    stratified = build_stratified_summary(film_rows, args.baseline_final_extension_dir)
    delta_summary = build_delta_summary(comparison)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_results.to_csv(args.output_dir / "round1_film_final_test_extension_variant_seed_results.csv", index=False)
    comparison.to_csv(args.output_dir / "round1_film_final_test_extension_comparison.csv", index=False)
    selected_counts.to_csv(args.output_dir / "round1_film_final_test_extension_selected_model_counts.csv", index=False)
    stratified.to_csv(args.output_dir / "round1_film_final_test_extension_stratified_summary.csv", index=False)
    delta_summary.to_csv(args.output_dir / "round1_film_final_test_extension_delta_summary.csv", index=False)

    feature_metadata = json.loads((Path(args.final_feature_dir) / "round1_feature_metadata.json").read_text(encoding="utf-8"))
    devices_used = sorted({str(meta.get("device", "")) for meta in task_meta if str(meta.get("device", "")).strip()})
    baseline_index_path = Path(args.baseline_final_extension_dir) / "prediction_index_round1_final_test_pilot_test.sqlite"
    actual_prediction_index_path = baseline_index_path if baseline_index_path.exists() else prediction_index_path(args.output_dir)
    metadata = {
        "status": "completed",
        "generated_at": display_time(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "commit_hash": git_commit_hash(),
        "output_dir": str(args.output_dir),
        "summary_copy_dir": str(args.summary_copy_dir),
        "inputs": {
            "pilot_test_sample_keys": str(Path(args.sample_dir) / "pilot_test_sample_keys.csv"),
            "final_test_only_feature_dir": str(args.final_feature_dir),
            "final_test_only_feature_manifest": str(feature_manifest),
            "film_dir": str(args.film_dir),
            "baseline_final_extension_dir": str(args.baseline_final_extension_dir),
            "oracle_labels_path": str(args.oracle_labels_path),
            "prediction_manifest_path": str(args.prediction_manifest_path),
            "prediction_index_path": str(actual_prediction_index_path),
            "prediction_index_reused_from": str(baseline_index_path) if baseline_index_path.exists() else "",
        },
        "film_source_metadata": source_meta,
        "variants": list(variants),
        "feature_groups": {key: list(value) for key, value in FEATURE_ARRAY_BY_FILM_VARIANT.items()},
        "seeds_evaluated": [int(seed) for seed in seeds],
        "sample_counts": {"pilot_test": int(len(pilot_test))},
        "checkpoint_model_paths": {variant: [str(checkpoint_path(args.film_dir, variant, seed)) for seed in seeds] for variant in variants},
        "pilot_test_used_for_selection": False,
        "pilot_test_evaluated": True,
        "trained_new_model": False,
        "changed_variant_by_test": False,
        "changed_seed_by_test": False,
        "changed_epoch_by_test": False,
        "changed_hyperparams_by_test": False,
        "rebuilt_p2a_feature": False,
        "used_final_test_only_feature_cache": True,
        "loaded_116m_prediction_manifest_to_memory": False,
        "saved_pseudo_image_tensor": False,
        "used_film": True,
        "used_gating": False,
        "used_attention": False,
        "used_concat_aux": False,
        "parallel_eval_used": bool(args.parallel_eval_used),
        "parallel_backend": "process_per_variant_seed" if bool(args.parallel_eval_used) else "serial_or_manual_tasks",
        "devices_requested": str(args.devices_requested),
        "devices_used": devices_used,
        "single_task_output_isolated": True,
        "feature_metadata_final_test_only": {
            "sample_sets": feature_metadata.get("sample_sets"),
            "final_test_only_sets": feature_metadata.get("final_test_only_sets"),
            "feature_constraints": feature_metadata.get("feature_constraints"),
        },
        "task_metadata": task_meta,
        "outputs": {name: str(Path(args.output_dir) / name) for name in required_outputs()},
    }
    write_json(args.output_dir / "round1_film_final_test_extension_metadata.json", metadata)
    write_summary_md(
        output_dir=args.output_dir,
        comparison=comparison,
        seed_results=seed_results,
        stratified=stratified,
        delta_summary=delta_summary,
        metadata=metadata,
    )
    copy_light_summaries(args.output_dir, args.summary_copy_dir)
    write_status(args.output_dir, {"status": "completed", "sample_count": int(len(pilot_test))})
    log_stage(f"P2e FiLM final test extension aggregation outputs written to {args.output_dir}")


def run_serial(args: argparse.Namespace) -> None:
    """函数功能：不使用 launcher 时串行完成 2 variants × seeds eval 后聚合。"""
    seeds = parse_seed_list(args.seeds)
    variants = parse_variant_list(args.variants)
    validate_film_sources(args.film_dir, variants, seeds)
    for variant in variants:
        for seed in seeds:
            child = argparse.Namespace(**vars(args))
            child.variant = variant
            child.seed = int(seed)
            child.run_single = True
            child.aggregate_only = False
            run_single(child)
    aggregate(args)


def main() -> None:
    """函数功能：根据 CLI 模式执行单任务、汇总或串行 fallback。"""
    args = parse_args()
    if args.aggregate_only:
        aggregate(args)
    elif args.run_single:
        run_single(args)
    else:
        run_serial(args)


if __name__ == "__main__":
    main()
