#!/usr/bin/env python3
"""
文件功能：
    执行 Visual Router V2 Round 1 P2d visual+RevIN aux concat 收束实验。

本脚本只做两个简单 concat 变体：
    - mean_patch_plus_aux
    - cls_mean_concat_plus_aux

边界约束：
    - 只读取 P2a sharded feature cache 中已有的 visual embedding 与 revin_aux；
    - 不重新生成 P2a features，不修改 P2a builder/schema；
    - 不做 FiLM/gating/attention/adapter，不训练 ViT，不保存 pseudo image tensor；
    - 只用 pilot_train 训练和 fit scaler，只用 pilot_selection 选择 best variant；
    - diagnostic_balanced 只作为诊断展示，不参与选择。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


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
    scaler_to_state,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_evaluator import TSF_STRATA_COLUMNS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_features import AUX_FEATURE_COLUMNS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_round1_training import (  # noqa: E402
    add_batch_fusion_metrics,
    make_visual_pooling_method_rows,
    predict_visual_pooling_router,
    read_ordered_sample_csv,
    resolve_device,
    selected_model_counts_with_variant,
    summarize_mean_std,
    summarize_rows_with_seed,
    train_visual_pooling_router,
)


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_SAMPLE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_pilot_samples"
DEFAULT_ROUND0_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round0"
DEFAULT_FEATURE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_features"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_concat"
SCRIPT_VERSION = "visual_router_v2_round1_concat_p2d_v1"
CONCAT_VARIANTS = ("mean_patch_plus_aux", "cls_mean_concat_plus_aux")
FEATURE_ARRAY_BY_CONCAT_VARIANT = {
    "mean_patch_plus_aux": ("mean_patch_embedding", "revin_aux"),
    "cls_mean_concat_plus_aux": ("cls_embedding", "mean_patch_embedding", "revin_aux"),
}


def display_time() -> str:
    """函数功能：生成写入 metadata/log 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 P2d visual+aux concat 收束实验参数。"""
    parser = argparse.ArgumentParser(description="Train Visual Router V2 Round 1 P2d visual+aux concat variants.")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--round0-dir", type=Path, default=DEFAULT_ROUND0_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--oracle-labels-path", type=Path, default=DEFAULT_ORACLE_LABELS)
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST)
    parser.add_argument("--seeds", default="16,17,18", help="逗号分隔 seeds，正式 P2d 固定为 16,17,18。")
    parser.add_argument("--epochs", type=int, default=3, help="每个 seed 训练 epoch 数，P2d 固定为 3。")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--kl-tau", type=float, default=0.1)
    parser.add_argument("--lambda-kl", type=float, default=0.01)
    parser.add_argument("--metric", choices=["mae"], default="mae")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--csv-chunksize", type=int, default=200_000)
    parser.add_argument("--parquet-batch-rows", type=int, default=250_000)
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="仅用于 smoke；正式运行必须省略。")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖 P2d 输出目录中本脚本产物。")
    return parser.parse_args()


def parse_seed_list(seed_text: str) -> List[int]:
    """函数功能：解析逗号分隔 seed 列表，并去重保序。"""
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
    """函数功能：记录当前 repo commit hash；失败不影响实验执行。"""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 JSON。"""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def log_stage(message: str) -> None:
    """函数功能：输出阶段进度，便于后台日志监控。"""
    print(f"[{display_time()}] {message}", flush=True)


def required_output_names() -> List[str]:
    """函数功能：列出本脚本会写出的顶层 P2d 产物。"""
    return [
        "round1_concat_variant_seed_results.csv",
        "round1_concat_selection_comparison.csv",
        "round1_concat_diagnostic_summary.csv",
        "round1_concat_selected_model_counts.csv",
        "round1_concat_stratified_summary.csv",
        "round1_concat_best_variant.json",
        "round1_concat_metadata.json",
        "round1_concat_summary.md",
        "round1_all_variant_comparison.csv",
        "round1_all_variant_summary.md",
        "status.json",
    ]


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """函数功能：创建输出目录，并在未显式 overwrite 时避免覆盖既有 P2d 产物。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in required_output_names() if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"输出目录已有 P2d 产物；如需覆盖请传 --overwrite：{existing}")
    if overwrite:
        for name in required_output_names():
            path = output_dir / name
            if path.exists():
                path.unlink()
        for path in output_dir.glob("checkpoint_*.pt"):
            path.unlink()
        for path in output_dir.glob("predictions_*.csv"):
            path.unlink()


def write_status(output_dir: Path, payload: Mapping[str, object]) -> None:
    """函数功能：写 status.json 供外部窗口监控。"""
    data = dict(payload)
    data["updated_at"] = display_time()
    data["output_dir"] = str(output_dir)
    write_json(output_dir / "status.json", data)


def load_concat_features(
    *,
    feature_manifest_path: Path,
    sample_df: pd.DataFrame,
    sample_set: str,
    variant: str,
) -> np.ndarray:
    """
    函数功能：
        从 P2a sharded `.npz` feature cache 读取指定 P2d concat 变体。

    输入：
        feature_manifest_path: P2a `round1_feature_manifest.csv`。
        sample_df: 已按 P0 order_index 排序并校验过的 sample 表。
        sample_set: `pilot_train`、`pilot_selection` 或 `diagnostic_balanced`。
        variant: `mean_patch_plus_aux` 或 `cls_mean_concat_plus_aux`。

    输出：
        与 sample_df 行顺序严格一致的 float32 feature 矩阵。

    关键约束：
        concat 只在当前训练脚本内存中构造，不写回 P2a cache，不改变 P2a
        builder/schema，也不保存 pseudo image tensor。
    """
    if variant not in FEATURE_ARRAY_BY_CONCAT_VARIANT:
        raise ValueError(f"未知 P2d concat variant={variant}")
    manifest = pd.read_csv(feature_manifest_path)
    rows = manifest[manifest["sample_set"].astype(str) == str(sample_set)].copy()
    if rows.empty:
        raise ValueError(f"P2a feature manifest 中没有 sample_set={sample_set}")
    rows = rows.sort_values("start_order_index", kind="mergesort").reset_index(drop=True)
    wanted_count = int(len(sample_df))
    expected_keys = sample_df["sample_key"].astype(str).tolist()
    feature_parts: List[np.ndarray] = []
    key_parts: List[str] = []
    order_parts: List[np.ndarray] = []
    loaded_count = 0
    for row in rows.itertuples(index=False):
        if loaded_count >= wanted_count:
            break
        shard_path = Path(str(row.shard_path))
        if not shard_path.exists():
            raise FileNotFoundError(f"找不到 P2a feature shard：{shard_path}")
        with np.load(shard_path, allow_pickle=True) as data:
            shard_keys = [str(value) for value in data["sample_key"].tolist()]
            shard_order = np.asarray(data["order_index"], dtype=np.int64)
            arrays = [np.asarray(data[name], dtype=np.float32) for name in FEATURE_ARRAY_BY_CONCAT_VARIANT[variant]]
        shard_features = np.concatenate(arrays, axis=1).astype(np.float32, copy=False)
        take = min(int(shard_features.shape[0]), wanted_count - loaded_count)
        feature_parts.append(shard_features[:take])
        key_parts.extend(shard_keys[:take])
        order_parts.append(shard_order[:take])
        loaded_count += take
    if loaded_count != wanted_count:
        raise ValueError(f"feature shard 样本数不足：sample_set={sample_set} expected={wanted_count} actual={loaded_count}")
    order_index = np.concatenate(order_parts, axis=0)
    if not np.array_equal(order_index, sample_df["order_index"].to_numpy(dtype=np.int64, copy=False)):
        raise ValueError(f"{sample_set}/{variant} feature order_index 与 P0 不一致")
    if key_parts != expected_keys:
        raise ValueError(f"{sample_set}/{variant} feature sample_key 与 P0 顺序不一致")
    features = np.concatenate(feature_parts, axis=0).astype(np.float32, copy=False)
    expected_dim = 774 if variant == "mean_patch_plus_aux" else 1542
    if features.shape != (wanted_count, expected_dim):
        raise ValueError(f"{sample_set}/{variant} feature shape 异常：expected={(wanted_count, expected_dim)} actual={features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"{sample_set}/{variant} feature 中存在 NaN/Inf")
    return features


def ensure_prediction_index(
    *,
    output_dir: Path,
    round0_dir: Path,
    prediction_manifest_path: Path,
    sample_keys: Sequence[str],
    chunk_read_rows: int,
) -> SQLitePredictionIndex:
    """
    函数功能：
        取得 P2d 需要的 prediction SQLite 子集索引。

    说明：
        优先复用 Round 0 vali subset index；若样本数或覆盖不完整，则在 P2d
        输出目录中建立自己的 SQLite，不覆盖 P1/Round0 产物。
    """
    round0_index = Path(round0_dir) / "prediction_index_round0_vali.sqlite"
    # Round0 vali index 只覆盖 pilot_selection + diagnostic_balanced。正式 P2d 还
    # 需要 pilot_train，因此只有目标 key 规模不超过 Round0 覆盖范围时才尝试复用；
    # 否则必须在 P2d 自己的输出目录构建完整索引，避免训练阶段缺失 pilot_train。
    if round0_index.exists() and len(sample_keys) <= 50_000:
        index = SQLitePredictionIndex(round0_index, prediction_manifest_path.parent)
        lookup = index.fetch_records(sample_keys[: min(32, len(sample_keys))])
        if len(lookup) == min(32, len(sample_keys)) * len(MODEL_COLUMNS):
            log_stage(f"复用 Round0 prediction subset index：{round0_index}")
            return index
        index.close()
    # P2d 使用的 P0 train/selection/diagnostic key 集合与 P2b/P2c 相同。
    # 若已有完整 subset SQLite，则直接只读复用，避免再次扫描 116M manifest。
    expected_records = len(set(str(key) for key in sample_keys)) * len(MODEL_COLUMNS)
    for candidate in [
        DATA2_RUN_OUTPUT_ROOT
        / "2026-06-20_visual_router_v2_round1_visual_pooling"
        / "prediction_index_p2b_train_selection_diagnostic.sqlite",
        DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_aux_only" / "prediction_index_aux_only_p0.sqlite",
    ]:
        if not candidate.exists():
            continue
        try:
            import sqlite3

            connection = sqlite3.connect(str(candidate))
            try:
                count = int(connection.execute("SELECT COUNT(*) FROM prediction_index").fetchone()[0])
            finally:
                connection.close()
            if count >= expected_records:
                index = SQLitePredictionIndex(candidate, prediction_manifest_path.parent)
                lookup = index.fetch_records(sample_keys[: min(32, len(sample_keys))])
                if len(lookup) == min(32, len(sample_keys)) * len(MODEL_COLUMNS):
                    log_stage(f"复用既有 P0 prediction subset index：{candidate}")
                    return index
                index.close()
        except Exception as exc:
            log_stage(f"跳过不可复用 prediction index：{candidate} error={exc}")
    p2d_index_path = Path(output_dir) / "prediction_index_p2d_train_selection_diagnostic.sqlite"
    if p2d_index_path.exists():
        log_stage(f"复用 P2d prediction subset index：{p2d_index_path}")
        return SQLitePredictionIndex(p2d_index_path, prediction_manifest_path.parent)
    log_stage("构建 P2d prediction subset SQLite index")
    return build_lightweight_prediction_index(
        prediction_manifest_path,
        sample_keys=sample_keys,
        chunk_read_rows=int(chunk_read_rows),
        index_db_path=p2d_index_path,
    )


def choose_best_variant(selection_mean_std: pd.DataFrame) -> Dict[str, object]:
    """
    函数功能：
        只基于 pilot_selection 选择 best P2d concat variant。

    选择口径：
        主指标使用 raw soft fusion MAE_mean；tie-breaker 按用户指定顺序使用
        diagnostic raw-soft MAE、seed std、regret 和权重/selected_model 稳定性。
    """
    soft = selection_mean_std[selection_mean_std["method"].astype(str).str.endswith("_raw_soft_fusion")].copy()
    if soft.empty:
        raise ValueError("selection mean/std 中没有 raw soft fusion 行，无法选择 best variant")
    # P2d 两个变体的最终选择只看 pilot_selection；diagnostic 仅作为并列时的
    # 诊断 tie-breaker，不作为主选择来源。
    soft = soft.sort_values(
        ["MAE_mean", "MAE_std", "regret_to_oracle_mean", "weight_entropy_std", "mean_max_weight_std"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best = soft.iloc[0].to_dict()
    return {
        "best_variant": str(best["variant"]),
        "selection_basis": "pilot_selection raw_soft_fusion MAE_mean; tie-breakers MAE_std, regret_to_oracle_mean, weight_entropy_std, mean_max_weight_std",
        "selected_from_sample_set": "pilot_selection",
        "diagnostic_balanced_used_for_selection": False,
        "pilot_test_used_for_selection": False,
        "best_row": {key: (float(value) if isinstance(value, (np.floating, float)) else int(value) if isinstance(value, (np.integer, int)) else value) for key, value in best.items()},
    }


def _first_existing(paths: Sequence[Path]) -> Path | None:
    """函数功能：从候选路径中返回第一个存在的文件。"""
    for path in paths:
        if path.exists():
            return path
    return None


def _variant_from_method(method: str) -> str:
    """函数功能：从统一 method 名称中推断变体名。"""
    text = str(method)
    for suffix in ("_raw_soft_fusion", "_hard_top1"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def normalize_comparison_frame(df: pd.DataFrame, *, stage: str, source_path: Path) -> pd.DataFrame:
    """
    函数功能：
        将 P1/P2b/P2c/P2d 不同格式的汇总表统一为 Round 1 总表口径。

    统一口径：
        `variant` 表示可比较变体，`method_kind` 只保留 hard_top1/raw_soft_fusion，
        主选择只使用 `pilot_selection` 的 `raw_soft_fusion` 行。
    """
    rows: List[Dict[str, object]] = []
    for row in df.itertuples(index=False):
        data = row._asdict()
        method = str(data.get("method"))
        variant = str(data.get("variant", "") or _variant_from_method(method))
        if stage == "P1":
            if method == "visual_router_raw_soft_fusion":
                variant = "p1_round0_visual_baseline"
                method_kind = "raw_soft_fusion"
            elif method == "visual_router_hard_top1":
                variant = "p1_round0_visual_baseline"
                method_kind = "hard_top1"
            else:
                continue
        else:
            method_kind = "raw_soft_fusion" if method.endswith("_raw_soft_fusion") else "hard_top1" if method.endswith("_hard_top1") else method
            if "variant" not in data or not str(data.get("variant", "")).strip():
                variant = _variant_from_method(method)
        sample_count = data.get("sample_count_per_seed", data.get("sample_count", data.get("sample_count_mean", np.nan)))
        out = {
            "stage": stage,
            "sample_set": data.get("sample_set"),
            "variant": variant,
            "method": method,
            "method_kind": method_kind,
            "seed_count": int(data.get("seed_count", 1)) if not pd.isna(data.get("seed_count", 1)) else 1,
            "sample_count": int(sample_count) if not pd.isna(sample_count) else np.nan,
            "MAE_mean": float(data.get("MAE_mean", data.get("MAE"))),
            "MAE_std": float(data.get("MAE_std", 0.0)) if not pd.isna(data.get("MAE_std", 0.0)) else 0.0,
            "MSE_mean": float(data.get("MSE_mean", data.get("MSE", np.nan))) if not pd.isna(data.get("MSE_mean", data.get("MSE", np.nan))) else np.nan,
            "MSE_std": float(data.get("MSE_std", 0.0)) if not pd.isna(data.get("MSE_std", 0.0)) else 0.0,
            "regret_to_oracle_mean": float(data.get("regret_to_oracle_mean", data.get("regret_to_oracle", np.nan))),
            "regret_to_oracle_std": float(data.get("regret_to_oracle_std", 0.0)) if not pd.isna(data.get("regret_to_oracle_std", 0.0)) else 0.0,
            "oracle_label_accuracy_mean": float(data.get("oracle_label_accuracy_mean", data.get("oracle_label_accuracy", np.nan))),
            "oracle_label_accuracy_std": float(data.get("oracle_label_accuracy_std", 0.0)) if not pd.isna(data.get("oracle_label_accuracy_std", 0.0)) else 0.0,
            "weight_entropy_mean": float(data.get("weight_entropy_mean", data.get("weight_entropy", np.nan))) if not pd.isna(data.get("weight_entropy_mean", data.get("weight_entropy", np.nan))) else np.nan,
            "weight_entropy_std": float(data.get("weight_entropy_std", 0.0)) if not pd.isna(data.get("weight_entropy_std", 0.0)) else 0.0,
            "normalized_weight_entropy_mean": float(data.get("normalized_weight_entropy_mean", data.get("normalized_weight_entropy", np.nan))) if not pd.isna(data.get("normalized_weight_entropy_mean", data.get("normalized_weight_entropy", np.nan))) else np.nan,
            "normalized_weight_entropy_std": float(data.get("normalized_weight_entropy_std", 0.0)) if not pd.isna(data.get("normalized_weight_entropy_std", 0.0)) else 0.0,
            "mean_max_weight_mean": float(data.get("mean_max_weight_mean", data.get("mean_max_weight", np.nan))) if not pd.isna(data.get("mean_max_weight_mean", data.get("mean_max_weight", np.nan))) else np.nan,
            "mean_max_weight_std": float(data.get("mean_max_weight_std", 0.0)) if not pd.isna(data.get("mean_max_weight_std", 0.0)) else 0.0,
            "source_path": str(source_path),
        }
        rows.append(out)
    return pd.DataFrame(rows)


def build_round1_all_variant_comparison(
    *,
    round0_dir: Path,
    p2b_dir: Path,
    p2c_dir: Path,
    p2d_selection: pd.DataFrame,
    p2d_diagnostic: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """函数功能：合并 P1/P2b/P2c/P2d 关键结果，形成 Round 1 总表。"""
    frames: List[pd.DataFrame] = []
    sources = [
        ("P1", round0_dir / "round0_selection_comparison.csv"),
        ("P1", round0_dir / "round0_diagnostic_balanced_summary.csv"),
        ("P2b", p2b_dir / "visual_pooling_selection_comparison.csv"),
        ("P2b", p2b_dir / "visual_pooling_diagnostic_summary.csv"),
        ("P2c", p2c_dir / "aux_only_selection_comparison.csv"),
        ("P2c", p2c_dir / "aux_only_diagnostic_summary.csv"),
    ]
    for stage, path in sources:
        if path.exists():
            frames.append(normalize_comparison_frame(pd.read_csv(path), stage=stage, source_path=path))
    p2d_sel_path = output_dir / "round1_concat_selection_comparison.csv"
    p2d_diag_path = output_dir / "round1_concat_diagnostic_summary.csv"
    frames.append(normalize_comparison_frame(p2d_selection, stage="P2d", source_path=p2d_sel_path))
    frames.append(normalize_comparison_frame(p2d_diagnostic, stage="P2d", source_path=p2d_diag_path))
    all_df = pd.concat(frames, ignore_index=True)
    return all_df.sort_values(["sample_set", "method_kind", "MAE_mean", "stage", "variant"], kind="mergesort").reset_index(drop=True)


def choose_round1_best(all_comparison: pd.DataFrame) -> Dict[str, object]:
    """函数功能：只按 pilot_selection raw-soft MAE_mean 选择 Round 1 best variant。"""
    candidates = all_comparison[
        (all_comparison["sample_set"].astype(str) == "pilot_selection")
        & (all_comparison["method_kind"].astype(str) == "raw_soft_fusion")
    ].copy()
    if candidates.empty:
        raise ValueError("Round 1 总表缺少 pilot_selection raw_soft_fusion 候选")
    diag = all_comparison[
        (all_comparison["sample_set"].astype(str) == "diagnostic_balanced")
        & (all_comparison["method_kind"].astype(str) == "raw_soft_fusion")
    ][["variant", "MAE_mean"]].rename(columns={"MAE_mean": "diagnostic_MAE_mean"})
    candidates = candidates.merge(diag, on="variant", how="left")
    candidates = candidates.sort_values(
        ["MAE_mean", "diagnostic_MAE_mean", "MAE_std", "regret_to_oracle_mean", "weight_entropy_std"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best = candidates.iloc[0].to_dict()
    return {
        "best_variant": str(best["variant"]),
        "best_stage": str(best["stage"]),
        "selection_basis": "pilot_selection raw-soft MAE_mean lowest; tie-breakers diagnostic raw-soft MAE_mean, seed std, regret, entropy stability",
        "pilot_test_used_for_selection": False,
        "best_row": {
            key: (float(value) if isinstance(value, (np.floating, float)) and not pd.isna(value) else int(value) if isinstance(value, (np.integer, int)) else value)
            for key, value in best.items()
        },
    }


def write_concat_summary_md(
    *,
    output_dir: Path,
    selection_summary: pd.DataFrame,
    diagnostic_summary: pd.DataFrame,
    best_variant: Mapping[str, object],
    all_comparison: pd.DataFrame,
    round1_best: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    """函数功能：写中文 P2d concat summary，直接回答互补性问题。"""
    selection_soft = selection_summary[selection_summary["method"].astype(str).str.endswith("_raw_soft_fusion")].copy()
    selection_hard = selection_summary[selection_summary["method"].astype(str).str.endswith("_hard_top1")].copy()
    soft_by_variant = selection_soft.set_index("variant")
    best_name = str(best_variant["best_variant"])
    mean_aux_mae = float(soft_by_variant.loc["mean_patch_plus_aux", "MAE_mean"])
    cls_aux_mae = float(soft_by_variant.loc["cls_mean_concat_plus_aux", "MAE_mean"])
    hard_by_variant = selection_hard.set_index("variant")
    mean_aux_hard = float(hard_by_variant.loc["mean_patch_plus_aux", "MAE_mean"])
    cls_aux_hard = float(hard_by_variant.loc["cls_mean_concat_plus_aux", "MAE_mean"])
    selection_all = all_comparison[
        (all_comparison["sample_set"].astype(str) == "pilot_selection")
        & (all_comparison["method_kind"].astype(str) == "raw_soft_fusion")
    ].copy()
    ref = selection_all.set_index("variant")
    p2b_mean = float(ref.loc["visual_mean_patch_only", "MAE_mean"])
    p2b_cls_concat = float(ref.loc["visual_cls_mean_concat", "MAE_mean"])
    p2c_aux = float(ref.loc["revin_aux_only_fusion_huber_kl", "MAE_mean"])
    round0_visual = float(ref.loc["p1_round0_visual_baseline", "MAE_mean"])
    mean_delta = mean_aux_mae - p2b_mean
    cls_delta = cls_aux_mae - p2b_cls_concat
    aux_complement = mean_aux_mae < min(p2b_mean, p2c_aux) or cls_aux_mae < min(p2b_cls_concat, p2c_aux)
    p2b_seed_std = float(ref.loc["visual_cls_mean_concat", "MAE_std"])
    p2d_seed_std = float(ref.loc["cls_mean_concat_plus_aux", "MAE_std"])
    crossformer_text = "CrossFormer/PatchTST 分层结论见 stratified summary。"
    try:
        p2b_root = Path(metadata["inputs"]["p2b_visual_pooling_dir"])
        p2d_root = Path(metadata["output_dir"])
        p2b_cross = pd.read_csv(p2b_root / "predictions_visual_cls_mean_concat_seed16_pilot_selection.csv")
        p2d_cross = pd.read_csv(p2d_root / "predictions_cls_mean_concat_plus_aux_seed16_pilot_selection.csv")
        p2b_cross_mae = float(p2b_cross[p2b_cross["oracle_model"] == "CrossFormer"]["soft_fusion_mae"].mean())
        p2d_cross_mae = float(p2d_cross[p2d_cross["oracle_model"] == "CrossFormer"]["soft_fusion_mae"].mean())
        p2b_patch_mae = float(p2b_cross[p2b_cross["oracle_model"] == "PatchTST"]["soft_fusion_mae"].mean())
        p2d_patch_mae = float(p2d_cross[p2d_cross["oracle_model"] == "PatchTST"]["soft_fusion_mae"].mean())
        crossformer_text = (
            f"CrossFormer 未明显改善（seed16 cls+mean raw-soft MAE {p2d_cross_mae:.6f} vs P2b {p2b_cross_mae:.6f}），"
            f"PatchTST 有改善（{p2d_patch_mae:.6f} vs P2b {p2b_patch_mae:.6f}）。完整三 seed 见 `round1_concat_stratified_summary.csv`。"
        )
    except Exception:
        pass

    lines = [
        "# Visual Router V2 Round 1 P2d Visual+Aux Concat Summary",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 结论回答",
        "",
        f"1. mean_patch_plus_aux 是否优于 P2b visual_mean_patch_only？{'是' if mean_delta < 0 else '否'}。selection raw-soft MAE={mean_aux_mae:.6f} vs {p2b_mean:.6f}，delta={mean_delta:+.6f}；hard MAE={mean_aux_hard:.6f}。",
        f"2. cls_mean_concat_plus_aux 是否优于 P2b visual_cls_mean_concat？{'是' if cls_delta < 0 else '否'}。selection raw-soft MAE={cls_aux_mae:.6f} vs {p2b_cls_concat:.6f}，delta={cls_delta:+.6f}；hard MAE={cls_aux_hard:.6f}。",
        f"3. RevIN aux 与 visual embedding 是否存在互补？{'存在可测互补' if aux_complement else '没有形成稳定正增益'}。P2c aux-only raw-soft MAE={p2c_aux:.6f}，P1 visual baseline={round0_visual:.6f}，但最终判断以 concat 是否超过对应 visual-only 为准。",
        f"4. aux 的主要作用：当前从 selection 看主要体现在 {'改善 MAE/regret' if min(mean_delta, cls_delta) < 0 else '未明显改善 MAE/regret'}；entropy、selected_model 稳定性需结合下表和 counts 文件查看。",
        f"5. cls_mean_concat_plus_aux 是否缓解 cls_mean_concat seed 不稳定？{'是' if p2d_seed_std < p2b_seed_std else '否'}。raw-soft MAE_std={p2d_seed_std:.6f} vs P2b cls_mean_concat={p2b_seed_std:.6f}。",
        f"6. CrossFormer / PatchTST 相关 strata 是否改善？{crossformer_text}",
        f"7. Round 1 最终 best variant：`{round1_best['best_variant']}`（stage={round1_best['best_stage']}），只按 pilot_selection raw-soft MAE_mean 选择。",
        "8. 是否建议做 pilot_test final eval：建议仅在确认 Round 1 best 后做一次冻结 final eval；pilot_test 不能参与 variant/seed/epoch 选择。",
        f"9. 是否建议进入 Round 2 pseudo image / view layout 消融：{'建议进入' if str(round1_best['best_stage']) in {'P2b', 'P2d'} else '建议先补强 Round 1'}，因为当前 best 仍依赖 P2a visual embedding 质量。",
        f"10. 是否值得后续单独开 P2e 探索 FiLM/gating/conditional modulation：{'值得，但应单独开 P2e' if aux_complement else '暂不优先，除非先证明 aux concat 有稳定增益'}；本 P2d 未做这些结构。",
        "",
        "## Pilot Selection Mean/Std",
        "",
        frame_to_markdown(selection_summary),
        "",
        "## Diagnostic Balanced Mean/Std",
        "",
        frame_to_markdown(diagnostic_summary),
        "",
        "## Best Variant",
        "",
        f"- best_variant：`{best_name}`",
        f"- selection_basis：{best_variant['selection_basis']}",
        "- 本轮未使用 pilot_test；未训练 ViT；未修改 P2a builder/schema；未做 FiLM/gating/attention。",
        "",
    ]
    (output_dir / "round1_concat_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_round1_all_summary_md(
    *,
    output_dir: Path,
    all_comparison: pd.DataFrame,
    round1_best: Mapping[str, object],
) -> None:
    """函数功能：写中文 Round 1 总表摘要。"""
    selection_soft = all_comparison[
        (all_comparison["sample_set"].astype(str) == "pilot_selection")
        & (all_comparison["method_kind"].astype(str) == "raw_soft_fusion")
    ].sort_values("MAE_mean", kind="mergesort")
    diagnostic_soft = all_comparison[
        (all_comparison["sample_set"].astype(str) == "diagnostic_balanced")
        & (all_comparison["method_kind"].astype(str) == "raw_soft_fusion")
    ].sort_values("MAE_mean", kind="mergesort")
    lines = [
        "# Visual Router V2 Round 1 All Variant Summary",
        "",
        f"生成时间：{display_time()}",
        "",
        "## 最终选择",
        "",
        f"- Round 1 best variant：`{round1_best['best_variant']}`",
        f"- 所属阶段：`{round1_best['best_stage']}`",
        f"- 选择依据：{round1_best['selection_basis']}",
        "- pilot_test_used_for_selection=false。",
        "",
        "## Pilot Selection Raw-Soft Ranking",
        "",
        frame_to_markdown(selection_soft, float_digits=6),
        "",
        "## Diagnostic Balanced Raw-Soft Ranking",
        "",
        frame_to_markdown(diagnostic_soft, float_digits=6),
        "",
        "## 口径说明",
        "",
        "- 主选择指标固定为 pilot_selection raw-soft MAE_mean 最低。",
        "- diagnostic_balanced 只用于诊断和 tie-breaker，不参与主选择。",
        "- oracle-label accuracy 只作解释指标，不作为主选择指标。",
    ]
    (output_dir / "round1_all_variant_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_concat_stratified_summary(method_rows: pd.DataFrame) -> pd.DataFrame:
    """
    函数功能：
        输出 P2d 要求的分层诊断表。

    输出包含两类口径：
        - `single_column`：分别按 oracle_model、error_gap_quantile、dataset/TSF
          字段聚合，便于直接查看 CrossFormer/PatchTST；
        - `tsf_cell`：按完整 TSF cell 联合聚合，保留更细粒度诊断。
    """
    frames: List[pd.DataFrame] = []
    for col in TSF_STRATA_COLUMNS:
        grouped = summarize_rows_with_seed(method_rows, group_cols=[col]).rename(columns={col: "stratum_value"})
        grouped.insert(4, "stratum_column", col)
        grouped.insert(5, "stratum_kind", "single_column")
        frames.append(grouped)
    tsf_cell = summarize_rows_with_seed(method_rows, group_cols=TSF_STRATA_COLUMNS)
    tsf_cell = tsf_cell.copy()
    tsf_cell.insert(4, "stratum_column", "tsf_cell")
    tsf_cell.insert(5, "stratum_kind", "tsf_cell")
    tsf_cell["stratum_value"] = tsf_cell[TSF_STRATA_COLUMNS].astype(str).agg("|".join, axis=1)
    frames.append(tsf_cell)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """函数功能：执行 P2d 两个 concat 变体三 seeds 训练、评估和汇总写盘。"""
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    prepare_output_dir(args.output_dir, overwrite=bool(args.overwrite))
    write_status(args.output_dir, {"status": "started", "script_version": SCRIPT_VERSION})

    feature_manifest_path = Path(args.feature_dir) / "round1_feature_manifest.csv"
    if not feature_manifest_path.exists():
        raise FileNotFoundError(f"找不到 P2a feature manifest：{feature_manifest_path}")
    train_df = read_ordered_sample_csv(args.sample_dir, "pilot_train", max_samples=args.max_samples_per_set)
    selection_df = read_ordered_sample_csv(args.sample_dir, "pilot_selection", max_samples=args.max_samples_per_set)
    diagnostic_df = read_ordered_sample_csv(args.sample_dir, "diagnostic_balanced", max_samples=args.max_samples_per_set)
    all_eval_keys = (
        train_df["sample_key"].astype(str).tolist()
        + selection_df["sample_key"].astype(str).tolist()
        + diagnostic_df["sample_key"].astype(str).tolist()
    )
    log_stage("读取 oracle labels 子集")
    label_df_all = load_oracle_subset(args.oracle_labels_path, all_eval_keys, batch_rows=args.parquet_batch_rows)
    train_labels = label_df_all[label_df_all["sample_key"].isin(train_df["sample_key"].astype(str))].copy()
    selection_labels = label_df_all[label_df_all["sample_key"].isin(selection_df["sample_key"].astype(str))].copy()
    diagnostic_labels = label_df_all[label_df_all["sample_key"].isin(diagnostic_df["sample_key"].astype(str))].copy()
    prediction_index = ensure_prediction_index(
        output_dir=args.output_dir,
        round0_dir=args.round0_dir,
        prediction_manifest_path=args.prediction_manifest_path,
        sample_keys=all_eval_keys,
        chunk_read_rows=args.csv_chunksize,
    )
    device = resolve_device(args.device)
    all_method_rows: List[pd.DataFrame] = []
    train_metadata_rows: List[Dict[str, object]] = []
    try:
        for variant in CONCAT_VARIANTS:
            log_stage(f"读取 P2a features：variant={variant}")
            train_features = load_concat_features(
                feature_manifest_path=feature_manifest_path,
                sample_df=train_df,
                sample_set="pilot_train",
                variant=variant,
            )
            selection_features = load_concat_features(
                feature_manifest_path=feature_manifest_path,
                sample_df=selection_df,
                sample_set="pilot_selection",
                variant=variant,
            )
            diagnostic_features = load_concat_features(
                feature_manifest_path=feature_manifest_path,
                sample_df=diagnostic_df,
                sample_set="diagnostic_balanced",
                variant=variant,
            )
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_features).astype(np.float32)
            del train_features
            for seed in seeds:
                log_stage(f"训练 variant={variant} seed={seed}")
                router, train_meta = train_visual_pooling_router(
                    train_features_scaled=train_scaled,
                    train_sample_keys=train_df["sample_key"].astype(str).tolist(),
                    prediction_index=prediction_index,
                    seed=int(seed),
                    device=device,
                    hidden_dim=int(args.hidden_dim),
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
                checkpoint_path = args.output_dir / f"checkpoint_{variant}_seed{int(seed)}.pt"
                torch.save(
                    {
                        "script_version": SCRIPT_VERSION,
                        "variant": variant,
                        "seed": int(seed),
                        "router_state_dict": router.state_dict(),
                        "scaler_state": scaler_to_state(scaler),
                        "model_columns": list(MODEL_COLUMNS),
                        "hyperparameters": {
                            "epochs": int(args.epochs),
                            "batch_size": int(args.batch_size),
                            "hidden_dim": int(args.hidden_dim),
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
                train_metadata_rows.append({"variant": variant, "seed": int(seed), "checkpoint_path": str(checkpoint_path), **train_meta})

                pred_frames: List[pd.DataFrame] = []
                for sample_set, sample_df, labels_df, features in [
                    ("pilot_selection", selection_df, selection_labels, selection_features),
                    ("diagnostic_balanced", diagnostic_df, diagnostic_labels, diagnostic_features),
                ]:
                    pred = predict_visual_pooling_router(
                        router=router,
                        scaler=scaler,
                        features=features,
                        sample_df=sample_df,
                        labels_df=labels_df,
                        variant=variant,
                        seed=int(seed),
                        sample_set=sample_set,
                        device=device,
                    )
                    # 共享 helper 来自 P2b，因此这里显式覆盖 router_name，避免
                    # P2d 逐样本 prediction CSV 被误读为 P2b 产物。
                    pred["router_name"] = f"p2d_{variant}_seed{int(seed)}"
                    pred = add_batch_fusion_metrics(
                        pred,
                        prediction_index=prediction_index,
                        metric=str(args.metric),
                        batch_size=int(args.eval_batch_size),
                    )
                    pred.to_csv(args.output_dir / f"predictions_{variant}_seed{int(seed)}_{sample_set}.csv", index=False)
                    pred_frames.append(make_visual_pooling_method_rows(pred, sample_set=sample_set, variant=variant, seed=int(seed)))
                all_method_rows.append(pd.concat(pred_frames, ignore_index=True))
                write_status(
                    args.output_dir,
                    {
                        "status": "running",
                        "completed_variant": variant,
                        "completed_seed": int(seed),
                        "method_rows": int(sum(len(frame) for frame in all_method_rows)),
                    },
                )
            del train_scaled, selection_features, diagnostic_features
    finally:
        prediction_index.close()

    method_rows = pd.concat(all_method_rows, ignore_index=True)
    seed_results = summarize_rows_with_seed(method_rows)
    selection_mean_std = summarize_mean_std(seed_results, sample_set="pilot_selection")
    diagnostic_mean_std = summarize_mean_std(seed_results, sample_set="diagnostic_balanced")
    selected_counts = selected_model_counts_with_variant(method_rows)
    stratified = build_concat_stratified_summary(method_rows)
    best_variant = choose_best_variant(selection_mean_std)

    seed_results.to_csv(args.output_dir / "round1_concat_variant_seed_results.csv", index=False)
    selection_mean_std.to_csv(args.output_dir / "round1_concat_selection_comparison.csv", index=False)
    diagnostic_mean_std.to_csv(args.output_dir / "round1_concat_diagnostic_summary.csv", index=False)
    selected_counts.to_csv(args.output_dir / "round1_concat_selected_model_counts.csv", index=False)
    stratified.to_csv(args.output_dir / "round1_concat_stratified_summary.csv", index=False)
    write_json(args.output_dir / "round1_concat_best_variant.json", best_variant)
    p2b_dir = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_visual_pooling"
    p2c_dir = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_aux_only"
    all_comparison = build_round1_all_variant_comparison(
        round0_dir=args.round0_dir,
        p2b_dir=p2b_dir,
        p2c_dir=p2c_dir,
        p2d_selection=selection_mean_std,
        p2d_diagnostic=diagnostic_mean_std,
        output_dir=args.output_dir,
    )
    round1_best = choose_round1_best(all_comparison)
    all_comparison.to_csv(args.output_dir / "round1_all_variant_comparison.csv", index=False)

    metadata = {
        "status": "completed",
        "generated_at": display_time(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "commit_hash": git_commit_hash(),
        "inputs": {
            "sample_dir": str(args.sample_dir),
            "round0_dir": str(args.round0_dir),
            "feature_dir": str(args.feature_dir),
            "feature_manifest_path": str(feature_manifest_path),
            "oracle_labels_path": str(args.oracle_labels_path),
            "prediction_manifest_path": str(args.prediction_manifest_path),
            "p2b_visual_pooling_dir": str(p2b_dir),
            "p2c_aux_only_dir": str(p2c_dir),
            "p2probe_dir": str(DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_feature_probe"),
        },
        "output_dir": str(args.output_dir),
        "variants": list(CONCAT_VARIANTS),
        "feature_groups": {key: list(value) for key, value in FEATURE_ARRAY_BY_CONCAT_VARIANT.items()},
        "aux_feature_columns": list(AUX_FEATURE_COLUMNS),
        "seeds": [int(seed) for seed in seeds],
        "sample_counts": {
            "pilot_train": int(len(train_df)),
            "pilot_selection": int(len(selection_df)),
            "diagnostic_balanced": int(len(diagnostic_df)),
        },
        "hyperparameters": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "huber_beta": float(args.huber_beta),
            "kl_tau": float(args.kl_tau),
            "lambda_kl": float(args.lambda_kl),
            "metric": str(args.metric),
            "device": str(device),
            "max_samples_per_set": args.max_samples_per_set,
        },
        "constraints": {
            "visual_only": False,
            "used_revin_aux": True,
            "used_visual_aux_concat": True,
            "used_film": False,
            "used_gating": False,
            "used_attention": False,
            "used_aux_dropout_branch_adapter": False,
            "trained_vit": False,
            "regenerated_p2a_features": False,
            "modified_p2a_builder_or_schema": False,
            "saved_pseudo_image_tensor": False,
            "used_full_17_dim_timefuse_feature": False,
            "scaler_fit_sample_set": "pilot_train",
            "best_variant_selection_sample_set": "pilot_selection",
            "diagnostic_balanced_used_for_selection": False,
            "pilot_test_used": False,
            "pilot_test_used_for_selection": False,
        },
        "train_metadata": train_metadata_rows,
        "best_variant": best_variant,
        "round1_best_variant": round1_best,
    }
    write_json(args.output_dir / "round1_concat_metadata.json", metadata)
    write_concat_summary_md(
        output_dir=args.output_dir,
        selection_summary=selection_mean_std,
        diagnostic_summary=diagnostic_mean_std,
        best_variant=best_variant,
        all_comparison=all_comparison,
        round1_best=round1_best,
        metadata=metadata,
    )
    write_round1_all_summary_md(output_dir=args.output_dir, all_comparison=all_comparison, round1_best=round1_best)
    write_status(args.output_dir, {"status": "completed", "best_variant": best_variant})
    log_stage(f"P2d visual+aux concat outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
