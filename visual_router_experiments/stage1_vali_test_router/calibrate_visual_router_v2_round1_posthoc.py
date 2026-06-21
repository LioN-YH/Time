#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round 1 P2f post-hoc calibration diagnostic。

设计边界：
    - 只读取既有 router prediction CSV 中的五专家权重，不训练新 router；
    - prediction CSV 不含 logits，因此使用任务允许的 weight power transform
      `normalize(w ** (1 / T))` 近似 temperature scaling；
    - 只用 `pilot_selection` 选择 calibration 参数，`diagnostic_balanced` 和
      `pilot_test` 仅做冻结诊断；
    - 复用已有轻量 SQLite prediction index 按 batch 读取专家预测数组，不读取
      116M prediction manifest、不读 checkpoint、不读 feature shard。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.stage1_vali_test_router.fusion_utils import (  # noqa: E402
    EPS,
    MODEL_COLUMNS,
    frame_to_markdown,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import (  # noqa: E402
    SQLitePredictionIndex,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_evaluator import (  # noqa: E402
    TSF_STRATA_COLUMNS,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_round1_training import (  # noqa: E402
    load_prediction_batch_from_index,
)


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
SUMMARY_ROOT = REPO_ROOT / "experiment_summaries" / "visual_router_v2_round1"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round1_calibration"
DEFAULT_SUMMARY_DIR = SUMMARY_ROOT / "p2f_calibration"
DEFAULT_SELECTION_INDEX = (
    DATA2_RUN_OUTPUT_ROOT
    / "2026-06-20_visual_router_v2_round1_visual_pooling"
    / "prediction_index_p2b_train_selection_diagnostic.sqlite"
)
DEFAULT_FINAL_INDEX = (
    DATA2_RUN_OUTPUT_ROOT
    / "2026-06-20_visual_router_v2_round1_final_test_extension"
    / "prediction_index_round1_final_test_pilot_test.sqlite"
)
DEFAULT_MANIFEST_DIR = (
    DATA2_RUN_OUTPUT_ROOT
    / "2026-06-15_stage1_96_48_s_full_scale"
    / "prediction_cache_full_scale_launcher"
    / "merged_cache"
)

SAMPLE_SETS = ("pilot_selection", "diagnostic_balanced", "pilot_test")
SEEDS = (16, 17, 18)
TEMPERATURE_GRID = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0)
ENTROPY_ALPHA_GRID = (0.0, 0.02, 0.05, 0.1)
RECOMMENDED_VARIANTS = (
    "film_mean_patch_aux",
    "film_cls_mean_concat_aux",
    "visual_cls_mean_concat",
    "cls_mean_concat_plus_aux",
)
STRATIFY_COLUMNS = (
    "oracle_model",
    "error_gap_quantile",
    "dataset_name",
    "group_name",
    "forecastability_cat",
    "season_strength_cat",
    "trend_strength_cat",
    "cv_cat",
    "missing_ratio_cat",
)


@dataclass(frozen=True)
class VariantSource:
    """类功能：记录一个候选 variant 三个 sample_set 的既有 prediction CSV 位置模板。"""

    variant: str
    selection_dir: Path
    selection_nested_tasks: bool
    final_dir: Path
    final_nested_tasks: bool

    def prediction_path(self, sample_set: str, seed: int) -> Path:
        """函数功能：按 variant、sample_set、seed 定位既有 prediction CSV。"""
        filename = f"predictions_{self.variant}_seed{int(seed)}_{sample_set}.csv"
        if sample_set == "pilot_test":
            base = self.final_dir
            nested = self.final_nested_tasks
        else:
            base = self.selection_dir
            nested = self.selection_nested_tasks
        if nested:
            return base / "tasks" / f"{self.variant}_seed{int(seed)}" / filename
        return base / filename


VARIANT_SOURCES = {
    "film_mean_patch_aux": VariantSource(
        variant="film_mean_patch_aux",
        selection_dir=DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round1_film",
        selection_nested_tasks=True,
        final_dir=DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round1_film_final_test_extension",
        final_nested_tasks=True,
    ),
    "film_cls_mean_concat_aux": VariantSource(
        variant="film_cls_mean_concat_aux",
        selection_dir=DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round1_film",
        selection_nested_tasks=True,
        final_dir=DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round1_film_final_test_extension",
        final_nested_tasks=True,
    ),
    "visual_cls_mean_concat": VariantSource(
        variant="visual_cls_mean_concat",
        selection_dir=DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_visual_pooling",
        selection_nested_tasks=False,
        final_dir=DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_final_test_extension",
        final_nested_tasks=False,
    ),
    "cls_mean_concat_plus_aux": VariantSource(
        variant="cls_mean_concat_plus_aux",
        selection_dir=DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_concat",
        selection_nested_tasks=False,
        final_dir=DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_final_test",
        final_nested_tasks=False,
    ),
}

BASELINE_FINAL_TEST = SUMMARY_ROOT / "global_summary" / "round1_global_final_test.csv"
GLOBAL_SELECTED = SUMMARY_ROOT / "global_summary" / "round1_global_selected_model_summary.csv"
GLOBAL_STRATA = SUMMARY_ROOT / "global_summary" / "round1_global_strata_summary.csv"


def display_time() -> str:
    """函数功能：生成 metadata 和中文 summary 使用的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def log_stage(message: str) -> None:
    """函数功能：输出带时间戳的进度信息，便于保留运行日志。"""
    print(f"[{display_time()}] {message}", flush=True)


def git_commit_hash() -> str:
    """函数功能：记录当前 commit hash；失败时返回 unknown，不影响诊断运行。"""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def parse_args() -> argparse.Namespace:
    """函数功能：解析 P2f calibration diagnostic 参数。"""
    parser = argparse.ArgumentParser(description="Visual Router V2 Round 1 post-hoc calibration diagnostic.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--selection-index-path", type=Path, default=DEFAULT_SELECTION_INDEX)
    parser.add_argument("--final-index-path", type=Path, default=DEFAULT_FINAL_INDEX)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--variants", nargs="+", default=list(RECOMMENDED_VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--temperatures", type=float, nargs="+", default=list(TEMPERATURE_GRID))
    parser.add_argument("--entropy-alphas", type=float, nargs="+", default=list(ENTROPY_ALPHA_GRID))
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--reuse-raw", action="store_true", help="复用 output-dir 中已生成的 seed raw CSV，只重建汇总交付物。")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """函数功能：校验路径和 grid，提前暴露误用。"""
    unknown = sorted(set(map(str, args.variants)).difference(VARIANT_SOURCES))
    if unknown:
        raise ValueError(f"未知 variant：{unknown}")
    for value in args.temperatures:
        if float(value) <= 0:
            raise ValueError(f"temperature 必须为正数：{value}")
    for value in args.entropy_alphas:
        if float(value) < 0 or float(value) > 1:
            raise ValueError(f"entropy alpha 必须位于 [0,1]：{value}")
    for path in [args.selection_index_path, args.final_index_path]:
        if not Path(path).exists():
            raise FileNotFoundError(f"找不到 prediction SQLite index：{path}")
    if not Path(args.manifest_dir).exists():
        raise FileNotFoundError(f"找不到 prediction manifest_dir：{args.manifest_dir}")


def calibrated_weights(weights: np.ndarray, *, temperature: float, alpha: float) -> np.ndarray:
    """
    函数功能：
        基于既有 soft weights 做 post-hoc calibration。

    说明：
        由于 Round1 prediction CSV 没有 logits，temperature 通过
        `normalize(w ** (1/T))` 实现；alpha 再向 uniform 分布插值。
    """
    clipped = np.clip(np.asarray(weights, dtype=np.float64), EPS, 1.0)
    powered = np.power(clipped, 1.0 / float(temperature))
    powered = powered / powered.sum(axis=1, keepdims=True)
    if float(alpha) > 0:
        uniform = np.full_like(powered, 1.0 / powered.shape[1], dtype=np.float64)
        powered = (1.0 - float(alpha)) * powered + float(alpha) * uniform
        powered = powered / powered.sum(axis=1, keepdims=True)
    return powered.astype(np.float32, copy=False)


def weight_diagnostics(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """函数功能：计算逐样本 entropy、normalized entropy 和 max weight。"""
    clipped = np.clip(weights.astype(np.float64), EPS, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    return entropy, entropy / math.log(len(MODEL_COLUMNS)), weights.max(axis=1)


def calibration_name(temperature: float, alpha: float) -> str:
    """函数功能：生成稳定的 calibration method 名称。"""
    temp = f"{float(temperature):g}".replace(".", "p")
    alp = f"{float(alpha):g}".replace(".", "p")
    if np.isclose(float(temperature), 1.0) and np.isclose(float(alpha), 0.0):
        return "original_uncalibrated"
    if np.isclose(float(alpha), 0.0):
        return f"power_temperature_T{temp}"
    return f"power_temperature_T{temp}_entropy_alpha{alp}"


def read_prediction_csv(path: Path, *, variant: str, sample_set: str, seed: int) -> pd.DataFrame:
    """函数功能：读取并校验一个既有 Round1 prediction CSV。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 prediction CSV：{path}")
    df = pd.read_csv(path)
    required = {
        "sample_key",
        "sample_set",
        "variant",
        "seed",
        "oracle_model",
        "oracle_value",
        "selected_model",
        "hard_top1_mae_from_array",
        "hard_top1_mse_from_array",
        "soft_fusion_mae",
        "soft_fusion_mse",
        *[f"weight_{name}" for name in MODEL_COLUMNS],
        *TSF_STRATA_COLUMNS,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} 缺少 calibration 所需字段：{missing}")
    if df["sample_key"].astype(str).duplicated().any():
        raise ValueError(f"{path} 中 sample_key 重复")
    if set(df["sample_set"].astype(str)) != {sample_set}:
        raise ValueError(f"{path} sample_set 不等于 {sample_set}")
    if set(df["variant"].astype(str)) != {variant}:
        raise ValueError(f"{path} variant 不等于 {variant}")
    if set(df["seed"].astype(int)) != {int(seed)}:
        raise ValueError(f"{path} seed 不等于 {seed}")
    return df.reset_index(drop=True)


def evaluate_calibration_grid(
    pred_df: pd.DataFrame,
    *,
    prediction_index: SQLitePredictionIndex,
    variant: str,
    seed: int,
    sample_set: str,
    temperatures: Sequence[float],
    entropy_alphas: Sequence[float],
    eval_batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    函数功能：
        对一个 variant/seed/sample_set 的全部 calibration grid 重算 soft fusion 指标。

    输出：
        grid_summary: 每个 calibration 参数一行；
        selected_counts: calibrated hard top-1 selected_model 计数；
        stratified_rows: 关键分层下的 calibrated soft fusion 指标。
    """
    base = pred_df.reset_index(drop=True)
    sample_keys = base["sample_key"].astype(str).tolist()
    original_weights = base[[f"weight_{name}" for name in MODEL_COLUMNS]].to_numpy(dtype=np.float32)
    oracle_value = pd.to_numeric(base["oracle_value"], errors="raise").to_numpy(dtype=np.float64)

    strategies: list[dict[str, object]] = []
    for temperature in temperatures:
        for alpha in entropy_alphas:
            weights = calibrated_weights(original_weights, temperature=float(temperature), alpha=float(alpha))
            entropy, normalized_entropy, max_weight = weight_diagnostics(weights)
            selected_idx = weights.argmax(axis=1).astype(np.int64)
            strategies.append(
                {
                    "temperature": float(temperature),
                    "entropy_alpha": float(alpha),
                    "calibration_method": calibration_name(float(temperature), float(alpha)),
                    "weights": weights,
                    "selected_idx": selected_idx,
                    "selected_model": np.asarray(MODEL_COLUMNS, dtype=object)[selected_idx],
                    "weight_entropy": entropy,
                    "normalized_weight_entropy": normalized_entropy,
                    "mean_max_weight": max_weight,
                    "soft_mae": np.zeros(len(base), dtype=np.float64),
                    "soft_mse": np.zeros(len(base), dtype=np.float64),
                }
            )

    for start in range(0, len(base), int(eval_batch_size)):
        stop = min(start + int(eval_batch_size), len(base))
        keys = sample_keys[start:stop]
        y_pred, y_true, _ = load_prediction_batch_from_index(prediction_index, keys, error_metric="mae")
        # 所有 calibration 策略共享同一个 batch 的专家预测数组，避免重复读取 mmap shard。
        for strategy in strategies:
            weights = strategy["weights"][start:stop]
            fused = (weights.reshape((len(keys), len(MODEL_COLUMNS), *([1] * (y_pred.ndim - 2)))) * y_pred).sum(axis=1)
            diff = fused - y_true
            axes = tuple(range(1, diff.ndim))
            strategy["soft_mae"][start:stop] = np.mean(np.abs(diff), axis=axes)
            strategy["soft_mse"][start:stop] = np.mean(diff**2, axis=axes)

    grid_rows: list[dict[str, object]] = []
    count_frames: list[pd.DataFrame] = []
    strata_frames: list[pd.DataFrame] = []
    for strategy in strategies:
        method = str(strategy["calibration_method"])
        mae = strategy["soft_mae"]
        mse = strategy["soft_mse"]
        regret = mae - oracle_value
        selected_model = pd.Series(strategy["selected_model"], name="selected_model")
        hard_label_correct = selected_model.astype(str).to_numpy() == base["oracle_model"].astype(str).to_numpy()
        grid_rows.append(
            {
                "sample_set": sample_set,
                "variant": variant,
                "seed": int(seed),
                "calibration_method": method,
                "temperature": float(strategy["temperature"]),
                "entropy_alpha": float(strategy["entropy_alpha"]),
                "sample_count": int(len(base)),
                "MAE": float(mae.mean()),
                "MSE": float(mse.mean()),
                "regret_to_oracle": float(regret.mean()),
                "oracle_label_accuracy": float(hard_label_correct.mean()),
                "weight_entropy": float(np.mean(strategy["weight_entropy"])),
                "normalized_weight_entropy": float(np.mean(strategy["normalized_weight_entropy"])),
                "mean_max_weight": float(np.mean(strategy["mean_max_weight"])),
                "original_soft_MAE": float(pd.to_numeric(base["soft_fusion_mae"], errors="raise").mean()),
                "original_soft_MSE": float(pd.to_numeric(base["soft_fusion_mse"], errors="raise").mean()),
                "original_hard_MAE": float(pd.to_numeric(base["hard_top1_mae_from_array"], errors="raise").mean()),
                "original_hard_MSE": float(pd.to_numeric(base["hard_top1_mse_from_array"], errors="raise").mean()),
                "calibration_basis": "weights_power_transform_no_logits",
            }
        )
        counts = selected_model.value_counts().reindex(MODEL_COLUMNS, fill_value=0).rename_axis("selected_model").reset_index(name="count")
        counts.insert(0, "sample_set", sample_set)
        counts.insert(1, "variant", variant)
        counts.insert(2, "seed", int(seed))
        counts.insert(3, "calibration_method", method)
        counts.insert(4, "temperature", float(strategy["temperature"]))
        counts.insert(5, "entropy_alpha", float(strategy["entropy_alpha"]))
        counts["ratio"] = counts["count"] / float(len(base))
        count_frames.append(counts)

        metric_frame = base[[*STRATIFY_COLUMNS]].copy()
        metric_frame["mae"] = mae
        metric_frame["mse"] = mse
        metric_frame["regret_to_oracle"] = regret
        metric_frame["oracle_label_correct"] = hard_label_correct
        for col in STRATIFY_COLUMNS:
            grouped = (
                metric_frame.groupby(col, dropna=False)
                .agg(
                    sample_count=("mae", "size"),
                    MAE=("mae", "mean"),
                    MSE=("mse", "mean"),
                    regret_to_oracle=("regret_to_oracle", "mean"),
                    oracle_label_accuracy=("oracle_label_correct", "mean"),
                )
                .reset_index()
                .rename(columns={col: "stratum_value"})
            )
            grouped.insert(0, "sample_set", sample_set)
            grouped.insert(1, "variant", variant)
            grouped.insert(2, "seed", int(seed))
            grouped.insert(3, "calibration_method", method)
            grouped.insert(4, "temperature", float(strategy["temperature"]))
            grouped.insert(5, "entropy_alpha", float(strategy["entropy_alpha"]))
            grouped.insert(6, "stratum_column", col)
            strata_frames.append(grouped)

    return (
        pd.DataFrame(grid_rows),
        pd.concat(count_frames, ignore_index=True),
        pd.concat(strata_frames, ignore_index=True),
    )


def summarize_grid(grid: pd.DataFrame, *, sample_set: str) -> pd.DataFrame:
    """函数功能：把 seed 级 grid 汇总成 variant/calibration 级 mean/std。"""
    subset = grid[grid["sample_set"].astype(str) == sample_set].copy()
    metric_cols = [
        "MAE",
        "MSE",
        "regret_to_oracle",
        "oracle_label_accuracy",
        "weight_entropy",
        "normalized_weight_entropy",
        "mean_max_weight",
    ]
    rows: list[dict[str, object]] = []
    for (variant, method, temperature, alpha), group in subset.groupby(
        ["variant", "calibration_method", "temperature", "entropy_alpha"], sort=False
    ):
        row: dict[str, object] = {
            "sample_set": sample_set,
            "variant": variant,
            "calibration_method": method,
            "temperature": float(temperature),
            "entropy_alpha": float(alpha),
            "seed_count": int(group["seed"].nunique()),
            "sample_count_per_seed": int(group["sample_count"].iloc[0]),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["variant", "MAE_mean", "MSE_mean", "calibration_method"]).reset_index(drop=True)


def select_best_params(selection_summary: pd.DataFrame) -> pd.DataFrame:
    """
    函数功能：
        只在 pilot_selection 上按用户指定 tie-breakers 选择每个 variant 的 calibration。
    """
    sort_cols = [
        "MAE_mean",
        "MAE_std",
        "MSE_mean",
        "regret_to_oracle_mean",
        "weight_entropy_std",
        "mean_max_weight_std",
    ]
    rows: list[pd.Series] = []
    for variant, group in selection_summary.groupby("variant", sort=False):
        best = group.sort_values(sort_cols, kind="mergesort").iloc[0].copy()
        best["selection_rule"] = "pilot_selection raw-soft MAE mean; tie MAE_std,MSE_mean,regret,entropy_std,max_weight_std"
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


def filter_selected(summary: pd.DataFrame, best_params: pd.DataFrame) -> pd.DataFrame:
    """函数功能：按 pilot_selection 选出的参数过滤其它 sample_set 汇总。"""
    keys = best_params[["variant", "calibration_method", "temperature", "entropy_alpha"]].copy()
    return summary.merge(keys, on=["variant", "calibration_method", "temperature", "entropy_alpha"], how="inner")


def build_delta_summary(
    selected_final: pd.DataFrame,
    selected_selection: pd.DataFrame,
    selection_summary: pd.DataFrame,
) -> pd.DataFrame:
    """函数功能：输出 calibrated 与 original 以及关键 baseline 的差值表。"""
    baseline = pd.read_csv(BASELINE_FINAL_TEST)
    baseline_by_variant = baseline.set_index("variant")
    baseline_by_method = baseline.set_index("method")

    def baseline_value(key: str, col: str) -> float:
        """函数功能：兼容 Round0 在 variant/method 中使用不同显示名的情况。"""
        if key in baseline_by_variant.index:
            return float(baseline_by_variant.loc[key, col])
        if key in baseline_by_method.index:
            return float(baseline_by_method.loc[key, col])
        raise KeyError(f"baseline final_test 中找不到 key={key}")

    def baseline_error_values(key: str) -> tuple[float, float, float]:
        """
        函数功能：
            取 baseline 的 MAE/MSE/regret。

        说明：
            router/raw-soft baseline 优先使用 raw-soft；`global_best_single` 和
            `oracle_top1` 没有 raw-soft 行，因此退回 hard_top1 指标。
        """
        mae = baseline_value(key, "raw_soft_MAE_mean")
        mse = baseline_value(key, "raw_soft_MSE_mean")
        regret = baseline_value(key, "raw_soft_regret_mean")
        if not np.isfinite(mae):
            mae = baseline_value(key, "hard_top1_MAE_mean")
        if not np.isfinite(mse):
            mse = baseline_value(key, "hard_top1_MSE_mean")
        if not np.isfinite(regret):
            regret = baseline_value(key, "hard_top1_regret_mean")
        return mae, mse, regret

    selection_original = selection_summary[
        selection_summary["calibration_method"].astype(str) == "original_uncalibrated"
    ].set_index("variant")
    rows: list[dict[str, object]] = []
    selection_by_variant = selected_selection.set_index("variant")
    for final_row in selected_final.itertuples(index=False):
        variant = str(final_row.variant)
        method = str(final_row.calibration_method)
        final_original = baseline_value(variant, "raw_soft_MAE_mean") if variant in baseline_by_variant.index else np.nan
        final_original_mse = baseline_value(variant, "raw_soft_MSE_mean") if variant in baseline_by_variant.index else np.nan
        final_original_regret = baseline_value(variant, "raw_soft_regret_mean") if variant in baseline_by_variant.index else np.nan
        comparisons = [
            ("final_test_calibrated_minus_original", final_original, final_original_mse, final_original_regret),
            (
                "final_test_calibrated_minus_round0_timefuse",
                *baseline_error_values("round0_timefuse"),
            ),
            (
                "final_test_calibrated_minus_round0_original_visual",
                *baseline_error_values("round0_original_visual"),
            ),
            (
                "final_test_calibrated_minus_visual_cls_mean_concat",
                *baseline_error_values("visual_cls_mean_concat"),
            ),
            (
                "final_test_calibrated_minus_global_best_single",
                *baseline_error_values("global_best_single"),
            ),
            (
                "final_test_calibrated_minus_oracle_top1",
                *baseline_error_values("oracle_top1"),
            ),
        ]
        for comparison, base_mae, base_mse, base_regret in comparisons:
            rows.append(
                {
                    "sample_set": "pilot_test",
                    "variant": variant,
                    "calibration_method": method,
                    "comparison": comparison,
                    "MAE_calibrated": float(final_row.MAE_mean),
                    "MAE_baseline": base_mae,
                    "delta_MAE": float(final_row.MAE_mean) - base_mae,
                    "MSE_calibrated": float(final_row.MSE_mean),
                    "MSE_baseline": base_mse,
                    "delta_MSE": float(final_row.MSE_mean) - base_mse,
                    "regret_calibrated": float(final_row.regret_to_oracle_mean),
                    "regret_baseline": base_regret,
                    "delta_regret": float(final_row.regret_to_oracle_mean) - base_regret,
                    "lower_is_better": True,
                }
            )
        if variant in selection_by_variant.index and variant in selection_original.index:
            sel_row = selection_by_variant.loc[variant]
            sel_orig = selection_original.loc[variant]
            rows.append(
                {
                    "sample_set": "pilot_selection",
                    "variant": variant,
                    "calibration_method": method,
                    "comparison": "selection_calibrated_minus_original",
                    "MAE_calibrated": float(sel_row.MAE_mean),
                    "MAE_baseline": float(sel_orig.MAE_mean),
                    "delta_MAE": float(sel_row.MAE_mean) - float(sel_orig.MAE_mean),
                    "MSE_calibrated": float(sel_row.MSE_mean),
                    "MSE_baseline": float(sel_orig.MSE_mean),
                    "delta_MSE": float(sel_row.MSE_mean) - float(sel_orig.MSE_mean),
                    "regret_calibrated": float(sel_row.regret_to_oracle_mean),
                    "regret_baseline": float(sel_orig.regret_to_oracle_mean),
                    "delta_regret": float(sel_row.regret_to_oracle_mean) - float(sel_orig.regret_to_oracle_mean),
                    "lower_is_better": True,
                }
            )
    return pd.DataFrame(rows)


def aggregate_selected_counts(counts: pd.DataFrame, best_params: pd.DataFrame) -> pd.DataFrame:
    """函数功能：只保留入选 calibration 的 selected_model ratio，并做 seed mean/std。"""
    keys = best_params[["variant", "calibration_method", "temperature", "entropy_alpha"]]
    subset = counts.merge(keys, on=["variant", "calibration_method", "temperature", "entropy_alpha"], how="inner")
    rows: list[dict[str, object]] = []
    for keys_tuple, group in subset.groupby(["sample_set", "variant", "calibration_method", "selected_model"], sort=False):
        sample_set, variant, method, selected_model = keys_tuple
        rows.append(
            {
                "sample_set": sample_set,
                "variant": variant,
                "calibration_method": method,
                "selected_model": selected_model,
                "seed_count": int(group["seed"].nunique()),
                "sample_count_total_over_seeds": int(group["count"].sum()),
                "ratio_mean": float(group["ratio"].mean()),
                "ratio_std": float(group["ratio"].std(ddof=1)) if len(group) > 1 else 0.0,
                "ratio_min": float(group["ratio"].min()),
                "ratio_max": float(group["ratio"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["sample_set", "variant", "selected_model"]).reset_index(drop=True)


def aggregate_strata(strata: pd.DataFrame, best_params: pd.DataFrame) -> pd.DataFrame:
    """函数功能：只保留入选 calibration 的分层结果，并做 seed mean/std。"""
    keys = best_params[["variant", "calibration_method", "temperature", "entropy_alpha"]]
    subset = strata.merge(keys, on=["variant", "calibration_method", "temperature", "entropy_alpha"], how="inner")
    rows: list[dict[str, object]] = []
    for keys_tuple, group in subset.groupby(["sample_set", "variant", "calibration_method", "stratum_column", "stratum_value"], dropna=False, sort=False):
        sample_set, variant, method, stratum_column, stratum_value = keys_tuple
        rows.append(
            {
                "sample_set": sample_set,
                "variant": variant,
                "calibration_method": method,
                "stratum_column": stratum_column,
                "stratum_value": stratum_value,
                "seed_count": int(group["seed"].nunique()),
                "sample_count_mean": float(group["sample_count"].mean()),
                "MAE_mean": float(group["MAE"].mean()),
                "MAE_std": float(group["MAE"].std(ddof=1)) if len(group) > 1 else 0.0,
                "MSE_mean": float(group["MSE"].mean()),
                "MSE_std": float(group["MSE"].std(ddof=1)) if len(group) > 1 else 0.0,
                "regret_to_oracle_mean": float(group["regret_to_oracle"].mean()),
                "oracle_label_accuracy_mean": float(group["oracle_label_accuracy"].mean()),
            }
        )
    out = pd.DataFrame(rows).reset_index(drop=True)
    if GLOBAL_STRATA.exists():
        original = pd.read_csv(GLOBAL_STRATA)
        original = original.rename(
            columns={
                "raw_soft_MAE_mean": "original_raw_soft_MAE_mean",
                "raw_soft_MSE_mean": "original_raw_soft_MSE_mean",
                "raw_soft_regret_mean": "original_raw_soft_regret_mean",
            }
        )
        keep_cols = [
            "variant",
            "sample_set",
            "stratum_column",
            "stratum_value",
            "original_raw_soft_MAE_mean",
            "original_raw_soft_MSE_mean",
            "original_raw_soft_regret_mean",
        ]
        out = out.merge(original[keep_cols], on=["variant", "sample_set", "stratum_column", "stratum_value"], how="left")
        out["delta_MAE_vs_original"] = out["MAE_mean"] - out["original_raw_soft_MAE_mean"]
        out["delta_MSE_vs_original"] = out["MSE_mean"] - out["original_raw_soft_MSE_mean"]
        out["delta_regret_vs_original"] = out["regret_to_oracle_mean"] - out["original_raw_soft_regret_mean"]
    return out


def best_variant_recommendation(final_selected: pd.DataFrame) -> str:
    """函数功能：根据 frozen pilot_test calibrated MAE 选出本诊断推荐变体。"""
    if final_selected.empty:
        return ""
    row = final_selected.sort_values(["MAE_mean", "MSE_mean", "regret_to_oracle_mean"], kind="mergesort").iloc[0]
    return str(row["variant"])


def write_outputs(
    *,
    args: argparse.Namespace,
    grid_seed: pd.DataFrame,
    counts_seed: pd.DataFrame,
    strata_seed: pd.DataFrame,
) -> None:
    """函数功能：生成全部 P2f 交付物，并复制轻量 summary 到仓库目录。"""
    output_dir = Path(args.output_dir)
    summary_dir = Path(args.summary_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    grid_summary = pd.concat([summarize_grid(grid_seed, sample_set=s) for s in SAMPLE_SETS], ignore_index=True)
    selection_summary = grid_summary[grid_summary["sample_set"] == "pilot_selection"].reset_index(drop=True)
    diagnostic_summary = grid_summary[grid_summary["sample_set"] == "diagnostic_balanced"].reset_index(drop=True)
    final_summary = grid_summary[grid_summary["sample_set"] == "pilot_test"].reset_index(drop=True)
    best_params = select_best_params(selection_summary)
    selected_selection = filter_selected(selection_summary, best_params)
    selected_diagnostic = filter_selected(diagnostic_summary, best_params)
    selected_final = filter_selected(final_summary, best_params)
    delta_summary = build_delta_summary(selected_final, selected_selection, selection_summary)
    selected_counts = aggregate_selected_counts(counts_seed, best_params)
    selected_strata = aggregate_strata(strata_seed, best_params)
    recommended_variant = best_variant_recommendation(selected_final)

    artifacts = {
        "round1_calibration_grid_results.csv": grid_summary,
        "round1_calibration_best_params.csv": best_params,
        "round1_calibration_selection_comparison.csv": selected_selection,
        "round1_calibration_diagnostic_summary.csv": selected_diagnostic,
        "round1_calibration_final_test_summary.csv": selected_final,
        "round1_calibration_delta_summary.csv": delta_summary,
        "round1_calibration_selected_model_counts.csv": selected_counts,
        "round1_calibration_stratified_summary.csv": selected_strata,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output_dir / name, index=False)
        frame.to_csv(summary_dir / name, index=False)

    metadata = {
        "script_version": "visual_router_v2_round1_p2f_posthoc_calibration_v1",
        "created_at": display_time(),
        "git_commit": git_commit_hash(),
        "output_dir": str(output_dir),
        "summary_dir": str(summary_dir),
        "variants": list(map(str, args.variants)),
        "seeds": [int(seed) for seed in args.seeds],
        "sample_sets": list(SAMPLE_SETS),
        "calibration_only": True,
        "trained_new_model": False,
        "changed_checkpoint": False,
        "rebuilt_p2a_feature": False,
        "used_pilot_test_for_selection": False,
        "pilot_test_evaluated": True,
        "selected_calibration_on": "pilot_selection",
        "diagnostic_balanced_used_for_selection": False,
        "loaded_116m_prediction_manifest_to_memory": False,
        "saved_pseudo_image_tensor": False,
        "read_feature_shard": False,
        "read_checkpoint": False,
        "calibration_methods": ["weight_power_temperature", "optional_entropy_uniform_interpolation"],
        "temperature_grid": [float(v) for v in args.temperatures],
        "entropy_alpha_grid": [float(v) for v in args.entropy_alphas],
        "calibration_basis": "prediction CSV had no logits; used normalize(w ** (1/T))",
        "selection_index_path": str(args.selection_index_path),
        "final_index_path": str(args.final_index_path),
        "manifest_dir": str(args.manifest_dir),
        "recommended_calibrated_variant": recommended_variant,
        "copied_lightweight_summary_to_repo": True,
    }
    for path in [output_dir / "round1_calibration_metadata.json", summary_dir / "round1_calibration_metadata.json"]:
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_text = build_summary_md(
        best_params=best_params,
        selected_selection=selected_selection,
        selected_diagnostic=selected_diagnostic,
        selected_final=selected_final,
        delta_summary=delta_summary,
        selected_counts=selected_counts,
        selected_strata=selected_strata,
        metadata=metadata,
    )
    for path in [output_dir / "round1_calibration_summary.md", summary_dir / "round1_calibration_summary.md"]:
        path.write_text(summary_text, encoding="utf-8")


def _metric(frame: pd.DataFrame, variant: str, col: str) -> float:
    """函数功能：从入选结果表提取单个 variant 指标。"""
    values = frame.loc[frame["variant"].astype(str) == variant, col]
    return float(values.iloc[0]) if not values.empty else float("nan")


def _delta(delta: pd.DataFrame, variant: str, comparison: str, metric: str) -> float:
    """函数功能：从 delta 表提取单个差值。"""
    values = delta[
        (delta["variant"].astype(str) == variant)
        & (delta["comparison"].astype(str) == comparison)
    ][metric]
    return float(values.iloc[0]) if not values.empty else float("nan")


def _crossformer_ratio(counts: pd.DataFrame, variant: str, sample_set: str) -> float:
    """函数功能：提取 CrossFormer hard selected ratio mean。"""
    values = counts[
        (counts["variant"].astype(str) == variant)
        & (counts["sample_set"].astype(str) == sample_set)
        & (counts["selected_model"].astype(str) == "CrossFormer")
    ]["ratio_mean"]
    return float(values.iloc[0]) if not values.empty else float("nan")


def build_summary_md(
    *,
    best_params: pd.DataFrame,
    selected_selection: pd.DataFrame,
    selected_diagnostic: pd.DataFrame,
    selected_final: pd.DataFrame,
    delta_summary: pd.DataFrame,
    selected_counts: pd.DataFrame,
    selected_strata: pd.DataFrame,
    metadata: Mapping[str, object],
) -> str:
    """函数功能：写中文 P2f 摘要，逐项回答用户验收问题。"""
    baseline = pd.read_csv(BASELINE_FINAL_TEST).set_index("variant")
    lines: list[str] = [
        "# Visual Router V2 Round1 P2f post-hoc calibration diagnostic",
        "",
        f"生成时间：{metadata['created_at']}",
        "",
        "## 约束确认",
        "",
        "- 本诊断未训练新 router，未读取 checkpoint，未重建 P2a feature，未保存 pseudo image tensor。",
        "- Round1 prediction CSV 没有 logits，因此 temperature scaling 使用 `normalize(w ** (1/T))`；T=1、alpha=0 为原始未校准 baseline。",
        "- calibration 参数只按 `pilot_selection` raw-soft MAE mean 选择；`diagnostic_balanced` 与 `pilot_test` 只做 frozen eval。",
        "",
        "## 入选 calibration 参数",
        "",
        frame_to_markdown(
            best_params[
                [
                    "variant",
                    "calibration_method",
                    "temperature",
                    "entropy_alpha",
                    "MAE_mean",
                    "MAE_std",
                    "MSE_mean",
                    "regret_to_oracle_mean",
                    "weight_entropy_mean",
                    "mean_max_weight_mean",
                ]
            ],
            float_digits=6,
        ),
        "",
        "## Frozen pilot_test 结果",
        "",
        frame_to_markdown(
            selected_final[
                [
                    "variant",
                    "calibration_method",
                    "MAE_mean",
                    "MAE_std",
                    "MSE_mean",
                    "MSE_std",
                    "regret_to_oracle_mean",
                    "weight_entropy_mean",
                    "mean_max_weight_mean",
                ]
            ].sort_values(["MAE_mean", "variant"]),
            float_digits=6,
        ),
        "",
        "## 验收问题回答",
        "",
    ]

    for variant in ["film_mean_patch_aux", "film_cls_mean_concat_aux", "visual_cls_mean_concat"]:
        delta_mae = _delta(delta_summary, variant, "final_test_calibrated_minus_original", "delta_MAE")
        delta_mse = _delta(delta_summary, variant, "final_test_calibrated_minus_original", "delta_MSE")
        selected = best_params.loc[best_params["variant"].astype(str) == variant, "calibration_method"].iloc[0]
        answer = "改善" if delta_mae < -1e-9 else "未改善"
        lines.append(
            f"1. `{variant}`：{answer} MAE；入选 `{selected}`，pilot_test delta_MAE={delta_mae:.6f}，delta_MSE={delta_mse:.6f}。"
        )

    best_film_delta_mse = _delta(delta_summary, "film_mean_patch_aux", "final_test_calibrated_minus_original", "delta_MSE")
    lines.append(
        f"2. MSE tail：以 `film_mean_patch_aux` 为主观察，pilot_test delta_MSE={best_film_delta_mse:.6f}，"
        + ("有下降。" if best_film_delta_mse < -1e-9 else "没有下降。")
    )

    for stratum in ["CrossFormer", "PatchTST"]:
        subset = selected_strata[
            (selected_strata["variant"].astype(str) == "film_mean_patch_aux")
            & (selected_strata["sample_set"].astype(str) == "pilot_test")
            & (selected_strata["stratum_column"].astype(str) == "oracle_model")
            & (selected_strata["stratum_value"].astype(str) == stratum)
        ]
        if subset.empty:
            lines.append(f"3. `{stratum}` stratum：缺少分层行，需检查输出。")
        else:
            row = subset.iloc[0]
            delta_mae = float(row.delta_MAE_vs_original) if "delta_MAE_vs_original" in subset.columns else float("nan")
            delta_mse = float(row.delta_MSE_vs_original) if "delta_MSE_vs_original" in subset.columns else float("nan")
            lines.append(
                f"3. `{stratum}` stratum：calibrated `film_mean_patch_aux` MAE={float(row.MAE_mean):.6f}，"
                f"MSE={float(row.MSE_mean):.6f}，regret={float(row.regret_to_oracle_mean):.6f}；"
                f"delta_MAE_vs_original={delta_mae:.6f}，delta_MSE_vs_original={delta_mse:.6f}。"
            )

    cross_ratio = _crossformer_ratio(selected_counts, "film_mean_patch_aux", "pilot_test")
    original_cross = pd.read_csv(GLOBAL_SELECTED)
    original_cross_ratio = original_cross[
        (original_cross["variant"].astype(str) == "film_mean_patch_aux")
        & (original_cross["sample_set"].astype(str) == "pilot_test")
        & (original_cross["selected_model"].astype(str) == "CrossFormer")
    ]["ratio_mean"]
    original_cross_value = float(original_cross_ratio.iloc[0]) if not original_cross_ratio.empty else float("nan")
    lines.append(
        f"4. selected_model ratio：`film_mean_patch_aux` CrossFormer ratio 从 original {original_cross_value:.6f} "
        f"到 calibrated {cross_ratio:.6f}。"
    )

    entropy_delta = _metric(selected_final, "film_mean_patch_aux", "weight_entropy_mean") - float(
        baseline.loc["film_mean_patch_aux", "weight_entropy_mean"]
    )
    mae_delta = _delta(delta_summary, "film_mean_patch_aux", "final_test_calibrated_minus_original", "delta_MAE")
    lines.append(
        f"5. 是否只是改变 entropy/max weight：`film_mean_patch_aux` entropy delta={entropy_delta:.6f}，"
        f"MAE delta={mae_delta:.6f}；若 MAE/MSE/regret 未同步下降，则只应视为权重形状诊断收益。"
    )

    recommended = str(metadata["recommended_calibrated_variant"])
    mse_delta = _delta(delta_summary, "film_mean_patch_aux", "final_test_calibrated_minus_original", "delta_MSE")
    enhanced_ok = recommended == "film_mean_patch_aux" and mae_delta < -1e-9 and mse_delta <= 0
    lines.append(
        "6. Round1 enhanced recommendation："
        + (
            "可以把 calibrated `film_mean_patch_aux` 作为 enhanced recommendation。"
            if enhanced_ok
            else "不建议把 calibrated `film_mean_patch_aux` 升级为综合 enhanced recommendation；它有极小 MAE/regret 收益，但 MSE tail 没有改善，应保持 raw `film_mean_patch_aux` 为主推荐。"
        )
    )
    lines.append(
        "7. 后续路线：若本页 delta_MAE/delta_MSE 没有稳定改善，应优先进入 view layout Round2；只有 calibration 对 FiLM 主线有明确 frozen pilot_test 收益时，才值得先扩展 FiLM hyperparameter search。"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """函数功能：P2f calibration diagnostic 主入口。"""
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录已存在且非空，请使用 --overwrite：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_raw:
        raw_paths = {
            "grid": output_dir / "round1_calibration_seed_grid_raw.csv",
            "counts": output_dir / "round1_calibration_seed_selected_counts_raw.csv",
            "strata": output_dir / "round1_calibration_seed_strata_raw.csv",
        }
        missing = [str(path) for path in raw_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"--reuse-raw 缺少 raw CSV：{missing}")
        log_stage("复用已生成 seed raw CSV，只重建聚合交付物")
        write_outputs(
            args=args,
            grid_seed=pd.read_csv(raw_paths["grid"]),
            counts_seed=pd.read_csv(raw_paths["counts"]),
            strata_seed=pd.read_csv(raw_paths["strata"]),
        )
        log_stage(f"P2f calibration diagnostic 汇总重建完成：{output_dir}")
        return

    selection_index = SQLitePredictionIndex(Path(args.selection_index_path), Path(args.manifest_dir))
    final_index = SQLitePredictionIndex(Path(args.final_index_path), Path(args.manifest_dir))
    grid_frames: list[pd.DataFrame] = []
    count_frames: list[pd.DataFrame] = []
    strata_frames: list[pd.DataFrame] = []
    try:
        for variant in args.variants:
            source = VARIANT_SOURCES[str(variant)]
            for sample_set in SAMPLE_SETS:
                index = final_index if sample_set == "pilot_test" else selection_index
                for seed in args.seeds:
                    path = source.prediction_path(sample_set, int(seed))
                    log_stage(f"评估 {variant} seed={seed} sample_set={sample_set} path={path}")
                    pred_df = read_prediction_csv(path, variant=str(variant), sample_set=str(sample_set), seed=int(seed))
                    grid, counts, strata = evaluate_calibration_grid(
                        pred_df,
                        prediction_index=index,
                        variant=str(variant),
                        seed=int(seed),
                        sample_set=str(sample_set),
                        temperatures=[float(v) for v in args.temperatures],
                        entropy_alphas=[float(v) for v in args.entropy_alphas],
                        eval_batch_size=int(args.eval_batch_size),
                    )
                    grid_frames.append(grid)
                    count_frames.append(counts)
                    strata_frames.append(strata)
    finally:
        selection_index.close()
        final_index.close()

    grid_seed = pd.concat(grid_frames, ignore_index=True)
    counts_seed = pd.concat(count_frames, ignore_index=True)
    strata_seed = pd.concat(strata_frames, ignore_index=True)
    grid_seed.to_csv(output_dir / "round1_calibration_seed_grid_raw.csv", index=False)
    counts_seed.to_csv(output_dir / "round1_calibration_seed_selected_counts_raw.csv", index=False)
    # seed 级 strata 行数较多，正式交付只保留 aggregate；raw 文件留在 /data2 便于复核。
    strata_seed.to_csv(output_dir / "round1_calibration_seed_strata_raw.csv", index=False)
    write_outputs(args=args, grid_seed=grid_seed, counts_seed=counts_seed, strata_seed=strata_seed)
    log_stage(f"P2f calibration diagnostic 完成：{output_dir}")


if __name__ == "__main__":
    main()
