#!/usr/bin/env python3
"""生成 Round2 spatial panel 相对 Round1 FiLM 的分层与错误尾部分析。

本脚本只读取既有 Round1/Round2f summary CSV 和逐样本 prediction CSV，
不训练模型、不运行 ViT、不生成 feature cache，也不改动任何 imageization 或
router head 逻辑。输出用于解释 `spatial_panel_3view + film_mean_patch_aux`
的收益来源和剩余弱点。
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiment_summaries" / "visual_router_v2_round2"

ROUND1_SUMMARY = (
    ROOT
    / "experiment_summaries"
    / "visual_router_v2_round1"
    / "p2e_film_final_test_extension"
)
ROUND1_SELECTION_SUMMARY = (
    ROOT
    / "experiment_summaries"
    / "visual_router_v2_round1"
    / "p2e_film"
)
ROUND2_SUMMARY = OUT_DIR / "p0_spatial_panel_mainline"

ROUND1_RUN = Path("/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round1_film_final_test_extension")
ROUND2_RUN = Path("/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline")

SEEDS = [16, 17, 18]
EXPERTS = ["CrossFormer", "DLinear", "ES", "NaiveForecaster", "PatchTST"]
STRATA_COLUMNS = [
    "oracle_model",
    "dataset_name",
    "group_name",
    "error_gap_quantile",
    "forecastability_cat",
    "season_strength_cat",
    "trend_strength_cat",
    "cv_cat",
    "missing_ratio_cat",
]


def _now_cst() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _prediction_paths() -> dict[str, list[Path]]:
    return {
        "round1_film_mean_patch_aux": [
            ROUND1_RUN
            / "tasks"
            / f"film_mean_patch_aux_seed{seed}"
            / f"predictions_film_mean_patch_aux_seed{seed}_pilot_test.csv"
            for seed in SEEDS
        ],
        "round2_spatial_panel_3view": [
            ROUND2_RUN
            / "tasks"
            / f"spatial_panel_3view_seed{seed}"
            / f"predictions_spatial_panel_3view_seed{seed}_pilot_test.csv"
            for seed in SEEDS
        ],
    }


def _load_predictions(paths: list[Path], system_name: str) -> pd.DataFrame:
    """读取逐 seed prediction CSV，并补齐统一系统名。

    这些 CSV 已由前序 frozen eval 产生；这里仅做只读统计。Round2 文件没有
    `order_index` 列，因此后续统一以稳定的 `sample_key` 对齐。
    """
    frames = []
    for path in paths:
        df = _read_csv(path)
        df["system_name"] = system_name
        df["source_path"] = str(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _build_sample_level(df: pd.DataFrame) -> pd.DataFrame:
    """把三 seed 预测折叠到 sample 级别，便于 tail overlap 和分层比较。"""
    meta_cols = [
        "sample_set",
        "config_name",
        "sample_key",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "oracle_model",
        "error_gap_quantile",
        "cluster",
        "group_name",
        "forecastability_cat",
        "season_strength_cat",
        "trend_strength_cat",
        "cv_cat",
        "missing_ratio_cat",
    ]
    numeric_cols = [
        "selected_value",
        "oracle_value",
        "regret_to_oracle",
        "weight_entropy",
        "normalized_weight_entropy",
        "max_weight",
        "hard_top1_mae_from_array",
        "hard_top1_mse_from_array",
        "soft_fusion_mae",
        "soft_fusion_mse",
        *[f"weight_{expert}" for expert in ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]],
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["oracle_label_correct_num"] = df["oracle_label_correct"].astype(float)

    grouped = df.groupby("sample_key", sort=False)
    # 样本元信息由固定 manifest 决定，三 seed 完全一致；取首行即可，避免
    # 对 75k 个 sample 做 Python 层 mode 聚合导致分析脚本过慢。
    base = grouped[meta_cols].first()
    means = grouped[numeric_cols + ["oracle_label_correct_num"]].mean()
    selected_counts = (
        df.groupby(["sample_key", "selected_model"], sort=False)
        .size()
        .reset_index(name="selected_count")
        .sort_values(["sample_key", "selected_count", "selected_model"], ascending=[True, False, True])
    )
    selected_mode = (
        selected_counts.drop_duplicates("sample_key")
        .set_index("sample_key")["selected_model"]
        .rename("selected_model_mode")
    )
    selected_unique = grouped["selected_model"].nunique().rename("selected_model_seed_unique_count")
    out = pd.concat([base, means, selected_mode, selected_unique], axis=1).reset_index(drop=True)
    # prediction CSV 中的 `regret_to_oracle` 是 hard selected expert regret。
    # raw-soft 主指标的 regret 需要按 fused MAE 相对 oracle value 重新派生。
    out["soft_fusion_regret"] = out["soft_fusion_mae"] - out["oracle_value"]
    out["hard_top1_regret"] = out["hard_top1_mae_from_array"] - out["oracle_value"]
    return out


def _selected_ratios(df: pd.DataFrame, subset: pd.Series | None = None) -> dict[str, float]:
    frame = df if subset is None else df.loc[subset]
    total = len(frame)
    counts = frame["selected_model"].value_counts()
    return {f"selected_ratio_{expert}": float(counts.get(expert, 0) / total) if total else 0.0 for expert in EXPERTS}


def _oracle_ratios(sample_df: pd.DataFrame, subset: pd.Series | None = None) -> dict[str, float]:
    frame = sample_df if subset is None else sample_df.loc[subset]
    total = len(frame)
    counts = frame["oracle_model"].value_counts()
    return {f"oracle_ratio_{expert}": float(counts.get(expert, 0) / total) if total else 0.0 for expert in EXPERTS}


def _aggregate_subset(sample_df: pd.DataFrame, pred_df: pd.DataFrame, subset: pd.Series) -> dict[str, Any]:
    frame = sample_df.loc[subset]
    seed_frame = pred_df[pred_df["sample_key"].isin(frame["sample_key"])]
    row: dict[str, Any] = {
        "sample_count": int(len(frame)),
        "raw_soft_MAE": float(frame["soft_fusion_mae"].mean()),
        "raw_soft_MSE": float(frame["soft_fusion_mse"].mean()),
        "raw_soft_regret": float(frame["soft_fusion_regret"].mean()),
        "hard_top1_MAE": float(frame["hard_top1_mae_from_array"].mean()),
        "hard_top1_MSE": float(frame["hard_top1_mse_from_array"].mean()),
        "hard_top1_regret": float(frame["hard_top1_regret"].mean()),
        "oracle_label_accuracy": float(frame["oracle_label_correct_num"].mean()),
        "weight_entropy": float(frame["weight_entropy"].mean()),
        "normalized_weight_entropy": float(frame["normalized_weight_entropy"].mean()),
        "mean_max_weight": float(frame["max_weight"].mean()),
        "soft_minus_hard_MAE": float(frame["soft_fusion_mae"].mean() - frame["hard_top1_mae_from_array"].mean()),
    }
    row.update(_selected_ratios(seed_frame))
    row.update(_oracle_ratios(frame))
    return row


def _load_overall_rows() -> pd.DataFrame:
    """读取 overall / seed stability 行，覆盖 selection 和 frozen pilot_test。"""
    rows: list[dict[str, Any]] = []
    specs = [
        (
            "round1_film_mean_patch_aux",
            "Round1 film_mean_patch_aux",
            {
                "pilot_selection": ROUND1_SELECTION_SUMMARY / "round1_film_variant_seed_results.csv",
                "pilot_test": ROUND1_SUMMARY / "round1_film_final_test_extension_variant_seed_results.csv",
            },
            "film_mean_patch_aux",
        ),
        (
            "round2_spatial_panel_3view",
            "Round2 spatial_panel_3view + film_mean_patch_aux",
            {
                "pilot_selection": ROUND2_SUMMARY / "round2_p0_spatial_variant_seed_results.csv",
                "pilot_test": ROUND2_SUMMARY / "round2_p0_spatial_variant_seed_results.csv",
            },
            "spatial_panel_3view",
        ),
    ]
    for system_name, display_name, seed_paths, variant in specs:
        for sample_set in ["pilot_selection", "pilot_test"]:
            seed_path = seed_paths[sample_set]
            seed_df = _read_csv(seed_path)
            for method_kind in ["raw_soft_fusion", "hard_top1"]:
                method = f"{variant}_{method_kind}"
                matched = seed_df[(seed_df["sample_set"] == sample_set) & (seed_df["method"] == method)]
                if matched.empty:
                    continue
                rows.append(
                    {
                        "metric_group": "overall",
                        "stratum_column": "overall",
                        "stratum_value": sample_set,
                        "method_kind": method_kind,
                        "system_name": system_name,
                        "display_name": display_name,
                        "sample_count": int(matched["sample_count"].iloc[0]),
                        "raw_soft_MAE": float(matched["MAE"].mean()) if method_kind == "raw_soft_fusion" else None,
                        "raw_soft_MSE": float(matched["MSE"].mean()) if method_kind == "raw_soft_fusion" else None,
                        "raw_soft_regret": float(matched["regret_to_oracle"].mean()) if method_kind == "raw_soft_fusion" else None,
                        "hard_top1_MAE": float(matched["MAE"].mean()) if method_kind == "hard_top1" else None,
                        "hard_top1_MSE": float(matched["MSE"].mean()) if method_kind == "hard_top1" else None,
                        "hard_top1_regret": float(matched["regret_to_oracle"].mean()) if method_kind == "hard_top1" else None,
                        "oracle_label_accuracy": float(matched["oracle_label_accuracy"].mean()),
                        "weight_entropy": float(matched["weight_entropy"].mean()),
                        "normalized_weight_entropy": float(matched["normalized_weight_entropy"].mean()),
                        "mean_max_weight": float(matched["mean_max_weight"].mean()),
                        "MAE_std": float(matched["MAE"].std(ddof=1)) if len(matched) > 1 else 0.0,
                        "MSE_std": float(matched["MSE"].std(ddof=1)) if len(matched) > 1 else 0.0,
                        "source_file": str(seed_path.relative_to(ROOT)),
                    }
                )
    return pd.DataFrame(rows)


def _build_strata_metrics(round1_pred: pd.DataFrame, round2_pred: pd.DataFrame) -> pd.DataFrame:
    r1_sample = _build_sample_level(round1_pred)
    r2_sample = _build_sample_level(round2_pred)
    shared_keys = set(r1_sample["sample_key"]).intersection(set(r2_sample["sample_key"]))
    if len(shared_keys) != len(r1_sample) or len(shared_keys) != len(r2_sample):
        raise RuntimeError(f"Round1/Round2 pilot_test sample_key 不完全一致：shared={len(shared_keys)}")

    # high-error/high-regret rate 使用各系统自身 top5% 阈值，用于定位每个 stratum 在本系统内的风险集中度。
    thresholds = {
        "round1_soft_mae_p95": float(r1_sample["soft_fusion_mae"].quantile(0.95)),
        "round2_soft_mae_p95": float(r2_sample["soft_fusion_mae"].quantile(0.95)),
        "round1_regret_p95": float(r1_sample["soft_fusion_regret"].quantile(0.95)),
        "round2_regret_p95": float(r2_sample["soft_fusion_regret"].quantile(0.95)),
    }

    rows: list[dict[str, Any]] = []
    for col in STRATA_COLUMNS:
        if col not in r1_sample.columns or col not in r2_sample.columns:
            continue
        values = sorted(set(r1_sample[col].dropna().astype(str)).union(set(r2_sample[col].dropna().astype(str))))
        for value in values:
            r1_mask = r1_sample[col].astype(str) == value
            r2_mask = r2_sample[col].astype(str) == value
            if int(r1_mask.sum()) == 0 or int(r2_mask.sum()) == 0:
                continue
            r1 = _aggregate_subset(r1_sample, round1_pred, r1_mask)
            r2 = _aggregate_subset(r2_sample, round2_pred, r2_mask)
            row: dict[str, Any] = {
                "metric_group": "strata",
                "stratum_column": col,
                "stratum_value": value,
                "sample_count": r2["sample_count"],
                "round1_raw_soft_MAE": r1["raw_soft_MAE"],
                "round2_raw_soft_MAE": r2["raw_soft_MAE"],
                "delta_raw_soft_MAE_round2_minus_round1": r2["raw_soft_MAE"] - r1["raw_soft_MAE"],
                "round1_raw_soft_MSE": r1["raw_soft_MSE"],
                "round2_raw_soft_MSE": r2["raw_soft_MSE"],
                "delta_raw_soft_MSE_round2_minus_round1": r2["raw_soft_MSE"] - r1["raw_soft_MSE"],
                "round1_raw_soft_regret": r1["raw_soft_regret"],
                "round2_raw_soft_regret": r2["raw_soft_regret"],
                "delta_raw_soft_regret_round2_minus_round1": r2["raw_soft_regret"] - r1["raw_soft_regret"],
                "round1_hard_top1_MAE": r1["hard_top1_MAE"],
                "round2_hard_top1_MAE": r2["hard_top1_MAE"],
                "delta_hard_top1_MAE_round2_minus_round1": r2["hard_top1_MAE"] - r1["hard_top1_MAE"],
                "round1_oracle_label_accuracy": r1["oracle_label_accuracy"],
                "round2_oracle_label_accuracy": r2["oracle_label_accuracy"],
                "delta_oracle_label_accuracy_round2_minus_round1": r2["oracle_label_accuracy"] - r1["oracle_label_accuracy"],
                "round1_weight_entropy": r1["weight_entropy"],
                "round2_weight_entropy": r2["weight_entropy"],
                "delta_weight_entropy_round2_minus_round1": r2["weight_entropy"] - r1["weight_entropy"],
                "round1_mean_max_weight": r1["mean_max_weight"],
                "round2_mean_max_weight": r2["mean_max_weight"],
                "delta_mean_max_weight_round2_minus_round1": r2["mean_max_weight"] - r1["mean_max_weight"],
                "round1_high_error_rate_top5": float((r1_sample.loc[r1_mask, "soft_fusion_mae"] >= thresholds["round1_soft_mae_p95"]).mean()),
                "round2_high_error_rate_top5": float((r2_sample.loc[r2_mask, "soft_fusion_mae"] >= thresholds["round2_soft_mae_p95"]).mean()),
                "round1_high_regret_rate_top5": float((r1_sample.loc[r1_mask, "soft_fusion_regret"] >= thresholds["round1_regret_p95"]).mean()),
                "round2_high_regret_rate_top5": float((r2_sample.loc[r2_mask, "soft_fusion_regret"] >= thresholds["round2_regret_p95"]).mean()),
            }
            for expert in EXPERTS:
                row[f"round1_selected_ratio_{expert}"] = r1[f"selected_ratio_{expert}"]
                row[f"round2_selected_ratio_{expert}"] = r2[f"selected_ratio_{expert}"]
                row[f"delta_selected_ratio_{expert}_round2_minus_round1"] = (
                    r2[f"selected_ratio_{expert}"] - r1[f"selected_ratio_{expert}"]
                )
                row[f"oracle_ratio_{expert}"] = r2[f"oracle_ratio_{expert}"]
            rows.append(row)

    overall = _load_overall_rows()
    strata = pd.DataFrame(rows)
    return pd.concat([overall, strata], ignore_index=True, sort=False)


def _distribution_rows(
    row_prefix: dict[str, Any],
    sample_df: pd.DataFrame,
    sample_keys: set[str],
    column: str,
    row_type: str,
) -> list[dict[str, Any]]:
    subset = sample_df[sample_df["sample_key"].isin(sample_keys)]
    total = len(subset)
    counts = Counter(subset[column].dropna().astype(str))
    return [
        {
            **row_prefix,
            "row_type": row_type,
            "category": category,
            "count": int(count),
            "ratio": float(count / total) if total else 0.0,
        }
        for category, count in sorted(counts.items())
    ]


def _build_tail_metrics(round1_pred: pd.DataFrame, round2_pred: pd.DataFrame) -> pd.DataFrame:
    r1 = _build_sample_level(round1_pred)
    r2 = _build_sample_level(round2_pred)
    r1_by_key = r1.set_index("sample_key")
    r2_by_key = r2.set_index("sample_key")

    rows: list[dict[str, Any]] = []
    metric_specs = [
        ("top_1pct_soft_MAE", "soft_fusion_mae", 0.99),
        ("top_5pct_soft_MAE", "soft_fusion_mae", 0.95),
        ("top_1pct_regret", "soft_fusion_regret", 0.99),
        ("top_5pct_regret", "soft_fusion_regret", 0.95),
    ]
    sample_frames = {
        "round1_film_mean_patch_aux": r1,
        "round2_spatial_panel_3view": r2,
    }
    for tail_name, metric_col, quantile in metric_specs:
        tail_sets: dict[str, set[str]] = {}
        thresholds: dict[str, float] = {}
        for basis_name, frame in sample_frames.items():
            threshold = float(frame[metric_col].quantile(quantile))
            thresholds[basis_name] = threshold
            tail_sets[basis_name] = set(frame.loc[frame[metric_col] >= threshold, "sample_key"])

        overlap = len(tail_sets["round1_film_mean_patch_aux"] & tail_sets["round2_spatial_panel_3view"])
        union = len(tail_sets["round1_film_mean_patch_aux"] | tail_sets["round2_spatial_panel_3view"])
        overlap_rate = overlap / union if union else 0.0

        for basis_name, keys in tail_sets.items():
            for eval_name, indexed in [
                ("round1_film_mean_patch_aux", r1_by_key),
                ("round2_spatial_panel_3view", r2_by_key),
            ]:
                subset = indexed.loc[list(keys)]
                row = {
                    "row_type": "summary",
                    "tail_name": tail_name,
                    "tail_basis_system": basis_name,
                    "evaluated_system": eval_name,
                    "tail_metric": metric_col,
                    "quantile": quantile,
                    "threshold": thresholds[basis_name],
                    "sample_count": int(len(subset)),
                    "tail_overlap_count": int(overlap),
                    "tail_union_count": int(union),
                    "tail_overlap_jaccard": float(overlap_rate),
                    "mean_soft_MAE": float(subset["soft_fusion_mae"].mean()),
                    "mean_soft_MSE": float(subset["soft_fusion_mse"].mean()),
                    "mean_regret": float(subset["soft_fusion_regret"].mean()),
                    "mean_hard_top1_MAE": float(subset["hard_top1_mae_from_array"].mean()),
                    "oracle_label_accuracy": float(subset["oracle_label_correct_num"].mean()),
                    "weight_entropy": float(subset["weight_entropy"].mean()),
                    "mean_max_weight": float(subset["max_weight"].mean()),
                    "category": "",
                    "count": "",
                    "ratio": "",
                }
                rows.append(row)
            prefix = {
                "tail_name": tail_name,
                "tail_basis_system": basis_name,
                "evaluated_system": basis_name,
                "tail_metric": metric_col,
                "quantile": quantile,
                "threshold": thresholds[basis_name],
                "sample_count": int(len(keys)),
                "tail_overlap_count": int(overlap),
                "tail_union_count": int(union),
                "tail_overlap_jaccard": float(overlap_rate),
                "mean_soft_MAE": "",
                "mean_soft_MSE": "",
                "mean_regret": "",
                "mean_hard_top1_MAE": "",
                "oracle_label_accuracy": "",
                "weight_entropy": "",
                "mean_max_weight": "",
            }
            frame = sample_frames[basis_name]
            rows.extend(_distribution_rows(prefix, frame, keys, "oracle_model", "oracle_model_distribution"))
            rows.extend(_distribution_rows(prefix, frame, keys, "selected_model_mode", "selected_model_distribution"))

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = _prediction_paths()
    for path_list in paths.values():
        for path in path_list:
            if not path.exists():
                raise FileNotFoundError(path)

    round1_pred = _load_predictions(paths["round1_film_mean_patch_aux"], "round1_film_mean_patch_aux")
    round2_pred = _load_predictions(paths["round2_spatial_panel_3view"], "round2_spatial_panel_3view")

    strata = _build_strata_metrics(round1_pred, round2_pred)
    tail = _build_tail_metrics(round1_pred, round2_pred)

    strata_path = OUT_DIR / "spatial_panel_strata_metrics.csv"
    tail_path = OUT_DIR / "spatial_panel_error_tail_metrics.csv"
    meta_path = OUT_DIR / "spatial_panel_strata_metadata.json"
    strata.to_csv(strata_path, index=False)
    tail.to_csv(tail_path, index=False)

    metadata = {
        "status": "completed",
        "generated_at": _now_cst(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "analysis_scope": "Read-only Round1/Round2f pilot_test strata and error-tail analysis.",
        "boundaries": {
            "trained_model": False,
            "ran_vit": False,
            "generated_feature_cache": False,
            "generated_new_samples": False,
            "modified_imageization": False,
            "modified_router_head": False,
            "used_test_for_training_or_selection": False,
        },
        "round1_system": "film_mean_patch_aux",
        "round2_system": "spatial_panel_3view + film_mean_patch_aux",
        "seeds": SEEDS,
        "sample_set": "pilot_test",
        "sample_count_per_seed": {
            "round1": int(len(round1_pred) / len(SEEDS)),
            "round2": int(len(round2_pred) / len(SEEDS)),
        },
        "prediction_sources": {k: [str(p) for p in v] for k, v in paths.items()},
        "summary_sources": [
            str((ROUND1_SELECTION_SUMMARY / "round1_film_variant_seed_results.csv").relative_to(ROOT)),
            str((ROUND1_SUMMARY / "round1_film_final_test_extension_variant_seed_results.csv").relative_to(ROOT)),
            str((ROUND1_SUMMARY / "round1_film_final_test_extension_stratified_summary.csv").relative_to(ROOT)),
            str((ROUND2_SUMMARY / "round2_p0_spatial_variant_seed_results.csv").relative_to(ROOT)),
            str((ROUND2_SUMMARY / "round2_p0_spatial_stratified_summary.csv").relative_to(ROOT)),
        ],
        "output_files": [str(strata_path.relative_to(ROOT)), str(tail_path.relative_to(ROOT)), str(meta_path.relative_to(ROOT))],
        "strata_columns": STRATA_COLUMNS,
        "tail_definitions": [
            "top 1% soft_fusion_mae",
            "top 5% soft_fusion_mae",
            "top 1% soft_fusion_regret",
            "top 5% soft_fusion_regret",
        ],
        "aggregation_note": "逐样本 prediction CSV 先按 sample_key 聚合三 seed 均值；selected_model 使用 seed mode，selected ratio 使用逐 seed 行计数。",
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote {strata_path}")
    print(f"wrote {tail_path}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
