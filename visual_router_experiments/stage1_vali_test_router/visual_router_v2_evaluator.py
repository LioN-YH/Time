#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round 0 evaluator 的共享工具。

设计边界：
    - 只消费既有 Visual Router / TimeFuse-style 预测、checkpoint、oracle label
      和 prediction cache，不训练新模型；
    - test 子集优先从 full-scale 逐样本预测 CSV 按 sample_key 抽取；
    - vali 子集只在固定 P0 sample_key 上重算 checkpoint forward，不读取 test
      oracle 作为可部署特征；
    - 所有指标都按传入样本顺序汇总，避免全量 manifest join。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from visual_router_experiments.stage1_vali_test_router.fusion_utils import MODEL_COLUMNS


WEIGHT_COLUMNS = [f"weight_{model_name}" for model_name in MODEL_COLUMNS]
TSF_STRATA_COLUMNS = [
    "dataset_name",
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


def read_sample_csv(path: Path) -> pd.DataFrame:
    """函数功能：读取 P0 sample CSV，并按 order_index 做稳定排序和唯一性校验。"""
    df = pd.read_csv(path)
    required = {"sample_set", "order_index", "sample_key", "split", *TSF_STRATA_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"sample CSV 缺少字段：{path} missing={missing}")
    if df["sample_key"].duplicated().any():
        dup = df.loc[df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"sample CSV 中 sample_key 重复：{path} 示例={dup}")
    if df["order_index"].duplicated().any():
        dup = df.loc[df["order_index"].duplicated(), "order_index"].head(10).tolist()
        raise ValueError(f"sample CSV 中 order_index 重复：{path} 示例={dup}")
    return df.sort_values("order_index").reset_index(drop=True)


def extract_csv_rows_by_sample_keys(
    csv_path: Path,
    *,
    sample_keys: Sequence[str],
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    函数功能：
        从大 CSV 中只抽取目标 sample_key 行。

    说明：
        该函数用于 5-8GB 的 full-scale prediction CSV。它只保留 P0 子集命中行，
        不把 full-scale CSV 全量读入内存，也不访问 116M prediction manifest。
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV：{csv_path}")
    key_order = {str(key): idx for idx, key in enumerate(sample_keys)}
    rows: List[pd.DataFrame] = []
    matched_count = 0
    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=int(chunksize)), start=1):
        matched = chunk[chunk["sample_key"].astype(str).isin(key_order)].copy()
        if not matched.empty:
            rows.append(matched)
            matched_count += int(len(matched))
        if chunk_idx == 1 or chunk_idx % 25 == 0:
            print(
                f"[extract_csv] path={csv_path.name} chunks={chunk_idx} matched={matched_count}/{len(key_order)}",
                flush=True,
            )
        if matched_count >= len(key_order):
            break
    if not rows:
        raise ValueError(f"CSV 中没有命中任何 P0 sample_key：{csv_path}")
    out = pd.concat(rows, ignore_index=True)
    if out["sample_key"].duplicated().any():
        dup = out.loc[out["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"CSV 抽取结果 sample_key 重复：{csv_path} 示例={dup}")
    present_keys = set(out["sample_key"].astype(str))
    missing = [key for key in sample_keys if str(key) not in present_keys]
    if missing:
        raise ValueError(f"CSV 抽取结果缺少 sample_key：{csv_path} missing_count={len(missing)} 示例={missing[:10]}")
    out["_order_index"] = out["sample_key"].astype(str).map(key_order)
    return out.sort_values("_order_index").drop(columns=["_order_index"]).reset_index(drop=True)


def align_with_sample_frame(sample_df: pd.DataFrame, pred_df: pd.DataFrame, *, required_col: str = "selected_model") -> pd.DataFrame:
    """函数功能：把预测行严格对齐到 P0 order_index，并补齐 sample CSV 的 TSF 字段。"""
    left = sample_df[["order_index", "sample_key", *TSF_STRATA_COLUMNS]].copy()
    merged = left.merge(pred_df, on="sample_key", how="left", suffixes=("_sample", ""))
    if required_col not in merged.columns:
        raise ValueError(f"对齐检查字段不存在：required_col={required_col} columns={list(merged.columns)}")
    missing = merged[merged[required_col].isna()]["sample_key"].head(10).tolist()
    if missing:
        raise ValueError(f"预测结果无法覆盖 sample CSV，示例 missing sample_key={missing}")
    for col in TSF_STRATA_COLUMNS:
        sample_col = f"{col}_sample"
        if sample_col in merged.columns:
            if col not in merged.columns:
                merged[col] = merged[sample_col]
            else:
                merged[col] = merged[col].fillna(merged[sample_col])
            merged = merged.drop(columns=[sample_col])
    return merged.sort_values("order_index").reset_index(drop=True)


def make_method_rows(
    *,
    sample_set: str,
    method: str,
    pred_df: pd.DataFrame,
    mae_col: str,
    mse_col: str | None,
    selected_model_col: str = "selected_model",
    include_weights: bool = True,
) -> pd.DataFrame:
    """
    函数功能：
        把 hard/soft/oracle/global rows 规整成统一逐样本 schema。

    说明：
        hard top-1 的 MAE/MSE 来自 selected expert；raw soft fusion 的 MAE/MSE
        来自数组加权复算列。oracle/global 也映射到同一 schema，便于统一分层。
    """
    rows = pd.DataFrame(
        {
            "sample_set": sample_set,
            "method": method,
            "sample_key": pred_df["sample_key"].astype(str),
            "selected_model": pred_df[selected_model_col].astype(str),
            "mae": pd.to_numeric(pred_df[mae_col], errors="raise"),
            "mse": pd.to_numeric(pred_df[mse_col], errors="raise") if mse_col is not None else np.nan,
            "oracle_model": pred_df["oracle_model"].astype(str),
            "oracle_mae": pd.to_numeric(pred_df["oracle_value"], errors="raise"),
            "oracle_label_correct": pred_df[selected_model_col].astype(str) == pred_df["oracle_model"].astype(str),
        }
    )
    rows["regret_to_oracle"] = rows["mae"] - rows["oracle_mae"]
    for col in TSF_STRATA_COLUMNS:
        rows[col] = pred_df[col].values
    if include_weights:
        for col in WEIGHT_COLUMNS:
            rows[col] = pd.to_numeric(pred_df[col], errors="coerce") if col in pred_df.columns else np.nan
        rows["weight_entropy"] = pd.to_numeric(pred_df["weight_entropy"], errors="coerce") if "weight_entropy" in pred_df.columns else np.nan
        rows["normalized_weight_entropy"] = (
            pd.to_numeric(pred_df["normalized_weight_entropy"], errors="coerce")
            if "normalized_weight_entropy" in pred_df.columns
            else np.nan
        )
        rows["mean_max_weight"] = pd.to_numeric(pred_df["max_weight"], errors="coerce") if "max_weight" in pred_df.columns else np.nan
    else:
        rows["weight_entropy"] = np.nan
        rows["normalized_weight_entropy"] = np.nan
        rows["mean_max_weight"] = np.nan
        for col in WEIGHT_COLUMNS:
            rows[col] = np.nan
    return rows


def summarize_method_rows(rows: pd.DataFrame, *, group_cols: Sequence[str] = ()) -> pd.DataFrame:
    """函数功能：按 method 或额外分层字段汇总 Round 0 指标。"""
    out_rows: List[Dict[str, object]] = []
    by_cols = ["sample_set", "method", *group_cols]
    for keys, group in rows.groupby(by_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(by_cols, keys)}
        row.update(
            {
                "sample_count": int(len(group)),
                "MAE": float(group["mae"].mean()),
                "MSE": float(group["mse"].mean()) if group["mse"].notna().any() else np.nan,
                "regret_to_oracle": float(group["regret_to_oracle"].mean()),
                "oracle_label_accuracy": float(group["oracle_label_correct"].mean()),
                "weight_entropy": float(group["weight_entropy"].mean()) if group["weight_entropy"].notna().any() else np.nan,
                "normalized_weight_entropy": (
                    float(group["normalized_weight_entropy"].mean())
                    if group["normalized_weight_entropy"].notna().any()
                    else np.nan
                ),
                "mean_max_weight": float(group["mean_max_weight"].mean()) if group["mean_max_weight"].notna().any() else np.nan,
            }
        )
        out_rows.append(row)
    return pd.DataFrame(out_rows).reset_index(drop=True)


def selected_model_counts(rows: pd.DataFrame) -> pd.DataFrame:
    """函数功能：输出每个 method 的 selected_model 计数和比例。"""
    count_df = (
        rows.groupby(["sample_set", "method", "selected_model"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = count_df.groupby(["sample_set", "method"])["count"].transform("sum")
    count_df["ratio"] = count_df["count"] / totals
    return count_df.sort_values(["sample_set", "method", "selected_model"]).reset_index(drop=True)


def add_oracle_and_global_rows(
    *,
    sample_set: str,
    sample_df: pd.DataFrame,
    label_df: pd.DataFrame,
    global_best_model: str,
) -> pd.DataFrame:
    """
    函数功能：
        为当前 P0 样本生成 oracle_top1 与 global_best_single 逐样本行。

    说明：
        global_best_model 必须由 selection/vali 学得，再用于 pilot_test；对
        pilot_selection 自身汇总时同样显式记录其来源，避免把 test 信息泄露进
        deployable baseline。
    """
    base = align_with_sample_frame(sample_df, label_df, required_col="oracle_model")
    oracle_pred = base.copy()
    oracle_pred["selected_model"] = oracle_pred["oracle_model"]
    oracle_pred["selected_value"] = oracle_pred["oracle_value"]
    oracle_pred["hard_top1_mse_from_array"] = np.nan
    oracle_rows = make_method_rows(
        sample_set=sample_set,
        method="oracle_top1",
        pred_df=oracle_pred,
        mae_col="selected_value",
        mse_col="hard_top1_mse_from_array",
        include_weights=False,
    )
    global_pred = base.copy()
    global_pred["selected_model"] = str(global_best_model)
    global_pred["selected_value"] = pd.to_numeric(global_pred[str(global_best_model)], errors="raise")
    global_pred["hard_top1_mse_from_array"] = np.nan
    global_rows = make_method_rows(
        sample_set=sample_set,
        method="global_best_single",
        pred_df=global_pred,
        mae_col="selected_value",
        mse_col="hard_top1_mse_from_array",
        include_weights=False,
    )
    return pd.concat([global_rows, oracle_rows], ignore_index=True)


def choose_global_best_model(selection_label_df: pd.DataFrame) -> str:
    """函数功能：只用 selection/vali 标签选择 global_best_single 专家。"""
    means = selection_label_df[MODEL_COLUMNS].mean()
    return str(means.idxmin())


def paired_diagnostics(
    *,
    sample_set: str,
    visual_rows: pd.DataFrame,
    timefuse_rows: pd.DataFrame,
    close_threshold: float = 1e-6,
    bad_regret_threshold: float = 0.1,
) -> pd.DataFrame:
    """函数功能：输出 Visual 与 TimeFuse 的 paired per-sample 诊断。"""
    visual = visual_rows[visual_rows["method"] == "visual_router_hard_top1"].copy()
    timefuse = timefuse_rows[timefuse_rows["method"] == "timefuse_hard_top1"].copy()
    merged = visual.merge(
        timefuse,
        on="sample_key",
        how="inner",
        suffixes=("_visual", "_timefuse"),
    )
    if len(merged) != len(visual) or len(merged) != len(timefuse):
        raise ValueError(f"paired diagnostics 无法一一对齐：sample_set={sample_set}")
    diff = merged["mae_visual"] - merged["mae_timefuse"]
    out = pd.DataFrame(
        {
            "sample_set": sample_set,
            "sample_key": merged["sample_key"],
            "visual_mae": merged["mae_visual"],
            "timefuse_mae": merged["mae_timefuse"],
            "visual_minus_timefuse_mae": diff,
            "visual_selected_model": merged["selected_model_visual"],
            "timefuse_selected_model": merged["selected_model_timefuse"],
            "selected_model_match": merged["selected_model_visual"] == merged["selected_model_timefuse"],
            "visual_wins": diff < -float(close_threshold),
            "timefuse_wins": diff > float(close_threshold),
            "both_close": diff.abs() <= float(close_threshold),
            "both_bad": (merged["regret_to_oracle_visual"] > float(bad_regret_threshold))
            & (merged["regret_to_oracle_timefuse"] > float(bad_regret_threshold)),
        }
    )
    for col in TSF_STRATA_COLUMNS:
        out[col] = merged[f"{col}_visual"].values
    return out


def paired_summary(paired_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：把 paired diagnostics 压成 summary 行，同时保留差值分布。"""
    rows = []
    for sample_set, group in paired_df.groupby("sample_set", sort=False):
        diff = group["visual_minus_timefuse_mae"]
        rows.append(
            {
                "sample_set": sample_set,
                "sample_count": int(len(group)),
                "visual_wins": int(group["visual_wins"].sum()),
                "timefuse_wins": int(group["timefuse_wins"].sum()),
                "both_close": int(group["both_close"].sum()),
                "both_bad": int(group["both_bad"].sum()),
                "selected_model_match_count": int(group["selected_model_match"].sum()),
                "selected_model_match_ratio": float(group["selected_model_match"].mean()),
                "visual_minus_timefuse_mae_mean": float(diff.mean()),
                "visual_minus_timefuse_mae_p05": float(diff.quantile(0.05)),
                "visual_minus_timefuse_mae_p50": float(diff.quantile(0.50)),
                "visual_minus_timefuse_mae_p95": float(diff.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def compare_round0_direction(main_df: pd.DataFrame, full_scale_reference: Mapping[str, float]) -> Tuple[bool, List[str]]:
    """函数功能：检查 pilot_test 是否复现 full-scale 的关键相对方向。"""
    table = main_df.set_index("method")
    messages: List[str] = []
    ok = True
    visual_hard_mae = float(table.loc["visual_router_hard_top1", "MAE"])
    timefuse_hard_mae = float(table.loc["timefuse_hard_top1", "MAE"])
    if not timefuse_hard_mae < visual_hard_mae:
        ok = False
        messages.append("pilot_test 未复现 TimeFuse hard MAE 优于 Visual hard MAE。")
    else:
        messages.append("pilot_test 复现 TimeFuse hard MAE 优于 Visual hard MAE。")

    visual_soft_mse = float(table.loc["visual_router_raw_soft_fusion", "MSE"])
    timefuse_soft_mse = float(table.loc["timefuse_raw_soft_fusion", "MSE"])
    if not visual_soft_mse < timefuse_soft_mse:
        ok = False
        messages.append("pilot_test 未复现 Visual raw-soft MSE 优于 TimeFuse raw-soft MSE。")
    else:
        messages.append("pilot_test 复现 Visual raw-soft MSE 优于 TimeFuse raw-soft MSE。")

    visual_acc = float(table.loc["visual_router_hard_top1", "oracle_label_accuracy"])
    timefuse_acc = float(table.loc["timefuse_hard_top1", "oracle_label_accuracy"])
    if not visual_acc < timefuse_acc:
        ok = False
        messages.append("pilot_test 未复现 Visual oracle-label accuracy 落后于 TimeFuse。")
    else:
        messages.append("pilot_test 复现 Visual oracle-label accuracy 落后于 TimeFuse。")

    messages.append(
        "full-scale reference: "
        + ", ".join(f"{key}={value:.6f}" for key, value in sorted(full_scale_reference.items()))
    )
    return ok, messages
