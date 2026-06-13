#!/usr/bin/env python3
"""
文件功能：
    基于 Stage 1 prediction cache manifest 计算 window-level oracle label 和 regret。

输入：
    - pilot/build_prediction_cache_pilot.py 生成的 manifest.csv。

输出：
    - window_oracle_labels.csv：每个 sample_key 的 MAE/MSE oracle 专家和各专家 regret。
    - window_oracle_summary.csv：按 split/dataset/metric 汇总 best single 与 oracle gap。

设计说明：
    该脚本只依赖 manifest 中的窗口级 MAE/MSE，不重新读取 y_true/y_pred 数组。
    它用于验证五专家 cache 是否已经足够支持 hard routing label 训练。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


METRICS = ["mae", "mse"]
MODEL_DISPLAY_ORDER = ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]


def parse_args() -> argparse.Namespace:
    """函数功能：解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Compute window-level oracle labels from prediction cache.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="prediction cache 目录，需包含 manifest.csv。",
    )
    return parser.parse_args()


def ordered_winner(row: pd.Series, model_columns: List[str]) -> str:
    """函数功能：按固定专家顺序处理并列最优，保证 oracle label 可复现。"""
    return str(row[model_columns].idxmin())


def load_manifest(cache_dir: Path) -> pd.DataFrame:
    """函数功能：读取并校验 manifest。"""
    manifest_path = cache_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 manifest：{manifest_path}")
    df = pd.read_csv(manifest_path)
    required = {
        "sample_key",
        "split",
        "dataset_name",
        "config_name",
        "item_id",
        "channel_id",
        "window_index",
        "model_name",
        "mae",
        "mse",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"manifest 缺少必要字段：{missing}")
    duplicate_count = int(df.duplicated(["sample_key", "model_name"]).sum())
    if duplicate_count:
        raise ValueError(f"manifest 存在 {duplicate_count} 条 sample_key/model_name 重复记录。")
    return df


def build_oracle_labels(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：计算每个 sample_key 的 oracle label 和专家 regret。"""
    index_cols = ["sample_key", "config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]
    model_order = [model for model in MODEL_DISPLAY_ORDER if model in set(manifest_df["model_name"])]
    rows = []
    for metric in METRICS:
        pivot = (
            manifest_df.pivot_table(
                index=index_cols,
                columns="model_name",
                values=metric,
                aggfunc="first",
            )
            .reindex(columns=model_order)
            .reset_index()
        )
        if pivot[model_order].isna().any().any():
            raise ValueError(f"{metric} pivot 存在缺失专家结果，不能计算 oracle。")
        pivot["metric"] = metric
        pivot["oracle_model"] = pivot.apply(ordered_winner, axis=1, model_columns=model_order)
        pivot["oracle_value"] = pivot[model_order].min(axis=1)
        for model in model_order:
            pivot[f"{model}_regret"] = pivot[model] - pivot["oracle_value"]
        rows.append(pivot)
    return pd.concat(rows, ignore_index=True).sort_values(["metric", "split", "dataset_name", "sample_key"])


def build_oracle_summary(labels_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：按 split/dataset/metric 汇总 best single 与 oracle gap。"""
    model_cols = [model for model in MODEL_DISPLAY_ORDER if model in labels_df.columns]
    rows: List[Dict[str, object]] = []
    group_cols = ["metric", "split", "dataset_name"]
    for keys, group in labels_df.groupby(group_cols, sort=False):
        metric, split, dataset_name = keys
        mean_by_model = group[model_cols].mean()
        best_model = str(mean_by_model.idxmin())
        best_value = float(mean_by_model[best_model])
        oracle_value = float(group["oracle_value"].mean())
        row: Dict[str, object] = {
            "metric": metric,
            "split": split,
            "dataset_name": dataset_name,
            "sample_count": int(len(group)),
            "best_single_model": best_model,
            "best_single_value": best_value,
            "oracle_value": oracle_value,
            "oracle_gap_abs": best_value - oracle_value,
            "oracle_gap_pct": (best_value - oracle_value) / best_value if best_value else 0.0,
        }
        for model in model_cols:
            row[f"{model}_win_rate"] = float((group["oracle_model"] == model).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["metric", "split", "dataset_name"]).reset_index(drop=True)


def main() -> None:
    """函数功能：执行 window-level oracle label 和 summary 生成。"""
    args = parse_args()
    manifest_df = load_manifest(args.cache_dir)
    labels_df = build_oracle_labels(manifest_df)
    summary_df = build_oracle_summary(labels_df)

    labels_df.to_csv(args.cache_dir / "window_oracle_labels.csv", index=False)
    summary_df.to_csv(args.cache_dir / "window_oracle_summary.csv", index=False)

    print(f"wrote oracle labels to {args.cache_dir / 'window_oracle_labels.csv'}")
    print(f"wrote oracle summary to {args.cache_dir / 'window_oracle_summary.csv'}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
