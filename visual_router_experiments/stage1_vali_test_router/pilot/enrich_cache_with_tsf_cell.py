#!/usr/bin/env python3
"""
文件功能：
    为 Stage 1 prediction cache 和 window oracle 结果补充 TSF cell 元信息。

输入：
    - cache 目录中的 manifest.csv；
    - 可选的 window_oracle_labels.csv；
    - Quito 的 item_clusters.csv。

输出：
    - manifest_with_tsf_cell.csv；
    - window_oracle_labels_with_tsf_cell.csv；
    - window_oracle_summary_by_tsf_cell.csv；
    - window_oracle_summary_by_dataset_tsf_cell.csv。

设计说明：
    dataset_name 表示数据来源/频率层级，不等于 TSF cell。TSF cell 由 item_id 映射到
    cluster/group_name。后续 router baseline 和分析必须能区分 dataset shortcut 与
    TSF cell shortcut，因此这里显式补齐这些字段。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


WORKSPACE = Path("/home/shiyuhong/Time")
DEFAULT_CLUSTER_PATH = WORKSPACE / "quito" / "examples" / "datasets" / "cluster_data" / "item_clusters.csv"
MODEL_DISPLAY_ORDER = ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]


def parse_args() -> argparse.Namespace:
    """函数功能：解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Enrich Stage 1 cache files with TSF cell metadata.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="prediction cache 目录，需包含 manifest.csv，可选包含 window_oracle_labels.csv。",
    )
    parser.add_argument(
        "--cluster-path",
        type=Path,
        default=DEFAULT_CLUSTER_PATH,
        help="item_id 到 TSF cell 的映射 CSV。",
    )
    return parser.parse_args()


def load_cluster_mapping(cluster_path: Path) -> pd.DataFrame:
    """函数功能：读取 TSF cell 映射并保留后续分析常用字段。"""
    if not cluster_path.exists():
        raise FileNotFoundError(f"找不到 cluster 映射：{cluster_path}")
    cluster_df = pd.read_csv(cluster_path)
    keep_cols = [
        "item_id",
        "cluster",
        "group_name",
        "forecastability_cat",
        "season_strength_cat",
        "trend_strength_cat",
        "cv_cat",
        "missing_ratio_cat",
    ]
    missing = sorted(set(keep_cols).difference(cluster_df.columns))
    if missing:
        raise ValueError(f"cluster 映射缺少字段：{missing}")
    return cluster_df[keep_cols].drop_duplicates("item_id")


def enrich_frame(frame: pd.DataFrame, cluster_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：按 item_id 合并 TSF cell 信息，并检查是否有缺失。"""
    existing_tsf_cols = [
        "cluster",
        "group_name",
        "forecastability_cat",
        "season_strength_cat",
        "trend_strength_cat",
        "cv_cat",
        "missing_ratio_cat",
    ]
    frame = frame.drop(columns=[col for col in existing_tsf_cols if col in frame.columns])
    enriched = frame.merge(cluster_df, on="item_id", how="left")
    missing_count = int(enriched["group_name"].isna().sum())
    if missing_count:
        missing_items = sorted(enriched.loc[enriched["group_name"].isna(), "item_id"].unique())[:10]
        raise ValueError(f"有 {missing_count} 行缺少 TSF cell 映射，示例 item_id={missing_items}")
    return enriched


def summarize_oracle(labels_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """函数功能：按指定分组汇总 best single、oracle gap 和专家胜率。"""
    model_cols = [model for model in MODEL_DISPLAY_ORDER if model in labels_df.columns]
    rows: List[Dict[str, object]] = []
    for keys, group in labels_df.groupby(["metric", *group_cols], sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metric = keys[0]
        group_values = keys[1:]
        mean_by_model = group[model_cols].mean()
        best_model = str(mean_by_model.idxmin())
        best_value = float(mean_by_model[best_model])
        oracle_value = float(group["oracle_value"].mean())
        row: Dict[str, object] = {
            "metric": metric,
            "sample_count": int(len(group)),
            "best_single_model": best_model,
            "best_single_value": best_value,
            "oracle_value": oracle_value,
            "oracle_gap_abs": best_value - oracle_value,
            "oracle_gap_pct": (best_value - oracle_value) / best_value if best_value else 0.0,
        }
        for col, value in zip(group_cols, group_values):
            row[col] = value
        for model in model_cols:
            row[f"{model}_win_rate"] = float((group["oracle_model"] == model).mean())
        rows.append(row)

    ordered_cols = ["metric", *group_cols, "sample_count", "best_single_model", "best_single_value", "oracle_value", "oracle_gap_abs", "oracle_gap_pct"]
    win_cols = [col for col in rows[0].keys() if col.endswith("_win_rate")] if rows else []
    return pd.DataFrame(rows)[ordered_cols + win_cols].sort_values(["metric", *group_cols]).reset_index(drop=True)


def main() -> None:
    """函数功能：执行 manifest/oracle labels 的 TSF cell enrich 与分组汇总。"""
    args = parse_args()
    cluster_df = load_cluster_mapping(args.cluster_path)

    manifest_path = args.cache_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 manifest：{manifest_path}")
    manifest_df = pd.read_csv(manifest_path)
    manifest_enriched = enrich_frame(manifest_df, cluster_df)
    manifest_enriched.to_csv(args.cache_dir / "manifest_with_tsf_cell.csv", index=False)

    labels_path = args.cache_dir / "window_oracle_labels.csv"
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        labels_enriched = enrich_frame(labels_df, cluster_df)
        labels_enriched.to_csv(args.cache_dir / "window_oracle_labels_with_tsf_cell.csv", index=False)

        by_cell = summarize_oracle(labels_enriched, ["split", "cluster", "group_name"])
        by_dataset_cell = summarize_oracle(labels_enriched, ["split", "dataset_name", "cluster", "group_name"])
        by_cell.to_csv(args.cache_dir / "window_oracle_summary_by_tsf_cell.csv", index=False)
        by_dataset_cell.to_csv(args.cache_dir / "window_oracle_summary_by_dataset_tsf_cell.csv", index=False)

        print(f"wrote enriched oracle labels and TSF summaries to {args.cache_dir}")
        print(by_dataset_cell.to_string(index=False))
    else:
        print(f"wrote enriched manifest to {args.cache_dir / 'manifest_with_tsf_cell.csv'}")
        print("window_oracle_labels.csv 不存在，跳过 oracle TSF cell 汇总。")


if __name__ == "__main__":
    main()
