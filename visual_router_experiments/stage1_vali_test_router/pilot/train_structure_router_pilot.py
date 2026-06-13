#!/usr/bin/env python3
"""
文件功能：
    使用 TimeFuse-derived 单变量元特征训练 Stage 1 最小结构特征 router pilot。

Pilot 限制：
    - 只作为非视觉数值 baseline，避免把结构化特征工程做成主线；
    - 每个 config_name 独立训练一个 router，严格遵守 Stage 1 动作空间边界；
    - feature scaler 和分类器只在 vali split 上 fit，再用于 test split；
    - 标签使用 oracle labels 中指定 metric 的 oracle_model。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


WORKSPACE = Path("/home/shiyuhong/Time")
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from visual_router_experiments.stage1_vali_test_router.evaluate_router_baselines import MODEL_COLUMNS  # noqa: E402


DEFAULT_FEATURE_CACHE_PATH = (
    RUN_OUTPUT_ROOT
    / "2026-06-13_113713_308023_visual_router_stage1_structure_feature_pilot"
    / "feature_cache.csv"
)
DEFAULT_LABELS_PATH = (
    RUN_OUTPUT_ROOT
    / "2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot"
    / "window_oracle_labels_with_tsf_cell.csv"
)

METADATA_COLUMNS = {
    "feature_version",
    "sample_key",
    "config_name",
    "split",
    "dataset_name",
    "item_id",
    "channel_id",
    "window_index",
    "history_length",
    "feature_type",
    "feature_dim",
}


def display_time() -> str:
    """函数功能：生成写入 metadata/summary 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析结构特征 router pilot 参数。"""
    parser = argparse.ArgumentParser(description="Train a minimal structure-feature router pilot for Stage 1.")
    parser.add_argument(
        "--feature-cache-path",
        type=Path,
        default=DEFAULT_FEATURE_CACHE_PATH,
        help="build_structure_feature_cache_pilot.py 生成的 feature_cache.csv。",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="带 TSF cell 元信息的 window_oracle_labels CSV。",
    )
    parser.add_argument(
        "--metric",
        default="mae",
        choices=["mae", "mse"],
        help="训练 router 使用的 oracle label 指标口径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认写入 feature cache 所在目录。",
    )
    return parser.parse_args()


def load_feature_cache(feature_cache_path: Path) -> tuple[pd.DataFrame, List[str]]:
    """函数功能：读取 feature cache，并识别数值特征列。"""
    if not feature_cache_path.exists():
        raise FileNotFoundError(f"找不到 feature cache：{feature_cache_path}")
    feature_df = pd.read_csv(feature_cache_path)
    missing_metadata = sorted(METADATA_COLUMNS.difference(feature_df.columns))
    if missing_metadata:
        raise ValueError(f"feature cache 缺少元信息字段：{missing_metadata}")
    feature_cols = [
        col
        for col in feature_df.columns
        if col not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(feature_df[col])
    ]
    if not feature_cols:
        raise ValueError("feature cache 中没有可用数值特征列")
    if feature_df["sample_key"].duplicated().any():
        dup_keys = feature_df.loc[feature_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"feature cache 中 sample_key 重复，示例：{dup_keys}")
    return feature_df, feature_cols


def load_labels(labels_path: Path, metric: str) -> pd.DataFrame:
    """函数功能：读取 oracle labels，并筛选指定 metric。"""
    if not labels_path.exists():
        raise FileNotFoundError(f"找不到 labels 文件：{labels_path}")
    labels_df = pd.read_csv(labels_path)
    required_cols = {
        "sample_key",
        "config_name",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "oracle_model",
        "oracle_value",
        "metric",
        *MODEL_COLUMNS,
    }
    missing_cols = sorted(required_cols.difference(labels_df.columns))
    if missing_cols:
        raise ValueError(f"labels 文件缺少字段：{missing_cols}")
    labels_df = labels_df[labels_df["metric"] == metric].copy()
    if labels_df.empty:
        raise ValueError(f"labels 文件中没有 metric={metric} 的记录")
    return labels_df


def join_feature_and_labels(feature_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    函数功能：
        用 sample_key 严格 join feature cache 与 oracle labels。
    """
    merged = feature_df.merge(
        labels_df,
        on=["sample_key", "config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"],
        how="inner",
        suffixes=("", "_label"),
    )
    if len(merged) != len(feature_df) or len(merged) != len(labels_df):
        missing_feature = sorted(set(labels_df["sample_key"]) - set(feature_df["sample_key"]))
        missing_label = sorted(set(feature_df["sample_key"]) - set(labels_df["sample_key"]))
        raise ValueError(
            f"feature/label join 不完整：missing_feature={missing_feature[:10]} missing_label={missing_label[:10]}"
        )
    return merged


def make_router() -> Pipeline:
    """
    函数功能：
        创建最小结构特征 router。

    说明：
        使用 StandardScaler + LogisticRegression 是为了建立轻量可解释的 baseline；
        不在这里做复杂模型搜索，避免偏离视觉路由主线。
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=16,
                ),
            ),
        ]
    )


def train_and_predict_config(config_df: pd.DataFrame, feature_cols: Sequence[str]) -> List[Dict[str, object]]:
    """函数功能：对单个 config_name 训练 vali->test 结构特征 router 并输出逐样本预测。"""
    config_name = str(config_df["config_name"].iloc[0])
    vali_df = config_df[config_df["split"] == "vali"].copy()
    test_df = config_df[config_df["split"] == "test"].copy()
    if vali_df.empty or test_df.empty:
        raise ValueError(f"config_name={config_name} 需要同时包含 vali/test 样本")

    labels_seen = sorted(vali_df["oracle_model"].unique().tolist())
    if len(labels_seen) < 2:
        raise ValueError(f"config_name={config_name} 的 vali oracle label 少于 2 类，无法训练分类 router")

    router = make_router()
    router.fit(vali_df[list(feature_cols)], vali_df["oracle_model"])
    selected_models = router.predict(test_df[list(feature_cols)])

    rows: List[Dict[str, object]] = []
    for selected_model, (_, row) in zip(selected_models, test_df.iterrows()):
        selected_model = str(selected_model)
        rows.append(
            {
                "router_name": "timefuse_single_variable_logistic_regression",
                "config_name": row["config_name"],
                "sample_key": row["sample_key"],
                "split": row["split"],
                "dataset_name": row["dataset_name"],
                "item_id": int(row["item_id"]),
                "channel_id": int(row["channel_id"]),
                "window_index": int(row["window_index"]),
                "selected_model": selected_model,
                "selected_value": float(row[selected_model]),
                "oracle_model": row["oracle_model"],
                "oracle_value": float(row["oracle_value"]),
                "regret_to_oracle": float(row[selected_model] - row["oracle_value"]),
                "oracle_label_correct": bool(selected_model == row["oracle_model"]),
                "vali_sample_count": int(len(vali_df)),
                "test_sample_count": int(len(test_df)),
                "vali_label_classes": ",".join(labels_seen),
            }
        )
    return rows


def summarize_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：汇总 router test MAE、oracle regret 和 label accuracy。"""
    rows: List[Dict[str, object]] = []
    for keys, group in pred_df.groupby(["router_name", "config_name"], sort=True):
        router_name, config_name = keys
        rows.append(
            {
                "router_name": router_name,
                "config_name": config_name,
                "sample_count": int(len(group)),
                "selected_value": float(group["selected_value"].mean()),
                "oracle_value": float(group["oracle_value"].mean()),
                "regret_to_oracle": float(group["regret_to_oracle"].mean()),
                "oracle_label_accuracy": float(group["oracle_label_correct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_summary(output_dir: Path, summary_df: pd.DataFrame, counts_df: pd.DataFrame, feature_cols: Sequence[str]) -> None:
    """函数功能：写出中文 Markdown 摘要。"""
    summary_lines = [
        "# Stage 1 TimeFuse-derived 结构特征 Router Pilot",
        "",
        f"生成时间：{display_time()}",
        "",
        "## Router",
        "",
        "- `StandardScaler`：只在 vali features 上 fit。",
        "- `LogisticRegression(class_weight='balanced')`：只在 vali oracle labels 上训练。",
        "- test 上根据预测专家名读取对应专家 MAE，计算 router test MAE。",
        "",
        "## Summary",
        "",
        "| router_name | config_name | sample_count | selected_value | oracle_value | regret_to_oracle | oracle_label_accuracy |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_df.itertuples(index=False):
        summary_lines.append(
            "| "
            f"{row.router_name} | {row.config_name} | {row.sample_count} | "
            f"{row.selected_value:.6f} | {row.oracle_value:.6f} | "
            f"{row.regret_to_oracle:.6f} | {row.oracle_label_accuracy:.6f} |"
        )

    summary_lines.extend(
        [
            "",
            "## Selected Expert Counts",
            "",
            "| router_name | config_name | selected_model | rows |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in counts_df.itertuples(index=False):
        summary_lines.append(f"| {row.router_name} | {row.config_name} | {row.selected_model} | {row.rows} |")

    summary_lines.extend(
        [
            "",
            "## Feature Columns",
            "",
            "```text",
            "\n".join(feature_cols),
            "```",
            "",
        ]
    )
    (output_dir / "structure_router_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行 TimeFuse-derived 结构特征 router pilot。"""
    args = parse_args()
    output_dir = args.output_dir or args.feature_cache_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_df, feature_cols = load_feature_cache(args.feature_cache_path)
    labels_df = load_labels(args.labels_path, args.metric)
    merged_df = join_feature_and_labels(feature_df, labels_df)

    rows: List[Dict[str, object]] = []
    for _, config_df in merged_df.groupby("config_name", sort=True):
        rows.extend(train_and_predict_config(config_df, feature_cols))

    pred_df = pd.DataFrame(rows)
    summary_df = summarize_predictions(pred_df)
    counts_df = (
        pred_df.groupby(["router_name", "config_name", "selected_model"], sort=True)
        .size()
        .reset_index(name="rows")
    )

    pred_df.to_csv(output_dir / "structure_router_predictions.csv", index=False)
    summary_df.to_csv(output_dir / "structure_router_summary.csv", index=False)
    counts_df.to_csv(output_dir / "structure_router_selected_model_counts.csv", index=False)

    metadata: Dict[str, object] = {
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "feature_cache_path": str(args.feature_cache_path),
        "labels_path": str(args.labels_path),
        "metric": args.metric,
        "feature_columns": list(feature_cols),
        "feature_dim": len(feature_cols),
        "router_name": "timefuse_single_variable_logistic_regression",
        "training_split": "vali",
        "evaluation_split": "test",
        "config_names": sorted(pred_df["config_name"].unique().tolist()),
    }
    (output_dir / "structure_router_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary(output_dir, summary_df, counts_df, feature_cols)

    print(f"wrote structure router pilot to {output_dir}")
    print(summary_df.to_string(index=False))
    print(counts_df.to_string(index=False))


if __name__ == "__main__":
    main()
