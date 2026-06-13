#!/usr/bin/env python3
"""
文件功能：
    基于 window-level oracle labels 评估 Stage 1 非视觉 router baseline。

输入：
    - window_oracle_labels_with_tsf_cell.csv

输出：
    - baseline_summary.csv：按 config_name 分开的 test 主指标；
    - baseline_predictions.csv：每个 test window 的 baseline 选择和误差；
    - baseline_summary_by_config.csv：按 config_name 分层指标，与 baseline_summary.csv 同口径；
    - baseline_summary_macro.csv：跨 config 的 macro average，仅用于总览，不作为可部署动作空间；
    - baseline_summary_by_dataset.csv：按 config_name + dataset_name 分层指标；
    - baseline_summary_by_tsf_cell.csv：按 config_name + TSF cell 分层指标；
    - baseline_summary_by_dataset_tsf_cell.csv：按 config_name + dataset_name + TSF cell 分层指标；
    - summary.md：中文摘要。

设计说明：
    该脚本不训练视觉模型，只用 vali split 学简单规则，然后在 test split 上评估。
    目的是建立后续 visual router 必须超过的 metadata/statistics baseline。
    正式 Stage 1 采用 per-config router，因此 baseline 训练也必须在每个 config_name
    内独立完成，避免把不同输入/输出长度的专家误合并为一个动作空间。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


MODEL_COLUMNS = ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]
DEFAULT_METRIC = "mae"


@dataclass(frozen=True)
class BaselineRule:
    """类功能：记录一个 baseline 的名称和分组字段。"""

    name: str
    group_cols: Tuple[str, ...]
    kind: str = "mean_metric_best"


BASELINE_RULES = [
    BaselineRule(name="global_best_single", group_cols=()),
    BaselineRule(name="dataset_only", group_cols=("dataset_name",)),
    BaselineRule(name="tsf_cell_only", group_cols=("group_name",)),
    BaselineRule(name="dataset_tsf_cell", group_cols=("dataset_name", "group_name")),
    BaselineRule(name="global_majority_label", group_cols=(), kind="majority_label"),
    BaselineRule(name="dataset_majority_label", group_cols=("dataset_name",), kind="majority_label"),
    BaselineRule(name="tsf_cell_majority_label", group_cols=("group_name",), kind="majority_label"),
    BaselineRule(name="dataset_tsf_cell_majority_label", group_cols=("dataset_name", "group_name"), kind="majority_label"),
]


def parse_args() -> argparse.Namespace:
    """函数功能：解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Evaluate non-visual router baselines.")
    parser.add_argument(
        "--labels-path",
        type=Path,
        required=True,
        help="带 TSF cell 元信息的 window_oracle_labels CSV。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="baseline 输出目录；默认写入 labels 文件所在目录。",
    )
    parser.add_argument(
        "--metric",
        choices=["mae", "mse"],
        default=DEFAULT_METRIC,
        help="用于学习和评估 baseline 的指标；第一版默认 MAE。",
    )
    return parser.parse_args()


def load_labels(labels_path: Path, metric: str) -> pd.DataFrame:
    """函数功能：读取 oracle labels，并保留指定指标的窗口记录。"""
    if not labels_path.exists():
        raise FileNotFoundError(f"找不到 labels 文件：{labels_path}")
    df = pd.read_csv(labels_path)
    required_cols = {
        "sample_key",
        "config_name",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "cluster",
        "group_name",
        "oracle_model",
        "oracle_value",
        "metric",
        *MODEL_COLUMNS,
    }
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(f"labels 文件缺少字段：{missing}")
    metric_df = df[df["metric"] == metric].copy()
    if metric_df.empty:
        raise ValueError(f"labels 文件中没有 metric={metric} 的记录。")
    split_set = set(metric_df["split"])
    if not {"vali", "test"}.issubset(split_set):
        raise ValueError(f"需要同时包含 vali/test split，实际 split={sorted(split_set)}")
    return metric_df


def choose_best_expert(group: pd.DataFrame, kind: str) -> str:
    """
    函数功能：
        从 vali 分组中选择一个专家。

    kind:
        - mean_metric_best：平均误差最低；
        - majority_label：oracle label 胜出次数最多，平手时按 MODEL_COLUMNS 顺序。
    """
    if kind == "mean_metric_best":
        means = group[MODEL_COLUMNS].mean()
        return str(means.idxmin())
    if kind == "majority_label":
        counts = group["oracle_model"].value_counts().reindex(MODEL_COLUMNS, fill_value=0)
        return str(counts.idxmax())
    raise ValueError(f"未知 baseline kind：{kind}")


def train_rule_map(vali_df: pd.DataFrame, rule: BaselineRule) -> Dict[Tuple[object, ...], str]:
    """函数功能：在 vali split 上为指定 baseline 学习 group -> expert 映射。"""
    if not rule.group_cols:
        return {(): choose_best_expert(vali_df, rule.kind)}
    mapping: Dict[Tuple[object, ...], str] = {}
    for keys, group in vali_df.groupby(list(rule.group_cols), dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        mapping[keys] = choose_best_expert(group, rule.kind)
    return mapping


def lookup_expert(row: pd.Series, rule: BaselineRule, mapping: Dict[Tuple[object, ...], str], fallback: str) -> str:
    """函数功能：在 test 样本上根据 group key 查找专家；未见组合退回 global best。"""
    if not rule.group_cols:
        return mapping[()]
    key = tuple(row[col] for col in rule.group_cols)
    return mapping.get(key, fallback)


def make_config_baseline_predictions(config_df: pd.DataFrame, config_name: str) -> List[Dict[str, object]]:
    """
    函数功能：
        对单个 config_name 的样本，用 vali 学规则并在 test 上生成逐窗口 baseline 记录。

    关键约束：
        Stage 1 的 router 动作空间只在同一个 config_name 内成立，因此这里不允许
        跨 config 训练 global/dataset/TSF-cell 规则。
    """
    vali_df = config_df[config_df["split"] == "vali"].copy()
    test_df = config_df[config_df["split"] == "test"].copy()
    if vali_df.empty or test_df.empty:
        raise ValueError(f"config_name={config_name} 需要同时包含 vali/test 样本。")
    global_fallback = choose_best_expert(vali_df, "mean_metric_best")

    rows: List[Dict[str, object]] = []
    for rule in BASELINE_RULES:
        mapping = train_rule_map(vali_df, rule)
        for _, row in test_df.iterrows():
            selected_model = lookup_expert(row, rule, mapping, global_fallback)
            rows.append(
                {
                    "baseline": rule.name,
                    "rule_kind": rule.kind,
                    "sample_key": row["sample_key"],
                    "config_name": row["config_name"],
                    "split": row["split"],
                    "dataset_name": row["dataset_name"],
                    "item_id": int(row["item_id"]),
                    "channel_id": int(row["channel_id"]),
                    "window_index": int(row["window_index"]),
                    "cluster": int(row["cluster"]),
                    "group_name": row["group_name"],
                    "selected_model": selected_model,
                    "selected_value": float(row[selected_model]),
                    "oracle_model": row["oracle_model"],
                    "oracle_value": float(row["oracle_value"]),
                    "regret_to_oracle": float(row[selected_model] - row["oracle_value"]),
                    "oracle_label_correct": bool(selected_model == row["oracle_model"]),
                }
            )

    # oracle top-1 是上限，不是可部署规则，但放入同一 predictions 表便于统一汇总。
    for _, row in test_df.iterrows():
        rows.append(
            {
                "baseline": "oracle_top1",
                "rule_kind": "upper_bound",
                "sample_key": row["sample_key"],
                "config_name": row["config_name"],
                "split": row["split"],
                "dataset_name": row["dataset_name"],
                "item_id": int(row["item_id"]),
                "channel_id": int(row["channel_id"]),
                "window_index": int(row["window_index"]),
                "cluster": int(row["cluster"]),
                "group_name": row["group_name"],
                "selected_model": row["oracle_model"],
                "selected_value": float(row["oracle_value"]),
                "oracle_model": row["oracle_model"],
                "oracle_value": float(row["oracle_value"]),
                "regret_to_oracle": 0.0,
                "oracle_label_correct": True,
            }
        )
    return rows


def make_baseline_predictions(labels_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：按 config_name 独立学习规则，并在对应 config 的 test split 上评估。"""
    rows: List[Dict[str, object]] = []
    for config_name, config_df in labels_df.groupby("config_name", dropna=False, sort=True):
        rows.extend(make_config_baseline_predictions(config_df, str(config_name)))
    return pd.DataFrame(rows)


def summarize_predictions(pred_df: pd.DataFrame, group_cols: Sequence[str] = ()) -> pd.DataFrame:
    """函数功能：汇总 baseline 的误差、oracle regret 和 label accuracy。"""
    rows: List[Dict[str, object]] = []
    base_group_cols = ["baseline", "rule_kind", *group_cols]
    for keys, group in pred_df.groupby(base_group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(base_group_cols, keys)}
        row.update(
            {
                "sample_count": int(len(group)),
                "selected_value": float(group["selected_value"].mean()),
                "oracle_value": float(group["oracle_value"].mean()),
                "regret_to_oracle": float(group["regret_to_oracle"].mean()),
                "oracle_label_accuracy": float(group["oracle_label_correct"].mean()),
            }
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    global_rows = summary[summary["baseline"] == "global_best_single"]
    if not global_rows.empty:
        if group_cols:
            # 每个分层组都应和同组内的 global_best_single 比较，避免跨 config 或跨 cell 比较。
            global_lookup = global_rows[[*group_cols, "selected_value"]].rename(columns={"selected_value": "global_selected_value"})
            summary = summary.merge(global_lookup, on=list(group_cols), how="left")
            summary["relative_improvement_vs_global"] = (
                summary["global_selected_value"] - summary["selected_value"]
            ) / summary["global_selected_value"]
            summary = summary.drop(columns=["global_selected_value"])
        else:
            global_value = float(global_rows.iloc[0]["selected_value"])
            summary["relative_improvement_vs_global"] = (global_value - summary["selected_value"]) / global_value
    else:
        summary["relative_improvement_vs_global"] = pd.NA
    return summary.sort_values(base_group_cols).reset_index(drop=True)


def summarize_macro_by_config(summary_by_config: pd.DataFrame) -> pd.DataFrame:
    """
    函数功能：
        基于 config-level summary 计算跨 config macro average。

    说明：
        macro average 只用于总览，不表示存在一个可以跨 config 自由选择专家的部署动作空间。
    """
    metric_cols = [
        "sample_count",
        "selected_value",
        "oracle_value",
        "regret_to_oracle",
        "oracle_label_accuracy",
        "relative_improvement_vs_global",
    ]
    rows: List[Dict[str, object]] = []
    for (baseline, rule_kind), group in summary_by_config.groupby(["baseline", "rule_kind"], sort=False):
        row: Dict[str, object] = {
            "baseline": baseline,
            "rule_kind": rule_kind,
            "config_count": int(group["config_name"].nunique()),
        }
        for col in metric_cols:
            row[col] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["baseline", "rule_kind"]).reset_index(drop=True)


def frame_to_markdown(df: pd.DataFrame, *, float_digits: int = 6) -> str:
    """函数功能：将 DataFrame 转为 Markdown 表格，避免依赖 tabulate。"""
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda value: f"{value:.{float_digits}f}")
        else:
            display_df[col] = display_df[col].astype(str)
    lines = [
        "| " + " | ".join(display_df.columns) + " |",
        "| " + " | ".join(["---"] * len(display_df.columns)) + " |",
    ]
    for row in display_df.values.tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary_md(output_dir: Path, summary_df: pd.DataFrame, macro_df: pd.DataFrame, metric: str) -> None:
    """函数功能：写出中文 baseline 摘要。"""
    compact = summary_df[
        [
            "config_name",
            "baseline",
            "sample_count",
            "selected_value",
            "oracle_value",
            "regret_to_oracle",
            "oracle_label_accuracy",
            "relative_improvement_vs_global",
        ]
    ].copy()
    lines = [
        "# Stage 1 非视觉 Router Baseline 汇总",
        "",
        f"指标口径：`{metric}`",
        "",
        "## 说明",
        "",
        "- 所有可部署 baseline 都只用 `vali` split 学规则，并在 `test` split 上评估。",
        "- baseline 训练和主汇总默认按 `config_name` 分层；不同历史-未来 config 不共享专家动作空间。",
        "- `oracle_top1` 使用 test 窗口事后最优专家，只作为上限，不是可部署方法。",
        "- macro average 仅用于跨 config 总览，不代表一个可跨 config 部署的 router。",
        "- 当前 baseline 用于判断后续 visual router 是否超过 dataset/TSF-cell shortcut。",
        "",
        "## Test 结果：按 Config",
        "",
        frame_to_markdown(compact),
        "",
        "## Test 结果：跨 Config Macro Average",
        "",
        frame_to_markdown(macro_df),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行 baseline 训练、test 评估和结果落盘。"""
    args = parse_args()
    output_dir = args.output_dir or args.labels_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_df = load_labels(args.labels_path, args.metric)
    pred_df = make_baseline_predictions(labels_df)
    by_config = summarize_predictions(pred_df, ["config_name"])
    macro_summary = summarize_macro_by_config(by_config)
    by_dataset = summarize_predictions(pred_df, ["config_name", "dataset_name"])
    by_tsf_cell = summarize_predictions(pred_df, ["config_name", "cluster", "group_name"])
    by_dataset_tsf_cell = summarize_predictions(pred_df, ["config_name", "dataset_name", "cluster", "group_name"])

    pred_df.to_csv(output_dir / "baseline_predictions.csv", index=False)
    by_config.to_csv(output_dir / "baseline_summary.csv", index=False)
    by_config.to_csv(output_dir / "baseline_summary_by_config.csv", index=False)
    macro_summary.to_csv(output_dir / "baseline_summary_macro.csv", index=False)
    by_dataset.to_csv(output_dir / "baseline_summary_by_dataset.csv", index=False)
    by_tsf_cell.to_csv(output_dir / "baseline_summary_by_tsf_cell.csv", index=False)
    by_dataset_tsf_cell.to_csv(output_dir / "baseline_summary_by_dataset_tsf_cell.csv", index=False)
    write_summary_md(output_dir, by_config, macro_summary, args.metric)

    print(f"wrote baseline outputs to {output_dir}")
    print(by_config.to_string(index=False))


if __name__ == "__main__":
    main()
