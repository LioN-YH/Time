#!/usr/bin/env python3
"""
文件功能：
    基于五模型三配置 per-item 汇总结果，审计 Visual Router Phase 1 的专家互补性上限。

输入：
    - experiment_logs/run_outputs/2026-06-11_230450_825063_five_model_three_config_summary/per_item_metrics.csv

输出：
    - config_oracle_summary.csv：每个配置、每个指标的 best single expert 与 per-item oracle top-1 对比。
    - tsf_cell_oracle_summary.csv：每个配置、TSF cell、指标的 cell 内 best single 与 oracle top-1 对比。
    - tsf_cell_win_rates.csv：每个配置、TSF cell、指标、专家的 oracle 胜率和平均 regret。
    - per_item_oracle_choices.csv：每个配置、item、指标的 oracle 选择明细。
    - summary.md：中文审计摘要和 Phase 1 可行性判断。
    - metadata.json：输入输出路径、生成时间和校验信息。

关键约束：
    当前输入只有 per-item 聚合误差，没有窗口级预测缓存，因此只能审计 oracle top-1
    的理论选择上限，不能计算窗口级 top-k 或 soft fusion 上限。
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


WORKSPACE = Path("/home/shiyuhong/Time")
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"
DEFAULT_INPUT_SUMMARY_DIR = (
    RUN_OUTPUT_ROOT / "2026-06-11_230450_825063_five_model_three_config_summary"
)

CONFIG_ORDER = ["96_48_S", "576_288_S", "1024_512_S"]
MODEL_ORDER = ["DLinear", "PatchTST", "CrossFormer", "ES", "SNaive"]
METRIC_ORDER = ["MAE", "MSE"]
INDEX_COLS = ["config_name", "item_id", "cluster", "group_name"]


def now_token() -> str:
    """函数功能：生成 run 目录中的本地时间戳，精确到微秒避免重名。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 Markdown/JSON 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Audit per-item oracle gap for visual-router Phase 1."
    )
    parser.add_argument(
        "--input-summary-dir",
        type=Path,
        default=DEFAULT_INPUT_SUMMARY_DIR,
        help="五模型三配置汇总目录，需包含 per_item_metrics.csv。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="审计输出目录；默认写入 experiment_logs/run_outputs 下的时间戳目录。",
    )
    return parser.parse_args()


def load_per_item(input_summary_dir: Path) -> pd.DataFrame:
    """函数功能：读取 per-item 指标并执行字段、重复键和缺失值校验。"""
    input_path = input_summary_dir / "per_item_metrics.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    df = pd.read_csv(input_path)
    required_cols = set(INDEX_COLS + ["model", *METRIC_ORDER])
    missing_cols = sorted(required_cols.difference(df.columns))
    if missing_cols:
        raise ValueError(f"per_item_metrics.csv 缺少必要字段：{missing_cols}")

    duplicate_count = int(df.duplicated(["config_name", "item_id", "model"]).sum())
    if duplicate_count:
        raise ValueError(f"存在 {duplicate_count} 行重复的 config/item/model 记录。")

    missing_metric_count = int(df[METRIC_ORDER].isna().sum().sum())
    if missing_metric_count:
        raise ValueError(f"MAE/MSE 中存在 {missing_metric_count} 个缺失值。")

    observed_models = set(df["model"].unique())
    missing_models = [model for model in MODEL_ORDER if model not in observed_models]
    if missing_models:
        raise ValueError(f"输入缺少以下专家模型：{missing_models}")

    # 只保留当前五专家和三配置，避免后续目录纳入新模型后无意改变本次 Phase 1 审计口径。
    filtered = df[df["model"].isin(MODEL_ORDER) & df["config_name"].isin(CONFIG_ORDER)].copy()
    expected_rows = len(CONFIG_ORDER) * len(MODEL_ORDER) * filtered["item_id"].nunique()
    if len(filtered) != expected_rows:
        raise ValueError(
            f"过滤后行数异常：期望 {expected_rows}，实际 {len(filtered)}。"
        )
    return filtered


def metric_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """函数功能：把长表转换为 item × expert 的误差矩阵。"""
    matrix = (
        df.pivot_table(
            index=INDEX_COLS,
            columns="model",
            values=metric,
            aggfunc="first",
        )
        .reindex(columns=MODEL_ORDER)
        .reset_index()
    )
    if matrix[MODEL_ORDER].isna().any().any():
        raise ValueError(f"{metric} 矩阵存在缺失专家结果，不能计算 oracle。")
    return matrix


def ordered_winner(row: pd.Series, models: Sequence[str]) -> str:
    """函数功能：按固定模型顺序处理极少数并列最优，保证输出可复现。"""
    values = row[list(models)]
    return str(values.idxmin())


def normalized_entropy(counts: Iterable[int]) -> float:
    """函数功能：计算专家胜率分布的归一化熵，用于衡量 winner 是否集中。"""
    counts = [count for count in counts if count > 0]
    if len(counts) <= 1:
        return 0.0
    total = float(sum(counts))
    entropy = -sum((count / total) * math.log(count / total) for count in counts)
    return float(entropy / math.log(len(MODEL_ORDER)))


def build_oracle_choices(df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：生成每个配置、item、指标的 oracle 选择明细。"""
    rows: List[pd.DataFrame] = []
    for metric in METRIC_ORDER:
        matrix = metric_matrix(df, metric)
        matrix["metric"] = metric
        matrix["oracle_model"] = matrix.apply(ordered_winner, axis=1, models=MODEL_ORDER)
        matrix["oracle_metric"] = matrix[MODEL_ORDER].min(axis=1)
        # per-item regret 是后续判断 visual router 犯错成本的基础量。
        for model in MODEL_ORDER:
            matrix[f"{model}_regret"] = matrix[model] - matrix["oracle_metric"]
        rows.append(matrix)

    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["metric", "config_name", "cluster", "item_id"]).reset_index(drop=True)


def summarize_config(choices: pd.DataFrame) -> pd.DataFrame:
    """函数功能：按配置汇总 best single expert 与 oracle top-1 gap。"""
    rows: List[Dict[str, object]] = []
    for (metric, config_name), group in choices.groupby(["metric", "config_name"], sort=False):
        mean_by_model = group[MODEL_ORDER].mean()
        best_model = str(mean_by_model.idxmin())
        best_metric = float(mean_by_model[best_model])
        oracle_metric = float(group["oracle_metric"].mean())
        winner_counts = group["oracle_model"].value_counts().reindex(MODEL_ORDER, fill_value=0)
        better_than_best_item_count = int((group["oracle_model"] != best_model).sum())
        row: Dict[str, object] = {
            "metric": metric,
            "config_name": config_name,
            "item_count": int(len(group)),
            "model_count": len(MODEL_ORDER),
            "best_single_model": best_model,
            "best_single_metric": best_metric,
            "oracle_top1_metric": oracle_metric,
            "oracle_gap_abs": best_metric - oracle_metric,
            "oracle_gap_pct": (best_metric - oracle_metric) / best_metric
            if best_metric
            else 0.0,
            "oracle_uses_non_best_item_count": better_than_best_item_count,
            "oracle_uses_non_best_item_rate": better_than_best_item_count / len(group),
            "winner_entropy_norm": normalized_entropy(winner_counts.tolist()),
        }
        for model in MODEL_ORDER:
            row[f"{model}_win_rate"] = float(winner_counts[model] / len(group))
        rows.append(row)
    return order_metric_config(pd.DataFrame(rows))


def summarize_tsf_cells(choices: pd.DataFrame) -> pd.DataFrame:
    """函数功能：按 TSF cell 汇总 cell 内 best single 与 oracle top-1 gap。"""
    rows: List[Dict[str, object]] = []
    group_cols = ["metric", "config_name", "cluster", "group_name"]
    for keys, group in choices.groupby(group_cols, sort=False):
        metric, config_name, cluster, group_name = keys
        mean_by_model = group[MODEL_ORDER].mean()
        best_model = str(mean_by_model.idxmin())
        best_metric = float(mean_by_model[best_model])
        oracle_metric = float(group["oracle_metric"].mean())
        winner_counts = group["oracle_model"].value_counts().reindex(MODEL_ORDER, fill_value=0)
        row: Dict[str, object] = {
            "metric": metric,
            "config_name": config_name,
            "cluster": int(cluster),
            "group_name": group_name,
            "item_count": int(len(group)),
            "cell_best_single_model": best_model,
            "cell_best_single_metric": best_metric,
            "oracle_top1_metric": oracle_metric,
            "oracle_gap_abs": best_metric - oracle_metric,
            "oracle_gap_pct": (best_metric - oracle_metric) / best_metric
            if best_metric
            else 0.0,
            "winner_entropy_norm": normalized_entropy(winner_counts.tolist()),
        }
        for model in MODEL_ORDER:
            row[f"{model}_win_rate"] = float(winner_counts[model] / len(group))
        rows.append(row)
    return order_metric_config(pd.DataFrame(rows)).sort_values(
        ["metric_order", "config_order", "cluster"]
    ).drop(columns=["metric_order", "config_order"])


def summarize_win_rates(choices: pd.DataFrame) -> pd.DataFrame:
    """函数功能：输出每个 TSF cell 内各专家的胜率和平均 regret。"""
    rows: List[Dict[str, object]] = []
    group_cols = ["metric", "config_name", "cluster", "group_name"]
    for keys, group in choices.groupby(group_cols, sort=False):
        metric, config_name, cluster, group_name = keys
        for model in MODEL_ORDER:
            win_count = int((group["oracle_model"] == model).sum())
            rows.append(
                {
                    "metric": metric,
                    "config_name": config_name,
                    "cluster": int(cluster),
                    "group_name": group_name,
                    "model": model,
                    "item_count": int(len(group)),
                    "win_count": win_count,
                    "win_rate": win_count / len(group),
                    "mean_regret_to_oracle": float(group[f"{model}_regret"].mean()),
                    "median_regret_to_oracle": float(group[f"{model}_regret"].median()),
                }
            )
    return order_metric_config(pd.DataFrame(rows)).sort_values(
        ["metric_order", "config_order", "cluster", "model_order"]
    ).drop(columns=["metric_order", "config_order", "model_order"])


def order_metric_config(df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：增加稳定排序键，使 CSV/Markdown 输出顺序一致。"""
    metric_rank = {metric: idx for idx, metric in enumerate(METRIC_ORDER)}
    config_rank = {config: idx for idx, config in enumerate(CONFIG_ORDER)}
    model_rank = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    result = df.copy()
    result["metric_order"] = result["metric"].map(metric_rank)
    result["config_order"] = result["config_name"].map(config_rank)
    if "model" in result.columns:
        result["model_order"] = result["model"].map(model_rank)
    return result.sort_values(
        [col for col in ["metric_order", "config_order", "model_order"] if col in result.columns]
    ).reset_index(drop=True)


def frame_to_markdown(df: pd.DataFrame, *, float_digits: int = 6) -> str:
    """
    函数功能：
        将 DataFrame 转为 GitHub Markdown 表格。

    设计说明：
        pandas.DataFrame.to_markdown 依赖可选包 tabulate；这里复用轻量格式化逻辑，
        避免为了审计报告额外改变实验环境。
    """
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda value: f"{value:.{float_digits}f}")
        else:
            display_df[col] = display_df[col].astype(str)

    headers = [str(col) for col in display_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display_df.values.tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown_summary(
    output_dir: Path,
    config_summary: pd.DataFrame,
    cell_summary: pd.DataFrame,
    win_rates: pd.DataFrame,
) -> None:
    """函数功能：写出中文审计摘要，直接给 Phase 1 是否继续提供判断依据。"""
    mae_config = config_summary[config_summary["metric"] == "MAE"].copy()
    mae_cells = cell_summary[cell_summary["metric"] == "MAE"].copy()
    mae_win_rates = win_rates[win_rates["metric"] == "MAE"].copy()

    best_gap = float(mae_config["oracle_gap_pct"].max())
    winner_models_by_cell = int(mae_cells["cell_best_single_model"].nunique())
    config_winner_models = int(mae_config["best_single_model"].nunique())
    max_cell_gap = float(mae_cells["oracle_gap_pct"].max())
    enough_complementarity = (
        best_gap >= 0.05 and winner_models_by_cell >= 3 and max_cell_gap >= 0.05
    )

    compact_config = mae_config[
        [
            "config_name",
            "best_single_model",
            "best_single_metric",
            "oracle_top1_metric",
            "oracle_gap_abs",
            "oracle_gap_pct",
            "oracle_uses_non_best_item_rate",
            "winner_entropy_norm",
        ]
    ]

    compact_cells = mae_cells.sort_values("oracle_gap_pct", ascending=False)[
        [
            "config_name",
            "cluster",
            "group_name",
            "cell_best_single_model",
            "cell_best_single_metric",
            "oracle_top1_metric",
            "oracle_gap_pct",
            "winner_entropy_norm",
        ]
    ].head(12)

    compact_winners = (
        mae_win_rates.sort_values(["config_name", "cluster", "win_rate"], ascending=[True, True, False])
        .groupby(["config_name", "cluster", "group_name"], as_index=False)
        .head(2)[
            [
                "config_name",
                "cluster",
                "group_name",
                "model",
                "win_rate",
                "mean_regret_to_oracle",
            ]
        ]
    )

    conclusion = (
        "通过：当前五专家池存在足够 per-item 互补性，值得进入 Visual Router Phase 1 的视觉路由训练。"
        if enough_complementarity
        else "谨慎：当前五专家池的 oracle gap 或 cell 差异不足，训练 router 前应优先扩充专家池或重新检查专家口径。"
    )

    lines = [
        "# Visual Router Phase 1 专家互补性 Oracle 审计",
        "",
        f"生成时间：{display_time()}",
        "",
        "## 口径",
        "",
        "- 输入为五模型三配置汇总目录中的 `per_item_metrics.csv`。",
        "- 专家池固定为 DLinear、PatchTST、CrossFormer、ES、SNaive。",
        "- 当前只有 per-item 聚合 MAE/MSE，没有窗口级预测缓存；因此只计算 per-item oracle top-1，不计算 top-k 或 soft fusion。",
        "- 结论以 MAE 为主，MSE 结果保存在 CSV 中供复核。",
        "",
        "## 配置级 MAE Oracle Gap",
        "",
        frame_to_markdown(compact_config),
        "",
        "## Oracle Gap 最大的 TSF Cell",
        "",
        frame_to_markdown(compact_cells),
        "",
        "## 每个 TSF Cell 的 Top-2 Oracle 胜率专家",
        "",
        frame_to_markdown(compact_winners, float_digits=4),
        "",
        "## 判断",
        "",
        f"- 最大配置级 MAE oracle 相对收益：{best_gap:.2%}。",
        f"- cell 内 best single expert 覆盖 {winner_models_by_cell} 个不同模型；配置级 best single 覆盖 {config_winner_models} 个不同模型。",
        f"- 最大 cell 级 MAE oracle 相对收益：{max_cell_gap:.2%}。",
        f"- 结论：{conclusion}",
        "",
        "## 下一步",
        "",
        "1. 若继续 Phase 1，先构造视觉结构图像特征与每个 item 的 oracle label / regret label。",
        "2. 采用 7-cell -> held-out-cell 协议评估 visual router zero-shot generalization。",
        "3. 若 router 只能学习 config-level 或 cell-level 常数策略，应补充 TSMixer/iTransformer 等专家或引入窗口级 prediction cache。",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行完整 oracle 审计流程并写出所有结果文件。"""
    args = parse_args()
    input_summary_dir = args.input_summary_dir
    output_dir = args.output_dir or RUN_OUTPUT_ROOT / f"{now_token()}_visual_router_phase1_oracle_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_item_df = load_per_item(input_summary_dir)
    choices = build_oracle_choices(per_item_df)
    config_summary = summarize_config(choices).drop(columns=["metric_order", "config_order"])
    cell_summary = summarize_tsf_cells(choices)
    win_rates = summarize_win_rates(choices)

    choices.to_csv(output_dir / "per_item_oracle_choices.csv", index=False)
    config_summary.to_csv(output_dir / "config_oracle_summary.csv", index=False)
    cell_summary.to_csv(output_dir / "tsf_cell_oracle_summary.csv", index=False)
    win_rates.to_csv(output_dir / "tsf_cell_win_rates.csv", index=False)

    metadata = {
        "generated_at": display_time(),
        "input_summary_dir": str(input_summary_dir),
        "input_per_item_rows": int(len(per_item_df)),
        "output_dir": str(output_dir),
        "configs": CONFIG_ORDER,
        "models": MODEL_ORDER,
        "metrics": METRIC_ORDER,
        "oracle_scope": "per_item_top1_only_no_window_prediction_cache",
        "tsf_cell_count": int(per_item_df["group_name"].nunique()),
        "item_count": int(per_item_df["item_id"].nunique()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown_summary(output_dir, config_summary, cell_summary, win_rates)

    print(f"wrote oracle audit outputs to {output_dir}")
    print(
        config_summary[config_summary["metric"] == "MAE"][
            [
                "config_name",
                "best_single_model",
                "best_single_metric",
                "oracle_top1_metric",
                "oracle_gap_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
