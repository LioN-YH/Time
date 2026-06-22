#!/usr/bin/env python3
"""
文件功能：
    汇总多个 PatchTST+Visual dual-branch fusion run 的 metrics.json。

输入：
    一个包含若干子目录的结果根目录，每个子目录内包含 `metrics.json`。

输出：
    fusion mode 级别的 mean/std 汇总 CSV、JSON 和 Markdown。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


METRIC_COLUMNS = [
    "patchtst_mae",
    "patchtst_mse",
    "dual_branch_mae",
    "dual_branch_mse",
    "delta_mae_vs_patchtst",
    "delta_mse_vs_patchtst",
]


def load_metric_rows(results_root: Path) -> pd.DataFrame:
    """函数功能：递归读取 metrics.json 并合并成 DataFrame。"""
    rows: List[Dict[str, object]] = []
    for metrics_path in sorted(results_root.rglob("metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["run_dir"] = str(metrics_path.parent)
        rows.append(payload)
    if not rows:
        raise ValueError(f"未找到 metrics.json：{results_root}")
    return pd.DataFrame(rows)


def frame_to_markdown(df: pd.DataFrame) -> str:
    """函数功能：避免依赖 tabulate 的轻量 Markdown 表格输出。"""
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{value:.8f}")
        else:
            display[col] = display[col].astype(str)
    lines = ["| " + " | ".join(display.columns) + " |", "| " + " | ".join(["---"] * len(display.columns)) + " |"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：按 fusion_mode 汇总多 seed mean/std 和 beats 比例。"""
    grouped_rows: List[Dict[str, object]] = []
    for fusion_mode, group in df.groupby("fusion_mode", sort=True):
        row: Dict[str, object] = {"fusion_mode": fusion_mode, "runs": int(len(group))}
        for col in METRIC_COLUMNS:
            row[f"mean_{col}"] = float(group[col].mean())
            row[f"std_{col}"] = float(group[col].std(ddof=0))
        row["beats_patchtst_mae_rate"] = float(group["beats_patchtst_mae"].astype(bool).mean())
        row["beats_patchtst_mse_rate"] = float(group["beats_patchtst_mse"].astype(bool).mean())
        grouped_rows.append(row)
    return pd.DataFrame(grouped_rows)


def build_markdown_summary(summary: pd.DataFrame) -> str:
    """
    函数功能：
        生成包含显式 beats 结论的 Markdown summary。

    关键约束：
        delta 定义保持为 PatchTST 指标减去 dual-branch 指标；正数表示双分支更好。
    """
    mae_beats = summary[summary["mean_delta_mae_vs_patchtst"] > 0.0].copy()
    mse_beats = summary[summary["mean_delta_mse_vs_patchtst"] > 0.0].copy()

    def _beat_lines(df: pd.DataFrame, metric: str) -> List[str]:
        delta_col = f"mean_delta_{metric}_vs_patchtst"
        if df.empty:
            return [f"- {metric.upper()}：没有双分支变体超过 PatchTST。"]
        lines = []
        for row in df.sort_values(delta_col, ascending=False).itertuples(index=False):
            lines.append(f"- {metric.upper()}：{row.fusion_mode} 超过 PatchTST，delta={getattr(row, delta_col):.8f}")
        return lines

    return "\n".join(
        [
            "# PatchTST + Visual Dual-Branch 汇总",
            "",
            "## 结论",
            "",
            *_beat_lines(mae_beats, "mae"),
            *_beat_lines(mse_beats, "mse"),
            "",
            "## 指标表",
            "",
            frame_to_markdown(summary),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    """函数功能：解析汇总脚本参数。"""
    parser = argparse.ArgumentParser(description="Summarize PatchTST+Visual dual-branch fusion metrics.")
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """函数功能：读取所有 run 指标并写出汇总产物。"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_metric_rows(args.results_root)
    summary = summarize(runs)
    runs.to_csv(args.output_dir / "dual_branch_run_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "dual_branch_summary.csv", index=False)
    (args.output_dir / "dual_branch_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "dual_branch_summary.md").write_text(build_markdown_summary(summary), encoding="utf-8")
    print(f"完成：汇总已写入 {args.output_dir}")


if __name__ == "__main__":
    main()
