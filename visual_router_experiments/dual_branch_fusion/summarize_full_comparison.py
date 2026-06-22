#!/usr/bin/env python3
"""
文件功能：
    汇总本轮剩余 PatchTST+Visual dual-branch run，并合并 1b20e72 历史 robust multiseed
    结果，生成完整比较大表。

输入：
    - remaining_root：本轮新增 run 目录。
    - historical_root：1b20e72 已完成 5 个 mode 的 run 目录。

输出：
    remaining_run_metrics.csv、remaining_summary.csv、full_dual_branch_comparison.csv/json/md、
    full_dual_branch_ranking.md。

关键约束：
    residual_scale sweep 以独立 method 进入大表；默认 scale=0.1 的历史
    patchtst_residual_visual 保持原 method 名，避免和本轮 scale sweep 混淆。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


REQUIRED_OUTPUTS = ["config.json", "metrics.json", "predictions.npz", "training_log.txt", "summary.md"]


def read_json(path: Path) -> Dict[str, object]:
    """函数功能：读取 UTF-8 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def scale_tag(value: float) -> str:
    """函数功能：将 residual_scale 转为稳定 method 后缀。"""
    text = f"{float(value):g}"
    return text.replace(".", "p")


def method_from_payload(metrics: Dict[str, object], config: Dict[str, object], run_dir: Path) -> str:
    """
    函数功能：
        生成用于大表聚合的 method 名。

    关键约束：
        本轮 residual_scale sweep 的目录名显式包含 `scale`，即使 scale=0.1 也作为
        `patchtst_residual_visual_scale0p1` 单独汇总；历史默认 run 不改名。
    """
    mode = str(metrics.get("fusion_mode") or config.get("fusion_mode"))
    residual_scale = float(config.get("residual_scale", 0.1))
    if mode == "patchtst_residual_visual" and "scale" in run_dir.name:
        return f"patchtst_residual_visual_scale{scale_tag(residual_scale)}"
    return mode


def note_for_method(method: str) -> str:
    """函数功能：为 full comparison 表补充简短解释，尤其标出 residual-safe 类方法。"""
    if method.startswith("patchtst_residual_visual_scale"):
        return "residual-safe prediction residual scale sweep; zero-init delta; h_ts=flattened_y_patchtst fallback"
    if method == "patchtst_residual_visual":
        return "residual-safe prediction residual; zero-init delta; h_ts=flattened_y_patchtst fallback"
    if method == "visual_residual":
        return "prediction residual from visual embedding; h_ts=flattened_y_patchtst fallback"
    if method == "gated_residual_feature":
        return "feature-level gated visual residual; not initialized to PatchTST prediction"
    return "feature/prediction fusion with h_ts=flattened_y_patchtst fallback"


def load_rows(root: Path, *, source: str) -> pd.DataFrame:
    """函数功能：递归读取一个结果根目录中的所有单 run 指标和配置。"""
    rows: List[Dict[str, object]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        missing = [name for name in REQUIRED_OUTPUTS if not (run_dir / name).exists()]
        metrics = read_json(metrics_path)
        config_path = run_dir / "config.json"
        config = read_json(config_path) if config_path.exists() else {}
        method = method_from_payload(metrics, config, run_dir)
        row: Dict[str, object] = {
            **metrics,
            "method": method,
            "source": source,
            "run_dir": str(run_dir),
            "residual_scale": float(config.get("residual_scale", metrics.get("residual_scale", 0.1))),
            "feature_standardization_enabled": bool(
                (config.get("feature_standardization") or {}).get("enabled", False)
                if isinstance(config.get("feature_standardization"), dict)
                else False
            ),
            "config_test_checkpoint": str(config.get("test_checkpoint", metrics.get("test_checkpoint", ""))),
            "missing_outputs": ",".join(missing),
        }
        rows.append(row)
    if not rows:
        raise ValueError(f"未找到 metrics.json：{root}")
    return pd.DataFrame(rows)


def summarize_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：按 method 生成验收要求的 mean/std/rate 大表。"""
    rows: List[Dict[str, object]] = []
    for method, group in df.groupby("method", sort=False):
        seeds = sorted(int(seed) for seed in group["seed"].tolist())
        row: Dict[str, object] = {
            "method": method,
            "seeds": "/".join(str(seed) for seed in seeds),
            "patchtst_mae_mean": float(group["patchtst_mae"].mean()),
            "patchtst_mae_std": float(group["patchtst_mae"].std(ddof=0)),
            "patchtst_mse_mean": float(group["patchtst_mse"].mean()),
            "patchtst_mse_std": float(group["patchtst_mse"].std(ddof=0)),
            "dual_branch_mae_mean": float(group["dual_branch_mae"].mean()),
            "dual_branch_mae_std": float(group["dual_branch_mae"].std(ddof=0)),
            "dual_branch_mse_mean": float(group["dual_branch_mse"].mean()),
            "dual_branch_mse_std": float(group["dual_branch_mse"].std(ddof=0)),
            "delta_mae_vs_patchtst_mean": float(group["delta_mae_vs_patchtst"].mean()),
            "delta_mae_vs_patchtst_std": float(group["delta_mae_vs_patchtst"].std(ddof=0)),
            "delta_mse_vs_patchtst_mean": float(group["delta_mse_vs_patchtst"].mean()),
            "delta_mse_vs_patchtst_std": float(group["delta_mse_vs_patchtst"].std(ddof=0)),
            "beats_patchtst_mae_rate": float(group["beats_patchtst_mae"].astype(bool).mean()),
            "beats_patchtst_mse_rate": float(group["beats_patchtst_mse"].astype(bool).mean()),
            "best_val_epoch_mean": float(group["best_val_epoch"].mean()),
            "note": note_for_method(str(method)),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["delta_mae_vs_patchtst_mean", "delta_mse_vs_patchtst_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)


def frame_to_markdown(df: pd.DataFrame) -> str:
    """函数功能：不依赖 tabulate 输出 Markdown 表格。"""
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


def build_comparison_markdown(summary: pd.DataFrame) -> str:
    """函数功能：生成完整大表 Markdown，并明确解释边界。"""
    mae_best = summary.iloc[0]
    mse_best = summary.sort_values("delta_mse_vs_patchtst_mean", ascending=False).iloc[0]
    has_mae_gain = bool((summary["delta_mae_vs_patchtst_mean"] > 0.0).any())
    has_mse_gain = bool((summary["delta_mse_vs_patchtst_mean"] > 0.0).any())
    boundary_line = (
        "本轮 fallback-level 融合出现正增益，但仍不能替代真实 PatchTST encoder hidden + visual embedding 的融合验证。"
        if has_mae_gain or has_mse_gain
        else "该负/弱结果不能否定真实 PatchTST encoder hidden + visual embedding 的融合可能性。"
    )
    return "\n".join(
        [
            "# Full PatchTST + Visual Dual-Branch Comparison",
            "",
            "说明：delta 定义为 PatchTST 指标减去 dual-branch 指标，正数表示双分支更好。",
            "边界：当前 h_ts 仍为 flattened_y_patchtst fallback，不是真实 PatchTST encoder hidden；本表只说明 fixed visual embedding + PatchTST frozen prediction fallback 的轻量融合效果。",
            "",
            "## 核心结论",
            "",
            f"- MAE 是否有 method 平均超过 PatchTST：{has_mae_gain}",
            f"- MSE 是否有 method 平均超过 PatchTST：{has_mse_gain}",
            f"- MAE 排名第一：{mae_best['method']}，delta={mae_best['delta_mae_vs_patchtst_mean']:.8f}",
            f"- MSE 排名第一：{mse_best['method']}，delta={mse_best['delta_mse_vs_patchtst_mean']:.8f}",
            f"- {boundary_line}",
            "",
            "## 完整大表",
            "",
            frame_to_markdown(summary),
            "",
        ]
    )


def build_ranking_markdown(summary: pd.DataFrame) -> str:
    """函数功能：单独输出按 MAE/MSE delta 排名和 residual-safe 方法标注。"""
    mae_rank = summary.sort_values(["delta_mae_vs_patchtst_mean", "delta_mse_vs_patchtst_mean"], ascending=[False, False])
    mse_rank = summary.sort_values("delta_mse_vs_patchtst_mean", ascending=False)
    residual_safe = summary[summary["method"].str.startswith("patchtst_residual_visual", na=False)]
    return "\n".join(
        [
            "# PatchTST + Visual Dual-Branch Ranking",
            "",
            "## MAE ranking",
            "",
            frame_to_markdown(mae_rank[["method", "delta_mae_vs_patchtst_mean", "dual_branch_mae_mean", "beats_patchtst_mae_rate", "note"]]),
            "",
            "## MSE ranking",
            "",
            frame_to_markdown(mse_rank[["method", "delta_mse_vs_patchtst_mean", "dual_branch_mse_mean", "beats_patchtst_mse_rate", "note"]]),
            "",
            "## residual-safe methods",
            "",
            frame_to_markdown(residual_safe[["method", "delta_mae_vs_patchtst_mean", "delta_mse_vs_patchtst_mean", "beats_patchtst_mse_rate", "note"]]),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    """函数功能：解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Build remaining and full dual-branch comparison tables.")
    parser.add_argument("--remaining_root", type=Path, required=True)
    parser.add_argument("--historical_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """函数功能：执行本轮汇总和历史+本轮完整大表生成。"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    remaining_runs = load_rows(args.remaining_root, source="remaining_2026_06_23")
    historical_runs = load_rows(args.historical_root, source="robust_multiseed_1b20e72")
    full_runs = pd.concat([historical_runs, remaining_runs], ignore_index=True)

    remaining_summary = summarize_by_method(remaining_runs)
    full_summary = summarize_by_method(full_runs)

    remaining_runs.to_csv(args.output_dir / "remaining_run_metrics.csv", index=False)
    remaining_summary.to_csv(args.output_dir / "remaining_summary.csv", index=False)
    full_summary.to_csv(args.output_dir / "full_dual_branch_comparison.csv", index=False)
    (args.output_dir / "full_dual_branch_comparison.json").write_text(
        json.dumps(full_summary.to_dict(orient="records"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "full_dual_branch_comparison.md").write_text(build_comparison_markdown(full_summary), encoding="utf-8")
    (args.output_dir / "full_dual_branch_ranking.md").write_text(build_ranking_markdown(full_summary), encoding="utf-8")

    missing = remaining_runs[remaining_runs["missing_outputs"].astype(str) != ""]
    if not missing.empty:
        raise RuntimeError("部分本轮 run 缺少验收产物：\n" + missing[["run_dir", "missing_outputs"]].to_string(index=False))
    print(f"完成：完整比较大表已写入 {args.output_dir}")


if __name__ == "__main__":
    main()
