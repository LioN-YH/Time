#!/usr/bin/env python3
"""
文件功能：
    汇总 Visual Router V2 Round2 staged full-scale validation thin slice / 1M gate 输出。

核心职责：
    - 复核 feature manifest、prediction SQLite、训练聚合 summary 是否存在；
    - 从逐 seed prediction CSV 生成 overall / strata / tail / router behavior 报告；
    - 固定后续 1M / near-full scale 可复用的 report schema 字段；
    - 只汇总轻量 CSV/JSON/Markdown，不复制 checkpoint、SQLite 或大规模逐样本输出。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = Path("/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_staged_fullscale_validation_thin_slice")
DEFAULT_SUMMARY_DIR = REPO_ROOT / "experiment_summaries" / "visual_router_v2_round2" / "staged_fullscale_validation"
DEFAULT_ARTIFACT_PREFIX = "round2_staged_fullscale"
MODEL_COLUMNS = ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]
STRATA_COLUMNS = [
    "oracle_model",
    "dataset_name",
    "group_name",
    "error_gap_quantile",
    "forecastability_cat",
    "season_strength_cat",
    "trend_strength_cat",
    "cv_cat",
]
SCRIPT_VERSION = "visual_router_v2_round2_staged_summary_v1"


def display_time() -> str:
    """函数功能：生成写入 summary/metadata 的本地时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_csv(text: str) -> List[str]:
    """函数功能：解析逗号分隔参数，并去重保序。"""
    values: List[str] = []
    for part in str(text).split(","):
        value = part.strip()
        if value and value not in values:
            values.append(value)
    if not values:
        raise ValueError("逗号分隔参数不能为空")
    return values


def parse_seeds(text: str) -> List[int]:
    """函数功能：解析 seed 列表。"""
    return [int(value) for value in parse_csv(text)]


def parse_args() -> argparse.Namespace:
    """函数功能：解析 staged summary 参数。"""
    parser = argparse.ArgumentParser(description="Summarize Round2 staged full-scale validation outputs.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--summary-copy-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--artifact-prefix", default=DEFAULT_ARTIFACT_PREFIX)
    parser.add_argument("--layouts", default="spatial_panel_3view,current_rgb_3view")
    parser.add_argument("--seeds", default="16")
    parser.add_argument("--sample-scale", choices=["smoke", "one_shard", "one_million"], default="smoke")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def git_commit_hash() -> str:
    """函数功能：记录当前 commit，失败时返回 unknown。"""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def feature_manifest_check(run_dir: Path, prefix: str, expected_layouts: Sequence[str]) -> Dict[str, object]:
    """函数功能：检查 shard-aware feature manifest 是否覆盖 layout/sample_set。"""
    path = run_dir / f"{prefix}_feature_manifest.csv"
    if not path.exists():
        return {"passed": False, "path": str(path), "error": "missing_feature_manifest"}
    frame = pd.read_csv(path)
    required = {"layout_name", "sample_set", "sample_count", "shard_path", "start_order_index", "end_order_index"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return {"passed": False, "path": str(path), "error": f"missing_columns={missing}"}
    missing_shards = [str(p) for p in frame["shard_path"].astype(str).map(Path) if not p.exists()]
    coverage = frame.groupby(["layout_name", "sample_set"], as_index=False)["sample_count"].sum()
    return {
        "passed": not missing_shards,
        "path": str(path),
        "row_count": int(len(frame)),
        "layouts": sorted(frame["layout_name"].astype(str).unique().tolist()),
        "expected_layouts": list(expected_layouts),
        "missing_shard_count": len(missing_shards),
        "missing_shards_preview": missing_shards[:5],
        "coverage": coverage.to_dict(orient="records"),
    }


def prediction_lookup_check(run_dir: Path, sample_manifest: Path) -> Dict[str, object]:
    """函数功能：检查 subset SQLite prediction lookup 是否覆盖 staged sample_key × 五专家。"""
    index_path = run_dir / "prediction_index_round2_layout_subset.sqlite"
    manifest = pd.read_csv(sample_manifest)
    sample_count = int(manifest["sample_key"].astype(str).nunique())
    if not index_path.exists():
        return {"passed": False, "path": str(index_path), "error": "missing_prediction_sqlite", "expected_records": sample_count * len(MODEL_COLUMNS)}
    connection = sqlite3.connect(str(index_path))
    try:
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", connection)["name"].astype(str).tolist()
        table_name = "prediction_index" if "prediction_index" in tables else tables[0]
        record_count = int(pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {table_name}", connection)["n"].iloc[0])
        unique_samples = int(pd.read_sql_query(f"SELECT COUNT(DISTINCT sample_key) AS n FROM {table_name}", connection)["n"].iloc[0])
        per_model = pd.read_sql_query(f"SELECT model_name, COUNT(*) AS n FROM {table_name} GROUP BY model_name", connection)
    finally:
        connection.close()
    expected = sample_count * len(MODEL_COLUMNS)
    return {
        "passed": record_count == expected and unique_samples == sample_count,
        "path": str(index_path),
        "sample_count": sample_count,
        "record_count": record_count,
        "expected_records": expected,
        "unique_samples": unique_samples,
        "per_model": per_model.to_dict(orient="records"),
    }


def load_predictions(run_dir: Path, layouts: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    """函数功能：读取 layout/seed task 逐样本 prediction CSV。"""
    frames: List[pd.DataFrame] = []
    missing: List[str] = []
    for layout in layouts:
        for seed in seeds:
            task_dir = run_dir / "tasks" / f"{layout}_seed{int(seed)}"
            paths = sorted(task_dir.glob(f"predictions_{layout}_seed{int(seed)}_staged_*.csv"))
            if not paths:
                missing.append(str(task_dir))
                continue
            for path in paths:
                frame = pd.read_csv(path)
                frame["layout_name"] = layout
                frame["seed"] = int(seed)
                frames.append(frame)
    if missing:
        raise FileNotFoundError("缺少 prediction CSV：" + "; ".join(missing[:10]))
    return pd.concat(frames, ignore_index=True)


def metric_columns(frame: pd.DataFrame) -> Dict[str, str]:
    """函数功能：兼容现有 prediction CSV 中的指标列名。"""
    candidates = {
        "soft_mae": ["raw_soft_fusion_MAE", "soft_fusion_mae", "soft_MAE", "MAE"],
        "soft_mse": ["raw_soft_fusion_MSE", "soft_fusion_mse", "soft_MSE", "MSE"],
        "soft_regret": ["raw_soft_fusion_regret_to_oracle"],
        "hard_mae": ["hard_top1_MAE", "hard_top1_mae_from_array", "MAE"],
        "hard_mse": ["hard_top1_MSE", "hard_top1_mse_from_array", "MSE"],
        "hard_regret": ["hard_top1_regret_to_oracle", "regret_to_oracle"],
        "entropy": ["weight_entropy"],
        "max_weight": ["max_weight", "mean_max_weight"],
    }
    resolved: Dict[str, str] = {}
    for key, names in candidates.items():
        for name in names:
            if name in frame.columns:
                resolved[key] = name
                break
    return resolved


def add_metric_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """函数功能：为 staged report 增加统一指标列。"""
    df = frame.copy()
    cols = metric_columns(df)
    for alias, source in cols.items():
        df[alias] = pd.to_numeric(df[source], errors="coerce")
    if "soft_regret" not in df.columns and {"soft_mae", "oracle_value"}.issubset(df.columns):
        # 现有 Round2 prediction CSV 中 `regret_to_oracle` 对应 hard top1；
        # raw-soft regret 需要由 soft fusion MAE 减 oracle MAE 现场计算。
        df["soft_regret"] = pd.to_numeric(df["soft_mae"], errors="coerce") - pd.to_numeric(df["oracle_value"], errors="coerce")
    if "oracle_label_correct" in df.columns:
        df["oracle_label_accuracy"] = df["oracle_label_correct"].astype(float)
    elif {"selected_model", "oracle_model"}.issubset(df.columns):
        df["oracle_label_accuracy"] = (df["selected_model"].astype(str) == df["oracle_model"].astype(str)).astype(float)
    if "max_weight" not in df.columns:
        weight_cols = [col for col in df.columns if col.endswith("_weight")]
        if weight_cols:
            df["max_weight"] = df[weight_cols].max(axis=1)
    return df


def mean_or_nan(series: pd.Series) -> float:
    """函数功能：计算均值，空列返回 NaN。"""
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if values.notna().any() else float("nan")


def build_overall(predictions: pd.DataFrame) -> pd.DataFrame:
    """函数功能：生成 overall report，包含 raw-soft、hard top1、entropy、per-seed 统计。"""
    rows: List[Dict[str, object]] = []
    group_cols = ["sample_set", "layout_name", "seed"]
    per_seed: List[Dict[str, object]] = []
    for keys, group in predictions.groupby(group_cols, sort=False):
        sample_set, layout, seed = keys
        per_seed.append(
            {
                "sample_set": sample_set,
                "layout_name": layout,
                "seed": int(seed),
                "sample_count": int(len(group)),
                "raw_soft_MAE": mean_or_nan(group.get("soft_mae", pd.Series(dtype=float))),
                "raw_soft_MSE": mean_or_nan(group.get("soft_mse", pd.Series(dtype=float))),
                "raw_soft_regret": mean_or_nan(group.get("soft_regret", pd.Series(dtype=float))),
                "hard_top1_MAE": mean_or_nan(group.get("hard_mae", pd.Series(dtype=float))),
                "hard_top1_MSE": mean_or_nan(group.get("hard_mse", pd.Series(dtype=float))),
                "hard_top1_regret": mean_or_nan(group.get("hard_regret", pd.Series(dtype=float))),
                "oracle_label_accuracy": mean_or_nan(group.get("oracle_label_accuracy", pd.Series(dtype=float))),
                "entropy": mean_or_nan(group.get("entropy", pd.Series(dtype=float))),
                "mean_max_weight": mean_or_nan(group.get("max_weight", pd.Series(dtype=float))),
                "raw_soft_vs_hard_top1_MAE_gap": mean_or_nan(group.get("soft_mae", pd.Series(dtype=float))) - mean_or_nan(group.get("hard_mae", pd.Series(dtype=float))),
            }
        )
    per_seed_df = pd.DataFrame(per_seed)
    metrics = [col for col in per_seed_df.columns if col not in {"sample_set", "layout_name", "seed", "sample_count"}]
    for keys, group in per_seed_df.groupby(["sample_set", "layout_name"], sort=False):
        sample_set, layout = keys
        row: Dict[str, object] = {
            "sample_set": sample_set,
            "layout_name": layout,
            "seed_count": int(group["seed"].nunique()),
            "sample_count_per_seed": int(group["sample_count"].iloc[0]),
            "per_seed_metrics": json.dumps(group.to_dict(orient="records"), ensure_ascii=False),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = mean_or_nan(group[metric])
            row[f"{metric}_std"] = float(pd.to_numeric(group[metric], errors="coerce").std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def build_strata(predictions: pd.DataFrame) -> pd.DataFrame:
    """函数功能：按固定 strata schema 生成 raw-soft 与 router 行为摘要。"""
    rows: List[Dict[str, object]] = []
    for column in STRATA_COLUMNS:
        if column not in predictions.columns:
            continue
        for keys, group in predictions.groupby(["sample_set", "layout_name", column], dropna=False, sort=False):
            sample_set, layout, value = keys
            rows.append(
                {
                    "sample_set": sample_set,
                    "layout_name": layout,
                    "stratum_column": column,
                    "stratum_value": value,
                    "sample_count": int(len(group)),
                    "raw_soft_MAE": mean_or_nan(group.get("soft_mae", pd.Series(dtype=float))),
                    "raw_soft_MSE": mean_or_nan(group.get("soft_mse", pd.Series(dtype=float))),
                    "raw_soft_regret": mean_or_nan(group.get("soft_regret", pd.Series(dtype=float))),
                    "hard_top1_MAE": mean_or_nan(group.get("hard_mae", pd.Series(dtype=float))),
                    "entropy": mean_or_nan(group.get("entropy", pd.Series(dtype=float))),
                    "mean_max_weight": mean_or_nan(group.get("max_weight", pd.Series(dtype=float))),
                    "oracle_label_accuracy": mean_or_nan(group.get("oracle_label_accuracy", pd.Series(dtype=float))),
                }
            )
    return pd.DataFrame(rows)


def _tail_group_rows(group: pd.DataFrame, metric: str, fraction: float) -> pd.DataFrame:
    """函数功能：按指标取 top tail，至少保留 1 行。"""
    if metric not in group.columns:
        return group.iloc[0:0].copy()
    count = max(1, int(np.ceil(len(group) * float(fraction))))
    return group.sort_values(metric, ascending=False, kind="mergesort").head(count).copy()


def _distribution_text(frame: pd.DataFrame, column: str) -> str:
    """函数功能：把 tail 内模型分布压成 JSON 字符串。"""
    if column not in frame.columns or frame.empty:
        return "{}"
    counts = frame[column].astype(str).value_counts().reindex(MODEL_COLUMNS, fill_value=0)
    return json.dumps({k: int(v) for k, v in counts.items()}, ensure_ascii=False)


def build_tail(predictions: pd.DataFrame) -> pd.DataFrame:
    """函数功能：生成 top 1%/5% soft MAE 与 regret tail 指标。"""
    rows: List[Dict[str, object]] = []
    for keys, group in predictions.groupby(["sample_set", "layout_name", "seed"], sort=False):
        sample_set, layout, seed = keys
        mae_1 = _tail_group_rows(group, "soft_mae", 0.01)
        mae_5 = _tail_group_rows(group, "soft_mae", 0.05)
        regret_1 = _tail_group_rows(group, "soft_regret", 0.01)
        regret_5 = _tail_group_rows(group, "soft_regret", 0.05)
        mae_1_keys = set(mae_1["sample_key"].astype(str)) if "sample_key" in mae_1 else set()
        regret_1_keys = set(regret_1["sample_key"].astype(str)) if "sample_key" in regret_1 else set()
        overlap = len(mae_1_keys & regret_1_keys) / max(1, len(mae_1_keys | regret_1_keys))
        rows.append(
            {
                "sample_set": sample_set,
                "layout_name": layout,
                "seed": int(seed),
                "sample_count": int(len(group)),
                "top1pct_soft_MAE": mean_or_nan(mae_1.get("soft_mae", pd.Series(dtype=float))),
                "top5pct_soft_MAE": mean_or_nan(mae_5.get("soft_mae", pd.Series(dtype=float))),
                "top1pct_regret": mean_or_nan(regret_1.get("soft_regret", pd.Series(dtype=float))),
                "top5pct_regret": mean_or_nan(regret_5.get("soft_regret", pd.Series(dtype=float))),
                "tail_overlap_top1pct_mae_regret_jaccard": float(overlap),
                "tail_oracle_model_distribution": _distribution_text(pd.concat([mae_1, regret_1], ignore_index=True).drop_duplicates("sample_key"), "oracle_model") if "sample_key" in group else "{}",
                "tail_selected_model_distribution": _distribution_text(pd.concat([mae_1, regret_1], ignore_index=True).drop_duplicates("sample_key"), "selected_model") if "sample_key" in group else "{}",
            }
        )
    return pd.DataFrame(rows)


def build_router_behavior(predictions: pd.DataFrame) -> pd.DataFrame:
    """函数功能：生成 selected_model ratio、entropy/max weight 和 soft-hard gap。"""
    rows: List[Dict[str, object]] = []
    for keys, group in predictions.groupby(["sample_set", "layout_name", "seed"], sort=False):
        sample_set, layout, seed = keys
        counts = group["selected_model"].astype(str).value_counts().reindex(MODEL_COLUMNS, fill_value=0) if "selected_model" in group.columns else pd.Series(0, index=MODEL_COLUMNS)
        row: Dict[str, object] = {
            "sample_set": sample_set,
            "layout_name": layout,
            "seed": int(seed),
            "sample_count": int(len(group)),
            "entropy": mean_or_nan(group.get("entropy", pd.Series(dtype=float))),
            "mean_max_weight": mean_or_nan(group.get("max_weight", pd.Series(dtype=float))),
            "raw_soft_vs_hard_top1_MAE_gap": mean_or_nan(group.get("soft_mae", pd.Series(dtype=float))) - mean_or_nan(group.get("hard_mae", pd.Series(dtype=float))),
        }
        for model in MODEL_COLUMNS:
            row[f"selected_ratio_{model}"] = float(counts[model] / max(1, len(group)))
        rows.append(row)
    return pd.DataFrame(rows)


def frame_to_markdown(frame: pd.DataFrame, max_rows: int = 12) -> str:
    """函数功能：小表转 Markdown，避免 summary 过长。"""
    if frame.empty:
        return "_empty_"
    head = frame.head(max_rows).copy()
    columns = [str(col) for col in head.columns]
    rows = [["" if pd.isna(value) else str(value) for value in row] for row in head.to_numpy(dtype=object)]
    widths = [len(col) for col in columns]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    header = "| " + " | ".join(col.ljust(widths[idx]) for idx, col in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = ["| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_summary_md(run_dir: Path, output_paths: Mapping[str, Path], checks: Mapping[str, object], metadata: Mapping[str, object]) -> None:
    """函数功能：写 staged validation 中文摘要。"""
    overall = pd.read_csv(output_paths["overall"])
    tail = pd.read_csv(output_paths["tail"])
    behavior = pd.read_csv(output_paths["router_behavior"])
    sample_scale = str(metadata.get("sample_scale", "smoke"))
    if sample_scale == "one_million":
        title = "# Visual Router V2 Round2 1M Staged Seed16 Gate"
        scope_line = "- 本次是 1M staged gate，不是 116M fullscale 正式结论。"
        next_steps = [
            "1. 若 staged_selection / staged_test 无灾难性退化，可启动 `spatial_panel_3view + film_mean_patch_aux` fullscale seed16。",
            "2. fullscale 完成后与 TimeFuse fullscale 做 single-seed first pass MAE/MSE 对比。",
            "3. 后续再补 `current_rgb_3view` fullscale baseline 或 seeds 17/18。",
        ]
    else:
        title = "# Visual Router V2 Round2 Staged Full-Scale Validation Thin Slice"
        scope_line = "- 本次是 staged thin slice，不是 1M 或 116M 正式长跑。"
        next_steps = [
            "1. `--sample-scale one_shard`：从同一 full-scale shard 扩到每集合 512 或更高，验证单 shard cache/lookup/report 稳定性。",
            "2. 1M staged planning：增加 shard list 参数，按 shard 生成 feature cache 与 subset SQLite，保持 train/selection/diagnostic/test 分离。",
            "3. near-full scale：只在 selection 冻结后做 test frozen eval，不用 test 选择 variant、seed、epoch 或超参。",
        ]
    lines = [
        title,
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 结论",
        "",
        scope_line,
        "- pipeline 已覆盖 staged sample manifest、shard-aware feature cache、subset SQLite prediction lookup、fixed FiLM train/eval aggregation 和 report schema。",
        "- 当前只比较 `spatial_panel_3view` 与 `current_rgb_3view`，后端固定 `film_mean_patch_aux`。",
        "",
        "## 检查",
        "",
        f"- feature manifest：passed={checks['feature_manifest']['passed']}，path=`{checks['feature_manifest']['path']}`",
        f"- prediction lookup：passed={checks['prediction_lookup']['passed']}，records={checks['prediction_lookup'].get('record_count')}/{checks['prediction_lookup'].get('expected_records')}",
        "",
        "## Overall",
        "",
        frame_to_markdown(overall),
        "",
        "## Tail",
        "",
        frame_to_markdown(tail),
        "",
        "## Router Behavior",
        "",
        frame_to_markdown(behavior),
        "",
        "## 下一步扩大方案",
        "",
        *next_steps,
    ]
    (run_dir / "round2_staged_fullscale_validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_light_outputs(paths: Sequence[Path], summary_dir: Path) -> None:
    """函数功能：复制轻量 summary 到仓库 experiment_summaries。"""
    summary_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, summary_dir / path.name)


def main() -> None:
    """函数功能：生成 staged full-scale report CSV/JSON/Markdown。"""
    args = parse_args()
    run_dir = Path(args.run_dir)
    layouts = parse_csv(args.layouts)
    seeds = parse_seeds(args.seeds)
    predictions = add_metric_aliases(load_predictions(run_dir, layouts, seeds))
    overall = build_overall(predictions)
    strata = build_strata(predictions)
    tail = build_tail(predictions)
    behavior = build_router_behavior(predictions)

    paths = {
        "overall": run_dir / "round2_staged_fullscale_overall_report.csv",
        "strata": run_dir / "round2_staged_fullscale_strata_report.csv",
        "tail": run_dir / "round2_staged_fullscale_tail_report.csv",
        "router_behavior": run_dir / "round2_staged_fullscale_router_behavior_report.csv",
        "metadata": run_dir / "round2_staged_fullscale_metadata.json",
        "summary": run_dir / "round2_staged_fullscale_validation_summary.md",
    }
    overall.to_csv(paths["overall"], index=False)
    strata.to_csv(paths["strata"], index=False)
    tail.to_csv(paths["tail"], index=False)
    behavior.to_csv(paths["router_behavior"], index=False)

    checks = {
        "feature_manifest": feature_manifest_check(run_dir, str(args.artifact_prefix), layouts),
        "prediction_lookup": prediction_lookup_check(run_dir, Path(args.sample_manifest)),
    }
    metadata = {
        "status": "completed",
        "generated_at": display_time(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "commit_hash": git_commit_hash(),
        "run_dir": str(run_dir),
        "sample_manifest": str(args.sample_manifest),
        "sample_scale": str(args.sample_scale),
        "layouts": layouts,
        "seeds": seeds,
        "backend": "film_mean_patch_aux",
        "report_schema_sections": ["overall", "strata", "tail", "router_behavior", "per_seed_metrics"],
        "checks": checks,
        "constraints": {
            "not_1m_run": str(args.sample_scale) != "one_million",
            "is_1m_staged_gate": str(args.sample_scale) == "one_million",
            "not_116m_full_scale_run": True,
            "loaded_116m_prediction_manifest_to_memory": False,
            "saved_pseudo_image_tensor": False,
            "test_used_for_training_or_selection": False,
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    write_json(paths["metadata"], metadata)
    write_summary_md(run_dir, paths, checks, metadata)
    copy_light_outputs(list(paths.values()), Path(args.summary_copy_dir))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
