#!/usr/bin/env python3
"""
文件功能：
    汇总五个模型在三组 QuitoBench 配置下的 evaluate 结果。

输入：
    - 15 个 evaluate JSON：
      DLinear / PatchTST / CrossFormer 使用 quito/outputs/default_baseline/；
      ES / SNaive 使用 quito/outputs/statistical_baseline/。
    - item 到 TSF cell / cluster 的映射：
      quito/examples/datasets/cluster_data/item_clusters.csv。

输出：
    - overall_mean_metrics.csv：每个模型每个配置的 item-level mean metrics。
    - overall_mean_mae_pivot.csv：论文式整体 mean MAE 透视表。
    - tsf_cell_metrics.csv：每个模型、配置、TSF cell 的分组指标。
    - tsf_cell_mae_pivot.csv：分 TSF cell 的 MAE 透视表。
    - per_item_metrics.csv：15 组 evaluate 展开的 per-item 明细。
    - checkpoint_lineage.csv：deep model evaluate 使用的 checkpoint 口径。
    - summary.md：中文汇总说明，便于直接查看。

关键口径：
    当前主表使用原始 default baseline，也就是 DLinear / PatchTST / CrossFormer
    在 quito/outputs/default_baseline/ 下的 validation MAE-best checkpoint evaluate 结果。
    后补的 validation MSE-best PatchTST 576_288_S 不进入主表，只在 summary.md 中说明。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yaml


WORKSPACE = Path("/home/shiyuhong/Time")
REPO_DIR = WORKSPACE / "quito"
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"
CLUSTER_PATH = REPO_DIR / "examples" / "datasets" / "cluster_data" / "item_clusters.csv"

CONFIG_ORDER = ["96_48_S", "576_288_S", "1024_512_S"]
MODEL_ORDER = ["DLinear", "PatchTST", "CrossFormer", "ES", "SNaive"]
METRIC_ORDER = ["MSE", "MAE", "MASE", "MASE_LEAK", "MAPE", "SMAPE", "SMASE"]


@dataclass(frozen=True)
class EvalSource:
    """函数功能：记录单个模型/配置 evaluate JSON 的来源和结果口径。"""

    model: str
    config_name: str
    eval_results_path: Path
    evaluate_config_path: Path
    source_family: str
    checkpoint_selection: str


def now_token() -> str:
    """函数功能：生成 run 目录中的时间戳，精确到微秒避免重名。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 Markdown/JSON 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def build_sources() -> List[EvalSource]:
    """函数功能：按固定顺序构造 15 个 evaluate JSON 来源。"""
    deep_specs = [
        ("DLinear", "dlinear", "eval_results_DLinear.json"),
        ("PatchTST", "patchtst", "eval_results_PatchTST.json"),
        ("CrossFormer", "crossformer", "eval_results_CrossFormer.json"),
    ]
    stat_specs = [
        ("ES", "es", "eval_results_ES.json"),
        ("SNaive", "snaive", "eval_results_NaiveForecaster.json"),
    ]

    sources: List[EvalSource] = []
    for model_name, model_dir, result_name in deep_specs:
        for config_name in CONFIG_ORDER:
            base = REPO_DIR / "outputs" / "default_baseline" / model_dir / config_name / "seed_16" / "EVALUATE" / "ver_0"
            sources.append(
                EvalSource(
                    model=model_name,
                    config_name=config_name,
                    eval_results_path=base / result_name,
                    evaluate_config_path=base / "config.yaml",
                    source_family="default_baseline",
                    checkpoint_selection="validation_mae_best",
                )
            )

    for model_name, model_dir, result_name in stat_specs:
        for config_name in CONFIG_ORDER:
            base = REPO_DIR / "outputs" / "statistical_baseline" / model_dir / config_name / "seed_16" / "EVALUATE" / "ver_0"
            sources.append(
                EvalSource(
                    model=model_name,
                    config_name=config_name,
                    eval_results_path=base / result_name,
                    evaluate_config_path=base / "config.yaml",
                    source_family="statistical_baseline",
                    checkpoint_selection="not_applicable_statistical_model",
                )
            )
    return sources


def validate_sources(sources: Iterable[EvalSource]) -> None:
    """函数功能：启动汇总前确认所有输入文件存在，避免静默漏项。"""
    missing: List[str] = []
    for source in sources:
        if not source.eval_results_path.exists():
            missing.append(str(source.eval_results_path))
        if not source.evaluate_config_path.exists():
            missing.append(str(source.evaluate_config_path))
    if not CLUSTER_PATH.exists():
        missing.append(str(CLUSTER_PATH))
    if missing:
        raise FileNotFoundError("以下输入文件不存在：\n" + "\n".join(missing))


def load_checkpoint_path(config_path: Path, source_family: str) -> Optional[str]:
    """函数功能：从 evaluate config 中提取 checkpoint 路径，统计模型返回空。"""
    if source_family == "statistical_baseline":
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    checkpoint_path = config.get("resume", {}).get("checkpoint_path")
    if isinstance(checkpoint_path, list):
        checkpoint_path = checkpoint_path[0] if checkpoint_path else None
    return checkpoint_path


def load_eval_rows(source: EvalSource) -> pd.DataFrame:
    """函数功能：把 Quito 原始 evaluate JSON 展开成 per-item DataFrame。"""
    with source.eval_results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows: List[Dict[str, object]] = []
    for item in payload.get("final_results", []):
        row: Dict[str, object] = {
            "model": source.model,
            "config_name": source.config_name,
            "seed": 16,
            "item_id": int(item["user_id"]),
            "n_samples": int(item.get("n_samples", 0)),
            "eval_time": float(item.get("eval_time", 0.0)),
            "source_family": source.source_family,
            "checkpoint_selection": source.checkpoint_selection,
            "eval_results_path": str(source.eval_results_path),
        }
        row.update(item.get("metrics", {}))
        rows.append(row)

    if not rows:
        raise ValueError(f"{source.eval_results_path} 中没有 final_results。")
    return pd.DataFrame(rows)


def merge_cluster_info(per_item_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：合并 TSF cell/cluster 映射，并检查是否存在缺失映射。"""
    cluster_df = pd.read_csv(CLUSTER_PATH)
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
    merged = per_item_df.merge(cluster_df[keep_cols], on="item_id", how="left")
    missing_count = int(merged["group_name"].isna().sum())
    if missing_count:
        raise ValueError(f"有 {missing_count} 行 evaluate 结果缺少 TSF cell 映射。")
    return merged


def metric_columns(df: pd.DataFrame) -> List[str]:
    """函数功能：按固定顺序返回当前结果中存在的数值指标列。"""
    return [metric for metric in METRIC_ORDER if metric in df.columns]


def summarize_overall(per_item_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：计算模型-配置级 overall mean metrics。"""
    metrics = metric_columns(per_item_df)
    rows: List[Dict[str, object]] = []
    group_cols = ["model", "config_name", "seed", "source_family", "checkpoint_selection", "eval_results_path"]
    for keys, group in per_item_df.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys))
        row["item_count"] = int(group["item_id"].nunique())
        row["sample_count"] = int(group["n_samples"].sum())
        for metric in metrics:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    return order_model_config(pd.DataFrame(rows))


def summarize_tsf_cells(per_item_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：计算模型-配置-TSF cell 级指标均值和 MAE 标准差。"""
    metrics = metric_columns(per_item_df)
    rows: List[Dict[str, object]] = []
    group_cols = ["model", "config_name", "seed", "cluster", "group_name"]
    for keys, group in per_item_df.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys))
        row["item_count"] = int(group["item_id"].nunique())
        row["sample_count"] = int(group["n_samples"].sum())
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
        if "MAE" in group:
            row["MAE_std"] = float(group["MAE"].std(ddof=0))
        rows.append(row)
    result = pd.DataFrame(rows)
    result["cluster"] = result["cluster"].astype(int)
    return order_model_config(result).sort_values(["config_order", "cluster", "model_order"]).drop(
        columns=["model_order", "config_order"]
    )


def order_model_config(df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：增加稳定排序键，使输出表顺序与实验叙述一致。"""
    model_rank = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    config_rank = {config: idx for idx, config in enumerate(CONFIG_ORDER)}
    df = df.copy()
    df["model_order"] = df["model"].map(model_rank)
    df["config_order"] = df["config_name"].map(config_rank)
    return df.sort_values(["model_order", "config_order"]).reset_index(drop=True)


def make_overall_mae_pivot(overall_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：生成模型 × 配置的 overall MAE 透视表。"""
    pivot = overall_df.pivot(index="model", columns="config_name", values="MAE")
    return pivot.reindex(index=MODEL_ORDER, columns=CONFIG_ORDER).reset_index()


def make_tsf_mae_pivot(tsf_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：生成 TSF cell × 模型配置的 MAE 透视表。"""
    table = tsf_df.pivot_table(
        index=["cluster", "group_name"],
        columns=["config_name", "model"],
        values="MAE_mean",
        aggfunc="first",
    )
    table = table.reindex(columns=pd.MultiIndex.from_product([CONFIG_ORDER, MODEL_ORDER]))
    table.columns = [f"{config}__{model}" for config, model in table.columns]
    return table.reset_index().sort_values(["cluster", "group_name"])


def build_checkpoint_lineage(sources: Iterable[EvalSource]) -> pd.DataFrame:
    """函数功能：记录每个 evaluate 来源及其 checkpoint 选择口径。"""
    rows = []
    for source in sources:
        checkpoint_path = load_checkpoint_path(source.evaluate_config_path, source.source_family)
        rows.append(
            {
                "model": source.model,
                "config_name": source.config_name,
                "seed": 16,
                "source_family": source.source_family,
                "checkpoint_selection": source.checkpoint_selection,
                "checkpoint_path": checkpoint_path or "not_applicable",
                "checkpoint_file": Path(checkpoint_path).name if checkpoint_path else "not_applicable",
                "evaluate_config_path": str(source.evaluate_config_path),
                "eval_results_path": str(source.eval_results_path),
            }
        )
    return order_model_config(pd.DataFrame(rows)).drop(columns=["model_order", "config_order"])


def frame_to_markdown(df: pd.DataFrame, *, float_digits: int = 6) -> str:
    """
    函数功能：
        将 DataFrame 转为 GitHub Markdown 表格。

    设计说明：
        pandas.DataFrame.to_markdown 依赖可选包 tabulate；当前实验环境没有安装该包。
        为避免为了写日志引入新依赖，这里实现一个轻量格式化函数。
    """
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda value: f"{value:.{float_digits}f}")
        else:
            display_df[col] = display_df[col].astype(str)

    headers = [str(col) for col in display_df.columns]
    rows = display_df.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown_summary(
    output_dir: Path,
    overall_pivot: pd.DataFrame,
    tsf_pivot: pd.DataFrame,
    checkpoint_df: pd.DataFrame,
) -> None:
    """函数功能：写出中文 Markdown 汇总，便于直接在日志中引用。"""
    lines = [
        "# 五模型三配置结果汇总",
        "",
        f"生成时间：{display_time()}",
        "",
        "## 口径",
        "",
        "- DLinear / PatchTST / CrossFormer 使用 `quito/outputs/default_baseline/` 的原始 evaluate 结果。",
        "- 上述 deep model evaluate checkpoint 为 validation MAE-best；这是当前 quick baseline 原始口径。",
        "- ES / SNaive 是统计模型，无训练 checkpoint。",
        "- `PatchTST 576_288_S` 另有 validation MSE-best 补评估，但不进入本主表；该补评估位于 `quito/outputs/default_baseline_mse_best/`。",
        "",
        "## Overall Mean MAE",
        "",
        frame_to_markdown(overall_pivot),
        "",
        "## TSF Cell Mean MAE",
        "",
        frame_to_markdown(tsf_pivot),
        "",
        "## Deep Model Checkpoint",
        "",
        frame_to_markdown(
            checkpoint_df[checkpoint_df["source_family"] == "default_baseline"][
                ["model", "config_name", "checkpoint_selection", "checkpoint_file"]
            ],
            float_digits=6,
        ),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """函数功能：解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Summarize five-model three-config QuitoBench results.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="汇总结果输出目录；默认写入 experiment_logs/run_outputs 下的时间戳目录。",
    )
    return parser.parse_args()


def main() -> None:
    """函数功能：执行完整汇总流程并写出所有结果文件。"""
    args = parse_args()
    output_dir = args.output_dir or RUN_OUTPUT_ROOT / f"{now_token()}_five_model_three_config_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = build_sources()
    validate_sources(sources)

    per_item_df = pd.concat([load_eval_rows(source) for source in sources], ignore_index=True)
    per_item_df = merge_cluster_info(per_item_df)

    expected_rows = len(sources) * 1290
    if len(per_item_df) != expected_rows:
        raise ValueError(f"per-item 行数异常：期望 {expected_rows}，实际 {len(per_item_df)}。")

    overall_df = summarize_overall(per_item_df).drop(columns=["model_order", "config_order"])
    tsf_df = summarize_tsf_cells(per_item_df)
    overall_pivot = make_overall_mae_pivot(overall_df)
    tsf_pivot = make_tsf_mae_pivot(tsf_df)
    checkpoint_df = build_checkpoint_lineage(sources)

    per_item_df.to_csv(output_dir / "per_item_metrics.csv", index=False)
    overall_df.to_csv(output_dir / "overall_mean_metrics.csv", index=False)
    overall_pivot.to_csv(output_dir / "overall_mean_mae_pivot.csv", index=False)
    tsf_df.to_csv(output_dir / "tsf_cell_metrics.csv", index=False)
    tsf_pivot.to_csv(output_dir / "tsf_cell_mae_pivot.csv", index=False)
    checkpoint_df.to_csv(output_dir / "checkpoint_lineage.csv", index=False)

    metadata = {
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "input_count": len(sources),
        "per_item_rows": int(len(per_item_df)),
        "tsf_cell_count": int(per_item_df["group_name"].nunique()),
        "cluster_path": str(CLUSTER_PATH),
        "main_table_checkpoint_policy": "deep models use validation_mae_best from quito/outputs/default_baseline",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown_summary(output_dir, overall_pivot, tsf_pivot, checkpoint_df)

    print(f"wrote summary outputs to {output_dir}")
    print(overall_pivot.to_string(index=False))


if __name__ == "__main__":
    main()
