#!/usr/bin/env python3
"""
文件功能：
    基于 window-level oracle labels 评估 Stage 1 非视觉 router baseline，并可同时训练
    TimeFuse-style 单层 fusor baseline。

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
    - timefuse_fusor_predictions.csv：TimeFuse-style fusor hard top-1 逐样本结果；
    - timefuse_fusor_raw_soft_fusion_predictions.csv：TimeFuse-style fusor raw soft fusion 数组级结果；
    - timefuse_fusor_summary.csv / timefuse_fusor_raw_soft_fusion_summary.csv：fusor 汇总；
    - timefuse_fusor_selected_model_counts.csv：fusor hard top-1 选中专家分布；
    - baseline_comparison.csv：统计规则、TimeFuse-style fusor hard/raw-soft 和 oracle 同表；
    - baseline_metadata.json：记录统计 baseline、fusor 训练口径、输入路径和 legacy baseline 说明；
    - summary.md：中文摘要。

设计说明：
    该脚本不训练视觉模型，只用 vali split 学简单规则，然后在 test split 上评估。
    目的是建立后续 visual router 必须超过的 metadata/statistics baseline。
    正式 Stage 1 采用 per-config router，因此 baseline 训练也必须在每个 config_name
    内独立完成，避免把不同输入/输出长度的专家误合并为一个动作空间。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


WORKSPACE = Path("/home/shiyuhong/Time")
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from visual_router_experiments.stage1_vali_test_router.fusion_utils import (  # noqa: E402
    MODEL_COLUMNS,
    load_feature_cache,
    load_prediction_lookup,
    run_timefuse_fusor_baseline,
)


DEFAULT_METRIC = "mae"
DEFAULT_FEATURE_CACHE_PATH = (
    RUN_OUTPUT_ROOT
    / "2026-06-13_113713_308023_visual_router_stage1_structure_feature_pilot"
    / "feature_cache.csv"
)
DEFAULT_LEGACY_LOGISTIC_SUMMARY_PATH = DEFAULT_FEATURE_CACHE_PATH.parent / "structure_router_summary.csv"


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
    parser.add_argument(
        "--timefuse-fusor",
        choices=["auto", "on", "off"],
        default="auto",
        help="是否训练 TimeFuse-style 单层 fusor baseline；auto 在 feature/prediction cache 存在且可对齐时运行。",
    )
    parser.add_argument(
        "--feature-cache-path",
        type=Path,
        default=DEFAULT_FEATURE_CACHE_PATH,
        help="build_structure_feature_cache_pilot.py 生成的 TimeFuse-derived feature_cache.csv。",
    )
    parser.add_argument(
        "--prediction-manifest-path",
        type=Path,
        default=None,
        help="五专家 prediction cache manifest；为空时默认使用 labels-path 同目录下的 manifest.csv。",
    )
    parser.add_argument("--fusor-epochs", type=int, default=5, help="TimeFuse-style fusor 训练 epoch，默认复刻 notebook 的 n_epochs=5。")
    parser.add_argument("--fusor-batch-size", type=int, default=64, help="TimeFuse-style fusor 训练 batch size。")
    parser.add_argument("--fusor-lr", type=float, default=5e-4, help="TimeFuse-style fusor Adam learning rate。")
    parser.add_argument("--fusor-beta", type=float, default=0.01, help="TimeFuse-style fusor SmoothL1Loss beta。")
    parser.add_argument("--seed", type=int, default=16, help="TimeFuse-style fusor 随机种子。")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="TimeFuse-style fusor 训练设备。")
    return parser.parse_args()


def display_time() -> str:
    """函数功能：生成 metadata 和 Markdown 中使用的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def resolve_device(device_arg: str) -> torch.device:
    """函数功能：解析 fusor 训练设备，auto 优先 CUDA。"""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 --device cuda，但当前 PyTorch CUDA 不可用")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    """函数功能：固定 TimeFuse-style fusor 相关随机源，保证 pilot 结果可复核。"""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def validate_fusor_args(args: argparse.Namespace) -> None:
    """函数功能：校验 TimeFuse-style fusor 训练超参，避免静默生成无效 baseline。"""
    if int(args.fusor_epochs) <= 0:
        raise ValueError("--fusor-epochs 必须为正整数")
    if int(args.fusor_batch_size) <= 0:
        raise ValueError("--fusor-batch-size 必须为正整数")
    if float(args.fusor_lr) <= 0:
        raise ValueError("--fusor-lr 必须为正数")
    if float(args.fusor_beta) <= 0:
        raise ValueError("--fusor-beta 必须为正数")


def resolve_prediction_manifest_path(args: argparse.Namespace) -> Path:
    """函数功能：解析 TimeFuse-style fusor 所需的 prediction cache manifest 路径。"""
    if args.prediction_manifest_path is not None:
        return args.prediction_manifest_path
    return args.labels_path.parent / "manifest.csv"


def should_attempt_timefuse_fusor(args: argparse.Namespace, prediction_manifest_path: Path) -> Tuple[bool, Optional[str]]:
    """
    函数功能：
        判断是否尝试运行 TimeFuse-style fusor。

    说明：
        `auto` 用于兼容没有同口径 feature cache 的大规模 baseline 目录；缺失时记录
        skip reason，不阻断统计 baseline 输出。`on` 则把缺失视为错误。
    """
    if args.timefuse_fusor == "off":
        return False, "命令行设置 --timefuse-fusor off"
    missing_paths = []
    if not args.feature_cache_path.exists():
        missing_paths.append(str(args.feature_cache_path))
    if not prediction_manifest_path.exists():
        missing_paths.append(str(prediction_manifest_path))
    if not missing_paths:
        return True, None
    reason = f"缺少 TimeFuse-style fusor 输入文件：{missing_paths}"
    if args.timefuse_fusor == "on":
        raise FileNotFoundError(reason)
    return False, reason


def run_timefuse_fusor_if_requested(
    *,
    args: argparse.Namespace,
    labels_df: pd.DataFrame,
    prediction_manifest_path: Path,
    device: torch.device,
) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    """
    函数功能：
        根据命令行参数运行或跳过 TimeFuse-style fusor baseline。

    返回：
        - fusor_result：实际输出表集合；跳过时为 None；
        - fusor_metadata：无论运行或跳过都写入 baseline_metadata.json。
    """
    attempt, skip_reason = should_attempt_timefuse_fusor(args, prediction_manifest_path)
    base_metadata: Dict[str, object] = {
        "requested_mode": str(args.timefuse_fusor),
        "status": "skipped",
        "skip_reason": skip_reason,
        "feature_cache_path": str(args.feature_cache_path),
        "prediction_manifest_path": str(prediction_manifest_path),
        "model": "nn.Linear(input_dim, output_dim) + softmax",
        "loss": "SmoothL1Loss",
        "beta": float(args.fusor_beta),
        "epochs": int(args.fusor_epochs),
        "batch_size": int(args.fusor_batch_size),
        "lr": float(args.fusor_lr),
        "seed": int(args.seed),
        "training_split": "vali",
        "evaluation_split": "test",
        "device": str(device),
    }
    if not attempt:
        return None, base_metadata

    set_seed(int(args.seed))
    try:
        feature_df, feature_cols = load_feature_cache(args.feature_cache_path)
        prediction_lookup = load_prediction_lookup(prediction_manifest_path)
        result = run_timefuse_fusor_baseline(
            feature_df=feature_df,
            labels_df=labels_df,
            prediction_lookup=prediction_lookup,
            metric=str(args.metric),
            feature_cols=feature_cols,
            epochs=int(args.fusor_epochs),
            batch_size=int(args.fusor_batch_size),
            lr=float(args.fusor_lr),
            beta=float(args.fusor_beta),
            seed=int(args.seed),
            device=device,
        )
    except Exception as exc:
        if args.timefuse_fusor == "on":
            raise
        base_metadata["skip_reason"] = f"auto 模式下 TimeFuse-style fusor 输入无法对齐或训练失败：{exc}"
        return None, base_metadata

    soft_summary_df = result["soft_summary_df"].copy()
    # 文件名和 comparison 中明确标注 raw soft fusion；训练权重未做 temperature/top-k 校准。
    soft_summary_df["router_name"] = soft_summary_df["router_name"].str.replace(
        "_soft_fusion",
        "_raw_soft_fusion",
        regex=False,
    )
    result["soft_summary_df"] = soft_summary_df

    base_metadata.update(
        {
            "status": "completed",
            "skip_reason": None,
            "feature_columns": list(feature_cols),
            "feature_dim": int(len(feature_cols)),
            "config_metadata": result["config_metadata"],
        }
    )
    return result, base_metadata


def build_baseline_comparison(
    *,
    statistics_summary: pd.DataFrame,
    fusor_result: Optional[Dict[str, object]],
    metric: str,
    output_dir: Path,
) -> pd.DataFrame:
    """函数功能：把统计规则、oracle 和 TimeFuse-style fusor 指标整理成公平同表。"""
    rows: List[Dict[str, object]] = []
    for row in statistics_summary.itertuples(index=False):
        rows.append(
            {
                "method": str(row.baseline),
                "method_family": str(row.rule_kind),
                "config_name": str(row.config_name),
                "sample_count": int(row.sample_count),
                "mae_like_value": float(row.selected_value),
                "oracle_value": float(row.oracle_value),
                "regret_to_oracle": float(row.regret_to_oracle),
                "oracle_label_accuracy": float(row.oracle_label_accuracy),
                "mean_weight_entropy": pd.NA,
                "mean_normalized_weight_entropy": pd.NA,
                "mean_max_weight": pd.NA,
                "source": str(output_dir / "baseline_summary.csv"),
            }
        )

    if fusor_result is not None:
        hard_summary = fusor_result["hard_summary_df"]
        for row in hard_summary.itertuples(index=False):
            rows.append(
                {
                    "method": str(row.router_name),
                    "method_family": "timefuse_style_hard_top1",
                    "config_name": str(row.config_name),
                    "sample_count": int(row.sample_count),
                    "mae_like_value": float(row.selected_value),
                    "oracle_value": float(row.oracle_value),
                    "regret_to_oracle": float(row.regret_to_oracle),
                    "oracle_label_accuracy": float(row.oracle_label_accuracy),
                    "mean_weight_entropy": float(row.mean_weight_entropy),
                    "mean_normalized_weight_entropy": float(row.mean_normalized_weight_entropy),
                    "mean_max_weight": float(row.mean_max_weight),
                    "source": str(output_dir / "timefuse_fusor_summary.csv"),
                }
            )

        soft_summary = fusor_result["soft_summary_df"]
        soft_metric_col = "soft_fusion_mae" if metric == "mae" else "soft_fusion_mse"
        for row in soft_summary.itertuples(index=False):
            soft_value = float(getattr(row, soft_metric_col))
            rows.append(
                {
                    "method": str(row.router_name),
                    "method_family": "timefuse_style_raw_soft_fusion",
                    "config_name": str(row.config_name),
                    "sample_count": int(row.sample_count),
                    "mae_like_value": soft_value,
                    "oracle_value": float(row.oracle_value),
                    "regret_to_oracle": float(soft_value - row.oracle_value),
                    "oracle_label_accuracy": pd.NA,
                    "mean_weight_entropy": float(row.mean_weight_entropy),
                    "mean_normalized_weight_entropy": float(row.mean_normalized_weight_entropy),
                    "mean_max_weight": float(row.mean_max_weight),
                    "source": str(output_dir / "timefuse_fusor_raw_soft_fusion_summary.csv"),
                }
            )

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        return comparison_df
    global_rows = comparison_df[comparison_df["method"] == "global_best_single"][["config_name", "mae_like_value"]]
    global_rows = global_rows.rename(columns={"mae_like_value": "global_best_single_value"})
    comparison_df = comparison_df.merge(global_rows, on="config_name", how="left")
    comparison_df["relative_improvement_vs_global_best_single"] = (
        comparison_df["global_best_single_value"] - comparison_df["mae_like_value"]
    ) / comparison_df["global_best_single_value"]
    return comparison_df.drop(columns=["global_best_single_value"]).sort_values(["config_name", "mae_like_value", "method"]).reset_index(drop=True)


def frame_to_markdown(df: pd.DataFrame, *, float_digits: int = 6) -> str:
    """函数功能：将 DataFrame 转为 Markdown 表格，避免依赖 tabulate。"""
    if df.empty:
        return "_无记录_"
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}")
        else:
            display_df[col] = display_df[col].map(lambda value: "" if pd.isna(value) else str(value))
    lines = [
        "| " + " | ".join(display_df.columns) + " |",
        "| " + " | ".join(["---"] * len(display_df.columns)) + " |",
    ]
    for row in display_df.values.tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary_md(
    *,
    output_dir: Path,
    summary_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    fusor_metadata: Mapping[str, object],
    metric: str,
) -> None:
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
    comparison_compact = comparison_df[
        [
            "method",
            "method_family",
            "config_name",
            "sample_count",
            "mae_like_value",
            "oracle_value",
            "regret_to_oracle",
            "oracle_label_accuracy",
            "mean_weight_entropy",
            "mean_normalized_weight_entropy",
            "mean_max_weight",
            "relative_improvement_vs_global_best_single",
        ]
    ].copy() if not comparison_df.empty else comparison_df
    lines = [
        "# Stage 1 Router Baseline 汇总",
        "",
        f"指标口径：`{metric}`",
        "",
        "## 说明",
        "",
        "- 所有可部署统计 baseline 都只用 `vali` split 学规则，并在 `test` split 上评估。",
        "- baseline 训练和主汇总默认按 `config_name` 分层；不同历史-未来 config 不共享专家动作空间。",
        "- `oracle_top1` 使用 test 窗口事后最优专家，只作为上限，不是可部署方法。",
        "- macro average 仅用于跨 config 总览，不代表一个可跨 config 部署的 router。",
        "- 旧 `timefuse_single_variable_logistic_regression` 仅作为 legacy/deprecated 历史口径保留，不作为新的主比较 baseline。",
        f"- TimeFuse-style fusor 状态：`{fusor_metadata.get('status', 'unknown')}`。",
    ]
    if fusor_metadata.get("skip_reason"):
        lines.append(f"- TimeFuse-style fusor 跳过原因：{fusor_metadata['skip_reason']}")
    lines.extend(
        [
            "",
            "## Test 结果：按 Config",
            "",
            frame_to_markdown(compact),
            "",
            "## Test 结果：跨 Config Macro Average",
            "",
            frame_to_markdown(macro_df),
            "",
            "## Unified Baseline Comparison",
            "",
            frame_to_markdown(comparison_compact),
            "",
            "## 输出文件",
            "",
            f"- `baseline_predictions.csv`: `{output_dir / 'baseline_predictions.csv'}`",
            f"- `baseline_summary.csv`: `{output_dir / 'baseline_summary.csv'}`",
            f"- `baseline_comparison.csv`: `{output_dir / 'baseline_comparison.csv'}`",
            f"- `timefuse_fusor_predictions.csv`: `{output_dir / 'timefuse_fusor_predictions.csv'}`",
            f"- `timefuse_fusor_raw_soft_fusion_predictions.csv`: `{output_dir / 'timefuse_fusor_raw_soft_fusion_predictions.csv'}`",
            f"- `baseline_metadata.json`: `{output_dir / 'baseline_metadata.json'}`",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行统计 baseline、TimeFuse-style fusor baseline 和结果落盘。"""
    args = parse_args()
    validate_fusor_args(args)
    device = resolve_device(args.device)
    output_dir = args.output_dir or args.labels_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    labels_df = load_labels(args.labels_path, args.metric)
    pred_df = make_baseline_predictions(labels_df)
    by_config = summarize_predictions(pred_df, ["config_name"])
    macro_summary = summarize_macro_by_config(by_config)
    by_dataset = summarize_predictions(pred_df, ["config_name", "dataset_name"])
    by_tsf_cell = summarize_predictions(pred_df, ["config_name", "cluster", "group_name"])
    by_dataset_tsf_cell = summarize_predictions(pred_df, ["config_name", "dataset_name", "cluster", "group_name"])

    prediction_manifest_path = resolve_prediction_manifest_path(args)
    fusor_result, fusor_metadata = run_timefuse_fusor_if_requested(
        args=args,
        labels_df=labels_df,
        prediction_manifest_path=prediction_manifest_path,
        device=device,
    )
    comparison_df = build_baseline_comparison(
        statistics_summary=by_config,
        fusor_result=fusor_result,
        metric=args.metric,
        output_dir=output_dir,
    )

    pred_df.to_csv(output_dir / "baseline_predictions.csv", index=False)
    by_config.to_csv(output_dir / "baseline_summary.csv", index=False)
    by_config.to_csv(output_dir / "baseline_summary_by_config.csv", index=False)
    macro_summary.to_csv(output_dir / "baseline_summary_macro.csv", index=False)
    by_dataset.to_csv(output_dir / "baseline_summary_by_dataset.csv", index=False)
    by_tsf_cell.to_csv(output_dir / "baseline_summary_by_tsf_cell.csv", index=False)
    by_dataset_tsf_cell.to_csv(output_dir / "baseline_summary_by_dataset_tsf_cell.csv", index=False)
    comparison_df.to_csv(output_dir / "baseline_comparison.csv", index=False)

    if fusor_result is not None:
        fusor_result["hard_pred_df"].to_csv(output_dir / "timefuse_fusor_predictions.csv", index=False)
        fusor_result["soft_pred_df"].to_csv(output_dir / "timefuse_fusor_raw_soft_fusion_predictions.csv", index=False)
        fusor_result["hard_summary_df"].to_csv(output_dir / "timefuse_fusor_summary.csv", index=False)
        fusor_result["soft_summary_df"].to_csv(output_dir / "timefuse_fusor_raw_soft_fusion_summary.csv", index=False)
        fusor_result["selected_counts_df"].to_csv(output_dir / "timefuse_fusor_selected_model_counts.csv", index=False)
    else:
        # 保持文件存在，便于外部流程判断本次是否跑过 fusor baseline。
        pd.DataFrame().to_csv(output_dir / "timefuse_fusor_predictions.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "timefuse_fusor_raw_soft_fusion_predictions.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "timefuse_fusor_summary.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "timefuse_fusor_raw_soft_fusion_summary.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "timefuse_fusor_selected_model_counts.csv", index=False)

    baseline_metadata: Dict[str, object] = {
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "labels_path": str(args.labels_path),
        "prediction_manifest_path": str(prediction_manifest_path),
        "feature_cache_path": str(args.feature_cache_path),
        "metric": args.metric,
        "training_split": "vali",
        "evaluation_split": "test",
        "model_columns": MODEL_COLUMNS,
        "statistics_baselines": [rule.name for rule in BASELINE_RULES] + ["oracle_top1"],
        "legacy_baseline": {
            "name": "timefuse_single_variable_logistic_regression",
            "status": "deprecated",
            "source": str(DEFAULT_LEGACY_LOGISTIC_SUMMARY_PATH),
        },
        "timefuse_fusor": fusor_metadata,
        "comparison_rows": int(len(comparison_df)),
    }
    (output_dir / "baseline_metadata.json").write_text(
        json.dumps(baseline_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary_md(
        output_dir=output_dir,
        summary_df=by_config,
        macro_df=macro_summary,
        comparison_df=comparison_df,
        fusor_metadata=fusor_metadata,
        metric=args.metric,
    )

    print(f"wrote baseline outputs to {output_dir}")
    print(by_config.to_string(index=False))
    if fusor_result is not None:
        print(fusor_result["hard_summary_df"].to_string(index=False))
        print(fusor_result["soft_summary_df"].to_string(index=False))


if __name__ == "__main__":
    main()
