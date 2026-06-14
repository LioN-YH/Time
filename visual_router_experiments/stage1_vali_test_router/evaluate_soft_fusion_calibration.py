#!/usr/bin/env python3
"""
文件功能：
    对 Stage 1 Visual Router 已输出的 test 权重做 soft fusion 校准评估。

设计约束：
    - 本脚本只读取 router 已预测出的五专家权重和同 sample_key 的专家预测数组；
    - 温度缩放和 top-k 截断只作用于 router 权重，不引入新的输入特征；
    - 不使用 test oracle error、未来 y 或专家误差来选择每个样本的权重；
    - 输出 raw soft、top-1 hard、top-2/top-3 fusion 和 temperature sweep 的统一对比表。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


WORKSPACE = Path("/home/shiyuhong/Time")
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from visual_router_experiments.stage1_vali_test_router.evaluate_router_baselines import MODEL_COLUMNS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router import (  # noqa: E402
    compute_array_metrics,
    load_prediction_lookup,
)


DEFAULT_ROUTER_PREDICTIONS_PATH = (
    RUN_OUTPUT_ROOT
    / "2026-06-14_025727_562553_visual_router_stage1_visual_router_smoke"
    / "visual_router_predictions.csv"
)
DEFAULT_LABELS_PATH = (
    RUN_OUTPUT_ROOT
    / "2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot"
    / "window_oracle_labels_with_tsf_cell.csv"
)
DEFAULT_PREDICTION_MANIFEST_PATH = (
    RUN_OUTPUT_ROOT
    / "2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot"
    / "manifest.csv"
)
DEFAULT_STRUCTURE_SUMMARY_PATH = (
    RUN_OUTPUT_ROOT
    / "2026-06-13_113713_308023_visual_router_stage1_structure_feature_pilot"
    / "structure_router_summary.csv"
)
EPS = 1e-8


@dataclass(frozen=True)
class CalibrationStrategy:
    """类功能：记录一种温度缩放和 top-k 截断组合。"""

    name: str
    family: str
    temperature: float
    top_k: Optional[int]

    @property
    def top_k_label(self) -> str:
        """函数功能：把 None top-k 显示为 all，方便 CSV/Markdown 阅读。"""
        return "all" if self.top_k is None else str(self.top_k)


@dataclass(frozen=True)
class SamplePredictionArrays:
    """类功能：缓存单个 sample_key 下五专家预测数组和共享 y_true。"""

    y_true: np.ndarray
    y_preds: np.ndarray


def now_token() -> str:
    """函数功能：生成 run 目录时间戳，精确到微秒避免输出目录重名。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 metadata 和 Markdown 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 soft fusion calibration 参数。"""
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 Visual Router soft fusion calibration.")
    parser.add_argument(
        "--router-predictions-path",
        type=Path,
        default=DEFAULT_ROUTER_PREDICTIONS_PATH,
        help="train_visual_router.py 输出的 visual_router_predictions.csv。",
    )
    parser.add_argument(
        "--prediction-manifest-path",
        type=Path,
        default=None,
        help="prediction cache manifest CSV；为空时优先读取 router metadata，再退回默认 pilot manifest。",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=None,
        help="window oracle labels CSV；用于定位 baseline summary，空时优先读取 router metadata。",
    )
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae", help="主排序和 regret 使用的指标。")
    parser.add_argument(
        "--temperatures",
        default="0.25,0.5,0.75,1.0,1.5,2.0",
        help="逗号分隔 softmax temperature；T<1 会 sharpen，T>1 会 flatten。",
    )
    parser.add_argument(
        "--top-k-values",
        default="all,1,2,3",
        help="逗号分隔 top-k 截断设置；支持 all,1,2,3。",
    )
    parser.add_argument("--output-root", type=Path, default=RUN_OUTPUT_ROOT, help="run 输出根目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式输出目录；默认基于 output-root 生成时间戳目录。")
    parser.add_argument("--print-rows", type=int, default=20, help="运行结束时打印多少行 comparison 预览。")
    return parser.parse_args()


def load_router_metadata(router_predictions_path: Path) -> Dict[str, object]:
    """函数功能：读取与 router predictions 同目录的 metadata；不存在时返回空字典。"""
    metadata_path = router_predictions_path.parent / "visual_router_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_optional_path(value: Optional[Path], metadata: Mapping[str, object], key: str, default_path: Path) -> Path:
    """
    函数功能：
        解析 labels/prediction manifest 路径。

    优先级：
        1. 命令行显式传入；
        2. router metadata 中记录的路径；
        3. 当前 Stage 1 pilot 默认路径。
    """
    if value is not None:
        return value
    metadata_value = metadata.get(key)
    if metadata_value:
        return Path(str(metadata_value))
    return default_path


def parse_temperatures(text: str) -> List[float]:
    """函数功能：解析并去重 temperature 列表。"""
    values: List[float] = []
    for part in str(text).split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value <= 0:
            raise ValueError(f"temperature 必须为正数，实际为 {value}")
        values.append(value)
    if not values:
        raise ValueError("--temperatures 不能为空")
    return sorted(set(values))


def parse_top_k_values(text: str) -> List[Optional[int]]:
    """函数功能：解析 top-k 设置，None 表示保留全部专家权重。"""
    values: List[Optional[int]] = []
    for part in str(text).split(","):
        stripped = part.strip().lower()
        if not stripped:
            continue
        if stripped in {"all", "none", "full"}:
            values.append(None)
            continue
        value = int(stripped)
        if value <= 0 or value > len(MODEL_COLUMNS):
            raise ValueError(f"top-k 必须落在 [1, {len(MODEL_COLUMNS)}] 或 all，实际为 {value}")
        values.append(value)
    if not values:
        raise ValueError("--top-k-values 不能为空")

    # 保留用户顺序，同时去掉重复项；None 需要特殊处理。
    deduped: List[Optional[int]] = []
    seen = set()
    for value in values:
        key = "all" if value is None else value
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def format_temperature(value: float) -> str:
    """函数功能：生成稳定、文件名友好的 temperature 文本。"""
    return f"{value:g}".replace(".", "p")


def strategy_name(temperature: float, top_k: Optional[int]) -> Tuple[str, str]:
    """函数功能：为 calibration 策略生成可读名称和家族。"""
    if top_k == 1:
        return "top1_hard", "top1_hard"
    temp_text = format_temperature(temperature)
    if top_k is None and np.isclose(temperature, 1.0):
        return "raw_soft", "raw_soft"
    if top_k is None:
        return f"soft_T{temp_text}", "temperature_soft"
    if np.isclose(temperature, 1.0):
        return f"top{top_k}_fusion", f"top{top_k}_fusion"
    return f"top{top_k}_fusion_T{temp_text}", f"top{top_k}_temperature_fusion"


def build_strategies(temperatures: Sequence[float], top_k_values: Sequence[Optional[int]]) -> List[CalibrationStrategy]:
    """
    函数功能：
        生成待评估策略清单。

    说明：
        top-1 hard 对 temperature 不敏感，因此只保留一个 `top1_hard`，避免同一结果
        在 sweep 中重复出现。
    """
    strategies: List[CalibrationStrategy] = []
    seen_names = set()
    for top_k in top_k_values:
        if top_k == 1:
            name, family = strategy_name(1.0, 1)
            if name not in seen_names:
                strategies.append(CalibrationStrategy(name=name, family=family, temperature=1.0, top_k=1))
                seen_names.add(name)
            continue
        for temperature in temperatures:
            name, family = strategy_name(float(temperature), top_k)
            if name in seen_names:
                continue
            strategies.append(CalibrationStrategy(name=name, family=family, temperature=float(temperature), top_k=top_k))
            seen_names.add(name)
    return strategies


def validate_router_predictions(pred_df: pd.DataFrame) -> None:
    """函数功能：校验 router predictions 是否包含校准所需字段。"""
    required_cols = {
        "router_name",
        "config_name",
        "sample_key",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "selected_model",
        "oracle_model",
        "oracle_value",
        *[f"weight_{model_name}" for model_name in MODEL_COLUMNS],
    }
    missing_cols = sorted(required_cols.difference(pred_df.columns))
    if missing_cols:
        raise ValueError(f"router predictions 缺少字段：{missing_cols}")
    if pred_df["sample_key"].duplicated().any():
        dup_keys = pred_df.loc[pred_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"router predictions sample_key 重复，示例：{dup_keys}")
    if not (pred_df["split"] == "test").all():
        split_values = sorted(pred_df["split"].astype(str).unique().tolist())
        raise ValueError(f"calibration 只应评估 test split，实际 split={split_values}")


def extract_weight_matrix(pred_df: pd.DataFrame) -> np.ndarray:
    """函数功能：按 MODEL_COLUMNS 顺序提取 router softmax 权重矩阵。"""
    weight_cols = [f"weight_{model_name}" for model_name in MODEL_COLUMNS]
    weights = pred_df[weight_cols].to_numpy(dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(MODEL_COLUMNS):
        raise ValueError(f"权重矩阵维度异常：{weights.shape}")
    if not np.isfinite(weights).all():
        raise ValueError("router 权重中存在非有限值")
    row_sums = weights.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(f"router 权重行和不为 1，示例={row_sums[:5]}")
    return weights


def calibrate_weights(weights: np.ndarray, *, temperature: float, top_k: Optional[int]) -> np.ndarray:
    """
    函数功能：
        对单行 router 权重执行 temperature scaling 和 top-k 截断重归一化。

    实现说明：
        训练脚本只保存 softmax 后概率，未保存 logits。这里使用 `log(p) / T`
        等价重构温度缩放后的概率；top-k 在温度缩放之后截断，排序不变但数值更清晰。
    """
    logits = np.log(np.clip(weights.astype(np.float64), EPS, 1.0)) / float(temperature)
    logits = logits - logits.max()
    scaled = np.exp(logits)
    scaled = scaled / scaled.sum()

    if top_k is None or top_k >= len(MODEL_COLUMNS):
        return scaled.astype(np.float64)
    top_indices = np.argsort(scaled)[::-1][: int(top_k)]
    truncated = np.zeros_like(scaled)
    truncated[top_indices] = scaled[top_indices]
    truncated_sum = truncated.sum()
    if truncated_sum <= 0:
        raise ValueError("top-k 截断后的权重和为 0")
    return (truncated / truncated_sum).astype(np.float64)


def weight_diagnostics(weights: np.ndarray) -> Dict[str, object]:
    """函数功能：计算单行校准权重的 entropy、max-weight 和 active model 诊断。"""
    clipped = np.clip(weights.astype(np.float64), EPS, 1.0)
    entropy = float(-(weights * np.log(clipped)).sum())
    max_idx = int(np.argmax(weights))
    active_indices = np.flatnonzero(weights > EPS)
    return {
        "selected_model": MODEL_COLUMNS[max_idx],
        "weight_entropy": entropy,
        "normalized_weight_entropy": float(entropy / np.log(len(MODEL_COLUMNS))),
        "max_weight": float(weights[max_idx]),
        "active_weight_count": int(len(active_indices)),
        "active_models": ",".join(MODEL_COLUMNS[int(idx)] for idx in active_indices),
    }


def load_sample_prediction_arrays(
    sample_keys: Sequence[str],
    prediction_lookup: Mapping[Tuple[str, str], Dict[str, object]],
) -> Dict[str, SamplePredictionArrays]:
    """
    函数功能：
        缓存每个 test sample_key 的五专家预测数组和共享 y_true。

    关键约束：
        这里只按 sample_key 读取 prediction cache 中已有数组用于评估融合误差；
        不根据 test 误差反向调整权重。
    """
    cache: Dict[str, SamplePredictionArrays] = {}
    for sample_key in sorted(set(str(key) for key in sample_keys)):
        preds: List[np.ndarray] = []
        y_true: Optional[np.ndarray] = None
        missing_models = [model_name for model_name in MODEL_COLUMNS if (sample_key, model_name) not in prediction_lookup]
        if missing_models:
            raise ValueError(f"prediction manifest 缺少 sample_key={sample_key} 的专家：{missing_models}")
        for model_name in MODEL_COLUMNS:
            record = prediction_lookup[(sample_key, model_name)]
            y_pred = np.load(record["y_pred_path"]).astype(np.float32)
            current_y_true = np.load(record["y_true_path"]).astype(np.float32)
            if y_pred.shape != current_y_true.shape:
                raise ValueError(f"y_pred/y_true shape 不一致：sample_key={sample_key} model={model_name}")
            if y_true is None:
                y_true = current_y_true
            elif not np.array_equal(y_true, current_y_true):
                raise ValueError(f"同一 sample_key 的 y_true 内容不一致：{sample_key}")
            preds.append(y_pred)
        assert y_true is not None
        stacked_preds = np.stack(preds, axis=0).astype(np.float32)
        if not (np.isfinite(stacked_preds).all() and np.isfinite(y_true).all()):
            raise ValueError(f"prediction cache 存在非有限值：sample_key={sample_key}")
        cache[sample_key] = SamplePredictionArrays(y_true=y_true, y_preds=stacked_preds)
    return cache


def evaluate_strategies(
    pred_df: pd.DataFrame,
    original_weights: np.ndarray,
    sample_cache: Mapping[str, SamplePredictionArrays],
    strategies: Sequence[CalibrationStrategy],
    metric: str,
) -> pd.DataFrame:
    """函数功能：逐策略逐样本计算校准后 fusion MAE/MSE 和权重诊断。"""
    rows: List[Dict[str, object]] = []
    for row_idx, row in pred_df.reset_index(drop=True).iterrows():
        sample_key = str(row["sample_key"])
        arrays = sample_cache[sample_key]
        sample_weights = original_weights[row_idx]
        for strategy in strategies:
            calibrated = calibrate_weights(sample_weights, temperature=strategy.temperature, top_k=strategy.top_k)
            weight_shape = (calibrated.shape[0], *([1] * (arrays.y_preds.ndim - 1)))
            fused_pred = (calibrated.reshape(weight_shape) * arrays.y_preds).sum(axis=0)
            metrics = compute_array_metrics(arrays.y_true, fused_pred)
            diagnostics = weight_diagnostics(calibrated)
            selected_model = str(diagnostics["selected_model"])
            output_row: Dict[str, object] = {
                "strategy_name": strategy.name,
                "strategy_family": strategy.family,
                "temperature": float(strategy.temperature),
                "top_k": strategy.top_k_label,
                "router_name": row["router_name"],
                "config_name": row["config_name"],
                "sample_key": sample_key,
                "split": row["split"],
                "dataset_name": row["dataset_name"],
                "item_id": int(row["item_id"]),
                "channel_id": int(row["channel_id"]),
                "window_index": int(row["window_index"]),
                "selected_model": selected_model,
                "selected_value": float(metrics[metric]),
                "mae": float(metrics["mae"]),
                "mse": float(metrics["mse"]),
                "oracle_model": row["oracle_model"],
                "oracle_value": float(row["oracle_value"]),
                "regret_to_oracle": float(metrics[metric] - float(row["oracle_value"])),
                "oracle_label_correct": bool(selected_model == row["oracle_model"]),
                **diagnostics,
            }
            for model_idx, model_name in enumerate(MODEL_COLUMNS):
                output_row[f"calibrated_weight_{model_name}"] = float(calibrated[model_idx])
                output_row[f"original_weight_{model_name}"] = float(sample_weights[model_idx])
            rows.append(output_row)
    return pd.DataFrame(rows)


def summarize_predictions(pred_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """函数功能：按策略和 config 汇总 calibration 指标。"""
    rows: List[Dict[str, object]] = []
    group_cols = ["strategy_name", "strategy_family", "config_name", "temperature", "top_k"]
    for keys, group in pred_df.groupby(group_cols, dropna=False, sort=False):
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(
            {
                "sample_count": int(len(group)),
                "metric": metric,
                "selected_value": float(group[metric].mean()),
                "mae": float(group["mae"].mean()),
                "mse": float(group["mse"].mean()),
                "oracle_value": float(group["oracle_value"].mean()),
                "regret_to_oracle": float(group["regret_to_oracle"].mean()),
                "oracle_label_accuracy": float(group["oracle_label_correct"].mean()),
                "mean_weight_entropy": float(group["weight_entropy"].mean()),
                "mean_normalized_weight_entropy": float(group["normalized_weight_entropy"].mean()),
                "mean_max_weight": float(group["max_weight"].mean()),
                "mean_active_weight_count": float(group["active_weight_count"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["config_name", "selected_value", "strategy_name"]).reset_index(drop=True)


def summarize_selected_model_counts(pred_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：汇总各策略 max-weight selected-model 分布。"""
    rows: List[Dict[str, object]] = []
    group_cols = ["strategy_name", "strategy_family", "config_name", "temperature", "top_k"]
    for keys, group in pred_df.groupby(group_cols, dropna=False, sort=False):
        base = {col: value for col, value in zip(group_cols, keys)}
        counts = group["selected_model"].value_counts().reindex(MODEL_COLUMNS, fill_value=0)
        for model_name, count in counts.items():
            rows.append(
                {
                    **base,
                    "selected_model": model_name,
                    "count": int(count),
                    "ratio": float(count / len(group)),
                }
            )
    return pd.DataFrame(rows).sort_values(["config_name", "strategy_name", "selected_model"]).reset_index(drop=True)


def build_comparison_table(
    *,
    calibration_summary: pd.DataFrame,
    labels_path: Path,
    output_dir: Path,
    metric: str,
) -> pd.DataFrame:
    """函数功能：把 calibration 结果与非视觉 baseline、结构特征 router、oracle 放到同一张表。"""
    rows: List[Dict[str, object]] = []
    baseline_path = labels_path.parent / "baseline_summary.csv"
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        for row in baseline_df.itertuples(index=False):
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
                    "mean_active_weight_count": pd.NA,
                    "source": str(baseline_path),
                }
            )

    if DEFAULT_STRUCTURE_SUMMARY_PATH.exists():
        structure_df = pd.read_csv(DEFAULT_STRUCTURE_SUMMARY_PATH)
        for row in structure_df.itertuples(index=False):
            rows.append(
                {
                    "method": str(row.router_name),
                    "method_family": "structure_feature_router",
                    "config_name": str(row.config_name),
                    "sample_count": int(row.sample_count),
                    "mae_like_value": float(row.selected_value),
                    "oracle_value": float(row.oracle_value),
                    "regret_to_oracle": float(row.regret_to_oracle),
                    "oracle_label_accuracy": float(row.oracle_label_accuracy),
                    "mean_weight_entropy": pd.NA,
                    "mean_normalized_weight_entropy": pd.NA,
                    "mean_max_weight": pd.NA,
                    "mean_active_weight_count": pd.NA,
                    "source": str(DEFAULT_STRUCTURE_SUMMARY_PATH),
                }
            )

    for row in calibration_summary.itertuples(index=False):
        rows.append(
            {
                "method": f"calibration_{row.strategy_name}",
                "method_family": str(row.strategy_family),
                "config_name": str(row.config_name),
                "sample_count": int(row.sample_count),
                "mae_like_value": float(getattr(row, metric)),
                "oracle_value": float(row.oracle_value),
                "regret_to_oracle": float(row.regret_to_oracle),
                "oracle_label_accuracy": float(row.oracle_label_accuracy),
                "mean_weight_entropy": float(row.mean_weight_entropy),
                "mean_normalized_weight_entropy": float(row.mean_normalized_weight_entropy),
                "mean_max_weight": float(row.mean_max_weight),
                "mean_active_weight_count": float(row.mean_active_weight_count),
                "source": str(output_dir / "soft_fusion_calibration_summary.csv"),
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
    """函数功能：将 DataFrame 转成 Markdown 表格，避免额外依赖 tabulate。"""
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
    comparison_df: pd.DataFrame,
    selected_counts_df: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    """函数功能：写出中文 calibration 摘要。"""
    metric = str(metadata["metric"])
    best_calibration = summary_df.sort_values(["selected_value", "strategy_name"]).iloc[0]
    soft_only = summary_df[summary_df["strategy_family"] != "top1_hard"].copy()
    best_soft = soft_only.sort_values(["selected_value", "strategy_name"]).iloc[0]
    global_rows = comparison_df[comparison_df["method"] == "global_best_single"]
    global_value = float(global_rows.iloc[0]["mae_like_value"]) if not global_rows.empty else np.nan

    if np.isfinite(global_value) and float(best_soft[metric]) < global_value:
        conclusion = (
            f"- 最佳 soft calibration `{best_soft.strategy_name}` 的 {metric.upper()}="
            f"{float(best_soft[metric]):.6f}，已经超过 `global_best_single={global_value:.6f}`。"
        )
    elif np.isfinite(global_value):
        conclusion = (
            f"- 最佳 soft calibration `{best_soft.strategy_name}` 的 {metric.upper()}="
            f"{float(best_soft[metric]):.6f}，仍未超过 `global_best_single={global_value:.6f}`。"
        )
    else:
        conclusion = f"- 最佳 soft calibration `{best_soft.strategy_name}` 的 {metric.upper()}={float(best_soft[metric]):.6f}。"

    compact_summary = summary_df[
        [
            "strategy_name",
            "config_name",
            "temperature",
            "top_k",
            "sample_count",
            "mae",
            "mse",
            "oracle_value",
            "regret_to_oracle",
            "oracle_label_accuracy",
            "mean_normalized_weight_entropy",
            "mean_max_weight",
            "mean_active_weight_count",
        ]
    ].head(20)
    compact_comparison = comparison_df[
        [
            "method",
            "config_name",
            "sample_count",
            "mae_like_value",
            "oracle_value",
            "regret_to_oracle",
            "oracle_label_accuracy",
            "mean_normalized_weight_entropy",
            "mean_max_weight",
            "relative_improvement_vs_global_best_single",
        ]
    ].head(24)
    compact_counts = selected_counts_df[selected_counts_df["strategy_name"].isin(["top1_hard", "raw_soft", str(best_soft.strategy_name)])]

    lines = [
        "# Stage 1 Soft Fusion Calibration Smoke",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 口径",
        "",
        f"- router_predictions: `{metadata['router_predictions_path']}`",
        f"- prediction_manifest: `{metadata['prediction_manifest_path']}`",
        f"- metric: `{metric}`",
        f"- temperatures: `{metadata['temperatures']}`",
        f"- top_k_values: `{metadata['top_k_values']}`",
        "- 校准只作用于 router 已输出权重，不把 test oracle error 或专家误差作为输入。",
        "",
        "## 结论",
        "",
        f"- 最佳 calibration overall：`{best_calibration.strategy_name}`，{metric.upper()}={float(best_calibration[metric]):.6f}。",
        conclusion,
        "",
        "## Calibration Summary",
        "",
        frame_to_markdown(compact_summary),
        "",
        "## Unified Comparison",
        "",
        frame_to_markdown(compact_comparison),
        "",
        "## Selected-Model 分布节选",
        "",
        frame_to_markdown(compact_counts),
        "",
        "## 输出文件",
        "",
        f"- `soft_fusion_calibration_predictions.csv`: `{output_dir / 'soft_fusion_calibration_predictions.csv'}`",
        f"- `soft_fusion_calibration_summary.csv`: `{output_dir / 'soft_fusion_calibration_summary.csv'}`",
        f"- `soft_fusion_calibration_selected_model_counts.csv`: `{output_dir / 'soft_fusion_calibration_selected_model_counts.csv'}`",
        f"- `soft_fusion_calibration_comparison.csv`: `{output_dir / 'soft_fusion_calibration_comparison.csv'}`",
        f"- `soft_fusion_calibration_metadata.json`: `{output_dir / 'soft_fusion_calibration_metadata.json'}`",
        "",
    ]
    (output_dir / "soft_fusion_calibration_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行 soft fusion calibration 评估、落盘和摘要输出。"""
    args = parse_args()
    output_dir = args.output_dir or args.output_root / f"{now_token()}_visual_router_stage1_soft_fusion_calibration_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    router_metadata = load_router_metadata(args.router_predictions_path)
    prediction_manifest_path = resolve_optional_path(
        args.prediction_manifest_path,
        router_metadata,
        "prediction_manifest_path",
        DEFAULT_PREDICTION_MANIFEST_PATH,
    )
    labels_path = resolve_optional_path(args.labels_path, router_metadata, "labels_path", DEFAULT_LABELS_PATH)

    pred_df = pd.read_csv(args.router_predictions_path)
    validate_router_predictions(pred_df)
    original_weights = extract_weight_matrix(pred_df)
    temperatures = parse_temperatures(args.temperatures)
    top_k_values = parse_top_k_values(args.top_k_values)
    strategies = build_strategies(temperatures, top_k_values)

    prediction_lookup = load_prediction_lookup(prediction_manifest_path)
    sample_cache = load_sample_prediction_arrays(pred_df["sample_key"].astype(str).tolist(), prediction_lookup)
    calibration_pred_df = evaluate_strategies(pred_df, original_weights, sample_cache, strategies, args.metric)
    summary_df = summarize_predictions(calibration_pred_df, args.metric)
    selected_counts_df = summarize_selected_model_counts(calibration_pred_df)
    comparison_df = build_comparison_table(
        calibration_summary=summary_df,
        labels_path=labels_path,
        output_dir=output_dir,
        metric=args.metric,
    )

    calibration_pred_df.to_csv(output_dir / "soft_fusion_calibration_predictions.csv", index=False)
    summary_df.to_csv(output_dir / "soft_fusion_calibration_summary.csv", index=False)
    selected_counts_df.to_csv(output_dir / "soft_fusion_calibration_selected_model_counts.csv", index=False)
    comparison_df.to_csv(output_dir / "soft_fusion_calibration_comparison.csv", index=False)

    metadata: Dict[str, object] = {
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "router_predictions_path": str(args.router_predictions_path),
        "prediction_manifest_path": str(prediction_manifest_path),
        "labels_path": str(labels_path),
        "metric": args.metric,
        "temperatures": temperatures,
        "top_k_values": ["all" if value is None else int(value) for value in top_k_values],
        "strategy_count": int(len(strategies)),
        "sample_count": int(len(pred_df)),
        "model_columns": MODEL_COLUMNS,
        "input_exclusions": ["future_y", "test_oracle_error_as_feature", "expert_error_as_feature"],
        "router_metadata": router_metadata,
    }
    (output_dir / "soft_fusion_calibration_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary_md(
        output_dir=output_dir,
        summary_df=summary_df,
        comparison_df=comparison_df,
        selected_counts_df=selected_counts_df,
        metadata=metadata,
    )

    print(f"wrote soft fusion calibration outputs to {output_dir}")
    print(comparison_df.head(int(args.print_rows)).to_string(index=False))


if __name__ == "__main__":
    main()
