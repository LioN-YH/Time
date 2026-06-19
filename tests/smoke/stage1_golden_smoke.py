#!/usr/bin/env python3
"""
文件功能：
    Stage 1 重构前 golden smoke。该脚本只读已有极小 packed cache fixture，
    用来锁定 sample_key 顺序、五专家顺序、数组 shape、hard top-1、raw soft
    fusion 指标，以及 packed_npy_v1 row index 读取行为。

输入：
    --fixture-root 指向已有 merged_cache 目录，默认使用 2026-06-14 的 4 sample
    full-scale dry-run packed fixture。

输出：
    标准输出打印中文检查日志；任一 golden 契约不一致时抛出 AssertionError。

关键约束：
    该脚本不生成预测、不训练 router、不写正式输出目录，只读取 fixture 现有
    manifest 和 npy 数组，适合作为 Stage 1 后续公共 batch reader 重构的等价门禁。
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.io.prediction_cache_reader import PredictionBatchReader  # noqa: E402
from time_router.evaluation.metrics import (  # noqa: E402
    compute_max_weight,
    compute_selected_counts,
    compute_weight_entropy,
    hard_top1_fusion,
    raw_soft_fusion,
)
from time_router.evaluation.prediction_rows import build_per_sample_fusion_rows  # noqa: E402
from time_router.evaluation.summary import build_fusion_summary  # noqa: E402


DEFAULT_FIXTURE_ROOT = (
    REPO_ROOT
    / "experiment_logs"
    / "run_outputs"
    / "2026-06-14_stage1_full_scale_dry_run_v2"
    / "merged_cache"
)
FUSION_UTILS_PATH = (
    REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "fusion_utils.py"
)
EXPECTED_MODEL_COLUMNS = ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]
EXPECTED_MANIFEST_MODEL_ORDER = ["CrossFormer", "DLinear", "ES", "NaiveForecaster", "PatchTST"]
EXPECTED_SAMPLE_KEYS = [
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win50",
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win151",
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win291",
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win873",
]
EXPECTED_Y_PRED_SHAPE = (4, 5, 48, 1)
EXPECTED_Y_TRUE_SHAPE = (4, 48, 1)
EXPECTED_HARD_MODELS = ["CrossFormer", "DLinear", "PatchTST", "DLinear"]
EXPECTED_SELECTED_COUNTS = {"DLinear": 2, "PatchTST": 1, "CrossFormer": 1, "ES": 0, "NaiveForecaster": 0}
EXPECTED_HARD_MAE = 0.41604843735694885
EXPECTED_HARD_MSE = 0.4563697576522827
EXPECTED_RAW_SOFT_MAE = 0.4102966785430908
EXPECTED_RAW_SOFT_MSE = 0.48815402388572693
EXPECTED_ROW_INDICES: Dict[str, Tuple[int, int]] = {
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win50": (1, 0),
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win151": (0, 0),
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win291": (0, 0),
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win873": (1, 0),
}
GOLDEN_WEIGHTS = np.asarray(
    [
        [0.10, 0.20, 0.55, 0.05, 0.10],
        [0.65, 0.10, 0.15, 0.05, 0.05],
        [0.05, 0.70, 0.10, 0.10, 0.05],
        [0.30, 0.30, 0.20, 0.10, 0.10],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    """函数功能：解析 smoke fixture 路径和数值容忍度参数。"""
    parser = argparse.ArgumentParser(description="Run Stage 1 golden smoke checks.")
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=DEFAULT_FIXTURE_ROOT,
        help="只读 merged_cache fixture 目录，需包含 manifest.csv 和 packed npy 数组。",
    )
    parser.add_argument("--atol", type=float, default=1e-6, help="MAE/MSE 浮点比对绝对容忍度。")
    return parser.parse_args()


def read_model_columns_from_source(source_path: Path) -> List[str]:
    """
    函数功能：
        直接从 fusion_utils.py 源码解析 MODEL_COLUMNS，避免为了一个 smoke 导入
        torch/sklearn 等训练依赖，同时仍能锁定正式专家顺序是否漂移。
    """
    text = source_path.read_text(encoding="utf-8")
    match = re.search(r"^MODEL_COLUMNS\s*=\s*(\[[^\n]+\])", text, flags=re.MULTILINE)
    if not match:
        raise AssertionError(f"未能在 {source_path} 中解析 MODEL_COLUMNS")
    value = ast.literal_eval(match.group(1))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AssertionError(f"MODEL_COLUMNS 不是字符串列表：{value!r}")
    return value


def assert_close(name: str, actual: float, expected: float, *, atol: float) -> None:
    """函数功能：带中文上下文的浮点 golden 断言。"""
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{name} 不一致：actual={actual:.12f} expected={expected:.12f} atol={atol}")


def load_manifest(fixture_root: Path) -> pd.DataFrame:
    """函数功能：读取并校验极小 golden fixture manifest。"""
    manifest_path = fixture_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 golden fixture manifest：{manifest_path}")
    df = pd.read_csv(manifest_path)
    required_cols = {
        "sample_key",
        "model_name",
        "y_true_path",
        "y_pred_path",
        "array_storage",
        "y_true_row_index",
        "y_pred_row_index",
        "mae",
        "mse",
    }
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise AssertionError(f"manifest 缺少字段：{missing}")
    if len(df) != len(EXPECTED_SAMPLE_KEYS) * len(EXPECTED_MODEL_COLUMNS):
        raise AssertionError(f"manifest 行数不一致：actual={len(df)}")
    if set(df["array_storage"].astype(str)) != {"packed_npy_v1"}:
        raise AssertionError("golden fixture 必须使用 packed_npy_v1")
    return df


def run_smoke(fixture_root: Path, *, atol: float) -> None:
    """函数功能：执行全部 Stage 1 golden smoke 检查并输出中文日志。"""
    fixture_root = fixture_root.resolve()
    print(f"开始 Stage 1 golden smoke：fixture_root={fixture_root}")

    model_columns = read_model_columns_from_source(FUSION_UTILS_PATH)
    if model_columns != EXPECTED_MODEL_COLUMNS:
        raise AssertionError(f"五专家顺序漂移：actual={model_columns} expected={EXPECTED_MODEL_COLUMNS}")
    print(f"通过：五专家顺序固定为 {model_columns}")

    manifest_df = load_manifest(fixture_root)
    sample_keys = manifest_df["sample_key"].drop_duplicates().astype(str).tolist()
    if sample_keys != EXPECTED_SAMPLE_KEYS:
        raise AssertionError(f"sample_key 顺序漂移：actual={sample_keys} expected={EXPECTED_SAMPLE_KEYS}")
    print(f"通过：sample_key 顺序固定，sample_count={len(sample_keys)}")

    reader = PredictionBatchReader(fixture_root=fixture_root, model_columns=model_columns)
    batch = reader.load(EXPECTED_SAMPLE_KEYS, verify_metrics=True)
    if batch.sample_keys != EXPECTED_SAMPLE_KEYS:
        raise AssertionError(f"reader sample_key 顺序漂移：actual={batch.sample_keys} expected={EXPECTED_SAMPLE_KEYS}")
    manifest_orders = batch.metadata["manifest_model_order_by_sample"]
    row_indices = batch.metadata["row_indices_by_sample_model"]
    for sample_key in EXPECTED_SAMPLE_KEYS:
        if manifest_orders[sample_key] != EXPECTED_MANIFEST_MODEL_ORDER:
            raise AssertionError(f"manifest 原始专家顺序漂移：sample_key={sample_key} order={manifest_orders[sample_key]}")
        for model_name in model_columns:
            if tuple(row_indices[sample_key][model_name]) != EXPECTED_ROW_INDICES[sample_key]:
                raise AssertionError(
                    f"row index 漂移：sample_key={sample_key} model={model_name} "
                    f"actual={row_indices[sample_key][model_name]} expected={EXPECTED_ROW_INDICES[sample_key]}"
                )
    y_pred, y_true = batch.y_pred, batch.y_true
    if tuple(y_pred.shape) != EXPECTED_Y_PRED_SHAPE:
        raise AssertionError(f"y_pred shape 漂移：actual={y_pred.shape} expected={EXPECTED_Y_PRED_SHAPE}")
    if tuple(y_true.shape) != EXPECTED_Y_TRUE_SHAPE:
        raise AssertionError(f"y_true shape 漂移：actual={y_true.shape} expected={EXPECTED_Y_TRUE_SHAPE}")
    print(f"通过：y_pred shape={y_pred.shape}，y_true shape={y_true.shape}")

    hard_result = hard_top1_fusion(y_pred=y_pred, y_true=y_true, weights=GOLDEN_WEIGHTS, model_columns=model_columns)
    hard_indices = hard_result.selected_indices
    hard_models = hard_result.selected_models
    if hard_indices is None or hard_models is None:
        raise AssertionError("hard top-1 helper 未返回 selected_indices / selected_models")
    if hard_models != EXPECTED_HARD_MODELS:
        raise AssertionError(f"hard top-1 选择漂移：actual={hard_models} expected={EXPECTED_HARD_MODELS}")
    if tuple(hard_result.fused_pred.shape) != EXPECTED_Y_TRUE_SHAPE:
        raise AssertionError(f"hard top-1 fused_pred shape 漂移：actual={hard_result.fused_pred.shape} expected={EXPECTED_Y_TRUE_SHAPE}")
    hard_mae = hard_result.mae
    hard_mse = hard_result.mse
    assert_close("hard top-1 MAE", hard_mae, EXPECTED_HARD_MAE, atol=atol)
    assert_close("hard top-1 MSE", hard_mse, EXPECTED_HARD_MSE, atol=atol)
    print(f"通过：hard top-1 选择={hard_models}，MAE={hard_mae:.9f}，MSE={hard_mse:.9f}")

    selected_counts = compute_selected_counts(hard_indices, model_columns)
    if selected_counts != EXPECTED_SELECTED_COUNTS:
        raise AssertionError(f"selected_counts 漂移：actual={selected_counts} expected={EXPECTED_SELECTED_COUNTS}")
    entropy = compute_weight_entropy(GOLDEN_WEIGHTS)
    max_weight = compute_max_weight(GOLDEN_WEIGHTS)
    if tuple(entropy.shape) != (len(EXPECTED_SAMPLE_KEYS),):
        raise AssertionError(f"weight entropy shape 漂移：actual={entropy.shape}")
    if tuple(max_weight.shape) != (len(EXPECTED_SAMPLE_KEYS),):
        raise AssertionError(f"max_weight shape 漂移：actual={max_weight.shape}")
    np.testing.assert_allclose(max_weight, np.max(GOLDEN_WEIGHTS, axis=1), rtol=0.0, atol=atol)
    print(f"通过：router weight diagnostics selected_counts={selected_counts}，max_weight={max_weight.tolist()}")

    raw_soft_result = raw_soft_fusion(y_pred=y_pred, y_true=y_true, weights=GOLDEN_WEIGHTS, model_columns=model_columns)
    if tuple(raw_soft_result.fused_pred.shape) != EXPECTED_Y_TRUE_SHAPE:
        raise AssertionError(f"raw soft fused_pred shape 漂移：actual={raw_soft_result.fused_pred.shape} expected={EXPECTED_Y_TRUE_SHAPE}")
    raw_soft_mae = raw_soft_result.mae
    raw_soft_mse = raw_soft_result.mse
    assert_close("raw soft fusion MAE", raw_soft_mae, EXPECTED_RAW_SOFT_MAE, atol=atol)
    assert_close("raw soft fusion MSE", raw_soft_mse, EXPECTED_RAW_SOFT_MSE, atol=atol)
    print(f"通过：raw soft fusion MAE={raw_soft_mae:.9f}，MSE={raw_soft_mse:.9f}")

    fusion_summary = build_fusion_summary(
        model_columns=model_columns,
        hard_result=hard_result,
        raw_soft_result=raw_soft_result,
        weights=GOLDEN_WEIGHTS,
    )
    assert_close("summary hard_mae", float(fusion_summary["hard_mae"]), EXPECTED_HARD_MAE, atol=atol)
    assert_close("summary hard_mse", float(fusion_summary["hard_mse"]), EXPECTED_HARD_MSE, atol=atol)
    assert_close("summary raw_soft_mae", float(fusion_summary["raw_soft_mae"]), EXPECTED_RAW_SOFT_MAE, atol=atol)
    assert_close("summary raw_soft_mse", float(fusion_summary["raw_soft_mse"]), EXPECTED_RAW_SOFT_MSE, atol=atol)
    if fusion_summary["selected_counts"] != EXPECTED_SELECTED_COUNTS:
        raise AssertionError(f"summary selected_counts 漂移：actual={fusion_summary['selected_counts']} expected={EXPECTED_SELECTED_COUNTS}")
    if fusion_summary["num_samples"] != len(EXPECTED_SAMPLE_KEYS):
        raise AssertionError(f"summary num_samples 漂移：actual={fusion_summary['num_samples']}")
    if fusion_summary["num_experts"] != len(EXPECTED_MODEL_COLUMNS):
        raise AssertionError(f"summary num_experts 漂移：actual={fusion_summary['num_experts']}")
    if fusion_summary["model_columns"] != EXPECTED_MODEL_COLUMNS:
        raise AssertionError(f"summary model_columns 漂移：actual={fusion_summary['model_columns']} expected={EXPECTED_MODEL_COLUMNS}")
    np.testing.assert_allclose(
        fusion_summary["mean_max_weight"],
        np.mean(np.max(GOLDEN_WEIGHTS, axis=1)),
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        fusion_summary["mean_entropy"],
        np.mean(compute_weight_entropy(GOLDEN_WEIGHTS)),
        rtol=0.0,
        atol=atol,
    )
    print(
        "通过：fusion summary "
        f"selected_counts={fusion_summary['selected_counts']}，"
        f"mean_entropy={fusion_summary['mean_entropy']:.9f}，"
        f"mean_max_weight={fusion_summary['mean_max_weight']:.9f}"
    )

    fusion_rows = build_per_sample_fusion_rows(
        sample_keys=EXPECTED_SAMPLE_KEYS,
        model_columns=model_columns,
        hard_result=hard_result,
        raw_soft_result=raw_soft_result,
        y_true=y_true,
        weights=GOLDEN_WEIGHTS,
    )
    required_row_fields = {
        "sample_key",
        "selected_model",
        "selected_index",
        "hard_mae",
        "hard_mse",
        "raw_soft_mae",
        "raw_soft_mse",
        "max_weight",
        "weight_entropy",
    }
    if len(fusion_rows) != len(EXPECTED_SAMPLE_KEYS):
        raise AssertionError(f"per-sample rows 数量漂移：actual={len(fusion_rows)} expected={len(EXPECTED_SAMPLE_KEYS)}")
    row_sample_keys = [row["sample_key"] for row in fusion_rows]
    row_selected_models = [row["selected_model"] for row in fusion_rows]
    row_selected_indices = [row["selected_index"] for row in fusion_rows]
    if row_sample_keys != EXPECTED_SAMPLE_KEYS:
        raise AssertionError(f"per-sample rows sample_key 顺序漂移：actual={row_sample_keys}")
    if row_selected_models != EXPECTED_HARD_MODELS:
        raise AssertionError(f"per-sample rows selected_model 漂移：actual={row_selected_models}")
    if row_selected_indices != hard_indices.astype(int).tolist():
        raise AssertionError(f"per-sample rows selected_index 漂移：actual={row_selected_indices}")
    for row in fusion_rows:
        missing_fields = sorted(required_row_fields.difference(row))
        if missing_fields:
            raise AssertionError(f"per-sample row 缺少字段：sample_key={row.get('sample_key')} missing={missing_fields}")
    row_hard_mae = [row["hard_mae"] for row in fusion_rows]
    row_raw_soft_mae = [row["raw_soft_mae"] for row in fusion_rows]
    np.testing.assert_allclose(
        row_hard_mae,
        np.mean(np.abs(hard_result.fused_pred - y_true), axis=(1, 2)),
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        [row["hard_mse"] for row in fusion_rows],
        np.mean((hard_result.fused_pred - y_true) ** 2, axis=(1, 2)),
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        row_raw_soft_mae,
        np.mean(np.abs(raw_soft_result.fused_pred - y_true), axis=(1, 2)),
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        [row["raw_soft_mse"] for row in fusion_rows],
        np.mean((raw_soft_result.fused_pred - y_true) ** 2, axis=(1, 2)),
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(np.mean([row["hard_mae"] for row in fusion_rows]), hard_result.mae, rtol=0.0, atol=atol)
    np.testing.assert_allclose(np.mean([row["raw_soft_mae"] for row in fusion_rows]), raw_soft_result.mae, rtol=0.0, atol=atol)
    np.testing.assert_allclose([row["max_weight"] for row in fusion_rows], np.max(GOLDEN_WEIGHTS, axis=1), rtol=0.0, atol=atol)
    np.testing.assert_allclose([row["weight_entropy"] for row in fusion_rows], compute_weight_entropy(GOLDEN_WEIGHTS), rtol=0.0, atol=atol)
    print(f"通过：per-sample fusion rows 行数={len(fusion_rows)}，selected_model={row_selected_models}")

    print("完成：Stage 1 golden smoke 全部通过")


def main() -> None:
    """函数功能：脚本入口。"""
    args = parse_args()
    run_smoke(args.fixture_root, atol=float(args.atol))


if __name__ == "__main__":
    main()
